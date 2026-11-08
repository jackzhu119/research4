"""Independent arithmetic/replay audit; no classifier fitting or parameter search."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/supplemental-spectra-benchmark-20261104-v1"
OUT = ROOT / "reports/supplemental-spectra-independent-audit-20261104-v1"
PROTOCOL = ROOT / "data/splits/supplemental_spectra_20261104_v1/bound_protocol.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def pure_metrics(ids, predictions, classes, truth, *, baseline=None, subset_macro=False):
    """Integer confusion counts; no repository metric implementation is called."""
    assert len(ids) == len(set(ids)) == len(predictions)
    y = np.array([truth[str(i)] for i in ids])
    vocabulary = {str(c): i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(y, predictions, strict=True):
        matrix[vocabulary[str(actual)], vocabulary[str(predicted)]] += 1
    support, predicted_counts = matrix.sum(1), matrix.sum(0)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted_counts, out=np.zeros(len(classes), float), where=predicted_counts != 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(classes), float), where=support != 0)
    denominator = support + predicted_counts
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(len(classes), float), where=denominator != 0)
    result = {"accuracy": float(diagonal.sum() / len(ids)),
              "macro_f1": float(f1[support > 0].mean() if subset_macro else f1.mean()),
              "balanced_accuracy": float(recall[support > 0].mean()),
              "confusion_matrix": matrix.tolist(), "class_order": classes.tolist(),
              "correct": int(diagonal.sum()), "total": len(ids),
              "class_counts": {str(c): int(support[i]) for i, c in enumerate(classes)},
              "class_recall": {str(c): float(recall[i]) for i, c in enumerate(classes)},
              "per_class": {str(c): {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])} for i, c in enumerate(classes)}}
    if baseline is not None:
        result.update(corrected_errors=int(np.sum((baseline != y) & (predictions == y))),
                      introduced_errors=int(np.sum((baseline == y) & (predictions != y))),
                      changed_predictions=int(np.sum(baseline != predictions)))
        original = pure_metrics(ids, baseline, classes, truth, subset_macro=subset_macro)
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            result[metric + "_gain_pp"] = 100 * (result[metric] - original[metric])
    return result


def compare_nested(left, right, path="", tolerance=2e-12):
    if isinstance(left, dict):
        for key, value in left.items():
            assert key in right, path + "/" + key
            compare_nested(value, right[key], path + "/" + key, tolerance)
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for i, value in enumerate(left):
            compare_nested(value, right[i], path + "/" + str(i), tolerance)
    elif isinstance(left, float):
        assert np.isfinite(left) and np.isfinite(right) and abs(left - right) <= tolerance, (path, left, right)
    else:
        assert left == right, (path, left, right)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    source = json.loads((RESULT / "source_lock.json").read_text())
    locked = json.loads((RESULT / "prediction_lock.json").read_text())
    assert source["protocol_sha256"] == locked["protocol_sha256"] == sha(PROTOCOL)
    assert locked["source_lock_sha256"] == sha(RESULT / "source_lock.json")
    tasks = {t["id"]: t for t in protocol["tasks"]}
    truth, packs = {}, {}
    for tid, task in tasks.items():
        assert sha(ROOT / task["truth"]["path"]) == task["truth"]["sha256"]
        with (ROOT / task["truth"]["path"]).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        truth[tid] = {r["record_id"]: r["y"] for r in rows}
        assert len(truth[tid]) == len(rows)
        packs[tid] = {}
        for role in ("source", "support", "query"):
            ref = task[role + "_pack"]
            assert sha(ROOT / ref["path"]) == ref["sha256"]
            packs[tid][role] = load(ROOT / ref["path"])
    raw = {}
    for ref in locked["predictions"]:
        path = RESULT / ref["path"]
        assert sha(path) == ref["sha256"]
        raw[ref["path"]] = load(path)
    assert len(raw) == len(locked["predictions"]) == 1020
    with gzip.open(RESULT / "scoring/all_prediction_metrics.json.gz", "rt", encoding="utf-8") as handle:
        stored_rows = json.load(handle)
    key_names = ("task", "family", "seed", "method", "shot", "draw", "stratum")
    stored = {tuple(r[k] for k in key_names): r for r in stored_rows}
    assert len(stored) == len(stored_rows) == 2175
    recomputed = []
    controls = defaultdict(int)
    native_refs = {}
    for ref in locked["predictions"]:
        if ref["method"] == "none":
            native_refs[(ref["task"], ref["family"], ref["seed"])] = ref
    expected_native = {(t, f, s) for t in tasks for f in protocol["candidates"] for s in [42, 43, 44]}
    assert set(native_refs) == expected_native and len(native_refs) == 48
    for ref in locked["predictions"]:
        pred = raw[ref["path"]]
        classes = np.array(tasks[ref["task"]]["classes"])
        assert np.array_equal(pred["classes"], classes)
        assert set(pred["record_ids"]) == set(packs[ref["task"]]["query"]["record_ids"])
        native = raw[native_refs[(ref["task"], ref["family"], ref["seed"])]["path"]]
        assert np.array_equal(pred["record_ids"], native["record_ids"])
        baseline = None if "baseline" not in ref else raw[ref["baseline"]["path"]]["predictions"]
        if ref["method"] in {"zero_label_control", "conservative"}:
            assert np.array_equal(pred["predictions"], native["predictions"])
            assert np.array_equal(pred["scores"], native["scores"])
            controls[ref["method"] + "_exact_native_identity"] += 1
        if ref["method"] == "raw_score_argmax_head_control":
            assert np.array_equal(pred["predictions"], classes[pred["scores"].argmax(1)])
            controls["raw_head_control_artifacts"] += 1
            controls["raw_head_native_label_disagreements"] += int(np.sum(pred["predictions"] != native["predictions"]))
        strata = {"all": pred["record_ids"].tolist(), **tasks[ref["task"]]["strata"]}
        for stratum, ids in strata.items():
            mask = np.array([str(i) in set(ids) for i in pred["record_ids"]])
            value = pure_metrics(pred["record_ids"][mask], pred["predictions"][mask], classes,
                truth[ref["task"]], baseline=None if baseline is None else baseline[mask], subset_macro=stratum != "all")
            header = {k: ref[k] for k in key_names[:-1]}
            key = tuple(header[k] for k in key_names[:-1]) + (stratum,)
            compare_nested(value, stored[key], str(key))
            recomputed.append({**header, "stratum": stratum, **value})
    assert len(recomputed) == 2175
    with gzip.open(OUT / "metrics_recomputed.json.gz", "xt", encoding="utf-8") as handle:
        json.dump(recomputed, handle, ensure_ascii=False, allow_nan=False)

    grouping = defaultdict(list)
    for row in recomputed:
        grouping[tuple(row[k] for k in ("task", "family", "method", "shot", "stratum"))].append(row)
    aggregates = json.loads((RESULT / "scoring/aggregate_metrics.json").read_text())
    for row in aggregates:
        values = grouping[tuple(row[k] for k in ("task", "family", "method", "shot", "stratum"))]
        assert len(values) == row["evaluations"] and row["seeds"] == [42, 43, 44]
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "corrected_errors", "introduced_errors",
                       "accuracy_gain_pp", "macro_f1_gain_pp", "balanced_accuracy_gain_pp"):
            numbers = [r[metric] for r in values if metric in r]
            if numbers:
                for stat, actual in (("mean", np.mean(numbers)), ("std", np.std(numbers)),
                                     ("min", np.min(numbers)), ("max", np.max(numbers))):
                    assert abs(float(actual) - row[metric + "_" + stat]) < 2e-12
    replays, support_scores, feature_sets = [], {}, defaultdict(dict)
    with threadpool_limits(limits=1):
        for ref in source["models"]:
            path = RESULT / ref["path"]
            assert sha(path) == ref["sha256"]
            model = joblib.load(path)
            before = joblib.hash(model)
            q = packs[ref["task"]]["query"]
            native = raw[native_refs[(ref["task"], ref["family"], ref["seed"])]["path"]]
            predicted, scores = model.predict(q["X"]), model.scores(q["X"])
            assert np.array_equal(predicted, native["predictions"])
            maxdiff = float(np.max(np.abs(scores - native["scores"])))
            assert maxdiff <= 1e-12
            assert np.allclose(model.scores(q["X"][:1]), scores[:1], rtol=1e-12, atol=1e-12)
            assert np.array_equal(model.predict(q["X"][::-1])[::-1], predicted)
            assert before == joblib.hash(model)
            s = packs[ref["task"]]["source"]
            assert model.metadata["source_ids"] == s["record_ids"].tolist()
            transformed = model.transform.preprocessor.transform(s["X"])
            assert np.allclose(model.transform.scaler.mean_, transformed.mean(0), rtol=1e-12, atol=1e-12)
            assert np.allclose(model.transform.scaler.var_, transformed.var(0), rtol=1e-12, atol=1e-12)
            assert np.array_equal(model.classes_, np.array(tasks[ref["task"]]["classes"]))
            if hasattr(model.estimator, "support_"):
                assert np.array_equal(model.estimator.support_vectors_, model.matrix(s["X"])[model.estimator.support_])
            if ref["family"] == "mcuve_svm":
                u = model.transform.selector
                sd = u.coefficients_.std(0, ddof=1)
                stability = u.coefficients_.mean(0) / np.maximum(sd, max(1., float(sd.max())) * 1e-12)
                ranking = np.argsort(-np.linalg.norm(np.abs(stability), axis=0), kind="stable")
                assert np.array_equal(ranking, u.ranking_)
                feature_sets[ref["task"]][str(ref["seed"])] = model.metadata["selected_indices"]
                support = packs[ref["task"]]["support"]
                support_scores[(ref["task"], ref["seed"])] = (model.scores(support["X"]), model.predict(support["X"]), model.classes_)
            replays.append({"task": ref["task"], "family": ref["family"], "seed": ref["seed"],
                "native_prediction_equal": True, "max_score_difference": maxdiff,
                "source_scaler_statistics_equal": True, "model_state_unchanged": True,
                "selected_features": ref["selected_features"], "model_sha256": ref["sha256"]})
            del model
        for ref in locked["predictions"]:
            if ref["method"] != "traditional":
                continue
            native = raw[native_refs[(ref["task"], ref["family"], ref["seed"])]["path"]]
            candidate = raw[ref["path"]]
            support = packs[ref["task"]]["support"]
            scores, support_native, classes = support_scores[(ref["task"], ref["seed"])]
            lookup = {str(r): i for i, r in enumerate(support["record_ids"])}
            indexes = np.array([lookup[r] for r in ref["support_ids"]])
            label_key = "y" if "y" in support else "labels"
            onehot = np.array([[float(label == c) for c in classes] for label in support[label_key][indexes]])
            design = np.column_stack([scores[indexes], np.ones(len(indexes))])
            delta = np.linalg.solve(design.T @ design + np.eye(len(classes) + 1), design.T @ (onehot - scores[indexes]))
            corrected = native["scores"] @ (np.eye(len(classes)) + delta[:-1]) + delta[-1]
            assert np.allclose(corrected, candidate["scores"], rtol=1e-12, atol=1e-12)
            changed = np.any(corrected != native["scores"], axis=1) & (corrected.argmax(1) != native["scores"].argmax(1))
            expected = native["predictions"].copy()
            expected[changed] = classes[corrected[changed].argmax(1)]
            assert np.array_equal(expected, candidate["predictions"])
            controls["traditional_reg1_closed_form_and_native_rank_guard_pass"] += 1
            unwrapped_refs = [r for r in locked["predictions"] if r["task"] == ref["task"]
                and r["seed"] == ref["seed"] and r["method"] == "traditional_unwrapped_anchor_argmax_diagnostic"
                and r["shot"] == ref["shot"] and r["draw"] == ref["draw"]]
            assert len(unwrapped_refs) == 1
            assert np.array_equal(raw[unwrapped_refs[0]["path"]]["predictions"], classes[corrected.argmax(1)])
    paired = []
    for family in protocol["candidates"]:
        for seed in [42, 43, 44]:
            a = raw[native_refs[("zaoren_legacy7", family, seed)]["path"]]
            b = raw[native_refs[("zaoren_expanded10", family, seed)]["path"]]
            b_lookup = {str(i): p for i, p in zip(b["record_ids"], b["predictions"], strict=True)}
            b_aligned = np.array([b_lookup[str(i)] for i in a["record_ids"]])
            labels = np.array([truth["zaoren_legacy7"][str(i)] for i in a["record_ids"]])
            a_ok, b_ok = a["predictions"] == labels, b_aligned == labels
            paired.append({"family": family, "seed": seed, "same_query_specimens": len(labels),
                "A_correct": int(a_ok.sum()), "B_old_subset_correct": int(b_ok.sum()),
                "A_wrong_to_B_correct": int(np.sum(~a_ok & b_ok)),
                "A_correct_to_B_wrong": int(np.sum(a_ok & ~b_ok)),
                "accuracy_gain_pp": 100 * float((b_ok.astype(int) - a_ok.astype(int)).mean()),
                "interpretation": "B adds three source samples AND three classes; no causal isolation of data benefit"})
    stability = {}
    for task, byseed in feature_sets.items():
        sets = {k: set(v) for k, v in byseed.items()}
        comparisons = {}
        keys = sorted(sets)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                comparisons[a + "_" + b] = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        stability[task] = {"selected_indices": byseed, "pairwise_jaccard": comparisons,
            "all_three_intersection": len(set.intersection(*sets.values())), "three_seed_union": len(set.union(*sets.values()))}
    save(OUT / "native_model_replay.json", replays)
    save(OUT / "paired_legacy171_A_vs_B.json", paired)
    save(OUT / "feature_selection_seed_stability.json", stability)
    save(OUT / "controls.json", dict(controls))
    summary = {"status": "PASS_INDEPENDENT_ARITHMETIC_AND_NATIVE_REPLAY", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(PROTOCOL), "source_lock_sha256": sha(RESULT / "source_lock.json"),
        "prediction_lock_sha256": sha(RESULT / "prediction_lock.json"), "audit_script_sha256": sha(__file__),
        "hashed_prediction_artifacts": len(raw), "independently_recomputed_evaluation_rows": len(recomputed),
        "aggregate_rows_recomputed": len(aggregates), "native_final_models_replayed": len(replays),
        "new_classifier_fits": 0, "new_hyperparameter_searches": 0,
        "calibration_regression_note": "Only closed-form replay of the already fixed reg1 corrections; no new calibration policy selection",
        "cpu_seconds": time.process_time(), "wall_seconds": time.monotonic() - start,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "warnings": ["Historical exposed development samples", "New-group query subset has only three specimens", "Repeated seeds/draws are not additional specimens"]}
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
