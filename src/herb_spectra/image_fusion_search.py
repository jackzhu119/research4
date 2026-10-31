"""Source-only probability fusion motivated by the user's model-fusion image.

Raw margins/logits are normalized separately before blending. Temperature
parameters come only from source OOF records. Target support labels are unused;
this is not a seven-sample target calibration experiment.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import numpy as np

from .calibration_exploration import encoded, fast_metrics


PROVIDERS = ("svm", "mlp", "mlp_sup100")
CLASS_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
PROBABILITY_METRICS = ("nll", "brier", "ece")
METRICS = CLASS_METRICS + PROBABILITY_METRICS


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def candidate_grid():
    result = []
    for provider in ("mlp", "mlp_sup100"):
        for probability in ("raw_softmax", "source_oof_temperature"):
            for weight in (0., .25, .5, .75, 1.):
                result.append({"id": f"f{len(result):03d}_fusion", "mlp_provider": provider,
                    "probability": probability, "svm_weight": weight})
    return result


def fallback_candidate():
    return next(c for c in candidate_grid() if c["mlp_provider"] == "mlp_sup100"
                and c["probability"] == "source_oof_temperature" and c["svm_weight"] == 0.)


def align_scores(scores, incoming_classes, classes):
    """Explicit class alignment; never infer class order from score magnitudes."""
    values = np.asarray(scores, dtype=float)
    incoming, target = list(map(str, incoming_classes)), list(map(str, classes))
    if (values.ndim != 2 or not len(values) or values.shape[1] != len(incoming)
            or not np.isfinite(values).all() or len(set(incoming)) != len(incoming)
            or len(set(target)) != len(target) or set(incoming) != set(target)):
        raise ValueError("Finite scores with identical unique class vocabularies are required")
    return values[:, [incoming.index(c) for c in target]]


def softmax_probabilities(scores, temperature=1.):
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 2 or len(values) < 1 or values.shape[1] < 2
            or not np.isfinite(values).all() or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Finite class scores and positive finite temperature are required")
    logits = values / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=1, keepdims=True)


def probability_metrics(probabilities, y):
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y)
    if (p.ndim != 2 or not len(p) or not np.isfinite(p).all()
            or labels.shape != (len(p),) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= p.shape[1])
            or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1., rtol=0, atol=1e-12)):
        raise ValueError("Known labels and normalized finite class probabilities are required")
    predicted = p.argmax(1)
    confidence = p.max(1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.
    for index in range(10):
        chosen = bins == index
        if chosen.any():
            ece += float(chosen.mean() * abs(np.mean(predicted[chosen] == labels[chosen])
                                           - np.mean(confidence[chosen])))
    return {"nll": float(-np.log(np.maximum(p[np.arange(len(p)), labels], 1e-15)).mean()),
            "brier": float(np.square(p - np.eye(p.shape[1])[labels]).sum(1).mean()),
            "ece": ece}


def fused_probabilities(scores_by_provider, temperatures, candidate, classes):
    """Prediction-only operation: no IDs, labels or fitted query statistics."""
    provider = candidate["mlp_provider"]
    if provider not in ("mlp", "mlp_sup100"):
        raise ValueError("Unknown MLP provider")
    weight = candidate["svm_weight"]
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("SVM fusion weight must lie in [0,1]")
    mode = candidate["probability"]
    if mode not in ("raw_softmax", "source_oof_temperature"):
        raise ValueError("Unknown probability normalization")
    probabilities = {}
    for name in ("svm", provider):
        values = align_scores(scores_by_provider[name], classes, classes)
        temperature = temperatures[name] if mode == "source_oof_temperature" else 1.
        probabilities[name] = softmax_probabilities(values, temperature)
    if probabilities["svm"].shape != probabilities[provider].shape:
        raise ValueError("Fusion requires the same specimens and class-aligned shapes")
    return weight * probabilities["svm"] + (1. - weight) * probabilities[provider]


def complementarity(y, svm_prediction, mlp_prediction, n_classes):
    y, first, second = (np.asarray(a, dtype=int) for a in (y, svm_prediction, mlp_prediction))
    if y.ndim != 1 or not len(y) or y.shape != first.shape or y.shape != second.shape:
        raise ValueError("Paired specimen predictions are required")
    if any(np.any((a < 0) | (a >= n_classes)) for a in (y, first, second)):
        raise ValueError("Unknown class index")
    good_first, good_second = first == y, second == y
    return {"n": len(y), "both_correct": int(np.sum(good_first & good_second)),
        "svm_only_correct": int(np.sum(good_first & ~good_second)),
        "mlp_only_correct": int(np.sum(~good_first & good_second)),
        "both_wrong": int(np.sum(~good_first & ~good_second)),
        "different_predictions": int(np.sum(first != second)),
        "both_wrong_same_prediction": int(np.sum(~good_first & ~good_second & (first == second))),
        "oracle_accuracy_upper_bound": float(np.mean(good_first | good_second)),
        "oracle_is_truth_based_unavailable_predictor": True}


def aggregate_development(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["candidate_id"])].append(row)
    return {stage: {c["id"]: {**{m: float(np.mean([r[m] for r in grouped[(stage, c["id"])]]))
                                         for m in METRICS},
                              "evaluations": len(grouped[(stage, c["id"])])}
                    for c in candidate_grid()}
            for stage in ("selection", "gate")}


def select_fusion(aggregate, *, minimum_macro_f1_gain=.002, max_accuracy_loss=.005,
                  max_balanced_accuracy_loss=.005, tie_tolerance=.002):
    candidates = candidate_grid()
    interiors = [c for c in candidates if 0 < c["svm_weight"] < 1]

    def endpoint(candidate, weight):
        return next(c["id"] for c in candidates if c["mlp_provider"] == candidate["mlp_provider"]
                    and c["probability"] == candidate["probability"] and c["svm_weight"] == weight)

    def passes(candidate, stage):
        value = aggregate[stage][candidate["id"]]
        for weight in (0., 1.):
            baseline = aggregate[stage][endpoint(candidate, weight)]
            if not (value["macro_f1"] - baseline["macro_f1"] > minimum_macro_f1_gain
                    and value["accuracy"] - baseline["accuracy"] >= -max_accuracy_loss
                    and value["balanced_accuracy"] - baseline["balanced_accuracy"] >= -max_balanced_accuracy_loss
                    and value["evaluations"] == baseline["evaluations"]):
                return False
        return True

    def rank(pool):
        top = max(aggregate["selection"][c["id"]]["macro_f1"] for c in pool)
        near = [c for c in pool if aggregate["selection"][c["id"]]["macro_f1"] >= top - tie_tolerance]
        return min(near, key=lambda c: (-aggregate["selection"][c["id"]]["accuracy"],
                     abs(c["svm_weight"] - .5), c["mlp_provider"], c["probability"], c["id"]))

    eligible = [c for c in interiors if passes(c, "selection")]
    proposed = rank(eligible) if eligible else None
    passed = proposed is not None and passes(proposed, "gate")
    return {"selected": proposed if passed else fallback_candidate(), "proposed": proposed,
            "diagnostic": rank(interiors), "eligible_selection_candidates": len(eligible),
            "passed_gate": passed, "fallback_to_prespecified_mlp_sup100": not passed,
            "selected_using_final_query": False}


def _context_fingerprint(context):
    return object_digest({k: context[k] for k in ("episode", "seed", "classes", "source_ids",
        "source_y", "prediction", "oof", "temperatures", "source_fingerprints", "lineage")})


def verify_context(context, *, final):
    classes = np.asarray(context["classes"])
    source, query = np.asarray(context["source_ids"]), np.asarray(context["prediction"]["record_ids"])
    if (len(classes) != 7 or len(np.unique(classes)) != 7 or len(set(source.tolist())) != len(source)
            or len(set(query.tolist())) != len(query) or set(source.tolist()) & set(query.tolist())):
        raise ValueError("Unique seven-class source/query specimen roles must be disjoint")
    if final and "y" in context["prediction"]:
        raise ValueError("Final prediction must not expose truth")
    if not final and "y" not in context["prediction"]:
        raise ValueError("Development query truth is required only for scoring")
    if len(source) != (513 if final else 307) or len(query) != (171 if final else 103):
        raise ValueError("Source/query role sizes differ from frozen protocol")
    source_y = np.asarray(context["source_y"])
    if (source_y.shape != (len(source),) or source_y.dtype.kind not in "iu"
            or set(source_y.tolist()) != set(range(7))):
        raise ValueError("Source labels must cover the fixed seven-class vocabulary")
    oof_ids = np.asarray(context["oof"]["record_ids"])
    if not set(oof_ids.tolist()) <= set(source.tolist()) or len(set(oof_ids.tolist())) != len(oof_ids):
        raise ValueError("OOF temperature records must be unique source records")
    counts = np.bincount(source_y, minlength=7)
    expected_oof_ids = set(source[counts[source_y] >= 3].tolist())
    if set(oof_ids.tolist()) != expected_oof_ids:
        raise ValueError("OOF records differ from the prespecified rare-anchor exclusion")
    oof_y = np.asarray(context["oof"]["y"])
    lookup = dict(zip(source.tolist(), source_y.tolist()))
    if (oof_y.shape != (len(oof_ids),) or oof_y.dtype.kind not in "iu"
            or any(lookup[str(i)] != y for i, y in zip(oof_ids, oof_y))):
        raise ValueError("OOF labels are not aligned with source records")
    for name in PROVIDERS:
        values = align_scores(context["prediction"]["scores_by_provider"][name], classes, classes)
        if values.shape[0] != len(query):
            raise ValueError("Provider prediction rows are not specimen aligned")
        if not np.isfinite(context["temperatures"][name]) or context["temperatures"][name] <= 0:
            raise ValueError("Source OOF temperature must be finite and positive")
        oof_scores = align_scores(context["oof"]["scores_by_provider"][name], classes, classes)
        if len(oof_scores) != len(oof_ids):
            raise ValueError("OOF scores do not match their source specimens")
        if not isinstance(context["source_fingerprints"][name], str):
            raise ValueError("Frozen source fingerprints are required for every provider")


def _prediction_writer(path, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = gzip.open(path, "wt", encoding="utf-8", newline="")
    fields = ["episode", "seed", "method", "candidate_id", "record_id", "y_true", "y_predicted",
              "is_correct", "target_support_labels_used"] + ["p_" + str(c) for c in classes]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return stream, writer


def evaluate_context(context, directory, *, methods, truth=None, final=False):
    verify_context(context, final=final)
    before = _context_fingerprint(context)
    query, classes = context["prediction"], np.asarray(context["classes"])
    # The separate final truth joins only at this scoring boundary.
    labels = encoded([truth[str(i)] for i in query["record_ids"]], classes) if final else np.asarray(query["y"], dtype=int)
    rows, controls = [], []
    for provider in ("mlp", "mlp_sup100"):
        first = np.asarray(query["scores_by_provider"]["svm"]).argmax(1)
        second = np.asarray(query["scores_by_provider"][provider]).argmax(1)
        controls.append({"episode": context["episode"], "seed": context["seed"], "mlp_provider": provider,
                         **complementarity(labels, first, second, len(classes))})
    path = Path(directory) / "predictions.csv.gz"
    stream, writer = _prediction_writer(path, classes)
    with stream:
        for method, candidate in methods.items():
            p = fused_probabilities(query["scores_by_provider"], context["temperatures"], candidate, classes)
            predicted = p.argmax(1)
            row = {"episode": context["episode"], "seed": context["seed"], "method": method,
                   "candidate_id": candidate["id"], "stage": "final" if final else
                       "selection" if context["episode"] < 4 else "gate", "target_support_labels_used": 0,
                   **fast_metrics(labels, predicted, len(classes), detailed=True),
                   **probability_metrics(p, labels)}
            rows.append(row)
            for i, actual, prediction, probability in zip(query["record_ids"], labels, predicted, p):
                writer.writerow({"episode": context["episode"], "seed": context["seed"], "method": method,
                    "candidate_id": candidate["id"], "record_id": str(i), "y_true": str(classes[actual]),
                    "y_predicted": str(classes[prediction]), "is_correct": int(actual == prediction),
                    "target_support_labels_used": 0,
                    **{"p_" + str(c): float(v) for c, v in zip(classes, probability)}})
    if _context_fingerprint(context) != before:
        raise AssertionError("Fusion mutated a source score/temperature/context")
    record = {"rows": rows, "complementarity": controls, "context_fingerprint_before": before,
        "context_fingerprint_after": _context_fingerprint(context), "source_ids": context["source_ids"],
        "query_ids": query["record_ids"], "class_order": classes, "temperatures": context["temperatures"],
        "source_fingerprints": context["source_fingerprints"], "lineage": context["lineage"],
        "predictions_sha256": digest(path), "target_support_labels_used": 0,
        "source_models_fitted_by_fusion": 0, "query_statistics_fitted": False}
    record["methods"] = methods
    record["fusion_formula"] = "w*softmax(svm_scores/T_svm)+(1-w)*softmax(mlp_scores/T_mlp); argmax fixed class order"
    write_json(Path(directory) / "evaluation.json", record)
    return rows, record


def _provenance(input_path, plan_path, protocol_path, score_run):
    root = Path(__file__).resolve().parents[2]
    files = ["src/herb_spectra/image_fusion_search.py", "scripts/run_image_fusion_search.py",
             "src/herb_spectra/image_probability.py", "src/herb_spectra/image_probability_search.py",
             "src/herb_spectra/calibration_exploration.py"]
    return {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
        "protocol_sha256": digest(protocol_path), "grid_sha256": object_digest(candidate_grid()),
        "score_run_manifest_sha256": digest(Path(score_run) / "run_manifest.json"),
        "score_selection_lock_sha256": digest(Path(score_run) / "selection_lock.json"),
        "source_sha256": {p: digest(root / p) for p in files}, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn")}}


def _verify_release(path, own_lock, score_lock):
    data = json.loads(Path(path).read_text())
    records = data.get("rounds", data.get("tracks"))
    if not isinstance(records, list):
        raise ValueError("Global release must explicitly list its named round records")
    root = Path(__file__).resolve().parents[2]
    def resolve(value):
        result = Path(value)
        return (result if result.is_absolute() else root / result).resolve()
    for lock in map(Path, (own_lock, score_lock)):
        matches = [r for r in records if isinstance(r, dict) and "selection_lock" in r
                   and resolve(r["selection_lock"]) == lock.resolve()]
        if len(matches) != 1:
            raise ValueError("Global release does not identify each exact frozen lock path once")
        record = matches[0]
        manifest = lock.parent / "run_manifest.json"
        protocol_snapshot = lock.parent / "protocol_snapshot.json"
        if (digest(lock) != record["selection_lock_sha256"]
                or digest(manifest) != record["run_manifest_sha256"]
                or digest(protocol_snapshot) != record["protocol_sha256"]
                or digest(resolve(record["protocol"])) != record["protocol_sha256"]
                or json.loads(manifest.read_text())["protocol_sha256"] != record["protocol_sha256"]):
            raise ValueError("Global release lock/protocol/manifest identity changed")
    return data


def _contexts(input_path, plan_path, score_run, scope):
    from .image_probability_search import get_score_contexts
    return get_score_contexts(input_path, plan_path, score_run, scope=scope)


def _summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return [{"method": method, "candidate_id": cases[0]["candidate_id"], "evaluations": len(cases),
        "seeds": sorted(set(r["seed"] for r in cases)), "target_support_labels_used": 0,
        **{metric + "_mean": float(np.mean([r[metric] for r in cases])) for metric in METRICS},
        **{metric + "_sd": float(np.std([r[metric] for r in cases], ddof=1)) for metric in METRICS}}
        for method, cases in grouped.items()]


def run_search(input_path, plan_path, protocol_path, score_run, output, *, stage="development",
               resume=False, release_path=None):
    if stage not in ("development", "final"):
        raise ValueError("Unknown stage")
    output, score_run = Path(output), Path(score_run)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol["candidate_grid_sha256"] != object_digest(candidate_grid()):
        raise ValueError("Frozen fusion grid differs")
    manifest = _provenance(input_path, plan_path, protocol_path, score_run)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing source/score/protocol identity changed, or resume was omitted")
    else:
        if stage != "development":
            raise ValueError("Final requires completed development")
        write_json(manifest_path, manifest)
        (output / "protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
        write_json(output / "candidate_grid.json", candidate_grid())
    event = {"event": "start", "stage": stage, "resume": resume,
             "utc": datetime.now(timezone.utc).isoformat(), "run_manifest_sha256": digest(manifest_path)}
    with (output / "stage_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    started = time.monotonic()
    lock_path = output / "selection_lock.json"
    if stage == "development":
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())
            all_rows = json.loads((output / "all_development_metrics.json").read_text())
            aggregate = aggregate_development(all_rows)
            if select_fusion(aggregate, **protocol["selection"]) != lock["choice"]:
                raise ValueError("Development choice no longer matches its metrics")
            for record in json.loads((output / "development_records.json").read_text()):
                if digest(output / record["path"]) != record["sha256"]:
                    raise ValueError("Development artifact changed")
            return {"status": "development_already_completed", "choice": lock["choice"]}
        contexts, _ = _contexts(input_path, plan_path, score_run, "development")
        if {(c["episode"], c["seed"]) for c in contexts} != {(i, s) for i in range(6) for s in (42, 43, 44)}:
            raise ValueError("Expected all18 prespecified source development contexts")
        all_rows, records, complement = [], [], []
        for context in contexts:
            relative = Path("development") / f"episode_{context['episode']:02d}" / f"seed_{context['seed']}"
            rows, record = evaluate_context(context, output / relative,
                methods={c["id"]: c for c in candidate_grid()}, final=False)
            all_rows.extend(rows)
            complement.extend(record["complementarity"])
            for name in ("evaluation.json", "predictions.csv.gz"):
                p = relative / name
                records.append({"path": str(p), "sha256": digest(output / p)})
        aggregate = aggregate_development(all_rows)
        choice = select_fusion(aggregate, **protocol["selection"])
        write_json(output / "all_development_metrics.json", all_rows)
        write_json(output / "development_aggregate.json", aggregate)
        write_json(output / "development_records.json", records)
        write_json(output / "development_complementarity.json", complement)
        write_json(lock_path, {"choice": choice, "source_only_development": True,
            "development_evaluations": len(all_rows), "run_manifest_sha256": digest(manifest_path),
            "score_selection_lock_sha256": digest(score_run / "selection_lock.json")})
        write_json(output / "status.json", {"status": "development_completed", "seconds": time.monotonic()-started,
            "development_evaluations": len(all_rows), "development_prediction_rows": sum(r["total"] for r in all_rows),
            "selection_lock_sha256": digest(lock_path), "final_prediction_scored": False,
            "new_source_model_fits": 0, "target_support_labels_used": 0})
        return {"status": "development_completed", "choice": choice}
    if not lock_path.exists() or release_path is None:
        raise ValueError("Final requires completed choice and a global release path")
    _verify_release(release_path, lock_path, score_run / "selection_lock.json")
    release_digest = digest(release_path)
    lock = json.loads(lock_path.read_text())
    own_lock_digest = digest(lock_path)
    if (output / "summary.json").exists() and resume:
        status = json.loads((output / "status.json").read_text())
        for path, value in status["final_artifacts_sha256"].items():
            if digest(output / path) != value:
                raise ValueError("Final artifact changed")
        return {"status": "final_already_completed", "summary": json.loads((output / "summary.json").read_text())}
    contexts, truth = _contexts(input_path, plan_path, score_run, "final")
    if {(c["episode"], c["seed"]) for c in contexts} != {(None, s) for s in (42, 43, 44)}:
        raise ValueError("Expected all3 final source contexts")
    methods = {"candidate_" + c["id"]: c for c in candidate_grid()}
    methods.update(selected=lock["choice"]["selected"], diagnostic=lock["choice"]["diagnostic"])
    for provider in ("mlp", "mlp_sup100"):
        for probability in ("raw_softmax", "source_oof_temperature"):
            methods[provider + "_" + probability] = next(c for c in candidate_grid() if
                c["mlp_provider"] == provider and c["probability"] == probability and c["svm_weight"] == 0.)
    for probability in ("raw_softmax", "source_oof_temperature"):
        methods["svm_" + probability] = next(c for c in candidate_grid() if
            c["mlp_provider"] == "mlp" and c["probability"] == probability and c["svm_weight"] == 1.)
    all_rows, records, complement, examples = [], [], [], []
    random_ids = None
    for context in contexts:
        relative = Path("final") / f"seed_{context['seed']}"
        rows, record = evaluate_context(context, output / relative, methods=methods, truth=truth, final=True)
        all_rows.extend(rows)
        complement.extend(record["complementarity"])
        if random_ids is None:
            random_ids = np.random.default_rng(20261102).choice(np.asarray(sorted(context["prediction"]["record_ids"])), 20, replace=False).tolist()
        with gzip.open(output / relative / "predictions.csv.gz", "rt", encoding="utf-8") as handle:
            examples.extend(row for row in csv.DictReader(handle) if row["record_id"] in random_ids
                            and row["method"] in ("selected", "diagnostic", "svm_raw_softmax", "mlp_sup100_source_oof_temperature"))
        for name in ("evaluation.json", "predictions.csv.gz"):
            p = relative / name
            records.append({"path": str(p), "sha256": digest(output / p)})
    if digest(lock_path) != own_lock_digest or digest(release_path) != release_digest:
        raise ValueError("Choice or release changed during final evaluation")
    write_json(output / "final_metrics.json", all_rows)
    write_json(output / "summary.json", _summary(all_rows))
    write_json(output / "final_complementarity.json", complement)
    write_json(output / "random_query_ids.json", random_ids)
    write_json(output / "final_records.json", records)
    with (output / "random_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(examples[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(examples)
    paths = [r["path"] for r in records] + ["final_metrics.json", "summary.json", "final_complementarity.json",
        "random_query_ids.json", "random_predictions.csv", "final_records.json"]
    write_json(output / "status.json", {"status": "completed", "final_seconds": time.monotonic()-started,
        "development_evaluations": lock["development_evaluations"], "final_evaluations": len(all_rows),
        "final_prediction_rows": sum(r["total"] for r in all_rows), "selection_lock_sha256": own_lock_digest,
        "global_release_sha256": release_digest, "target_support_labels_used": 0, "new_source_model_fits": 0,
        "final_artifacts_sha256": {p: digest(output / p) for p in paths},
        "same_batch_retrospective": True, "query_labels_used_for_fit": False})
    return {"status": "completed", "summary": _summary(all_rows)}
