"""Bounded no-fit runtime readiness; full step/prediction replay follows."""
from datetime import datetime, timezone
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import joblib
import torch
from herb_spectra.leakage_guard import array_fingerprint, source_fingerprint

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "empirical/prepared"
ARMS = ("A", "Aextra", "B", "C")
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
CHECKS, MAXIMUM = 0, 0.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(a, b, message, atol=1e-10):
    global MAXIMUM
    a, b = np.asarray(a), np.asarray(b)
    check(a.shape == b.shape, message + " shape")
    if a.size:
        MAXIMUM = max(MAXIMUM, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
    check(np.allclose(a, b, rtol=1e-10, atol=atol), message)


def snv(values):
    return (values - values.mean(1, keepdims=True)) / values.std(1, ddof=1, keepdims=True)


def state(model):
    result = hashlib.sha256()
    values = model.state_dict() if hasattr(model, "state_dict") else model
    for name, value in sorted(values.items()):
        a = value.detach().cpu().contiguous().numpy()
        result.update(name.encode()); result.update(str(a.dtype).encode())
        result.update(str(a.shape).encode()); result.update(a.tobytes())
    return result.hexdigest()


def stable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {k: stable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    return value


def native(fitted, query):
    z = fitted.scaler.transform(snv(query["X"])).astype(np.float32)
    fitted.model.eval()
    before = state(fitted.model)
    with torch.no_grad():
        scores = fitted.model({"alpha": torch.as_tensor(z)}).numpy()
    check(state(fitted.model) == before, "native query inference is stateless")
    return np.asarray(CLASSES)[scores.argmax(1)], scores


def main():
    torch.set_num_threads(1)
    manifest, lock, result = read(RUN / "frozen_manifest.json"), read(RUN / "diagnostic_prediction_lock.json"), read(RUN / "results.json")
    for relative, expected in manifest["files_sha256"].items():
        check(sha(RUN / relative) == expected, "frozen input " + relative)
    for relative, expected in manifest["source_sha256"].items():
        check(sha(ROOT / relative) == expected, "frozen code " + relative)
    check(result["score_new_fits"] == 0 and result["pipeline_fits"] == 240, "score has no fits")
    check(result["diagnostic_prediction_lock_sha256"] == sha(RUN / "diagnostic_prediction_lock.json"), "score binds prediction lock")
    check(lock["outer_truth_parsed"] is False and lock["main_selection_or_budget_changed"] is False, "all locks before parsing")
    prep = read(RUN / "preparation.json")
    contexts = {c["split_id"]: c for c in prep["contexts"]}
    outer = next(c for c in prep["contexts"] if c["stage"] == "outer")
    with np.load(RUN / outer["source_pack"], allow_pickle=False) as a:
        source = {k: a[k] for k in a.files}
    with np.load(RUN / outer["query_pack"], allow_pickle=False) as a:
        query = {k: a[k] for k in a.files}
    positions = {rid: i for i, rid in enumerate(source["record_ids"])}
    original = dict(zip(source["record_ids"].tolist(), source["source_labels"].tolist()))
    names = ["observed", "query_truth_permuted_trap", *[n["run"] for n in lock["nulls"]]]
    check(len(names) == 12 and [n["permutation_seed"] for n in lock["nulls"]] == list(range(1000, 1010)), "all fixed nulls")
    for key, path in [("observed_round_lock_sha256", "fits/observed/round_prediction_lock.json"),
                      ("query_truth_trap_round_lock_sha256", "fits/query_truth_permuted_trap/round_prediction_lock.json")]:
        check(lock[key] == sha(RUN / path), "round dependency")
    for item in lock["nulls"]:
        check(item["round_lock_sha256"] == sha(RUN / "fits" / item["run"] / "round_prediction_lock.json"), "null dependency")
    loaded, signatures, pipelines = {}, {}, 0
    for name in names:
        labels = dict(original)
        if name.startswith("null_"):
            seed = next(n["permutation_seed"] for n in lock["nulls"] if n["run"] == name)
            rng = np.random.default_rng(seed)
            for block in prep["inner_query_blocks"]:
                old = [original[rid] for rid in block]
                new = rng.permutation(np.asarray(old)).tolist()
                check(Counter(old) == Counter(new), "null class counts in block")
                labels.update(zip(block, new))
        round_lock = read(RUN / "fits" / name / "round_prediction_lock.json")
        choice = read(RUN / "fits" / name / "source_selection_lock.json")
        check(round_lock["source_selection_lock_sha256"] == sha(RUN / "fits" / name / "source_selection_lock.json"), "source lock")
        check(len(choice["inner_records"]) == 16 and len(round_lock["finals"]) == 4 and round_lock["pipeline_fits"] == 20,
              "20 actual pipelines per repeat")
        check(round_lock["outer_truth_read"] is False, "repeat prediction lock before truth")
        for record in [*choice["inner_records"], *round_lock["finals"].values()]:
            directory = RUN / record["model_directory"]
            for filename, expected in record["evidence_sha256"].items():
                check(sha(directory / filename) == expected, "actual learned evidence bytes")
            metadata = read(directory / "training.json")
            for key, filename in [("pipeline_sha256", "pipeline.joblib"), ("batch_trace_sha256", "batch_trace.jsonl.gz"),
                                  ("batch_schedule_sha256", "batch_schedule.json"), ("fit_scope_sha256", "fit_scope.jsonl")]:
                check(metadata[key] == sha(directory / filename), "metadata evidence binding")
            ids = contexts[metadata["split_id"]]["source_ids"]
            ix = [positions[rid] for rid in ids]
            y = np.asarray([labels[rid] for rid in ids], dtype=source["source_labels"].dtype)
            check(metadata["source_ids"] == ids, "current-source IDs/order")
            check(metadata["raw_target_source_fingerprint"] == source_fingerprint(source["X"][ix], y), "actual randomized/current source labels")
            counts = Counter(y)
            check(metadata["source_class_counts"] == dict(counts), "current-source class counts")
            close(metadata["target_class_weights"], [len(ids) / (7 * counts[c]) for c in CLASSES], "current-source class weights")
            fitted = joblib.load(directory / "pipeline.joblib")
            norm = snv(source["X"][ix])
            close(fitted.scaler.mean_, norm.mean(0), "source-only scaler mean")
            close(fitted.scaler.var_, norm.var(0), "source-only scaler variance")
            z = fitted.scaler.transform(norm).astype(np.float32)
            encoded = np.asarray([CLASSES.index(v) for v in y], dtype=np.int64)
            events = [json.loads(line) for line in (directory / "fit_scope.jsonl").read_text().splitlines()]
            target_ops = [e for e in events if e["event"] == "fit_start" and e["component"] == "target_torch_training"]
            check(len(target_ops) == 1 and target_ops[0]["fit_sample_ids"] == ids, "actual target operation current source")
            check(target_ops[0]["data_fingerprint"] == source_fingerprint(z, encoded), "actual target operation values and labels")
            for field in ("initial_encoder_sha256", "initial_zy_head_sha256", "fresh_target_optimizer_initial_sha256", "target_order_sha256"):
                check(isinstance(metadata[field], str) and len(metadata[field]) >= 32, "paired field present")
            check(state(fitted.model) == metadata["fitted_state_sha256"], "saved model state")
            for field, values in metadata["BN_after_reset"].items():
                check(np.all(np.asarray(values) == (1 if field.endswith("running_var") else 0)), "BN reset actual values")
            key = record.get("key", "outer/" + record["arm"])
            if name in {"observed", "query_truth_permuted_trap"}:
                signature = {"model": state(fitted.model), "target_scaler": stable(fitted.scaler.__dict__),
                             "aux_scalers": {n: stable(s.__dict__) for n, s in fitted.auxiliary_scalers.items()},
                             "pretrained_state": stable(fitted.pretrained_state), "schedule": sha(directory / "batch_schedule.json")}
                with gzip.open(directory / "batch_trace.jsonl.gz", "rb") as handle:
                    signature["decoded_trace"] = hashlib.sha256(handle.read()).hexdigest()
                signatures[name, key] = signature
            if key.startswith("outer/") and name == "observed":
                loaded[record["arm"]] = (fitted, record)
            pipelines += 1
    check(pipelines == 240, "all 240 actual weights checked")
    for key in [k for name, k in signatures if name == "observed"]:
        check(signatures["observed", key] == signatures["query_truth_permuted_trap", key], "all20 C actual learned objects identical")
    d_replays = {}
    for arm, (fitted, record) in loaded.items():
        original_pred, original_scores = native(fitted, query)
        stored = record["predictions"]
        check(original_pred.tolist() == [r["y_predicted"] for r in stored], "observed final native predictions")
        close(original_scores, [[r["score_" + c] for c in CLASSES] for r in stored], "observed final native logits", atol=1e-6)
        changed = {k: v.copy() for k, v in query.items()}; changed["X"][0, 0] += .01
        changed_pred, changed_scores = native(fitted, changed)
        check(np.array_equal(original_pred[1:], changed_pred[1:]), "D other170 native labels unchanged")
        close(original_scores[1:], changed_scores[1:], "D other170 native logits unchanged", atol=1e-6)
        check(all(lock["D_postfit_single_query"][arm][k] for k in ("model_state_unchanged", "other170_predictions_unchanged", "other170_scores_unchanged")), "stored D matches actual")
        d_replays[arm] = True
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LIMITED_ACTUAL_EMPIRICAL_READINESS_PASS_FULL_REPLAY_PENDING",
        "checks": CHECKS, "maximum_difference": MAXIMUM, "new_audit_fits": 0,
        "actual_saved_pipelines_checked": pipelines, "actual_C_learned_object_pairs": 20,
        "D_native_replayed_arms": d_replays, "all_null_current_label_scopes_checked": True,
        "prediction_lock_sha256": sha(RUN / "diagnostic_prediction_lock.json"),
        "results_sha256": sha(RUN / "results.json"), "script_sha256": sha(__file__),
        "scope": "All240 checkpoint/trace hashes, current-source scaler/classweights/target-operation label/value scopes, raw20C object equality,4D native replays. No new fitting.",
        "pending": ["Native replay all240 source/final predictions and source winners", "Every optimizer minibatch X/y fingerprint and auxiliary scopes", "Independent final/null metric recomputation"],
        "limitations": ["Limited outer0 seed42 2/4epoch4prestep diagnostic; does not select/change main grid", "Engineering-only transfer, unresolved units/ancestors, no new confirmation"],
    }
    output = HERE / "empirical_runtime_readiness.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output), "checks": CHECKS, "new_audit_fits": 0}))


if __name__ == "__main__":
    main()
