"""Deferred independent NN audit: cold plan now, arrays only after healthy stages.

No training or optimizer entry points are present. ``plan`` uses only the
standard library and does not inspect scientific arrays or Q files.
``audit`` requires the registered terminal-Q authority, exclusive original Q
lock, six healthy stage closures and all frozen prediction arms before NumPy
can be imported or any truth file opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"


ROOT = Path(__file__).resolve().parents[1]
TASKS = {"zaoren_legacy7": (513, 171, 7), "huangqi_legacy4": (241, 82, 4),
         "tianma_legacy4": (242, 83, 4), "zaoren_expanded10": (516, 174, 10)}
ARMS = ("alpha_single", "alpha_capacity3", "alpha_n_time")
SEEDS = (42, 43, 44)
EPOCHS = (24, 48, 96)
STAGES = ("prepare", "bind", "source", "fit", "predict", "score")
FORBIDDEN_NUMERICAL_MODULES = ("numpy", "torch", "pyscf", "scipy", "sklearn")
POINTS = {"alpha": 128, "n": 128, "time": 1024, "alpha_long": 1024}
BRANCHES = {"alpha_single": ("alpha",), "alpha_capacity3": ("alpha", "alpha", "alpha_long"),
            "alpha_n_time": ("alpha", "n", "time")}


class AuditBudget:
    """An explicit independent audit stage, never a reset of NN fit budgets.

    CPU/RSS snapshots concern this one process; OS CPU/AS caps and a wall alarm
    are installed only in the actual audit CLI, never in cold plan/tests.
    A hard-killed process has no healthy closure and must remain incomplete.
    """
    def __init__(self, output, cpu_seconds, wall_seconds):
        require(10 <= cpu_seconds <= 300 and 10 <= wall_seconds <= 900, "Audit budget must remain explicitly bounded")
        self.output, self.cpu_seconds, self.wall_seconds = output, float(cpu_seconds), float(wall_seconds)
        self.started = time.monotonic(); self.rss_bytes = int(1.5 * 1024**3)

    def __enter__(self):
        self.original_signals = {s: signal.getsignal(s) for s in (signal.SIGXCPU, signal.SIGALRM)}
        def stop(signum, frame):
            raise RuntimeError("EXPLICIT_AUDIT_RESOURCE_LIMIT: " + str(signum))
        for sig in self.original_signals:
            signal.signal(sig, stop)
        cpu_hard = min(int(self.cpu_seconds), resource.getrlimit(resource.RLIMIT_CPU)[1])
        if cpu_hard < 0:
            cpu_hard = int(self.cpu_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, cpu_hard - 2), cpu_hard))
        old_as = resource.getrlimit(resource.RLIMIT_AS)[1]
        as_hard = min(10 * 1024**3, old_as) if old_as >= 0 else 10 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (as_hard, as_hard))
        self.as_bytes = as_hard
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        signal.setitimer(signal.ITIMER_REAL, self.wall_seconds)
        self.check()
        return self

    def record(self):
        return {"process_CPU_seconds_including_startup": time.process_time(),
                "wall_seconds_since_audit_gate": time.monotonic() - self.started,
                "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "CPU_cap_seconds": self.cpu_seconds, "wall_cap_seconds": self.wall_seconds,
                "RSS_best_effort_cap_bytes": self.rss_bytes, "AS_hard_cap_bytes": self.as_bytes,
                "accounting": "Independent explicitly bounded audit stage; not added to or reset in the frozen NN ledger",
                "healthy_closure_required": True}

    def check(self):
        require(time.process_time() < self.cpu_seconds and time.monotonic() - self.started < self.wall_seconds,
                "Independent audit CPU/wall budget reached")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= self.rss_bytes,
                "Independent audit RSS guard reached")

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                try:
                    self.check()
                except BaseException as closed_error:
                    if self.output.exists():
                        write_new(self.output / "failure.json", {"status": "FAILED_RESOURCE_CHECK_AT_CLOSE",
                            "error": str(closed_error), "resource": self.record()})
                    raise
                write_new(self.output / "audit_stage_closed.json", {"status": "EXITED_HEALTHY", "resource": self.record(),
                    "overhead_note": "Resource snapshot precedes only this final receipt write and process return"})
            elif self.output.exists():
                write_new(self.output / "failure.json", {"status": "PARTIAL_OR_FAILED_NO_HEALTHY_AUDIT_CLOSURE",
                    "error_type": kind.__name__, "error": str(error), "resource": self.record()})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in self.original_signals.items():
                signal.signal(sig, handler)
        return False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def reference(record):
    path = (ROOT / record["path"]).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == record["sha256"], "Changed artifact reference")
    return path


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_new(path, data):
    require(data, "Cannot write an empty audit CSV")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)


def expected_grid():
    return {f"{task}.{arm}.seed{seed}" for task in TASKS for arm in ARMS for seed in SEEDS}


def validate_grid(models, predictions):
    expected = expected_grid()
    for name, records in (("model", models), ("prediction", predictions)):
        require(len(records) == len(expected) and {r["model_key"] for r in records} == expected,
                "Incomplete/duplicate " + name + " task/arm/seed grid")
        for row in records:
            require(row["model_key"] == f"{row['task_id']}.{row['arm']}.seed{row['seed']}", "Grid identity differs")


def metrics(truth, predicted, vocabulary):
    require(len(truth) == len(predicted) > 0 and len(vocabulary) == len(set(vocabulary)), "Invalid metric dimensions")
    require(set(truth) | set(predicted) <= set(vocabulary), "Unknown metric class")
    matrix = Counter(zip(truth, predicted)); supports = Counter(truth); positives = Counter(predicted)
    recalls, f1s, per_class = [], [], []
    for label in vocabulary:
        correct = matrix[label, label]
        precision = correct / positives[label] if positives[label] else 0.
        recall = correct / supports[label] if supports[label] else 0.
        denominator = supports[label] + positives[label]
        f1 = 2 * correct / denominator if denominator else 0.
        if supports[label]:
            recalls.append(recall); f1s.append(f1)
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": f1, "support": supports[label]})
    return {"accuracy_percent": 100 * sum(a == b for a, b in zip(truth, predicted)) / len(truth),
            "macro_f1": statistics.mean(f1s), "balanced_accuracy_percent": 100 * statistics.mean(recalls),
            "N": len(truth), "correct": sum(a == b for a, b in zip(truth, predicted)),
            "per_class": per_class, "confusion": [{"true_class": a, "predicted_class": b, "count": matrix[a, b]}
                                                       for a in vocabulary for b in vocabulary]}


def align(ids, expected_ids, labels):
    require(len(ids) == len(set(ids)) == len(labels) and set(ids) == set(expected_ids)
            and len(expected_ids) == len(set(expected_ids)), "Duplicate/missing prediction IDs")
    lookup = dict(zip(ids, labels)); return [lookup[i] for i in expected_ids]


def pair(truth, baseline, candidate):
    require(len(truth) == len(baseline) == len(candidate) > 0, "Pairing dimensions differ")
    corrected = sum(a != t and b == t for t, a, b in zip(truth, baseline, candidate))
    harmed = sum(a == t and b != t for t, a, b in zip(truth, baseline, candidate))
    return {"corrected": corrected, "harmed": harmed, "net_correct": corrected - harmed,
            "accuracy_delta_pp": 100 * (corrected - harmed) / len(truth)}


def validate_expanded_strata(all_ids, old_ids, legacy_ids, new_ids):
    require(len(all_ids) == len(set(all_ids)) == 174 and len(old_ids) == len(set(old_ids)) == 171,
            "Fixed expanded/old query counts changed")
    require(len(legacy_ids) == len(set(legacy_ids)) == 171 and len(new_ids) == len(set(new_ids)) == 3
            and set(legacy_ids) == set(old_ids) and set(new_ids) == set(all_ids) - set(old_ids)
            and set(legacy_ids) | set(new_ids) == set(all_ids),
            "Expanded old171/new3 identities differ from the fixed seven-class query")


def numeric_equal(left, right, message):
    require(math.isfinite(float(left)) and abs(float(left) - float(right)) < 2e-10, message)


def terminal_authority(run):
    registration = read(run / "registration.json")
    authority_path = Path(registration["authority_path"]).resolve()
    require(authority_path.is_file() and sha(authority_path) == registration["authority_sha256"], "Root authority changed")
    authority = read(authority_path)
    require(authority.get("schema") == "research4-post-quantum-fusion-authority-v1"
            and authority.get("root_execution_authorized") is True
            and authority.get("quantum_terminal_reviewed") is True
            and authority.get("implementation_review_passed") is True, "No reviewed post-Q root authority")
    qroot = Path(authority["quantum_worktree_root"]).resolve()
    qref = authority["quantum_ledger"]; qpath = Path(qref["path"]).resolve()
    require(qpath.is_relative_to(qroot / "reports") and sha(qpath) == qref["sha256"], "Terminal Q receipt changed")
    phases = read(qpath)["phases"]
    require(phases and all(p.get("status") != "RUNNING" and p.get("actual_cpu_seconds") is not None for p in phases),
            "Q remains active or has unresolved accounting; no numerical audit allowed")
    lock = Path(authority["quantum_global_lock"]).resolve()
    require(lock == qroot / ".runtime/pyscf-v4-global.lock" and lock.is_file(), "Wrong original quantum lock")
    for record in registration["code_refs"]:
        reference(record)
    reference(registration["protocol"])
    return authority, lock


def closed_chain(run):
    ledger = read(run / "resource_ledger.json")
    require([r["stage"] for r in ledger["phases"]] == list(STAGES), "Six complete stages required")
    outputs = {}
    for row in ledger["phases"]:
        require(row["status"] == "EXITED_HEALTHY" and row.get("actual_child_CPU_seconds") is not None,
                "Failed/unknown stage cannot be audited as a complete experiment")
        closed = read(reference(row["closed_stage"]))
        require(closed["stage"] == row["stage"] and closed["status"] == "DONE_HEALTHY_CLOSED_STAGE"
                and closed["exit_code"] == 0 and closed["stop_reason"] is None
                and closed["actual_child_CPU_seconds"] == row["actual_child_CPU_seconds"], "Unhealthy closure chain")
        reference(closed["supervisor_proof"])
        result = read(reference(closed["worker_result"]))
        require(result["status"] == "DONE" and result["stage"] == row["stage"]
                and result["proof_sha256"] == closed["supervisor_proof"]["sha256"], "Worker/proof identity differs")
        outputs[row["stage"]] = {k: v for k, v in result.items() if isinstance(v, dict) and {"path", "sha256"} <= set(v)}
        for value in outputs[row["stage"]].values():
            reference(value)
    require(ledger["closed_child_CPU_seconds"] + ledger["closed_supervisor_CPU_seconds_observed"]
            <= ledger["limits"]["cpu_seconds"], "Run exceeded its closed CPU budget")
    completion = read(run / "completion.json")
    require(completion["status"] == "DONE_DEVELOPMENT_ONLY" and completion["source_fits"] == 27
            and completion["final_fits"] == 36, "Experiment not complete")
    return outputs, ledger


def source_epoch_audit(source_lock, source_manifest, np):
    require(len(source_lock["models"]) == 27 and len(source_lock["locks"]) == 3, "Incomplete source epoch search")
    tasks = {t["task_id"]: t for t in source_manifest["tasks"]}
    labels, predictions = {}, defaultdict(dict)
    for task_id, task in tasks.items():
        if task_id == "zaoren_expanded10":
            require(not task["inner_folds"], "Singleton source classes cannot have epoch CV")
            continue
        with np.load(reference(task["source_pack"]), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
        labels[task_id] = dict(zip(metadata["record_ids"], metadata["source_labels"], strict=True))
        require(len(labels[task_id]) == TASKS[task_id][0], "Source size changed")
    observed_grid = set()
    for item in source_lock["models"]:
        task_id, seed, fold = item["task_id"], item["seed"], item["fold"]
        require(task_id != "zaoren_expanded10" and seed in SEEDS and fold in (0, 1, 2), "Unknown source grid")
        require((task_id, seed, fold) not in observed_grid, "Duplicate source model")
        observed_grid.add((task_id, seed, fold))
        training = read(reference(item["training"])); reference(item["model"])
        require(training["arm"] == "alpha_single" and training["epochs"] == 96 and training["seed"] == seed
                and training["source_ids"] == tasks[task_id]["inner_folds"][fold]["train_ids"], "Source selection used another arm/budget/source")
        expected_ids = tasks[task_id]["inner_folds"][fold]["valid_ids"]
        require(training["validation_ids"] == expected_ids and not training["query_features_received"]
                and not training["query_labels_received"], "Source fold accessed a different validation/query role")
        with np.load(reference(item["source_epoch_predictions"]), allow_pickle=False) as archive:
            ids, classes = archive["record_ids"].astype(str).tolist(), archive["class_labels"].astype(str).tolist()
            require(ids == expected_ids and classes == training["class_labels"], "Source OOF columns/rows changed")
            for epoch in EPOCHS:
                logits = archive["epoch_" + str(epoch)]
                require(logits.shape == (len(ids), len(classes)) and bool(np.isfinite(logits).all()), "Invalid OOF logits")
                bucket = predictions[(task_id, seed, epoch)]
                require(not set(bucket) & set(ids), "Repeated source OOF entity")
                bucket.update(zip(ids, [classes[int(i)] for i in logits.argmax(1)], strict=True))
    require(observed_grid == {(t, s, f) for t in labels for s in SEEDS for f in (0, 1, 2)}, "Source grid has missing seeds/folds")
    report = {}
    for task_id, truth in labels.items():
        by_epoch = defaultdict(list); seed_rows = []
        for seed in SEEDS:
            for epoch in EPOCHS:
                predicted = predictions[(task_id, seed, epoch)]
                require(set(predicted) == set(truth), "Incomplete pooled source OOF")
                score = metrics(list(truth.values()), [predicted[i] for i in truth], sorted(set(truth.values())))
                converted = {"epoch": epoch, "seed": seed, "macro_f1": score["macro_f1"],
                    "balanced_accuracy": score["balanced_accuracy_percent"] / 100, "accuracy": score["accuracy_percent"] / 100}
                seed_rows.append(converted); by_epoch[epoch].append(converted)
        locked = source_lock["locks"][task_id]
        for actual in seed_rows:
            found = [r for r in locked["seed_metrics"] if (r["seed"], r["epoch"]) == (actual["seed"], actual["epoch"])]
            require(len(found) == 1, "Missing/duplicate source epoch metric")
            for key in ("accuracy", "macro_f1", "balanced_accuracy"):
                numeric_equal(actual[key], found[0][key], "Source metric differs")
        candidates = [{"epoch": epoch, **{m: statistics.mean(r[m] for r in by_epoch[epoch])
                      for m in ("accuracy", "macro_f1", "balanced_accuracy")}} for epoch in EPOCHS]
        chosen = min(candidates, key=lambda r: (-r["macro_f1"], -r["balanced_accuracy"], -r["accuracy"], r["epoch"]))
        require(chosen["epoch"] == locked["epoch"] and locked["scope"] == "SOURCE_OOF_ONLY", "Wrong independently chosen epoch")
        report[task_id] = {"selected_epoch": chosen["epoch"], "candidates": candidates}
    require(source_lock["expanded10_epoch"] == report["zaoren_legacy7"]["selected_epoch"], "Expanded epoch not inherited")
    return report


def replay_saved_models(models, training, source_tasks, tasks, locked, output, np, budget):
    """Actual no-fit replay, called only by the explicit deferred option.

    The reference artifact stores logits, not probabilities. Probability checks
    compare softmax derived from frozen logits to softmax of replayed logits.
    """
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from herb_spectra.upgrade_neural import Encoder
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    def state_hash(model):
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode()); digest.update(value.tobytes())
        return digest.hexdigest()

    class ReplayNet(torch.nn.Module):
        def __init__(self, inputs, classes):
            super().__init__(); self.inputs = inputs
            self.encoders = torch.nn.ModuleList([Encoder(12, 2, .35, True, False) for _ in inputs])
            self.head = torch.nn.Linear(16 * len(inputs), classes)
        def forward(self, values):
            return self.head(torch.cat([encoder(values[mode]) for encoder, mode in zip(self.encoders, self.inputs)], dim=1))

    def pack(record):
        with np.load(reference(record), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            values = {mode: archive["X_" + mode] for mode in POINTS}
        return metadata, values

    def softmax(value):
        value = np.asarray(value, dtype=np.float64)
        shifted = np.exp(value - value.max(1, keepdims=True))
        return shifted / shifted.sum(1, keepdims=True)

    cached = {}
    prediction_by_key = {r["model_key"]: r for r in locked["predictions"]}
    reports = []; total_cpu = time.process_time(); total_wall = time.monotonic()
    for item in models["models"]:
        budget.check(); started_cpu = time.process_time(); started_wall = time.monotonic()
        key, task_id, arm, seed = item["model_key"], item["task_id"], item["arm"], item["seed"]
        metadata = training[key]; inputs = BRANCHES[arm]
        if task_id not in cached:
            cached[task_id] = (pack(source_tasks[task_id]["source_pack"]), pack(tasks[task_id]["query_pack"]))
        (source_meta, source_values), (query_meta, query_values) = cached[task_id]
        require(source_meta["record_ids"] == metadata["source_ids"] and query_meta["record_ids"] == tasks[task_id]["query_ids"],
                "Replay source/query row binding differs")
        query = {}
        for mode in dict.fromkeys(inputs):
            require(metadata["source_axis_metadata"][mode] == source_meta["axis_metadata"][mode]
                    == query_meta["axis_metadata"][mode], "Replay physical axis/quantity contract changed")
            source = np.asarray(source_values[mode], dtype=np.float64)
            raw = np.asarray(query_values[mode], dtype=np.float64)
            require(source.shape == (len(metadata["source_ids"]), POINTS[mode])
                    and raw.shape == (len(query_meta["record_ids"]), POINTS[mode])
                    and bool(np.isfinite(source).all()) and bool(np.isfinite(raw).all()), "Replay feature shape/values invalid")
            scaler = metadata["source_scalers"][mode]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            require(mean.shape == scale.shape == (POINTS[mode],) and bool(np.isfinite(mean).all())
                    and bool(np.isfinite(scale).all()) and bool((scale > 0).all()), "Invalid frozen source scaler")
            actual_mean, variance = source.mean(0), source.var(0)
            eps = np.finfo(np.float64).eps; n = len(source)
            constant = variance <= n * eps * variance + (n * actual_mean * eps) ** 2
            actual_scale = np.sqrt(variance); actual_scale[constant] = 1.
            require(bool(np.allclose(mean, actual_mean, rtol=1e-10, atol=1e-10))
                    and bool(np.allclose(scale, actual_scale, rtol=1e-6, atol=1e-12)), "Saved scaler not consistent with current source values")
            query[mode] = torch.as_tensor(((raw - mean) / scale).astype(np.float32))
        torch.manual_seed(seed); model = ReplayNet(inputs, len(metadata["class_labels"]))
        require(state_hash(model) == metadata["initial_state_sha256"], "Independently rebuilt initialization differs")
        require(sum(p.numel() for p in model.parameters()) == metadata["parameter_count"], "Replay parameter count changed")
        model.load_state_dict(torch.load(reference(item["model"]), map_location="cpu", weights_only=True))
        model.eval(); before = state_hash(model)
        require(before == metadata["fitted_state_sha256"], "Saved fitted state dict differs")
        with torch.no_grad():
            logits = model(query).cpu().numpy()
            reversed_logits = model({m: value.flip(0) for m, value in query.items()}).cpu().numpy()[::-1]
            first_logits = model({m: value[:1] for m, value in query.items()}).cpu().numpy()
        require(before == state_hash(model) and not model.training, "Replay mutated parameters/BN state or used train mode")
        with np.load(reference(prediction_by_key[key]["predictions"]), allow_pickle=False) as archive:
            expected = archive["logits"]
            require(archive["record_ids"].astype(str).tolist() == query_meta["record_ids"]
                    and archive["class_labels"].astype(str).tolist() == metadata["class_labels"], "Replay output row/class order changed")
        require(logits.shape == expected.shape and bool(np.isfinite(logits).all()), "Invalid replay logits")
        require(bool(np.allclose(logits, expected, rtol=1e-5, atol=1e-5)), "Replay logits exceed the fixed tolerance")
        require(bool(np.array_equal(logits.argmax(1), expected.argmax(1))), "Replay changes frozen class predictions")
        require(bool(np.allclose(reversed_logits, logits, rtol=1e-5, atol=1e-5))
                and bool(np.allclose(first_logits, logits[:1], rtol=1e-5, atol=1e-5)), "Replay depends on query order/batch size")
        require(bool(np.allclose(softmax(logits), softmax(expected), rtol=1e-5, atol=1e-5)), "Derived softmax replay differs")
        row = {"model_key": key, "task_id": task_id, "arm": arm, "seed": seed, "records": len(logits),
            "status": "ACTUAL_NO_FIT_NATIVE_REPLAY_PASS", "source_scaler_statistics_pass": True,
            "max_absolute_logit_difference": float(np.max(np.abs(logits - expected))),
            "max_absolute_derived_softmax_difference": float(np.max(np.abs(softmax(logits) - softmax(expected)))),
            "logit_atol": 1e-5, "logit_rtol": 1e-5, "derived_probability_atol": 1e-5, "derived_probability_rtol": 1e-5,
            "recorded_artifact_contains_probabilities": False, "native_predictions_exact": True,
            "fitted_state_and_BN_unchanged": True, "source_model_sha256": item["model"]["sha256"],
            "actual_process_CPU_seconds": time.process_time() - started_cpu,
            "actual_wall_seconds": time.monotonic() - started_wall}
        with (output / "model_replay_progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        reports.append(row); del model, query; budget.check()
    require(len(reports) == 36, "Actual native replay is incomplete")
    summary = {"status": "DONE_ACTUAL_NO_FIT_REPLAY", "models_replayed": 36, "reports": reports,
        "CPU_seconds": time.process_time() - total_cpu, "wall_seconds": time.monotonic() - total_wall,
        "Torch_version": torch.__version__, "classifier_fits": 0, "device": "cpu"}
    write_new(output / "native_model_replay.json", summary)
    return summary


def audit(run, output, *, replay_models=False, cpu_seconds=180, wall_seconds=600):
    started = time.monotonic()
    require(run.is_relative_to(ROOT / "reports") and output.is_relative_to(ROOT / "reports"), "Audit paths must remain inside this worktree's reports")
    authority, qlock_path = terminal_authority(run)  # No NumPy or prediction/truth arrays above this gate.
    require(not output.exists(), "Fresh independent audit directory required")
    with (run / ".run.lock").open("a") as local, qlock_path.open("a") as qlock, AuditBudget(output, cpu_seconds, wall_seconds) as budget:
        fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(qlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages, ledger = closed_chain(run)
        require(ledger["limits"] == authority["limits"], "Resource ledger limits differ from the frozen authority")
        source_lock = read(reference(stages["source"]["source_lock"]))
        models = read(reference(stages["fit"]["model_lock"]))
        locked = read(reference(stages["predict"]["prediction_lock"]))
        require(locked["models"] == stages["fit"]["model_lock"]
                and models["source_epoch_lock"] == stages["source"]["source_lock"], "Source/model/prediction chain differs")
        source_manifest = read(reference(source_lock["source_manifest"]))
        source_tasks = {t["task_id"]: t for t in source_manifest["tasks"]}
        require(set(source_tasks) == set(TASKS) and len(source_manifest["tasks"]) == len(TASKS), "Incomplete/duplicate source manifest")
        manifest = read(reference(locked["prediction_manifest"]))
        tasks = {t["task_id"]: t for t in manifest["tasks"]}
        require(set(tasks) == set(TASKS) and len(manifest["tasks"]) == len(TASKS), "Task set changed/duplicated")
        validate_grid(models["models"], locked["predictions"])
        by_key, training = {}, {}
        for item in models["models"]:
            reference(item["model"]); metadata = read(reference(item["training"])); training[item["model_key"]] = metadata
            require(metadata["task_id"] == item["task_id"] and metadata["arm"] == item["arm"]
                    and metadata["seed"] == item["seed"] and not metadata["validation_ids"]
                    and not metadata["query_features_received"] and not metadata["query_labels_received"], "Invalid final training identity")
            require(metadata["source_ids"] == source_tasks[item["task_id"]]["source_ids"]
                    and len(metadata["source_ids"]) == TASKS[item["task_id"]][0]
                    and len(set(metadata["source_ids"])) == len(metadata["source_ids"]), "Final training source role changed")
            owner = "zaoren_legacy7" if item["task_id"] == "zaoren_expanded10" else item["task_id"]
            require(item["epochs"] == metadata["epochs"] == source_lock["locks"][owner]["epoch"], "Unequal/unlocked epoch budget")
            by_key[item["model_key"]] = item
        pairing = []
        for task_id in TASKS:
            for seed in SEEDS:
                capacity = training[f"{task_id}.alpha_capacity3.seed{seed}"]
                multi = training[f"{task_id}.alpha_n_time.seed{seed}"]
                for field in ("task_id", "seed", "epochs", "source_ids", "source_batch_order_sha256", "initial_state_sha256", "initial_encoder_sha256", "parameter_count", "source_class_weights"):
                    require(capacity[field] == multi[field], "C/M same-capacity control differs: " + field)
                require(capacity["branch_spec"]["branch_points"] == multi["branch_spec"]["branch_points"] == [128, 128, 1024], "C/M lengths differ")
                pairing.append({"task_id": task_id, "seed": seed, "epochs": capacity["epochs"],
                                "same_parameter_count": capacity["parameter_count"], "same_initial_state_and_order": True})
        output.mkdir(parents=True)
        write_new(output / "execution_start.json", {"status": "RUNNING_NOT_A_COMPLETED_RESULT", "resource": budget.record(),
            "replay_models_requested": bool(replay_models), "classifier_fits": 0})
        # This delayed import occurs only after terminal-Q lock and complete stage/model metadata checks.
        import numpy as np
        source_check = source_epoch_audit(source_lock, source_manifest, np)
        for task_id, task in source_tasks.items():
            with np.load(reference(task["source_pack"]), allow_pickle=False) as archive:
                source_metadata = json.loads(str(archive["metadata_json"]))
            with np.load(reference(tasks[task_id]["query_pack"]), allow_pickle=False) as archive:
                query_metadata = json.loads(str(archive["metadata_json"]))
            require(source_metadata["record_ids"] == task["source_ids"]
                    and query_metadata["record_ids"] == tasks[task_id]["query_ids"], "Prepared metadata IDs changed")
            require(not set(query_metadata) & {"source_labels", "y", "labels", "truth"}, "Prediction pack exposed truth")
            source_aliases = set(source_metadata["record_ids"]) | set(source_metadata["entity_ids"])
            query_aliases = set(query_metadata["record_ids"]) | set(query_metadata["entity_ids"])
            for values in source_metadata["identity_aliases"]:
                source_aliases.update(values)
            for values in query_metadata["identity_aliases"]:
                query_aliases.update(values)
            require(source_aliases == set(task["source_alias_union"]) and not source_aliases & query_aliases,
                    "Prepared source/query physical aliases overlap or changed")
            expected_classes = sorted(set(source_metadata["source_labels"]))
            counts = Counter(source_metadata["source_labels"])
            for arm in ARMS:
                for seed in SEEDS:
                    metadata = training[f"{task_id}.{arm}.seed{seed}"]
                    require(metadata["class_labels"] == expected_classes, "Source class vocabulary changed")
                    expected_weights = [len(source_metadata["record_ids"]) / (len(expected_classes) * counts[c]) for c in expected_classes]
                    require(all(abs(a - b) <= 1e-6 * max(1., abs(a))
                                for a, b in zip(expected_weights, metadata["source_class_weights"], strict=True)), "Class weights are not source-only balanced weights")
        predictions = {}
        for item in locked["predictions"]:
            budget.check()
            prediction_metadata = read(reference(item["prediction_metadata"]))
            require(prediction_metadata["model_key"] == item["model_key"]
                    and prediction_metadata["status"] == "LABEL_FREE_PREDICTIONS_AWAIT_INDEPENDENT_SCORING"
                    and prediction_metadata["query_ids"] == tasks[item["task_id"]]["query_ids"]
                    and prediction_metadata["query_labels_received"] is False
                    and prediction_metadata["inference_audit"].get("status") == "TECHNICAL_CHECK_PASS"
                    and prediction_metadata["inference_audit"].get("samples_checked") == TASKS[item["task_id"]][1]
                    and prediction_metadata["inference_audit"].get("eval_mode") is True
                    and prediction_metadata["inference_audit"].get("parameters_and_buffers_unchanged") is True
                    and prediction_metadata["inference_audit"].get("gradients_enabled_during_forward") is False,
                    "Prediction metadata identity/pointwise audit is not healthy")
            with np.load(reference(item["predictions"]), allow_pickle=False) as archive:
                require(set(archive.files) == {"record_ids", "class_labels", "logits"}, "Unexpected prediction fields")
                ids = archive["record_ids"].astype(str).tolist(); classes = archive["class_labels"].astype(str).tolist()
                logits = archive["logits"]
                require(classes == training[item["model_key"]]["class_labels"] and len(classes) == TASKS[item["task_id"]][2], "Changed class-column order")
                require(logits.shape == (len(ids), len(classes)) and bool(np.isfinite(logits).all()), "Invalid logits")
                labels = [classes[int(i)] for i in logits.argmax(1)]
                predictions[item["model_key"]] = (classes, align(ids, tasks[item["task_id"]]["query_ids"], labels))
        replay = (replay_saved_models(models, training, source_tasks, tasks, locked, output, np, budget)
                  if replay_models else {"status": "NOT_RUN_HASH_VERIFICATION_ONLY", "models_replayed": 0,
                                        "model_weight_files_sha_verified": 36, "Torch_imported": False})
        # Every prediction/grid/column has been validated before the first truth open.
        true, all_scores, paired_rows, new_rows = {}, [], [], []
        for task_id, task in tasks.items():
            rows = csv_rows(reference(task["truth"])); truth = {r["record_id"]: r for r in rows}
            ids = task["query_ids"]
            require(len(truth) == len(rows) == len(ids) == TASKS[task_id][1] and set(truth) == set(ids), "Truth rows differ")
            true[task_id] = truth
            subsets = {"overall": list(range(len(ids)))}
            if task_id == "zaoren_expanded10":
                subsets["legacy_exposed171"] = [i for i, identifier in enumerate(ids) if truth[identifier]["stratum"] == "legacy_exposed"]
                subsets["new3_descriptive_only"] = [i for i, identifier in enumerate(ids) if truth[identifier]["stratum"] == "new_three_record_group"]
                require(len(subsets["legacy_exposed171"]) == 171 and len(subsets["new3_descriptive_only"]) == 3, "Expanded subsets differ")
                validate_expanded_strata(ids, tasks["zaoren_legacy7"]["query_ids"],
                    [ids[i] for i in subsets["legacy_exposed171"]], [ids[i] for i in subsets["new3_descriptive_only"]])
            labels = [truth[i]["y"] for i in ids]
            for seed in SEEDS:
                arms = {}
                for arm in ARMS:
                    classes, predicted = predictions[f"{task_id}.{arm}.seed{seed}"]; arms[arm] = predicted
                    for name, positions in subsets.items():
                        score = metrics([labels[i] for i in positions], [predicted[i] for i in positions], classes)
                        all_scores.append({"task_id": task_id, "subset": name, "arm": arm, "seed": seed, **score})
                    if task_id == "zaoren_expanded10":
                        new_rows.extend({"record_id": ids[i], "truth": labels[i], "prediction": predicted[i], "arm": arm,
                            "seed": seed, "correct": labels[i] == predicted[i], "independent_new_query_specimens": 3}
                            for i in subsets["new3_descriptive_only"])
                for base, candidate in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
                    for name, positions in subsets.items():
                        paired_rows.append({"task_id": task_id, "subset": name, "seed": seed, "baseline": base, "candidate": candidate,
                            "N": len(positions), **pair([labels[i] for i in positions], [arms[base][i] for i in positions], [arms[candidate][i] for i in positions])})
        scored = csv_rows(run / "score/per_seed_metrics.csv")
        score_by_key = {(r["task_id"], r["subset"], r["arm"], int(r["seed"])): r for r in scored}
        require(len(score_by_key) == len(scored) == len(all_scores) == 54, "Incomplete per-seed scores")
        for actual in all_scores:
            stored = score_by_key[(actual["task_id"], actual["subset"], actual["arm"], actual["seed"])]
            for field in ("N", "correct", "accuracy_percent", "macro_f1", "balanced_accuracy_percent"):
                numeric_equal(actual[field], stored[field], "Independent score differs: " + field)
        class_key = lambda r: (r["task_id"], r["subset"], r["arm"], int(r["seed"]), r["class"])
        class_rows = csv_rows(run / "score/per_class_metrics.csv")
        class_by_key = {class_key(r): r for r in class_rows}
        confusion_key = lambda r: (r["task_id"], r["subset"], r["arm"], int(r["seed"]), r["true_class"], r["predicted_class"])
        confusion_rows = csv_rows(run / "score/confusion_matrix_counts.csv")
        confusion_by_key = {confusion_key(r): r for r in confusion_rows}
        require(len(class_by_key) == len(class_rows) == sum(len(r["per_class"]) for r in all_scores), "Missing/duplicate per-class rows")
        require(len(confusion_by_key) == len(confusion_rows) == sum(len(r["confusion"]) for r in all_scores), "Missing/duplicate confusion rows")
        for actual in all_scores:
            prefix = {k: actual[k] for k in ("task_id", "subset", "arm", "seed")}
            for row in actual["per_class"]:
                stored = class_by_key[class_key({**prefix, **row})]
                for field in ("precision", "recall", "f1", "support"):
                    numeric_equal(row[field], stored[field], "Per-class metric differs")
            for row in actual["confusion"]:
                numeric_equal(row["count"], confusion_by_key[confusion_key({**prefix, **row})]["count"], "Confusion count differs")
        groups = defaultdict(list)
        for row in all_scores:
            groups[(row["task_id"], row["subset"], row["arm"])].append(row)
        summaries = csv_rows(run / "score/model_comparison.csv")
        require(len(summaries) == len(groups) == 18, "Missing/duplicate summary group")
        summary_keys = set()
        for stored in summaries:
            key = tuple(stored[k] for k in ("task_id", "subset", "arm")); require(key not in summary_keys, "Duplicate summary")
            summary_keys.add(key); values = groups[key]
            require(sorted(r["seed"] for r in values) == list(SEEDS), "A best seed replaced the full seed set")
            for metric in ("accuracy_percent", "macro_f1", "balanced_accuracy_percent"):
                for suffix, operation in (("mean", statistics.mean), ("sampleSD", statistics.stdev)):
                    numeric_equal(operation(r[metric] for r in values), stored[metric + "_" + suffix], "Summary differs")
        stored_pairs = csv_rows(run / "score/paired_corrected_harmed.csv")
        pair_key = lambda r: (r["task_id"], r["subset"], int(r["seed"]), r["baseline"], r["candidate"])
        pair_by_key = {pair_key(r): r for r in stored_pairs}
        require(len(pair_by_key) == len(stored_pairs) == len(paired_rows) == 54, "Incomplete paired contrasts")
        for actual in paired_rows:
            for field in ("N", "corrected", "harmed", "net_correct", "accuracy_delta_pp"):
                numeric_equal(actual[field], pair_by_key[pair_key(actual)][field], "Paired count differs")
        retention = []
        for arm in ARMS:
            for seed in SEEDS:
                ids = tasks["zaoren_legacy7"]["query_ids"]
                old_truth = [true["zaoren_legacy7"][i]["y"] for i in ids]
                _, old = predictions[f"zaoren_legacy7.{arm}.seed{seed}"]
                _, expanded = predictions[f"zaoren_expanded10.{arm}.seed{seed}"]
                expanded_old = align(tasks["zaoren_expanded10"]["query_ids"], tasks["zaoren_expanded10"]["query_ids"], expanded)
                lookup = dict(zip(tasks["zaoren_expanded10"]["query_ids"], expanded_old))
                require(set(ids) <= set(lookup) and all(true["zaoren_expanded10"][i]["y"] == t for i, t in zip(ids, old_truth)), "Old171 identity/truth changed")
                retention.append({"arm": arm, "seed": seed, "N": 171,
                    "A_correct": sum(a == t for a, t in zip(old, old_truth)),
                    "B_old_correct": sum(lookup[i] == t for i, t in zip(ids, old_truth)),
                    **pair(old_truth, old, [lookup[i] for i in ids]),
                    "interpretation": "Known-class retention after adding three source classes; not a fresh confirmation"})
        require(len(new_rows) == 27, "New-three query predictions incomplete")
        write_new(output / "source_epoch_audit.json", source_check); write_new(output / "capacity_pairing_audit.json", pairing)
        write_new(output / "all_metrics_recomputed.json", all_scores)
        csv_new(output / "paired_modalities_recomputed.csv", paired_rows)
        csv_new(output / "old171_A_vs_B_paired.csv", retention); csv_new(output / "new3_predictions.csv", new_rows)
        summary = {"status": "PASS_INDEPENDENT_POST_Q_NUMERICAL_AUDIT", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_models_audited": 27, "final_model_metadata_audited": 36, "prediction_artifacts_recomputed": 36,
            "source_epoch_candidates": list(EPOCHS), "all_seeds": list(SEEDS), "per_seed_score_rows": len(all_scores),
            "summary_rows": len(summaries), "paired_modality_rows": len(paired_rows), "paired_old171_rows": len(retention),
            "new3_repeated_prediction_rows": len(new_rows), "independent_new_query_specimens": 3,
            "classifier_fits_by_audit": 0, "Torch_imported_by_audit": bool(replay_models), "PySCF_imported_by_audit": False,
            "native_model_replay": {k: v for k, v in replay.items() if k != "reports"},
            "root_authority_used": True, "closed_stage_CPU_seconds": ledger["closed_child_CPU_seconds"],
            "audit_cpu_seconds": time.process_time(), "audit_wall_seconds": time.monotonic() - started,
            "audit_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "historical_96_0234_direct_comparison": False, "source_code_sha256": sha(__file__)}
        summary["independent_audit_stage_resources"] = budget.record()
        write_new(output / "summary.json", summary)
        return summary


def plan(output=None):
    require(not any(name in sys.modules for name in FORBIDDEN_NUMERICAL_MODULES), "Cold plan must not load numerical packages")
    result = {"status": "NOT_RUN_AUDIT_PREPARED_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "actual_classifier_fits": 0, "actual_predictions_or_arrays_read": 0, "actual_metrics_recomputed": 0,
        "expected_after_healthy_run": {"source_models": 27, "final_models": 36, "query_prediction_artifacts": 36,
            "per_seed_metrics": 54, "paired_modalities": 54, "old171_pairs": 9, "new3_repeated_predictions": 27},
        "requirements": ["actual terminal-Q root authority and original Q global lock", "all six stages healthy with full hash closure",
            "complete source fold/seed/epoch and final task/arm/seed grids", "all prediction IDs/class columns checked before truth join"],
        "data_claim": "No numerical audit or NN experiment has been executed by this preparation",
        "optional_native_replay": {"CLI_flag": "--replay-models", "status": "NOT_RUN", "actual_models_replayed": 0,
            "Torch_import": "only inside explicit actual replay after terminal-Q/stage/hash gates",
            "logits_atol": 1e-5, "logits_rtol": 1e-5,
            "probability_reference": "softmax derived from frozen logits; probability arrays are not stored",
            "default_independent_audit_CPU_cap_seconds": 180, "default_wall_cap_seconds": 600,
            "RSS_best_effort_bytes": 1610612736, "AS_hard_bytes": 10737418240,
            "healthy_closure_file": "audit_stage_closed.json"},
        "source_refs": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in (Path(__file__), ROOT / "scripts/run_supplemental_multirepresentation_20261104.py",
                      ROOT / "scripts/draft_supplemental_multirepresentation_20261104.py",
                      ROOT / "data/research/supplemental_multirepresentation_draft_20261104.json")]}
    if output is not None:
        write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cold = sub.add_parser("plan"); cold.add_argument("--output", type=Path)
    actual = sub.add_parser("audit"); actual.add_argument("--run", type=Path, required=True); actual.add_argument("--output", type=Path, required=True)
    actual.add_argument("--replay-models", action="store_true", help="Actually replay all36 saved weights, without fitting; deferred until after Q")
    actual.add_argument("--cpu-seconds", type=int, default=180, help="Explicit independent audit process CPU cap, 10..300")
    actual.add_argument("--wall-seconds", type=int, default=600, help="Explicit independent audit wall cap, 10..900")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.output)
    else:
        result = audit(args.run.resolve(), args.output.resolve(), replay_models=args.replay_models,
                       cpu_seconds=args.cpu_seconds, wall_seconds=args.wall_seconds)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
