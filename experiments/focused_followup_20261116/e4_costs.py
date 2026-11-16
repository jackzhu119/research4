#!/usr/bin/env python3
"""Audit every archived attempt and conditionally profile twelve cold starts.

Cost gaps, never outer scores, determine the need for supplemental benchmarks.
The original background queue is read only and is never resumed by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sqlite3
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(os.environ.get("FOCUSED_RESULT_ROOT", ROOT / "reports/focused_followup_20261116"))
CACHE = Path(os.environ.get("FOCUSED_SOURCE_ROOT", ROOT / ".cache/focused_followup_20261116"))
BG = CACHE / "background"
METHODS = {"baseline_svm": ("baseline", "svm_alpha_snv"), "baseline_lda": ("baseline", "lda_alpha_raw_snv"),
           "selection": ("selection", "alpha_snv"), "spacing": ("spacing", "alpha_inner_gap_k62_c2")}
EXPECTED_FITS = {"baseline_svm": 121, "baseline_lda": 21, "selection": 4801, "spacing": 481}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def records(path):
    path = Path(path)
    if path.is_file():
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)


def table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def task_receipt(status, extra):
    task = os.environ.get("FOCUSED_TASK_ID", "E4")
    def reference(path):
        try:
            return str(Path(path).relative_to(ROOT))
        except ValueError:
            return str(path)
    if "artifacts" not in extra:
        extra["artifacts"] = [reference(REPORT / "e4" / n) for n in (
            "cost_audit.json", "archived_attempt_costs.csv", "archived_final_representation_costs.csv",
            "mc_cost_common_contexts.csv", "orphan_receipts_not_double_counted.json", "queue_schema.json", "REPORT.md")]
    dump(REPORT / "task_results" / (task + ".json"), {"task_id": task, "module": "E4", "status": status, **extra})


def db_rows():
    path = BG / "snapshot_source/queue.sqlite3"
    if not path.is_file():
        return [], [], []
    # immutable query URI prevents WAL/journal writes into frozen evidence.
    con = sqlite3.connect("file:" + str(path) + "?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    attempts = [dict(r) for r in con.execute("SELECT * FROM attempts")]
    jobs = [dict(r) for r in con.execute("SELECT * FROM jobs")]
    schema = [dict(r) for r in con.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")]
    con.close()
    return attempts, jobs, schema


def audit():
    out = REPORT / "e4"
    attempts, jobs, schema = db_rows()
    byid = {a["id"]: a for a in attempts}
    job_info = {}
    for j in jobs:
        try:
            job_info[j["id"]] = json.loads(j["job_json"])
        except (KeyError, ValueError):
            job_info[j["id"]] = {}
    costs, orphan = {}, []
    for row in records(BG / "all_attempt_receipts.jsonl.gz"):
        r = row["receipt"]
        aid = r.get("attempt_id")
        if not aid and "/attempts/" in row["archive_path"]:
            aid = row["archive_path"].split("/attempts/", 1)[1].split("/", 1)[0]
        if not aid:
            orphan.append({"path": row["archive_path"], "sha256": row["sha256"], "reason": "receipt has no immutable attempt ID; not added to attempt sum"})
            continue
        cpu = float(r.get("cpu_seconds_this_attempt", r.get("process_cpu_seconds", r.get("cpu_seconds", 0))) or 0)
        candidate = {"attempt_id": aid, "job_id": r.get("job_id"), "status": r.get("status"), "cpu_seconds_receipt": cpu,
                     "wall_seconds_receipt": float(r.get("wall_seconds", 0) or 0), "receipt_sha256": row["sha256"],
                     "receipt_path": row["archive_path"], "error": r.get("error"), "exit_code": r.get("exit_code")}
        # A canonical job receipt and its archived attempt copy are one process.
        # Never sum cumulative lineage totals and per-attempt totals together.
        if aid not in costs or cpu > costs[aid]["cpu_seconds_receipt"]:
            costs[aid] = candidate
    fits = defaultdict(lambda: {"classifier": 0, "pls": 0, "source": []})
    for row in records(BG / "all_attempt_heartbeats.jsonl.gz"):
        r = row.get("heartbeat", {})
        aid = r.get("attempt_id")
        if not aid and "/attempts/" in row["archive_path"]:
            aid = row["archive_path"].split("/attempts/", 1)[1].split("/", 1)[0]
        if aid:
            fits[aid]["classifier"] = max(fits[aid]["classifier"], int(r.get("svm_or_reference_fits", 0) or 0))
            fits[aid]["pls"] = max(fits[aid]["pls"], int(r.get("PLS_coefficient_fits", 0) or 0))
            fits[aid]["source"].append(row["archive_path"])
    result_rows = []
    results = {}
    for row in records(BG / "all_result_summaries.jsonl.gz"):
        r = row.get("result", {})
        if r.get("status") != "complete" or not r.get("complete"):
            continue
        jid = r.get("job_id")
        # Keep only small provenance here; millions of archived per-specimen
        # predictions are analyzed by E0 and need not all remain in E4 memory.
        results[jid] = {"job": r.get("job", {})}
        aid = r.get("attempt_id")
        if aid:
            fits[aid]["classifier"] = max(fits[aid]["classifier"], int(r.get("new_classifier_fits_this_attempt", 0) or 0))
            fits[aid]["pls"] = max(fits[aid]["pls"], int(r.get("new_pls_coefficient_fits_this_attempt", 0) or 0))
            fits[aid]["source"].append(row["archive_path"])
        info = r.get("job", job_info.get(jid, {}))
        actual_split_hash = None
        split_path = None
        for name in ("split_frozen.json", "split_frozen.json.gz"):
            candidate = BG / "jobs" / str(jid) / name
            if candidate.is_file():
                split_path = candidate
                break
        if split_path is not None:
            frozen = load(split_path)
            actual_split_hash = hashlib.sha256(json.dumps({"train": frozen["outer_train_ids"], "test": frozen["outer_test_ids"]}, separators=(",", ":")).encode()).hexdigest()
        result_rows.append({"job_id": jid, "attempt_id": aid, "family": r.get("family"), "variant": r.get("variant"),
                            "mc_draws": r.get("mc_draws"), "repeat": info.get("repeat"), "seed": r.get("seed"), "fold": r.get("fold"),
                            "final_dimension": r.get("selected_predictor_count"), "candidate_dimension": r.get("candidate_dimension"),
                            "actual_split_ids_sha256": actual_split_hash,
                            "input_dependency": "all612 alpha points for full-spectrum SNV; selected representation does not prove reduced acquisition",
                            "classifier_fits_completing_attempt": r.get("new_classifier_fits_this_attempt"), "pls_fits_completing_attempt": r.get("new_pls_coefficient_fits_this_attempt")})
    all_ids = sorted(set(byid) | set(costs))
    rows = []
    for aid in all_ids:
        a, r = byid.get(aid, {}), costs.get(aid, {})
        jid = a.get("job_id", r.get("job_id"))
        info = job_info.get(jid, results.get(jid, {}).get("job", {}))
        actual = a.get("verified_actual_cpu")
        row = {"attempt_id": aid, "job_id": jid, "status": a.get("status", r.get("status")), "family": info.get("family"),
               "variant": info.get("variant"), "mc_draws": info.get("mc_draws"), "seed": info.get("seed"), "fold": info.get("fold"),
               "verified_actual_cpu_seconds_ledger": actual, "receipt_cpu_seconds": r.get("cpu_seconds_receipt"),
               "charged_cpu_seconds_ledger": a.get("charged_cpu"), "uncertain_upper_cpu_seconds": a.get("uncertain_upper_cpu"),
               "wall_seconds_receipt": r.get("wall_seconds_receipt"), "exit_code": a.get("exit_code", r.get("exit_code")),
               "exit_signal": a.get("exit_signal"), "final_source": a.get("final_source"), "error": r.get("error"),
               "classifier_fits_observed_lower_bound": fits[aid]["classifier"] if aid in fits else None,
               "pls_fits_observed_lower_bound": fits[aid]["pls"] if aid in fits else None,
               "receipt_path": r.get("receipt_path"), "receipt_sha256": r.get("receipt_sha256")}
        rows.append(row)
    table(out / "archived_attempt_costs.csv", rows, ["attempt_id", "job_id", "status", "family", "variant", "mc_draws", "seed", "fold",
          "verified_actual_cpu_seconds_ledger", "receipt_cpu_seconds", "charged_cpu_seconds_ledger", "uncertain_upper_cpu_seconds", "wall_seconds_receipt",
          "exit_code", "exit_signal", "final_source", "error", "classifier_fits_observed_lower_bound", "pls_fits_observed_lower_bound", "receipt_path", "receipt_sha256"])
    table(out / "archived_final_representation_costs.csv", result_rows, ["job_id", "attempt_id", "family", "variant", "mc_draws", "repeat", "seed", "fold", "final_dimension", "candidate_dimension", "actual_split_ids_sha256", "input_dependency", "classifier_fits_completing_attempt", "pls_fits_completing_attempt"])
    dump(out / "orphan_receipts_not_double_counted.json", orphan)
    dump(out / "queue_schema.json", schema)
    status_counts = Counter(a.get("status", "unknown") for a in rows)
    totals = {
        "archived_attempt_count": len(rows), "ledger_attempt_count": len(attempts), "receipt_attempt_count": len(costs),
        "verified_actual_cpu_seconds_ledger": sum(float(a.get("verified_actual_cpu", 0) or 0) for a in attempts),
        "charged_cpu_seconds_ledger": sum(float(a.get("charged_cpu", 0) or 0) for a in attempts),
        "uncertain_upper_cpu_seconds_ledger": sum(float(a.get("uncertain_upper_cpu", 0) or 0) for a in attempts),
        "observed_active_cpu_seconds_ledger": sum(float(a.get("observed_cpu", 0) or 0) for a in attempts if a.get("status") in ("running", "launching")),
        "receipt_cpu_seconds_sum_unique_attempts": sum(r["cpu_seconds_receipt"] for r in costs.values()),
        "receipt_wall_seconds_sum_unique_attempts": sum(r["wall_seconds_receipt"] for r in costs.values()),
        "classifier_fits_observed_lower_bound": sum(r["classifier"] for r in fits.values()),
        "pls_fits_observed_lower_bound": sum(r["pls"] for r in fits.values()),
        "status_counts": dict(status_counts), "job_status_counts": dict(Counter(j.get("status", "unknown") for j in jobs)),
        "jobs_with_retries": sum(int(j.get("attempt_count", 0) or 0) > 1 for j in jobs),
        "retry_attempts": sum(max(0, int(j.get("attempt_count", 0) or 0) - 1) for j in jobs),
        "receipt_status_cpu_seconds": dict((s, sum(r["cpu_seconds_receipt"] for r in costs.values() if r.get("status") == s)) for s in {r.get("status") for r in costs.values()}),
        "no_lineage_double_count": "Only one receipt per immutable attempt ID; cpu_seconds_lineage is never summed. Ledger conservative charges and actual CPU are separate columns.",
        "wall_time_scope": "sum of process wall times, not parallel study elapsed time; competing-resource wall time is not an algorithm speed comparison",
    }
    # Pair only exact repeated split contexts for MC20/100/500.  These are CPU
    # descriptions, not a claim that hardware/cache conditions match.
    keyed = defaultdict(dict)
    for row in result_rows:
        if row["family"] and row["attempt_id"] in costs:
            key = (row["family"], row["variant"], row["seed"], row["fold"])
            keyed[key][int(row["mc_draws"] or 0)] = {**costs[row["attempt_id"]], "actual_split_ids_sha256": row["actual_split_ids_sha256"]}
    pairs = []
    for key, levels in keyed.items():
        if len(set(levels) & {20, 100, 500}) > 1:
            hashes = {v["actual_split_ids_sha256"] for v in levels.values()}
            verified_split = len(hashes) == 1 and None not in hashes
            pairs.append({"family": key[0], "variant": key[1], "seed": key[2], "fold": key[3],
                          "mc_levels": sorted(levels), "cpu_by_mc": {str(k): v["cpu_seconds_receipt"] for k, v in levels.items()},
                          "actual_split_ids_verified_equal": verified_split,
                          "comparability": "actual archived ID lists match; cache/resume/hardware equivalence unknown; no speedup claim" if verified_split else "seed/fold coincidence only; missing/different actual ID evidence; not a verified paired cost comparison"})
    table(out / "mc_cost_common_contexts.csv", pairs, ["family", "variant", "seed", "fold", "mc_levels", "cpu_by_mc", "actual_split_ids_verified_equal", "comparability"])
    # Original worker emits total invocation CPU and counters but not complete
    # independent stage durations/memory/model byte measurements.  Supplemental
    # profiling fills this gap; it cannot retroactively time failed old fits.
    gaps = ["old attempts lack exhaustive exclusive stage CPU/wall measurements", "old attempts do not all record peak RSS and serialized pipeline bytes",
            "fit counters are completing-attempt or last-heartbeat lower bounds when failure happened before final receipt"]
    benchmarks = []
    for p in sorted((out / "benchmarks").glob("*/cost_receipt.json")):
        benchmarks.append(load(p))
    dump(out / "cost_audit.json", {"status": "COMPLETED" if len(benchmarks) == 12 else "PARTIAL_BUDGET" if benchmarks else "BLOCKED_MISSING_ARTIFACT",
        "archived_totals": totals, "benchmark_required": True, "reason": gaps, "benchmark_runs_completed": len(benchmarks),
        "benchmark_matrix": {"contexts": [0, 1, 2], "methods": list(METHODS), "theoretical_classifier_fits_per_context": EXPECTED_FITS,
                              "max_theoretical_classifier_fits": 16272, "estimated_pls_fits": 18600},
        "benchmark_scope": "new sequential single-worker MC20 cold starts or explicitly labeled resume; fills measurement gaps, not replacement archived performance",
        "benchmark_receipts": benchmarks, "input_dependency": "SNV uses all612 alpha inputs per spectrum even if classifier keeps62 features",
        "acquisition_time_reduction_inferred": False, "historical_stage_costs_reconstructed": False})
    (out / "REPORT.md").write_text("# E4: Computational cost evidence\n\n"
        "Archived immutable attempt IDs are the accounting unit. Per-attempt receipts and supervisor ledger totals are reported separately; cumulative lineage fields are never added again. Failed, interrupted and retried attempts remain in the table. Process wall-time sums differ from concurrent study elapsed time. MC20/100/500 rows pair archived seed/fold contexts but do not claim identical cache or hardware conditions.\n\n"
        "The original worker records total CPU and fit counters; it does not exhaustively measure exclusive stage durations, peak RSS and model bytes. The fixed three-context/four-method supplemental benchmark is therefore required when remaining budget permits. New profiles distinguish cold start from actual resume and cannot reconstruct missing historical stage times. No score determines this decision.\n\n"
        "Rowwise SNV depends on all612 alpha points. A final62-dimensional classifier is not evidence for a62-point acquisition protocol or a proportional reduction in instrument time.\n", encoding="utf-8")
    task_receipt("COMPLETED", {"scope": "archived cost audit and conditional benchmark decision", "benchmark_required": True,
                              "new_classifier_fits": 0, "new_pls_fits": 0, "summary": "e4/cost_audit.json"})
    return totals


def frozen_context(index):
    path = REPORT / "e0/common_contexts.json"
    if not path.is_file():
        raise FileNotFoundError("E0 common-context freeze is required: " + str(path))
    manifest = load(path)
    contexts = [c for c in manifest["contexts"] if int(c["fold"]) == 0]
    # Manifest order is already frozen by actual minimum source ordinal.
    if len(contexts) < 3:
        raise ValueError("Fewer than three frozen common complete fold0 contexts")
    return contexts[index], digest(path)


def archived_split(context):
    jid = context["methods"]["selection/alpha_snv"]
    if isinstance(jid, dict):
        jid = jid.get("job_id", jid.get("id"))
    for suffix in ["json", "json.gz"]:
        path = BG / "jobs" / str(jid) / ("split_frozen." + suffix)
        if path.is_file():
            return load(path), digest(path)
    raise FileNotFoundError("Missing actual archived benchmark split " + str(jid))


def benchmark(index, method):
    # Include source verification, data hashing and dynamic scientific-library
    # imports in a cold start.  The whole process is separately charged by the
    # experiment ledger, including interpreter/module imports before this call.
    cpu0, wall0 = time.process_time(), time.monotonic()
    audit_path = REPORT / "e4/cost_audit.json"
    if not audit_path.is_file():
        audit()
    if not load(audit_path).get("benchmark_required"):
        task_receipt("REUSED_VERIFIED", {"reason": "existing cost evidence sufficient", "new_classifier_fits": 0, "new_pls_fits": 0})
        return
    context, context_hash = frozen_context(index)
    split, split_hash = archived_split(context)
    ident = "context" + str(index) + "_" + method
    out = REPORT / "e4/benchmarks" / ident
    if (out / "cost_receipt.json").is_file():
        old = load(out / "cost_receipt.json")
        if old["context_manifest_sha256"] != context_hash or old["split_sha256"] != split_hash:
            raise ValueError("Completed benchmark source context changed")
        task_receipt("REUSED_VERIFIED", {"reason": "same benchmark identity complete", "new_classifier_fits": 0, "new_pls_fits": 0})
        return
    out.mkdir(parents=True, exist_ok=True)
    source = BG / "snapshot_source/study_worker.py"
    if not source.is_file():
        # Recovery keeps pinned source small files in a separate source view on
        # some transports; require exact worker, never substitute newer code.
        source = BG / "source/study_worker.py"
    expected_source = load(BG / "snapshot_source/protocol.json").get("sealed_files", {}).get("study_worker.py")
    if expected_source and digest(source) != expected_source:
        raise ValueError("Pinned worker hash does not match frozen protocol")
    spec = importlib.util.spec_from_file_location("focused_e4_frozen_worker", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    import sklearn
    from threadpoolctl import threadpool_info, threadpool_limits
    import joblib
    phases = defaultdict(lambda: [0.0, 0.0, 0])
    events = []
    stage = {"value": "initialization"}
    def measured(label, fn, *args, **kwargs):
        c, w = time.process_time(), time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            cpu, wall = time.process_time() - c, time.monotonic() - w
            key = stage["value"] + "/" + label
            phases[key][0] += cpu
            phases[key][1] += wall
            phases[key][2] += 1
            events.append({"stage": stage["value"], "operation": label, "cpu_seconds": cpu, "wall_seconds": wall})
    # Original source's predict combines fit and inference; instrument estimator
    # method calls to keep those exclusive phases distinguishable.
    original_svc, original_lda = mod.SVC, mod.LinearDiscriminantAnalysis
    class ProfileSVC(original_svc):
        def fit(self, *a, **kw): return measured("classifier_fit", super().fit, *a, **kw)
        def predict(self, *a, **kw): return measured("inference_predict", super().predict, *a, **kw)
        def decision_function(self, *a, **kw): return measured("inference_scores", super().decision_function, *a, **kw)
    class ProfileLDA(original_lda):
        def fit(self, *a, **kw): return measured("classifier_fit", super().fit, *a, **kw)
        def predict(self, *a, **kw):
            self._profiling_inside_predict = True
            try:
                return measured("inference_predict", super().predict, *a, **kw)
            finally:
                self._profiling_inside_predict = False
        def decision_function(self, *a, **kw):
            # sklearn LDA.predict internally invokes decision_function.  That
            # nested work is already inside predict, so count it only once.
            if getattr(self, "_profiling_inside_predict", False):
                return super().decision_function(*a, **kw)
            return measured("inference_scores", super().decision_function, *a, **kw)
    mod.SVC, mod.LinearDiscriminantAnalysis = ProfileSVC, ProfileLDA
    class ProfileWorker(mod.Worker):
        def load_data(self):
            measured("load_and_split_validation", super().load_data)
            ids = {str(s): i for i, s in enumerate(self.ids)}
            frozen_train = np.array([ids[s] for s in split["outer_train_ids"]], dtype=int)
            frozen_test = np.array([ids[s] for s in split["outer_test_ids"]], dtype=int)
            if self.ids[self.train].tolist() != split["outer_train_ids"] or self.ids[self.test].tolist() != split["outer_test_ids"]:
                raise ValueError("Frozen actual outer IDs do not replay")
            inner = []
            for original, frozen in zip(self.inner, split["inner"]):
                if self.ids[original[0]].tolist() != frozen["fit_ids"] or self.ids[original[1]].tolist() != frozen["validation_ids"]:
                    raise ValueError("Frozen actual inner IDs do not replay")
                inner.append((np.array([ids[s] for s in frozen["fit_ids"]], int), np.array([ids[s] for s in frozen["validation_ids"]], int)))
            if len(inner) != 10 or set(frozen_train) & set(frozen_test):
                raise ValueError("Benchmark isolation/inner-fold count invalid")
            self.train, self.test, self.inner = frozen_train, frozen_test, inner
        def transformed(self, *a, **kw):
            stage["value"] = str(a[4] if len(a) > 4 else kw["stage"])
            return measured("transform_inclusive_of_ranking", super().transformed, *a, **kw)
        def rank(self, *a, **kw):
            return measured("ranking", super().rank, *a, **kw)
        def predict(self, *a, **kw):
            result = super().predict(*a, **kw)
            if stage["value"] == "outer":
                self._profile_model = result[2]
                self._profile_info = a[-1]
            return result
    family, variant = METHODS[method]
    input_path = BG / "inputs/current1672_paired612.npz"
    job = {"id": "focused-e4-" + ident, "ordinal": index * 4 + list(METHODS).index(method), "family": family, "variant": variant,
           "seed": int(context["seed"]), "fold": 0, "mc_draws": 20, "task": "core300", "data_path": str(input_path), "data_sha256": digest(input_path)}
    prior_attempts = list(out.glob("profile_attempt_*.json"))
    execution_mode = "resume_with_verified_checkpoint" if any((out / n).exists() for n in ("checkpoint.json", "rank_cache", "inner_scores")) else "cold_start"
    aid = uuid.uuid4().hex
    worker, error = None, None
    try:
        worker = ProfileWorker(job, out, None, None, int(float(os.environ.get("FOCUSED_CPU_REMAINING_SECONDS", 990000))))
        with threadpool_limits(limits=1):
            result = worker.execute()
        model_path = out / "serialized_pipeline.joblib"
        # Serialize a standard sklearn estimator rather than a local profiling
        # subclass.  Learned state is copied exactly after all predictions.
        original_class = original_svc if method != "baseline_lda" else original_lda
        serial_model = original_class.__new__(original_class)
        serial_model.__dict__.update(worker._profile_model.__dict__)
        serial_model.__dict__.pop("_profiling_inside_predict", None)
        joblib.dump({"model": serial_model, "selected_global_indices": worker._profile_info["selected_global_indices"],
                     "scaler_mean": worker._profile_info["scaler_mean"], "scaler_scale": worker._profile_info["scaler_scale"], "rowwise_snv_ddof": 1}, model_path)
        used_cpu, used_wall = time.process_time() - cpu0, time.monotonic() - wall0
        exclusive = []
        for key, (cpu, wall, calls) in phases.items():
            name, operation = key.rsplit("/", 1)
            if operation == "transform_inclusive_of_ranking":
                nested = phases.get(name + "/ranking", [0, 0, 0])
                exclusive.append({"stage": name, "operation": "preprocessing_scaling_selection_excluding_ranking", "cpu_seconds": max(0, cpu - nested[0]), "wall_seconds": max(0, wall - nested[1]), "calls": calls})
            else:
                exclusive.append({"stage": name, "operation": operation, "cpu_seconds": cpu, "wall_seconds": wall, "calls": calls})
        table(out / "exclusive_stage_timings.csv", exclusive)
        measured_cpu = sum(r["cpu_seconds"] for r in exclusive)
        cost = {"status": "COMPLETED", "benchmark_id": ident, "context_index": index, "method": method,
            "context_manifest_sha256": context_hash, "split_sha256": split_hash, "input_sha256": digest(input_path), "source_code_sha256": digest(source),
            "execution_mode": execution_mode, "prior_attempt_count": len(prior_attempts), "cpu_seconds_this_attempt": used_cpu,
            "wall_seconds_this_attempt": used_wall, "new_classifier_fits_this_attempt": worker.nfits, "new_pls_fits_this_attempt": worker.rank_fits,
            "theoretical_classifier_fits_without_dedup": EXPECTED_FITS[method], "fit_count_limit": "actual cache/signature dedup may reduce counts; counters are actual new fits",
            "search_and_io_overhead_cpu_seconds": max(0, used_cpu - measured_cpu), "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "serialized_pipeline_bytes": model_path.stat().st_size, "serialized_pipeline_sha256": digest(model_path),
            "final_dimension": result["selected_predictor_count"], "input_dependency_points": 612,
            "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count()},
            "software": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__}, "threads": threadpool_info(),
            "thread_configuration": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
            "score_used_for_continuation": False}
        dump(out / "cost_receipt.json", cost)
        task_receipt("COMPLETED", {"new_classifier_fits": worker.nfits, "new_pls_fits": worker.rank_fits,
                                   "summary": str((out / "cost_receipt.json").relative_to(REPORT)), "execution_mode": execution_mode,
                                   "artifacts": [str(out.relative_to(ROOT))]})
    except Exception as exc:
        error = {"error": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        dump(out / ("profile_attempt_" + aid + ".json"), {"attempt_id": aid, "benchmark_id": ident,
             "status": "failed" if error else "complete", "cpu_seconds_this_attempt": time.process_time() - cpu0,
             "wall_seconds_this_attempt": time.monotonic() - wall0, "new_classifier_fits": worker.nfits if worker else 0,
             "new_pls_fits": worker.rank_fits if worker else 0, "execution_mode": execution_mode, "error": error,
             "events": events})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-context", type=int, choices=range(3))
    parser.add_argument("--method", choices=list(METHODS))
    args = parser.parse_args(argv)
    if args.benchmark_context is not None:
        if args.method is None:
            parser.error("--method required with --benchmark-context")
        benchmark(args.benchmark_context, args.method)
    else:
        print(json.dumps(audit(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
