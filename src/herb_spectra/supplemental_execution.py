"""Staged execution for a checksummed, separately prepared supplemental protocol.

Only explicit release files unlock source fitting and query prediction. Query
truth is opened exclusively by ``score``. This is an auditable API boundary,
not an operating-system sandbox against arbitrary code.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time
import traceback
import warnings

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from .leakage_guard import FitScope, LeakageError
from .paper_c_calibration import candidate_grid as calibration_grid, choose_development
from .supplemental_benchmark import (
    calibrate_scores, file_sha256, fit_source_model, load_role_pack, object_sha256,
    pack_labels, predict_numeric, preflight_role_packs, score_predictions, write_new_json,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, settings, *, prior_cpu=0., prior_wall=0., reserve_for_evaluation=False):
        self.cpu_limit = float(settings["cpu_seconds"])
        self.wall_limit = float(settings["wall_seconds"])
        self.rss_limit = float(settings["rss_bytes"])
        self.cpu_gate = self.cpu_limit - (float(settings.get("evaluation_cpu_reserve_seconds", 120)) if reserve_for_evaluation else 0)
        self.wall_gate = self.wall_limit - (float(settings.get("evaluation_wall_reserve_seconds", 120)) if reserve_for_evaluation else 0)
        self.prior_cpu, self.prior_wall = prior_cpu, prior_wall
        self.wall_start = time.monotonic()
        self.cpu_start = 0.  # Include this process's import/startup CPU time.
        self.completed_fits = 0

    def record(self):
        return {"cpu_seconds_this_process": time.process_time() - self.cpu_start,
                "cpu_seconds_total": self.prior_cpu + time.process_time() - self.cpu_start,
                "wall_seconds_this_process": time.monotonic() - self.wall_start,
                "wall_seconds_total": self.prior_wall + time.monotonic() - self.wall_start,
                "peak_rss_bytes_this_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "completed_source_model_fits": self.completed_fits,
                "limits": {"cpu_seconds": self.cpu_limit, "wall_seconds": self.wall_limit,
                           "rss_bytes": self.rss_limit, "threads": 1},
                "stage_stop": {"cpu_seconds": self.cpu_gate, "wall_seconds": self.wall_gate},
                "rss_guard": "best_effort_before_and_after_fit; external resource monitor recommended"}

    def check(self):
        current = self.record()
        if current["cpu_seconds_total"] >= self.cpu_gate:
            raise BudgetExceeded("Registered cumulative CPU budget exhausted")
        if current["wall_seconds_total"] >= self.wall_gate:
            raise BudgetExceeded("Registered cumulative wall-time budget exhausted")
        if current["peak_rss_bytes_this_process"] > self.rss_limit:
            raise BudgetExceeded("Observed RSS exceeded registered best-effort limit")


def repo_path(repo, value):
    root = Path(repo).resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise LeakageError("Protocol paths must remain in this independent worktree")
    return path


def verify_ref(repo, ref):
    path = repo_path(repo, ref["path"])
    if file_sha256(path) != ref["sha256"]:
        raise LeakageError("Frozen file bytes changed: " + ref["path"])
    return path


def code_identity(repo, protocol):
    refs = protocol["code_files"]
    if isinstance(refs, dict):
        refs = [{"path": p, "sha256": h} for p, h in sorted(refs.items())]
    if not refs:
        raise LeakageError("A source-code hash inventory is required")
    for ref in refs:
        verify_ref(repo, ref)
    return object_sha256(refs)


def load_protocol(repo, protocol_path):
    path = repo_path(repo, protocol_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema", "dataset_version", "seeds", "candidates", "calibration", "resources", "code_files", "tasks"):
        if key not in protocol:
            raise LeakageError("Bound protocol lacks " + key)
    if protocol["seeds"] != [42, 43, 44]:
        raise LeakageError("All three predeclared seeds 42/43/44 are required")
    code = code_identity(repo, protocol)
    verify_ref(repo, protocol["intake_manifest"])
    conservative = protocol["calibration"].get("conservative", {})
    if "candidate_grid" in conservative and conservative["candidate_grid"] != calibration_grid():
        raise LeakageError("Registered conservative grid differs from the reused implementation")
    if conservative.get("source_episode_model_seed", 42) != 42:
        raise LeakageError("Source calibration uses the explicitly fixed model seed 42")
    return protocol, file_sha256(path), code


def require_release(repo, release_path, protocol_hash, stage, *, source_lock_hash=None):
    if release_path is None:
        raise LeakageError("Root-reviewed release file is required before execution")
    release = json.loads(repo_path(repo, release_path).read_text(encoding="utf-8"))
    if release.get("protocol_sha256") != protocol_hash or stage not in release.get("allowed_stages", []):
        raise LeakageError("Release does not unlock this frozen protocol/stage")
    if source_lock_hash is not None and release.get("source_lock_sha256") != source_lock_hash:
        raise LeakageError("Query release must identify the complete source-model lock")
    if release.get("data_ready") is not True:
        raise LeakageError("Release must explicitly record root's data-READY review")
    return release


def role_packs(repo, task):
    packs = {}
    for role in ("source", "support", "query"):
        ref = task[role + "_pack"]
        path = verify_ref(repo, ref)
        packs[role] = load_role_pack(path, role=role,
            expected_sha256=ref["sha256"], expected_axis_sha256=task.get("axis_sha256"))
    return packs


def indices_for(ids, requested):
    ids = np.asarray(ids).astype(str)
    requested = [str(v) for v in requested]
    lookup = {v: i for i, v in enumerate(ids)}
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(lookup):
        raise LeakageError("Requested IDs are missing, duplicated, or outside their role pack")
    return np.array([lookup[v] for v in requested], dtype=int)


def preflight(repo, protocol_path):
    protocol, protocol_hash, code = load_protocol(repo, protocol_path)
    summaries = {}
    for task in protocol["tasks"]:
        packs = role_packs(repo, task)
        summary = preflight_role_packs(packs)
        if summary["classes"] != task["classes"]:
            raise LeakageError("Task class vocabulary changed after split lock")
        for role, expected in task.get("expected_roles", {}).items():
            if len(packs[role]["X"]) != int(expected):
                raise LeakageError("Prepared role count differs from the registered task")
        source_ids = packs["source"]["record_ids"]
        source_labels = pack_labels(packs["source"])
        validation_ids = []
        for fold in task.get("inner_folds", []):
            train = indices_for(source_ids, fold["train_ids"])
            valid = indices_for(source_ids, fold["valid_ids"])
            if set(train) & set(valid) or set(train) | set(valid) != set(range(len(source_ids))):
                raise LeakageError("Inner fold does not partition this source pack")
            if set(source_labels[train]) != set(task["classes"]):
                raise LeakageError("Inner source fold loses a declared class")
            if set(source_labels[valid]) != set(task["classes"]):
                raise LeakageError("Inner validation fold loses a declared class")
            validation_ids.extend(fold["valid_ids"])
        if task.get("inner_folds") and (len(validation_ids) != len(source_ids) or set(validation_ids) != set(source_ids)):
            raise LeakageError("OOF source validation does not cover each source entity once")
        if not task.get("inherit_selection_from") and not task.get("inner_folds"):
            raise LeakageError("A non-inherited task requires frozen source-only folds")
        for episode in task.get("source_calibration_episodes", []):
            base, support, query = (indices_for(source_ids, episode[k])
                                    for k in ("base_ids", "support_ids", "query_ids"))
            if (set(base) & set(support) or set(base) & set(query) or set(support) & set(query)
                    or set(base) | set(support) | set(query) != set(range(len(source_ids)))):
                raise LeakageError("Source calibration pseudoroles must partition only this source pack")
            if any(set(source_labels[idx]) != set(task["classes"]) for idx in (base, support, query)):
                raise LeakageError("Every source calibration pseudorole must cover the task vocabulary")
            for shot, requested in episode["shots"].items():
                subset = indices_for(source_ids[support], requested)
                if any(np.sum(source_labels[support][subset] == c) != int(shot) for c in task["classes"]):
                    raise LeakageError("Source calibration shots do not match independent support counts")
        support_ids, support_labels = packs["support"]["record_ids"], pack_labels(packs["support"])
        for draw in task.get("calibration_draws", []):
            for shot, requested in draw["shots"].items():
                subset = indices_for(support_ids, requested)
                if any(np.sum(support_labels[subset] == c) != int(shot) for c in task["classes"]):
                    raise LeakageError("Target draw shot count differs from available independent support")
        summaries[task["id"]] = summary
    return {"status": "PREFLIGHT_ONLY_NO_MODEL_FIT", "protocol_sha256": protocol_hash,
            "code_version": code, "tasks": summaries}


def environment():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        versions[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "threads": 1, "gpu": "NOT_USED", "paid_compute": "NOT_USED"}


def compressed_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)


def fit_logged(out, task, pack, indices, config, seed, fit_id, protocol, code, budget, cache):
    budget.check()
    directory = out / "fits" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / (fit_id + ".jsonl.partial")
    packed_trace = directory / (fit_id + ".jsonl.gz")
    if trace.exists() or packed_trace.exists():
        raise LeakageError("Fit evidence already exists; historical traces cannot be replaced")
    start_cpu, start_wall, start_utc = time.process_time(), time.monotonic(), utc()
    try:
        model = fit_source_model(config, pack["X"][indices], pack_labels(pack)[indices],
            pack["axis"], pack["record_ids"][indices], seed=seed,
            dataset_version=protocol["dataset_version"], split_id=task["id"] + "/" + fit_id.rsplit("_candidate", 1)[0],
            log_path=trace, code_version=code, physical_quantity=task["quantity"], cache=cache)
        budget.completed_fits += 1
        model.metadata.update(configuration_sha256=object_sha256(config), started_utc=start_utc, ended_utc=utc(),
            actual_process_cpu_seconds=time.process_time() - start_cpu,
            actual_wall_seconds=time.monotonic() - start_wall,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
        compressed_json(directory / (fit_id + ".metadata.json.gz"), model.metadata)
        return model
    finally:
        if trace.exists():
            with trace.open("rb") as source, gzip.open(packed_trace, "xb") as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
            trace.unlink()  # Only the just-created temporary trace, losslessly archived above.
        budget.check()


def candidate_configs(protocol):
    configs = [c for group in protocol["candidates"].values() for c in group]
    identifiers = [c["id"] for c in configs]
    if not configs or len(set(identifiers)) != len(identifiers):
        raise LeakageError("Registered model candidates need unique IDs")
    return configs


def source_selection(repo, task, protocol, out, code, budget):
    source = load_role_pack(verify_ref(repo, task["source_pack"]), role="source")
    y, classes = pack_labels(source), np.array(task["classes"])
    configs = candidate_configs(protocol)
    pooled, all_metrics = {}, []
    for seed in protocol["seeds"]:
        records = {c["id"]: {"record_ids": source["record_ids"].copy(), "classes": classes.copy(),
                    "predictions": np.full(len(y), "", dtype=source["y" if "y" in source else "labels"].dtype),
                    "scores": np.full((len(y), len(classes)), np.nan)} for c in configs}
        for fold_no, fold in enumerate(task["inner_folds"]):
            train, valid = (indices_for(source["record_ids"], fold[k]) for k in ("train_ids", "valid_ids"))
            cache = {}
            for config in configs:
                fit_id = f"seed{seed}_fold{fold_no}_candidate{config['id']}"
                model = fit_logged(out, task, source, train, config, seed, fit_id, protocol, code, budget, cache)
                prediction = predict_numeric(model, source["X"][valid], source["record_ids"][valid])
                if not np.array_equal(prediction["classes"], classes):
                    raise LeakageError("Source fold changes the fixed class order")
                records[config["id"]]["predictions"][valid] = prediction["predictions"]
                records[config["id"]]["scores"][valid] = prediction["scores"]
        for config in configs:
            result = records[config["id"]]
            if np.any(result["predictions"] == "") or not np.isfinite(result["scores"]).all():
                raise LeakageError("A source candidate has incomplete OOF predictions")
            metrics = score_predictions(source["record_ids"], y, result)
            row = {"candidate_id": config["id"], "family": config["family"], "seed": seed, **metrics}
            all_metrics.append(row)
            path = out / "selection" / task["id"] / f"{config['id']}_seed{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **result)
            pooled[(config["id"], seed)] = {"path": str(path.relative_to(out)), "sha256": file_sha256(path)}
        budget.check()
    means = {}
    for config in configs:
        values = [r for r in all_metrics if r["candidate_id"] == config["id"]]
        if len(values) != len(protocol["seeds"]):
            raise LeakageError("Candidate selection requires all registered seeds")
        means[config["id"]] = {m: float(np.mean([r[m] for r in values]))
                               for m in ("macro_f1", "accuracy", "balanced_accuracy")}
    chosen = {}
    for family, candidates in protocol["candidates"].items():
        ranking = sorted(range(len(candidates)), key=lambda i: (
            -means[candidates[i]["id"]]["macro_f1"], -means[candidates[i]["id"]]["accuracy"],
            -means[candidates[i]["id"]]["balanced_accuracy"], i))
        chosen[family] = candidates[ranking[0]]
    summary = {"task": task["id"], "status": "DONE_SOURCE_ONLY", "selected": chosen,
               "means": means, "per_seed_metrics": all_metrics,
               "selection_rule": "mean pooled-OOF Macro-F1; Accuracy; Balanced Accuracy; frozen candidate order",
               "query_labels_read": False, "query_predictions_generated": False}
    write_new_json(out / "selection" / task["id"] / "summary.json", summary)
    return summary


def source_calibration(repo, task, config, protocol, out, code, budget):
    settings = task.get("calibration", {})
    if settings.get("conservative_policy") == "global_no_op":
        return {"fallback_to_identity": True, "selected": None, "status": "NOT_RUN",
                "reason": settings["reason"]}
    episodes = task.get("source_calibration_episodes", [])
    if not episodes:
        return {"fallback_to_identity": True, "selected": None, "status": "NOT_RUN",
                "reason": "no_registered_source_calibration_episodes"}
    source = load_role_pack(verify_ref(repo, task["source_pack"]), role="source")
    rows = []
    for episode_no, episode in enumerate(episodes):
        base, support, query = (indices_for(source["record_ids"], episode[k])
                                for k in ("base_ids", "support_ids", "query_ids"))
        if set(base) & set(support) or set(base) & set(query) or set(support) & set(query):
            raise LeakageError("Source calibration pseudoroles overlap")
        model = fit_logged(out, task, source, base, config, 42, f"pseudo{episode_no}_candidate{config['id']}",
                           protocol, code, budget, {})
        reference = model.scores(source["X"][base])
        support_pred = predict_numeric(model, source["X"][support], source["record_ids"][support])
        query_pred = predict_numeric(model, source["X"][query], source["record_ids"][query])
        native_metrics = score_predictions(source["record_ids"][query], pack_labels(source)[query], query_pred)
        for shot, requested in episode["shots"].items():
            subset = indices_for(source["record_ids"][support], requested)
            support_ids = source["record_ids"][support][subset]
            for candidate in calibration_grid():
                budget.check()
                fit_scope = FitScope(support_ids, f"{task['id']}/pseudo{episode_no}/K{shot}/{candidate['id']}",
                    protocol["dataset_version"], code_version=code)
                calibration_cpu, calibration_wall = time.process_time(), time.monotonic()
                adapted = calibrate_scores("conservative", reference, model.classes_, query_pred["scores"],
                    query_pred["predictions"], support_scores=support_pred["scores"][subset],
                    support_labels=pack_labels(source)[support][subset],
                    support_native=support_pred["predictions"][subset], limited_candidate=candidate,
                    fit_scope=fit_scope, support_ids=support_ids)
                prediction = {**query_pred, "predictions": adapted["predictions"]}
                metrics = score_predictions(source["record_ids"][query], pack_labels(source)[query], prediction,
                                             baseline=query_pred["predictions"])
                rows.append({"stage": episode["stage"], "episode": episode_no, "shot": int(shot),
                    "candidate_id": candidate["id"], **{m: metrics[m] for m in ("accuracy", "macro_f1", "balanced_accuracy")},
                    **{m + "_gain": metrics[m] - native_metrics[m] for m in ("accuracy", "macro_f1", "balanced_accuracy")},
                    "corrected": metrics["corrected_errors"], "harmed": metrics["introduced_errors"],
                    "adapter_status": adapted["status"], "adapter_reason": adapted["reason"],
                    "candidate": candidate, "configuration_sha256": object_sha256(candidate),
                    "actual_process_cpu_seconds": time.process_time() - calibration_cpu,
                    "actual_wall_seconds": time.monotonic() - calibration_wall,
                    "fit_events": fit_scope.entries})
    if {r["stage"] for r in rows} != {"selection", "gate"}:
        raise LeakageError("Conservative calibration needs both predeclared source selection and gate episodes")
    policy = {**choose_development(rows, calibration_grid()), "status": "DONE_SOURCE_ONLY", "evaluations": len(rows)}
    directory = out / "calibration_selection" / task["id"]
    compressed_json(directory / "source_episode_results.json.gz", rows)
    write_new_json(directory / "policy.json", policy)
    return policy


def run_source(repo, protocol_path, output, release_path):
    protocol, protocol_hash, code = load_protocol(repo, protocol_path)
    release = require_release(repo, release_path, protocol_hash, "source")
    preflight_result = preflight(repo, protocol_path)
    out = repo_path(repo, output)
    if Path(repo).resolve() / "reports" not in out.parents:
        raise LeakageError("New run output must be an independent reports subdirectory")
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(protocol["resources"], reserve_for_evaluation=True)
    write_new_json(out / "protocol_snapshot.json", protocol)
    write_new_json(out / "release_snapshot.json", release)
    write_new_json(out / "preflight.json", preflight_result)
    write_new_json(out / "environment.json", environment())
    started = utc()
    summaries, models, policies, completed_tasks = {}, [], {}, []
    try:
        with threadpool_limits(limits=1):
            for task in protocol["tasks"]:
                parent = task.get("inherit_selection_from")
                if parent:
                    if parent not in summaries:
                        raise LeakageError("Inherited source task must be completely selected first")
                    summaries[task["id"]] = {"status": "INHERITED_SOURCE_CONFIGURATION", "parent": parent,
                                              "selected": summaries[parent]["selected"]}
                else:
                    summaries[task["id"]] = source_selection(repo, task, protocol, out, code, budget)
                source = load_role_pack(verify_ref(repo, task["source_pack"]), role="source")
                for family, config in summaries[task["id"]]["selected"].items():
                    for seed in protocol["seeds"]:
                        fit_id = f"final_seed{seed}_candidate{config['id']}"
                        model = fit_logged(out, task, source, np.arange(len(source["X"])), config, seed,
                                           fit_id, protocol, code, budget, {})
                        directory = out / "source_models" / task["id"] / family / str(seed)
                        directory.mkdir(parents=True, exist_ok=False)
                        path = directory / "model.joblib"
                        joblib.dump(model, path, compress=3)
                        reference_path = directory / "source_reference.npz"
                        np.savez_compressed(reference_path, scores=model.scores(source["X"]),
                                            record_ids=source["record_ids"], classes=model.classes_)
                        models.append({"task": task["id"], "family": family, "seed": seed, "config": config,
                            "path": str(path.relative_to(out)), "sha256": file_sha256(path),
                            "source_reference": {"path": str(reference_path.relative_to(out)), "sha256": file_sha256(reference_path)},
                            "source_pack_sha256": task["source_pack"]["sha256"],
                            "selected_features": model.metadata["selected_features"],
                            "source_fit_seconds": model.metadata["fit_total_seconds"]})
                policies[task["id"]] = source_calibration(repo, task,
                    summaries[task["id"]]["selected"]["mcuve_svm"], protocol, out, code, budget)
                completed_tasks.append(task["id"])
                budget.check()
        lock = {"status": "DONE_SOURCE_LOCKED_NO_QUERY_PREDICTIONS", "protocol_sha256": protocol_hash,
                "code_version": code, "started_utc": started, "ended_utc": utc(),
                "models": models, "selection": summaries, "conservative_policies": policies,
                "completed_task_ids": completed_tasks,
                "task_status": {t["id"]: "DONE" for t in protocol["tasks"]},
                "cost": budget.record(), "query_truth_read": False}
        write_new_json(out / "source_lock.json", lock)
        return {"status": lock["status"], "source_lock_sha256": file_sha256(out / "source_lock.json"), "cost": lock["cost"]}
    except BudgetExceeded as exc:
        complete_models = [m for m in models if m["task"] in completed_tasks]
        partial_models = [m for m in models if m["task"] not in completed_tasks]
        resource_lock = {"status": "PARTIAL_RESOURCE_LIMIT_COMPLETE_TASKS_ONLY", "protocol_sha256": protocol_hash,
            "code_version": code, "started_utc": started, "ended_utc": utc(), "models": complete_models,
            "partial_models_not_evaluable": partial_models, "completed_task_ids": completed_tasks,
            "task_status": {t["id"]: ("DONE" if t["id"] in completed_tasks else "NOT_RUN_RESOURCE_LIMIT")
                            for t in protocol["tasks"]},
            "selection": summaries, "conservative_policies": policies,
            "resource_stop_reason": str(exc), "cost": budget.record(), "query_truth_read": False,
            "partial_task_policy": "never evaluate an incomplete candidate/family/seed/calibration search"}
        write_new_json(out / "source_lock.json", resource_lock)
        return {"status": resource_lock["status"], "source_lock_sha256": file_sha256(out / "source_lock.json"),
                "completed_task_ids": completed_tasks, "cost": budget.record()}
    except BaseException as exc:
        write_new_json(out / "source_failure.json", {"status": "PARTIAL", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "cost": budget.record(),
            "completed_models": models, "completed_selection": summaries, "ended_utc": utc(),
            "query_predictions_generated": False, "query_truth_read": False})
        raise


def output_ref(out, ref):
    path = (out / ref["path"]).resolve()
    if out.resolve() not in path.parents or file_sha256(path) != ref["sha256"]:
        raise LeakageError("Locked run artifact changed or escaped its directory")
    return path


def save_prediction(out, relative, prediction):
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LeakageError("A frozen prediction artifact cannot be overwritten")
    arrays = {k: prediction[k] for k in ("record_ids", "classes", "predictions", "scores")}
    np.savez_compressed(path, **arrays)
    return {"path": str(path.relative_to(out)), "sha256": file_sha256(path)}


def run_predict(repo, protocol_path, output, release_path):
    protocol, protocol_hash, code = load_protocol(repo, protocol_path)
    out = repo_path(repo, output)
    source_lock_path = out / "source_lock.json"
    source_lock_hash = file_sha256(source_lock_path)
    release = require_release(repo, release_path, protocol_hash, "predict", source_lock_hash=source_lock_hash)
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != protocol_hash or lock["code_version"] != code:
        raise LeakageError("Source lock belongs to different protocol/code bytes")
    # All source tasks/models/policies must exist before any query prediction.
    completed_tasks = lock.get("completed_task_ids", [t["id"] for t in protocol["tasks"]])
    if not completed_tasks or set(completed_tasks) - {t["id"] for t in protocol["tasks"]}:
        raise LeakageError("No complete task is eligible for query release")
    expected = {(t, f, s) for t in completed_tasks for f in protocol["candidates"] for s in protocol["seeds"]}
    actual = {(m["task"], m["family"], m["seed"]) for m in lock["models"]}
    if actual != expected or len(actual) != len(lock["models"]):
        raise LeakageError("All task/family/seed source models must be locked together")
    for model in lock["models"]:
        output_ref(out, model)
        output_ref(out, model["source_reference"])
    directory = out / "predictions"
    directory.mkdir(exist_ok=False)
    write_new_json(directory / "release.json", release)
    budget = Budget(protocol["resources"], prior_cpu=lock["cost"]["cpu_seconds_total"],
                    prior_wall=lock["cost"]["wall_seconds_total"])
    tasks = {t["id"]: t for t in protocol["tasks"]}
    all_predictions = []
    try:
        with threadpool_limits(limits=1):
            for model_ref in lock["models"]:
                budget.check()
                task, family, seed = tasks[model_ref["task"]], model_ref["family"], model_ref["seed"]
                packs = role_packs(repo, task)
                preflight_role_packs(packs)
                model = joblib.load(output_ref(out, model_ref))
                native = predict_numeric(model, packs["query"]["X"], packs["query"]["record_ids"])
                prefix = f"predictions/{task['id']}/{family}/seed{seed}"
                native_ref = save_prediction(out, prefix + "/native.npz", native)
                common = {"task": task["id"], "family": family, "seed": seed,
                          "source_model_sha256": model_ref["sha256"]}
                all_predictions.append({**common, **native_ref, "method": "none", "shot": 0, "draw": -1,
                    "native_vote_vs_argmax_disagreements": native["native_vote_vs_argmax_disagreements"]})
                if family != "mcuve_svm":
                    continue
                head_control = {**native, "predictions": native["classes"][native["scores"].argmax(1)]}
                head_ref = save_prediction(out, prefix + "/raw_argmax_head_control.npz", head_control)
                all_predictions.append({**common, **head_ref, "baseline": native_ref,
                    "method": "raw_score_argmax_head_control", "shot": 0, "draw": -1,
                    "interpretation": "no support labels; score-head diagnostic, never calibration gain"})
                support = predict_numeric(model, packs["support"]["X"], packs["support"]["record_ids"])
                with np.load(output_ref(out, model_ref["source_reference"]), allow_pickle=False) as archive:
                    reference = archive["scores"]
                    if not np.array_equal(archive["classes"], native["classes"]):
                        raise LeakageError("Source score reference has a different vocabulary")
                policy = lock["conservative_policies"][task["id"]]
                zero_controls = {}
                for zero_method in ("none", "traditional", "conservative", "zero_label_control"):
                    zero = calibrate_scores(zero_method, reference, native["classes"], native["scores"], native["predictions"])
                    if (not np.array_equal(zero["scores"], native["scores"])
                            or not np.array_equal(zero["predictions"], native["predictions"])):
                        raise LeakageError("Zero-support calibration changed frozen native predictions")
                    zero_controls[zero_method] = {"exact_native_identity": True, "reason": zero["reason"]}
                write_new_json(out / prefix / "zero_support_controls.json", zero_controls)
                for draw in task.get("calibration_draws", []):
                    draw_id = int(draw["draw_id"])
                    for shot, requested in draw["shots"].items():
                        subset = indices_for(packs["support"]["record_ids"], requested)
                        subset_ids = packs["support"]["record_ids"][subset]
                        counts = {c: int(np.sum(pack_labels(packs["support"])[subset] == c)) for c in task["classes"]}
                        if any(n != int(shot) for n in counts.values()):
                            raise LeakageError("Target support draw does not match its uniform independent shot count")
                        for method in ("zero_label_control", "traditional", "conservative"):
                            budget.check()
                            fit_id = f"{task['id']}/{family}/seed{seed}/draw{draw_id}/K{shot}/{method}"
                            scope = FitScope(subset_ids, fit_id, protocol["dataset_version"], code_version=code)
                            calibration_cpu, calibration_wall = time.process_time(), time.monotonic()
                            if method == "conservative" and policy["fallback_to_identity"]:
                                adapted = {"predictions": native["predictions"].copy(), "scores": native["scores"].copy(),
                                    "status": "NO_OP", "reason": policy.get("reason", "source_development_gate_rejected")}
                            else:
                                adapted = calibrate_scores(method, reference, native["classes"], native["scores"],
                                    native["predictions"], support_scores=support["scores"][subset],
                                    support_labels=pack_labels(packs["support"])[subset],
                                    support_native=support["predictions"][subset],
                                    limited_candidate=policy.get("selected"), fit_scope=scope, support_ids=subset_ids)
                            prediction = {**native, "predictions": adapted["predictions"], "scores": adapted["scores"]}
                            if method == "zero_label_control" and (
                                    not np.array_equal(prediction["predictions"], native["predictions"])
                                    or not np.array_equal(prediction["scores"], native["scores"])):
                                raise LeakageError("Zero-label structural control changed native predictions")
                            prediction_ref = save_prediction(out,
                                prefix + f"/draw{draw_id}_K{shot}_{method}.npz", prediction)
                            diagnostic = {k: v for k, v in adapted.items() if k not in {"scores", "predictions", "unwrapped_predictions"}}
                            diagnostic.update(actual_process_cpu_seconds=time.process_time() - calibration_cpu,
                                actual_wall_seconds=time.monotonic() - calibration_wall)
                            all_predictions.append({**common, **prediction_ref, "baseline": native_ref, "method": method,
                                "shot": int(shot), "draw": draw_id, "support_ids": subset_ids.tolist(),
                                "support_counts": counts, "adapter": diagnostic, "fit_events": scope.entries,
                                "support_native_vs_argmax_disagreements": int(np.sum(support["predictions"][subset]
                                    != native["classes"][support["scores"][subset].argmax(1)]))})
                            if method == "traditional" and "unwrapped_predictions" in adapted:
                                unwrapped = {**prediction, "predictions": adapted["unwrapped_predictions"]}
                                unwrapped_ref = save_prediction(out,
                                    prefix + f"/draw{draw_id}_K{shot}_traditional_unwrapped.npz", unwrapped)
                                all_predictions.append({**common, **unwrapped_ref, "baseline": native_ref,
                                    "method": "traditional_unwrapped_anchor_argmax_diagnostic", "shot": int(shot),
                                    "draw": draw_id, "support_ids": subset_ids.tolist(),
                                    "interpretation": "same fitted reg1 Anchor; includes unguarded score-head rule, diagnostic"})
                budget.check()
        prediction_lock = {"status": "DONE_PREDICTIONS_LOCKED_NOT_SCORED", "protocol_sha256": protocol_hash,
            "source_lock_sha256": source_lock_hash, "code_version": code, "ended_utc": utc(),
            "predictions": all_predictions, "cost": budget.record(), "query_truth_read": False}
        write_new_json(out / "prediction_lock.json", prediction_lock)
        return {"status": prediction_lock["status"], "predictions": len(all_predictions),
                "prediction_lock_sha256": file_sha256(out / "prediction_lock.json"), "cost": budget.record()}
    except BaseException as exc:
        write_new_json(directory / "prediction_failure.json", {"status": "PARTIAL", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "cost": budget.record(),
            "completed_predictions": all_predictions, "query_truth_read": False})
        raise


def read_truth(repo, task):
    path = verify_ref(repo, task["truth"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"record_id", "y"}.issubset(rows[0]):
        raise LeakageError("Independent truth CSV requires record_id,y")
    return np.array([r["record_id"] for r in rows]), np.array([r["y"] for r in rows])


def subset_metrics(truth_ids, truth_labels, prediction, requested, baseline=None):
    subset = indices_for(prediction["record_ids"], requested)
    kept = {k: np.asarray(prediction[k])[subset] for k in ("record_ids", "predictions", "scores")}
    kept["classes"] = prediction["classes"]
    truth_lookup = dict(zip(truth_ids, truth_labels, strict=True))
    truth = np.array([truth_lookup[i] for i in kept["record_ids"]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = score_predictions(kept["record_ids"], truth, kept,
                                    baseline=None if baseline is None else baseline[subset])
    metrics["metric_warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    averaging_classes = np.unique(truth)
    _, _, f1, _ = precision_recall_fscore_support(truth, kept["predictions"], labels=averaging_classes, zero_division=0)
    metrics["macro_f1"] = float(np.mean(f1))
    metrics["macro_average_class_order"] = averaging_classes.tolist()
    metrics["confusion_matrix_note"] = "Full source vocabulary retained, including predictions into other strata"
    return metrics


def plot_model_summaries(directory, aggregate, rows):
    """Descriptive mean/SD plots; seeds and repeated query specimens are explicit."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "BLOCKED_MISSING_MATPLOTLIB", "reason": str(exc)}
    artifacts = []
    baseline = [r for r in aggregate if r["method"] == "none" and r["stratum"] == "all"]
    for task in sorted({r["task"] for r in baseline}):
        values = [r for r in baseline if r["task"] == task]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, metric, label in zip(axes, ("accuracy", "macro_f1"), ("Accuracy (%)", "Macro-F1 (%)")):
            labels = [r["family"] for r in values]
            ax.bar(np.arange(len(values)), [100 * r[metric + "_mean"] for r in values],
                   yerr=[100 * r[metric + "_std"] for r in values], capsize=3)
            ax.set_xticks(np.arange(len(values)), labels, rotation=20, ha="right")
            ax.set_ylabel(label)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", alpha=.25)
        fig.suptitle(task + "; mean/population SD over seeds 42/43/44\nSame exploratory query specimens per seed")
        for extension in ("png", "pdf"):
            path = directory / (task + "_model_comparison." + extension)
            fig.savefig(path, dpi=160)
            artifacts.append({"path": path.name, "sha256": file_sha256(path)})
        plt.close(fig)
        model_rows = [r for r in rows if r["task"] == task and r["method"] == "none" and r["stratum"] == "all"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        for ax, family in zip(axes.flat, sorted({r["family"] for r in model_rows})):
            selected = [r for r in model_rows if r["family"] == family]
            matrix = np.mean([r["confusion_matrix"] for r in selected], axis=0)
            ax.imshow(matrix, cmap="Blues")
            ax.set_title(family + "; mean per-seed counts")
            ax.set_xticks(range(len(selected[0]["class_order"])), selected[0]["class_order"], rotation=45, ha="right")
            ax.set_yticks(range(len(selected[0]["class_order"])), selected[0]["class_order"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
        fig.suptitle(task + "; repeated predictions of the same query specimens")
        path = directory / (task + "_confusion_matrices.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.append({"path": path.name, "sha256": file_sha256(path)})
    return {"status": "DONE_DESCRIPTIVE_PLOTS", "artifacts": artifacts}


def run_score(repo, protocol_path, output):
    protocol, protocol_hash, code = load_protocol(repo, protocol_path)
    out = repo_path(repo, output)
    lock_path = out / "prediction_lock.json"
    prediction_lock_hash = file_sha256(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != protocol_hash or lock["code_version"] != code:
        raise LeakageError("Prediction lock belongs to different protocol/code bytes")
    if file_sha256(out / "source_lock.json") != lock["source_lock_sha256"]:
        raise LeakageError("Source model lock changed after prediction")
    source_lock = json.loads((out / "source_lock.json").read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in protocol["tasks"]}
    expected = set()
    for model in source_lock["models"]:
        task_id, family, seed = model["task"], model["family"], model["seed"]
        expected.add((task_id, family, seed, "none", 0, -1))
        if family == "mcuve_svm":
            expected.add((task_id, family, seed, "raw_score_argmax_head_control", 0, -1))
            for draw in tasks[task_id].get("calibration_draws", []):
                for shot in draw["shots"]:
                    for method in ("zero_label_control", "traditional", "conservative",
                                   "traditional_unwrapped_anchor_argmax_diagnostic"):
                        expected.add((task_id, family, seed, method, int(shot), int(draw["draw_id"])))
    actual_keys = [tuple(ref[k] for k in ("task", "family", "seed", "method", "shot", "draw"))
                   for ref in lock["predictions"]]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected:
        raise LeakageError("Prediction lock lacks complete registered task/family/seed/calibration arms")
    for ref in lock["predictions"]:
        output_ref(out, ref)
        if "baseline" in ref:
            output_ref(out, ref["baseline"])
        with np.load(output_ref(out, ref), allow_pickle=False) as archive:
            if not np.array_equal(archive["classes"], np.array(tasks[ref["task"]]["classes"])):
                raise LeakageError("A prediction changed its frozen source class vocabulary")
    directory = out / "scoring"
    directory.mkdir(exist_ok=False)
    budget = Budget(protocol["resources"], prior_cpu=lock["cost"]["cpu_seconds_total"],
                    prior_wall=lock["cost"]["wall_seconds_total"])
    truths, rows = {}, []
    for ref in lock["predictions"]:
        budget.check()
        task = tasks[ref["task"]]
        if task["id"] not in truths:
            truths[task["id"]] = read_truth(repo, task)  # First and only stage allowed to open query truth.
        truth_ids, truth_labels = truths[task["id"]]
        with np.load(output_ref(out, ref), allow_pickle=False) as archive:
            prediction = {k: archive[k] for k in archive.files}
        baseline = None
        if "baseline" in ref:
            with np.load(output_ref(out, ref["baseline"]), allow_pickle=False) as archive:
                if not np.array_equal(archive["record_ids"], prediction["record_ids"]):
                    raise LeakageError("Calibration baseline identity/order changed")
                baseline = archive["predictions"]
        summary = {k: ref[k] for k in ("task", "family", "seed", "method", "shot", "draw")}
        metrics = score_predictions(truth_ids, truth_labels, prediction, baseline=baseline)
        if baseline is not None:
            native_prediction = {**prediction, "predictions": baseline}
            native_metrics = score_predictions(truth_ids, truth_labels, native_prediction)
            metrics.update({m + "_gain_pp": 100 * (metrics[m] - native_metrics[m])
                            for m in ("accuracy", "macro_f1", "balanced_accuracy")})
        rows.append({**summary, "stratum": "all", **metrics})
        for name, stratum in task.get("strata", {}).items():
            requested = stratum["record_ids"] if isinstance(stratum, dict) else stratum
            subset = subset_metrics(truth_ids, truth_labels, prediction, requested, baseline)
            if baseline is not None:
                native_subset = subset_metrics(truth_ids, truth_labels, {**prediction, "predictions": baseline}, requested)
                subset.update({m + "_gain_pp": 100 * (subset[m] - native_subset[m])
                               for m in ("accuracy", "macro_f1", "balanced_accuracy")})
            rows.append({**summary, "stratum": name, **subset})
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("task", "family", "method", "shot", "stratum"))].append(row)
    aggregate = []
    for key, values in groups.items():
        result = dict(zip(("task", "family", "method", "shot", "stratum"), key, strict=True))
        result.update(evaluations=len(values), seeds=sorted({r["seed"] for r in values}),
            interpretation="paired development draws/seeds, not independent new cohorts")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "corrected_errors", "introduced_errors",
                       "accuracy_gain_pp", "macro_f1_gain_pp", "balanced_accuracy_gain_pp"):
            present = [r[metric] for r in values if metric in r]
            if present:
                result[metric + "_mean"] = float(np.mean(present))
                result[metric + "_std"] = float(np.std(present, ddof=0))
                result[metric + "_min"] = float(np.min(present))
                result[metric + "_max"] = float(np.max(present))
        aggregate.append(result)
    compressed_json(directory / "all_prediction_metrics.json.gz", rows)
    write_new_json(directory / "aggregate_metrics.json", aggregate)
    fields = ["task", "family", "method", "shot", "stratum", "evaluations", "accuracy_mean", "accuracy_std",
              "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "balanced_accuracy_std",
              "corrected_errors_mean", "introduced_errors_mean", "accuracy_gain_pp_mean", "macro_f1_gain_pp_mean",
              "balanced_accuracy_gain_pp_mean"]
    with (directory / "model_calibration_comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregate)
    plots = plot_model_summaries(directory, aggregate, rows)
    write_new_json(directory / "figures.json", plots)
    feature_fields = ["task", "family", "seed", "selected_features", "source_fit_seconds", "source_model_sha256"]
    with (directory / "model_feature_cost.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows({**{k: m[k] for k in feature_fields[:-1]}, "source_model_sha256": m["sha256"]}
                         for m in source_lock["models"])
    budget.check()
    result = {"status": "DONE_INDEPENDENT_METRIC_RECOMPUTATION", "protocol_sha256": protocol_hash,
              "prediction_lock_sha256": prediction_lock_hash, "evaluations": len(rows),
              "query_truth_first_opened_in": "score", "cost": budget.record(),
              "historical_96_0234_direct_comparability": False,
              "plots": plots["status"], "summary_sd": "population SD; correlated seeds/draws, not a confidence interval",
              "interpretation": "RETROSPECTIVE_EXPLORATORY; expanded new-group subset has only three query specimens"}
    write_new_json(directory / "score_lock.json", result)
    return result
