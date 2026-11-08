"""Deferred numerical leaf for an independently authorized historical NN audit.

Imports are standard-library only. The parent owns original NN/time provenance,
the archived/current Q prefix proof and actual global lock, and a NEW bounded
audit grant. This module never calls the old top-level auditor, authority,
budget, claim or training entry points. It writes only a fresh child directory
of the parent's separately authorized output.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER_SHA256 = "7ef9829efd2bdfefc75e4a7e59813f6713ba8779342176aedc3adca899bc6306"
STAGES = ("prepare", "bind", "source", "fit", "predict", "score")
ARMS = ("alpha_single", "alpha_capacity3", "alpha_n_time")
SEEDS = (42, 43, 44)
TASKS = {"zaoren_legacy7": (513, 171, 7), "huangqi_legacy4": (241, 82, 4),
         "tianma_legacy4": (242, 83, 4), "zaoren_expanded10": (516, 174, 10)}


def require(value, message):
    if not value:
        raise ValueError(message)


def small_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_token(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_entry(run, output, old, context, budget):
    """Check the parent's explicit proof token without reopening any Q object."""
    require(isinstance(context, dict) and all(context.get(k) is True for k in (
        "historical_training_valid", "current_Q_prefix_verified", "current_grant_verified")),
        "Historical training, current Q prefix and current root grant must be verified first")
    require(sha_token(context.get("current_grant_sha256"))
            and context.get("leaf_code_sha256") == small_sha(__file__),
            "Current root grant must bind this exact numerical leaf")
    require(callable(getattr(budget, "check", None)), "Current separately bounded audit budget is required")
    require(Path(old.__file__).resolve().is_relative_to(ROOT / "scripts")
            and small_sha(old.__file__) == FROZEN_HELPER_SHA256
            and Path(old.ROOT).resolve() == ROOT, "Do not replace or monkeypatch the frozen helper/root")
    require(run.is_relative_to(ROOT / "reports") and run.is_dir()
            and output.is_relative_to(ROOT / "reports") and output != ROOT / "reports"
            and run != output and not output.is_relative_to(run) and not run.is_relative_to(output),
            "Historical input and newly authorized output must be separate reports directories")
    require(output.is_dir(), "Parent must first create its fresh grant-bound output")
    required = {"stages", "source_lock", "models_lock", "prediction_lock", "source_manifest", "prediction_manifest"}
    require(required <= set(context), "Missing frozen historical context")
    require(set(context["stages"]) == set(STAGES), "Parent must verify the complete historical stage chain")
    require(not (output / "historical_numerical").exists(), "Never append to or overwrite a prior numerical audit")


def metadata_checks(old, context, budget):
    """All final model identities and pairing checks precede NumPy imports."""
    source_lock, models, locked = (context[k] for k in ("source_lock", "models_lock", "prediction_lock"))
    stages = context["stages"]
    require(locked["models"] == stages["fit"]["model_lock"]
            and models["source_epoch_lock"] == stages["source"]["source_lock"],
            "Historical source/model/prediction reference chain differs")
    source_manifest, manifest = context["source_manifest"], context["prediction_manifest"]
    sources = {t["task_id"]: t for t in source_manifest["tasks"]}
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    require(set(sources) == set(tasks) == set(TASKS)
            and len(source_manifest["tasks"]) == len(manifest["tasks"]) == len(TASKS),
            "Incomplete or duplicate historical tasks")
    old.validate_grid(models["models"], locked["predictions"])
    training = {}
    for item in models["models"]:
        budget.check()
        old.reference(item["model"])
        meta = old.read(old.reference(item["training"]))
        training[item["model_key"]] = meta
        require(meta["task_id"] == item["task_id"] and meta["arm"] == item["arm"]
                and meta["seed"] == item["seed"] and not meta["validation_ids"]
                and meta["query_features_received"] is False and meta["query_labels_received"] is False,
                "Historical final model identity or source-only scope differs")
        require(meta["source_ids"] == sources[item["task_id"]]["source_ids"]
                and len(meta["source_ids"]) == len(set(meta["source_ids"])) == TASKS[item["task_id"]][0],
                "Historical final fit source IDs changed")
        owner = "zaoren_legacy7" if item["task_id"] == "zaoren_expanded10" else item["task_id"]
        require(item["epochs"] == meta["epochs"] == source_lock["locks"][owner]["epoch"],
                "Historical final epoch differs from its source-only lock")
    pairing = []
    for task_id in TASKS:
        for seed in SEEDS:
            capacity = training[f"{task_id}.alpha_capacity3.seed{seed}"]
            multi = training[f"{task_id}.alpha_n_time.seed{seed}"]
            for field in ("task_id", "seed", "epochs", "source_ids", "source_batch_order_sha256",
                          "initial_state_sha256", "initial_encoder_sha256", "parameter_count", "source_class_weights"):
                require(capacity[field] == multi[field], "Historical C/M pairing differs: " + field)
            require(capacity["branch_spec"]["branch_points"] == multi["branch_spec"]["branch_points"]
                    == [128, 128, 1024], "Historical C/M input lengths differ")
            pairing.append({"task_id": task_id, "seed": seed, "epochs": capacity["epochs"],
                "same_parameter_count": capacity["parameter_count"], "same_initial_state_and_order": True})
    return sources, tasks, training, pairing


def run_checks(run: Path, output: Path, old, context: dict, budget, replay_models: bool) -> dict:
    """Actually reverify historical artifacts only after a NEW parent release.

    The parent has already checked old healthy closures/timestamps and current Q
    authority under its original actual global lock. This leaf never compares
    historical training times with today's clock and never writes old ledgers.
    """
    started_cpu, started_wall = time.process_time(), time.monotonic()
    run, output = Path(run).resolve(), Path(output).resolve()
    validate_entry(run, output, old, context, budget)
    budget.check()
    sources, tasks, training, pairing = metadata_checks(old, context, budget)
    source_lock, models, locked = (context[k] for k in ("source_lock", "models_lock", "prediction_lock"))
    directory = output / "historical_numerical"
    directory.mkdir(exist_ok=False)
    old.write_new(directory / "historical_numerical_start.json", {
        "status": "RUNNING_NOT_A_COMPLETED_RESULT", "current_grant_sha256": context["current_grant_sha256"],
        "leaf_code_sha256": context["leaf_code_sha256"], "frozen_helper_sha256": FROZEN_HELPER_SHA256,
        "replay_models_requested": bool(replay_models), "classifier_fits": 0})
    # Every guard above is standard-library metadata only; no real arrays above.
    import numpy as np

    source_check = old.source_epoch_audit(source_lock, context["source_manifest"], np)
    budget.check()
    time_audits = {}
    for task_id, task in sources.items():
        budget.check()
        with np.load(old.reference(task["source_pack"]), allow_pickle=False) as archive:
            source_meta = json.loads(str(archive["metadata_json"]))
        with np.load(old.reference(tasks[task_id]["query_pack"]), allow_pickle=False) as archive:
            query_meta = json.loads(str(archive["metadata_json"]))
        require(source_meta["record_ids"] == task["source_ids"]
                and query_meta["record_ids"] == tasks[task_id]["query_ids"], "Historical prepared role IDs changed")
        require(not set(query_meta) & {"source_labels", "y", "labels", "truth", "origin_group_ids"},
                "Historical inference metadata contains truth")
        source_time, query_time = source_meta["axis_metadata"]["time"], query_meta["axis_metadata"]["time"]
        require(source_time == query_time, "Historical source/query time contract differs")
        time_audits[task_id] = old.validate_time_repair(source_time)
        source_aliases = set(source_meta["record_ids"]) | set(source_meta["entity_ids"])
        query_aliases = set(query_meta["record_ids"]) | set(query_meta["entity_ids"])
        for values in source_meta["identity_aliases"]:
            source_aliases.update(values)
        for values in query_meta["identity_aliases"]:
            query_aliases.update(values)
        require(source_aliases == set(task["source_alias_union"]) and not source_aliases & query_aliases,
                "Historical fit/query identity aliases overlap or changed")
        classes = sorted(set(source_meta["source_labels"])); counts = Counter(source_meta["source_labels"])
        for arm in ARMS:
            for seed in SEEDS:
                meta = training[f"{task_id}.{arm}.seed{seed}"]
                require(meta["source_axis_metadata"]["time"] == source_time
                        and meta["class_labels"] == classes, "Historical training axis/class vocabulary differs")
                weights = [len(source_meta["record_ids"]) / (len(classes) * counts[c]) for c in classes]
                require(all(abs(a - b) <= 1e-6 * max(1., abs(a))
                            for a, b in zip(weights, meta["source_class_weights"], strict=True)),
                        "Historical class weights are not current-source-only balanced weights")
    predictions = {}
    for item in locked["predictions"]:
        budget.check()
        meta = old.read(old.reference(item["prediction_metadata"]))
        audit = meta["inference_audit"]
        require(meta["model_key"] == item["model_key"]
                and meta["status"] == "LABEL_FREE_PREDICTIONS_AWAIT_INDEPENDENT_SCORING"
                and meta["query_ids"] == tasks[item["task_id"]]["query_ids"]
                and meta["query_labels_received"] is False and audit.get("status") == "TECHNICAL_CHECK_PASS"
                and audit.get("samples_checked") == TASKS[item["task_id"]][1]
                and audit.get("eval_mode") is True and audit.get("parameters_and_buffers_unchanged") is True
                and audit.get("gradients_enabled_during_forward") is False,
                "Historical pointwise inference metadata is not healthy")
        with np.load(old.reference(item["predictions"]), allow_pickle=False) as archive:
            require(set(archive.files) == {"record_ids", "class_labels", "logits"}, "Unknown historical prediction fields")
            ids = archive["record_ids"].astype(str).tolist()
            classes = archive["class_labels"].astype(str).tolist(); logits = archive["logits"]
            require(classes == training[item["model_key"]]["class_labels"]
                    and len(classes) == TASKS[item["task_id"]][2], "Historical prediction class columns differ")
            require(logits.shape == (len(ids), len(classes)) and bool(np.isfinite(logits).all()),
                    "Invalid historical frozen logits")
            labels = [classes[int(i)] for i in logits.argmax(1)]
            predictions[item["model_key"]] = (classes, old.align(ids, tasks[item["task_id"]]["query_ids"], labels))
    # Only after all36 frozen ID/class/logit checks may optional native replay run.
    if replay_models:
        replay_dir = directory / "historical_native_replay"; replay_dir.mkdir(exist_ok=False)
        replay = old.replay_saved_models(models, training, sources, tasks, locked, replay_dir, np, budget)
    else:
        replay = {"status": "NOT_RUN_HASH_VERIFICATION_ONLY", "models_replayed": 0,
                  "model_weight_files_sha_verified": 36, "Torch_imported": False}
    budget.check()

    # The first and only target-truth join is below the complete prediction gate.
    true, all_scores, pairs, new_rows = {}, [], [], []
    for task_id, task in tasks.items():
        budget.check()
        truth_rows = old.csv_rows(old.reference(task["truth"]))
        truth = {r["record_id"]: r for r in truth_rows}; ids = task["query_ids"]
        require(len(truth) == len(truth_rows) == len(ids) == TASKS[task_id][1]
                and set(truth) == set(ids), "Historical truth rows differ from frozen IDs")
        true[task_id] = truth
        subsets = {"overall": list(range(len(ids)))}
        if task_id == "zaoren_expanded10":
            subsets["legacy_exposed171"] = [i for i, v in enumerate(ids) if truth[v]["stratum"] == "legacy_exposed"]
            subsets["new3_descriptive_only"] = [i for i, v in enumerate(ids) if truth[v]["stratum"] == "new_three_record_group"]
            old.validate_expanded_strata(ids, tasks["zaoren_legacy7"]["query_ids"],
                [ids[i] for i in subsets["legacy_exposed171"]], [ids[i] for i in subsets["new3_descriptive_only"]])
        labels = [truth[i]["y"] for i in ids]
        for seed in SEEDS:
            arms = {}
            for arm in ARMS:
                classes, pred = predictions[f"{task_id}.{arm}.seed{seed}"]; arms[arm] = pred
                for name, positions in subsets.items():
                    score = old.metrics([labels[i] for i in positions], [pred[i] for i in positions], classes)
                    all_scores.append({"task_id": task_id, "subset": name, "arm": arm, "seed": seed, **score})
                if task_id == "zaoren_expanded10":
                    new_rows.extend({"record_id": ids[i], "truth": labels[i], "prediction": pred[i],
                        "arm": arm, "seed": seed, "correct": labels[i] == pred[i],
                        "independent_new_query_specimens": 3} for i in subsets["new3_descriptive_only"])
            for baseline, candidate in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
                for name, positions in subsets.items():
                    pairs.append({"task_id": task_id, "subset": name, "seed": seed,
                        "baseline": baseline, "candidate": candidate, "N": len(positions),
                        **old.pair([labels[i] for i in positions], [arms[baseline][i] for i in positions],
                                   [arms[candidate][i] for i in positions])})
    difference = 0.
    def compare(left, right, message):
        nonlocal difference
        require(math.isfinite(float(left)) and math.isfinite(float(right)), message)
        difference = max(difference, abs(float(left) - float(right)))
        old.numeric_equal(left, right, message)
    key = lambda r: (r["task_id"], r["subset"], r["arm"], int(r["seed"]))
    stored = old.csv_rows(run / "score/per_seed_metrics.csv"); lookup = {key(r): r for r in stored}
    require(len(lookup) == len(stored) == len(all_scores) == 54, "Incomplete historical per-seed metrics")
    for actual in all_scores:
        for field in ("N", "correct", "accuracy_percent", "macro_f1", "balanced_accuracy_percent"):
            compare(actual[field], lookup[key(actual)][field], "Historical per-seed metric differs")
    class_key = lambda r: (*key(r), r["class"])
    stored_class = old.csv_rows(run / "score/per_class_metrics.csv")
    class_lookup = {class_key(r): r for r in stored_class}
    confusion_key = lambda r: (*key(r), r["true_class"], r["predicted_class"])
    stored_confusion = old.csv_rows(run / "score/confusion_matrix_counts.csv")
    confusion_lookup = {confusion_key(r): r for r in stored_confusion}
    require(len(class_lookup) == len(stored_class) == sum(len(r["per_class"]) for r in all_scores) == 405,
            "Historical per-class grid differs")
    require(len(confusion_lookup) == len(stored_confusion) == sum(len(r["confusion"]) for r in all_scores) == 3429,
            "Historical confusion grid differs")
    class_rows, confusion_rows = [], []
    for actual in all_scores:
        prefix = {k: actual[k] for k in ("task_id", "subset", "arm", "seed")}
        for row in actual["per_class"]:
            merged = {**prefix, **row}; class_rows.append(merged)
            for field in ("precision", "recall", "f1", "support"):
                compare(row[field], class_lookup[class_key(merged)][field], "Historical per-class metric differs")
        for row in actual["confusion"]:
            merged = {**prefix, **row}; confusion_rows.append(merged)
            compare(row["count"], confusion_lookup[confusion_key(merged)]["count"], "Historical confusion count differs")
    groups = defaultdict(list)
    for row in all_scores:
        groups[row["task_id"], row["subset"], row["arm"]].append(row)
    summaries = old.csv_rows(run / "score/model_comparison.csv"); seen = set(); rebuilt_summary = []
    require(len(summaries) == len(groups) == 18, "Historical summary group count differs")
    for stored in summaries:
        group_key = tuple(stored[k] for k in ("task_id", "subset", "arm"))
        require(group_key not in seen, "Duplicate historical summary group"); seen.add(group_key)
        values = groups[group_key]
        require(sorted(r["seed"] for r in values) == list(SEEDS), "Historical summary selects an incomplete/best seed")
        rebuilt = dict(zip(("task_id", "subset", "arm"), group_key))
        for metric in ("accuracy_percent", "macro_f1", "balanced_accuracy_percent"):
            for suffix, operation in (("mean", statistics.mean), ("sampleSD", statistics.stdev)):
                value = operation(r[metric] for r in values)
                rebuilt[metric + "_" + suffix] = value
                compare(value, stored[metric + "_" + suffix], "Historical seed mean/sampleSD differs")
        rebuilt_summary.append(rebuilt)
    stored_pairs = old.csv_rows(run / "score/paired_corrected_harmed.csv")
    pair_key = lambda r: (r["task_id"], r["subset"], int(r["seed"]), r["baseline"], r["candidate"])
    pair_lookup = {pair_key(r): r for r in stored_pairs}
    require(len(pair_lookup) == len(stored_pairs) == len(pairs) == 54, "Historical paired contrasts incomplete")
    for actual in pairs:
        for field in ("N", "corrected", "harmed", "net_correct", "accuracy_delta_pp"):
            compare(actual[field], pair_lookup[pair_key(actual)][field], "Historical corrected/harmed count differs")
    retention = []
    old_ids = tasks["zaoren_legacy7"]["query_ids"]
    old_truth = [true["zaoren_legacy7"][i]["y"] for i in old_ids]
    for arm in ARMS:
        for seed in SEEDS:
            _, baseline = predictions[f"zaoren_legacy7.{arm}.seed{seed}"]
            _, expanded = predictions[f"zaoren_expanded10.{arm}.seed{seed}"]
            expanded_lookup = dict(zip(tasks["zaoren_expanded10"]["query_ids"], expanded, strict=True))
            require(set(old_ids) <= set(expanded_lookup)
                    and all(true["zaoren_expanded10"][i]["y"] == t for i, t in zip(old_ids, old_truth, strict=True)),
                    "Historical old171 identity/truth changed after class expansion")
            retention.append({"arm": arm, "seed": seed, "N": 171,
                "A_correct": sum(a == t for a, t in zip(baseline, old_truth, strict=True)),
                "B_old_correct": sum(expanded_lookup[i] == t for i, t in zip(old_ids, old_truth, strict=True)),
                **old.pair(old_truth, baseline, [expanded_lookup[i] for i in old_ids]),
                "interpretation": "Same known-class171 retention; exposed development, not new confirmation"})
    require(len(retention) == 9 and len(new_rows) == 27, "Historical retention/new-three rows incomplete")
    budget.check()
    old.write_new(directory / "historical_source_epoch_audit.json", source_check)
    old.write_new(directory / "historical_capacity_pairing_audit.json", pairing)
    old.write_new(directory / "historical_time_axis_repair_audit.json", time_audits)
    old.write_new(directory / "historical_all_metrics_recomputed.json", all_scores)
    basic = [{k: v for k, v in r.items() if k not in ("per_class", "confusion")} for r in all_scores]
    for filename, rows in (("historical_per_seed_metrics.csv", basic),
                           ("historical_model_comparison.csv", rebuilt_summary),
                           ("historical_per_class_metrics.csv", class_rows),
                           ("historical_confusion_counts.csv", confusion_rows),
                           ("historical_paired_contrasts.csv", pairs),
                           ("historical_old171_retention.csv", retention),
                           ("historical_new3_predictions.csv", new_rows)):
        old.csv_new(directory / filename, rows)
    summary = {"status": "PASS_HISTORICAL_NUMERICAL_REVERIFICATION", "source_models_audited": 27,
        "final_model_metadata_audited": 36, "prediction_artifacts_recomputed": 36, "all_seeds": list(SEEDS),
        "per_seed_score_rows": len(all_scores), "summary_rows": len(rebuilt_summary),
        "per_class_rows": len(class_rows), "confusion_rows": len(confusion_rows),
        "paired_modality_rows": len(pairs), "paired_old171_rows": len(retention),
        "new3_repeated_prediction_rows": len(new_rows), "independent_new_query_specimens": 3,
        "maximum_recomputed_metric_difference": difference, "classifier_fits_by_leaf": 0,
        "Torch_imported_by_leaf": bool(replay_models), "PySCF_imported_by_leaf": False,
        "native_model_replay": {k: v for k, v in replay.items() if k != "reports"},
        "current_grant_sha256": context["current_grant_sha256"], "leaf_code_sha256": context["leaf_code_sha256"],
        "frozen_helper_sha256": FROZEN_HELPER_SHA256, "original_training_budgets_or_claims_modified": False,
        "historical_96_0234_direct_comparison": False, "numerical_output_directory": str(directory.relative_to(ROOT)),
        "leaf_process_CPU_seconds": time.process_time() - started_cpu,
        "leaf_wall_seconds": time.monotonic() - started_wall,
        "healthy_stage_resource_closure_required_from_parent": True}
    budget.check()
    old.write_new(directory / "historical_numerical_summary.json", summary)
    return summary
