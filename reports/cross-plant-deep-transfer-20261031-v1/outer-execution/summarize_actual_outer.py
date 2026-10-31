"""Count immutable outer fits/scoring commands, without training or rescoring."""
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
from herb_spectra import cross_plant_transfer as trial

out = Path("reports/cross-plant-deep-transfer-20261031-v1/neural")
destination = out.parent / "outer-execution"
records = trial.read(out / "outer_model_records.json")
metas = [trial.read(out / r["model_directory"] / "training.json") for r in records]
assert len(records) == 60
batch_counts, visits, operations = Counter(), Counter(), Counter()
for record, metadata in zip(records, metas):
    directory = out / record["model_directory"]
    trial.check_pipeline_evidence(directory)
    assert metadata["profile"]["epochs"] == 96
    assert metadata["target_model_parameters"] == 3603
    with gzip.open(directory / "batch_trace.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            batch = json.loads(line)
            ids = batch["fit_sample_ids"]
            allowed = metadata["source_ids"] if batch["stage"] == "target" else metadata["auxiliary_identity"][batch["head"]]["record_ids"]
            assert len(ids) == len(set(ids)) and set(ids) <= set(allowed)
            if batch["stage"] == "pretrain":
                assert len(ids) == 32
            batch_counts[batch["stage"]] += 1
            visits[batch["stage"]] += len(ids)
    for line in (directory / "fit_scope.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "fit_start":
            operations[row["component"]] += 1
        assert row["event"] not in ("fit_failed", "fit_rejected")
assert batch_counts == {"target": 63360, "pretrain": 5400}
paired = {}
for metadata in metas:
    paired.setdefault((metadata["split_id"], metadata["seed"]), []).append(metadata)
for group in paired.values():
    assert len(group) == 4
    for key in ("initial_encoder_sha256", "initial_zy_head_sha256", "fresh_target_optimizer_initial_sha256", "target_order_sha256", "raw_target_source_fingerprint"):
        assert len({m[key] for m in group}) == 1
events = [json.loads(line) for line in (out / "driver_events.jsonl").read_text().splitlines()]
timings = {}
for label in ("cross-plant-main-outer-prediction", "cross-plant-main-prediction-lock-verification", "cross-plant-main-independent-score"):
    start = next(e for e in events if e["event"] == "command_start" and e["label"] == label)
    end = next(e for e in events if e["event"] == "command_completed" and e["label"] == label)
    assert end["returncode"] == 0
    timings[label] = (datetime.fromisoformat(end["utc"]) - datetime.fromisoformat(start["utc"])).total_seconds()
result = {"completed_at_utc": trial.utc(), "new_summary_fit_operations": 0, "new_metric_scoring_calls": 0,
          "actual_outer_model_fits": 60, "policy_alias_cases": 15, "outer_prediction_rows": 12825,
          "paired_context_seed_groups_checked": len(paired), "pretrain_trajectories": 45,
          "all_target_epochs": 96, "actual_optimizer_step_counts": dict(batch_counts),
          "actual_FitScope_learning_operation_counts": dict(operations),
          "actual_sample_presentations_not_unique_N": dict(visits), "command_wall_seconds": timings,
          "sum_pipeline_fit_setup_seconds": sum(m["elapsed_seconds"] for m in metas),
          "sum_target_training_seconds": sum(m["target_training_seconds"] for m in metas),
          "sum_pretrain_seconds": sum(m["pretrain_seconds"] for m in metas),
          "runtime_model_failures": 0, "FitScope_rejections": 0,
          "code_source_sha256": trial.source_versions(),
          "selection_lock_sha256": trial.sha(out / "selection_lock.json"),
          "prediction_lock_sha256": trial.sha(out / "prediction_lock.json"),
          "scoring_manifest_sha256": trial.sha(out / "scoring_manifest.json"),
          "new_independent_confirmation_created": False}
trial.write(destination / "outer_execution_summary.json", result)
print(json.dumps(result, ensure_ascii=False))
