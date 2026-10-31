"""Count saved source executions without fitting or opening outer truth."""
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
from herb_spectra import cross_plant_transfer as trial

out = Path("reports/cross-plant-deep-transfer-20261031-v1/neural")
destination = out.parent / "source-execution"
records = trial.read(out / "source_records.json")
lock = trial.read(out / "selection_lock.json")
metas = [trial.read(out / r["model_directory"] / "training.json") for r in records]
assert len(records) == 240
batch_counts, presentations, scopes = Counter(), Counter(), Counter()
for record, metadata in zip(records, metas):
    directory = out / record["model_directory"]
    trial.check_pipeline_evidence(directory)
    assert metadata["target_model_parameters"] == 3603
    assert metadata["profile"]["epochs"] in (48, 96)
    assert metadata["source_versions"] == trial.source_versions()
    with gzip.open(directory / "batch_trace.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids = row["fit_sample_ids"]
            allowed = metadata["source_ids"] if row["stage"] == "target" else metadata["auxiliary_identity"][row["head"]]["record_ids"]
            assert len(set(ids)) == len(ids) and set(ids) <= set(allowed)
            if row["stage"] == "pretrain":
                assert len(ids) == 32
            batch_counts[row["stage"]] += 1
            presentations[row["stage"]] += len(ids)
    with (directory / "fit_scope.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event["event"] == "fit_start":
                scopes[event["component"]] += 1
            assert event["event"] not in ("fit_failed", "fit_rejected")
assert batch_counts == {"target": 103680, "pretrain": 21600}
paired = {}
for metadata in metas:
    key = (metadata["split_id"], metadata["profile"]["id"], metadata["seed"])
    paired.setdefault(key, []).append(metadata)
for group in paired.values():
    assert len(group) == 4
    for key in ("initial_encoder_sha256", "initial_zy_head_sha256", "fresh_target_optimizer_initial_sha256", "target_order_sha256", "raw_target_source_fingerprint"):
        assert len({metadata[key] for metadata in group}) == 1
driver = [json.loads(line) for line in (out / "driver_events.jsonl").read_text().splitlines()]
start = next(row for row in driver if row["event"] == "command_start")
end = next(row for row in driver if row["event"] == "command_completed")
assert end["returncode"] == 0
result = {"completed_at_utc": trial.utc(), "new_summary_fit_operations": 0,
          "source_pipeline_fits": len(records), "actual_driver_wall_seconds": (datetime.fromisoformat(end["utc"]) - datetime.fromisoformat(start["utc"])).total_seconds(),
          "sum_pipeline_fit_setup_seconds": sum(m["elapsed_seconds"] for m in metas),
          "sum_target_training_seconds": sum(m["target_training_seconds"] for m in metas),
          "sum_pretrain_seconds": sum(m["pretrain_seconds"] for m in metas),
          "pretrain_trajectories": sum(m["pretrain_steps"] > 0 for m in metas),
          "pipeline_arm_counts": dict(Counter(m["arm"] for m in metas)),
          "pipeline_profile_counts": dict(Counter(m["profile"]["id"] for m in metas)),
          "pipeline_seed_counts": dict(Counter(m["seed"] for m in metas)),
          "actual_optimizer_step_counts": dict(batch_counts), "actual_sample_presentations_not_unique_N": dict(presentations),
          "actual_FitScope_learning_operation_counts": dict(scopes), "paired_context_profile_seed_groups_checked": len(paired),
          "source_failures": 0, "source_rejections": 0, "outer_predictions_or_scores_created": False,
          "selection_lock_sha256": trial.sha(out / "selection_lock.json"),
          "source_records_sha256": trial.sha(out / "source_records.json"),
          "source_predictions_sha256": trial.sha(out / "source_predictions.csv.gz"),
          "shared_source_choices": [{"outer_fold": f["outer_fold"], "shared_profile": f["shared_profile"]["id"], "policy_arm": f["policy_arm"]} for f in lock["outer_folds"]],
          "profile_choice_basis": "Pooled684 source inner queries per seed, three-seed mean, A-only shared profile",
          "cost_caveat": "Pipeline seconds exclude final serialization/inference; driver wall includes process startup and trial I/O. Steps/batch size matched, not exact FLOPs.",
          "interpretation": "Source policy chose Aextra for two folds and A for three; no external-data arm passed the predeclared development guard."}
trial.write(destination / "source_execution_summary.json", result)
print(json.dumps(result, ensure_ascii=False))
