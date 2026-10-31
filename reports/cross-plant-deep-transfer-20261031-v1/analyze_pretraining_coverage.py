"""Read locked training schedules; count source coverage without fitting models."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    run = root / "neural"
    out = root / "pretraining-coverage"
    if out.exists():
        raise FileExistsError("Preserve the previous coverage analysis")
    if not (run / "scoring_manifest.json").exists():
        raise RuntimeError("Wait for all training and locked separate scoring")
    records = []
    for phase, file in [("source", "source_records.json"), ("outer", "outer_model_records.json")]:
        for record in json.loads((run / file).read_text()):
            directory = run / record["model_directory"]
            meta = json.loads((directory / "training.json").read_text())
            schedule_path = directory / "batch_schedule.json"
            assert hashlib.sha256(schedule_path.read_bytes()).hexdigest() == meta["batch_schedule_sha256"]
            schedule = json.loads(schedule_path.read_text())["pretrain"]
            for pool, steps in meta["pretrain_pool_steps"].items():
                ids = meta["auxiliary_identity"][pool]["record_ids"]
                batches = [entry["indices"] for entry in schedule if entry["pool"] == pool]
                assert len(batches) == steps
                assert all(len(batch) == len(set(batch)) == meta["profile"]["pretrain_batch_size"] for batch in batches)
                observed = {i for batch in batches for i in batch}
                assert all(0 <= i < len(ids) for i in observed)
                records.append({"phase": phase, "split_id": meta["split_id"], "arm": meta["arm"],
                                "seed": meta["seed"], "profile": meta["profile"]["id"], "pool": pool,
                                "eligible_source_observations": len(ids), "optimizer_unique_observations": len(observed),
                                "optimizer_presentations": sum(map(len, batches)), "pretrain_steps": steps})
    out.mkdir()
    with (out / "actual_coverage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for arm in ("Aextra", "B", "C"):
        for pool in sorted({row["pool"] for row in records if row["arm"] == arm}):
            rows = [row for row in records if row["arm"] == arm and row["pool"] == pool]
            summary.append({"arm": arm, "pool": pool, "training_trajectories": len(rows),
                            "unique_optimizer_observations_per_trajectory_min": min(row["optimizer_unique_observations"] for row in rows),
                            "unique_optimizer_observations_per_trajectory_max": max(row["optimizer_unique_observations"] for row in rows),
                            "presentations_total_not_independent_samples": sum(row["optimizer_presentations"] for row in rows)})
    (out / "summary.json").write_text(json.dumps({"research_fits": 0, "summary": summary,
        "note": "Declared eligible pool size and actual gradient coverage are distinct. Scalers fit all eligible source rows. Repeated presentations, profiles, folds and seeds never add independent N."}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
