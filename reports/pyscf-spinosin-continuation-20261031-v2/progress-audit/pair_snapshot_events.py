"""Pair only already captured snapshot events; never reread a live source."""
import hashlib
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent
events = [json.loads(line) for line in (OUT / "active_events.jsonl.snapshot").read_text().splitlines()]
review = json.loads((OUT / "progress_snapshot_review.json").read_text())
selected = {row["cycle"]: row for row in review["audited_geometries"]}
pairs = []
release = timing = None
for event in events:
    if event["event"] == "gradient_started_after_cache_release":
        release, timing = event, None
    elif event["event"] == "gradient_kernel_complete":
        timing = event
    elif event["event"] == "geometry_cycle" and event["cycle"] in selected:
        row = selected[event["cycle"]]
        assert release is not None and timing is not None
        assert math.isclose(release["SCF_energy_hartree"], row["energy_hartree"], abs_tol=5e-9, rel_tol=0)
        assert math.isclose(event["energy_hartree"], row["energy_hartree"], abs_tol=1e-12, rel_tol=0)
        pairs.append({"cycle": event["cycle"], "paired_energy_hartree": row["energy_hartree"],
                      "SCF_gradient_event_energies_match": True,
                      "gradient_cpu_seconds": timing["gradient_cpu_seconds"],
                      "gradient_wall_seconds": timing["gradient_wall_seconds"],
                      "cache_release_rss_mb_before": release["current_rss_mb_before_release"],
                      "cache_release_rss_mb_after": release["current_rss_mb_after_release"],
                      "gradient_started_at_utc": release["utc"], "gradient_finished_at_utc": timing["utc"]})
assert len(pairs) == len(selected)
target = OUT / "paired_snapshot_timings.json"
assert not target.exists()
target.write_text(json.dumps({"status": "PASS_CAPTURED_EVENTS_ONLY", "new_live_source_reads": 0,
                              "new_quantum_jobs": 0, "pairs": pairs,
                              "source_snapshot_sha256": hashlib.sha256((OUT / "active_events.jsonl.snapshot").read_bytes()).hexdigest()}, indent=2) + "\n")
print(json.dumps(pairs))
