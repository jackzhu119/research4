"""Print a compact live status without loading experimental result tables."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from species_background_budget import proc_stat, status

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/research/species_background_20261108"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=BASE / "control")
    args = parser.parse_args()
    state = status(args.state_dir)
    counts = Counter(value["status"] for value in state["jobs"].values())
    manifest = json.loads((BASE / "jobs.json").read_text())
    counts["not_started"] = len(manifest["jobs"]) - len(state["jobs"])
    owner_state = (proc_stat(state["owner"]["pid"]) or {}).get("state")
    supervisor_running = state["supervisor_identity_matches"] and owner_state not in (None, "Z", "X")
    result = {"status": state["status"], "supervisor_pid": state["owner"]["pid"],
              "supervisor_running": supervisor_running,
              "active_worker_running": state["active_identity_matches"],
              "cpu_seconds_used": round(state["spent_cpu_seconds"], 3),
              "cpu_seconds_remaining": round(state["remaining_cpu_seconds"], 3),
              "job_counts": dict(counts), "planned_jobs": len(manifest["jobs"]),
              "active_job": (state.get("active") or {}).get("job_id"),
              "heartbeat_beijing": datetime.fromtimestamp(state["heartbeat_unix"], ZoneInfo("Asia/Shanghai")).isoformat(),
              "scope": "training-only development; fixed calibration and prediction pools excluded"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
