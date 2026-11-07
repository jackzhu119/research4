"""Actual stdlib-only supervisor handshake control; performs no quantum work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_research4_research72h_budget_20261107.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(GUARD) != args.guard_sha256:
        raise ValueError("Frozen new supervisor source differs")
    spec = importlib.util.spec_from_file_location("actual72h_guard_control", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    output = args.output.resolve()
    prefix = ROOT / "reports/research72h-20261107-v1"
    if not output.is_relative_to(prefix) or output == prefix or output.exists():
        raise ValueError("Fresh output inside new72h report prefix required")
    output.mkdir(parents=True)
    value = {
        "schema": "research4-actual72h-supervisor-handshake-control-v1",
        "status": "GUARD_HANDSHAKE_CONTROL_PASSED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_worker": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "guard_sha256": args.guard_sha256,
        "resource_supervision": supervision,
        "actual_worker_pid": os.getpid(),
        "actual_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "new_quantum_evaluations": 0,
        "new_model_fits": 0,
    }
    with (output / "result.json").open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": value["status"], "new_quantum_evaluations": 0}))


if __name__ == "__main__":
    main()
