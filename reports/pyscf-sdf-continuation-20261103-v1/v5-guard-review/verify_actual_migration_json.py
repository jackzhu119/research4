#!/usr/bin/env python3
"""Read-only actual migration JSON check; no budget changes or QM imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "reports/pyscf-sdf-continuation-20261103-v1/cpu_budget.json"
NEW = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5/cpu_budget.json"

if __name__ == "__main__":
    old_bytes, new_bytes = OLD.read_bytes(), NEW.read_bytes()
    old, new = json.loads(old_bytes), json.loads(new_bytes)
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    assert old_sha == "4533c152abb267c685a451a2690c9e6be8e94e2fe7e88e9194c8f2dda7384e35"
    assert new["schema"] == "research4-pyscf-actual-cpu-budget-v5"
    assert len(old["phases"]) == 13
    assert new["phases"][:13] == old["phases"]
    for key in ("created_at_utc", "run_deadline_at_utc", "quantum_cpu_cap_seconds", "run_wall_cap_seconds"):
        assert new[key] == old[key], key
    assert new["migration"]["source_v4_sha256"] == old_sha
    assert new["migration"]["new_budget_created"] is False
    assert new["migration"]["prior_phase_statuses_rewritten"] is False
    assert new["migration"]["prior_actual_CPU_preserved"] == 3605.273693
    assert OLD.read_bytes() == old_bytes
    receipt = {
        "status": "ACTUAL_MIGRATED_V4_PREFIX_AND_BOUNDS_READ_ONLY_PASS",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_V4_sha256": old_sha,
        "V5_mutable_ledger_snapshot_sha256_at_read": hashlib.sha256(new_bytes).hexdigest(),
        "original_phases_preserved_exactly": 13, "original_CPU_seconds_carried": 3605.273693,
        "CPU_cap_unchanged": 129600, "wall_cap_unchanged_seconds": 64800,
        "deadline_unchanged": new["run_deadline_at_utc"],
        "old_failed_monitor_rows_unchanged": True,
        "observation_is_not_a_final_live_pool_accounting": True,
        "new_quantum_evaluations": 0, "active_HDF_or_NPY_read": False,
        "actual_budget_created_or_modified_by_reviewer": False,
    }
    output = Path(__file__).with_name("actual_migration_read_only.json")
    with output.open("x") as handle:
        handle.write(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
