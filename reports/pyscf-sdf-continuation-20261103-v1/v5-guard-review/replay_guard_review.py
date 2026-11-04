#!/usr/bin/env python3
"""Independent pure V5 reaping/migration review; no actual budget or QM.

The complete migration fixture uses private temporary files with an explicitly
invalid synthetic authorization plus mocked authorization/prior verification.
Its output cannot be run as an authorized research ledger. Existing real V4
history is read as JSON only and never modified.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"
OLD = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261101.py"
EXPECTED = "33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee"
OLD_SHA = "354ff51b5c2af2810033656d9972948262cfebb4dabb3978fca2d021f97e2b8f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)}


def pure_complete_migration(r):
    with tempfile.TemporaryDirectory(prefix="synthetic-v5-migration-review-") as name:
        root = Path(name)
        prior = root / "prior/ledger.json"
        write(prior, {"scope": "PURE_FIXTURE_NO_ACTUAL_PRIOR_BUDGET"})
        prior.with_suffix(".lock").touch()
        source = root / "old.json"
        checkpoint = root / "source.chk"
        checkpoint.write_bytes(b"PURE_FIXTURE_NOT_A_QUANTUM_CHECKPOINT")
        prior_info = {"scope": "SYNTHETIC_PRIOR_INFO"}
        old = {
            "schema": "research4-pyscf-actual-cpu-budget-v4", "runner_sha256": OLD_SHA,
            "created_at_utc": "2026-11-04T07:20:50.391742+00:00",
            "run_deadline_at_utc": "2026-11-05T01:20:50.391742+00:00",
            "quantum_cpu_cap_seconds": 129600, "run_wall_cap_seconds": 64800,
            "consumed_cpu_seconds": 12.5, "remaining_quantum_cpu_seconds": 129587.5,
            "phases": [
                {"phase": "SYNTHETIC_SUCCESS", "status": "EXITED", "exit_code": 0,
                 "actual_cpu_seconds": 2.5, "ended_at_utc": "2026-11-04T07:21:00+00:00"},
                {"phase": "SYNTHETIC_OLD_REAP_FAILURE", "status": "PARTIAL_OR_FAILED",
                 "exit_code": 0, "monitor_failed": True, "hard_killed": True,
                 "actual_cpu_seconds": 10., "ended_at_utc": "2026-11-04T07:22:00+00:00"}],
            "source_checkpoint_path": str(checkpoint), "source_checkpoint_sha256": sha(checkpoint),
            "source_base_commit": "d"*40, "authorization_path": str(root / "old_auth.json"),
            "authorization_sha256": "e"*64, "prior_closed_ledger": prior_info,
            "prior_budget_is_historical_only": True,
        }
        write(source, old)
        source_bytes = source.read_bytes()
        authorization_file = root / "synthetic_auth.json"
        # Deliberately not a valid V5 authorization; the temporary ledger can
        # never launch a real phase through the unmocked supervisor.
        write(authorization_file, {"scope": "SYNTHETIC_MOCKED_AUTHORIZATION_ONLY"})
        auth = {
            "migrate_from_closed_v4_path": str(source), "migrate_from_closed_v4_sha256": sha(source),
            "prior_closed_ledger_path": str(prior), "actual_cpu_cleanup_reserve_seconds": 120,
            "max_quantum_cpu_seconds": 129600, "max_run_wall_seconds": 64800,
            **{"original_"+key: old[key] for key in
               ("created_at_utc", "run_deadline_at_utc", "quantum_cpu_cap_seconds", "run_wall_cap_seconds", "consumed_cpu_seconds")},
        }
        args = SimpleNamespace(
            ledger=root / "v5.json", authorization=authorization_file,
            authorization_sha256=sha(authorization_file), source_checkpoint=checkpoint,
            source_checkpoint_sha256=sha(checkpoint), source_base_commit="f"*40,
            migrate_from_closed_v4=source, migrate_from_sha256=sha(source))
        checks = {}
        with patch.object(r, "validate_authorization", return_value=auth), \
                patch.object(r, "validate_prior_closed_ledger", return_value=prior_info), \
                patch.object(r, "frozen_v4_module", return_value=SimpleNamespace(validate_ledger=lambda v: None)):
            migrated = r.migrate_from_closed_v4(args)
            assert source.read_bytes() == source_bytes
            assert migrated["phases"] == old["phases"]
            for key in ("created_at_utc", "run_deadline_at_utc", "quantum_cpu_cap_seconds",
                        "run_wall_cap_seconds", "consumed_cpu_seconds", "remaining_quantum_cpu_seconds"):
                assert migrated[key] == old[key], key
            assert migrated["migration"]["new_budget_created"] is False
            assert migrated["phases"][1]["monitor_failed"] is True
            checks["complete_migration_preserves_all_old_rows_costs_caps_times_remaining"] = True
            r.validate_ledger(migrated)
            for label, mutate in (
                ("cap_reset", lambda v: v.update(quantum_cpu_cap_seconds=129601)),
                ("deadline_reset", lambda v: v.update(run_deadline_at_utc="2026-11-06T01:20:50.391742+00:00")),
                ("old_failure_relabel", lambda v: v["phases"][1].update(monitor_failed=False)),
                ("cost_reset", lambda v: v.update(consumed_cpu_seconds=0.)),
                ("old_phase_omitted", lambda v: v["phases"].pop(0)),
            ):
                bad = deepcopy(migrated)
                mutate(bad)
                try:
                    r.validate_ledger(bad)
                except (ValueError, RuntimeError, TypeError, KeyError):
                    checks[label+"_rejected"] = True
                else:
                    raise AssertionError("Invalid migration accepted: "+label)
            # A legitimate known new cost accumulates on the carried old total.
            advanced = deepcopy(migrated)
            advanced["phases"].append({"phase": "SYNTHETIC_NEW_PHASE", "status": "EXITED", "actual_cpu_seconds": 1.5})
            advanced.update(consumed_cpu_seconds=14., remaining_quantum_cpu_seconds=129586.)
            r.validate_ledger(advanced)
            checks["new_known_cost_accumulates_on_old_total_without_new_cap"] = True
            source.write_bytes(source_bytes+b" ")
            try:
                r.validate_ledger(migrated)
            except ValueError:
                checks["changed_original_ledger_bytes_rejected"] = True
            else:
                raise AssertionError("Changed original ledger accepted")
        return {"checks": checks, "scope": "PRIVATE_SYNTHETIC_FILES_WITH_MOCKED_AUTH_AND_PRIOR_VERIFIER",
                "actual_budget_created_or_changed": False}


def run(output):
    output = Path(output).resolve()
    output.mkdir()
    assert sha(SOURCE) == EXPECTED and sha(OLD) == OLD_SHA
    spec = importlib.util.spec_from_file_location("independent_v5_guard_review", SOURCE)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    old_functions, new_functions = source_functions(OLD), source_functions(SOURCE)
    names = ("parse_proc_stat", "read_process_identity", "identity_matches", "signal_group",
             "signal_bound_group", "actual_cpu_observation", "process_group_snapshot", "read_cgroup_memory",
             "resource_runtime_snapshot", "validate_prior_closed_ledger", "prior_budget_lock", "phase_limits")
    for name in names:
        assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False), name
    checks = pure_complete_migration(r)
    real_path = ROOT / "reports/pyscf-sdf-continuation-20261103-v1/cpu_budget.json"
    old_sha = sha(real_path)
    history = json.loads(real_path.read_text())
    total = sum(row["actual_cpu_seconds"] for row in history["phases"])
    assert len(history["phases"]) == 13 and abs(total-3605.273693) < 1e-7
    assert total == history["consumed_cpu_seconds"]
    assert history["quantum_cpu_cap_seconds"] == 129600
    assert history["remaining_quantum_cpu_seconds"] == 125994.726307
    assert history["run_deadline_at_utc"] == "2026-11-05T01:20:50.391742+00:00"
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                              "-p", "test_pyscf_actual_cpu_budget_v5.py", "-v"],
                             cwd=ROOT, capture_output=True, text=True)
    log = output / "pure_tests.log"
    log.write_text(process.stdout+process.stderr)
    assert process.returncode == 0, process.stdout+process.stderr
    assert sha(SOURCE) == EXPECTED and sha(real_path) == old_sha
    assert not any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules)
    receipt = {
        "status": "V5_GUARD_STATIC_AND_PURE_MIGRATION_REVIEW_PASS",
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "guard_sha256": EXPECTED,
        "old_guard_unchanged_sha256": OLD_SHA, "review_script_sha256": sha(Path(__file__)),
        "pure_test_exit_code": process.returncode, "pure_test_log_sha256": sha(log),
        "test_source_sha256": sha(ROOT / "tests/test_pyscf_actual_cpu_budget_v5.py"),
        "unchanged_AST_functions": list(names), "synthetic_complete_migration": checks,
        "actual_closed_V4_JSON_read_only": {"sha256": old_sha, "phase_count": 13,
            "actual_CPU_seconds": total, "remaining_CPU_seconds": history["remaining_quantum_cpu_seconds"],
            "cap_CPU_seconds": 129600, "deadline": history["run_deadline_at_utc"],
            "failed_monitor_rows_preserved": [row["phase"] for row in history["phases"] if row.get("monitor_failed")]},
        "new_quantum_evaluations": 0, "actual_ledger_migration_performed": False,
        "actual_process_signals_sent": False, "active_molecular_payloads_read": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "replay")
    run(parser.parse_args().output)
