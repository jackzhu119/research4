#!/usr/bin/env python3
"""Review the final bounded watchdog with pure/mock tests only; no QM.

Preserves the earlier controller receipt and repeats the complete private-lock
and pure-test review for the final frozen revision.
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED = "d0a67096215102a6e7b46ce9510e1eee6e3e81517de5f3e09e5f86f44c13bd6e"
SUPERVISOR_SHA = "354ff51b5c2af2810033656d9972948262cfebb4dabb3978fca2d021f97e2b8f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    supervisor = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261101.py"
    assert sha(supervisor) == SUPERVISOR_SHA
    tree = ast.parse(supervisor.read_text())
    child = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "child_limits")
    calls = [n for n in ast.walk(child) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "prctl"]
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[0]) == 1
    assert isinstance(calls[0].args[1], ast.Attribute)
    assert calls[0].args[1].attr == "SIGKILL"
    assert any(isinstance(n, ast.Attribute) and n.attr == "getppid" for n in ast.walk(child))
    spec = importlib.util.spec_from_file_location("independent_final_serial_replay",
                                                HERE / "replay_serial_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_SHA = EXPECTED
    output = HERE / "watchdog-final"
    module.run(output)
    receipt = {
        "status": "FINAL_CONTROLLER_WATCHDOG_STATIC_AND_PURE_REVIEW_PASS",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "controller_source_sha256": EXPECTED,
        "wrapper_script_sha256": sha(Path(__file__)),
        "unchanged_supervisor_sha256": SUPERVISOR_SHA,
        "PDEATHSIG_SIGKILL_and_parent_PID_gate_statically_verified": True,
        "watchdog_scope": "Own Popen supervisor, wall allowance+2s, TERM/wait3s then KILL/wait3s, failure prevents automatic continuation",
        "watchdog_execution_test": "Mock Popen and clock only",
        "new_quantum_evaluations": 0,
        "actual_research_global_lock_touched": False,
        "active_molecular_HDF_or_NPY_read": False,
    }
    (output / "watchdog_review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
