#!/usr/bin/env python3
"""Independent restart review with tiny synthetic arrays, without any QM.

Six actual Python processes exercise file persistence and lock release, but
every MF, integral and source-check callback is a synthetic fixture. This is
NOT a cross-process PySCF restoration or a molecular vibrational calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MODULE = ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py"
TEST = ROOT / "tests/test_pyscf_incremental_rhf_checkpoint.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_module():
    spec = importlib.util.spec_from_file_location("independent_checkpoint_fixture", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child(cache):
    fixture = fixture_module()
    inc = fixture.inc
    f = fixture.Fixture()
    patches = (patch.object(inc, "_source_identity", return_value={"source": "SYNTHETIC_ONLY"}),
               patch.object(inc, "_check_stock_callable", return_value=None))
    with patches[0], patches[1]:
        context = inc.IncrementalPureJContext(
            f.hobj, cache, {"snapshot": "INDEPENDENT_SYNTHETIC_FIXTURE_ONLY"},
            max_new_units=1, rhf_module=f.rhf,
            execution_evidence={"scope": "SYNTHETIC_ONLY_NOT_SUPERVISED_QUANTUM"})
        try:
            with context:
                result = f.rhf._partial_hess_ejk(f.hobj, with_k=False)
        except inc.CheckpointSegmentComplete:
            status = "SYNTHETIC_SEGMENT_SAVED_NOT_FULL_RHF"
            maxdiff = None
        else:
            status = "SYNTHETIC_FULL_RHF_ARRAYS_MATCH_NOT_QUANTUM"
            reference = f.reference()
            maxdiff = max(float(np.max(np.abs(a - b))) for a, b in zip(result, reference))
            for a, b in zip(result, reference):
                np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-13)
        assert f.rhf._partial_hess_ejk is f.original
        assert not any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules)
    return {"status": status, "progress": context.progress(),
            "helper_restored": True, "new_quantum_evaluations": 0,
            "final_array_max_abs_difference": maxdiff}


def parent():
    source_sha, test_sha = sha(MODULE), sha(TEST)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p",
         "test_pyscf_incremental_rhf_checkpoint.py", "-v"],
        cwd=ROOT, capture_output=True, text=True)
    log = OUT / "incremental_independent_tests.log"
    with log.open("x") as handle:
        handle.write(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    stages = []
    with tempfile.TemporaryDirectory(prefix="independent-pure-cross-process-", dir=ROOT / "outputs") as temporary:
        cache = Path(temporary) / "cache"
        for index in range(6):
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--child", str(cache)], cwd=ROOT,
                                       capture_output=True, text=True)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            data = json.loads(completed.stdout)
            assert data["progress"]["total_units_completed"] == index + 1
            assert data["progress"]["new_units_this_context"] == 1
            assert data["progress"]["complete_RHF_helper"] is (index == 5)
            assert data["new_quantum_evaluations"] == 0
            stages.append({"process_number": index + 1, "exit_code": completed.returncode,
                           "stdout": data, "stderr": completed.stderr})
        shutil.copytree(cache, OUT / "synthetic-cross-process-cache")
    assert source_sha == sha(MODULE) and test_sha == sha(TEST), "Review source changed"
    fmod = fixture_module()
    pins = {name: sha(ROOT / ".runtime/quantum-upgrade-site/pyscf" / name)
            for name in fmod.inc.PINNED}
    assert pins == fmod.inc.PINNED
    result = {
        "status": "STATIC_AND_SYNTHETIC_IMPLEMENTATION_REVIEW_PASS_REAL_QM_GATES_PENDING",
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - timer,
        "review_script_sha256": sha(Path(__file__)), "incremental_module_sha256": source_sha,
        "test_module_sha256": test_sha, "pure_unittest_exit_code": process.returncode,
        "pure_unittest_log_sha256": sha(log), "actual_pinned_source_hashes": pins,
        "six_separate_python_processes_synthetic_fixture_only": stages,
        "new_quantum_evaluations": 0, "PySCF_cross_process_MF_restore": "NOT_RUN",
        "water_and_ethanol_full_Hessian_mode_gate": "NOT_RUN",
        "spinosin_frequency_calculation": "NOT_RUN",
        "scope": "RHF pre-XC engineering implementation only, no scientific-result release",
        "producer_unit_supervision_validation_owner": "Root consumer/controller",
    }
    with (OUT / "incremental_implementation_review.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "module_sha256": source_sha,
                      "separate_synthetic_processes": len(stages),
                      "max_final_array_difference": stages[-1]["stdout"]["final_array_max_abs_difference"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path)
    arguments = parser.parse_args()
    if arguments.child:
        print(json.dumps(child(arguments.child), sort_keys=True))
    else:
        parent()
