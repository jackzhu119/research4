#!/usr/bin/env python3
"""Pure controller review with fake phases and private lock files; no QM.

Actual global/cache locks are tested only under an isolated temporary ROOT.
The real research global lock and active molecular payloads are never touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/run_pyscf_serial_segments_20261103.py"
EXPECTED_SHA = "8c300c503842536af097289e3d1583def5ba3d9abd69c54479155d2bffd3926c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejected_lock(call):
    try:
        with call():
            raise AssertionError("Contending guard incorrectly acquired the lock")
    except BlockingIOError:
        return True


def run(output):
    output = Path(output).resolve()
    output.mkdir()
    assert sha(SOURCE) == EXPECTED_SHA
    spec = importlib.util.spec_from_file_location("independent_serial_review", SOURCE)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="research4-private-lock-review-") as directory:
        private_root = Path(directory)
        (private_root / ".runtime").mkdir()
        cache = private_root / "outputs/cache"
        global_path = private_root / ".runtime/pyscf-v4-global.lock"
        with patch.object(controller, "ROOT", private_root):
            with controller._closed_cache_guard(cache):
                checks["same_cache_contention_rejected"] = rejected_lock(
                    lambda: controller._closed_cache_guard(cache))
                checks["different_cache_still_excluded_by_global_guard"] = rejected_lock(
                    lambda: controller._closed_cache_guard(private_root / "outputs/other"))
                assert not (private_root / "outputs/other").exists()
            # Even when the global guard is free, a live helper's cache lock
            # must reject archival and release the newly acquired global lock.
            with (cache / "cache.lock").open("a") as helper:
                fcntl.flock(helper, fcntl.LOCK_EX | fcntl.LOCK_NB)
                checks["active_helper_cache_lock_rejected"] = rejected_lock(
                    lambda: controller._closed_cache_guard(cache))
                with global_path.open("a") as free_global:
                    fcntl.flock(free_global, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    checks["failed_cache_acquisition_releases_global_guard"] = True
            with controller._closed_cache_guard(cache):
                checks["locks_reacquire_after_normal_release"] = True
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                              "-p", "test_pyscf_serial_segments.py", "-v"],
                             cwd=ROOT, capture_output=True, text=True)
    log = output / "pure_tests.log"
    log.write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    assert sha(SOURCE) == EXPECTED_SHA
    assert not any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules)
    receipt = {
        "status": "SERIAL_CONTROLLER_STATIC_AND_PURE_REVIEW_PASS",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": EXPECTED_SHA, "review_script_sha256": sha(Path(__file__)),
        "test_sha256": sha(ROOT / "tests/test_pyscf_serial_segments.py"),
        "pure_test_exit_code": process.returncode, "pure_test_log_sha256": sha(log),
        "real_filesystem_lock_checks_in_private_ROOT": checks,
        "actual_research_global_lock_touched": False, "new_quantum_evaluations": 0,
        "supervisor_subprocesses_launched_by_review": 0,
        "active_molecular_HDF_or_NPY_read": False,
        "scientific_Hessian_or_mode_validation": "NOT_RUN_BY_REVIEWER",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "replay")
    run(parser.parse_args().output)
