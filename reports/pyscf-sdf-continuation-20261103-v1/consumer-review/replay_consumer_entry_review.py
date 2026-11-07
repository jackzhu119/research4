#!/usr/bin/env python3
"""Metadata/pure consumer review; integral/SCF/gradient entrypoints are forbidden.

This imports PySCF and builds tiny basis metadata but performs no electronic
structure calculation. Scientific restore-gradient/full-Hessian gates belong
to the root launcher and are not represented by this receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/pyscf_resumable_hessian_worker_20261103.py"
EXPECTED_SHA = "d23c4a3185aa4f666e22d671034728c1ca2443676a05fc8bec0587d6c8d6bf03"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise AssertionError("Electronic structure/integral call forbidden in metadata review")


def run(output):
    output = Path(output).resolve()
    output.mkdir()
    assert sha(SOURCE) == EXPECTED_SHA
    spec = importlib.util.spec_from_file_location("independent_consumer_replay", SOURCE)
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    worker = consumer.state_module()
    worker.install_sites()
    from pyscf import gto
    from pyscf.grad import rks as grad_rks
    from pyscf.hessian import rks as hess_rks

    molecule = gto.Mole()
    molecule.build(atom=[("O", (0., 0., 0.)), ("H", (0., 0., .96)),
                         ("H", (.91, 0., -.31))], unit="Angstrom", basis="def2-svp",
                   spin=0, charge=0, symmetry=False, verbose=0)
    with patch.object(gto.Mole, "intor", side_effect=forbidden), \
            patch.object(grad_rks.Gradients, "kernel", side_effect=forbidden):
        mf = worker.make_scf(molecule, worker.method_settings(), None)
        with patch.object(mf, "kernel", side_effect=forbidden):
            hobj = mf.Hessian()
            origins = consumer.stock_operator_gate(mf, hobj)
            assert type(hobj) is hess_rks.Hessian
            assert not mf.converged and mf.mo_coeff is None
            original = mf.get_hcore
            mf.get_hcore = lambda *args, **kwargs: None
            try:
                consumer.stock_operator_gate(mf, hobj)
            except ValueError:
                pass
            else:
                raise AssertionError("Custom hcore accepted")
            mf.get_hcore = original
            mf.with_x2c = object()
            try:
                consumer.stock_operator_gate(mf, hobj)
            except ValueError:
                pass
            else:
                raise AssertionError("X2C decorator accepted")
            del mf.with_x2c
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                              "-p", "test_pyscf_resumable_hessian_worker.py", "-v"],
                             cwd=ROOT, capture_output=True, text=True)
    log = output / "pure_tests.log"
    log.write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    assert sha(SOURCE) == EXPECTED_SHA
    receipt = {
        "status": "CONSUMER_ENTRY_METADATA_AND_PURE_REVIEW_PASS_REAL_QM_PENDING",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": EXPECTED_SHA, "review_script_sha256": sha(Path(__file__)),
        "test_sha256": sha(ROOT / "tests/test_pyscf_resumable_hessian_worker.py"),
        "pure_test_exit_code": process.returncode, "pure_test_log_sha256": sha(log),
        "actual_factory_metadata_gate": "PASS",
        "custom_hcore_and_x2c_rejection": "PASS", "stock_core_origins": origins,
        "SCF_gradient_integral_entrypoints_mocked_to_raise": True,
        "new_quantum_evaluations": 0,
        "scientific_restore_gradient_and_full_Hessian_gates": "NOT_RUN_BY_REVIEWER",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "replay")
    run(parser.parse_args().output)
