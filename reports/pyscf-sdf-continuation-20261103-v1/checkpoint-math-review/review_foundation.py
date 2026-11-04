#!/usr/bin/env python3
"""Independent tiny NumPy/static review; never imports PySCF or runs QM.

This verifies only the partition identity and the existing synthetic algebra
replay. It does not validate libcint, an implemented restart, or spectroscopy.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RHF = ROOT / ".runtime/quantum-upgrade-site/pyscf/hessian/rhf.py"
VHF = ROOT / ".runtime/quantum-upgrade-site/pyscf/scf/_vhf.py"
RHF_SHA = "2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    if any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules):
        raise RuntimeError("No PySCF import is permitted in this review")
    if sha(RHF) != RHF_SHA:
        raise RuntimeError("Reference RHF source differs")
    tree = ast.parse(RHF.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_partial_hess_ejk")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "_get_jk"
                and isinstance(n.args[1], ast.Constant)
                and n.args[1].value == "int2e_ipip1")
    assert ast.literal_eval(call.args[2]) == 9
    assert ast.literal_eval(call.args[3]) == "s2kl"
    assert isinstance(call.args[4], ast.List)
    assert ast.literal_eval(call.args[4].elts[0]) == "lk->s1ij"
    assert isinstance(call.args[4].elts[1], ast.Name)
    assert call.args[4].elts[1].id == "dm0"

    # Four-index synthetic ERI, independently contracted in the global and
    # first-index-sliced forms. The k/l tensor symmetry is the s2kl assumption.
    rng = np.random.default_rng(20261103)
    nao = 5
    eri = rng.normal(size=(9, nao, nao, nao, nao))
    eri = (eri + eri.swapaxes(-1, -2)) / 2
    cocc = rng.normal(size=(nao, 2))
    dm = 2 * cocc @ cocc.T
    global_j = np.einsum("cijkl,lk->cij", eri, dm, optimize=False)
    cuts = [0, 1, 3, 5]
    comparisons = []
    for atom, (p0, p1) in enumerate(zip(cuts[:-1], cuts[1:])):
        local_j = np.einsum("cijkl,lk->cij", eri[:, p0:p1], dm,
                            optimize=False)
        global_diag = 2 * np.einsum("cij,ij->c", global_j[:, p0:p1], dm[p0:p1])
        local_diag = 2 * np.einsum("cij,ij->c", local_j, dm[p0:p1])
        err_j = float(np.max(np.abs(local_j - global_j[:, p0:p1])))
        err_diag = float(np.max(np.abs(local_diag - global_diag)))
        assert err_j <= 1e-12 and err_diag <= 1e-12
        comparisons.append({"synthetic_atom": atom, "AO_range": [p0, p1],
                            "J_max_abs_difference": err_j,
                            "precontracted_diagonal_max_abs_difference": err_diag})

    replay_path = (ROOT / "reports/pyscf-sdf-continuation-20261103-v1/"
                   "incremental-rhf-design/synthetic-check/replay_synthetic_lower_rows.py")
    spec = importlib.util.spec_from_file_location("independent_closed_array_replay", replay_path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    result = replay.run(ROOT)
    assert result["return_array_bitwise_comparisons"] == 72
    assert result["returned_buffer_cache_nonalias_checks"] == 24
    assert result["negative_contracts_rejected"] == 18
    assert result["new_quantum_evaluations"] == 0
    assert not result["pyscf_imported"]
    if any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules):
        raise RuntimeError("Unexpected PySCF import")
    return {
        "status": "FOUNDATION_PASS_IMPLEMENTATION_AND_REAL_QUANTUM_GATES_PENDING",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "Static source and tiny synthetic NumPy only",
        "reference_rhf_sha256": sha(RHF),
        "reference_vhf_sha256": sha(VHF),
        "review_script_sha256": sha(Path(__file__)),
        "reviewed_synthetic_replay_sha256": sha(replay_path),
        "full_ERI_partition_checks": comparisons,
        "lower_row_replay": result,
        "new_quantum_evaluations": 0,
        "libcint_first_shell_partition_validation": "NOT_RUN",
        "production_checkpoint_implementation_review": "PENDING",
        "cross_process_exact_MF_quantum_restore": "NOT_RUN",
        "spinosin_vibrational_frequencies": "NOT_COMPUTED",
        "IR_intensities": "NOT_COMPUTED",
        "new_DFT_method_or_speedup_claim": False,
    }


if __name__ == "__main__":
    output = Path(__file__).with_name("foundation_review.json")
    if output.exists():
        raise FileExistsError("Preserve the first closed review receipt")
    output.write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n")
    print(output)
