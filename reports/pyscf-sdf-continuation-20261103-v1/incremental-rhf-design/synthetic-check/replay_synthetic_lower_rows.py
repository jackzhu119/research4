#!/usr/bin/env python3
"""Synthetic algebra only: no PySCF import, integral, SCF, gradient or Hessian.

The frozen V9 function body is exercised with a fake integral provider. This
checks row-state arithmetic and serialization, not real electronic structure,
the PySCF shell-slice implementation, or cross-process quantum restoration.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import types

import numpy as np


V9_SHA256 = "e3215e563b4429212084cfccbcec98e6a9f482bd8c3352d53d3064402649b450"
V9_RELATIVE = Path("scripts/pyscf_spinosin_pure_j_worker_20261101.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint64),
        np.ascontiguousarray(b).view(np.uint64),
    )


def roundtrip(array: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    buffer.seek(0)
    return np.load(buffer, allow_pickle=False)


class QuietLog:
    def timer_debug1(self, *args):
        return (0.0, 0.0)

    def timer(self, *args):
        return None


def frozen_body(source: Path):
    if sha(source) != V9_SHA256:
        raise ValueError("Frozen V9 source hash differs")
    node = next(
        item for item in ast.parse(source.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "pure_j_partial_body"
    )
    namespace = {
        "np": np,
        "require_plain_pbe": lambda h: None,
        "require_none_derivative_abi": lambda r: None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 "<frozen-v9-synthetic-algebra-only>", "exec"), namespace)
    return namespace["pure_j_partial_body"]


def fixture(seed: int, zero_integrals: bool = False):
    rng = np.random.default_rng(seed)
    widths = [2, 1, 3, 2, 1]
    n, nao = len(widths), sum(widths)
    cuts = np.r_[0, np.cumsum(widths)]
    mo = np.linalg.qr(rng.normal(size=(nao, nao)))[0]
    occ = np.r_[np.full(4, 2.0), np.zeros(nao - 4)]
    energies = np.sort(rng.normal(size=nao))

    def values(shape):
        value = rng.normal(size=shape)
        if zero_integrals:
            value[:] = 0.0
            value.ravel()[::2] = -0.0
        return value

    global_vj = values((9, nao, nao))
    vj1 = [values((9, nao, nao)) for _ in range(n)]
    vj2 = [values((9, widths[i], nao)) for i in range(n)]
    saa = values((3, 3, nao, nao))
    sab = values((3, 3, nao, nao))
    core = values((n, n, 3, 3, nao, nao))
    mol = types.SimpleNamespace(
        natm=n, nbas=n,
        aoslice_by_atom=lambda: np.array(
            [[i, i + 1, cuts[i], cuts[i + 1]] for i in range(n)]),
    )
    mf = types.SimpleNamespace(mo_coeff=mo, mo_occ=occ, mo_energy=energies)
    hobj = types.SimpleNamespace(
        mol=mol, base=mf,
        hcore_generator=lambda m: lambda i, j: core[i, j].copy(),
    )

    def fake_get_jk(m, intor, *args, **kwargs):
        if intor == "int2e_ipip1":
            sliced = kwargs.get("shls_slice")
            if sliced is None:
                return [global_vj.copy()]
            i = int(sliced[0])
            # This merely slices an arbitrary precomputed synthetic tensor.
            # It does NOT validate a real libcint/PySCF shell-slice operation.
            return [global_vj[:, cuts[i]:cuts[i + 1], :].copy()]
        i = int(kwargs["shls_slice"][0])
        return [(vj1 if intor == "int2e_ip1ip2" else vj2)[i].copy()]

    rhf = types.SimpleNamespace(
        logger=types.SimpleNamespace(
            new_logger=lambda *a: QuietLog(),
            process_clock=lambda: 0.0, perf_counter=lambda: 0.0),
        get_ovlp=lambda m: (saa.copy(), sab.copy(), None),
        _get_jk=fake_get_jk,
    )
    mocc = mo[:, occ > 0]
    dm = np.dot(mocc, mocc.T) * 2
    dme = np.einsum("pi,qi,i->pq", mocc, mocc, energies[occ > 0]) * 2

    def global_diagonal(partitioned: bool):
        terms = []
        full = None if partitioned else global_vj.reshape(3, 3, nao, nao)
        for i in range(n):
            p0, p1 = cuts[i:i + 2]
            if partitioned:
                shls = (i, i + 1) + (0, n) * 3
                value = fake_get_jk(mol, "int2e_ipip1", 9, "s2kl",
                                    ["lk->s1ij", dm], shls_slice=shls)[0]
                block = value.reshape(3, 3, p1 - p0, nao)
            else:
                block = full[:, :, p0:p1]
            terms.append(np.einsum("xypq,pq->xy", block, dm[p0:p1]) * 2)
        return np.stack(terms)

    def lower_row(i: int, diagonal: np.ndarray):
        p0, p1 = cuts[i:i + 2]
        value = vj1[i].copy()
        value[:, :, p0:p1] += vj2[i].transpose(0, 2, 1) * 0.5
        value = value.reshape(3, 3, nao, nao)
        e1 = np.zeros((i + 1, 3, 3))
        ej = np.zeros_like(e1)
        # Keep zero += diagonal, then the original per-cell operation order.
        ej[i] += diagonal[i]
        e1[i] -= np.einsum("xypq,pq->xy", saa[:, :, p0:p1], dme[p0:p1]) * 2
        for j in range(i + 1):
            q0, q1 = cuts[j:j + 2]
            ej[j] += np.einsum("xypq,pq->xy", value[:, :, q0:q1], dm[q0:q1]) * 4
            e1[j] -= np.einsum("xypq,pq->xy", sab[:, :, p0:p1, q0:q1],
                               dme[p0:p1, q0:q1]) * 2
            e1[j] += np.einsum("xypq,pq->xy", core[i, j], dm)
        return e1, ej

    def restore(shards, diagonal, expected_binding, actual_binding):
        if expected_binding != actual_binding:
            raise ValueError("Synthetic binding marker differs")
        if diagonal.shape != (n, 3, 3) or diagonal.dtype != np.float64:
            raise ValueError("Wrong synthetic diagonal shape/dtype")
        if not np.isfinite(diagonal).all():
            raise ValueError("Nonfinite synthetic diagonal")
        if [item[0] for item in shards] != list(range(len(shards))):
            raise ValueError("Rows must be a unique contiguous lower-row prefix")
        e1 = np.zeros((n, n, 3, 3))
        ej = np.zeros_like(e1)

        def assign(i, a, b):
            if a.shape != (i + 1, 3, 3) or b.shape != a.shape:
                raise ValueError("Wrong row shape")
            if a.dtype != np.float64 or b.dtype != np.float64:
                raise ValueError("Wrong row dtype")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite row")
            e1[i, :i + 1] = a
            ej[i, :i + 1] = b
            for j in range(i):
                e1[j, i] = e1[i, j].T
                ej[j, i] = ej[i, j].T

        for i, a, b in shards:
            assign(i, a, b)
        for i in range(len(shards), n):
            assign(i, *lower_row(i, diagonal))
        return e1, ej, np.zeros_like(e1)

    return n, nao, rhf, hobj, global_diagonal, lower_row, restore


def run(repository: Path):
    if any(name == "pyscf" or name.startswith("pyscf.") for name in sys.modules):
        raise RuntimeError("No PySCF import is allowed in this synthetic check")
    source = repository / V9_RELATIVE
    body = frozen_body(source)
    cases = []
    comparisons = mutation_checks = rejected = 0
    maxdiff = 0.0
    for label, zero in [("mixed_random_float64", False),
                        ("signed_zero_integral_fixture", True)]:
        n, nao, rhf, hobj, diagonal_fn, row_fn, restore = fixture(20261103, zero)
        reference = body(rhf, hobj)
        global_diag = diagonal_fn(False)
        sliced_diag = diagonal_fn(True)
        if not bitwise_equal(global_diag, sliced_diag):
            raise AssertionError("Synthetic first-AO partition differs")
        per_case = 0
        for partitioned, diagonal in [(False, global_diag), (True, sliced_diag)]:
            rows = [row_fn(i, diagonal) for i in range(n)]
            for prefix in range(n + 1):
                shards = [(i, roundtrip(rows[i][0]), roundtrip(rows[i][1]))
                          for i in range(prefix)]
                restored_diag = roundtrip(diagonal)
                before = [payload_sha(x) for _, a, b in shards for x in (a, b)]
                diag_before = payload_sha(restored_diag)
                actual = restore(shards, restored_diag, "test-binding", "test-binding")
                for a, b in zip(actual, reference):
                    comparisons += 1
                    maxdiff = max(maxdiff, float(np.max(np.abs(a - b))))
                    if not bitwise_equal(a, b):
                        raise AssertionError("Restored synthetic return differs in float64 bits")
                # Real RKS mutates returned e1 in place. Check no cache alias.
                actual[0][:] += actual[1]
                if before != [payload_sha(x) for _, a, b in shards for x in (a, b)]:
                    raise AssertionError("Returned matrices alias synthetic row caches")
                if diag_before != payload_sha(restored_diag):
                    raise AssertionError("Returned matrices alias synthetic diagonal cache")
                mutation_checks += 1
                per_case += 1
        rows = [row_fn(i, global_diag) for i in range(n)]
        wrong_shape = np.zeros((2, 3, 3))
        nonfinite = rows[0][0].copy(); nonfinite[0, 0, 0] = np.nan
        bad = [
            ([(1, *rows[1])], global_diag, "x", "x"),
            ([(0, *rows[0]), (0, *rows[0])], global_diag, "x", "x"),
            ([(1, *rows[1]), (0, *rows[0])], global_diag, "x", "x"),
            ([], global_diag[:1], "x", "x"),
            ([], global_diag.astype(np.float32), "x", "x"),
            ([], global_diag, "x", "changed"),
            ([(0, wrong_shape, wrong_shape.copy())], global_diag, "x", "x"),
            ([(0, nonfinite, rows[0][1])], global_diag, "x", "x"),
            ([(0, rows[0][0].astype(np.float32), rows[0][1])], global_diag, "x", "x"),
        ]
        for arguments in bad:
            try:
                restore(*arguments)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("A malformed synthetic contract was accepted")
        cases.append({"fixture": label, "synthetic_atoms": n, "synthetic_AO": nao,
                      "global_vs_first_AO_sliced_diagonal_bitwise_equal": True,
                      "prefix_restorations_checked": per_case,
                      "negative_contracts_rejected": len(bad)})
    if any(name == "pyscf" or name.startswith("pyscf.") for name in sys.modules):
        raise AssertionError("Unexpected PySCF import")
    return {
        "status": "SYNTHETIC_ALGEBRA_ONLY_NOT_PYSCF_VALIDATION",
        "source_V9_path": str(source), "source_V9_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "python_version": sys.version, "numpy_version": np.__version__,
        "cases": cases,
        "total_prefix_restorations_checked": sum(x["prefix_restorations_checked"] for x in cases),
        "return_array_bitwise_comparisons": comparisons,
        "maximum_e1_ej_ek_absolute_difference": maxdiff,
        "returned_buffer_cache_nonalias_checks": mutation_checks,
        "negative_contracts_rejected": rejected,
        "binding_check_scope": "Opaque synthetic marker only; no actual MO/grid restoration implemented",
        "first_AO_slice_scope": "Partition of an arbitrary precomputed tensor only; not libcint/PySCF shell-slice validation",
        "cross_process_quantum_restore": "NOT_RUN",
        "new_quantum_evaluations": 0, "pyscf_imported": False,
        "old_scientific_files_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior synthetic-check receipts; choose a fresh output")
    started = datetime.now(timezone.utc).isoformat()
    result = run(args.repository.resolve())
    result.update(started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
