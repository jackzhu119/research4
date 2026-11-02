#!/usr/bin/env python3
"""Source/hash and integer storage audit only: no PySCF import or QM execution."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve an earlier audit; use another output path")
    root = Path(__file__).resolve().parents[3]
    site = root / ".runtime/quantum-upgrade-site/pyscf"
    source_ranges = {
        "hessian/rhf.py": ["286-360"],
        "hessian/rks.py": ["45-84", "330-409"],
        "scf/cphf.py": ["105-145"],
        "scf/_response_functions.py": ["45-101"],
        "lib/linalg_helper.py": ["1221-1224", "1299-1375", "1411-1447"],
        "dft/numint.py": ["1418-1530", "2887-2928", "2930-2963"],
        "lib/numpy_helper.py": ["764-820"],
        "grad/rks.py": ["213-232"],
        "gto/eval_gto.py": ["26"],
        "lib/vhf/nr_direct.c": ["66-93", "238-264", "361-427", "442-465"],
        "lib/vhf/nr_direct.h": ["22"],
    }
    sources = {name: {"path": str((site / name).relative_to(root)),
                      "sha256": sha(site / name), "lines": lines}
               for name, lines in source_ranges.items()}
    natm, nao, nmo, nocc, itemsize, cycles = 75, 762, 762, 160, 8, 50
    as_limit = 12 * 1024**3
    full_h1 = natm * 3 * nao**2 * itemsize
    full_mo1 = natm * 3 * nao * nocc * itemsize
    full_moe1 = natm * 3 * nocc**2 * itemsize
    persistent = full_h1 + full_mo1 + full_moe1
    assert full_h1 == 1045159200 and persistent == 1310695200
    blocks = []
    for block in (2, 4, 8):
        rhs = block * 3
        b = rhs * nmo * nocc * itemsize
        dm = rhs * nao**2 * itemsize
        row = {
            "atoms_per_full_block": block, "rhs_per_full_block": rhs,
            "number_of_blocks": math.ceil(natm / block),
            "last_block_atoms": natm % block or block,
            "requested_tolerance_full_block": 1e-8 * block,
            "mo_rhs_block_bytes": b,
            "krylov_xs_plus_ax_50_no_deflation_bytes": 2 * cycles * b,
            "estimated_control_copies_10_rhs_blocks_bytes": 10 * b,
            "one_AO_density_batch_bytes": dm,
            "four_live_AO_batch_matrices_estimate_bytes": 4 * dm,
            "largest_reduced_krylov_h_bytes": (cycles * rhs)**2 * itemsize,
            "conditional_response_calls_at_K_iterations": {
                str(k): math.ceil(natm / block) * (k + 1) for k in (5, 8, 10)
            },
        }
        # These are budget assumptions, not measurements or allocator bounds.
        fixed = persistent + 2 * cycles * b + 10 * b + 4 * dm + 100000000 + 600000000
        row["native_thread_scenarios"] = {}
        for nt in (1, 2):
            subtotal = fixed + nt * 3205000000
            reserved = subtotal + 1024**3
            row["native_thread_scenarios"][str(nt)] = {
                "modeled_working_set_bytes": subtotal,
                "modeled_plus_one_GiB_unmodeled_reserve_bytes": reserved,
                "AS_margin_after_reserve_bytes": as_limit - reserved,
                "not_a_measured_peak_or_strict_upper_bound": True,
            }
        blocks.append(row)
    grid_points = 1200 * 56
    one_grid_ao = grid_points * nao * itemsize
    # During aow = _make_dR_dao_w(...), the old list is retained until RHS returns:
    # old3 AO components + new3 + one temporary scale_ao result may coexist.
    partial = {
        "conditional_grid_points_if_ngrids_large_and_memory_allows": grid_points,
        "one_grid_AO_component_bytes": one_grid_ao,
        "full_vmat_bytes": natm * 3 * 3 * nao**2 * itemsize,
        "AO_second_derivative_10_components_bytes": 10 * one_grid_ao,
        "ao_dm0_4_components_bytes": 4 * one_grid_ao,
        "old_new_aow_plus_temporary_7_components_bytes": 7 * one_grid_ao,
    }
    partial["visible_array_live_set_estimate_bytes"] = sum(partial[k] for k in (
        "full_vmat_bytes", "AO_second_derivative_10_components_bytes",
        "ao_dm0_4_components_bytes", "old_new_aow_plus_temporary_7_components_bytes"))
    partial["plus_runtime_600MB_and_unmodeled_1GiB_bytes"] = (
        partial["visible_array_live_set_estimate_bytes"] + 600000000 + 1024**3)
    payload = {
        "audit_created_UTC": datetime.now(timezone.utc).isoformat(),
        "status": "DONE_SOURCE_ONLY_NOT_A_QUANTUM_RESULT",
        "physical_backend": "SCREENED_DIRECT_RKS_V4",
        "frozen_V6_sha256": sha(root / "scripts/pyscf_spinosin_memory_bounded_direct_worker_20261031.py"),
        "quantum_jobs_started": 0, "active_checkpoints_read": 0,
        "dimensions": {"natom": natm, "nao": nao, "nmo": nmo, "nocc": nocc,
                       "float_item_bytes": itemsize, "max_cycle": cycles},
        "AS_limit_bytes": as_limit,
        "persistent_full_H1AO_MO1_MOE1_bytes": {
            "H1AO": full_h1, "MO1": full_mo1, "MOE1": full_moe1, "total": persistent},
        "memory_model_assumptions": {
            "native_driver_per_thread_bytes": 3205000000,
            "ground_XC_cache_and_grid_bookkeeping_bytes": 100000000,
            "runtime_library_baseline_bytes": 600000000,
            "additional_allocator_BLAS_unmodeled_reserve_bytes": 1024**3,
            "all_assumptions_are_estimates_not_measured_peak": True},
        "blocks": blocks,
        "separate_partial_GGA_Hessian_phase_risk": partial,
        "performance_input": {
            "source": "Parent-reported actual Spinosin direct1 SCF timings; not remeasured by this audit",
            "initial_SCF_5_cycles_plus_extra_approx_CPU_minutes": 20,
            "one_native_J_approx_CPU_minutes": 3,
            "caveat": "SCF J is not measured CPHF J; per-call cost and Krylov rank vary with batch size"},
        "recommendations": {
            "B8_native1_CPHF": "CONDITIONAL_BOUNDED_PILOT_AFTER_9_ATOM_ENGINEERING_GATE",
            "B8_native2_CPHF": "DO_NOT_DEFAULT_INSUFFICIENT_UNMEASURED_MARGIN",
            "whole_Spinosin_Hessian": "BLOCKED_PENDING_SEPARATE_GRID_BLOCK_MEMORY_GATE",
            "V7_implementation": "NOT_REQUESTED_NOT_CREATED",
            "restore": "SAVE_ONLY_CROSS_PROCESS_RESTORE_NOT_IMPLEMENTED"},
        "sources": sources,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "quantum_jobs_started": 0, "active_checkpoints_read": 0}))


if __name__ == "__main__":
    main()
