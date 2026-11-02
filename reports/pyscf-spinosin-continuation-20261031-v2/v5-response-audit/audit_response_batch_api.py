"""Read source/constants and integer array sizes; never call quantum methods."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SITE = ROOT / ".runtime/quantum-upgrade-site"
sys.path.insert(0, str(SITE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor(shape):
    number = math.prod(shape) * 8
    return {"shape": list(shape), "dtype": "float64", "bytes": number, "GiB": number / 2 ** 30}


def lines(path, begin, end):
    return [{"line": n, "text": text} for n, text in enumerate(Path(path).read_text().splitlines(), 1)
            if begin <= n <= end]


def run():
    target = OUT / "public_api_response_review.json"
    if target.exists():
        raise FileExistsError("Preserve this read-only method snapshot")
    # Imports inspect defaults only; no molecule, SCF, integral or response call.
    from pyscf.lib import linalg_helper
    from pyscf.scf.hf import SCF
    native_tol = float(SCF.conv_tol_cpscf)
    krylov_signature = inspect.signature(linalg_helper.krylov)
    lindep = float(krylov_signature.parameters["lindep"].default)
    krylov_memory = float(krylov_signature.parameters["max_memory"].default)
    rhf = SITE / "pyscf/hessian/rhf.py"
    text = rhf.read_text()
    for expected in ("mo1s = [None] * mol.natm", "e1s = [None] * mol.natm", "mo1s[ia] = mo1[k]",
                     "e1s[ia] = e1[k].reshape(3,nocc,nocc)", "h1vo.append(_ao2mo(h1ao[ia]))",
                     "tol = mf.conv_tol_cpscf * (ia1 - ia0)", "vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)",
                     "dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)"):
        assert expected in text, expected
    sources = {}
    for relative in ("pyscf/hessian/rhf.py", "pyscf/hessian/rks.py", "pyscf/scf/cphf.py", "pyscf/lib/linalg_helper.py", "pyscf/scf/hf.py"):
        path = SITE / relative
        snapshot = OUT / "source_snapshots" / relative.replace("/", "__")
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources[relative] = {"sha256": sha(path), "snapshot": str(snapshot.relative_to(OUT))}
    natm, nao, nmo, nocc = 75, 762, 762, 160
    h1ao = tensor((natm, 3, nao, nao))
    mo1 = tensor((natm, 3, nao, nocc))
    moe1 = tensor((natm, 3, nocc, nocc))
    partial = tensor((natm, natm, 3, 3))
    source = {
        "status": "DESIGN_SOURCE_REVIEW_PASS_PENDING_REAL_V4_V5_WATER_AND_TARGET_VALIDATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "new_quantum_jobs": 0,
        "integral_or_response_operator_calls": 0, "active_checkpoint_read": False,
        "pyscf_version": importlib.metadata.version("pyscf"), "script_sha256": sha(__file__),
        "public_API": {
            "call": "pyscf.hessian.rhf.solve_mo1 or the original HessianBase.solve_mo1 bound method",
            "return_type": "two lists, each length mol.natm; entries indexed by global atom ID",
            "unrequested_entries": None,
            "per_requested_atom_mo1_shape": [3, nao, nocc], "per_requested_atom_moe1_shape": [3, nocc, nocc],
            "atmlst_meaning": "Select nuclear-perturbation RHS, not an electronic subsystem or AO/MO truncation",
            "h1ao_argument": "Keep full globally indexed h1ao; do not pass a short atom-sublist",
            "response_operator": "gen_vind uses complete global mo_coeff/mocc, full AO dm1, and the full mean-field vresp",
            "correct_assembly": "Merge all requested mo1/moe1 global-index entries, return complete lists to unchanged hess_elec over all75 atoms",
            "cross_coupling": "Original hess_elec loops every ia,ja pair and contracts h1ao[ia] with global mo1[ja]/moe1[ja]",
            "forbidden_shortcut": "Do not call kernel(atmlst=smallblock) and join diagonal Hessian blocks; that discards cross-atom terms"},
        "numerical_stopping_policy": {
            "actual_imported_class_default_conv_tol_cpscf": native_tol,
            "V5_explicit_requested_conv_tol_cpscf": 1e-8,
            "per_native_subbatch_requested_tol_formula": "mf.conv_tol_cpscf * number_of_atoms_in_that_subbatch",
            "requested_tol_for_two_atoms": 2e-8, "requested_tol_for_last_single_atom": 1e-8,
            "nominal_full75_requested_tol": 75e-8,
            "actual_imported_krylov_default_lindep": lindep,
            "actual_imported_krylov_default_max_memory_MB": krylov_memory,
            "break_condition": "max_innerprod < lindep OR max_innerprod < tol**2",
            "final_response_equation_residual_independently_measured": False,
            "warning": "max_innerprod is an internal Krylov trial-vector norm criterion, not a recorded independent final equation residual. Return alone does not establish measured residual<=requested tol.",
            "not_converged_action": "lib.krylov raises RuntimeError when max_cycle ends without stopping; do not swallow or call that a complete block",
            "expected_finite_precision_difference": "Smaller RHS batches change the Krylov subspace/numerical stopping path; actual paired V4/V5 water validation is required despite mathematical linear-response equivalence"},
        "sizing_assumptions_not_a_measured_peak": {"atoms": natm, "AO": nao,
            "MO_assumed_full_rank": nmo, "occupied_closed_shell_orbitals": nocc,
            "active_MO_arrays_not_read": True},
        "unavoidable_global_results_for_unchanged_assembly": {"H1_AO": h1ao, "MO1_AO": mo1,
            "MO_energy1": moe1, "partial_hessian": partial,
            "numeric_payload_bytes_sum": h1ao["bytes"]+mo1["bytes"]+moe1["bytes"]+partial["bytes"],
            "not_a_total_peak_memory_or_disk_measurement": True},
        "batch_workspace_scaling": {
            "two_atom_block_count_for75": 38,
            "full_batch_rhs": 225, "two_atom_rhs": 6, "last_single_atom_rhs": 3,
            "full_raw_Krylov_RHS_or_vector_batch": tensor((225, nmo, nocc)),
            "two_atom_raw_Krylov_RHS_or_vector_batch": tensor((6, nmo, nocc)),
            "full_dm1_or_vresp_output": tensor((225, nao, nao)),
            "two_atom_dm1_or_vresp_output": tensor((6, nao, nao)),
            "native_min_memory_floor_MB": 2000,
            "native_blocksize_lower_bound_at_fullrank762x160": int(2000e6/8/(nmo*nocc*3*6)),
            "hypothetical_50_iteration_two_Krylov_lists_no_rank_deflation_full_bytes": 2*50*225*nmo*nocc*8,
            "hypothetical_50_iteration_two_Krylov_lists_no_rank_deflation_two_atom_bytes": 2*50*6*nmo*nocc*8,
            "warning": "QR rank deflation/convergence affect retained vectors. These are dimension-based workspace scales, not measured peaks or guaranteed allocations. Blocking need not improve total CPU."},
        "save_only_provenance_requirements": [
            "Bind exact geometry (including atom order), settings, PySCF version, worker SHA, MO coefficient/energy/occupation array dtype/shape/byte SHA.",
            "Keep globally indexed atom IDs for every completed response block; reject missing/duplicate/out-of-range entries before final assembly.",
            "Save partial Hessian, H1 and completed response arrays to ignored outputs with report-side hashes. Avoid stacking/copying all H1 solely to save it.",
            "Mark cross-process restoration NOT_IMPLEMENTED_SAVE_ONLY; no load routine or Hessian-resume command is justified by merely saved blocks.",
            "Preserve original H1/partial/RKS assembly semantics. Saved source energy/gradient/method must not be relabelled for a different model."],
        "evidence": {
            "original_cross_atom_Hessian_assembly": lines(rhf, 39, 91),
            "public_solve_global_indices": lines(rhf, 286, 341),
            "global_response_operator": lines(rhf, 343, 361),
            "public_bound_wrapper_keeps_maxcycle_levelshift": lines(rhf, 545, 551),
            "native_Krylov_storage_and_stopping": lines(SITE / "pyscf/lib/linalg_helper.py", 1290, 1345)},
        "source_files": sources}
    target.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": source["status"], "global_required_array_payload_bytes": source["unavoidable_global_results_for_unchanged_assembly"]["numeric_payload_bytes_sum"],
                      "nominal_block_tol": 2e-8, "native_lindep": lindep, "quantum_jobs": 0}))


if __name__ == "__main__":
    run()
