"""Source/API/size audit only: never evaluate integrals or create wavefunctions."""
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
for site in (".runtime/quantum-upgrade-site", ".runtime/quantum-pilot-site"):
    sys.path.insert(0, str(ROOT / site))
import pyscf
from pyscf.hessian import rhf, rks
from pyscf.lib import linalg_helper
from pyscf.scf import cphf, hf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_row(name, shape):
    size = math.prod(shape) * 8
    return {"name": name, "shape": list(shape), "dtype": "float64",
            "payload_bytes": size, "MiB": size / 2**20, "GiB": size / 2**30}


sources = {}
for module in (rhf, rks, cphf, linalg_helper):
    path = Path(inspect.getsourcefile(module))
    sources[module.__name__] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
functions = {"rhf.hess_elec": rhf.hess_elec, "rks.partial_hess_elec": rks.partial_hess_elec,
             "rks.make_h1": rks.make_h1, "rhf.solve_mo1": rhf.solve_mo1,
             "HessianBase.solve_mo1": rhf.HessianBase.solve_mo1,
             "HessianBase.kernel": rhf.HessianBase.kernel, "cphf.solve": cphf.solve,
             "cphf.solve_withs1": cphf.solve_withs1, "lib.krylov": linalg_helper.krylov}
signatures = {name: {"signature": str(inspect.signature(func)), "start_line": inspect.getsourcelines(func)[1]}
              for name, func in functions.items()}
assert rks.Hessian.hess_elec is rhf.hess_elec
assert all(name in inspect.signature(rhf.hess_elec).parameters for name in ("h1ao", "mo1", "mo_e1"))
assert not any(name in inspect.signature(rhf.HessianBase.kernel).parameters for name in ("h1ao", "mo1", "mo_e1"))
node = ast.parse(inspect.getsource(rks.make_h1))
assert not any(isinstance(n, ast.Name) and n.id == "chkfile" and isinstance(n.ctx, ast.Load) for n in ast.walk(node))

natom, nao, nmo, nocc = 75, 762, 762, 160
rows = [array_row("H1AO_all_atoms", (natom, 3, nao, nao)),
        array_row("H1AO_one_atom", (3, nao, nao)),
        array_row("MO1_returned_AO_representation_all_atoms", (natom, 3, nao, nocc)),
        array_row("MO_E1_occupied_matrix_all_atoms", (natom, 3, nocc, nocc)),
        array_row("partial_or_complete_Hessian", (natom, natom, 3, 3)),
        array_row("MO_coeff", (nao, nmo)),
        array_row("RKS_partial_XC_vmat_internal_not_checkpoint", (natom, 3, 3, nao, nao)),
        array_row("all_atom_CPHF_rhs", (3*natom, nmo, nocc))]
chunk_rows = []
for atoms in (1, 2, 4, 5, 10, 75):
    b = 3 * atoms * nmo * nocc * 8
    nd = 3 * atoms * 50
    chunk_rows.append({"atoms": atoms, "RHS_vectors": 3*atoms, "rhs_payload_bytes": b,
                       "upper_bound_xs_plus_ax_bytes_50_cycles_no_deflation": 2*50*b,
                       "upper_bound_dense_Krylov_reduced_matrix_bytes": nd*nd*8,
                       "effective_native_CPHF_tolerance": float(hf.SCF.conv_tol_cpscf) * atoms})
payload_total = sum(row["payload_bytes"] for row in rows if row["name"] in
                    {"H1AO_all_atoms", "MO1_returned_AO_representation_all_atoms",
                     "MO_E1_occupied_matrix_all_atoms", "partial_or_complete_Hessian"})
b_all = rows[-1]["payload_bytes"]
native_min_block = max(2, int(2000e6 / 8 / (nmo*nocc*3*6)))
result = {"status": "DONE_SOURCE_AND_INTEGER_SIZE_AUDIT_ONLY",
          "created_at_utc": datetime.now(timezone.utc).isoformat(), "pyscf_version": pyscf.__version__,
          "quantum_and_integral_jobs": 0, "wavefunction_objects_constructed": 0,
          "active_checkpoints_read": 0, "large_arrays_allocated_or_saved": 0,
          "source_sha256": sources, "public_signatures": signatures,
          "RKS_hess_elec_is_RHF_function": True,
          "native_kernel_accepts_cached_H1_or_response_arguments": False,
          "hess_elec_accepts_cached_H1_and_responses": True,
          "hess_elec_always_recomputes_partial_first": True,
          "make_h1_chkfile_parameter_unused_in_RKS_body": True,
          "native_CPHF_mid_iteration_restart_exposed": False,
          "completed_atom_blocks_may_be_saved_by_new_validated_wrapper": True,
          "new_checkpointed_wrapper_implemented_by_this_audit": False,
          "assumed_molecular_dimensions_not_an_active_wavefunction_measurement":
              {"atoms": natom, "AO": nao, "MO_full_rank_assumption": nmo, "doubly_occupied_orbitals": nocc},
          "arrays": rows, "logical_complete_intermediate_payload_bytes": payload_total,
          "NPY_HDF_headers_and_transients_excluded_from_sizes": True,
          "native_solve_mo1_minimum_memory_floor_decimal_MB": 2000,
          "native_blksize_at_that_floor": native_min_block,
          "default_all75_atoms_fit_one_native_block": native_min_block >= natom,
          "actual_imported_krylov_default_max_memory_MB": inspect.signature(linalg_helper.krylov).parameters["max_memory"].default,
          "actual_SCF_class_conv_tol_cpscf": float(hf.SCF.conv_tol_cpscf),
          "default_all_atom_krylov_incore_prediction":
              inspect.signature(linalg_helper.krylov).parameters["max_memory"].default*1e6/b_all > 14,
          "CPHF_chunk_arithmetic_not_measured_peak_RAM": chunk_rows,
          "cross_process_automatic_resume_status": "BLOCKED_NOT_IMPLEMENTED_OR_VALIDATED",
          "scientific_frequency_results_created": False}
target = OUT / "api_and_storage_audit.json"
if target.exists():
    raise FileExistsError("Preserve this source-audit snapshot")
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "H1_plus_responses_plus_partial_payload_bytes": payload_total,
                  "native_min_block": native_min_block, "actual_Krylov_default_MB": result["actual_imported_krylov_default_max_memory_MB"],
                  "new_quantum_jobs": 0}))
