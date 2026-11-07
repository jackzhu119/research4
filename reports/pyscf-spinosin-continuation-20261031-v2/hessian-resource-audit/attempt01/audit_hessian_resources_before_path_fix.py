"""Read-only PySCF Hessian source and dimension audit; no integrals or SCF."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SITE = ROOT / ".runtime/quantum-upgrade-site"
sys.path.insert(0, str(SITE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size(shape):
    value = math.prod(shape) * 8
    return {"shape": list(shape), "dtype": "float64", "bytes": value,
            "decimal_GB": value / 1e9, "binary_GiB": value / 2 ** 30}


def excerpt(text, start, end):
    return [{"line": i, "text": line} for i, line in enumerate(text.splitlines(), 1)
            if start <= i <= end]


def run():
    result_path = OUT / "hessian_resource_audit.json"
    if result_path.exists():
        raise FileExistsError("Preserve source audit snapshots")
    old_path = ROOT / "reports/pyscf-spinosin-pilot-20261031/spinosin-cached-optimization/result.json"
    old = json.loads(old_path.read_text())
    nao, naux, natm, nocc = old["basis_functions"], old["density_fit_cache"]["auxiliary_functions"], old["atoms"], old["electrons"] // 2
    assert (nao, naux, natm, nocc) == (762, 2459, 75, 160)
    sources = {}
    for relative in ("pyscf/df/hessian/rhf.py", "pyscf/df/hessian/rks.py", "pyscf/hessian/rhf.py", "pyscf/hessian/rks.py", "pyscf/df/grad/rhf.py", "pyscf/grad/rhf.py", "pyscf/df/df.py", "pyscf/df/df_jk.py"):
        source = SITE / relative
        destination = OUT / "source_snapshots" / relative.replace("/", "__")
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(source.read_bytes())
        sources[relative] = {"sha256": sha(source), "snapshot": str(destination.relative_to(OUT)),
                             "bytes": source.stat().st_size}
    df_source = (SITE / "pyscf/df/hessian/rhf.py").read_text()
    assert "rho_ip1[p0:p1] = tmp_ip1.transpose(2,3,0,1)" in df_source
    assert "rho0_Pij['%.4d'%i] = coef3c = coef3c.reshape(naux,p1-p0,nao)" in df_source
    assert "wj_ip1_pij['%.4d'%i] = lib.einsum('xqp,pij->qixj', int2c_ip1, coef3c)" in df_source
    rho_ip1 = size((nao, nao, naux, 3))
    rho0 = size((naux, nao, nao))
    wj_ip1 = size((naux, nao, 3, nao))
    vj_buf = size((natm, 3, nao, nao))
    free = os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize
    minimum_h1 = rho0["bytes"] + wj_ip1["bytes"] + vj_buf["bytes"]
    out = {"status": "BLOCKED_DEFAULT_DF_HESSIAN_DISK_SOURCE_AUDIT_NOT_A_CALCULATION",
           "created_at_utc": datetime.now(timezone.utc).isoformat(), "pyscf_version": importlib.metadata.version("pyscf"),
           "script_sha256": sha(__file__), "new_quantum_jobs": 0, "integral_evaluations": 0,
           "dimension_source": {"path": str(old_path.relative_to(ROOT)), "sha256": sha(old_path),
                                "nao": nao, "naux": naux, "natm": natm, "closed_shell_occupied_orbitals": nocc},
           "assumptions": {"xc": "PBE", "with_k": False, "auxbasis_response": 2,
                           "all_75_atoms_in_Hessian": True, "HDF5_default_uncompressed_numeric_datasets": True},
           "tmp_filesystem_free_bytes_at_audit": free,
           "DF_partial_hessian": {
               "rho_ip1": {**rho_ip1, "created_unconditionally": True, "fully_written_for_all_atoms_even_with_k_false": True,
                           "read_for_pure_J_first_order_auxiliary_response": True},
               "wk_ip2": {**size((nao, naux, 3, nao)), "dataset_declared_even_with_k_false": True,
                          "with_k_false_written": False, "counted_as_allocated_disk_lower_bound": False},
               "wk_ip2_P_buffer": {**size((naux, 3, nocc, nocc)), "allocated_with_auxbasis_response_gt_1_even_with_k_false": True,
                                    "is_virtual_array_allocation_not_measured_RSS": True},
               "mandatory_written_rho_ip1_exceeds_current_tmp_free_space": rho_ip1["bytes"] > free},
           "DF_first_derivative_H1_generator": {
               "rho0_Pij": {**rho0, "complete_group_datasets_cover_all_AO_rows": True},
               "wj_ip1_pij": {**wj_ip1, "complete_group_datasets_cover_all_AO_rows_when_auxbasis_response_true": True},
               "vj1_buf": {**vj_buf, "numeric_buffer_written_to_same_HDF5_file": True},
               "simultaneous_mandatory_numeric_payload_bytes_lower_bound": minimum_h1,
               "same_binary_GiB_lower_bound": minimum_h1 / 2 ** 30,
               "lower_bound_exceeds_current_tmp_free_space": minimum_h1 > free,
               "not_added_to_partial_stage_minimum": "The temporary file scopes differ; do not sum all sequential stages into one claimed peak."},
           "RAM_sizing_not_peak_measurement": {
               "H1_AO_array": size((natm, 3, nao, nao)),
               "DF_SCF_cached_tensor_if_CPHF_rebuilds": size((naux, nao * (nao + 1) // 2)),
               "largest_nominal_aux480_derivative_block": size((3, nao, nao, min(480, naux))),
               "note": "Other transient arrays, CPHF vectors and thread/runtime virtual memory are additional. These arrays do not necessarily coexist. No Spinosin Hessian was run to measure peak RAM."},
           "existing_low_storage_switch": {
               "max_memory": "Changes block sizes, not mandatory global HDF5 dimensions.",
               "atmlst": "Does not avoid all-atom rho0_Pij/wj_ip1_pij construction in _gen_jk and is not a complete normal-mode calculation when atoms are omitted.",
               "auxbasis_response_zero": "Changes the derivative approximation and still leaves unconditional allocations; not an equivalent allowed shortcut.",
               "native_streaming_option_found_in_reviewed_source": False},
           "no_DF_RKS_analytic_Hessian": {
               "source_available": True, "screened_direct_integral_contractions": True,
               "forced_DF_rho_ip1_rho0_wj_HDF5_tensors_present": False,
               "integral_contract_function": "pyscf.hessian.rhf._get_jk -> pyscf.scf._vhf.direct_bindm",
               "same_as_current_DF_derivative_method": False,
               "not_run": True, "runtime_and_peak_memory_not_measured": True,
               "required_gates_before_possible_future_use": [
                   "Record a distinct no-DF PBE/def2-SVP/grid1/no-D3 protocol and implementation provenance; do not relabel a DF result.",
                   "Run and independently verify small-molecule no-DF Opt/Freq, including raw/symmetrized Hessian and complete positive modes.",
                   "At the candidate geometry perform a fresh converged no-DF SCF and no-DF nuclear gradient with the same charge/spin/basis/XC/grid.",
                   "If no-DF gradient or optimizer conditions are not met, optimize with no-DF before claiming a no-DF stationary-geometry frequency result.",
                   "Apply explicit CPU, memory and disk limits, then a real Hessian/CPHF convergence and complete 219-mode check.",
                   "Compare DF/no-DF differences transparently; no empirical peak matching, no IR intensities without verified dipole derivatives."]},
           "evidence": {
               "DF_partial_global_datasets": excerpt(df_source, 136, 183),
               "DF_partial_auxiliary_response_reads": excerpt(df_source, 235, 256),
               "DF_H1_global_groups_and_writes": excerpt(df_source, 405, 445),
               "no_DF_screened_direct_contract": excerpt((SITE / "pyscf/hessian/rhf.py").read_text(), 261, 270)},
           "source_files": sources,
           "conclusion": "Default full DF-RKS analytic Hessian is blocked by current disk, independent of optimization convergence. A no-DF calculation is a distinct approximation requiring actual SCF/gradient/optimization consistency and software validation; it has not been executed by this audit."}
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "rho_ip1_bytes": rho_ip1["bytes"], "H1_disk_lower_bound_bytes": minimum_h1,
                      "H1_disk_lower_bound_GiB": minimum_h1 / 2 ** 30, "tmp_free_bytes": free, "quantum_jobs": 0}))


if __name__ == "__main__":
    run()
