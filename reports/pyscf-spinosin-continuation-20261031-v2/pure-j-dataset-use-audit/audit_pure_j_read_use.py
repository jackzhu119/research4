"""Read-use/control-flow audit only; no PySCF numerical method is called."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SITE = ROOT / ".runtime/quantum-upgrade-site"
sys.path.insert(0, str(SITE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size(shape):
    number = math.prod(shape) * 8
    return {"shape": list(shape), "float64_bytes": number, "GiB": number / 2**30}


def ref(line, guards, effect):
    return {"line": line, "guards": guards, "effect": effect}


def run():
    target = OUT / "pure_j_dataset_read_use.json"
    if target.exists():
        raise FileExistsError("Preserve the source audit")
    rhf, rks = SITE / "pyscf/df/hessian/rhf.py", SITE / "pyscf/df/hessian/rks.py"
    text = rhf.read_text()
    assert text.count("ftmp['vj1_buf']") == 1
    for statement in ("vj1_buf = ftmp['vj1_buf'] = vj1_buf", "vj1 = -np.asarray(vj1_buf[ia])",
                      "wk1_Pij = rho_ip1[p0:p1].transpose(2,3,0,1)",
                      "rhoj1_P = np.einsum('pxij,ji->px', wk1_Pij, dm0[:,p0:p1])"):
        assert statement in text
    for original in (rhf, rks):
        (OUT / ("snapshot_" + original.name.replace(".py", "") + "_df_hessian.py")).write_bytes(original.read_bytes())
    # Language-semantics negative control: a chained assignment does not read
    # a Dataset or change the local array to a Dataset handle.
    class Sink:
        def __init__(self): self.saved = {}; self.read_count = 0
        def __setitem__(self, key, value): self.saved[key] = tuple(value)
        def __getitem__(self, key): self.read_count += 1; raise AssertionError("not used")
    f = Sink(); original_value = [1., 2.]; local = original_value
    local = f["vj1_buf"] = local
    assert local is original_value and f.read_count == 0
    # A tiny HDF5 metadata-only probe establishes installed default storage
    # behavior; it neither declares a huge array nor writes numeric payload.
    import h5py
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "unwritten_tiny_dataset.h5"
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("unused", (2,3,3,2), "f8")
            handle.flush()
            probe = {"h5py_version": h5py.__version__, "small_shape": [2,3,3,2],
                     "declared_data_bytes": dataset.size * dataset.dtype.itemsize,
                     "allocated_numeric_storage_bytes": dataset.id.get_storage_size(),
                     "allocation_time_code": dataset.id.get_create_plist().get_alloc_time(),
                     "default_is_contiguous": dataset.chunks is None,
                     "actual_small_file_bytes": path.stat().st_size,
                     "quantum_calls": 0}
        assert probe["allocated_numeric_storage_bytes"] == 0
    nao, naux, natm, nocc = 762, 2459, 75, 160
    records = [
        {"function": "_partial_hess_ejk", "name": "rho_ip1", "kind": "HDF5 dataset", **size((nao,nao,naux,3)),
         "create": ref(137, [], "Declare dataset"),
         "writers": [ref(149, ["for all requested atom AO rows"], "Write tmp_ip1.transpose(2,3,0,1)")],
         "readers": [ref(243, ["auxbasis_response != 0"], "Read AO-row block into wk1_Pij")],
         "downstream_pure_J_use": [ref(244, ["auxbasis_response != 0"], "Contract dm0 to rhoj1_P"),
             ref(261, ["auxbasis_response != 0"], "J _ej auxiliary term"),
             ref(263, ["auxbasis_response != 0"], "J _ej auxiliary term"),
             ref(265, ["auxbasis_response != 0"], "J _ej auxiliary term"),
             ref(267, ["auxbasis_response > 1"], "Add _ej*2 to ej"),
             ref(268, ["auxbasis_response > 1"], "Add transposed _ej*2 to ej")],
         "pure_J_written": True, "pure_J_read": True, "delete_without_replacement": False},
        {"function": "_partial_hess_ejk", "name": "wk_ip2_Ipk", "HDF5_name": "wk_ip2", **size((nao,naux,3,nao)),
         "create": ref(172, [], "Declare dataset"),
         "writers": [ref(181, ["with_k"], "Store exchange response tensor")],
         "readers": [ref(254, ["auxbasis_response != 0", "with_k"], "Exchange contraction")],
         "pure_J_written": False, "pure_J_read": False, "delete_without_replacement": True,
         "savings_limit": "Only unwritten-dataset metadata; do not claim recovery of the declared31.91GiB numeric payload."},
        {"function": "_partial_hess_ejk", "name": "wk_ip2_P__", "kind": "np.empty RAM/virtual allocation", **size((naux,3,nocc,nocc)),
         "create": ref(174, ["auxbasis_response > 1"], "Allocate despite with_k=False"),
         "writers": [ref(183, ["with_k", "auxbasis_response > 1"], "Exchange contraction")],
         "readers": [ref(357, ["auxbasis_response > 1", "with_k"], "Exchange _ek contraction"),
                     ref(358, ["auxbasis_response > 1", "with_k"], "Exchange _ek contraction")],
         "pure_J_written": False, "pure_J_read": False, "delete_without_replacement": True,
         "savings_limit": "Avoids unused ndarray address-space reservation, not proven RSS and not HDF5 disk payload."},
        {"function": "_partial_hess_ejk", "name": "rhok_ip1_IkP", "kind": "HDF5 group",
         "create": ref(138, [], "Empty group"), "writers": [ref(152, ["with_k"], "Exchange data")],
         "readers": [ref(206, ["with_k"], "Exchange vk1 contraction")],
         "pure_J_written": False, "pure_J_read": False, "delete_without_replacement": True,
         "savings_limit": "Only empty-group metadata; line255 is a comment, not a reader."},
        {"function": "_partial_hess_ejk", "name": "rhok_ip1_PkI", "kind": "HDF5 group",
         "create": ref(139, [], "Empty group"), "writers": [ref(153, ["with_k"], "Exchange data")],
         "readers": [ref(167, ["with_k"], "Exchange vk2 contraction")],
         "pure_J_written": False, "pure_J_read": False, "delete_without_replacement": True,
         "savings_limit": "Only empty-group metadata."},
        {"function": "_gen_jk", "name": "rho0_Pij", "kind": "HDF5 group of atom AO-row datasets", **size((naux,nao,nao)),
         "create": ref(406, [], "Create group"), "writers": [ref(418, [], "Store solved three-center coef3c for every AO row")],
         "readers": [ref(435, [], "_load_dim0 -> coef3c"),
                     ref(462, ["auxbasis_response != 0"], "_load_dim0 -> coef3c")],
         "downstream_pure_J_use": [ref(438, [], "vj1_buf J contribution"),
             ref(464, ["auxbasis_response != 0"], "vj1 auxiliary term"),
             ref(466, ["auxbasis_response != 0"], "vj1 auxiliary term")],
         "pure_J_written": True, "pure_J_read": True, "delete_without_replacement": False},
        {"function": "_gen_jk", "name": "wj_ip1_pij", "kind": "HDF5 group of atom AO-row datasets", **size((naux,nao,3,nao)),
         "create": ref(407, [], "Create group"), "writers": [ref(423, ["auxbasis_response != 0"], "Store int2c_ip1*coef3c contraction")],
         "readers": [ref(463, ["auxbasis_response != 0"], "_load_dim0 -> pij")],
         "downstream_pure_J_use": [ref(467, ["auxbasis_response != 0"], "vj1 pure-J auxiliary term; no with_k guard")],
         "pure_J_written": True, "pure_J_read": True, "delete_without_replacement": False},
        {"function": "_gen_jk", "name": "vj1_buf RAM ndarray", **size((natm,3,nao,nao)),
         "create": ref(430, [], "Allocate RAM J buffer"), "writers": [ref(438, [], "Accumulate J contraction")],
         "readers": [ref(450, [], "Read RAM ndarray for vj1")],
         "downstream_pure_J_use": [ref(474, [], "Symmetrize vj1"), ref(478, [], "Yield vj1 to RKS caller")],
         "pure_J_written": True, "pure_J_read": True, "delete_without_replacement": False},
        {"function": "_gen_jk", "name": "ftmp['vj1_buf'] disk copy", **size((natm,3,nao,nao)),
         "create": ref(443, [], "Chained assignment writes an HDF5 copy but leaves local vj1_buf as original ndarray"),
         "writers": [ref(443, [], "Write one complete RAM buffer copy")], "readers": [],
         "pure_J_written": True, "pure_J_read": False, "delete_without_replacement": True,
         "savings_limit": "Remove the dead HDF5 write only; retain the live RAM J buffer and all its contractions."},
        {"function": "_gen_jk", "name": "vk1_buf", "kind": "np.zeros RAM allocation", **size((3,nao,nao)),
         "create": ref(429, [], "Allocate exchange-only work buffer"),
         "writers": [ref(441, ["with_k"], "Exchange contraction")],
         "readers": [ref(455, ["with_k"], "Exchange vk1 update")],
         "pure_J_written": False, "pure_J_read": False, "delete_without_replacement": True,
         "savings_limit": "Unused K RAM work buffer; no HDF5 payload saving."}]
    rho, rho0, wj, vjcopy = [next(row for row in records if row["name"] == name)["float64_bytes"] for name in
                           ("rho_ip1", "rho0_Pij", "wj_ip1_pij", "ftmp['vj1_buf'] disk copy")]
    report = {"status": "READ_USE_AUDIT_COMPLETE_REQUIRED_PURE_J_HDF5_PAYLOAD_REMAINS_BLOCKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "pyscf_version": importlib.metadata.version("pyscf"),
        "new_integral_SCF_gradient_response_Hessian_calls": 0, "source_files_modified": False,
        "scenario": {"xc": "PBE", "with_k": False, "auxbasis_response": 2, "natm": natm, "nao": nao, "naux": naux, "nocc": nocc},
        "source_files_sha256": {str(rhf.relative_to(ROOT)): sha(rhf), str(rks.relative_to(ROOT)): sha(rks)},
        "per_object_read_write_control_flow": records,
        "RKS_uses_J_results": {"rks_partial_lines": [57,58,59,60], "rks_H1_lines": [105,106,107],
                               "rks_K_H1_ignored_for_pure_PBE_lines": [108,109]},
        "chain_assignment_negative_control": {"local_object_identity_preserved": local is original_value, "sink_getitem_calls": f.read_count},
        "tiny_unwritten_HDF5_default_storage_probe": probe,
        "disk_arithmetic_not_runtime_peak_measurement": {
            "partial_required_J_rho_ip1_bytes": rho,
            "H1_current_written_payload_bytes": rho0+wj+vjcopy,
            "H1_dead_disk_copy_removable_bytes": vjcopy,
            "H1_required_J_payload_after_dead_copy_removal_bytes": rho0+wj,
            "H1_required_J_payload_after_dead_copy_removal_GiB": (rho0+wj)/2**30,
            "removing_only_dead_K_metadata_or_allocations_unblocks_full_DF_Hessian": False,
            "both_partial_and_H1_stages_still_exceed_about9GiB_available_disk": True,
            "do_not_sum_distinct_temp_lifetimes_into_one_peak": True},
        "strict_source_identity_observation_not_an_implementation": {
            "first_expression_line": 147, "store_line": 149, "reload_line": 243, "second_expression_line": 244,
            "same_atom_iteration_and_frozen_dm0": True,
            "observation": "The149 transpose is inverted by243. The244 contraction to rhoj1_P has the same operands/einsum expression as147 rhoj1[i0].",
            "original_rhoj1_released_at_line": 158,
            "conclusion": "Retaining an existing precontraction might enable a different storage schedule, but rho_ip1 is genuinely read as written; deleting it without replacing that data flow is wrong. No formula/storage rewrite was implemented or validated here."},
        "decision": "Some exchange-only allocations and one unused J-buffer disk copy are removable in principle, but the31.91GiB partial rho_ip1 and42.55GiB H1 rho0/wj datasets remain genuinely read in pure-J/auxresponse2. Simple dead-store skipping does not unblock this environment.",
        "independent_crosscheck": "A second read-only agent independently checked line numbers, guards, chained assignment and the identical precontraction observation; no quantum calls or files changed.",
        "script_sha256": sha(__file__)}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "unused_disk_copy_bytes": vjcopy,
                      "remaining_H1_required_disk_bytes": rho0+wj, "remaining_H1_required_GiB": (rho0+wj)/2**30,
                      "unused_K_array_virtual_bytes": size((naux,3,nocc,nocc))["float64_bytes"], "quantum_calls": 0}))


if __name__ == "__main__":
    run()
