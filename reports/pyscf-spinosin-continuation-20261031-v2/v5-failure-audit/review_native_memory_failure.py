"""Read ended failure artifacts and native source; no quantum or response call."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = OUT.parent
SOURCE = RUN / "water-direct-v5-optfreq"
SITE = ROOT / ".runtime/quantum-upgrade-site"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run():
    target = OUT / "native_memory_failure_review.json"
    if target.exists():
        raise FileExistsError("Preserve failure audit")
    ledger_bytes = (RUN / "cpu_budget.json").read_bytes()
    ledger = json.loads(ledger_bytes)
    phase = next(row for row in ledger["phases"] if row["phase"] == "water_direct_v5_block_validation")
    assert phase["status"] != "RUNNING" and phase["ended_at_utc"] and phase["exit_code"] == 1
    assert phase["address_space_limit_gib"] == 12 and phase["threads"] == 4
    (OUT / "ledger_snapshot_with_failed_source.json").write_bytes(ledger_bytes)
    driver = Path(phase["log"])
    assert sha(driver) == phase["log_sha256"]
    allocation_bytes = [int(value) for value in re.findall(r"malloc\((\d+)\) failed in CVHFallocate_JKArray", driver.read_text())]
    assert allocation_bytes == [533459336, 533459336]
    assert not (SOURCE / "result.json").exists() and not (SOURCE / "exception_traceback.txt").exists()
    manifest_path = SOURCE / "hessian_cache_manifest.json"
    manifest = read(manifest_path)
    assert manifest["partial_hessian"]["native_function_returned_normally"]
    assert manifest["H1AO_complete"] and len(manifest["H1AO_atoms"]) == 3
    assert manifest["response_blocks"] == []
    assert manifest["response_operator"]["shared_fx_created_once"]
    records = [manifest["partial_hessian"]["array"]] + [row["array"] for row in manifest["H1AO_atoms"]]
    cache_checks = []
    for record in records:
        path = Path(record["path"])
        values = np.load(path, allow_pickle=False)
        assert sha(path) == record["file_sha256"] and list(values.shape) == record["shape"] and np.isfinite(values).all()
        cache_checks.append({"path": str(path), "file_sha256": sha(path), "shape": list(values.shape), "finite": True})
    source_C = OUT / "native_source_snapshots/nr_direct.c"
    source_H = OUT / "native_source_snapshots/nr_direct.h"
    text = source_C.read_text()
    assert "(400000000 - di*di*di*di*ncomp - cache_size) / n_dm" in text
    assert "malloc(sizeof(double) * (size_limit + 136*136))" in text
    assert "#pragma omp for nowait schedule(dynamic, 1)" in text
    official = read(OUT / "native_source_provenance.json")
    for row in official:
        content = (OUT / "native_source_snapshots" / row["file"]).read_bytes()
        assert hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest() == row["github_blob_sha"]
    python_driver = SITE / "pyscf/scf/_vhf.py"
    pytext = python_driver.read_text()
    assert "def nr_direct_drv(" in pytext
    signature = pytext[pytext.index("def nr_direct_drv("):pytext.index("def nr_direct_drv(")+260]
    assert "max_memory" not in signature
    helper_spec = importlib.util.spec_from_file_location("failed_water_sites_only", ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py")
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper); helper.isolated_sites()
    from pyscf.scf import chkfile
    checkpoint = SOURCE / "calculation.chk"
    checkpoint_before = sha(checkpoint)
    molecule = chkfile.load_mol(str(checkpoint))
    scf = chkfile.load(str(checkpoint), "scf")
    assert molecule.natm == 3 and molecule.nelectron == 10 and molecule.spin == molecule.charge == 0
    assert scf["mo_coeff"].shape == (24,24) and scf["mo_energy"].shape == scf["mo_occ"].shape == (24,)
    assert all(np.isfinite(scf[k]).all() for k in ("mo_coeff", "mo_energy", "mo_occ"))
    assert checkpoint_before == sha(checkpoint)
    n_dm_assumed = 6
    one = allocation_bytes[0]
    size_limit_inferred = one//8 - 136*136
    thread_payload = one*n_dm_assumed
    all_four_payload = thread_payload*4
    address_cap = 12*2**30
    v4_result_path = RUN / "water-direct-v4-optfreq/result.json"
    v4 = read(v4_result_path)
    assert v4["status"] == "DONE" and v4["frequency_validation_passed"] and v4["actual_pyscf_threads"] == 4
    verdict = {
        "status": "SOURCE_SUPPORTED_NATIVE_BUFFER_AS_INCOMPATIBILITY_V6_VALIDATION_REQUIRED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "new_quantum_or_response_calls": 0,
        "failed_source_phase": phase, "batch_not_asserted_final": True,
        "original_failure_scene_modified": False, "synthetic_result_json_created_for_failed_source": False,
        "observed_malloc_failure_bytes": allocation_bytes,
        "observed_partial_Hessian_and_all_H1_saves_valid": cache_checks,
        "completed_response_blocks": 0,
        "raw_manifest_status_left_unchanged": manifest["status"],
        "why_no_Python_result_or_cleanup": "Official native allocation failure calls exit(1), bypassing Python exception/finally cleanup.",
        "failed_closed_checkpoint_read_only": {"path": str(checkpoint.relative_to(ROOT)), "sha256": checkpoint_before,
            "atoms": 3, "electrons": 10, "AO_MO_shape": [24,24], "all_arrays_finite": True},
        "native_buffer_source_mechanism": {
            "official_tag": "v2.14.0", "per_thread_nominal_double_budget": 400000000,
            "size_limit": "(400000000 - di**4*ncomp - cache_size) // n_dm",
            "allocation_per_DM": "8 * (size_limit + 136**2)",
            "allocated_in_each_OpenMP_thread_for_all_DM_scripts": True,
            "mf_max_memory_controls_this_C_constant": False,
            "Python_native_driver_has_max_memory_argument": False,
            "C_exit_on_failure": True},
        "exact_arithmetic_under_expected_first_block_6_DM": {
            "expected_requested_atom_directions": [2,3], "n_dm_assumed": n_dm_assumed,
            "C_call_DM_count_or_thread_count_instrumented_in_failed_process": False,
            "inferred_size_limit_doubles": size_limit_inferred,
            "allocation_bytes_for6arrays_per_thread": thread_payload,
            "allocation_bytes_for4threads_if_simultaneously_live": all_four_payload,
            "address_space_cap_bytes": address_cap,
            "remaining_address_bytes_for_all_other_mappings": address_cap-all_four_payload,
            "recorded_peak_RSS_bytes_not_exact_failure_VmSize": phase["peak_rss_kib"]*1024,
            "tiny_water_response_dm1_bytes_at6rhs": 6*24*24*8,
            "interpretation": "A normal6-RHS tiny-water density batch can invoke a per-thread hardcoded multi-GB native buffer budget. With4threads, almost the entire12GiB address cap is consumed before Python/libraries/stacks/other arrays."},
        "why_V4_success_does_not_establish_4thread_large_system_safety": {
            "same_setting_4threads_V4_water_DONE": True,
            "V4_frequency_result_sha256": sha(v4_result_path),
            "source_AO_block_size": 64, "water_AO": 24,
            "source_has_nowait_work_loop_followed_by_perthread_free": True,
            "hypothesis_not_measured": "For tiny water few integral work blocks, idle native threads may free their private buffers before all other threads finish allocations. Allocation scheduling/fragmentation or actual active-thread counts can differ. No failed-process VmSize/call-level thread trace was captured, so the exact V4/V5 difference is not proven."},
        "evidence_does_not_show": ["No verified malformed MO/H1 shape or density-batch truncation was found.",
            "Pure guard tests did not validate the numerical/native path; V5 water is a real failed gate.",
            "The current audit has not measured a full runtime VmSize trace or run a repaired native wrapper."],
        "bounded_repair_design_for_new_version_only": [
            "Keep the12GiB address cap and exact screened-direct integral method/tolerance.",
            "A process-local context wrapper around _vhf.nr_direct_drv can request one native OpenMP thread for SCF/CPHF and derivative direct_bindm/direct_mapdm calls, restoring prior thread state in finally.",
            "Keep outer XC/other computation thread choices separately recorded; do not silently alter frozenV4/V5 or the installed library.",
            "Actual V6 water Opt/Freq and native-thread instrumentation are required before treating the repaired route as validated.",
            "Do not invent frequencies/results for this failed run or call saved partial/H1 arrays a resumable Hessian."],
        "file_provenance": {str(source_C.relative_to(ROOT)): sha(source_C), str(source_H.relative_to(ROOT)): sha(source_H),
            str(python_driver.relative_to(ROOT)): sha(python_driver),
            str((SITE / "pyscf/lib/libcvhf.so").relative_to(ROOT)): sha(SITE / "pyscf/lib/libcvhf.so"),
            str(driver.relative_to(ROOT)): sha(driver), str(manifest_path.relative_to(ROOT)): sha(manifest_path),
            "ledger_snapshot_sha256": hashlib.sha256(ledger_bytes).hexdigest()},
        "official_source_provenance": official, "script_sha256": sha(__file__)}
    target.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": verdict["status"], "single_malloc_bytes": one,
                      "four_thread_private_buffer_bytes_under6dm": all_four_payload,
                      "AS12GiB_bytes": address_cap, "other_mapping_headroom_bytes": address_cap-all_four_payload,
                      "new_quantum_calls": 0}))


if __name__ == "__main__":
    run()
