"""Read-only forensic inventory; never parses molecular HDF/NPY or runs QM."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "reports/pyscf-spinosin-pure-j-20261101-v4"
PHASE = OLD / "spinosin-pure-j-frequency-001"
V3 = ROOT / "reports/pyscf-spinosin-continuation-20261031-v3"
PIDS = (18564, 18565)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def main():
    assert not (OUT / "audit.json").exists(), "Preserve previous audit output"
    now = datetime.now(timezone.utc)
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    process_absence = {str(pid): not Path(f"/proc/{pid}").exists() for pid in PIDS}
    assert all(process_absence.values()), "No abandoned-file reads while original PIDs exist"
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    paths = {
        "v4-ledger.json": OLD / "cpu_budget.json",
        "v4-progress.json": PHASE / "progress.json",
        "v4-rhf-helper-manifest.json": PHASE / "rhf_helper_cache_manifest.json",
        "v4-hessian-manifest.json": PHASE / "hessian_cache_manifest.json",
        "v4-supervisor-proof.json": OLD / "execution/spinosin-pure-j-frequency-001.log.supervisor-proof.json",
        "v4-execution-lock.json": OLD / "execution_lock.json",
        "v3-native-optimization-result.json": V3 / "spinosin-native4-optimization-001/result.json",
        "v3-closed-ledger.json": V3 / "cpu_budget.json",
    }
    source_records = {}
    snapshots = OUT / "source_snapshots"
    snapshots.mkdir(exist_ok=False)
    for name, source in paths.items():
        raw = source.read_bytes()
        (snapshots / name).write_bytes(raw)
        source_records[rel(source)] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "snapshot": str((snapshots / name).relative_to(OUT)),
        }
    ledger = read(paths["v4-ledger.json"])
    progress = read(paths["v4-progress.json"])
    rhf = read(paths["v4-rhf-helper-manifest.json"])
    hessian = read(paths["v4-hessian-manifest.json"])
    proof = read(paths["v4-supervisor-proof.json"])
    optimization = read(paths["v3-native-optimization-result.json"])
    historical_v3 = read(paths["v3-closed-ledger.json"])
    assert proof["child_pid"] == 18565 and proof["supervisor_pid"] == 18564
    assert int(proof["child_start_ticks"]) / os.sysconf("SC_CLK_TCK") > uptime
    target = next(x for x in ledger["phases"] if x["phase"] == "spinosin-pure-j-frequency-001")
    controls = [x for x in ledger["phases"] if x["phase"] != target["phase"]]
    assert len(controls) == 2 and all(x["status"] == "EXITED" and x["exit_code"] == 0 for x in controls)
    known_cpu = sum(x["actual_cpu_seconds"] for x in controls)
    assert abs(known_cpu - 112.63092) < 1e-8
    assert target["status"] == "RUNNING" and target.get("actual_cpu_seconds") is None

    monitor = ROOT / "outputs/pyscf-spinosin-pure-j-20261101-v4/root-monitor-events.jsonl"
    monitored = [json.loads(x) for x in monitor.read_text().splitlines() if x.strip()]
    eligible = [x for x in monitored if x.get("identity") == "matched" and x.get("pid") == 18565]
    assert eligible
    last_monitor = eligible[-1]
    observed_cpu = max(x.get("actual_phase_CPU_seconds_observed") or 0 for x in eligible)
    (snapshots / "last-monitor-observation.json").write_text(json.dumps(last_monitor, ensure_ascii=False, indent=2) + "\n")
    source_records[rel(monitor)] = {"sha256": sha(monitor), "bytes": monitor.stat().st_size,
                                  "snapshot_scope": "last identity-matched observation only"}

    binary_candidates = [
        V3 / "spinosin-native4-optimization-001/calculation.chk",
        V3 / "spinosin-native4-optimization-001/optimized_coordinates_angstrom.npy",
        PHASE / "calculation.chk", PHASE / "gradient_hartree_per_bohr.npy",
        PHASE / "normal_mode_geometry_angstrom.npy",
    ]
    write_new(OUT / "forensic_release.json", {
        "schema": "research4-process-absence-forensic-byte-inventory-v1",
        "observed_at_utc": now.isoformat(), "issuer": "/root/resumption_forensics under root task",
        "status": "OLD_PROCESSES_ABSENT_NOT_A_TERMINAL_WAIT4_RECORD",
        "boot_id": boot_id, "uptime_seconds": uptime, "old_PID_absence": process_absence,
        "proof_old_child_start_ticks": proof["child_start_ticks"],
        "allow_listed_file_byte_hashes_only": [rel(x) for x in binary_candidates],
        "HDF_or_NPY_content_parsing_authorized": False,
        "new_quantum_evaluation_authorized": False, "old_ledger_mutation_authorized": False,
        "termination_time_exit_code_final_CPU_or_cause_inferred": False,
    })
    inventory = []
    expected_gradient = hessian.get("binding", {}).get("input_evidence", {}).get("gradient_npy_sha256")
    expectations = {
        rel(binary_candidates[0]): optimization.get("checkpoint_sha256"),
        rel(PHASE / "gradient_hartree_per_bohr.npy"): expected_gradient,
        rel(PHASE / "normal_mode_geometry_angstrom.npy"): progress.get("normal_mode_geometry_angstrom_npy_sha256"),
    }
    for path in binary_candidates:
        record = {"path": rel(path), "exists": path.is_file(), "content_parsed": False}
        if path.is_file():
            record.update(bytes=path.stat().st_size, sha256=sha(path))
            expected = expectations.get(rel(path))
            record.update(published_expected_sha256=expected, matches_published_sha256=(record["sha256"] == expected) if expected else None)
            if expected:
                assert record["matches_published_sha256"]
        inventory.append(record)
    missing_names = ["result.json", "hessian_hartree_per_bohr2.npy", "normal_modes.npy", "frequencies.csv"]
    missing = {name: not (PHASE / name).exists() for name in missing_names}
    assert all(missing.values())
    assert not rhf.get("arrays") and not hessian.get("partial_hessian")
    assert not hessian.get("H1AO_atoms") and not hessian.get("response_blocks")
    cache_roots = [Path(hessian["cache_root"]), Path(hessian["cache_root"] + "-rhf-pure-j-parts")]
    cache_inventory = []
    for path in cache_roots:
        files = [x for x in path.rglob("*") if x.is_file()] if path.exists() else []
        cache_inventory.append({"path": rel(path), "exists": path.exists(),
                                "files": [{"path": rel(x), "bytes": x.stat().st_size} for x in files]})
        assert not files
    release_sha = sha(OUT / "forensic_release.json")
    audit = {
        "schema": "research4-interrupted-runtime-evidence-v1",
        "observed_at_utc": now.isoformat(),
        "observed_at_client_time": now.astimezone(ZoneInfo("America/Los_Angeles")).isoformat(),
        "status": "PARTIAL_ABANDONED_RUNTIME_OBSERVED_NO_AVAILABLE_SPINOSIN_FREQUENCIES",
        "old_ledger_status_preserved_verbatim": target["status"],
        "current_original_processes_absent": process_absence, "current_boot_id": boot_id,
        "current_uptime_seconds": uptime,
        "old_proof_start_ticks_exceed_current_uptime_ticks": True,
        "observation_scope": "New process timebase and absent old processes; exact interruption mechanism/time is not known",
        "old_main_final_actual_CPU_seconds": None, "old_main_exit_code": None,
        "old_main_ended_at_utc": None, "old_main_termination_cause": "UNKNOWN_NOT_RECORDED",
        "closed_V4_software_control_actual_CPU_seconds": known_cpu,
        "last_recorded_main_process_CPU_observation_seconds": observed_cpu,
        "last_recorded_monitor_observation": last_monitor,
        "observational_minimum_closed_plus_main_CPU_seconds_not_final_accounting": known_cpu + observed_cpu,
        "CPU_observation_warning": "Process counter observation is not final wait4 and cannot replace or close the old budget ledger",
        "old_remaining_CPU_metadata_not_reusable": ledger["remaining_quantum_cpu_seconds"],
        "old_deadline_at_utc": ledger["run_deadline_at_utc"],
        "old_deadline_already_past_at_observation": datetime.fromisoformat(ledger["run_deadline_at_utc"]) < now,
        "old_V3_closed_actual_CPU_seconds": historical_v3["consumed_cpu_seconds"],
        "available_source_progress": {key: progress.get(key) for key in (
            "utc", "event", "process_cpu_seconds", "scf_converged", "final_gradient_thresholds_passed",
            "energy_hartree", "atom_rms_hartree_per_bohr", "atom_max_hartree_per_bohr", "frequencies_status")},
        "progress_native_signature_is_latched_not_current_kernel_progress": True,
        "RHF_helper_manifest_status": rhf["status"], "complete_RHF_tuple_available": False,
        "complete_RKS_partial_available": False, "H1_atoms_available": 0, "response_blocks_available": 0,
        "available_complete_Hessian_or_modes": False, "missing_main_artifacts": missing,
        "cache_directory_inventory": cache_inventory,
        "binary_byte_inventory_no_HDF_NPY_parse": inventory,
        "old_native_optimum_is_preferred_existing_geometry": {
            "result_path": rel(paths["v3-native-optimization-result.json"]),
            "result_sha256": sha(paths["v3-native-optimization-result.json"]),
            "status": optimization["status"], "geometry_optimized": optimization.get("geometry_optimized"),
            "fresh_gradient_thresholds_passed": optimization.get("final_gradient_thresholds_passed"),
            "energy_hartree": optimization.get("energy_hartree"),
            "checkpoint_byte_hash_matches_its_closed_result": inventory[0]["matches_published_sha256"],
        },
        "recovery": {
            "Hessian_resume": "NOT_POSSIBLE_FROM_CURRENT_AVAILABLE_CACHES",
            "abandoned_checkpoint_content_validation": "NOT_RUN_BYTE_HASH_ONLY",
            "old_closed_optimum_checkpoint_role": "SCF_WARMSTART_ONLY_NOT_HESSIAN_RESTORATION",
            "next_actual_calculation_requires": [
                "New independent finite authorization/ledger with historical unknown cost disclosed",
                "Trusted molecule identity and same-method input geometry validation",
                "New fresh SCF/gradient before a new full Hessian attempt",
                "Consider validated incremental immutable RHF checkpoints to avoid losing the entire helper",
            ],
        },
        "IR_intensities_status": "NOT_COMPUTED", "new_quantum_evaluations": 0,
        "old_scientific_files_modified": False, "old_ledgers_filled_or_reclassified": False,
        "forensic_byte_release_sha256": release_sha, "source_records": source_records,
    }
    write_new(OUT / "audit.json", audit)
    with (OUT / "checkpoint_inventory.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "exists", "bytes", "sha256", "published_expected_sha256", "matches_published_sha256", "content_parsed"])
        writer.writeheader(); writer.writerows(inventory)
    (OUT / "README.md").write_text(
        "# 中断取证与可恢复范围\n\n"
        f"观察时间（用户时区）：{audit['observed_at_client_time']}；机器UTC及原始时间戳保留在JSON。\n\n"
        "旧监督进程18564及工作进程18565均已不存在。当前机器uptime短于旧启动证明的start ticks；旧V4账本的RUNNING是持久化快照，不代表当前仍在运行。这些观察与运行环境被重建／恢复相符，但不能确定旧任务的精确停止原因、停止时间、退出码或最终CPU。旧账本及其状态逐字保留，没有补造终止或wait4。\n\n"
        f"V4水／乙醇已闭合控制实际CPU合计{known_cpu:.5f}秒。最后主任务监测为{last_monitor['utc']}，记录进程CPU {observed_cpu:.2f}秒；它只能作为已观测过程消耗下界，不能替代最终wait4。旧remaining={ledger['remaining_quantum_cpu_seconds']}尚未结算主任务，且期限已过，不能被当成新可用余额。\n\n"
        "当前可用产物有已收敛SCF及通过阈值的fresh-gradient记录；没有主任务result.json、完整Hessian、正常模式或频率文件。RHF辅助manifest仍WAITING且数组为空，RKS partial为空，H1／response为空，两缓存目录也无文件。文件名normal_mode_geometry只是计划用于未来模式分析的坐标，不是已计算的振型。无记录不等于没有低频振动。\n\n"
        "已先写入forensic_release.json，再仅按字节哈希旧检查点／坐标／梯度文件；没有导入h5py、NumPy或PySCF，也没有解析HDF／NPY内容。V3已闭合native优化checkpoint字节SHA匹配其真实结果，应优先复用其最佳完整精度几何和SCF初猜。旧V4遗留calculation.chk尚未做内部内容核验；即使后续核验成功，它也只提供SCF初猜，不构成Hessian续算。\n\n"
        "下一步需要新的明确有限预算，披露旧主任务最终成本未知，保留旧记录；在同一方法下重做fresh SCF／梯度后开展完整Hessian。建议先验证可跨进程恢复的逐原子RHF检查点，以免环境再次中断时丢失整个昂贵helper。新上传SDF是另一路结构来源，身份／立体化学核验后也不能默默覆盖已有最优DFT几何。\n\n"
        "本次只有取证与哈希，没有新量子计算、停止信号、Git操作、原始光谱／模型修改。IR强度NOT_COMPUTED；无可用斯皮诺素理论频率可用于实验峰归属或改变96.0234%分类结果。\n"
    )
    unchanged = []
    for source, metadata in source_records.items():
        unchanged.append(sha(ROOT / source) == metadata["sha256"])
    assert all(unchanged)
    write_new(OUT / "readback_checks.json", {
        "status": "PASS", "all_original_text_sources_unchanged": True,
        "source_file_count": len(source_records), "original_PIDs_still_absent": all(not Path(f"/proc/{pid}").exists() for pid in PIDS),
        "binary_files_only_hashed_not_parsed": True, "old_ledger_not_modified": True,
        "new_quantum_evaluations": 0,
    })
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "files_sha256.json":
            files[str(path.relative_to(OUT))] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    write_new(OUT / "files_sha256.json", {"scope": "Paths relative to this interruption-audit directory; index self-excluded", "files": files})
    print(json.dumps({"status": audit["status"], "source_files_unchanged": True,
                      "output_files_indexed": len(files), "index_sha256": sha(OUT / "files_sha256.json"),
                      "final_main_CPU": None, "Hessian_resume_available": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
