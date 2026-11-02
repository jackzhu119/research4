#!/usr/bin/env python3
"""Audit closed metadata and CSV records only; no quantum engine or array load."""
from datetime import datetime, timezone
from decimal import Decimal, getcontext
import csv
import hashlib
import json
from pathlib import Path
import re
import signal

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("reports/pyscf-spinosin-continuation-20261031-v3")
OUT = Path(__file__).resolve().parent
PHASE = "spinosin-native4-frequency-001"
getcontext().prec = 40


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dec(value):
    return Decimal(str(value))


def micros(value):
    return dec(value).quantize(Decimal("0.000001"))


def main():
    for name in ("README.md", "audit.json", "source_sha256.json", "index.json"):
        require(not (OUT / name).exists(), "Existing evidence refused: " + name)
    sources = {}

    def read(relative, expected=None, kind="json"):
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), "Source missing or symlink: " + str(relative))
        digest = sha(path)
        require(expected is None or digest == expected, "Source SHA differs: " + str(relative))
        sources[str(relative)] = {"sha256": digest, "bytes": path.stat().st_size, "read_kind": kind}
        if kind == "json":
            return json.loads(path.read_text())
        if kind == "csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        if kind == "text":
            return path.read_text()
        if kind == "binary_SHA256_only_not_HDF5_parse":
            return None
        raise ValueError("Unreviewed read type")

    ledger = read(BASE / "cpu_budget.json", "19fbbc80b5e839dc645f10a467bb54f32fdebae14309a95d824520cd885d3d2c")
    state = read(BASE / "pipeline-native4-001/state.json", "a2346f01bb7997c0f5fdf84e23ee603cea8f1be8e484007791d76217bd23eca5")
    require(state["ended_at_utc"] and state["ledger_sha256_at_end"] == sources[str(BASE / "cpu_budget.json")]["sha256"], "Pipeline is not closed against this ledger")
    require(all(row["status"] != "RUNNING" and row.get("ended_at_utc") for row in ledger["phases"]), "Ledger has an open phase")
    rows = [row for row in ledger["phases"] if row["phase"] == PHASE]
    require(len(rows) == 1, "Closed frequency phase must be unique")
    row = rows[0]
    result = read(BASE / PHASE / "result.json", "fc428a4c22b05d00246b68418817e2bd6a9fa9c4a5459a0dfc29d1d28a3393fa")
    manifest = read(BASE / PHASE / "hessian_cache_manifest.json", "c96d9450775e98180352935f0beba4e0df40eba7f9821a80802c783b902fe00a")
    proof = read(BASE / "execution" / (PHASE + ".log.supervisor-proof.json"), row["supervisor_proof_sha256"])
    execution_log = read(BASE / "execution" / (PHASE + ".log"), row["log_sha256"], "text")
    event_text = read(BASE / PHASE / "events.jsonl", kind="text")
    read(BASE / PHASE / "pyscf.log", kind="text")
    events = [json.loads(line) for line in event_text.splitlines() if line.strip()]
    worker_source = read(Path("scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"), result["script_sha256"], "text")
    supervisor_source = read(Path("scripts/run_pyscf_monitored_budget_20261031.py"), proof["supervisor_script_sha256"], "text")
    require('raise BudgetStop("Worker interrupted by resource supervisor signal " + str(signum))' in worker_source, "Signal handler wording differs")
    summary = read(BASE / "final-analysis/summary.json")
    analysis_readme = read(BASE / "final-analysis/README.md", kind="text")
    validated = read(BASE / "final-analysis/spinosin_validated_all_frequencies.csv", kind="csv")
    low = read(BASE / "final-analysis/spinosin_validated_0p1_1p5_thz_modes.csv", kind="csv")
    computed = read(BASE / "final-analysis/computed_frequencies_including_unvalidated.csv", kind="csv")
    pairs = read(BASE / "final-analysis/experiment_theory_pairs.csv", kind="csv")
    controls = read(BASE / "final-analysis/validation_controls_only.csv", kind="csv")
    read(BASE / "resource-limit-disclosure/index.json", "a5288e879bf4d95eb83f8b7110cfb3d00dcf7788851ad9b181bef7aacd180ec2")
    require(row["status"] == "PARTIAL_OR_FAILED" and row["exit_code"] == 2 and result["status"] == "PARTIAL", "Actual exit differs")
    require(result["interrupted"] is True and result["frequencies_status"] == "NOT_RUN", "Actual scientific state differs")
    match = re.fullmatch(r"Worker interrupted by resource supervisor signal (\d+)", result["stop_reason"])
    require(match and int(match.group(1)) == 24 and signal.Signals(24).name == "SIGXCPU", "Observed handler signal differs")
    require(result["stop_reason"] in execution_log and manifest["stop_reason"] == result["stop_reason"], "Stop evidence differs")
    require(manifest["exception_type"] == "BudgetStop" and manifest["status"] == "PARTIAL", "Capture did not record partial interruption")
    require(result["resource_supervision"]["proof"] == proof and result["resource_supervision"]["proof_sha256"] == row["supervisor_proof_sha256"], "Launch proof binding differs")
    require(proof["cpu_soft_limit_seconds"] == row["cpu_soft_limit_seconds"] == 86280 and proof["cpu_hard_limit_seconds"] == row["cpu_hard_limit_seconds"] == 86400, "Historical CPU configuration differs")
    stop_flags = {key: row[key] for key in ("wall_terminated", "hard_killed", "memory_terminated", "monitor_failed")}
    require(all(flag is False for flag in stop_flags.values()), "Actual parent stop flags differ")
    counters = result["native_direct_driver_control"]
    require(counters["native_calls_entered"] == counters["native_calls_returned"] == 155, "Native counter evidence differs")
    require(manifest["partial_hessian"] is None and manifest["H1AO_atoms"] == [] and manifest["response_blocks"] == [], "Completed intermediate evidence differs")
    require(result["hessian_cache_manifest_sha256"] == sources[str(BASE / PHASE / "hessian_cache_manifest.json")]["sha256"], "Final result cache seal differs")
    cache = Path(manifest["cache_root"])
    cache_entries = sorted(p.name for p in cache.iterdir()) if cache.is_dir() else []
    require(cache.is_dir() and cache_entries == [], "Cache root contains unreviewed files")
    absent_names = ["frequencies.csv", "normal_modes.npy", "hessian_hartree_per_bohr2.npy", "hessian_used_for_analysis.npy"]
    absent = {name: not (ROOT / BASE / PHASE / name).exists() for name in absent_names}
    require(all(absent.values()), "Unexpected frequency or complete Hessian artifact")
    read(BASE / PHASE / "calculation.chk", result["checkpoint_sha256"], "binary_SHA256_only_not_HDF5_parse")
    require(result["scf_converged"] and result["final_gradient_thresholds_passed"], "Fresh SCF/gradient did not finish")
    require(not validated and not low and all(mode["identity"] == "water" for mode in computed), "Frequency identities or empty result evidence differs")
    require(summary["computed_Spinosin_frequency_rows_including_unvalidated"] == summary["validated_spinosin_frequency_rows"] == 0, "Analyzer Spinosin count differs")
    require(summary["water_and_ethanol_are_validation_only"] is True and {item["identity"] for item in controls} == {"water", "validation_ethanol"}, "Validation scope differs")
    require(all(not item["theoretical_mode"] and not item["theoretical_frequency_thz"] and item["comparison_status"] == "BLOCKED_NO_VALIDATED_SPINOSIN_MODES_IN_RANGE" for item in pairs), "An experimental comparison falsely contains theory")
    require("未计算保持空表，不推断零吸收" in analysis_readme, "Existing report lacks empty-table caveat")
    require(any(event["event"] == "hessian_started" for event in events) and not any(event["event"] == "frequencies_complete" for event in events), "Execution event milestones differ")
    actual, worker = micros(row["actual_cpu_seconds"]), dec(result["process_cpu_seconds"])
    hard, soft = dec(row["cpu_hard_limit_seconds"]), dec(row["cpu_soft_limit_seconds"])
    total = sum((micros(item["actual_cpu_seconds"]) for item in ledger["phases"]), Decimal(0))
    require(total == Decimal("111023.488913") and actual == Decimal("89154.115469"), "Recomputed closed costs differ")
    remaining = dec(ledger["quantum_cpu_cap_seconds"]) - total
    require(remaining == Decimal("61776.511087") and abs(total - dec(ledger["consumed_cpu_seconds"])) < Decimal("1e-8"), "Actual full-cost charge differs")
    now = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema": "research4-closed-spinosin-frequency-audit-v1", "created_at_utc": now,
        "audit_status": "DONE_READ_ONLY_CLOSED_PHASE_AUDIT", "scientific_frequency_status": "PARTIAL_NO_SPINOSIN_FREQUENCIES_COMPUTED",
        "phase": PHASE, "started_at_utc": row["started_at_utc"], "ended_at_utc": row["ended_at_utc"],
        "root_closed_release": {"ledger_sha256": sources[str(BASE / "cpu_budget.json")]["sha256"], "pipeline_state_sha256": sources[str(BASE / "pipeline-native4-001/state.json")]["sha256"], "no_RUNNING_ledger_rows": True, "pipeline_has_terminal_status_and_end_time": True},
        "parent_exit": {"status": row["status"], "exit_code": row["exit_code"], **stop_flags},
        "signal_evidence": {"observed_handler_signal_number": 24, "observed_handler_signal_name": "SIGXCPU", "verbatim_stop_reason": result["stop_reason"], "exception_type": manifest["exception_type"], "sender_identity": "NOT_RECORDED", "exact_delivery_and_overshoot_cause": "UNPROVEN", "generic_handler_word_supervisor_is_not_sender_proof": True, "siginfo_or_sender_trace_saved": False},
        "CPU_accounting": {"configured_phase_grant_seconds": row["granted_cpu_seconds"], "configured_startup_soft_seconds": int(soft), "configured_startup_hard_seconds": int(hard), "actual_wait4_CPU_seconds": float(actual), "worker_process_CPU_seconds": float(worker), "wait4_above_hard_seconds": float(actual-hard), "wait4_above_hard_percent": float((actual-hard)/hard*100), "wait4_above_soft_seconds": float(actual-soft), "parent_minus_worker_seconds": float(actual-worker), "strict_phase_wait4_bound_86400_achieved": False, "actual_whole_closed_V3_CPU_seconds": float(total), "configured_whole_V3_budget_seconds": ledger["quantum_cpu_cap_seconds"], "remaining_closed_V3_budget_seconds": float(remaining), "actual_usage_fully_charged_not_clipped_to_configured_limits": True, "whole_closed_V3_cost_below_configured_budget": True, "raw_ledger_wait4_decimal": str(row["actual_cpu_seconds"]), "raw_ledger_total_decimal": str(ledger["consumed_cpu_seconds"]), "presentation_note": "wait4 costs are rounded to microseconds for presentation; archived JSON float serialization is retained above. No historical value was rewritten."},
        "wall_and_memory_observations": {"elapsed_wall_seconds": row["elapsed_wall_seconds"], "configured_wall_seconds": row["wall_limit_seconds"], "sampled_group_RSS_peak_bytes": row["peak_group_rss_bytes_sampled"], "sampled_cgroup_peak_bytes": row["peak_cgroup_memory_bytes_sampled"], "AS_24_GiB_is_virtual_not_physical_RAM": True, "RSS_guard_is_best_effort_not_hard_RSS": True, "false_stop_flags_do_not_prove_all_possible_memory_pressure_absent": True},
        "completed_SCF_and_gradient_metadata": {"scf_converged": result["scf_converged"], "energy_hartree": result["energy_hartree"], "fresh_gradient_thresholds_passed": result["final_gradient_thresholds_passed"], "atom_RMS_hartree_per_bohr": result["atom_rms_hartree_per_bohr"], "atom_max_hartree_per_bohr": result["atom_max_hartree_per_bohr"], "this_is_not_a_positive_frequency_minimum_verification": True, "gradient_arrays_reloaded_by_this_auditor": False},
        "Hessian_completion": {"native_driver_calls_entered": 155, "native_driver_calls_returned": 155, "native_batch_signature_counts": counters["density_batch_signature_counts"], "native_calls_do_not_prove_full_partial_Hessian_returned": True, "partial_hessian_manifest_record": None, "completed_H1AO_atom_records": 0, "completed_response_block_records": 0, "cache_root_exists": True, "cache_root_entries": cache_entries, "expected_final_artifacts_absent": absent, "complete_or_resumable_Hessian_available": False, "cross_process_Hessian_restore": "NOT_IMPLEMENTED_SAVE_ONLY", "SCF_checkpoint": {"sha256": result["checkpoint_sha256"], "bytes": (ROOT / BASE / PHASE / "calculation.chk").stat().st_size, "auditor_access": "BYTES_HASH_ONLY_NO_HDF5_OR_ORBITAL_PARSE", "use": "Potential SCF initial guess only; requires future geometry/method/SCF/gradient checks. Not a vibration/Hessian restart."}},
        "frequency_evidence": {"computed_Spinosin_frequency_records": 0, "validated_Spinosin_frequency_records": 0, "validated_Spinosin_low_band_records": 0, "physical_vibrational_or_low_frequency_peak_count_determined": False, "expected_219_is_not_an_observed_count": True, "empty_CSV_does_not_mean_zero_absorption_or_absent_low_frequency_modes": True, "computed_main_table_water_validation_records": len(computed), "water_and_ethanol_validation_only": True, "experimental_candidate_pair_rows": len(pairs), "candidate_theory_fields_all_empty_and_BLOCKED": True, "IR_intensity_status": "NOT_COMPUTED", "chemical_assignment_confirmed": False},
        "scope": {"new_quantum_or_integral_evaluations": 0, "NPY_array_loads": 0, "checkpoint_HDF5_parses": 0, "active_process_signals": 0, "frozen_sources_old_reports_results_or_ledger_modified": False, "classification_or_raw_spectra_modified": False, "Git_operations": 0},
        "builder_source_sha256": sha(Path(__file__)), "source_hash_manifest": "source_sha256.json"
    }
    readme = f"""# 斯皮诺素频率阶段：闭合结果与失败证据

审计状态：**DONE（只读证据核查）**。量子阶段状态：**PARTIAL；尚未得到斯皮诺素振动频率**。生成时间：{now}。本附录只新增文件，不更改冻结源、原账本、结果和已有报告。

本次依据root绑定的最终账本SHA `19fbbc80b5e839dc645f10a467bb54f32fdebae14309a95d824520cd885d3d2c` 与pipeline state SHA `a2346f01bb7997c0f5fdf84e23ee603cea8f1be8e484007791d76217bd23eca5`，确认所有账本阶段已经结束，控制器状态为PARTIAL并有结束时间。

## 实际退出及CPU用量

频率阶段 `{PHASE}` 在{row['ended_at_utc']}结束，父进程exit code **2**。worker和缓存manifest记录实际捕获 **signal 24（SIGXCPU）**。原始文本为：

> {result['stop_reason']}

**这段泛用handler文本不能证明信号由supervisor发送。** 冻结源码1058—1066行对SIGTERM／SIGXCPU统一使用这句话，没有保存siginfo／发送者轨迹。能够确认的是捕获了24号信号并经BudgetStop路径保存PARTIAL；确切发送者、信号递送过程及CPU超额根因均**未直接记录／未证实**。不能以措辞推断墙钟监控、内存保护或某个人主动终止。

| 资源核查项 | CPU秒 |
|---|---:|
| 启动配置soft | 86280 |
| 启动配置hard／阶段额度 | 86400 |
| 父进程wait4实际user＋system | {actual} |
| worker process CPU计时 | {worker} |
| wait4超出hard | {actual-hard} |
| wait4超出soft | {actual-soft} |
| wait4与worker计时数值差 | {actual-worker} |
| 全部已闭合V3阶段实测合计 | {total} |
| V3新有限预算配置 | 172800 |
| 按完整实测扣除后的余额 | {remaining} |

该阶段wait4超出配置hard约 **{(actual-hard)/hard*100:.9f}%**；严格86400秒实测上界未实现。退出失败没有把真实消耗截断为86400，原账本完整计入实际费用。四阶段V3合计低于其172800秒配置预算；余额不是已经完成的科研结果，也不构成自动启动新计算的声明。父进程标志 `wall_terminated`、`hard_killed`、`memory_terminated`、`monitor_failed` 均为false。它们描述父监控记录，不能反推所有潜在内存压力不存在。

实际墙钟{row['elapsed_wall_seconds']:.6f}秒，配置43200秒。24 GiB AS仍是虚拟地址空间，不是物理RAM；RSS轮询属于best-effort而非hard RSS。优化阶段的独立CPU披露见[先前附录](../resource-limit-disclosure/README.md)，该附录只讨论当时已结束的优化前缀，不能据此判断后来频率阶段合规。

## 已完成与未完成的计算边界

本频率尝试先完成同方法fresh SCF和梯度，元数据记录能量{result['energy_hartree']} Eh、原子梯度RMS {result['atom_rms_hartree_per_bohr']}、最大值{result['atom_max_hartree_per_bohr']} Eh/Bohr，阈值检查通过。本审计没有重新读取梯度数组或计算导数；这些是已保存元数据，不是新增复算结果，也不能验证正频极小点。

native driver记录 **155次进入、155次返回**。这是底层调用计数，不等于完整partial Hessian、Hessian核或振动计算返回。最终manifest为`PARTIAL`，`partial_hessian=null`、`H1AO_atoms=[]`、`response_blocks=[]`；本次闭合缓存目录为空。标准捕获器只有原函数正常返回并校验、保存后才建立完成记录，因此**没有可恢复的完整partial Hessian、H1／响应块或全Hessian**。预计中间量尺寸、geometry文件或正常模式准备文件都不能作为已计算振型。

最终raw Hessian、analysis Hessian、frequencies.csv、normal_modes.npy均不存在。SCF checkpoint实际保存，SHA为`{result['checkpoint_sha256']}`，大小5704080字节；本审计只做字节SHA核验，没有打开HDF5或提取轨道。它最多可供后续合规SCF初猜使用，不能宣称从已有Hessian／振动继续；任何后续工作仍需严格几何、方法、SCF及梯度门槛。跨进程Hessian加载仍`NOT_IMPLEMENTED_SAVE_ONLY`，没有伪造恢复命令。

## 频率表与理论解释的限制

[原分析汇总](../final-analysis/summary.json)的斯皮诺素已计算／已验证频率记录均为0。两份斯皮诺素CSV为空，仅表示**未计算成功**，绝不表示该分子没有低频振动、没有吸收或某成分不存在。75原子非线性分子的219个预期模式不是实际得到219个模式。主频率表中的3条真实记录均来自水；水及乙醇21模式仅作数值验证，不能填入斯皮诺素机制结果。

20个实验候选对应表均保留空理论字段，状态为`BLOCKED_NO_VALIDATED_SPINOSIN_MODES_IN_RANGE`；没有可用斯皮诺素理论峰进行化学归属。原[分析README](../final-analysis/README.md)已明确“未计算保持空表，不推断零吸收”，不存在把空CSV解释为无低频峰的结论。IR始终`NOT_COMPUTED`；没有修改96.0234%的分类结果、光谱或标签。

完整机器可读审计见[audit.json](audit.json)，来源及冻结源码SHA见[source_sha256.json](source_sha256.json)，本目录文件索引见[index.json](index.json)。索引不包含自身SHA，由上级交付绑定。没有启动量子计算、积分、信号或Git操作；所有科学来源仅只读。
"""
    for relative, item in sources.items():
        require(sha(ROOT / relative) == item["sha256"], "Source changed during closed audit: " + relative)
    for name, value in (("audit.json", audit), ("source_sha256.json", {"status": "CLOSED_SOURCE_BYTES_VERIFIED", "created_at_utc": now, "files": sources})):
        with (OUT / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    with (OUT / "README.md").open("x") as handle:
        handle.write(readme)
    files = [OUT / name for name in ("README.md", "audit.json", "source_sha256.json", "audit_closure.py")]
    index = {"status": "FROZEN_READ_ONLY_FREQUENCY_CLOSURE_AUDIT", "created_at_utc": now, "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(files)]}
    with (OUT / "index.json").open("x") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"audit_status": audit["audit_status"], "frequency_status": audit["scientific_frequency_status"], "index_sha256": sha(OUT / "index.json"), "source_count": len(sources)}))


if __name__ == "__main__":
    main()
