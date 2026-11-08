#!/usr/bin/env python3
"""Build an append-only disclosure from a fixed, closed optimization snapshot.

Standard library only. Never opens the active ledger, arrays, or checkpoints;
never imports a quantum engine. Existing output files are refused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, getcontext
import hashlib
import json
import os
from pathlib import Path

getcontext().prec = 40
ROOT = Path(__file__).resolve().parents[3]
BASE = Path("reports/pyscf-spinosin-continuation-20261031-v3")
PHASE = "spinosin-native4-optimization-001"
FIXED = {
    "closed_ledger_snapshot": (BASE / "optimization-completion-snapshot/ledger_at_optimization_completion.json", "59d1ae734106f91d3b0dfd038502c4930a26c1c17d7ccd9ca27636a5e85aef9a"),
    "snapshot_metadata": (BASE / "optimization-completion-snapshot/metadata.json", None),
    "optimization_result": (BASE / "spinosin-native4-optimization-001/result.json", "299539676308f3bac4bc01d3a0c1554a9b636228d91f7acd05e9fa08c0b7adeb"),
    "optimization_launch_proof": (BASE / "execution/spinosin-native4-optimization-001.log.supervisor-proof.json", "116e56835b2bc2cad256b00bc076b732291ef97620ebcd9c5a07236e4630f0d1"),
    "scientific_optimization_audit": (BASE / "opt-native4-completion-audit/science-point-check/science_audit.json", "e795aac15b254236790a8f10c292fc9318d9e565ec78e8ae1cca042b41f3fb6a"),
    "operational_CPU_audit": (BASE / "cpu-limit-overshoot-audit/closed_opt_cpu_audit.json", "9cbf236d894dae65a40732d693322a0c5c73744a63f4159979915e4aadb4699d"),
    "operational_CPU_audit_index": (BASE / "cpu-limit-overshoot-audit/index.json", "f525821bd6141e0b201013eebbcb2fbe7b168b992e707916a726142af260cf7a"),
    "kernel_clock_audit": (BASE / "cpu-overshoot-kernel-audit/audit.json", "5fe3cc4f0a6559bf64c106aee2e72cb061b6422915072245f4345644cb5b4e52"),
    "kernel_clock_audit_index": (BASE / "cpu-overshoot-kernel-audit/index.json", "e4067c1daee67bda6ee108e536cc9e71bbf37bf4eed3a98fd156eafc35dc0c91"),
    "frozen_authorization": (BASE / "authorization.json", "4d2a665bfeab6b1a7fb13d0fac8525640a228e74713b014596511b8d34276763"),
    "frozen_resource_protocol": (Path("docs/PYSCF_MONITORED_DIRECT_PROTOCOL_20261031.md"), "952c08e74bdda2338b7493c52426f32f03e47cf2d218643d22bd23f00b3ea863"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value) -> Decimal:
    return Decimal(str(value))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build(output: Path) -> dict:
    output = output.resolve()
    require(not output.is_symlink(), "Symlink output refused")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "disclosure.json", "index.json"):
        require(not (output / name).exists(), "Existing output refused: " + name)
    sources, data = {}, {}
    for label, (relative, expected) in FIXED.items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), "Missing or symlink source: " + str(relative))
        digest = sha(path)
        require(expected is None or digest == expected, "Fixed source SHA mismatch: " + label)
        sources[label] = {"path": str(relative), "sha256": digest, "bytes": path.stat().st_size}
        if path.suffix == ".json":
            data[label] = json.loads(path.read_text())
    # Verify every static file in both earlier audit seals; no live ledger is read.
    seals = []
    for label in ("operational_CPU_audit_index", "kernel_clock_audit_index"):
        index_path = ROOT / sources[label]["path"]
        entries = data[label]["files"]
        for entry in entries:
            relative = Path(entry["path"])
            path = ROOT / relative if relative.parts[0] == "reports" else index_path.parent / relative
            require(path.is_file() and not path.is_symlink(), "Sealed source missing")
            require(sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], "Earlier audit seal mismatch")
        seals.append({"source_id": label, "files_verified": len(entries)})

    ledger, result = data["closed_ledger_snapshot"], data["optimization_result"]
    metadata, proof = data["snapshot_metadata"], data["optimization_launch_proof"]
    operational, kernel = data["operational_CPU_audit"], data["kernel_clock_audit"]
    authorization, science = data["frozen_authorization"], data["scientific_optimization_audit"]
    require(metadata["verified_ledger_sha256"] == sources["closed_ledger_snapshot"]["sha256"], "Snapshot seal differs")
    require(all(row["status"] == "EXITED" and row.get("ended_at_utc") for row in ledger["phases"]), "Snapshot is not a closed phase prefix")
    selected = [row for row in ledger["phases"] if row["phase"] == PHASE]
    require(len(selected) == 1, "Optimization phase must be unique")
    row = selected[0]
    require(row["exit_code"] == 0 and result["status"] == "DONE", "Actual optimization did not complete")
    require(result["geometry_optimized"] and result["optimizer_converged"] and result["scf_converged"] and result["final_gradient_thresholds_passed"], "Optimization/fresh gradient flags differ")
    require(science["status"] == "PASS_SCIENTIFIC_OPTIMIZATION_COMPLETION" and science["fresh_gradient_independently_completed"], "Independent optimization audit differs")
    require(science["source_result_sha256"] == sources["optimization_result"]["sha256"], "Scientific audit result binding differs")
    require(row["supervisor_proof_sha256"] == sources["optimization_launch_proof"]["sha256"] == result["resource_supervision"]["proof_sha256"], "Proof bindings differ")
    require(result["resource_supervision"]["proof"] == proof, "Recorded launch proof differs")
    require(row["cpu_soft_limit_seconds"] == proof["cpu_soft_limit_seconds"] == 21480, "Soft CPU limit differs")
    require(row["cpu_hard_limit_seconds"] == proof["cpu_hard_limit_seconds"] == row["granted_cpu_seconds"] == 21600, "Hard CPU/grant differs")
    require(result["resource_supervision"]["observed_CPU_limits_seconds"] == [21480, 21600], "Worker startup limits differ")
    require(authorization["max_quantum_cpu_seconds"] == ledger["quantum_cpu_cap_seconds"] == 172800, "Separate finite budget differs")
    require(ledger["authorization_sha256"] == sources["frozen_authorization"]["sha256"], "Authorization binding differs")
    require(proof["address_space_hard_limit_bytes"] == proof["address_space_soft_limit_bytes"] == 24 * 2**30, "AS configuration differs")
    require(proof["rss_stop_bytes"] == 10 * 2**30 and proof["cgroup_kill_bytes"] == 12 * 2**30 and proof["expected_cgroup_memory_max_bytes"] == 16 * 2**30, "Historical memory policy differs")

    actual, worker, hard, soft = map(number, (row["actual_cpu_seconds"], result["process_cpu_seconds"], row["cpu_hard_limit_seconds"], row["cpu_soft_limit_seconds"]))
    over = actual - hard
    require(over > 0, "Expected discrepancy is absent; do not silently change scope")
    require(abs(actual - number(operational["actual_parent_wait4_CPU_seconds"])) < Decimal("1e-9"), "Operational CPU evidence differs")
    require(abs(worker - number(kernel["confirmed"]["worker_reported_process_CPU_seconds"])) < Decimal("1e-9"), "Kernel audit worker evidence differs")
    require(kernel["unproven"]["exact_cause_of_81_second_overshoot"] == "UNPROVEN", "Root-cause status differs")
    completed = sum((number(item["actual_cpu_seconds"]) for item in ledger["phases"]), Decimal(0))
    require(abs(completed - number(ledger["consumed_cpu_seconds"])) < Decimal("1e-9"), "Historical actual CPU sum differs")
    require(completed < number(ledger["quantum_cpu_cap_seconds"]), "Historical prefix exceeds separate budget")
    stop_flags = {key: row[key] for key in ("wall_terminated", "hard_killed", "memory_terminated", "monitor_failed")}
    require(all(value is False for value in stop_flags.values()), "Actual archived stop flags differ")
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema": "research4-static-resource-limit-disclosure-v1",
        "created_at_utc": now,
        "status": "DONE_STATIC_DISCLOSURE_CLOSED_OPTIMIZATION_ONLY",
        "artifact_kind": "ADDITIVE_RESOURCE_DISCLOSURE_NOT_WORKER_RESULT_OR_FINAL_BATCH_REPORT",
        "audited_phase": PHASE,
        "closed_phase_started_at_utc": row["started_at_utc"],
        "closed_phase_ended_at_utc": row["ended_at_utc"],
        "configured_CPU_seconds": {"phase_grant": int(hard), "RLIMIT_CPU_soft_at_startup": int(soft), "RLIMIT_CPU_hard_at_startup": int(hard)},
        "actual_CPU_seconds": {"parent_wait4_user_plus_system": float(actual), "worker_process_clock": float(worker)},
        "derived_accounting": {
            "parent_above_hard_seconds": float(over),
            "parent_above_hard_percent": float(over / hard * 100),
            "parent_above_soft_seconds": float(actual - soft),
            "worker_above_hard_seconds": float(worker - hard),
            "parent_minus_worker_seconds": float(actual - worker),
            "parent_wait4_over_hard_ratio": float(actual / hard),
            "decimal_parent_above_hard_seconds": str(over),
            "decimal_parent_above_hard_percent": str(over / hard * 100),
            "formulas": {"overshoot": "parent_wait4_CPU - configured_hard_CPU", "overshoot_percent": "100 * overshoot / configured_hard_CPU"},
            "arithmetic_note": "Decimal arithmetic on serialized source values; extra displayed digits do not imply additional timing accuracy.",
            "parent_worker_difference_interpretation": "Different measurement intervals/clocks; this numerical difference is not a proven causal decomposition."
        },
        "phase_exit": {"status": row["status"], "exit_code": row["exit_code"], **stop_flags},
        "strict_phase_wait4_bound_21600_seconds_achieved": False,
        "actual_wait4_usage_fully_charged_in_historical_snapshot": True,
        "scientific_optimization_status": "DONE_OPTIMIZER_AND_INDEPENDENT_FRESH_GRADIENT_PASSED",
        "scientific_completion_is_not_CPU_limit_compliance": True,
        "exact_overshoot_root_cause": "UNPROVEN",
        "root_cause_limitations": ["No final raw PROF clock or signal-delivery trace from this long phase.", "Source-supported clock/accounting mechanisms are hypotheses, not measured causes.", "Short ordinary CPU probes in the prior audit do not establish exact enforcement for a six-hour quantum phase."],
        "separate_V3_budget_historical_configuration": {
            "quantum_CPU_seconds": authorization["max_quantum_cpu_seconds"], "quantum_CPU_hours": 48,
            "run_wall_seconds": authorization["max_run_wall_seconds"], "run_wall_hours": 24,
            "max_phase_CPU_seconds": authorization["max_phase_cpu_seconds"], "max_phase_wall_seconds": authorization["max_phase_wall_seconds"],
            "prior_V2_CPU_seconds_preserved_separately": ledger["prior_consumed_cpu_seconds"],
            "closed_prefix_snapshot_CPU_seconds": float(completed),
            "closed_prefix_snapshot_remaining_CPU_seconds": ledger["remaining_quantum_cpu_seconds"],
            "closed_prefix_only_below_48_CPU_hour_budget": True,
            "not_a_current_or_final_batch_cost_or_compliance_claim": True
        },
        "historical_memory_configuration": {
            "RLIMIT_AS_soft_and_hard_bytes": proof["address_space_hard_limit_bytes"], "RLIMIT_AS_GiB": 24,
            "AS_is_virtual_address_space_not_physical_RAM": True,
            "rss_termination_threshold_bytes": proof["rss_stop_bytes"], "cgroup_termination_threshold_bytes": proof["cgroup_kill_bytes"],
            "original_cgroup_kernel_memory_boundary_bytes": proof["expected_cgroup_memory_max_bytes"],
            "monitor_interval_seconds": proof["monitor_interval_seconds"], "termination_grace_seconds": proof["memory_kill_grace_seconds"],
            "guard_semantics": "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT",
            "sampling_cannot_exclude_transient_overshoot_or_guarantee_headroom": True,
            "startup_proof_is_not_future_monitor_health_or_a_physical_memory_measurement": True,
            "closed_opt_sampled_peak_group_RSS_bytes": row["peak_group_rss_bytes_sampled"],
            "closed_opt_sampled_peak_cgroup_bytes": row["peak_cgroup_memory_bytes_sampled"]
        },
        "frequency_phase_completion_and_resource_compliance": "NOT_ASSESSED_NO_ACTIVE_FREQUENCY_FILES_READ",
        "frequency_or_minimum_claim_from_this_appendix": False,
        "expected_219_is_not_an_observed_frequency_count": True,
        "scope": {"new_quantum_evaluations": 0, "active_ledger_reads": 0, "active_HDF_NPY_checkpoint_reads": 0, "active_process_signals": 0, "existing_files_or_limits_modified": False, "Git_operations": 0},
        "source_evidence": sources,
        "prior_audit_seal_verification": seals,
        "source_files_unchanged_during_generation": True,
        "builder_source_sha256": sha(Path(__file__))
    }
    def link(label):
        return os.path.relpath(ROOT / sources[label]["path"], output)
    readme = f"""# 资源限额披露：已结束的斯皮诺素优化阶段

这是独立追加的静态披露，依据[已封存的优化完成账本快照]({link('closed_ledger_snapshot')})及[优化结果]({link('optimization_result')})；不修改冻结程序、原账本、结果或限额。生成时间：{now}。

**该阶段配置的21600秒CPU硬限额，未成为实际wait4用量的严格上界。** 科学优化及独立fresh SCF／梯度已完成，不等于CPU限额严格达标。

| 已结束阶段 `{PHASE}` | CPU秒 |
|---|---:|
| 配置阶段额度／启动时RLIMIT_CPU hard | 21600 |
| 启动时RLIMIT_CPU soft | 21480 |
| 父进程wait4实测 user＋system | {actual} |
| worker报告process CPU计时 | {worker} |
| wait4超出配置hard | {over} |
| wait4超出配置soft | {actual-soft} |
| worker计时超出配置hard | {worker-hard} |
| wait4与worker计时数值差 | {actual-worker} |

超出hard的比例为 **{over/hard*100:.9f}%**（`100 × (21681.889694 − 21600) / 21600`）；不是准确率变化。不同计时的统计范围／时钟不同，二者之差不能单独证明启动、收尾或某种内核机制贡献了多少时间。机器可读原值、公式与来源SHA见[disclosure.json](disclosure.json)。

父进程记录 `EXITED`、exit code `0`；归档标志 `wall_terminated`、`hard_killed`、`memory_terminated`、`monitor_failed` 均为false。真实21681.889694秒已完整计入快照，未改小或截断为21600。退出成功或没有stop标志均不能证明CPU用量未超配置。启动证明记录的soft／hard确为21480／21600。

超额的**准确根因仍未证实（UNPROVEN）**。[运行与普通CPU短测审查]({link('operational_CPU_audit')})、[内核／时钟源码审查]({link('kernel_clock_audit')})保留了证据和限制。不同CPU时钟／计账机制属于源码支持的可能解释；没有该长阶段最终raw PROF计数及完整信号递送轨迹，不能据此确认因果。普通小进程短测不是六小时量子阶段严格限额的验证。

## 新有限预算与历史快照的范围

[冻结授权]({link('frozen_authorization')})与[冻结资源协议]({link('frozen_resource_protocol')})定义的是**另行授权的有限V3批次**：48 CPU小时（172800秒）、24墙钟小时，最多4线程；单阶段另有有限额度。并非无限运行，也未重置或抹去V2已发生用量。V2成本在原账本及来源链中单独保留。

本历史快照只包括已结束阶段前缀，其实测CPU合计为{completed}秒，低于该新批次172800秒配置预算；快照所记剩余额度为{ledger['remaining_quantum_cpu_seconds']}秒。**这不是当前或最终全批次成本，也不保证未来阶段不超配置。** 不把后续阶段申请额度当成实际消耗，也不把本附录准备完成当成量子计算完成。

## 地址空间、物理内存与尽力监测

历史启动配置的 **24 GiB RLIMIT_AS是虚拟地址空间上限，不是24 GiB物理RAM，也不是RSS限额**。原有16 GiB cgroup内存边界保持不变。父监视器按0.1秒采样，group RSS高于10 GiB请求终止、约1秒宽限内执行强制停止；cgroup用量高于12 GiB触发停止。其语义明确为 `BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT`，不是内核强制的10 GiB驻留内存硬限额；快速原生分配可能在采样间隔内越界，不能保证固定空余内存。历史启动证明不是当前监视器健康状态。

已结束优化阶段记录的采样峰值group RSS为{row['peak_group_rss_bytes_sampled']}字节，cgroup为{row['peak_cgroup_memory_bytes_sampled']}字节；这些是采样记录，不保证捕捉到所有瞬时峰值。这里只引用归档授权、协议、证明与历史快照，未读取当前系统或活动计算内存状态。

## 科学结论及本附录边界

[独立科学优化审查]({link('scientific_optimization_audit')})证明真实优化器收敛并完成同方法独立fresh SCF／梯度，和上述资源超额同时如实保留。**本附录不评价频率阶段是否完成、是否满足资源限额或是否为正频极小点。** 未读取活动频率HDF、NPY、checkpoint或实时账本，未启动任何SCF、梯度或Hessian计算；没有更改阈值、信号或模型。非线性75原子分子预期219个振动模式不等于已经得到219个模式，水和乙醇的软件验证模式也不能计作斯皮诺素结果。

可复算命令（必须指定全新输出目录，已有输出拒绝覆盖）：

```bash
python reports/pyscf-spinosin-continuation-20261031-v3/resource-limit-disclosure/build_disclosure.py --output /tmp/research4-resource-disclosure-fresh
```

本目录[index.json](index.json)保存文件SHA-256；索引不包含自身哈希，其哈希由上级交付记录绑定。生成器只使用Python标准库及固定静态来源，不导入量子引擎。
"""
    for label, item in sources.items():
        require(sha(ROOT / item["path"]) == item["sha256"], "Source changed during preparation: " + label)
    for name, text in (("disclosure.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n"), ("README.md", readme)):
        with (output / name).open("x") as handle:
            handle.write(text)
    outputs = [output / "README.md", output / "disclosure.json"]
    if Path(__file__).resolve().parent == output:
        outputs.append(Path(__file__).resolve())
    index = {"status": "FROZEN_ADDITIVE_CLOSED_OPT_RESOURCE_DISCLOSURE", "created_at_utc": now, "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(outputs)], "source_file_count": len(sources)}
    with (output / "index.json").open("x") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"status": payload["status"], "output": str(output), "index_sha256": sha(output / "index.json"), "source_files": len(sources), "sealed_audit_files_verified": sum(item["files_verified"] for item in seals)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))
