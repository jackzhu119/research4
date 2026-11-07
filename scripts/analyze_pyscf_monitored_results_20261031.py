"""Final, failure-aware V3 evidence audit; no SCF or Hessian evaluation.

Uses the frozen V2 pure array/plot helpers without editing or monkeypatching
them. Startup supervision proofs are audited as historical evidence; they are
not claims that RSS polling was a hard limit or that a former PID is still live.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/analyze_pyscf_continuation_v2_20261031.py"
SPEC = importlib.util.spec_from_file_location("frozen_failure_aware_v2_helpers", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(CORE)
sha, read, write, csvwrite = CORE.sha, CORE.read, CORE.write, CORE.csvwrite
SUPERVISOR = ROOT / "scripts/run_pyscf_monitored_budget_20261031.py"
WORKER = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
ETHANOL_VALIDATOR = ROOT / "scripts/validate_pyscf_native4_large_block_engineering_20261031.py"
PEAK_PATH, PEAK_SHA256 = CORE.PEAK_PATH, CORE.PEAK_SHA256
GUARD = "BEST_EFFORT_RSS_GUARD_NOT_HARD_RSS_LIMIT"


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def authorize(root, release, validator=None):
    """Validate the final V3 and bound prior V2 ledger independently."""
    ledger_path = root / "cpu_budget.json"; ledger = read(ledger_path)
    if (release.get("allow_analysis_export") is not True or release.get("cpu_budget_sha256") != sha(ledger_path)
            or not release.get("created_at_utc")):
        raise ValueError("Dated root release must bind the final monitored V3 ledger")
    if datetime.fromisoformat(release["created_at_utc"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Final release must include an actual timezone offset")
    if ledger.get("schema") != "research4-pyscf-monitored-budget-v3":
        raise ValueError("An independent monitored V3 ledger is required")
    if any(row.get("status") in ("RUNNING", "BUDGET_VIOLATION_REQUIRES_REVIEW") for row in ledger["phases"]):
        raise ValueError("No final export during active work or unresolved budget violation")
    pipeline_path = release.get("pipeline_state_path")
    if not pipeline_path: raise ValueError("Final V3 release must bind the actual closed automatic pipeline state")
    pipeline_path = resolved(pipeline_path)
    if not pipeline_path.is_relative_to(root): raise ValueError("Pipeline state must belong to the V3 run")
    pipeline_state = read(pipeline_path)
    if (sha(pipeline_path) != release.get("pipeline_state_sha256")
            or pipeline_state.get("status") not in ("DONE", "PARTIAL") or not pipeline_state.get("ended_at_utc")
            or pipeline_state.get("ledger_sha256_at_end") != sha(ledger_path)):
        raise ValueError("Automatic pipeline is still active, between stages or bound to another final ledger")
    if validator is None:
        spec = importlib.util.spec_from_file_location("official_monitored_v3_budget_audit", SUPERVISOR)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        validator = module.validate_ledger
    authorization = validator(ledger)
    if ledger.get("runner_sha256") != sha(SUPERVISOR): raise ValueError("Recorded V3 supervisor source changed")
    auth_path = resolved(ledger["authorization_path"])
    if sha(auth_path) != ledger["authorization_sha256"]: raise ValueError("V3 authorization bytes changed")
    source = resolved(ledger["source_checkpoint_path"])
    if sha(source) != ledger["source_checkpoint_sha256"]: raise ValueError("Immutable original checkpoint changed")
    prior_path = resolved(authorization["prior_closed_ledger_path"]); prior = read(prior_path)
    if (sha(prior_path) != authorization["prior_closed_ledger_sha256"]
            or prior.get("schema") != "research4-pyscf-continuation-budget-v2"
            or any(row.get("status") == "RUNNING" for row in prior["phases"])):
        raise ValueError("Bound prior V2 ledger differs or is still active")
    sums = {}
    for label, document in [("V3", ledger), ("prior_V2", prior)]:
        values = [float(row["actual_cpu_seconds"]) for row in document["phases"]]
        if (not all(math.isfinite(value) and value >= 0 for value in values)
                or not math.isclose(sum(values), document["consumed_cpu_seconds"], abs_tol=1e-7, rel_tol=1e-12)
                or document["consumed_cpu_seconds"] > document["quantum_cpu_cap_seconds"]):
            raise ValueError("Independent completed CPU sum/cap failed: " + label)
        sums[label] = sum(values)
    if not math.isclose(sums["prior_V2"], authorization["prior_consumed_cpu_seconds"], abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError("Prior actual CPU differs from explicitly bound usage")
    hashes = {str(path): sha(path) for path in (ledger_path, auth_path, source, prior_path, SUPERVISOR,
        WORKER, ETHANOL_VALIDATOR, Path(__file__), CORE_PATH, Path(CORE.LEGACY.__file__))}
    hashes[str(pipeline_path)] = sha(pipeline_path)
    return ledger, authorization, hashes, sums


def audit_closed_proof(root, ledger_path, row, input_hashes):
    """Audit archived startup contract, not live process or future monitor health."""
    log = resolved(row["log"])
    if not log.is_relative_to(root) or sha(log) != row["log_sha256"]:
        raise ValueError("Closed phase driver log path/hash differs")
    input_hashes[str(log)] = sha(log)
    proof_path = resolved(row["supervisor_proof_path"])
    if not proof_path.is_relative_to(root): raise ValueError("Proof path outside actual V3 run")
    if row["status"] == "FAILED_TO_START" and not row.get("supervisor_proof_sha256"):
        if proof_path.is_file(): input_hashes[str(proof_path)] = sha(proof_path)
        return {"status": "UNBOUND_OR_ABSENT_FAILED_TO_START", "frequency_acceptance_allowed": False,
                "startup_proof_does_not_prove_future_monitor_health": True}
    if not proof_path.is_relative_to(root) or sha(proof_path) != row.get("supervisor_proof_sha256"):
        raise ValueError("Closed phase proof path/hash differs from final ledger")
    proof = read(proof_path)
    expected = {"schema": "research4-pyscf-resource-supervisor-proof-v3", "guard_semantics": GUARD,
        "address_space_soft_limit_bytes": 24*1024**3, "address_space_hard_limit_bytes": 24*1024**3,
        "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3,
        "expected_cgroup_memory_max_bytes": 16*1024**3, "phase": row["phase"], "threads": row["threads"]}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("Archived resource proof differs from reviewed policy")
    if (resolved(proof["ledger_path"]) != ledger_path.resolve()
            or resolved(proof["supervisor_script_path"]) != SUPERVISOR
            or proof.get("supervisor_script_sha256") != sha(SUPERVISOR)
            or proof.get("child_pid") != row.get("pid")):
        raise ValueError("Proof supervisor, child or ledger identity differs")
    if any(type(proof.get(key)) is not int or proof[key] <= 0 for key in
           ("supervisor_pid", "supervisor_start_ticks", "child_pid", "child_start_ticks")):
        raise ValueError("Archived process/start-tick identities are incomplete")
    for key in ("cpu_soft_limit_seconds", "cpu_hard_limit_seconds", "wall_limit_seconds"):
        if type(proof.get(key)) is not int or proof[key] != row.get(key):
            raise ValueError("Proof CPU/wall budget differs: " + key)
    for key, maximum in (("monitor_interval_seconds", .1), ("memory_kill_grace_seconds", 1.0)):
        value = proof.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= maximum:
            raise ValueError("Unreviewed proof monitor timing")
    input_hashes[str(proof_path)] = sha(proof_path)
    eligible = row["status"] == "EXITED" and row.get("exit_code") == 0 and not any(
        row.get(key) for key in ("wall_terminated", "memory_terminated", "monitor_failed", "hard_killed"))
    return {"status": "ARCHIVED_STARTUP_CONTRACT_HASH_VERIFIED", "proof_sha256": sha(proof_path),
            "frequency_acceptance_allowed": bool(eligible), "AS_is_virtual_not_resident_limit": True,
            "RSS_monitor_best_effort_not_hard_limit": True,
            "startup_proof_does_not_prove_future_monitor_health": True}


def failed_geometry_attempt(path, phase, input_hashes):
    meta = read(path); input_hashes[str(path)] = sha(path)
    record = {"phase": phase, "cycle": meta.get("cycle"), "scf_converged": meta.get("scf_converged"),
              "claimed_energy_hartree": meta.get("energy_hartree"),
              "accepted_as_completed_converged_point": False, "reason": "SCF_NOT_CONVERGED"}
    for suffix in (".gradient.npy", ".coordinates_angstrom.npy", ".coordinates_bohr.npy", ".xyz"):
        artifact = path.with_suffix(suffix)
        if artifact.is_file(): input_hashes[str(artifact)] = sha(artifact)
    return record


def frequency_parent_audit(root, ledger, row, result, input_hashes):
    """Tie the true optimized parent to one successful supervised ledger owner."""
    CORE.frequency_parent_guard(root, row, result, input_hashes)
    if result.get("identity") == "water": return
    command = row["command"]
    path = resolved(command[command.index("--optimization-result") + 1])
    matches = []
    for candidate in ledger["phases"]:
        args = candidate.get("command", [])
        if "--output" in args and resolved(args[args.index("--output")+1]) == path.parent:
            matches.append(candidate)
    if len(matches) != 1 or matches[0].get("status") != "EXITED" or matches[0].get("exit_code") != 0:
        raise ValueError("Frequency parent lacks one successful closed supervised optimization owner")
    parent = read(path); owner = matches[0]
    if (parent.get("phase") != "spinosin-optimize" or parent.get("script_sha256") != sha(WORKER)
            or [resolved(value) for value in owner["command"] if str(value).endswith(".py")] != [WORKER]):
        raise ValueError("Frequency parent did not execute the recorded optimization source")
    proof = audit_closed_proof(root, root / "cpu_budget.json", owner, input_hashes)
    if not proof["frequency_acceptance_allowed"]: raise ValueError("Parent optimization supervision did not close cleanly")
    CORE.audit_final_gradient(path.parent, parent, input_hashes)
    input_hashes[str(path)] = sha(path)


def cost_plot(output, phases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    indices = np.arange(len(phases)); labels = [row["phase"] for row in phases]
    axes[0].bar(indices, [row["actual_cpu_seconds"]/60 for row in phases], color="#286090")
    axes[0].set(ylabel="Actual wait4 CPU / minutes", title="Actual compute cost; not wall time x threads")
    measured = [(i, row["peak_group_RSS_bytes_sampled"]) for i, row in enumerate(phases) if row["peak_group_RSS_bytes_sampled"] is not None]
    axes[1].bar([i for i, _ in measured], [value/1024**3 for _, value in measured], color="#ac5423")
    axes[1].set(ylabel="Sampled process-group RSS / GiB", title="Best-effort observed samples; missing values omitted")
    for axis in axes:
        axis.set(xticks=indices, xticklabels=labels)
        axis.tick_params(axis="x", labelrotation=30, labelsize=6)
    CORE.LEGACY.save_figure(fig, output, "actual_phase_costs"); plt.close(fig)


def computed_frequency_plot(output, computed):
    """Display signed computed values without promoting an unstable minimum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    target = [row for row in computed if row["identity"] == "spinosin"]
    for phase in dict.fromkeys(row["phase"] for row in target):
        records = [row for row in target if row["phase"] == phase]
        for axis, key in zip(axes, ("frequency_cm1", "frequency_thz")):
            axis.plot([row["mode"] for row in records], [row[key] for row in records], ".-", ms=2, lw=.8, label=phase)
    for axis, unit in zip(axes, ("cm$^{-1}$", "THz")):
        axis.axhline(0, c="gray", ls=":")
        axis.set(xlabel="Actual computed normal-mode index", ylabel="Signed mode frequency / " + unit)
        if target: axis.legend(fontsize=6)
        else: axis.text(.5, .5, "Spinosin frequencies NOT_RUN\nor no finalized computed arrays", transform=axis.transAxes, ha="center")
    fig.suptitle("Actual computed Spinosin modes, including unvalidated negatives; not IR intensities")
    CORE.LEGACY.save_figure(fig, output, "computed_Spinosin_signed_frequencies"); plt.close(fig)


def run(root, output, release_path, budget_validator=None):
    root, output, release_path = root.resolve(), output.resolve(), release_path.resolve()
    if output.exists(): raise FileExistsError("Use a fresh output directory; preserve prior analysis")
    ledger, authorization, input_hashes, cpu_sums = authorize(root, read(release_path), budget_validator)
    input_hashes[str(release_path)] = sha(release_path)
    if sha(PEAK_PATH) != PEAK_SHA256: raise ValueError("Historical experimental candidate bytes changed")
    experimental = CORE.LEGACY.experimental_rows(PEAK_PATH)
    if len(experimental) != 20: raise ValueError("All20 original candidate rows are required")
    input_hashes[str(PEAK_PATH)] = sha(PEAK_PATH)
    phases, steps, failed_steps, computed, water, accepted, replays, mode_sets, failures, proofs, controls = [], [], [], [], [], [], [], [], [], [], []
    for row in ledger["phases"]:
        if row["status"] not in ("EXITED", "PARTIAL_OR_FAILED", "FAILED_TO_START"):
            raise ValueError("Unknown or unresolved closed phase state")
        proof = audit_closed_proof(root, root / "cpu_budget.json", row, input_hashes)
        proofs.append({"phase": row["phase"], **proof})
        location = CORE.phase_location(root, row)
        result = CORE.completed_phase_result(row, location, input_hashes)
        if location is not None:
            for name in ("result.json", "input.json", "progress.json", "hessian_cache_manifest.json"):
                path = location / name
                if path.is_file(): input_hashes[str(path)] = sha(path)
        identity = result.get("identity", "UNKNOWN")
        settings = result.get("settings", {})
        if identity in CORE.COMPOSITIONS and not (CORE.method_matches(settings)
                and settings.get("density_fitting") is False
                and settings.get("density_fit_backend") == "SCREENED_DIRECT_RKS_V4"
                and settings.get("engineering_worker_version") == "SUPERVISED_PARALLEL_DIRECT_V8_SAVE_ONLY"
                and settings.get("resource_supervision_required") is True):
            raise ValueError("Unexpected molecular electronic-structure method")
        if result.get("script_sha256"):
            script_paths = [resolved(value) for value in row["command"] if str(value).endswith(".py")]
            expected_script = ETHANOL_VALIDATOR if identity == "validation_ethanol" else WORKER
            if (len(script_paths) != 1 or script_paths[0] != expected_script
                    or sha(expected_script) != result["script_sha256"]):
                raise ValueError("Executed worker program/source SHA differs")
            input_hashes[str(expected_script)] = sha(expected_script)
        elif identity != "UNKNOWN": raise ValueError("Identified worker lacks recorded source SHA")
        supervision = result.get("resource_supervision", {})
        if supervision and supervision.get("proof_sha256") != proof.get("proof_sha256"):
            raise ValueError("Worker resource proof differs from its actual parent ledger")
        if identity != "UNKNOWN" and not result["worker_result_missing"] and not supervision:
            raise ValueError("Final worker result lacks its recorded resource-supervision evidence")
        if identity == "validation_ethanol" and result.get("engineering_helper_source_sha256") != sha(WORKER):
            raise ValueError("Engineering validator helper source changed")
        gradient = CORE.audit_final_gradient(location, result, input_hashes) if identity in CORE.COMPOSITIONS else {}
        phases.append({"phase": row["phase"], "identity": identity, "budget_status": row["status"],
            "scientific_status": result.get("status", "PARTIAL"), "worker_result_missing": result["worker_result_missing"],
            "actual_cpu_seconds": row["actual_cpu_seconds"], "wall_seconds": row["elapsed_wall_seconds"],
            "threads": row["threads"], "exit_code": row["exit_code"], "geometry_optimized": result.get("geometry_optimized", False),
            "frequencies_status": result.get("frequencies_status", "NOT_RUN"), "IR_intensity_status": "NOT_COMPUTED",
            "fresh_gradient_RMS_hartree_per_bohr": gradient.get("atom_rms_hartree_per_bohr"),
            "fresh_gradient_max_hartree_per_bohr": gradient.get("atom_max_hartree_per_bohr"),
            "backend": result.get("settings", {}).get("density_fit_backend"),
            "peak_group_RSS_bytes_sampled": row.get("peak_group_rss_bytes_sampled"),
            "peak_cgroup_bytes_sampled": row.get("peak_cgroup_memory_bytes_sampled"),
            "memory_terminated": row.get("memory_terminated"), "monitor_failed": row.get("monitor_failed"),
            "stop_reason": result.get("stop_reason", row.get("failure")),
            "scope": "MOLECULAR_SOFTWARE_VALIDATION_ONLY" if identity in ("water", "validation_ethanol") else "SPINOSIN_OR_FAILED_UNIDENTIFIED_PHASE"})
        if identity in ("water", "validation_ethanol"):
            controls.append({"phase": row["phase"], "identity": identity, "validation_only": True,
                "scientific_status": result.get("status", "PARTIAL"), "geometry_optimized": result.get("geometry_optimized", False),
                "numerical_equivalence_passed": result.get("numerical_equivalence_passed"),
                "stationary_minimum_verified": result.get("stationary_minimum_verified", False),
                "expected_modes_if_completed_not_a_computed_count": 3 if identity == "water" else 21,
                "computed_main_table_modes": result.get("normal_mode_count") if identity == "water" else None,
                "mode_count_is_not_a_Spinosin_result": True, "IR_intensity_status": "NOT_COMPUTED"})
            if location is not None:
                for path in location.rglob("*"):
                    if path.is_file(): input_hashes[str(path)] = sha(path)
        if result["worker_result_missing"] or row["status"] != "EXITED" or result.get("status") != "DONE":
            failures.append({"phase": row["phase"], "identity": identity, "parent_exit_code": row.get("exit_code"),
                "budget_status": row["status"], "derived_scientific_status": result.get("status", "PARTIAL"),
                "worker_result_missing": result["worker_result_missing"], "no_worker_result_json_fabricated": True,
                "snapshot_event": result.get("snapshot_event"), "stop_reason": result.get("stop_reason", row.get("failure")),
                "observed_closed_artifacts": result.get("observed_closed_artifacts", {}),
                "memory_terminated": row.get("memory_terminated"), "monitor_failed": row.get("monitor_failed")})
        if identity in CORE.COMPOSITIONS:
            for path in sorted((location / "geometry_steps").glob("step_*.json")):
                if read(path).get("scf_converged") is True: steps.append(CORE.audit_step(path, identity, row["phase"], input_hashes))
                else: failed_steps.append(failed_geometry_attempt(path, row["phase"], input_hashes))
        # Ethanol's21 modes and other engineering results never enter molecular
        # water/Spinosin arrays, even if an auxiliary program used DONE wording.
        if identity in CORE.COMPOSITIONS and result.get("frequencies_status") == "DONE":
            frequency_parent_audit(root, ledger, row, result, input_hashes)
            rows, modes, geometry, replay = CORE.load_modes(location, result, input_hashes)
            eligible = CORE.valid_modes(result, identity) and proof["frequency_acceptance_allowed"]
            rows = [{**mode, "accepted_for_comparison": bool(eligible)} for mode in rows]
            computed.extend(rows); replays.append({"phase": row["phase"], **replay})
            if eligible:
                if identity == "water": water.extend(rows)
                else:
                    accepted.extend(rows); mode_sets.append((row["phase"], rows, modes, geometry))
    low = [mode for mode in accepted if .1 <= mode["frequency_thz"] <= 1.5]
    comparisons = CORE.LEGACY.comparison_rows(experimental, low)
    output.mkdir(parents=True)
    frequency_fields = ["phase", "identity", "mode", "frequency_cm1", "frequency_thz", "imaginary", "accepted_for_comparison", "IR_intensity_status"]
    csvwrite(output / "phase_status_and_resources.csv", phases, list(phases[0]) if phases else ["phase", "scientific_status"])
    csvwrite(output / "actual_paired_geometry_steps.csv", steps, list(steps[0]) if steps else ["phase", "cycle", "energy_hartree"])
    csvwrite(output / "unconverged_geometry_attempts.csv", failed_steps, list(failed_steps[0]) if failed_steps else ["phase", "cycle", "scf_converged"])
    csvwrite(output / "validation_controls_only.csv", controls, list(controls[0]) if controls else ["phase", "identity", "validation_only"])
    for name, rows in [("computed_frequencies_including_unvalidated.csv", computed), ("water_validation_frequencies.csv", water),
                       ("spinosin_validated_all_frequencies.csv", accepted), ("spinosin_validated_0p1_1p5_thz_modes.csv", low)]:
        csvwrite(output / name, rows, frequency_fields)
    csvwrite(output / "experimental_candidate_positions.csv", experimental, list(experimental[0]))
    csvwrite(output / "experiment_theory_pairs.csv", comparisons, list(comparisons[0]))
    write(output / "actual_failure_evidence.json", failures)
    write(output / "archived_supervision_proof_audit.json", proofs)
    write(output / "saved_matrix_frequency_replays.json", replays)
    CORE.plot_reports(output, steps, water, accepted, experimental, mode_sets)
    cost_plot(output, phases)
    computed_frequency_plot(output, computed)
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_only": True,
        "new_SCF_gradient_Hessian_evaluations": 0, "V3_actual_quantum_CPU_seconds": cpu_sums["V3"],
        "bound_prior_V2_actual_quantum_CPU_seconds": cpu_sums["prior_V2"],
        "combined_V2_V3_actual_CPU_seconds_not_new_classification_experiments": sum(cpu_sums.values()),
        "closed_supervisor_phase_count": len(phases), "failed_or_partial_phase_count": len(failures),
        "ended_phases_without_worker_result": sum(row["worker_result_missing"] for row in phases),
        "no_worker_result_json_fabricated": True, "complete_paired_gradient_points": len(steps),
        "Spinosin_paired_geometry_evaluations": sum(row["identity"] == "spinosin" for row in steps),
        "water_validation_geometry_evaluations": sum(row["identity"] == "water" for row in steps),
        "unconverged_geometry_attempt_records": len(failed_steps),
        "computed_frequency_rows_including_unvalidated": len(computed), "validated_water_frequency_rows": len(water),
        "computed_Spinosin_frequency_rows_including_unvalidated": sum(row["identity"] == "spinosin" for row in computed),
        "computed_water_validation_frequency_rows": sum(row["identity"] == "water" for row in computed),
        "validated_spinosin_frequency_rows": len(accepted), "validated_spinosin_low_band_modes": len(low),
        "Spinosin_frequency_status": "VALIDATED_ISOLATED_MOLECULE_MODES_ONLY" if accepted else "BLOCKED_NO_VALIDATED_SPINOSIN_MODES",
        "water_and_ethanol_are_validation_only": True, "repeated_backend_runs_are_not_new_molecules_or_samples": True,
        "water_validation_phase_count": sum(row["identity"] == "water" for row in phases),
        "ethanol_validation_phase_count": sum(row["identity"] == "validation_ethanol" for row in phases),
        "experimental_candidate_rows": len(experimental), "all_candidate_mode_pair_rows": len(comparisons),
        "NOT_RUN_is_not_a_computed_zero_spectrum": True, "chemical_attribution_confirmed": False,
        "IR_intensities_status": "NOT_COMPUTED", "classification_modified": False,
        "RSS_monitor_best_effort_not_hard_limit": True, "input_sha256": input_hashes}
    for path, expected in input_hashes.items():
        if sha(path) != expected: raise ValueError("Input evidence changed during final audit: " + path)
    write(output / "summary.json", summary)
    (output / "README.md").write_text("# V3监控批次实际计算结果审计\n\n本工具只读已结束阶段，复算保存梯度统计及Hessian小矩阵正常模式，不启动SCF、梯度或Hessian求值。root release绑定最终V3账本，原V2账本、授权、CPU、执行代码、日志及历史启动监督证明分别核验。历史证明不等于前PID仍存活，也不能把BEST_EFFORT RSS轮询声称为硬居民内存保证。\n\n"
        + f"实际配对点{len(steps)}，计算频率记录{len(computed)}，合格斯皮诺素模式{len(accepted)}（其中0.1—1.5THz共{len(low)}），部分／失败阶段{len(failures)}。\n\n"
        + "computed_frequencies_including_unvalidated.csv保留真实负频及未通过最低点检查的频率；不修改或抹除。仅完整219个正频、同物理方法父优化和fresh gradient／原始Hessian审计均通过，才进入斯皮诺素实验峰比较。未计算保持空表，不推断零吸收。\n\n"
        + "水与乙醇只作软件验证；乙醇21模式不会成为斯皮诺素219模式，多次后端复测不是新增分子或独立样品。IR NOT_COMPUTED，模式位置图不是吸收强度；低频振型箭头和XYZ动画仅来自合格真实模式，幅度任意可视缩放。\n\n"
        + "原生失败没有result.json时，只以实际父进程退出和input／progress快照派生PARTIAL，不向原目录补造结果。源SCF不收敛的梯度尝试单独记录，不当作合格点。\n\n"
        + "原生CPHF正常返回和请求容差不当作独立测得的残差，中间缓存完成不等于频率或正频极小点验证通过。带正负号的频率图保留已计算负频，不将其包装为稳定分子的吸收峰。\n\n"
        + "全部20实验候选与全部合格频段模式作完整对照，不按接近程度挑构象或阈值。孤立分子不包含药材固态晶格、分子间氢键、集体振动与复杂基质；频率接近不确认化学归属，仍需纯品／含量等实验证据。不修改96.0234%分类模型。\n")
    write(output / "files_sha256.json", {str(path.relative_to(output)): {"sha256": sha(path), "bytes": path.stat().st_size}
                                         for path in sorted(output.rglob("*")) if path.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "output", "release"): parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(); result = run(args.run_root, args.output, args.release)
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
