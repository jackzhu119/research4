"""Read one ended opt phase and ordinary probe evidence; never touch live CHKs."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
RUN = ROOT / "reports/pyscf-spinosin-continuation-20261031-v3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    target = REPORT / "closed_opt_cpu_audit.json"
    if target.exists():
        raise FileExistsError("Preserve this audit; use a new report for a later review")
    ledger = read(RUN / "cpu_budget.json")
    matches = [row for row in ledger["phases"] if row["phase"] == "spinosin-native4-optimization-001"]
    if len(matches) != 1:
        raise ValueError("Exactly one real optimization row required")
    row = matches[0]
    if row["status"] != "EXITED" or row["exit_code"] != 0 or not row.get("ended_at_utc"):
        raise ValueError("Only the actually closed successful opt phase may be audited here")
    location = RUN / "spinosin-native4-optimization-001"
    result_path = location / "result.json"
    result = read(result_path)
    proof_path, log_path = Path(row["supervisor_proof_path"]), Path(row["log"])
    proof = read(proof_path)
    if (sha(proof_path) != row["supervisor_proof_sha256"] or sha(log_path) != row["log_sha256"]
            or result["resource_supervision"]["observed_CPU_limits_seconds"] != [21480,21600]
            or proof["cpu_soft_limit_seconds"] != 21480 or proof["cpu_hard_limit_seconds"] != 21600):
        raise ValueError("Closed opt resource bindings differ from the observed limits")
    if result["status"] != "DONE" or not all(result.get(key) is True for key in
            ("optimizer_converged", "scf_converged", "final_gradient_thresholds_passed", "geometry_optimized")):
        raise ValueError("Do not substitute resource speculation for actual scientific status")
    actual, hard = float(row["actual_cpu_seconds"]), row["cpu_hard_limit_seconds"]
    closed_cpu = sum(float(value["actual_cpu_seconds"]) for value in ledger["phases"] if value["status"] != "RUNNING")
    if not math.isclose(closed_cpu, ledger["consumed_cpu_seconds"], abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError("Completed quantum CPU totals do not reconcile")
    active = [value for value in ledger["phases"] if value["status"] == "RUNNING"]
    phases = [{"phase": value["phase"], "requested_cpu_hard_seconds": value["cpu_hard_limit_seconds"]}
              for value in active]
    projection = closed_cpu + sum(value["requested_cpu_hard_seconds"] for value in phases)
    sources = {str(path.relative_to(ROOT)): sha(path) for path in (result_path, proof_path, log_path,
        location / "events.jsonl", ROOT/"scripts/run_pyscf_monitored_budget_20261031.py",
        ROOT/"scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py")}
    (REPORT / "closed_opt_row_snapshot.json").write_text(json.dumps(row, indent=2)+"\n")
    findings = {"status": "DONE_OPERATIONAL_AUDIT_ACTUAL_OPT_SCIENCE_PRESERVED",
        "created_at_UTC": datetime.now(timezone.utc).isoformat(),
        "source_ledger_path": str(RUN/"cpu_budget.json"),
        "whole_ledger_is_active_not_claimed_final": bool(active),
        "audited_phase": row["phase"], "actual_scientific_status": "DONE_FRESH_GRADIENT_PASSED",
        "configured_grant_CPU_seconds": row["granted_cpu_seconds"],
        "observed_startup_soft_hard_CPU_seconds": [21480,21600],
        "actual_parent_wait4_CPU_seconds": actual,
        "actual_worker_process_clock_CPU_seconds": result["process_cpu_seconds"],
        "parent_minus_worker_startup_shutdown_CPU_seconds": actual-result["process_cpu_seconds"],
        "actual_CPU_above_configured_hard_seconds": actual-hard,
        "percent_above_configured_hard": (actual-hard)/hard*100,
        "CPU_above_soft_seconds": actual-row["cpu_soft_limit_seconds"],
        "actual_cost_fully_charged_in_original_ledger": True,
        "completed_phase_CPU_sum_seconds": closed_cpu,
        "original_V3_quantum_CPU_cap_seconds": ledger["quantum_cpu_cap_seconds"],
        "active_phase_configurations_only_not_measured_actual_cost": phases,
        "completed_cost_plus_active_requested_hard_seconds": projection,
        "cap_minus_that_configuration_projection_seconds": ledger["quantum_cpu_cap_seconds"]-projection,
        "this_projection_is_not_a_guarantee_of_future_enforcement": True,
        "root_cause": "UNPROVEN_NO_FINAL_RAW_PROF_CLOCK_OR_DELIVERY_TRACE",
        "source_supported_candidate": "RLIMIT uses raw PROF clock; wait4/process_time may use adjusted scheduler-runtime-based clocks",
        "kernel_audit_is_separate": str(RUN/"cpu-overshoot-kernel-audit"),
        "Python_signal_latency_alone_does_not_explain_SIGKILL_non_delivery": True,
        "parent_did_not_sample_CPU_for_emergency_stop": True,
        "future_design_candidate_not_implemented": "independently monitor aggregate adjusted process CPU as well as retaining kernel limits",
        "recovery": "NOT_RUN_ORIGINAL_OPT_DONE_NO_RECOVERY_NEEDED",
        "new_quantum_evaluations": 0, "active_frequency_checkpoint_reads": 0,
        "active_frequency_signals": 0, "limits_changed": False, "old_ledger_rows_modified": False,
        "closed_input_hashes": sources,
        "ordinary_probe_evidence_sha256": sha(REPORT/"ordinary_probe_results.json")}
    target.write_text(json.dumps(findings, indent=2)+"\n")
    for relative, expected in sources.items():
        if sha(ROOT/relative) != expected:
            raise ValueError("Closed evidence changed during audit")
    print(json.dumps({key: findings[key] for key in
        ("actual_parent_wait4_CPU_seconds", "actual_CPU_above_configured_hard_seconds",
         "percent_above_configured_hard", "actual_cost_fully_charged_in_original_ledger", "root_cause")}))


if __name__ == "__main__":
    main()
