"""Read-only closed-phase and public-source audit; no generated CPU workload/QM.

The live frequency directory/checkpoint is never read. Only the completed
optimization row is selected from a timestamped mutable-ledger snapshot.
"""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import platform
import re
import sys
import time


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
RUN=OUT.parent
PHASE="spinosin-native4-optimization-001"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def main():
    target=OUT/"audit.json"
    if target.exists():raise FileExistsError("Preserve this one-time audit")
    start=now();inputs={}
    def record(path):
        inputs[str(path.relative_to(ROOT))]={"sha256":sha(path),"bytes":path.stat().st_size}
        return path
    ledger_path=RUN/"cpu_budget.json";ledger_bytes=ledger_path.read_bytes();ledger=json.loads(ledger_bytes)
    selected=[row for row in ledger["phases"] if row["phase"]==PHASE]
    assert len(selected)==1
    row=selected[0]
    assert row["status"]=="EXITED" and row["exit_code"]==0 and row.get("ended_at_utc")
    (OUT/"selected_closed_optimization_row.json").write_text(json.dumps(row,indent=2)+"\n")
    result_path=RUN/PHASE/"result.json";result=json.loads(record(result_path).read_text())
    proof_path=Path(row["supervisor_proof_path"]);proof=json.loads(record(proof_path).read_text())
    assert sha(proof_path)==row["supervisor_proof_sha256"]==result["resource_supervision"]["proof_sha256"]
    assert result["status"]=="DONE" and result["optimizer_converged"] is True and result["geometry_optimized"] is True
    assert result["scf_converged"] is True and result["final_gradient_thresholds_passed"] is True
    assert proof["cpu_soft_limit_seconds"]==row["cpu_soft_limit_seconds"]==21480
    assert proof["cpu_hard_limit_seconds"]==row["cpu_hard_limit_seconds"]==row["granted_cpu_seconds"]==21600
    assert result["resource_supervision"]["observed_CPU_limits_seconds"]==[21480,21600]
    assert all(not row.get(key) for key in ("wall_terminated","hard_killed","memory_terminated","monitor_failed"))
    worker=ROOT/"scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
    supervisor=ROOT/"scripts/run_pyscf_monitored_budget_20261031.py"
    assert sha(record(worker))==result["script_sha256"]
    assert sha(record(supervisor))==proof["supervisor_script_sha256"]==ledger["runner_sha256"]
    log_path=Path(row["log"]);assert sha(record(log_path))==row["log_sha256"]
    worker_cpu=float(result["process_cpu_seconds"]);parent_cpu=float(row["actual_cpu_seconds"])
    assert parent_cpu>21600 and worker_cpu>21600 and parent_cpu>=worker_cpu
    sources=json.loads((OUT/"source_metadata.json").read_text())
    for source in sources:
        path=record(OUT/source["file"]);data=path.read_bytes()
        assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\x00"+data).hexdigest()==source["git_blob_sha"]
        source["sha256"]=sha(path);source["bytes"]=len(data)
    config_path=Path("/proc/config.gz")
    config_raw=config_path.read_bytes();config=gzip.decompress(config_raw)
    config_lines=config.decode().splitlines()
    keys=("CONFIG_VIRT_CPU_ACCOUNTING_GEN","CONFIG_VIRT_CPU_ACCOUNTING_NATIVE","CONFIG_TICK_CPU_ACCOUNTING",
          "CONFIG_POSIX_CPU_TIMERS_TASK_WORK","CONFIG_NO_HZ_FULL","CONFIG_HZ","CONFIG_HZ_250")
    config_selected={key:[{"line":i+1,"text":line} for i,line in enumerate(config_lines)
                           if line.startswith(key+"=") or line=="# "+key+" is not set"] for key in keys}
    info=time.get_clock_info("process_time")
    clock_metadata={key:getattr(info,key) for key in ("implementation","monotonic","adjustable","resolution")}
    # Read each available clock once. This is not a benchmark or the old worker.
    clock_readings={"this_auditor_process_PROF_clockid_minus8_ns":time.clock_gettime_ns(-8),
                    "this_auditor_process_SCHED_clockid_minus6_ns":time.clock_gettime_ns(-6)}
    package_hits=[]
    for directory in (ROOT/".runtime/quantum-upgrade-site/pyscf",ROOT/".runtime/quantum-pilot-site/geometric"):
        for path in directory.rglob("*"):
            if path.suffix not in (".py",".c",".h"):continue
            for i,line in enumerate(path.read_text(errors="replace").splitlines(),1):
                if re.search(r"\b(setrlimit|prlimit|RLIMIT_CPU|SIGXCPU)\b",line):
                    package_hits.append({"file":str(path.relative_to(ROOT)),"line":i,"text":line})
    audit={"status":"DONE_SOURCE_AND_CLOSED_PHASE_AUDIT_ROOT_CAUSE_UNPROVEN",
        "started_at_utc":start,"completed_at_utc":now(),"quantum_evaluations":0,"generated_CPU_workload_tests":0,
        "active_frequency_file_or_checkpoint_reads":0,"historical_rows_or_limits_modified":False,
        "fallback_recovery":"NOT_RUN_ORIGINAL_OPTIMIZATION_ACTUALLY_DONE",
        "phase":PHASE,"selected_closed_phase":row,"ledger_snapshot_SHA256_at_read":hashlib.sha256(ledger_bytes).hexdigest(),
        "ledger_can_contain_later_active_phase_not_a_claim_of_global_closure":True,
        "confirmed":{"requested_phase_CPU_seconds":21600,"observed_startup_RLIMIT_CPU_seconds":[21480,21600],
            "parent_wait4_actual_CPU_seconds":parent_cpu,"worker_reported_process_CPU_seconds":worker_cpu,
            "wait4_CPU_above_requested_hard_seconds":parent_cpu-21600,
            "wait4_CPU_above_requested_hard_percent":100*(parent_cpu-21600)/21600,
            "worker_CPU_above_requested_hard_seconds":worker_cpu-21600,
            "wait4_minus_worker_timer_seconds":parent_cpu-worker_cpu,"actual_physics_optimization_DONE":True,
            "strict_wait4_bound_of21600_s_was_not_achieved":True,
            "global_48_CPU_hour_cap_was_not_exceeded_by_closed_phase_totals_at_snapshot":ledger["consumed_cpu_seconds"]<=ledger["quantum_cpu_cap_seconds"]},
        "source_supported_clock_mechanism":{"Linux_RLIMIT_CPU_clock":"CPUCLOCK_PROF (raw accounting user+system)",
            "Linux_CLOCK_PROCESS_CPUTIME_ID_clock":"CPUCLOCK_SCHED (scheduler sum_exec_runtime)",
            "Linux_rusage_path":"adjusted thread-group CPU accounting; generic path scales user+system to scheduler runtime",
            "clock_encoding":"make_process_cpuclock(pid,clock)=((~pid)<<3)|clock; PROF0 => -8 and SCHED2 => -6 for pid0",
            "clock_accounting_and_timer_delivery_can_differ":"SOURCE_SUPPORTED_POSSIBILITY_NOT_OLD_PHASE_MEASUREMENT",
            "public_source_version":"gregkh/linux v6.18.57 and CPython v3.12.14",
            "matching_uname_tag_does_not_prove_vendor_build_is_unmodified":True},
        "unproven":{"exact_cause_of_81_second_overshoot":"UNPROVEN",
            "old_worker_final_raw_PROF_counter":"NOT_RECORDED",
            "all_midrun_limit_changes_excluded_by_syscall_trace":"NOT_RECORDED_NO_STRACE",
            "configured_NO_HZ_FULL_implies_every_CPU_was_running_full_dynticks":False,
            "Python_signal_delay_alone_explains_hard_SIGKILL_not_firing":False,
            "new_frozen_supervisor_or_active_frequency_parameters_modified":False},
        "runtime":{"uname":platform.uname()._asdict(),"python":sys.version,
            "current_process_time_clock":clock_metadata,"non_benchmark_clock_reads":clock_readings,
            "kernel_config_gzip_sha256":hashlib.sha256(config_raw).hexdigest(),
            "kernel_config_decompressed_sha256":hashlib.sha256(config).hexdigest(),
            "kernel_config_selected_lines":config_selected,"absent_config_line_is_not_an_explicit_notset_assertion":True},
        "pyscf_geometric_source_scan_hits":package_hits,
        "source_scan_does_not_exclude_unobserved_native_or_runtime_behavior":True,
        "sources":sources,"immutable_input_evidence":inputs,"auditor_source_sha256":sha(__file__),
        "related_ordinary_process_probe":"../cpu-limit-overshoot-audit/ (separate audit, not a quantum experiment)",
        "followup_without_changing_current_run":"Future supervisors can separately poll measured process/group CPU with a predeclared cleanup margin; no such change was applied to the active phase."}
    assert not package_hits
    target.write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":audit["status"],"confirmed":audit["confirmed"],
        "runtime_clock":clock_metadata,"config":config_selected,"sources":len(sources)},indent=2))


if __name__=="__main__":main()
