#!/usr/bin/env python3
"""One finite, local-only evidence job after authenticated QC closure.

Default is read-only preflight. No fit, inference, quantum work, Git, signals,
budget initialization or automatic retries are performed by this waiter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PREFIX = ROOT / "reports/species-correction-20261108-v1"
DRIVER = ROOT / "scripts/run_research4_autonomous72h_20261107.py"
DRIVER_SHA = "44dde1f73beafc9591923374dbe60782b0daae625e2d15d59c62de4c2bb9dd63"
GUARD_SHA = "643a431b7229eca94ac50645d467e6c3472f976b8ac48cc2f36be68241319e68"
QC_SHA = "23c76f8df7edc01ab031398a3fa12fb7466772912a3f53530637c4a3d6c09ee5"
DEADLINE = "2026-11-10T02:50:37.289380+00:00"
SCHEMA = "research4-species-afterq-watch-v1"
PHASE = "r72-species-correction-evidence-001"


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def driver_module():
    require(sha(DRIVER) == DRIVER_SHA, "Frozen driver changed")
    spec = importlib.util.spec_from_file_location("species_afterq_frozen_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(PREFIX), "Only new evidence reports may be written")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temp.replace(path)


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def parent_status(expected, identity):
    require(expected["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "BOOT_CHANGED: root must reconcile the old queue before a new waiter")
    try:
        actual = identity(expected["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return "PARENT_ABSENT_REQUIRES_FULL_CLOSURE_AUDIT"
    require(all(actual[k] == expected[k] for k in ("pid", "start_ticks", "uid", "boot_id")),
            "PID_REUSED_OR_IDENTITY_CHANGED: no automatic takeover")
    return "PARENT_TERMINAL_REQUIRES_FULL_CLOSURE_AUDIT" if actual["state"] in ("Z", "X", "x") else "WAITING_FOR_QC_PARENT"


def validate(config, d, *, waiting=True):
    require(config.get("schema") == SCHEMA and config.get("watcher_source_sha256") == sha(SELF), "Exact waiter source required")
    require(d.checked(config["driver_ref"]) == DRIVER and config["driver_ref"]["sha256"] == DRIVER_SHA,
            "Only the reviewed frozen QC driver may establish closure")
    require(config["guard_ref"]["sha256"] == GUARD_SHA, "Exact registered guardian required")
    guard = d.guard_module(config["guard_ref"])
    require(d.path(config["ledger_path"]) == d.LEDGER and config["deadline_utc"] == DEADLINE, "Existing ledger and deadline required")
    require(config["qc_config_ref"]["sha256"] == QC_SHA, "Exact current QC queue required")
    qc = d.read(d.checked(config["qc_config_ref"])); receipt = d.read(d.checked(config["qc_receipt_ref"]))
    require(receipt["config_sha256"] == QC_SHA and receipt["identity"]["uid"] == os.getuid(), "Real current QC launch receipt required")
    for ref in qc.get("source_refs", []) + qc.get("immutable_inputs", []) + config.get("immutable_inputs", []):
        d.checked(ref)
    worker = d.checked(config["worker_ref"])
    require(worker == ROOT / "scripts/research4_species_correction_analysis_20261108.py", "Only the declared evidence worker is allowed")
    argv = config["argv"]
    require(isinstance(argv, list) and all(isinstance(t, str) and "\0" not in t for t in argv), "Shell-free argv required")
    require(argv.count("{output}") == 1 and argv.count("--output") == 1 and argv[argv.index("--output") + 1] == "{output}", "One fresh output placeholder required")
    require(len(argv) >= 3 and Path(argv[0]).name.startswith("python") and argv[1] in ("-B", "-u")
            and d.path(argv[2]) == worker and not any(t in argv for t in ("-c", "-m")), "Direct pinned worker invocation required")
    require(all("{" not in t or t == "{output}" for t in argv), "No unbound template arguments")
    for key in ("workroot", "output", "log"):
        require(d.path(config[key]).is_relative_to(PREFIX), "Fresh species evidence paths required")
    require(d.path(config["workroot"]) != d.path(config["output"]), "Worker output must be distinct from waiter metadata")
    result_name = config.get("result_file", "result.json")
    require(isinstance(result_name, str) and Path(result_name).name == result_name and result_name.endswith(".json"), "Simple JSON result basename required")
    require(isinstance(config.get("success_statuses"), list) and config["success_statuses"] and all(isinstance(s, str) and s for s in config["success_statuses"]), "Explicit scientific success statuses required")
    require(d.path(config["qc_state_path"]) == d.path(qc["workroot"]) / "state.json", "QC state must belong to the bound launch")
    require(config["phase"] == PHASE and type(config["cpu_seconds"]) is int and 125 <= config["cpu_seconds"] <= 2000
            and type(config["wall_seconds"]) is int and 61 <= config["wall_seconds"] <= 1200
            and config["threads"] == 1, "One finite 2000CPU/1200wall/one-thread evidence phase required")
    ledger = d.read(d.LEDGER); guard.validate_ledger(ledger, allow_running=waiting)
    require(ledger["run_deadline_at_utc"] == DEADLINE, "Budget deadline changed")
    return guard, qc, receipt, ledger


def authenticate_qc_closure(config, d, guard, qc):
    ledger = d.read(d.LEDGER); guard.validate_ledger(ledger)
    state = d.read(config["qc_state_path"])
    require(state.get("status") == "FINITE_QUEUE_COMPLETE_WITH_ALL_RESULTS_PRESERVED", "QC parent absence is not completed science evidence")
    d.resume_state(state, QC_SHA, qc, guard)
    require(set(state.get("jobs", {})) == {j["id"] for j in qc["jobs"]}, "All QC queue entries must have authenticated terminal states")
    d.process_scan(qc)
    require(not any(row["phase"] == PHASE for row in ledger["phases"]), "Evidence phase already exists; root review required, never rerun")
    require(ledger["remaining_quantum_cpu_seconds"] >= config["cpu_seconds"] + 4, "Insufficient remaining registered CPU")
    require((datetime.fromisoformat(DEADLINE) - utc()).total_seconds() > config["wall_seconds"] + 2, "Insufficient authorized wall window")
    return {"qc_state_sha256": sha(d.path(config["qc_state_path"])), "all_terminal_jobs_authenticated": True,
            "independent_process_scan_clear": True, "remaining_CPU_before_launch": ledger["remaining_quantum_cpu_seconds"]}


def execute(config, config_path, config_hash, d):
    work = d.path(config["workroot"]); work.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / ".runtime/research4-species-afterq-watcher.lock"
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard, qc, receipt, _ = validate(config, d)
        state = {"schema": SCHEMA, "config_sha256": config_hash, "source_sha256": sha(SELF),
                 "identity": d.identity(os.getpid()), "started_at_utc": utc().isoformat(), "status": "WAITING_FOR_QC_PARENT",
                 "no_Git_no_fit_no_inference_no_Q_in_waiter": True, "host_shutdown_survival_guarantee": False}
        require(not (work / "state.json").exists(), "Preserve previous waiter state; root reconciliation is required")
        started_cpu = time.process_time()
        try:
            while True:
                require(sha(config_path) == config_hash and sha(SELF) == config["watcher_source_sha256"], "Waiter/config source changed")
                require(utc() < datetime.fromisoformat(DEADLINE), "Authorized deadline reached")
                require(time.process_time() - started_cpu < 30, "Finite waiter CPU ceiling reached")
                status = parent_status(receipt["identity"], d.identity)
                if status != "WAITING_FOR_QC_PARENT":
                    break
                state.update(status=status, last_observed_at_utc=utc().isoformat())
                save(work / "state.json", state); time.sleep(10)
            guard, qc, _, _ = validate(config, d, waiting=False)
            state["qc_closure"] = authenticate_qc_closure(config, d, guard, qc)
            require(not d.path(config["output"]).exists() and not d.path(config["log"]).exists(), "Never overwrite evidence output/log")
            actual = [str(d.path(config["output"])) if t == "{output}" else t for t in config["argv"]]
            command = [sys.executable, "-B", str(d.GUARD), "run", "--ledger", str(d.LEDGER), "--phase", PHASE,
                       "--log", str(d.path(config["log"])), "--cpu-seconds", str(config["cpu_seconds"]),
                       "--wall-seconds", str(config["wall_seconds"]), "--threads", "1", "--memory-gib", "24", "--", *actual]
            state.update(status="STARTING_SINGLE_GUARDED_EVIDENCE_JOB", actual_argv=actual); save(work / "state.json", state)
            with (work / "guardian-console.log").open("x") as console:
                child = subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status="RUNNING_GUARDED_EVIDENCE_JOB", guard_identity=d.identity(child.pid)); save(work / "state.json", state)
                while child.poll() is None:
                    require(utc() < datetime.fromisoformat(DEADLINE), "Deadline reached with live guardian; root must reconcile, no signals sent")
                    require(time.process_time() - started_cpu < 30, "Waiter CPU ceiling with live guardian; no automatic retry")
                    time.sleep(2)
                code = child.returncode
            state["guard_exit_code"] = code
            state["closure"] = guard.verify_closed_row(d.LEDGER, PHASE, expected_worker_path=d.path(config["worker_ref"]["path"]),
                expected_worker_sha256=config["worker_ref"]["sha256"], expected_output=d.path(config["output"]), allow_failure=True)
            result = d.path(config["output"]) / config.get("result_file", "result.json")
            if result.exists():
                state["result"] = {"path": str(result), "sha256": sha(result)}
            passed = code == 0 and state["closure"]["scientific_success"] and result.exists()
            if passed:
                passed = d.read(result).get("status") in config["success_statuses"]
            state["status"] = "DONE_LOCAL_EVIDENCE_NOT_AUTO_PUBLISHED" if passed else "PARTIAL_OR_FAILED_CLOSED_PRESERVED"
        except BaseException as exc:
            state.update(status="BLOCKED_REQUIRES_ROOT_RECONCILIATION", error_type=type(exc).__name__, error=str(exc))
        finally:
            state.update(updated_at_utc=utc().isoformat(), watcher_CPU_seconds=time.process_time() - started_cpu)
            save(work / "state.json", state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(); d = driver_module()
    config_path = d.checked({"path": str(args.config), "sha256": args.config_sha256}); config = d.read(config_path)
    _, _, receipt, _ = validate(config, d)
    status = parent_status(receipt["identity"], d.identity)
    if not args.execute:
        require(not args.detach, "Detachment requires --execute")
        print(json.dumps({"status": "READ_ONLY_PREFLIGHT_NO_EXECUTION", "QC_parent": status})); return
    work = d.path(config["workroot"]); work.mkdir(parents=True, exist_ok=True)
    if args.detach:
        claim = work / "launch-claim.json"
        exclusive(claim, {"config_sha256": args.config_sha256, "source_sha256": sha(SELF), "at_utc": utc().isoformat(),
                          "launcher_identity": d.identity(os.getpid()), "claim_is_not_actual_worker_completion": True})
        argv = [sys.executable, "-B", str(SELF), "--config", str(config_path), "--config-sha256", args.config_sha256, "--execute"]
        with (work / "waiter-console.log").open("x") as console:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"status": "ACTUAL_DETACHED_WAITER_OBSERVED_NOT_ANALYSIS_COMPLETED", "identity": d.identity(child.pid),
                   "config_sha256": args.config_sha256, "source_sha256": sha(SELF), "at_utc": utc().isoformat(),
                   "no_host_shutdown_survival_claim": True, "no_90000_CPU_budget_reinitialization": True}
        exclusive(work / "detached-launch.json", receipt); print(json.dumps(receipt)); return
    print(json.dumps({"status": execute(config, config_path, args.config_sha256, d)["status"]}))


if __name__ == "__main__":
    main()
