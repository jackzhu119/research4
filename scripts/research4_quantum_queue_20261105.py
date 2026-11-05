"""Prepared, bounded queue for already materialized Research4 quantum jobs.

Default mode only checks the configuration. This is not an operating-system
startup hook or a user-notification service. Root must invoke the saved resume
command after environment recovery. No scientific index or unknown SHA is made.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
SCHEMA = "research4-materialized-quantum-queue-v1"
GRANT_SCHEMA = "research4-materialized-quantum-queue-grant-v1"
MAX_JOBS, MAX_INVOCATIONS, MAX_SELF_CPU, MAX_WALL = 128, 3, 600, 86400
INTERVAL = 1800
WORKERS = {"pyscf_xc_chunk_worker_20261105_v7.py", "pyscf_staged_consumer_20261105_v7.py"}
FROZEN_A = {
    "scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py": "da1e2b0267c48e748a2bcb7d3f2ee3110d8e2288aa340d23dab7a12c994ddf8e",
    "scripts/pyscf_xc_chunk_worker_20261105_v7.py": "00e710eb0d637dbd073939bea5d4496993cb68f29efad1439322df3972415f02"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc)


def aware(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "An actual timezone-aware deadline is required")
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path(value):
    supplied = Path(value)
    require(not supplied.is_symlink(), "Symlink queue artifact rejected")
    result = supplied.resolve() if supplied.is_absolute() else (ROOT / supplied).resolve()
    require(result.is_relative_to(ROOT), "Queue files must stay inside this repository")
    return result


def read(file, expected=None, maximum=4 * 1024**2):
    file = path(file)
    require(file.is_file() and file.stat().st_size <= maximum, "Missing/oversized queue metadata")
    require(expected is None or sha(file) == expected, "Bound metadata SHA changed: " + str(file))
    return json.loads(file.read_text())


def save(file, value):
    file = path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".writing-" + str(os.getpid()))
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, file)
    descriptor = os.open(file.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checked(ref):
    file = path(ref["path"])
    require(file.is_file() and sha(file) == ref["sha256"], "Bound source/payload changed: " + str(file))
    require("bytes" not in ref or file.stat().st_size == ref["bytes"], "Bound artifact size changed")
    return file


def artifact(file):
    file = path(file)
    return {"path": str(file), "sha256": sha(file), "bytes": file.stat().st_size}


def own_rss():
    line = next(v for v in Path("/proc/self/status").read_text().splitlines() if v.startswith("VmRSS:"))
    return int(line.split()[1]) * 1024


@contextmanager
def self_limits(grant):
    """Per-process timers are NOT inherited by forked guardian processes.

    Never lower an inherited RLIMIT_CPU hard cap: that would also cap the
    guardian, which must be able to supervise a much longer quantum job.
    A second CPU/wall expiry has default terminate action if cleanup stalls.
    """
    cpu = grant["launcher_cpu_seconds_per_invocation"]
    wall = min(grant["wall_seconds_per_invocation"], aware(grant["expires_at_utc"]).timestamp() - time.time())
    require(type(cpu) is int and 10 <= cpu <= MAX_SELF_CPU and 0 < wall <= MAX_WALL, "Invalid own CPU/wall envelope")
    require(all(signal.getitimer(timer) == (0., 0.) for timer in (signal.ITIMER_PROF, signal.ITIMER_REAL)), "Do not replace an existing process timer")
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGPROF, signal.SIGALRM)}
    def expire(sig, _frame):
        timer = signal.ITIMER_PROF if sig == signal.SIGPROF else signal.ITIMER_REAL
        signal.signal(sig, signal.SIG_DFL)
        signal.setitimer(timer, 2.)
        raise TimeoutError("Launcher own CPU/wall limit reached; no success inferred")
    try:
        signal.signal(signal.SIGPROF, expire)
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_PROF, max(.01, cpu - time.process_time() - 2))
        signal.setitimer(signal.ITIMER_REAL, wall)
        yield
    finally:
        for timer in (signal.ITIMER_PROF, signal.ITIMER_REAL):
            signal.setitimer(timer, 0.)
        for sig, handler in originals.items():
            signal.signal(sig, handler)


def option(command, name):
    require(command.count(name) == 1 and command.index(name) + 1 < len(command), "Unique option required: " + name)
    return command[command.index(name) + 1]


def load_guard():
    require(sha(GUARD) == GUARD_SHA, "Frozen cumulative budget guard changed")
    spec = importlib.util.spec_from_file_location("queue_original_budget_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_config(config, grant, config_sha, execute=False):
    require(config.get("schema") == SCHEMA and grant.get("schema") == GRANT_SCHEMA, "Unsupported queue/grant")
    require(grant.get("config_sha256") == config_sha and grant.get("launcher_sha256") == sha(__file__), "Source/config-bound root grant required")
    require(grant.get("no_new_quantum_budget") is True and grant.get("no_deadline_extension") is True,
            "A queue cannot allocate resources or reset the research clock")
    require(config.get("monitor_interval_seconds") == INTERVAL, "Progress is fixed at1800seconds")
    require(type(grant.get("max_invocations")) is int and 1 <= grant["max_invocations"] <= MAX_INVOCATIONS, "Invocation count must be bounded")
    require(type(grant.get("launcher_cpu_seconds_per_invocation")) is int and 10 <= grant["launcher_cpu_seconds_per_invocation"] <= MAX_SELF_CPU, "Launcher own CPU limit invalid")
    require(type(grant.get("wall_seconds_per_invocation")) is int and 180 <= grant["wall_seconds_per_invocation"] <= MAX_WALL, "Launcher wall limit invalid")
    require(type(grant.get("launcher_rss_stop_bytes")) is int and 16 * 1024**2 <= grant["launcher_rss_stop_bytes"] <= 512 * 1024**2, "Launcher own best-effort RSS guard invalid")
    require(aware(grant["expires_at_utc"]) <= aware(config["original_quantum_deadline_at_utc"]), "Queue cannot extend Q deadline")
    require(config["original_quantum_cpu_cap_seconds"] == 216000, "Original cumulative CPU cap must be retained")
    jobs = config["jobs"]
    require(isinstance(jobs, list) and 0 < len(jobs) <= MAX_JOBS, "A finite materialized job list is required")
    require(len({j["id"] for j in jobs}) == len(jobs) and len({j["phase"] for j in jobs}) == len(jobs), "Duplicate job/phase")
    refs = {str(checked(r)): r["sha256"] for r in config["source_refs"]}
    for filename in refs:
        if Path(filename).suffix == ".py":
            for statement in ast.parse(Path(filename).read_text()).body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant) and statement.value.value == "NOT_FROZEN":
                    require(not any(isinstance(t, ast.Name) and t.id.endswith("_SHA") for t in statement.targets), "A downstream source pin is still NOT_FROZEN")
    require(len(refs) == len(config["source_refs"]), "Duplicate source ref")
    require(refs.get(str(GUARD)) == GUARD_SHA, "Original guard must be explicitly source-bound")
    require(all(refs.get(str(ROOT / name)) == expected for name, expected in FROZEN_A.items()), "Reviewed A/core sources required")
    require(config.get("pythonpath") == [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"], "Keep the reviewed Python site order")
    queue_directory = path(config["queue_directory"])
    require(queue_directory.is_relative_to(ROOT / "reports") and queue_directory != ROOT / "reports",
            "Queue metadata needs its own separate report directory")
    require(path(config["state"]) == queue_directory / "state.json"
            and path(config["monitor"]) == queue_directory / "monitor.jsonl", "Only the two reserved queue checkpoint paths may be written")
    protected = [path(config["ledger"]), *[path(r["path"]) for r in config["source_refs"]]]
    for job in jobs:
        protected += [path(job["result"]), path(job["result"]).parent / "input.json", path(job["log"]),
                      path(job["log"] + ".queue-console.log")]
        protected += [path(r["path"]) for r in job.get("inputs", [])]
    require(all(not file.is_relative_to(queue_directory) for file in protected), "Queue writes must be isolated from scientific inputs/results/logs and sources")
    require(len({str(path(j["result"])) for j in jobs}) == len(jobs)
            and len({str(path(j["log"])) for j in jobs}) == len(jobs), "Duplicate output/log target")
    for job in jobs:
        command = job["payload_argv"]
        worker = path(job["worker"]["path"])
        require(worker.name in WORKERS and refs.get(str(worker)) == job["worker"]["sha256"], "Only explicitly pinned V7 Q workers supported")
        require(isinstance(command, list) and len(command) >= 4 and all(isinstance(t, str) for t in command), "Arguments must be materialized strings; no shell")
        require(path(command[0]) == path(config["python_executable"]) and path(command[1]) == worker, "Exact Python/worker argv required")
        require(command.count("--execute") == 1 and option(command, "--stage") == job["stage"]
                and path(option(command, "--output")) == path(job["result"]).parent
                and int(option(command, "--threads")) == job["threads"] == 2, "Job argv/identity differs")
        require(type(job["cpu_seconds"]) is int and 125 <= job["cpu_seconds"] <= 122400
                and type(job["wall_seconds"]) is int and 61 <= job["wall_seconds"] <= 43200, "Invalid bounded job request")
        require(job["memory_gib"] == 24 and path(job["result"]).name == "result.json", "Original resource/result contract required")
        path(job["log"])
    for key in ("ledger", "state", "monitor"):
        path(config[key])
    if execute:
        require(grant.get("root_execution_approved") is True and grant.get("root_serial_scheduling_confirmed") is True,
                "Prepared queue is disabled until root explicitly approves serial execution")
        require(utc() < aware(grant["expires_at_utc"]), "Original grant expired")
    return config


def verify_stage(job, guard, ledger):
    """No PID absence or output-file existence is a success criterion."""
    rows = [r for r in ledger["phases"] if r.get("phase") == job["phase"]]
    if not rows:
        return None
    require(len(rows) == 1 and guard.healthy_closed_row(rows[0]), "FAILED/RUNNING/orphan stage requires forensic closure and a fresh root attempt")
    row = rows[0]
    require(row["command"] == job["payload_argv"] and path(row["log"]) == path(job["log"]), "Actual guardian payload argv/log changed")
    result_path = path(job["result"])
    result = read(result_path)
    require(result.get("stage") == job["stage"] and result.get("status") == job["success_status"], "Scientific stage is incomplete")
    proof_phase = result["resource_supervision"]["proof"]["phase"]
    require(proof_phase == job["phase"], "Result belongs to another actual phase")
    evidence = {"source_worker": result["source_worker"], "resource_supervision": result["resource_supervision"],
                "output_directory": str(result_path.parent), "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "stage": job["stage"]}
    guard.verify_closed_phase(evidence, expected_worker_path=path(job["worker"]["path"]),
        expected_worker_sha256=job["worker"]["sha256"], expected_result_schema=job["result_schema"],
        allowed_result_statuses=(job["success_status"],), expected_output=result_path.parent,
        expected_snapshot_sha256=option(job["payload_argv"], "--snapshot-manifest-sha256"))
    input_path = result_path.parent / "input.json"
    read(input_path)
    sealed = [artifact(result_path), artifact(input_path), artifact(path(result["resource_supervision"]["proof_path"])), artifact(path(row["log"]))]
    for key in ("plans_artifact", "partial_differential"):
        if key in result:
            sealed.append(artifact(checked(result[key])))
    for key, name in (("raw_hessian_sha256", "hessian_hartree_per_bohr2.npy"),
                      ("analysis_hessian_sha256", "hessian_used_for_analysis.npy"),
                      ("global_assembly_capture_sha256", "global_assembly_capture.json")):
        if key in result:
            sealed.append(artifact(checked({"path": str(result_path.parent / name), "sha256": result[key]})))
    if job["stage"] == "global-assemble":
        for name in ("normal_modes.npy", "reduced_mass_amu.npy", "frequencies.csv", "normal_mode_geometry_angstrom.npy",
                     "normal_mode_geometry_bohr.npy", "normal_mode_geometry.xyz"):
            sealed.append(artifact(result_path.parent / name))
    for ref in job.get("inputs", []):
        checked(ref)
    arrays = result.get("arrays", [])
    if "cache_manifest_sha256" in result:
        manifest_path = result_path.parent / "stage-cache/manifest.json"
        manifest = read(manifest_path, result["cache_manifest_sha256"])
        sealed.append(artifact(manifest_path))
        arrays = [*arrays, *manifest.get("arrays", [])]
    for item in arrays:
        require(Path(item["file"]).name == item["file"], "Unsafe payload filename")
        sealed.append(artifact(checked({"path": str(result_path.parent / "stage-cache" / item["file"]), "sha256": item["file_sha256"]})))
    unique = {r["path"]: r for r in sealed}
    return [unique[key] for key in sorted(unique)]


def verify_existing(job, guard, ledger, expected):
    require(expected, "Existing phase needs a persisted or root-config-bound seal before it can be skipped")
    for ref in expected:
        checked(ref)
    actual = verify_stage(job, guard, ledger)
    require(actual == expected, "Do not silently re-seal a changed historical result/input/proof/payload")
    return expected


def conflicting_processes():
    """Best effort; root must also keep Git/ML scheduling serial."""
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            stat_text = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if stat_text[0] in ("Z", "X", "x"):
                continue
            command = (directory / "cmdline").read_bytes().split(b"\0")
            names = [Path(t.decode(errors="replace")).name for t in command if t]
            git = bool(names and names[0] == "git")
            ml = any(n.endswith(".py") and n.startswith(("train_", "run_supplemental_", "run_cross_plant_", "run_calibration_", "run_benchmark_")) for n in names)
            if git or ml:
                found.append({"pid": int(directory.name), "category": "GIT" if git else "ML"})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


@contextmanager
def queue_lock():
    file = ROOT / ".runtime/research4-materialized-quantum-queue.lock"
    file.parent.mkdir(exist_ok=True)
    require(not file.is_symlink(), "Symlink launcher lock rejected")
    with file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args, config, grant):
    guard = load_guard()
    with queue_lock():
        ledger = read(config["ledger"])
        guard.validate_ledger(ledger)
        require(ledger["quantum_cpu_cap_seconds"] == config["original_quantum_cpu_cap_seconds"]
                and ledger["run_deadline_at_utc"] == config["original_quantum_deadline_at_utc"], "Original quantum envelope changed")
        state_path = path(config["state"])
        state = read(state_path) if state_path.exists() else {"schema": SCHEMA, "config_sha256": args.config_sha256,
            "grant_sha256": args.grant_sha256, "invocations": [], "jobs": {}}
        require(state["config_sha256"] == args.config_sha256 and state["grant_sha256"] == args.grant_sha256, "Existing state belongs to another immutable queue/grant")
        require(all(r["status"] == "MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE" for r in state["invocations"]), "Unclosed/blocked launcher invocation requires root forensic review and a fresh approved configuration; never infer success from a missing PID")
        require(len(state["invocations"]) < grant["max_invocations"] and args.attempt not in {r["id"] for r in state["invocations"]}, "Fresh bounded root invocation required")
        start = time.monotonic()
        end = min(aware(grant["expires_at_utc"]).timestamp(), args.invocation_deadline)
        monotonic_end = args.invocation_monotonic_deadline
        own_cpu = grant["launcher_cpu_seconds_per_invocation"]
        invocation = {"id": args.attempt, "status": "RUNNING", "pid": os.getpid(), "started_at_utc": utc().isoformat(),
            "per_invocation_self_CPU_cap_seconds": own_cpu, "wall_deadline_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
            "own_CPU_envelope_not_quantum_CPU_or_observed_actual_cost": True}
        invocation["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        invocation["start_ticks"] = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        state["resume_argv"] = [str(path(config["python_executable"])), str(Path(__file__).resolve()),
            "--config", str(path(args.config)), "--config-sha256", args.config_sha256, "--grant", str(path(args.grant)),
            "--grant-sha256", args.grant_sha256, "--attempt", args.attempt + "-resume", "--execute"]
        state["resume_requires_environment_ready_and_no_unresolved_invocation_or_Q_phase"] = True
        state["invocations"].append(invocation)
        child = None
        interrupted = [False]
        handlers = {s: signal.signal(s, lambda *_: interrupted.__setitem__(0, True)) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGXCPU)}
        next_progress = time.monotonic() + INTERVAL
        def persist(event):
            nonlocal next_progress
            current = read(config["ledger"])
            invocation["self_CPU_observed_seconds"] = time.process_time()
            state["next_stage"] = next((j["id"] for j in config["jobs"] if state["jobs"].get(j["id"], {}).get("status") != "HEALTHY_CLOSED"), None)
            state["last_event"] = event
            save(state_path, state)
            record = {"utc": utc().isoformat(), "event": event, "next_stage": state["next_stage"],
                "closed_quantum_CPU_seconds": current["consumed_cpu_seconds"], "remaining_quantum_CPU_before_open_phase": current["remaining_quantum_cpu_seconds"],
                "open_phases": [r["phase"] for r in current["phases"] if r.get("status") == "RUNNING"],
                "launcher_self_CPU_observed_seconds": invocation["self_CPU_observed_seconds"],
                "launcher_RSS_observed_bytes": own_rss(), "platform_push_service_created": False}
            monitor = path(config["monitor"])
            monitor.parent.mkdir(parents=True, exist_ok=True)
            with monitor.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
            print(json.dumps(record), flush=True)
            if event == "PROGRESS_1800_SECONDS":
                while next_progress <= time.monotonic():
                    next_progress += INTERVAL
        try:
            persist("QUEUE_STARTED")
            for job in config["jobs"]:
                ledger = read(config["ledger"])
                guard.validate_ledger(ledger)
                existing = [r for r in ledger["phases"] if r.get("phase") == job["phase"]]
                if existing:
                    expected = state["jobs"].get(job["id"], {}).get("seal") or job.get("existing_seal")
                    verify_existing(job, guard, ledger, expected)
                    state["jobs"][job["id"]] = {"status": "HEALTHY_CLOSED", "skipped_after_full_reverification": True, "seal": expected}
                    continue
                require(not interrupted[0] and min(end - time.time(), monotonic_end - time.monotonic()) >= 181
                        and time.process_time() < own_cpu - 5 and own_rss() < grant["launcher_rss_stop_bytes"], "Launcher resource/stop boundary reached")
                require(not conflicting_processes(), "Git/ML conflict: root serial scheduling required")
                require(sha(path(args.config)) == args.config_sha256 and sha(path(args.grant)) == args.grant_sha256
                        and sha(__file__) == grant["launcher_sha256"], "Queue/grant/launcher bytes changed before a new job")
                for ref in [*config["source_refs"], *job.get("inputs", [])]:
                    checked(ref)
                require(not path(job["log"]).exists() and not path(job["result"]).parent.exists(), "Unclosed output/log already exists; preserve and create a fresh root attempt")
                wall = min(job["wall_seconds"], int(min(end - time.time(), monotonic_end - time.monotonic())) - 120)
                command = [str(path(config["python_executable"])), str(GUARD), "run", "--ledger", str(path(config["ledger"])),
                    "--phase", job["phase"], "--cpu-seconds", str(job["cpu_seconds"]), "--wall-seconds", str(wall),
                    "--threads", "2", "--memory-gib", "24", "--log", str(path(job["log"])), "--", *job["payload_argv"]]
                state["jobs"][job["id"]] = {"status": "STARTING", "actual_guardian_argv": command}
                persist("GUARDED_STAGE_STARTING")
                console = path(job["log"] + ".queue-console.log")
                console.parent.mkdir(parents=True, exist_ok=True)
                with console.open("x") as handle:
                    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path(p)) for p in config["pythonpath"]))
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    state["jobs"][job["id"]].update(status="RUNNING", guardian_pid=child.pid)
                    save(state_path, state)
                    stopping = False
                    while child.poll() is None:
                        if time.monotonic() >= next_progress:
                            persist("PROGRESS_1800_SECONDS")
                        if interrupted[0] or time.time() >= end - 60 or time.monotonic() >= monotonic_end - 60 or time.process_time() >= own_cpu - 5 or own_rss() >= grant["launcher_rss_stop_bytes"] or conflicting_processes():
                            if not stopping:
                                child.terminate()
                                stopping = True
                        require(time.time() < end and time.monotonic() < monotonic_end, "Guardian not closed by outer wall deadline; retain unresolved state and do not retry")
                        time.sleep(1)
                    returncode = child.wait()
                child = None
                state["jobs"][job["id"]].update(status="GUARDIAN_EXITED_UNVERIFIED", actual_guardian_returncode=returncode)
                require(returncode == 0, "Actual guardian failed; keep failed cost and stop without retry")
                ledger = read(config["ledger"])
                guard.validate_ledger(ledger)
                sealed = verify_stage(job, guard, ledger)
                require(sealed, "Guardian exit alone is insufficient; scientific stage verification failed")
                state["jobs"][job["id"]] = {"status": "HEALTHY_CLOSED", "actual_guardian_returncode": returncode, "seal": sealed}
                persist("GUARDED_STAGE_HEALTHY_CLOSED")
            invocation["status"] = "MATERIALIZED_QUEUE_CLOSED_NOT_ALL_SCIENCE_COMPLETE"
        except BaseException as error:
            invocation.update(status="BLOCKED_REQUIRES_ROOT_REVIEW", reason=type(error).__name__ + ": " + str(error))
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    code = child.wait(timeout=max(.01, min(10., end - time.time(), monotonic_end - time.monotonic())))
                    invocation["exception_cleanup_actual_guardian_returncode"] = code
                except subprocess.TimeoutExpired:
                    invocation.update(status="BLOCKED_GUARDIAN_UNRESOLVED_NO_AUTOMATIC_RESUME", unresolved_guardian_pid=child.pid)
            raise
        finally:
            invocation.update(ended_observed_at_utc=utc().isoformat(), self_CPU_observed_seconds=time.process_time(), elapsed_wall_observed_seconds=time.monotonic() - start)
            persist("QUEUE_INVOCATION_CLOSED_OR_BLOCKED")
            for sig, old in handlers.items():
                signal.signal(sig, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--grant", required=True)
    parser.add_argument("--grant-sha256", required=True)
    parser.add_argument("--attempt", default="PREPARATION_ONLY")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, grant = read(args.config, args.config_sha256), read(args.grant, args.grant_sha256)
    require(not {path(args.config), path(args.grant)} & {path(config["state"]), path(config["monitor"])}, "Queue checkpoint cannot overwrite its configuration or root grant")
    if not args.execute:
        validate_config(config, grant, args.config_sha256, False)
        print(json.dumps({"status": "NOT_RUN_CONFIGURATION_CHECKED", "jobs": len(config["jobs"]), "background_start_hook_created": False}))
        return
    require(args.attempt != "PREPARATION_ONLY", "Root must provide a fresh invocation identity")
    require(grant.get("root_execution_approved") is True and grant.get("root_serial_scheduling_confirmed") is True,
            "Root execution grant required before activating even the own-process timers")
    args.invocation_deadline = min(aware(grant["expires_at_utc"]).timestamp(), time.time() + grant["wall_seconds_per_invocation"])
    args.invocation_monotonic_deadline = time.monotonic() + grant["wall_seconds_per_invocation"]
    with self_limits(grant):
        validate_config(config, grant, args.config_sha256, True)
        run(args, config, grant)


if __name__ == "__main__":
    main()
