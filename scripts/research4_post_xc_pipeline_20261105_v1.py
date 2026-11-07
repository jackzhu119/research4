"""Finite, explicitly root-started post-XC pipeline; default is NOT_RUN.

No service/startup hook is installed. --detach uses a new process session, which
can survive loss of a terminal, but cannot survive termination of the machine.
All quantum phases use the unchanged cumulative V6b resource guard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts/research4_quantum_queue_20261105_v2.py"
QUEUE_SHA = "3af76bd231b4a6caa1bf70fe788c2ad2be1dd9cec8b84d181a17e53efeb79dad"
XC = ROOT / "scripts/pyscf_xc_chunk_worker_20261105_v7.py"
XC_SHA = "00e710eb0d637dbd073939bea5d4496993cb68f29efad1439322df3972415f02"
CONSUMER = ROOT / "scripts/pyscf_staged_consumer_20261105_v7.py"
CONSUMER_SHA = "6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe"
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v7_v2.py"
ANALYZER_SHA = "76e2eddebf681f9adf2ab7c5f9fa0bdc126b0c93f081cc93a606009f31c94705"
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
SCHEMA = "research4-post-XC-finite-pipeline-config-v1"
GRANT_SCHEMA = "research4-post-XC-finite-pipeline-grant-v1"
STATE_SCHEMA = "research4-post-XC-finite-pipeline-state-v1"
RESULT_SCHEMA = "research4-staged-full-rks-hessian-result-v7"
SNAPSHOT = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
RELEASE = "82d2138c8e7e514b20dd574b411a1093b8506f29f2698be0474c1c5aebe9b029"
DEADLINE = "2026-11-05T23:20:50.391742+00:00"
STAGES = ("rks-partial", "h1ao", *[f"response-{i:02d}" for i in range(10)], "global-assemble")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(file, expected, name):
    require(sha(file) == expected, "Frozen source changed: " + str(file))
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_modules():
    q = load(QUEUE, QUEUE_SHA, "post_XC_frozen_queue_metadata")
    return q, q.load_guard(), load(XC, XC_SHA, "post_XC_frozen_components"), load(CONSUMER, CONSUMER_SHA, "post_XC_frozen_consumer")


def ref(q, file):
    return {"path": str(q.path(file)), "sha256": q.sha(q.path(file))}


def immutable_json(q, file, value):
    file = q.path(file)
    if file.exists():
        require(q.read(file) == value, "Immutable materialized dependency changed")
    else:
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return ref(q, file)


def validate(q, config, grant, config_sha, execute=False):
    require(config.get("schema") == SCHEMA and grant.get("schema") == GRANT_SCHEMA, "Fresh pipeline schemas required")
    require(grant.get("config_sha256") == config_sha and grant.get("driver_sha256") == sha(__file__), "Exact source/config-bound root grant required")
    require(grant.get("no_new_quantum_budget") is True and grant.get("no_deadline_extension") is True, "No new pool or deadline")
    require(config["quantum_cpu_cap_seconds"] == 216000 and config["quantum_deadline_at_utc"] == DEADLINE
            and grant["expires_at_utc"] == DEADLINE, "Original cumulative budget/deadline required")
    require(type(grant["max_invocations"]) is int and 1 <= grant["max_invocations"] <= 2, "Finite root invocations required")
    require(10 <= grant["launcher_cpu_seconds_per_invocation"] <= 600
            and 181 <= grant["wall_seconds_per_invocation"] <= 86400
            and 16 * 1024**2 <= grant["launcher_rss_stop_bytes"] <= 256 * 1024**2, "Bounded launcher resources required")
    require(type(grant["conflict_wait_seconds"]) is int and 0 <= grant["conflict_wait_seconds"] <= 1800, "Conflict waits must be bounded")
    exceptions = grant.get("opaque_process_identity_exceptions", [])
    require(isinstance(exceptions, list) and len(exceptions) <= 2, "Only explicit known daemon visibility exceptions")
    for item in exceptions:
        require(item.get("uid") == 0 and item.get("comm") in ("dockerd", "containerd")
                and type(item.get("pid")) is int and item["pid"] > 0
                and type(item.get("start_ticks")) is int and item["start_ticks"] > 0
                and isinstance(item.get("boot_id"), str) and len(item["boot_id"]) >= 8, "Exact approved daemon identity required")
    work = q.path(config["work_directory"])
    target_root = ROOT / "reports/pyscf-spinosin-grid-checkpoint-20261105-v7"
    require(work.is_relative_to(target_root) and work != target_root, "Separate directory under the actual frozen target namespace required")
    require(q.path(config["state"]) == work / "state.json" and q.path(config["monitor"]) == work / "monitor.jsonl", "Narrow checkpoint paths required")
    require(config["snapshot_manifest_sha256"] == SNAPSHOT and config["gradient_result"]["sha256"] == GRADIENT
            and config["control_release"]["sha256"] == RELEASE and config["threads"] == 2, "Same exact target source/G/control/method required")
    require(config["pythonpath"] == [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"], "Frozen site order required")
    require(isinstance(config["phase_prefix"], str) and 1 <= len(config["phase_prefix"]) <= 100
            and all(c.isalnum() or c in "-_" for c in config["phase_prefix"]), "Safe bounded fresh phase prefix required")
    require(set(config.get("existing_results", {})).issubset(set(STAGES) - {"global-assemble"}), "Only declared qualified predecessor imports supported")
    expected = {str(QUEUE): QUEUE_SHA, str(XC): XC_SHA, str(CONSUMER): CONSUMER_SHA,
        str(ANALYZER): ANALYZER_SHA, str(GUARD): GUARD_SHA}
    actual = {str(q.checked(item)): item["sha256"] for item in config["source_refs"]}
    require(all(actual.get(p) == s for p, s in expected.items()), "Required frozen source refs missing")
    for stage in STAGES:
        limits = config["phase_requests"][stage]
        require(type(limits["cpu_seconds"]) is int and 125 <= limits["cpu_seconds"] <= 122400
                and type(limits["wall_seconds"]) is int and 181 <= limits["wall_seconds"] <= 43200, "Finite per-phase requests required")
    require(len(config["phase_requests"]) == len(STAGES), "Only the13 predetermined quantum stages are supported")
    q.checked(config["master_XC_config"])
    for key in ("plan_result", "RHF_result", "gradient_result", "control_release"):
        q.checked(config[key])
    protected = [q.path(config["snapshot"]), q.path(config["ledger"]),
        *[q.path(config[k]["path"]) for k in ("plan_result", "RHF_result", "gradient_result", "control_release")]]
    require(all(not work.is_relative_to(p) and not p.is_relative_to(work) for p in protected), "Protect original source, results and ledger")
    if execute:
        require(grant.get("root_execution_approved") is True and grant.get("root_serial_scheduling_confirmed") is True,
                "Disabled until explicit root serial execution grant")
        require(q.utc() < q.aware(DEADLINE), "Original deadline expired")


def component_index(q, guard, xc, config, state):
    master = q.read(config["master_XC_config"]["path"], config["master_XC_config"]["sha256"])
    require(len(master["jobs"]) == 31, "Exactly31 real XC producers required")
    plan = q.read(config["plan_result"]["path"], config["plan_result"]["sha256"])
    physical = plan["physical_binding"]
    indices = {"xc-diag-chunk": [], "xc-deriv2-chunk": []}
    refs = []
    for job in master["jobs"]:
        require(job["stage"] in indices, "Only original XC chunk jobs")
        result = q.read(job["result"])
        closure = xc.qualify_component(ref(q, job["result"]), job["stage"], physical, 75, 762, guard)
        require(result["status"] == "XC_CHUNK_COMPUTED" and closure, "An XC producer is incomplete")
        start = int(q.option(job["payload_argv"], "--start-block"))
        require(q.option(job["payload_argv"], "--block-count") == "1", "Original one-block chunks required")
        indices[job["stage"]].append(start)
        refs.append(ref(q, job["result"]))
    require(sorted(indices["xc-diag-chunk"]) == list(range(20))
            and sorted(indices["xc-deriv2-chunk"]) == list(range(11)), "Exact full-grid coverage required")
    clean = lambda value: {"path": str(q.path(value["path"])), "sha256": value["sha256"]}
    value = {"schema": "research4-v7-XC-component-index-v1", "RHF_partial": clean(config["RHF_result"]),
        "XC_plan": clean(config["plan_result"]), "XC_chunks": refs}
    if config.get("component_index"):
        record = clean(config["component_index"])
        supplied = q.read(q.checked(record))
        require(set(supplied) == set(value) and supplied["schema"] == value["schema"]
                and supplied["RHF_partial"] == value["RHF_partial"] and supplied["XC_plan"] == value["XC_plan"]
                and sorted(supplied["XC_chunks"], key=lambda r: r["path"]) == sorted(refs, key=lambda r: r["path"]),
                "Root canonical index must contain exactly the same real31 producer refs")
    else:
        record = immutable_json(q, q.path(config["work_directory"]) / "metadata/XC_components.json", value)
    verified = xc._assembly_sources(record, physical, 75, 762, guard)
    require(verified, "Full healthy XC/RHF grid-source chain required")
    state["complete_XC_component_index"] = record
    return record, physical


def common_argv(config, worker, stage, output):
    return [str(ROOT / config["python_executable"]), str(worker), "--execute", "--stage", stage,
        "--threads", "2", "--snapshot", config["snapshot"], "--snapshot-manifest-sha256", SNAPSHOT,
        "--gradient-check-result", config["gradient_result"]["path"], "--gradient-check-result-sha256", GRADIENT,
        "--control-release", config["control_release"]["path"], "--control-release-sha256", RELEASE,
        "--output", str(output)]


def stage_job(q, config, stage, deps):
    output = q.path(config["work_directory"]) / "science" / stage
    actual_stage = "response-block" if stage.startswith("response-") else stage
    worker, worker_sha = (XC, XC_SHA) if stage == "rks-partial" else (CONSUMER, CONSUMER_SHA)
    command = common_argv(config, worker, actual_stage, output)
    if stage == "rks-partial":
        command += ["--component-index", deps["components"]["path"], "--component-index-sha256", deps["components"]["sha256"], "--xc-max-memory-mb", "2000"]
    else:
        command += ["--partial-result", deps["partial"]["path"], "--partial-result-sha256", deps["partial"]["sha256"]]
        if stage != "h1ao":
            command += ["--h1-result", deps["h1"]["path"], "--h1-result-sha256", deps["h1"]["sha256"]]
        if stage.startswith("response-"):
            command += ["--block-index", str(int(stage.split("-")[1]))]
        if stage == "global-assemble":
            command += ["--response-index", deps["responses"]["path"], "--response-index-sha256", deps["responses"]["sha256"]]
    return {"id": stage, "phase": config["phase_prefix"] + "-" + stage, "stage": actual_stage,
        "worker": {"path": str(worker), "sha256": worker_sha}, "payload_argv": command,
        "result_schema": RESULT_SCHEMA, "success_status": "FREQUENCIES_COMPUTED" if stage == "global-assemble" else "STAGE_COMPLETE",
        "result": str(output / "result.json"), "log": str(q.path(config["work_directory"]) / "execution" / (stage + ".log")),
        "threads": 2, "memory_gib": 24, **config["phase_requests"][stage]}


def imported_healthy_job(q, guard, config, stage, record, physical, xc, consumer, deps):
    """Root may bind an already completed stage; never infer its future hash."""
    value = q.read(q.checked(record))
    actual_stage = "response-block" if stage.startswith("response-") else stage
    require(value["stage"] == actual_stage, "Imported producer stage differs")
    if stage == "rks-partial":
        require(value["component_index"] == deps["components"], "Imported A uses another component index")
        xc.qualify_partial(record, physical, 75, 762, guard)
        expected_worker, expected_sha = XC, XC_SHA
    elif stage == "h1ao":
        require(value["partial_result_dependency"] == deps["partial"], "Imported H1 uses another A")
        consumer.qualify_stage(record, actual_stage, physical, 75, 762, guard)
        expected_worker, expected_sha = CONSUMER, CONSUMER_SHA
    elif stage.startswith("response-"):
        require(value["block_index"] == int(stage.split("-")[1]), "Imported response block belongs to another global RHS block")
        consumer.qualify_response(record, physical, 75, 762, guard, deps["partial"], deps["h1"])
        expected_worker, expected_sha = CONSUMER, CONSUMER_SHA
    else:
        raise ValueError("Only qualified A/H1/response predecessors can be imported")
    proof = value["resource_supervision"]["proof"]
    require(q.path(proof["ledger_path"]) == q.path(config["ledger"]), "Imported producer must use the same cumulative ledger")
    ledger = q.read(config["ledger"])
    rows = [row for row in ledger["phases"] if row["phase"] == proof["phase"]]
    require(len(rows) == 1 and guard.healthy_closed_row(rows[0]), "Imported real producer not healthy closed")
    row = rows[0]
    require(value["source_worker"]["sha256"] == expected_sha
            and q.path(value["source_worker"]["path"]) == expected_worker, "Imported scientific source differs")
    return {"id": stage, "phase": row["phase"], "stage": actual_stage,
        "worker": {"path": str(expected_worker), "sha256": expected_sha}, "payload_argv": row["command"],
        "result_schema": RESULT_SCHEMA, "success_status": "STAGE_COMPLETE", "result": record["path"],
        "log": row["log"], "threads": 2, "memory_gib": 24, **config["phase_requests"][stage]}


def wait_for_clear(q, grant, end, mono_end, publish, interrupted):
    stop = min(time.monotonic() + grant["conflict_wait_seconds"], mono_end - 181)
    while True:
        observed = q.process_observation(opaque_exceptions=grant.get("opaque_process_identity_exceptions", []))
        if not observed["conflicts"]:
            return
        publish("WAITING_FOR_EXTERNAL_CONFLICT_CLEAR", {"process_observation": observed})
        require(not interrupted and time.monotonic() < stop and time.time() < end - 181,
                "Bounded conflict wait exhausted; no new quantum phase launched")
        time.sleep(min(10., max(.01, stop - time.monotonic())))


def launch_quantum(q, guard, config, grant, job, state, end, mono_end, interrupted, publish):
    ledger = q.read(config["ledger"])
    guard.validate_ledger(ledger)
    require(ledger["quantum_cpu_cap_seconds"] == 216000 and ledger["run_deadline_at_utc"] == DEADLINE, "Original pool changed")
    prior = state["stages"].get(job["id"])
    if prior:
        require(prior["status"] == "HEALTHY_CLOSED", "Unknown/failed/open checkpoint requires explicit root review, never automatic retry")
        q.verify_existing(job, guard, ledger, prior["seal"])
        publish("STAGE_REVERIFIED_AND_SKIPPED", {"stage": job["id"]})
        return ref(q, job["result"])
    require(not any(r["phase"] == job["phase"] for r in ledger["phases"]), "Unsealed prior phase requires root recovery")
    wait_for_clear(q, grant, end, mono_end, publish, interrupted)
    require(not interrupted and min(end - time.time(), mono_end - time.monotonic()) >= 181, "Finite invocation cleanup boundary reached")
    require(not q.path(job["result"]).parent.exists() and not q.path(job["log"]).exists(), "Never overwrite or retry an unclosed scientific output")
    # Expected new payload plus a modest free-space reserve; not a peak-memory claim.
    required = 1045159200 if job["id"] == "h1ao" else 32 * 1024**2
    fs = os.statvfs(config["work_directory"])
    require(fs.f_bavail * fs.f_frsize > required + 256 * 1024**2, "Insufficient disk headroom; stop before new quantum work")
    wall = min(job["wall_seconds"], int(min(end - time.time(), mono_end - time.monotonic())) - 120)
    command = [str(ROOT / config["python_executable"]), str(GUARD), "run", "--ledger", config["ledger"],
        "--phase", job["phase"], "--cpu-seconds", str(job["cpu_seconds"]), "--wall-seconds", str(wall),
        "--threads", "2", "--memory-gib", "24", "--log", job["log"], "--", *job["payload_argv"]]
    stage = state["stages"][job["id"]] = {"status": "STARTING", "materialized_job": job, "actual_guardian_argv": command}
    publish("GUARDED_STAGE_STARTING", {"stage": job["id"]})
    console = q.path(job["log"] + ".driver-console.log")
    console.parent.mkdir(parents=True, exist_ok=True)
    child = None
    identity = None
    try:
        with console.open("x") as handle:
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in config["pythonpath"]))
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                identity = q.proc_identity(child.pid)
            except (FileNotFoundError, ProcessLookupError):
                require(child.poll() is not None, "Live guardian identity unavailable")
            stage.update(status="RUNNING", guardian_identity=identity)
            publish("GUARDED_STAGE_RUNNING", {"stage": job["id"]})
            next_progress = time.monotonic() + 1800
            stop_requested = False
            def stop_record(record):
                stage.setdefault("stop_requests", []).append(record)
                publish("EXACT_GUARDIAN_STOP_REQUEST", {"stage": job["id"], "record": record})
            while child.poll() is None:
                sample = q.resource_stop_sample(interrupted, end, mono_end,
                    grant["launcher_cpu_seconds_per_invocation"], grant["launcher_rss_stop_bytes"])
                if sample["reasons"] and not stop_requested:
                    stop_requested = q.signal_owned_guardian(child, identity, sample, stop_record)
                require(time.time() < end and time.monotonic() < mono_end, "Hard launcher deadline; leave unresolved checkpoint for root")
                if time.monotonic() >= next_progress:
                    publish("ACTIVE_GUARDED_STAGE_PROGRESS", {"stage": job["id"], "resource_sample": sample})
                    next_progress += 1800
                time.sleep(1)
            code = child.wait()
        stage["actual_guardian_returncode"] = code
        if code != 0:
            stage["guardian_exit_origin"] = ("LAUNCHER_CALL_RECORDED" if any(r.get("action") == "SIGTERM_CALL_RETURNED" for r in stage.get("stop_requests", [])) else "UNKNOWN_EXTERNAL_TO_LAUNCHER")
        require(code == 0, "Guardian failed; retain actual failed cost and stop without automatic retry")
        ledger = q.read(config["ledger"])
        guard.validate_ledger(ledger)
        sealed = q.verify_stage(job, guard, ledger)
        require(sealed, "Healthy real producer and actual scientific result required")
        stage.update(status="HEALTHY_CLOSED", seal=sealed)
        publish("GUARDED_STAGE_HEALTHY_CLOSED", {"stage": job["id"]})
        return ref(q, job["result"])
    except BaseException as error:
        stage.update(status="BLOCKED_REQUIRES_ROOT_REVIEW", error_type=type(error).__name__)
        publish("GUARDED_STAGE_BLOCKED_NO_RETRY", {"stage": job["id"], "error_type": type(error).__name__})
        if child is not None and child.poll() is None and identity is not None:
            sample = q.resource_stop_sample(interrupted, end, mono_end,
                grant["launcher_cpu_seconds_per_invocation"], grant["launcher_rss_stop_bytes"])
            sample["reasons"].append("EXCEPTION_CLEANUP_" + type(error).__name__)
            q.signal_owned_guardian(child, identity, sample, lambda rec: publish("EXCEPTION_CLEANUP_STOP_REQUEST", {"stage": job["id"], "record": rec}))
            try:
                child.wait(timeout=max(.01, min(10., end-time.time(), mono_end-time.monotonic())))
            except subprocess.TimeoutExpired:
                stage["guardian_unresolved"] = True
                publish("GUARDIAN_UNRESOLVED_ROOT_REVIEW_REQUIRED", {"stage": job["id"]})
        raise


def run_analysis(q, config, grant, global_ref, state, end, mono_end, interrupted, publish):
    prior = state.get("analysis")
    if prior:
        require(prior["status"] == "HEALTHY_CLOSED", "Unclosed analysis never automatically retried")
        for item in prior["seal"]:
            q.checked(item)
        return
    wait_for_clear(q, grant, end, mono_end, publish, interrupted)
    require(min(end-time.time(), mono_end-time.monotonic()) > 181, "No analysis after original deadline")
    output = q.path(config["work_directory"]) / "independent-modes"
    require(not output.exists(), "Fresh analysis directory required")
    wall = min(900, int(min(end-time.time(), mono_end-time.monotonic())) - 60)
    command = [str(ROOT / config["python_executable"]), str(ANALYZER), "--result", global_ref["path"],
        "--result-sha256", global_ref["sha256"], "--output", str(output), "--cpu-seconds", "300", "--wall-seconds", str(wall)]
    state["analysis"] = {"status": "RUNNING", "actual_argv": command, "new_quantum_evaluations": 0}
    publish("INDEPENDENT_ANALYSIS_STARTING", {})
    console = q.path(config["work_directory"]) / "analysis-console.log"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in config["pythonpath"]))
    child = None
    identity = None
    try:
        with console.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                identity = q.proc_identity(child.pid)
            except (FileNotFoundError, ProcessLookupError):
                require(child.poll() is not None, "Live analysis identity unavailable")
            state["analysis"]["child_identity"] = identity
            publish("INDEPENDENT_ANALYSIS_CHILD_STARTED", {"child_identity": identity})
            analysis_stop = min(mono_end, time.monotonic()+wall+10)
            # The frozen analyzer enforces CPU/AS/RSS/finite wall and the Q lock.
            while child.poll() is None:
                require(not interrupted and time.time() < end and time.monotonic() < analysis_stop, "Analysis launcher stop/deadline")
                time.sleep(1)
            code = child.wait()
        require(code == 0, "Independent analysis failed; preserve outputs and stop")
        index_path = output / "artifact_index.json"
        index = q.read(index_path)
        require(index["analysis_source_sha256"] == ANALYZER_SHA and index["V7_worker_sha256"] == CONSUMER_SHA, "Independent analyzer/source export differs")
        sealed = [q.artifact(index_path), q.artifact(console)]
        for item in index["files"]:
            require(not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts, "Unsafe analysis artifact")
            sealed.append(q.artifact(q.checked({"path": str(output/item["path"]), "sha256": item["sha256"], "bytes": item["bytes"]})))
        checks = q.read(output / "checks.json")
        require(checks["schema"] == "research4-V7-independent-staged-modes-analysis-v1"
                and checks["status"] == "DONE_COMPUTED_MODES_AUDITED" and checks["normal_modes"] == 219
                and checks["normal_mode_count_complete"] is True and checks["new_quantum_evaluations"] == 0
                and checks["source_result_sha256"] == global_ref["sha256"]
                and checks["IR_intensities_status"] == "NOT_COMPUTED", "Actual full219 mode audit required")
        state["analysis"].update(status="HEALTHY_CLOSED", seal=sealed,
            qualified_for_mechanism=checks["qualified_for_mechanism"], mechanism_pairing_status=checks["mechanism_pairing_status"])
        publish("INDEPENDENT_ANALYSIS_HEALTHY_CLOSED", {})
    except BaseException:
        state["analysis"]["status"] = "BLOCKED_REQUIRES_ROOT_REVIEW"
        publish("INDEPENDENT_ANALYSIS_BLOCKED", {})
        if child is not None and child.poll() is None and identity is not None:
            sample = q.resource_stop_sample(interrupted, end, mono_end,
                grant["launcher_cpu_seconds_per_invocation"], grant["launcher_rss_stop_bytes"])
            sample["reasons"].append("INDEPENDENT_ANALYSIS_EXCEPTION_CLEANUP")
            q.signal_owned_guardian(child, identity, sample, lambda rec: publish("ANALYSIS_OWNED_CHILD_STOP_REQUEST", {"record": rec}))
            try:
                child.wait(timeout=max(.01, min(10., end-time.time(), mono_end-time.monotonic())))
            except subprocess.TimeoutExpired:
                state["analysis"]["child_unresolved"] = True
                publish("INDEPENDENT_ANALYSIS_UNRESOLVED", {})
        raise


def run(args, q, config, grant):
    q, guard, xc, consumer = metadata_modules()
    with q.queue_lock():
        state_path = q.path(config["state"])
        state = q.read(state_path) if state_path.exists() else {"schema": STATE_SCHEMA,
            "config_sha256": args.config_sha256, "grant_sha256": args.grant_sha256, "invocations": [], "stages": {}}
        require(state["config_sha256"] == args.config_sha256 and state["grant_sha256"] == args.grant_sha256, "Checkpoint belongs to another root-bound pipeline")
        require(all(i["status"] == "PIPELINE_HEALTHY_CLOSED" for i in state["invocations"]), "Unclosed/blocked prior invocation requires root reconciliation")
        require(len(state["invocations"]) < grant["max_invocations"] and args.attempt not in [i["id"] for i in state["invocations"]], "Finite fresh root invocation required")
        end = min(q.aware(DEADLINE).timestamp(), args.invocation_deadline)
        mono_end = args.invocation_monotonic_deadline
        invocation = {"id": args.attempt, "status": "RUNNING", "identity": q.proc_identity(os.getpid()), "started_at_utc": q.utc().isoformat()}
        state["invocations"].append(invocation)
        interrupted = []
        def receive(sig, _frame):
            if len(interrupted) < 8:
                interrupted.append({"signal": int(sig), "observed_at_monotonic_seconds": time.monotonic(), "sender": "UNKNOWN"})
        old_handlers = {s: signal.signal(s, receive) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGXCPU)}
        def publish(event, details):
            state["last_event"] = event
            invocation["self_CPU_observed_seconds"] = time.process_time()
            q.save(state_path, state)
            record = {"utc": q.utc().isoformat(), "event": event, "details": details,
                "launcher_CPU_seconds": time.process_time(), "launcher_RSS_bytes": q.own_rss()}
            monitor = q.path(config["monitor"])
            monitor.parent.mkdir(parents=True, exist_ok=True)
            with monitor.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n"); handle.flush()
            print(json.dumps(record), flush=True)
        try:
            publish("FINITE_PIPELINE_STARTED", {})
            guard.validate_ledger(q.read(config["ledger"]))
            xc.require_control_release(config["control_release"]["path"], RELEASE)
            components, physical = component_index(q, guard, xc, config, state)
            deps = {"components": components}
            for stage in STAGES:
                require(sha(__file__) == grant["driver_sha256"] and q.sha(q.path(args.config)) == args.config_sha256
                        and q.sha(q.path(args.grant)) == args.grant_sha256, "Driver/config/grant changed before stage")
                for item in config["source_refs"]:
                    q.checked(item)
                if stage == "global-assemble":
                    value = {"schema": "research4-v7-closed-response-results-index-v1", "results": [state["results"][f"response-{i:02d}"] for i in range(10)]}
                    deps["responses"] = immutable_json(q, q.path(config["work_directory"])/"metadata/response_results.json", value)
                    consumer.qualify_response_index(deps["responses"], physical, 75, 762, guard, deps["partial"], deps["h1"])
                job = stage_job(q, config, stage, deps)
                if stage in config.get("existing_results", {}) and stage not in state["stages"]:
                    job = imported_healthy_job(q, guard, config, stage, config["existing_results"][stage], physical, xc, consumer, deps)
                    sealed = q.verify_stage(job, guard, q.read(config["ledger"]))
                    require(sealed, "Imported stage must have a real full healthy seal")
                    state["stages"][stage] = {"status": "HEALTHY_CLOSED", "seal": sealed, "materialized_job": job,
                        "root_bound_existing_result": config["existing_results"][stage]}
                elif stage in state["stages"] and state["stages"][stage].get("root_bound_existing_result"):
                    job = state["stages"][stage]["materialized_job"]
                record = launch_quantum(q, guard, config, grant, job, state, end, mono_end, interrupted, publish)
                state.setdefault("results", {})[stage] = record
                if stage == "rks-partial":
                    xc.qualify_partial(record, physical, 75, 762, guard); deps["partial"] = record
                elif stage == "h1ao":
                    consumer.qualify_stage(record, "h1ao", physical, 75, 762, guard); deps["h1"] = record
                elif stage.startswith("response-"):
                    consumer.qualify_response(record, physical, 75, 762, guard, deps["partial"], deps["h1"])
                else:
                    result = q.read(record["path"], record["sha256"])
                    require(result["normal_mode_count"] == 219 and result["status"] == "FREQUENCIES_COMPUTED", "Actual full219 signed modes required")
                publish("DYNAMIC_DEPENDENCY_HEALTHY_AND_BOUND", {"stage": stage, "result": record})
            guard.validate_ledger(q.read(config["ledger"]))
            run_analysis(q, config, grant, state["results"]["global-assemble"], state, end, mono_end, interrupted, publish)
            invocation["status"] = "PIPELINE_HEALTHY_CLOSED"
            publish("PIPELINE_HEALTHY_CLOSED_COMPUTED_NOT_CHEMICALLY_CONFIRMED", {})
        except BaseException as error:
            invocation.update(status="BLOCKED_REQUIRES_ROOT_REVIEW_NO_AUTOMATIC_RETRY", error_type=type(error).__name__, reason=str(error))
            publish("PIPELINE_BLOCKED_PRESERVED", {"error_type": type(error).__name__})
            raise
        finally:
            invocation["ended_observed_at_utc"] = q.utc().isoformat()
            publish("PIPELINE_INVOCATION_CLOSED_OR_BLOCKED", {})
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def prepare(directory, work_directory):
    """Materialize a disabled template only; no quantum state is decoded."""
    q, _, _, _ = metadata_modules()
    base = ROOT / "reports/pyscf-spinosin-grid-checkpoint-20261105-v7"
    directory, work = q.path(directory), q.path(work_directory)
    require(not directory.exists() and not work.exists(), "Fresh config/work directories required")
    old = q.read(base / "queue-configs/spinosin-xc-blocks-recovery-003/config.json")
    old_grant = q.read(base / "queue-configs/spinosin-xc-blocks-recovery-002/root_grant_healthy_direct_closed.json",
        "592bece82327d8871057d65be4c4f0cf4bccc9f6501ce0581888d1db5295911a")
    prefix = "v7-spinosin-C001-" + work.name
    limits = {s: {"cpu_seconds": 600 if s == "rks-partial" else 7200 if s == "h1ao" else 3600,
        "wall_seconds": 900 if s == "rks-partial" else 10800 if s == "h1ao" else 5400} for s in STAGES}
    sources = [dict(item) for item in old["source_refs"]]
    sources.append(ref(q, Path(__file__).resolve()))
    config = {"schema": SCHEMA, "status": "NOT_RUN_DISABLED_PREPARATION", "work_directory": str(work),
        "state": str(work/"state.json"), "monitor": str(work/"monitor.jsonl"), "phase_prefix": prefix,
        "quantum_cpu_cap_seconds": 216000, "quantum_deadline_at_utc": DEADLINE, "ledger": old["ledger"],
        "python_executable": ".runtime/bin/python", "pythonpath": old["pythonpath"], "threads": 2,
        "snapshot": old["jobs"][0]["payload_argv"][old["jobs"][0]["payload_argv"].index("--snapshot")+1],
        "snapshot_manifest_sha256": SNAPSHOT,
        "gradient_result": ref(q, ROOT/"reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-restore-gradient-001/result.json"),
        "control_release": old["root_actual_control_release"],
        "plan_result": ref(q, base/"spinosin-C001-xc-plan-001/result.json"),
        "RHF_result": ref(q, base/"spinosin-C001-rhf-partial-001/result.json"),
        "master_XC_config": ref(q, base/"queue-configs/spinosin-xc-blocks-recovery-003/config.json"),
        "component_index": {"path": str(base/"spinosin-C001-XC-components-001.json"),
            "sha256": "98373692422b5c300f34fc8daf30a99b0c49f1e8f72a72a02e78cd62256e6dfb"},
        "existing_results": {"rks-partial": {"path": str(base/"spinosin-C001-rks-partial-001/result.json"),
            "sha256": "7bbcfc8b62ad81d53a65b09f0b8809d9646a62130e759a69f8d794e8f29ee8d9"}},
        "phase_requests": limits, "source_refs": sources,
        "request_ceilings_are_not_allocations": True, "all_frequencies_or_mechanism_claimed": False}
    directory.mkdir(parents=True)
    config_ref = immutable_json(q, directory/"config.json", config)
    grant = {"schema": GRANT_SCHEMA, "root_execution_approved": False, "root_serial_scheduling_confirmed": False,
        "config_sha256": config_ref["sha256"], "driver_sha256": sha(__file__), "no_new_quantum_budget": True,
        "no_deadline_extension": True, "expires_at_utc": DEADLINE, "max_invocations": 1,
        "launcher_cpu_seconds_per_invocation": 600, "launcher_rss_stop_bytes": 256*1024**2,
        "wall_seconds_per_invocation": 86400, "conflict_wait_seconds": 600,
        "opaque_process_identity_exceptions": old_grant["opaque_process_identity_exceptions"],
        "opaque_exception_source_grant": ref(q, base/"queue-configs/spinosin-xc-blocks-recovery-002/root_grant_healthy_direct_closed.json"),
        "status": "NOT_RUN_DISABLED_ROOT_MUST_VALIDATE_AND_ENABLE"}
    grant_ref = immutable_json(q, directory/"root_grant_TEMPLATE_DISABLED.json", grant)
    validate(q, config, grant, config_ref["sha256"], False)
    print(json.dumps({"status": "NOT_RUN_DISABLED_PREPARED", "config": config_ref, "grant": grant_ref}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config"); parser.add_argument("--config-sha256")
    parser.add_argument("--grant"); parser.add_argument("--grant-sha256")
    parser.add_argument("--prepare"); parser.add_argument("--work-directory")
    parser.add_argument("--attempt", default="PREPARATION_ONLY")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        require(args.work_directory and not args.execute and not args.detach and not args.config and not args.grant,
                "Preparation only creates a disabled fresh template")
        prepare(args.prepare, args.work_directory); return
    require(args.config and args.config_sha256 and args.grant and args.grant_sha256, "Explicit config/grant hashes required")
    q, _, _, _ = metadata_modules()
    config, grant = q.read(args.config, args.config_sha256), q.read(args.grant, args.grant_sha256)
    validate(q, config, grant, args.config_sha256, args.execute or args.detach)
    if not args.execute and not args.detach:
        print(json.dumps({"status": "NOT_RUN_CONFIGURATION_CHECKED", "quantum_stages": 13, "startup_service_created": False})); return
    require(args.attempt != "PREPARATION_ONLY", "Explicit fresh root invocation identity required")
    if args.detach:
        require(not args.execute, "Use --detach or --execute, not both")
        file = q.path(config["work_directory"])/"detached-driver-console.log"
        file.parent.mkdir(parents=True, exist_ok=True)
        command = [str(ROOT/config["python_executable"]), str(Path(__file__).resolve()), "--config", str(q.path(args.config)),
            "--config-sha256", args.config_sha256, "--grant", str(q.path(args.grant)), "--grant-sha256", args.grant_sha256,
            "--attempt", args.attempt, "--execute"]
        with file.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"status": "DETACHED_LAUNCH_REQUESTED_NOT_SCIENCE_DONE", "pid": child.pid,
            "source_sha256": sha(__file__), "config_sha256": args.config_sha256, "grant_sha256": args.grant_sha256,
            "identity": q.proc_identity(child.pid), "console": str(file), "machine_lifetime_not_guaranteed": True}
        immutable_json(q, q.path(config["work_directory"])/"detached-launch.json", record)
        print(json.dumps(record)); return
    args.invocation_deadline = min(q.aware(DEADLINE).timestamp(), time.time()+grant["wall_seconds_per_invocation"])
    args.invocation_monotonic_deadline = time.monotonic()+grant["wall_seconds_per_invocation"]
    with q.self_limits(grant):
        run(args, q, config, grant)


if __name__ == "__main__":
    main()
