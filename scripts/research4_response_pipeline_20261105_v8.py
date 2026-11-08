"""Finite V8 response continuation. Default is metadata-only NOT_RUN.

Only root may authorize execution. --detach creates a terminal-independent session;
it does not promise survival of a cloud machine shutdown. No quantum retry,
budget increase, scientific source modification, or service installation occurs.
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
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
CONSUMER = ROOT / "scripts/pyscf_staged_consumer_20261105_v8.py"
CONSUMER_SHA = "3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585"
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v8.py"
ANALYZER_SHA = "da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6"
SCHEMA = "research4-v8-finite-response-pipeline-config-v1"
GRANT_SCHEMA = "research4-v8-finite-response-pipeline-grant-v1"
STATE_SCHEMA = "research4-v8-finite-response-pipeline-state-v1"
RESULT_SCHEMA = "research4-staged-full-rks-hessian-result-v8"
CAP = 390160.8308359999
DEADLINE = "2026-11-09T12:30:00+00:00"
SNAPSHOT = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
RESPONSES = tuple(f"response-{i:02d}" for i in range(10))
STAGES = (*RESPONSES, "global-assemble")

class BoundaryReached(RuntimeError):
    """Normal healthy-stage boundary exit; no active Q has been started."""

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


def metadata_modules():
    q = load(QUEUE, QUEUE_SHA, "v8_pipeline_frozen_queue_metadata")
    return q, load(GUARD, GUARD_SHA, "v8_pipeline_actual_guard"), load(CONSUMER, CONSUMER_SHA, "v8_pipeline_actual_consumer")


def validate(q, config, grant, config_sha, execute=False):
    require(config.get("schema") == SCHEMA and grant.get("schema") == GRANT_SCHEMA, "Explicit V8 pipeline schemas required")
    require(grant.get("config_sha256") == config_sha and grant.get("driver_sha256") == sha(__file__), "Source/config-bound root grant required")
    require(grant.get("no_new_quantum_budget") is True and grant.get("no_deadline_extension") is True,
            "Only the already registered96h V8 pool is allowed")
    require(config.get("quantum_cpu_cap_seconds") == CAP and config.get("quantum_deadline_at_utc") == DEADLINE
            and grant.get("expires_at_utc") == DEADLINE, "V8 fixed cap and deadline required")
    require(type(grant.get("max_invocations")) is int and 1 <= grant["max_invocations"] <= 8, "Finite invocation count required")
    require(type(grant.get("launcher_cpu_seconds_per_invocation")) is int and 10 <= grant["launcher_cpu_seconds_per_invocation"] <= 600
            and type(grant.get("wall_seconds_per_invocation")) is int and 181 <= grant["wall_seconds_per_invocation"] <= 86400
            and type(grant.get("launcher_rss_stop_bytes")) is int and 16*1024**2 <= grant["launcher_rss_stop_bytes"] <= 256*1024**2,
            "Bounded own-process resources required")
    require(type(grant.get("conflict_wait_seconds")) is int and 0 <= grant["conflict_wait_seconds"] <= 1800, "Finite conflict wait required")
    exceptions = grant.get("opaque_process_identity_exceptions", [])
    require(isinstance(exceptions, list) and len(exceptions) <= 2 and len({x.get("pid") for x in exceptions}) == len(exceptions), "At most two distinct exact known daemon exceptions")
    for x in exceptions:
        require(x.get("uid") == 0 and x.get("comm") in ("dockerd", "containerd") and type(x.get("pid")) is int
                and x["pid"] > 0 and type(x.get("start_ticks")) is int and x["start_ticks"] > 0
                and isinstance(x.get("boot_id"), str) and len(x["boot_id"]) >= 8, "Exact approved daemon identity required")
    work = q.path(config["work_directory"])
    require(work.is_relative_to(ROOT/"reports") and work != ROOT/"reports", "Separate report namespace required")
    require(q.path(config["state"]) == work/"state.json" and q.path(config["monitor"]) == work/"monitor.jsonl", "Narrow checkpoint paths required")
    require(not q.path(config["ledger"]).is_relative_to(work) and not work.is_relative_to(q.path(config["snapshot"])), "Protect source and ledger")
    require(config.get("snapshot_manifest_sha256") == SNAPSHOT and config["gradient_result"]["sha256"] == GRADIENT
            and config.get("threads") == 2 and config.get("pythonpath") == [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"], "Exact source/G/site/thread contract required")
    require(type(config.get("phase_prefix")) is str and 1 <= len(config["phase_prefix"]) <= 100
            and all(c.isalnum() or c in "-_" for c in config["phase_prefix"]), "Safe finite phase prefix required")
    require(set(config.get("existing_results", {})).issubset(RESPONSES), "Only genuinely complete V8 response predecessors may be imported")
    expected = {str(QUEUE):QUEUE_SHA, str(GUARD):GUARD_SHA, str(CONSUMER):CONSUMER_SHA, str(ANALYZER):ANALYZER_SHA}
    refs = {str(q.checked(v)):v["sha256"] for v in config["source_refs"]}
    require(len(refs) == len(config["source_refs"]) and all(refs.get(p)==s for p,s in expected.items()), "All frozen orchestration/science sources required")
    for key in ("partial_result", "h1_result", "gradient_result"):
        q.checked(config[key])
    require(set(config["phase_requests"]) == set(STAGES), "Exactly10 responses and1 global stage required")
    for stage in STAGES:
        v=config["phase_requests"][stage]
        require(type(v["cpu_seconds"]) is int and 125 <= v["cpu_seconds"] <= 122400
                and type(v["wall_seconds"]) is int and 181 <= v["wall_seconds"] <= 43200, "Finite phase request required")
    require(config["phase_requests"][RESPONSES[0]] == {"cpu_seconds":21600,"wall_seconds":14400}, "Root approved first nativeB8 pilot ceiling required")
    require(config.get("forecast_multiplier") == 1.2 and config.get("global_CPU_reserve_seconds") == 1800,
            "Fixed measured-cost feasibility gate required")
    if config.get("control_release"):
        q.checked(config["control_release"])
    if execute:
        require(config.get("control_release") and grant.get("root_execution_approved") is True
                and grant.get("root_serial_scheduling_confirmed") is True, "Actual new controls and enabled root serial grant required")
        require(q.utc() < q.aware(DEADLINE), "96h V8 deadline expired")


def common_argv(config, stage, output):
    return [str(ROOT/config["python_executable"]), str(CONSUMER), "--execute", "--stage", stage,
        "--threads", "2", "--snapshot", config["snapshot"], "--snapshot-manifest-sha256", SNAPSHOT,
        "--gradient-check-result", config["gradient_result"]["path"], "--gradient-check-result-sha256", GRADIENT,
        "--control-release", config["control_release"]["path"], "--control-release-sha256", config["control_release"]["sha256"],
        "--partial-result", config["partial_result"]["path"], "--partial-result-sha256", config["partial_result"]["sha256"],
        "--h1-result", config["h1_result"]["path"], "--h1-result-sha256", config["h1_result"]["sha256"],
        "--output", str(output)]


def stage_job(q, config, stage, response_index=None):
    require(stage in STAGES, "Unapproved phase")
    output=q.path(config["work_directory"])/"science"/stage
    actual="response-block" if stage in RESPONSES else "global-assemble"
    command=common_argv(config,actual,output)
    if stage in RESPONSES:
        command += ["--block-index",str(RESPONSES.index(stage))]
    else:
        require(response_index is not None, "Actual complete response index required")
        command += ["--response-index",response_index["path"],"--response-index-sha256",response_index["sha256"]]
    return {"id":stage,"phase":config["phase_prefix"]+"-"+stage,"stage":actual,
        "worker":{"path":str(CONSUMER),"sha256":CONSUMER_SHA},"payload_argv":command,
        "result_schema":RESULT_SCHEMA,"success_status":"STAGE_COMPLETE" if stage in RESPONSES else "FREQUENCIES_COMPUTED",
        "result":str(output/"result.json"),"log":str(q.path(config["work_directory"])/"execution"/(stage+".log")),
        "threads":2,"memory_gib":24,**config["phase_requests"][stage]}


def imported_response(q, guard, consumer, config, stage, record, physical):
    value,_,_=consumer.closed_stage(record,"response-block",physical,75,762,guard)
    require(value["block_index"]==RESPONSES.index(stage),"Imported RHS block differs")
    consumer.qualify_response(record,physical,75,762,guard,config["partial_result"],config["h1_result"])
    proof=value["resource_supervision"]["proof"]
    require(q.path(proof["ledger_path"])==q.path(config["ledger"]),"Imported response must use actual V8 ledger")
    rows=[p for p in q.read(config["ledger"])["phases"] if p["phase"]==proof["phase"]]
    require(len(rows)==1 and guard.healthy_closed_row(rows[0]),"Imported response not truly healthy closed")
    row=rows[0]
    return {"id":stage,"phase":row["phase"],"stage":"response-block","worker":{"path":str(CONSUMER),"sha256":CONSUMER_SHA},
        "payload_argv":row["command"],"result_schema":RESULT_SCHEMA,"success_status":"STAGE_COMPLETE",
        "result":record["path"],"log":row["log"],"threads":2,"memory_gib":24,**config["phase_requests"][stage]}


def feasibility(config, state, ledger):
    completed=[k for k in RESPONSES if state["stages"].get(k,{}).get("status")=="HEALTHY_CLOSED"]
    pending=[k for k in RESPONSES if k not in completed]
    if not completed or not pending:
        return None
    costs=[]
    for k in completed:
        phase=state["stages"][k]["materialized_job"]["phase"]
        rows=[p for p in ledger["phases"] if p["phase"]==phase]
        require(len(rows)==1 and rows[0]["status"]=="EXITED" and rows[0]["exit_code"]==0,
                "Forecast requires real healthy wait4 costs")
        cost=rows[0]["actual_cpu_seconds"]
        require(isinstance(cost,(int,float)) and cost>0,"Finite positive actual pilot cost required")
        costs.append(cost)
    reference=max(costs)
    needed=reference*len(pending)*config["forecast_multiplier"]+config["global_CPU_reserve_seconds"]
    remaining=ledger["quantum_cpu_cap_seconds"]-ledger["consumed_cpu_seconds"]
    return {"measured_reference_CPU_seconds":reference,"pending_response_blocks":len(pending),
        "forecast_CPU_seconds":needed,"actual_remaining_CPU_seconds":remaining,"feasible":remaining>=needed,
        "forecast_is_not_completion_or_budget_increase":True}


def ensure_full_window(job, end, mono_end):
    if min(end-time.time(),mono_end-time.monotonic()) < job["wall_seconds"]+180:
        raise BoundaryReached("Full next-stage wall request plus cleanup does not fit this bounded invocation")

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
    require(ledger["quantum_cpu_cap_seconds"] == CAP and ledger["run_deadline_at_utc"] == DEADLINE, "Registered V8 pool changed")
    prior = state["stages"].get(job["id"])
    if prior:
        require(prior["status"] == "HEALTHY_CLOSED", "Unknown/failed/open checkpoint requires explicit root review, never automatic retry")
        q.verify_existing(job, guard, ledger, prior["seal"])
        publish("STAGE_REVERIFIED_AND_SKIPPED", {"stage": job["id"]})
        return ref(q, job["result"])
    require(not any(r["phase"] == job["phase"] for r in ledger["phases"]), "Unsealed prior phase requires root recovery")
    ensure_full_window(job,end,mono_end)
    wait_for_clear(q, grant, end, mono_end, publish, interrupted)
    ensure_full_window(job,end,mono_end)
    require(not interrupted,"Explicit launcher signal before next stage")
    require(not q.path(job["result"]).parent.exists() and not q.path(job["log"]).exists(), "Never overwrite or retry an unclosed scientific output")
    # Expected new payload plus a modest free-space reserve; not a peak-memory claim.
    required = 32 * 1024**2
    fs = os.statvfs(config["work_directory"])
    require(fs.f_bavail * fs.f_frsize > required + 256 * 1024**2, "Insufficient disk headroom; stop before new quantum work")
    wall = job["wall_seconds"]
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
    ensure_full_window({"wall_seconds":900},end,mono_end)
    wait_for_clear(q, grant, end, mono_end, publish, interrupted)
    ensure_full_window({"wall_seconds":900},end,mono_end)
    output = q.path(config["work_directory"]) / "independent-modes"
    require(not output.exists(), "Fresh analysis directory required")
    wall = 900
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
        require(index["analysis_source_sha256"] == ANALYZER_SHA and index["V8_worker_sha256"] == CONSUMER_SHA, "Independent analyzer/source export differs")
        sealed = [q.artifact(index_path), q.artifact(console)]
        for item in index["files"]:
            require(not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts, "Unsafe analysis artifact")
            sealed.append(q.artifact(q.checked({"path": str(output/item["path"]), "sha256": item["sha256"], "bytes": item["bytes"]})))
        checks = q.read(output / "checks.json")
        require(checks["schema"] == "research4-V8-independent-staged-modes-analysis-v1"
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
    q,guard,consumer=metadata_modules()
    with q.queue_lock():
        state_path=q.path(config['state'])
        state=q.read(state_path) if state_path.exists() else {'schema':STATE_SCHEMA,'config_sha256':args.config_sha256,
            'grant_sha256':args.grant_sha256,'invocations':[],'stages':{},'results':{}}
        require(state['config_sha256']==args.config_sha256 and state['grant_sha256']==args.grant_sha256,
                'Checkpoint belongs to a different immutable config/grant')
        allowed=('BOUNDARY_CLOSED','PIPELINE_HEALTHY_CLOSED')
        require(all(i['status'] in allowed for i in state['invocations']),
                'Unclosed/failed invocation needs explicit root forensics, never missing-PID inference')
        require(len(state['invocations'])<grant['max_invocations'] and args.attempt not in [i['id'] for i in state['invocations']],
                'Finite fresh invocation required')
        end=min(q.aware(DEADLINE).timestamp(),args.invocation_deadline);mono_end=args.invocation_monotonic_deadline
        inv={'id':args.attempt,'status':'RUNNING','identity':q.proc_identity(os.getpid()),'started_at_utc':q.utc().isoformat()}
        state['invocations'].append(inv);interrupted=[]
        def receive(sig,_frame):
            if len(interrupted)<8:interrupted.append({'signal':int(sig),'observed_monotonic':time.monotonic(),'sender':'UNKNOWN'})
        handlers={sig:signal.signal(sig,receive) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGXCPU)}
        def publish(event,details):
            state['last_event']=event;inv['self_CPU_observed_seconds']=time.process_time();q.save(state_path,state)
            row={'utc':q.utc().isoformat(),'event':event,'details':details,'launcher_CPU_seconds':time.process_time(),'launcher_RSS_bytes':q.own_rss()}
            p=q.path(config['monitor']);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            print(json.dumps(row),flush=True)
        try:
            publish('FINITE_V8_RESPONSE_PIPELINE_STARTED',{})
            ledger=q.read(config['ledger']);guard.validate_ledger(ledger)
            require(ledger['quantum_cpu_cap_seconds']==CAP and ledger['run_deadline_at_utc']==DEADLINE,'Registered V8 pool changed')
            require(not any(x['status']=='RUNNING' for x in ledger['phases']),'An existing live/orphan Q phase needs root inspection')
            consumer.require_control_release(config['control_release']['path'],config['control_release']['sha256'],guard)
            av=q.read(config['partial_result']['path'],config['partial_result']['sha256']);physical=av['physical_binding']
            consumer.qualify_partial(config['partial_result'],physical,75,762,guard)
            consumer.qualify_stage(config['h1_result'],'h1ao',physical,75,762,guard)
            hv=q.read(config['h1_result']['path'],config['h1_result']['sha256'])
            require(hv['partial_result_dependency']==config['partial_result'],'Actual H1 belongs to another A')
            publish('ACTUAL_LEGACY_A_H1_AND_V8_CONTROL_RELEASE_QUALIFIED',{})
            for stage in STAGES:
                require(sha(__file__)==grant['driver_sha256'] and q.sha(q.path(args.config))==args.config_sha256
                        and q.sha(q.path(args.grant))==args.grant_sha256,'Driver/config/grant changed')
                for v in config['source_refs']:q.checked(v)
                index=None
                if stage=='global-assemble':
                    require(all(state['stages'].get(k,{}).get('status')=='HEALTHY_CLOSED' for k in RESPONSES),
                            'All10 genuine healthy responses required before index')
                    value={'schema':'research4-v8-closed-response-results-index-v1','results':[state['results'][k] for k in RESPONSES]}
                    index=immutable_json(q,q.path(config['work_directory'])/'metadata/response_results.json',value)
                    consumer.qualify_response_index(index,physical,75,762,guard,config['partial_result'],config['h1_result'])
                job=stage_job(q,config,stage,index)
                if stage in config.get('existing_results',{}) and stage not in state['stages']:
                    job=imported_response(q,guard,consumer,config,stage,config['existing_results'][stage],physical)
                    seal=q.verify_stage(job,guard,q.read(config['ledger']))
                    require(seal,'Actual imported response full seal required')
                    state['stages'][stage]={'status':'HEALTHY_CLOSED','seal':seal,'materialized_job':job,
                        'root_bound_existing_result':config['existing_results'][stage]}
                elif state['stages'].get(stage,{}).get('root_bound_existing_result'):
                    job=state['stages'][stage]['materialized_job']
                if stage in RESPONSES and state['stages'].get(stage,{}).get('status')!='HEALTHY_CLOSED':
                    forecast=feasibility(config,state,q.read(config['ledger']))
                    if forecast is not None:
                        state['latest_cost_forecast']=forecast;publish('ACTUAL_PILOT_COST_FEASIBILITY_GATE',forecast)
                        require(forecast['feasible'],'BLOCKED measured response-cost forecast exceeds fixed remaining V8 pool; preserve pilot and await root review')
                record=launch_quantum(q,guard,config,grant,job,state,end,mono_end,interrupted,publish)
                state['results'][stage]=record
                if stage in RESPONSES:
                    consumer.qualify_response(record,physical,75,762,guard,config['partial_result'],config['h1_result'])
                else:
                    v=q.read(record['path'],record['sha256'])
                    require(v['status']=='FREQUENCIES_COMPUTED' and v['normal_mode_count']==219,
                            'Actual complete219 signed modes required; minimum qualification separate')
                publish('ACTUAL_DYNAMIC_RESULT_DEPENDENCY_BOUND',{'stage':stage,'result':record})
            require(not any(x['status']=='RUNNING' for x in q.read(config['ledger'])['phases']),'No mode analysis while Q open')
            run_analysis(q,config,grant,state['results']['global-assemble'],state,end,mono_end,interrupted,publish)
            inv['status']='PIPELINE_HEALTHY_CLOSED';publish('PIPELINE_HEALTHY_CLOSED_COMPUTED_NOT_CHEMICALLY_CONFIRMED',{})
        except BoundaryReached as err:
            inv.update(status='BOUNDARY_CLOSED',reason=str(err));publish('HEALTHY_STAGE_BOUNDARY_CLOSED_NO_ACTIVE_Q_CANCELLED',{'reason':str(err)})
        except BaseException as err:
            inv.update(status='BLOCKED_REQUIRES_ROOT_REVIEW_NO_AUTOMATIC_RETRY',error_type=type(err).__name__,reason=str(err))
            publish('PIPELINE_BLOCKED_PRESERVED',{'error_type':type(err).__name__});raise
        finally:
            inv['ended_observed_at_utc']=q.utc().isoformat();publish('PIPELINE_INVOCATION_CLOSED_OR_BLOCKED',{})
            for sig,handler in handlers.items():signal.signal(sig,handler)


def prepare(directory,work_directory,release=None,release_sha=None):
    q,_,_=metadata_modules();directory=q.path(directory);work=q.path(work_directory)
    require(not directory.exists(),'Fresh preparation namespace required')
    b=ROOT/'reports/pyscf-spinosin-grid-checkpoint-20261105-v7'
    release_ref=None
    if release is not None:
        require(release_sha is not None,'Actual release SHA required');release_ref={'path':str(q.path(release)),'sha256':release_sha};q.checked(release_ref)
    cfg={'schema':SCHEMA,'status':'DISABLED_ROOT_APPROVAL_REQUIRED','created_at_utc':q.utc().isoformat(),
        'work_directory':str(work),'state':str(work/'state.json'),'monitor':str(work/'monitor.jsonl'),
        'ledger':str(ROOT/'reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json'),
        'quantum_cpu_cap_seconds':CAP,'quantum_deadline_at_utc':DEADLINE,'python_executable':'.runtime/bin/python',
        'pythonpath':['.runtime/quantum-pilot-site','.runtime/quantum-upgrade-site'],'threads':2,
        'snapshot':str(ROOT/'reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/exact-mf-snapshot'),
        'snapshot_manifest_sha256':SNAPSHOT,'gradient_result':ref(q,ROOT/'reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-restore-gradient-001/result.json'),
        'partial_result':ref(q,b/'spinosin-C001-rks-partial-001/result.json'),
        'h1_result':ref(q,b/'post-xc-pipeline-001/science/h1ao/result.json'),'control_release':release_ref,
        'source_refs':[ref(q,p) for p in (QUEUE,GUARD,CONSUMER,ANALYZER)],'existing_results':{},
        'phase_prefix':'v8-spinosin-C001-response-pipeline-001','forecast_multiplier':1.2,'global_CPU_reserve_seconds':1800,
        'phase_requests':{k:{'cpu_seconds':21600,'wall_seconds':14400} for k in RESPONSES}}
    cfg['phase_requests']['global-assemble']={'cpu_seconds':1800,'wall_seconds':3600}
    directory.mkdir(parents=True);cp=directory/'config_TEMPLATE_DISABLED.json';immutable_json(q,cp,cfg)
    old=q.read(b/'queue-configs/spinosin-xc-blocks-recovery-002/root_grant_healthy_direct_closed.json')
    grant={'schema':GRANT_SCHEMA,'created_at_utc':q.utc().isoformat(),'driver_sha256':sha(__file__),'config_sha256':q.sha(cp),
        'root_execution_approved':False,'root_serial_scheduling_confirmed':False,'no_new_quantum_budget':True,'no_deadline_extension':True,
        'expires_at_utc':DEADLINE,'max_invocations':4,'launcher_cpu_seconds_per_invocation':600,'wall_seconds_per_invocation':86400,
        'launcher_rss_stop_bytes':256*1024**2,'conflict_wait_seconds':600,
        'opaque_process_identity_exceptions':old['opaque_process_identity_exceptions'],
        'scope':'Root must bind actual new V8 control release, source/config and approved registered V8 pool; no Q started by preparation.'}
    gp=directory/'root_grant_TEMPLATE_DISABLED.json';immutable_json(q,gp,grant)
    return {'status':'NOT_RUN_PREPARATION_ONLY','source_sha256':sha(__file__),'config':ref(q,cp),'grant':ref(q,gp),
        'actual_control_release_supplied':release_ref is not None,'service_or_autostart_created':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',type=Path);p.add_argument('--work-directory',type=Path)
    p.add_argument('--control-release',type=Path);p.add_argument('--control-release-sha256')
    p.add_argument('--config');p.add_argument('--config-sha256');p.add_argument('--grant');p.add_argument('--grant-sha256')
    p.add_argument('--attempt',default='PREPARATION_ONLY');p.add_argument('--execute',action='store_true');p.add_argument('--detach',action='store_true')
    args=p.parse_args()
    if args.prepare:
        require(args.work_directory is not None and not args.execute and not args.detach and args.config is None,'Preparation is metadata only')
        print(json.dumps(prepare(args.prepare,args.work_directory,args.control_release,args.control_release_sha256)));return
    require(all((args.config,args.config_sha256,args.grant,args.grant_sha256)),'Explicit config/grant and hashes required')
    q,_,_=metadata_modules();config=q.read(args.config,args.config_sha256);grant=q.read(args.grant,args.grant_sha256)
    validate(q,config,grant,args.config_sha256,args.execute or args.detach)
    if not args.execute and not args.detach:
        print(json.dumps({'status':'NOT_RUN_CONFIGURATION_CHECKED','quantum_stages':11,'actual_control_release_supplied':bool(config.get('control_release'))}));return
    require(args.attempt!='PREPARATION_ONLY','Explicit finite invocation identity required')
    require(1 <= len(args.attempt) <= 100 and all(c.isalnum() or c in '-_' for c in args.attempt),
            'Safe finite invocation identity required')
    if args.detach:
        require(not args.execute,'Use detach or execute, not both')
        f=q.path(config['work_directory'])/'execution'/('driver-'+args.attempt+'.console.log');f.parent.mkdir(parents=True,exist_ok=True)
        command=[str(ROOT/config['python_executable']),str(Path(__file__).resolve()),'--config',str(q.path(args.config)),
            '--config-sha256',args.config_sha256,'--grant',str(q.path(args.grant)),'--grant-sha256',args.grant_sha256,'--attempt',args.attempt,'--execute']
        with f.open('x') as handle:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        record={'status':'DETACHED_LAUNCH_REQUESTED_NOT_SCIENCE_DONE','identity':q.proc_identity(child.pid),'source_sha256':sha(__file__),
            'config_sha256':args.config_sha256,'grant_sha256':args.grant_sha256,'console':str(f),'machine_shutdown_survival_not_guaranteed':True}
        immutable_json(q,q.path(config['work_directory'])/('detached-launch-'+args.attempt+'.json'),record)
        q.save(q.path(config['work_directory'])/'current-launch.json',record)
        print(json.dumps(record));return
    args.invocation_deadline=min(q.aware(DEADLINE).timestamp(),time.time()+grant['wall_seconds_per_invocation'])
    args.invocation_monotonic_deadline=time.monotonic()+grant['wall_seconds_per_invocation']
    with q.self_limits(grant):run(args,q,config,grant)


if __name__=='__main__':main()
