"""Finite JSON/log observer; no quantum runs and no Git before DONE_PUBLISHED.

Only a healthy, published, complete 219-mode job may launch the separately
frozen analyzer. All subprocesses obey the remaining observation deadline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5"
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
SCHEMA = "research4-spinosin-completion-observer-v1"
JOB_PATH = REPORT / "run_bounded_spinosin_backup_job.py"
JOB_SHA = "ffad90a7554887b929bc208f4e941f080697bab5d04dc697361ac4361b3d3198"
JOB_MANIFEST_SHA = "80374c96a26fc77112cd05e6695d37d932466202dfa12a8fef5fca00924358f2"
ANALYZER_PATH = ROOT / "scripts/analyze_spinosin_v5_modes_20261104.py"
ANALYZER_SHA = "a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154"
CONSUMER_PATH = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
CONSUMER_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
GATE_SHA = "28832b15aea8b2a0c7947d133c6be752b545babc5081bcdcd6b7f3deb547692b"
CONTROLLER_SHA = "a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7"
ANALYSIS = REPORT / "final-mode-analysis"
FINAL_DOC = ROOT / "docs/PYSCF_SPINOSIN_FINAL_STATUS_20261104.md"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def now():
    return datetime.now(timezone.utc)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def read_json(path, *, maximum=32*1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Missing, symlink or oversized JSON: " + str(path))
    return json.loads(path.read_text())


def checked(item, *, expected=None):
    original = Path(item["path"])
    if original.is_symlink():
        raise ValueError("Symlink frozen input rejected")
    path = original.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != item["sha256"]:
        raise ValueError("Frozen source/input hash changed: " + str(path))
    if expected is not None and item["sha256"] != expected:
        raise ValueError("Unexpected registered source/input SHA")
    return path


def load_module(path, expected, name):
    if sha(path) != expected:
        raise ValueError("Frozen execution source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError("Preserve existing observer evidence: " + str(path))
    tmp = path.with_name(path.name+".writing")
    with tmp.open("w") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def terminal_publication(log_text):
    markers = []
    for line in log_text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get("status") == "DONE_PUBLISHED":
            if set(value) != {"status", "commit"} or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("commit", ""))):
                raise ValueError("Malformed terminal publication marker")
            markers.append(value["commit"])
    if len(markers) > 1:
        raise ValueError("Multiple terminal publication markers require manual review")
    return markers[0] if markers else None


def console_tail(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Symlink root console rejected")
    with path.open("rb") as handle:
        handle.seek(max(0,path.stat().st_size-16*1024**2))
        return handle.read(16*1024**2).decode("utf-8",errors="replace")


def observation_deadline(ledger, research):
    original = read_json(checked({"path": ledger["migration"]["source_v4_path"],
                                  "sha256": ledger["migration"]["source_v4_sha256"]}))
    if ledger["run_deadline_at_utc"] != original["run_deadline_at_utc"]:
        raise ValueError("Quantum deadline changed after migration")
    deadline = datetime.fromisoformat(original["run_deadline_at_utc"])+timedelta(seconds=600)
    return min(deadline, datetime.fromisoformat(research["authorized_end_at_utc"]))


def poll_decision(status, log_text, deadline, current):
    if current >= deadline:
        return "BLOCKED_OBSERVATION_DEADLINE", None
    revision = terminal_publication(log_text)
    if revision is not None:
        if status.get("status") != "DONE":
            raise ValueError("Published terminal marker and job JSON disagree")
        return "PUBLISHED", revision
    if status.get("status") == "PARTIAL":
        return "BLOCKED_PARTIAL_JOB", None
    return "WAIT", None


def readonly_git(deadline,*args):
    remaining=min(180,int((deadline-now()).total_seconds())-5)
    if remaining<1:
        raise TimeoutError("No observation time remains for a read-only Git query")
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True,timeout=remaining,
                env={**os.environ,"GIT_TERMINAL_PROMPT":"0"}).strip()


def verify_git(revision, job, deadline=None):
    query=job.git if deadline is None else lambda *args:readonly_git(deadline,*args)
    if query("branch", "--show-current") != BRANCH or query("rev-parse", "HEAD") != revision:
        raise ValueError("Published terminal commit is not the current research HEAD")
    remote = query("ls-remote", "origin", "refs/heads/"+BRANCH).split()
    if len(remote) != 2 or remote[0] != revision or remote[1] != "refs/heads/"+BRANCH:
        raise ValueError("Remote research branch does not equal terminal commit")
    if query("diff", "--cached", "--name-only"):
        raise ValueError("Another task has staged work; no automatic publication")


def qualify_closed_job(status, job_manifest):
    """JSON/log checks only; no NPY/HDF is opened by this observer."""
    if (status.get("status") != "DONE" or status.get("manifest_sha256") != JOB_MANIFEST_SHA
            or type(status.get("round")) is not int or not 1 <= status["round"] <= 17):
        raise ValueError("Job status lacks the frozen terminal identity")
    ledger = read_json(REPORT/"cpu_budget.json")
    if any(r.get("status") == "RUNNING" or r.get("actual_cpu_seconds") is None for r in ledger["phases"]):
        raise ValueError("A quantum phase is still active or has unknown final CPU")
    controller_path = REPORT/"auto-segments"/f"round-{status['round']:03d}"/"result.json"
    controller = read_json(controller_path)
    if (controller.get("status") != "DONE" or controller.get("completed_units") != 150
            or controller.get("total_units") != 150 or controller.get("source_sha256") != CONTROLLER_SHA
            or controller.get("worker_sha256") != CONSUMER_SHA
            or controller.get("snapshot_manifest_sha256") != SNAPSHOT_SHA
            or controller.get("final_result") != status.get("final_result")):
        raise ValueError("Final controller is not the complete frozen Spinosin task")
    result_path = checked(controller["final_result"])
    if not result_path.is_relative_to(controller_path.parent):
        raise ValueError("Final result is outside its controller round")
    result = read_json(result_path)
    if (result.get("status") != "FREQUENCIES_COMPUTED" or result.get("frequencies_status") != "DONE"
            or result.get("normal_mode_count") != 219 or result.get("atoms") != 75
            or len(result.get("frequencies_cm1", [])) != 219
            or result.get("complete_RHF_helper") is not True or result.get("complete_global_Hessian") is not True
            or result.get("stable_MF_state_unchanged") is not True
            or result.get("snapshot_manifest_sha256") != SNAPSHOT_SHA
            or result.get("source_worker", {}).get("sha256") != CONSUMER_SHA
            or result.get("IR_intensities_status") != "NOT_COMPUTED"):
        raise ValueError("Final scientific result lacks complete actual signed 219 modes")
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in result["frequencies_cm1"]):
        raise ValueError("Nonfinite signed frequencies are not valid completed evidence")
    worker = load_module(CONSUMER_PATH, CONSUMER_SHA, "completion_observer_closed_consumer")
    evidence = {"schema": worker.EXECUTION_SCHEMA, "source_worker": result["source_worker"],
                "resource_supervision": result["resource_supervision"], "output_directory": str(result_path.parent),
                "snapshot_manifest_sha256": SNAPSHOT_SHA, "phase": result["phase"]}
    closure = worker.verify_closed_execution(evidence, expected_snapshot_sha=SNAPSHOT_SHA, expected_output=result_path.parent)
    cache = Path(controller["cache_directory"]).resolve()
    if cache != ROOT/"outputs/pyscf-spinosin-resumable-20261104-v5/spinosin-C001-restart":
        raise ValueError("Unexpected exact-state RHF cache source")
    units = worker.verify_cached_unit_producers(cache,SNAPSHOT_SHA)
    if len(units)!=150 or any(v.get("qualification")!="CLOSED_HEALTHY_V5_PRODUCER" for v in units):
        raise ValueError("All150 unit producers must be independently healthy; no automatic forensic release")
    # These gates read JSON/log only; the analyzer will independently validate
    # snapshot array bytes after the same genuine closed-producer gates.
    checked(job_manifest["snapshot_manifest"], expected=SNAPSHOT_SHA)
    gradient = checked(job_manifest["gradient_check"], expected=GRADIENT_SHA)
    checked(job_manifest["scientific_gate"], expected=GATE_SHA)
    worker.qualify_gradient_receipt(gradient, GRADIENT_SHA, SNAPSHOT_SHA, result["boot_id"])
    snapshot = read_json(Path(job_manifest["snapshot_manifest"]["path"]))
    state_worker = worker.state_module()
    source_closure = state_worker.validate_closed_producer(snapshot, Path(job_manifest["snapshot_manifest"]["path"]))
    return {"final_result": record(result_path), "controller_result": record(controller_path),
            "final_closed_producer": closure, "SCF_source_closed_producer": source_closure,
            "healthy_RHF_unit_producer_count":len(units),
            "quantum_actual_cpu_seconds": ledger["consumed_cpu_seconds"], "IR_intensities_status": "NOT_COMPUTED"}


def verify_analysis(output, final_sha):
    checks = read_json(output/"checks.json")
    if (checks.get("status") != "DONE_COMPUTED_MODES_AUDITED" or checks.get("case") != "Spinosin"
            or checks.get("schema")!="research4-V5-independent-modes-analysis-v1"
            or checks.get("source_result_sha256") != final_sha or checks.get("normal_modes") != 219
            or checks.get("snapshot_manifest_sha256") != SNAPSHOT_SHA
            or checks.get("analysis_source_sha256") != ANALYZER_SHA
            or checks.get("IR_intensities_status") != "NOT_COMPUTED"
            or checks.get("chemical_attribution_verified") is not False
            or checks.get("PySCF_imported") is not False
            or checks.get("Spinosin_target_status")!="ACTUAL_CLOSED_FREQUENCIES_AUDITED"
            or checks.get("new_quantum_evaluations") != 0 or checks.get("theory_or_source_peaks_enter_classifier") is not False):
        raise ValueError("Postprocessor did not produce the qualified descriptive audit")
    profile=checks.get("postprocessing_environment",{})
    if (len(profile.get("CPU_affinity",[]))!=1 or profile.get("BLAS_threads")!="1"
            or not 0<profile.get("max_virtual_address_space_bytes",0)<=1024**3
            or checks.get("peak_rss_kib",1024**3)>1024**2):
        raise ValueError("Analyzer did not retain its one-CPU/small-memory resource profile")
    index = read_json(output/"artifact_index.json")
    if index.get("analysis_source_sha256") != ANALYZER_SHA:
        raise ValueError("Analysis output index source differs")
    for item in index.get("files", []):
        target = (output/item["path"]).resolve()
        if not target.is_relative_to(output.resolve()) or sha(target) != item["sha256"]:
            raise ValueError("Closed analysis artifact changed")
    if not index.get("files"):
        raise ValueError("Empty analysis output index")
    if not checks.get("qualified_for_mechanism") and (output/"all_experimental_theory_pairs.csv").exists():
        raise ValueError("Unqualified minimum must not receive mechanism peak pairing")
    return checks


def subprocess_until(command, log_path, deadline, *, maximum_wall=900, analysis=False):
    remaining = min(maximum_wall, int((deadline-now()).total_seconds())-5)
    if remaining < 1:
        raise TimeoutError("No observation time remains for the requested subprocess")
    env = {**os.environ, **{name:"1" for name in THREADS}, "GIT_TERMINAL_PROMPT":"0"}
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0,0))
        if analysis:
            if hasattr(os,"sched_getaffinity"):
                os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
            resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining+1))
            resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    with Path(log_path).open("xb") as log:
        child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True,preexec_fn=limits)
        try:
            return child.wait(timeout=remaining)
        except BaseException:
            if child.returncode is None:
                # This original Popen child is unreaped, so its session ID
                # cannot be recycled. Never signal a reaped/reusable PID.
                try:
                    os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
            raise


def report_text(state, checks=None):
    lines=["# Spinosin频率任务最终状态", "", "状态："+state["status"], "",
           "本报告根据实际闭合日志和产物生成，不代表化学成分确认。"]
    if checks is None:
        lines += ["", "尚未生成可用的Spinosin振动后处理结果。阻塞原因："+state.get("reason","未知"),
                  "", "保留原任务状态与恢复线索；未启动新的量子计算，也未扩大CPU或墙钟预算。"]
    else:
        lines += ["", f"实际审计219个有符号正常模式；虚频{checks['imaginary_frequency_count']}个，零频{checks['zero_frequency_count']}个。",
                  f"0.1–1.5 THz模式数：{checks['low_frequency_mode_count']}。机制辅助资格：{checks['qualified_for_mechanism']}。",
                  "", "|产物|路径|", "|---|---|",
                  "|完整频率表|reports/pyscf-spinosin-resumable-20261104-v5/final-mode-analysis/all_frequencies_cm1_THz.csv|",
                  "|低频模式表|reports/pyscf-spinosin-resumable-20261104-v5/final-mode-analysis/normal_modes_0p1_to_1p5_THz.csv|",
                  "|数值与来源核验|reports/pyscf-spinosin-resumable-20261104-v5/final-mode-analysis/checks.json|",
                  "", "孤立分子PBE/def2-SVP、grid1、无D3；单构象不等于全局最低能构象。分子频率不等于固态药材的晶格、氢键及集体振动。",
                  "频率接近只能作为描述性比较，不能确认峰的化学归属；IR强度NOT_COMPUTED，振动可能没有IR活性。",
                  "理论频率和候选峰没有用于分类器，既有96.0234%结果保持不变。"]
    return "\n".join(lines)+"\n"


def narrow_observer_directory(observer_dir):
    observer_dir=Path(observer_dir).resolve()
    if (not observer_dir.is_relative_to(REPORT) or observer_dir==REPORT
            or observer_dir.is_relative_to(REPORT/"auto-segments")
            or not observer_dir.relative_to(REPORT).parts[0].startswith("completion-observer")):
        raise ValueError("Observer output must be a narrow completion-observer* subtree")
    return observer_dir


def safe_publish_paths(spec, observer_dir, launch):
    observer_dir=narrow_observer_directory(observer_dir)
    paths = [Path(__file__).resolve(), ROOT/"tests/test_spinosin_completion_observer.py",
             ANALYZER_PATH, ROOT/"tests/test_analyze_spinosin_v5_modes.py", observer_dir, ANALYSIS, FINAL_DOC,launch]
    for value in spec.get("additional_publish_paths", []):
        path = Path(value).resolve()
        if not path.is_relative_to(REPORT) or path == REPORT or path.is_relative_to(REPORT/"auto-segments"):
            raise ValueError("Additional publication paths must be narrow non-quantum preflight scopes")
        paths.append(path)
    if any(not path.exists() for path in paths):
        raise ValueError("A requested publication scope is missing")
    return paths


def publish_child(request_path, request_sha):
    request_path = checked({"path":str(request_path),"sha256":request_sha})
    request = read_json(request_path)
    if request.get("observer_source_sha256") != sha(__file__) or now() >= datetime.fromisoformat(request["deadline"]):
        raise ValueError("Expired or changed publication request")
    if request.get("expected_head")!=request.get("root_terminal_commit"):
        raise ValueError("Only the exact published root-job commit can parent this publication")
    job = load_module(JOB_PATH,JOB_SHA,"completion_observer_publisher")
    console = console_tail(REPORT/"execution/bounded-backup-job.root-console.log")
    if terminal_publication(console) != request["root_terminal_commit"]:
        raise ValueError("Root job terminal publication marker changed")
    verify_git(request["expected_head"],job,datetime.fromisoformat(request["deadline"]))
    launch=checked(request["launch_manifest"])
    launch_spec=read_json(launch)
    if (launch_spec.get("schema")!=SCHEMA
            or checked(launch_spec["observer_source"])!=Path(__file__).resolve()):
        raise ValueError("Publication child does not match its registered observer source")
    checked(launch_spec["analyzer_source"],expected=ANALYZER_SHA)
    narrow_observer_directory(launch_spec["observer_output"])
    job_spec=read_json(checked(launch_spec["job_manifest"],expected=JOB_MANIFEST_SHA))
    qualification=qualify_closed_job(read_json(REPORT/"auto-segments/job-status.json"),job_spec)
    checked(request["analysis_index"])
    checked(request["analysis_checks"])
    verify_analysis(ANALYSIS,qualification["final_result"]["sha256"])
    paths = [Path(p).resolve() for p in request["paths"]]
    approved=safe_publish_paths(launch_spec,Path(launch_spec["observer_output"]).resolve(),launch)
    if paths!=approved:
        raise ValueError("Publication paths differ from the registered narrow scopes")
    revision = job.publish(paths,request["expected_head"],request["message"])
    print(json.dumps({"status":"OBSERVER_PUBLISHED","commit":revision}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--publish-request",type=Path)
    parser.add_argument("--publish-request-sha256")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.publish_request is not None:
        if not args.execute or args.manifest is not None:
            parser.error("Private publication child requires --execute and no observation manifest")
        publish_child(args.publish_request,args.publish_request_sha256)
        return
    if args.manifest is None or not args.manifest_sha256:
        parser.error("Frozen observer manifest and SHA required")
    launch=checked({"path":str(args.manifest),"sha256":args.manifest_sha256})
    spec=read_json(launch)
    if spec.get("schema")!=SCHEMA or checked(spec["observer_source"])!=Path(__file__).resolve():
        raise ValueError("Observer launch manifest/source differs")
    checked(spec["analyzer_source"],expected=ANALYZER_SHA)
    job_spec=read_json(checked(spec["job_manifest"],expected=JOB_MANIFEST_SHA))
    checked(job_spec["job_source"],expected=JOB_SHA)
    research=read_json(checked(job_spec["research_authorization"]))
    deadline=observation_deadline(read_json(REPORT/"cpu_budget.json"),research)
    monotonic_deadline=time.monotonic()+max(0,(deadline-now()).total_seconds())
    interval=spec.get("poll_seconds",30)
    if type(interval) not in (int,float) or not 1<=interval<=60:
        raise ValueError("Poll interval must be1..60seconds")
    observer_dir=narrow_observer_directory(spec["observer_output"])
    if observer_dir.exists():
        raise ValueError("Use a new narrow observer output directory")
    if not args.execute:
        print(json.dumps({"status":"NOT_RUN","deadline":deadline.isoformat(),"no_quantum_runs":True}))
        return
    observer_dir.mkdir(parents=True)
    state={"schema":SCHEMA,"status":"OBSERVING","started_at_utc":now().isoformat(),"deadline":deadline.isoformat(),
           "observer_source":record(__file__),"launch_manifest":record(launch),"new_quantum_evaluations":0,
           "target_arrays_read_by_observer":False,"IR_intensities_status":"NOT_COMPUTED"}
    try:
        while now()<deadline and time.monotonic()<monotonic_deadline:
            try:
                status=read_json(REPORT/"auto-segments/job-status.json")
                console=console_tail(REPORT/"execution/bounded-backup-job.root-console.log")
                decision,revision=poll_decision(status,console,deadline,now())
            except (OSError,json.JSONDecodeError):
                decision,revision="WAIT",None
            state.update(last_observed_at_utc=now().isoformat(),last_job_status=status if "status" in locals() else None)
            write_json(observer_dir/"status.json",state)
            if decision!="WAIT":
                break
            time.sleep(min(interval,max(0,(deadline-now()).total_seconds())))
        else:
            decision,revision="BLOCKED_OBSERVATION_DEADLINE",None
        if decision!="PUBLISHED":
            state.update(status=decision,reason="Root task has no qualified DONE_PUBLISHED terminal commit")
            raise RuntimeError(state["reason"])
        job=load_module(JOB_PATH,JOB_SHA,"completion_observer_closed_job")
        verify_git(revision,job,deadline)
        qualification=qualify_closed_job(status,job_spec)
        state.update(root_terminal_commit=revision,closed_job_qualification=qualification)
        if ANALYSIS.exists():
            raise FileExistsError("Do not overwrite existing final mode analysis")
        checked(spec["analyzer_source"],expected=ANALYZER_SHA)
        analyzer_log=observer_dir/"analyzer.log"
        returncode=subprocess_until([str(ROOT/".runtime/bin/python"),str(ANALYZER_PATH),"--result",qualification["final_result"]["path"],
                           "--result-sha256",qualification["final_result"]["sha256"],"--output",str(ANALYSIS)],analyzer_log,deadline,analysis=True)
        if returncode!=0:
            raise RuntimeError("Frozen analyzer failed; see its actual log")
        checks=verify_analysis(ANALYSIS,qualification["final_result"]["sha256"])
        state.update(status="MODES_AUDITED_READY_FOR_PUBLICATION",analysis_checks=record(ANALYSIS/"checks.json"),
                     analysis_index=record(ANALYSIS/"artifact_index.json"),analyzer_returncode=returncode,analyzer_log=record(analyzer_log),
                     qualified_for_mechanism=checks["qualified_for_mechanism"],mechanism_pairing_status=checks["mechanism_pairing_status"])
        if FINAL_DOC.exists():
            raise FileExistsError("Preserve existing final status report")
        FINAL_DOC.write_text(report_text(state,checks))
        write_json(observer_dir/"status.json",state)
        # Root's frozen publisher owns branch CAS, narrow staging and non-force
        # push. Run it in a bounded private session to cap total Git wall time.
        request={"observer_source_sha256":sha(__file__),"deadline":deadline.isoformat(),"root_terminal_commit":revision,
                 "expected_head":revision,"paths":[str(p) for p in safe_publish_paths(spec,observer_dir,launch)],
                 "launch_manifest":record(launch),
                 "analysis_index":state["analysis_index"],"analysis_checks":state["analysis_checks"],
                 "message":"Audit completed Spinosin modes and publish final descriptive evidence"}
        request_path=observer_dir/"publish-request.json";write_json(request_path,request,exclusive=True)
        runtime=ROOT/"outputs/spinosin-completion-observer";runtime.mkdir(parents=True,exist_ok=True)
        pub_log=runtime/(observer_dir.name+"-publish.log")
        code=subprocess_until([sys.executable,__file__,"--publish-request",str(request_path),"--publish-request-sha256",sha(request_path),"--execute"],
                              pub_log,deadline,maximum_wall=900)
        if code!=0:
            raise RuntimeError("Safe publication child failed; manual branch/remote review required")
        commits=[]
        for line in pub_log.read_text().splitlines():
            try:
                v=json.loads(line)
            except ValueError:
                continue
            if v.get("status")=="OBSERVER_PUBLISHED":commits.append(v["commit"])
        if len(commits)!=1 or not re.fullmatch(r"[0-9a-f]{40}",commits[0]):
            raise ValueError("Publication child has no unique successful commit receipt")
        # Published status in the committed tree truthfully says ready; exact
        # successful commit is in this terminal stdout, avoiding self-hash loops.
        print(json.dumps({"status":"OBSERVER_DONE_PUBLISHED","commit":commits[0],"analysis":str(ANALYSIS),
                          "qualified_for_mechanism":checks["qualified_for_mechanism"]}),flush=True)
    except BaseException as exc:
        state.update(status=state["status"] if state["status"].startswith("BLOCKED") else "BLOCKED_POSTPROCESSING_OR_PUBLICATION",
                     reason=str(exc),exception_type=type(exc).__name__,ended_at_utc=now().isoformat())
        write_json(observer_dir/"status.json",state)
        if not FINAL_DOC.exists():
            FINAL_DOC.write_text(report_text(state))
        print(json.dumps({"status":state["status"],"reason":state["reason"],"Git_not_performed_before_root_DONE_PUBLISHED":True}),flush=True)
        raise


if __name__=="__main__":
    main()
