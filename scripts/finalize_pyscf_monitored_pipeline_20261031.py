"""Finite, root-permitted finalization of an ended V3 pipeline; never launches QM.

Only an ended controller plus a closed, exactly bound ledger permits one frozen
analysis invocation. Optional Git delivery is narrowly scoped and never forces,
resets, merges, stages unrelated files or changes an existing scientific report.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN_REL = Path("reports/pyscf-spinosin-continuation-20261031-v3")
DOC_REL = Path("docs/PYSCF_SPINOSIN_MONITORED_RESULTS_20261031.md")
ANALYZER_REL = Path("scripts/analyze_pyscf_monitored_results_20261031.py")
SUPERVISOR_REL = Path("scripts/run_pyscf_monitored_budget_20261031.py")
CONTROLLER_REL = Path("scripts/run_pyscf_v3_pipeline_20261031.py")
FINISHER_REL = Path("scripts/finalize_pyscf_monitored_pipeline_20261031.py")
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
SCHEMA = "research4-pyscf-conditional-finisher-permit-v1"
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 25 * 1024**2, 150 * 1024**2
POLL_SECONDS, MAX_ITERATIONS = 15, 5761


def utc():
    return datetime.now(timezone.utc)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_time(value):
    if not isinstance(value, str):
        raise ValueError("Dated ISO string with timezone required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Actual UTC offset required")
    return result


def validate_permit(path, expected_sha, push=False):
    if Path(path).is_symlink():
        raise ValueError("Root permit must be an ordinary immutable file")
    path = resolve(path)
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError("Explicit root permit SHA differs from exact file bytes")
    value = read(path)
    if value.get("schema") != SCHEMA or value.get("allow_conditional_finalization") is not True:
        raise ValueError("A real root conditional permit is required; templates do not authorize work")
    parse_time(value.get("created_at_utc", ""))
    run_root = (ROOT / RUN_REL).resolve()
    for location in (ROOT/RUN_REL, ROOT/DOC_REL):
        if not location.resolve().is_relative_to(ROOT.resolve()) or any(
                parent.is_symlink() for parent in (location, *location.parents) if parent != ROOT):
            raise ValueError("Fixed report/document paths must not escape through symlinks")
    expected_paths = {"run_root": run_root,
        "pipeline_state_path": run_root / "pipeline-native4-001/state.json",
        "ledger_path": run_root / "cpu_budget.json", "analysis_output_path": run_root / "final-analysis",
        "documentation_path": (ROOT / DOC_REL).resolve()}
    for key, expected in expected_paths.items():
        if resolve(value.get(key, "")) != expected:
            raise ValueError("Unreviewed finisher path: " + key)
        raw = Path(value.get(key, ""))
        raw = raw if raw.is_absolute() else ROOT/raw
        if raw.is_symlink() or any(parent.is_symlink() for parent in raw.parents if parent != ROOT):
            raise ValueError("Finisher evidence/output paths must not traverse symlinks")
    if not path.is_relative_to(run_root):
        raise ValueError("Root permit must be saved in the actual V3 report directory")
    if value.get("expected_branch") != BRANCH or type(value.get("allow_git_push")) is not bool:
        raise ValueError("Fixed research branch and explicit Git permission required")
    if push and value["allow_git_push"] is not True:
        raise ValueError("--push requires explicit root permit authorization")
    if (value.get("poll_interval_seconds") != POLL_SECONDS
            or type(value.get("max_poll_iterations")) is not int
            or not 1 <= value["max_poll_iterations"] <= MAX_ITERATIONS):
        raise ValueError("Finisher polling must be15s and bounded within original24h")
    sources = value.get("source_sha256", {})
    for relative in (FINISHER_REL, ANALYZER_REL, SUPERVISOR_REL, CONTROLLER_REL):
        if sources.get(str(relative)) != sha(ROOT / relative):
            raise ValueError("Final frozen source differs from root permit: " + str(relative))
    for relative, expected in sources.items():
        source = resolve(relative)
        if not source.is_relative_to(ROOT) or sha(source) != expected:
            raise ValueError("Source binding changed: " + str(relative))
    ledger = read(expected_paths["ledger_path"])
    if ledger.get("schema") != "research4-pyscf-monitored-budget-v3":
        raise ValueError("Only the actual V3 monitored ledger may be finalized")
    authorization = resolve(ledger["authorization_path"])
    if (value.get("authorization_sha256") != ledger.get("authorization_sha256")
            or sha(authorization) != value["authorization_sha256"]):
        raise ValueError("Original V3 authorization binding differs")
    if value.get("run_deadline_at_utc") != ledger.get("run_deadline_at_utc"):
        raise ValueError("Root permit may not extend the original V3 deadline")
    deadline = parse_time(value["run_deadline_at_utc"])
    plan = expected_paths["pipeline_state_path"].parent / "plan.json"
    if sha(plan) != value.get("pipeline_plan_sha256"):
        raise ValueError("Actual frozen pipeline plan differs from root permit")
    return {**value, "_permit_path": path, "_permit_sha256": expected_sha,
            "_run_root": run_root, "_state_path": expected_paths["pipeline_state_path"],
            "_ledger_path": expected_paths["ledger_path"],
            "_analysis_output": expected_paths["analysis_output_path"],
            "_document_path": expected_paths["documentation_path"], "_deadline": deadline}


def closed_pipeline_snapshot(run_root, pipeline_state, ledger_path):
    run_root, pipeline_state, ledger_path = map(resolve, (run_root, pipeline_state, ledger_path))
    if not pipeline_state.is_relative_to(run_root) or ledger_path != run_root / "cpu_budget.json":
        raise ValueError("Final evidence must remain in the actual run")
    if not pipeline_state.is_file():
        return None
    state, ledger = read(pipeline_state), read(ledger_path)
    if ledger.get("schema") != "research4-pyscf-monitored-budget-v3":
        raise ValueError("Unexpected final ledger schema")
    if any(row.get("status") == "BUDGET_VIOLATION_REQUIRES_REVIEW" for row in ledger["phases"]):
        raise ValueError("Unresolved budget violation cannot be automatically released")
    if (state.get("status") not in ("DONE", "PARTIAL") or not state.get("ended_at_utc")
            or any(row.get("status") == "RUNNING" for row in ledger["phases"])):
        return None
    parse_time(state["ended_at_utc"])
    ledger_sha, state_sha = sha(ledger_path), sha(pipeline_state)
    if state.get("ledger_sha256_at_end") != ledger_sha:
        raise ValueError("Terminal controller state binds another ledger; do not export stage gaps")
    return {"ledger_sha256": ledger_sha, "pipeline_state_sha256": state_sha,
            "ledger": ledger, "pipeline_state": state}


def wait_for_closure(permit, clock=utc, sleep=time.sleep, on_poll=None):
    started = clock()
    seconds = (permit["_deadline"] - started).total_seconds()
    limit = min(permit["max_poll_iterations"], max(1, math.ceil(max(0, seconds)/POLL_SECONDS)+1))
    for iteration in range(1, limit+1):
        if clock() >= permit["_deadline"]:
            raise TimeoutError("Original V3 deadline reached; no budget or polling extension")
        snapshot = closed_pipeline_snapshot(permit["_run_root"], permit["_state_path"], permit["_ledger_path"])
        if on_poll is not None:
            on_poll(iteration, snapshot is not None)
        if snapshot is not None:
            if snapshot["pipeline_state"].get("plan_sha256") != permit["pipeline_plan_sha256"]:
                raise ValueError("Closed controller plan differs from root permit")
            return snapshot
        if iteration < limit:
            sleep(min(POLL_SECONDS, max(0, (permit["_deadline"]-clock()).total_seconds())))
    raise TimeoutError("Finite authorized polling iteration limit reached")


def build_release(permit, snapshot):
    return {"created_at_utc": utc().isoformat(), "allow_analysis_export": True,
        "cpu_budget_sha256": snapshot["ledger_sha256"],
        "pipeline_state_path": str(permit["_state_path"]),
        "pipeline_state_sha256": snapshot["pipeline_state_sha256"],
        "root_permit_path": str(permit["_permit_path"]), "root_permit_sha256": permit["_permit_sha256"],
        "final_analysis_source_sha256": permit["source_sha256"][str(ANALYZER_REL)],
        "finisher_source_sha256": permit["source_sha256"][str(FINISHER_REL)],
        "root_release_is_conditional_on_closed_controller_and_ledger": True,
        "scientific_results_are_not_created_by_this_release": True}


@contextmanager
def finalization_locks(permit):
    """Match supervisor lock order: V3 ledger, V3 global, then bound old V2."""
    # The ledger's actual authorization path is authoritative, not a guessed name.
    ledger = read(permit["_ledger_path"])
    authorization = read(resolve(ledger["authorization_path"]))
    locks = [permit["_ledger_path"].with_suffix(".lock"), ROOT / ".runtime/pyscf-v3-global.lock",
             resolve(authorization["prior_closed_ledger_path"]).with_suffix(".lock")]
    with ExitStack() as stack:
        for path in locks:
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def remaining_seconds(deadline, cap=300):
    remaining = (deadline - utc()).total_seconds()
    if remaining <= 0:
        raise TimeoutError("Original V3 deadline reached during finalization")
    return min(float(cap), remaining)


def candidate_delivery_files(run_root, doc_path, candidates):
    run_root, doc_path = resolve(run_root), resolve(doc_path)
    values, total, seen = [], 0, set()
    for candidate in sorted(map(Path, candidates), key=str):
        if candidate.is_symlink():
            raise ValueError("Do not deliver symlinks")
        path = resolve(candidate)
        if path in seen:
            continue
        seen.add(path)
        if (not path.is_relative_to(ROOT) or not (path.is_relative_to(run_root) or path == doc_path)
                or not path.is_file()):
            raise ValueError("Deleted, non-file or out-of-scope Git candidate")
        if any(parent.is_symlink() for parent in candidate.parents if parent != ROOT):
            raise ValueError("Do not traverse symlinked delivery directories")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError("Delivery file exceeds25MiB: " + str(path.relative_to(ROOT)))
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Total Git delivery exceeds150MiB")
        values.append({"path": str(path.relative_to(ROOT)), "bytes": size, "sha256": sha(path)})
    return values


def redact(value):
    return re.sub(r"(https?://)[^/\s@]+@", r"\1[REDACTED]@", value)


def git_output(arguments, deadline, executor=None):
    runner = executor or subprocess.run
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_TRACE") and key != "GIT_CURL_VERBOSE"}
    result = runner(["git", *arguments], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=remaining_seconds(deadline, 120), env=env)
    if result.returncode:
        raise RuntimeError("Git command failed: " + redact(result.stderr.decode(errors="replace")))
    return result.stdout


def branch_guard(deadline):
    branch = git_output(["branch", "--show-current"], deadline).decode().strip()
    if branch != BRANCH:
        raise RuntimeError("Research branch changed; stop Git delivery without modifying index")
    return branch


def publish_results(permit, deadline, delivery_callback=None):
    if permit.get("allow_git_push") is not True:
        raise ValueError("No root authorization for Git delivery")
    branch_guard(deadline)
    if git_output(["diff", "--cached", "--name-only", "-z"], deadline):
        raise RuntimeError("Existing staged files; stop Git without resetting or unstaging them")
    head = git_output(["rev-parse", "HEAD"], deadline).decode().strip()
    origin = git_output(["remote", "get-url", "origin"], deadline).decode().strip()
    if not re.fullmatch(r"(?:https://(?:[^/@]+@)?github\.com/|git@github\.com:)jackzhu119/research4(?:\.git)?/?", origin):
        raise RuntimeError("Origin is not the explicitly authorized research4 repository")
    scopes = [str(permit["_run_root"].relative_to(ROOT)), str(permit["_document_path"].relative_to(ROOT))]
    paths = []
    for command in (["ls-files", "--others", "--exclude-standard", "-z", "--", *scopes],
                    ["diff", "--name-only", "-z", "--", *scopes]):
        paths.extend(ROOT / os.fsdecode(raw) for raw in git_output(command, deadline).split(b"\0") if raw)
    files = candidate_delivery_files(permit["_run_root"], permit["_document_path"], paths)
    if not files:
        return {"status": "NO_NEW_SCOPED_FILES", "push_attempted": False}
    branch_guard(deadline)
    if git_output(["diff", "--cached", "--name-only", "-z"], deadline):
        raise RuntimeError("Concurrent staged files detected before Git add")
    names = [entry["path"] for entry in files]
    git_output(["add", "--", *names], deadline)
    staged = {os.fsdecode(raw) for raw in git_output(["diff", "--cached", "--name-only", "-z"], deadline).split(b"\0") if raw}
    if staged != set(names):
        raise RuntimeError("Concurrent staged changes; stop without resetting the shared index")
    branch_guard(deadline)
    if git_output(["rev-parse", "HEAD"], deadline).decode().strip() != head:
        raise RuntimeError("HEAD changed during delivery; stop without rewriting history")
    audited_blobs = {}
    for entry in files:
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Closed result bytes changed before commit")
        work_blob = git_output(["hash-object", "--", entry["path"]], deadline).strip()
        index_blob = git_output(["rev-parse", ":"+entry["path"]], deadline).strip()
        if work_blob != index_blob:
            raise RuntimeError("Index differs from audited closed result bytes")
        audited_blobs[entry["path"]] = work_blob
    # --only keeps a concurrent unrelated staged path out of our commit even if
    # it arrives in the last instant after the shared-index checks. Never reset
    # or unstage it; the post-commit guard will stop before pushing.
    git_output(["-c", "core.hooksPath=/dev/null", "commit", "--only", "-m", "research: finalize monitored Spinosin DFT evidence", "--", *names], deadline)
    commit = git_output(["rev-parse", "HEAD"], deadline).decode().strip()
    result = {"status": "COMMITTED_NOT_YET_PUSHED", "commit_sha": commit, "branch": BRANCH,
              "files": files, "push_attempted": False, "force_push": False, "main_modified": False}
    if delivery_callback:
        delivery_callback(result)
    committed_names = {os.fsdecode(raw) for raw in git_output(
        ["diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit], deadline).split(b"\0") if raw}
    if committed_names != set(names):
        raise RuntimeError("Actual commit scope differs from closed-file manifest; stop before push")
    for name, expected_blob in audited_blobs.items():
        if git_output(["rev-parse", commit+":"+name], deadline).strip() != expected_blob:
            raise RuntimeError("Actual commit bytes differ from audited closed result; stop before push")
    branch_guard(deadline)
    if git_output(["diff", "--cached", "--name-only", "-z"], deadline):
        raise RuntimeError("Unexpected staged files after commit; stop before push")
    if git_output(["rev-parse", "HEAD"], deadline).decode().strip() != commit:
        raise RuntimeError("HEAD changed before push; stop delivery")
    result["push_attempted"] = True
    if delivery_callback:
        delivery_callback(result)
    output = git_output(["push", "--porcelain", "origin", commit+":refs/heads/"+BRANCH], deadline)
    result.update(status="PUSHED", push_output=redact(output.decode(errors="replace")), completed_at_utc=utc().isoformat())
    if delivery_callback:
        delivery_callback(result)
    return result


def write_scientific_document(path, summary, snapshot, release):
    def value(key):
        if key not in summary:
            raise ValueError("Actual analyzer summary lacks required evidence: " + key)
        return summary[key]
    if (value("analysis_only") is not True or value("new_SCF_gradient_Hessian_evaluations") != 0
            or value("IR_intensities_status") != "NOT_COMPUTED"
            or value("chemical_attribution_confirmed") is not False
            or value("classification_modified") is not False):
        raise ValueError("Actual analyzer scope differs from the root-permitted evidence-only report")
    rows = [("V3实际CPU秒", value("V3_actual_quantum_CPU_seconds")),
        ("绑定V2实际CPU秒", value("bound_prior_V2_actual_quantum_CPU_seconds")),
        ("斯皮诺素配对几何／梯度点", value("Spinosin_paired_geometry_evaluations")),
        ("实际计算的斯皮诺素频率记录（含未验证及负频）", value("computed_Spinosin_frequency_rows_including_unvalidated")),
        ("通过完整检查的斯皮诺素模式记录", value("validated_spinosin_frequency_rows")),
        ("通过检查且位于0.1—1.5THz的模式", value("validated_spinosin_low_band_modes")),
        ("失败／部分完成阶段", value("failed_or_partial_phase_count")),
        ("已结束但没有worker结果文件的阶段", value("ended_phases_without_worker_result"))]
    text = "# 斯皮诺素受监控DFT批次：实际结束结果\n\n"
    text += "本报告由已关闭控制器、最终CPU账本及保存计算文件的独立复核生成；没有启动新的SCF、梯度或Hessian计算。\n\n"
    text += f"控制器最终状态：`{snapshot['pipeline_state']['status']}`。斯皮诺素频率状态：`{value('Spinosin_frequency_status')}`。\n\n"
    text += "| 实际核查项目 | 结果 |\n|---|---:|\n" + "".join(f"| {label} | {result} |\n" for label, result in rows)
    text += "\n未计算、未收敛或尚未通过完整正频检查的结果均保留其实际状态；空频率表不代表零吸收。水和乙醇仅作软件／数值验证，不属于斯皮诺素机制证据。负频及缺少worker结果的失败证据保存在独立分析目录，未补造计算结果。\n\n"
    text += "仅通过同方法父优化、fresh梯度、完整219个正频及原始Hessian复核的孤立分子模式，才进入实验候选峰比较。孤立分子与固态药材中的晶格、氢键和集体振动不同；频率接近不确认化学归属。IR强度为`NOT_COMPUTED`，不绘制虚构吸收强度。\n\n"
    text += "本批次不修改酸枣仁光谱、标签、分类器或既有96.0234%分类结果，也不声称分子计算提高了分类准确率。RSS轮询属于尽力监测，24GiB地址空间不等于24GiB物理内存。\n\n"
    text += "实际证据：[分析汇总](../"+str((RUN_REL/"final-analysis/summary.json"))+")、[阶段成本与状态](../"+str((RUN_REL/"final-analysis/phase_status_and_resources.csv"))+")、[失败证据](../"+str((RUN_REL/"final-analysis/actual_failure_evidence.json"))+")、[理论与实验候选位置对照](../"+str((RUN_REL/"final-analysis/experiment_theory_pairs.csv"))+")。\n\n"
    text += f"最终账本SHA-256：`{release['cpu_budget_sha256']}`。闭合控制器状态SHA-256：`{release['pipeline_state_sha256']}`。\n\n"
    text += f"root条件授权SHA-256：`{release['root_permit_sha256']}`。分析源码SHA-256：`{release['final_analysis_source_sha256']}`。\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def run(args, analysis_executor=None):
    permit = validate_permit(args.permit, args.permit_sha256, push=args.push)
    delivery = resolve(args.delivery_dir)
    if not delivery.is_relative_to((ROOT/"outputs").resolve()) or delivery.exists():
        raise ValueError("Use one fresh ignored outputs/ delivery directory; do not retry or overwrite")
    delivery.mkdir(parents=True)
    result = {"status": "WAITING_FOR_CLOSED_PIPELINE", "started_at_utc": utc().isoformat(),
              "root_permit_sha256": permit["_permit_sha256"], "quantum_jobs_started": 0,
              "analysis_attempts": 0, "Git_delivery": {"status": "NOT_REQUESTED"}}
    def event(kind, data):
        with (delivery/"events.jsonl").open("a") as handle:
            handle.write(json.dumps({"utc": utc().isoformat(), "event": kind, **data}, ensure_ascii=False)+"\n")
    try:
        snapshot = wait_for_closure(permit, on_poll=lambda i, closed: event("poll", {"iteration": i, "closed": closed}))
        with finalization_locks(permit):
            validate_permit(args.permit, args.permit_sha256, push=args.push)
            current = closed_pipeline_snapshot(permit["_run_root"], permit["_state_path"], permit["_ledger_path"])
            if current is None or any(current[key] != snapshot[key] for key in ("ledger_sha256", "pipeline_state_sha256")):
                raise ValueError("Closed evidence changed before finalization lock")
            spec = importlib.util.spec_from_file_location("finisher_official_budget_validation", ROOT/SUPERVISOR_REL)
            budget = importlib.util.module_from_spec(spec); spec.loader.exec_module(budget)
            budget.validate_ledger(snapshot["ledger"])
            release = build_release(permit, snapshot)
            release_path = permit["_run_root"] / "final_analysis_release.json"
            if permit["_analysis_output"].exists() or permit["_document_path"].exists():
                raise FileExistsError("Preserve existing final analysis/document; no automatic retry")
            write_new(release_path, release)
            command = [str(ROOT/".runtime/bin/python"), str(ROOT/ANALYZER_REL), "--run-root", str(permit["_run_root"]),
                       "--output", str(permit["_analysis_output"]), "--release", str(release_path)]
            result["analysis_attempts"] = 1
            event("analysis_requested_once", {"command": command, "release_sha256": sha(release_path)})
            executor = analysis_executor or subprocess.run
            completed = executor(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=remaining_seconds(permit["_deadline"], 600))
            (delivery/"analysis.log").write_bytes(completed.stdout)
            if completed.returncode:
                raise RuntimeError("Frozen final analyzer failed; no automatic retry")
            validate_permit(args.permit, args.permit_sha256, push=args.push)
            if sha(permit["_ledger_path"]) != snapshot["ledger_sha256"] or sha(permit["_state_path"]) != snapshot["pipeline_state_sha256"]:
                raise ValueError("Closed source evidence changed during final analysis")
            summary = read(permit["_analysis_output"] / "summary.json")
            remaining_seconds(permit["_deadline"])
            write_scientific_document(permit["_document_path"], summary, snapshot, release)
            result.update(status="SCIENTIFIC_REPORT_DONE", final_ledger_sha256=snapshot["ledger_sha256"],
                closed_pipeline_state_sha256=snapshot["pipeline_state_sha256"],
                analysis_source_sha256=permit["source_sha256"][str(ANALYZER_REL)],
                summary_path=str(permit["_analysis_output"]/"summary.json"),
                document_path=str(permit["_document_path"]), document_sha256=sha(permit["_document_path"]))
            if args.push:
                try:
                    def delivery_status(status):
                        result["Git_delivery"] = dict(status)
                        event("actual_Git_delivery", status)
                    result["Git_delivery"] = publish_results(permit, permit["_deadline"],
                        delivery_callback=delivery_status)
                except Exception as exc:
                    result["Git_delivery"] = {**result["Git_delivery"], "status": "BLOCKED_OR_FAILED_PRESERVE_SHARED_INDEX", "reason": redact(str(exc))}
                    result["status"] = "SCIENTIFIC_REPORT_DONE_GIT_BLOCKED_OR_FAILED"
    except Exception as exc:
        result.update(status="PARTIAL_FINALIZATION", exception_type=type(exc).__name__, stop_reason=redact(str(exc)))
    finally:
        result["ended_at_utc"] = utc().isoformat()
        write_new(delivery/"delivery.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permit", required=True, type=Path)
    parser.add_argument("--permit-sha256", required=True)
    parser.add_argument("--delivery-dir", required=True, type=Path)
    parser.add_argument("--push", action="store_true", help="Requires separate explicit root permit flag")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "SCIENTIFIC_REPORT_DONE" else 2)


if __name__ == "__main__":
    main()
