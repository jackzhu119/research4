"""Finite serial controller for reviewed, actual-CPU-supervised RHF segments.

No quantum package is imported and no calculation starts on import. The run
subcommand executes only the frozen V5 supervisor, one payload at a time. Each
healthy closed segment is sealed with its small immutable RHF checkpoints.
The controller does not commit/push Git, extend a ledger, or repair failed rows.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"
SUPERVISOR_SHA = "33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee"
WORKER = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
CORE_SHA = "ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55"
SCHEMA = "research4-finite-serial-segment-controller-v1"
STOP_FLAGS = ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed")


def utc():
    return datetime.now(timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    temporary = path.with_name("."+path.name+".writing-"+uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if immutable:
        os.link(temporary, path)
        temporary.unlink()
    else:
        os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_verified_worker(expected_sha):
    if sha(SUPERVISOR) != SUPERVISOR_SHA or sha(WORKER) != expected_sha:
        raise ValueError("Frozen supervisor/consumer source SHA differs")
    spec = importlib.util.spec_from_file_location("research4_verified_consumer_for_serial_controller", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.INCREMENTAL_SHA != CORE_SHA:
        raise ValueError("Consumer does not bind the reviewed incremental core")
    return module


def verify_artifact(record):
    path = Path(record["path"]).resolve()
    if not record.get("sha256") or sha(path) != record["sha256"]:
        raise ValueError("Gate evidence artifact SHA differs: "+str(path))
    return path


def verify_gate(path, expected_sha, worker_sha, state_sha):
    path = verify_artifact({"path": str(path), "sha256": expected_sha})
    gate = read(path)
    if (gate.get("schema") != "research4-resumable-hessian-gate-v1" or gate.get("status") != "PASS"
            or gate.get("approved_molecule") != "Spinosin" or gate.get("incremental_module_sha256") != CORE_SHA
            or gate.get("consumer_worker_sha256") != worker_sha or gate.get("state_worker_sha256") != state_sha):
        raise ValueError("Explicit root gate does not qualify these exact sources/Spinosin")
    for name, count in (("water", 3), ("ethanol", 21)):
        result = gate["tests"][name]
        errors = [result.get("full_hessian_max_abs_diff"), result.get("all_modes_max_abs_diff_cm1")]
        if (result.get("status") != "PASS" or result.get("actual_separate_process_resume") is not True
                or result.get("same_state_gradient_verified") is not True or result.get("normal_mode_count") != count
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in errors)
                or errors[0] > 1e-6 or errors[1] > .01 or not result.get("artifacts")):
            raise ValueError("Missing actual full-Hessian/all-mode cross-process gate: "+name)
        for artifact in result["artifacts"]:
            verify_artifact(artifact)
    review = gate["independent_review"]
    if review.get("status") != "PASS":
        raise ValueError("Independent implementation review is required")
    verify_artifact(review["report"])
    return gate


def available_budget(ledger, *, requested_cpu, requested_wall, now=None):
    """Read only a closed V5 ledger; never mutate or increase its budget."""
    if ledger.get("schema") != "research4-pyscf-actual-cpu-budget-v5" or ledger.get("runner_sha256") != SUPERVISOR_SHA:
        raise ValueError("Expected unchanged actual-CPU V5 ledger")
    phases = ledger["phases"]
    if any(p.get("status") == "RUNNING" or p.get("status") == "BUDGET_VIOLATION_REQUIRES_REVIEW"
           or p.get("residual_processes_require_review_after_main_payload_exit") for p in phases):
        raise ValueError("Unresolved/running/review-required phase; no parallel or automatic retry")
    costs = [p.get("actual_cpu_seconds") for p in phases]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in costs):
        raise ValueError("Unknown completed CPU cost")
    if not math.isclose(sum(costs), ledger["consumed_cpu_seconds"], abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError("Ledger actual CPU sum differs")
    deadline = datetime.fromisoformat(ledger["run_deadline_at_utc"])
    if deadline.tzinfo is None:
        raise ValueError("Explicit timezone required on finite deadline")
    cpu = min(requested_cpu, math.floor(ledger["quantum_cpu_cap_seconds"]-sum(costs))-10)
    wall = min(requested_wall, math.floor((deadline-(now or utc())).total_seconds())-2)
    return None if cpu < 125 or wall < 61 else {"cpu_seconds": cpu, "wall_seconds": wall}


def checkpoint_records(cache):
    """Inspect completion JSON only; caller first verifies closed producers."""
    cache = Path(cache)
    if not cache.exists() or not (cache / "header.json").exists():
        if cache.exists() and any(cache.iterdir()):
            raise ValueError("Nonempty cache without immutable header")
        return []
    header = read(cache / "header.json")
    if (header.get("schema") != "research4-incremental-pure-pbe-rhf-v10"
            or header.get("natm") != 75 or header.get("total_units") != 150):
        raise ValueError("Controller is restricted to the complete 75-atom Spinosin state")
    paths = sorted((cache / "units").glob("*/attempt-*/complete.json"))
    values = []
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 131072:
            raise ValueError("Unsafe completion record")
        record = read(path)
        index = record.get("index")
        if type(index) is not int or not 0 <= index < 150:
            raise ValueError("Invalid completion index")
        kind, atom = ("diag", index) if index < 75 else ("row", index-75)
        if path.parent.parent.name != f"{index:06d}-{kind}-{atom:04d}":
            raise ValueError("Completion index/path differ")
        if (record.get("header_sha256") != sha(cache / "header.json")
                or record.get("binding_sha256") != header["binding_sha256"]):
            raise ValueError("Completion header/state binding differs")
        values.append((index, path, record))
    values.sort(key=lambda v: v[0])
    if [v[0] for v in values] != list(range(len(values))):
        raise ValueError("Duplicate, hole or noncontinuous checkpoint prefix")
    return values


def _copy_immutable(source, target, expected_sha):
    source, target = Path(source), Path(target)
    if source.is_symlink() or source.stat().st_size > 2**20 or sha(source) != expected_sha:
        raise ValueError("Unsafe/changed checkpoint artifact: "+str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or sha(target) != expected_sha:
            raise ValueError("Archive conflicts with an earlier immutable artifact")
        return
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Checkpoint changed during closed-artifact read")
    with target.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    directory = target.parent
    while directory.is_relative_to(ROOT) and directory != ROOT:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = directory.parent


def archive_checkpoint(cache, archive, phase, dependencies, *, verify_closed, expected_units):
    with _closed_cache_guard(cache):
        producer_checks = verify_closed()
        if len(checkpoint_records(cache)) != expected_units:
            raise ValueError("Cache prefix changed before its locked closed-producer archive")
        return _archive_checkpoint_unlocked(cache, archive, phase, dependencies, producer_checks)


def _archive_checkpoint_unlocked(cache, archive, phase, dependencies, producer_checks):
    """Called only after the V5 phase is healthy/closed and producer-qualified."""
    cache, archive = Path(cache).resolve(), Path(archive).resolve()
    files = {"header.json": sha(cache / "header.json")}
    records = checkpoint_records(cache)
    for _, path, record in records:
        for name, entry in record["arrays"].items():
            if name not in ("jdiag", "e1", "ej") or entry.get("file") != name+".npy":
                raise ValueError("Unexpected checkpoint payload filename")
            payload = path.parent / entry["file"]
            files[str(payload.relative_to(cache))] = entry["file_sha256"]
        files[str(path.relative_to(cache))] = sha(path)
    # Completion records are copied last so an interrupted archive is not a seal.
    for name in sorted(files, key=lambda n: n.endswith("complete.json")):
        _copy_immutable(cache / name, archive / name, files[name])
    seal = {"schema": "research4-closed-rhf-checkpoint-archive-v1", "phase": phase,
            "created_at_utc": utc().isoformat(), "repository_root": str(ROOT),
            "original_cache_directory": str(cache), "archive_directory": str(archive),
            "completed_units": len(records), "total_units": 150, "files": files,
            "dependencies": dependencies, "active_arrays_read": False, "producer_checks": producer_checks,
            "scope": "QUALIFIED_COMPLETED_RHF_INTERMEDIATES_NOT_A_COMPLETE_HESSIAN"}
    seals = archive / "seals"
    seals.mkdir(parents=True, exist_ok=True)
    seal_path = seals / (phase+".json")
    atomic_json(seal_path, seal, immutable=True)
    result = {"path": str(seal_path), "sha256": sha(seal_path), "completed_units": len(records)}
    result["restore_command"] = shlex.join([str(ROOT / ".runtime/bin/python"), str(Path(__file__).resolve()),
          "restore-cache", "--seal", str(seal_path), "--seal-sha256", result["sha256"], "--cache-dir", str(cache)])
    return result


@contextmanager
def _closed_cache_guard(target):
    """Archive/restore only while no real V5 payload or helper owns the cache."""
    global_path = ROOT / ".runtime/pyscf-v4-global.lock"
    global_path.parent.mkdir(exist_ok=True)
    with global_path.open("a") as global_handle:
        fcntl.flock(global_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path(target).mkdir(parents=True, exist_ok=True)
        with (Path(target) / "cache.lock").open("a") as cache_handle:
            fcntl.flock(cache_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def restore_cache(seal_path, expected_sha, target):
    target = Path(target).resolve()
    if not target.is_relative_to(ROOT / "outputs") or target == ROOT / "outputs":
        raise ValueError("Dedicated ignored cache path required")
    with _closed_cache_guard(target):
        return _restore_cache_unlocked(seal_path, expected_sha, target)


def _restore_cache_unlocked(seal_path, expected_sha, target):
    seal_path = verify_artifact({"path": str(seal_path), "sha256": expected_sha})
    seal = read(seal_path)
    target = Path(target).resolve()
    if (seal.get("schema") != "research4-closed-rhf-checkpoint-archive-v1"
            or str(target) != seal.get("original_cache_directory")
            or not target.is_relative_to(ROOT / "outputs") or target == ROOT / "outputs"):
        raise ValueError("Restore requires the exact original dedicated ignored-cache path")
    archive = Path(seal["archive_directory"]).resolve()
    if seal_path.parent.parent != archive:
        raise ValueError("Seal/archive location changed; review absolute source bindings")
    for artifact in seal["dependencies"]:
        verify_artifact(artifact)
    for name in seal["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name == "cache.lock":
            raise ValueError("Unsafe archive-relative path")
    target.mkdir(parents=True, exist_ok=True)
    for name in sorted(seal["files"], key=lambda n: n.endswith("complete.json")):
        _copy_immutable(archive / name, target / name, seal["files"][name])
    count = len(checkpoint_records(target))
    if count != seal["completed_units"]:
        raise ValueError("Restored continuous prefix count differs")
    return {"status": "RESTORED_CLOSED_RHF_PREFIX_NOT_NEW_QM", "completed_units": count,
            "cache_directory": str(target), "seal_sha256": expected_sha}


def run_supervisor(command, console_path, emit):
    """One finite V5 child; signals are relayed for V5's own cleanup/accounting."""
    maximum_wall = int(command[command.index("--wall-seconds")+1])
    with Path(console_path).open("x") as console:
        process = subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+maximum_wall+2
        next_heartbeat = 0.
        requested = {"signal": None}
        def relay(signum, frame):
            requested["signal"] = signum
            if process.poll() is None:
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass
        previous = {s: signal.signal(s, relay) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    # V5 normally ends within its own finite wall limit. If its
                    # monitor freezes, never wait forever or launch another QM.
                    # Its frozen child-limits contract arms parent-death SIGKILL.
                    emit("SUPERVISOR_WALL_WATCHDOG", supervisor_pid=process.pid)
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    return process.returncode, "CONTROLLER_SUPERVISOR_WALL_WATCHDOG"
                if now >= next_heartbeat:
                    emit("SUPERVISOR_RUNNING", supervisor_pid=process.pid, stop_signal=requested["signal"])
                    next_heartbeat = now+15
                time.sleep(min(1., max(.01, deadline-now)))
            return process.returncode, requested["signal"]
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def continuation_command(args):
    namespace = vars(args).copy()
    namespace["output"] = Path(str(args.output)+"-continue-"+utc().strftime("%Y%m%dT%H%M%SZ"))
    namespace.pop("dry_run", None)
    command = [str(args.python), str(Path(__file__).resolve()), "run"]
    for key, value in namespace.items():
        if key in ("action", "python") or value is None:
            continue
        command.extend(["--"+key.replace("_", "-"), str(value)])
    return shlex.join(command)


def execute(args, *, worker_loader=load_verified_worker, phase_runner=run_supervisor):
    if not 1 <= args.max_segments <= 17 or args.phase_cpu_seconds < 125 or args.phase_wall_seconds < 61:
        raise ValueError("At most 17 finite segments; safe CPU/wall minimums required")
    worker = worker_loader(args.worker_sha256)
    gate = verify_gate(args.gate_report, args.gate_report_sha256, args.worker_sha256, worker.STATE_SHA)
    verify_artifact({"path": str(args.snapshot / "manifest.json"), "sha256": args.snapshot_manifest_sha256})
    worker.qualify_gradient_receipt(args.gradient_check_result, args.gradient_check_result_sha256,
                                   args.snapshot_manifest_sha256, worker.boot_id())
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "reports"):
        raise ValueError("A fresh report directory under reports/ is required")
    cache = args.rhf_cache.resolve()
    if not cache.is_relative_to(ROOT / "outputs") or cache == ROOT / "outputs":
        raise ValueError("A dedicated ignored outputs/ RHF cache is required")
    if not args.phase_prefix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Simple unique phase prefix required")
    # An explicit root release can qualify preserved units from a reviewed old
    # failure. It never turns that old row into EXITED/0 or releases a new failure.
    forensic = worker.load_forensic_release(args.forensic_unit_release, args.forensic_unit_release_sha256,
                                           args.snapshot_manifest_sha256)
    def qualify_prefix():
        return worker.verify_cached_unit_producers(cache, args.snapshot_manifest_sha256, forensic_release=forensic)
    # Producer qualification precedes even JSON prefix inspection.
    qualify_prefix()
    count = len(checkpoint_records(cache))
    output.mkdir(parents=True)
    state = {"schema": SCHEMA, "status": "READY", "started_at_utc": utc().isoformat(),
             "source_sha256": sha(__file__), "worker_sha256": args.worker_sha256,
             "gate_report": {"path": str(args.gate_report.resolve()), "sha256": args.gate_report_sha256},
             "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "cache_directory": str(cache),
             "initial_completed_units": count, "completed_units": count, "total_units": 150,
             "max_segments": args.max_segments, "max_new_units_per_segment": 10, "segments": [],
             "recovery_command": continuation_command(args), "automatic_Git": False}
    def emit(event, **values):
        state.update(updated_at_utc=utc().isoformat(), event=event, **values)
        atomic_json(output / "progress.json", state)
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"utc": utc().isoformat(), "event": event, **values}, allow_nan=False)+"\n")
        print(json.dumps({"utc": utc().isoformat(), "event": event, **values}, allow_nan=False), flush=True)
    try:
        for ordinal in range(1, args.max_segments+1):
            ledger = read(args.ledger)
            limits = available_budget(ledger, requested_cpu=args.phase_cpu_seconds, requested_wall=args.phase_wall_seconds)
            if limits is None:
                state.update(status="PARTIAL", stop_reason="INSUFFICIENT_REMAINING_FINITE_CPU_OR_WALL_BUDGET")
                break
            existing = {p["phase"] for p in ledger["phases"]}
            serial = 1
            while f"{args.phase_prefix}-segment-{serial:03d}" in existing:
                serial += 1
            phase = f"{args.phase_prefix}-segment-{serial:03d}"
            segment = output / phase
            log = output / (phase+".supervisor-payload.log")
            command = [str(args.python), str(SUPERVISOR), "--ledger", str(args.ledger.resolve()),
                       "--phase", phase, "--cpu-seconds", str(limits["cpu_seconds"]),
                       "--wall-seconds", str(limits["wall_seconds"]), "--threads", "4", "--memory-gib", "24",
                       "--log", str(log), "--", str(args.python), str(WORKER), "--phase", "hessian-segment",
                       "--snapshot", str(args.snapshot.resolve()), "--snapshot-manifest-sha256", args.snapshot_manifest_sha256,
                       "--output", str(segment), "--threads", "4", "--max-new-units", "10", "--rhf-cache", str(cache),
                       "--gradient-check-result", str(args.gradient_check_result.resolve()),
                       "--gradient-check-result-sha256", args.gradient_check_result_sha256]
            if args.forensic_unit_release is not None:
                command.extend(["--forensic-unit-release", str(args.forensic_unit_release.resolve()),
                                "--forensic-unit-release-sha256", args.forensic_unit_release_sha256])
            emit("SEGMENT_READY", phase=phase, limits=limits, completed_units=count, command=command)
            if args.dry_run:
                state.update(status="NOT_RUN", stop_reason="EXPLICIT_DRY_RUN_NO_QM_LAUNCHED")
                break
            returncode, stop_signal = phase_runner(command, output / (phase+".supervisor-console.log"), emit)
            rows = [p for p in read(args.ledger)["phases"] if p.get("phase") == phase]
            if returncode != 0 or len(rows) != 1 or not worker.healthy_closed_row(rows[0]):
                raise RuntimeError("Supervisor did not close as healthy EXITED/0; no automatic retry")
            result_path = segment / "result.json"
            result = read(result_path)
            evidence = {"schema": worker.EXECUTION_SCHEMA, "source_worker": result["source_worker"],
                        "resource_supervision": result["resource_supervision"], "output_directory": str(segment),
                        "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "phase": result["phase"]}
            qualified = worker.verify_closed_execution(evidence, expected_snapshot_sha=args.snapshot_manifest_sha256,
                                                       expected_output=segment)
            qualify_prefix()
            after = len(checkpoint_records(cache))
            progress = result["incremental_after"]
            if progress.get("total_units_completed") != after or progress.get("total_units") != 150:
                raise RuntimeError("Result progress differs from durable continuous prefix")
            if not count <= after <= min(count+10, 150):
                raise RuntimeError("A segment exceeded its fixed 10-unit scheduling limit")
            status = result["status"]
            if status == "CHECKPOINT_SEGMENT_COMPLETE":
                if not count < after <= min(count+10, 150) or after == 150 or result.get("frequencies_status") != "NOT_RUN":
                    raise RuntimeError("Segment has no bounded progress or falsely claims scientific completion")
            elif status in ("FREQUENCIES_COMPUTED", "DONE"):
                frequencies = result.get("frequencies_cm1", [])
                if (after != 150 or result.get("frequencies_status") != "DONE" or result.get("normal_mode_count") != 219
                        or result.get("complete_global_Hessian") is not True or len(frequencies) != 219
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in frequencies)):
                    raise RuntimeError("Terminal result lacks all Spinosin RHF units and 219 actual modes")
            else:
                raise RuntimeError("Unexpected scientific partial/failure; do not automatically continue")
            dependencies = [{"path": str(args.snapshot.resolve()/"manifest.json"), "sha256": args.snapshot_manifest_sha256},
                            {"path": str(args.gradient_check_result.resolve()), "sha256": args.gradient_check_result_sha256},
                            {"path": str(result_path), "sha256": sha(result_path)},
                            {"path": str(args.gate_report.resolve()), "sha256": args.gate_report_sha256},
                            {"path": str(args.ledger.resolve()), "sha256": sha(args.ledger)}]
            # Ledger is mutable across subsequent segments, so bind this phase's
            # immutable closed row separately rather than require an old ledger SHA.
            closed_row = output / (phase+".closed-ledger-row.json")
            atomic_json(closed_row, rows[0], immutable=True)
            dependencies[-1] = {"path": str(closed_row), "sha256": sha(closed_row)}
            if args.forensic_unit_release is not None:
                dependencies.append({"path": str(args.forensic_unit_release.resolve()),
                                     "sha256": args.forensic_unit_release_sha256})
            seal = archive_checkpoint(cache, output / "checkpoint-archive", phase, dependencies,
                 verify_closed=qualify_prefix,
                 expected_units=after)
            state["segments"].append({"phase": phase, "status": status, "before_units": count, "after_units": after,
                                      "closed_producer": qualified, "archive_seal": seal})
            count = after
            emit("SEGMENT_CLOSED_AND_ARCHIVED", phase=phase, completed_units=count, archive_seal=seal)
            if status in ("FREQUENCIES_COMPUTED", "DONE"):
                state.update(status="DONE", final_result={"path": str(result_path), "sha256": sha(result_path)},
                             frequency_validation_passed=result.get("frequency_validation_passed"),
                             imaginary_frequency_count=result.get("imaginary_frequency_count"))
                break
            if stop_signal is not None:
                state.update(status="PARTIAL", stop_reason="CONTROLLER_STOP_SIGNAL_AFTER_CLOSED_SEGMENT")
                break
        else:
            state.update(status="PARTIAL", stop_reason="FINITE_SEGMENT_COUNT_EXHAUSTED")
    except BaseException as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), exception_type=type(exc).__name__)
    state.update(completed_units=count, ended_at_utc=utc().isoformat())
    emit("CONTROLLER_STOPPED", status=state["status"], completed_units=count, stop_reason=state.get("stop_reason"))
    atomic_json(output / "result.json", state, immutable=True)
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    for name in ("ledger", "snapshot", "gradient-check-result", "gate-report", "output", "rhf-cache"):
        run.add_argument("--"+name, type=Path, required=True)
    for name in ("snapshot-manifest-sha256", "gradient-check-result-sha256", "gate-report-sha256", "worker-sha256"):
        run.add_argument("--"+name, required=True)
    run.add_argument("--python", type=Path, default=ROOT / ".runtime/bin/python")
    run.add_argument("--phase-prefix", default="spinosin-resumable")
    run.add_argument("--max-segments", type=int, default=17)
    run.add_argument("--phase-cpu-seconds", type=int, default=7200)
    run.add_argument("--phase-wall-seconds", type=int, default=3600)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--forensic-unit-release", type=Path)
    run.add_argument("--forensic-unit-release-sha256")
    restore = commands.add_parser("restore-cache")
    restore.add_argument("--seal", type=Path, required=True)
    restore.add_argument("--seal-sha256", required=True)
    restore.add_argument("--cache-dir", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if args.action == "restore-cache":
        print(json.dumps(restore_cache(args.seal, args.seal_sha256, args.cache_dir), indent=2))
    else:
        result = execute(args)
        raise SystemExit(0 if result["status"] in ("DONE", "NOT_RUN") else 2)


if __name__ == "__main__":
    main()
