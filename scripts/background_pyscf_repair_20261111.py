#!/usr/bin/env python3
"""Detach and supervise the PySCF repair queue, then verify and publish it.

No score, elapsed-time, CPU or retry-count limit stops the computation. Scientific
quality is reported by the runner rather than guessed here. Exit 42 means an
input/code integrity mismatch and requires investigation. Detachment survives
the invoking shell, not suspension or destruction of the cloud instance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/pyscf_repair_20261111"
RUNTIME = OUTPUT / ".background"
STATE = RUNTIME / "state.json"
RECEIPT = OUTPUT / "background_finalization.json"
BRANCH = "research/pyscf-frequency-repair-20261111"
BASE = "research/b-mechanism-round5-20261111"
PYTHON = ROOT / ".venv-pyscf/bin/python"
RUNNER = ROOT / "scripts/run_pyscf_repair_20261111.py"
INTEGRITY_ERROR = 42
SCIENTIFIC_SCRIPTS = (
    "run_pyscf_repair_20261111.py",
    "pyscf_repair_core_20261111.py",
    "pyscf_repair_modes_20261111.py",
)
REQUIREMENTS = ROOT / "requirements-pyscf-repair-20261111.txt"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_message(value):
    message = str(value)
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        secret = os.environ.get(name)
        if secret and len(secret) >= 8:
            message = message.replace(secret, "[REDACTED]")
    return message[-4000:]


def git(*arguments, input_text=None):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, input=input_text,
        capture_output=True, check=False,
    )


def require_branch():
    result = git("branch", "--show-current")
    if result.returncode or result.stdout.strip() != BRANCH:
        raise ValueError("Integrity error: checkout must remain on " + BRANCH)


def scientific_snapshot(required=True):
    """Authenticate code and model inputs without freezing evolving results."""
    paths = [ROOT / "scripts" / name for name in SCIENTIFIC_SCRIPTS]
    for path in paths:
        if required and not path.is_file():
            raise ValueError("Integrity error: missing scientific script " + str(path))
    paths = [path for path in paths if path.is_file()]
    for folder in ("inputs", "history_controls"):
        paths += [p for p in (ROOT / "data/research/pyscf_repair_20261111" / folder).rglob("*")
                  if p.is_file() and not p.is_symlink()]
    if required and not any("/inputs/" in str(path) for path in paths):
        raise ValueError("Integrity error: authenticated molecular inputs are missing")
    for path in (OUTPUT / "frozen_protocol.json", OUTPUT / "protocol.json", REQUIREMENTS):
        if path.is_file():
            paths.append(path)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"scientific_input_sha256": digest, "scientific_files": hashes}


def check_snapshot(state):
    current = scientific_snapshot()
    if current != {key: state.get(key) for key in current}:
        raise ValueError("Integrity error: frozen molecular inputs or scientific code changed")
    if not PYTHON.is_file():
        raise ValueError("Integrity error: prepared .venv-pyscf Python is missing")
    return current


def eligible_files():
    """Only this repair's source, inputs and inspectable result artifacts."""
    paths = list((ROOT / "scripts").glob("*pyscf_repair*20261111.py"))
    paths += list((ROOT / "docs").glob("PYSCF_*_20261111.md"))
    paths += [ROOT / "tests/test_pyscf_repair_20261111.py", REQUIREMENTS]
    for folder in ("inputs", "history_controls"):
        paths += list((ROOT / "data/research/pyscf_repair_20261111" / folder).rglob("*"))
    paths += list(OUTPUT.rglob("*"))
    selected = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if path == OUTPUT / ".gitignore":
            selected.append(relative.as_posix())
            continue
        if any(part.startswith(".") or part in {
            "local-only", "local_only", "checkpoints", "checkpoint", "__pycache__", "cache",
        } for part in relative.parts):
            continue
        if path.name.endswith((".log", ".lock", ".pid", ".tmp", ".partial", ".pyc", ".chk", ".h5")):
            continue
        selected.append(relative.as_posix())
    return sorted(set(selected))


def process_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return stat[0] != "Z"
    except (ValueError, TypeError, OSError):
        return False


def matching_process(pid, terms):
    if not process_alive(pid):
        return False
    try:
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return all(str(term).encode() in command for term in terms)
    except OSError:
        return False


def supervisor_alive(state):
    return bool(state and matching_process(state.get("supervisor_pid"),
                [Path(__file__).name, "--supervise", state.get("instance", "")]))


def phases():
    return [
        ("computing", [str(PYTHON), str(RUNNER), "--resume"]),
        ("verifying", [str(PYTHON), str(RUNNER), "--verify"]),
    ]


def status():
    state = read_json(STATE)
    receipt = read_json(RECEIPT)
    result = {
        "state": state,
        "supervisor_alive": supervisor_alive(state),
        "worker_alive": bool(state and process_alive(state.get("worker_pid"))),
        "finalization": None if not receipt else {
            key: receipt.get(key) for key in
            ("status", "finalized_at_utc", "branch", "scientific_input_sha256", "artifact_count")
        },
        "cloud_lifetime_guaranteed": False,
    }
    if state:
        try:
            result["observed_supervisor_sid"] = os.getsid(state["supervisor_pid"])
        except (TypeError, KeyError, OSError):
            result["observed_supervisor_sid"] = None
    # The runner owns the schema. Never emit its complete per-point audit here.
    for filename in ("progress.json", "execution_state.json", "status.json"):
        progress = read_json(OUTPUT / filename)
        if isinstance(progress, dict):
            result[filename] = {key: value for key, value in progress.items()
                                if key not in {"attempts", "artifact_sha256", "points", "jobs"}}
    return result


def check_index(files):
    result = git("diff", "--cached", "--name-only", "-z")
    if result.returncode:
        raise ValueError("Integrity error: cannot inspect the Git index")
    foreign = [name for name in result.stdout.split("\0") if name and name not in set(files)]
    if foreign:
        raise ValueError("Integrity error: unrelated staged files: " + ", ".join(foreign))


def publish(state, update):
    """Retry Git independently, so a publication failure never refits molecules."""
    require_branch()
    check_snapshot(state)
    files = [name for name in eligible_files() if ROOT / name != RECEIPT]
    check_index(files + [str(RECEIPT.relative_to(ROOT))])
    head = git("rev-parse", "HEAD")
    if head.returncode:
        raise RuntimeError("Cannot identify the commit before finalization")
    atomic_json(RECEIPT, {
        "schema_version": 1, "status": "FINALIZED_LOCAL", "finalized_at_utc": utc_now(),
        "branch": BRANCH, "base_branch": BASE,
        "source_head_before_finalization": head.stdout.strip(),
        "scientific_input_sha256": state["scientific_input_sha256"],
        "validations": state["completed_phases"],
        "phase_attempt_costs": state["phase_attempt_costs"],
        "artifact_count": len(files),
        "artifact_sha256": {name: sha256(ROOT / name) for name in files},
        "quality_interpretation": "Scientific qualification is recorded by the runner; completion does not override warnings.",
        "publication": "Receipt and listed files are committed together and pushed without force.",
        "runtime_state_path": str(STATE.relative_to(ROOT)),
        "cloud_lifetime_guaranteed": False,
    })
    committed = False
    attempt = 0
    while True:
        require_branch()
        check_snapshot(state)
        attempt += 1
        update(phase="publishing", publication_attempt=attempt, worker_pid=None)
        if not committed:
            files = eligible_files()
            check_index(files)
            # NUL pathspec stdin avoids ARG_MAX when full derivative points exist.
            added = git("add", "--pathspec-from-file=-", "--pathspec-file-nul",
                        input_text="\0".join(files) + "\0")
            if added.returncode:
                update(last_error=safe_message(added.stderr), retry_after_seconds=30)
                time.sleep(30)
                continue
            changed = git("diff", "--cached", "--quiet")
            if changed.returncode not in (0, 1):
                update(last_error=safe_message(changed.stderr), retry_after_seconds=30)
                time.sleep(30)
                continue
            if changed.returncode == 1:
                result = git("commit", "-m", "Finalize PySCF frequency repair calculations and numerical verification")
                if result.returncode:
                    update(last_error=safe_message(result.stderr), retry_after_seconds=30)
                    time.sleep(30)
                    continue
            committed = True
            update(final_commit_sha=git("rev-parse", "HEAD").stdout.strip())
        pushed = git("push", "origin", "HEAD:refs/heads/" + BRANCH)
        if pushed.returncode == 0:
            update(publication="PUSHED", published_at_utc=utc_now(), last_error=None)
            return
        update(publication="RETRYING_PUSH", last_error=safe_message(pushed.stderr), retry_after_seconds=30)
        time.sleep(30)


def supervise(instance):
    require_branch()
    frozen = scientific_snapshot()
    if not PYTHON.is_file():
        raise ValueError("Integrity error: .venv-pyscf must be prepared before launch")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 0
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    previous = read_json(STATE)
    if previous and previous.get("scientific_input_sha256") != frozen["scientific_input_sha256"]:
        lock.close()
        raise ValueError("Integrity error: prior runtime state has different scientific inputs")
    prior_sessions = [] if not previous else list(previous.get("prior_sessions", []))
    if previous:
        prior_sessions.append({
            "instance": previous.get("instance"), "status": previous.get("status"),
            "phase_attempt_costs": previous.get("phase_attempt_costs", []),
            "interrupted_attempt": previous.get("active_phase_attempt"),
            "interrupted_attempt_cpu_seconds": None,
        })
    state = {
        "schema_version": 1, "instance": instance, "status": "RUNNING",
        "supervisor_pid": os.getpid(), "supervisor_sid": os.getsid(0),
        "supervisor_ppid_at_start": os.getppid(), "started_at_utc": utc_now(),
        "completed_phases": [], "phase_attempt_costs": [], "prior_sessions": prior_sessions,
        "active_phase_attempt": None, "quality_runtime_cpu_cutoffs": None,
        "retry_count_limit": None, "integrity_error_exit_code": INTEGRITY_ERROR,
        "cloud_lifetime_guaranteed": False, **frozen,
    }

    def update(**values):
        state.update(values)
        state["heartbeat_at_utc"] = utc_now()
        atomic_json(STATE, state)

    env = os.environ.copy()
    env.update({name: "2" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    )})
    env["PYTHONUNBUFFERED"] = "1"
    update(phase="starting", worker_pid=None, last_error=None)
    try:
        # An orphan worker may have outlived an interrupted supervisor. Wait for
        # its checkpoints instead of introducing a competing chemistry process.
        orphan_pid = previous.get("worker_pid") if previous else None
        if matching_process(orphan_pid, [RUNNER.name]):
            update(phase="waiting_for_previous_worker", worker_pid=orphan_pid)
            while matching_process(orphan_pid, [RUNNER.name]):
                update(worker_pid=orphan_pid)
                time.sleep(5)
            update(worker_pid=None)
        for phase, command in phases():
            attempt = 0
            while True:
                require_branch()
                check_snapshot(state)
                attempt += 1
                update(phase=phase, phase_attempt=attempt, command=command, last_error=None,
                       active_phase_attempt={"phase": phase, "attempt": attempt,
                                             "started_at_utc": utc_now(), "cpu_seconds": None})
                usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                wall_before = time.monotonic()
                try:
                    worker = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
                except OSError as error:
                    update(last_error=safe_message(error), active_phase_attempt=None,
                           retry_after_seconds=30, worker_pid=None)
                    time.sleep(30)
                    continue
                update(worker_pid=worker.pid)
                while worker.poll() is None:
                    update(worker_pid=worker.pid)
                    time.sleep(5)
                code = worker.wait()
                usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                state["phase_attempt_costs"].append({
                    "phase": phase, "attempt": attempt, "exit_code": code,
                    "cpu_seconds": usage_after.ru_utime + usage_after.ru_stime
                                   - usage_before.ru_utime - usage_before.ru_stime,
                    "wall_seconds": time.monotonic() - wall_before,
                    "finished_at_utc": utc_now(),
                    "measurement": "RUSAGE_CHILDREN difference after wait; overlapping engine CPU must not be added",
                })
                update(worker_pid=None, active_phase_attempt=None, last_exit_code=code)
                if code == 0:
                    state["completed_phases"].append({"phase": phase, "attempts": attempt,
                                                       "finished_at_utc": utc_now()})
                    update()
                    break
                if code == INTEGRITY_ERROR:
                    update(status="BLOCKED_INTEGRITY", last_error=f"{phase} refused mismatched scientific inputs or artifacts")
                    return INTEGRITY_ERROR
                delay = min(30, 5 * 2 ** min(attempt - 1, 3))
                update(last_error=f"{phase} exited {code}; resuming durable checkpoints",
                       retry_after_seconds=delay)
                time.sleep(delay)
        publish(state, update)
        update(status="DONE", phase="completed", worker_pid=None, finished_at_utc=utc_now())
        return 0
    except ValueError as error:
        update(status="BLOCKED_INTEGRITY", last_error=safe_message(error), worker_pid=None)
        return INTEGRITY_ERROR
    except Exception as error:
        update(status="SUPERVISOR_ERROR", last_error=safe_message(repr(error)))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def launch():
    require_branch()
    scientific_snapshot()
    if not PYTHON.is_file():
        raise ValueError("Integrity error: .venv-pyscf must be prepared before launch")
    current = read_json(STATE)
    if current:
        check_snapshot(current)
    if supervisor_alive(current):
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    if current and current.get("status") == "DONE":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    instance = uuid.uuid4().hex
    command = [str(PYTHON), str(Path(__file__).resolve()), "--supervise", "--instance", instance]
    with (RUNTIME / "supervisor.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    # This bounds only the handshake and never terminates the independent child.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        snapshot = read_json(STATE)
        if snapshot and snapshot.get("instance") == instance:
            if snapshot.get("status") in {"BLOCKED_INTEGRITY", "SUPERVISOR_ERROR"}:
                print(json.dumps(status(), ensure_ascii=False, indent=2))
                return INTEGRITY_ERROR
            if snapshot.get("worker_pid") and snapshot.get("supervisor_sid") == child.pid:
                print(json.dumps(status(), ensure_ascii=False, indent=2))
                return 0
        if child.poll() is not None:
            print(json.dumps(status(), ensure_ascii=False, indent=2))
            return child.returncode
        time.sleep(0.2)
    print(json.dumps({"launch_handshake": "PENDING", "supervisor_pid": child.pid,
                      "computation_terminated": False, "state_path": str(STATE)}))
    return 0


def verify():
    current = status()
    state = current["state"]
    if not state:
        print(json.dumps({"verified": False, "reason": "No supervisor state"}))
        return 1
    check_snapshot(state)
    valid = state.get("quality_runtime_cpu_cutoffs") is None and state.get("retry_count_limit") is None
    if current["supervisor_alive"]:
        valid = valid and state.get("supervisor_pid") == current.get("observed_supervisor_sid")
    else:
        receipt = read_json(RECEIPT)
        valid = valid and state.get("status") == "DONE" and bool(receipt)
        if receipt:
            valid = valid and all((ROOT / name).is_file() and sha256(ROOT / name) == expected
                                  for name, expected in receipt.get("artifact_sha256", {}).items())
    print(json.dumps({"verified": bool(valid), **current}, ensure_ascii=False, indent=2))
    return 0 if valid else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--instance", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.dry_run:
            frozen = scientific_snapshot(required=False)
            print(json.dumps({"branch": BRANCH, "base_branch": BASE, "phases": phases(),
                              "quality_runtime_cpu_cutoffs": None, "retry_count_limit": None,
                              "cloud_lifetime_guaranteed": False,
                              "scientific_input_sha256": frozen["scientific_input_sha256"],
                              "authenticated_file_count": len(frozen["scientific_files"]),
                              "eligible_file_count": len(eligible_files()),
                              "publication": "Only repair files, non-force branch push after independent verification"},
                             ensure_ascii=False, indent=2))
            return 0
        if args.status:
            print(json.dumps(status(), ensure_ascii=False, indent=2))
            return 0
        if args.verify:
            require_branch()
            return verify()
        if args.supervise:
            return supervise(args.instance or uuid.uuid4().hex)
        if args.launch or args.resume:
            return launch()
        parser.error("choose --launch, --resume, --status, --verify or --dry-run")
    except ValueError as error:
        print(safe_message(error), file=sys.stderr)
        return INTEGRITY_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
