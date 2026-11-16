#!/usr/bin/env python3
"""Resume the frozen task queue and publish checkpoints from GitHub Actions.

The segment boundary is a checkpoint for the hosted runner's six-hour lifetime,
not an experiment stopping rule.  No metric is inspected by this scheduler.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/focused_followup_20261116"
REPORT = ROOT / "reports/focused_followup_20261116"
RUNTIME = REPORT / "runtime"
STATE = RUNTIME / "state.json"
TERMINAL_GOOD = {"COMPLETED", "REUSED_VERIFIED", "SKIPPED_BY_CONFIG"}
TERMINAL_BLOCKED = {
    "BLOCKED_MISSING_ARTIFACT", "BLOCKED_WAITING_NEW_BATCH_DATA",
    "BLOCKED_MISSING_RAW_SIGNAL", "BLOCKED_MISSING_EXPORT_DEFINITION",
    "BLOCKED_FAILED_TASK", "BLOCKED_DEPENDENCY", "BLOCKED_INTERRUPTED_TASK",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def git(*arguments, check=True):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=check,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def cpu_children():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def live_group_cpu(group_id):
    """Include CPU of live descendants and their already-reaped children."""
    if group_id is None:
        return 0.0
    ticks = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id:
                ticks += sum(int(fields[i]) for i in (11, 12, 13, 14))
        except (OSError, ValueError, IndexError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK")


def output_hashes(task, result_path):
    paths = set()
    patterns = list(task.get("outputs", []))
    if result_path.exists():
        paths.add(result_path)
        result = json.loads(result_path.read_text())
        patterns.extend(result.get("artifacts", []))
    for pattern in patterns:
        for match in glob.glob(str(ROOT / pattern), recursive=True):
            path = Path(match)
            if path.is_dir():
                paths.update(p for p in path.rglob("*") if p.is_file())
            elif path.is_file():
                paths.add(path)
    hashes = {}
    for path in sorted(paths):
        relative = path.resolve().relative_to(ROOT)
        if relative.parts[0] not in {"experiments", "reports"}:
            raise ValueError(f"Task output is outside approved result paths: {relative}")
        if relative.parts[0] == "experiments" and relative.parts[1] != EXPERIMENT.name:
            raise ValueError(f"Task output is outside this experiment: {relative}")
        if relative.parts[0] == "reports" and relative.parts[1] != REPORT.name:
            raise ValueError(f"Task output is outside this report: {relative}")
        hashes[str(relative)] = sha256(path)
    return hashes


def verify_outputs(record):
    hashes = record.get("immutable_output_sha256", record.get("output_sha256", {}))
    return all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in hashes.items())


def verify_frozen_files():
    """Refuse code drift and changes to protected paper/data evidence."""
    for name in ("code_lock.json", "protected_baseline.json"):
        lock = EXPERIMENT / name
        if not lock.exists():
            raise FileNotFoundError(f"Required frozen file manifest is missing: {lock.relative_to(ROOT)}")
        manifest = json.loads(lock.read_text())
        files = manifest.get("files", manifest)
        if not isinstance(files, dict):
            raise ValueError(f"Invalid frozen file manifest: {name}")
        for relative, expected in files.items():
            path = ROOT / relative
            path.resolve().relative_to(ROOT)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Frozen file changed or missing: {relative}")


class Runner:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.cpu_start = time.process_time() + cpu_children()
        self.active_group = None
        self.stop_requested = False
        self.queue = json.loads(args.queue.read_text())
        self.protocol = json.loads(args.protocol.read_text())
        self.tasks = self.queue["tasks"] if isinstance(self.queue, dict) else self.queue
        ids = [t["id"] for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate frozen task IDs")
        self.identity = {"queue_sha256": sha256(args.queue), "protocol_sha256": sha256(args.protocol)}
        self.budget = float(self.protocol.get("CPU_BUDGET_SECONDS", self.protocol.get("cpu_budget_seconds", 990000)))
        if self.budget != 990000:
            raise ValueError("This task is authorized for a cumulative 990000 CPU seconds")
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {
            "schema": "focused-followup-runtime-v1", "identity": self.identity,
            "created_at_utc": utc(), "tasks": {}, "segments": [],
            "cumulative_cpu_seconds": float(self.protocol.get("PRELAUNCH_CPU_CHARGE_SECONDS", 0.0)),
        }
        if self.state["identity"] != self.identity:
            raise ValueError("Frozen protocol/queue mismatch; refusing to overwrite another task")
        self.cpu_base = float(self.state["cumulative_cpu_seconds"])
        self.restored_attempts = []
        for identifier, record in list(self.state["tasks"].items()):
            if record.get("status") == "RUNNING":
                self.restore_interrupted(identifier, record)
            elif record.get("status") in TERMINAL_GOOD and not verify_outputs(record):
                record["status"] = "BLOCKED_MISSING_ARTIFACT"
                record["detail"] = "A previously saved output hash no longer matches; no silent recomputation."
        self.segment = {
            "index": len(self.state["segments"]), "started_at_utc": utc(),
            "code_commit": git("rev-parse", "HEAD").stdout.strip(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "cpu_base_seconds": self.cpu_base,
        }
        self.state["segments"].append(self.segment)
        self.state["status"] = "RUNNING"
        self.state["cpu_budget_seconds"] = self.budget
        self.restore_evicted_cache_tasks()
        verify_frozen_files()

    def restore_evicted_cache_tasks(self):
        cache = ROOT / ".cache/focused_followup_20261116"
        sentinels = {
            "R0": cache / "background/recovery_verification.json",
            "R1": cache / "v4/train_only_features.npz",
        }
        for identifier, sentinel in sentinels.items():
            record = self.state["tasks"].get(identifier)
            if record and record.get("status") in TERMINAL_GOOD and not sentinel.is_file():
                self.state.setdefault("cache_restore_attempts", []).append({
                    "task_id": identifier, "reason": "source_cache_missing_or_evicted",
                    "missing_sentinel": str(sentinel.relative_to(ROOT)),
                    "prior_attempt": record,
                })
                # Source restoration re-authenticates the same immutable
                # bytes. Completed classifier/ranking records stay frozen.
                del self.state["tasks"][identifier]

    def restore_interrupted(self, identifier, record):
        """Retry only a verified hosted timeout, with a finite attempt count."""
        old_run = self.state["segments"][record["segment_index"]].get("github_run_id")
        repository = os.environ.get("GITHUB_REPOSITORY")
        reason = None
        details = None
        if repository and old_run and str(old_run) != os.environ.get("GITHUB_RUN_ID"):
            response = subprocess.run(
                ["gh", "api", f"repos/{repository}/actions/runs/{old_run}"],
                text=True, capture_output=True, check=False,
            )
            if response.returncode == 0:
                details = json.loads(response.stdout)
                if details.get("conclusion") == "timed_out":
                    reason = "github_hosted_run_timed_out"
        record["interruption_classification"] = reason or "unclassified_interruption_no_auto_retry"
        record["unobserved_cpu_tail"] = True
        if reason and details:
            last = dt.datetime.fromisoformat(self.state["updated_at_utc"].replace("Z", "+00:00"))
            end = dt.datetime.fromisoformat(details["updated_at"].replace("Z", "+00:00"))
            upper = max(0.0, (end - last).total_seconds()) * 4
            record["conservative_unobserved_cpu_charge_seconds"] = upper
            self.cpu_base += upper
        history = self.state.setdefault("interrupted_attempts", [])
        prior = sum(row.get("task_id") == identifier for row in history)
        if reason and prior < 2:
            record["status"] = "RETRY_IDENTIFIED_INFRASTRUCTURE"
            history.append({"task_id": identifier, **record})
            del self.state["tasks"][identifier]
            self.restored_attempts.append(identifier)
        else:
            record["status"] = "BLOCKED_INTERRUPTED_TASK"
            record["detail"] = "No classified infrastructure retry remains; partial artifacts and logs retained."

    def cpu(self):
        return self.cpu_base + max(0.0, time.process_time() + cpu_children() - self.cpu_start) + live_group_cpu(self.active_group)

    def save(self):
        self.state["cumulative_cpu_seconds"] = self.cpu()
        self.state["updated_at_utc"] = utc()
        self.segment["measured_cpu_seconds"] = self.state["cumulative_cpu_seconds"] - self.cpu_base
        self.segment["wall_seconds"] = time.monotonic() - self.started
        atomic_json(STATE, self.state)

    def publish(self, reason):
        self.save()
        if not self.args.publish:
            return
        branch = self.args.branch or git("branch", "--show-current").stdout.strip()
        if branch != "research/focused-followup-20261116":
            raise ValueError("Automatic publication is restricted to this experiment branch")
        git("add", "--", str(EXPERIMENT.relative_to(ROOT)), str(REPORT.relative_to(ROOT)))
        staged = git("diff", "--cached", "--name-only").stdout.splitlines()
        approved = (str(EXPERIMENT.relative_to(ROOT)) + "/", str(REPORT.relative_to(ROOT)) + "/")
        if any(not path.startswith(approved) for path in staged):
            raise RuntimeError("Refusing checkpoint commit with staged files outside the experiment/report directories")
        if git("diff", "--cached", "--quiet", check=False).returncode:
            git("-c", "user.name=research4 experiment bot", "-c", "user.email=research4-bot@users.noreply.github.com",
                "commit", "-m", f"focused follow-up: {reason}")
        last = None
        for attempt in range(3):
            pushed = git("push", "origin", f"HEAD:refs/heads/{branch}", check=False)
            if pushed.returncode == 0:
                return
            last = pushed.stderr[-4000:]
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise RuntimeError("Checkpoint push failed after three infrastructure attempts: " + (last or "unknown"))

    def run_task(self, task):
        verify_frozen_files()
        identifier = task["id"]
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in identifier)
        log_path = RUNTIME / "logs" / f"{safe_id}.log"
        result_path = REPORT / "task_results" / f"{identifier}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "FOCUSED_TASK_ID": identifier, "FOCUSED_PROTOCOL_PATH": str(self.args.protocol),
            "FOCUSED_RESULT_ROOT": str(REPORT),
            "FOCUSED_SOURCE_ROOT": str(ROOT / ".cache/focused_followup_20261116"),
            "FOCUSED_CPU_REMAINING_SECONDS": str(max(0.0, self.budget - self.cpu())),
            "FOCUSED_MAX_WORKERS": "4", "PYTHONUNBUFFERED": "1",
        })
        if "command" in task:
            command = task["command"]
            if not isinstance(command, list):
                raise ValueError("Task command must be an argument array, never shell text")
        else:
            script = ROOT / task["script"]
            script.resolve().relative_to(EXPERIMENT)
            command = [sys.executable, str(script), *map(str, task.get("args", []))]
        before_cpu = self.cpu()
        before_wall = time.monotonic()
        record = {
            "status": "RUNNING", "started_at_utc": utc(), "command": command,
            "code_commit": self.segment["code_commit"], "segment_index": self.segment["index"],
            "log": str(log_path.relative_to(ROOT)), "cpu_start_seconds": before_cpu,
        }
        self.state["tasks"][identifier] = record
        self.publish(f"start {identifier}")
        next_heartbeat = time.monotonic() + 60
        next_publish = time.monotonic() + 300
        budget_interrupted = False
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.active_group = process.pid
            while process.poll() is None:
                now = time.monotonic()
                if not budget_interrupted and self.cpu() >= self.budget:
                    # This is the authorized cumulative cap, never a score or
                    # pilot-time criterion. Python modules receive SIGINT so
                    # their finally blocks can preserve completed artifacts.
                    budget_interrupted = True
                    os.killpg(process.pid, signal.SIGINT)
                    record["budget_interrupt_at_utc"] = utc()
                if now >= next_heartbeat:
                    record["cpu_seconds_so_far"] = self.cpu() - before_cpu
                    record["wall_seconds_so_far"] = now - before_wall
                    self.save()
                    next_heartbeat = now + 60
                if now >= next_publish:
                    self.publish(f"heartbeat {identifier}")
                    next_publish = now + 300
                time.sleep(1)
            returncode = process.wait()
            self.active_group = None
        record.update({
            "returncode": returncode, "finished_at_utc": utc(),
            "cpu_seconds": self.cpu() - before_cpu,
            "wall_seconds": time.monotonic() - before_wall,
        })
        if budget_interrupted:
            record["status"] = "PARTIAL_BUDGET"
            record["detail"] = "Authorized cumulative CPU cap reached during task; partial artifacts and original log retained."
        elif returncode:
            record["status"] = "BLOCKED_FAILED_TASK"
            record["detail"] = "Command failed; original log retained and no score-driven or unclassified retry performed."
        else:
            try:
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                status = result.get("status", "COMPLETED")
                if not isinstance(status, str):
                    raise ValueError("Module status must be a string")
                record["status"] = status
                record["detail"] = result.get("detail", "Command completed; inspect saved module evidence for scientific completeness.")
                record["observed_output_sha256"] = output_hashes(task, result_path)
                if not record["observed_output_sha256"]:
                    record["evidence_limitation"] = "No output hashes declared; command success alone is not verified evidence completeness."
            except (OSError, ValueError, TypeError, KeyError) as error:
                record["status"] = "BLOCKED_FAILED_TASK"
                record["detail"] = "Module evidence receipt could not be validated: " + str(error)
        # Snapshot the module's original status and hashes. Summary tables may
        # change later, so only the immutable receipt and explicitly immutable
        # task artifacts are enforced when resuming the queue.
        receipt = RUNTIME / "task_receipts" / f"{safe_id}.segment_{self.segment['index']:04d}.json"
        atomic_json(receipt, {"task_id": identifier, **record})
        immutable = {str(receipt.relative_to(ROOT)): sha256(receipt)}
        for pattern in task.get("immutable_outputs", []):
            for match in glob.glob(str(ROOT / pattern), recursive=True):
                path = Path(match)
                if path.is_file():
                    immutable[str(path.resolve().relative_to(ROOT))] = sha256(path)
        record["immutable_output_sha256"] = immutable
        verify_frozen_files()
        self.publish(f"finish {identifier}: {record['status']}")

    def run(self):
        self.publish("segment startup")
        if self.args.install_dependencies:
            setup_id = f"SETUP_SEGMENT_{self.segment['index']:04d}"
            self.run_task({"id": setup_id, "command": [sys.executable, "-m", "pip", "install", "-r", str(EXPERIMENT / "requirements.txt")]})
            if self.state["tasks"][setup_id]["status"] != "COMPLETED":
                self.state["status"] = "PARTIAL_BUDGET" if self.state["tasks"][setup_id]["status"] == "PARTIAL_BUDGET" else "BLOCKED_INFRASTRUCTURE"
                self.state["remaining_tasks"] = [t["id"] for t in self.tasks if t["id"] not in self.state["tasks"]]
                self.publish("dependency setup failed")
                return
        cached_source = self.state["tasks"].get("R0", {})
        if cached_source.get("status") in TERMINAL_GOOD:
            cache_id = f"CACHE_VERIFY_SEGMENT_{self.segment['index']:04d}"
            # Run source authentication as a measured subprocess. A restored
            # cache is never trusted solely because its completion flag exists.
            verification = (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(EXPERIMENT)!r}); "
                "from recovery import verify_cached; "
                f"verify_cached(Path({str(ROOT / '.cache/focused_followup_20261116/background')!r})); "
                "print('Restored source cache authentication passed', flush=True)"
            )
            self.run_task({"id": cache_id, "command": [sys.executable, "-c", verification]})
            if self.state["tasks"][cache_id]["status"] != "COMPLETED":
                self.state["status"] = "PARTIAL_BUDGET" if self.state["tasks"][cache_id]["status"] == "PARTIAL_BUDGET" else "BLOCKED_INFRASTRUCTURE"
                self.state["remaining_tasks"] = [t["id"] for t in self.tasks if t["id"] not in self.state["tasks"]]
                self.publish("restored source cache authentication failed")
                return
        for task in self.tasks:
            identifier = task["id"]
            if identifier in self.state["tasks"]:
                continue
            dependency_ids = task.get("requires", task.get("depends_on", []))
            missing = [d for d in dependency_ids if self.state["tasks"].get(d, {}).get("status") not in TERMINAL_GOOD]
            if missing:
                self.state["tasks"][identifier] = {
                    "status": "BLOCKED_DEPENDENCY", "dependencies": missing,
                    "detail": "Required task did not produce verified usable evidence.",
                }
                self.publish(f"blocked dependency {identifier}")
                continue
            reservation = float(task.get("cpu_reservation_seconds", 0))
            if self.cpu() + reservation >= self.budget:
                self.state["status"] = "PARTIAL_BUDGET"
                break
            wall_reservation = float(task.get("wall_reservation_seconds", 900))
            if self.stop_requested or time.monotonic() - self.started + wall_reservation >= self.args.max_wall_seconds:
                self.state["status"] = "WAITING_CONTINUATION"
                break
            self.run_task(task)
            if self.state["tasks"][identifier]["status"] == "PARTIAL_BUDGET":
                self.state["status"] = "PARTIAL_BUDGET"
                break
        else:
            blocked = [k for k, v in self.state["tasks"].items() if v["status"] not in TERMINAL_GOOD]
            self.state["status"] = "FINISHED_WITH_BLOCKED_TASKS" if blocked else "COMPLETED"
            self.state["blocked_tasks"] = blocked
        self.state["remaining_tasks"] = [
            t["id"] for t in self.tasks
            if t["id"] not in self.state["tasks"] or self.state["tasks"][t["id"]]["status"] == "PARTIAL_BUDGET"
        ]
        self.segment["finished_at_utc"] = utc()
        self.publish(f"segment {self.state['status']}")
        outputs = os.environ.get("GITHUB_OUTPUT")
        if outputs:
            with open(outputs, "a", encoding="utf-8") as stream:
                stream.write(f"needs_continue={'true' if self.state['status'] == 'WAITING_CONTINUATION' else 'false'}\n")
                stream.write(f"study_status={self.state['status']}\n")
        print(json.dumps({"status": self.state["status"], "cumulative_cpu_seconds": self.state["cumulative_cpu_seconds"], "remaining": len(self.state["remaining_tasks"])}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=EXPERIMENT / "queue.json")
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "protocol.json")
    parser.add_argument("--max-wall-seconds", type=float, default=16200)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--install-dependencies", action="store_true")
    parser.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME"))
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / ".cache/focused_followup_20261116/orchestrator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(args)
        def request_checkpoint(signum, frame):
            runner.stop_requested = True
        signal.signal(signal.SIGTERM, request_checkpoint)
        signal.signal(signal.SIGINT, request_checkpoint)
        try:
            runner.run()
        except Exception as exc:
            runner.state["status"] = "BLOCKED_INFRASTRUCTURE"
            runner.state["error"] = str(exc)
            runner.save()
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
