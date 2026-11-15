#!/usr/bin/env python3
"""Keep a resumable worker running while the host environment remains alive.

This is a local supervisor, not a cloud scheduler or a resource reservation.
There are deliberately no runtime, CPU-budget, score, or retry-count stop rules.
The worker owns scientific progress, checkpoint integrity, and checkpoint resume.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    """Replace one complete JSON document; never truncate the current document."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def cpu_quota() -> dict:
    info = {"visible_cpus": os.cpu_count(), "quota_equivalent_cpus": None}
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            info["quota_equivalent_cpus"] = int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--restart-delay", type=float, default=5.0)
    parser.add_argument("--max-restart-delay", type=float, default=60.0,
                        help="Backoff delay cap; this never stops retries.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a worker command is required after --")
    if args.heartbeat_seconds <= 0 or args.restart_delay < 0:
        parser.error("heartbeat must be positive and restart delay nonnegative")
    if args.max_restart_delay < args.restart_delay:
        parser.error("max restart delay must be at least restart delay")
    args.cwd = args.cwd.resolve(strict=True)
    if not args.cwd.is_dir():
        parser.error("cwd must be a directory")
    args.state_dir = args.state_dir.resolve()
    return args


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state_dir = args.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.state_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.lock_stream = None
        self.child: subprocess.Popen | None = None
        self.received_signal: int | None = None
        self.stop_reason: str | None = None
        self.forwarded_signal = False
        self.last_heartbeat = 0.0
        self.state = {
            "schema_version": 1,
            "supervisor_pid": os.getpid(),
            "supervisor_started_at": utc_now(),
            "worker_pid": None,
            "worker_process_group": None,
            "attempt": 0,
            "phase": "starting",
            "command": args.command,
            "cwd": str(args.cwd),
            "resource_observation": cpu_quota(),
            "runtime_stop_limit": None,
            "cpu_budget_stop_limit": None,
            "metric_stop_limit": None,
            "retry_count_stop_limit": None,
            "checkpoint_owner": "worker",
            "persistence_scope": "current host lifetime; no cloud uptime guarantee",
        }

    def event(self, name: str, **fields) -> None:
        entry = {"at": utc_now(), "event": name,
                 "supervisor_pid": os.getpid(), **fields}
        try:
            with (self.state_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
        except OSError as exc:
            print(f"Cannot append supervisor event: {exc}", file=sys.stderr, flush=True)

    def publish(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_heartbeat < self.args.heartbeat_seconds:
            return
        self.state["heartbeat_at"] = utc_now()
        self.state["heartbeat_unix"] = time.time()
        heartbeat = {key: self.state[key] for key in (
            "supervisor_pid", "worker_pid", "attempt", "phase",
            "heartbeat_at", "heartbeat_unix")}
        for path, payload in ((self.state_dir / "status.json", self.state),
                              (self.state_dir / "heartbeat.json", heartbeat)):
            try:
                atomic_json(path, payload)
            except OSError as exc:
                # A temporary state-write failure must not terminate training.
                print(f"Cannot publish {path.name}: {exc}", file=sys.stderr, flush=True)
        self.last_heartbeat = now

    def signal_handler(self, signum: int, _frame) -> None:
        # Keep handlers simple; the main loop records and forwards the signal.
        if self.received_signal is None:
            self.received_signal = signum

    def check_stop(self) -> bool:
        if self.stop_reason:
            return True
        if self.received_signal is not None:
            self.stop_reason = f"signal:{signal.Signals(self.received_signal).name}"
        elif (self.state_dir / "STOP").exists():
            self.stop_reason = "STOP_file"
        if self.stop_reason:
            self.state.update(phase="stopping", stop_reason=self.stop_reason,
                              stop_requested_at=utc_now())
            self.event("stop_requested", reason=self.stop_reason,
                       worker_pid=self.state["worker_pid"])
            self.publish(force=True)
            return True
        return False

    def forward_stop(self) -> None:
        if self.child is None or self.forwarded_signal:
            return
        signum = self.received_signal or signal.SIGTERM
        try:
            os.killpg(self.child.pid, signum)
            self.event("signal_forwarded", worker_pid=self.child.pid,
                       signal=signal.Signals(signum).name)
        except ProcessLookupError:
            pass
        self.forwarded_signal = True

    def pause(self, seconds: float) -> bool:
        """Wait with heartbeats; no wall-clock expiry ever terminates a worker."""
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if self.check_stop():
                return False
            self.publish()
            time.sleep(min(0.2, max(0.0, until - time.monotonic())))
        return not self.check_stop()

    def acquire(self) -> bool:
        self.lock_stream = (self.state_dir / "supervisor.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_stream.close()
            self.lock_stream = None
            print("Supervisor or its surviving worker already holds this run lock.",
                  file=sys.stderr, flush=True)
            return False
        try:
            previous = json.loads((self.state_dir / "status.json").read_text())
            attempt = previous.get("attempt", 0)
            if isinstance(attempt, int) and attempt >= 0:
                self.state["attempt"] = attempt
        except (OSError, ValueError, TypeError):
            pass
        return True

    def run(self) -> int:
        if not self.acquire():
            return 2
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.event("supervisor_started", command=self.args.command,
                   resource_observation=self.state["resource_observation"])
        failure_streak = 0
        try:
            while not self.check_stop():
                self.state["attempt"] += 1
                attempt = self.state["attempt"]
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                log_path = self.log_dir / f"worker-{attempt:08d}-{stamp}.log"
                self.state.update(phase="launching", worker_pid=None,
                                  worker_process_group=None, log_path=str(log_path),
                                  worker_started_at=utc_now())
                self.publish(force=True)
                try:
                    with log_path.open("xb") as log_stream:
                        log_stream.write((json.dumps({"attempt": attempt,
                            "at": utc_now(), "command": self.args.command},
                            ensure_ascii=False) + "\n").encode())
                        log_stream.flush()
                        self.child = subprocess.Popen(
                            self.args.command, cwd=self.args.cwd,
                            stdin=subprocess.DEVNULL, stdout=log_stream,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            # The worker retains the run lock if the supervisor
                            # itself is killed; a replacement cannot duplicate it.
                            pass_fds=(self.lock_stream.fileno(),),
                        )
                        self.forwarded_signal = False
                        self.state.update(phase="running", worker_pid=self.child.pid,
                                          worker_process_group=self.child.pid)
                        self.event("worker_started", attempt=attempt,
                                   worker_pid=self.child.pid, log_path=str(log_path))
                        self.publish(force=True)
                        while self.child.poll() is None:
                            if self.check_stop():
                                self.forward_stop()
                            self.publish()
                            time.sleep(min(0.2, self.args.heartbeat_seconds))
                        returncode = self.child.wait()
                        self.event("worker_exited", attempt=attempt,
                                   worker_pid=self.child.pid, returncode=returncode)
                        self.state.update(last_worker_returncode=returncode,
                                          last_worker_exited_at=utc_now())
                        self.child = None
                        failure_streak = 0 if returncode == 0 else failure_streak + 1
                except OSError as exc:
                    failure_streak += 1
                    self.event("worker_spawn_failed", attempt=attempt,
                               error=f"{type(exc).__name__}: {exc}")
                    self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                    self.child = None
                self.state.update(worker_pid=None, worker_process_group=None)
                if self.check_stop():
                    break
                # Capping the delay does not cap the number of retries.
                exponent = min(max(failure_streak - 1, 0), 20)
                delay = min(self.args.max_restart_delay,
                            self.args.restart_delay * (2 ** exponent))
                self.state.update(phase="restart_wait", restart_delay_seconds=delay)
                self.event("worker_restart_scheduled", attempt=attempt,
                           delay_seconds=delay, failure_streak=failure_streak)
                self.publish(force=True)
                if not self.pause(delay):
                    break
            self.state.update(phase="stopped", stopped_at=utc_now(),
                              worker_pid=None, worker_process_group=None)
            self.event("supervisor_stopped", reason=self.stop_reason)
            self.publish(force=True)
            return 0
        finally:
            # Closing our fd does not release a surviving worker's inherited lock.
            if self.lock_stream is not None:
                self.lock_stream.close()


def main(argv: list[str] | None = None) -> int:
    return Supervisor(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
