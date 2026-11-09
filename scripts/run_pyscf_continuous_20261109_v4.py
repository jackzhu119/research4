#!/usr/bin/env python3
"""Continuous PySCF queue with explicit authorization and preserved historical costs.

Example: python scripts/run_pyscf_continuous_20261109_v4.py --config CONFIG --detach
Each argv is a list (never a shell command). {output} is the job directory and
{repo} is the repository. Result quality is recorded independently of whether the next job executes.
Detachment survives terminal closure, not cloud-host shutdown. Open phases after
a crash require explicit forensic accounting; this program never guesses costs.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import tempfile
import stat
from contextlib import contextmanager
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SCHEMA = "research4-continuous-pyscf-v4"
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
BOOT = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
HZ = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
RESERVE = 2000.0
CPU_TAIL_RESERVE = 30.0
POLL = 0.1
MAX_FILE = 5 * 1024**2
MAX_UPLOAD = 500 * 1024**2


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def proc(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    owner = Path(f"/proc/{pid}").stat().st_uid
    return {"pid": int(pid), "ppid": int(fields[1]), "pgrp": int(fields[2]),
            "sid": int(fields[3]), "state": fields[0], "start_ticks": int(fields[19]),
            "uid": owner, "boot_id": BOOT,
            "cpu_seconds": sum(int(fields[i]) for i in (11, 12, 13, 14)) / HZ,
            "rss_bytes": int(fields[21]) * PAGE}


def same(left, right):
    return all(left.get(k) == right.get(k) for k in ("pid", "start_ticks", "uid", "boot_id"))


def alive(identity):
    if not identity or not isinstance(identity.get("pid"), int):
        return False
    try:
        actual = proc(identity["pid"])
        return same(identity, actual) and actual["state"] not in ("Z", "X", "x")
    except (FileNotFoundError, ProcessLookupError):
        return False


def descendants(parent):
    """All current descendants, including adopted double-forks (Linux subreaper)."""
    rows = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                item = proc(int(entry.name))
                if item["uid"] == os.getuid():
                    rows[item["pid"]] = item
            except (FileNotFoundError, ProcessLookupError):
                pass
    owned = {parent}
    while True:
        added = {pid for pid, row in rows.items() if row["ppid"] in owned}
        if added <= owned:
            break
        owned |= added
    return [rows[pid] for pid in owned - {parent} if pid in rows]


def signal_owned(rows, sig):
    for row in rows:
        try:
            if same(row, proc(row["pid"])):
                os.kill(row["pid"], sig)
        except (FileNotFoundError, ProcessLookupError):
            pass


def exec_limited(specification):
    """Fresh, single-threaded exec shim; never use preexec_fn with upload threads."""
    require(os.getppid() == specification["parent_pid"], "Payload parent changed before resource setup")
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0, "Cannot arm parent-death containment")
    require(os.getppid() == specification["parent_pid"], "Payload parent exited during resource setup")
    resource.setrlimit(resource.RLIMIT_CPU, (specification["cpu_soft"], specification["cpu_hard"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if specification.get("address_space_bytes"):
        resource.setrlimit(resource.RLIMIT_AS, (specification["address_space_bytes"],) * 2)
    argv = specification["argv"]
    os.execvpe(argv[0], argv, os.environ)


def exec_limited_file(filename):
    """Read a private one-use specification instead of placing JSON in argv."""
    path = Path(filename)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        details = os.fstat(descriptor)
        require(details.st_uid == os.getuid() and stat.S_ISREG(details.st_mode),
                "Resource specification must be an owned regular file")
        require(details.st_mode & 0o077 == 0 and details.st_size <= 64 * 1024**2,
                "Resource specification must be private and bounded")
        with os.fdopen(descriptor, "r", closefd=False) as stream:
            specification = json.load(stream)
    finally:
        os.close(descriptor)
    path.unlink()
    exec_limited(specification)


def limited_argv(argv, cpu_soft, cpu_hard, address_space_bytes=None):
    """Keep the exec shim's argv constant-size even for many payload arguments."""
    descriptor, filename = tempfile.mkstemp(prefix="research4-exec-spec-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"argv": argv, "cpu_soft": cpu_soft, "cpu_hard": cpu_hard,
                       "address_space_bytes": address_space_bytes,
                       "parent_pid": os.getpid()}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(filename).unlink(missing_ok=True)
        raise
    return [sys.executable, str(SELF), "--exec-payload-file", filename]


def report_directory(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT / "reports") and path != ROOT / "reports",
            "Dedicated checkpoint output must remain inside repository reports")
    require(not any(item.is_symlink() for item in [path, *path.parents]),
            "Symlink checkpoint output rejected")
    return path


def job_output_directory(job, directory):
    explicit = job.get("output_dir", job.get("legacy_output_dir"))
    return report_directory(explicit) if explicit is not None else directory / "jobs" / job["id"]


def next_attempt_index(output):
    existing = []
    for path in (Path(output) / "attempts").glob("attempt-*.log"):
        match = re.fullmatch(r"attempt-([0-9]+)\.log", path.name)
        if match:
            existing.append(int(match.group(1)))
    return max(existing, default=0) + 1


def config_load(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, "Frozen configuration hash changed")
    config = read(path)
    cap = config.get("total_cpu_seconds")
    require(type(cap) is int and 2001 <= cap <= 90000, "Total CPU cap must be 2001..90000")
    directory = Path(config["run_dir"])
    directory = (ROOT / directory).resolve() if not directory.is_absolute() else directory.resolve()
    require(directory.is_relative_to(ROOT / "reports") and directory != ROOT / "reports",
            "A dedicated reports subdirectory is required")
    require(not any(p.is_symlink() for p in [directory, *directory.parents]), "Symlink run path rejected")
    jobs = config.get("jobs")
    require(isinstance(jobs, list) and 1 <= len(jobs) <= 100, "Finite job list required")
    if config.get("budget_model") == "prospective_new_authorization":
        # A new actual user directive can authorize future work while an old
        # interrupted run's unobserved tail remains explicitly unsettled.
        require(cap <= 28000, "Prospective continuation window is at most 28000 CPU seconds")
        authorization_ref = config.get("authorization_ref", {})
        authorization_path = (ROOT / authorization_ref.get("path", "")).resolve()
        require(authorization_path.is_relative_to(ROOT) and authorization_path.is_file() and
                sha(authorization_path) == authorization_ref.get("sha256"),
                "Explicit prospective user authorization reference/hash required")
        authorization = read(authorization_path)
        instruction = authorization.get("user_instruction_verbatim", "")
        require(isinstance(instruction, str) and bool(instruction.strip()) and
                hashlib.sha256(instruction.encode("utf-8")).hexdigest() == authorization.get("user_instruction_sha256"),
                "Actual continuation instruction and its exact text hash required")
        allowance = authorization.get("prospective_maximum_cpu_seconds")
        require(type(allowance) is int and cap <= allowance <= 28000 and
                authorization.get("prior_unclosed_costs_preserved") is True,
                "Prospective allowance cannot reset or certify old unclosed costs")
        allocations = authorization.get("allocations")
        if allocations is not None:
            require(isinstance(allocations, list) and allocations, "Explicit nonempty allocation list required")
            allocated = {}
            for allocation in allocations:
                name, maximum = allocation.get("run_id"), allocation.get("cap")
                require(isinstance(name, str) and SAFE.fullmatch(name) and name not in allocated and
                        type(maximum) is int and maximum > 0, "Unique finite prospective allocations required")
                allocated[name] = maximum
            require(sum(allocated.values()) <= allowance, "Parallel allocations exceed the one prospective batch")
            require(config.get("allocation_id") in allocated and cap <= allocated[config["allocation_id"]],
                    "Queue must fit its explicitly assigned allocation")
        else:
            require(config.get("allocation_id") in (None, "single"), "Undeclared prospective allocation")
    else:
        require(config.get("budget_model") in (None, "original_grant_carryforward"), "Unknown budget model")
        prior = config.get("prior_charged_cpu_seconds", 0.0)
        original = config.get("original_grant_cpu_seconds", 90000)
        require(isinstance(prior, (float, int)) and math.isfinite(prior) and prior >= 0, "Prior costs cannot be refunded")
        require(type(original) is int and 0 < original <= 90000 and prior + cap <= original,
                "This queue plus preserved prior costs must fit the original CPU grant")
    seen = set()
    for job in jobs:
        name = job.get("id")
        if "output_dir" in job and "legacy_output_dir" in job:
            require(report_directory(job["output_dir"]) == report_directory(job["legacy_output_dir"]),
                    "Conflicting explicit output directories")
        job_output_directory(job, directory)
        require(isinstance(name, str) and SAFE.fullmatch(name) and name not in seen, "Unique safe job ID required")
        require(isinstance(job.get("argv"), list) and job["argv"] and
                all(isinstance(x, str) and x and "\x00" not in x for x in job["argv"]), "Explicit argv list required")
        require(job.get("cpu_seconds") is None or (type(job["cpu_seconds"]) is int and job["cpu_seconds"] >= 5), "Optional phase grant must be finite")
        require(type(job.get("wall_seconds", 172800)) is int and 5 <= job.get("wall_seconds", 172800) <= 7 * 86400, "Finite execution wall bound required")
        require(set(job.get("depends_on", [])) <= seen, "Dependencies must be earlier jobs")
        require(isinstance(job.get("success_statuses", []), list) and
                all(isinstance(x, str) for x in job.get("success_statuses", [])), "Quality status list must contain strings")
        require(type(job.get("max_attempts", 2)) is int and 1 <= job.get("max_attempts", 2) <= 10, "Finite automatic recovery attempts required")
        require(job.get("threads", 2) in (1, 2), "One or two native threads required")
        seen.add(name)
    require(type(config.get("maximum_wall_seconds", 7 * 86400)) is int and
            60 <= config.get("maximum_wall_seconds", 7 * 86400) <= 7 * 86400, "Finite queue lifetime required")
    require(type(config.get("checkpoint_interval_seconds", 900)) is int and
            60 <= config.get("checkpoint_interval_seconds", 900) <= 3600,
            "Checkpoint interval must be 60..3600 seconds")
    frozen = {}
    for item in config.get("frozen_refs", []):
        require(isinstance(item, dict) and isinstance(item.get("path"), str), "Explicit frozen path/hash required")
        target = (ROOT / item["path"]).resolve()
        require(target.is_relative_to(ROOT) and target.is_file() and not target.is_symlink(), "Frozen file unavailable")
        require(re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) and sha(target) == item["sha256"],
                "Frozen source/input hash changed: " + item["path"])
        frozen[target] = item["sha256"]
    for job in jobs:
        for token in job["argv"]:
            if token.endswith(".py") and "{" not in token:
                target = (ROOT / token).resolve()
                require(target in frozen, "Every Python worker source must be explicitly frozen")
    upload = config.get("git_return", {})
    if upload.get("enabled"):
        branch = upload.get("branch", "")
        require(branch.startswith("research/") and re.fullmatch(r"[A-Za-z0-9_./-]+", branch)
                and ".." not in branch and not branch.endswith("/"), "Safe research result branch required")
        require(len(upload.get("source_files", [])) <= 30, "Finite source whitelist required")
        for filename in upload.get("source_files", []):
            target = (ROOT / filename).resolve()
            require(not Path(filename).is_absolute() and target.is_relative_to(ROOT) and
                    Path(filename).parts[0] in ("scripts", "docs", "tests") and target.is_file()
                    and not target.is_symlink() and target.stat().st_size < MAX_FILE,
                    "Only explicit small scripts/docs/tests may accompany results")
            require(target in frozen, "Git-return source whitelist must also be hash-frozen")
    return config, directory, digest, path


def verify_ledger(ledger, digest):
    require(ledger.get("schema") == SCHEMA and ledger.get("config_sha256") == digest,
            "Different or changed queue ledger")
    require(ledger.get("supervisor_sha256") == sha(SELF), "Frozen supervisor source changed")
    require(not ledger.get("accounting_blocked"), "Unknown process/CPU closure requires forensic review")
    require(all(row.get("status") != "RUNNING" for row in ledger.get("jobs", {}).values()),
            "Unclosed RUNNING phase: never infer closure from missing PID or a new boot")
    used = sum(row.get("actual_cpu_seconds", 0) for row in ledger.get("jobs", {}).values())
    require(math.isclose(used, ledger.get("payload_cpu_seconds", -1), abs_tol=1e-6), "Closed actual costs changed")
    require(ledger.get("preparation_cpu_seconds_reserved") == RESERVE,
            "Conservative preparation reservation changed")


class Queue:
    def __init__(self, config, directory, digest, config_path):
        self.config, self.directory, self.digest, self.config_path = config, directory, digest, config_path
        self.continuation_authorization = (read((ROOT / config["authorization_ref"]["path"]).resolve())
                                           if config.get("budget_model") == "prospective_new_authorization" else None)
        self.ledger_path = directory / "ledger.json"
        self.stop = None
        self.reaped_science_cpu = 0.0
        self.started = time.monotonic()
        self.active = None
        self.ledger = None
        self.upload_thread = None
        self.upload_receipt = None
        self.upload_pids = set()
        self.last_upload = time.monotonic()
        self.upload_cancel = False
        self.process_lock = threading.RLock()

    @contextmanager
    def allocation_claim(self):
        """Persist one authorization/allocation identity; never reset its budget by moving directories."""
        if self.config.get("budget_model") != "prospective_new_authorization":
            yield
            return
        reference = self.config["authorization_ref"]
        authorization_path = (ROOT / reference["path"]).resolve()
        allocation = self.config.get("allocation_id", "single")
        directory = authorization_path.parent / ".allocation-claims" / reference["sha256"]
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / (allocation + ".json")
        with (directory / (allocation + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Prospective allocation already has an active driver") from error
            identity = {"authorization_sha256": reference["sha256"], "allocation_id": allocation,
                        "config_sha256": self.digest, "run_dir": str(self.directory),
                        "total_cpu_seconds": self.config["total_cpu_seconds"]}
            if marker.exists():
                previous = read(marker)
                require(all(previous.get(key) == value for key, value in identity.items()),
                        "Prospective allocation already claimed by another configuration or ledger")
            else:
                save(marker, {**identity, "first_claimed_at_utc": now(),
                              "first_driver_identity": proc(os.getpid()), "budget_reset_allowed": False})
            yield

    def process_rows(self):
        with self.process_lock:
            rows = descendants(os.getpid())
            excluded = set(self.upload_pids)
            excluded |= {row["pid"] for row in rows if row["sid"] in self.upload_pids}
        while True:
            more = {row["pid"] for row in rows if row["ppid"] in excluded}
            if more <= excluded:
                break
            excluded |= more
        return ([row for row in rows if row["pid"] not in excluded],
                [row for row in rows if row["pid"] in excluded])

    def checkpoint(self, final=False, schedule=True):
        if self.upload_thread is not None and not self.upload_thread.is_alive():
            self.upload_thread.join()
            self.upload_thread = None
            self.ledger.setdefault("git_returns", []).append(self.upload_receipt)
            self.event("GIT_RETURN_FINISHED", status=self.upload_receipt["status"])
        interval = self.config.get("checkpoint_interval_seconds", 900)
        if (schedule and self.config.get("git_return", {}).get("enabled") and self.upload_thread is None
                and not self.upload_cancel and (final or time.monotonic() - self.last_upload >= interval)):
            self.last_upload = time.monotonic()
            self.upload_receipt = None
            self.upload_thread = threading.Thread(target=self.upload, name="small-result-return", daemon=False)
            self.upload_thread.start()

    def finish_upload(self):
        while self.upload_thread is not None:
            _, uploader = self.process_rows()
            if self.stop or (self.directory / "STOP").exists() or self.used(sum(p["cpu_seconds"] for p in uploader)) >= self.config["total_cpu_seconds"] - CPU_TAIL_RESERVE:
                self.upload_cancel = True
                signal_owned(uploader, signal.SIGKILL)
            self.checkpoint(schedule=False)
            if self.upload_thread is not None:
                self.upload_thread.join(timeout=POLL)

    def overhead(self):
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return max(0.0, own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime - self.reaped_science_cpu)

    def used(self, active_cpu=0.0):
        return RESERVE + self.ledger["payload_cpu_seconds"] + self.previous_overhead + self.overhead() + active_cpu

    def persist(self):
        self.ledger["updated_at_utc"] = now()
        self.ledger["supervisor_cpu_seconds"] = self.previous_overhead + self.overhead()
        self.ledger["consumed_cpu_seconds"] = self.used(self.active.get("sampled_cpu_seconds", 0) if self.active else 0)
        self.ledger["actual_observed_cpu_seconds_excluding_preparation_reservation"] = self.ledger["consumed_cpu_seconds"] - RESERVE
        self.ledger["remaining_cpu_seconds"] = max(0.0, self.config["total_cpu_seconds"] - self.ledger["consumed_cpu_seconds"])
        if self.config.get("budget_model") == "prospective_new_authorization":
            self.ledger["combined_original_grant_charged_cpu_seconds"] = None
            self.ledger["historical_cost_accounting_status"] = "PRESERVED_UNSETTLED_INTERRUPTED_TAIL"
            self.ledger["prospective_current_run_charged_cpu_seconds"] = self.ledger["consumed_cpu_seconds"]
            self.ledger["prior_final_cost_exact"] = None
            self.ledger["prior_unclosed_costs_preserved"] = True
            prior_recorded = self.continuation_authorization.get("prior_last_recorded_charge_including_reservations")
            self.ledger["prior_last_recorded_charge_including_reservations"] = prior_recorded
            self.ledger["combined_recorded_charge_excluding_unknown_tail"] = (prior_recorded + self.ledger["consumed_cpu_seconds"]
                if isinstance(prior_recorded, (float, int)) and math.isfinite(prior_recorded) else None)
        else:
            self.ledger["combined_original_grant_charged_cpu_seconds"] = self.config.get("prior_charged_cpu_seconds", 0) + self.ledger["consumed_cpu_seconds"]
        save(self.ledger_path, self.ledger)

    def event(self, event, **details):
        self.persist()
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"at_utc": now(), "event": event, **details}, ensure_ascii=False) + "\n")

    def interrupted(self, sig, _frame):
        self.stop = "SUPERVISOR_" + signal.Signals(sig).name

    def prerequisites(self, job):
        replacements, missing = {}, []
        for entry in job.get("output_prerequisites", []):
            item = {"path": entry} if isinstance(entry, str) else entry
            requested = item["path"]
            candidates = [requested, *item.get("fallback_paths", [])]
            chosen = None
            for value in candidates:
                path = Path(value.replace("{repo}", str(ROOT)))
                path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
                if path.is_file() and path.stat().st_size:
                    try:
                        if path.suffix.lower() == ".json":
                            require(isinstance(read(path), dict), "Usable JSON input required")
                        chosen = str(path)
                        break
                    except (ValueError, OSError):
                        continue
            if chosen is None:
                if not item.get("optional", False):
                    missing.append(requested)
            else:
                replacements[requested] = chosen
        return replacements, missing

    def run_job(self, job):
        aggregate = {"status": "RUNNING", "started_at_utc": now(), "actual_cpu_seconds": 0.0,
                     "attempts": [], "quality_does_not_block_downstream": True}
        self.ledger["jobs"][job["id"]] = aggregate
        output = job_output_directory(job, self.directory)
        start_index = next_attempt_index(output)
        aggregate["output_directory"] = str(output)
        aggregate["first_new_attempt_index"] = start_index
        for ordinal in range(1, job.get("max_attempts", 2) + 1):
            index = start_index + ordinal - 1
            if self.stop or (self.directory / "STOP").exists():
                aggregate["status"] = "STOPPED"
                break
            if self.used() >= self.config["total_cpu_seconds"] - CPU_TAIL_RESERVE:
                aggregate["status"] = "CPU_BUDGET_STOPPED"
                self.stop = "TOTAL_CPU_GUARD"
                break
            self.run_attempt(job, aggregate, index)
            attempt = aggregate["attempts"][-1]
            if attempt["status"] == "COMPLETED":
                aggregate["status"] = ("COMPLETED" if attempt.get("quality_matches_requested_status") and not attempt.get("quality_warnings_present")
                                       else "COMPLETED_WITH_WARNINGS")
                break
            if attempt.get("stop_reason") in ("TOTAL_CPU_GUARD", "QUEUE_WALL_GUARD"):
                self.stop = attempt["stop_reason"]
                aggregate["status"] = "RESOURCE_STOPPED"
                break
            if self.stop or (self.directory / "STOP").exists():
                aggregate["status"] = "STOPPED"
                break
            aggregate["status"] = "RECOVERING" if ordinal < job.get("max_attempts", 2) else "FAILED_CONTINUED"
            self.event("JOB_ATTEMPT_FAILED_CONTINUING", job=job["id"], attempt=index,
                       next_attempt=ordinal < job.get("max_attempts", 2))
        aggregate["ended_at_utc"] = now()
        self.event("JOB_FINISHED_CONTINUING", job=job["id"], status=aggregate["status"])

    def run_attempt(self, job, aggregate, attempt_index):
        config_load(self.config_path, self.digest)
        output = job_output_directory(job, self.directory)
        output.mkdir(parents=True, exist_ok=True)
        attempt_directory = output / "attempts"
        attempt_directory.mkdir(exist_ok=True)
        # A phase never receives a fresh grant: every attempt shares the one remaining pool.
        cpu_grant = max(1, int(self.config["total_cpu_seconds"] - self.used() - CPU_TAIL_RESERVE))
        argv = [arg.replace("{output}", str(output)).replace("{repo}", str(ROOT)) for arg in job["argv"]]
        row = {"status": "RUNNING", "started_at_utc": now(), "argv": argv,
               "remaining_pool_cpu_at_start": cpu_grant, "sampled_cpu_seconds": 0.0,
               "attempt_index": attempt_index,
               "output_directory": str(output), "guardian_identity": proc(os.getpid())}
        aggregate["attempts"].append(row)
        aggregate["status"] = "RUNNING"
        self.active = row
        self.event("JOB_STARTING", job=job["id"])
        threads = job.get("threads", 2)
        environment = dict(os.environ, PYTHONUNBUFFERED="1", R4_JOB_OUTPUT=str(output), R4_ATTEMPT=str(attempt_index))
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = str(threads)

        child = None
        reaped = 0.0
        direct_exit = None
        reason, terminate_at = None, None
        began = time.monotonic()
        try:
            with (attempt_directory / ("attempt-%03d.log" % attempt_index)).open("xb") as log:
                child = subprocess.Popen(limited_argv(argv, cpu_grant, cpu_grant + 1, 12 * 1024**3),
                                         cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row["worker_identity"] = proc(child.pid)
                aggregate["worker_identity"] = row["worker_identity"]
                aggregate["current_attempt"] = attempt_index
                self.event("JOB_RUNNING", job=job["id"])
                last_saved = 0.0
                while True:
                    while True:
                        try:
                            # Never steal the uploader thread's Git child status.
                            # Only science children reparented to this subreaper
                            # can be reaped here; worker-waited children are part
                            # of its eventual wait4 rusage instead.
                            science_rows, _ = self.process_rows()
                            reapable = [p["pid"] for p in science_rows if p["ppid"] == os.getpid()]
                            found = (0, 0, None)
                            for candidate in reapable:
                                try:
                                    found = os.wait4(candidate, os.WNOHANG)
                                except ChildProcessError:
                                    continue
                                if found[0]:
                                    break
                            pid, code, usage = found
                        except ChildProcessError:
                            pid = 0
                        if not pid:
                            break
                        cost = usage.ru_utime + usage.ru_stime
                        reaped += cost
                        self.reaped_science_cpu += cost
                        if pid == child.pid:
                            direct_exit = os.waitstatus_to_exitcode(code)
                            child.returncode = direct_exit
                    rows, uploader = self.process_rows()
                    living = [p for p in rows if p["state"] not in ("Z", "X", "x")]
                    sampled = reaped + sum(p["cpu_seconds"] for p in rows)
                    row["sampled_cpu_seconds"] = sampled
                    aggregate["sampled_current_attempt_cpu_seconds"] = sampled
                    row["peak_rss_bytes"] = max(row.get("peak_rss_bytes", 0), sum(p["rss_bytes"] for p in rows))
                    elapsed = time.monotonic() - began
                    if direct_exit is not None and not rows:
                        break
                    if self.used(sampled + sum(p["cpu_seconds"] for p in uploader)) >= self.config["total_cpu_seconds"] - CPU_TAIL_RESERVE:
                        reason = reason or "TOTAL_CPU_GUARD"
                    if sum(p["rss_bytes"] for p in rows) >= 10 * 1024**3:
                        reason = reason or "RSS_GUARD"
                    if elapsed >= job.get("wall_seconds", 172800) - 1:
                        reason = reason or "ATTEMPT_WALL_GUARD"
                    if time.monotonic() - self.started >= self.config.get("maximum_wall_seconds", 7 * 86400) - 2:
                        reason = reason or "QUEUE_WALL_GUARD"
                    if self.stop or (self.directory / "STOP").exists():
                        reason = reason or self.stop or "STOP_FILE"
                        self.upload_cancel = True
                        signal_owned(uploader, signal.SIGKILL)
                    if direct_exit is not None and living:
                        reason = reason or "WORKER_EXITED_WITH_LIVE_DESCENDANTS"
                    if reason and terminate_at is None:
                        terminate_at = time.monotonic()
                        signal_owned(rows, signal.SIGTERM)
                        row["stop_reason"] = reason
                        self.event("JOB_STOP_REQUESTED", job=job["id"], reason=reason)
                    if terminate_at is not None and time.monotonic() - terminate_at >= 0.3:
                        signal_owned(rows, signal.SIGKILL)
                    if terminate_at is not None and time.monotonic() - terminate_at > 10:
                        raise RuntimeError("Owned process closure could not be established")
                    if elapsed - last_saved >= 2:
                        self.persist()
                        last_saved = elapsed
                    self.checkpoint()
                    time.sleep(POLL)
            require(direct_exit is not None and not self.process_rows()[0], "Incomplete wait4 closure")
            row.update(actual_cpu_seconds=reaped, exit_code=direct_exit, ended_at_utc=now(),
                       actual_cpu_source="WAIT4_ALL_OWNED_REAPED_CHILDREN_USER_PLUS_SYSTEM",
                       all_owned_processes_closed=True, wall_seconds=time.monotonic() - began)
            self.ledger["payload_cpu_seconds"] += reaped
            aggregate["actual_cpu_seconds"] += reaped
            self.active = None
            result_path = output / "result.json"
            try:
                result = read(result_path) if result_path.is_file() else {}
                require(isinstance(result, dict), "Result must be a JSON object")
            except (ValueError, OSError) as error:
                result = {}
                row["result_error"] = str(error)
            row["result_status"] = result.get("status")
            if result_path.is_file():
                row["result_sha256"] = sha(result_path)
            row["status"] = ("COMPLETED" if direct_exit == 0 and not reason and result else "FAILED_CLOSED")
            row["quality_matches_requested_status"] = result.get("status") in job.get("success_statuses", [])
            row["quality_flags"] = result.get("quality_flags", result.get("warnings", []))
            row["quality_warnings_present"] = bool(row["quality_flags"] or result.get("warnings") or
                result.get("optimizer_converged") is False or result.get("native_pyscf_converged") is False)
            aggregate.update(last_result_status=result.get("status"), result_status=result.get("status"),
                             last_exit_code=direct_exit, last_stop_reason=reason, updated_at_utc=now())
            if reason:
                row["stop_reason"] = reason
            self.event("JOB_CLOSED", job=job["id"], status=row["status"], actual_cpu_seconds=reaped)
        except BaseException as error:
            remaining, _ = self.process_rows()
            signal_owned(remaining, signal.SIGKILL)
            if child is None:
                row.update(status="FAILED_CLOSED", actual_cpu_seconds=0.0, ended_at_utc=now(),
                           error=type(error).__name__ + ": " + str(error), all_owned_processes_closed=True)
                aggregate["last_error"] = row["error"]
                self.active = None
                self.event("JOB_LAUNCH_FAILED", job=job["id"])
                return
            # An exception after Popen can conceal final costs. Fail closed rather
            # than manufacture a zero or permit another job/restart.
            row["error"] = type(error).__name__ + ": " + str(error)
            row["status"] = "RUNNING"
            self.ledger["accounting_blocked"] = True
            self.event("ACCOUNTING_BLOCKED", job=job["id"])
            raise

    def git(self, argv):
        require(not self.upload_cancel, "Checkpoint upload cancelled by resource/stop guard")
        with self.process_lock:
            child = subprocess.Popen(limited_argv(["git", "-c", "core.quotePath=false", "-C", str(ROOT), *argv], 60, 65),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            self.upload_pids.add(child.pid)
        try:
            out, err = child.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            raise RuntimeError("Git return timed out")
        finally:
            # A reparented Git descendant retains its session identity even
            # after its original parent was reaped. Close that session before
            # removing its exclusion from scientific accounting.
            deadline = time.monotonic() + 2
            while True:
                leftover = [row for row in descendants(os.getpid()) if row["sid"] == child.pid]
                if not leftover:
                    with self.process_lock:
                        self.upload_pids.discard(child.pid)
                    break
                signal_owned(leftover, signal.SIGKILL)
                for row in leftover:
                    if row["ppid"] == os.getpid():
                        try:
                            os.wait4(row["pid"], os.WNOHANG)
                        except ChildProcessError:
                            pass
                if time.monotonic() > deadline:
                    self.ledger["accounting_blocked"] = True
                    self.stop = "UPLOAD_PROCESS_CLOSURE_UNKNOWN"
                    raise RuntimeError("Git descendants could not be closed; accounting blocked")
                time.sleep(POLL)
        require(child.returncode == 0, "Git return failed: " + err.decode(errors="replace")[:1000])
        return out.decode().strip()

    def git_add_paths(self, paths):
        """NUL-delimited literal pathspecs avoid ARG_MAX and glob expansion."""
        control = self.directory / "local-only" / "git-upload-control"
        control.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix="pathspec-", suffix=".nul", dir=control)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                for path in paths:
                    require("\x00" not in path, "NUL byte cannot occur in a Git path")
                    stream.write(os.fsencode(path) + b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            self.git(["--literal-pathspecs", "add", "--force",
                      "--pathspec-from-file=" + filename, "--pathspec-file-nul"])
        finally:
            Path(filename).unlink(missing_ok=True)

    def upload(self):
        """Serialize index/commit/push across simultaneous scientific drivers."""
        if not self.config.get("git_return", {}).get("enabled"):
            return
        lock_path = ROOT / ".git" / "pyscf-checkpoint-upload.lock"
        try:
            with lock_path.open("a") as lock:
                deadline = time.monotonic() + 2.0
                while True:
                    if self.upload_cancel or self.stop or (self.directory / "STOP").exists():
                        self.upload_receipt = {"status": "UPLOAD_CANCELLED_LOCAL_RESULTS_PRESERVED", "at_utc": now()}
                        return
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            self.upload_receipt = {"status": "UPLOAD_DEFERRED_OTHER_DRIVER_LOCAL_RESULTS_PRESERVED", "at_utc": now()}
                            return
                        time.sleep(POLL)
                self.upload_locked()
        except Exception as error:
            self.upload_receipt = {"status": "UPLOAD_FAILED_LOCAL_RESULTS_PRESERVED", "error": str(error), "at_utc": now()}

    def upload_locked(self):
        settings = self.config.get("git_return", {})
        if not settings.get("enabled"):
            return
        try:
            require(self.used() < self.config["total_cpu_seconds"] - CPU_TAIL_RESERVE, "Upload skipped near CPU cap")
            branch = settings["branch"]
            require(self.git(["branch", "--show-current"]) == branch, "Never push another branch or main")
            paths, total, omissions = [], 0, []
            excluded = {"local-only", "checkpoints", "checkpoint", "cache", "caches", ".runtime", "venv"}
            roots = {self.directory}
            roots.update(job_output_directory(job, self.directory) for job in self.config["jobs"]
                         if "output_dir" in job or "legacy_output_dir" in job)
            visited = set()
            for checkpoint_root in sorted(roots):
                for path in sorted(checkpoint_root.rglob("*")):
                    if path in visited:
                        continue
                    visited.add(path)
                    relative = path.relative_to(checkpoint_root)
                    if (not path.is_file() or path.is_symlink() or excluded.intersection(relative.parts)
                            or path.suffix.lower() in (".chk", ".h5", ".hdf5", ".lock", ".tmp")
                            or path.name == "STOP" or path.stat().st_size >= MAX_FILE):
                        if path.is_file():
                            omissions.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                              "reason": "LOCAL_CHECKPOINT_CONTROL_SYMLINK_OR_FILE_SIZE_LIMIT"})
                        continue
                    total += path.stat().st_size
                    require(total <= MAX_UPLOAD, "Result return exceeds cumulative 500 MiB")
                    paths.append(str(path.relative_to(ROOT)))
            paths += settings.get("source_files", [])
            manifest = self.directory / "git_return_manifest.json"
            save(manifest, {"status": "PARTIAL_LIVE_CHECKPOINT" if self.active else "CLOSED_QUEUE_SNAPSHOT",
                            "at_utc": now(), "excluded_files_retained_locally": omissions,
                            "included_paths": sorted(set(paths)),
                            "explicit_checkpoint_roots": [str(path.relative_to(ROOT)) for path in sorted(roots)],
                            "maximum_file_bytes_exclusive": MAX_FILE,
                            "maximum_snapshot_bytes": MAX_UPLOAD})
            paths.append(str(manifest.relative_to(ROOT)))
            paths = sorted(set(paths))
            require(sum((ROOT / p).stat().st_size for p in paths) <= MAX_UPLOAD,
                    "Result/source return exceeds cumulative 500 MiB")
            allowed = set(paths)
            staged = set(filter(None, self.git(["diff", "--cached", "--name-only"]).splitlines()))
            require(staged <= allowed, "Unrelated staged files block return")
            if paths:
                self.git_add_paths(paths)
                changes = self.git(["diff", "--cached", "--name-only"])
                if changes:
                    self.git(["commit", "-m", "Save autonomous PySCF queue actual results"])
                commit = self.git(["rev-parse", "HEAD"])
                self.git(["push", "origin", "HEAD:refs/heads/" + branch])
                remote = self.git(["ls-remote", "--heads", "origin", "refs/heads/" + branch]).split()[0]
                require(remote == commit, "Remote result commit not verified")
                receipt = {"status": "PUSH_REMOTE_HEAD_VERIFIED", "commit": commit, "at_utc": now()}
            else:
                receipt = {"status": "NO_SMALL_RESULTS", "at_utc": now()}
        except Exception as error:
            receipt = {"status": "UPLOAD_FAILED_LOCAL_RESULTS_PRESERVED", "error": str(error), "at_utc": now()}
        self.upload_receipt = receipt

    def run(self, resume=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.allocation_claim(), (self.directory / "driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0,
                    "Linux child-subreaper support required")
            if self.ledger_path.exists():
                require(resume, "Existing ledger requires explicit --resume; terminal jobs are never retried")
                self.ledger = read(self.ledger_path)
                verify_ledger(self.ledger, self.digest)
            else:
                self.ledger = {"schema": SCHEMA, "config_sha256": self.digest, "supervisor_sha256": sha(SELF),
                               "created_at_utc": now(), "total_cpu_seconds": self.config["total_cpu_seconds"],
                               "preparation_cpu_seconds_reserved": RESERVE,
                               "preparation_reservation_is_conservative_not_measured_CPU": True,
                               "payload_cpu_seconds": 0.0, "supervisor_cpu_seconds": 0.0, "jobs": {},
                               "prior_charged_cpu_seconds": (None if self.config.get("budget_model") == "prospective_new_authorization"
                                                             else self.config.get("prior_charged_cpu_seconds", 0)),
                               "original_grant_cpu_seconds": self.config.get("original_grant_cpu_seconds", 90000),
                               "budget_model": self.config.get("budget_model", "original_grant_carryforward"),
                               "authorization_ref": self.config.get("authorization_ref"),
                               "allocation_id": self.config.get("allocation_id"),
                               "historical_accounting": self.config.get("historical_accounting"),
                               "scientific_quality_stops_queue": False}
                save(self.directory / "config.frozen.json", self.config)
            self.previous_overhead = self.ledger["supervisor_cpu_seconds"]
            self.ledger.update(status="RUNNING", driver_identity=proc(os.getpid()),
                               detachment_survives_host_shutdown=False)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, self.interrupted)
            self.event("DRIVER_STARTED")
            try:
                for job in self.config["jobs"]:
                    if job["id"] in self.ledger["jobs"]:
                        continue
                    if self.stop or (self.directory / "STOP").exists():
                        self.ledger["status"] = ("CPU_BUDGET_STOPPED" if self.stop == "TOTAL_CPU_GUARD" else
                                                 "WALL_BUDGET_STOPPED" if self.stop == "QUEUE_WALL_GUARD" else "STOPPED")
                        break
                    if self.used() >= self.config["total_cpu_seconds"] - CPU_TAIL_RESERVE:
                        self.ledger["status"] = "CPU_BUDGET_STOPPED"
                        break
                    # Prior quality is advisory. Only unavailable actual input files can defer work.
                    replacements, missing = self.prerequisites(job)
                    if missing:
                        self.ledger["jobs"][job["id"]] = {"status": "MISSING_INPUT_CONTINUED", "missing_inputs": missing, "actual_cpu_seconds": 0.0}
                        self.event("JOB_MISSING_INPUT_CONTINUED", job=job["id"])
                        continue
                    resolved = dict(job, argv=[replacements.get(arg, arg) for arg in job["argv"]])
                    self.run_job(resolved)
                    self.checkpoint(final=True)
                    self.finish_upload()
                    if self.stop or (self.directory / "STOP").exists():
                        self.ledger["status"] = ("CPU_BUDGET_STOPPED" if self.stop == "TOTAL_CPU_GUARD" else
                                                 "WALL_BUDGET_STOPPED" if self.stop == "QUEUE_WALL_GUARD" else "STOPPED")
                        break
                else:
                    unsuccessful = [name for name, row in self.ledger["jobs"].items()
                                    if row.get("status") in ("FAILED_CONTINUED", "MISSING_INPUT_CONTINUED")]
                    self.ledger["unresolved_jobs_after_all_attempts"] = unsuccessful
                    self.ledger["status"] = "WORKLIST_EXHAUSTED_WITH_UNRESOLVED_JOBS" if unsuccessful else "ALL_WORK_COMPLETE"
                self.event("DRIVER_FINISHED", status=self.ledger["status"])
                self.checkpoint(final=True)
                self.finish_upload()
            finally:
                self.upload_cancel = True
                self.finish_upload()
                self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--resume", action="store_true",
                        help="Continue only unstarted jobs in a fully closed ledger; unknown RUNNING costs block recovery")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--run", action="store_true", help="Internal foreground guardian")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    config, directory, digest, config_path = config_load(args.config, args.config_sha256)
    ledger_path = directory / "ledger.json"
    if args.status:
        value = read(ledger_path) if ledger_path.exists() else {"status": "PREPARED_NOT_RUNNING"}
        identity = value.get("driver_identity")
        value["driver_identity_currently_alive"] = bool(identity and alive(identity))
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return
    if args.run:
        Queue(config, directory, digest, config_path).run(resume=args.resume)
        return
    if not args.detach:
        print(json.dumps({"status": "VALIDATED_NOT_RUNNING", "config_sha256": digest, "jobs": len(config["jobs"]),
                          "total_cpu_seconds": config["total_cpu_seconds"], "reserved_cpu_seconds": RESERVE}))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger_path.exists():
            require(args.resume, "Existing ledger requires explicit --resume")
            existing = read(ledger_path)
            verify_ledger(existing, digest)
            require(not alive(existing.get("driver_identity", {})), "This queue already has a live driver")
            require(existing.get("status") not in ("ALL_WORK_COMPLETE", "WORKLIST_EXHAUSTED_WITH_UNRESOLVED_JOBS"), "All configured work already attempted")
        launches = directory / "launches"
        launches.mkdir(exist_ok=True)
        for old in launches.glob("*.json"):
            require(not alive(read(old)["identity"]), "A detached launch already owns this queue")
        receipt = launches / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
        argv = [sys.executable, str(SELF), "--run", "--config", str(config_path), "--config-sha256", digest]
        if args.resume:
            argv.append("--resume")
        with (directory / "driver.log").open("ab") as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        value = {"status": "DETACHED_PROCESS_STARTED", "identity": proc(child.pid), "config_sha256": digest,
                 "at_utc": now(), "host_shutdown_survival_guaranteed": False, "argv": argv}
        save(receipt, value)
        print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--exec-payload-file":
        exec_limited_file(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--exec-payload":
        exec_limited(json.loads(sys.argv[2]))
    else:
        main()
