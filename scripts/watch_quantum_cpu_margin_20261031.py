"""Request graceful quantum checkpoint saving before the kernel hard CPU cap."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def proc_cpu_seconds(stat_text, ticks):
    # Process names can contain spaces and parentheses; fields after the final ')'
    # start at Linux stat field3. utime/stime are fields14/15.
    fields = stat_text.rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / ticks


def active_phase(ledger, phase):
    matches = [row for row in ledger["phases"] if row["phase"] == phase]
    return matches[-1] if matches and matches[-1].get("status") == "RUNNING" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--margin-seconds", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.margin_seconds <= 120:
        raise ValueError("Fresh output and bounded5..120s margin required")
    ledger = json.loads(args.ledger.read_text())
    row = active_phase(ledger, args.phase)
    if row is None:
        raise ValueError("Requested phase is not running")
    pid = row["pid"]
    threshold = row["granted_cpu_seconds"] - args.margin_seconds
    if threshold <= 0:
        raise ValueError("Margin exceeds phase allowance")
    ticks = os.sysconf("SC_CLK_TCK")
    started = time.monotonic()
    result = {"phase": args.phase, "pid": pid, "grace_cpu_margin_seconds": args.margin_seconds,
              "requested_cpu_threshold_seconds": threshold, "quantum_jobs_launched": 0}
    while time.monotonic() - started <= row["wall_limit_seconds"] + 10:
        current = active_phase(json.loads(args.ledger.read_text()), args.phase)
        if current is None:
            result["outcome"] = "PHASE_EXITED_WITHOUT_MARGIN_SIGNAL"
            break
        if current.get("pid") != pid:
            raise RuntimeError("Worker identity changed; refuse signal")
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"pyscf_pilot_" not in cmd:
                raise RuntimeError("Unrecognized worker; refuse signal")
            cpu = proc_cpu_seconds(Path(f"/proc/{pid}/stat").read_text(), ticks)
        except FileNotFoundError:
            result["outcome"] = "WORKER_ALREADY_EXITED"
            break
        if cpu >= threshold:
            os.killpg(pid, signal.SIGTERM)
            result.update(outcome="GRACEFUL_RESOURCE_STOP_REQUESTED", cpu_seconds_when_requested=cpu,
                          signal="SIGTERM", reason="Allow Python signal handler to save PARTIAL state before hard CPU cap")
            break
        time.sleep(2)
    else:
        result["outcome"] = "WATCHER_WALL_BOUND_REACHED"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
