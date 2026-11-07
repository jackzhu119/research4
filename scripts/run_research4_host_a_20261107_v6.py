"""Run the approved host-A queue and return small results after real closure.

Default validates only. It does not modify quantum implementations or budgets.
--detach starts a finite parent; no platform restart/uptime guarantee is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "research/pyscf-spinosin-mechanism-20261031"
WORK_DIRECTORY = ROOT / "reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006"
A_LEDGER_RELATIVE = "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json"
CAP = 390160.8308359999
STRICT_DEADLINE = datetime.fromisoformat("2026-11-09T08:25:17.730193+00:00")
CONFIG = ROOT / "reports/fixed-quadrature-convergence-20261107-v2/queue-configs/host-a-006/config_ROOT_APPROVED.json"
GRANT = CONFIG.with_name("root_grant_ROOT_APPROVED.json")
CONFIG_SHA = 'f58adfd21b3fa9f4b29a742657b6c3e0663bc8c1d92ccbb827f582d1e13e0aa0'
GRANT_SHA = 'c1836e5f31795dc2a9d303d0e327158dbe273e62fdfc5296efce655a23856561'
DRIVER = ROOT / "scripts/research4_cloud_followup_queue_20261106_v2.py"
DRIVER_SHA = "06a3c5f22854f78bf8f868bf85b3956506b8514110147b977550ba838eb3780f"
UPLOADER = ROOT / "scripts/research4_host_a_git_return_20261107_v6.py"
PYTHON = ROOT / ".runtime/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing-" + str(os.getpid()))
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(pid):
    proc = Path("/proc") / str(pid)
    raw = (proc / "stat").read_text(); values = raw[raw.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(values[19]), "state": values[0],
            "uid": proc.stat().st_uid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def require_frozen_bindings():
    """Root fills reviewed hashes only after the exact config/grant are frozen."""
    for value in (CONFIG_SHA, GRANT_SHA):
        require(isinstance(value,str) and len(value)==64 and all(ch in "0123456789abcdef" for ch in value),
                "ROOT must freeze and fill the A006 config/grant hashes before validation or execution")


def validate_queue_namespace(config):
    work=Path(config["work_directory"])
    work=(work if work.is_absolute() else ROOT/work).resolve()
    ledger=Path(config["ledger"])
    ledger=(ledger if ledger.is_absolute() else ROOT/ledger).resolve()
    require(work==WORK_DIRECTORY and not any(p.is_symlink() for p in (WORK_DIRECTORY,*WORK_DIRECTORY.parents)),
            "A006 wrapper requires its exact separate output namespace without symlinks")
    require(ledger==ROOT/A_LEDGER_RELATIVE and config.get("quantum_cpu_cap_seconds")==CAP,
            "A006 retains the original registered V8 ledger and fixed cumulative cap")
    deadline=datetime.fromisoformat(config["deadline_utc"])
    require(deadline.tzinfo is not None and deadline==STRICT_DEADLINE,
            "A006 retains the exact stricter earlier authorization cutoff")
    return work


def validate_fresh_attempt(work,attempt,*,detached=False):
    paths=[work/("host-a-wrapper-"+attempt+".json")]
    for index in range(1,5):
        name=attempt+"-q"+str(index)
        paths.extend((work/("host-a-queue-"+name+".console.log"),work/"closed-budget-snapshots"/(name+".json")))
    if detached:
        paths.extend((work/("host-a-wrapper-"+attempt+".console.log"),
                      work/("host-a-wrapper-"+attempt+".launch.json")))
    require(not any(p.exists() or p.is_symlink() for p in paths),
            "Fresh A006 attempt required; preserve previous wrapper, queue and closed-budget records")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--uploader-sha256", required=True)
    parser.add_argument("--return-manifest", type=Path, required=True)
    parser.add_argument("--return-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    require(not args.detach or args.execute,"--detach requires an explicit --execute invocation")
    require_frozen_bindings()
    require(args.attempt.replace("-", "").replace("_", "").isalnum() and len(args.attempt) <= 60, "Fresh safe attempt required")
    for path, expected in [(CONFIG, CONFIG_SHA), (GRANT, GRANT_SHA), (DRIVER, DRIVER_SHA),
                           (UPLOADER, args.uploader_sha256), (args.return_manifest, args.return_manifest_sha256)]:
        require(sha(path) == expected, "Bound source/input changed: " + str(path))
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == BRANCH,
            "Host A must remain on its own research branch")
    config = json.loads(CONFIG.read_text()); work = validate_queue_namespace(config)
    ledger_path = ROOT / A_LEDGER_RELATIVE
    ledger = json.loads(ledger_path.read_text())
    require(not any(row["status"] == "RUNNING" for row in ledger["phases"]), "Do not start another queue over an open quantum phase")
    base = [str(PYTHON), str(DRIVER), "--config", str(CONFIG), "--config-sha256", CONFIG_SHA,
            "--grant", str(GRANT), "--grant-sha256", GRANT_SHA]
    if not args.execute:
        subprocess.run(base, cwd=ROOT, check=True)
        print(json.dumps({"status": "CHECK_ONLY_NOT_RUN", "return_manifest_sha256": args.return_manifest_sha256}))
        return
    require(datetime.now(timezone.utc) < datetime.fromisoformat(config["deadline_utc"]), "Finite authorization expired")
    validate_fresh_attempt(work,args.attempt,detached=args.detach)
    work.mkdir(parents=True, exist_ok=True)
    if args.detach:
        console = work / ("host-a-wrapper-" + args.attempt + ".console.log")
        receipt = work / ("host-a-wrapper-" + args.attempt + ".launch.json")
        require(not console.exists() and not receipt.exists(), "Preserve previous launch records")
        command = [str(PYTHON), str(Path(__file__).resolve()), "--execute", "--attempt", args.attempt,
                   "--uploader-sha256", args.uploader_sha256, "--return-manifest", str(args.return_manifest),
                   "--return-manifest-sha256", args.return_manifest_sha256]
        with console.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        value = {"status": "ACTUAL_WRAPPER_STARTED_NOT_SCIENCE_COMPLETE", "utc": now(),
                 "identity": identity(child.pid), "source_sha256": sha(__file__),
                 "config_sha256": CONFIG_SHA, "console": str(console), "machine_shutdown_survival_not_guaranteed": True}
        save(receipt, value); print(json.dumps(value)); return
    record = work / ("host-a-wrapper-" + args.attempt + ".json")
    require(not record.exists(), "Fresh wrapper invocation required; inspect earlier checkpoints")
    state = {"status": "RUNNING", "utc": now(), "identity": identity(os.getpid()), "queue_invocations": [],
             "source_sha256": sha(__file__), "Git_status": "SEE_SEPARATE_ACTUAL_GIT_RETURN_RECEIPTS"}
    save(record, state)
    for index in range(4):
        name = args.attempt + "-q" + str(index + 1)
        log = work / ("host-a-queue-" + name + ".console.log")
        require(not log.exists(), "Preserve prior queue consoles")
        with log.open("x") as handle:
            child = subprocess.Popen([*base, "--execute", "--attempt", name], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            state["queue_invocations"].append({"attempt": name, "identity": identity(child.pid), "log": str(log)})
            save(record, state); code = child.wait()
        actual = json.loads(ledger_path.read_text())
        require(not any(row["status"] == "RUNNING" for row in actual["phases"]), "Unresolved quantum phase: preserve, do not upload or retry")
        budget_copy = work / "closed-budget-snapshots" / (name + ".json")
        budget_copy.parent.mkdir(parents=True, exist_ok=True)
        with budget_copy.open("xb") as handle:
            handle.write(ledger_path.read_bytes())
            handle.flush(); os.fsync(handle.fileno())
        queue = json.loads(Path(config["state"]).read_text()); last = queue["invocations"][-1]
        state.update(status=last["status"], queue_returncode=code, quantum_phases_closed=True, ended_at_utc=now(),
                     closed_budget_snapshot={"path": str(budget_copy.relative_to(ROOT)), "sha256": sha(budget_copy)})
        save(record, state)
        returned = False
        for retry in range(3):
            require(sha(UPLOADER) == args.uploader_sha256, "Reviewed uploader changed; preserve results without running new source")
            command = [str(PYTHON), str(UPLOADER), "--repo", str(ROOT), "--manifest", str(args.return_manifest),
                       "--manifest-sha256", args.return_manifest_sha256, "--attempt", name + "-upload" + str(retry + 1), "--execute"]
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode == 0:
                returned = True; break
            if retry < 2:
                time.sleep(20)
        save(ROOT / ".runtime/research4-host-a-006-wrapper-latest.json", {**state, "actual_upload_succeeded": returned})
        require(returned, "Results remain local; automatic bounded upload attempts failed; no science rerun")
        require(code == 0, "Closed queue error saved and backed up; explicit forensics needed before recovery")
        if last["status"] != "HEALTHY_BOUNDARY_CLOSED":
            return
    raise RuntimeError("Four finite queue invocations reached; retain all results for explicit review")


if __name__ == "__main__":
    main()
