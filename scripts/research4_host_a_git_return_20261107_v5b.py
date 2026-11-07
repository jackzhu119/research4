#!/usr/bin/env python3
"""A-only fork returning allowlisted small results after real quantum closure.

Forked from the separately frozen B helper 3f118f7c; B source is unchanged.
The single exact historical preexec failure is accepted without nonexistent
proof only as a known failed start, never as verified scientific payload.

Default is read-only validation. No force-push, main write, automatic merge,
credential discovery, checkpoint deletion, or cross-host budget inheritance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

EXPECTED_ORIGIN = "https://github.com/jackzhu119/research4.git"
BRANCH_A = "research/pyscf-spinosin-mechanism-20261031"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}\Z")
MAX_FILE = 5 * 1024**2
MAX_TOTAL = 20 * 1024**2
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".xyz", ".sdf", ".png", ".pdf", ".npy"}
FORBIDDEN_SUFFIXES = {".chk", ".h5", ".hdf", ".hdf5", ".npz", ".joblib", ".pt", ".pth", ".pkl", ".pickle", ".key", ".pem", ".p12", ".pyc"}
FORKED_B_HELPER_SHA256 = "3f118f7cbae979e67c1e219ca447e1d8193dc170bff60e0c4b70a237fb5c6a07"
V8_GUARD_SHA256 = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
A_LEDGER_RELATIVE = "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json"
KNOWN_PREEXEC_PHASE = "v8-control-water-response-000-001"
KNOWN_PREEXEC_ROW_SHA256 = "03a2bdade1ba879cb0e6c4ec0167c9b921432599016004f3b996c144a3e66517"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class SafetyError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise SafetyError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def proc_identity(proc):
    stat = (proc / "stat").read_text(); fields = stat[stat.rfind(")") + 2:].split()
    uid = next(v for v in (proc / "status").read_text().splitlines() if v.startswith("Uid:"))
    return {"pid": int(proc.name), "start_ticks": int(fields[19]), "uid": int(uid.split()[1]),
            "comm": stat[stat.find("(") + 1:stat.rfind(")")], "state": fields[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def git(repo_root, *args):
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=120)
    require(result.returncode == 0, "Git operation failed: " + args[0] + "; code=" + str(result.returncode) + ". Inspect the local index or transport; environment authorization is separate.")
    return result.stdout.strip()


def expected_branch(manifest):
    require(manifest.get("schema") == "research4-root-host-a-result-return-v1"
            and manifest.get("role") == "A" and manifest.get("run_id") == "host-a-005",
            "Only the exact ROOT-approved Host-A return manifest is allowed")
    require(manifest.get("upload", {}).get("enabled") is True
            and manifest["upload"].get("branch") == BRANCH_A and BRANCH_A != "main",
            "Host-A upload must target its fixed research branch")
    return BRANCH_A


def validate_origin(repo_root):
    actual = git(repo_root, "remote", "get-url", "origin")
    if EXPECTED_ORIGIN.startswith(("https://", "http://")):
        if actual.startswith("git@github.com:"):
            canonical = "https://github.com/" + actual.split(":", 1)[1]
        else:
            parsed = urlsplit(actual)
            require(parsed.hostname == "github.com" and parsed.scheme in ("https", "ssh"), "Expected GitHub origin required")
            canonical = "https://github.com/" + parsed.path.lstrip("/")
        require(canonical.removesuffix(".git") == EXPECTED_ORIGIN.removesuffix(".git"), "Origin repository mismatch")
    else:
        # Test fixtures patch EXPECTED_ORIGIN to an isolated local bare repo.
        require(actual == EXPECTED_ORIGIN, "Origin repository mismatch")


def work_path(repo_root, manifest):
    expected_branch(manifest)
    repo_root = Path(repo_root).resolve()
    raw = Path(manifest["work_directory"])
    require(not raw.is_absolute() and ".." not in raw.parts, "Relative work directory required")
    require(not any((repo_root / part).is_symlink() for part in [raw, *raw.parents]),
            "Symlink output prefix rejected")
    work = (repo_root / raw).resolve()
    require(work == repo_root / "reports/fixed-quadrature-20261107-v1/queue-host-a-005",
            "A upload has a fixed narrow output prefix")
    require(work.is_relative_to(repo_root / "reports") and not work.is_symlink(), "Report namespace only")
    return work


def validate_file(repo_root, work, file):
    require(not file.is_symlink() and file.is_file(), "Symlink or nonfile result rejected")
    resolved = file.resolve()
    require(resolved.is_relative_to(work) and resolved.is_relative_to(repo_root), "Result escapes allowed output prefix")
    for parent in file.relative_to(repo_root).parents:
        require(not (repo_root / parent).is_symlink(), "Symlink parent rejected")
    parts = [part.lower() for part in file.relative_to(work).parts]
    forbidden = {".git", ".runtime", "venv", ".venv", "__pycache__", "cache", "caches", ".env", ".ssh", ".aws", ".azure"}
    secret_names = ("secret", "credential", "token", "private_key", "api_key", "apikey", "password", "passwd", "id_rsa", "id_ed25519")
    require(not any(p in forbidden or p.startswith(".env.") or any(s in p for s in secret_names) for p in parts), "Secret/cache/runtime path rejected")
    require(file.suffix.lower() not in FORBIDDEN_SUFFIXES and file.suffix.lower() in ALLOWED_SUFFIXES, "Only approved small scientific result formats may be returned")
    require(file.stat().st_size <= MAX_FILE, "Result exceeds five MiB; preserve it locally with a small hash manifest")


def collect_return_files(repo_root, manifest):
    repo_root = Path(repo_root).resolve()
    expected_branch(manifest)
    work = work_path(repo_root, manifest)
    allowlist = manifest.get("upload", {}).get("allowlist")
    require(isinstance(allowlist, list) and 1 <= len(allowlist) <= 64 and all(isinstance(v, str) for v in allowlist), "Explicit finite upload allowlist required")
    selected = set()
    for pattern in allowlist:
        raw = Path(pattern)
        require(not raw.is_absolute() and ".." not in raw.parts and not pattern.startswith("."), "Allowlist must remain relative to the run directory")
        for file in work.glob(pattern):
            if file.is_dir():
                continue
            validate_file(repo_root, work, file)
            selected.add(file.resolve())
    extra = manifest.get("upload", {}).get("resource_protocol_files", [])
    require(isinstance(extra, list) and len(extra) <= 16, "Finite explicit resource protocol files required")
    if extra:
        require(manifest.get("role") == "A", "B cannot upload A resource protocol")
        protocol = repo_root / "reports/fixed-quadrature-20261107-v1/resource-protocol"
        for relative in extra:
            raw = Path(relative)
            require(not raw.is_absolute() and ".." not in raw.parts and raw.suffix == ".json", "Explicit small protocol JSON paths only")
            file = repo_root / raw
            require(file.parent.resolve() == protocol.resolve(), "A protocol upload prefix is fixed")
            validate_file(repo_root, protocol, file)
            selected.add(file.resolve())
    require(selected, "No allowlisted scientific artifacts exist")
    require(sum(p.stat().st_size for p in selected) <= MAX_TOTAL, "Return exceeds twenty MiB total")
    return sorted(selected)


def require_no_active_quantum(repo_root, manifest):
    expected_branch(manifest)
    repo_root = Path(repo_root).resolve()
    canonical = repo_root / A_LEDGER_RELATIVE
    declared = Path(manifest.get("ledger", ""))
    require(not declared.is_absolute() and ".." not in declared.parts
            and (repo_root / declared).resolve() == canonical.resolve(),
            "Exact actual Host-A registered ledger required")
    # Validate with the unmodified original supervisor before interpreting
    # any exceptional row. This never changes the historical ledger.
    guard_path = repo_root / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
    require(sha(guard_path) == V8_GUARD_SHA256, "Original frozen V8 ledger validator changed")
    spec = importlib.util.spec_from_file_location("host_a_original_v8_ledger_validator", guard_path)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    original_ledger = json.loads(canonical.read_text())
    guard.validate_ledger(original_ledger)
    ledger_candidates = [canonical]
    observed = 0
    exceptions = []
    for p in dict.fromkeys(ledger_candidates):
        if not p.exists():
            continue
        observed += 1
        value = json.loads(p.read_text())
        phases = value.get("phases")
        require(isinstance(phases, list), "Unrecognized quantum ledger; refuse upload")
        closed = {"EXITED", "PARTIAL_OR_FAILED", "FAILED_CLOSED", "CLOSED", "HEALTHY_CLOSED", "DONE", "COMPLETED", "FAILED", "PREPARED"}
        for row in phases:
            status = str(row.get("status", "")).upper()
            if status == "FAILED_TO_START":
                if known_historical_preexec_failure(row, p):
                    exceptions.append({"phase": row["phase"], "canonical_row_sha256": KNOWN_PREEXEC_ROW_SHA256,
                        "status": "KNOWN_PREEXEC_FAILURE_WITHOUT_SCIENTIFIC_PAYLOAD_OR_PROOF",
                        "actual_cpu_seconds_recorded_in_unchanged_ledger": row["actual_cpu_seconds"],
                        "missing_proof_expected_before_exec": True,
                        "verified_scientific_payload": False,
                        "not_a_newly_waived_failure_or_refund": True})
                    continue
                strict_failed_start_proof(row, p)
            else:
                require(status in closed, "Active, unknown or unresolved quantum phase; defer Git return")
    require(observed > 0, "No actual local quantum ledger; closure cannot be verified")
    require_no_live_quantum_processes()
    return {"frozen_native_V8_ledger_validator_sha256": V8_GUARD_SHA256,
            "actual_A_ledger_sha256": sha(canonical), "native_ledger_validation": "PASS_READ_ONLY",
            "historical_preexec_exceptions": exceptions,
            "source_budget_or_rows_modified": False, "no_active_quantum_observed": True}


def canonical_row_sha256(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strict_failed_start_proof(row, ledger_path):
    """Unchanged general failed-start proof requirements for every other row."""
    require(isinstance(row.get("actual_cpu_seconds"), (int, float)) and row["actual_cpu_seconds"] >= 0
            and row.get("started_at_utc") and row.get("ended_at_utc") and row.get("failure")
            and row.get("exit_code") is None and isinstance(row.get("command"), list),
            "Incomplete failed-start accounting; defer Git return")
    log = Path(row.get("log", "")); proof = Path(row.get("supervisor_proof_path", ""))
    require(log.is_file() and proof.is_file() and row.get("log_sha256") == sha(log),
            "Failed-start source/log closure evidence missing")
    evidence = json.loads(proof.read_text())
    require(evidence.get("phase") == row.get("phase")
            and Path(evidence.get("ledger_path", "")).resolve() == Path(ledger_path).resolve(),
            "Failed-start proof does not bind this real phase/ledger")


def known_historical_preexec_failure(row, ledger_path):
    """One complete immutable row, not a general missing-proof exemption."""
    if canonical_row_sha256(row) != KNOWN_PREEXEC_ROW_SHA256:
        return False
    ledger = Path(ledger_path).resolve()
    require(ledger == Path(__file__).resolve().parents[1] / A_LEDGER_RELATIVE,
            "Known preexec exception belongs only to the actual A ledger")
    require(row.get("phase") == KNOWN_PREEXEC_PHASE and row.get("status") == "FAILED_TO_START"
            and row.get("failure") == "SubprocessError('Exception occurred in preexec_fn.')"
            and type(row.get("actual_cpu_seconds")) is float and row["actual_cpu_seconds"] == 0.0
            and row.get("exit_code") is None and row.get("started_at_utc") and row.get("ended_at_utc"),
            "Exact historical preexec failure identity differs")
    expected_log = ledger.parent / "controls" / (KNOWN_PREEXEC_PHASE + ".log")
    log = Path(row.get("log", "")); proof = Path(row.get("supervisor_proof_path", ""))
    require(log == expected_log and log.is_file() and not log.is_symlink()
            and log.stat().st_size == 0 and row.get("log_sha256") == EMPTY_SHA256
            and sha(log) == EMPTY_SHA256, "Known preexec failure must retain its actual empty closed log")
    require(proof == expected_log.with_name(expected_log.name + ".supervisor-proof.json") and not proof.exists(),
            "Known preexec failure has no proof because execution never began; unexpected evidence requires review")
    return True


def require_no_live_quantum_processes():
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            stat = (proc / "stat").read_text(); fields = stat[stat.rfind(")") + 2:].split()
            if fields[0] in ("Z", "X", "x"):
                continue
            args = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if not args or not Path(args[0]).name.startswith(("python", "pypy")):
                continue
            scripts = [Path(v).name for v in args[1:] if v.endswith(".py")]
            names = [v for v in scripts if any(s in v for s in ("pyscf", "cloud_followup_worker", "cloud_orbital_stability", "cloud_small_molecule", "parallel_b_small", "parallel_b_worker", "parallel_b_quantum", "fixed_quadrature_worker")) and not v.startswith(("analyze_", "summarize_"))]
            require(not names, "Live quantum-looking process observed; defer Git return")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            # Managed runtime root daemons can hide cmdline. This narrow,
            # double-read identity exception does not include root Python or
            # an unknown process and never relies on PID alone.
            try:
                first = proc_identity(proc); second = proc_identity(proc)
                identity_keys = ("pid", "start_ticks", "uid", "comm", "boot_id")
                require(all(first[k] == second[k] for k in identity_keys) and first["uid"] == 0 and first["comm"] in ("dockerd", "containerd"), "Opaque process is not an exact stable non-scientific root daemon")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, SafetyError) as unknown:
                raise SafetyError("Process visibility insufficient to prove quantum closure") from unknown




def execute_return(repo_root, manifest, manifest_path, manifest_sha256, attempt="return-001"):
    repo_root = Path(repo_root).resolve(); manifest_path = Path(manifest_path).resolve()
    require(SAFE_ID.fullmatch(attempt) is not None, "Safe fresh return attempt required")
    require(manifest_path.is_relative_to(repo_root) and sha(manifest_path) == manifest_sha256, "Actual manifest/source binding required")
    require(json.loads(manifest_path.read_text()) == manifest, "Supplied manifest differs from bound file")
    branch = expected_branch(manifest)
    require(git(repo_root, "branch", "--show-current") == branch, "Current branch does not match the exact role/run result branch; main is forbidden")
    require(not git(repo_root, "diff", "--cached", "--name-only"), "Preexisting staged work must not be included")
    validate_origin(repo_root); quantum_closure = require_no_active_quantum(repo_root, manifest)
    files = collect_return_files(repo_root, manifest)
    snapshots = {str(p.relative_to(repo_root)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}
    receipt_dir = work_path(repo_root, manifest) / "git-return" / attempt
    require(not receipt_dir.parent.is_symlink() and receipt_dir.resolve().is_relative_to(work_path(repo_root, manifest)), "Symlink or escaping return-receipt directory rejected")
    require(not receipt_dir.exists(), "Preserve earlier upload attempts and receipts")
    receipt_dir.mkdir(parents=True)
    receipt = {"schema": "research4-actual-result-return-v1", "started_at_utc": now(), "branch": branch,
               "manifest_sha256": manifest_sha256, "source_helper_sha256": sha(__file__), "artifact_hashes": snapshots,
               "force_push": False, "main_modified": False, "actual_result_upload": "NOT_YET_PERFORMED",
               "A_quantum_closure": quantum_closure, "A_only_fork_of_B_helper_sha256": FORKED_B_HELPER_SHA256}
    receipt_path = receipt_dir / "return_receipt.json"
    try:
        require_no_active_quantum(repo_root, manifest)
        git(repo_root, "add", "--sparse", "--", *snapshots)
        staged = set(git(repo_root, "diff", "--cached", "--name-only").splitlines())
        require(staged.issubset(snapshots), "Staging differs from explicit allowlist")
        for rel, saved in snapshots.items():
            require(sha(repo_root / rel) == saved["sha256"], "Output changed while preparing return")
        # Native calculation logs are scientific evidence, not editable source.
        # Preserve their bytes, including trailing spaces, and retain the later
        # committed-blob SHA checks for every file. Lint structured text only.
        text_metadata = [v for v in snapshots if Path(v).suffix.lower() in
                         {".json", ".jsonl", ".csv", ".md", ".txt", ".xyz", ".sdf"}]
        if text_metadata:
            git(repo_root, "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol",
                "diff", "--cached", "--check", "--", *text_metadata)
        receipt["raw_native_logs_preserved_verbatim_not_whitespace_normalized"] = True
        if staged:
            git(repo_root, "commit", "-m", f"Save {manifest['role']} {manifest['run_id']} actual bounded cloud results")
        science_commit = git(repo_root, "rev-parse", "HEAD")
        for relative, saved in snapshots.items():
            blob = subprocess.run(["git", "-C", str(repo_root), "cat-file", "blob", science_commit + ":" + relative], capture_output=True, timeout=30)
            require(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == saved["sha256"], "Committed scientific blob differs from its sealed artifact; empty-index recovery cannot fabricate a commit")
        receipt.update(science_commit=science_commit, all_committed_scientific_blobs_match=True,
                       reused_existing_local_science_commit=not bool(staged))
        git(repo_root, "push", "origin", "HEAD:refs/heads/" + branch)
        heads = git(repo_root, "ls-remote", "--heads", "origin", branch).splitlines()
        require(any(line.split()[0] == science_commit and line.split()[1] == "refs/heads/" + branch for line in heads), "Remote research branch did not verify")
        receipt.update(status="DONE_ACTUAL_PUSH_AND_REMOTE_REF_VERIFIED", science_commit=science_commit,
                       actual_result_upload="DONE", verified_at_utc=now(), remote_download_of_content="NOT_RUN")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        # This second commit truthfully records the first, without self-SHA claims.
        require_no_active_quantum(repo_root, manifest)
        git(repo_root, "add", "--sparse", "--", str(receipt_path.relative_to(repo_root)))
        require(set(git(repo_root, "diff", "--cached", "--name-only").splitlines()) == {str(receipt_path.relative_to(repo_root))}, "Unexpected staged closing metadata")
        git(repo_root, "commit", "-m", f"Record actual {manifest['role']} {manifest['run_id']} result push receipt")
        git(repo_root, "push", "origin", "HEAD:refs/heads/" + branch)
        final_commit = git(repo_root, "rev-parse", "HEAD")
        final_heads = git(repo_root, "ls-remote", "--heads", "origin", branch).splitlines()
        require(any(line.split()[0] == final_commit for line in final_heads), "Closing metadata remote ref did not verify")
        return {**receipt, "final_metadata_commit": final_commit, "receipt_path": str(receipt_path)}
    except BaseException as error:
        failure = {**receipt, "status": "PARTIAL_OR_FAILED_UPLOAD_PRESERVED", "ended_at_utc": now(), "error_type": type(error).__name__, "reason": str(error)}
        (receipt_dir / "failure_receipt.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--attempt", default="return-001")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve(); source = args.manifest.resolve()
    require(sha(source) == args.manifest_sha256, "Manifest hash mismatch")
    manifest = json.loads(source.read_text()); expected_branch(manifest)
    if args.execute:
        print(json.dumps(execute_return(repo, manifest, source, args.manifest_sha256, args.attempt)))
    else:
        require(git(repo, "branch", "--show-current") == expected_branch(manifest), "Current branch mismatch")
        closure = require_no_active_quantum(repo, manifest)
        files = collect_return_files(repo, manifest)
        print(json.dumps({"status": "CHECK_ONLY_NO_GIT_WRITE_NO_UPLOAD", "files": len(files),
                          "bytes": sum(p.stat().st_size for p in files), "A_quantum_closure": closure}))


if __name__ == "__main__":
    main()
