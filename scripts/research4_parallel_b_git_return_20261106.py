#!/usr/bin/env python3
"""Return bounded, allowlisted small results after real quantum closure.

Default is read-only validation. No force-push, main write, automatic merge,
credential discovery, checkpoint deletion, or cross-host budget inheritance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
from urllib.parse import urlsplit

EXPECTED_ORIGIN = "https://github.com/jackzhu119/research4.git"
SOURCE_BRANCH_B = "research/parallel-worker-b-20261106"
BRANCH_A = "research/pyscf-spinosin-mechanism-20261031"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}\Z")
MAX_FILE = 5 * 1024**2
MAX_TOTAL = 20 * 1024**2
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".xyz", ".sdf", ".png", ".pdf", ".npy"}
FORBIDDEN_SUFFIXES = {".chk", ".h5", ".hdf", ".hdf5", ".npz", ".joblib", ".pt", ".pth", ".pkl", ".pickle", ".key", ".pem", ".p12", ".pyc"}


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
    require(result.returncode == 0, "Git operation failed: " + args[0] + "; code=" + str(result.returncode) + ". Check this account's repository authorization; credentials are not inherited.")
    return result.stdout.strip()


def expected_branch(manifest):
    role = manifest.get("role", "B")
    run_id = manifest.get("run_id", "")
    require(SAFE_ID.fullmatch(run_id) is not None, "Safe explicit run_id required")
    if role == "A":
        value = BRANCH_A
    else:
        require(role == "B", "Only explicitly bound A/B roles are allowed")
        value = SOURCE_BRANCH_B + "-" + run_id
    declared = manifest.get("upload", {}).get("branch")
    require(declared == value and value != "main", "Upload branch must equal the fixed role/run binding")
    return value


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
    repo_root = Path(repo_root).resolve()
    raw = Path(manifest["work_directory"])
    require(not raw.is_absolute() and ".." not in raw.parts, "Relative work directory required")
    lexical = repo_root / raw
    require(not any((repo_root / part).is_symlink() for part in [raw, *raw.parents]), "Symlink output prefix rejected")
    work = lexical.resolve()
    role = manifest.get("role", "B")
    if role == "B":
        expected = repo_root / "reports/parallel-cloud-20261106/worker-b" / manifest["run_id"]
        require(work == expected.resolve(), "B output prefix must match its run_id")
    else:
        require(role == "A" and work == repo_root / "reports/cloud-followup-20261106-v1/queue-host-a-001", "A upload has a fixed narrow output prefix")
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
        protocol = repo_root / "reports/cloud-followup-20261106-v1/resource-protocol"
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
    repo_root = Path(repo_root).resolve()
    ledger_candidates = []
    if manifest.get("role", "B") == "A":
        ledger_candidates.append(repo_root / "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json")
    declared = manifest.get("ledger", manifest.get("ledger_path"))
    if declared:
        p = Path(declared)
        p = p if p.is_absolute() else repo_root / p
        require(p.resolve().is_relative_to(repo_root), "Ledger outside this repository rejected")
        ledger_candidates.append(p)
    else:
        ledger_candidates.append(work_path(repo_root, manifest) / "cpu_budget.json")
    observed = 0
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
                require(isinstance(row.get("actual_cpu_seconds"), (int, float)) and row["actual_cpu_seconds"] >= 0
                        and row.get("started_at_utc") and row.get("ended_at_utc") and row.get("failure")
                        and row.get("exit_code") is None and isinstance(row.get("command"), list),
                        "Incomplete failed-start accounting; defer Git return")
                log = Path(row.get("log", "")); proof = Path(row.get("supervisor_proof_path", ""))
                require(log.is_file() and proof.is_file() and row.get("log_sha256") == sha(log), "Failed-start source/log closure evidence missing")
                evidence = json.loads(proof.read_text())
                require(evidence.get("phase") == row.get("phase") and Path(evidence.get("ledger_path", "")).resolve() == p.resolve(), "Failed-start proof does not bind this real phase/ledger")
            else:
                require(status in closed, "Active, unknown or unresolved quantum phase; defer Git return")
    require(observed > 0, "No actual local quantum ledger; closure cannot be verified")
    require_no_live_quantum_processes()


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
            names = [v for v in scripts if any(s in v for s in ("pyscf", "cloud_followup_worker", "cloud_orbital_stability", "cloud_small_molecule", "parallel_b_small", "parallel_b_worker", "parallel_b_quantum")) and not v.startswith(("analyze_", "summarize_"))]
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


def claim_result_branch(repo_root, run_id, release_path):
    """Atomically allocate the single released remote B run before any Q.

    The first non-force remote branch creation wins. A public owner claim alone
    cannot authorize another clone: its private local ownership marker is also
    required, and no new CPU allowance is created by retrying this function.
    """
    repo_root = Path(repo_root).resolve(); release_path = Path(release_path).resolve()
    require(SAFE_ID.fullmatch(run_id) is not None, "Safe root-released run ID required")
    require(release_path.is_relative_to(repo_root), "Release outside repository rejected")
    release = json.loads(release_path.read_text()); release_sha = sha(release_path)
    require(release.get("release_enabled") is True and release.get("remote_run_id") == run_id, "Only the one exact ROOT-released remote allocation can be claimed")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(release["deadline_utc"]), "Original B authorization expired")
    for ref in release.get("source_refs", []):
        source = repo_root / ref["path"]
        require(source.resolve().is_relative_to(repo_root) and sha(source) == ref["sha256"], "Released source changed before allocation claim")
    branch = SOURCE_BRANCH_B + "-" + run_id
    require(git(repo_root, "branch", "--show-current") in (SOURCE_BRANCH_B, branch), "B release/per-run branch required for ownership claim")
    require(not git(repo_root, "diff", "--cached", "--name-only"), "Preserve preexisting staged work before ownership claim")
    own_prefix = f"reports/parallel-cloud-20261106/worker-b/{run_id}/"
    changed = [v for v in git(repo_root, "diff", "--name-only", "-z").split("\0") if v]
    require(all(v.startswith(own_prefix) for v in changed), "Tracked source or another run has changes; preserve it before ownership claim")
    release_blob = subprocess.run(["git", "-C", str(repo_root), "cat-file", "blob", "HEAD:" + str(release_path.relative_to(repo_root))], capture_output=True, timeout=30)
    require(release_blob.returncode == 0 and hashlib.sha256(release_blob.stdout).hexdigest() == release_sha, "Allocation release must be part of the actual published source lineage")
    validate_origin(repo_root); require_no_live_quantum_processes()
    work = work_path(repo_root, {"role": "B", "run_id": run_id, "work_directory": f"reports/parallel-cloud-20261106/worker-b/{run_id}"})
    marker = repo_root / ".runtime/worker-b-ownership" / (run_id + ".json")
    require(not any(p.is_symlink() for p in (repo_root / ".runtime", marker.parent, marker)), "Symlink ownership marker/prefix rejected")
    claim_file = work / "owner_claim.json"
    remote = git(repo_root, "ls-remote", "--heads", "origin", branch).splitlines()
    if remote:
        require(marker.is_file() and claim_file.is_file(), "BLOCK_ALREADY_CLAIMED: remote allocation belongs to another host; no local owner marker")
        saved = json.loads(marker.read_text()); public = json.loads(claim_file.read_text())
        require(all(saved.get(k) == public.get(k) and saved.get(k) is not None for k in ("run_id", "nonce", "release_sha256", "source_commit")), "BLOCK_ALREADY_CLAIMED: ownership proof mismatch")
        require(saved["run_id"] == run_id and saved["release_sha256"] == release_sha and saved.get("branch") == branch, "BLOCK_ALREADY_CLAIMED: local proof does not bind this released branch")
        require(saved.get("repository_root") == str(repo_root), "BLOCK_ALREADY_CLAIMED: local marker belongs to another checkout")
        if saved.get("remote_verified") is not True:
            # A successful push followed by a transport failure at ls-remote
            # may be repaired only by verifying this same local claim HEAD.
            local_head = git(repo_root, "rev-parse", "HEAD")
            require({p.name for p in work.iterdir()} == {"owner_claim.json"} and any(v.split()[0] == local_head for v in remote), "BLOCK_ALREADY_CLAIMED: unverified local nonce did not win the actual branch")
            blob = subprocess.run(["git", "-C", str(repo_root), "cat-file", "blob", local_head + ":" + str(claim_file.relative_to(repo_root))], capture_output=True, timeout=30)
            require(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == sha(claim_file), "Unverified claim blob differs from local ownership proof")
            saved.update(remote_verified=True, claim_commit=local_head, remote_head=local_head, verified_at_utc=now())
            marker.write_text(json.dumps(saved, indent=2) + "\n")
        if not (work / "cpu_budget_B.json").is_file():
            existing = {p.name for p in work.iterdir()}
            if existing != {"owner_claim.json"}:
                allowed = {"owner_claim.json", "manifest.json", "authorization.json", "prepared-launch.json"}
                require(existing == allowed, "Existing allocation lacks its actual CPU ledger; preserve for forensic recovery")
                manifest_file, auth_file = work / "manifest.json", work / "authorization.json"
                m, a = json.loads(manifest_file.read_text()), json.loads(auth_file.read_text())
                prepared = json.loads((work / "prepared-launch.json").read_text())
                require(m.get("run_id") == run_id and m.get("root_release", {}).get("sha256") == release_sha
                        and a.get("manifest_sha256") == sha(manifest_file)
                        and prepared.get("manifest_sha256") == sha(manifest_file)
                        and prepared.get("authorization_sha256") == sha(auth_file)
                        and prepared.get("new_CPU_consumed") == 0
                        and m.get("ownership", {}).get("local_marker_sha256") == sha(marker),
                        "Pre-compute resume metadata is not bound to the same release/owner; root review required")
        require(git(repo_root, "branch", "--show-current") == branch, "Existing owner must resume its original local result branch")
        return {"status": "EXISTING_LOCAL_OWNER_VERIFIED_NO_NEW_ALLOCATION", "branch": branch, "owner_claim": str(claim_file), "local_marker": str(marker)}
    # Retry a failed pre-compute claim push without generating another nonce.
    if marker.exists() or claim_file.exists():
        require(marker.is_file() and claim_file.is_file(), "Incomplete local claim; root review required")
        saved = json.loads(marker.read_text()); public = json.loads(claim_file.read_text())
        require(saved.get("repository_root") == str(repo_root) and all(saved.get(k) == public.get(k) for k in ("run_id", "nonce", "release_sha256", "source_commit")) and saved.get("release_sha256") == release_sha, "Local failed-claim lineage mismatch")
        require({p.name for p in work.iterdir()} == {"owner_claim.json"} and not saved.get("remote_verified"), "Only a preserved pre-compute claim may retry transport")
        require(git(repo_root, "branch", "--show-current") == branch, "Failed local claim branch mismatch")
    else:
        require(not work.exists(), "Fresh remote claim must not overwrite a run directory")
        source_commit = git(repo_root, "rev-parse", "HEAD")
        source_heads = git(repo_root, "ls-remote", "--heads", "origin", SOURCE_BRANCH_B).splitlines()
        require(any(line.split()[0] == source_commit for line in source_heads), "Checkout must equal the actual published B release before claiming")
        git(repo_root, "push", "--dry-run", "origin", "HEAD:refs/heads/" + branch)
        if git(repo_root, "branch", "--show-current") != branch:
            git(repo_root, "switch", "-c", branch)
        work.mkdir(parents=True); marker.parent.mkdir(parents=True, exist_ok=True)
        public = {"schema": "research4-remote-b-owner-claim-v1", "run_id": run_id, "nonce": secrets.token_hex(24),
                  "release_sha256": release_sha, "source_commit": source_commit, "created_at_utc": now(),
                  "owner_uid": os.getuid(), "owner_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  "new_CPU_consumed": 0, "not_a_quantum_result": True}
        with claim_file.open("x") as handle: handle.write(json.dumps(public, indent=2) + "\n")
        saved = {**public, "repository_root": str(repo_root), "branch": branch, "remote_verified": False}
        with marker.open("x") as handle: handle.write(json.dumps(saved, indent=2) + "\n")
        git(repo_root, "add", "--sparse", "--", str(claim_file.relative_to(repo_root)))
        require(set(git(repo_root, "diff", "--cached", "--name-only").splitlines()) == {str(claim_file.relative_to(repo_root))}, "Only ownership claim may be staged")
        git(repo_root, "commit", "-m", "Claim the single released remote B allocation " + run_id)
    require_no_live_quantum_processes()
    claim_commit = git(repo_root, "rev-parse", "HEAD")
    try:
        git(repo_root, "push", "origin", "HEAD:refs/heads/" + branch)
    except SafetyError as error:
        raise SafetyError("B claim push failed or another host won; no Q may start. Preserve local claim/marker; never force-push.") from error
    confirmed = git(repo_root, "ls-remote", "--heads", "origin", branch).splitlines()
    require(any(v.split()[0] == claim_commit for v in confirmed), "B ownership remote head did not verify")
    saved.update(remote_verified=True, claim_commit=claim_commit, remote_head=claim_commit, verified_at_utc=now())
    marker.write_text(json.dumps(saved, indent=2) + "\n")
    return {"status": "DONE_ATOMIC_REMOTE_BRANCH_CLAIM_NO_Q", "branch": branch, "claim_commit": claim_commit,
            "owner_claim": str(claim_file), "local_marker": str(marker)}


def execute_return(repo_root, manifest, manifest_path, manifest_sha256, attempt="return-001"):
    repo_root = Path(repo_root).resolve(); manifest_path = Path(manifest_path).resolve()
    require(SAFE_ID.fullmatch(attempt) is not None, "Safe fresh return attempt required")
    require(manifest_path.is_relative_to(repo_root) and sha(manifest_path) == manifest_sha256, "Actual manifest/source binding required")
    require(json.loads(manifest_path.read_text()) == manifest, "Supplied manifest differs from bound file")
    branch = expected_branch(manifest)
    require(git(repo_root, "branch", "--show-current") == branch, "Current branch does not match the exact role/run result branch; main is forbidden")
    require(not git(repo_root, "diff", "--cached", "--name-only"), "Preexisting staged work must not be included")
    validate_origin(repo_root); require_no_active_quantum(repo_root, manifest)
    files = collect_return_files(repo_root, manifest)
    snapshots = {str(p.relative_to(repo_root)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}
    receipt_dir = work_path(repo_root, manifest) / "git-return" / attempt
    require(not receipt_dir.parent.is_symlink() and receipt_dir.resolve().is_relative_to(work_path(repo_root, manifest)), "Symlink or escaping return-receipt directory rejected")
    require(not receipt_dir.exists(), "Preserve earlier upload attempts and receipts")
    receipt_dir.mkdir(parents=True)
    receipt = {"schema": "research4-actual-result-return-v1", "started_at_utc": now(), "branch": branch,
               "manifest_sha256": manifest_sha256, "source_helper_sha256": sha(__file__), "artifact_hashes": snapshots,
               "force_push": False, "main_modified": False, "actual_result_upload": "NOT_YET_PERFORMED"}
    receipt_path = receipt_dir / "return_receipt.json"
    try:
        require_no_active_quantum(repo_root, manifest)
        git(repo_root, "add", "--sparse", "--", *snapshots)
        staged = set(git(repo_root, "diff", "--cached", "--name-only").splitlines())
        require(staged.issubset(snapshots), "Staging differs from explicit allowlist")
        for rel, saved in snapshots.items():
            require(sha(repo_root / rel) == saved["sha256"], "Output changed while preparing return")
        git(repo_root, "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol", "diff", "--cached", "--check")
        if staged:
            git(repo_root, "commit", "-m", f"Save {manifest.get('role', 'B')} {manifest['run_id']} actual bounded cloud results")
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
        git(repo_root, "commit", "-m", f"Record actual {manifest.get('role', 'B')} {manifest['run_id']} result push receipt")
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
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--claim-run-id")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--attempt", default="return-001")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.claim_run_id:
        require(args.execute and args.release is not None and args.manifest is None, "Explicit claim execution/release required; no mixed return action")
        print(json.dumps(claim_result_branch(repo, args.claim_run_id, args.release))); return
    require(args.manifest is not None and args.manifest_sha256 is not None, "Explicit result manifest/hash required")
    source = args.manifest.resolve()
    require(sha(source) == args.manifest_sha256, "Manifest hash mismatch")
    manifest = json.loads(source.read_text())
    if args.execute:
        print(json.dumps(execute_return(repo, manifest, source, args.manifest_sha256, args.attempt)))
    else:
        require(git(repo, "branch", "--show-current") == expected_branch(manifest), "Current branch mismatch")
        files = collect_return_files(repo, manifest)
        print(json.dumps({"status": "CHECK_ONLY_NO_GIT_WRITE_NO_UPLOAD", "files": len(files), "bytes": sum(p.stat().st_size for p in files)}))


if __name__ == "__main__":
    main()
