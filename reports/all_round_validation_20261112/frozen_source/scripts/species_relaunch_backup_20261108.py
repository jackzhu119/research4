"""Publish bounded, sealed TRAIN-only evidence; never run scientific work.

Invoked only as a serial budget-supervised job. Network failure is recorded as
NOT_BACKED_UP, not retried here. Live ledgers are projected, never staged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
BRANCH = "research/species-relaunch-90000-20261108"
BASE = "data/research/species_relaunch_20261108_v2"
FILE_CAP, DELTA_CAP = 10 * 1024**2, 40 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def inside(path, root):
    path = Path(path).resolve()
    need(path.is_relative_to(root), "Artifact outside approved run root")
    return path


def permitted(path):
    name = path.name.lower()
    return (not any(word in name for word in (".partial", ".tmp", ".lock"))
            and (path.suffix.lower() in {".json", ".log", ".md", ".csv", ".pt", ".joblib", ".png", ".pdf"}
                 or name.endswith((".csv.gz", ".jsonl.gz"))))


def git(*args):
    result = subprocess.run(["git", *args], cwd=REPO, shell=False, timeout=60,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    need(result.returncode == 0, "Git operation failed: " + args[0])
    return result.stdout


def remote_allowed(url):
    if url.startswith("git@github.com:"):
        return url.removeprefix("git@github.com:").removesuffix(".git") == "jackzhu119/research4"
    parsed = urlsplit(url)
    return (parsed.scheme in {"https", "ssh"} and parsed.hostname == "github.com"
            and parsed.path.strip("/").removesuffix(".git") == "jackzhu119/research4")


def candidates(root, output):
    protocol, lock, manifest = [read(root / p) for p in ("protocol.json", "input_lock.json", "jobs.json")]
    state_bytes = (root / "control/state.json").read_bytes()
    state = json.loads(state_bytes)
    need(manifest["protocol_sha256"] == sha(root / "protocol.json"), "Protocol hash mismatch")
    need(protocol["input_lock_sha256"] == sha(root / "input_lock.json"), "Input lock hash mismatch")
    need(protocol["worker_code_sha256"][str(Path(__file__).resolve().relative_to(REPO))] == sha(__file__), "Publisher source drift")
    need(sha(root / "train_only_features.npz") == lock["training_feature_sha256"]
         and sha(root / "train_only_split.csv") == lock["training_split_sha256"], "TRAIN input drift")
    jobs = manifest["jobs"]
    need(state["manifest_sha256"] == objhash({"manifest": manifest, "normalized_jobs": jobs,
                                            "budget_cpu_seconds": 88000.}), "Frozen manifest mismatch")
    current = [j for j in jobs if j.get("kind") == "backup" and str(output / "receipt.json") in j["outputs"]]
    need(len(current) == 1 and current[0]["max_retries"] == 0, "Undeclared backup destination")
    current_id = current[0]["id"]
    need(not state.get("active") or state["active"]["job_id"] == current_id, "Another queue job is active")
    need(all(v.get("status") != "running" or key == current_id for key, v in state["jobs"].items()), "Other job marked running")
    selected, evidence = {}, []
    hash_cache = {}

    def digest(path):
        path = Path(path).resolve()
        need(path.is_relative_to(REPO), "Digest outside repository")
        if path not in hash_cache:
            hash_cache[path] = sha(path)
        return hash_cache[path]

    def add(path, expected=None):
        path = inside(path, root)
        need(path.is_file(), "Sealed artifact missing")
        actual = digest(path)
        need(expected is None or actual == expected, "Sealed artifact hash mismatch")
        selected[str(path.relative_to(REPO))] = {"sha256": actual, "bytes": path.stat().st_size,
                                                "eligible": permitted(path) and path.stat().st_size <= FILE_CAP}

    for job in jobs:
        record = state["jobs"].get(job["id"], {})
        if record.get("status") != "completed":
            continue
        receipt_path = root / "control/receipts" / (job["id"] + ".json")
        receipt = read(receipt_path)
        need(digest(receipt_path) == record["receipt_sha256"] and receipt["job_sha256"] == objhash(job), "Job receipt mismatch")
        need(receipt["input_hashes"] == state["input_hashes"][job["id"]], "Job input seal mismatch")
        for path, expected in receipt["input_hashes"].items():
            # Scientific source inputs live outside this run root.
            need(digest(path) == expected, "Closed-job input changed")
        add(receipt_path); add(receipt["log_path"], receipt["log_sha256"])
        for path, expected in receipt["output_hashes"].items():
            add(path, expected)
            if Path(path).name == "complete.json":
                done = read(path)
                for name, checksum in done.get("artifacts_sha256", done.get("artifacts", {})).items():
                    add(Path(path).parent / name, checksum)
        for path in receipt["output_hashes"]:
            if Path(path).name == "summary.json":
                for model in read(path).get("model_receipts", {}).values():
                    cache = inside(model["cache_path"], root)
                    add(cache / "complete.json", model["complete_sha256"])
                    for name, checksum in read(cache / "complete.json")["artifacts"].items():
                        add(cache / name, checksum)
        evidence.append({"job_id": job["id"], "receipt_sha256": record["receipt_sha256"]})
    projection = {"observed_utc": datetime.now(timezone.utc).isoformat(), "not_terminal_live_ledger_projection": True,
                  "source_hashes": {"protocol": sha(root / "protocol.json"), "manifest": sha(root / "jobs.json"),
                                    "input_lock": sha(root / "input_lock.json")},
                  "completed_jobs": evidence, "state_file_sha256_at_read": hashlib.sha256(state_bytes).hexdigest(),
                  "state": {key: state.get(key) for key in ("run_id", "status", "budget_cpu_seconds",
                      "spent_cpu_seconds", "remaining_cpu_seconds", "heartbeat_unix", "owner", "active", "termination_reason")},
                  "job_statuses": {key: value.get("status") for key, value in state["jobs"].items()},
                  "preparation_cpu": read(root / "preparation_cpu.json")}
    write(output / "budget_progress_snapshot.json", projection)
    add(output / "budget_progress_snapshot.json")
    return selected


def publish(selected, execute):
    need(git("branch", "--show-current").decode().strip() == BRANCH, "Research branch mismatch")
    need(remote_allowed(git("remote", "get-url", "origin").decode().strip()), "Remote repository mismatch")
    need(not git("diff", "--cached", "--name-only", "-z"), "Existing staged files; index left untouched")
    before = git("rev-parse", "HEAD").decode().strip()
    algorithm = git("rev-parse", "--show-object-format").decode().strip()
    need(algorithm in {"sha1", "sha256"}, "Unsupported Git object format")
    tracked = {}
    for entry in git("ls-tree", "-rz", "HEAD", "--", BASE).split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            tracked[path.decode()] = metadata.split()[2].decode()
    chosen, omitted, existing, total = {}, {}, {}, 0
    for name, info in sorted(selected.items()):
        if not info["eligible"]:
            omitted[name] = {**info, "reason": "file_size_or_extension_policy"}; continue
        content = (REPO / name).read_bytes()
        blob = hashlib.new(algorithm, ("blob %d\0" % len(content)).encode() + content).hexdigest()
        if name in tracked:
            need(tracked[name] == blob, "Committed immutable evidence changed")
            existing[name] = {**info, "git_blob": blob}; continue
        if total + info["bytes"] > DELTA_CAP:
            omitted[name] = {**info, "reason": "delta_cap"}; continue
        chosen[name] = {**info, "git_blob": blob}; total += info["bytes"]
    if not execute:
        return {"status": "NOT_RUN", "selected": chosen, "not_backed_up": omitted, "already_committed": existing}
    if chosen:
        need(git("branch", "--show-current").decode().strip() == BRANCH
             and git("rev-parse", "HEAD").decode().strip() == before, "Branch or HEAD changed before staging")
        git("add", "--", *chosen)
        staged = set(git("diff", "--cached", "--name-only", "-z").decode().strip("\0").split("\0"))
        need(staged == set(chosen), "Unexpected staging; no commit performed")
        need(git("branch", "--show-current").decode().strip() == BRANCH
             and git("rev-parse", "HEAD").decode().strip() == before, "Branch or HEAD changed before commit")
        git("commit", "-m", "research: preserve sealed species relaunch development evidence")
    commit = git("rev-parse", "HEAD").decode().strip()
    for name, info in {**existing, **chosen}.items():
        need(git("rev-parse", commit + ":" + name).decode().strip() == info["git_blob"], "Committed Git blob mismatch")
    need(git("branch", "--show-current").decode().strip() == BRANCH
         and git("rev-parse", "HEAD").decode().strip() == commit, "Branch or HEAD changed before push")
    git("push", "origin", commit + ":refs/heads/" + BRANCH)
    remote = git("ls-remote", "origin", "refs/heads/" + BRANCH).decode().split()
    need(len(remote) == 2 and remote[0] == commit, "Remote head not independently verified")
    return {"status": "DONE", "before_commit": before, "verified_remote_commit": commit,
            "published_new": chosen, "already_committed": existing, "not_backed_up": omitted,
            "delta_bytes": total, "scope": "Only listed immutable artifacts; not all run data or this live receipt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); root, output = args.root.resolve(), args.output.resolve()
    need(root == REPO / BASE and output.is_relative_to(root / "backups"), "Unapproved root/output")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    selected = {}
    try:
        selected = candidates(root, output)
        result = publish(selected, args.execute)
    except Exception as error:
        # Never print Git stderr, URLs, tokens or arbitrary exception content.
        result = {"status": "NOT_BACKED_UP", "failure_type": type(error).__name__,
                  "scientific_retry_requested": False, "remote_verified": False,
                  "local_commit_may_have_completed": True, "candidate_hashes_not_verified_remote": selected,
                  "detail": str(error) if isinstance(error, ValueError) else "Bounded operation failed"}
    result.update(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), branch=BRANCH)
    write(output / "receipt.json", result)
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "complete.json"}
    write(output / "complete.json", {"status": "complete", "backup_status": result["status"], "artifacts": artifacts})
    print(json.dumps({"backup_status": result["status"], "receipt": str(output / "receipt.json")}))
    # A closed backup failure does not restart any expensive scientific job.


if __name__ == "__main__":
    main()
