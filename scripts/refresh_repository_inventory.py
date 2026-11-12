#!/usr/bin/env python3
"""Refresh Git metadata in the repository inventory without contacting a remote.

Fetch refs separately before running this script. Commit dates describe Git history,
not the time a training job was observed. Curated experiment statuses are preserved.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
REMOTE_PREFIX = "refs/remotes/origin/"


def git(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY), *arguments], text=True
    ).rstrip("\n")


def collect_refs(exclusions):
    refs = {}
    lines = git(
        "for-each-ref",
        "--format=%(refname)%09%(objectname)%09%(tree)%09%(committerdate:iso-strict)%09%(parent)%09%(subject)",
        REMOTE_PREFIX,
    ).splitlines()
    main_ref = REMOTE_PREFIX + "main"
    main_sha = git("rev-parse", "--verify", main_ref)
    trees = defaultdict(list)
    for line in lines:
        ref, sha, tree_sha, commit_time, parents, subject = line.split("\t", 5)
        name = ref.removeprefix(REMOTE_PREFIX)
        if name == "HEAD" or any(name.startswith(prefix) for prefix in exclusions):
            continue
        behind, ahead = map(
            int, git("rev-list", "--left-right", "--count", f"{main_ref}...{ref}").split()
        )
        refs[name] = {
            "name": name,
            "sha": sha,
            "tree_sha": tree_sha,
            "commit_time": commit_time,
            "parents": parents.split(),
            "subject": subject,
            "ahead_main": ahead,
            "behind_main": behind,
            "merged_into_main": ahead == 0,
            "ref_state": "present",
        }
        trees[tree_sha].append(name)
    for name, metadata in refs.items():
        metadata["identical_tree_to"] = sorted(
            other for other in trees[metadata["tree_sha"]] if other != name
        )
    return refs, main_sha


def read_inventory(path):
    if not path.exists():
        return {"branches": []}
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict) or not isinstance(inventory.get("branches"), list):
        raise ValueError("inventory must be an object with a branches array")
    records = inventory["branches"]
    if any(not isinstance(row, dict) or not isinstance(row.get("name"), str) for row in records):
        raise ValueError("each branch must have a string name")
    if len({row["name"] for row in records}) != len(records):
        raise ValueError("inventory contains duplicate branch names")
    return inventory


def check_inventory(inventory, refs, exclusions):
    records = {row["name"]: row for row in inventory["branches"]}
    problems = []
    for name, metadata in refs.items():
        if name not in records:
            problems.append(f"missing branch: {name}")
        elif records[name].get("sha") != metadata["sha"]:
            problems.append(f"SHA changed: {name}")
        elif records[name].get("ref_state") == "not_in_local_refs":
            problems.append(f"branch has returned to local refs: {name}")
    for name, row in records.items():
        if any(name.startswith(prefix) for prefix in exclusions):
            continue
        if name not in refs and row.get("ref_state") != "not_in_local_refs":
            problems.append(f"absent local ref needs review: {name}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Inventory covers {len(refs)} local origin branches at their recorded SHAs.")
    print("This checks local Git refs only; training statuses were not checked.")
    return 0


def refresh_inventory(inventory, refs, main_sha, exclusions, path):
    records = {row["name"]: row for row in inventory["branches"]}
    for name, metadata in refs.items():
        record = records.setdefault(name, {"name": name, "status": "needs_review"})
        record.update(metadata)
    for name, record in records.items():
        if name not in refs and not any(name.startswith(prefix) for prefix in exclusions):
            record["ref_state"] = "not_in_local_refs"
    inventory["branches"] = [records[name] for name in sorted(records)]
    inventory["branch_count"] = len(refs)
    inventory["main_sha"] = main_sha
    inventory["exclude_branches"] = exclusions
    inventory["git_inventory_updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        json.dump(inventory, output, ensure_ascii=False, indent=2)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)
    print(f"Updated Git metadata for {len(refs)} local origin branches: {path}")
    print("Curated training statuses and other manual fields were preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "docs/repository/branches.json")
    parser.add_argument("--check", action="store_true", help="check branch coverage and SHAs without writing")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="PREFIX",
        help="exclude branch-name prefixes; repeatable and combined with exclude_branches in the inventory",
    )
    arguments = parser.parse_args()
    try:
        if arguments.check and not arguments.output.exists():
            raise ValueError(f"inventory does not exist: {arguments.output}")
        inventory = read_inventory(arguments.output)
        stored_exclusions = inventory.get("exclude_branches", [])
        if not isinstance(stored_exclusions, list) or any(not isinstance(item, str) or not item for item in stored_exclusions):
            raise ValueError("exclude_branches must be a list of nonempty branch-name prefixes")
        if any(not prefix for prefix in arguments.exclude):
            raise ValueError("--exclude requires a nonempty prefix")
        exclusions = sorted(set(stored_exclusions + arguments.exclude))
        refs, main_sha = collect_refs(exclusions)
        if arguments.check:
            return check_inventory(inventory, refs, exclusions)
        refresh_inventory(inventory, refs, main_sha, exclusions, arguments.output)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Cannot refresh repository inventory: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
