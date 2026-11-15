#!/usr/bin/env python3
"""Reject withdrawn mixed-kind classifier outputs in a current Git tree.

This check reads tree metadata and the existing withdrawal rules. It neither
fetches nor checks out historical results, changes spectra, or fits models.
Corrected 1305 aliases such as OLD-* are not rejected by their names.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RULE_PATH = ROOT / "maintenance/withdrawn_results_rules.json"


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def withdrawn_reason(path, blob, rules):
    if path in rules["delete_files"]:
        return "withdrawn_file"
    if any(path.startswith(prefix) for prefix in rules["delete_prefixes"]):
        return "withdrawn_prefix"
    if any(re.search(pattern, path) for pattern in rules["delete_regexes"]):
        return "withdrawn_pattern"
    if rules["obsolete_generator_blob_only"].get(path) == blob:
        return "withdrawn_generator_blob"
    return None


def inspect(ref, rules):
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    matches = []
    count = 0
    for record in git("ls-tree", "-r", "-z", sha).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, kind, raw_blob = metadata.split()
        if kind != b"blob":
            continue
        count += 1
        path = raw_path.decode("utf-8")
        blob = raw_blob.decode("ascii")
        reason = withdrawn_reason(path, blob, rules)
        if reason:
            matches.append({"path": path, "blob": blob, "reason": reason})
    return {
        "ref": ref,
        "commit": sha,
        "tracked_files": count,
        "withdrawn_matches": matches,
        "status": "PASS" if not matches else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", action="append", help="Current branch/ref to inspect; repeatable, default HEAD")
    parser.add_argument("--output", type=Path, help="Write a JSON receipt")
    arguments = parser.parse_args()
    rule_bytes = RULE_PATH.read_bytes()
    rules = json.loads(rule_bytes)
    results = [inspect(ref, rules) for ref in arguments.ref or ["HEAD"]]
    receipt = {
        "schema": "research4-withdrawn-result-reentry-check-v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "rules": str(RULE_PATH.relative_to(ROOT)),
        "rules_sha256": hashlib.sha256(rule_bytes).hexdigest(),
        "trees": results,
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "scope": "Specified current Git trees against the approved path and generator-blob withdrawal rules; no model rerun, remote fetch or historical checkout.",
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(f"{row['status']} {row['ref']} {row['commit']}: {row['tracked_files']} files, {len(row['withdrawn_matches'])} withdrawn matches")
        for match in row["withdrawn_matches"]:
            print(f"  {match['reason']}: {match['path']}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
