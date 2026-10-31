"""Read-only byte audit against the previous research commit, not a Git-index check."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = "ea7b6b7917c91aae43e3f2ded0beee269bfc3a1a"
OUT = Path(__file__).with_name("protected_base_audit.json")
started = datetime.now(timezone.utc).isoformat()
tree = subprocess.check_output(["git", "ls-tree", "-rlz", BASE], cwd=ROOT)
items = []
for line in tree.split(b"\0"):
    if not line:
        continue
    info, name = line.split(b"\t", 1)
    mode, kind, sha, size = info.split()
    if kind != b"blob":
        raise RuntimeError("Unexpected non-blob base tree entry")
    items.append((os.fsdecode(name), mode.decode(), sha.decode(), int(size)))


def inspect(item):
    name, mode, expected, old_size = item
    p = ROOT / name
    allowed = name in ("AGENTS.md", "README.md") or name.startswith("docs/")
    try:
        if mode == "120000":
            if not p.is_symlink():
                raise RuntimeError("Expected symlink changed type")
            data = os.fsencode(os.readlink(p))
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            current_mode = "120000"
        else:
            if p.is_symlink() or not p.is_file():
                raise RuntimeError("Base regular file missing or changed type")
            size = p.stat().st_size
            h = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            actual = h.hexdigest()
            current_mode = "100755" if p.stat().st_mode & 0o111 else "100644"
        changed = actual != expected or current_mode != mode
        return {"path": name, "allowed_documentation_update": allowed, "changed": changed,
                "base_blob": expected, "actual_git_blob": actual,
                "base_mode": mode, "actual_mode": current_mode, "base_bytes": old_size}
    except Exception as e:
        return {"path": name, "allowed_documentation_update": allowed, "changed": True,
                "base_blob": expected, "actual_git_blob": None, "error": str(e), "base_bytes": old_size}


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(inspect, items))
changed = [v for v in results if v["changed"]]
violations = [v for v in changed if not v["allowed_documentation_update"]]
report = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
          "base_commit": BASE, "base_tree_sha256": hashlib.sha256(tree).hexdigest(),
          "actual_git_blob_hashes_computed_from_worktree_bytes": True,
          "base_blob_count": len(items), "base_blob_bytes": sum(v[3] for v in items),
          "unchanged_blob_count": sum(not v["changed"] for v in results),
          "allowed_documentation_changes": [v for v in changed if v["allowed_documentation_update"]],
          "protected_data_history_model_source_script_test_changes": violations,
          "status": "PASS" if not violations else "FAIL", "new_research_fits": 0,
          "git_index_mutated": False, "main_merged": False}
with OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({k: v for k, v in report.items() if k not in
                  ("allowed_documentation_changes", "protected_data_history_model_source_script_test_changes")}, indent=2))
if violations:
    raise SystemExit(1)
