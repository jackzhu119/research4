"""Preserve the prelaunch candidate and require regrid centers in v4 revision2.

Only depends_on is changed for signed regrid points. No prior artifact is
rewritten and no allocation is registered, science run or Git command invoked.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "reports/parallel-cloud-20261106/worker-b/B20261106-01/followup-validation-v4"


def main():
    source = ROOT / "scripts/research4_b_validation_supervisor_20261107.py"
    spec = importlib.util.spec_from_file_location("r4_validation_dep_revision", source)
    s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
    s.require(not (WORK / "cpu_budget.json").exists() and not (WORK / "state.json").exists()
              and not (ROOT / ".runtime/worker-b-validation-allocation.json").exists(),
              "A launched allocation cannot change manifest")
    old_spec_path = WORK / "reviewed_spec_ROOT_AUTHORIZED.json"
    old_manifest_path = WORK / "manifest_ROOT_AUTHORIZED.json"
    s.require(s.sha(old_spec_path) == "9347b094aafbe1a1c51cc763eeff7e6545b4c4037197e85fdf2b5da0e6e1fe8a"
              and s.sha(old_manifest_path) == "5979494426548bf9d19b2e67d20148997356a3188042f52889e7788f4d4846fe",
              "Pinned prelaunch candidate changed")
    old_spec = s.read(old_spec_path); manifest = s.read(old_manifest_path)
    s.validate(manifest, s.sha(old_manifest_path))
    revised = json.loads(json.dumps(old_spec)); changes = []
    ids = {j["id"] for j in revised["jobs"]}
    for job in revised["jobs"]:
        if "-regrid-" in job["id"] and job["id"].endswith(("-minus", "-plus")):
            center = job["id"].split("-regrid-")[0] + "-regrid-center"
            s.require(center in ids and not job["depends_on"], "Unexpected candidate dependency scope")
            changes.append({"job_id": job["id"], "old_depends_on": [], "new_depends_on": [center]})
            job["depends_on"] = [center]
    s.require(len(changes) == 28, "Expected exactly28 signed regrid points")
    snapshot = WORK / "preparation/source_snapshots" / (Path(__file__).name + ".txt")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as handle: handle.write(Path(__file__).read_bytes())
    revision_path = WORK / "preparation/dependency_revision_v2.json"
    s.save(revision_path, {"schema": "research4-b-validation-prelaunch-dependency-revision-v2",
        "reason": "Require every signed fixed/regrid point to depend on a successful matching center; preserve SCF qualification failures and avoid spending points without qualified E0.",
        "old_spec": s.ref(old_spec_path), "old_manifest": s.ref(old_manifest_path), "changes": changes,
        "source": s.ref(__file__), "source_snapshot": s.ref(snapshot), "new_quantum_evaluations": 0,
        "old_candidate_superseded_never_launched": True}, exclusive=True)
    revised["source_refs"].append(s.ref(__file__))
    revised["input_refs"].extend([s.ref(old_spec_path), s.ref(old_manifest_path), s.ref(snapshot), s.ref(revision_path)])
    new_spec_path = WORK / "reviewed_spec_v2_ROOT_AUTHORIZED.json"; s.save(new_spec_path, revised, exclusive=True)
    manifest["jobs"] = revised["jobs"]
    manifest["source_refs"].append(s.ref(__file__))
    manifest["input_refs"].extend([s.ref(old_manifest_path), s.ref(new_spec_path), s.ref(snapshot), s.ref(revision_path)])
    manifest["prelaunch_dependency_revision"] = s.ref(revision_path)
    target = WORK / "manifest_v2_ROOT_AUTHORIZED.json"; s.save(target, manifest, exclusive=True)
    s.validate(manifest, s.sha(target))
    s.save(WORK / "preparation/active_manifest_selection.json", {
        "schema": "research4-b-validation-active-manifest-selection-v2",
        "superseded_candidate": {"manifest": s.ref(old_manifest_path), "status": "NOT_EXECUTED_SUPERSEDED"},
        "active_manifest": s.ref(target), "active_spec": s.ref(new_spec_path),
        "launch_current_receipt_and_return_must_use_active_manifest": True,
        "allocation_claim_and_ledger_absent_at_selection": True, "new_quantum_evaluations": 0}, exclusive=True)
    print(json.dumps({"status": "REVISED_PREPARED_NOT_RUN", "manifest": s.ref(target), "spec": s.ref(new_spec_path),
                      "dependency_changes": len(changes), "new_quantum_evaluations": 0, "git_mutations": 0}, indent=2))


if __name__ == "__main__":
    main()
