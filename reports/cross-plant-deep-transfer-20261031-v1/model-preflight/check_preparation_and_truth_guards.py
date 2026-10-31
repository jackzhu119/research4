"""Zero-fit preparation, membership and actual named-truth-hook checks."""
from pathlib import Path
import json
import numpy as np
from herb_spectra import cross_plant_transfer as trial
from herb_spectra.leakage_guard import LeakageError

out = Path("reports/cross-plant-deep-transfer-20261031-v1/neural")
destination = out.parent / "model-preflight"
prep = trial.verify_run(out)
checks = 0
for context in prep["contexts"]:
    source = trial.read_pack(out / context["source_pack"], True)
    query = trial.read_pack(out / context["query_pack"], False)
    assert source["record_ids"].tolist() == context["source_ids"]
    assert query["record_ids"].tolist() == context["query_ids"]
    assert not set(context["source_ids"]) & set(context["query_ids"])
    assert len(source["X"]) == (342 if context["stage"] == "inner" else 684)
    assert len(query["X"]) == (342 if context["stage"] == "inner" else 171)
    assert tuple(np.unique(source["source_labels"])) == trial.CLASSES
    checks += 6
for name, pool in prep["auxiliary"].items():
    pack = trial.read_pack(out / pool["path"], True)
    assert pack["record_ids"].tolist() == pool["record_ids"]
    assert np.unique(pack["source_labels"]).tolist() == pool["class_order"]
    assert len(pack["X"]) == {"huangqi": 403, "tianma": 405, "ginseng": 42}[name]
    checks += 3
paths = [out / context["truth_path"] for context in prep["contexts"]]
before = {str(path): trial.sha(path) for path in paths}
with trial.forbid_named_truth(paths) as attempts:
    for path in paths:
        for mode in ("rb", "ab"):
            try:
                path.open(mode)
            except LeakageError:
                checks += 1
            else:
                raise AssertionError("Actual named truth-file guard did not reject " + mode)
after = {str(path): trial.sha(path) for path in paths}
assert before == after and len(attempts) == 30
checks += 2
result = {"completed_at_utc": trial.utc(), "status": "TECHNICAL_PREPARATION_AND_NAMED_HOOK_PASS_EXPLORATORY_ONLY",
          "checks": checks, "research_fits": 0, "contexts": len(prep["contexts"]),
          "real_named_read_attempts_rejected": 15, "real_named_write_attempts_rejected": 15,
          "truth_hashes_before": before, "truth_hashes_after": after,
          "all_original_truth_bytes_unchanged": True, "denied_attempts": attempts,
          "OS_sandbox": False, "source_versions": trial.source_versions(),
          "run_manifest_sha256": trial.sha(out / "run_manifest.json")}
trial.write(destination / "prepared_boundary_checks.json", result)
print(json.dumps({k: result[k] for k in ("status", "research_fits", "checks", "contexts")}))
