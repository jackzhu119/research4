"""Read-only review of frozen small empirical-control implementation/inputs."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "neural"
DIAG = HERE.parent / "empirical/prepared"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)
    manifest = read(DIAG / "frozen_manifest.json")
    check(sha(DIAG / "frozen_manifest.json") == "2d06a5fcfc61f4d2feed3e118f711c38c19114a03106290eb1f45dc05d9ee85a", "frozen diagnostic manifest")
    for relative, expected in manifest["files_sha256"].items():
        check(sha(DIAG / relative) == expected, "frozen input " + relative)
    for relative, expected in manifest["source_sha256"].items():
        check(sha(ROOT / relative) == expected, "frozen implementation " + relative)
    for absolute, expected in manifest["main_files_sha256"].items():
        check(sha(absolute) == expected, "parent main input bindings")
    plan = read(DIAG / "plan.json")
    check(plan["outer_fold"] == 0 and plan["seed"] == 42, "one fixed fold/seed")
    check(plan["arms"] == ["A", "Aextra", "B", "C"], "all four arms")
    check(plan["target_profiles"] == [{"id": "diagnostic2", "epochs": 2}, {"id": "diagnostic4", "epochs": 4}], "small fixed budget")
    check(plan["pretrain_steps"] == 4 and plan["pretrain_batch_size"] == 32, "pretraining small fixed budget")
    check(plan["pipeline_budget"] == {"inner_per_repeat": 16, "outer_per_repeat": 4,
          "per_repeat": 20, "repeats": 12, "total": 240}, "actual declared candidate count")
    check(plan["pretrained_trajectories"] == 180 and plan["pretrain_optimizer_steps"] == 720, "separate pretraining count")
    check(plan["null_permutation_seeds"] == list(range(1000, 1010)), "all fixed null seeds")
    check(plan["all_12_prediction_locks_before_scoring"] is True and plan["main_candidates_and_budget_unchanged_by_diagnostic"] is True,
          "locks and unchanged main budget")
    prep, main_prep = read(DIAG / "preparation.json"), read(MAIN / "preparation.json")
    expected = {c["split_id"]: c for c in main_prep["contexts"] if c["outer_fold"] == 0}
    check(len(prep["contexts"]) == 3 and {c["split_id"] for c in prep["contexts"]} == set(expected), "three exact contexts")
    for context in prep["contexts"]:
        original = expected[context["split_id"]]
        # The diagnostic intentionally omits main per-context hashes and truth
        # paths: copied input hashes live in its top frozen manifest; source
        # labels for inner scoring come from the684 current-source mapping.
        expected_context = {k: v for k, v in original.items()
                            if k not in {"source_sha256", "query_sha256", "truth_path", "truth_sha256"}}
        check(context == expected_context, "diagnostic routing metadata matches main")
        for role in ("source", "query"):
            check((DIAG / context[role + "_pack"]).read_bytes() == (MAIN / original[role + "_pack"]).read_bytes(), "original source/query bytes copied exactly")
    for name, pool in prep["auxiliary"].items():
        check((DIAG / pool["path"]).read_bytes() == (MAIN / main_prep["auxiliary"][name]["path"]).read_bytes(), "source-only auxiliary bytes")
    outer = next(c for c in prep["contexts"] if c["stage"] == "outer")
    with np.load(DIAG / outer["source_pack"], allow_pickle=False) as archive:
        source_ids = archive["record_ids"].astype(str).tolist()
        source_labels = archive["source_labels"].astype(str).tolist()
    with np.load(DIAG / outer["query_pack"], allow_pickle=False) as archive:
        query_ids = archive["record_ids"].astype(str).tolist()
        check(set(archive.files) == {"X", "axis", "record_ids"}, "query contains no labels")
    blocks = prep["inner_query_blocks"]
    check(len(blocks) == 2 and len(blocks[0]) == len(blocks[1]) == 342, "two permutation blocks")
    check(set(blocks[0]) | set(blocks[1]) == set(source_ids) and not set(blocks[0]) & set(blocks[1]), "blocks partition current source")
    labels = dict(zip(source_ids, source_labels))
    for block in blocks:
        check(set(labels[rid] for rid in block) == {f"ZY{i:02d}" for i in range(1, 8)}, "rare classes preserved within each permutation block")
    truth_arrays = []
    for filename in ("outer0_original.csv", "outer0_permuted_trap.csv"):
        path = DIAG / "scoring_truth" / filename
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        check([r["record_id"] for r in rows] == query_ids, "frozen truth routing order")
        check(path.stat().st_mode & 0o222 == 0, "truth original files marked read-only")
        truth_arrays.append([r["label"] for r in rows])
    check(truth_arrays[0] != truth_arrays[1] and sorted(truth_arrays[0]) == sorted(truth_arrays[1]), "two actual query truth variants differ and preserve counts")
    log_path = HERE.parent / "empirical/unit_tests.log"
    log = log_path.read_text()
    check(len(re.findall(r"^test_.*\.\.\. ok$", log, re.M)) == 10 and bool(re.search(r"Ran 10 tests in [\d.]+s\s+OK\s*$", log)), "actual ten synthetic fixtures passed")
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_EMPIRICAL_DRIVER_AND_PREPARATION_REVIEW_PASS_LIMITED_SCOPE",
        "checks": checks, "new_audit_fits": 0, "new_research_scores": 0,
        "frozen_manifest_sha256": sha(DIAG / "frozen_manifest.json"),
        "source_sha256": manifest["source_sha256"], "actual_test_log_sha256": sha(log_path),
        "script_sha256": sha(__file__),
        "authorized_controls_already_started_at_report_completion": (DIAG / "fits").exists(),
        "static_review": [
            "All20 current-source candidate/final trajectories rerun for C; actual two fixed query-truth file versions and named denied access probe differ",
            "C signature includes model/encoder/head/BN/scalers/pretrained checkpoint/loss/order/decoded actual optimizer trace; excludes only paths/timestamps/container IDs",
            "D modifies one query after fitting; state and other170 rows compared, explicitly not full-source retraining",
            "Ten target-label permutations within two fixed source blocks preserve every current inner class count; external labels unchanged",
            "All12 prediction locks and actual trace/weight dependencies precede separate scoring; source-majority baseline fixed by source labels",
            "Full main code/protocol/preparation hashes frozen before diagnostic outcomes and unchanged by control budget",
        ],
        "limitations": [
            "Runtime empirical C/D/H results still require independent replay after completion",
            "One fold, one seed,2/4epochs and4presteps cannot certify full120-step/48-or96epoch behavior",
            "C compares two frozen actual truth-file variants via denied active file probe; does not overwrite the canonical main truth file",
            "Hashing frozen file bytes for integrity is separate from parsing labels; named Python I/O hooks are not an OS sandbox",
            "Diagnostic outcomes cannot select or revise main models, seeds, profiles or budgets",
            "An initial auditor assertion demanded unused main truth/hash keys in diagnostic routing metadata; corrected to declared diagnostic schema while retaining exact copied-byte checks. Initial script/log preserved separately.",
        ],
    }
    output = HERE / "empirical_preflight_review.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output), "checks": checks, "new_audit_fits": 0}))


if __name__ == "__main__":
    main()
