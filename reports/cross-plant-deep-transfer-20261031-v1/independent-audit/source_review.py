"""Independently reconstruct fixed transfer inputs; never fit any estimator."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
from herb_spectra import cross_plant_transfer as core
from herb_spectra.leakage_guard import LeakageError

from rebuild_common_grid import reconstruct

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "neural"
CHECKS = 0
VALUES = 0
MAXIMUM = 0.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def same(actual, expected, message):
    global VALUES, MAXIMUM
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, message + " shape")
    check(np.array_equal(a, b), message + " exact values")
    if a.dtype.kind in "fiu":
        VALUES += a.size
        if a.size:
            MAXIMUM = max(MAXIMUM, float(np.abs(a.astype(float) - b.astype(float)).max()))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def last_truth_guard():
    """Exercise real all-truth hash guard without loading weights or labels."""
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        dependencies = [("prediction_sha256", "outer_predictions.csv.gz"),
                        ("outer_model_records_sha256", "outer_model_records.json"),
                        ("outer_role_aliases_sha256", "outer_role_aliases.json"),
                        ("selection_lock_sha256", "selection_lock.json"),
                        ("outer_release_sha256", "outer_release_snapshot.json"),
                        ("run_manifest_sha256", "run_manifest.json"),
                        ("protocol_sha256", "protocol.json")]
        lock = {"outer_truth_parsed": False}
        for key, name in dependencies:
            (out / name).write_text("[]\n")
            lock[key] = sha(out / name)
        contexts = []
        for fold in range(5):
            name = f"outer{fold}.csv"
            (out / name).write_text("record_id,label\nfixture,ZY01\n")
            contexts.append({"stage": "outer", "truth_path": name, "truth_sha256": sha(out / name)})
        (out / "prediction_lock.json").write_text(json.dumps(lock))
        with patch.object(core, "verify_selected", return_value=({"contexts": contexts}, {})):
            core.verify_predictions_before_truth(out)
            (out / contexts[-1]["truth_path"]).write_text("record_id,label\nfixture,ZY07\n")
            with patch.object(core, "_truth", side_effect=AssertionError("Premature label parsing")) as parser:
                try:
                    core.verify_predictions_before_truth(out)
                except LeakageError as error:
                    check("All outer truth bytes" in str(error), "last truth mismatch rejected")
                else:
                    raise AssertionError("Last truth mutation accepted")
                check(parser.call_count == 0, "no labels parsed before all byte checks")
    return {"synthetic_last_of_five_truth_file_tampered": True,
            "actual_guard_entrypoint_exercised": True, "truth_parser_calls": 0,
            "model_fits_or_predictions": 0, "scope": "Unit guard, not a full empirical experiment"}


def main():
    manifest = read(RUN / "run_manifest.json")
    preparation = read(RUN / "preparation.json")
    protocol = read(RUN / "protocol.json")
    check(manifest["protocol_sha256"] == sha(RUN / "protocol.json"), "protocol hash")
    check(manifest["preparation_sha256"] == sha(RUN / "preparation.json"), "preparation hash")
    for relative, expected in manifest["source_sha256"].items():
        check(sha(ROOT / relative) == expected, "frozen source " + relative)
        if "cross_plant_transfer" not in relative:
            original = subprocess.check_output(["git", "show", "3dc8b91162b160f7b9b22512f394178de0327fcb:" + relative], cwd=ROOT)
            check(hashlib.sha256(original).hexdigest() == expected, "historical dependency unchanged " + relative)
    check(manifest["source_sha256"]["src/herb_spectra/cross_plant_transfer.py"] ==
          "79f15d8fa8d08a1744740dd4892a963546403bf60212222e9a1b8897f87d6e01", "reviewed prototype core frozen")
    check(protocol["arms"] == ["A", "Aextra", "B", "C"], "four fixed arms")
    check(protocol["roles"] == ["A", "Aextra", "B", "C", "source_policy"], "policy alias")
    check(protocol["seeds"] == [42, 43, 44], "all fixed seeds")
    profiles = protocol["profiles"]
    check([p["epochs"] for p in profiles] == [48, 96], "two fixed epochs")
    check({k: v for k, v in profiles[0].items() if k not in {"id", "epochs"}} ==
          {k: v for k, v in profiles[1].items() if k not in {"id", "epochs"}}, "profiles differ only epochs")
    for p in profiles:
        check(p["residual"] is True and p["attention"] is False, "compact ResNet")
        check((p["width"], p["blocks"], p["dropout"], p["target_batch_size"], p["pretrain_batch_size"], p["pretrain_steps"]) ==
              (12, 2, .35, 64, 32, 120), "architecture and budgets")
    check(protocol["source_pipeline_budget"] == 240 and protocol["outer_pipeline_budget"] == 60, "pipeline budgets")
    check(protocol["pretrain_trajectories_if_complete"] == 225 and protocol["cache_reuse"] is False, "no hidden cache")
    check(protocol["policy_guard"] == {"macro_f1_margin": .005, "balanced_accuracy_not_lower": True,
          "accuracy_not_lower": True, "tolerance": 1e-12, "fixed_arm_order": ["Aextra", "B", "C"]}, "source guard fixed")
    check(protocol["external_class_heads"] == {"huangqi": 4, "tianma": 4, "ginseng": 2}, "external heads")
    check(preparation["unit_compatibility_verified"] is False and preparation["independent_confirmation_established"] is False,
          "scientific status not promoted")

    axis, reconstructed = reconstruct()
    same(protocol["axis"], axis, "fixed engineering axis")
    canonical_ids = reconstructed["zaoren"]["record_ids"]
    canonical_y = reconstructed["zaoren"]["source_labels"]
    lookup = {rid: index for index, rid in enumerate(canonical_ids)}
    splits = read(ROOT / protocol["historical_splits"])
    check(sha(ROOT / protocol["historical_splits"]) == protocol["historical_splits_sha256"], "historical memberships frozen")
    expected_contexts = {}
    for fold in splits:
        for inner, part in enumerate(fold["inner_folds"]):
            expected_contexts[f"outer{fold['fold']:02d}.inner{inner:02d}"] = (
                "inner", fold["fold"], inner, part["train_sample_ids"], part["validation_sample_ids"], fold["test_sample_ids"])
        expected_contexts[f"outer{fold['fold']:02d}.final"] = (
            "outer", fold["fold"], None, fold["train_sample_ids"], fold["test_sample_ids"], [])
    check(len(preparation["contexts"]) == 15, "15 contexts")
    check({c["split_id"] for c in preparation["contexts"]} == set(expected_contexts), "exact contexts")
    for context in preparation["contexts"]:
        stage, fold, inner, source_ids, query_ids, forbidden = expected_contexts[context["split_id"]]
        check((context["stage"], context["outer_fold"], context["inner_fold"]) == (stage, fold, inner), "context roles")
        check(context["source_ids"] == source_ids and context["query_ids"] == query_ids, "membership and order")
        check(not set(source_ids) & set(query_ids), "source/query disjoint")
        check(not (set(source_ids) | set(query_ids)) & set(forbidden), "outer heldouts absent from inner")
        for role, ids in [("source", source_ids), ("query", query_ids)]:
            path = RUN / context[role + "_pack"]
            check(sha(path) == context[role + "_sha256"], "input pack hash")
            pack = load(path)
            check(set(pack) == {"X", "record_ids", "axis"} | ({"source_labels"} if role == "source" else set()), "strict input schema")
            indices = [lookup[rid] for rid in ids]
            same(pack["record_ids"], ids, "source/query ordered IDs")
            same(pack["axis"], axis, "source/query exact axis")
            same(pack["X"], reconstructed["zaoren"]["raw_common"][indices], "independent pointwise interpolation")
            if role == "source":
                same(pack["source_labels"], canonical_y[indices], "current-source labels")
        truth_path = RUN / context["truth_path"]
        check(sha(truth_path) == context["truth_sha256"], "truth serialization hash")
        with truth_path.open(newline="") as handle:
            truth = list(csv.DictReader(handle))
        check(truth == [{"record_id": rid, "label": str(canonical_y[lookup[rid]])} for rid in query_ids], "truth correctly bound, preparation-only role")

    for name, specification in preparation["auxiliary"].items():
        check(name in {"huangqi", "tianma", "ginseng"}, "allowed auxiliary task")
        path = RUN / specification["path"]
        check(sha(path) == specification["sha256"], "auxiliary input hash")
        pack = load(path)
        check(set(pack) == {"X", "record_ids", "axis", "source_labels"}, "auxiliary strict schema")
        expected = reconstructed[name]
        ids = np.asarray([name + "::" + rid for rid in expected["record_ids"]])
        same(pack["record_ids"], ids, "auxiliary namespaced IDs")
        same(pack["source_labels"], expected["source_labels"], "separate native source labels")
        same(pack["axis"], axis, "auxiliary axis")
        same(pack["X"], expected["raw_common"], "auxiliary independent interpolation")
        check(specification["record_ids"] == ids.tolist(), "auxiliary membership")
        check(specification["class_order"] == np.unique(expected["source_labels"]).tolist(), "auxiliary head vocabulary")
        check(not set(ids) & set(canonical_ids), "no target IDs in external pools")

    data_index_path = HERE.parent / "data-audit/files_sha256.json"
    for relative, specification in read(data_index_path).items():
        path = data_index_path.parent / relative
        check(sha(path) == specification["sha256"] and path.stat().st_size == specification["bytes"], "sealed data audit unchanged")
    tests_path = ROOT / "tests/test_cross_plant_transfer.py"
    check(sha(tests_path) == "129c22c8d9a86171a60e6c7ad5c03094d2ba0306bf3b25788ac36503957ebde5", "actual synthetic fixture code")
    log_path = HERE.parent / "model-preflight/synthetic_tests.log"
    log = log_path.read_text()
    check(len(re.findall(r"^test_.*\.\.\. ok$", log, re.M)) == 14, "14 actual successful synthetic tests")
    check(bool(re.search(r"Ran 14 tests in [\d.]+s\s+OK\s*$", log)), "actual test summary")
    for directory in (RUN / "source_models", RUN / "outer_models"):
        check(not directory.exists(), "zero main research models at preflight")

    guard = last_truth_guard()
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_CORE_AND_PREPARATION_REVIEW_PASS_EXPLORATORY_ONLY",
        "new_audit_fits": 0, "new_research_scores": 0, "checks": CHECKS,
        "numeric_values": VALUES, "maximum_difference": MAXIMUM,
        "reviewed_source_sha256": {**manifest["source_sha256"], str(tests_path.relative_to(ROOT)): sha(tests_path)},
        "frozen_artifacts_sha256": {name: sha(RUN / name) for name in ("protocol.json", "preparation.json", "run_manifest.json")},
        "actual_test_log_sha256": sha(log_path), "data_audit_index_sha256": sha(data_index_path),
        "source_pipelines": 240, "outer_pipelines": 60, "policy_alias_cases": 15,
        "expected_source_prediction_rows": 82080, "expected_outer_prediction_rows": 12825,
        "last_truth_guard_fixture": guard, "script_sha256": sha(__file__),
        "scientific_independence_or_physical_compatibility_certified": False,
        "review_scope": ["Four original canonical NPZ and all15 exact historical contexts independently reconstructed",
                         "All30 source/query packs and3 separately labelled auxiliary packs, coordinates and truth bindings",
                         "Read-only implementation review of per-source scalers, separate pretraining heads, actual batch scopes, paired initializations, resetBN/fresh optimizer and inference state",
                         "A-only shared48/96 profile selection using pooled684 source OOF/seed, fixed guard and source/outer/prediction hash gates",
                         "Actual14 synthetic fixture log bound to frozen code; extra no-fit last-truth-file rejection guard"],
        "limitations": ["Main research fitting and empirical C/D/H diagnostic not yet run at this review",
                        "Unknown teacher absorption units and ginseng ancestors remain unresolved; only engineering shape transfer",
                        "Independent reviewer reads labels to verify fixed packaging, never chooses hyperparameters or fits models",
                        "Named Python truth I/O guard is not an OS sandbox; original855 remains exposed development",
                        "Interpolated128 bins are not newly measured independent frequencies; same representation in all4 arms"],
    }
    output = HERE / "source_review.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output),
                      "checks": CHECKS, "numeric_values": VALUES, "maximum_difference": MAXIMUM,
                      "new_audit_fits": 0}))


if __name__ == "__main__":
    main()
