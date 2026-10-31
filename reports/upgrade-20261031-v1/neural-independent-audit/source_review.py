"""Independent, read-only pre-source audit. No estimator or adapter is fitted."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/upgrade-20261031-v1/neural"
DEST = Path(__file__).resolve().parent
PROTOCOL = ROOT / "data/research/upgrade_neural_protocol_20261031.json"
checks = 0


def require(condition, detail):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def truth(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len({r["record_id"] for r in rows}) == len(rows), "Repeated scoring truth ID")
    return {r["record_id"]: r["label"] for r in rows}


def main():
    target = DEST / "source_review.json"
    if target.exists():
        raise FileExistsError("Preserve the completed review; use a separate revision")
    protocol, prep, candidate = load(PROTOCOL), load(RUN / "preparation.json"), load(RUN / "source_candidate_manifest.json")
    require(sha(PROTOCOL) == candidate["protocol_sha256"], "Candidate protocol changed")
    require(sha(RUN / "preparation.json") == candidate["preparation_sha256"], "Prepared contexts changed")
    require(protocol["samples"] == 855 and protocol["outer_folds"] == 5 and protocol["inner_folds"] == 2, "Unexpected evaluation population")
    require(protocol["seeds"] == [42, 43, 44] and protocol["primary_seed"] == 42, "Seed selection changed")
    require(len(protocol["families"]) == 8 and len(protocol["profiles"]) == 2, "Budget changed")
    for relative, expected in candidate["source_sha256"].items():
        require(sha(ROOT / relative) == expected, "Source changed: " + relative)
    for field, expected_field in (("input", "input_sha256"), ("alignment", "alignment_sha256"),
                                  ("lineage", "lineage_sha256"), ("historical_splits", "historical_splits_sha256")):
        require(sha(ROOT / protocol[field]) == protocol[expected_field], "Input changed: " + field)
    boundary = load(RUN / "boundary_test_provenance.json")
    require(boundary["tested_implementation_sha256"] == candidate["source_sha256"]["src/herb_spectra/upgrade_neural.py"], "Tests apply to another implementation")
    require(sha(ROOT / boundary["test_log"]) == boundary["test_log_sha256"], "Boundary log changed")
    require(sha(ROOT / "tests/test_upgrade_neural.py") == boundary["test_module_sha256"], "Boundary test source changed")
    require(boundary["tests"] >= 9 and (ROOT / boundary["test_log"]).read_text().rstrip().endswith("OK"), "Boundary suite not completed")
    require(not (RUN / "selection_lock.json").exists() and not (RUN / "outer_predictions.csv.gz").exists(), "Review must precede research selection and outer inference")
    data = read_npz(ROOT / protocol["input"])
    ids = data["record_ids"].astype(str).tolist()
    labels = data["y"].astype(str).tolist()
    require(len(ids) == len(set(ids)) == 855, "DB specimen identity")
    require(Counter(labels) == Counter({"ZY01":350, "ZY02":5, "ZY03":100, "ZY04":100, "ZY05":100, "ZY06":100, "ZY07":100}), "DB classes changed")
    index = {v:i for i,v in enumerate(ids)}
    label_by_id = dict(zip(ids, labels, strict=True))
    alignment = load(ROOT / protocol["alignment"])
    binding_path = ROOT / alignment["original_sheet_value_binding_file"]
    require(sha(binding_path) == alignment["original_sheet_value_binding_sha256"], "Original-value bindings changed")
    with binding_path.open(encoding="utf-8-sig", newline="") as handle:
        bindings = list(csv.DictReader(handle))
    require(len(bindings) == 3420, "Expected 855 x four representation bindings")
    raw_hashes = {}
    for row in bindings:
        mode, rid = row["modality"], row["record_id"]
        require(array_sha(data["X_" + mode][index[rid]]) == row["model_matrix_row_sha256"], "ID/value mismatch " + rid + "/" + mode)
        require(array_sha(data["axis_" + mode]) == row["model_axis_sha256"], "Axis mismatch " + mode)
        raw_hashes[row["source_file"]] = row["source_hash"]
    require(len(raw_hashes) == 7, "Unexpected source workbook set")
    for relative, expected in raw_hashes.items():
        require(sha(ROOT / relative) == expected, "Original workbook changed " + relative)
    old = read_npz(ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz")
    old_ids = old["groups"].astype(str).tolist()
    require(set(old_ids) == set(ids), "Old baseline population changed")
    positions = np.asarray([index[r] for r in old_ids])
    require(np.array_equal(data["X_alpha"][positions], old["X_alpha"]), "Old alpha numeric values changed")
    require(np.array_equal(data["X_n"][positions], old["X_n"]), "Old refractive numeric values changed")
    fixed_values = {}
    for mode, maximum in protocol["maximum_points"].items():
        values = data["X_" + mode]
        if maximum is None:
            fixed_values[mode] = values
        else:
            blocks = np.array_split(np.arange(values.shape[1]), min(maximum, values.shape[1]))
            fixed_values[mode] = np.stack([values[:, b].mean(1) for b in blocks], axis=1)
            require(prep["transforms"][mode]["blocks_start_stop"] == [[int(b[0]), int(b[-1])+1] for b in blocks], "Fixed block boundaries changed")
    old_splits = load(ROOT / protocol["historical_splits"])
    require(len(prep["contexts"]) == 15, "Missing fixed contexts")
    with (ROOT / protocol["lineage"]).open(encoding="utf-8-sig", newline="") as handle:
        lineage = [r for r in csv.DictReader(handle) if r["dataset_id"] == "zaoren"]
    require(len(lineage) == 3420, "Wrong zaoren lineage population")
    row_by_id = {}
    for row in lineage:
        require(row["physical_sample_id"] == row["record_id"], "Declared specimen mapping changed")
        require(row["independence_evidence"] == "user_declared_independent_20261031", "Independent evidence overstated")
        row_by_id[row["record_id"]] = row
    context_summaries = []
    numeric_cells = 0
    for context in prep["contexts"]:
        outer = old_splits[context["outer_fold"]]
        if context["stage"] == "inner":
            expected = outer["inner_folds"][context["inner_fold"]]
            source_ids, inference_ids = expected["train_sample_ids"], expected["validation_sample_ids"]
            require(not (set(source_ids) | set(inference_ids)) & set(outer["test_sample_ids"]), "Outer record entered inner role")
        else:
            source_ids, inference_ids = outer["train_sample_ids"], outer["test_sample_ids"]
        require(context["source_ids"] == source_ids and context["inference_ids"] == inference_ids, "Saved split changed")
        require(not set(source_ids) & set(inference_ids), "Role overlap")
        require(context["lineage_status"] == "EXPLORATORY", "User declaration improperly becomes confirmed independence")
        for role, want_ids in (("source", source_ids), ("inference", inference_ids)):
            p = ROOT / context[role + "_pack"]
            require(sha(p) == context[role + "_pack_sha256"], "Input pack changed")
            pack = read_npz(p)
            expected_keys = {"record_ids", "X_alpha", "X_n", "X_time", "X_frequency"} | ({"source_labels"} if role == "source" else set())
            require(set(pack) == expected_keys, "Labels or metadata entered input feature pack")
            require(pack["record_ids"].astype(str).tolist() == want_ids, "Pack rows changed")
            pos = np.asarray([index[r] for r in want_ids])
            for mode, values in fixed_values.items():
                require(np.isfinite(pack["X_" + mode]).all() and np.array_equal(pack["X_" + mode], values[pos]), "Source/inference values not bound to specimen IDs")
                numeric_cells += int(pack["X_" + mode].size)
            if role == "source":
                require(pack["source_labels"].astype(str).tolist() == [label_by_id[r] for r in want_ids], "Source labels mismatched")
                require(min(Counter(pack["source_labels"].tolist()).values()) == (2 if context["stage"] == "inner" else 4), "ZY02 feasibility changed")
        truths = truth(ROOT / context["scoring_truth"])
        require(truths == {r:label_by_id[r] for r in inference_ids}, "Independent scoring truth misaligned")
        context_summaries.append({"split_id":context["split_id"], "source":len(source_ids), "inference":len(inference_ids), "lineage":"EXPLORATORY"})
    require(prep["os_enforced_file_access_isolation"] is False, "OS isolation overstatement")
    DEST.mkdir(parents=True, exist_ok=True)
    result = {"status":"TECHNICAL_SOURCE_REVIEW_PASS_EXPLORATORY_DATA", "allow_source_training":True,
              "completed_at_utc":datetime.now(timezone.utc).isoformat(), "checks":checks,
              "numeric_cells_compared":numeric_cells, "new_estimator_fits":0, "outer_predictions_scored":False,
              "protocol_sha256":sha(PROTOCOL), "candidate_manifest_sha256":sha(RUN / "source_candidate_manifest.json"),
              "preparation_sha256":sha(RUN / "preparation.json"), "boundary_tests":boundary["tests"],
              "guard_source_sha256":sha(ROOT / "src/herb_spectra/leakage_guard.py"),
              "review_script_sha256":sha(Path(__file__)), "contexts":context_summaries,
              "manual_code_review":["Each scaler, MC selector and every internal PLS fit is current-source scoped", "Numeric feature matrices exclude IDs, paths and labels", "Coupled profiles are selected by per-outer, two-inner-fold and three-seed means", "All fresh neural models and optimizers are constructed inside each fit", "Discontinuous MC inputs use an MLP branch, not a physical convolution", "Inference uses eval/no_grad and state checks; outer pointwise audit covers all171", "Prediction files contain no truth; a separate scoring process joins labels after prediction lock"],
              "limitations":["Historical855 data and prior scores exposed; no new independent confirmation", "Biological independence remains user declared and single-batch", "API/file boundaries are not an OS security sandbox", "Wrappers guard declared entry points, not arbitrary Python code", "Standalone learned-cache loading must retain the complete run-manifest dependency checks", "Coupled profiles and family comparison use different compute, not strict factor isolation"]}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v, list)},ensure_ascii=False))


if __name__ == "__main__":
    main()
