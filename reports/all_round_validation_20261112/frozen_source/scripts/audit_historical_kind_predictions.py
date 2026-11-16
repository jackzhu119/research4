#!/usr/bin/env python3
"""Reinterpret frozen historical OOF labels; never fit or relabel a model.

The input is read directly from the local Git object database. Fetch the
historical research commit first if it is not already available. No network
request, source-workbook write, or historical-result write is performed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path


HISTORICAL_COMMIT = "3fc996e99cce8ff5799b5a2af9b565f982dfc899"
PREDICTION_PATH = (
    "reports/staged-nested-optimization/2026-10-31/core/outer_predictions.csv.gz"
)
EXPECTED_PREDICTION_SHA256 = (
    "0f6d5843408ce2193abaf8ea9344d36ee1718ad7ef3603c72c10eb104a167c3c"
)
CLASSES = [f"ZY{i:02d}" for i in range(1, 8)]
KINDS = ["枣仁", "酸枣仁"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def confusion_summary(truth: list[str], prediction: list[str], labels: list[str]) -> dict:
    counts = collections.Counter(zip(truth, prediction))
    matrix = [[counts[(a, b)] for b in labels] for a in labels]
    support = collections.Counter(truth)
    correct = sum(a == b for a, b in zip(truth, prediction))
    recalls, f1s = [], []
    for label in labels:
        tp = counts[(label, label)]
        fn = sum(count for (a, b), count in counts.items() if a == label and b != label)
        fp = sum(count for (a, b), count in counts.items() if b == label and a != label)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "n": len(truth),
        "correct": correct,
        "accuracy": correct / len(truth),
        "macro_f1": sum(f1s) / len(labels),
        "balanced_accuracy": sum(recalls) / len(labels),
        "labels": labels,
        "matrix_orientation": "rows=true, columns=predicted",
        "confusion_matrix": matrix,
        "support": {label: support[label] for label in labels},
        "recall": dict(zip(labels, recalls)),
    }


def run(root: Path, source_ref: str) -> dict:
    commit = git_bytes(root, "rev-parse", f"{source_ref}^{{commit}}").decode().strip()
    compressed = git_bytes(root, "show", f"{commit}:{PREDICTION_PATH}")
    prediction_hash = sha256(compressed)
    if prediction_hash != EXPECTED_PREDICTION_SHA256:
        raise ValueError("Frozen historical prediction hash differs; do not silently reuse it")

    registry_path = root / "data/active_workbooks.json"
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    groups = {
        row["origin_group_id"]: row
        for row in registry["workbooks"]
        if row.get("origin_group_id") in CLASSES
    }
    if set(groups) != set(CLASSES):
        raise ValueError("Current active registry must contain every original ZY group")
    mapping = {group: groups[group]["species"] for group in CLASSES}
    expected_mapping = {group: "枣仁" if group in CLASSES[:2] else "酸枣仁" for group in CLASSES}
    if mapping != expected_mapping:
        raise ValueError("Current registry no longer matches the 2026-11-08 kind correction")
    if not all(groups[group].get("species_verified") for group in CLASSES):
        raise ValueError("The kind mapping must retain explicit user confirmation")
    expected_counts = {group: groups[group]["record_count"] for group in CLASSES}
    if sum(expected_counts.values()) != 855:
        raise ValueError("Historical audit requires the original 855 ZY records")

    rows = list(csv.DictReader(io.StringIO(gzip.decompress(compressed).decode("utf-8"))))
    methods = {}
    method_ids = {}
    for method in ("R0", "R2"):
        selected = [row for row in rows if row["method"] == method]
        ids = [row["record_id"] for row in selected]
        if len(selected) != 855 or len(set(ids)) != 855:
            raise ValueError(f"{method} must contain exactly one OOF row per original record")
        if collections.Counter(row["y_true"] for row in selected) != expected_counts:
            raise ValueError(f"{method} original class counts do not match the registry")
        for row in selected:
            if row["y_predicted"] not in CLASSES or row["y_true"] != row["record_id"].split("-")[0]:
                raise ValueError(f"Invalid historical record/class binding: {row['record_id']}")
            if int(row["is_correct"]) != int(row["y_true"] == row["y_predicted"]):
                raise ValueError("Saved correctness differs from independently recomputed values")
        method_ids[method] = set(ids)
        methods[method] = {
            "historical_seven_class_oof": confusion_summary(
                [row["y_true"] for row in selected],
                [row["y_predicted"] for row in selected],
                CLASSES,
            ),
            "kind_collapsed_historical_predictions": confusion_summary(
                [mapping[row["y_true"]] for row in selected],
                [mapping[row["y_predicted"]] for row in selected],
                KINDS,
            ),
            "original_seven_class_correctness_within_true_kind": {},
        }
        for kind in KINDS:
            subset = [row for row in selected if mapping[row["y_true"]] == kind]
            correct = sum(row["y_true"] == row["y_predicted"] for row in subset)
            methods[method]["original_seven_class_correctness_within_true_kind"][kind] = {
                "n": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
                "prediction_label_space": CLASSES,
                "interpretation": "Historical seven-class decisions evaluated within this true-kind subset; no restriction of predicted classes and no new within-kind classifier fit.",
            }
    if method_ids["R0"] != method_ids["R2"]:
        raise ValueError("R0 and R2 evaluated record identities differ")

    confirmation = root / registry["confirmation_record"]
    return {
        "schema": "research4_historical_kind_relabel_audit_v1",
        "audit_date": "2026-11-08",
        "status": "RETROSPECTIVE_HISTORICAL_PREDICTION_REINTERPRETATION",
        "new_model_fits": 0,
        "source": {
            "git_source_ref_requested": source_ref,
            "git_commit": commit,
            "path_in_commit": PREDICTION_PATH,
            "github_blob_url": f"https://github.com/jackzhu119/research4/blob/{commit}/{PREDICTION_PATH}",
            "sha256_compressed_file": prediction_hash,
            "sha256_decompressed_csv": sha256(gzip.decompress(compressed)),
            "total_file_rows_all_methods": len(rows),
            "selected_methods": ["R0", "R2"],
        },
        "current_kind_mapping_evidence": {
            "registry_path": str(registry_path.relative_to(root)),
            "registry_version": registry["version"],
            "registry_sha256": sha256(registry_bytes),
            "confirmation_path": str(confirmation.relative_to(root)),
            "confirmation_sha256": sha256(confirmation.read_bytes()),
            "verification_level": "user_confirmation_not_independent_taxonomic_audit",
            "group_to_kind": mapping,
            "original_group_counts": expected_counts,
        },
        "methods": methods,
        "historical_accuracy_gain_decomposition_R2_minus_R0": {
            kind: {
                "n": methods["R0"]["original_seven_class_correctness_within_true_kind"][kind]["n"],
                "correct_count_change": (
                    methods["R2"]["original_seven_class_correctness_within_true_kind"][kind]["correct"]
                    - methods["R0"]["original_seven_class_correctness_within_true_kind"][kind]["correct"]
                ),
                "accuracy_percentage_point_change": 100 * (
                    methods["R2"]["original_seven_class_correctness_within_true_kind"][kind]["accuracy"]
                    - methods["R0"]["original_seven_class_correctness_within_true_kind"][kind]["accuracy"]
                ),
                "scope": "Retrospective decomposition of the original seven-class decisions, not improvement from a refitted within-kind model.",
            }
            for kind in KINDS
        },
        "interpretation_limits": [
            "Historical R0/R2 were trained to predict seven anonymous groups from 355 枣仁 and 500 酸枣仁 records, not pure 酸枣仁 origins.",
            "Collapsing saved seven-class labels is a retrospective audit; it is not a newly trained binary classifier or new external-validation result.",
            "Within-kind subset accuracy uses the original seven-class predictions; it is not accuracy of a refitted five-origin 酸枣仁 model.",
            "All 855 records have prior development exposure and each original source group has only one user-declared batch; kind, origin and source-lot effects are not isolated.",
            "The nine newly added 枣仁 records, 黄芪 and 天麻 are outside this historical prediction audit.",
            "No source workbook, original label, saved model, historical result or prediction was changed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default=HISTORICAL_COMMIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/species_revision_20261108/historical_relabel_audit.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = run(root, args.source_ref)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    for method, result in audit["methods"].items():
        collapsed = result["kind_collapsed_historical_predictions"]
        print(f"{method}: retrospective kind collapse {collapsed['correct']}/{collapsed['n']}")


if __name__ == "__main__":
    main()
