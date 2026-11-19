#!/usr/bin/env python3
"""Export allowlisted LOWO results without rebuilding the six existing books.

The export is mechanical: fixed results and their original Chinese kind labels
are retained. Each source is checked against both the current manifest and its
immutable Git object; every exported cell is read back before receipts change.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import numbers
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/active_workbooks.json").exists())
BASE = ROOT / "paper/revision_20261115"
CURRENT = ROOT / "paper/current"
SOURCE_PREFIX = "results/kind_lowo_20261115"
SOURCE_PIN = "627a0975bdc61ef7cd95742265c3b500c879deae"
TARGET = BASE / "Supplementary_LOWO.xlsx"
MERGED_RECEIPT = BASE / "SUBMISSION_TABLES_VERIFICATION.json"
RECEIPT = BASE / "LOWO_TABLES_VERIFICATION.json"
QA_RECEIPT = BASE / "KIND_LOWO_VERIFICATION.json"
OLD_BOOKS = ["Supplementary_Tables.xlsx", *[f"Supplementary_B{i}.xlsx" for i in range(1, 6)]]
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="34495E")
BODY_FONT = Font(name="Calibri", size=11, color="000000")
INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
NUMBER = re.compile(r"[+-]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def csv_cell(value):
    if value == "":
        return None
    if INTEGER.fullmatch(value):
        number = int(value)
        return number if len(value.lstrip("+-")) <= 15 else value
    if NUMBER.fullmatch(value) and ("." in value or "e" in value.lower()):
        number = float(value)
        return number if math.isfinite(number) else value
    return value


def flatten(value, prefix=""):
    result = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.extend(flatten(item, path))
        else:
            result.append([path, json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item])
    return result


def main():
    require((CURRENT / SOURCE_PREFIX).is_dir(), "LOWO sources have not yet been promoted to paper/current; export not run")
    manifest_path = CURRENT / "manifest.json"
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    allow = {record["path"]: record for record in manifest["files"]}
    prior = json.loads(MERGED_RECEIPT.read_text())
    old_books = [deepcopy(record) for record in prior["workbooks"] if record["file"] in OLD_BOOKS]
    old_sources = [deepcopy(record) for record in prior["files"] if record["workbook"] in OLD_BOOKS]
    require([record["file"] for record in old_books] == OLD_BOOKS, "Expected six previously verified supplementary books")
    require(len(old_sources) == 51, "Existing six-book source records changed")
    for record in old_books:
        path = BASE / record["file"]
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], f"Old workbook changed: {path.name}")
    for record in old_sources:
        require(sha(ROOT / record["source"]) == record["sha256"], f"Prior table source changed: {record['source']}")
    used = {}

    def verified(relative):
        current_relative = f"{SOURCE_PREFIX}/{relative}"
        require(current_relative in allow, f"Source not allowlisted: {current_relative}")
        entry = allow[current_relative]
        path = CURRENT / current_relative
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == entry["sha256"], f"Current source SHA mismatch: {current_relative}")
        require(entry["source_commit"] == SOURCE_PIN, f"LOWO source commit differs: {current_relative}")
        pinned = subprocess.check_output(["git", "-C", str(ROOT), "show", SOURCE_PIN + ":" + entry["source_path"]])
        require(raw == pinned, f"Current LOWO source differs from immutable original: {current_relative}")
        used[current_relative] = {"source": str(path.relative_to(ROOT)), "sha256": digest,
                                  "source_bytes": len(raw), "source_commit": SOURCE_PIN,
                                  "immutable_source_path": entry["source_path"],
                                  "source_verification": "Current manifest allowlist SHA-256 and immutable Git bytes"}
        return raw, used[current_relative]

    tables = []

    def add_csv(relative, sheet, scope):
        raw, source = verified(relative)
        values = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        require(len(values) > 1, f"Empty source CSV: {relative}")
        headers, rows = values[0], [[csv_cell(value) for value in row] for row in values[1:]]
        require(all(len(row) == len(headers) for row in rows), f"Ragged source CSV: {relative}")
        tables.append({"sheet": sheet, "headers": headers, "rows": rows, "scope": scope, **source})

    summaries = {}
    for task, prefix in (("binary_kinds", "Binary"), ("four_kinds", "Four")):
        for filename, suffix, scope in (
            ("oof_predictions.csv", "OOF", "Complete per-record predictions; each task record tested exactly once"),
            ("fold_metrics.csv", "Folds", "One complete workbook held out per fold; fold accuracy equals true-kind recall"),
            ("kind_recall.csv", "Kind_recall", "Kind recalls from pooled out-of-workbook predictions"),
            ("pooled_confusion.csv", "Confusion", "Pooled confusion counts; original Chinese true and predicted kind labels"),
            ("pooled_confusion_normalized.csv", "Confusion_norm", "Pooled confusion normalized within each true-kind row"),
        ):
            add_csv(f"{task}/{filename}", f"{prefix}_{suffix}", scope)
        raw, source = verified(f"{task}/summary.json")
        summary = summaries[task] = json.loads(raw)
        baseline = summary["training_majority_baseline"]
        tables.append({"sheet": f"{prefix}_Majority", "headers": ["true_kind", *baseline["class_order"]],
                       "rows": [[kind, *values] for kind, values in zip(baseline["class_order"], baseline["confusion_counts"])],
                       "scope": "Training-majority baseline confusion counts; majority kind determined separately from each fold's training records", **source})
    add_csv("record_index.csv", "Record_index", "All 1672 corrected record identities; historical roles retained only as provenance")
    add_csv("feature_axis.csv", "Feature_axis", "612 original shared frequency coordinates; absorption values only are classifier predictors")
    raw, source = verified("protocol.json")
    protocol = json.loads(raw)
    tables.append({"sheet": "Protocol", "headers": ["key", "value"], "rows": flatten(protocol),
                   "scope": "Flattened fixed protocol; no hyperparameter tuning or outcome-based fold selection", **source})
    raw, source = verified("summary.json")
    combined = json.loads(raw)
    summary_rows = []
    for task in ("binary_kinds", "four_kinds"):
        data = summaries[task]
        require(combined["tasks"][task] == data, "Combined/task source summaries differ")
        metrics, baseline = data["metrics"], data["training_majority_baseline"]
        summary_rows.append([task, data["n_folds"], metrics["n_records"], metrics["n_correct"],
                             metrics["accuracy"], metrics["macro_f1"], metrics["balanced_accuracy"],
                             data["equal_weight_workbook_mean_accuracy"], baseline["accuracy"], baseline["macro_f1"],
                             baseline["balanced_accuracy"], data["equal_weight_workbook_mean_majority_accuracy"],
                             data["all_records_tested_exactly_once"]])
    tables.append({"sheet": "Task_summary", "headers": ["task", "n_folds", "n_records", "n_correct", "pooled_accuracy", "pooled_macro_f1", "pooled_balanced_accuracy", "equal_weight_workbook_mean_accuracy", "training_majority_pooled_accuracy", "training_majority_pooled_macro_f1", "training_majority_pooled_balanced_accuracy", "training_majority_equal_weight_workbook_mean_accuracy", "all_records_tested_exactly_once"],
                   "rows": summary_rows, "scope": "Pooled specimen-level metrics and equal-weight workbook mean accuracy are distinct estimands", **source})
    references = []
    for relative in ("planned_folds.json", "binary_kinds/fold_splits.json", "four_kinds/fold_splits.json",
                     "dataset_audit.json", "verification.json", "independent_audit.json"):
        _, source = verified(relative)
        references.append(source)
    qa_raw, qa_source = verified("independent_audit.json")
    qa_original = json.loads(qa_raw)
    require(qa_original["status"] == "PASS", "Independent LOWO audit has not passed")
    require(len(tables) == 16 and len({table["sheet"] for table in tables}) == 16, "Expected 16 data sheets")
    require(all(re.fullmatch(r"[A-Za-z0-9_]{1,31}", table["sheet"]) for table in tables), "Invalid ASCII sheet name")
    notes = [
        ["LOWO supplementary tables", "Fixed corrected-kind validation; complete positive and negative results"],
        ["Task scope", "Binary: 10 complete-workbook folds and 864 specimens. Four kinds: 18 folds and 1672 specimens. Each record has exactly one OOF prediction per eligible task."],
        ["Fold interpretation", "A heldout workbook has one true kind; fold accuracy equals its true-kind recall. Task macro-F1 and balanced accuracy are computed after pooling all task OOF predictions."],
        ["Weighting", "Pooled accuracy weights specimens; equal-weight workbook mean accuracy weights workbooks. These quantities are reported separately."],
        ["Original roles", "Frozen TRAIN/CAL/PRED labels are provenance only. All eligible corrected records enter this new validation protocol; original roles are not overwritten."],
        ["Independence", "The two tasks overlap in records. Their 28 model fits/folds are not 28 independent specimens or an independent replication count. Workbook grouping does not establish prospective new-batch performance."],
        ["Predictors", "612 original absorption coordinates in 0.1-1.5 THz; paired coordinates checked. SNV ddof=1 precedes training-fold-only StandardScaler and fixed C=10 balanced RBF SVC."],
        ["Source policy", "Original CSV field names and Chinese kind labels are retained. Every source matches the current manifest SHA-256 and the immutable source commit."],
        ["Immutable source commit", SOURCE_PIN],
        ["Full split IDs", "Complete pre-fit plan and executed train/test ID lists remain in the indexed immutable/current JSON sources; repeated long ID lists are not duplicated in this workbook."],
    ]
    index_headers = ["Sheet_or_reference", "Repository_source", "Data_rows", "Columns", "Source_SHA256", "Scope", "Source_commit", "Immutable_source_path"]
    index_rows = [[table["sheet"], table["source"], len(table["rows"]), len(table["headers"]), table["sha256"], table["scope"], SOURCE_PIN, table["immutable_source_path"]] for table in tables]
    index_rows.extend([["Reference_only", record["source"], None, None, record["sha256"], "Full provenance, splits or independent QA; not counted as an extra data sheet", SOURCE_PIN, record["immutable_source_path"]] for record in references])
    source_index = [[*row, *([None] * (len(index_headers)-len(row)))] for row in notes] + [index_headers] + index_rows
    workbook = Workbook(write_only=True)
    workbook.properties.title = "Supplementary LOWO classification results"
    workbook.properties.description = "Verified mechanical export of complete fixed leave-one-workbook-out validation results"
    workbook.properties.creator = "Ziyuan Zhu; Jiayi Ren; Shuyu Zhou; Mengting Zhu; Yin Shen"

    def cell(sheet, value, header=False):
        result = WriteOnlyCell(sheet, value=value)
        if isinstance(value, str):
            require(len(value) <= 32767, "Excel string limit exceeded")
            result.data_type = "s"
        result.font = HEADER_FONT if header else BODY_FONT
        if header:
            result.fill = HEADER_FILL
        result.alignment = Alignment(vertical="top", wrap_text=header)
        return result

    index = workbook.create_sheet("Source_Index")
    index.freeze_panes = f"A{len(notes)+2}"
    index.sheet_view.showGridLines = False
    for col, width in enumerate([27, 83, 13, 13, 70, 85, 45, 85], 1):
        index.column_dimensions[get_column_letter(col)].width = width
    for number, row in enumerate(source_index, 1):
        index.append([None if value is None else cell(index, value, number == len(notes)+1) for value in row])
    index.auto_filter.ref = f"A{len(notes)+1}:H{len(source_index)}"
    for table in tables:
        sheet = workbook.create_sheet(table["sheet"])
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(table['headers']))}{len(table['rows'])+1}"
        sheet.row_dimensions[1].height = 32
        for number, header in enumerate(table["headers"], 1):
            sample = [str(row[number-1]) for row in table["rows"][:100] if row[number-1] is not None]
            sheet.column_dimensions[get_column_letter(number)].width = min(48, max(13, len(header)+2, max(map(len, sample), default=0)+2))
        sheet.append([cell(sheet, header, True) for header in table["headers"]])
        for row in table["rows"]:
            sheet.append([None if value is None else cell(sheet, value) for value in row])
    workbook.save(TARGET)
    require(TARGET.stat().st_size < 10*1024*1024, "LOWO supplementary book exceeds 10 MiB")
    with ZipFile(TARGET) as archive:
        require(archive.testzip() is None, "XLSX CRC failure")
    reopened = load_workbook(TARGET, read_only=True, data_only=False)
    require(reopened.sheetnames == ["Source_Index", *[table["sheet"] for table in tables]], "Sheet list changed")
    numeric_cells = string_cells = total_cells = 0

    def readback(name, expected):
        nonlocal numeric_cells, string_cells, total_cells
        for row_number, (actual, original) in enumerate(zip(reopened[name].iter_rows(values_only=True), expected, strict=True), 1):
            # Write-only XLSX omits trailing empty cells; restore those positions.
            require(len(actual) <= len(original), f"Readback width mismatch: {name}/{row_number}")
            actual = tuple(actual) + (None,) * (len(original)-len(actual))
            for saved, value in zip(actual, original, strict=True):
                total_cells += 1
                if isinstance(value, numbers.Real) and not isinstance(value, bool):
                    require(isinstance(saved, numbers.Real) and math.isclose(saved, value, rel_tol=1e-14, abs_tol=1e-15), f"Numeric readback mismatch: {name}/{row_number}")
                    numeric_cells += 1
                else:
                    require(saved == value, f"Readback mismatch: {name}/{row_number}")
                    string_cells += isinstance(value, str)
    readback("Source_Index", source_index)
    records = []
    for table in tables:
        readback(table["sheet"], [table["headers"], *table["rows"]])
        records.append({key: value for key, value in table.items() if key not in ("headers", "rows")} |
                       {"workbook": TARGET.name, "data_rows": len(table["rows"]), "columns": len(table["headers"]),
                        "readback_data_rows": len(table["rows"]), "all_cells_readback_verified": True})
    reopened.close()
    for relative, record in used.items():
        require(sha(CURRENT / relative) == record["sha256"], f"Source changed during export: {relative}")
    require(sha(manifest_path) == manifest_sha, "Current manifest changed during export")
    for record in old_books:
        require(sha(BASE / record["file"]) == record["sha256"], "Existing workbook changed during export")
    info = {"file": TARGET.name, "bytes": TARGET.stat().st_size, "sha256": sha(TARGET), "sheets": 17,
            "data_sheets": 16, "data_rows": sum(record["data_rows"] for record in records), "zip_crc": "PASS",
            "all_rows_and_cells_readback_verified": True, "file_below_10_MiB": True}
    receipt = {"schema": "research4-lowo-supplement-verification-v1", "status": "PASS", **info,
               "source_pin": SOURCE_PIN, "source_manifest_sha256": manifest_sha,
               "all_source_hashes_verified": True, "all_source_bytes_unchanged": True,
               "all_index_and_data_cells_readback_verified": True, "numeric_cells_readback_checked": numeric_cells,
               "string_cells_readback_checked": string_cells, "total_cells_readback_checked": total_cells,
               "numeric_readback_tolerance": {"relative": 1e-14, "absolute": 1e-15},
               "existing_six_workbooks_byte_unchanged": True, "existing_workbooks": old_books,
               "new_model_training": False, "files": records, "reference_sources": references,
               "builder_source_sha256": sha(__file__)}
    qa = deepcopy(qa_original)
    qa["immutable_source"] = {"source_pin": SOURCE_PIN, **qa_source}
    QA_RECEIPT.write_bytes(encoded(qa))
    receipt["independent_analysis_qa"] = {"file": QA_RECEIPT.name, "sha256": sha(QA_RECEIPT), "status": qa["status"]}
    RECEIPT.write_bytes(encoded(receipt))
    merged = deepcopy(prior)
    merged["workbooks"] = [*old_books, info]
    merged["files"] = [*old_sources, *records]
    merged["total_sheets"] = sum(record["sheets"] for record in merged["workbooks"])
    merged["data_sheets"] = sum(record["data_sheets"] for record in merged["workbooks"])
    merged["total_data_rows"] = sum(record["data_rows"] for record in merged["files"])
    previous_totals = prior.get("lowo_extension", {}).get("prior_readback_totals", {
        "numeric_cells_readback_checked": prior["numeric_cells_readback_checked"],
        "string_cells_readback_checked": prior["string_cells_readback_checked"]})
    for key, value in previous_totals.items():
        merged[key] = value + receipt[key]
    merged["source_manifest_sha256"] = manifest_sha
    merged["all_files_below_10_MiB"] = all(record["file_below_10_MiB"] for record in merged["workbooks"])
    merged["lowo_extension"] = {"verification_file": RECEIPT.name, "verification_sha256": sha(RECEIPT),
                               "source_pin": SOURCE_PIN, "prior_readback_totals": previous_totals,
                               "prior_six_workbook_records_preserved": True, "prior_51_source_records_preserved": True}
    require(merged["workbooks"][:6] == old_books and merged["files"][:51] == old_sources, "Prior table records changed")
    require(merged["data_sheets"] == len(merged["files"]) == 67, "Merged sheet totals inconsistent")
    MERGED_RECEIPT.write_bytes(encoded(merged))
    print(json.dumps({"status": "PASS", **info, "merged_data_sheets": merged["data_sheets"],
                      "merged_total_data_rows": merged["total_data_rows"], "all_existing_six_bytes_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
