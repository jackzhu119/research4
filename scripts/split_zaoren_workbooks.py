"""Split current numbered workbooks into disjoint same-batch research roles.

Only sample columns are selected. Retained XLSX cells keep their original XML
payloads, including exact numeric decimal strings, styles and header suffixes;
their column coordinates are compacted. These derived files do not replace the
active dataset registry. ZY identifiers encode the anonymous class, so the
prediction workbooks are dedicated holdouts, not blinded unidentified samples.
"""
import argparse
from collections import Counter
from copy import copy
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import re
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from herb_spectra.dataset_registry import REPO, file_sha256, select_workbooks


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ID = re.compile(r"^(ZY[0-9]{2}-[0-9]{3,})(.*)$")
ROWS = re.compile(rb'<row\b[^>]*>.*?</row>', re.DOTALL)
CELLS = re.compile(rb'<c\b[^>]*(?:/>|>.*?</c>)', re.DOTALL)
COORD = re.compile(rb'\br="([A-Z]+)([0-9]+)"')
ROLES = ("training", "calibration", "prediction")
ROLE_DIRS = {"training": "训练样本专用", "calibration": "校准专用", "prediction": "预测专用"}
MANIFEST_FIELDS = ["record_id", "physical_sample_id", "origin_group_id", "role", "source_workbook"]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields=MANIFEST_FIELDS):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_split_plan(sources, metadata_path, seed=20261101):
    """Use explicit independent physical IDs; no spectrum values affect roles."""
    sources = sorted((s for s in sources if s.dataset == "zaoren"), key=lambda s: s.origin_group_id)
    if not sources or len({s.dataset_version for s in sources}) != 1:
        raise ValueError("One current zaoren dataset version is required")
    with Path(metadata_path).open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    rng = np.random.default_rng(seed)
    records, counts, seen_records, seen_physical = [], {}, set(), set()
    for source in sources:
        rows = [r for r in metadata if r.get("source_workbook") == source.relative_path
                and r.get("sheet") == "吸收系数"]
        rows.sort(key=lambda r: r.get("record_id", ""))
        if len(rows) != source.record_count or len(rows) < 5:
            raise ValueError(f"{source.origin_group_id}: metadata count differs or fewer than five specimens")
        for row in rows:
            identifier, physical = row.get("record_id", "").strip(), row.get("physical_sample_id", "").strip()
            if (not ID.fullmatch(identifier) or not identifier.startswith(source.origin_group_id + "-")
                    or row.get("origin_group_id") != source.origin_group_id):
                raise ValueError("Metadata record or origin group differs from its registered workbook")
            if not physical or row.get("origin_group_verified", "").lower() not in {"true", "1", "yes"}:
                raise ValueError("Explicit physical identity and verified anonymous origin group are required")
            if identifier in seen_records or physical in seen_physical:
                raise ValueError("Duplicate record or physical specimen; cannot split repeated physical groups")
            seen_records.add(identifier); seen_physical.add(physical)
        # A single PCG64 stream visits sorted classes and sorted record IDs.
        ordered = rng.permutation(len(rows))
        n_train, n_calibration = int(len(rows) * .6), int(len(rows) * .2)
        boundary = n_train + n_calibration
        assignments = [("training", ordered[:n_train]), ("calibration", ordered[n_train:boundary]),
                       ("prediction", ordered[boundary:])]
        counts[source.origin_group_id] = {role: len(indices) for role, indices in assignments}
        if any(count < 1 for count in counts[source.origin_group_id].values()):
            raise ValueError("Every origin must have a sample in each role")
        for role, indices in assignments:
            for index in sorted(indices):
                row = rows[index]
                records.append({"record_id": row["record_id"], "physical_sample_id": row["physical_sample_id"],
                                "origin_group_id": source.origin_group_id, "role": role,
                                "source_workbook": source.relative_path})
    plan = {"schema_version": 1, "split_id": "zaoren-same-batch-threeway-v1",
            "dataset_version": sources[0].dataset_version, "split_seed": int(seed),
            "rng": "NumPy default_rng PCG64; sorted class then sorted record ID; first60% training, next20% calibration, last20% prediction",
            "ratio": {"training": .6, "calibration": .2, "prediction": .2},
            "scope": "retrospective within-batch sample holdout; not independent cross-batch validation",
            "prior_exposure": "All current855 samples have previously been included in internal CV; this is not a pristine external confirmation set",
            "prediction_is_blinded": False,
            "prediction_id_caveat": "ZY header prefixes reveal anonymous origin groups; model features must contain only numerical spectra",
            "counts_by_origin": counts,
            "role_counts": dict(Counter(row["role"] for row in records)),
            "metadata_sha256": file_sha256(metadata_path),
            "sources": [{"path": s.relative_path, "sha256": s.sha256, "origin_group_id": s.origin_group_id,
                         "record_count": s.record_count} for s in sources], "records": records}
    validate_split_plan(plan, sources)
    return plan


def validate_split_plan(plan, sources):
    """Validate an explicit plan before any export or model access."""
    sources = {s.relative_path: s for s in sources if s.dataset == "zaoren"}
    records, physical, counts = set(), set(), Counter()
    for row in plan["records"]:
        source = sources.get(row["source_workbook"])
        if source is None or row["role"] not in ROLES or row["origin_group_id"] != source.origin_group_id:
            raise ValueError("Unknown source, role or origin in split plan")
        identifier = row["record_id"]
        if not ID.fullmatch(identifier) or not identifier.startswith(source.origin_group_id + "-"):
            raise ValueError("Invalid canonical record identifier in split plan")
        if identifier in records or not row["physical_sample_id"] or row["physical_sample_id"] in physical:
            raise ValueError("Physical specimens and records must be unique and disjoint across roles")
        records.add(identifier); physical.add(row["physical_sample_id"])
        counts[(source.relative_path, row["role"])] += 1
    for source in sources.values():
        if sum(counts[(source.relative_path, role)] for role in ROLES) != source.record_count:
            raise ValueError("A workbook is not completely partitioned")
        expected = plan["counts_by_origin"][source.origin_group_id]
        if any(counts[(source.relative_path, role)] != expected[role] or expected[role] < 1 for role in ROLES):
            raise ValueError("Actual split records disagree with declared role counts")
    if dict(Counter(row["role"] for row in plan["records"])) != plan["role_counts"]:
        raise ValueError("Total role counts disagree with split records")


def _column(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _column_number(value):
    result = 0
    for char in value.decode() if isinstance(value, bytes) else value:
        result = result * 26 + ord(char) - 64
    return result


def _text(cell, strings):
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter("t"))
    value = cell.findtext("v", "")
    return strings[int(value)] if cell.get("t") == "s" else value


def _canonical_cell(cell):
    # Preserve original row number and the complete remaining XML cell payload.
    return COORD.sub(lambda m: b'r="' + m.group(2) + b'"', cell, count=1)


def _layout(data, strings, expected_ids):
    header = next((m for m in ROWS.finditer(data) if re.search(rb'\br="1"', m.group(0).split(b">", 1)[0])), None)
    if header is None:
        raise ValueError("Every worksheet requires a row-one header")
    identifiers, common = {}, set()
    for raw in CELLS.findall(header.group(0)):
        coord = COORD.search(raw)
        if coord is None:
            raise ValueError("Cell without an explicit coordinate")
        column = coord.group(1).decode()
        label = _text(ET.fromstring(raw), strings).strip()
        match = ID.fullmatch(label)
        if match:
            identifier = match.group(1)
            if identifier in identifiers.values():
                raise ValueError("Duplicate specimen header in a worksheet")
            identifiers[column] = identifier
        else:
            if _column_number(column) > 1 and label and "参考" not in label and label.lower() not in {"ref", "reference"}:
                raise ValueError(f"Unrecognized non-reference sample header: {label}")
            common.add(column)
    if set(identifiers.values()) != set(expected_ids):
        raise ValueError("Four-sheet specimen IDs must exactly match the complete partition")
    if re.search(rb"<f\b", data):
        raise ValueError("Formula workbooks require a separate reference-aware exporter")
    for tag in [b"cols", b"mergeCells", b"autoFilter", b"conditionalFormatting", b"dataValidations", b"hyperlinks", b"drawing", b"tableParts"]:
        if re.search(rb"<" + tag + rb"\b", data):
            raise ValueError("Worksheet contains column-addressed metadata requiring a reference-aware exporter")
    return identifiers, common


def _filter_sheet(data, strings, expected_ids, selected_ids):
    identifiers, common = _layout(data, strings, expected_ids)
    retained = common | {column for column, identifier in identifiers.items() if identifier in selected_ids}
    columns = sorted(retained, key=_column_number)
    remap = {column: _column(index + 1) for index, column in enumerate(columns)}
    expected_digest = hashlib.sha256()
    cells_retained, last_row = 0, 0

    def replace_row(match):
        nonlocal cells_retained, last_row
        raw = match.group(0)
        opening, rest = raw.split(b">", 1)
        row_match = re.search(rb'\br="([0-9]+)"', opening)
        if row_match is None:
            raise ValueError("Row without explicit index")
        last_row = max(last_row, int(row_match.group(1)))
        opening = re.sub(rb'\s+spans="[^"]*"', b"", opening) + b">"
        kept = []
        for cell in CELLS.findall(rest):
            coord = COORD.search(cell)
            if coord is None:
                raise ValueError("Cell without explicit coordinate")
            old_column = coord.group(1).decode()
            if old_column not in remap:
                continue
            expected_digest.update(_canonical_cell(cell))
            kept.append(COORD.sub(lambda m: b'r="' + remap[old_column].encode() + m.group(2) + b'"', cell, count=1))
            cells_retained += 1
        return opening + b"".join(kept) + b"</row>"

    output = ROWS.sub(replace_row, data)
    dimension = b"A1:" + _column(len(columns)).encode() + str(last_row).encode()
    output = re.sub(rb'(<dimension\b[^>]*\bref=")[^"]*(")', lambda m: m.group(1) + dimension + m.group(2), output, count=1)
    return output, {"retained_cells": cells_retained,
                    "canonical_cell_sha256": expected_digest.hexdigest(),
                    "retained_coordinate_reference_columns": sorted(common, key=_column_number),
                    "retained_sample_count": len(selected_ids), "last_row": last_row,
                    "old_to_new_columns": remap}


def _verify_sheet(data, expected):
    digest, count = hashlib.sha256(), 0
    for row in ROWS.finditer(data):
        for cell in CELLS.findall(row.group(0)):
            digest.update(_canonical_cell(cell)); count += 1
    if count != expected["retained_cells"] or digest.hexdigest() != expected["canonical_cell_sha256"]:
        raise AssertionError("An exported selected numeric/header/reference cell changed")


def export_workbook(source, roles, output):
    """Export all three roles in one source pass and verify exact retained cells."""
    expected_ids = {identifier for ids in roles.values() for identifier in ids}
    if sum(len(ids) for ids in roles.values()) != len(expected_ids):
        raise ValueError("Role specimen IDs overlap")
    if len(expected_ids) != source.record_count:
        raise ValueError("Incomplete source workbook partition")
    original_sha = file_sha256(source.path)
    if source.sha256 is not None and original_sha != source.sha256:
        raise ValueError("Canonical source workbook changed")
    destinations = {role: output / ROLE_DIRS[role] / (source.path.stem + "_" + ROLE_DIRS[role] + ".xlsx") for role in ROLES}
    if any(path.exists() or path.resolve() == source.path.resolve() for path in destinations.values()):
        raise FileExistsError("Refusing to overwrite a source or prior derived workbook")
    temporary, writers, summaries = {}, {}, {role: {} for role in ROLES}
    for role, path in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary[role] = path.with_suffix(".xlsx.partial")
        writers[role] = ZipFile(temporary[role], "w", compression=ZIP_DEFLATED, compresslevel=6)
    try:
        with ZipFile(source.path) as original:
            if original.testzip() is not None:
                raise ValueError("Source ZIP CRC failed")
            strings = []
            if "xl/sharedStrings.xml" in original.namelist():
                strings = ["".join(node.text or "" for node in item.iter(f"{{{MAIN}}}t"))
                           for item in ET.fromstring(original.read("xl/sharedStrings.xml"))]
            relations = ET.fromstring(original.read("xl/_rels/workbook.xml.rels"))
            targets = {item.get("Id"): item.get("Target") for item in relations}
            workbook = ET.fromstring(original.read("xl/workbook.xml"))
            sheets = {}
            for sheet in workbook.find(f"{{{MAIN}}}sheets"):
                target = targets[sheet.get(f"{{{REL}}}id")]
                member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
                sheets[member] = sheet.get("name")
            if set(sheets.values()) != {"时域", "频域", "折射率", "吸收系数"}:
                raise ValueError("Exactly the four canonical spectral worksheets are required")
            for info in original.infolist():
                data = original.read(info.filename)
                for role in ROLES:
                    if info.filename in sheets:
                        filtered, result = _filter_sheet(data, strings, expected_ids, set(roles[role]))
                        summaries[role][sheets[info.filename]] = {**result, "zip_member": info.filename}
                        writers[role].writestr(copy(info), filtered)
                    else:
                        writers[role].writestr(copy(info), data)
            for writer in writers.values():
                writer.comment = original.comment
                writer.close()
            for role in ROLES:
                with ZipFile(temporary[role]) as derived:
                    if original.namelist() != derived.namelist() or derived.testzip() is not None:
                        raise AssertionError("Derived ZIP structure or CRC failed")
                    for info in original.infolist():
                        after = derived.read(info.filename)
                        if info.filename in sheets:
                            _verify_sheet(after, summaries[role][sheets[info.filename]])
                        elif original.read(info.filename) != after:
                            raise AssertionError("An unrelated XLSX package member changed")
        if file_sha256(source.path) != original_sha:
            raise AssertionError("Canonical source bytes changed during export")
        for role in ROLES:
            temporary[role].replace(destinations[role])
        return [{"role": role, "origin_group_id": source.origin_group_id,
                 "path": str(destinations[role].relative_to(output)),
                 "sha256": file_sha256(destinations[role]), "bytes": destinations[role].stat().st_size,
                 "source_path": source.relative_path, "source_sha256": original_sha,
                 "sample_count": len(roles[role]), "sheets": summaries[role],
                 "verification": {"retained_cell_payloads_exact": True, "all_four_sheets_paired": True,
                                  "all_unrelated_zip_members_exact": True, "source_unchanged": True}}
                for role in ROLES]
    except BaseException:
        for writer in writers.values():
            writer.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def write_plan(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "split_plan.json", plan)
    _write_csv(output / "split_manifest.csv", plan["records"])
    for role in ("training", "calibration"):
        directory = output / ROLE_DIRS[role]; directory.mkdir(exist_ok=True)
        _write_csv(directory / "labels.csv", [r for r in plan["records"] if r["role"] == role])
    directory = output / "evaluation"; directory.mkdir(exist_ok=True)
    _write_csv(directory / "prediction_truth.csv", [r for r in plan["records"] if r["role"] == "prediction"])


def write_readme(output, plan):
    rows = ["| 组 | 训练样本专用 | 校准专用 | 预测专用 |", "| --- | ---: | ---: | ---: |"]
    for group, counts in plan["counts_by_origin"].items():
        rows.append(f"| {group} | {counts['training']} | {counts['calibration']} | {counts['prediction']} |")
    totals = plan["role_counts"]
    rows.append(f"| 合计 | {totals['training']} | {totals['calibration']} | {totals['prediction']} |")
    text = ("# 酸枣仁三用途拆分衍生文件\n\n"
            "本目录由当前7份编号版按独立样品拆分生成，不是新增观测，也不替换原始有效工作簿。"
            "不要将本目录与原编号版拼接计数；有效来源仍以`data/active_workbooks.json`为准。\n\n"
            + "\n".join(rows) + "\n\n"
            "- 训练样本专用：只用这些样品选择预处理、特征和模型参数，随后冻结。\n"
            "- 校准专用：训练冻结后，仅从该池抽取少量有标签样品拟合校准器；不回填预测集。\n"
            "- 预测专用：同一固定预测集比较校准前后结果；其真值表单独放在`evaluation/prediction_truth.csv`，只用于最终评分。\n\n"
            "每个来源各有3份衍生工作簿，共21份；四个谱表保持同一份样品归属。"
            "坐标、参考列、选中样品原数值、ZY编号及厚度后缀保留，列位置紧凑排列。"
            "保留单元格的XML载荷摘要、文件SHA-256与原件未变校验见`workbook_verification.json`。\n\n"
            f"固定随机种子为{plan['split_seed']}，按类别及记录ID排序后使用NumPy PCG64分配60%训练、20%校准、20%预测。"
            "逐样品归属见`split_manifest.csv`，完整规则及来源hash见`split_plan.json`。\n\n"
            "ZY02只有5份，拆分为3／1／1，因此全部7类只能统一开展每类1份的校准；不能声称已具备7类3／5／10-shot条件。"
            "ID的ZY前缀及文件分组仍透露匿名类别，这些是用途隔离文件而非盲样。模型输入只能包含光谱数值，不能包含文件名、ID或标签列。\n\n"
            "用户确认每产地目前仅一批，故这是同批次样品留出实验，不是真实跨批次校准。"
            "全部855份数据此前已参与内部交叉验证；此次为回顾性重新划分，不是从未查看的外部确认集。\n")
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; prior splits cannot be overwritten")
    sources = select_workbooks(repo=args.repo, manifest=args.manifest)
    sources = sorted((s for s in sources if s.dataset == "zaoren"), key=lambda s: s.origin_group_id)
    metadata = args.metadata or args.repo / "data/metadata_template.csv"
    plan = build_split_plan(sources, metadata, args.seed)
    write_plan(args.output, plan)
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "counts": plan["role_counts"],
                          "plan": str(args.output / "split_plan.json")}, ensure_ascii=False))
        return
    results = []
    for source in sources:
        roles = {role: [row["record_id"] for row in plan["records"]
                        if row["source_workbook"] == source.relative_path and row["role"] == role]
                 for role in ROLES}
        results.extend(export_workbook(source, roles, args.output))
        print(json.dumps({"exported": source.origin_group_id, "role_counts": {r: len(ids) for r, ids in roles.items()}}, ensure_ascii=False), flush=True)
    _write_json(args.output / "workbook_verification.json", {"status": "passed", "workbooks": results,
                "split_plan_sha256": file_sha256(args.output / "split_plan.json"), "source_workbooks_unchanged": True,
                "role_sample_overlap": 0, "unique_independent_sample_ids": sum(plan["role_counts"].values())})
    write_readme(args.output, plan)
    print(json.dumps({"status": "completed", "workbooks": len(results), "counts": plan["role_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
