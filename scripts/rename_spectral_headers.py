"""Give spectral header IDs a workbook-specific prefix without rewriting data.

Only matching row-one header cells are changed inside the XLSX ZIP package.
All remaining XML and other package members must be byte-for-byte unchanged.
This assigns record identifiers; it does not establish physical-sample identity.
"""
import argparse
from copy import copy
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import re
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ROW = re.compile(rb'<row\b[^>]*\br="1"[^>]*>.*?</row>', re.DOTALL)
CELL = re.compile(rb'<c\b[^>]*>.*?</c>', re.DOTALL)
LABEL = re.compile(r"^(?:样品|样本)\s*([0-9]+)(.*)$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def first_row(data):
    match = ROW.search(data)
    if match is None:
        raise ValueError("A standard row-one XML header is required")
    return match


def cell_text(element, strings):
    kind = element.get("t")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in element.iter("t"))
    value = element.find("v")
    if value is None:
        return ""
    if kind == "s":
        return strings[int(value.text)]
    return value.text or ""


def rename_one(source, group, directory):
    source = Path(source)
    destination = directory / (source.stem + "_编号版.xlsx")
    mapping_file = directory / (source.stem + "_编号对照.csv")
    if source.resolve() == destination.resolve() or destination.exists() or mapping_file.exists():
        raise FileExistsError("Choose a fresh output directory; inputs and outputs are preserved")
    temporary = destination.with_suffix(".xlsx.partial")
    records, patches, sheet_counts = [], {}, {}
    with ZipFile(source) as original:
        strings = []
        if "xl/sharedStrings.xml" in original.namelist():
            root = ET.fromstring(original.read("xl/sharedStrings.xml"))
            strings = ["".join(node.text or "" for node in item.iter(f"{{{MAIN}}}t"))
                       for item in root]
        relations = ET.fromstring(original.read("xl/_rels/workbook.xml.rels"))
        targets = {item.get("Id"): item.get("Target") for item in relations}
        workbook = ET.fromstring(original.read("xl/workbook.xml"))
        expected_indices = None
        for sheet in workbook.find(f"{{{MAIN}}}sheets"):
            target = targets[sheet.get(f"{{{REL}}}id")]
            member = (target.lstrip("/") if target.startswith("/")
                      else posixpath.normpath("xl/" + target))
            if not member.startswith("xl/worksheets/"):
                continue
            data = original.read(member)
            row = first_row(data)
            indices, changes = set(), []

            def replace_cell(match):
                raw = match.group(0)
                element = ET.fromstring(raw)
                old = cell_text(element, strings)
                label = LABEL.fullmatch(old.strip())
                if label is None:
                    return raw
                number = int(label.group(1))
                if number < 1 or number in indices:
                    raise ValueError(f"Nonpositive or duplicate sample number in {sheet.get('name')}")
                indices.add(number)
                identifier = f"{group}-{number:03d}"
                suffix = label.group(2)
                new = identifier + suffix
                opener = raw[:raw.index(b">") + 1]
                opener = re.sub(rb'\s+t="[^"]*"', b"", opener)
                replacement = (opener[:-1] + b' t="inlineStr"><is><t>'
                               + escape(new).encode("utf-8") + b"</t></is></c>")
                records.append({"source_file": source.name, "group_id": group,
                                "original_index": number, "new_id": identifier,
                                "sheet": sheet.get("name"), "cell": element.get("r"),
                                "old_header": old, "new_header": new,
                                "preserved_suffix": suffix})
                changes.append((element.get("r"), raw, replacement))
                return replacement

            updated_row = CELL.sub(replace_cell, row.group(0))
            if not indices:
                continue
            if expected_indices is None:
                expected_indices = indices
            elif indices != expected_indices:
                raise ValueError("Sample-number sets differ between worksheets; review pairing first")
            sheet_counts[sheet.get("name")] = len(indices)
            updated = data[:row.start()] + updated_row + data[row.end():]
            patches[member] = {"data": updated, "changes": changes,
                               "original_sha256": digest(data), "output_sha256": digest(updated)}
        if not patches:
            raise ValueError("No sample headers were found; no workbook was changed")
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
                output.comment = original.comment
                for info in original.infolist():
                    output.writestr(copy(info), patches[info.filename]["data"]
                                    if info.filename in patches else original.read(info.filename))
            with ZipFile(temporary) as output:
                if output.namelist() != original.namelist():
                    raise AssertionError("XLSX package member list changed")
                for info in original.infolist():
                    before, after = original.read(info.filename), output.read(info.filename)
                    if info.filename not in patches:
                        if before != after:
                            raise AssertionError("An unrelated XLSX member changed")
                        continue
                    before_row, after_row = first_row(before), first_row(after)
                    if (before[:before_row.start()] != after[:after_row.start()]
                            or before[before_row.end():] != after[after_row.end():]):
                        raise AssertionError("Data outside the header row changed")
                    restored = after_row.group(0)
                    for _, old_cell, new_cell in patches[info.filename]["changes"]:
                        if restored.count(new_cell) != 1:
                            raise AssertionError("Renamed cell is missing or ambiguous")
                        restored = restored.replace(new_cell, old_cell, 1)
                    if restored != before_row.group(0):
                        raise AssertionError("An unrelated row-one cell changed")
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    with mapping_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    sample_ids = sorted({record["new_id"] for record in records})
    return {"source_file": source.name, "source_sha256": digest(source.read_bytes()),
            "group_id": group, "output_file": destination.name,
            "output_sha256": digest(destination.read_bytes()),
            "mapping_file": mapping_file.name, "record_count": len(sample_ids),
            "sample_ids": sample_ids, "sheet_counts": sheet_counts,
            "header_cells_changed": len(records),
            "verification": {"all_non_header_xml_bytes_unchanged": True,
                             "all_unrelated_package_members_unchanged": True,
                             "coordinate_and_reference_headers_unchanged": True,
                             "sample_id_sets_match_across_sheets": True,
                             "original_suffixes_preserved": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", action="append", required=True,
                        help="Repeat PREFIX=PATH, e.g. ZY01=样本1.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    entries, groups = [], set()
    for item in args.workbook:
        group, source = item.split("=", 1)
        if not re.fullmatch(r"ZY[0-9]{2}", group) or group in groups:
            raise ValueError("Each workbook requires its own unique ZY01-style prefix")
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        groups.add(group)
        entries.append((group, source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.output_dir / "编号核验.json"
    if manifest_file.exists():
        raise FileExistsError("A previous numbering manifest exists; use a fresh output directory")
    results = [rename_one(source, group, args.output_dir) for group, source in entries]
    identifiers = [identifier for result in results for identifier in result["sample_ids"]]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Identifiers overlap across workbooks")
    manifest = {"identifier_scope": "workbook-specific spectrum records, not verified physical specimens",
                "unique_record_ids": len(identifiers), "cross_workbook_ids_unique": True,
                "workbooks": results}
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"workbooks": len(results), "unique_record_ids": len(identifiers),
                      "manifest": str(manifest_file)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
