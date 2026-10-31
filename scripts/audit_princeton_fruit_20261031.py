#!/usr/bin/env python3
"""Read-only, hash-bound audit of the public Princeton Meta-Sticker archive.

Never executes upstream code, emits no waveform arrays or per-observation target
values, and never trains a model. The upstream archive has no detected license;
keep the original ZIP and any numerical derivatives outside Git.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import openpyxl
from scipy.io import loadmat

COMMIT = "16943a2c499ce7918afb1e20a223267d94ad7a93"
SOURCE_URL = "https://github.com/PrincetonUniversity/-Sub-Terahertz-Fruit-Sensing"
SHEETS = {"Mango DM, Brix": ("MA", ("DM", "Brix")),
          "Pear Brix": ("PR", ("Brix",)),
          "Persimmon Brix": ("PM", ("Brix",))}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_timepoint(text: str) -> dt.datetime:
    """Interpret explicit workbook month/day and AM/PM in the archive's 2023 year."""
    return dt.datetime.strptime(text + "_2023", "%b%d_%I%p_%Y")


def workbook_targets(payload: bytes):
    book = openpyxl.load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
    target_lookup, summaries = {}, {}
    for sheet_name, (fruit_type, target_names) in SHEETS.items():
        rows = list(book[sheet_name].values)
        header = rows[0]
        assert list(header[1:]) == [fruit_type + str(i) for i in range(1, 11)]
        observed = collections.defaultdict(list)
        missing = []
        for row in rows[1:]:
            timestamp = parse_timepoint(row[0])
            for fruit_id, value in zip(header[1:], row[1:]):
                key = (fruit_id, timestamp.isoformat())
                if key in target_lookup:
                    raise ValueError(f"Duplicate workbook key {key}")
                if value is None:
                    target_lookup[key] = None
                    missing.append({"fruit_id": fruit_id, "timestamp": timestamp.isoformat()})
                    continue
                values = ([float(x.strip()) for x in value.split(",")]
                          if isinstance(value, str) else [float(value)])
                if len(values) != len(target_names) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid target schema {key}")
                target_lookup[key] = dict(zip(target_names, values))
                for name, val in zip(target_names, values):
                    observed[name].append(val)
        summaries[fruit_type] = {
            "sheet": sheet_name, "physical_fruit_ids": list(header[1:]),
            "physical_fruit_ids_count": len(header) - 1,
            "timepoints": len(rows) - 1,
            "missing_target_cells": missing,
            "targets": {name: {"count": len(vals), "min": min(vals), "max": max(vals),
                                "mean": float(np.mean(vals)), "std": float(np.std(vals))}
                        for name, vals in observed.items()},
        }
    return target_lookup, summaries


def mat_identity(member: str):
    parts = member.split("/")
    date = dt.datetime.strptime(parts[2], "%b%d%Y")
    hour = dt.datetime.strptime(parts[3], "%I%p").hour
    timestamp = date.replace(hour=hour).isoformat()
    filename = parts[-1]
    if len(parts) == 5:
        role = filename.split("_", 1)[0]
        if role not in {"Background", "Metal"}:
            raise ValueError(f"Unknown reference {member}")
        return {"role": role, "timestamp": timestamp, "fruit_type": "",
                "fruit_id": "", "condition": "", "canonical_name": True}
    if len(parts) != 7 or parts[4] not in {"MA", "PR", "PM"} or parts[5] not in {"Tag", "NT"}:
        raise ValueError(f"Unknown MAT path {member}")
    match = re.fullmatch(r"(MA|PR|PM)(10|[1-9])_(?:(NT|T)_)?(\d{4})_(am|pm)\.mat", filename)
    if match is None:
        raise ValueError(f"Unknown MAT name {member}")
    fruit_id = match[1] + match[2]
    if fruit_id[:2] != parts[4] or match[4] != date.strftime("%m%d"):
        raise ValueError(f"Inconsistent MAT metadata {member}")
    if (match[5] == "am") != (hour < 12):
        raise ValueError(f"Inconsistent AM/PM {member}")
    if match[3] is not None and (match[3] == "NT") != (parts[5] == "NT"):
        raise ValueError(f"Inconsistent Tag/NT {member}")
    return {"role": "Fruit", "timestamp": timestamp, "fruit_type": parts[4],
            "fruit_id": fruit_id, "condition": parts[5], "canonical_name": match[3] is not None}


def audit(archive: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty audit output {out}")
    original_hash = sha256(archive.read_bytes())
    rows, array_groups, identity_groups = [], collections.defaultdict(list), collections.defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        workbook_name = next(n for n in members if n.endswith(".xlsx"))
        targets, target_summary = workbook_targets(z.read(workbook_name))
        for member in members:
            if not member.endswith(".mat"):
                continue
            payload = z.read(member)
            parsed = loadmat(io.BytesIO(payload))
            variables = {k: v for k, v in parsed.items() if not k.startswith("__")}
            if len(variables) != 1:
                raise ValueError(f"Expected one waveform variable {member}")
            variable, array = next(iter(variables.items()))
            array = np.asarray(array)
            identity = mat_identity(member)
            row = {"member": member, **identity, "bytes": len(payload),
                   "file_sha256": sha256(payload), "variable": variable,
                   "shape": "x".join(map(str, array.shape)), "dtype": str(array.dtype),
                   "finite": bool(np.isfinite(array).all()),
                   "nonconstant": bool(np.ptp(array) > 0),
                   "array_sha256": sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()),
                   "target_present": "", "duplicate_identity": False}
            if row["shape"] != "1x7000" or row["dtype"] != "float64":
                raise ValueError(f"Unexpected waveform shape/dtype {member}")
            if identity["role"] == "Fruit":
                key = (identity["fruit_id"], identity["timestamp"])
                if key not in targets:
                    raise ValueError(f"No workbook timepoint for {member}")
                row["target_present"] = targets[key] is not None
                identity_groups[key + (identity["condition"],)].append(len(rows))
            rows.append(row)
            array_groups[row["array_sha256"]].append(len(rows)-1)
        duplicate_identities = []
        chosen = []
        for key, ids in sorted(identity_groups.items()):
            if len(ids) > 1:
                if len({rows[i]["array_sha256"] for i in ids}) != 1:
                    raise ValueError(f"Conflicting duplicate fruit/time/condition {key}")
                for i in ids:
                    rows[i]["duplicate_identity"] = True
                duplicate_identities.append({"fruit_id": key[0], "timestamp": key[1],
                                             "condition": key[2], "members": [rows[i]["member"] for i in ids]})
            selected = sorted(ids, key=lambda i: not rows[i]["canonical_name"])[0]
            chosen.append(rows[selected])
        missing = []
        for (fruit_id, timestamp), values in sorted(targets.items()):
            for condition in ("NT", "Tag"):
                if (fruit_id, timestamp, condition) not in identity_groups:
                    missing.append({"fruit_id": fruit_id, "timestamp": timestamp,
                                    "condition": condition, "target_present": values is not None})
        nested_names = [n for n in members if n.endswith("SupportingInfo.zip")]
        nested_hashes = collections.Counter(sha256(z.read(n)) for n in nested_names)
        nested_evidence = []
        # Audit every distinct nested calibration archive, without extracting files.
        for nested_name in nested_names:
            payload = z.read(nested_name)
            digest = sha256(payload)
            if any(r["sha256"] == digest for r in nested_evidence):
                continue
            with zipfile.ZipFile(io.BytesIO(payload)) as nested:
                reference = nested.read("Reference.picotd").decode(errors="replace")
                lines = reference.splitlines()
                data = [line.split() for line in lines if re.match(r"^\d+\.\d+\s", line)]
                blank = nested.read("Blank.xml").decode(errors="replace")
                nested_evidence.append({"member": nested_name, "sha256": digest,
                    "same_byte_copies": nested_hashes[digest], "members": nested.namelist(),
                    "reference_header": lines[:4],
                    "reference_axis_points": len(data),
                    "reference_axis_first": float(data[0][0]), "reference_axis_last": float(data[-1][0]),
                    "reference_axis_step": float(data[1][0])-float(data[0][0]),
                    "explicit_axis_unit_found": False,
                    "blank_xml_units_field": re.findall(r"<Units>(.*?)</Units>", blank),
                    "unit_warning": "Blank.xml Units=mm is a product unit, not an explicit waveform time unit; 2021 reference is not proof of 2023 MAT axis or calibration."})
        counts = collections.Counter(row["role"] for row in rows)
        duplicate_arrays = [[rows[i]["member"] for i in ids] for ids in array_groups.values() if len(ids) > 1]
        distinct = {
            ft: {"physical_fruit_ids_count": len({r["fruit_id"] for r in chosen if r["fruit_type"] == ft}),
                 "unique_waveforms_by_condition": dict(collections.Counter(r["condition"] for r in chosen if r["fruit_type"] == ft)),
                 "labelled_unique_waveforms_by_condition": dict(collections.Counter(r["condition"] for r in chosen if r["fruit_type"] == ft and r["target_present"])),
                 "fruit_time_pairs_with_any_trace": len({(r["fruit_id"], r["timestamp"]) for r in chosen if r["fruit_type"] == ft}),
                 "timepoints": len({r["timestamp"] for r in chosen if r["fruit_type"] == ft})}
            for ft in ("MA", "PR", "PM")}
        report = {
            "audit_version": "princeton-fruit-feasibility-20261031-v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "archive": str(archive), "archive_sha256": original_hash, "archive_bytes": archive.stat().st_size,
            "source_url": SOURCE_URL, "commit": COMMIT,
            "archive_members": len(members), "archive_uncompressed_bytes": sum(i.file_size for i in z.infolist()),
            "extensions": dict(collections.Counter(Path(n).suffix.lower() for n in members)),
            "detected_license_files": [n for n in members if re.search(r"(^|/)(LICENSE|LICENCE|COPYING)(\.|$)", n, re.I)],
            "workbook_sha256": sha256(z.read(workbook_name)), "targets": target_summary,
            "mat_files": len(rows), "mat_roles": dict(counts),
            "all_waveforms_finite": all(r["finite"] for r in rows),
            "all_waveforms_nonconstant": all(r["nonconstant"] for r in rows),
            "waveform_shapes": dict(collections.Counter(r["shape"] for r in rows)),
            "variable_names": dict(collections.Counter(r["variable"] for r in rows)),
            "physical_fruit_ids_count": len({r["fruit_id"] for r in chosen}),
            "distinct_timepoints": len({r["timestamp"] for r in chosen}),
            "duplicate_identity_keys_count": len(duplicate_identities),
            "duplicate_identity_excess_files": sum(len(r["members"])-1 for r in duplicate_identities),
            "duplicate_identity_keys": duplicate_identities,
            "exact_array_duplicate_groups": duplicate_arrays,
            "unique_fruit_waveforms": len(chosen),
            "distinct_by_fruit_type": distinct,
            "expected_missing_waveforms": missing,
            "nested_supporting_archives": len(nested_names), "nested_supporting_unique_byte_hashes": len(nested_hashes),
            "nested_supporting_evidence": nested_evidence,
            "models_trained": 0,
            "eligibility": {"exploratory_grouped_waveform_regression": "CONDITIONAL",
                "validated_physical_spectral_regression": "BLOCKED_AXIS_AND_REFERENCE_PROCESSING_UNVERIFIED",
                "acid_jujube_origin_transfer": "NOT_COMPATIBLE_WITHOUT_ADDITIONAL_PHYSICS_AND_TASK_DESIGN",
                "numeric_data_redistribution": "BLOCKED_NO_LICENSE_DETECTED"},
            "array_exported": False, "per_observation_targets_exported": False,
        }
    report["input_unchanged"] = sha256(archive.read_bytes()) == original_hash
    with (out/"mat_inventory.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mat_files","physical_fruit_ids_count","distinct_timepoints",
        "duplicate_identity_excess_files","unique_fruit_waveforms","input_unchanged","models_trained")},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.archive,args.output)


if __name__ == "__main__":
    main()
