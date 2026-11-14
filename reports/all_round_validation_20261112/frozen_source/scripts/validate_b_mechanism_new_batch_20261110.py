#!/usr/bin/env python3
"""Validate prospective batch metadata; never fit or evaluate a model.

Spectrum references use ``path.csv`` (two numeric columns) or
``path.xlsx#sheet=Absorption&column=sample_column``.  The first spreadsheet
column is the measured axis. Blank scientific metadata is never imputed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b_mechanism_20261110"
FIELDS = [
    "physical_sample_id", "user_confirmed_species", "origin_group_id",
    "true_batch_id", "collection_date", "instrument_id", "measurement_date",
    "preparation_protocol", "thickness", "moisture", "spectrum_file",
    "label_verification_basis", "support_or_query_role",
]
REQUIRED = [f for f in FIELDS if f not in {"thickness", "moisture"}]
EXPERIMENT_STATUS = "NOT_RUN_NEEDS_NEW_BATCH_DATA"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spectrum_values(reference: str, metadata_dir: Path):
    """Return exact referenced axis/signal and file hash, without directory search."""
    raw_path, _, fragment = reference.partition("#")
    path = Path(raw_path)
    if not path.is_absolute():
        path = metadata_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"spectrum file does not exist: {path}")
    options = parse_qs(fragment, strict_parsing=True) if fragment else {}
    if path.suffix.lower() == ".csv":
        selected_column = "csv-two-columns"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows:
            raise ValueError("empty spectrum CSV")
        if options:
            raise ValueError("CSV references must have exactly two columns and no fragment")
        # Exactly one optional header row; malformed later rows are not dropped.
        try:
            float(rows[0][0]); float(rows[0][1])
        except (ValueError, IndexError):
            rows = rows[1:]
        if any(len(r) != 2 for r in rows):
            raise ValueError("spectrum CSV requires exactly axis,signal columns")
        pairs = [(float(r[0]), float(r[1])) for r in rows]
    elif path.suffix.lower() == ".xlsx":
        if set(options) != {"sheet", "column"} or any(len(v) != 1 for v in options.values()):
            raise ValueError("XLSX reference requires #sheet=exact_sheet&column=exact_header")
        from openpyxl import load_workbook
        selected_column = json.dumps({k: v[0] for k, v in options.items()}, sort_keys=True, ensure_ascii=False)
        book = load_workbook(path, data_only=True, read_only=True)
        try:
            sheet = book[options["sheet"][0]]
            rows_iter = sheet.iter_rows(values_only=True)
            header = next(rows_iter)
            hits = [i for i, value in enumerate(header) if str(value) == options["column"][0]]
            if len(hits) != 1 or hits[0] == 0:
                raise ValueError("sample column must match exactly one non-axis column")
            col = hits[0]
            pairs = [(float(row[0]), float(row[col])) for row in rows_iter]
        finally:
            book.close()
    else:
        raise ValueError("supported spectrum formats are CSV or explicitly selected XLSX columns")
    if len(pairs) < 2:
        raise ValueError("spectrum requires at least two coordinates")
    if not all(math.isfinite(a) and math.isfinite(y) for a, y in pairs):
        raise ValueError("spectrum contains a non-finite value")
    axis = [p[0] for p in pairs]
    if not all(b > a for a, b in zip(axis, axis[1:])):
        raise ValueError("spectrum axis must be strictly increasing and unique")
    return {"path": str(path), "sha256": sha256(path), "selected_column": selected_column,
            "n_points": len(pairs), "axis_min": axis[0], "axis_max": axis[-1],
            "axis_sha256": hashlib.sha256(json.dumps(axis, separators=(",", ":")).encode()).hexdigest()}


def validate(metadata: Path, historical_manifest: Path | None = None) -> dict:
    errors, warnings = [], []
    if not metadata.is_file():
        return {"experiment_status": EXPERIMENT_STATUS, "validation_status": "BLOCKED",
                "errors": [f"metadata does not exist: {metadata}"], "n_samples": 0}
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        rows = [{k: (v or "").strip() for k, v in row.items() if k is not None}
                for row in reader if any((v or "").strip() for v in row.values() if isinstance(v, str))]
    missing = [name for name in FIELDS if name not in headers]
    if missing:
        errors.append(f"missing metadata columns: {missing}")
    if len(set(headers)) != len(headers):
        errors.append("duplicate metadata column names")
    old_ids = set()
    old_hash = None
    if historical_manifest and historical_manifest.is_file():
        old_hash = sha256(historical_manifest)
        with historical_manifest.open(encoding="utf-8-sig", newline="") as stream:
            old_ids = {r.get("physical_sample_id", "") for r in csv.DictReader(stream)} - {""}
    seen_ids, seen_spectra, spectra, counts = set(), set(), {}, {}
    for line, row in enumerate(rows, start=2):
        prefix = f"line {line}"
        for name in REQUIRED:
            if not row.get(name):
                errors.append(f"{prefix}: missing {name}; supply source records, never infer")
        physical_id = row.get("physical_sample_id", "")
        if physical_id in seen_ids:
            errors.append(f"{prefix}: duplicate physical_sample_id {physical_id}; roles cannot overlap")
        if physical_id:
            seen_ids.add(physical_id)
        if physical_id in old_ids:
            errors.append(f"{prefix}: physical_sample_id overlaps historical master dataset: {physical_id}")
        role = row.get("support_or_query_role", "")
        if role not in {"support", "query"}:
            errors.append(f"{prefix}: role must be support or query")
        parsed_dates = {}
        for name in ("collection_date", "measurement_date"):
            if row.get(name):
                try:
                    parsed_dates[name] = date.fromisoformat(row[name])
                except ValueError:
                    errors.append(f"{prefix}: {name} must be an actual ISO YYYY-MM-DD date")
        if len(parsed_dates) == 2 and parsed_dates["measurement_date"] < parsed_dates["collection_date"]:
            errors.append(f"{prefix}: measurement precedes collection")
        for name in ("thickness", "moisture"):
            if not row.get(name):
                warnings.append(f"{prefix}: {name} unavailable; retained missing")
            else:
                try:
                    value = float(row[name])
                    if not math.isfinite(value) or value < 0 or (name == "thickness" and value == 0):
                        raise ValueError()
                except ValueError:
                    errors.append(f"{prefix}: {name} must be finite and nonnegative (thickness > 0)")
        reference = row.get("spectrum_file", "")
        if reference:
            try:
                spectrum = spectrum_values(reference, metadata.parent)
                # The same file may contain many physical columns, but not the same selected column twice.
                spectrum_key = (spectrum["path"], spectrum["selected_column"])
                if spectrum_key in seen_spectra:
                    errors.append(f"{prefix}: spectral column reused by multiple physical IDs")
                seen_spectra.add(spectrum_key)
                spectra[reference] = spectrum
            except (ValueError, OSError, KeyError, StopIteration, ImportError, IndexError) as exc:
                errors.append(f"{prefix}: invalid spectrum reference: {exc}")
        group = "|".join(row.get(k, "") for k in
                         ("user_confirmed_species", "origin_group_id", "true_batch_id"))
        counts.setdefault(group, {"support": 0, "query": 0})
        if role in {"support", "query"}:
            counts[group][role] += 1
    for group, tally in counts.items():
        if tally["support"] < 10 or tally["query"] < 20:
            warnings.append(f"{group}: prospective plan is 10 support + 20 independent query; actual {tally}")
    status = "BLOCKED" if errors else ("VALIDATED_NOT_EVALUATED" if rows else EXPERIMENT_STATUS)
    return {"experiment_status": EXPERIMENT_STATUS, "validation_status": status,
            "n_samples": len(rows), "n_unique_physical_sample_ids": len(seen_ids),
            "metadata_path": str(metadata.resolve()), "metadata_sha256": sha256(metadata),
            "historical_manifest_sha256": old_hash, "group_role_counts": counts,
            "spectrum_audit": spectra, "errors": errors, "warnings": warnings,
            "no_model_fitted": True, "no_batch_id_generated": True,
            "limitations": ["Metadata validation is not independent botanical identification.",
                            "Instrument calibration, axis units, thickness/moisture units and batch provenance need source-document review.",
                            "No new-batch evaluation has been run."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=OUT / "new_batch_metadata_template.csv")
    parser.add_argument("--historical-manifest", type=Path,
                        default=ROOT / "data/research/species_revision_20261108/split_manifest.csv")
    parser.add_argument("--report", type=Path, default=OUT / "new_batch_validation.json")
    parser.add_argument("--dry-run", action="store_true", help="Read/validate only; write no files")
    parser.add_argument("--resume", action="store_true", help="Reuse an exactly matching validation receipt")
    parser.add_argument("--verify", action="store_true", help="Compare current evidence with a saved receipt; never fit")
    args = parser.parse_args(argv)
    result = validate(args.metadata, args.historical_manifest)
    if args.verify:
        if not args.report.is_file():
            result["errors"].append("no saved validation receipt; run once without --verify")
        else:
            saved = json.loads(args.report.read_text(encoding="utf-8"))
            if saved != result:
                result["errors"].append("saved receipt differs from current metadata/spectrum audit")
    elif not args.dry_run:
        if args.resume and args.report.exists():
            saved = json.loads(args.report.read_text(encoding="utf-8"))
            if saved != result:
                raise SystemExit("Input hash/audit changed; select a new --report path for a fresh validation.")
        else:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
