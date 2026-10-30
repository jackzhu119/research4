"""Create scan-to-sample annotation rows without inferring scientific labels."""
import argparse
import csv
import hashlib
from pathlib import Path
import warnings

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata_template.csv"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep existing annotations; choose a new output path")
    warnings.filterwarnings("ignore", message="Workbook contains no default style")
    rows = []
    for path in sorted(args.data_root.rglob("*.xlsx")):
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb["吸收系数"]
            ws.reset_dimensions()
            header = next(ws.iter_rows(values_only=True), ())
            for i, name in enumerate(header):
                if i == 0 or name is None or not str(name).strip() or "参考" in str(name):
                    continue
                key = f"{path.as_posix()}|{ws.title}|{i + 1}"
                rows.append({"scan_id": hashlib.sha256(key.encode()).hexdigest()[:20],
                             "source_workbook": path.as_posix(), "sheet": ws.title,
                             "excel_column": get_column_letter(i + 1), "scan_header": str(name),
                             "physical_sample_id": "", "origin": "", "origin_verified": "",
                             "batch_id": "", "instrument_id": "", "measurement_date": "",
                             "species_verified": "", "assay_value": "", "assay_unit": "",
                             "assay_method": "", "notes": ""})
        finally:
            wb.close()
    if not rows:
        raise ValueError("No spectrum columns found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"Created {len(rows)} unlabelled scan annotation rows: {args.output}")


if __name__ == "__main__":
    main()
