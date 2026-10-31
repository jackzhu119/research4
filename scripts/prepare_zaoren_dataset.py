"""Audit and prepare only registered paired n/alpha acid-jujube workbooks."""
import argparse
import json
from pathlib import Path

from herb_spectra.zaoren_data import prepare_zaoren_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata_template.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-duplicate-limit", type=int, default=20)
    args = parser.parse_args()
    if args.near_duplicate_limit < 0:
        parser.error("--near-duplicate-limit cannot be negative")
    report = prepare_zaoren_dataset(manifest=args.manifest, metadata=args.metadata,
                                    output=args.output, near_duplicate_limit=args.near_duplicate_limit)
    print(json.dumps({"quality_status": report["quality_status"],
                      "record_count": report.get("record_count"),
                      "npz_written": report["npz_written"],
                      "readiness": report.get("readiness"),
                      "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["quality_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
