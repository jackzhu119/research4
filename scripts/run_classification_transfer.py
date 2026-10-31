"""Evaluate known-origin, genuinely new-batch classification calibration.

The input source/development/target NPZ files must have disjoint physical IDs
and verified, role-disjoint batch IDs. A record ID is never invented as a batch.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from herb_spectra.classification import DatasetValidationError, load_reviewed_dataset
from herb_spectra.classification_transfer import (METHODS, TransferProtocolError, prepare_domains,
                                                run_classification_transfer)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("source", "development", "target"):
        parser.add_argument("--" + role, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=["alpha", "n", "concat"], default="alpha")
    parser.add_argument("--model", choices=["svc", "plsda"], default="svc")
    parser.add_argument("--selector", choices=["full", "mcuve20", "mcuve100", "stable", "cars"], default="full")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Positive independent support samples per class; 0-shot source reference is automatic")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--query-fraction", type=float, default=.5)
    parser.add_argument("--minimum-query-per-class", type=int, default=10)
    parser.add_argument("--source-cv-splits", type=int, default=3)
    parser.add_argument("--cars-cv", type=int, default=3)
    parser.add_argument("--regularizations", type=float, nargs="+", default=[.1, 1., 10., 100.])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty experiment output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        registry = {role: {"path": str(getattr(args, role)),
                           "sha256": hashlib.sha256(getattr(args, role).read_bytes()).hexdigest()}
                    for role in ("source", "development", "target")}
        write_json(args.output / "inputs.json", registry)
        data = {role: load_reviewed_dataset(getattr(args, role)) for role in registry}
        if args.preflight_only:
            prepared = prepare_domains(**data)
            status = {"status": "ready_for_classification_transfer", "preflight_only": True,
                      "domains": {role: {"physical_samples": len(item["y"]),
                                          "batch_ids": sorted(set(item["batch_ids"].tolist()))}
                                  for role, item in prepared.items()},
                      "models_fitted": 0}
            write_json(args.output / "status.json", status)
            print(json.dumps(status, ensure_ascii=False))
            return 0
        result = run_classification_transfer(**data, modality=args.representation, model=args.model,
            selector=args.selector, shots=tuple(args.shots), repeats=args.repeats,
            query_fraction=args.query_fraction, minimum_query_per_class=args.minimum_query_per_class,
            source_cv_splits=args.source_cv_splits, cars_cv=args.cars_cv,
            regularizations=tuple(args.regularizations), methods=tuple(args.methods), seed=args.seed,
            memory=str(args.output / ".cache"))
    except (DatasetValidationError, TransferProtocolError, FileNotFoundError, KeyError) as exc:
        status = {"status": "blocked", "reason": str(exc),
                  "issues": list(getattr(exc, "issues", [str(exc)])), "classification_results_available": False}
        write_json(args.output / "status.json", status)
        print(json.dumps(status, ensure_ascii=False))
        return 2
    except (ValueError, RuntimeError) as exc:
        status = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        write_json(args.output / "status.json", status)
        print(json.dumps(status, ensure_ascii=False))
        return 1
    predictions = result.pop("prediction_rows")
    write_json(args.output / "results.json", result)
    write_json(args.output / "partitions.json", result["partitions"])
    write_json(args.output / "summary.json", result["summary"])
    with (args.output / "sample_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "shots_per_class", "repeat", "sample_id",
                                                    "y_true", "y_predicted", "input_record_ids"], lineterminator="\n")
        writer.writeheader()
        for row in predictions:
            writer.writerow({**row, "input_record_ids": json.dumps(row["input_record_ids"], ensure_ascii=False)})
    with (args.output / "record_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "shots_per_class", "repeat", "sample_id",
                                                    "record_id", "y_true", "y_predicted"], lineterminator="\n")
        writer.writeheader()
        for row in predictions:
            for identifier in row["input_record_ids"]:
                writer.writerow({**{k: row[k] for k in ["method", "shots_per_class", "repeat", "sample_id",
                                                        "y_true", "y_predicted"]}, "record_id": identifier})
    status = {"status": "completed", "scope": "verified known-class cross-batch protocol",
              "evaluations": len(result["rows"]), "sample_predictions": len(predictions),
              "uncertainty_scope": result["uncertainty_scope"]}
    write_json(args.output / "status.json", status)
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
