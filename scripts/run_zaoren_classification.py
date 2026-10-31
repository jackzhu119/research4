"""Run reviewed, physical-sample-grouped acid jujube classification paths."""
import argparse
import hashlib
import json
from pathlib import Path

from herb_spectra.classification import (DEFAULT_METHODS, METHODS, DatasetValidationError,
    load_reviewed_dataset, run_grouped_classification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reviewed paired-spectrum NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representations", nargs="+", choices=["alpha", "n", "concat"], default=["alpha"])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--cars-cv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20261031)
    parser.add_argument("--identity-scope", default="user-declared independent samples; internal CV")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    try:
        data = load_reviewed_dataset(args.input)
    except DatasetValidationError as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"status": "blocked", "models_not_run": True, "issues": exc.issues,
                  "input": str(args.input), "scope": "no model was fitted"}
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 2
    report = run_grouped_classification(data, representations=args.representations,
        methods=args.methods, outer_splits=args.outer_splits, inner_splits=args.inner_splits,
        cars_cv=args.cars_cv, seed=args.seed, split_seed=args.split_seed, output_dir=args.output,
        preflight_only=args.preflight_only, identity_scope=args.identity_scope)
    report["input"] = str(args.input)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "physical_samples": report["physical_sample_count"],
                      "cases": [{"representation": r["representation"], "method": r["method"],
                                 "status": r["status"]} for r in report["results"]]}, ensure_ascii=False))
    return 0 if report["status"] in {"completed", "ready", "partial", "not_evaluable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
