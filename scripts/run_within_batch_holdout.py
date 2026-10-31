"""Run a frozen three-role, retrospective same-batch calibration comparison."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from herb_spectra.within_batch_holdout import (fit_frozen_predictions, prepare_holdout,
                                             score_prediction_bundle)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True, help="Frozen research protocol JSON recorded before fitting")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selectors", nargs="+", choices=["full", "mcuve20"], default=["full", "mcuve20"])
    parser.add_argument("--shots", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draw-seed", type=int, default=42)
    parser.add_argument("--source-cv-splits", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=1.)
    parser.add_argument("--frequency-min", type=float, default=.1)
    parser.add_argument("--frequency-max", type=float, default=1.5)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite existing experiment results")
    if not args.protocol.is_file():
        raise FileNotFoundError("A predeclared protocol JSON is required")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("Protocol JSON must be a nonempty object")
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    expected = {"selectors": [row["selector"] for row in protocol["models"]],
        "shots": protocol["calibration"]["shots_per_class"],
        "repeats": protocol["calibration"]["repeats"], "seed": protocol["model_seed"],
        "draw_seed": protocol["support_seed"],
        "source_cv_splits": protocol["training"]["cross_validation_folds"],
        "regularization": protocol["calibration"]["regularization"],
        "frequency_min": protocol["frequency"]["lower_inclusive"],
        "frequency_max": protocol["frequency"]["upper_inclusive"]}
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("CLI options disagree with frozen protocol; record a distinct protocol for another experiment")
    if (plan.get("dataset_version") != protocol["dataset_version"]
            or plan.get("split_seed") != protocol["split_seed"]
            or plan.get("counts_by_origin") != protocol["counts_by_origin"]):
        raise ValueError("Split plan disagrees with frozen protocol")
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    roles, truth, preparation = prepare_holdout(arrays, plan,
        frequency_min=args.frequency_min, frequency_max=args.frequency_max)
    if preparation["feature_count"] != protocol["frequency"]["expected_points"]:
        raise ValueError("Frequency point count disagrees with frozen protocol")
    bundle = fit_frozen_predictions(**roles, selectors=tuple(args.selectors), shots=tuple(args.shots),
        repeats=args.repeats, seed=args.seed, draw_seed=args.draw_seed,
        source_cv_splits=args.source_cv_splits, regularization=args.regularization,
        output_dir=args.output)
    # Persist predictions before the evaluation boundary consumes query truth.
    write_json(args.output / "prediction_bundle.json", bundle)
    scored = score_prediction_bundle(bundle, truth)
    rows = scored.pop("prediction_rows")
    with (args.output / "sample_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    write_json(args.output / "results.json", scored)
    write_json(args.output / "summary.json", scored["summary"])
    write_json(args.output / "preparation.json", preparation)
    write_json(args.output / "inputs.json", {
        "input": {"path": str(args.input), "sha256": digest(args.input)},
        "split_plan": {"path": str(args.split_plan), "sha256": digest(args.split_plan)},
        "protocol": {"path": str(args.protocol), "sha256": digest(args.protocol)},
        "python": platform.python_version(), "dependencies": {name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn", "joblib"]},
        "argv": sys.argv,
        "source_files_sha256": {str(path): digest(path) for path in [
            Path(__file__), Path(__file__).resolve().parents[1] / "src/herb_spectra/within_batch_holdout.py",
            Path(__file__).resolve().parents[1] / "src/herb_spectra/classification.py",
            Path(__file__).resolve().parents[1] / "src/herb_spectra/classification_transfer.py",
            Path(__file__).resolve().parents[1] / "src/herb_spectra/calibration.py"]}})
    write_json(args.output / "status.json", {"status": "completed", "scope": scored["scope"],
        "evaluations": len(scored["rows"]), "prediction_rows": len(rows),
        "source_model_count": len(bundle["models"]), "query_truth_used_for_fitting": False})
    print(json.dumps({"status": "completed", "evaluations": len(scored["rows"]),
                      "summary": scored["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
