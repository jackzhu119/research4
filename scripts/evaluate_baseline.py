"""Evaluate a reviewed numeric dataset; never infer labels from raw filenames."""
import argparse
import json
from pathlib import Path

import numpy as np
from herb_spectra.validation import nested_evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Reviewed NPZ with X, y, groups, labels_verified")
    parser.add_argument("--model", choices=["svc", "svr", "pls"], default="svc")
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        if "labels_verified" not in data:
            raise ValueError("Verify origin/assay labels and physical-sample IDs before evaluation")
        verified = np.asarray(data["labels_verified"])
        if verified.dtype != np.dtype(bool) or verified.size == 0 or not verified.all():
            raise ValueError("labels_verified must contain explicit true Boolean values")
        X, y, groups = data["X"], data["y"], data["groups"]
    result = nested_evaluate(X, y, groups, model=args.model,
                             outer_splits=args.outer_splits,
                             inner_splits=args.inner_splits, seed=args.seed)
    result["predictions"] = result["predictions"].tolist()
    result["configuration"] = {"model": args.model, "outer_splits": args.outer_splits,
                               "inner_splits": args.inner_splits, "seed": args.seed,
                               "scope": "within-dataset grouped CV, not external-domain validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
