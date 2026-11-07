"""Run the isolated, released supplemental benchmark without altering history."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import json
from pathlib import Path
from herb_spectra.supplemental_execution import preflight, run_source, run_predict, run_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", required=True, help="Repo-relative bound protocol JSON")
    parser.add_argument("--stage", choices=("preflight", "source", "predict", "score"), default="preflight")
    parser.add_argument("--output", help="New repo-relative reports directory")
    parser.add_argument("--release", help="Repo-relative root-reviewed release JSON")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight(args.repo, args.protocol)
    else:
        if not args.output:
            parser.error("--output is required for a released execution stage")
        if args.stage == "source":
            result = run_source(args.repo, args.protocol, args.output, args.release)
        elif args.stage == "predict":
            result = run_predict(args.repo, args.protocol, args.output, args.release)
        else:
            result = run_score(args.repo, args.protocol, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
