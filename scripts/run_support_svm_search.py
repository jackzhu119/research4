"""Run the source-frozen support-weighted SVM parameter adaptation round."""
import argparse
import json
from pathlib import Path

from herb_spectra.support_svm_search import run_support_svm_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "split-plan", "protocol", "source-run", "final-source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    args = parser.parse_args()
    result = run_support_svm_search(args.input, args.split_plan, args.protocol, args.source_run,
        args.final_source, args.output, resume=args.resume, stage=args.stage)
    print(json.dumps({"status": "completed", "summary": result}))


if __name__ == "__main__":
    main()
