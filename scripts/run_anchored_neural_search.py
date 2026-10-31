"""Run source-anchored few-label updates after locking the dense source model."""
import argparse
import json
from pathlib import Path

from herb_spectra.anchored_neural_search import run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "split-plan", "protocol", "source-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_search(args.input, args.split_plan, args.protocol, args.source_run,
        args.output, stage=args.stage, resume=args.resume)
    print(json.dumps({"stage": args.stage, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
