"""Run a frozen source-only paper-C comparison, then released evaluation."""
import argparse
import json
from pathlib import Path
from herb_spectra.paper_c_benchmark import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("selection", "evaluation"), default="selection")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    result = run(args.protocol, args.output, stage=args.stage, release_path=args.release)
    print(json.dumps(result, ensure_ascii=False), flush=True)
