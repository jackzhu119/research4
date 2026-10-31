"""Run the fixed historical strong MLP on the original five/two folds."""
import argparse
import json
from pathlib import Path

from herb_spectra.paper_c_strong_mlp import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("selection", "evaluation"), default="selection")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.output, stage=args.stage, release_path=args.release), ensure_ascii=False), flush=True)
