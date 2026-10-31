"""Run separate guarded stages of paired external-plant neural transfer."""
import argparse
import json
from pathlib import Path
from herb_spectra.cross_plant_transfer import prepare, selection, evaluation, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "selection", "evaluation", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.stage in ("selection", "evaluation") and args.release is None:
        parser.error("Source and outer stages require separate root release files")
    operation = {"prepare": lambda: prepare(args.output), "selection": lambda: selection(args.output, args.release),
                 "evaluation": lambda: evaluation(args.output, args.release), "score": lambda: score(args.output)}[args.stage]
    print(json.dumps(operation(), ensure_ascii=False), flush=True)
