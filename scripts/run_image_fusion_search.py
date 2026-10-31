"""Run source-only probability fusion after its score-provider cache is ready."""
import argparse
import json
from pathlib import Path

from herb_spectra.image_fusion_search import run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "split-plan", "protocol", "score-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    result = run_search(args.input, args.split_plan, args.protocol, args.score_run, args.output,
                        stage=args.stage, resume=args.resume, release_path=args.release)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
