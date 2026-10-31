"""Run source-anchored FO-MAML refinement after the MLP development lock."""
import argparse
import json
from pathlib import Path

from herb_spectra.meta_refinement_search import run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "split-plan", "protocol", "base-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_search(args.input, args.split_plan, args.protocol, args.base_run, args.output,
                        stage=args.stage, resume=args.resume)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
