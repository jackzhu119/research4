"""Execute the predeclared source-only dense neural baseline search."""
import argparse
import json
from pathlib import Path

from herb_spectra.neural_base_search import run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "split-plan", "protocol", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    args = parser.parse_args()
    result = run_search(args.input,args.split_plan,args.protocol,args.output,resume=args.resume,stage=args.stage)
    print(json.dumps({"status":"completed","selected_candidate":result["candidate"]["id"],
                      "stage":result.get("stage","final_completed"),
                      "accuracy_mean":result.get("accuracy_mean")}),flush=True)


if __name__ == "__main__":
    main()
