"""Run the recorded, CPU-sized paper-inspired few-shot classification pilot."""
import argparse
import json
from pathlib import Path

from herb_spectra.paper_fewshot_classification import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input","split-plan","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    summaries = run_pilot(args.input,args.split_plan,args.protocol,args.output,resume=args.resume)
    print(json.dumps({"status":"completed","summary_rows":len(summaries)}))


if __name__ == "__main__":
    main()
