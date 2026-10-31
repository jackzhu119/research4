"""Run separately frozen development or released final calibration evaluation."""
import argparse
import json
from pathlib import Path

from herb_spectra.paper_c_calibration import evaluate, select


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("select","evaluate"))
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--release",type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.phase=="select":
        value=select(repo,args.protocol,args.output)
        print(json.dumps({"status":"selected","lock":str(args.output/"selection_lock.json"),
                          "fallback":{k:v["fallback_to_identity"] for k,v in value["choices"].items()}}))
    else:
        if args.release is None:
            parser.error("--release is required for evaluation")
        value=evaluate(repo,args.protocol,args.output,args.release)
        print(json.dumps({"status":"evaluated","summary_rows":len(value)}))


if __name__=="__main__":
    main()
