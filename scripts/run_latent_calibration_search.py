"""Run latent-readout adaptation after a verified source-only spectral search."""
import argparse
import json
from pathlib import Path

from herb_spectra.latent_calibration_search import run_latent_search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--split-plan",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    rows=run_latent_search(args.input,args.split_plan,args.protocol,args.source_run,args.output,resume=args.resume)
    print(json.dumps({"status":"completed","summary_rows":len(rows),"output":str(args.output)}),flush=True)


if __name__=="__main__":
    main()
