"""Run source-only feature/classifier search; final requires global release."""
import argparse
import json
from pathlib import Path

from herb_spectra.image_feature_search import run_search


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("input","split-plan","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--stage",choices=("development","final"),default="development")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    result=run_search(args.input,args.split_plan,args.protocol,args.output,stage=args.stage,resume=args.resume)
    print(json.dumps({"stage":args.stage,"completed":True,"result":result},ensure_ascii=False),flush=True)
