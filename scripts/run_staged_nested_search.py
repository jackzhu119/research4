"""Run source-only staged nested selection or globally released outer evaluation."""
import argparse
import json
from pathlib import Path
from herb_spectra.staged_nested_search import run_search


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("input","protocol","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--stage",choices=("selection","evaluation"),default="selection")
    parser.add_argument("--release",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    value=run_search(args.input,args.protocol,args.output,stage=args.stage,resume=args.resume,release_path=args.release)
    print(json.dumps({"status":"completed","stage":args.stage,"result":value},ensure_ascii=False),flush=True)
