"""Run source-only R4 inner selection, then released outer evaluation."""
import argparse
import json
from pathlib import Path

from herb_spectra.staged_nested_combinations import run_search


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input","protocol","core-protocol","core-run","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--stage",choices=("selection","evaluation"),default="selection")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--release",type=Path)
    a=p.parse_args()
    result=run_search(a.input,a.protocol,a.core_protocol,a.core_run,a.output,stage=a.stage,resume=a.resume,release_path=a.release)
    print(json.dumps({"stage":a.stage,"result":result},ensure_ascii=False),flush=True)
