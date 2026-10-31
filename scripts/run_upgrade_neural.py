"""Prepare, source-select, infer and independently score the new neural pilot."""
import argparse
import json
from pathlib import Path
from herb_spectra.upgrade_neural import prepare, select, evaluate, score


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","selection","evaluation","score"))
    p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--release",type=Path)
    a=p.parse_args()
    if a.stage in ("selection","evaluation") and a.release is None:p.error("Source training and outer inference require separate root releases")
    result={"prepare":lambda:prepare(a.protocol,a.output),"selection":lambda:select(a.protocol,a.output,a.release),"evaluation":lambda:evaluate(a.protocol,a.output,a.release),"score":lambda:score(a.protocol,a.output)}[a.stage]()
    print(json.dumps(result,ensure_ascii=False),flush=True)
