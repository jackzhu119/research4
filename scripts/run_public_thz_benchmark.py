"""Run the public-THz grouped-development benchmark in separate guarded stages."""
import argparse,json
from pathlib import Path
from herb_spectra.public_thz_benchmark import prepare,selection,evaluation,score
p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','selection','evaluation','score']);p.add_argument('--output',type=Path,required=True);p.add_argument('--release',type=Path)
a=p.parse_args()
if a.stage in ['selection','evaluation'] and not a.release:p.error('A root stage release JSON is required')
result={'prepare':lambda:prepare(a.output),'selection':lambda:selection(a.output,a.release),'evaluation':lambda:evaluation(a.output,a.release),'score':lambda:score(a.output)}[a.stage]()
print(json.dumps(result,ensure_ascii=False),flush=True)
