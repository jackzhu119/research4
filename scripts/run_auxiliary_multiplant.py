"""Separate preparation, source selection, inference and scoring for two plants."""
import argparse
import json
from pathlib import Path
from herb_spectra.auxiliary_multiplant import default_protocol,prepare,selection,evaluation,score


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['init-protocol','prepare','selection','evaluation','score'])
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--release',type=Path)
    a=p.parse_args()
    if a.stage=='init-protocol':
        if a.protocol.exists():raise FileExistsError('Protocol already exists')
        a.protocol.write_text(json.dumps(default_protocol(),ensure_ascii=False,indent=2)+'\n');print(a.protocol)
    else:
        if a.output is None:p.error('--output required')
        if a.stage in ['selection','evaluation'] and a.release is None:p.error('--release required')
        result={'prepare':lambda:prepare(a.protocol,a.output),'selection':lambda:selection(a.protocol,a.output,a.release),
                'evaluation':lambda:evaluation(a.protocol,a.output,a.release),'score':lambda:score(a.protocol,a.output)}[a.stage]()
        print(json.dumps(result,ensure_ascii=False),flush=True)
