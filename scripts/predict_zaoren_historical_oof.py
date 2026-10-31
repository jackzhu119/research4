"""Read a numbered research workbook and replay only its original held-out model.

This does not fit a deployment model or validate a new cohort. It preserves the
historical SVC vote and strict ID/value binding; new specimens use the separate
deployment prediction tool.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from herb_spectra.deployment_io import load_prediction_input
from predict_paper_c_candidate import DEFAULT_INDEX, predict_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=Path,default=DEFAULT_INDEX)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    index=json.loads(a.index.read_text())
    data=load_prediction_input(a.input,index['frequency_axis'])
    rows=predict_records(index,data['X_alpha'],data['axis'],data['record_ids'],mode='heldout-replay')
    a.output.mkdir(parents=True)
    path=a.output/'predictions.csv'
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    lock={'created_at_utc':datetime.now(timezone.utc).isoformat(),'mode':'historical-heldout-replay',
          'input_sha256':data['source_sha256'],'historical_index_sha256':sha(a.index),
          'prediction_sha256':sha(path),'records':len(rows),'new_model_fits':0,
          'ground_truth_received_by_prediction':False,'independent_confirmation':False,
          'source_seen_by_replayed_fold_model':False,'historically_exposed_development_data':True,
          'script_sha256':sha(__file__),'qc':data['qc']}
    (a.output/'prediction_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'records':len(rows),'mode':lock['mode'],'output':str(a.output),'new_model_fits':0},ensure_ascii=False))

if __name__=='__main__':main()
