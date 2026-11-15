#!/usr/bin/env python3
"""Verify pinned material LOWO evidence after extending the paper allowlist.

No fitting is performed. The original README/manifest hashes describe the
historical analysis context, while all spectral data and frozen identities
must remain byte-identical in the current checkout.
"""
from pathlib import Path
import csv,hashlib,io,json,subprocess
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score

ROOT=Path(__file__).resolve().parents[1]
SOURCE='627a0975bdc61ef7cd95742265c3b500c879deae'
CONTEXT='ae4e20561f652369926cee25e666ea5cc3e0a49c'
PREFIX='reports/kind_lowo_20261115'
CONTEXT_PATHS={'paper/current/README.md','paper/current/manifest.json'}
def sha(data):return hashlib.sha256(data).hexdigest()
def original(path,commit=SOURCE):return subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{path}'])
def rows(path):return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
def main():
 base=ROOT/PREFIX;receipt=json.loads((base/'verification.json').read_text())
 script='scripts/validate_kind_lowo_20261115.py'
 assert sha((ROOT/script).read_bytes())==sha(original(script))==receipt['script_sha256']
 for name,expected in receipt['input_sha256'].items():
  data=original(name,CONTEXT) if name in CONTEXT_PATHS else (ROOT/name).read_bytes()
  assert sha(data)==expected,name
 names=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',SOURCE,PREFIX],text=True).splitlines()
 for name in names:assert (ROOT/name).read_bytes()==original(name),name
 dataset=json.loads((base/'dataset_audit.json').read_text())
 for book in dataset['workbook_checks']:assert sha((ROOT/book['path']).read_bytes())==book['sha256']
 plans=json.loads((base/'planned_folds.json').read_text());summaries=json.loads((base/'summary.json').read_text())['tasks'];checks={}
 for task,fold_count,record_count in [('binary_kinds',10,864),('four_kinds',18,1672)]:
  summary=summaries[task];pred=rows(base/task/'oof_predictions.csv');splits=json.loads((base/task/'fold_splits.json').read_text())
  assert len(pred)==len({r['record_id'] for r in pred})==record_count
  actual=[{k:f[k] for k in ['fold_id','heldout_workbook','train_record_ids','test_record_ids']} for f in splits]
  assert actual==plans[task] and len(splits)==fold_count
  for fold in splits:
   assert not set(fold['train_record_ids'])&set(fold['test_record_ids'])
   assert not set(fold['train_workbooks'])&set(fold['test_workbooks'])
   assert fold['all_task_kinds_represented_in_training']
  labels=summary['metrics']['class_order'];true=[r['true_kind'] for r in pred];guess=[r['predicted_kind'] for r in pred]
  metric={'accuracy':accuracy_score(true,guess),'macro_f1':f1_score(true,guess,labels=labels,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(true,guess)}
  for name,value in metric.items():assert abs(value-summary['metrics'][name])<1e-14
  assert confusion_matrix(true,guess,labels=labels).tolist()==summary['metrics']['confusion_counts']
  assert sum(int(r['correct']) for r in pred)==summary['metrics']['n_correct']
  checks[task]={'folds':fold_count,'records':record_count,'metrics':metric,'one_oof_prediction_per_record':True}
 current=ROOT/'paper/current';manifest=json.loads((current/'manifest.json').read_text())
 copied=[r for r in manifest['files'] if r['path'].startswith('results/kind_lowo_20261115/')]
 assert len(copied)==22,'LOWO publication entry is incomplete'
 for row in copied:
  assert row['source_commit']==SOURCE and (current/row['path']).read_bytes()==original(row['source_path'])
 result={'status':'PASS','source_commit':SOURCE,'original_context_commit':CONTEXT,'original_context_checked_from_git':sorted(CONTEXT_PATHS),'pinned_result_files':len(names),'current_allowlisted_lowo_files':len(copied),'active_workbooks_unchanged':18,'tasks':checks,'new_model_fits_in_verification':0,'independent_deep_input_and_scaler_audit':'reports/kind_lowo_20261115/independent_audit.json'}
 out=ROOT/'outputs/revision_20261115/kind_lowo_current_context_verification.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
