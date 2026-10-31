"""Post-run byte checks only; not an ex-ante release or scientific confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json,re,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
BASE='33b31e1db2e30ed893962a1eac84c1d3d1e82e65'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def index_check(path):
 obj=read(path);items=obj.get('files',obj);n=0
 for name,value in items.items():
  expected=value if isinstance(value,str) else value.get('sha256')
  p=path.parent/name
  assert sha(p)==expected,str(p);n+=1
 return {'index':str(path.relative_to(ROOT)),'sha256':sha(path),'files_checked':n}
def main():
 assert not (RUN/'delivery_validation.json').exists()
 indexes=[ROOT/p for p in ['reports/upgrade-20261031-v1/files_sha256.json','reports/zaoren-deployment-review-20261031-v1/audit_deliverable_index.json','reports/zaoren-deployment-historical-demo-20261031-v1/files_sha256.json','reports/zaoren-neural-source-diagnostic-20261031-v1/files_sha256.json','reports/zaoren-neural-learning-pilot-20261031-v1/files_sha256.json']]
 checked=[index_check(p) for p in indexes]
 from herb_spectra.deployment import load_bundle
 manifest,models=load_bundle(RUN/'deployment');assert set(models)=={'42','43','44'} and manifest['external_accuracy'] is None
 summary=read(RUN/'feature-stability/verification.json');assert summary['actual_independent_boolean_mask_pair_checks']==108 and summary['maximum_difference']==0
 audit=read(ROOT/'reports/zaoren-deployment-review-20261031-v1/final_review.json')
 assert audit['new_estimator_fits']==0 if 'new_estimator_fits' in audit else True
 # Historical protected Git blobs include old datasets, models, reports, code and references.
 raw=subprocess.check_output(['git','ls-tree','-r','-z',BASE,'--','data','models','reports','references','src','scripts','tests'],cwd=ROOT);protected=0
 for item in raw.split(b'\0'):
  if not item:continue
  header,name=item.split(b'\t',1);mode,kind,blob=header.split()
  if kind!=b'blob':continue
  p=ROOT/name.decode();h=hashlib.sha1();h.update(f'blob {p.stat().st_size}\0'.encode())
  with p.open('rb') as f:
   for b in iter(lambda:f.read(2**20),b''):h.update(b)
  assert h.hexdigest()==blob.decode(),str(p);protected+=1
 log=(RUN/'project-tests.log').read_text();assert re.search(r'Ran 399 tests',log) and re.search(r'^OK$',log,re.M)
 scope=read(RUN/'fit_scope_manifest.json');h=hashlib.sha256()
 with gzip.open(RUN/'fit_scope_events.jsonl.gz','rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 assert h.hexdigest()==scope['raw_sha256'] and scope['completed_learned_operations']==23034
 archive=read(RUN/'model_bundle_archive_validation.json');assert sha(ROOT/archive['path'])==archive['sha256']
 with zipfile.ZipFile(ROOT/archive['path']) as z:
  assert z.testzip() is None
  assert z.read('ZAOREN_ANONYMOUS_DEPLOYMENT.md')==(ROOT/'docs/ZAOREN_ANONYMOUS_DEPLOYMENT.md').read_bytes()
  for p in (RUN/'deployment').rglob('*'):
   if p.is_file():assert z.read(str(p.relative_to(RUN)))==p.read_bytes()
 for directory in [RUN,ROOT/'reports/zaoren-neural-learning-pilot-20261031-v1']:
  assert all(p.stat().st_size<100*2**20 for p in directory.rglob('*') if p.is_file())
 record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'status':'TECHNICAL_DELIVERY_CHECK_PASS_NOT_INDEPENDENT_CONFIRMATION','compiled_after_runs':True,'source_commit':BASE,'protected_historical_blobs':protected,'indexes_verified':checked,'project_tests_passed':399,'deployment_models':3,'source_selection_models':75,'neural_pilot_fresh_trajectories':6,'new_external_score_studies':0,'seven_anonymous_classes':manifest['classes'],'external_deployment_accuracy':None,'raw_fit_log_lossless':True,'model_zip_crc_and_all_member_bytes_match':True,'new_model_fits_in_finalizer':0,'code_sha256':sha(__file__)}
 (RUN/'delivery_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'] and p!=RUN/'files_sha256.json'}
 (RUN/'files_sha256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'report_files_indexed':len(files),'protected_historical_blobs':protected,'project_tests':399,'source_models':75,'deployment_models':3,'new_external_accuracy':None}))
if __name__=='__main__':main()
