"""Compact exact nested-search evidence; omit only reconstructible MC tensors.

Run after global outer evaluation is complete. No fitting is performed.
"""
from pathlib import Path
import gzip,hashlib,json,shutil,tarfile
import numpy as np
import joblib

ROOT=Path.cwd();RUN=ROOT/'outputs/staged-nested-20261031/core'
DEST=RUN/'compact_archive';MODELS=RUN/'deliverable_models'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def copy(p,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)
def build():
 status=json.loads((RUN/'status.json').read_text())
 if status['status']!='completed':raise ValueError('Complete released outer evaluation before final archiving')
 DEST.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
 original={};omitted=[];metadata_refs={}
 with np.load(ROOT/'outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz',allow_pickle=False) as source:
  source_labels={str(i):str(y) for i,y in zip(source['groups'],source['y'])}
 for p in sorted((RUN/'selection').rglob('*')):
  if not p.is_file():continue
  rel=str(p.relative_to(RUN));original[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
  if p.suffix=='.json' and p.parent.name.startswith('inner_'):
   metadata=json.loads(p.read_text())
   if 'coefficient_file' in metadata:
    metadata_refs.setdefault(str(Path(metadata['coefficient_file']).resolve()),[]).append({'metadata_path':rel,
     'metadata_sha256':sha(p),'source_ids':metadata['source_ids'],
     'source_labels':[source_labels[i] for i in metadata['source_ids']],
     'mc':metadata['mc'],'preprocessing':metadata['preprocessing'],
     'source_value_fingerprint':metadata['source_value_fingerprint'],
     'source_label_fingerprint':metadata['source_label_fingerprint']})
 for p in sorted((RUN/'selection').glob('**/coefficients_*.npz')):
  with np.load(p,allow_pickle=False) as x:coef=x['coefficients']
  mean=coef.mean(0);sd=coef.std(0,ddof=1);floor=max(1.,float(sd.max()))*1e-12
  numeric_sha=hashlib.sha256(np.ascontiguousarray(coef).tobytes()).hexdigest()
  relative=Path('coefficient_summaries')/(numeric_sha+'.npz');target=DEST/relative;target.parent.mkdir(exist_ok=True)
  if not target.exists():np.savez_compressed(target,mean=mean,standard_deviation=sd,floor=np.asarray(floor),n_runs=np.asarray(len(coef)))
  with np.load(target,allow_pickle=False) as z:
   assert np.array_equal(z['mean'],mean) and np.array_equal(z['standard_deviation'],sd) and float(z['floor'])==floor
  omitted.append({'original_path':str(p.relative_to(RUN)),'original_sha256':sha(p),'original_bytes':p.stat().st_size,
   'shape':list(coef.shape),'dtype':str(coef.dtype),'coefficient_numeric_sha256':numeric_sha,
   'archived_summary':str(relative),'summary_sha256':sha(target),'model_seed':42,
   'protocol_sha256':sha(RUN/'protocol_snapshot.json'),
   'source_records':metadata_refs.get(str(p.resolve()),[]),
   'omission_reason':'Per-MC coefficient matrix is a rebuildable local cache; actual mean/std/floor and all signed-stability/importance/ranking/statistics are archived.'})
 members=0
 for outer in sorted((RUN/'selection').glob('outer_*')):
  target=DEST/(outer.name+'_selection_evidence.tar.gz')
  with target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
   with tarfile.open(fileobj=zipped,mode='w') as tar:
    for p in sorted(outer.rglob('*')):
     if p.is_file() and not (p.name.startswith('coefficients_') and p.suffix=='.npz'):
      tar.add(p,arcname=str(p.relative_to(RUN)),recursive=False);members+=1
  with tarfile.open(target,'r:gz') as tar:
   for m in tar.getmembers():
    stream=tar.extractfile(m);assert stream is not None
    assert hashlib.sha256(stream.read()).hexdigest()==original[m.name]['sha256']
 for p in sorted(RUN.iterdir()):
  if p.is_file() and p.suffix in ('.json','.jsonl','.gz','.log','.csv'):
   copy(p,DEST/p.name)
 for p in [ROOT/'outputs/staged-nested-20261031/core_arithmetic_audit.py',ROOT/'outputs/staged-nested-20261031/core_outer_arithmetic_audit.py',Path(__file__).resolve()]:
  if p.exists():copy(p,DEST/p.name)
 rows=json.loads((RUN/'outer_metrics.json').read_text());splits=json.loads((RUN/'splits.json').read_text());artifacts={}
 for row in rows:
  path=Path(row['model_path']);key=(row['outer_fold'],path.name)
  if key not in artifacts:
   target=MODELS/f"outer_{row['outer_fold']:02d}"/path.name;copy(path,target)
   model=joblib.load(target)
   metadata={'source_ids':splits[row['outer_fold']]['train_sample_ids'],
    'outer_query_ids':splits[row['outer_fold']]['test_sample_ids'],'outer_fold':row['outer_fold'],
    'original_path':str(path),'model_sha256':sha(path),'run_manifest_sha256':sha(RUN/'run_manifest.json'),
    'source_code_sha256':json.loads((RUN/'run_manifest.json').read_text())['source_sha256'],
    'model_metadata':model.metadata,'roles':[]}
   artifacts[key]={'target':target,'metadata':metadata}
  artifacts[key]['metadata']['roles'].append({'method':row['method'],'recipe':row['config']})
 model_manifest=[]
 for item in artifacts.values():
  path=item['target'];meta=item['metadata'];dump(path.with_suffix('.json'),meta)
  model_manifest.append({'path':str(path.relative_to(MODELS)),'sha256':sha(path),'bytes':path.stat().st_size,
   'source_samples':len(meta['source_ids']),'method_roles':[r['method'] for r in meta['roles']]})
 dump(DEST/'original_selection_files.json',original);dump(DEST/'omitted_mc_coefficient_matrices.json',omitted)
 summary_count=len(list((DEST/'coefficient_summaries').glob('*.npz')))
 with (DEST/'coefficient_summaries.tar.gz').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as tar:
   for p in sorted((DEST/'coefficient_summaries').glob('*.npz')):tar.add(p,arcname=str(p.relative_to(DEST)),recursive=False)
 with tarfile.open(DEST/'coefficient_summaries.tar.gz','r:gz') as tar:
  for member in tar.getmembers():
   stream=tar.extractfile(member);assert stream is not None
   assert hashlib.sha256(stream.read()).hexdigest()==sha(DEST/member.name)
 shutil.rmtree(DEST/'coefficient_summaries')
 compressed=[]
 for p in sorted(DEST.glob('*.json')):
  if p.stat().st_size>4*1024*1024:
   original_bytes=p.read_bytes();target=p.with_suffix(p.suffix+'.gz')
   with target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:zipped.write(original_bytes)
   with gzip.open(target,'rb') as f:assert f.read()==original_bytes
   compressed.append({'original_name':p.name,'original_sha256':sha(p),'original_bytes':len(original_bytes),'compressed_name':target.name,'compressed_sha256':sha(target)})
   p.unlink()
 dump(DEST/'compressed_json_files.json',compressed)
 dump(MODELS/'manifest.json',{'models':model_manifest,'unique_model_count':len(model_manifest),
  'all_outer_method_roles':len(rows),'role_aliases_not_duplicate_models':True})
 dump(DEST/'archive_validation.json',{'status':'pass','exact_selection_members_archived':members,
  'original_coefficient_matrices_local_only':len(omitted),'coefficient_summary_files':summary_count,
  'mean_std_floor_summaries_numeric_exact':True,'exact_tar_member_hashes_verified':True,
  'outer_models':len(model_manifest),'new_fits_in_archiving':0})
 (DEST/'README.md').write_text('''# Staged nested-search evidence\n\nAll inner metrics and raw predictions are preserved byte-for-byte; large JSON files are gzip-compressed with original SHA/size listed in `compressed_json_files.json`. Five outer-fold tar archives retain original training statistics, frequency indices, signed stability, importance and rankings.\n\nThe original per-MC coefficient matrices remain in the cloud cache and are intentionally omitted from GitHub. `omitted_mc_coefficient_matrices.json.gz` records every original SHA/shape/dtype/seed and ordered source identity; `coefficient_summaries.tar.gz` extracts to `coefficient_summaries/` and stores exact mean, sample standard deviation and prefix-specific floor. Their original numerical arrays were independently audited before archiving. Rebuild full matrices using the frozen input, ordered source IDs and protocol; do not claim this archive contains the raw matrices.\n\nOuter model roles may alias one underlying fitted pipeline. `deliverable_models/manifest.json` records every unique source model and each role mapping. No outer test label was used for selection, and no fitting occurred during archiving.\n''',encoding='utf-8')
 for base in (DEST,MODELS):
  index={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p!=base/'files_sha256.json'}
  dump(base/'files_sha256.json',index)
  for name,h in index.items():assert sha(base/name)==h
 print(json.dumps({'archive_files':len(list(DEST.rglob('*'))),'unique_models':len(model_manifest),'omitted_matrices':len(omitted)}))
if __name__=='__main__':build()
