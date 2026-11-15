#!/usr/bin/env python3
"""Lossless SVM support-vector row references. No fitting, ranking, or quantization.

encode_job(job_dir, root=None, compact=False, progress_callback=None) -> status dict
load_artifact(job_dir, root=None) -> original array-name -> ndarray
restore_job(job_dir, root=None, output_path=None, progress_callback=None) -> status dict

Default encoding keeps the original. --compact unlinks it only after on-disk
round-trip verification and fresh completed/non-testing/no-active-attempt checks.
The original ZIP-container SHA is provenance; a restored ZIP may have a different
container SHA. Every restored array's dtype, shape, layout and C-order bits match.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,sqlite3,time,zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
import numpy as np

FORMAT_VERSION='svm-support-training-row-reference-v1'
ORIGINAL_NAME='fitted_artifacts.npz'
LOSSLESS_NAME='fitted_artifacts.lossless.npz'
RECIPE_NAME='fitted_artifacts.recipe.json'
LOCK_NAME='.model_artifact_codec.lock'
SUPPORTED_WORKER_SHA256='570d4af4e39083d0e3ec60afae0a21eb137bd2ea272a49168bc346368fd998cd'
_DATA_CACHE={}
class CodecError(RuntimeError):pass

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def _json(path):return json.loads(Path(path).read_text())
def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def _emit(callback,stage,**extra):
 if callback:
  try:callback({'stage':stage,**extra})
  except Exception:pass

def _fsync_dir(directory):
 fd=os.open(str(directory),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def _atomic_json(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 try:
  with tmp.open('w')as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path);_fsync_dir(path.parent)
 finally:
  if tmp.exists():tmp.unlink()
def _atomic_npz(path,arrays):
 path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 try:
  with tmp.open('wb')as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path);_fsync_dir(path.parent)
 finally:
  if tmp.exists():tmp.unlink()
def _arrays(path):
 with np.load(path,allow_pickle=False)as z:
  result={k:z[k].copy(order='K') for k in z.files}
 if any(a.dtype.hasobject for a in result.values()):raise CodecError('object arrays are unsupported')
 return result

def _array_metadata(a):
 a=np.asarray(a)
 return {'shape':list(a.shape),'dtype':a.dtype.str,'fortran_order':bool(a.flags.f_contiguous and not a.flags.c_contiguous),'sha256_c_order_bytes':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
def _manifest(arrays):return {k:_array_metadata(v) for k,v in arrays.items()}
def _verify(arrays,manifest):
 if set(arrays)!=set(manifest):raise CodecError('restored array names differ')
 for name,a in arrays.items():
  if _array_metadata(a)!=manifest[name]:raise CodecError('array bitwise verification failed: '+name)
 return True

def _root_for(job_dir,root):
 job_dir=Path(job_dir).resolve();root=Path(root).resolve() if root else job_dir.parent.parent
 return job_dir,root
@contextmanager
def _job_lock(job_dir):
 with (Path(job_dir)/LOCK_NAME).open('a')as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise CodecError('artifact is locked by another codec operation')
  yield

def _require_completed(job_dir,root):
 if job_dir.parent!=(root/'jobs').resolve():raise CodecError('compaction requires a canonical root/jobs/<id> directory')
 result=_json(job_dir/'result.json')
 if result.get('status')!='complete' or result.get('complete') is not True:raise CodecError('result is not complete')
 if result.get('testing_only') or result.get('job',{}).get('testing'):raise CodecError('testing jobs cannot be compacted')
 db=root/'queue.sqlite3'
 if not db.exists():raise CodecError('completion database is unavailable')
 with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True,timeout=10)as c:
  row=c.execute('SELECT status,job_json FROM jobs WHERE id=?',(job_dir.name,)).fetchone()
  if row is None or row[0] not in ('completed','complete','done'):raise CodecError('database job is not completed')
  job=json.loads(row[1])
  if job.get('testing'):raise CodecError('database marks a testing job')
  active=c.execute("SELECT COUNT(*) FROM attempts WHERE job_id=? AND status NOT IN ('succeeded','complete','completed','done','failed','interrupted','cancelled','abandoned','lost','terminated','orphaned_closed')",(job_dir.name,)).fetchone()[0]
  if active:raise CodecError('job has an active or unrecognized attempt state')
 frozen=_json(job_dir/'job_frozen.json')
 if frozen['job']!=job:raise CodecError('database and frozen job configuration differ')
 if result.get('identity_sha256')!=frozen['identity_sha256']:raise CodecError('result/frozen identity mismatch')
 return result,job,frozen

def _data(root,relative_path,expected_sha):
 path=(root/relative_path).resolve();stat=path.stat();finger=(str(path),stat.st_size,stat.st_mtime_ns,expected_sha)
 if finger in _DATA_CACHE:return _DATA_CACHE[finger]
 if _sha(path)!=expected_sha:raise CodecError('shared input SHA mismatch')
 data=_arrays(path)
 for name in ('record_ids','material_labels','roles','X_alpha','X_n'):
  if name not in data:raise CodecError('missing input '+name)
 if data['X_alpha'].shape[1]!=612 or data['X_n'].shape!=data['X_alpha'].shape:raise CodecError('unsupported paired input shape')
 if len(set(map(str,data['record_ids'])))!=len(data['record_ids']):raise CodecError('duplicate input identifiers')
 _DATA_CACHE.clear();_DATA_CACHE[finger]=data
 return data

def _snv(x):
 return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,ddof=1,keepdims=True),np.finfo(float).eps)
def _pool(alpha,n,pool,preprocessing):
 if pool=='alpha_raw':return (_snv(alpha) if preprocessing=='snv' else alpha),np.arange(612,1224)
 if pool=='alpha_snv':return _snv(alpha),np.arange(612,1224)
 if pool=='joint_both_snv':return np.concatenate([_snv(n),_snv(alpha)],1),np.arange(1224)
 if pool=='joint_nraw_asnv':return np.concatenate([n,_snv(alpha)],1),np.arange(1224)
 if pool=='n_raw':return n,np.arange(612)
 if pool=='n_snv':return _snv(n),np.arange(612)
 if pool=='snv_plus_mean_sd':return np.column_stack([_snv(alpha),alpha.mean(1),alpha.std(1,ddof=1)]),np.arange(614)+2000
 if pool=='mean_sd_only':return np.column_stack([alpha.mean(1),alpha.std(1,ddof=1)]),np.arange(2)+3000
 if pool.startswith('context_'):return np.concatenate([n if pool.endswith('raw_n') else _snv(n),_snv(alpha)],1),np.arange(1224)
 raise CodecError('unsupported frozen pool '+str(pool))
def _seedof(*values):return int(hashlib.sha256('|'.join(map(str,values)).encode()).hexdigest()[:8],16)

def _training_matrix(root,job,selection,arrays,input_info,worker_info):
 """Replay frozen transforms only, using saved means/scales and final indices."""
 if worker_info['sha256']!=SUPPORTED_WORKER_SHA256:raise CodecError('unsupported worker version; original retained')
 if _sha(root/worker_info['relative_path'])!=worker_info['sha256']:raise CodecError('frozen worker dependency changed')
 data=_data(root,input_info['relative_path'],input_info['sha256']);material=job.get('family')=='material' or job.get('task','core300') in ('material','material4','material_binary')
 if material:
  mask=np.isin(data['material_labels'],['酸枣仁','枣仁']) if str(job.get('variant','')).startswith('binary') else np.ones(len(data['record_ids']),bool)
 else:mask=(data['material_labels']=='酸枣仁')&(data['roles']=='train')
 ids=data['record_ids'][mask];alpha=data['X_alpha'][mask];n=data['X_n'][mask]
 if not material and len(ids)!=300:raise CodecError('core population differs from frozen 300')
 locations={str(r):i for i,r in enumerate(ids)};train_ids=list(map(str,arrays['train_record_ids']))
 if len(set(train_ids))!=len(train_ids):raise CodecError('duplicate train identifiers')
 try:fit=np.array([locations[r] for r in train_ids],dtype=int)
 except KeyError as e:raise CodecError('saved train identifier outside population: '+str(e))
 base=selection['base'];chosen=selection['chosen'];x,mapping=_pool(alpha,n,base['pool'],chosen['preprocessing']);rep=base['representation'];k=int(chosen['K'])
 if rep=='uniform':
  selected=np.rint(np.linspace(0,611,k)).astype(int);x=x[:,selected];mapping=mapping[selected]
 elif rep=='random':
  selected=np.sort(np.random.default_rng(_seedof('coverage_random',int(job['seed']),k)).choice(612,k,replace=False));x=x[:,selected];mapping=mapping[selected]
 elif rep=='binmean':
  x=np.column_stack([x[:,ix].mean(1) for ix in np.array_split(np.arange(612),k)]);mapping=np.arange(k)+10000
 elif rep not in ('full','mc'):raise CodecError('unsupported representation '+str(rep))
 mean=arrays['scaler_mean'];scale=arrays['scaler_scale']
 if mean.shape!=(x.shape[1],) or scale.shape!=mean.shape:raise CodecError('saved scaler width mismatch')
 # StandardScaler.transform performs these two in-place floating operations.
 zt=x[fit].copy();zt-=mean;zt/=scale
 final=np.asarray(arrays['selected_global_indices'])
 if rep=='mc':
  lookup={int(g):i for i,g in enumerate(mapping)}
  try:columns=np.array([lookup[int(g)] for g in final],dtype=int)
  except KeyError as e:raise CodecError('final predictor outside frozen pool: '+str(e))
  zt=zt[:,columns]
 else:
  if not np.array_equal(mapping,final):raise CodecError('final mapping differs from frozen transform')
 if not np.isfinite(zt).all():raise CodecError('nonfinite reconstructed training values')
 return zt

def _support_rows(zt,support):
 if support.ndim!=2 or zt.ndim!=2 or support.shape[1]!=zt.shape[1] or support.dtype!=zt.dtype:raise CodecError('support/training shape or dtype mismatch')
 index={}
 for i,row in enumerate(zt):index.setdefault(hashlib.sha256(row.tobytes()).digest(),[]).append(i)
 rows=[];ambiguous=0
 for row in support:
  bits=row.tobytes();candidates=index.get(hashlib.sha256(bits).digest(),[]);exact=[i for i in candidates if zt[i].tobytes()==bits]
  if not exact:raise CodecError('a support vector has no bitwise-equal frozen training row')
  rows.append(exact[0]);ambiguous+=int(len(exact)>1)
 return np.asarray(rows,dtype=np.int32),ambiguous

def _read_encoded(job_dir,root,recipe=None):
 recipe=recipe or _json(job_dir/RECIPE_NAME)
 if recipe.get('format_version')!=FORMAT_VERSION:raise CodecError('unsupported codec format')
 if recipe['job_id']!=job_dir.name:raise CodecError('recipe job identifier mismatch')
 encoded=job_dir/recipe['encoded_container']['name']
 if _sha(encoded)!=recipe['encoded_container']['sha256']:raise CodecError('encoded container SHA mismatch')
 arrays=_arrays(encoded)
 if 'support_training_rows' not in arrays:raise CodecError('support row reference missing')
 rows=arrays.pop('support_training_rows')
 if _array_metadata(rows)!=recipe['support_training_rows_manifest']:raise CodecError('support row index SHA mismatch')
 if set(arrays)!=set(recipe['original_arrays'])-{'support_vectors'}:raise CodecError('stored array names differ from recipe')
 zt=_training_matrix(root,recipe['job'],recipe['selection'],arrays,recipe['input'],recipe['worker'])
 if rows.ndim!=1 or rows.dtype.kind not in ('i','u') or np.any(rows<0) or np.any(rows>=len(zt)):raise CodecError('invalid support training rows')
 restored=zt[rows]
 original=recipe['original_arrays']['support_vectors']
 if original['fortran_order']:restored=np.asfortranarray(restored)
 arrays['support_vectors']=restored
 _verify(arrays,recipe['original_arrays'])
 return arrays

def load_artifact(job_dir,root=None):
 """Load all original arrays, verifying a compact recipe whenever applicable."""
 job_dir,root=_root_for(job_dir,root)
 if (job_dir/ORIGINAL_NAME).exists():
  arrays=_arrays(job_dir/ORIGINAL_NAME)
  if (job_dir/RECIPE_NAME).exists():_verify(arrays,_json(job_dir/RECIPE_NAME)['original_arrays'])
  return arrays
 return _read_encoded(job_dir,root)

def encode_job(job_dir,root=None,compact=False,progress_callback:Callable|None=None):
 """Encode a completed real job. A failure always leaves its original untouched."""
 started=time.process_time();job_dir,root=_root_for(job_dir,root);original=job_dir/ORIGINAL_NAME;encoded=job_dir/LOSSLESS_NAME;recipe_path=job_dir/RECIPE_NAME
 answer={'job_id':job_dir.name,'compact_requested':bool(compact),'raw_bytes':original.stat().st_size if original.exists() else 0,'encoded_bytes':0,'recipe_bytes':0,'saved_bytes':0}
 try:
  with _job_lock(job_dir):
   result,job,frozen=_require_completed(job_dir,root)
   _emit(progress_callback,'validated_completion',job_id=job_dir.name)
   if original.exists() and original.is_symlink():raise CodecError('original artifact must not be a symlink')
   if encoded.exists() and recipe_path.exists():
    recipe=_json(recipe_path);restored=_read_encoded(job_dir,root,recipe);_emit(progress_callback,'verified_existing_encoding',job_id=job_dir.name)
    if original.exists():
     old=_arrays(original);_verify(old,recipe['original_arrays']);original_sha=_sha(original)
    else:old=None;original_sha=None
   else:
    if not original.exists():raise CodecError('original and complete encoding are unavailable')
    original_sha=_sha(original);old=_arrays(original)
    _emit(progress_callback,'loaded_original',job_id=job_dir.name,raw_bytes=original.stat().st_size)
    if 'support_vectors' not in old:return {**answer,'status':'skipped','reason':'no_support_vectors','cpu_seconds':time.process_time()-started}
    if 'support_training_rows' in old:raise CodecError('reserved row-index array name already exists')
    selection=_json(job_dir/'selection.json')
    if selection['identity_sha256']!=frozen['identity_sha256']:raise CodecError('selection/frozen identity mismatch')
    identity={k:frozen[k] for k in ('job','data_sha256','spec_sha256','code_sha256')}
    if _digest(identity)!=frozen['identity_sha256']:raise CodecError('frozen identity checksum differs')
    input_info={'relative_path':'inputs/current1672_paired612.npz','sha256':frozen['data_sha256']}
    worker_info={'relative_path':'study_worker.py','sha256':frozen['code_sha256']}
    zt=_training_matrix(root,job,selection,old,input_info,worker_info)
    _emit(progress_callback,'reconstructed_training_matrix',job_id=job_dir.name)
    rows,ambiguous=_support_rows(zt,old['support_vectors']);stored={k:v for k,v in old.items() if k!='support_vectors'};stored['support_training_rows']=rows
    manifest=_manifest(old);restored={k:v for k,v in stored.items() if k!='support_training_rows'};restored['support_vectors']=zt[rows]
    if manifest['support_vectors']['fortran_order']:restored['support_vectors']=np.asfortranarray(restored['support_vectors'])
    _verify(restored,manifest)
    _emit(progress_callback,'in_memory_bitwise_roundtrip_verified',job_id=job_dir.name)
    _atomic_npz(encoded,stored)
    recipe={'format_version':FORMAT_VERSION,'job_id':job_dir.name,'job_identity_sha256':frozen['identity_sha256'],'job':job,'selection':selection,'input':input_info,'worker':worker_info,'codec':{'name':Path(__file__).name,'sha256':_sha(__file__)},'original_container':{'name':ORIGINAL_NAME,'sha256':original_sha,'bytes':original.stat().st_size},'encoded_container':{'name':LOSSLESS_NAME,'sha256':_sha(encoded),'bytes':encoded.stat().st_size},'original_arrays':manifest,'support_training_rows_manifest':_array_metadata(rows),'support_row_matching':{'method':'SHA256 plus exact C-order row-byte comparison; first bitwise-equal training row','ambiguous_equal_row_count':ambiguous,'interpretation':'References reproduce vector values and order exactly; duplicate equal training rows do not establish unique physical support identity.'},'restore_steps':['Verify encoded container, shared paired input and frozen worker SHA256.','Load unchanged arrays and support_training_rows.','Build the original analytical population and row order from fixed input, job task and saved train_record_ids.','Replay frozen per-spectrum/coverage transforms; subtract saved scaler_mean and divide by saved scaler_scale; select saved final global coordinates.','Take training-matrix rows in saved support-vector order; preserve support dtype/layout.','Verify every original array shape, dtype, layout and C-order-byte SHA256; optionally write a full NPZ.'],'verification':{'in_memory_all_arrays_bitwise':True,'on_disk_all_arrays_bitwise':False},'container_note':'A restored NPZ ZIP container may have a different SHA256 because packaging/compression metadata differ. Original container SHA256 is provenance only. All data arrays are restored bitwise; no quantization, model fitting or ranking is performed.'}
    # Persist the recipe after the encoded file is fsynced; then verify real disk bytes.
    _atomic_json(recipe_path,recipe);restored=_read_encoded(job_dir,root,recipe);_verify(restored,manifest)
    recipe['verification']['on_disk_all_arrays_bitwise']=True;_atomic_json(recipe_path,recipe)
    _emit(progress_callback,'on_disk_all_arrays_verified',job_id=job_dir.name)
   answer.update(raw_bytes=original.stat().st_size if original.exists() else recipe['original_container']['bytes'],encoded_bytes=encoded.stat().st_size,recipe_bytes=recipe_path.stat().st_size)
   answer['equivalent_storage_reduction_bytes']=answer['raw_bytes']-answer['encoded_bytes']-answer['recipe_bytes']
   if compact and original.exists() and answer['equivalent_storage_reduction_bytes']<=0:return {**answer,'status':'skipped','reason':'no_net_storage_saving','original_preserved':True,'all_arrays_bitwise_verified':True,'cpu_seconds':time.process_time()-started}
   if compact and original.exists():
    _require_completed(job_dir,root)
    if _sha(original)!=original_sha:raise CodecError('original changed during encoding')
    _read_encoded(job_dir,root,recipe) # Last verification immediately before irreversible file unlink.
    original.unlink();_fsync_dir(job_dir);answer['status']='compacted'
   elif not original.exists():answer['status']='already_compact'
   else:answer['status']='encoded'
   answer['saved_bytes']=answer['equivalent_storage_reduction_bytes'] if answer['status']=='compacted' else 0
   answer['all_arrays_bitwise_verified']=True;answer['original_preserved']=original.exists();answer['cpu_seconds']=time.process_time()-started
   _emit(progress_callback,'complete',**answer)
   return answer
 except Exception as e:
  return {**answer,'status':'error','reason':type(e).__name__+': '+str(e),'original_preserved':original.exists(),'cpu_seconds':time.process_time()-started}

def restore_job(job_dir,root=None,output_path=None,progress_callback:Callable|None=None):
 """Write a complete NPZ from verified arrays; compact evidence is retained."""
 started=time.process_time();job_dir,root=_root_for(job_dir,root);output=Path(output_path).resolve() if output_path else job_dir/ORIGINAL_NAME
 try:
  with _job_lock(job_dir):
   recipe=_json(job_dir/RECIPE_NAME);arrays=_read_encoded(job_dir,root,recipe);_emit(progress_callback,'decoded_and_verified',job_id=job_dir.name)
   if output.exists():
    _verify(_arrays(output),recipe['original_arrays']);status='already_restored'
   else:
    output.parent.mkdir(parents=True,exist_ok=True);_atomic_npz(output,arrays);_verify(_arrays(output),recipe['original_arrays']);status='restored'
   return {'status':status,'job_id':job_dir.name,'output':str(output),'all_arrays_bitwise_verified':True,'restored_container_sha256':_sha(output),'original_container_sha256_provenance':recipe['original_container']['sha256'],'container_sha_equality_required':False,'cpu_seconds':time.process_time()-started}
 except Exception as e:return {'status':'error','job_id':job_dir.name,'reason':type(e).__name__+': '+str(e),'cpu_seconds':time.process_time()-started}

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 e=sub.add_parser('encode');e.add_argument('job_dir');e.add_argument('--root');e.add_argument('--compact',action='store_true')
 r=sub.add_parser('restore');r.add_argument('job_dir');r.add_argument('--root');r.add_argument('--output')
 v=sub.add_parser('verify');v.add_argument('job_dir');v.add_argument('--root')
 a=p.parse_args()
 if a.command=='encode':result=encode_job(a.job_dir,a.root,a.compact)
 elif a.command=='restore':result=restore_job(a.job_dir,a.root,a.output)
 else:
  try:
   jd,rr=_root_for(a.job_dir,a.root);arrays=_read_encoded(jd,rr) if (jd/RECIPE_NAME).exists() else load_artifact(jd,rr);result={'status':'verified','arrays':{k:_array_metadata(v) for k,v in arrays.items()}}
  except Exception as e:result={'status':'error','reason':type(e).__name__+': '+str(e)}
 print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return int(result['status']=='error')
if __name__=='__main__':raise SystemExit(main())
