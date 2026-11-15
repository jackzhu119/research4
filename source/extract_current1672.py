"""Extract current registry-addressed measured alpha/n exports, without training.
Only attachment registry and frozen split-manifest entries determine inputs/labels.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,warnings
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string
warnings.filterwarnings('ignore',message='Workbook contains no default style')
B=Path(__file__).resolve().parent
SRC=Path('/workspace/applied_spectroscopy_revision/source/research4_Applied_Spectroscopy')
REGPATH=SRC/'evidence/active_workbooks.json';METAPATH=SRC/'evidence/original_snapshots/data/research/species_revision_20261108/split_manifest.csv'
reg=json.loads(REGPATH.read_text());rest=json.loads((B/'workbook_restoration_manifest.json').read_text());meta=pd.read_csv(METAPATH,keep_default_na=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert rest['verified']==rest['required']==18
assert len(meta)==meta.record_id.nunique()==meta.physical_sample_id.nunique()==1672
restored={x['origin_group_id']:x for x in rest['records']}
with np.load(SRC/'evidence/acid_train_300_spectra.npz',allow_pickle=False)as core:
 core_ids=core['record_ids'].copy();core_axis=core['axis'].copy();core_alpha=core['X_alpha'].copy();core_n=core['X_n'].copy()

def read_one(w):
 gid=w['origin_group_id'];entry=restored[gid];path=Path(entry['target_path']);assert sha(path)==w['sha256']
 mm=meta.loc[meta.origin_group_id.eq(gid)].copy();assert len(mm)==w['record_count'];assert mm.source_workbook.eq(w['path']).all();assert mm.species.eq(w['species']).all()
 indices=[column_index_from_string(c)-1 for c in mm.excel_column]
 wb=load_workbook(path,read_only=True,data_only=True)
 blocks={};axes={};headers={}
 for sheetname,key in [('吸收系数','X_alpha'),('折射率','X_n')]:
  assert sheetname in wb.sheetnames,(gid,sheetname)
  ws=wb[sheetname];ws.reset_dimensions();it=ws.iter_rows(values_only=True);header=next(it,())
  assert len(header)>max(indices),(gid,sheetname,'missing sample header')
  headers[key]=[str(header[i])for i in indices]
  rows=[];axis=[]
  for row in it:
   if not row or row[0] is None:continue
   freq=float(row[0])
   if freq<.1:continue
   if freq>1.5:break
   assert len(row)>max(indices),(gid,sheetname,'row missing sample columns')
   axis.append(freq);rows.append([float(row[i])if row[i]is not None else np.nan for i in indices])
  arr=np.asarray(rows,float).T;ax=np.asarray(axis,float)
  assert arr.shape==(len(mm),612),(gid,sheetname,arr.shape)
  assert np.isfinite(arr).all(),(gid,sheetname,'nonfinite values')
  assert np.max(np.abs(ax-core_axis))<=1e-12,(gid,sheetname,'axis differs')
  blocks[key]=arr;axes[key]=ax
 wb.close()
 assert np.array_equal(axes['X_alpha'],axes['X_n']),(gid,'alpha/n axis mismatch')
 for j,rid in enumerate(mm.record_id):
  if gid.startswith('ZY'):assert headers['X_alpha'][j]==rid and headers['X_n'][j]==rid,(gid,rid,'numbered header mismatch')
 return gid,mm.record_id.tolist(),blocks,{'group_id':gid,'path':str(path),'sha256':w['sha256'],'records':len(mm),'alpha_shape':list(blocks['X_alpha'].shape),'n_shape':list(blocks['X_n'].shape),'core_axis_max_abs_difference':float(max(np.max(np.abs(ax-core_axis))for ax in axes.values())),'numbered_ID_headers_checked':gid.startswith('ZY'),'metadata_excel_columns_preserved':True}

alpha=np.empty((1672,612));refr=np.empty((1672,612));pos={x:i for i,x in enumerate(meta.record_id)};receipts=[]
with ThreadPoolExecutor(max_workers=4)as pool:
 for fut in as_completed([pool.submit(read_one,w)for w in reg['workbooks']]):
  gid,ids,blocks,receipt=fut.result();ii=[pos[x]for x in ids];alpha[ii]=blocks['X_alpha'];refr[ii]=blocks['X_n'];receipts.append(receipt);print('EXTRACTED',gid,len(ids),flush=True)
ci=[pos[x]for x in core_ids];diff_a=float(np.max(np.abs(alpha[ci]-core_alpha)));diff_n=float(np.max(np.abs(refr[ci]-core_n)))
assert diff_a<=1e-12 and diff_n<=1e-12
sha_bygroup={w['origin_group_id']:w['sha256']for w in reg['workbooks']}
npz=B/'inputs/current1672_paired612.npz';np.savez_compressed(npz,record_ids=meta.record_id.to_numpy(dtype=str),group_ids=meta.origin_group_id.to_numpy(dtype=str),physical_sample_ids=meta.physical_sample_id.to_numpy(dtype=str),material_labels=meta.species.to_numpy(dtype=str),roles=meta.role.to_numpy(dtype=str),X_alpha=alpha,X_n=refr,axis=core_axis,workbook_sha=np.asarray([sha_bygroup[g]for g in meta.origin_group_id],dtype=str))
meta.to_csv(B/'inputs/current1672_frozen_split_manifest.csv',index=False)
with np.load(npz,allow_pickle=False)as check:
 assert np.array_equal(check['X_alpha'],alpha)and np.array_equal(check['X_n'],refr)
report={'status':'verified','records':1672,'shape_each_modality':[1672,612],'finite_alpha':bool(np.isfinite(alpha).all()),'finite_n':bool(np.isfinite(refr).all()),'roles':{str(k):int(v)for k,v in meta.role.value_counts().items()},'materials':{str(k):int(v)for k,v in meta.species.value_counts().items()},'unique_record_ids':int(meta.record_id.nunique()),'unique_physical_ids':int(meta.physical_sample_id.nunique()),'axis':{'length':612,'min':float(core_axis.min()),'max':float(core_axis.max()),'unit':'nominal THz as exported; physical confirmation still required'},'core300_comparison':{'records':300,'max_abs_alpha_difference':diff_a,'max_abs_n_difference':diff_n,'tolerance':1e-12,'passed':True},'NPZ_path':str(npz),'NPZ_sha256':sha(npz),'NPZ_keys':['record_ids','group_ids','physical_sample_ids','material_labels','roles','X_alpha','X_n','axis','workbook_sha'],'registry_path':str(REGPATH),'registry_sha256':sha(REGPATH),'metadata_path':str(METAPATH),'metadata_sha256':sha(METAPATH),'frozen_metadata_copy_sha256':sha(B/'inputs/current1672_frozen_split_manifest.csv'),'workbooks':sorted(receipts,key=lambda r:r['group_id']),'no_training':True,'no_interpolation_smoothing_or_imputation':True,'reference_columns_excluded':True,'raw_and_repository_untouched':True}
(B/'current1672_extraction_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('NPZ COMPLETE',npz,'core300 diff',diff_a,diff_n,flush=True)
