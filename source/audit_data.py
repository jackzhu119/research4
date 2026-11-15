"""Independent bounded integrity audit. No model fit; no N x N distance matrix.
Exact duplicate hashes cover all1672 rows; similarities use fixed sampled pairs.
"""
from pathlib import Path
from collections import defaultdict
import json,hashlib,time
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parent;clock=time.perf_counter();cpu=time.process_time()
P=B/'inputs/current1672_paired612.npz';V=json.loads((B/'current1672_extraction_verification.json').read_text())
sha=hashlib.sha256(P.read_bytes()).hexdigest();assert sha==V['NPZ_sha256']
with np.load(P,allow_pickle=False)as a:data={k:a[k].copy()for k in a.files}
meta=pd.read_csv(B/'inputs/current1672_frozen_split_manifest.csv',keep_default_na=False)
N=1672;ids=data['record_ids'];groups=data['group_ids'];physical=data['physical_sample_ids'];materials=data['material_labels'];roles=data['roles']
assert len(ids)==len(set(ids))==len(set(physical))==N
for key,col in [('record_ids','record_id'),('group_ids','origin_group_id'),('physical_sample_ids','physical_sample_id'),('material_labels','species'),('roles','role')]:assert np.array_equal(data[key],meta[col].to_numpy(dtype=str)),key
assert set(materials)=={'天麻','枣仁','酸枣仁','黄芪'}and set(roles)=={'train','calibration','prediction'}
assert data['X_alpha'].shape==data['X_n'].shape==(N,612)and np.isfinite(data['X_alpha']).all()and np.isfinite(data['X_n']).all()
assert np.all(np.diff(data['axis'])>0)and len(data['axis'])==612
assert all(x==x.strip()for x in materials)
# Hash little-endian binary values with signed zeros normalized; preserve every other number.
def hashes(X):
 out=[]
 for row in X:
  x=np.asarray(row,dtype='<f8').copy();x[x==0]=0.;out.append(hashlib.sha256(x.tobytes()).hexdigest())
 return out
ha=hashes(data['X_alpha']);hn=hashes(data['X_n']);hj=[hashlib.sha256((x+y).encode()).hexdigest()for x,y in zip(ha,hn)]
exact=[];exact_report={}
for modality,hs in [('alpha',ha),('n',hn),('paired_alpha_n',hj)]:
 g=defaultdict(list)
 for i,h in enumerate(hs):g[h].append(i)
 dup=[ii for ii in g.values()if len(ii)>1];pairs=sum(len(ii)*(len(ii)-1)//2 for ii in dup)
 exact_report[modality]={'distinct_row_hashes':len(g),'duplicate_hash_groups':len(dup),'duplicate_unique_specimens':sum(len(ii)for ii in dup),'duplicate_unordered_pairs':pairs}
 for ii in dup:
  for k,i in enumerate(ii):
   for j in ii[k+1:]:exact.append({'modality':modality,'record_id_1':ids[i],'record_id_2':ids[j],'group_1':groups[i],'group_2':groups[j],'material_1':materials[i],'material_2':materials[j],'role_1':roles[i],'role_2':roles[j],'same_physical_ID':physical[i]==physical[j],'cross_role':roles[i]!=roles[j]})
pd.DataFrame(exact,columns=['modality','record_id_1','record_id_2','group_1','group_2','material_1','material_2','role_1','role_2','same_physical_ID','cross_role']).to_csv(B/'exact_duplicate_rowhash_pairs.csv',index=False)
role_overlaps=[]
rr=['train','calibration','prediction']
for x,r1 in enumerate(rr):
 for r2 in rr[x+1:]:
  i=np.where(roles==r1)[0];j=np.where(roles==r2)[0]
  role_overlaps.append({'role1':r1,'role2':r2,'shared_record_IDs':len(set(ids[i])&set(ids[j])),'shared_physical_IDs':len(set(physical[i])&set(physical[j])),'shared_source_groups':len(set(groups[i])&set(groups[j]))})
# Fixed bounded samples:20k uniform unordered pairs plus1000 within-material TRAIN/PRED pairs.
rng=np.random.default_rng(20261110);strata=defaultdict(set)
while len(strata['global_uniform_pairs'])<20000:
 ar=rng.integers(0,N,size=(3000,2))
 for i,j in ar:
  if i==j:continue
  strata['global_uniform_pairs'].add(tuple(sorted((int(i),int(j)))))
  if len(strata['global_uniform_pairs'])>=20000:break
for material in sorted(set(materials)):
 it=np.flatnonzero((materials==material)&(roles=='train'));ip=np.flatnonzero((materials==material)&(roles=='prediction'));name=f'{material}_TRAIN_vs_PRED'
 while len(strata[name])<min(1000,len(it)*len(ip)):
  for i,j in zip(rng.choice(it,1200,replace=True),rng.choice(ip,1200,replace=True)):
   strata[name].add(tuple(sorted((int(i),int(j)))))
   if len(strata[name])>=min(1000,len(it)*len(ip)):break
pairs=sorted(set.union(*strata.values()));ix=np.array([i for i,j in pairs]);jx=np.array([j for i,j in pairs]);loc={p:i for i,p in enumerate(pairs)}
df=pd.DataFrame({'record_id_1':ids[ix],'record_id_2':ids[jx],'material_1':materials[ix],'material_2':materials[jx],'group_1':groups[ix],'group_2':groups[jx],'role_1':roles[ix],'role_2':roles[jx],'same_material':materials[ix]==materials[jx],'same_group':groups[ix]==groups[jx]})
for modality,key in [('alpha','X_alpha'),('n','X_n')]:
 X=data[key];rel=np.empty(len(pairs));corr=np.empty(len(pairs))
 for start in range(0,len(pairs),512):
  stop=min(start+512,len(pairs));A=X[ix[start:stop]];C=X[jx[start:stop]]
  rel[start:stop]=np.linalg.norm(A-C,axis=1)/((np.linalg.norm(A,axis=1)+np.linalg.norm(C,axis=1))/2)
  aa=A-A.mean(axis=1,keepdims=True);cc=C-C.mean(axis=1,keepdims=True);den=np.linalg.norm(aa,axis=1)*np.linalg.norm(cc,axis=1)
  corr[start:stop]=np.divide(np.sum(aa*cc,axis=1),den,out=np.full(stop-start,np.nan),where=den>0)
 df[f'{modality}_symmetric_relative_RMS']=rel;df[f'{modality}_Pearson_r']=corr
 assert np.isfinite(rel).all()and np.isfinite(corr).all()
summary=[]
for name,ss in {'all_unique_sampled_pairs':set(pairs),**strata}.items():
 zz=df.iloc[[loc[p]for p in ss]]
 for modality in ['alpha','n']:
  r=zz[f'{modality}_Pearson_r'];distance=zz[f'{modality}_symmetric_relative_RMS'];n=len(zz)
  summary.append({'sample_scope':name,'modality':modality,'sampled_unique_unordered_pairs':n,'Pearson_r_ge_0.99_n':int((r>=.99).sum()),'Pearson_r_ge_0.99_fraction':float((r>=.99).mean()),'Pearson_r_ge_0.999_n':int((r>=.999).sum()),'Pearson_r_ge_0.999_fraction':float((r>=.999).mean()),'Pearson_r_ge_0.9999_n':int((r>=.9999).sum()),'relative_RMS_le_0.001_n':int((distance<=.001).sum()),'relative_RMS_le_0.01_n':int((distance<=.01).sum()),'both_relative_RMS_le_0.01_and_Pearson_r_ge_0.999_n':int(((distance<=.01)&(r>=.999)).sum()),'minimum_sampled_relative_RMS':float(distance.min()),'median_sampled_Pearson_r':float(r.median()),'maximum_sampled_Pearson_r':float(r.max())})
pd.DataFrame(summary).to_csv(B/'sampled_pair_similarity_summary.csv',index=False)
df.to_csv(B/'bounded_sampled_pair_similarities.csv.gz',index=False,compression='gzip')
flag=df.loc[((df.alpha_symmetric_relative_RMS<=.01)&(df.alpha_Pearson_r>=.999))|((df.n_symmetric_relative_RMS<=.01)&(df.n_Pearson_r>=.999))]
flag.to_csv(B/'sampled_near_highcorr_flagged_pairs.csv',index=False)
counts=[]
for material,mm in meta.groupby('species'):
 counts.append({'material':material,'N':len(mm),'source_groups':mm.origin_group_id.nunique(),'TRAIN':int(mm.role.eq('train').sum()),'CAL':int(mm.role.eq('calibration').sum()),'PRED':int(mm.role.eq('prediction').sum())})
npz_core=Path('/workspace/applied_spectroscopy_revision/source/research4_Applied_Spectroscopy/evidence/acid_train_300_spectra.npz')
with np.load(npz_core,allow_pickle=False)as core:
 position={rid:i for i,rid in enumerate(ids)};ci=[position[x]for x in core['record_ids']];da=float(np.max(np.abs(data['X_alpha'][ci]-core['X_alpha'])));dn=float(np.max(np.abs(data['X_n'][ci]-core['X_n'])));assert da<=1e-12 and dn<=1e-12
out={'schema':'independent_data_audit_v1','NPZ_sha256':sha,'NPZ_keys':list(data),'metadata_and_material_strings_match':True,'exact_material_strings':sorted(set(materials)),'material_role_counts':counts,'all_values_finite':True,'unique_record_IDs':N,'unique_physical_IDs':N,'no_full_pairwise_matrix_constructed':True,'complete_exact_hash_check':exact_report,'cross_role_ID_intersections':role_overlaps,'core300_recheck':{'max_abs_alpha':da,'max_abs_n':dn,'tolerance':1e-12},'bounded_similarity_check':{'seed':20261110,'RNG':'numpy.default_rng','unique_pairs_evaluated':len(pairs),'all_possible_unordered_pairs':N*(N-1)//2,'pair_fraction_evaluated':len(pairs)/(N*(N-1)//2),'design':'20,000 uniform unordered pairs plus1,000 TRAIN/PRED pairs sampled within each of4 materials; de-duplicated union.','chunk_rows':512,'summary':summary,'near_highcorr_flagged_pair_n':len(flag),'limitation':'This sample cannot rule out all near duplicates in1,396,956 possible pairs. High Pearson similarity of smooth spectral shapes is not evidence of identical physical specimens or an acquisition artifact. Raw magnitude distance is retained alongside shape correlation.'},'material_task_scope':{'binary_N':864,'binary_roles':{'TRAIN':516,'CAL':174,'PRED':174},'four_material_N':1672,'four_material_roles':{'TRAIN':999,'CAL':334,'PRED':339},'all18_source_groups_in_all3_roles':True,'batch_ID_nonempty_n':int(meta.batch_id.ne('').sum()),'interpretation':'The archived100% binary and97.64% four-material task scores concern specimen splits among represented source groups. No ID overlap or exact spectra duplicates does not remove source-batch/preparation/instrument confounding; no batch identifiers or independent botanical/chemical verification are supplied.'},'model_fits':0,'quantum_calculations':0,'audit_wall_seconds':time.perf_counter()-clock,'audit_CPU_seconds':time.process_time()-cpu}
(B/'data_integrity_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
# Append the independent audit receipt to the existing eligibility, without changing its input selection.
el=B/'data_eligibility.json';e=json.loads(el.read_text());e['independent_data_integrity_audit']={'file':str(B/'data_integrity_audit.json'),'script':str(B/'audit_data.py'),'status':'passed_structural_and_exact_hash_checks; bounded_similarity_descriptive','near_similarity_summary':str(B/'sampled_pair_similarity_summary.csv'),'exact_duplicate_pairs':exact_report,'sampled_near_highcorr_flagged_pair_n':len(flag)};el.write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n')
print('AUDIT COMPLETE',json.dumps({'rows':N,'exact_duplicate_report':exact_report,'unique_sampled_pairs':len(pairs),'near_highcorr_flagged_pairs':len(flag),'CPU_seconds':out['audit_CPU_seconds']},ensure_ascii=False),flush=True)
