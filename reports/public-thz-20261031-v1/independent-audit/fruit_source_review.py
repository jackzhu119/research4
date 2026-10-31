"""Independent read-only, zero-fit preflight for local-only fruit regression."""
import csv,datetime,hashlib,io,json,zipfile
import openpyxl
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold
from audit_common import Audit,ROOT,OUT,read,sha,utc,write_new

B=ROOT/'outputs/public-thz-fruit-regression-20261031-v1'
D=ROOT/'data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit'
R=ROOT/'reports/public-thz-20261031-v1/fruit-regression-preflight'
TASKS={'MA_Brix':('MA','y_Brix'),'PM_Brix':('PM','y_Brix'),'PR_Brix':('PR','y_Brix'),'MA_DM':('MA','y_DM')}

def main():
 a=Audit();v=read(R/'validation.json');m=read(B/'run_manifest.json');p=read(B/'protocol.json');prep=read(B/'preparation.json')
 a.check(sha(B/'run_manifest.json')==v['local_run_manifest_sha256']=='cbaad570b88803e9618c94c80720c0f0e5338e0568454992e64f6e1ff8573a11','frozen run manifest')
 for k in ['protocol','preparation']:a.check(sha(B/(k+'.json'))==m[k+'_sha256']==v[k+'_sha256'],'frozen '+k)
 for f,h in v['source_sha256'].items():a.check(sha(ROOT/f)==h==m['source_sha256'][f],'frozen source '+f)
 a.check(sha(ROOT/'tests/test_public_fruit_regression.py')==v['test_source_sha256']=='8b7d9c6bc8cfaafed8ddb304e031dc0d6e2af25b961c00799ffab513d2124dd8','frozen actual tests')
 log=(R/'unit_tests_with_outer_truth_guard.log').read_text();a.check('Ran 15 tests' in log and log.rstrip().endswith('OK'),'actual fifteen synthetic fixtures pass')
 archive=D/'author_repository.zip';input=D/'prepared/fruit_fixed_Tag_time_index_v1.npz'
 a.check(sha(archive)=='902cca56a744187dd3eb3bbed41aa1ce28eb5e75e5b003946d29007a7fed603f','original downloaded archive')
 a.check(sha(input)=='b3c5d8b6272038377cfdfe272a6d084895b9817325fdd9ec34bab6c9db1ec27b'==prep['input_sha256'],'prepared raw-waveform input')
 with np.load(input,allow_pickle=False) as z:data={k:z[k] for k in z.files}
 labels={}
 with zipfile.ZipFile(archive) as z:
  workbook=next(n for n in z.namelist() if n.endswith('.xlsx'));book=openpyxl.load_workbook(io.BytesIO(z.read(workbook)),read_only=True,data_only=True)
  for sheet,kind,names in [('Mango DM, Brix','MA',['DM','Brix']),('Pear Brix','PR',['Brix']),('Persimmon Brix','PM',['Brix'])]:
   rows=list(book[sheet].values);header=rows[0];a.equal(header[1:],[f'{kind}{i}' for i in range(1,11)],'source workbook fruit headers')
   for row in rows[1:]:
    timestamp=datetime.datetime.strptime(row[0]+'_2023','%b%d_%I%p_%Y').isoformat()
    for fruit,val in zip(header[1:],row[1:],strict=True):
     if val is not None:labels[(fruit,timestamp)]=dict(zip(names,[float(t.strip()) for t in val.split(',')] if isinstance(val,str) else [float(val)],strict=True))
  for i,member in enumerate(data['source_member']):
   values=[v for k,v in loadmat(io.BytesIO(z.read(str(member)))).items() if not k.startswith('__')];a.check(len(values)==1,'one MAT waveform')
   a.equal(values[0].ravel(),data['X'][i],'verbatim MAT waveform')
   fruit,timestamp=str(data['groups'][i]),str(data['timestamp'][i]);truth=labels[(fruit,timestamp)]
   a.equal(data['y_Brix'][i],truth['Brix'],'source workbook Brix binding')
   if fruit.startswith('MA'):a.equal(data['y_DM'][i],truth['DM'],'source workbook DM binding')
   else:a.check(np.isnan(data['y_DM'][i]),'unavailable DM remains NaN')
   parts=str(member).split('/');a.check(parts[4]==data['fruit_type'][i] and parts[5]=='Tag' and str(parts[-1]).split('_')[0]==fruit,'source member to fruit and condition identity')
 blocks=np.column_stack([data['X'][:,ix].mean(1) for ix in np.array_split(np.arange(7000),128)])
 a.check(p['source_estimator_fit_budget']==4*5*3*3*42==7560,'fixed source fit budget')
 a.check(len(p['R1'])==34 and len(p['R2_template'])==8 and p['MC_PLScall_budget_with_transform_cache']==10800,'34+8 grid and continuous-target MC PLS budget')
 a.equal(p['seeds'],[42,43,44],'all seeds')
 a.equal(p['outer_roles'],['dummy_mean','PLSR','full_SVR','MLP','MC_SVR','source_policy'],'six roles')
 a.check(p['fresh_confirmation'] is False and p['medicinal_THZ_TDS_transfer_claim'] is False and p['cross_time_or_new_batch_claim'] is False and 'LOCAL_ONLY' in p['redistribution'],'scope and license limitations')
 a.check(prep['license_verified'] is False and prep['same_four_dates_overlap_train_query'] is True,'no licence or cross-time claims invented')
 expected=[];contexts=prep['contexts'];a.check(len(contexts)==80,'twenty outer plus sixty inner')
 for task,(kind,target) in TASKS.items():
  population=np.flatnonzero(data['fruit_type']==kind);groups=data['groups'][population]
  a.check(len(population)==40 and len(set(groups))==10,'forty observations ten fruits per task')
  for outer,(tr,te) in enumerate(GroupKFold(5).split(population,groups=groups)):
   source,query=population[tr],population[te];expected.append((task,outer,None,source,query,target))
   for inner,(st,sv) in enumerate(GroupKFold(3).split(source,groups=data['groups'][source])):expected.append((task,outer,inner,source[st],source[sv],target))
 for c,(task,outer,inner,s,q,target) in zip(contexts,expected,strict=True):
  a.check((c['task'],c['outer'],c['inner'])==(task,outer,inner),'fixed source/query context')
  a.equal(c['source_ids'],data['record_id'][s],'source ordered IDs');a.equal(c['query_ids'],data['record_id'][q],'query ordered IDs')
  a.check(not set(data['groups'][s])&set(data['groups'][q]),'persistent fruits cannot cross roles')
  for ix in [s,q]:
   for fruit in set(data['groups'][ix]):a.check(int(np.sum(data['groups'][ix]==fruit))==4,'all four dates follow fruit')
  for role,ix in [('source',s),('query',q)]:
   f=B/c[role+'_pack'];a.check(sha(f)==c[role+'_pack_sha256'],'frozen pack bytes')
   with np.load(f,allow_pickle=False) as z:
    a.check(set(z.files)=={'record_ids','X','feature_index'}|({'source_targets'} if role=='source' else set()),'strict metadata exclusion')
    a.equal(z['record_ids'],data['record_id'][ix],'actual pack IDs');a.equal(z['X'],blocks[ix],'fixed index-block means');a.equal(z['feature_index'],np.arange(128),'index units only')
    if role=='source':a.equal(z['source_targets'],data[target][ix],'actual measured source targets')
  stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=['record_id','target'],lineterminator='\n');w.writeheader();w.writerows({'record_id':str(r),'target':float(y)} for r,y in zip(data['record_id'][q],data[target][q],strict=True))
  a.check(hashlib.sha256(stream.getvalue().encode('utf-8-sig')).hexdigest()==c['truth_sha256']==sha(B/c['truth']),'truth serialization bytes checked without semantic truth CSV parsing')
 a.check(not (B/'fit_scope_events.jsonl').exists() and not (B/'selection_lock.json').exists(),'zero public fruit fits at review')
 result={'completed_at_utc':utc(),'status':'FRUIT_SOURCE_REVIEW_PASS_FOR_LOCAL_ONLY_EXPLORATORY_ENGINEERING_REGRESSION','checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'new_audit_fits':0,'new_fruit_fits_at_review':0,'run_manifest_sha256':sha(B/'run_manifest.json'),'source_sha256':m['source_sha256'],'source_estimator_fit_budget':7560,'internal_regression_MC_PLS_budget':10800,'expected_outer_prediction_rows':2880,'source_review':['All120 raw MAT waveforms independently reconstructed from archive and targets bound to actual workbook cells','All80 prepared contexts/packs/metadata exclusions/group partitions checked','Each persistent fruit and its four dates stay in one role; each task has ten fruit IDs not forty independent fruits','Source-only target/input scaling, grouped pooled OOF source selection, source cache and all-truth-before-score source inspected','Actual15 synthetic tests logged; no research fits executed by auditor'], 'limitations':['Repository has no verified numeric-data redistribution license: raw, derived arrays, models, traces and per-record predictions remain ignored/local only','Author files do not establish physical time calibration, frequency axis or target assay units; model input is128 engineering index blocks','Four dates overlap source and query fruit groups, so this is not future-date or new-batch generalization','This exploratory fruit reflection regression is separate from medicinal THz-TDS classification and does not validate acid-jujube transfer','MC subsamples dependent observations inside source fruits; its rows cannot be counted as independent fruits','Review certifies only this frozen preparation/source; scoring requires independent source-selection review and a new root outer gate'],'script_sha256':sha(__file__)}
 write_new(OUT/'fruit_source_review.json',result);print(json.dumps({k:result[k] for k in ['status','checks','numeric_values','maximum_difference','new_audit_fits']}))

if __name__=='__main__':main()
