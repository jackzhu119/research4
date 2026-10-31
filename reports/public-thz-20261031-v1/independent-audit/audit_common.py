"""Independent no-fit evidence checks for the public THz study."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
RAW=ROOT/'data/external/ginseng_hvbp5mdtxx_v2/original'
DATA=ROOT/'data/external/public_thz_20261031_v1/ginseng'
CLASSES=('0','1')

class Audit:
 def __init__(self):self.checks=0;self.numeric_values=0;self.maximum_difference=0.
 def check(self,value,why):
  self.checks+=1
  if not value:raise AssertionError(why)
 def equal(self,a,b,why):
  aa,bb=np.asarray(a),np.asarray(b);self.check(aa.shape==bb.shape,why+' shape')
  if aa.dtype.kind in 'biufc' and bb.dtype.kind in 'biufc':
   self.numeric_values+=aa.size;difference=float(np.max(np.abs(aa-bb))) if aa.size else 0.;self.maximum_difference=max(self.maximum_difference,difference)
   self.check(np.array_equal(aa,bb),why+' exact values')
  else:self.check(np.array_equal(aa,bb),why+' exact metadata')
 def close(self,a,b,why,atol=1e-12,rtol=1e-12):
  aa,bb=np.asarray(a),np.asarray(b);self.check(aa.shape==bb.shape,why+' shape');self.numeric_values+=aa.size
  difference=float(np.max(np.abs(aa-bb))) if aa.size else 0.;self.maximum_difference=max(self.maximum_difference,difference)
  self.check(np.allclose(aa,bb,atol=atol,rtol=rtol),why+' within predeclared tolerance')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,obj):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()
def csvrows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify_dataset(audit):
 historical=ROOT/'data/external/ginseng_hvbp5mdtxx_v2/annotated/inspection.json'
 audit.check(sha(historical)=='cfea731b774a69888f8380951881448cb312564e928698485e1b935632ec8a62','pinned historical inspection bytes')
 raw_hashes={r['file']:r['sha256'] for r in read(historical)['member_manifest']}
 audit.check(len(raw_hashes)==18,'exact eighteen original CSV members')
 xalpha=[];xn=[];labels=[];axis=None
 for group in range(1,7):
  for prefix,target in [('a',xalpha),('n',xn)]:
   name=f'data_{prefix}{group}.csv';p=RAW/name;audit.check(sha(p)==raw_hashes[name],name+' original SHA')
   with p.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.reader(f));audit.check(len(rows)==66,name+' header and65rows')
    audit.check([v.strip() for v in rows[0]]==['Frequency (THz)','','','','','','',''],name+' explicit raw header')
    arr=np.asarray([[float(s) for s in row] for row in rows[1:]],dtype=float)
   audit.check(arr.shape==(65,8) and np.isfinite(arr).all(),name+' finite table shape')
   if axis is None:axis=arr[:,0].copy()
   audit.equal(arr[:,0],axis,name+' raw paired frequency axis');target.extend(arr[:,1:].T)
  name=f'label_{group}.csv';audit.check(sha(RAW/name)==raw_hashes[name],name+' original SHA')
  with (RAW/name).open(encoding='utf-8-sig',newline='') as f:
   values=[float(s) for row in csv.reader(f) for s in row if s!='']
  audit.check(len(values)==7 and set(values)<={0.,1.},name+' seven binary labels')
  labels.extend(str(int(s)) for s in values)
 data={'X_alpha':np.asarray(xalpha),'X_n':np.asarray(xn),'axis':axis,'y':np.array(labels),
       'record_ids':np.array([f'G{g}_S{s:02}' for g in range(1,7) for s in range(1,8)]),
       'export_groups':np.repeat([f'G{g}' for g in range(1,7)],7)}
 audit.check(sha(RAW/'Dataset.rar')=='3b1bbcff42c0825361c372cab329abee2c21c29d3b4b30ac7c1366865e7c3fcc','pinned user-uploaded RAR')
 manifest=read(DATA/'manifest.json')
 audit.check(manifest['scientific_status']=='EXPLORATORY_OBSERVATION_LEVEL_UNKNOWN_UPSTREAM','unknown upstream scope retained')
 for k in ['verified_independent_plants','verified_independent_batches']:audit.check(manifest[k] is None,k+' not invented')
 audit.check(manifest['records']==42 and manifest['new_independent_samples_added']==0,'forty-two observation records; no new independent samples')
 audit.check(manifest['class_counts']=={'0':22,'1':20},'source label counts')
 audit.check(manifest['export_group_is_biological_batch'] is False,'export group not batch')
 audit.check(manifest['geographic_origins_available'] is False,'no origin labels invented')
 audit.check(manifest['modalities_expand_record_count'] is False,'two representations do not double sample count')
 audit.check(sha(DATA/'observations.npz')==manifest['prepared_arrays_sha256'],'prepared bytes')
 with np.load(DATA/'observations.npz',allow_pickle=False) as z:
  audit.check(set(z.files)==set(data),'strict six prepared fields')
  for k,v in data.items():audit.equal(z[k],v,'independent raw reconstruction '+k)
 lineage=csvrows(DATA/'observation_lineage.csv');audit.check(len(lineage)==84,'record x modality lineage rows')
 lookup={rid:i for i,rid in enumerate(data['record_ids'])}
 for row in lineage:
  i=lookup[row['record_id']];group=i//7+1;prefix='a' if row['modality']=='alpha' else 'n'
  audit.check(row['modality'] in ['alpha','n'],'known modality')
  audit.check(all(row[k]=='' for k in ['physical_sample_id','parent_sample_id','plant_id','preparation_id','batch_id','measurement_id']),'unknown ancestor fields not invented')
  audit.check(row['label']==data['y'][i] and row['export_group']==data['export_groups'][i],'lineage label/file group binding')
  audit.check(int(row['source_column_1based'])==i%7+2 and int(row['label_item_1based'])==i%7+1,'source column and label item binding')
  audit.check(row['source_hash']==raw_hashes[f'data_{prefix}{group}.csv'],'lineage source SHA')
  audit.check(row['physical_quantity_unit']==('cm^-1' if prefix=='a' else 'dimensionless'),'native units retained')
 audit.check((data['X_alpha']<0).sum()==141,'negative absorption values preserved')
 for name in ['X_alpha','X_n']:
  for i in range(42):
   for j in range(i+1,42):audit.check(not np.array_equal(data[name][i],data[name][j]),name+' no exact pair')
 return data,manifest

def metrics(y,pred,classes=CLASSES):
 y=np.asarray(y);pred=np.asarray(pred);lookup={s:i for i,s in enumerate(classes)};cm=np.zeros((len(classes),len(classes)),int)
 for t,p in zip(y,pred,strict=True):cm[lookup[str(t)],lookup[str(p)]]+=1
 tp=cm.diagonal().astype(float);support=cm.sum(1);pred_support=cm.sum(0)
 precision=np.divide(tp,pred_support,out=np.zeros_like(tp),where=pred_support!=0)
 recall=np.divide(tp,support,out=np.zeros_like(tp),where=support!=0)
 f1=np.divide(2*tp,support+pred_support,out=np.zeros_like(tp),where=(support+pred_support)!=0)
 return {'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall[support>0].mean()),'confusion_matrix':cm.tolist(),
         'per_class':{s:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,s in enumerate(classes)}}

if __name__=='__main__':
 a=Audit();d,m=verify_dataset(a)
 path=OUT/'independent_dataset_review.json'
 write_new(path,{'completed_at_utc':utc(),'status':'TECHNICAL_RAW_VALUE_BINDING_PASS_EXPLORATORY_UNKNOWN_UPSTREAM','checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'observations':42,'verified_independent_plants':None,'verified_independent_batches':None,'negative_alpha_cells':141,'new_audit_fits':0,'raw_data_source':'existing uploaded source archive; not a fresh remote numeric download','script_sha256':sha(__file__),'prepared_manifest_sha256':sha(DATA/'manifest.json')})
 print(json.dumps({'checks':a.checks,'numeric_values':a.numeric_values,'maximum_difference':a.maximum_difference,'new_audit_fits':0}))
