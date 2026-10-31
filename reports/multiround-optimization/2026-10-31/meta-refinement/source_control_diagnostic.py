"""Read-only source-control diagnostic; no method selection or fitting.

Reports already predeclared supervised continuation pairs alongside FO-MAML.
The m000 control was saved but absent from the target-adaptation summary.
"""
from copy import deepcopy
from datetime import datetime,timezone
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from herb_spectra.neural_base_search import load_locked_contexts,model_scores
from herb_spectra.paper_fewshot_classification import state_fingerprint
from herb_spectra.calibration_exploration import encoded

ROOT=Path('/workspace/research4');BASE=ROOT/'outputs/multiround-20261031/meta-refinement'
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p,k=7):
 cm=np.bincount(y*k+p,minlength=k*k).reshape(k,k);d=cm.diagonal();s=cm.sum(1);t=cm.sum(0)
 return {'accuracy':float(d.sum()/s.sum()),'balanced_accuracy':float(np.mean(d/s)),
         'macro_f1':float(np.mean(np.divide(2*d,s+t,out=np.zeros(k,dtype=float),where=(s+t)>0))),
         'confusion_matrix':cm.tolist(),'correct':int(d.sum()),'total':int(s.sum()),'class_recall':(d/s).tolist()}
contexts,truth=load_locked_contexts(ROOT/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz',
 ROOT/'data/processed/zaoren_threeway_20261031_v1/split_plan.json',BASE.parent/'neural-base',scope='final')
allrows=[];metricrows=[];disagreements=[];artifact_hashes=[]
for context in contexts:
 classes=np.asarray(context['classes']);query=context['prediction'];seed=context['seed']
 assert 'y' not in query
 modelset={'base_b006':context['model']}
 for meta in ('m000','m002'):
  directory=BASE/'final'/f'seed_{seed}'/meta;metadata=json.loads((directory/'training.json').read_text())
  for strategy in ('fomaml','supervised'):
   path=directory/(strategy+'.pt');assert sha(path)==metadata['artifacts'][strategy]['sha256']
   model=deepcopy(context['model']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
   assert state_fingerprint(model)==metadata['artifacts'][strategy]['fingerprint']
   modelset[meta+'_'+strategy]=model
   artifact_hashes.append({'seed':seed,'meta_id':meta,'strategy':strategy,'path':str(path.relative_to(ROOT)),
    'sha256':sha(path),'fingerprint':state_fingerprint(model)})
 scores={name:model_scores(model,query['X']) for name,model in modelset.items()}
 predictions={name:values.argmax(1) for name,values in scores.items()}
 # Only scoring uses the separate truth dictionary, after all predictions exist.
 y=encoded([truth[str(i)] for i in query['record_ids']],classes)
 for name,p in predictions.items():
  metricrows.append({'seed':seed,'source_model':name,**metrics(y,p)})
  for identifier,label,pred,logits in zip(query['record_ids'],y,p,scores[name]):
   allrows.append({'seed':seed,'source_model':name,'record_id':str(identifier),'y_true':str(classes[label]),
    'y_predicted':str(classes[pred]),'is_correct':int(label==pred),**{f'logit_{c}':float(v) for c,v in zip(classes,logits)}})
 names=list(predictions)
 for i,first in enumerate(names):
  for second in names[i+1:]:
   a,b=predictions[first],predictions[second]
   disagreements.append({'seed':seed,'first':first,'second':second,'different_predictions':int(np.sum(a!=b)),
    'second_corrected_first':int(np.sum((a!=y)&(b==y))),'second_new_errors':int(np.sum((a==y)&(b!=y))),
    'max_absolute_logit_difference':float(np.max(np.abs(scores[first]-scores[second])))})
 for item in artifact_hashes:
  if item['seed']==seed:assert sha(ROOT/item['path'])==item['sha256']
 assert state_fingerprint(context['model'])==context['fingerprint']
means=[]
for name in [r['source_model'] for r in metricrows[:5]]:
 cases=[r for r in metricrows if r['source_model']==name]
 means.append({'source_model':name,'seeds':[r['seed'] for r in cases],
  **{key+'_mean':float(np.mean([r[key] for r in cases])) for key in ('accuracy','balanced_accuracy','macro_f1')}})
with (BASE/'source_control_diagnostic.csv').open('w',encoding='utf-8-sig',newline='') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(allrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(allrows)
report={'status':'pass','created_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only diagnostic of predeclared source training controls; no new method selection or fitting',
 'new_fits':0,'query_specimens':171,'source_training_samples':513,'prediction_rows':len(allrows),
 'source_model_metrics':metricrows,'source_model_means':means,'pairwise_disagreement':disagreements,
 'artifacts':artifact_hashes,'csv_sha256':sha(BASE/'source_control_diagnostic.csv'),'script_sha256':sha(Path(__file__)),
 'model_states_unchanged':True,'no_support_labels_used':True,'all_methods_predeclared_before_final_scoring':True,
 'limits':['Same171 historical query specimens, one source batch per group.','Three seeds are not independent batches.',
 'Exact equal accuracy does not imply equal model weights or scores.','Source learning gains are not target-support calibration gains.']}
(BASE/'source_control_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'means':means,'rows':len(allrows),'new_fits':0},ensure_ascii=False))
