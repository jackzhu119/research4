"""Describe saved training-source feature masks; never create a new input mask."""
import argparse,csv,hashlib,json
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
 out=r/'feature-stability';out.mkdir(exist_ok=False)
 lock=json.loads((r/'selection_lock.json').read_text());prep=json.loads((r/'preparation.json').read_text());axis=np.array(prep['axis'])
 records=json.loads((r/'selection_records.json').read_text());chosen=lock['selected']['id'];masks=[];rows=[];sources={}
 for item in records:
  if item['candidate_id']!=chosen:continue
  f=Path(item['model']['path']).with_suffix('.json');assert digest(f)==item['model']['metadata_sha256'];d=json.loads(f.read_text());sources[str(f)]=digest(f)
  masks.append({'scope':'development_fold','fold':item['fold'],'seed':item['seed'],'indices':set(d['selected_indices'])})
 for seed in [42,43,44]:
  f=r/'deployment'/f'model_seed{seed}.json';d=json.loads(f.read_text());sources[str(f)]=digest(f)
  masks.append({'scope':'full_development855','fold':None,'seed':seed,'indices':set(d['selected_indices'])})
 assert len(masks)==18 and all(len(m['indices'])==123 for m in masks)
 for m in masks:
  for i in sorted(m['indices']):rows.append({k:v for k,v in m.items() if k!='indices'}|{'feature_index':i,'frequency_axis_value':float(axis[i])})
 with (out/'selected_features.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 comparisons=[]
 for a,b in combinations(masks,2):
  if a['scope']!=b['scope']:continue
  kind='same_source_different_seed' if a['scope']=='full_development855' or a['fold']==b['fold'] else 'overlapping_development_training_folds'
  comparisons.append({'scope':a['scope'],'comparison':kind,'fold_a':a['fold'],'seed_a':a['seed'],'fold_b':b['fold'],'seed_b':b['seed'],'intersection':len(a['indices']&b['indices']),'union':len(a['indices']|b['indices']),'jaccard':len(a['indices']&b['indices'])/len(a['indices']|b['indices'])})
 with (out/'jaccard_pairs.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparisons[0]),lineterminator='\n');w.writeheader();w.writerows(comparisons)
 full=[m for m in masks if m['scope']=='full_development855'];fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
 for i,m in enumerate(full):axes[0].vlines(axis[sorted(m['indices'])],i+.1,i+.8,label=f"seed{m['seed']}")
 axes[0].set_yticks([.45,1.45,2.45],['seed42','seed43','seed44']);axes[0].set_title('Full-development masks: same 855 source specimens, three MC seeds')
 count=np.zeros(len(axis),int)
 for m in masks:
  if m['scope']=='development_fold':count[list(m['indices'])]+=1
 axes[1].plot(axis,count);axes[1].set_ylim(0,15.5);axes[1].set_ylabel('Selected source models / 15');axes[1].set_xlabel('Stored frequency-axis value (unit requires instrument documentation)')
 axes[1].set_title('Five overlapping development-training folds x three seeds; descriptive only')
 for ax in axes:ax.set_xlim(axis[0],axis[-1]);ax.grid(alpha=.2)
 fig.savefig(out/'feature_stability.png',dpi=180);fig.savefig(out/'feature_stability.pdf');plt.close(fig)
 summary={}
 for scope,kind in sorted({(x['scope'],x['comparison']) for x in comparisons}):
  v=[x['jaccard'] for x in comparisons if (x['scope'],x['comparison'])==(scope,kind)]
  summary[scope+'/'+kind]={'pairs':len(v),'mean_jaccard':float(np.mean(v)),'minimum':min(v),'maximum':max(v)}
 result={'selected_deployment_candidate':chosen,'models':18,'source_masks_only':True,'selected_features_per_model':123,'feature_rows':len(rows),'jaccard':summary,'input_metadata_sha256':sources,'new_model_fits':0,'new_mask_constructed_for_prediction':False,'stability_is_new_batch_validation':False,'script_sha256':digest(__file__)}
 (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (out/'README.md').write_text('# 冻结模型的源内选点一致性\n\n只读取已锁配置的15个开发源模型和3个全855模型，不重新训练、不取频点并集供预测。表中的Jaccard描述同源不同MC种子和重叠开发训练折的选点差异；这些不是独立批次、化学峰归属或稳定筛选提准证据。所有模型各保留123点。\n')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
