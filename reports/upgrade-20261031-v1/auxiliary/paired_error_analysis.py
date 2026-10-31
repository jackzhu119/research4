"""Fixed-model, all-seed paired error accounting; never changes a prediction."""
from pathlib import Path
import csv,gzip,hashlib,json
BASE=Path(__file__).resolve().parent
OUT=BASE/'tables-and-figures'
with gzip.open(BASE/'scored_outer_predictions.csv.gz','rt',encoding='utf-8',newline='') as h:rows=list(csv.DictReader(h))
metrics=json.loads((BASE/'pooled_metrics.json').read_text());result=[]
for dataset,families in metrics.items():
    for seed in ['42','43','44']:
        baseline={r['record_id']:r for r in rows if (r['dataset_id'],r['seed'],r['family'])==(dataset,seed,'full_svm')}
        for family in families:
            if family=='full_svm':continue
            candidate={r['record_id']:r for r in rows if (r['dataset_id'],r['seed'],r['family'])==(dataset,seed,family)}
            assert baseline.keys()==candidate.keys();corrected=harmed=0
            for rid,b in baseline.items():
                c=candidate[rid];assert b['y_true']==c['y_true'];old=b['y_predicted']==b['y_true'];new=c['y_predicted']==c['y_true'];corrected+=int(not old and new);harmed+=int(old and not new)
            old=metrics[dataset]['full_svm'][seed];new=metrics[dataset][family][seed];assert corrected-harmed==new['correct']-old['correct']
            result.append({'dataset_id':dataset,'seed':seed,'candidate_family':family,'N':len(baseline),'baseline_correct':old['correct'],'candidate_correct':new['correct'],'errors_corrected':corrected,'correct_harmed':harmed,'net_correct':corrected-harmed,'accuracy_delta_pp':100*(new['accuracy']-old['accuracy']),'macro_f1_delta':new['macro_f1']-old['macro_f1'],'balanced_accuracy_delta_pp':100*(new['balanced_accuracy']-old['balanced_accuracy'])})
p=OUT/'paired_error_analysis.csv'
if p.exists():raise FileExistsError(p)
with p.open('w',encoding='utf-8-sig',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(OUT/'paired_error_analysis_validation.json').write_text(json.dumps({'status':'TECHNICAL_CHECK_PASS','all36_paired_net_changes_equal_metric_counts':len(result)==36,'scored_predictions_sha256':sha(BASE/'scored_outer_predictions.csv.gz'),'metrics_sha256':sha(BASE/'pooled_metrics.json'),'analysis_script_sha256':sha(__file__),'table_sha256':sha(p),'model_changes':0,'seed_selection':'all fixed42/43/44','inference':'descriptive paired errors; no significance or generalization success claimed'},indent=2)+'\n')
print(json.dumps([r for r in result if r['seed']=='42' and r['candidate_family']=='mcuve_svm'],indent=2))
