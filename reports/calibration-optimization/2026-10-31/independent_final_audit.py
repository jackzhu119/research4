"""Independent, read-only score and lineage audit; never fits or selects a model."""
import csv,gzip,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np

repo=Path(__file__).resolve().parents[2]
root=repo/'outputs/zaoren-calibration-optimization-20261031/run'
output=root.parent/'independent_final_audit.json'
checks=0; max_difference=0.; failures=[]
def ck(name, condition):
 global checks
 checks+=1
 if not condition:
  failures.append(name)
  raise AssertionError(name)
def close(name,left,right):
 global max_difference
 difference=abs(float(left)-float(right));max_difference=max(max_difference,difference)
 ck(name,difference<=1e-12)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def key(row):return(row['scope'],row['selector'],int(row['shots_per_class']),int(row['repeat']),row['method'])
def summary_key(row):return(row['scope'],row['selector'],int(row['shots_per_class']),row['method'])
def evaluate(pred,classes):
 cm=[[0 for _ in classes] for _ in classes];vocab={c:i for i,c in enumerate(classes)}
 for identifier,label in pred.items():cm[vocab[truth[identifier]]][vocab[label]]+=1
 true_counts=[sum(row) for row in cm]; predicted_counts=[sum(cm[i][j] for i in range(len(classes))) for j in range(len(classes))]
 diagonal=[cm[i][i] for i in range(len(classes))];n=sum(true_counts)
 recall=[diagonal[i]/true_counts[i] if true_counts[i] else 0. for i in range(len(classes))]
 f1=[2*diagonal[i]/(true_counts[i]+predicted_counts[i]) if true_counts[i]+predicted_counts[i] else 0. for i in range(len(classes))]
 return{'accuracy':sum(diagonal)/n,'balanced_accuracy':statistics.mean(r for r,c in zip(recall,true_counts) if c),'macro_f1':statistics.mean(f1),'confusion_matrix':cm,'class_recall':recall,'class_counts':true_counts,'correct':sum(diagonal),'total':n}

status=readj(root/'status.json');ck('run_completed',status['status']=='completed')
protocol_path=repo/'data/research/calibration_optimization_protocol_20261031.json';protocol=readj(protocol_path)
plan=readj(repo/protocol['fixed_split']);assignment={r['record_id']:r for r in plan['records']}
with np.load(repo/protocol['source_input_npz'],allow_pickle=False) as data:
 truth={str(i):str(y) for i,y in zip(data['record_ids'],data['y'])}
ck('canonical_input_IDs_855',len(truth)==855 and set(truth)==set(assignment))
ck('original_labels_equal_anonymous_split_groups',all(truth[i]==r['origin_group_id'] for i,r in assignment.items()))
manifest=readj(root/'run_manifest.json')
ck('protocol_sha_matches',sha(protocol_path)==manifest['protocol_sha256'])
ck('plan_sha_matches',sha(repo/protocol['fixed_split'])==manifest['plan_sha256'])
ck('input_sha_matches',sha(repo/protocol['source_input_npz'])==manifest['input_sha256'])
for filename,digest in manifest['source_code_sha256'].items():ck('source_code_hash_'+filename,sha(repo/filename)==digest)
lock=readj(root/'selection_lock.json');lock_sha=readj(root/'selection_lock_sha256.json')
ck('selection_lock_sha',sha(root/'selection_lock.json')==lock_sha['sha256']==status['selection_lock_sha256'])
ck('lock_marked_before_final',lock_sha['created_before_final_scoring'] is True)
random_ids=readj(root/'random_query_ids.json')
grid={c['id']:c for c in readj(root/'candidate_grid.json')}
ck('candidate_grid_object_sha',hashlib.sha256(json.dumps(list(grid.values()),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==manifest['grid_sha256'])
all_rows=readj(root/'all_final_metrics.json'); recorded={key(r):r for r in all_rows}
ck('unique_recorded_cases',len(recorded)==len(all_rows)==status['final_evaluations'])
recomputed={}; random_raw=[]; total_raw_rows=0; file_reports=[]; model_reports=[]
scopes={scope['id']:scope for scope in protocol['scopes']}
for scope_id,scope in scopes.items():
 excluded=set(scope.get('exclude_groups',[]))
 expected_query={i for i,r in assignment.items() if r['role']=='prediction' and truth[i] not in excluded}
 expected_source={i for i,r in assignment.items() if r['role']=='training' and truth[i] not in excluded}
 expected_pool={i for i,r in assignment.items() if r['role']=='calibration' and truth[i] not in excluded}
 ck(scope_id+'_expected_sizes',len(expected_query)==scope['role_counts']['prediction'] and len(expected_source)==scope['role_counts']['training'] and len(expected_pool)==scope['role_counts']['calibration'])
 expected_random=np.random.default_rng(protocol['final']['random_query_seed']).choice(np.asarray(sorted(expected_query)),min(len(expected_query),protocol['final']['random_query_examples']),replace=False).tolist()
 ck(scope_id+'_random_ids_predeclared_seed',random_ids[scope_id]==expected_random)
 for selector in protocol['selectors']:
  directory=root/'final'/scope_id/selector;meta=readj(directory/'metadata.json');classes=meta['class_order']
  ck(scope_id+selector+'_metadata_class_order',classes==sorted({truth[i] for i in expected_source}))
  ck(scope_id+selector+'_fixed_query_ids',len(meta['query_ids'])==len(expected_query) and set(meta['query_ids'])==expected_query)
  ck(scope_id+selector+'_source_ids',len(meta['source_sample_ids'])==len(expected_source) and set(meta['source_sample_ids'])==expected_source)
  ck(scope_id+selector+'_source_frozen',meta['source_fingerprint_before']==meta['source_fingerprint_after'])
  ck(scope_id+selector+'_no_query_fit',meta['query_labels_used_to_fit'] is False and meta['query_statistics_fitted'] is False)
  artifact=Path(meta['source_artifact']['path']);artifact=artifact if artifact.is_absolute() else repo/artifact
  ck(scope_id+selector+'_artifact_sha',sha(artifact)==meta['source_artifact']['sha256'])
  model=joblib.load(artifact)
  ck(scope_id+selector+'_reloaded_fingerprint',joblib.hash(model)==meta['source_fingerprint_before'])
  draws={(int(d['repeat']),int(d['shots_per_class'])):d['support_ids'] for d in meta['draws']}
  ck(scope_id+selector+'_draw_count',len(draws)==protocol['final']['support_draws']*len(scope['shots']))
  for (repeat,k),ids in draws.items():
   ck('draw_IDs',len(ids)==len(set(ids))==len(classes)*k and set(ids)<=expected_pool and not(set(ids)&expected_query))
   ck('draw_balanced_supports',Counter(truth[i] for i in ids)=={c:k for c in classes})
   smaller=[j for j in scope['shots'] if j<k]
   for j in smaller:ck('nested_support_ids',set(draws[(repeat,j)])<set(ids))
  per_model_rows=[]
  for k in scope['shots']:
   file=directory/f'predictions_k{k}.csv.gz'; ck('gzip_sha',sha(file)==meta['predictions_files'][file.name])
   groups=defaultdict(dict); candidate_ids={}; count=0
   with gzip.open(file,'rt',encoding='utf-8',newline='') as f:
    reader=csv.DictReader(f)
    for row in reader:
     case=key(row);identifier=row['record_id'];predicted=row['y_predicted'];actual=row['y_true']
     ck('raw_context',case[0]==scope_id and case[1]==selector and case[2]==k)
     ck('raw_truth_matches_canonical',identifier in expected_query and truth[identifier]==actual)
     ck('raw_pred_known_class',predicted in classes)
     ck('raw_is_correct',int(row['is_correct'])==int(actual==predicted))
     ck('unique_ID_per_case',identifier not in groups[case])
     groups[case][identifier]=predicted;candidate_ids.setdefault(case,row['candidate_id'])
     ck('case_candidate_consistent',candidate_ids[case]==row['candidate_id'])
     if int(row['repeat'])==0 and row['method'] in{'source_identity','development_selected'} and identifier in set(random_ids[scope_id]):random_raw.append(row)
     count+=1
   total_raw_rows+=count
   for case,pred in groups.items():
    ck('complete_query_per_case',set(pred)==expected_query)
    row=recorded[case];ck('candidate_ID_in_metric',candidate_ids[case]==row['candidate_id'])
    metrics=evaluate(pred,classes)
    for name in('accuracy','balanced_accuracy','macro_f1'):close(name,metrics[name],row[name])
    ck('CM',metrics['confusion_matrix']==row['confusion_matrix'])
    ck('class_counts',metrics['class_counts']==row['class_counts'])
    ck('correct_count',metrics['correct']==row['correct'] and metrics['total']==row['total'])
    for left,right in zip(metrics['class_recall'],row['class_recall']):close('class_recall',left,right)
    baseline=groups[(scope_id,selector,k,case[3],'source_identity')];bm=evaluate(baseline,classes)
    for name in('accuracy','balanced_accuracy','macro_f1'):close('gain_'+name,metrics[name]-bm[name],row[name+'_gain'])
    corrected=sum(baseline[i]!=truth[i] and pred[i]==truth[i] for i in expected_query)
    harmed=sum(baseline[i]==truth[i] and pred[i]!=truth[i] for i in expected_query)
    ck('paired_errors',corrected==row['corrected_errors'] and harmed==row['new_errors'])
    close('paired_accuracy_arithmetic',(corrected-harmed)/len(expected_query),row['accuracy_gain'])
    candidate=grid[candidate_ids[case]]
    if case[-1].startswith('matched_zero_support_'):matched=pred
    elif candidate['family']=='prototype_shrinkage':matched=groups[(scope_id,selector,k,case[3],'matched_zero_support_'+candidate['id'])]
    else:matched=baseline
    matched_accuracy=sum(matched[i]==truth[i] for i in expected_query)/len(expected_query)
    close('matched_control_gain',metrics['accuracy']-matched_accuracy,row['gain_vs_matched_control'])
    if case[-1]=='development_selected':ck('locked_global_selection',candidate['id']==lock['choices'][scope_id][selector][str(k)]['selected']['id'])
    if case[-1].startswith('family_'):ck('locked_family_selection',candidate['id']==lock['choices'][scope_id][selector][str(k)]['family_winners'][case[-1].removeprefix('family_')]['candidate']['id'])
    rebuilt={**row,**metrics,'corrected_errors':corrected,'new_errors':harmed,'gain_vs_matched_control':metrics['accuracy']-matched_accuracy}
    recomputed[case]=rebuilt;per_model_rows.append(rebuilt)
   methods={case[-1] for case in groups}
   ck('all_repeats_recorded',all(len({case[3] for case in groups if case[-1]==method})==protocol['final']['support_draws'] for method in methods))
   file_reports.append({'file':str(file.relative_to(root)),'rows':count,'cases':len(groups),'query_count':len(expected_query),'methods':len(methods),'sha256':sha(file)})
  ck('metadata_evaluation_count',len(per_model_rows)==meta['evaluations'])
  model_reports.append({'scope':scope_id,'selector':selector,'source_fingerprint':meta['source_fingerprint_before'],'artifact_sha256_verified':True,'reloaded_fingerprint_verified':True,'query_count':len(expected_query),'source_count':len(expected_source),'calibration_pool_count':len(expected_pool),'evaluations':len(per_model_rows),'baseline_accuracy':meta['baseline_metrics']['accuracy']})
ck('all_cases_recomputed',set(recomputed)==set(recorded))
ck('all_raw_rows_count',total_raw_rows==status['prediction_rows']==sum(r['total'] for r in all_rows))
# Recalculate summary means, population SD, ranges, gains and draw sign counts without project helpers.
summary=readj(root/'summary.json');summaries={summary_key(r):r for r in summary};grouped=defaultdict(list)
for row in recomputed.values():grouped[summary_key(row)].append(row)
ck('summary_keys',set(grouped)==set(summaries) and len(summary)==len(summaries))
for sk,cases in grouped.items():
 row=summaries[sk];ck('summary_evaluations',row['evaluations']==len(cases))
 for metric in('accuracy','balanced_accuracy','macro_f1'):
  values=[r[metric] for r in cases]
  for suffix,value in [('mean',statistics.mean(values)),('sd',statistics.pstdev(values)),('min',min(values)),('max',max(values))]:close('summary_'+metric+suffix,value,row[metric+'_'+suffix])
  close('summary_gain',statistics.mean(r[metric+'_gain'] for r in cases),row[metric+'_gain_mean'])
 for label,op in [('positive',lambda x:x>0),('negative',lambda x:x<0),('zero',lambda x:x==0)]:ck('summary_signs',row[label+'_accuracy_draws']==sum(op(r['accuracy_gain']) for r in cases))
 close('summary_corrected',statistics.mean(r['corrected_errors'] for r in cases),row['paired_corrected_mean'])
 close('summary_harmed',statistics.mean(r['new_errors'] for r in cases),row['paired_harmed_mean'])
 close('summary_matched_gain',statistics.mean(r['gain_vs_matched_control'] for r in cases),row['gain_vs_matched_zero_support_mean'])
# Random rows must reproduce every selected ID, including errors, across all intended methods/scopes/K/selectors.
with (root/'random_query_predictions.csv').open(encoding='utf-8-sig',newline='') as f:random_saved=list(csv.DictReader(f))
normalized=lambda row:tuple((k,str(row[k])) for k in sorted(row))
ck('random_rows_exact_raw_subset',Counter(normalized(r) for r in random_saved)==Counter(normalized(r) for r in random_raw))
expected_random_rows=sum(protocol['final']['random_query_examples']*len(protocol['selectors'])*len(s['shots'])*2 for s in protocol['scopes'])
ck('all_preselected_random_rows_included',len(random_saved)==expected_random_rows)
# Check final per-model metrics/summary sidecars are identical to globally archived subsets.
for meta in model_reports:
 d=root/'final'/meta['scope']/meta['selector'];subset=[r for r in all_rows if r['scope']==meta['scope'] and r['selector']==meta['selector']]
 ck('per_model_metrics_sidecar',readj(d/'metrics.json')==subset)
 ck('per_model_summary_sidecar',readj(d/'summary.json')==[r for r in summary if r['scope']==meta['scope'] and r['selector']==meta['selector']])
report={'status':'pass','auditor':'independent subagent /root/calibration_label_audit','scope':'Independent streaming recomputation of every final per-ID prediction using original NPZ labels and manual confusion-matrix formulas; no project scoring helpers, fitting, tuning, or label changes.','checks_evaluated':checks,'failed_checks':failures,'maximum_numeric_difference':max_difference,'prediction_rows':total_raw_rows,'cases':len(recomputed),'summary_rows':len(summary),'files':file_reports,'models':model_reports,'random_example_rows':len(random_saved),'random_example_incorrect_rows':sum(int(r['is_correct'])==0 for r in random_saved),'random_examples_seed_verified':True,'all_random_examples_preserved_including_errors':True,'all_matched_zero_support_controls_checked':True,'canonical_labels_consistent':True,'named_geographic_origins_verified':False,'selection_lock_sha256':lock_sha['sha256'],'selection_candidates_match_locked_choices':True,'original_fixed_threeway_split_unchanged':True,'limits':['Anonymous origin labels are consistent with supplied workbooks and metadata, not independently confirmed geographic growth origins.','This audit verifies recorded lock provenance, deterministic random IDs, source hashes and final arithmetic; no external execution attestation or independent new-batch evidence.','Repeated supports share fixed queries and are not independent specimens or batches.','Does not rerun development model fitting or choose a replacement method based on final scores.']}
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in['status','checks_evaluated','maximum_numeric_difference','prediction_rows','cases','summary_rows','random_example_rows','random_example_incorrect_rows']},ensure_ascii=False))
