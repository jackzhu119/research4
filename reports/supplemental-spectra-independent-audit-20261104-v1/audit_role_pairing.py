"""Read-only role, paired-cohort and locked-calibration audit after scoring.

No estimators are fitted/deserialized, no result is selected, and no quantum
files or large workbooks are accessed. New output is this independent receipt.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'reports/supplemental-spectra-benchmark-20261104-v1'
BOUND = ROOT/'data/splits/supplemental_spectra_20261104_v1/bound_protocol.json'


def require(value,message):
    if not value:raise AssertionError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(base,ref):
    path=(base/ref['path']).resolve()
    require(path.is_relative_to(base.resolve()) and sha(path)==ref['sha256'],'Changed/outside locked artifact: '+str(path))
    return path


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def alias_check(rolepacks):
    parent={}
    def find(k):
        parent.setdefault(k,k)
        if parent[k]!=k:parent[k]=find(parent[k])
        return parent[k]
    def join(a,b):parent[find(a)]=find(b)
    rows=[]
    for role,p in rolepacks.items():
        for i,(rid,eid,legacy) in enumerate(zip(p['record_ids'],p['entity_ids'],p['legacy_match_ids'],strict=True)):
            keys=[str(rid),str(eid),*json.loads(str(legacy))]
            for k in keys[1:]:join(keys[0],str(k))
            rows.append((role,keys))
    components=defaultdict(set)
    for role,keys in rows:components[find(keys[0])].add(role)
    require(all(len(v)==1 for v in components.values()),'Identity aliases span roles')
    return len(components)


def main():
    protocol=json.loads(BOUND.read_text());source=json.loads((RUN/'source_lock.json').read_text())
    prediction=json.loads((RUN/'prediction_lock.json').read_text());score=json.loads((RUN/'scoring/score_lock.json').read_text())
    require(sha(BOUND)==source['protocol_sha256']==prediction['protocol_sha256']==score['protocol_sha256'],'Protocol chain differs')
    require(source['code_version']==prediction['code_version'],'Source/prediction code version differs')
    require(prediction['source_lock_sha256']==sha(RUN/'source_lock.json'),'Prediction did not bind source lock')
    require(score['prediction_lock_sha256']==sha(RUN/'prediction_lock.json'),'Score did not bind prediction lock')
    require(source['query_truth_read'] is False and prediction['query_truth_read'] is False,'Earlier stage reports truth access')
    require(score['query_truth_first_opened_in']=='score','Scoring truth boundary changed')
    require(json.loads((RUN/'protocol_snapshot.json').read_text())==protocol,'Actual protocol snapshot differs')
    release=json.loads((RUN/'predictions/release.json').read_text())
    require(release['source_lock_sha256']==sha(RUN/'source_lock.json') and release['allowed_stages']==['predict'],'Prediction release chain differs')
    tasks={t['id']:t for t in protocol['tasks']};packs={};truths={};role_audit={};input_hashes={}
    expected_counts={'zaoren_legacy7':(513,171,171),'zaoren_expanded10':(516,174,174),
                     'huangqi_legacy4':(241,80,82),'tianma_legacy4':(242,80,83)}
    for tid,t in tasks.items():
        own={role:npz(checked(ROOT,t[role+'_pack'])) for role in ('source','support','query')}
        for role,p in own.items():
            require(len(set(p['record_ids']))==len(p['record_ids']),'Repeated role record ID')
            require(len(p['record_ids'])==expected_counts[tid][('source','support','query').index(role)],'Role denominator changed')
            require(np.array_equal(p['axis'],own['source']['axis']),'Role axes differ')
            input_hashes[t[role+'_pack']['path']]=t[role+'_pack']['sha256']
        require(not {'y','labels','y_true','origin_group_ids'}&set(own['query']),'Query payload contains truth')
        truth_path=checked(ROOT,t['truth']);input_hashes[t['truth']['path']]=t['truth']['sha256']
        rows=list(csv.DictReader(truth_path.open(encoding='utf-8-sig')))
        require([r['record_id'] for r in rows]==own['query']['record_ids'].tolist(),'Truth/query order differs')
        truth={r['record_id']:r['y'] for r in rows};truths[tid]=truth
        components=alias_check(own)
        require(components==sum(len(p['record_ids']) for p in own.values()),'Noncanonical repeated role entity')
        role_audit[tid]={'counts':dict(zip(('source','support','query'),expected_counts[tid])),
                        'identity_components':components,'query_label_keys_present':False,
                        'query_class_counts':dict(Counter(truth.values())),'truth_sha256':sha(truth_path),
                        'exposure':t['exposure']}
        packs[tid]=own

    # Reconstruct the historical membership map independently of the new draft.
    legacy_rows=list(csv.DictReader((ROOT/protocol['legacy_zaoren_split']).open(encoding='utf-8-sig')))
    role_names={'training':'source','calibration':'support','prediction':'query'}
    for row in legacy_rows:
        require(row['record_id'] in set(packs['zaoren_legacy7'][role_names[row['role']]]['record_ids']),
                'A changed an old registered role')
    a,b=packs['zaoren_legacy7'],packs['zaoren_expanded10']
    paired={}
    for role in ('source','support','query'):
        pb={str(i):k for k,i in enumerate(b[role]['record_ids'])}
        indices=[pb[str(i)] for i in a[role]['record_ids']]
        for key in ('X','entity_ids','legacy_match_ids','axis'):
            observed=b[role][key] if key=='axis' else b[role][key][indices]
            require(np.array_equal(a[role][key],observed),'A/B old-role payload differs: '+role+'/'+key)
        if role!='query':require(np.array_equal(a[role]['y'],b[role]['y'][indices]),'A/B old source/support labels differ')
        else:require(all(truths['zaoren_legacy7'][str(i)]==truths['zaoren_expanded10'][str(i)] for i in a[role]['record_ids']),
                     'A/B old query truth differs')
        extra=[str(i) for i in b[role]['record_ids'] if str(i) not in set(a[role]['record_ids'])]
        require(len(extra)==3,'Expanded role has unexpected new count')
        labels=(dict(zip(map(str,b[role]['record_ids']),map(str,b[role]['y']),strict=True)) if role!='query' else truths['zaoren_expanded10'])
        require(Counter(labels[i] for i in extra)=={'ZY08':1,'ZY09':1,'ZY10':1},'New groups do not each have one specimen per role')
        paired[role]={'old_records_exactly_equal':len(indices),'new_record_ids':extra,'new_class_counts':dict(Counter(labels[i] for i in extra))}
    require(not tasks['zaoren_expanded10']['inner_folds'] and not tasks['zaoren_expanded10']['source_calibration_episodes'],'B silently used singleton CV/meta selection')
    require(source['selection']['zaoren_expanded10']['status']=='INHERITED_SOURCE_CONFIGURATION','B inheritance absent')
    require(source['selection']['zaoren_expanded10']['selected']==source['selection']['zaoren_legacy7']['selected'],'B changed A configuration')

    # Every auxiliary role is a subset of the registered old measurement data.
    for tid,sp in [('huangqi_legacy4','huangqi'),('tianma_legacy4','tianma')]:
        old=npz(ROOT/f'data/multiplant_database/{sp}/alpha.npz');mask=(old['axis']>=.1)&(old['axis']<=1.5)
        positions={str(i):k for k,i in enumerate(old['record_ids'])};seen=[]
        for role,p in packs[tid].items():
            indices=[positions[str(i)] for i in p['record_ids']];seen+=p['record_ids'].tolist()
            require(np.array_equal(p['X'],old['X'][indices][:,mask]),'Auxiliary alpha changed instead of legacy reuse')
            require(np.array_equal(p['axis'],old['axis'][mask]),'Auxiliary axis differs')
            labels=p['y'] if role!='query' else np.array([truths[tid][str(i)] for i in p['record_ids']])
            require(np.array_equal(labels,old['filename_groups'][indices]),'Auxiliary filename-group label changed')
        require(len(seen)==len(set(seen))==len(old['record_ids']) and set(seen)==set(old['record_ids']),'Auxiliary records duplicated or missing')
        role_audit[tid]['all_records_equal_registered_legacy']=True
        role_audit[tid]['new_specimens_from_reupload']=0

    seeds=protocol['seeds'];families=list(protocol['candidates']);models=source['models']
    expected_models={(t,f,s) for t in tasks for f in families for s in seeds}
    require({(m['task'],m['family'],m['seed']) for m in models}==expected_models and len(models)==48,'Missing task/family/seed final models')
    models_by={(m['task'],m['family'],m['seed']):m for m in models}
    for m in models:
        checked(RUN,m);checked(RUN,m['source_reference'])
        require(m['config']==source['selection'][m['task']]['selected'][m['family']],'Final model config differs from source lock')
        require(m['source_pack_sha256']==tasks[m['task']]['source_pack']['sha256'],'Final model source pack differs')
    require(datetime.fromisoformat(source['ended_utc'])<datetime.fromisoformat(prediction['ended_utc']),'Source lock does not precede prediction lock')
    refs=prediction['predictions'];require(len(refs)==1020,'Prediction artifact count differs')
    baseline={};loaded={};counts=Counter();zero_count=0;noop_count=0;rank_guard_rows=0;head_counts={}
    expected=set()
    for tid in tasks:
        for family in families:
            for seed in seeds:
                expected.add((tid,family,seed,'none',0,-1))
                if family=='mcuve_svm':
                    expected.add((tid,family,seed,'raw_score_argmax_head_control',0,-1))
                    for draw in tasks[tid]['calibration_draws']:
                        for k in draw['shots']:
                            for method in ('zero_label_control','traditional','conservative','traditional_unwrapped_anchor_argmax_diagnostic'):
                                expected.add((tid,family,seed,method,int(k),draw['draw_id']))
    actual=[tuple(r[k] for k in ('task','family','seed','method','shot','draw')) for r in refs]
    require(len(set(actual))==len(refs) and set(actual)==expected,'Prediction arms/draw/shot set differs')
    for ref in refs:
        data=npz(checked(RUN,ref));key=(ref['task'],ref['family'],ref['seed'])
        require(np.array_equal(data['record_ids'],packs[ref['task']]['query']['record_ids']),'Prediction query identity/order changed')
        require(np.array_equal(data['classes'],np.array(tasks[ref['task']]['classes'])),'Prediction vocabulary changed')
        require(data['scores'].shape==(len(data['record_ids']),len(data['classes'])) and np.isfinite(data['scores']).all(),'Incomplete scores')
        require(set(data['predictions'])<=set(data['classes']),'Prediction outside source vocabulary')
        require(ref['source_model_sha256']==models_by[key]['sha256'],'Prediction used different source weight')
        loaded[tuple(ref[k] for k in ('task','family','seed','method','shot','draw'))]=data
        counts[ref['task']+'/'+ref['method']]+=1
        if ref['method']=='none':baseline[key]=data
    require(len(baseline)==48,'Missing native base prediction')
    for ref in refs:
        tid=ref['task'];key=(tid,ref['family'],ref['seed']);native=baseline[key]
        d=loaded[tuple(ref[k] for k in ('task','family','seed','method','shot','draw'))]
        if ref['method']=='raw_score_argmax_head_control':
            require(np.array_equal(d['predictions'],d['classes'][native['scores'].argmax(1)]),'Head control not raw argmax')
            head_counts[tid+'/'+str(ref['seed'])]=int(np.sum(d['predictions']!=native['predictions']))
            control_path=RUN/ref['path'];controls=json.loads((control_path.parent/'zero_support_controls.json').read_text())
            require(set(controls)=={'none','traditional','conservative','zero_label_control'} and all(v['exact_native_identity'] for v in controls.values()),'Missing zero-support exact-identity controls')
            zero_count+=len(controls)
        if ref['shot']>0:
            expected_ids=tasks[tid]['calibration_draws'][ref['draw']]['shots'][str(ref['shot'])]
            require(ref['support_ids']==expected_ids and len(expected_ids)==len(set(expected_ids)),'Support draw changed or oversampled')
            lookup={str(i):str(y) for i,y in zip(packs[tid]['support']['record_ids'],packs[tid]['support']['y'],strict=True)}
            require(set(expected_ids)<=set(lookup),'Support ID outside its locked pool')
            require(Counter(lookup[i] for i in expected_ids)=={c:ref['shot'] for c in tasks[tid]['classes']},'Uniform shot class count differs')
            for event in ref.get('fit_events',[]):
                if 'fit_sample_ids' in event:require(event['fit_sample_ids']==expected_ids,'Adaptation fit touched other rows')
        if ref['method']=='zero_label_control' or ref['method']=='conservative' and source['conservative_policies'][tid]['fallback_to_identity']:
            require(np.array_equal(d['predictions'],native['predictions']) and np.array_equal(d['scores'],native['scores']),'No-op/zero-label changed native output')
            noop_count+=1
        if ref['method'] in ('traditional','conservative'):
            changed=d['predictions']!=native['predictions']
            ranking=d['scores'].argmax(1)!=native['scores'].argmax(1)
            numerical=np.any(d['scores']!=native['scores'],axis=1)
            require(not np.any(changed&~(ranking&numerical)),'Native prediction changed without label-induced score-winner change')
            rank_guard_rows+=len(changed)
        if ref['method']=='traditional_unwrapped_anchor_argmax_diagnostic':
            require(np.array_equal(d['predictions'],d['classes'][d['scores'].argmax(1)]),'Unwrapped diagnostic not score argmax')
    require(tasks['zaoren_legacy7']['shots_available']==[1] and tasks['zaoren_expanded10']['shots_available']==[1],'Tiny source task enabled unsupported shots')
    require(all(tasks[t]['shots_available']==[1,3,5] for t in ('huangqi_legacy4','tianma_legacy4')),'Auxiliary support shots missing')
    require(source['conservative_policies']['zaoren_expanded10']['fallback_to_identity'],'B no-op rule changed')
    new_support=set(paired['support']['new_record_ids'])
    require(all(new_support<=set(d['shots']['1']) for d in tasks['zaoren_expanded10']['calibration_draws']),'B fabricated new-class support replicates')

    native_pairs=[]
    for family in families:
        for seed in seeds:
            da=baseline[('zaoren_legacy7',family,seed)];db=baseline[('zaoren_expanded10',family,seed)]
            bi={str(i):k for k,i in enumerate(db['record_ids'])};idx=[bi[str(i)] for i in da['record_ids']]
            y=np.array([truths['zaoren_legacy7'][str(i)] for i in da['record_ids']]);pa=da['predictions'];pb=db['predictions'][idx]
            newids=paired['query']['new_record_ids'];newidx=[bi[i] for i in newids]
            newy=np.array([truths['zaoren_expanded10'][i] for i in newids])
            native_pairs.append({'family':family,'seed':seed,'paired_old_query_n':171,'A_correct':int(np.sum(pa==y)),
                                 'B_old_correct':int(np.sum(pb==y)),'B_old_minus_A_accuracy_pp':100*float(np.mean(pb==y)-np.mean(pa==y)),
                                 'B_corrected_old_errors':int(np.sum((pa!=y)&(pb==y))),
                                 'B_harmed_old_correct':int(np.sum((pa==y)&(pb!=y))),
                                 'B_new_query_n':3,'B_new_correct':int(np.sum(db['predictions'][newidx]==newy))})
    with gzip.open(RUN/'scoring/all_prediction_metrics.json.gz','rt') as h:metric_rows=json.load(h)
    require(len(metric_rows)==2175==score['evaluations'],'Metric evaluation count differs')
    denominator_counts=Counter()
    for row in metric_rows:
        tid=row['task'];stratum=row['stratum']
        expected_n=len(packs[tid]['query']['record_ids']) if stratum=='all' else len(tasks[tid]['strata'][stratum])
        require(row['total']==expected_n and sum(row['class_counts'].values())==expected_n,'Metric denominator differs')
        require(sum(map(sum,row['confusion_matrix']))==expected_n and np.trace(row['confusion_matrix'])==row['correct'],'Confusion count denominator differs')
        denominator_counts[tid+'/'+stratum+'/'+str(expected_n)]+=1
    receipt={'status':'PASS_ROLE_PAIRING_LINEAGE_AND_CALIBRATION_LOCKS','audited_at_utc':datetime.now(timezone.utc).isoformat(),
             'no_fits_or_model_deserialization':True,'no_quantum_or_git':True,'query_truth_read_only_after_authorized_scoring':True,
             'boundary_evidence':'Actual bound hashes, stage locks, recorded API flags and reviewed code; not an OS-level file-access proof',
             'protocol_sha256':sha(BOUND),'source_lock_sha256':sha(RUN/'source_lock.json'),
             'prediction_lock_sha256':sha(RUN/'prediction_lock.json'),'score_lock_sha256':sha(RUN/'scoring/score_lock.json'),
             'source_end_utc':source['ended_utc'],'prediction_lock_end_utc':prediction['ended_utc'],
             'roles':role_audit,'A_B_pairing':paired,'B_configuration_equals_A_source_choice':True,
             'final_models':48,'seeds':seeds,'prediction_artifacts':len(refs),'prediction_method_counts':dict(counts),
             'zero_support_identity_assertions':zero_count,'actual_zero_label_or_no_op_artifacts_equal_native':noop_count,
             'wrapped_calibration_rows_checked_for_rank_change':rank_guard_rows,'raw_head_disagreements':head_counts,
             'metric_evaluations':len(metric_rows),'metric_denominators':dict(denominator_counts),
             'native_paired_old171_comparisons':native_pairs,'input_hashes':input_hashes,
             'limitations':['Old855/HQ403/TM405 are exposed development data; reuploads add no independent N',
                            'B new query has only3 physical specimens,1 per new class; seeds/draws are not independent sample replications',
                            'A7 vs B10 full-query metrics have different class spaces/denominators and are not direct same-task accuracy gains',
                            'Paired A/B old171 compares known-class retention after enrollment, with the same specimens and labels',
                            'Main historical96.0234% nested855 protocol remains a separate retrospective reference']}
    path=OUT/'role_pairing_audit.json'
    with path.open('x') as h:json.dump(receipt,h,ensure_ascii=False,indent=2);h.write('\n')
    print(json.dumps({'status':receipt['status'],'receipt':str(path.relative_to(ROOT)),'sha256':sha(path),
                      'predictions':len(refs),'metric_denominators':dict(denominator_counts),'paired_native':native_pairs},ensure_ascii=False))


if __name__=='__main__':main()
