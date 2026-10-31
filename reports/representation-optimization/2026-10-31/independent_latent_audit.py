"""Independent latent-stage selection and per-specimen score audit; no fitting."""
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time

import joblib
import numpy as np


REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'outputs/representation-optimization-20261031'
ROOT=BASE/'adaptation'
SOURCE=BASE/'source'
OUTPUT=BASE/'independent_latent_audit.json'
START=time.monotonic()
checks=0; maximum=0.; failures=[]
METRICS=('accuracy','balanced_accuracy','macro_f1')
DEV=METRICS+tuple('matched_'+v for v in METRICS)


def ck(condition,message):
    global checks
    checks+=1
    if not condition:
        failures.append(message)
        raise AssertionError(message)


def close(a,b,message):
    global maximum
    delta=abs(float(a)-float(b));maximum=max(maximum,delta)
    ck(delta<=1e-12,message)


def readj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def objsha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def score(pred,truth,classes):
    vocabulary={c:i for i,c in enumerate(classes)}
    cm=[[0]*len(classes) for _ in classes]
    for identifier,label in pred.items():cm[vocabulary[truth[identifier]]][vocabulary[label]]+=1
    counts=[sum(row) for row in cm]; guesses=[sum(row[i] for row in cm) for i in range(len(classes))]
    diag=[cm[i][i] for i in range(len(classes))]
    recall=[d/n if n else 0. for d,n in zip(diag,counts)]
    f1=[2*d/(n+p) if n+p else 0. for d,n,p in zip(diag,counts,guesses)]
    return {'accuracy':sum(diag)/len(pred),'balanced_accuracy':math.fsum(recall)/len(classes),
            'macro_f1':math.fsum(f1)/len(classes),'confusion_matrix':cm,'class_recall':recall,
            'class_counts':counts,'correct':sum(diag),'total':len(pred)}


def compare_details(actual,saved):
    for key in ['confusion_matrix','class_counts','correct','total']:ck(actual[key]==saved[key],'Detailed score mismatch: '+key)
    for key in METRICS:close(actual[key],saved[key],'Scalar score mismatch: '+key)
    for a,b in zip(actual['class_recall'],saved['class_recall']):close(a,b,'Recall mismatch')


def main():
    manifest=readj(ROOT/'latent_run_manifest.json');protocol=readj(REPO/'data/research/representation_calibration_protocol_20261031.json')
    plan_path=REPO/protocol['fixed_split'];plan=readj(plan_path)
    npz_path=REPO/'outputs/approved-zaoren-20261031/data-preparation/dataset.npz'
    for path,key in [(npz_path,'input_sha256'),(plan_path,'plan_sha256'),(REPO/'data/research/representation_calibration_protocol_20261031.json','protocol_sha256')]:ck(sha(path)==manifest[key],'Input SHA mismatch')
    for path,digest in manifest['source_code_sha256'].items():ck(sha(REPO/path)==digest,'Executed source SHA changed: '+path)
    ck(platform.python_version()==manifest['python'],'Python version mismatch')
    for name,version in manifest['dependencies'].items():ck(importlib.metadata.version(name)==version,'Dependency version mismatch')
    with np.load(npz_path,allow_pickle=False) as d:
        ids=d['record_ids'].copy(); y=d['y'].copy(); axis=d['axis'].copy(); alpha=d['X_alpha'].copy()
    truth=dict(zip(ids.tolist(),y.tolist()));classes=sorted(set(y.tolist()))
    roles={role:{r['record_id'] for r in plan['records'] if r['role']==role} for role in ['training','calibration','prediction']}
    ck([len(v) for v in roles.values()]==[513,171,171],'Outer role counts')
    ck(all(not roles[a]&roles[b] for a,b in [('training','calibration'),('training','prediction'),('calibration','prediction')]),'Outer role overlap')
    source_lock=readj(SOURCE/'source_selection_lock.json')
    source_ids=[c['id'] for c in source_lock['candidates_for_adaptation']]
    ck(source_ids==manifest['source_candidate_ids'],'Source cohort mismatch')
    ck(sha(SOURCE/'source_selection_lock.json')==manifest['source_selection_lock_sha256'],'Source lock mismatch')
    ck(sha(SOURCE/'source_run_manifest.json')==manifest['source_run_manifest_sha256'],'Source manifest mismatch')
    candidates=[]
    def add(family,**kwargs):candidates.append({'id':f'l{len(candidates):03d}_{family}','family':family,**kwargs})
    add('identity')
    for anchor,ridge,weight in product([.1,1.,10.],[.1,1.,10.],[1.,5.,20.]):add('anchored_ridge',source_anchor=anchor,ridge=ridge,support_weight=weight)
    for neighbors,boost,blend in product([3,7,15],[1.,5.,20.],[.1,.25,.5]):add('local_memory',neighbors=neighbors,support_boost=boost,blend=blend)
    for pseudo in [1.,3.,10.,30.,100.]:add('mahalanobis_centroid',source_pseudocount=pseudo)
    ck(candidates==readj(ROOT/'latent_candidate_grid.json'),'Independent candidate grid mismatch')
    ck(objsha(candidates)==manifest['latent_grid_sha256']==protocol['candidates']['latent_grid_sha256'],'Latent grid SHA mismatch')
    lookup={c['id']:c for c in candidates};identity=candidates[0]['id'];config=protocol['calibration_search']
    recorded_aggregate=readj(ROOT/'latent_development_aggregate.json');partitions=readj(SOURCE/'development_partitions.json')
    recomputed={};dev_rows=0;dev_meta_count=0
    for source in source_ids:
        accum=defaultdict(lambda:{name:[] for name in DEV})
        for episode in range(config['episodes']):
            directory=ROOT/'development'/source;path=directory/f'episode_{episode:02d}.csv.gz';meta=readj(directory/f'episode_{episode:02d}.json')
            ck(meta['partition']==partitions[episode],'Development partition differs from Stage A')
            ck(meta['new_source_fit_calls']==0 and meta['candidate_fit_calls']==1200 and not meta['failures'],'Development fit/failure count')
            ck(meta['source_fingerprint_before']==meta['source_fingerprint_after'] and meta['context_fingerprint_before']==meta['context_fingerprint_after'],'Development freeze assertion')
            source_meta=readj(SOURCE/'development'/source/f'episode_{episode:02d}.json')
            ck(meta['source_model_sha256']==source_meta['model_sha256']==sha(SOURCE/'development'/source/f'episode_{episode:02d}.joblib'),'Cached source artifact mismatch')
            ck(meta['source_fingerprint_before']==source_meta['source_fingerprint_before'],'Cached source fingerprint mismatch')
            ck(sha(path)==meta['metrics_file_sha256'],'Development metric-file SHA')
            pool=meta['partition']['support_pool'];pool_y=np.array([truth[v] for v in pool]);rng=np.random.default_rng(config['support_seed']+episode)
            for repeat in range(config['support_draws']):
                sampled=[]
                for label in classes:
                    members=np.flatnonzero(pool_y==label);sampled.extend(np.array(pool)[rng.permutation(members)[:1]].tolist())
                draw=meta['draws'][repeat]
                ck(draw['repeat']==repeat and draw['support_ids']==sampled,'Seeded support draw mismatch')
                ck(len(set(sampled))==7 and set(sampled)<=set(pool) and not set(sampled)&set(meta['partition']['query']),'Support membership or overlap')
            seen=set()
            with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
                for row in csv.DictReader(f):
                    stage='selection' if episode<config['selection_episodes'] else 'gate'
                    ck(row['source_id']==source and int(row['episode'])==episode and row['stage']==stage,'Development row identifiers')
                    ck(row['candidate_id'] in lookup and row['family']==lookup[row['candidate_id']]['family'],'Unknown development candidate')
                    ck(row['status']=='pass' and row['error']=='','Development candidate failure')
                    pair=(int(row['repeat']),row['candidate_id']);ck(pair not in seen,'Duplicate development evaluation');seen.add(pair)
                    for name in DEV:
                        value=float(row[name]);ck(0<=value<=1,'Development metric outside bounds');accum[(stage,row['candidate_id'])][name].append(value)
                    if row['candidate_id']==identity:
                        for name in METRICS:
                            close(row[name],source_meta['metrics'][name],'Identity differs from Stage-A source prediction')
                            close(row['matched_'+name],row[name],'Identity matched-control mismatch')
                    dev_rows+=1
            ck(len(seen)==1200,'Incomplete development checkpoint')
            dev_meta_count+=1
        stats={stage:{} for stage in ['selection','gate']}
        for (stage,cid),values in accum.items():
            mean={name:math.fsum(nums)/len(nums) for name,nums in values.items()}
            mean.update(evaluations=len(values['accuracy']),failures=0);stats[stage][cid]=mean
            for name in DEV:close(mean[name],recorded_aggregate[source][stage][cid][name],'Development aggregate mismatch')
            ck(mean['evaluations']==recorded_aggregate[source][stage][cid]['evaluations'] and recorded_aggregate[source][stage][cid]['failures']==0,'Aggregate counts')
        recomputed[source]=stats
    lock=readj(ROOT/'latent_selection_lock.json')
    rank={'identity':0,'mahalanobis_centroid':1,'local_memory':2,'anchored_ridge':3}
    def complexity(c):return(rank[c['family']],c.get('blend',0),c.get('support_boost',0),c.get('support_weight',0),-c.get('source_anchor',0),-c.get('ridge',0),-c.get('source_pseudocount',0),-c.get('neighbors',0),c['id'])
    expected_choices={}
    for source,stats in recomputed.items():
        def passes(cid,stage):
            v,b=stats[stage][cid],stats[stage][identity]
            return(not v['failures'] and v['evaluations']==b['evaluations'] and v['accuracy']-b['accuracy']>config['minimum_accuracy_gain'] and v['macro_f1']-b['macro_f1']>=-config['max_macro_f1_loss'] and v['balanced_accuracy']-b['balanced_accuracy']>=-config['max_balanced_accuracy_loss'] and v['accuracy']-v['matched_accuracy']>config['minimum_accuracy_gain'] and v['macro_f1']-v['matched_macro_f1']>=-config['max_macro_f1_loss'] and v['balanced_accuracy']-v['matched_balanced_accuracy']>=-config['max_balanced_accuracy_loss'])
        def best(which):
            maximum=max(stats['selection'][cid]['accuracy'] for cid in which)
            return min([lookup[cid] for cid in which if stats['selection'][cid]['accuracy']>=maximum-config['tie_tolerance']],key=complexity)
        eligible=[c['id'] for c in candidates if passes(c['id'],'selection')]
        proposed=best(eligible) if eligible else lookup[identity]
        selected=proposed if proposed['family']!='identity' and passes(proposed['id'],'gate') else lookup[identity]
        saved=lock['choices'][source]
        ck(saved['proposed']==proposed and saved['selected']==selected,'Independent global adapter choice')
        families={}
        for family in rank:
            available=[c['id'] for c in candidates if c['family']==family and stats['selection'][c['id']]['failures']==0]
            winner=best(available) if available else lookup[identity]
            ck(saved['family_winners'][family]['candidate']==winner and saved['family_winners'][family]['passes_gate']==passes(winner['id'],'gate'),'Independent family adapter choice/gate')
            families[family]=winner
        expected_choices[source]={'selected':selected,'proposed':proposed,'families':families}
    for path,key in [(SOURCE/'source_selection_lock.json','source_selection_lock_sha256'),(ROOT/'latent_run_manifest.json','latent_run_manifest_sha256')]:ck(sha(path)==lock[key],'Latent lock provenance')
    ck(lock['latent_grid_sha256']==objsha(candidates) and lock['development_candidate_fit_calls']==108000,'Latent lock grid/budget')
    lock_sha=sha(ROOT/'latent_selection_lock.json');status=readj(ROOT/'status.json')
    ck(status['latent_selection_lock_sha256']==lock_sha==readj(ROOT/'latent_selection_lock_sha256.json')['sha256'],'Latent selection lock SHA')
    all_metrics=readj(ROOT/'all_final_metrics.json');saved_cases={(v['source_id'],v['method'],v['repeat']):v for v in all_metrics}
    ck(len(saved_cases)==2400,'Saved case count')
    metadata={v['source_candidate']['id']:v for v in readj(ROOT/'final_metadata.json')}
    raw_cases={};raw_count=0;random_expected=[];random_ids=readj(ROOT/'random_query_ids.json')
    rng=np.random.default_rng(protocol['final']['random_query_seed']);expected_random=rng.choice(np.array(sorted(roles['prediction'])),20,replace=False).tolist()
    ck(random_ids==expected_random,'Precommitted random IDs mismatch')
    all_lookup={str(v):i for i,v in enumerate(ids)};mask=(axis>=.1)&(axis<=1.5)
    for source in source_ids:
        meta=metadata[source];directory=ROOT/'final'/source
        ck(set(meta['source_ids'])==roles['training'] and set(meta['query_ids'])==roles['prediction'],'Final source/query IDs')
        ck(meta['source_model_fingerprint_before']==meta['source_model_fingerprint_after'] and meta['context_fingerprint_before']==meta['context_fingerprint_after'],'Final freeze assertion')
        ck(not meta['final_query_labels_used_for_fit'] and not meta['final_query_statistics_fitted'],'Final query-use assertion')
        ck(sha(directory/'source_model.joblib')==meta['source_model_sha256'] and sha(directory/'predictions.csv.gz')==meta['predictions_sha256'],'Final artifact SHA')
        model=joblib.load(directory/'source_model.joblib');ck(joblib.hash(model)==meta['source_model_fingerprint_before'],'Final model fitted-state fingerprint')
        replay=model.predict(alpha[[all_lookup[v] for v in meta['query_ids']]][:,mask]).tolist();ck(joblib.hash(model)==meta['source_model_fingerprint_before'],'Prediction replay mutated final source')
        expected_methods={'source_original':lookup[identity],'development_selected':expected_choices[source]['selected']}
        expected_methods.update({'family_'+family:c for family,c in expected_choices[source]['families'].items() if family!='identity'})
        for c in list(expected_methods.values()):
            if c['family']!='identity':expected_methods['matched_zero_'+c['id']]=c
        cal_ids=[str(v) for v in ids if str(v) in roles['calibration']];cal_y=np.array([truth[v] for v in cal_ids]);rng=np.random.default_rng(protocol['final']['support_seed'])
        for repeat in range(protocol['final']['support_draws']):
            expected=[]
            for label in classes:expected.extend(np.array(cal_ids)[rng.permutation(np.flatnonzero(cal_y==label))[:1]].tolist())
            ck(meta['draws'][repeat]['repeat']==repeat and meta['draws'][repeat]['support_ids']==expected,'Final seeded support draw mismatch')
            ck(set(expected)<=roles['calibration'] and not set(expected)&roles['prediction'] and len(set(expected))==7,'Final support membership')
        with gzip.open(directory/'predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                key=(source,row['method'],int(row['repeat']));identifier=row['record_id']
                ck(row['source_id']==source and row['method'] in expected_methods,'Final method identifier')
                c=expected_methods[row['method']];ck(row['candidate_id']==c['id'],'Final candidate differs from locked choice')
                ck(identifier in roles['prediction'] and row['y_true']==truth[identifier],'Final truth/membership mismatch')
                ck(row['y_predicted'] in classes and int(row['is_correct'])==int(row['y_predicted']==row['y_true']),'Final correctness mismatch')
                expected_used=0 if row['method'].startswith('matched_zero_') or c['family']=='identity' else 7
                ck(int(row['support_labels_used'])==expected_used,'Actual support-label count mismatch')
                case=raw_cases.setdefault(key,{});ck(identifier not in case,'Duplicate per-ID case row');case[identifier]=row['y_predicted'];raw_count+=1
                if int(row['repeat'])==0 and identifier in random_ids and row['method'] in ['source_original','development_selected']:random_expected.append(row)
        ck(raw_cases[(source,'source_original',0)]==dict(zip(meta['query_ids'],replay)),'Final source replay differs from saved CSV')
        compare_details(score(raw_cases[(source,'source_original',0)],truth,classes),meta['source_baseline_metrics'])
    recalculated={}
    reference_accuracy=score(raw_cases[(source_ids[0],'source_original',0)],truth,classes)['accuracy']
    for key,pred in raw_cases.items():
        source,method,repeat=key;ck(set(pred)==roles['prediction'],'Final case query coverage')
        saved=saved_cases[key];actual=score(pred,truth,classes);compare_details(actual,saved)
        baseline=raw_cases[(source,'source_original',repeat)];base=score(baseline,truth,classes)
        if method.startswith('matched_zero_'):matched=pred
        elif saved['candidate_id']==identity:matched=baseline
        else:matched=raw_cases[(source,'matched_zero_'+saved['candidate_id'],repeat)]
        for metric in METRICS:close(actual[metric]-base[metric],saved[metric+'_gain_vs_own_source'],'Own-source gain mismatch')
        close(actual['accuracy']-reference_accuracy,saved['accuracy_gain_vs_reference'],'Reference gain mismatch')
        close(actual['accuracy']-score(matched,truth,classes)['accuracy'],saved['accuracy_gain_vs_matched'],'Matched-zero gain mismatch')
        corrected=sum(baseline[i]!=truth[i] and pred[i]==truth[i] for i in pred);harmed=sum(baseline[i]==truth[i] and pred[i]!=truth[i] for i in pred)
        ck(corrected==saved['corrected_errors'] and harmed==saved['new_errors'],'Paired changed-error counts')
        ck(actual['correct']-base['correct']==corrected-harmed,'Paired error arithmetic')
        recalculated[key]={**actual,**{m+'_gain_vs_own_source':actual[m]-base[m] for m in METRICS},'accuracy_gain_vs_reference':actual['accuracy']-reference_accuracy,'accuracy_gain_vs_matched':actual['accuracy']-score(matched,truth,classes)['accuracy'],'corrected_errors':corrected,'new_errors':harmed}
    summary=readj(ROOT/'summary.json')
    for saved in summary:
        rows=[v for (s,m,_),v in recalculated.items() if s==saved['source_id'] and m==saved['method']]
        ck(len(rows)==saved['evaluations']==100,'Summary draw count')
        for metric in METRICS:
            values=[r[metric] for r in rows]
            for name,value in [('mean',statistics.fmean(values)),('sd',statistics.pstdev(values)),('min',min(values)),('max',max(values))]:close(value,saved[metric+'_'+name],'Summary statistic mismatch')
            close(statistics.fmean(r[metric+'_gain_vs_own_source'] for r in rows),saved[metric+'_gain_vs_own_source_mean'],'Summary source gain')
        for field,source_field in [('accuracy_gain_vs_reference_mean','accuracy_gain_vs_reference'),('accuracy_gain_vs_matched_mean','accuracy_gain_vs_matched'),('corrected_mean','corrected_errors'),('harmed_mean','new_errors')]:close(statistics.fmean(r[source_field] for r in rows),saved[field],'Summary paired mean')
        ck(sum(r['accuracy_gain_vs_own_source']>0 for r in rows)==saved['positive_draws'] and sum(r['accuracy_gain_vs_own_source']<0 for r in rows)==saved['negative_draws'] and sum(r['accuracy_gain_vs_own_source']==0 for r in rows)==saved['unchanged_draws'],'Summary gain signs')
    with (ROOT/'random_query_predictions.csv').open(encoding='utf-8-sig',newline='') as f:random_saved=list(csv.DictReader(f))
    ck(random_saved==random_expected,'Random examples missing, changed or correct-only filtered')
    sensitivity=readj(ROOT/'full_range_sensitivity/results.json');sensitivity_results=[]
    for saved in sensitivity:
        c=saved['candidate'];path=ROOT/'full_range_sensitivity'/(c['id']+'.predictions.csv')
        ck(sha(path)==saved['predictions_sha256'],'Sensitivity prediction SHA')
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        ck({r['record_id'] for r in rows}==roles['prediction'] and len(rows)==171,'Sensitivity query set')
        for row in rows:ck(row['y_true']==truth[row['record_id']] and int(row['is_correct'])==int(row['y_true']==row['y_predicted']),'Sensitivity truth/correctness mismatch')
        actual=score({r['record_id']:r['y_predicted'] for r in rows},truth,classes);compare_details(actual,saved)
        ck(saved['source_count']==513 and saved['query_count']==171 and saved['feature_count']==2578 and saved['support_labels_used']==0,'Sensitivity scope/counts')
        ck(saved['source_fingerprint_before']==saved['source_fingerprint_after'] and saved['negative_absorption_not_removed'] is True and saved['physical_effective_range_verified'] is False,'Sensitivity scope/freeze assertion')
        close(actual['accuracy']-reference_accuracy,saved['accuracy_gain_vs_reference'],'Sensitivity reference gain')
        sensitivity_results.append({'candidate_id':c['id'],'correct':actual['correct'],'accuracy':actual['accuracy'],'macro_f1':actual['macro_f1'],'balanced_accuracy':actual['balanced_accuracy'],'ZY02_recall':actual['class_recall'][1]})
    events=[json.loads(line) for line in (BASE/'latent_run.log').read_text().splitlines() if line.startswith('{')]
    locked=next(i for i,v in enumerate(events) if v.get('stage')=='all_latent_choices_locked')
    finals=[i for i,v in enumerate(events) if v.get('stage')=='latent_final_source_completed']
    ck(len(finals)==3 and all(i>locked for i in finals) and events[locked]['sha256']==lock_sha,'Choice lock did not precede final events')
    starts=[json.loads(v) for v in (BASE/'latent_run_invocations.jsonl').read_text().splitlines() if v.startswith('{')]
    stagea=[json.loads(v) for v in (BASE/'source_run_invocations.jsonl').read_text().splitlines() if v.startswith('{')]
    completed_a=next(v for v in stagea if v.get('event')=='complete');start_b=next(v for v in starts if v.get('event')=='start')
    ck(datetime.fromisoformat(completed_a['utc'])<=datetime.fromisoformat(start_b['utc']),'Stage A completion not before Stage B start')
    ck(raw_count==410400 and len(raw_cases)==2400 and dev_rows==108000 and dev_meta_count==90,'Final workload counts')
    return {'status':'pass','auditor':'independent Stage-B aggregation/selection and complete per-specimen metric audit; no project aggregation/selection helper imported','checks_evaluated':checks,'failed_checks':failures,'maximum_numeric_difference':maximum,'development_files':dev_meta_count,'development_rows':dev_rows,'final_prediction_rows':raw_count,'final_cases':len(raw_cases),'summary_rows':len(summary),'source_count':len(source_ids),'independently_recomputed_choices':expected_choices,'latent_selection_lock_sha256':lock_sha,'random_example_rows':len(random_saved),'random_incorrect_rows_preserved':sum(int(r['is_correct'])==0 for r in random_saved),'final_source_replays':3,'new_fits_in_audit':0,'sensitivity_predictions_checked':sum(v['correct']*0+171 for v in sensitivity_results),'sensitivity_results':sensitivity_results,'seconds':time.monotonic()-START,'limits':['Development per-case adapter predictions are not stored; audited their metric aggregation and choices, while identity metrics were independently linked to Stage-A raw predictions.','Source/context freeze assertions checked in all development/final metadata; three final source models independently reloaded and source predictions replayed without fitting. No saved adapter contexts exist to independently replay all calibration fits.','Full-export sensitivity high scores do not establish physical band validity, geographic origins or calibration benefit; rare class has one final query specimen.','Current same-batch query data have prior research exposure; no claim of fresh external confirmation or independent batches.']}


if __name__=='__main__':
    try:result=main()
    except Exception as exc:
        result={'status':'fail','checks_evaluated':checks,'failed_checks':failures,'error':repr(exc),'maximum_numeric_difference':maximum,'seconds':time.monotonic()-START}
        OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');raise
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','checks_evaluated','development_rows','final_prediction_rows','final_cases','maximum_numeric_difference','seconds']},ensure_ascii=False))
