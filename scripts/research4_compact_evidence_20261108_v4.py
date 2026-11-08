"""Bounded TRAIN-only OOF readout and serial, sealed-result publication.

No model is fitted here. Only completed jobs with supervisor/output seals enter
metrics. Repeated predictions are not additional independent specimens.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'data/research/evidence_validation_20261108_v4'
BRANCH = 'research/evidence-driven-validation-90000-20261108'
RUN = 'evidence-validation-20261108-v4'
SCIENCE = {'matched_classical', 'neural'}
FILE_CAP, DELTA_CAP = 10*1024**2, 64*1024**2
SUMMARY_CAP = 2*1024**2


def need(test, message):
    if not test:
        raise ValueError(message)


def read(path, cap=16*1024**2):
    path=Path(path);need(path.stat().st_size<=cap,'Metadata exceeds bounded size')
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path, value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    Path(path).write_text(text,encoding='utf-8')


def contained(path, root):
    path=Path(path).resolve();need(path.is_relative_to(root),'Path outside approved run')
    return path


def context(root):
    root=Path(root).resolve();need(root==BASE,'Exact new run root required')
    manifest=read(root/'jobs.json');protocol=read(root/'protocol.json')
    lock=read(root/'input_lock.json');state=read(root/'control/state.json',cap=48*1024**2)
    need(manifest['run_id']==RUN and manifest['cpu_budget_seconds']==88000,'Finite new grant required')
    need(manifest['protocol_sha256']==sha(root/'protocol.json'),'Protocol changed')
    need(protocol['input_lock_sha256']==sha(root/'input_lock.json'),'Input lock changed')
    pins=protocol['worker_code_sha256']
    need(pins[str(Path(__file__).resolve().relative_to(REPO))]==sha(__file__),'Readout/publisher source changed')
    jobs=[]
    for j in manifest['jobs']:
        row=dict(j);row['cwd']=str(Path(row.get('cwd',manifest.get('cwd',str(REPO)))).resolve());jobs.append(row)
    frozen={'manifest':manifest,'normalized_jobs':jobs,'budget_cpu_seconds':88000.}
    need(state['manifest_sha256']==object_hash(frozen),'Supervisor manifest seal changed')
    need(sha(root/'train_only_features.npz')==lock['training_feature_sha256'] and
         sha(root/'train_only_split.csv')==lock['training_split_sha256'],'TRAIN pack changed')
    return manifest,protocol,lock,state,jobs


def closed(root, job, state, cache=None):
    cache={}if cache is None else cache
    def digest(path):
        key=str(Path(path).resolve())
        if key not in cache:cache[key]=sha(path)
        return cache[key]
    record=state['jobs'].get(job['id'],{})
    if record.get('status')!='completed':return None
    path=root/'control/receipts'/(job['id']+'.json');receipt=read(path)
    need(digest(path)==record['receipt_sha256'] and receipt['job_sha256']==object_hash(job),'Closed job receipt differs')
    need(receipt['input_hashes']==state['input_hashes'][job['id']],'Closed job input seal differs')
    for name,checksum in receipt['input_hashes'].items():
        p=Path(name).resolve();need(p.is_relative_to(REPO) and digest(p)==checksum,'Pinned job input changed')
    files={str(path):digest(path)}
    if receipt.get('log_path'):
        files[receipt['log_path']]=receipt['log_sha256']
    for name,checksum in receipt['output_hashes'].items():
        p=contained(name,root);need(p.is_file() and digest(p)==checksum,'Closed output differs')
        files[str(p)]=checksum
        if p.name=='complete.json':
            done=read(p)
            for relative,expected in done.get('artifacts_sha256',done.get('artifacts',{})).items():
                child=contained(p.parent/relative,root)
                # The sealed complete.json retains the full cache inventory.
                # Compact readouts need only published scientific outputs,
                # not each transform checkpoint or fitted model payload.
                if child.name not in {'summary.json','validation_predictions.csv.gz','predictions.csv.gz',
                    'selected_features.csv.gz','inner_search.csv.gz','query_truth.json',
                    'job_manifest.json','fit_lock.json','prediction_lock.json','REPORT.md','metrics.csv'}:continue
                need(child.is_file() and digest(child)==expected,'Completion artifact differs')
                files[str(child)]=expected
    for name,checksum in files.items():
        p=contained(name,root);need(p.is_file() and digest(p)==checksum,'Closed artifact differs')
    return receipt,files


def metrics(confusion):
    labels=sorted({x for pair in confusion for x in pair})
    n=sum(confusion.values());correct=sum(confusion[(x,x)] for x in labels)
    per=[]
    for label in labels:
        support=sum(v for (a,b),v in confusion.items()if a==label)
        predicted=sum(v for (a,b),v in confusion.items()if b==label)
        tp=confusion[(label,label)]
        precision=tp/predicted if predicted else 0.;recall=tp/support if support else 0.
        f1=2*tp/(support+predicted)if support+predicted else 0.
        per.append({'label':label,'support':support,'precision':precision,'recall':recall,'f1':f1})
    return {'correct':correct,'denominator':n,'accuracy':correct/n if n else 0.,
        'macro_f1':sum(x['f1']for x in per)/len(per)if per else 0.,
        'balanced_accuracy':sum(x['recall']for x in per)/len(per)if per else 0.,
        'class_order':labels,'confusion_matrix':[[confusion[(a,b)]for b in labels]for a in labels],
        'per_class':per}


def field(row, summary, names, default=None):
    for source in (row,summary,summary.get('config',{}),summary.get('configuration',{})):
        for key in names:
            if key in source and str(source[key])!='':return source[key]
    return default


def grouping(row, summary, kind):
    scope=str(field(row,summary,['selection_scope','selection'],'primary'))
    method=str(field(row,summary,['method','family','model'],'NN'if kind=='neural'else'unknown'))
    need(method!='unknown','Prediction method absent')
    representation=str(field(row,summary,['representation'],'alpha'))
    if kind=='matched_classical':
        if method.startswith(('concat_','block_','weighted_')):representation='n_alpha'
        elif method.startswith('n_'):representation='n'
    return {'kind':kind,'method':method,'selection_scope':scope,
        'k':str(field(row,summary,['k','fixed_k','n_features'],'selected'if scope=='primary'else'')),
        'representation':representation,
        'source_count':int(field(row,summary,['source_count'],240 if kind=='matched_classical' else 190)),
        'auxiliary':str(field(row,summary,['auxiliary','auxiliary_mode'],'')),
        'target_epochs':int(field(row,summary,['target_epochs_actual','target_epochs'],192)) if kind=='neural' else 0,
        'calibration':str(field(row,summary,['calibration','setting'],'none')),
        'shots_per_class':int(field(row,summary,['shots_per_class','shots'],0))}


def truth_ids(root):
    with (root/'train_only_split.csv').open(encoding='utf-8-sig',newline='')as f:
        rows=list(csv.DictReader(f))
    need(len(rows)==999 and all(x['role']=='train'for x in rows),'Only999TRAIN records allowed')
    need(len({x['record_id']for x in rows})==999 and
         len({x['physical_sample_id']for x in rows})==999 and
         all(x['physical_sample_id']for x in rows),'Unique independent TRAIN specimen identities required')
    truth={r['record_id']:r['origin_group_id']for r in rows if r['species']=='酸枣仁'}
    need(len(truth)==300 and len(set(truth.values()))==5,'Pure-acid300TRAIN required')
    return truth


def aggregate(groups, expected_ids):
    result=[]
    for key, by_seed in sorted(groups.items()):
        descriptor=json.loads(key);expected_seeds={42,43,44}if descriptor['kind']=='neural'else{20261031,20261108,42}
        draws_by_seed=defaultdict(list)
        calibrated=descriptor['shots_per_class']>0
        expected_draws=set(range(20))if calibrated else{-1}
        for seed_draw,folds in sorted(by_seed.items()):
            seed,draw=seed_draw
            pooled={};invalid=[]
            for fold,predictions in folds.items():
                for rid, value in predictions.items():
                    if rid in pooled:invalid.append('duplicate_across_folds')
                    pooled[rid]=value
            complete=set(folds)==set(range(5)) and set(pooled)==expected_ids and not invalid
            m=metrics(Counter(pooled.values()))if complete else None
            draws_by_seed[seed].append({'draw':draw,'folds':sorted(folds),'unique_specimens':len(pooled),
                         'status':'complete'if complete else'partial','metrics':m})
        reps=[]
        for seed,draws in sorted(draws_by_seed.items()):
            full_draws={x['draw']for x in draws if x['status']=='complete'}
            complete=full_draws==expected_draws
            available=[x['metrics']for x in draws if x['metrics']is not None]
            if not calibrated:
                m=available[0]if available else None
            else:
                m={name:sum(x[name]for x in available)/len(available)for name in
                   ['accuracy','macro_f1','balanced_accuracy']}if available else None
                if m is not None:
                    m.update(denominator_per_support_draw=len(expected_ids),
                        correct_mean_over_support_draws=sum(x['correct']for x in available)/len(available),
                        support_draw_ranges={name:[min(x[name]for x in available),max(x[name]for x in available)]
                            for name in ['accuracy','macro_f1','balanced_accuracy']})
            reps.append({'seed':seed,'status':'complete'if complete else'partial',
                'expected_support_draws':len(expected_draws),'complete_support_draws':len(full_draws),
                'unique_specimens_per_support_draw':len(expected_ids),'metrics':m})
        full={r['seed']for r in reps if r['status']=='complete'}
        finished=full==expected_seeds
        available=[r['metrics']for r in reps if r['metrics']is not None]
        mean={name:sum(x[name]for x in available)/len(available)for name in
              ['accuracy','macro_f1','balanced_accuracy']}if available else {}
        result.append({**descriptor,'status':'complete'if finished else'partial',
            'eligible_for_development_selection':finished and descriptor['selection_scope']=='primary',
            'unique_specimens':len(expected_ids),'expected_repeats':sorted(expected_seeds),
            'complete_repeats':sorted(full),'mean_metrics':mean,'repeats':reps})
    return result


def paired_classical(groups, results, expected_ids):
    """Paired repeat predictions; uncertainty resamples physical specimens."""
    eligible={json.dumps({k:r[k]for k in ['kind','method','selection_scope','k','representation',
        'source_count','auxiliary','target_epochs','calibration','shots_per_class']},sort_keys=True,
        ensure_ascii=False,separators=(',',':')):r for r in results
        if r['kind']=='matched_classical'and r['eligible_for_development_selection']}
    baseline=[key for key,r in eligible.items()if r['method']=='alpha_snv20']
    if not baseline:return []
    basekey=baseline[0];base=groups[basekey]
    ids=sorted(expected_ids);seeds=(20261031,20261108,42);effects=[]
    def pooled(data,seed):
        value={}
        for fold in data[(seed,-1)].values():value.update(fold)
        need(set(value)==expected_ids,'Paired source coverage differs')
        return value
    before={seed:pooled(base,seed)for seed in seeds}
    for key,record in sorted(eligible.items()):
        if key==basekey:continue
        after={seed:pooled(groups[key],seed)for seed in seeds}
        corrected=damaged=0;per_id=[]
        for rid in ids:
            delta=0
            for seed in seeds:
                actual,pred=before[seed][rid];other_truth,other=after[seed][rid]
                need(actual==other_truth,'Paired labels differ')
                a=actual==pred;b=actual==other
                corrected+=int(not a and b);damaged+=int(a and not b);delta+=int(b)-int(a)
            per_id.append(delta/len(seeds))
        rng=random.Random(20261108)
        bootstrap=sorted(sum(per_id[i]for i in rng.choices(range(len(ids)),k=len(ids)))/len(ids)
                         for _ in range(500))
        effects.append({'baseline':'alpha_snv20','comparison':record['method'],
            'independent_physical_specimens':len(ids),'partition_repeats':len(seeds),
            'repeated_prediction_corrected':corrected,'repeated_prediction_damaged':damaged,
            'mean_accuracy_delta':sum(per_id)/len(ids),
            'cluster_bootstrap_accuracy_delta_95_interval':[bootstrap[12],bootstrap[487]],
            'bootstrap_draws':500,'bootstrap_seed':20261108,
            'uncertainty_unit':'physical_sample_id cluster retaining all three repeat predictions; exploratory, not external confirmation'})
    return effects


def summarize(root, output):
    root=Path(root).resolve();output=contained(output,root)
    manifest,protocol,lock,state,jobs=context(root);truth=truth_ids(root)
    output.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    counts=Counter();evidence=[];digest_cache={}
    for job in jobs:
        if job.get('kind')not in SCIENCE:continue
        record=state['jobs'].get(job['id'],{});status=record.get('status','not_run')
        if status!='completed':counts[status]+=1;continue
        receipt,files=closed(root,job,state,digest_cache)
        summaries=[Path(p)for p in files if Path(p).name=='summary.json']
        need(len(summaries)==1,'Exactly one sealed science summary required')
        summary=read(summaries[0],cap=2*1024**2)
        truth_paths=[Path(p)for p in files if Path(p).name=='query_truth.json']
        query_truth=read(truth_paths[0])if truth_paths else {}
        predictions=[Path(p)for p in files if Path(p).name.endswith('.csv.gz')and 'prediction'in Path(p).name]
        need(predictions,'Sealed science predictions required')
        # A worker may additionally seal search/feature CSVs. Only prediction
        # filenames enter this readout, and every row must identify its method.
        for path in predictions:
            with gzip.open(path,'rt',encoding='utf-8-sig',newline='')as f:
                for row in csv.DictReader(f):
                    rid=str(field(row,summary,['record_id']))
                    actual=str(field(row,summary,['y_true','truth','actual'],query_truth.get(rid)))
                    predicted=str(field(row,summary,['y_pred','prediction','predicted']))
                    need(rid in truth and actual==truth[rid]and predicted in set(truth.values()),'TRAIN identity or label mismatch')
                    descriptors=[grouping(row,summary,job['kind'])]
                    if job['kind']=='matched_classical':
                        primary=str(row.get('is_primary_k','')).lower()in('1','true')
                        if str(row.get('family','')).startswith('fullalpha_'):
                            descriptors[0].update(selection_scope='primary',k='full')
                        else:
                            descriptors[0].update(selection_scope='fixed_k',k=str(row['k']))
                            if primary:
                                descriptors.append({**descriptors[0],'selection_scope':'primary','k':'inner_selected'})
                    draw=int(field(row,summary,['support_repeat','calibration_repeat','draw','repeat'], -1))
                    seed=int(field(row,summary,['repeat_seed','seed']))
                    fold=int(field(row,summary,['outer_fold','fold']))
                    need(0<=fold<5,'Invalid outer fold')
                    for descriptor in descriptors:
                        key=json.dumps(descriptor,sort_keys=True,ensure_ascii=False,separators=(',',':'))
                        mapping=groups[key][(seed,draw)][fold]
                        need(rid not in mapping,'Duplicate sample in OOF cell')
                        mapping[rid]=(actual,predicted)
        counts['complete']+=1;evidence.append({'job_id':job['id'],'receipt_sha256':record['receipt_sha256']})
    result=aggregate(groups,set(truth))
    report={'schema':'research4-compact-evidence-v1','run_id':RUN,
        'scope':'TRAIN_only_exploratory_development_not_blind_or_cross_batch',
        'job_counts':dict(counts),'source_specimens':300,
        'repeat_policy':'Pool exact300 unique specimens per fivefold repeat; average complete repeats; no sample-count inflation.',
        'selection_policy':'Primary K/parameters chosen inside inner folds only. Fixed-K rows are ablations, not outer-selected candidates.',
        'support_policy':'All20 support draws require full fivefold coverage, then metrics averaged within model seed. Draws/repeated predictions are not new independent specimens.',
        'completed_job_receipts':evidence,'results':result,
        'paired_classical_effects':paired_classical(groups,result,set(truth))}
    encoded=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    need(len(encoded.encode())<=SUMMARY_CAP,'Compact summary exceeded2MiB; refuse giant expansion')
    (output/'summary.json').write_text(encoded,encoding='utf-8')
    columns=['kind','method','selection_scope','k','representation','source_count','auxiliary','target_epochs','calibration',
             'shots_per_class','status','eligible_for_development_selection','unique_specimens','accuracy','macro_f1','balanced_accuracy']
    with(output/'metrics.csv').open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in result:writer.writerow({k:row.get(k,row['mean_metrics'].get(k,''))for k in columns})
    lines=['# TRAIN-only exploratory evidence','',f'Completed scientific jobs: {counts["complete"]}. Independent acid specimens:300.',
        '','Only complete three-repeat/fivefold groups are eligible. Primary selection uses inner data; fixed-K scores are ablations.',
        '','| Family | Method | Scope/K | Representation | Source | Auxiliary | Accuracy | Macro-F1 | Status |',
        '|---|---|---|---|---:|---|---:|---:|---|']
    for row in result:
        m=row['mean_metrics'];a=f'{100*m["accuracy"]:.2f}%'if m else'—';f=f'{m["macro_f1"]:.4f}'if m else'—'
        lines.append(f'| {row["kind"]} | {row["method"]} | {row["selection_scope"]}/{row["k"]} | {row["representation"]} | {row["source_count"]} | {row["auxiliary"]} | {a} | {f} | {row["status"]} |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(output/'complete.json',{'status':'complete','artifacts':{p.name:sha(p)for p in output.iterdir()if p.is_file()and p.name!='complete.json'}})
    return {'status':'DONE_COMPACT_READOUT','scientific_jobs':dict(counts),'groups':len(result)}


def git(*args):
    r=subprocess.run(['git',*args],cwd=REPO,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,timeout=90,shell=False)
    need(r.returncode==0,'Bounded Git operation failed: '+args[0])
    return r.stdout


def allowed_remote(value):
    if value.startswith('git@github.com:'):return value.removeprefix('git@github.com:').removesuffix('.git')=='jackzhu119/research4'
    parsed=urlsplit(value)
    return parsed.scheme in('https','ssh')and parsed.hostname=='github.com'and parsed.path.strip('/').removesuffix('.git')=='jackzhu119/research4'


def select_delta(candidates, tracked, algorithm):
    chosen,existing,omitted={}, {}, {};total=0
    for name,info in sorted(candidates.items(),key=lambda item:(item[1]['priority'],item[0])):
        if info['bytes']>=FILE_CAP:
            omitted[name]={**info,'reason':'oversized_local_only_hash_preserved'};continue
        content=(REPO/name).read_bytes()
        blob=hashlib.new(algorithm,('blob %d\0'%len(content)).encode()+content).hexdigest()
        if name in tracked:
            need(tracked[name]==blob,'Committed immutable evidence changed');existing[name]={**info,'git_blob':blob};continue
        if total+info['bytes']>DELTA_CAP:
            omitted[name]={**info,'reason':'delta_cap_local_only_hash_preserved'};continue
        chosen[name]={**info,'git_blob':blob};total+=info['bytes']
    return chosen,existing,omitted,total


def backup(root, output, *, execute=False):
    root=Path(root).resolve();output=contained(output,root)
    output.mkdir(parents=True,exist_ok=False);started=datetime.now(timezone.utc).isoformat()
    selected={}
    try:
        manifest,protocol,lock,state,jobs=context(root)
        current=[j for j in jobs if j.get('kind')=='backup'and str(output/'receipt.json')in j['outputs']]
        need(len(current)==1 and current[0].get('max_retries')==0,'Declared nonretry backup required')
        need(not state.get('active')or state['active']['job_id']==current[0]['id'],'Only serial own backup may publish')
        def add(path, expected=None, priority=3):
            need(not Path(path).is_symlink(),'Symlink artifact excluded')
            p=contained(path,root);need(p.is_file(),'Sealed regular file required')
            need(p.name!='state.json'and p.suffix not in('.npz','.npy','.hdf','.chk','.xlsx'),'Live/raw/giant inputs excluded')
            need(not any(x in p.name for x in('.partial','.tmp','.lock')),'Unclosed file excluded')
            checksum=sha(p);need(expected is None or checksum==expected,'Selected artifact seal differs')
            name=str(p.relative_to(REPO));selected[name]={'sha256':checksum,'bytes':p.stat().st_size,
                'priority':min(priority,selected.get(name,{}).get('priority',priority))}
        for name in ['protocol.json','input_lock.json','jobs.json']:add(root/name,priority=1)
        summary_jobs=[j for j in jobs if j.get('kind')=='summary'and state['jobs'].get(j['id'],{}).get('status')=='completed']
        latest_summary=summary_jobs[-1]['id']if summary_jobs else None
        digest_cache={}
        for job in jobs:
            if job.get('kind')=='backup':continue
            seal=closed(root,job,state,digest_cache)
            if seal is None:continue
            receipt,files=seal
            for name,checksum in files.items():
                path=Path(name)
                priority=(0 if job['id']==latest_summary else 2 if job.get('kind')=='summary'else
                          3 if path.suffix in('.json','.md') else 4 if path.name.endswith('.csv.gz') else 5)
                add(path,checksum,priority)
            if job.get('kind')in SCIENCE:
                summary_paths=[Path(p)for p in files if Path(p).name=='summary.json']
                if summary_paths:
                    summary=read(summary_paths[0],cap=2*1024**2)
                    for model in summary.get('results',[]):
                        if model.get('is_primary_k')and model.get('model_path'):
                            add(summary_paths[0].parent/model['model_path'],model['model_sha256'],priority=3)
                    for reference in summary.get('model_receipts',{}).values():
                        cache=contained(reference['cache_path'],root);receipt_path=cache/'complete.json'
                        add(receipt_path,reference['complete_sha256'],priority=3)
                        for relative,checksum in read(receipt_path)['artifacts'].items():
                            model=contained(cache/relative,root)
                            if model.suffix in('.pt','.joblib','.json'):
                                add(model,checksum,priority=3)
        projection={'status':'IMMUTABLE_PROGRESS_PROJECTION_NOT_LIVE_LEDGER','at_utc':datetime.now(timezone.utc).isoformat(),
            'manifest_sha256':sha(root/'jobs.json'),'source_state_digest_at_read':object_hash(state),
            'queue_status':state['status'],'budget_cpu_seconds':state['budget_cpu_seconds'],
            'spent_cpu_seconds':state['spent_cpu_seconds'],'job_statuses':dict(Counter(x['status']for x in state['jobs'].values()))}
        write(output/'progress_snapshot.json',projection);add(output/'progress_snapshot.json',priority=0)
        need(git('branch','--show-current').decode().strip()==BRANCH,'Research branch mismatch')
        need(allowed_remote(git('remote','get-url','origin').decode().strip()),'Remote mismatch')
        need(not git('diff','--cached','--name-only','-z'),'Unknown staged data; leave untouched')
        before=git('rev-parse','HEAD').decode().strip();algorithm=git('rev-parse','--show-object-format').decode().strip()
        need(algorithm in('sha1','sha256'),'Git object algorithm unsupported')
        tracked={}
        for item in git('ls-tree','-rz','HEAD','--',str(BASE.relative_to(REPO))).split(b'\0'):
            if item:
                meta,name=item.split(b'\t',1);tracked[name.decode()]=meta.split()[2].decode()
        chosen,existing,omitted,total=select_delta(selected,tracked,algorithm)
        if execute:
            if chosen:
                need(git('branch','--show-current').decode().strip()==BRANCH and git('rev-parse','HEAD').decode().strip()==before,'Branch/HEAD changed before add')
                git('add','--',*chosen)
                staged=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
                need(staged==set(chosen),'Unexpected index content; no commit')
                need(git('branch','--show-current').decode().strip()==BRANCH and git('rev-parse','HEAD').decode().strip()==before,'Branch/HEAD changed before commit')
                git('commit','-m','research: preserve compact evidence validation results')
            commit=git('rev-parse','HEAD').decode().strip()
            for name,info in {**existing,**chosen}.items():need(git('rev-parse',commit+':'+name).decode().strip()==info['git_blob'],'Committed blob differs')
            need(git('branch','--show-current').decode().strip()==BRANCH and git('rev-parse','HEAD').decode().strip()==commit,'Branch/HEAD changed before push')
            git('push','origin',commit+':refs/heads/'+BRANCH)
            remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()
            need(len(remote)==2 and remote[0]==commit,'Remote commit not confirmed')
            status='DONE'
        else:status='NOT_RUN';commit=None
        result={'status':status,'verified_remote_commit':commit,'published_new':chosen,'already_committed':existing,
            'not_backed_up':omitted,'delta_bytes':total,'priority_policy':'Newest sealed readout before old evidence/predictions; preserve explicit omissions'}
    except Exception as error:
        result={'status':'NOT_BACKED_UP','failure_type':type(error).__name__,
            'detail':str(error)if isinstance(error,ValueError)else'Bounded operation failed',
            'scientific_retry_requested':False,'remote_verified':False}
    result.update(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),branch=BRANCH)
    write(output/'receipt.json',result)
    write(output/'complete.json',{'status':'complete','backup_status':result['status'],
        'artifacts':{p.name:sha(p)for p in output.iterdir()if p.is_file()and p.name!='complete.json'}})
    return {'status':result['status'],'receipt':str(output/'receipt.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['summarize','backup'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    result=summarize(args.root,args.output)if args.action=='summarize'else backup(args.root,args.output,execute=args.execute)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
