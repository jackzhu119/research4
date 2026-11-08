#!/usr/bin/env python3
"""Stream sealed legacy base predictions into a bounded, cache-backed readout.

Scientific jobs, original outputs and original CPU ledgers are read-only. This
tool does not summarize calibration, fit models, or launch scientific work.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
BRANCH = 'research/cache-aware-continuation-20261108'
TARGETS = {'suanzaoren': ('酸枣仁', 300), 'huangqi': ('黄芪', 241), 'tianma': ('天麻', 242)}
ARMS = ('A', 'Aextra', 'B', 'SVM_reference')
SEEDS = {42, 43, 44}
SUMMARY_CAP = 2 * 1024**2
FILE_CAP, DELTA_CAP = 8 * 1024**2, 64 * 1024**2


def need(test, message):
    if not test:
        raise ValueError(message)


def read(path, cap=32 * 1024**2):
    path = Path(path)
    need(path.stat().st_size <= cap, 'Bounded metadata limit exceeded')
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def inside(path, root):
    path = Path(path).resolve()
    need(path.is_relative_to(root), 'Artifact escaped its approved root')
    return path


def stamp(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'dev': s.st_dev, 'inode': s.st_ino}


def argument(job, name):
    argv = job['command']
    need(name in argv and argv.index(name) + 1 < len(argv), 'Scientific argument missing: ' + name)
    return argv[argv.index(name) + 1]


def load_context(root):
    root = Path(root).resolve()
    need(root == REPO, 'Only the independent runtime repository is allowed')
    original = read(root/'original_manifest.json')
    prior = read(root/'evidence/prior_inner_closed.json')
    remaining = read(root/'remaining_jobs.json') if (root/'remaining_jobs.json').exists() else {'jobs': []}
    current = read(root/'work/control/state.json') if (root/'work/control/state.json').exists() else {'jobs': {}, 'input_hashes': {}}
    jobs = original['jobs']
    need(len({j['id'] for j in jobs}) == len(jobs), 'Original job IDs are not unique')
    lookup = {j['id']: j for j in jobs}
    for job in remaining['jobs']:
        if job.get('kind') in ('neural', 'classical'):
            need(job['id'] in lookup and objhash(job) == objhash(lookup[job['id']]),
                 'A resumed scientific descriptor differs from the original')
    oldroot = Path(original['protocol_path']).resolve().parent
    return root, original, prior, remaining, current, oldroot


def expected_truth(original):
    neural = [j for j in original['jobs'] if j.get('kind') == 'neural']
    need(neural, 'No original neural job descriptors')
    paths = {argument(j, '--split') for j in neural}
    need(len(paths) == 1, 'Scientific jobs use different TRAIN pools')
    split = Path(next(iter(paths)))
    with split.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    need(len(rows) == 999 and all(r['role'] == 'train' for r in rows), 'Only the frozen999TRAIN pool is admissible')
    need(len({r['record_id'] for r in rows}) == 999 and len({r['physical_sample_id'] for r in rows}) == 999,
         'Independent specimen identity collision')
    result = {}
    for target, (species, count) in TARGETS.items():
        truth = {r['record_id']: r['origin_group_id'] for r in rows if r['species'] == species}
        need(len(truth) == count, 'Target TRAIN count changed')
        result[target] = truth
    return result


def closed_job(root, original_job, prior, current, oldroot, digest_cache):
    jid = original_job['id']
    old = prior['jobs'].get(jid, {})
    new = current['jobs'].get(jid, {})
    need(not (old.get('status') == 'completed' and new.get('status') == 'completed'),
         'Scientific job was completed twice across the two ledgers')
    if new.get('status') == 'completed':
        state, row, receiptroot = current, new, root/'work/control/receipts'
    elif old.get('status') == 'completed':
        state, row, receiptroot = prior, old, oldroot/'control/receipts'
    else:
        return None
    receiptpath = receiptroot/(jid+'.json')
    def digest(path):
        key = str(Path(path).resolve())
        if key not in digest_cache:
            digest_cache[key] = sha(path)
        return digest_cache[key]
    need(digest(receiptpath) == row['receipt_sha256'], 'Supervisor closure receipt changed')
    receipt = read(receiptpath)
    need(receipt['job_sha256'] == objhash(original_job), 'Supervisor receipt sealed a different scientific job')
    need(receipt['input_hashes'] == state['input_hashes'][jid], 'Supervisor input seal differs')
    for name, checksum in receipt['input_hashes'].items():
        need(digest(name) == checksum, 'Original frozen scientific input changed')
    out = Path(argument(original_job, '--output')).resolve()
    need(out.is_relative_to(oldroot), 'Scientific output escaped original run')
    completepath, summarypath = out/'complete.json', out/'summary.json'
    need(receipt['output_hashes'][str(completepath)] == digest(completepath), 'Science completion seal differs')
    need(receipt['output_hashes'][str(summarypath)] == digest(summarypath), 'Science summary seal differs')
    done = read(completepath)
    need(done.get('status') == 'complete', 'Scientific producer did not finish')
    artifacts = done.get('artifacts', done.get('artifacts_sha256', {}))
    pred = out/'validation_predictions.csv.gz'; truth = out/'validation_truth.json'
    expected_pred = receipt['output_hashes'][str(pred)]
    need(artifacts['validation_predictions.csv.gz'] == expected_pred, 'Prediction output seals disagree')
    need(artifacts['validation_truth.json'] == digest(truth), 'Validation truth seal changed')
    return {'out': out, 'receipt_path': receiptpath, 'receipt_sha256': row['receipt_sha256'],
            'complete_sha256': digest(completepath), 'summary_sha256': digest(summarypath),
            'prediction_sha256': expected_pred, 'truth_sha256': digest(truth),
            'ledger_source': 'continued' if state is current else 'prior'}


class HashingReader(io.RawIOBase):
    def __init__(self, path):
        self.handle = Path(path).open('rb'); self.hash = hashlib.sha256(); self.bytes = 0
    def readable(self): return True
    def readinto(self, buffer):
        n = self.handle.readinto(buffer)
        if n:
            self.hash.update(memoryview(buffer)[:n]); self.bytes += n
        return n
    def close(self):
        self.handle.close(); super().close()


def base_cache(root, job, proof, allowed_truth):
    jid = job['id']; need(re.fullmatch(r'[A-Za-z0-9_.-]+', jid), 'Unsafe cache identifier')
    pred, truthpath = proof['out']/'validation_predictions.csv.gz', proof['out']/'validation_truth.json'
    declaration = {'job_sha256': objhash(job), 'receipt_sha256': proof['receipt_sha256'],
        'prediction_sha256': proof['prediction_sha256'], 'truth_sha256': proof['truth_sha256'],
        'prediction_stamp': stamp(pred), 'truth_stamp': stamp(truthpath)}
    cachepath = root/'readout_cache'/(jid+'.json')
    if cachepath.exists():
        cached = read(cachepath, cap=512*1024)
        supplied = cached.pop('cache_sha256', None)
        need(supplied == objhash(cached), 'Compact base cache was altered')
        if cached['declaration'] == declaration:
            validate_base_rows(cached['rows'], cached['query_truth'], allowed_truth)
            return cached, cachepath, True
    truth = read(truthpath, cap=128*1024)
    need(truth and all(rid in allowed_truth and allowed_truth[rid] == label for rid, label in truth.items()),
         'Producer truth is outside the frozen target TRAIN pool')
    rows = []
    prefixes = tuple((arm+',source_only,-1,0,').encode() for arm in ARMS)
    before = stamp(pred)
    with HashingReader(pred) as raw, io.BufferedReader(raw, buffer_size=1024**2) as buffered:
        with gzip.GzipFile(fileobj=buffered, mode='rb') as gz:
            header = next(csv.reader([gz.readline().decode('utf-8-sig')]))
            need(header[:6] == ['arm','method','repeat','shots','record_id','predicted'], 'Unexpected frozen prediction schema')
            for line in gz:
                if not line.startswith(prefixes):
                    continue
                values = next(csv.reader([line.decode('utf-8')]))
                need(len(values) == len(header), 'Prediction column count differs')
                arm, method, repeat, shots, rid, predicted = values[:6]
                need(method == 'source_only' and repeat == '-1' and shots == '0', 'Base filter failed')
                rows.append([arm, rid, truth.get(rid), predicted])
        need(raw.hash.hexdigest() == proof['prediction_sha256'] and raw.bytes == before['bytes'],
             'Streamed prediction hash/size differs from producer closure')
    need(stamp(pred) == before == declaration['prediction_stamp'], 'Closed prediction changed during scan')
    validate_base_rows(rows, truth, allowed_truth)
    result = {'schema': 'research4-base-prediction-cache-v1', 'declaration': declaration,
        'query_truth': truth, 'rows': rows, 'scope': 'producer-derived source_only rows; calibration is intentionally not summarized'}
    sealed = {**result, 'cache_sha256': objhash(result)}
    if cachepath.exists():
        preserved = cachepath.parent/'superseded'; preserved.mkdir(parents=True, exist_ok=True)
        cachepath.rename(preserved/(jid+'.'+str(time.time_ns())+'.json'))
    atomic(cachepath, sealed)
    return result, cachepath, False


def validate_base_rows(rows, truth, allowed):
    need(truth and all(rid in allowed and allowed[rid] == label for rid, label in truth.items()), 'Base truth mismatch')
    seen = {arm: set() for arm in ARMS}; classes = set(allowed.values())
    for row in rows:
        need(isinstance(row, list) and len(row) == 4, 'Invalid compact prediction row')
        arm, rid, actual, predicted = row
        need(arm in ARMS and rid in truth and actual == truth[rid] and predicted in classes, 'Base label/arm mismatch')
        need(rid not in seen[arm], 'Duplicate query sample in an arm')
        seen[arm].add(rid)
    need(all(value == set(truth) for value in seen.values()), 'Base arm lacks exact query coverage')


def metrics(confusion, labels):
    n = sum(confusion.values()); correct = sum(confusion[(x,x)] for x in labels)
    per = []
    for label in labels:
        support = sum(v for (a,b),v in confusion.items() if a == label)
        predicted = sum(v for (a,b),v in confusion.items() if b == label)
        tp = confusion[(label,label)]
        per.append({'label': label, 'support': support, 'precision': tp/predicted if predicted else 0.,
            'recall': tp/support if support else 0., 'f1': 2*tp/(support+predicted) if support+predicted else 0.})
    return {'correct': correct, 'denominator': n, 'accuracy': correct/n if n else 0.,
        'macro_f1': sum(x['f1'] for x in per)/len(per), 'balanced_accuracy': sum(x['recall'] for x in per)/len(per),
        'class_order': labels, 'confusion_matrix': [[confusion[(a,b)] for b in labels] for a in labels], 'per_class': per}


def aggregate(groups, truth_by_target):
    result = []
    for key, by_seed in sorted(groups.items()):
        target, profile, steps, arm = key; truth = truth_by_target[target]
        reps = []; full = []; perclass = []
        for seed, folds in sorted(by_seed.items()):
            pooled = {}; overlap = False
            for fold, records in folds.items():
                for rid, pair in records.items():
                    if rid in pooled: overlap = True
                    pooled[rid] = pair
            complete = set(folds) == set(range(5)) and set(pooled) == set(truth) and not overlap
            m = metrics(Counter(pooled.values()), sorted(set(truth.values()))) if complete else None
            if complete:
                full.append(m); perclass.append(m['per_class'])
            reps.append({'seed': seed, 'status': 'complete' if complete else 'partial', 'folds': sorted(folds),
                'unique_specimens': len(pooled), 'metrics': {k:m[k] for k in ['correct','denominator','accuracy','macro_f1','balanced_accuracy']} if m else None})
        completed = {x['seed'] for x in reps if x['status'] == 'complete'}
        mean = {k:sum(x[k] for x in full)/len(full) for k in ['accuracy','macro_f1','balanced_accuracy']} if full else {}
        per = [{**{k: row[k] for k in ['label','support']}, **{k:sum(p[i][k] for p in perclass)/len(perclass)
            for k in ['precision','recall','f1']}} for i,row in enumerate(perclass[0])] if perclass else []
        result.append({'target': target, 'profile': profile, 'pretrain_steps': steps, 'arm': arm,
            'status': 'complete' if completed == SEEDS else 'partial', 'complete_seeds': sorted(completed),
            'expected_seeds': sorted(SEEDS), 'independent_specimens': len(truth), 'mean_metrics': mean,
            'per_seed': reps, 'mean_per_class': per})
    return result


def summarize(root, output):
    root, original, prior, remaining, current, oldroot = load_context(root)
    output = inside(output, root); output.mkdir(parents=True, exist_ok=False)
    truth = expected_truth(original); digest_cache = {}; groups = defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    counts = Counter(); cache_hits = cache_misses = 0; closed_ids = []
    for job in original['jobs']:
        if job.get('kind') == 'classical':
            counts['classical_retained_completed' if prior['jobs'].get(job['id'],{}).get('status') == 'completed' else 'classical_unresolved'] += 1
            continue
        if job.get('kind') != 'neural': continue
        proof = closed_job(root, job, prior, current, oldroot, digest_cache)
        if proof is None:
            status = current.get('jobs',{}).get(job['id'],{}).get('status',prior['jobs'].get(job['id'],{}).get('status','not_run'))
            counts['neural_'+status] += 1; continue
        target = argument(job,'--target'); profile = argument(job,'--profile')
        steps = int(argument(job,'--pretrain-steps')); seed = int(argument(job,'--seed')); fold = int(argument(job,'--outer-fold'))
        need(target in TARGETS and seed in SEEDS and 0 <= fold < 5, 'Unknown original neural configuration')
        cached, cachepath, hit = base_cache(root, job, proof, truth[target])
        cache_hits += int(hit); cache_misses += int(not hit)
        for arm, rid, actual, predicted in cached['rows']:
            cell = groups[(target,profile,steps,arm)][seed][fold]
            need(rid not in cell, 'OOF record duplicated within a frozen fold')
            cell[rid] = (actual,predicted)
        counts['neural_completed'] += 1
        closed_ids.append({'job_id':job['id'],'receipt_sha256':proof['receipt_sha256'],
                           'ledger_source':proof['ledger_source'],'base_cache_sha256':sha(cachepath)})
    results = aggregate(groups, truth); paired = []
    by_config = defaultdict(dict)
    for row in results:
        if row['status'] == 'complete': by_config[(row['target'],row['profile'],row['pretrain_steps'])][row['arm']] = row
    for key, arms in sorted(by_config.items()):
        if 'B' not in arms: continue
        for control in ('A','Aextra','SVM_reference'):
            if control in arms:
                paired.append({'target':key[0],'profile':key[1],'pretrain_steps':key[2],'comparison':'B-'+control,
                    'delta':{metric:arms['B']['mean_metrics'][metric]-arms[control]['mean_metrics'][metric]
                             for metric in ['accuracy','macro_f1','balanced_accuracy']}})
    report = {'schema':'research4-cache-aware-base-readout-v1','scope':'TRAIN_only_base_predictions_not_calibration_or_external_confirmation',
        'run_counts':dict(counts),'cache':{'hits':cache_hits,'new_stream_scans':cache_misses,
            'policy':'First scan hashes full compressed producer predictions; later reuse requires sealed cache, source receipt SHA and unchanged size/mtime/ctime/inode.'},
        'independent_specimens':{k:len(v) for k,v in truth.items()},
        'repeat_policy':'Exact fivefold specimen coverage per seed; then mean across all three seeds. Repeated predictions are not extra specimens.',
        'classic_policy':'Retained original completed classical jobs; no new classical metric recomputation in this neural-base readout.',
        'calibration_policy':'NOT_SUMMARIZED; all calibration records remain in the original compressed scientific outputs.',
        'closed_neural_jobs':closed_ids,'results':results,'paired_transfer_differences':paired}
    text = json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    need(len(text.encode()) < SUMMARY_CAP,'Refuse to emit an oversized repeated-result JSON')
    (output/'summary.json').write_text(text,encoding='utf-8')
    columns = ['target','profile','pretrain_steps','arm','status','independent_specimens','accuracy','macro_f1','balanced_accuracy']
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in results:writer.writerow({k:row.get(k,row['mean_metrics'].get(k,'')) for k in columns})
    lines = ['# Compact retained neural-base evidence','',f'Closed neural jobs: {counts["neural_completed"]}. '
        f'Cache hits: {cache_hits}; new one-pass stream scans: {cache_misses}.','',
        'Only source_only, zero-shot predictions are summarized. Calibration is not evaluated by this report.',
        'These are previously exposed TRAIN development samples, not external or independent-batch confirmation.','',
        '| Target | Profile | Pretraining | Arm | Accuracy | Macro-F1 | Complete seeds | Status |',
        '|---|---|---:|---|---:|---:|---|---|']
    for row in results:
        m=row['mean_metrics'];a=f'{100*m["accuracy"]:.2f}%' if m else '—';f=f'{m["macro_f1"]:.4f}' if m else '—'
        lines.append(f'| {row["target"]} | {row["profile"]} | {row["pretrain_steps"]} | {row["arm"]} | {a} | {f} | {row["complete_seeds"]} | {row["status"]} |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_new(output/'complete.json',{'status':'complete','artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='complete.json'}})
    return {'status':'DONE_COMPACT_BASE_READOUT','jobs':dict(counts),'cache_hits':cache_hits,'new_stream_scans':cache_misses,'groups':len(results)}


def git(root, *args):
    result = subprocess.run(['git',*args],cwd=root,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,timeout=120,shell=False)
    need(result.returncode == 0,'Bounded Git operation failed: '+args[0])
    return result.stdout


def remote_allowed(value):
    if value.startswith('git@github.com:'):return value.removeprefix('git@github.com:').removesuffix('.git')=='jackzhu119/research4'
    p=urlsplit(value)
    return p.scheme in ('https','ssh') and p.hostname=='github.com' and p.path.strip('/').removesuffix('.git')=='jackzhu119/research4'


def publish(root, output, execute=False):
    root, original, prior, remaining, current, oldroot = load_context(root)
    output = inside(output,root);output.mkdir(parents=True,exist_ok=False)
    selected={};omitted={};started=datetime.now(timezone.utc).isoformat()
    try:
        def copy_candidate(source, relative, expected, priority):
            source=Path(source);need(source.is_file() and sha(source)==expected,'Closed publication source differs')
            name=str(relative);target=inside(root/relative,root)
            if source.stat().st_size >= FILE_CAP:
                omitted[str(source)]={'sha256':expected,'bytes':source.stat().st_size,'reason':'file_cap_local_only'};return
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():need(sha(target)==expected,'Immutable runtime artifact changed')
            else:shutil.copyfile(source,target)
            selected[name]={'sha256':expected,'bytes':target.stat().st_size,'priority':priority}
        latest=None
        for job in remaining['jobs']:
            if job.get('kind') in ('summary','compact_summary','readout') and current.get('jobs',{}).get(job['id'],{}).get('status')=='completed':latest=job['id']
        for job in remaining['jobs']:
            record=current.get('jobs',{}).get(job['id'],{})
            if record.get('status')!='completed' or job.get('kind') in ('backup','publish'):continue
            receiptpath=root/'work/control/receipts'/(job['id']+'.json');need(sha(receiptpath)==record['receipt_sha256'],'New publication receipt changed')
            receipt=read(receiptpath);need(receipt['job_sha256']==objhash(job),'New publication descriptor differs')
            priority=0 if job['id']==latest else 2
            copy_candidate(receiptpath,receiptpath.relative_to(root),record['receipt_sha256'],priority)
            if job.get('kind')=='neural':
                proof=closed_job(root,job,prior,current,oldroot,{})
                for name in ('summary.json','complete.json','validation_truth.json'):
                    path=proof['out']/name
                    expected=proof['summary_sha256'] if name=='summary.json' else proof['complete_sha256'] if name=='complete.json' else proof['truth_sha256']
                    copy_candidate(path,Path('published_science')/job['id']/name,expected,3)
                pred=proof['out']/'validation_predictions.csv.gz'
                omitted[str(pred)]={'sha256':proof['prediction_sha256'],'bytes':pred.stat().st_size,'reason':'full_calibration_predictions_retained_original_local_only'}
                cache=root/'readout_cache'/(job['id']+'.json')
                if cache.exists():
                    data=read(cache,cap=512*1024);seal=data.pop('cache_sha256',None)
                    need(seal==objhash(data) and data['declaration']['receipt_sha256']==proof['receipt_sha256'],'Base cache publication seal differs')
                    checksum=sha(cache)
                    copy_candidate(cache,Path('published_base')/job['id']/(checksum+'.json'),checksum,3)
            else:
                for name,expected in receipt['output_hashes'].items():
                    path=inside(name,root)
                    copy_candidate(path,path.relative_to(root),expected,priority)
                    if path.name=='complete.json':
                        for relative,checksum in read(path).get('artifacts',{}).items():
                            child=inside(path.parent/relative,root)
                            copy_candidate(child,child.relative_to(root),checksum,priority)
        # Initial readouts are root-sealed before continuation begins. Their
        # completion file is the explicit local derivative seal, not a claimed
        # producer-supervisor receipt.
        for done in sorted((root/'reports').glob('*/complete.json')):
            for relative,checksum in read(done).get('artifacts',{}).items():
                p=inside(done.parent/relative,root);copy_candidate(p,p.relative_to(root),checksum,1)
            copy_candidate(done,done.relative_to(root),sha(done),1)
        need(git(root,'branch','--show-current').decode().strip()==BRANCH,'Dedicated runtime branch required')
        need(remote_allowed(git(root,'remote','get-url','origin').decode().strip()),'Remote repository mismatch')
        need(not git(root,'diff','--cached','--name-only','-z'),'Unknown index content; leave untouched')
        before=git(root,'rev-parse','HEAD').decode().strip();algorithm=git(root,'rev-parse','--show-object-format').decode().strip()
        tracked={}
        for raw in git(root,'ls-tree','-rz','HEAD').split(b'\0'):
            if raw:
                meta,name=raw.split(b'\t',1);tracked[name.decode()]=meta.split()[2].decode()
        chosen={};existing={};total=0
        for name,info in sorted(selected.items(),key=lambda p:(p[1]['priority'],p[0])):
            content=(root/name).read_bytes();blob=hashlib.new(algorithm,('blob %d\0'%len(content)).encode()+content).hexdigest()
            if name in tracked:
                need(tracked[name]==blob,'Committed runtime artifact changed');existing[name]={**info,'git_blob':blob};continue
            if total+info['bytes']>DELTA_CAP:
                omitted[name]={**info,'reason':'delta_cap_local_only'};continue
            chosen[name]={**info,'git_blob':blob};total+=info['bytes']
        commit=None
        if execute:
            if chosen:
                need(git(root,'branch','--show-current').decode().strip()==BRANCH and git(root,'rev-parse','HEAD').decode().strip()==before,'Branch/HEAD changed before add')
                git(root,'add','--',*chosen)
                staged=set(git(root,'diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
                need(staged==set(chosen),'Unexpected staged files; no commit')
                need(git(root,'branch','--show-current').decode().strip()==BRANCH and git(root,'rev-parse','HEAD').decode().strip()==before,'Branch/HEAD changed before commit')
                git(root,'commit','-m','research: retain compact base evidence from unchanged continuation')
            commit=git(root,'rev-parse','HEAD').decode().strip()
            for name,info in {**chosen,**existing}.items():need(git(root,'rev-parse',commit+':'+name).decode().strip()==info['git_blob'],'Committed runtime blob differs')
            need(git(root,'branch','--show-current').decode().strip()==BRANCH and git(root,'rev-parse','HEAD').decode().strip()==commit,'Branch/HEAD changed before push')
            git(root,'push','origin',commit+':refs/heads/'+BRANCH)
            remote=git(root,'ls-remote','origin','refs/heads/'+BRANCH).decode().split()
            need(len(remote)==2 and remote[0]==commit,'Remote runtime commit not verified')
        result={'status':'DONE' if execute else 'NOT_RUN','verified_remote_commit':commit,'new':chosen,
            'already_committed_files':len(existing),'not_backed_up':omitted,'delta_bytes':total,
            'scope':'Only listed sealed runtime derivatives and small continued science records; calibration prediction archives stay original.'}
    except Exception as error:
        result={'status':'NOT_BACKED_UP','error_type':type(error).__name__,'detail':str(error) if isinstance(error,ValueError) else 'Bounded operation failed',
                'scientific_retry_requested':False,'remote_verified':False}
    result.update(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),branch=BRANCH)
    write_new(output/'receipt.json',result)
    write_new(output/'complete.json',{'status':'complete','backup_status':result['status'],
        'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='complete.json'}})
    return {'status':result['status'],'receipt':str(output/'receipt.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['summarize','publish'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    value=summarize(args.root,args.output) if args.action=='summarize' else publish(args.root,args.output,args.execute)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
