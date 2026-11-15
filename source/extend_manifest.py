#!/usr/bin/env python3
"""Generate frozen, score-independent experiment blocks; never read results."""
import argparse, fcntl, hashlib, json, os
from pathlib import Path

FAMILIES = ['baseline','selection','spacing','coverage','rank_context','retain_drop','amplitude','permutation','material']
MC_FAMILIES = {'selection','spacing','rank_context','retain_drop','permutation'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path, value):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,separators=(',',':')); f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def seed_for(repeat):
    return int.from_bytes(hashlib.sha256(f'thz-mechanism-20261110-repeat-{repeat}'.encode()).digest()[:4],'big') % (2**31-1)

def make_jobs(root,spec,repeat,draws,families):
    seed=seed_for(repeat)
    result=[]
    for family in families:
        if family not in spec['enabled_families']:continue
        for variant in spec['variants'][family]:
            once=spec.get('single_run_variants',[])
            if isinstance(once,dict):once=once.get(family,[])
            if repeat>0 and variant in once:continue
            folds=[0] if family=='material' else range(5)
            for fold in folds:
                config={'repeat':repeat,'seed':seed,'fold':fold,'family':family,'variant':variant,'mc_draws':draws if family in MC_FAMILIES else 20,'task':'material' if family=='material' else 'core300','data_path':str(root/'inputs/current1672_paired612.npz')}
                uid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
                config['id']=f'r{repeat:05d}_f{fold}_{family}_{uid}'
                result.append(config)
    return result

def extend(root,initial=False):
    root=Path(root).resolve(); spec=json.loads((root/'worker_spec.json').read_text())
    with (root/'manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'job_manifest.json'
        if path.exists():
            manifest=json.loads(path.read_text())
            assert manifest['worker_spec_sha256']==digest(root/'worker_spec.json')
            if initial:return manifest
        else:
            manifest={'schema':'thz-frozen-queue-v1','worker_spec_sha256':digest(root/'worker_spec.json'),'rule':'sha256 seed stream; fixed method order; initial repeats0..2 MC20 then 0..2 MC100 then repeat0 MC500; extensions 32 repeats MC20 plus every8th MC100/every32nd MC500; no score reads','blocks':0,'next_repeat':3,'jobs':[]}
        jobs=[]
        if not manifest['blocks']:
            for repeat in range(3):jobs += make_jobs(root,spec,repeat,20,FAMILIES)
            for repeat in range(3):jobs += make_jobs(root,spec,repeat,100,[f for f in FAMILIES if f in MC_FAMILIES and f!='permutation'])
            jobs += make_jobs(root,spec,0,500,[f for f in FAMILIES if f in MC_FAMILIES and f!='permutation'])
        else:
            first=manifest['next_repeat']
            for repeat in range(first,first+32):
                jobs += make_jobs(root,spec,repeat,20,FAMILIES)
                if repeat%8==0:jobs += make_jobs(root,spec,repeat,100,[f for f in FAMILIES if f in MC_FAMILIES and f!='permutation'])
                if repeat%32==0:jobs += make_jobs(root,spec,repeat,500,[f for f in FAMILIES if f in MC_FAMILIES and f!='permutation'])
            manifest['next_repeat']=first+32
        old_ids={j['id'] for j in manifest['jobs']}
        for job in jobs:
            if job['id'] not in old_ids:
                job['ordinal']=len(manifest['jobs']);manifest['jobs'].append(job);old_ids.add(job['id'])
        manifest['blocks']+=1
        atomic(path,manifest)
        print(json.dumps({'jobs':len(manifest['jobs']),'blocks':manifest['blocks'],'next_repeat':manifest['next_repeat']}))
        return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--initial',action='store_true');a=p.parse_args();extend(a.root,a.initial)
