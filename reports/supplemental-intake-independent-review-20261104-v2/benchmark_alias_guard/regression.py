"""Four tiny synthetic role-pack checks; no scientific input or model fitting."""
from pathlib import Path
import os,resource,sys,json,hashlib,importlib.util

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(int(1.5*1024**3),int(1.5*1024**3)))
import numpy as np
from herb_spectra.leakage_guard import LeakageError

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OLD=OUT.parent/'reviewed_benchmark_guard.py'
NEW=ROOT/'src/herb_spectra/supplemental_benchmark.py'
OLD_SHA='8d36fc545931fd9987d96f2e40bf48302493b7d07d825d512b6578cd622a306c'
NEW_SHA='a15605d7367121b262605dcdb6cad75ab4ca90c4bfd85c856a12317d3578046e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,digest,name):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def roles():
    rng=np.random.default_rng(20261104);axis=np.array([.1,.2,.3]);out={}
    for role in ('source','support','query'):
        p={'X':rng.normal(size=(2,3)),'axis':axis.copy(),
           'record_ids':np.array([role+'-1',role+'-2'],dtype='U80'),
           'entity_ids':np.array(['ent-'+role+'-1','ent-'+role+'-2'],dtype='U80'),
           'batch_ids':np.array(['user-declared-batch']*2),
           'legacy_match_ids':np.array(['[]','[]'],dtype='U160')}
        if role!='query':p['y']=np.array(['A','B'])
        out[role]=p
    return out
def check(module,packs):
    try:return {'accepted':True,'status':module.preflight_role_packs(packs)['status']}
    except LeakageError as exc:return {'accepted':False,'exception':'LeakageError','reason':str(exc)}
def main():
    assert not(OUT/'results.json').exists()
    old=load(OLD,OLD_SHA,'herb_spectra.review_before_physical_alias');new=load(NEW,NEW_SHA,'herb_spectra.review_after_physical_alias')
    results=[]
    for case in ('legacy_to_old_record_different_X','entity_to_record_different_X','indirect_three_role_alias_chain','legal_self_alias_clean_baseline'):
        p=roles()
        if case=='legacy_to_old_record_different_X':
            p['source']['record_ids'][0]='OLD-HQ03-S0001';p['source']['entity_ids'][0]='OLD-HQ03-S0001'
            p['query']['legacy_match_ids'][0]=json.dumps(['OLD-HQ03-S0001'])
        elif case=='entity_to_record_different_X':
            p['source']['entity_ids'][0]='SHARED-ENTITY';p['query']['record_ids'][0]='SHARED-ENTITY'
        elif case=='indirect_three_role_alias_chain':
            p['source']['record_ids'][0]='CHAIN-A';p['source']['entity_ids'][0]='CHAIN-B'
            p['support']['record_ids'][0]='CHAIN-C';p['support']['entity_ids'][0]='CHAIN-D';p['support']['legacy_match_ids'][0]=json.dumps(['CHAIN-B'])
            p['query']['record_ids'][0]='CHAIN-E';p['query']['entity_ids'][0]='CHAIN-F';p['query']['legacy_match_ids'][0]=json.dumps(['CHAIN-D'])
        else:
            for role,pack in p.items():
                pack['entity_ids']=pack['record_ids'].copy()
                pack['legacy_match_ids']=np.array([json.dumps([str(r),str(r)])for r in pack['record_ids']])
        before=check(old,p);after=check(new,p)
        assert before['accepted']is True
        assert after['accepted']is (case=='legal_self_alias_clean_baseline')
        assert not np.array_equal(p['source']['X'][0],p['query']['X'][0])
        results.append({'case':case,'before':before,'after':after,'source_query_X_differ':True})
    assert sha(OLD)==OLD_SHA and sha(NEW)==NEW_SHA
    report={'status':'PASS_FOUR_INDEPENDENT_SYNTHETIC_ALIAS_REGRESSIONS','old_source_sha256':OLD_SHA,'new_source_sha256':NEW_SHA,
        'cases':results,'old_alias_leaks_reproduced':3,'new_alias_leaks_rejected':3,'clean_self_alias_case_preserved':True,
        'scientific_array_or_workbook_reads':0,'scientific_or_synthetic_model_fits':0,'quantum_evaluations':0,
        'resource_AS_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'peak_RSS_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'regression_source_sha256':sha(__file__)}
    with (OUT/'results.json').open('x')as f:f.write(json.dumps(report,indent=2)+'\n')
    (OUT/'new_guard_source_snapshot.py').write_bytes(NEW.read_bytes())
    with (OUT/'index.json').open('x')as f:f.write(json.dumps({'status':'SEALED_INDEPENDENT_GUARD_REGRESSION','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file()and p.name!='index.json']},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'results_sha256':sha(OUT/'results.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
