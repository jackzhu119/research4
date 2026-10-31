"""Copy closed auxiliary evidence, with byte-checked model archives. No training."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,shutil,tarfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def jread(p):return json.loads(Path(p).read_text())
def jwrite(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def copy_same(source,dest):
    if dest.exists():
        if sha(source)!=sha(dest):raise RuntimeError('Refuse to replace closed evidence '+str(dest))
    else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
def archive_models(out,dest,stage):
    inventory=[]
    for dataset in ['huangqi','tianma']:
        for fold in range(5):
            roots=sorted((out/stage).glob(dataset+f'.outer{fold:02d}.*'))
            files=sorted(p for r in roots for p in r.rglob('*') if p.is_file())
            expected=480 if stage=='selection' else 72
            if len(files)!=expected:raise RuntimeError(f'Incomplete {stage} archive {dataset}/{fold}: {len(files)}')
            p=dest/('source-models' if stage=='selection' else 'evaluation-models')/f'{dataset}_outer{fold:02d}.tar.gz';p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():raise FileExistsError(p)
            entries={str(f.relative_to(out)):sha(f) for f in files}
            with tarfile.open(p,'w:gz',compresslevel=3) as h:
                for f in files:h.add(f,arcname=str(f.relative_to(out)),recursive=False)
            checked=0
            with tarfile.open(p,'r:gz') as h:
                for member in h.getmembers():
                    if not member.isfile() or member.name not in entries:raise RuntimeError('Unexpected archive member')
                    blob=h.extractfile(member).read();assert hashlib.sha256(blob).hexdigest()==entries[member.name];checked+=1
            assert checked==len(entries)
            inventory.append({'archive':str(p.relative_to(dest)),'sha256':sha(p),'bytes':p.stat().st_size,'members_sha256':entries,'member_count':len(entries)})
    jwrite(dest/(stage+'_archive_manifest.json'),{'created_at_utc':datetime.now(timezone.utc).isoformat(),'original_output_prefix':str(out),'archives':inventory,'member_hash_checks':sum(x['member_count'] for x in inventory)})
def pack_scope(path,dest):
    with Path(path).open('rb') as source,gzip.open(dest,'wb',compresslevel=6) as target:shutil.copyfileobj(source,target)
    h=hashlib.sha256()
    with gzip.open(dest,'rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==sha(path)
    return {'path':str(dest),'compressed_sha256':sha(dest),'raw_sha256':h.hexdigest(),'raw_bytes':Path(path).stat().st_size}
def main(out,dest,phase):
    dest.mkdir(parents=True,exist_ok=True)
    if phase=='source':
        assert jread(out/'selection_lock.json')['inner_trials']==1200
        for name in ['candidate_manifest.json','run_manifest.json','protocol_snapshot.json','preparation.json','split_manifest.json','boundary_test_provenance.json','unit_tests.log','source_release_snapshot.json','inner_metrics.json','inner_predictions.csv.gz','selection_lock.json','source_training.log']:
            copy_same(out/name,dest/name)
        frozen=jread(out/'run_manifest.json')['source_sha256']
        for name,expected in frozen.items():
            assert sha(ROOT/name)==expected;copy_same(ROOT/name,dest/'source_snapshot'/name)
        copy_same(ROOT/'tests/test_auxiliary_multiplant.py',dest/'source_snapshot/tests/test_auxiliary_multiplant.py')
        prep=jread(out/'preparation.json')
        jwrite(dest/'prepared_cache_index.json',{'note':'Large derived matrices omitted from Git, not original raw data. Rebuild with frozen prepare in fresh directory; each matrix must match this SHA. Truth included separately only after prediction lock and scoring.','contexts':[{k:c[k] for k in ['split_id','stage','source_pack','source_pack_sha256','inference_pack','inference_pack_sha256','scoring_truth','scoring_truth_sha256']} for c in prep['contexts']]})
        archive_models(out,dest,'selection')
        scope=pack_scope(out/'fit_scope_events.jsonl',dest/'source_fit_scope_events.jsonl.gz');jwrite(dest/'source_scope_snapshot.json',scope)
    elif phase=='final':
        assert jread(out/'prediction_lock.json')['fitted_model_count']==180
        for p in out.iterdir():
            if p.is_file() and p.name!='fit_scope_events.jsonl':copy_same(p,dest/p.name)
        for p in (out/'scoring_truth').iterdir():copy_same(p,dest/'scoring_truth'/p.name)
        archive_models(out,dest,'evaluation')
        scope=pack_scope(out/'fit_scope_events.jsonl',dest/'fit_scope_events.jsonl.gz')
        prep=jread(out/'preparation.json');allowed={c['split_id']:set(c['source_ids']) for c in prep['contexts']};stages={c['split_id']:c['stage'] for c in prep['contexts']}
        start=collections.Counter();end=collections.Counter();kinds=collections.Counter();stage_counts=collections.Counter();rows=0
        with (out/'fit_scope_events.jsonl').open() as h:
            for line in h:
                r=json.loads(line);rows+=1;ids=set(r['fit_sample_ids']);assert ids and ids<=allowed[r['split_id']]
                key=(r['component'],r['split_id'],r['code_version'],r['data_fingerprint'],r['train_id_digest'])
                if r['event']=='fit_start':start[key]+=1;kinds[r['component']]+=1;stage_counts[stages[r['split_id']]]+=1
                elif r['event']=='fit_completed':end[key]+=1
                else:raise RuntimeError('Noncompleted actual fit event '+r['event'])
        assert start==end and stage_counts['inner']==39000
        jwrite(dest/'fit_scope_manifest.json',{'status':'TECHNICAL_CHECK_PASS','actual_successful_models':1380,'source_models':1200,'outer_models':180,'outer_alias_role_sets':30,'v2_failed_classifier_fits_additional':1,'v2_failed_run_learning_operations_additional':2,'v2_failure_archive':'source_inference_failure_v2/archive_manifest.json','unit_toy_fits_excluded':True,'learned_operations':sum(start.values()),'operations_by_stage':dict(stage_counts),'operations_by_component':dict(kinds),'start_end_rows':rows,'all_actual_fit_ids_inside_current_source':True,'starts_equal_completions':True,'source_only_standardization_is_counted_as_learning':True,'scientific_status':'EXPLORATORY_USER_DECLARED_INDEPENDENCE_BATCH_UNKNOWN','independent_confirmation':False,'OS_enforced_truth_isolation':False,**scope})
    else:raise ValueError(phase)
    print(json.dumps({'phase':phase,'destination':str(dest),'status':'TECHNICAL_CHECK_PASS'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--phase',choices=['source','final'],required=True);a=p.parse_args();main(a.output,a.destination,a.phase)
