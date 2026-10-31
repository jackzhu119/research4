"""Exact-byte R4 evidence archive and released outer-source model delivery."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tarfile


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/staged-nested-20261031'
RUN=BASE/'combinations'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    destination=RUN/'compact_archive';models=RUN/'deliverable_models'
    destination.mkdir(exist_ok=True);models.mkdir(exist_ok=True)
    status=json.loads((RUN/'status.json').read_text())
    assert status['status']=='completed'
    for relative,digest in status['result_sha256'].items():assert sha(RUN/relative)==digest
    for path in RUN.iterdir():
        if path.is_file() and path.suffix in ['.json','.jsonl','.csv','.gz']:
            shutil.copy2(path,destination/path.name)
    selected=[p for p in sorted((RUN/'selection').rglob('*')) if p.is_file() and p.suffix not in ['.joblib','.pt']]
    source={str(p.relative_to(RUN)):sha(p) for p in selected}
    archive=destination/'selection_evidence.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for p in selected:tar.add(p,arcname=str(p.relative_to(RUN)),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            stream=tar.extractfile(member);assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest()==source[member.name]
    original={};model_rows=[]
    for part in ['selection','evaluation']:
        for p in sorted((RUN/part).rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(RUN);original[str(relative)]={'sha256':sha(p),'bytes':p.stat().st_size}
            if part=='evaluation':
                target=models/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
                if p.suffix in ['.joblib','.pt']:
                    model_rows.append({'run_relative_path':str(relative),'sha256':sha(p),'bytes':p.stat().st_size})
                else:
                    q=destination/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for name in ['combinations_arithmetic_audit.py','combinations_arithmetic_audit_v1.py','cnn_source_audit.py','cnn_source_audit.json',
        'cnn_full_source_audit.py','cnn_full_source_audit.json','combinations-tests.log','combinations-selection.log',
        'combinations-evaluation.log','combinations_invocations.jsonl','evaluation_invocations.jsonl']:
        p=BASE/name
        if p.exists():shutil.copy2(p,destination/name)
    for name in ['src/herb_spectra/staged_nested_combinations.py','scripts/run_staged_nested_combinations.py',
        'tests/test_staged_nested_combinations.py','data/research/staged_nested_combinations_protocol_20261031.json',
        'src/herb_spectra/staged_nested_search.py','data/research/staged_nested_protocol_20261031.json']:
        p=ROOT/name;q=destination/'source_snapshot'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    shutil.copy2(__file__,destination/Path(__file__).name)
    write(destination/'original_run_files.json',original)
    write(models/'manifest.json',{'source_binary_files':model_rows,'outer_callers':5,'source_samples_each':684,
        'family_pipelines':35,'shared_CNN_training_invocations':5,'standalone_CNN_states':5,
        'source_pipeline_inference':'raw complete612 source-preprocessed spectra; SVC standard vote, CNN head logit argmax',
        'selection_policy':'each outer own selection_lock family; cannot pick highest outer diagnostic'})
    write(destination/'archive_validation.json',{'status':'pass','selection_tar_members':len(source),
        'all_original_nonbinary_selection_evidence_restorable_byte_exact':True,'outer_source_binary_files':len(model_rows),
        'cached_selection_binary_models_not_duplicated':True,'new_model_fits_in_export':0})
    for base in [destination,models]:
        index={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p!=base/'files_sha256.json'}
        write(base/'files_sha256.json',index)
        for name,digest in index.items():assert sha(base/name)==digest
    print(json.dumps({'archive':str(destination),'models':str(models),'selection_tar_members':len(source),'source_binaries':len(model_rows)}))


if __name__=='__main__':main()
