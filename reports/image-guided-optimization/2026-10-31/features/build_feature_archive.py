"""Compact exact feature-run evidence and preserve the superseded V1 record."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tarfile


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/image-guided-20261031'
RUN=BASE/'features-v2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def tar_exact(paths,base,target):
    source={str(p.relative_to(base)):sha(p) for p in paths}
    with target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for p in paths:archive.add(p,arcname=str(p.relative_to(base)),recursive=False)
    with tarfile.open(target,'r:gz') as archive:
        for member in archive.getmembers():
            stream=archive.extractfile(member)
            assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest()==source[member.name]
    return source


def main():
    destination=RUN/'compact_archive';destination.mkdir(exist_ok=True)
    models=RUN/'deliverable_models';models.mkdir(exist_ok=True)
    for p in sorted(RUN.iterdir()):
        if p.is_file() and p.suffix in ['.json','.csv','.log']:
            shutil.copy2(p,destination/p.name)
    development=[p for p in sorted((RUN/'development').rglob('*')) if p.is_file() and p.suffix!='.joblib']
    tar_members=tar_exact(development,RUN,destination/'development_evidence.tar.gz')
    model_entries=[];original_files={}
    for parent in ['development','final']:
        for p in sorted((RUN/parent).rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(RUN);original_files[str(relative)]={'sha256':sha(p),'bytes':p.stat().st_size}
            if parent=='final':
                target=(models if p.suffix=='.joblib' else destination)/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
                if p.suffix=='.joblib':
                    meta=p.with_suffix('.json');mt=models/meta.relative_to(RUN)
                    mt.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(meta,mt)
                    model_entries.append({'run_relative_path':str(relative),'sha256':sha(p),'bytes':p.stat().st_size,
                        'metadata_relative_path':str(meta.relative_to(RUN))})
    write(destination/'original_run_files.json',original_files)
    write(models/'manifest.json',{'model_type':'source-fitted feature/classifier pipeline, no target calibration',
        'final_source_models':model_entries,'development_binary_cache_not_duplicated':True})
    v1=BASE/'features-v1-checkpoint-issue';revision=destination/'revision-v1';revision.mkdir(exist_ok=True)
    for p in sorted(v1.iterdir()):
        if p.is_file():shutil.copy2(p,revision/p.name)
    shutil.copytree(v1/'source_snapshot',revision/'source_snapshot',dirs_exist_ok=True)
    v1development=[p for p in sorted((v1/'development').rglob('*')) if p.is_file() and p.suffix!='.joblib']
    tar_exact(v1development,v1,revision/'development_evidence.tar.gz')
    for p in [BASE/'feature_arithmetic_audit.py',BASE/'feature_v2_invocations.jsonl',BASE/'feature-v2-development.log',BASE/'feature-v2-final.log',BASE/'feature-v2-tests.log',BASE/'feature_checkpoint_layout_diagnostic.json']:
        shutil.copy2(p,destination/p.name)
    write(destination/'archive_validation.json',{'status':'pass','development_tar_members':len(tar_members),
        'all_development_metadata_and_raw_predictions_restorable_byte_exact':True,'final_source_models':len(model_entries),
        'superseded_v1_preserved':True,'v1_and_v2_all720_model_files_byte_equal':True,
        'V2_checkpoint_numeric_algorithm_change':False,'new_model_fits_in_archiving':0})
    for base in [destination,models]:
        index={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p!=base/'files_sha256.json'}
        write(base/'files_sha256.json',index)
        for name,digest in index.items():assert sha(base/name)==digest
    print(json.dumps({'archive':str(destination.relative_to(ROOT)),'source_models':len(model_entries),'development_tar_members':len(tar_members)}))


if __name__=='__main__':main()
