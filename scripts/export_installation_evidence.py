"""Export concise, source-backed evidence from a successful unified run."""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--runtime-dir',type=Path,default=Path('.runtime'))
    parser.add_argument('--unit-log',type=Path,default=Path('outputs/integration-unit-tests-final.log'))
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):
        parser.error('Choose an empty/new evidence output directory')
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run/'run_summary.json').read_text())
    if summary.get('selected_paths_passed') is not True or summary.get('exit_code')!=0:
        raise ValueError('Only successful selected-path runs may be exported as installation evidence')
    doctor=json.loads((args.run/'environment.json').read_text())
    if not doctor.get('open_source_ready'):raise ValueError('Environment is not ready')
    for filename in ['run_summary.json','RUN_REPORT.md','environment.json']:
        shutil.copyfile(args.run/filename,out/filename)
    shutil.copyfile(args.runtime_dir/'manifest.json',out/'environment_manifest.json')
    for name in summary['case_groups']:
        source=args.run/name
        if not source.exists():continue
        target=out/name;target.mkdir(exist_ok=True)
        for filename in ['results.json','versions.json','example_overlap.json']:
            if (source/filename).exists():shutil.copyfile(source/filename,target/filename)
        for path in source.glob('*.warnings.log'):shutil.copyfile(path,target/path.name)
        saved_rows=json.loads(json.dumps(summary['case_groups'][name]))
        for row in saved_rows:
            case=source/row['id']
            if not case.is_dir():continue
            case_target=target/row['id'];case_target.mkdir(exist_ok=True)
            for filename in ['check.m','adaptation.json','run.log']:
                path=case/filename
                if not path.exists():continue
                if filename=='run.log' and path.stat().st_size>256_000:
                    saved=filename+'.gz'
                    (case_target/saved).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
                else:
                    saved=filename;shutil.copyfile(path,case_target/saved)
                if filename=='run.log':row['saved_log']=str(Path(row['id'])/saved)
        (target/'results.json').write_text(json.dumps(saved_rows,ensure_ascii=False,indent=2)+'\n')
    inventory=args.run/'sources/source_inventory.json'
    if inventory.exists():shutil.copyfile(inventory,out/'source_inventory.json')
    log_dir=out/'logs';log_dir.mkdir(exist_ok=True)
    for step in summary['steps']:
        path=args.run/step['log']
        if path.exists():shutil.copyfile(path,log_dir/path.name)
    if 'ssa_full' in summary['case_groups']:
        for path in (args.run/'ssa_full').rglob('run.log'):
            shutil.copyfile(path,log_dir/'ssa_full_complete.log')
    if args.unit_log.exists():shutil.copyfile(args.unit_log,out/'unit_tests.log')
    rebuild=Path('outputs/bootstrap-rebuild')
    for filename in ['manifest.json','doctor.json']:
        if (rebuild/filename).exists():shutil.copyfile(rebuild/filename,out/('rebuilt_octave_'+filename))
    cache=Path('outputs/ssa-cache-revalidation.json')
    if cache.exists():shutil.copyfile(cache,out/'historical_ssa_cache_revalidation.json')
    rows=[]
    for name,values in summary['case_groups'].items():
        if name=='original_diagnostics':continue
        if not values or any(r.get('status')!='pass' or r.get('stage')=='isolated_defect' for r in values):
            raise ValueError(f'Unsuccessful or diagnostic cases in readiness group: {name}')
        rows.extend(values)
    text=args.unit_log.read_text() if args.unit_log.exists() else ''
    match=re.search(r'Ran (\d+) tests?',text)
    result={'exported_at_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'open_source_ready':True,'live_runnable_checks':len(rows),
            'unit_tests_latest':int(match.group(1)) if match else None,
            'cached_full_cases':len(summary.get('cached_cases',[])),
            'fresh_full_ssa_executed':'ssa_full' in summary['case_groups'],
            'matlab_native_execution_verified':False,'all_original_files_unmodified_runnable':False,
            'research_performance_verified':False,
            'scope':'main algorithm families via explicit compatibility/correction/reimplementation paths; not every archived helper'}
    (out/'installation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    hashes=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='files_sha256.json']
    (out/'files_sha256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
