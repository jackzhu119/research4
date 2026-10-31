"""Collect logs, harnesses, source hashes and numerical cross-checks for the report."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import gzip
import json
from pathlib import Path
import re
import shutil

import numpy as np
from scipy.io import loadmat


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,help='Source inventory; default is run-root/source_inventory.json')
    args=parser.parse_args();root=args.run_root;out=args.output
    out.mkdir(parents=True,exist_ok=True)
    phases=['first-pass','second-pass','focused','vectorized','final-pass','ssa-full','targeted','targeted-corrected','texture-array','python','python-cnn-corrected']
    cases=[]
    for phase in phases:
        result_file=root/phase/'results.json'
        if not result_file.exists():continue
        for result in json.loads(result_file.read_text()):
            row=dict(result,phase=phase)
            if phase=='focused' and row['id'].startswith('compat_ssa'):
                row['attribution']='superseded adapter regex error, not an original algorithm failure'
            if phase=='focused' and row['id'].startswith('fixture_text_'):
                row['attribution']='synthetic XLSX reader interaction; rerun with direct MAT input'
            if phase=='python' and row['id']=='python_cnn_structure_migration':
                row['attribution']='initial layout attempt; use python-cnn-corrected column-major run as final result'
            if phase=='targeted' and row['id']=='pca_centering_defect':
                row['attribution']='harness requested cluster assignments instead of centroids; corrected and rerun'
            if phase in {'targeted','targeted-corrected'} and row['id']=='texture_image_independence_defect':
                row['attribution']='PNG fixture decoded as grayscale, exposing original missing Gray branch; rerun with a true-color fixture'
            log_source=root/phase/row.get('log','') if row.get('log') else None
            if log_source and log_source.is_file():
                target=out/'logs'/phase/(row['id']+'.log');target.parent.mkdir(parents=True,exist_ok=True)
                if log_source.stat().st_size>1024**2:
                    compressed=target.with_suffix('.log.gz')
                    with log_source.open('rb') as src, compressed.open('wb') as dst:
                        with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0) as gz:
                            shutil.copyfileobj(src,gz)
                    raw=log_source.read_bytes()
                    digest=hashlib.sha256(raw).hexdigest()
                    excerpt=(f'EXCERPT: original log {len(raw)} bytes; full output in {compressed.name}; original SHA-256 {digest}\n\n').encode()+raw[:8192]+b'\n\n[... middle omitted; see compressed complete log ...]\n\n'+raw[-16384:]
                    target.write_bytes(excerpt)
                    row.update(full_log_gzip=str(compressed.relative_to(out)),full_log_bytes=len(raw),full_log_sha256=digest)
                else:
                    shutil.copyfile(log_source,target)
                row['saved_log']=str(target.relative_to(out))
                log=log_source.read_text();matches=re.findall(r'METRICS_JSON=([^\n]+)',log)
                if matches:row['metrics']=json.loads(matches[-1])
                match=re.search(r'交叉验准确率([0-9.]+)',log)
                if match:row['metrics']['tuning_reused_cv_accuracy_percent']=float(match.group(1))
                match=re.search(r'rightridio\s*=\s*\n\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',log)
                if match:row['metrics']['example_class_recall_displayed']=[float(x) for x in match.groups()]
                if row['status']=='timeout' and 'ssa' in row['id']:
                    iterations=[int(x) for x in re.findall(r'^([0-9]+)$',log,re.M)]
                    if iterations:row['metrics']['last_completed_iteration']=max(iterations)
                for filename in ['check.m','adaptation.json']:
                    source=root/phase/row['id']/filename
                    if source.exists():
                        dest=out/'harnesses'/phase/row['id']/filename
                        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            cases.append(row)
    if (root/'python/unit_tests.log').exists():
        shutil.copyfile(root/'python/unit_tests.log',out/'python_unit_tests.log')
    for name in ['octave-environment.log','example_overlap.json']:
        source=root/name
        if not source.exists() and name=='example_overlap.json':source=root/'python'/name
        if source.exists():shutil.copyfile(source,out/name)
    shutil.copyfile(root/'python/versions.json',out/'python_versions.json')
    source_inventory=json.loads((args.inventory or root/'source_inventory.json').read_text())
    for r in source_inventory:
        r.pop('folder',None)
        for m in r['members']:m.pop('path',None)
    (out/'source_hashes.json').write_text(json.dumps(source_inventory,ensure_ascii=False,indent=2)+'\n')
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    fields=['phase','id','stage','status','duration_seconds','archive','source_member','error','attribution','saved_log','metrics']
    with (out/'test_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in cases:
            record={k:row.get(k,'') for k in fields};record['metrics']=json.dumps(record['metrics'],ensure_ascii=False);writer.writerow(record)
    checks={}
    p=next((root/'second-pass/compat_pls_original').rglob('cross_language_pls.mat'))
    a=loadmat(p)['pred'].ravel();b=np.load(root/'python/python_pls_predictions.npy')
    checks['pls_octave_python_max_abs_prediction_difference']=float(np.max(np.abs(a-b)))
    checks['pls_prediction_rows']=len(a)
    def find(phase,test_id):return next(x for x in cases if x['phase']==phase and x['id']==test_id)
    original=find('focused','compat_audio_bp_complete');vector=find('vectorized','vectorized_audio_bp')
    checks['bp_original_loop_seconds']=original['duration_seconds']
    checks['bp_vectorized_seconds']=vector['duration_seconds']
    checks['bp_single_run_time_ratio']=original['duration_seconds']/vector['duration_seconds']
    checks['bp_displayed_class_recall_matches']=original['metrics'].get('example_class_recall_displayed')==vector['metrics'].get('example_class_recall_displayed')
    checks['bp_gradient_max_abs_difference']=find('vectorized','bp_gradient_equivalence')['metrics']['max_gradient_difference']
    checks['cars_original_error_ratio']=find('second-pass','verify_cars_error_scaling')['metrics']['raw_to_reported_ratio']
    checks['cars_corrected_error_ratio']=find('second-pass','patched_cars_error_scaling')['metrics']['raw_to_reported_ratio']
    checks['cars_selected_variables_unchanged']=find('second-pass','verify_cars_error_scaling')['metrics']['variables']==find('second-pass','patched_cars_error_scaling')['metrics']['variables']
    (out/'numerical_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    stage_counts=defaultdict(Counter)
    for row in cases:stage_counts[row.get('stage','python')][row['status']]+=1
    summary={'execution_attempts':len(cases),'status_counts':dict(Counter(x['status'] for x in cases)),
             'stage_counts':{k:dict(v) for k,v in stage_counts.items()},
             'native_matlab':'not installed, not executed; user approved Octave/Python testing',
             'scope':'execution attempts include shared algorithms, adapter retries and deliberately failing checks; not an algorithm pass rate'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    manifests=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name!='evidence_sha256.json':
            manifests.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/'evidence_sha256.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
