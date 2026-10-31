#!/usr/bin/env python3
"""Compile post-run delivery evidence; this is not a pre-evaluation permit.

Verify immutable component indexes, original protected Git blobs, and saved OOF
counts independently. No model or preprocessing fit is invoked. Refuse to
replace a completed delivery. Scientific eligibility is never inferred from a
technical check.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/upgrade-20261031-v1'
START='ee47a7caf92c7e3288f9ef4542d2be44ac895527'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(name,data):
    p=RUN/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def ref(path):
    p=Path(path)
    return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}

def verify_index(path):
    obj=read(path);entries=obj.get('files',obj)
    checked=0
    for name,record in entries.items():
        if not isinstance(record,(str,dict)):continue
        expected=record if isinstance(record,str) else record.get('sha256')
        if not expected:continue
        p=(Path(path).parent/name).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Index escapes repository')
        if sha(p)!=expected:raise ValueError(f'Component index mismatch: {p}')
        checked+=1
    if not checked:raise ValueError('Empty component evidence index')
    return {**ref(path),'checked_files':checked}

def protected_blobs():
    # Compare bytes to the actual starting Git tree, including every historical
    # raw input, weight and report; no trust in a self-written success flag.
    raw=subprocess.check_output(['git','ls-tree','-r','-z',START,'--','data','models','reports','references','src','scripts','tests'],cwd=ROOT)
    count=0
    for item in raw.split(b'\0'):
        if not item:continue
        header,name=item.split(b'\t',1);mode,kind,blob=header.split()
        if kind!=b'blob':continue
        p=ROOT/name.decode()
        if not p.is_file():raise ValueError(f'Historical file missing: {p}')
        h=hashlib.sha1();h.update(f'blob {p.stat().st_size}\0'.encode())
        with p.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        if h.hexdigest()!=blob.decode():raise ValueError(f'Historical bytes changed: {p}')
        count+=1
    return count

def independent_neural_metrics():
    with np.load(ROOT/'data/multiplant_database/zaoren/aligned_modalities.npz',allow_pickle=False) as z:
        truth=dict(zip(z['record_ids'].tolist(),z['y'].tolist()))
    groups=defaultdict(list)
    with gzip.open(RUN/'neural/outer_predictions.csv.gz','rt',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):groups[(row['family'],row['seed'])].append(row)
    metrics=read(RUN/'neural/pooled_metrics_by_seed.json');order=[f'ZY{i:02}' for i in range(1,8)]
    lookup={v:i for i,v in enumerate(order)};results=[];maximum=0.
    if len(groups)!=27:raise ValueError('Incomplete role/seed prediction inventory')
    for (family,seed),rows in sorted(groups.items()):
        ids=[r['record_id'] for r in rows]
        if len(ids)!=855 or len(set(ids))!=855 or set(ids)!=set(truth):raise ValueError('Wrong OOF denominator')
        cm=np.zeros((7,7),dtype=int)
        for r in rows:cm[lookup[truth[r['record_id']]],lookup[r['y_predicted']]]+=1
        tp=np.diag(cm).astype(float);actual=cm.sum(axis=1);predicted=cm.sum(axis=0)
        recalls=np.divide(tp,actual,out=np.zeros(7),where=actual!=0)
        f1s=np.divide(2*tp,actual+predicted,out=np.zeros(7),where=(actual+predicted)!=0)
        numbers={'accuracy':float(tp.sum()/855),'macro_f1':float(f1s.mean()),'balanced_accuracy':float(recalls.mean())}
        saved=metrics[family][seed]
        for key,value in numbers.items():
            diff=abs(value-saved[key]);maximum=max(maximum,diff)
            if diff>1e-12:raise ValueError('Independent metric arithmetic differs')
        if saved['confusion_matrix']!=cm.tolist():raise ValueError('Confusion matrix differs')
        results.append({'family':family,'seed':int(seed),'correct':int(tp.sum()),'total':855,**numbers})
    return {'cases':27,'raw_prediction_rows':23085,'maximum_numeric_difference':maximum,'independent_confusion_count_arithmetic':True,'results':results}

def compile_delivery():
    if (RUN/'delivery_validation.json').exists():raise FileExistsError('Delivery already finalized')
    if any((RUN/n).exists() for n in ['fit_scope_manifest.json','selection_lock.json','leakage_audit.json','files_sha256.json']):raise FileExistsError('Partial or completed delivery exists; preserve before a new version')
    at=datetime.now(timezone.utc).isoformat()
    indexes=[ROOT/'data/multiplant_database/files_sha256.json']
    indexes += [RUN/name/'files_sha256.json' for name in ['neural','adversarial-audit','source-peaks','source-attributions','quantum','quantum-additional-candidates','paper-assets']]
    indexes += [RUN/'neural-independent-audit/audit_deliverable_index.json']
    # Auxiliary must be complete and independently reviewed before finalizing.
    indexes += [RUN/'auxiliary/files_sha256.json',RUN/'auxiliary-independent-audit/audit_deliverable_index.json']
    verified=[verify_index(p) for p in indexes]
    testlog=RUN/'project-tests-release.log'
    finaltests=testlog.read_text();match=re.search(r'Ran (\d+) tests',finaltests)
    if not match or int(match[1])<358 or not re.search(r'^OK$',finaltests,re.M):raise ValueError('Final project tests incomplete')
    testcount=int(match[1])
    aux=read(RUN/'auxiliary-independent-audit/final_review.json')
    if aux.get('status') != 'TECHNICAL_AUXILIARY_OUTER_REPLAY_PASS_EXPLORATORY_DATA':raise ValueError('Auxiliary independent review incomplete')
    expected_aux={'actual_outer_models_replayed':180,'SVC_models_replayed':60,'neural_models_replayed':120,'actual_model_roles':180,'aliased_policy_roles':30,'raw_prediction_rows':16968,'pooled_plant_method_seed_cases':42,'real_specimens_per_plant_method_seed':{'huangqi':403,'tianma':405}}
    if any(aux.get(k)!=v for k,v in expected_aux.items()):raise ValueError('Incomplete auxiliary outer replay counts')
    auxinner=read(RUN/'auxiliary-independent-audit/selection_review.json')
    if auxinner.get('status')!='TECHNICAL_AUXILIARY_INNER_REVIEW_PASS_EXPLORATORY_DATA' or auxinner.get('models_replayed')!=1200:raise ValueError('Incomplete auxiliary source review counts')
    historic=read(RUN/'state-audit/repository_initial_snapshot.json')
    for name,expected in historic['historic_inputs_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError(f'Initial input hash changed: {name}')
    execution_sources=[]
    for name in ['neural/run_manifest.json','auxiliary/run_manifest.json']:
        path=RUN/name;mapping=read(path)['source_sha256']
        for source,expected in mapping.items():
            if sha(ROOT/source)!=expected:raise ValueError(f'Executed source drifted: {source}')
        execution_sources.append({**ref(path),'current_source_hashes_checked':len(mapping)})
    bundle=read(RUN/'reading_bundle_manifest.json')
    if sha(ROOT/bundle['path'])!=bundle['sha256']:raise ValueError('Reading bundle changed')
    for name,expected in bundle['source_members'].items():
        if sha(ROOT/name)!=expected:raise ValueError(f'Reading bundle source drifted: {name}')
    with zipfile.ZipFile(ROOT/bundle['path']) as z:
        if z.testzip() is not None:raise ValueError('Reading bundle CRC failed')
    protected=protected_blobs();metrics=independent_neural_metrics()
    for folder in [ROOT/'data/multiplant_database',RUN]:
        if any(p.stat().st_size>=100*2**20 for p in folder.rglob('*') if p.is_file()):raise ValueError('GitHub file size exceeded')
    source_scopes=['neural/fit_scope_manifest.json','adversarial-audit/fit_scope_manifest.json','source-peaks/fit_scope_manifest.json','source-attributions/fit_scope_manifest.json','auxiliary/fit_scope_manifest.json']
    locks=['neural/source_candidate_manifest.json','neural/selection_lock.json','neural/prediction_lock.json','auxiliary/candidate_manifest.json','auxiliary/selection_lock.json','auxiliary/prediction_lock.json']
    gates=['minimum_training_gate.json','neural_source_training_gate.json','neural_outer_evaluation_release.json','auxiliary_source_training_gate.json','auxiliary_source_training_gate_v3.json','auxiliary_outer_evaluation_release.json']
    auditrefs=['leakage-core/core_guard_validation.json','adversarial-audit/leakage_audit.json','neural-independent-audit/source_review.json','neural-independent-audit/selection_review.json','neural-independent-audit/final_review.json','auxiliary-independent-audit/source_review.json','auxiliary-independent-audit/source_review_v3.json','auxiliary-independent-audit/selection_review.json','auxiliary-independent-audit/final_review.json']
    # Resolve every evidence reference before creating any root completion file.
    checked_refs={name:ref(RUN/name) for name in source_scopes+locks+gates+auditrefs}
    write('fit_scope_manifest.json',{'compiled_at_utc':at,'compiled_after_component_runs':True,'status':'TECHNICAL_INSTRUMENTED_SCOPE_EVIDENCE','components':[ref(RUN/p) for p in source_scopes],'neural_actual_completed_learned_operations':11250,'auxiliary_successful_completed_learned_operations':42420,'auxiliary_failed_v2_completed_operations':2,'arbitrary_uninstrumented_code_certified':False,'scientific_status':'EXPLORATORY'})
    write('selection_lock.json',{'compiled_at_utc':at,'compiled_after_component_runs':True,'is_ex_ante_global_permission':False,'purpose':'Index of distinct component locks, each issued before its own prohibited evaluation; not a retroactive pre-registration','locks':[ref(RUN/p) for p in locks],'actual_release_files':[ref(RUN/p) for p in gates],'each_component_outer_results_used_to_select_its_own_candidates':False,'historically_exposed_zaoren':True,'fresh_confirmation':False})
    write('leakage_audit.json',{'compiled_at_utc':at,'compiled_after_component_runs':True,'status':'FINITE_TECHNICAL_CHECKS_PASS_EXPLORATORY_SCIENTIFIC_STATUS','components':[ref(RUN/p) for p in auditrefs],'project_tests_passed':testcount,'project_test_log':ref(testlog),'actual_scope':'A/B/E/F/G intentional failure fixtures; C/D/H/I empirical representative fixed-fold MC-SVM; actual source fit tracing; all main and auxiliary outer model pointwise/state checks and independent replay','full_C_D_retraining_all_families_all_folds':'NOT_RUN','OS_enforced_truth_sandbox':False,'absolute_zero_leakage_proven':False,'independent_confirmation_cohorts':0,'unresolved':['Real growth origins and species authentication','Common upstream lineage outside user declarations','One source batch per acid-jujube class','Unknown auxiliary batch counts','Cross-plant units/instrument/reference compatibility'],'blocked_studies':['Cross-plant transfer','Real new-batch few-shot calibration','Three medicinal-candidate DFT frequencies and theory-prior classification']})
    result={'compiled_at_utc':at,'status':'DELIVERY_BYTE_AND_ARITHMETIC_CHECK_PASS_NOT_SCIENTIFIC_CONFIRMATION','finalizer':ref(Path(__file__)),'component_indexes':verified,'starting_commit':START,'protected_historical_blobs_checked':protected,'initial_input_hashes_unchanged':True,'executed_sources_match_current_tree':execution_sources,'all_new_files_below_100MiB':True,'independent_main_neural_metrics':metrics,'auxiliary_final_review':ref(RUN/'auxiliary-independent-audit/final_review.json'),'project_tests_passed':testcount,'new_model_fits_in_finalizer':0,'reading_bundle_manifest':ref(RUN/'reading_bundle_manifest.json'),'unchanged_main_expected':'709dc9a90f4f7ba3ae104503e612e5228de5f20d'}
    write('delivery_validation.json',result)
    files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'] and p.name!='files_sha256.json'}
    # Include component index files too; exclude only this root self-index.
    for p in sorted(RUN.rglob('files_sha256.json')):
        if p.parent!=RUN:files[str(p.relative_to(RUN))]=sha(p)
    write('files_sha256.json',files)
    print(json.dumps({'indexed_files':len(files),'verified_component_files':sum(x['checked_files'] for x in verified),'protected_historical_blobs':protected,'metric_cases':27,'maximum_numeric_difference':metrics['maximum_numeric_difference']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();compile_delivery()
