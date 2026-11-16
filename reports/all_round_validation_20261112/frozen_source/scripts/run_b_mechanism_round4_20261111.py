#!/usr/bin/env python3
"""Frozen fourth B study: loss mass, active C bounds, weak-score motion and Support composition."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import fcntl
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
OUTPUT = ROOT / 'reports/b_mechanism_round4_20261111'
OLDROOT = ROOT / 'reports/b_mechanism_20261110'
ROUND2ROOT = ROOT / 'reports/b_mechanism_followup_20261110'
ROUND3ROOT = ROOT / 'reports/b_mechanism_round3_20261110'
BASE_SHA = '69354eb7e2ec726ac5f5568fdac1035afff36da6'
import run_b_mechanism_gap_20261110 as initial
import b_round4_models_20261111 as models
import b_round4_weak_20261111 as weak
import b_round4_support_20261111 as support
from b_round4_common_20261111 import canonical_hash, file_hash, validate_roles, verify_task


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    initial.atomic_json(path, value)


def build_design(metadata, old_jobs=None, phase2_jobs=None):
    if old_jobs is None:
        old_jobs = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    if phase2_jobs is None:
        phase2_jobs = json.loads((ROUND2ROOT / 'frozen_protocol.json').read_text())['design']
    selected = [job for job in old_jobs if job['experiment'] == 'E03' or job.get('ratio') == 'B']
    if len(selected) != 45:
        raise ValueError('Expected all 15 five-origin and 30 balanced binary historical contexts')
    jobs = []
    for family in ('H01', 'H03', 'H04'):
        for old in selected:
            job = copy.deepcopy(old)
            job.update(job_id=family + '_' + old['job_id'], family=family,
                       parent_job_id=old['job_id'],
                       phase2_job_id=('F02_' if family == 'H01' else 'F01_') + old['job_id'])
            if family == 'H03':
                job['phase3_job_id'] = 'G03_' + old['job_id']
            if family == 'H04':
                job['composition_plan'] = support.composition_plan(job, metadata)
            jobs.append(job)
    for old in phase2_jobs:
        if old['family'] == 'F03':
            job = copy.deepcopy(old)
            job.update(job_id='H02_' + old['job_id'].removeprefix('F03_'), family='H02',
                       phase2_job_id=old['job_id'],
                       phase3_job_id='G04_' + old['job_id'].removeprefix('F03_'))
            jobs.append(job)
    jobs.sort(key=lambda job: (job['family'], job['job_id']))
    if len(jobs) != 375 or len({job['job_id'] for job in jobs}) != 375:
        raise ValueError('Expected 375 unique fourth-round task bundles')
    for job in jobs:
        validate_roles(job, metadata)
    return jobs


def executor(job):
    return models if job['family'] in ('H01', 'H02') else weak if job['family'] == 'H03' else support


def expected_endpoints(job):
    return executor(job).expected_endpoints(job)


def previous_evidence():
    evidence = {}
    for name, root in (('first', OLDROOT), ('second', ROUND2ROOT), ('third', ROUND3ROOT)):
        receipt_path = root / 'background_finalization.json'
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt['artifact_sha256'].items():
            actual = ROOT / path
            if not actual.is_file() or file_hash(actual) != expected:
                raise ValueError('Previously sealed evidence changed: ' + path)
        evidence[name] = {'finalization_sha256': file_hash(receipt_path),
                          'frozen_protocol_sha256': file_hash(root / 'frozen_protocol.json'),
                          'verified_prior_artifacts': len(receipt['artifact_sha256'])}
    return evidence


def prepare(write=False):
    registry, master, metadata, workbook_hashes = initial.base_inputs()
    features = initial.feature_data(metadata, prepare=write)
    if features is None:
        raise ValueError('Verified feature cache is absent; reconstruct with --prepare')
    jobs = build_design(metadata)
    planned_models = sum(executor(job).fit_count(job) for job in jobs)
    planned_adapters = sum(executor(job).adapter_count(job) for job in jobs)
    if (planned_models, planned_adapters) != (1160, 2340):
        raise ValueError('Fourth-round objective/reuse accounting differs from 1160 classifiers and 2340 adapters')
    evidence = previous_evidence()
    names = ('run_b_mechanism_round4_20261111.py', 'b_round4_common_20261111.py',
             'b_round4_models_20261111.py', 'b_round4_weak_20261111.py', 'b_round4_support_20261111.py',
             'b_round3_common_20261110.py', 'b_round3_margin_20261110.py',
             'b_followup_common_20261110.py', 'b_followup_models_20261110.py',
             'b_followup_calibration_20261110.py', 'b_mechanism_engine_20261110.py',
             'run_b_mechanism_gap_20261110.py')
    paths = [ROOT / 'scripts' / name for name in names] + sorted((ROOT / 'src/herb_spectra').glob('*.py'))
    protocol = {
        'protocol_id': 'b-mechanism-round4-20261111-v1', 'base_commit': BASE_SHA,
        'registry_sha256': initial.sha(initial.REGISTRY), 'master_manifest_sha256': initial.sha(initial.MANIFEST),
        'workbook_sha256': workbook_hashes, 'array_sha256': features[3]['array_sha256'],
        'prior_evidence': evidence,
        'science_code_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
        'runtime': {'python': sys.version, 'libraries': {name: importlib.metadata.version(name) for name in
                      ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib')}},
        'design': jobs, 'endpoint_contract': {job['job_id']: expected_endpoints(job) for job in jobs},
        'planned_new_model_fits': planned_models, 'planned_new_adapter_fits': planned_adapters,
        'H01': {'k_per_class': [1, 3, 5, 10], 'coordinates': 'frozen Source scaler and numeric gamma',
                'source_C': 10.0, 'source_only_mass_C': '10*(N_source+N_support)/N_source',
                'augmented_fixed_C': 10.0, 'augmented_mass_normalized_C': '10*N_source/(N_source+N_support)',
                'selection': 'derived loss-mass interventions, no Query-based C tuning'},
        'H02': {'fixed_C_doses': [0.1, 1.0, 10.0], 'class_weight': ['balanced', None],
                'coordinates': 'original Source scaler and numeric gamma',
                'reuse': 'C10 saved endpoints; equal-count ratio B has exactly unit balanced weights at every C',
                'constraint_tolerances': {'rtol': 1e-7, 'atol': 1e-9},
                'purpose': 'test conditional C-box activation and weight interaction; report every dose without choosing best C'},
        'H03': {'new_fits': 0, 'weak_actual_blend': 0.05,
                'cases': '34 F01 weak episodes plus 16 positive copied-label G03 weak cases per context',
                'zero_noise_dedup': 'G03 zero dose is the exact F01 K10 endpoint and is excluded',
                'certificate': 'source winning gap > 2*max(abs(weak_score-source_score)) is sufficient for unchanged argmax',
                'first_crossing': 'analytic fixed score-path class boundary; no blend grid or Query-label choice',
                'scope': 'finite saved-score numerical explanation, not population robustness or useful adaptation proof'},
        'H04': {'binary_support_total': 10, 'five_class_support_total': 25,
                'draws': [0, 101, 102], 'inverse_frequency_weights': 'N/(n_classes*n_support_class), sum weights N',
                'ridge_lambda': 1.0, 'balanced_cells': 'exact K5 historical endpoint; weighted and ordinary objectives identical',
                'per_class_K_not_applicable': -1, 'source_query_roles': 'fixed and paired across Support composition',
                'scope': 'label-stratified experimental Support composition, not inferred natural class prior'},
        'inference': 'Exploratory internal master TRAIN; no new real batches or Query-based parameter/sample choice',
        'execution': {'single_thread': True, 'quality_cpu_wall_time_stop_cap': None, 'finite_frozen_matrix': True}}
    protocol['input_hash'] = canonical_hash(protocol)
    if write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / '.gitignore').write_text('.background/\n.cache/\n*.lock\n*.log\n')
        target = OUTPUT / 'frozen_protocol.json'
        if target.exists() and json.loads(target.read_text()) != protocol:
            raise ValueError('Frozen fourth-round inputs, code, Support composition or endpoint contract changed')
        if not target.exists():
            atomic_json(target, protocol)
            indexed = metadata.set_index('record_id', drop=False)
            rows = []
            for job in jobs:
                for role in ('source', 'support', 'query'):
                    for order, record_id in enumerate(job[role + '_ids']):
                        item = indexed.loc[record_id]
                        rows.append({'job_id': job['job_id'], 'family': job['family'], 'role': role,
                                     'order': order, 'record_id': record_id, 'physical_sample_id': item.physical_sample_id,
                                     'origin_group_id': item.origin_group_id, 'species': item.species, 'master_role': item.role})
            pd.DataFrame(rows).to_csv(OUTPUT / 'split_locks.csv', index=False)
        atomic_json(OUTPUT / 'preflight.json', {
            'status': 'DONE', 'base_commit': BASE_SHA, 'previous_evidence': evidence,
            'active_records': len(metadata), 'species_counts': metadata.species.value_counts().to_dict(),
            'master_role_counts': metadata.role.value_counts().to_dict(),
            'feature_cache_byte_identical': features[3]['byte_identical_to_historical_npz'],
            'actual_batch_id_count': int(metadata.batch_id.ne('').sum()), 'sample_shortfalls': [],
            'planned_tasks': 375, 'planned_new_model_fits': planned_models,
            'planned_new_adapter_fits': planned_adapters, 'no_score_cpu_time_gate': True})
        pd.DataFrame([{'job_id': job['job_id'], 'family': job['family'], 'status': 'FEASIBLE',
                       'source_n': len(job['source_ids']), 'support_n': len(job['support_ids']), 'query_n': len(job['query_ids']),
                       'new_model_fits': executor(job).fit_count(job), 'new_adapter_fits': executor(job).adapter_count(job),
                       'endpoints': len(expected_endpoints(job))} for job in jobs]).to_csv(OUTPUT / 'feasibility.csv', index=False)
    return protocol, features


def verify_all(protocol):
    completed = new_models = new_adapters = 0
    for job in protocol['design']:
        path = OUTPUT / 'jobs' / job['job_id']
        if (path / 'COMPLETE.json').exists():
            result = verify_task(path, protocol['input_hash'], job, expected_endpoints(job))
            if (result['new_model_fits'] != executor(job).fit_count(job)
                    or result.get('new_adapter_fits', 0) != executor(job).adapter_count(job)):
                raise ValueError('Task fitting/reuse counts differ from frozen fourth-round design')
            completed += 1
            new_models += result['new_model_fits']
            new_adapters += result.get('new_adapter_fits', 0)
    locks = pd.read_csv(OUTPUT / 'split_locks.csv', dtype=str, keep_default_na=False)
    if not locks.master_role.eq('train').all():
        raise ValueError('Non-TRAIN identity in fourth-round role table')
    for job in protocol['design']:
        subset = locks.loc[locks.job_id.eq(job['job_id'])]
        for role in ('source', 'support', 'query'):
            if subset.loc[subset.role.eq(role), 'record_id'].tolist() != job[role + '_ids']:
                raise ValueError('Frozen fourth-round physical sample role lock changed')
    receipt = {'status': 'DONE' if completed == len(protocol['design']) else 'PARTIAL',
               'verified_completed_jobs': completed, 'planned_jobs': len(protocol['design']),
               'new_model_fits': new_models, 'new_adapter_fits': new_adapters,
               'all_completed_artifacts_verified': True, 'input_hash': protocol['input_hash'], 'verified_at_utc': now()}
    atomic_json(OUTPUT / 'artifact_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def run(protocol, features, pilot=False):
    X, axis, metadata = features[:3]
    path = OUTPUT / 'execution_state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'input_hash': protocol['input_hash'], 'planned_jobs': len(protocol['design']), 'created_at_utc': now(), 'attempts': []}
    if state['input_hash'] != protocol['input_hash']:
        raise ValueError('Fourth-round execution ledger hash changed')
    jobs = protocol['design']
    if pilot:
        jobs = [next(job for job in jobs if job['family'] == family) for family in ('H01', 'H02', 'H03', 'H04')]
    state.update(status='RUNNING', quality_cpu_runtime_cutoffs=None)
    completed = 0
    for job in jobs:
        path_job = OUTPUT / 'jobs' / job['job_id']
        if path_job.exists():
            verify_task(path_job, protocol['input_hash'], job, expected_endpoints(job))
            completed += 1
            continue
        attempt = {'job_id': job['job_id'], 'family': job['family'], 'started_at_utc': now(), 'status': 'RUNNING'}
        state['attempts'].append(attempt)
        state.update(current_job=job['job_id'], heartbeat_at_utc=now())
        atomic_json(path, state)
        cpu, wall = time.process_time(), time.monotonic()
        try:
            executor(job).execute(job, X, metadata, axis, OUTPUT, protocol['input_hash'], OLDROOT, ROUND2ROOT, ROUND3ROOT)
            verify_task(path_job, protocol['input_hash'], job, expected_endpoints(job))
            attempt['status'] = 'DONE'
            completed += 1
        except Exception as error:
            attempt.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        finally:
            attempt.update(attempt_cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
                           finished_at_utc=now())
            state.update(completed_jobs=len(list((OUTPUT / 'jobs').glob('*/COMPLETE.json'))), heartbeat_at_utc=now(),
                         measured_attempt_cpu_seconds=sum(item.get('attempt_cpu_seconds', 0) for item in state['attempts']))
            atomic_json(path, state)
            print(json.dumps({'job': job['job_id'], 'status': attempt['status'], 'completed': state['completed_jobs'],
                              'planned': len(protocol['design'])}, ensure_ascii=False), flush=True)
    state.update(status=('PILOT_DONE' if pilot else 'DONE') if completed == len(jobs) else 'PARTIAL', current_job=None, heartbeat_at_utc=now())
    atomic_json(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('dry-run', 'prepare', 'pilot', 'resume', 'verify'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('choose --dry-run, --prepare, --pilot, --resume or --verify')
    try:
        if args.dry_run:
            protocol, _ = prepare(write=False)
            print(json.dumps({'planned_jobs': len(protocol['design']), 'new_model_fits': protocol['planned_new_model_fits'],
                              'new_adapter_fits': protocol['planned_new_adapter_fits'], 'no_fits': True, 'input_hash': protocol['input_hash']}))
            return 0
        OUTPUT.mkdir(parents=True, exist_ok=True)
        lock = (OUTPUT / 'runner.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, features = prepare(write=True)
        if args.verify:
            verify_all(protocol)
        elif args.prepare:
            print(json.dumps({'status': 'FROZEN_NO_FIT', 'planned_jobs': len(protocol['design']), 'input_hash': protocol['input_hash']}))
        else:
            return 0 if run(protocol, features, pilot=args.pilot)['status'] in ('PILOT_DONE', 'DONE') else 1
        return 0
    except (ValueError, AssertionError) as error:
        print('INTEGRITY_BLOCKED: ' + str(error), file=sys.stderr, flush=True)
        return 42


if __name__ == '__main__':
    raise SystemExit(main())
