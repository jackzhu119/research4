#!/usr/bin/env python3
"""Third TRAIN-only B study: frozen objectives, labels and margin mechanisms."""
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
OUTPUT = ROOT / 'reports/b_mechanism_round3_20261110'
OLDROOT = ROOT / 'reports/b_mechanism_20261110'
ROUND2ROOT = ROOT / 'reports/b_mechanism_followup_20261110'
BASE_SHA = 'ad21deecbb53371f9b4547646a2aa985d8f617fb'
import run_b_mechanism_gap_20261110 as initial
import b_round3_regularization_20261110 as regularization
import b_round3_labels_20261110 as labels
import b_round3_margin_20261110 as margins
from b_round3_common_20261110 import canonical_hash, file_hash, validate_roles, verify_task


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
        raise ValueError('Expected the complete fixed 15 five-origin and 30 balanced binary contexts')
    jobs = []
    for family in ('G01', 'G02', 'G03'):
        for old in selected:
            job = copy.deepcopy(old)
            job.update(job_id=family + '_' + old['job_id'], family=family,
                       parent_job_id=old['job_id'],
                       phase2_job_id=('F01_' if family == 'G01' else 'F02_') + old['job_id'])
            if family == 'G03':
                job['dependency_job_id'] = 'G02_' + old['job_id']
                job['noise_plan'] = labels.noise_plan(job, metadata)
            jobs.append(job)
    for old in phase2_jobs:
        if old['family'] == 'F03':
            job = copy.deepcopy(old)
            job.update(job_id='G04_' + old['job_id'].removeprefix('F03_'), family='G04',
                       phase2_job_id=old['job_id'])
            jobs.append(job)
    if len(jobs) != 375 or len({job['job_id'] for job in jobs}) != 375:
        raise ValueError('Expected 375 unique third-round task bundles')
    for job in jobs:
        validate_roles(job, metadata)
    return jobs


def executor(job):
    return regularization if job['family'] == 'G01' else margins if job['family'] == 'G04' else labels


def expected_endpoints(job):
    return executor(job).expected_endpoints(job)


def previous_evidence():
    evidence = {}
    for name, root in (('first_round', OLDROOT), ('second_round', ROUND2ROOT)):
        receipt_path = root / 'background_finalization.json'
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt['artifact_sha256'].items():
            actual = ROOT / path
            if not actual.is_file() or file_hash(actual) != expected:
                raise ValueError('Previously sealed scientific evidence changed: ' + path)
        evidence[name] = {'finalization_sha256': file_hash(receipt_path),
                          'frozen_protocol_sha256': file_hash(root / 'frozen_protocol.json'),
                          'verified_prior_artifacts': len(receipt['artifact_sha256'])}
    return evidence


def prepare(write=False):
    registry, master, metadata, workbook_hashes = initial.base_inputs()
    features = initial.feature_data(metadata, prepare=write)
    if features is None:
        raise ValueError('Verified source feature cache is missing; use --prepare to reconstruct it')
    jobs = build_design(metadata)
    planned_models = sum(executor(job).fit_count(job) for job in jobs)
    planned_adapters = sum(executor(job).adapter_count(job) for job in jobs)
    evidence = previous_evidence()
    source_names = ('run_b_mechanism_round3_20261110.py', 'b_round3_common_20261110.py',
                    'b_round3_regularization_20261110.py', 'b_round3_labels_20261110.py',
                    'b_round3_margin_20261110.py', 'b_followup_common_20261110.py',
                    'b_followup_models_20261110.py', 'b_followup_calibration_20261110.py',
                    'b_mechanism_engine_20261110.py', 'run_b_mechanism_gap_20261110.py')
    source_paths = [ROOT / 'scripts' / name for name in source_names]
    source_paths += sorted((ROOT / 'src/herb_spectra').glob('*.py'))
    protocol = {
        'protocol_id': 'b-mechanism-round3-20261110-v1', 'base_commit': BASE_SHA,
        'registry_sha256': initial.sha(initial.REGISTRY), 'master_manifest_sha256': initial.sha(initial.MANIFEST),
        'workbook_sha256': workbook_hashes, 'array_sha256': features[3]['array_sha256'],
        'prior_evidence': evidence,
        'science_code_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
        'runtime': {'python': sys.version, 'libraries': {name: importlib.metadata.version(name) for name in
                      ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib')}},
        'design': jobs, 'endpoint_contract': {job['job_id']: expected_endpoints(job) for job in jobs},
        'planned_new_model_fits': planned_models, 'planned_new_adapter_fits': planned_adapters,
        'G01': {'loss': ['sum_lambda1', 'mean_lambda1_equivalent_sum_lambdaN'],
                'intercept_penalty': ['same_lambda', 'zero'], 'draws': [0, 101, 102],
                'k_per_class': [1, 3, 5, 10], 'full_pool_k10': 'canonical draw0 only',
                'old_sum_penalized_endpoints': 'reuse exact second-round scores and adapters'},
        'G02': {'k_per_class': [1, 3, 5, 10], 'gamma': 'frozen actual numeric source gamma',
                'scaler': 'fit Source+Support spectra in all adaptation arms',
                'arms': ['Source-only labels with unlabeled Support statistics',
                         'Source+Support true labels (historical exact endpoint)',
                         'Source+Support with Support-only class-cyclic copied labels'],
                'unlabeled_scope': 'Support pool was originally class-stratified; this arm does not fit Support label values'},
        'G03': {'k_per_class': 10, 'corrupted_per_class': [1, 2, 4, 10],
                'realized_corruption_fraction': [0.1, 0.2, 0.4, 1.0], 'mask_seeds': [201, 202, 203, 204, 205],
                'full_corruption': 'one canonical201 endpoint reused from G02; no repeated all-mask fits',
                'deduplication': 'unique copied Support-label objective signatures frozen before any new Query scoring',
                'semantics': 'artificial balanced class-cyclic diagnostic labels; not an audit finding of wrong original labels'},
        'G04': {'new_fits': 0, 'source_constraints': 'dual alpha versus per-class effective C bounds',
                'near_bound_rtol': 1e-7, 'near_bound_atol': 1e-9,
                'tolerance_role': 'descriptive numerical diagnostic only; never a training stopping rule',
                'saved_scores': 'exact second-round F03 native and argmax endpoints'},
        'inference': 'Exploratory internal master TRAIN; no Query-based choice of methods, masks, samples or parameters; no real new batches',
        'execution': {'single_thread': True, 'quality_cpu_wall_time_stop_cap': None, 'finite_frozen_matrix': True}}
    digest = canonical_hash(protocol)
    protocol['input_hash'] = digest
    if write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / '.gitignore').write_text('.background/\n.cache/\n*.lock\n*.log\n')
        target = OUTPUT / 'frozen_protocol.json'
        if target.exists() and json.loads(target.read_text()) != protocol:
            raise ValueError('Frozen third-round code, inputs, labels or endpoint design changed')
        if not target.exists():
            atomic_json(target, protocol)
            indexed = metadata.set_index('record_id', drop=False)
            locks = []
            for job in jobs:
                for role in ('source', 'support', 'query'):
                    for order, record_id in enumerate(job[role + '_ids']):
                        item = indexed.loc[record_id]
                        locks.append({'job_id': job['job_id'], 'family': job['family'], 'role': role,
                                      'order': order, 'record_id': record_id,
                                      'physical_sample_id': item.physical_sample_id,
                                      'origin_group_id': item.origin_group_id, 'species': item.species, 'master_role': item.role})
            pd.DataFrame(locks).to_csv(OUTPUT / 'split_locks.csv', index=False)
        atomic_json(OUTPUT / 'preflight.json', {
            'status': 'DONE', 'base_commit': BASE_SHA, 'previous_evidence': evidence,
            'active_records': len(metadata), 'species_counts': metadata.species.value_counts().to_dict(),
            'master_role_counts': metadata.role.value_counts().to_dict(),
            'feature_cache_byte_identical': features[3]['byte_identical_to_historical_npz'],
            'actual_batch_id_count': int(metadata.batch_id.ne('').sum()), 'sample_shortfalls': [],
            'planned_tasks': 375, 'planned_new_model_fits': planned_models,
            'planned_new_adapter_fits': planned_adapters, 'no_score_cpu_time_gate': True})
        pd.DataFrame([{'job_id': job['job_id'], 'family': job['family'], 'status': 'FEASIBLE',
                       'source_n': len(job['source_ids']), 'support_n': len(job['support_ids']),
                       'query_n': len(job['query_ids']), 'new_model_fits': executor(job).fit_count(job),
                       'new_adapter_fits': executor(job).adapter_count(job),
                       'endpoints': len(expected_endpoints(job))} for job in jobs]).to_csv(OUTPUT / 'feasibility.csv', index=False)
    return protocol, features


def verify_all(protocol):
    completed, model_count, adapter_count = 0, 0, 0
    for job in protocol['design']:
        destination = OUTPUT / 'jobs' / job['job_id']
        if (destination / 'COMPLETE.json').exists():
            result = verify_task(destination, protocol['input_hash'], job, expected_endpoints(job))
            if (result['new_model_fits'] != executor(job).fit_count(job)
                    or result.get('new_adapter_fits', 0) != executor(job).adapter_count(job)):
                raise ValueError('Per-task fitting/reuse counts differ from the frozen plan')
            completed += 1
            model_count += result['new_model_fits']
            adapter_count += result.get('new_adapter_fits', 0)
    locks = pd.read_csv(OUTPUT / 'split_locks.csv', dtype=str, keep_default_na=False)
    if not locks.master_role.eq('train').all():
        raise ValueError('Non-TRAIN identity in frozen role table')
    for job in protocol['design']:
        block = locks.loc[locks.job_id.eq(job['job_id'])]
        for role in ('source', 'support', 'query'):
            if block.loc[block.role.eq(role), 'record_id'].tolist() != job[role + '_ids']:
                raise ValueError('Frozen physical sample role assignment changed')
    receipt = {'status': 'DONE' if completed == len(protocol['design']) else 'PARTIAL',
               'verified_completed_jobs': completed, 'planned_jobs': len(protocol['design']),
               'new_model_fits': model_count, 'new_adapter_fits': adapter_count,
               'all_completed_artifacts_verified': True, 'input_hash': protocol['input_hash'], 'verified_at_utc': now()}
    atomic_json(OUTPUT / 'artifact_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def run(protocol, features, pilot=False):
    X, axis, metadata = features[:3]
    path = OUTPUT / 'execution_state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'input_hash': protocol['input_hash'], 'planned_jobs': len(protocol['design']),
        'created_at_utc': now(), 'attempts': []}
    if state['input_hash'] != protocol['input_hash']:
        raise ValueError('Scientific attempt ledger hash changed')
    chosen = protocol['design']
    if pilot:
        chosen = [next(job for job in chosen if job['family'] == family) for family in ('G01', 'G02', 'G03', 'G04')]
    state.update(status='RUNNING', quality_cpu_runtime_cutoffs=None)
    completed = 0
    for job in chosen:
        destination = OUTPUT / 'jobs' / job['job_id']
        if destination.exists():
            verify_task(destination, protocol['input_hash'], job, expected_endpoints(job))
            completed += 1
            continue
        attempt = {'job_id': job['job_id'], 'family': job['family'], 'started_at_utc': now(), 'status': 'RUNNING'}
        state['attempts'].append(attempt)
        state.update(current_job=job['job_id'], heartbeat_at_utc=now())
        atomic_json(path, state)
        cpu, wall = time.process_time(), time.monotonic()
        try:
            executor(job).execute(job, X, metadata, axis, OUTPUT, protocol['input_hash'], OLDROOT, ROUND2ROOT)
            verify_task(destination, protocol['input_hash'], job, expected_endpoints(job))
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
            print(json.dumps({'job': job['job_id'], 'status': attempt['status'],
                              'completed': state['completed_jobs'], 'planned': len(protocol['design'])}, ensure_ascii=False), flush=True)
    state.update(status=('PILOT_DONE' if pilot else 'DONE') if completed == len(chosen) else 'PARTIAL',
                 current_job=None, heartbeat_at_utc=now())
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
            print(json.dumps({'planned_jobs': len(protocol['design']),
                              'new_model_fits': protocol['planned_new_model_fits'],
                              'new_adapter_fits': protocol['planned_new_adapter_fits'], 'no_fits': True,
                              'input_hash': protocol['input_hash']}))
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
            state = run(protocol, features, pilot=args.pilot)
            return 0 if state['status'] in ('PILOT_DONE', 'DONE') else 1
        return 0
    except (ValueError, AssertionError) as error:
        print('INTEGRITY_BLOCKED: ' + str(error), file=sys.stderr, flush=True)
        return 42


if __name__ == '__main__':
    raise SystemExit(main())
