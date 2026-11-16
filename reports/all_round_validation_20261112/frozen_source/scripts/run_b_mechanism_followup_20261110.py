#!/usr/bin/env python3
"""Run the preregistered second B study, independently resumable per task."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import fcntl
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
OUTPUT = ROOT / 'reports/b_mechanism_followup_20261110'
OLDROOT = ROOT / 'reports/b_mechanism_20261110'
BASE_SHA = '76966534df314e5bcb108a1fdccd611d1ad52910'
from b_followup_common_20261110 import canonical_hash, file_hash, verify_task, validate_roles
import b_followup_calibration_20261110 as calibration
import b_followup_models_20261110 as models
import run_b_mechanism_gap_20261110 as previous


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    previous.atomic_json(path, value)


def build_design(metadata, old_jobs=None):
    if old_jobs is None:
        old_jobs = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    selected = [job for job in old_jobs if job['experiment'] == 'E03' or job.get('ratio') == 'B']
    if len(selected) != 45:
        raise ValueError('Expected fixed 15 five-class and 30 balanced-ratio binary contexts')
    jobs = []
    for family in ('F01', 'F02'):
        for original in selected:
            job = copy.deepcopy(original)
            job.update(job_id=family + '_' + original['job_id'], family=family,
                       parent_job_id=original['job_id'])
            if family == 'F02':
                job['exact_gamma_reuse_k'] = models.gamma_reuse_plan(job, OLDROOT)
            jobs.append(job)
    train = metadata.loc[metadata.role.eq('train')]
    acid_groups = ('ZY03', 'ZY04', 'ZY05', 'ZY06', 'ZY07')
    old_map = {job['job_id']: job for job in old_jobs}
    for target in acid_groups:
        for seed in range(42, 50):
            rng = np.random.default_rng(seed)
            target_order = rng.permutation(sorted(train.loc[train.origin_group_id.eq(target), 'record_id'])).tolist()
            zaoren_order = rng.permutation(sorted(train.loc[train.origin_group_id.eq('ZY01'), 'record_id'])).tolist()
            other_order = rng.permutation(sorted(train.loc[train.origin_group_id.isin(set(acid_groups) - {target}), 'record_id'])).tolist()
            for ratio, (acid_n, zaoren_n) in {'A': (60, 180), 'B': (120, 120), 'C': (180, 60)}.items():
                for coverage in (0, 1):
                    old_id = f'E01_{target}_s{seed}_r{ratio}_c{coverage}'
                    original = {'job_id': old_id, 'experiment': 'E01', 'target_origin': target,
                                'seed': seed, 'coverage': coverage, 'ratio': ratio, 'fold': None,
                                'source_ids': (zaoren_order[30:30 + zaoren_n] + other_order[:acid_n - 20 * coverage]
                                               + (target_order[30:50] if coverage else [])),
                                'support_ids': zaoren_order[20:30] + target_order[20:30],
                                'query_ids': zaoren_order[:20] + target_order[:20],
                                'classes': ['枣仁', '酸枣仁'], 'label_column': 'species',
                                'source_class_counts': {'枣仁': zaoren_n, '酸枣仁': acid_n}}
                    if seed <= 44 and original != old_map[old_id]:
                        raise ValueError('Historical role sequence was not reproduced exactly')
                    original.update(job_id='F03_' + old_id, family='F03',
                                    parent_job_id=old_id if seed <= 44 else None)
                    jobs.append(original)
    for original in old_jobs:
        if original['experiment'] == 'E03':
            job = copy.deepcopy(original)
            job.update(job_id='F04_' + original['job_id'], family='F04', parent_job_id=original['job_id'])
            jobs.append(job)
    if len(jobs) != 345 or len({job['job_id'] for job in jobs}) != 345:
        raise ValueError('Expected 345 unique follow-up task bundles')
    for job in jobs:
        validate_roles(job, metadata)
    return jobs


def expected_endpoints(job):
    return calibration.expected_endpoints(job) if job['family'] == 'F01' else models.expected_endpoints(job)


def source_snapshot():
    # Verify prior scientific/report artifacts against the completed publication receipt.
    receipt_path = OLDROOT / 'background_finalization.json'
    receipt = json.loads(receipt_path.read_text())
    for name, expected in receipt['artifact_sha256'].items():
        path = ROOT / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError('Prior completed evidence changed: ' + name)
    return {'completed_parent_commit': BASE_SHA,
            'parent_finalization_receipt_sha256': file_hash(receipt_path),
            'parent_artifacts_verified': len(receipt['artifact_sha256']),
            'parent_protocol_sha256': file_hash(OLDROOT / 'frozen_protocol.json'),
            'parent_result_sha256': {path.parent.name: file_hash(path) for path in sorted((OLDROOT / 'jobs').glob('*/result.json'))}}


def prepare(write=False):
    registry, master, metadata, workbooks = previous.base_inputs()
    features = previous.feature_data(metadata, prepare=False)
    if features is None:
        if not write:
            raise ValueError('Verified feature cache is absent; run --prepare to reconstruct it')
        # Cache location is ignored and contains no frozen historical result files.
        features = previous.feature_data(metadata, prepare=True)
    jobs = build_design(metadata)
    audit = source_snapshot()
    scientific = [Path(__file__), ROOT / 'scripts/b_followup_common_20261110.py',
                  ROOT / 'scripts/b_followup_calibration_20261110.py', ROOT / 'scripts/b_followup_models_20261110.py',
                  ROOT / 'scripts/run_b_mechanism_gap_20261110.py', ROOT / 'scripts/b_mechanism_engine_20261110.py']
    scientific += sorted((ROOT / 'src/herb_spectra').glob('*.py'))
    protocol = {
        'protocol_id': 'b-mechanism-followup-20261110-v1', 'base_commit': BASE_SHA,
        'input_registry_sha256': previous.sha(previous.REGISTRY),
        'input_manifest_sha256': previous.sha(previous.MANIFEST),
        'input_workbook_sha256': workbooks, 'array_sha256': features[3]['array_sha256'],
        'science_code_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in scientific},
        'parent_evidence': audit, 'design': jobs,
        'endpoint_contract': {job['job_id']: expected_endpoints(job) for job in jobs},
        'planned_new_model_fits': sum(models.fit_count(job) for job in jobs if job['family'] != 'F01'),
        'planned_new_adapter_fits': 12150,
        'selection': 'F01/F02 all old 15 E03 plus all30 E01 ratio B, selected by design not query score',
        'F01': {'support_draws': [0] + list(range(101, 111)), 'k_per_class': [1, 3, 5, 10],
                'k10_deduplication': 'true labels only canonical draw0; shuffled labels remain separate copied-label diagnostics',
                'score_units': [0.1, 1.0, 10.0], 'source_rms': 'global positive RMS after per-row class-score centering, SOURCE-only'},
        'F02': {'scaler': ['source_frozen', 'source_plus_support_refit'],
                'gamma': ['old_numeric_frozen', 'scale_recomputed'], 'k_per_class': [1, 3, 5, 10],
                'reuse': 'refit_scaler/recomputed_gamma is the saved original full-pipeline model'},
        'F03': {'sampling_seeds': list(range(42, 50)), 'class_weight': ['balanced', None],
                'ratios': ['A', 'B', 'C'], 'coverage': [0, 1], 'old_balanced_models_reused': 90},
        'F04': {'source_sample_sizes': [50, 100, 150, 190, 240], 'class_balance': 'equal per group',
                'reuse': '190 original source, 240 old Source190+Support50 K10 refit'},
        'inference': 'exploratory internal TRAIN reuse; no new batches, no confirmation, no query-based selection',
        'execution': {'single_thread': True, 'cpu_stop_cap_seconds': None, 'quality_or_wall_time_gate': None,
                      'finite_frozen_matrix': True, 'separate_cost_ledger': True}}
    if protocol['planned_new_model_fits'] != 723:
        raise ValueError('New classifier fit count must be 723 after exact-objective deduplication')
    digest = canonical_hash(protocol)
    protocol['input_hash'] = digest
    if write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / '.gitignore').write_text('.background/\n.cache/\n*.lock\n*.log\n')
        path = OUTPUT / 'frozen_protocol.json'
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('Frozen follow-up input/code/endpoint protocol changed; unsafe resume refused')
        if not path.exists():
            atomic_json(path, protocol)
            lookup = metadata.set_index('record_id', drop=False)
            rows = []
            for job in jobs:
                for role in ('source', 'support', 'query'):
                    for order, record_id in enumerate(job[role + '_ids']):
                        item = lookup.loc[record_id]
                        rows.append({'job_id': job['job_id'], 'family': job['family'], 'role': role,
                                     'order': order, 'record_id': record_id, 'physical_sample_id': item.physical_sample_id,
                                     'species': item.species, 'origin_group_id': item.origin_group_id, 'master_role': item.role})
            pd.DataFrame(rows).to_csv(OUTPUT / 'split_locks.csv', index=False)
        atomic_json(OUTPUT / 'preflight.json', {
            'status': 'DONE', 'base_commit': BASE_SHA, 'parent_evidence': audit,
            'active_records': len(metadata), 'species_counts': metadata.species.value_counts().to_dict(),
            'master_role_counts': metadata.role.value_counts().to_dict(),
            'batch_ids_supplied': int(metadata.batch_id.ne('').sum()),
            'feature_byte_identical_to_historical': features[3]['byte_identical_to_historical_npz'],
            'planned_tasks': len(jobs), 'planned_new_classifier_fits': 723, 'planned_new_adapter_fits': 12150,
            'sample_shortfalls': [], 'cpu_or_quality_stopping_threshold': None,
            'historical_data_models_results_unchanged': True})
        pd.DataFrame([{'job_id': job['job_id'], 'family': job['family'], 'status': 'FEASIBLE',
                       'source_n': len(job['source_ids']), 'support_n': len(job['support_ids']), 'query_n': len(job['query_ids']),
                       'new_classifier_fits': 0 if job['family'] == 'F01' else models.fit_count(job),
                       'endpoints': len(expected_endpoints(job))} for job in jobs]).to_csv(OUTPUT / 'feasibility.csv', index=False)
    return protocol, features


def verify_all(protocol):
    completed, fits, adapters = 0, 0, 0
    for job in protocol['design']:
        path = OUTPUT / 'jobs' / job['job_id']
        if (path / 'COMPLETE.json').exists():
            result = verify_task(path, protocol['input_hash'], job, expected_endpoints(job))
            completed += 1
            fits += result['new_model_fits']
            adapters += result.get('new_adapter_fits', 0)
    locks = pd.read_csv(OUTPUT / 'split_locks.csv', dtype=str, keep_default_na=False)
    if not locks.master_role.eq('train').all():
        raise ValueError('Non-TRAIN identity in split lock')
    for job in protocol['design']:
        block = locks.loc[locks.job_id.eq(job['job_id'])]
        for role in ('source', 'support', 'query'):
            if block.loc[block.role.eq(role), 'record_id'].tolist() != job[role + '_ids']:
                raise ValueError('Frozen split lock changed')
    status = 'DONE' if completed == len(protocol['design']) else 'PARTIAL'
    if status == 'DONE' and (fits != 723 or adapters != 12150):
        raise ValueError('Actual scientific fit counts disagree with frozen matrix')
    receipt = {'status': status, 'verified_completed_jobs': completed, 'planned_jobs': 345,
               'new_model_fits': fits, 'new_adapter_fits': adapters, 'verified_at_utc': now(),
               'input_hash': protocol['input_hash'], 'all_completed_artifacts_verified': True}
    atomic_json(OUTPUT / 'artifact_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def run(protocol, features, pilot=False):
    X, axis, metadata = features[:3]
    path = OUTPUT / 'execution_state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'input_hash': protocol['input_hash'], 'planned_jobs': 345, 'attempts': [], 'created_at_utc': now()}
    if state['input_hash'] != protocol['input_hash']:
        raise ValueError('Execution ledger hash mismatch')
    chosen = protocol['design']
    if pilot:
        chosen = [next(job for job in chosen if job['family'] == family) for family in ('F01', 'F02', 'F03', 'F04')]
    state.update(status='RUNNING', quality_cpu_runtime_cutoffs=None)
    completed = 0
    for job in chosen:
        destination = OUTPUT / 'jobs' / job['job_id']
        if destination.exists():
            verify_task(destination, protocol['input_hash'], job, expected_endpoints(job))
            completed += 1
            continue
        attempt = {'job_id': job['job_id'], 'family': job['family'], 'status': 'RUNNING', 'started_at_utc': now()}
        state['attempts'].append(attempt)
        state.update(current_job=job['job_id'], heartbeat_at_utc=now())
        atomic_json(path, state)
        cpu, wall = time.process_time(), time.monotonic()
        try:
            executor = calibration.execute if job['family'] == 'F01' else models.execute
            executor(job, X, metadata, axis, OUTPUT, protocol['input_hash'], OLDROOT)
            verify_task(destination, protocol['input_hash'], job, expected_endpoints(job))
            attempt['status'] = 'DONE'
            completed += 1
        except Exception as error:
            attempt.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        finally:
            attempt.update(attempt_cpu_seconds=time.process_time() - cpu,
                           wall_seconds=time.monotonic() - wall, finished_at_utc=now())
            state.update(completed_jobs=len(list((OUTPUT / 'jobs').glob('*/COMPLETE.json'))),
                         heartbeat_at_utc=now(),
                         measured_attempt_cpu_seconds=sum(item.get('attempt_cpu_seconds', 0) for item in state['attempts']))
            atomic_json(path, state)
            print(json.dumps({'job': job['job_id'], 'status': attempt['status'],
                              'completed': state['completed_jobs'], 'planned': 345}, ensure_ascii=False), flush=True)
    state.update(status='PILOT_DONE' if pilot else ('DONE' if completed == len(chosen) else 'PARTIAL'),
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
            print(json.dumps({'planned_jobs': len(protocol['design']), 'new_model_fits': 723,
                              'new_adapter_fits': 12150, 'input_hash': protocol['input_hash'], 'no_fits': True}))
            return 0
        OUTPUT.mkdir(parents=True, exist_ok=True)
        lock = (OUTPUT / 'runner.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, features = prepare(write=True)
        if args.verify:
            verify_all(protocol)
        elif args.prepare:
            print(json.dumps({'status': 'FROZEN_NO_FIT', 'planned_jobs': 345, 'input_hash': protocol['input_hash']}))
        else:
            state = run(protocol, features, pilot=args.pilot)
            if not args.pilot and state['status'] != 'DONE':
                return 1
        return 0
    except (ValueError, AssertionError) as error:
        print('INTEGRITY_BLOCKED: ' + str(error), file=sys.stderr, flush=True)
        return 42


if __name__ == '__main__':
    raise SystemExit(main())
