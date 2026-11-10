#!/usr/bin/env python3
"""Frozen TRAIN-only B-mechanism study; resumable and independent of chat."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
OUTPUT = ROOT / 'reports/b_mechanism_20261110'
MANIFEST = ROOT / 'data/research/species_revision_20261108/split_manifest.csv'
REGISTRY = ROOT / 'data/active_workbooks.json'
MASTER_PROTOCOL = ROOT / 'data/research/species_revision_20261108/protocol.json'
BASE_SHA = 'b051733989161517a68c4a60cce75766880c95f5'


class IntegrityError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def base_inputs():
    registry = json.loads(REGISTRY.read_text())
    master = json.loads(MASTER_PROTOCOL.read_text())
    metadata = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    require(len(metadata) == 1672, 'Active manifest must contain 1672 records')
    require(metadata.record_id.is_unique, 'Duplicate record_id')
    require(metadata.physical_sample_id.is_unique and metadata.physical_sample_id.ne('').all(),
            'Missing or duplicate physical_sample_id')
    require(metadata.species.value_counts().to_dict() == {'酸枣仁': 500, '天麻': 405, '黄芪': 403, '枣仁': 364},
            'Current species counts disagree')
    require(sha(REGISTRY) == master['active_registry_sha256'], 'Registry differs from frozen master')
    require(sha(MANIFEST) == master['manifest_sha256'], 'Master roles changed')
    require(sha(ROOT / 'data/metadata_template.csv') == master['metadata_sha256'], 'Metadata changed')
    hashes = {}
    for entry in registry['workbooks']:
        path = ROOT / entry['path']
        require(path.is_file(), f'Missing active workbook: {entry["path"]}')
        actual = sha(path)
        require(actual == entry['sha256'] == master['workbook_hashes'][entry['path']],
                f'Active workbook hash changed: {entry["path"]}')
        hashes[entry['path']] = actual
        group = metadata.loc[metadata.origin_group_id.eq(entry['origin_group_id'])]
        require(len(group) == entry['record_count'], 'Workbook count/manifest disagree')
        require(group.species.eq(entry['species']).all(), 'Species/group mapping disagrees')
    for group in ('ZY01', 'ZY02'):
        require(metadata.loc[metadata.origin_group_id.eq(group), 'species'].eq('枣仁').all(), 'Wrong ZY kind')
    for group in ('ZY03', 'ZY04', 'ZY05', 'ZY06', 'ZY07'):
        require(metadata.loc[metadata.origin_group_id.eq(group), 'species'].eq('酸枣仁').all(), 'Wrong ZY kind')
    return registry, master, metadata, hashes


def feature_data(metadata, prepare=False):
    cache = OUTPUT / '.cache'
    path = cache / 'features.npz'
    if not path.exists():
        if not prepare:
            return None
        command = [sys.executable, str(ROOT / 'scripts/prepare_species_revision_features.py'),
                   '--root', str(ROOT), '--output', str(cache)]
        subprocess.run(command, cwd=ROOT, check=True)
    require((cache / 'feature_audit.json').exists(), 'Feature cache has no reconstruction audit')
    rebuilt = json.loads((cache / 'feature_audit.json').read_text())
    require(sha(path) == rebuilt['features_sha256'], 'Feature cache byte hash changed')
    require(rebuilt['registry_sha256'] == sha(REGISTRY), 'Feature registry mismatch')
    historical = json.loads((ROOT / 'data/research/species_revision_20261108/feature_audit.json').read_text())
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    ids = arrays['record_ids'].astype(str)
    require(len(ids) == len(set(ids)) == len(metadata), 'Feature IDs not unique or complete')
    require(set(ids) == set(metadata.record_id), 'Feature/manifest IDs disagree')
    aligned = metadata.set_index('record_id', drop=False).loc[ids].reset_index(drop=True)
    require(np.array_equal(arrays['species'], aligned.species), 'Feature species mismatch')
    require(np.array_equal(arrays['origin_group_ids'], aligned.origin_group_id), 'Feature origins mismatch')
    axis, X = arrays['axis'], arrays['X_alpha']
    require(X.shape == (1672, 612) and axis.shape == (612,), 'Expected original 612 alpha coordinates')
    require(np.isfinite(X).all() and np.isfinite(axis).all() and (np.diff(axis) > 0).all(),
            'Nonfinite spectrum or invalid coordinates')
    array_hashes = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for key, value in arrays.items()}
    same_bytes = sha(path) == historical['features_sha256']
    comparison = {
        'historical_npz_sha256': historical['features_sha256'], 'rebuilt_npz_sha256': sha(path),
        'byte_identical_to_historical_npz': same_bytes,
        'array_sha256': array_hashes,
        'historical_array_comparison': 'byte-identical archive establishes equal contents' if same_bytes else
            'Historical NPZ is absent: array equality with that absent archive is not established. '
            'Current arrays were independently reconstructed from identical workbook bytes and checked against '
            'historical axis/count/workbook audit plus distinct-reader source-cell checks.',
        'historical_workbook_checks_match': rebuilt['workbook_checks'] == historical['workbook_checks'],
        'independent_spot_checks': rebuilt.get('independent_spot_checks'),
        'shape': list(X.shape), 'axis_range': [float(axis[0]), float(axis[-1])],
    }
    require(comparison['historical_workbook_checks_match'], 'Historical workbook extraction audit differs')
    return X, axis, aligned, comparison, rebuilt


def preflight(prepare=False):
    registry, master, metadata, workbooks = base_inputs()
    from b_mechanism_engine_20261110 import build_design
    jobs = build_design(metadata)
    require(len(jobs) == 105, 'Expected 90 factorial + 15 five-source jobs')
    features = feature_data(metadata, prepare=prepare)
    train = metadata.loc[metadata.role.eq('train')]
    counts = train.groupby(['origin_group_id', 'species']).size().reset_index(name='train_count')
    runtime = {'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
               'disk_free_bytes': shutil.disk_usage(ROOT).free,
               'memory_bytes': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
               'single_thread_libraries': {name: os.environ[name] for name in
                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
               'dependencies': {name: importlib.metadata.version(name) for name in
                                ('numpy', 'scipy', 'scikit-learn', 'pandas', 'openpyxl', 'matplotlib')}}
    audit = {
        'experiment': 'B-mechanism-gaps-20261110', 'status': 'DONE' if features else 'PARTIAL',
        'base_commit': BASE_SHA, 'registry_sha256': sha(REGISTRY), 'master_manifest_sha256': sha(MANIFEST),
        'master_protocol_sha256': sha(MASTER_PROTOCOL), 'workbook_sha256': workbooks,
        'active_records': len(metadata), 'species_counts': metadata.species.value_counts().to_dict(),
        'master_role_counts': metadata.role.value_counts().to_dict(),
        'train_species_counts': train.species.value_counts().to_dict(),
        'train_group_counts': counts.to_dict('records'),
        'unique_physical_samples': int(metadata.physical_sample_id.nunique()),
        'physical_id_basis': metadata.physical_sample_id_basis.value_counts().to_dict(),
        'actual_batch_id_nonmissing': int(metadata.batch_id.ne('').sum()),
        'batch_scope': 'No true batch identifiers. Source group and collection batch are confounded.',
        'planned_source_models': len(jobs), 'planned_complete_pipeline_fits': len(jobs) * 5,
        'historical_audit': 'docs/B_MECHANISM_EXISTING_EVIDENCE_AUDIT.md',
        'feature_comparison': features[3] if features else {'status': 'NOT_REBUILT'},
        'environment': runtime,
        'compute_policy': {'original_requested_cpu_seconds': 10000, 'effective_cpu_stop_limit': None,
                           'authority': 'Latest user request: do not interrupt background progression using thresholds',
                           'no_quality_or_pilot_timing_gate': True, 'separate_new_cpu_ledger': True},
    }
    if prepare:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / '.gitignore').write_text('.cache/\n.background/\n*.lock\n*.log\n')
        atomic_json(OUTPUT / 'preflight.json', audit)
        atomic_json(OUTPUT / 'feature_audit_rebuilt.json', features[4])
        rows = []
        for job in jobs:
            rows.append({'job_id': job['job_id'], 'experiment': job['experiment'], 'status': 'FEASIBLE',
                         'source_n': len(job['source_ids']), 'support_n': len(job['support_ids']),
                         'query_n': len(job['query_ids']), 'sample_shortfall': 0,
                         'basis': 'All explicit IDs and roles checked by deterministic design builder'})
        pd.DataFrame(rows).to_csv(OUTPUT / 'feasibility.csv', index=False)
    return audit, jobs, features


def freeze(jobs, features):
    science_paths = [Path(__file__), ROOT / 'scripts/b_mechanism_engine_20261110.py']
    science_paths += sorted((ROOT / 'src/herb_spectra').glob('*.py'))
    inputs = {'registry': sha(REGISTRY), 'master_manifest': sha(MANIFEST),
              'master_protocol': sha(MASTER_PROTOCOL), 'array_sha256': features[3]['array_sha256'],
              'science_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in science_paths}}
    protocol = {'protocol_id': 'b-mechanism-20261110-v1', 'base_commit': BASE_SHA,
                'inputs': inputs, 'design': jobs, 'source_model': 'alpha612 SNV(ddof=1) StandardScaler RBF-SVM C10 gamma=scale balanced',
                'k_per_class': [0, 1, 3, 5, 10], 'outer_split_seeds': [20261031, 20261108, 42],
                'e01_sampling_seeds': [42, 43, 44],
                'methods': ['Identity', 'Bias-only', 'Anchor lambda1', 'Weak lambda1000 blend0.05', 'Full pipeline retrain'],
                'budget_policy': 'No automatic CPU, accuracy, wall-time or pilot gate by latest user request; record actual CPU independently',
                'query_policy': 'master TRAIN only; no Query-label or original CAL/PRED model selection',
                'e05': 'NOT_RUN_NEEDS_NEW_BATCH_DATA'}
    digest = canonical_hash(protocol)
    protocol['input_hash'] = digest
    path = OUTPUT / 'frozen_protocol.json'
    if path.exists():
        require(json.loads(path.read_text()) == protocol, 'Frozen scientific inputs/protocol/code changed; refusing unsafe resume')
    else:
        atomic_json(path, protocol)
        lookup = features[2].set_index('record_id')
        rows = []
        for job in jobs:
            for role in ('source', 'support', 'query'):
                for order, record_id in enumerate(job[role + '_ids']):
                    item = lookup.loc[record_id]
                    rows.append({'job_id': job['job_id'], 'experiment': job['experiment'],
                                 'role': role, 'order': order, 'record_id': record_id,
                                 'physical_sample_id': item.physical_sample_id,
                                 'origin_group_id': item.origin_group_id, 'species': item.species,
                                 'master_role': item.role})
        pd.DataFrame(rows).to_csv(OUTPUT / 'split_locks.csv', index=False)
    return digest


def usage():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def run_jobs(jobs, features, digest, pilot=False):
    from b_mechanism_engine_20261110 import execute_job, verify_job
    X, axis, metadata = features[:3]
    ledger_path = OUTPUT / 'execution_state.json'
    state = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'schema_version': 1, 'input_hash': digest, 'planned_jobs': len(jobs), 'attempts': [],
        'status': 'RUNNING', 'created_at_utc': now(), 'latest_user_no_cutoff_policy': True}
    require(state['input_hash'] == digest, 'CPU ledger belongs to changed inputs')
    selected = jobs[:1] if pilot else jobs
    completed = []
    for job in selected:
        destination = OUTPUT / 'jobs' / job['job_id']
        if (destination / 'result.json').exists():
            verify_job(destination, digest)
            completed.append(job['job_id'])
            continue
        attempt = {'job_id': job['job_id'], 'started_at_utc': now(), 'status': 'RUNNING'}
        state['attempts'].append(attempt)
        state.update(current_job=job['job_id'], heartbeat_at_utc=now())
        atomic_json(ledger_path, state)
        started, cpu = time.monotonic(), usage()
        try:
            execute_job(job, X, metadata, OUTPUT, digest, resume=True, axis=axis)
            verify_job(destination, digest)
            attempt['status'] = 'DONE'
            completed.append(job['job_id'])
        except Exception as error:
            # Isolate failures; a bad result never causes query/seed/protocol replacement.
            attempt['status'] = 'FAILED'
            attempt['error'] = repr(error)
            attempt['traceback'] = traceback.format_exc()
            print(traceback.format_exc(), flush=True)
        finally:
            attempt.update(finished_at_utc=now(), attempt_cpu_seconds=usage() - cpu,
                           wall_seconds=time.monotonic() - started)
            state.update(heartbeat_at_utc=now(),
                         completed_jobs=len(list((OUTPUT / 'jobs').glob('*/result.json'))),
                         measured_attempt_cpu_seconds=sum(a.get('attempt_cpu_seconds', 0) for a in state['attempts']))
            atomic_json(ledger_path, state)
            print(json.dumps({'job': job['job_id'], 'status': attempt['status'],
                              'completed': state['completed_jobs'], 'planned': len(jobs)}, ensure_ascii=False), flush=True)
    state.update(status='PILOT_DONE' if pilot else ('DONE' if len(completed) == len(jobs) else 'PARTIAL'),
                 current_job=None, heartbeat_at_utc=now())
    atomic_json(ledger_path, state)
    return state


def verify_all(jobs, digest):
    from b_mechanism_engine_20261110 import verify_job
    done = []
    for job in jobs:
        path = OUTPUT / 'jobs' / job['job_id']
        if (path / 'result.json').exists():
            verify_job(path, digest)
            done.append(job['job_id'])
    locks = pd.read_csv(OUTPUT / 'split_locks.csv', dtype=str, keep_default_na=False)
    require(locks.master_role.eq('train').all(), 'Non-TRAIN record used')
    for job in jobs:
        actual = locks.loc[locks.job_id.eq(job['job_id'])]
        for role in ('source', 'support', 'query'):
            require(actual.loc[actual.role.eq(role), 'record_id'].tolist() == job[role + '_ids'], 'Split lock changed')
    receipt = {'verified_at_utc': now(), 'input_hash': digest, 'verified_completed_jobs': len(done),
               'planned_jobs': len(jobs), 'status': 'DONE' if len(done) == len(jobs) else 'PARTIAL',
               'all_completed_artifacts_verified': True, 'independent_metrics': 'verify_job recomputes each saved prediction group'}
    atomic_json(OUTPUT / 'artifact_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='audit/rebuild features and freeze IDs without fitting')
    parser.add_argument('--pilot', action='store_true', help='one complete reusable task; no performance gate')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('choose --dry-run, --prepare, --pilot, --resume or --verify')
    try:
        if args.dry_run:
            audit, jobs, features = preflight(prepare=False)
            print(json.dumps({'preflight': audit, 'task_count': len(jobs), 'no_fits': True}, ensure_ascii=False, indent=2))
            return 0
        OUTPUT.mkdir(parents=True, exist_ok=True)
        lock = (OUTPUT / 'runner.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit, jobs, features = preflight(prepare=True)
        digest = freeze(jobs, features)
        if args.verify:
            verify_all(jobs, digest)
        elif args.prepare:
            print(json.dumps({'status': 'FROZEN_NO_FIT', 'input_hash': digest, 'planned_jobs': len(jobs)}))
        else:
            state = run_jobs(jobs, features, digest, pilot=args.pilot)
            if not args.pilot:
                # The supervisor handles reporting/publishing; missing jobs are retried without refitting completed jobs.
                return 0 if state['status'] == 'DONE' else 1
        return 0
    except (IntegrityError, AssertionError, ValueError) as error:
        print('INTEGRITY_BLOCKED: ' + str(error), file=sys.stderr, flush=True)
        return 42


if __name__ == '__main__':
    raise SystemExit(main())
