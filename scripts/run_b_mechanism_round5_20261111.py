#!/usr/bin/env python3
"""Fifth frozen B study: honest policy selection, acquisition and score geometry."""
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
OUTPUT = ROOT / 'reports/b_mechanism_round5_20261111'
OLDROOT = ROOT / 'reports/b_mechanism_20261110'
ROUND2ROOT = ROOT / 'reports/b_mechanism_followup_20261110'
ROUND3ROOT = ROOT / 'reports/b_mechanism_round3_20261110'
ROUND4ROOT = ROOT / 'reports/b_mechanism_round4_20261111'
BASE_SHA = 'bbab5a9de3f3e79fe8b12dd5e06c29547e3b9778'
import run_b_mechanism_gap_20261110 as initial
import b_round5_selector_20261111 as selector
import b_round5_acquisition_20261111 as acquisition
import b_round5_scores_20261111 as scores
from b_round5_common_20261111 import canonical_hash, file_hash, validate_roles, verify_task


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    initial.atomic_json(path, value)


def build_design(metadata, X=None, axis=None, old_jobs=None):
    if X is None or axis is None:
        cached = initial.feature_data(metadata, prepare=False)
        if cached is None:
            raise ValueError('A verified feature cache is required to freeze label-blind acquisition')
        X, axis, metadata = cached[:3]
    if old_jobs is None:
        old_jobs = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    selected = [job for job in old_jobs if job['experiment'] == 'E03' or job.get('ratio') == 'B']
    if len(selected) != 45:
        raise ValueError('Expected the complete 15 five-origin and 30 balanced binary contexts')
    round4_jobs = json.loads((ROUND4ROOT / 'frozen_protocol.json').read_text())['design']
    old_compositions = {job['parent_job_id']: job['composition_plan'] for job in round4_jobs if job['family'] == 'H04'}
    jobs = []
    for family in ('I01', 'I02', 'I03', 'I04'):
        for old in selected:
            job = copy.deepcopy(old)
            job.update(job_id=family + '_' + old['job_id'], family=family,
                       parent_job_id=old['job_id'], phase2_job_id='F01_' + old['job_id'],
                       phase3_job_id='G01_' + old['job_id'], phase4_job_id='H04_' + old['job_id'])
            if family == 'I01':
                job['inner_plan'] = selector.inner_plan(job, metadata)
            elif family == 'I02':
                job['acquisition_plan'] = acquisition.acquisition_plan(
                    job, X, metadata, axis, OLDROOT, ROUND2ROOT, ROUND3ROOT, ROUND4ROOT)
            elif family == 'I04':
                job['composition_plan'] = copy.deepcopy(old_compositions[old['job_id']])
                job['inherited_composition_plan_source'] = 'H04_' + old['job_id']
                job['planned_current_adapter_fits'] = scores.adapter_count(job)
            validate_roles(job, metadata)
            jobs.append(job)
    if len(jobs) != 180 or len({job['job_id'] for job in jobs}) != 180:
        raise ValueError('Expected 180 unique fifth-round task bundles')
    return jobs


def executor(job):
    return selector if job['family'] == 'I01' else acquisition if job['family'] == 'I02' else scores


def expected_endpoints(job):
    return executor(job).expected_endpoints(job)


def previous_evidence():
    evidence = {}
    roots = (('first', OLDROOT), ('second', ROUND2ROOT), ('third', ROUND3ROOT), ('fourth', ROUND4ROOT))
    for name, root in roots:
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
    registry, master, metadata, workbooks = initial.base_inputs()
    features = initial.feature_data(metadata, prepare=write)
    if features is None:
        raise ValueError('Verified feature cache is missing; reconstruct with --prepare')
    X, axis, aligned = features[:3]
    jobs = build_design(aligned, X, axis)
    model_count = sum(executor(job).fit_count(job) for job in jobs)
    adapter_count = sum(executor(job).adapter_count(job) for job in jobs)
    evidence = previous_evidence()
    names = ('run_b_mechanism_round5_20261111.py', 'b_round5_common_20261111.py',
             'b_round5_selector_20261111.py', 'b_round5_acquisition_20261111.py', 'b_round5_scores_20261111.py',
             'b_round4_common_20261111.py', 'b_round4_models_20261111.py', 'b_round4_support_20261111.py',
             'b_round3_common_20261110.py', 'b_round3_regularization_20261110.py',
             'b_followup_common_20261110.py', 'b_followup_models_20261110.py', 'b_followup_calibration_20261110.py',
             'b_mechanism_engine_20261110.py', 'run_b_mechanism_gap_20261110.py', 'prepare_species_revision_features.py')
    paths = [ROOT / 'scripts' / name for name in names] + sorted((ROOT / 'src/herb_spectra').glob('*.py'))
    protocol = {
        'protocol_id': 'b-mechanism-round5-20261111-v1', 'base_commit': BASE_SHA,
        'registry_sha256': initial.sha(initial.REGISTRY), 'master_manifest_sha256': initial.sha(initial.MANIFEST),
        'workbook_sha256': workbooks, 'array_sha256': features[3]['array_sha256'], 'prior_evidence': evidence,
        'science_code_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
        'runtime': {'python': sys.version, 'libraries': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib')}},
        'design': jobs, 'endpoint_contract': {job['job_id']: expected_endpoints(job) for job in jobs},
        'planned_new_model_fits': model_count, 'planned_new_adapter_fits': adapter_count,
        'I01': {'scope': 'selection uses outer Source only; outer Support/Query excluded',
                'episode_seeds': [501, 502, 503],
                'pure_per_class_roles': {'inner_source': 20, 'inner_support': 10, 'inner_validation': 8},
                'binary_per_class_roles': {'inner_source': 90, 'inner_support': 10, 'inner_validation': 20},
                'candidate_methods': ['identity', 'bias', 'mean_anchor', 'full_refit'],
                'choice': 'maximum mean inner validation Macro-F1 per K; round12 ties prefer identity,bias,mean_anchor,full_refit',
                'outer_application': 'choose first, then reuse exact saved outer candidate endpoint',
                'limitation': 'smaller Source-only inner problems may not transport to target Support domain'},
        'I02': {'ranking_inputs': 'candidate spectra in frozen Source coordinates, Source scores and opaque hash keys; no candidate labels/Query',
                'policies': ['random_601', 'random_602', 'random_603', 'uncertainty', 'diversity'],
                'total_label_budgets': 'n_classes times [1,3,5,10]',
                'ranking': 'nested label-blind order; diversity is max-min coverage anchored by the Source centroid; class counts revealed after all orders fixed',
                'full_refit': 'C10, Source-frozen scaler/gamma, class_weight None to hold per-point C fixed',
                'guarded_mean_anchor': 'sum-loss lambda=number selected; Identity fallback if any Support class missing',
                'reuse': 'complete/canonical historical subsets and repeated canonical ordered-ID objectives do not refit',
                'scope': 'candidate pool was historically class-stratified; only within-pool ranking is label-blind'},
        'I03': {'output_anchor': 'original OVR Source scores; no changed OVO argmax baseline',
                'residual_input': ['Source-centered trace1 OVR', 'Source-centered trace1 OVO', 'Source-only OVO PCA min(C,P) trace1'],
                'lambda': 'sum loss N, equivalent to mean residual loss with unit ridge prior',
                'episodes': 'draw0/101/102, K1/3/5/10 nested; complete K10 only draw0',
                'negative_control': 'binary normalized feature Grams are algebraically equivalent; solver-coordinate fits are numerical controls, not independent algorithms',
                'basis_cost': 'Source feature-statistic/PCA fits recorded separately from spectral classifier and calibration fits'},
        'I04': {'same_ids': 'exact inherited H04 Support compositions and draws',
                'weights': 'N/(n_classes*n_support_class), total N',
                'lambda_conditions': ['historical weighted lambda1', 'mean loss: N', 'ESS heuristic: N*N/ESS'],
                'ESS': '(sum weights)^2/sum squared weights; weight concentration, not independent physical sample count',
                'balanced_reuse': 'ESS=N gives equal lambda N; reuse prior G01 mean K5 endpoints',
                'heuristic_scope': 'prespecified extra shrinkage, not an optimized or universally valid rule',
                'legacy_plan_costs': 'inherited H04 plan fit-count fields are provenance only; current counts use executor/feasibility/sealed results'},
        'inference': 'Exploratory master TRAIN, all physical samples historically exposed; no new independent batch evidence or Query-based method selection',
        'execution': {'single_thread': True, 'quality_cpu_wall_time_stop_cap': None, 'finite_frozen_matrix': True}}
    protocol['input_hash'] = canonical_hash(protocol)
    if write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / '.gitignore').write_text('.background/\n.cache/\n*.lock\n*.log\n')
        target = OUTPUT / 'frozen_protocol.json'
        if target.exists() and json.loads(target.read_text()) != protocol:
            raise ValueError('Frozen fifth-round inputs, plans, code or endpoint contract changed')
        if not target.exists():
            atomic_json(target, protocol)
            indexed = aligned.set_index('record_id', drop=False)
            rows = []
            for job in jobs:
                for role in ('source', 'support', 'query'):
                    for order, record_id in enumerate(job[role + '_ids']):
                        row = indexed.loc[record_id]
                        rows.append({'job_id': job['job_id'], 'family': job['family'], 'role': role,
                                     'order': order, 'record_id': record_id, 'physical_sample_id': row.physical_sample_id,
                                     'origin_group_id': row.origin_group_id, 'species': row.species, 'master_role': row.role})
            pd.DataFrame(rows).to_csv(OUTPUT / 'split_locks.csv', index=False)
        atomic_json(OUTPUT / 'preflight.json', {
            'status': 'DONE', 'base_commit': BASE_SHA, 'previous_evidence': evidence,
            'active_records': len(metadata), 'species_counts': metadata.species.value_counts().to_dict(),
            'master_role_counts': metadata.role.value_counts().to_dict(), 'sample_shortfalls': [],
            'feature_cache_byte_identical': features[3]['byte_identical_to_historical_npz'],
            'actual_batch_id_count': int(metadata.batch_id.ne('').sum()), 'planned_tasks': len(jobs),
            'planned_new_model_fits': model_count, 'planned_new_adapter_fits': adapter_count,
            'no_score_cpu_time_gate': True})
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
                raise ValueError('Task fitting/reuse counts disagree with frozen fifth-round plan')
            completed += 1
            new_models += result['new_model_fits']
            new_adapters += result.get('new_adapter_fits', 0)
    locks = pd.read_csv(OUTPUT / 'split_locks.csv', dtype=str, keep_default_na=False)
    if not locks.master_role.eq('train').all():
        raise ValueError('Non-TRAIN identity in frozen fifth-round role table')
    for job in protocol['design']:
        block = locks.loc[locks.job_id.eq(job['job_id'])]
        for role in ('source', 'support', 'query'):
            if block.loc[block.role.eq(role), 'record_id'].tolist() != job[role + '_ids']:
                raise ValueError('Frozen fifth-round role lock changed')
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
        raise ValueError('Fifth-round attempt ledger has different inputs')
    jobs = protocol['design']
    if pilot:
        jobs = [next(job for job in jobs if job['family'] == family) for family in ('I01', 'I02', 'I03', 'I04')]
    state.update(status='RUNNING', quality_cpu_runtime_cutoffs=None)
    completed = 0
    for job in jobs:
        target = OUTPUT / 'jobs' / job['job_id']
        if target.exists():
            verify_task(target, protocol['input_hash'], job, expected_endpoints(job))
            completed += 1
            continue
        attempt = {'job_id': job['job_id'], 'family': job['family'], 'started_at_utc': now(), 'status': 'RUNNING'}
        state['attempts'].append(attempt)
        state.update(current_job=job['job_id'], heartbeat_at_utc=now())
        atomic_json(path, state)
        cpu, wall = time.process_time(), time.monotonic()
        try:
            executor(job).execute(job, X, metadata, axis, OUTPUT, protocol['input_hash'], OLDROOT, ROUND2ROOT, ROUND3ROOT, ROUND4ROOT)
            verify_task(target, protocol['input_hash'], job, expected_endpoints(job))
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
    state.update(status=('PILOT_DONE' if pilot else 'DONE') if completed == len(jobs) else 'PARTIAL',
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
