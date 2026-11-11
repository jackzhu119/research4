#!/usr/bin/env python3
"""Independently summarize sealed round-five predictions; never fit a model.

Intervals resample physical samples while carrying their repeated outer Query
predictions together. Inner Source episodes and acquisition policies are not
independent new specimens or a fresh external validation set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import summarize_b_mechanism_followup_20261110 as shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'reports/b_mechanism_round5_20261111'
REPORT = ROOT / 'docs/B_MECHANISM_ROUND5_RESULTS_20261111.md'
FAMILIES = {'I01': 45, 'I02': 45, 'I03': 45, 'I04': 45}
KS = (1, 3, 5, 10)
SELECTOR_CANDIDATES = ('identity', 'bias', 'mean_anchor', 'full_refit')
SCORE_METHODS = ('raw_mean_anchor', 'ovr_trace1', 'ovo_trace1', 'ovo_pca_trace1')
ESS_METHODS = ('weighted_lambda1', 'weighted_mean', 'weighted_ess')
LIMITS = (
    '本轮根据前四轮已曝光结果设计，所有实验仅使用master TRAIN，仍属探索性内部研究。'
    '内层Source重复划分、Support获取顺序、输入几何及收缩条件不增加独立实物数。'
    '描述性95%区间按physical_sample_id聚类并保留相关预测，不覆盖来源/批次总体、'
    '协议选择和长期开发偏倚；来源与批次混杂，不能称真实新批次验证或确认性因果结论。'
)


def protocol_payload(output):
    path = Path(output) / 'frozen_protocol.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def truthy(values):
    return values.astype(str).str.lower().isin(('true', '1', '1.0'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def numeric(values):
    return pd.to_numeric(values, errors='coerce')


def _read_csv(path):
    try:
        return pd.read_csv(path, keep_default_na=False, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _method_rows(rows, method, k=None, draw=None):
    selected = rows[rows.method.eq(method)]
    if k is not None:
        selected = selected[selected.k.eq(k)]
    if draw is not None:
        selected = selected[selected.draw.eq(draw)]
    return selected


def _check_same_endpoint(left, right, message):
    if len(left) != len(right) or set(left.physical_sample_id) != set(right.physical_sample_id):
        raise ValueError(message + ': Query coverage differs')
    a = left.set_index('physical_sample_id')
    b = right.set_index('physical_sample_id').loc[a.index]
    if not a.pred_label.eq(b.pred_label).all():
        raise ValueError(message + ': predictions differ')
    if 'scores_json' in a and 'scores_json' in b:
        av = np.asarray([json.loads(value) for value in a.scores_json])
        bv = np.asarray([json.loads(value) for value in b.scores_json])
        if not np.allclose(av, bv, rtol=0, atol=1e-12):
            raise ValueError(message + ': saved scores differ')
    return len(a)


def _verify_selector(directory, job, outer_predictions):
    """Independently verify Source-only roles, inner metrics and frozen choice."""
    policy_path = directory / 'source_policy.json'
    plan_path = directory / 'inner_plan.json'
    if not policy_path.is_file() or not plan_path.is_file():
        raise ValueError('Complete I01 bundle lacks its Source-only selection evidence')
    plan, policy = json.loads(plan_path.read_text()), json.loads(policy_path.read_text())
    if job.get('inner_plan') is not None and plan != job['inner_plan']:
        raise ValueError('I01 inner role plan differs from the frozen task')
    source = set(job['source_ids'])
    excluded = set(job['support_ids']) | set(job['query_ids'])
    episode_map = {}
    for episode in plan['episodes']:
        seed = int(episode['episode_seed'])
        parts = [episode[name] for name in ('inner_source_ids', 'inner_support_ids', 'inner_validation_ids')]
        sets = [set(part) for part in parts]
        if (any(len(part) != len(unique) for part, unique in zip(parts, sets))
                or sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
                or set.union(*sets) != source or set.union(*sets) & excluded):
            raise ValueError('I01 Source-only role exclusion or disjointness failed')
        episode_map[seed] = episode
    if set(episode_map) != {501, 502, 503}:
        raise ValueError('I01 Source episode seed coverage differs')
    metric_path = directory / 'inner_validation_metrics.csv'
    prediction_path = directory / 'inner_validation_predictions.csv.gz'
    if not metric_path.is_file() or not prediction_path.is_file():
        raise ValueError('I01 inner validation predictions/metrics are missing')
    inner_metrics, inner_predictions = _read_csv(metric_path), _read_csv(prediction_path)
    required = {'episode_seed', 'k', 'method', 'record_id', 'true_label', 'pred_label'}
    if not required.issubset(inner_predictions):
        raise ValueError('I01 inner prediction fields missing: ' + str(required - set(inner_predictions)))
    required_metric = {'episode_seed', 'k', 'method', 'macro_f1', 'accuracy', 'balanced_accuracy'}
    if not required_metric.issubset(inner_metrics):
        raise ValueError('I01 inner metric fields missing')
    inner_metrics = inner_metrics.copy()
    inner_metrics['episode_seed'] = numeric(inner_metrics.episode_seed).astype(int)
    inner_metrics['k'] = numeric(inner_metrics.k).astype(int)
    inner_predictions = inner_predictions.copy()
    inner_predictions['episode_seed'] = numeric(inner_predictions.episode_seed).astype(int)
    inner_predictions['k'] = numeric(inner_predictions.k).astype(int)
    positive = inner_metrics[inner_metrics.k.isin(KS)]
    expected = {(seed, k, method) for seed in episode_map for k in KS for method in SELECTOR_CANDIDATES}
    if (set(zip(positive.episode_seed, positive.k, positive.method)) != expected
            or positive.duplicated(['episode_seed', 'k', 'method']).any()):
        raise ValueError('I01 inner candidate/K/episode matrix incomplete')
    checks = 0
    for (seed, k, method), rows in inner_predictions[inner_predictions.k.isin(KS)].groupby(
            ['episode_seed', 'k', 'method'], sort=False):
        if (int(seed), int(k), method) not in expected:
            raise ValueError('I01 unplanned inner validation endpoint')
        if len(rows) != len(set(rows.record_id)) or set(rows.record_id) != set(episode_map[int(seed)]['inner_validation_ids']):
            raise ValueError('I01 inner validation ID denominator differs')
        rows = rows.copy()
        if 'physical_sample_id' not in rows:
            rows['physical_sample_id'] = rows.record_id
        fresh = shared.independent_metrics(rows, job['classes'])
        saved = positive[(positive.episode_seed.eq(seed)) & positive.k.eq(k) & positive.method.eq(method)]
        if len(saved) != 1:
            raise ValueError('I01 duplicated inner metric row')
        for field in ('macro_f1', 'accuracy', 'balanced_accuracy'):
            if not np.isclose(float(saved.iloc[0][field]), fresh[field], atol=1e-12, rtol=0):
                raise ValueError('Independent I01 validation metric mismatch: ' + field)
            checks += 1
    if set(zip(inner_predictions.episode_seed, inner_predictions.k, inner_predictions.method)) & expected != expected:
        raise ValueError('I01 inner prediction candidate coverage is incomplete')
    choices = []
    for seed, episode in episode_map.items():
        choices.append({'analysis': 'source_only_inner_role_counts', 'family': 'I01', 'job_id': job['job_id'],
                        'experiment': job['experiment'], 'episode_seed': seed,
                        'outer_source_count': len(source),
                        'inner_source_count': len(episode['inner_source_ids']),
                        'inner_support_count': len(episode['inner_support_ids']),
                        'inner_validation_count': len(episode['inner_validation_ids']),
                        'inner_source_class_counts_json': json.dumps(episode.get('source_counts', {}), ensure_ascii=False),
                        'new_independent_physical_samples': 0,
                        'outer_support_query_excluded': True})
    for first, left in episode_map.items():
        for second, right in episode_map.items():
            if first >= second:
                continue
            choices.append({'analysis': 'source_inner_episode_overlap', 'family': 'I01', 'job_id': job['job_id'],
                            'experiment': job['experiment'], 'episode_seed_before': first,
                            'episode_seed_after': second,
                            'shared_inner_source_ids': len(set(left['inner_source_ids']) & set(right['inner_source_ids'])),
                            'shared_inner_support_ids': len(set(left['inner_support_ids']) & set(right['inner_support_ids'])),
                            'shared_inner_validation_ids': len(set(left['inner_validation_ids']) & set(right['inner_validation_ids'])),
                            'validation_before_used_as_source_after': len(set(left['inner_validation_ids']) & set(right['inner_source_ids'])),
                            'validation_after_used_as_source_before': len(set(right['inner_validation_ids']) & set(left['inner_source_ids'])),
                            'interpretation': 'Source holdouts reuse the same outer Source specimens; episodes are dependent internal repetitions'})
    for k in KS:
        scored = positive[positive.k.eq(k)].groupby('method').macro_f1.mean().to_dict()
        rounded = {name: round(float(scored[name]), 12) for name in SELECTOR_CANDIDATES}
        selected = max(SELECTOR_CANDIDATES, key=lambda name: (rounded[name], -SELECTOR_CANDIDATES.index(name)))
        record = policy[str(k)]
        if record['selected_method'] != selected:
            raise ValueError('I01 Source-only policy does not follow the frozen validation/tie rule')
        reported = record.get('mean_validation_macro_f1_by_method', {})
        for name, score in reported.items():
            if name in scored and not np.isclose(float(score), float(scored[name]), atol=1e-12, rtol=0):
                raise ValueError('I01 recorded validation score differs from independent average')
        chosen = _method_rows(outer_predictions, 'source_selected', k)
        reference = _method_rows(outer_predictions, selected, 0 if selected == 'identity' else k)
        checks += _check_same_endpoint(reference, chosen, 'I01 selected outer candidate does not match Source-only choice')
        choices.append({'analysis': 'source_policy_choice', 'family': 'I01', 'job_id': job['job_id'],
                        'experiment': job['experiment'], 'k': k, 'selected_method': selected,
                        'mean_validation_macro_f1': float(scored[selected]),
                        'all_validation_scores_json': json.dumps(scored, sort_keys=True),
                        'inner_episode_count': len(episode_map),
                        'inner_validation_predictions': int((inner_predictions.k.eq(k) & inner_predictions.method.eq(selected)).sum()),
                        'inner_validation_unique_source_records': int(inner_predictions.loc[
                            inner_predictions.k.eq(k) & inner_predictions.method.eq(selected), 'record_id'].nunique()),
                        'selection_scope': 'outer_Source_only; no outer Support/Query labels',
                        'interpretation': 'Maximum inner validation score is a selection statistic, not an unbiased outer performance estimate'})
    return checks, pd.DataFrame(choices)


def _verify_acquisition(directory, job, predictions, metrics):
    plan = job.get('acquisition_plan')
    path = directory / 'acquisition_plan.json'
    if path.is_file():
        saved = json.loads(path.read_text())
        if plan is not None and saved != plan:
            raise ValueError('I02 acquisition plan differs from frozen task')
        plan = saved
    if plan is None:
        raise ValueError('Complete I02 bundle has no acquisition plan')
    pool = job['support_ids']
    episodes = plan.get('episodes', [])
    if len(episodes) != 20:
        raise ValueError('I02 expected five policies and four budgets')
    if (plan.get('selection_uses_candidate_true_labels', False)
            or plan.get('selection_uses_query', False)
            or not plan.get('labels_revealed_after_all_rankings', True)):
        raise ValueError('I02 frozen acquisition plan violates the label-blind selector interface')
    expected_policies = {'random_601', 'random_602', 'random_603', 'uncertainty', 'diversity'}
    expected_cells = {(policy, len(job['classes']) * k) for policy in expected_policies for k in KS}
    if (set((item['policy'], int(item['budget'])) for item in episodes) != expected_cells
            or len({(item['policy'], int(item['budget'])) for item in episodes}) != len(episodes)):
        raise ValueError('I02 policy/total-budget matrix differs from frozen design')
    endpoint_path = directory / 'endpoints.json'
    endpoints = {item['endpoint_id']: item for item in json.loads(endpoint_path.read_text())}
    episode_by_cell = {(item['policy'], int(item['budget'])): item for item in episodes}
    for episode in episodes:
        selected = episode.get('selected_ids', episode.get('selection_ids'))
        adapted = episode.get('adaptation_ids')
        budget = int(episode['budget'])
        if (selected is None or adapted is None or len(selected) != budget
                or len(set(selected)) != budget or set(selected) != set(adapted)
                or not set(selected).issubset(pool) or adapted != [sample for sample in pool if sample in set(selected)]):
            raise ValueError('I02 selected identities or canonical adaptation order changed')
    for policy, episodes_by_policy in pd.DataFrame(episodes).groupby('policy'):
        episodes_by_policy = episodes_by_policy.sort_values('budget')
        prior = set()
        for episode in episodes_by_policy.to_dict('records'):
            current = set(episode.get('selected_ids', episode.get('selection_ids')))
            if not prior.issubset(current):
                raise ValueError('I02 label budget selections are not nested')
            prior = current
    identity = _method_rows(predictions, 'identity', 0)
    checks = 0
    guarded = metrics[metrics.method.eq('guarded_mean_anchor')]
    for metric in metrics[metrics.k.eq(-1)].to_dict('records'):
        cell = (metric['policy'], int(float(metric['budget'])))
        if cell not in episode_by_cell:
            raise ValueError('I02 saved endpoint is outside the frozen policy/budget cells')
        episode = episode_by_cell[cell]
        endpoint = endpoints[metric['endpoint_id']]
        if endpoint['support_ids'] != episode['adaptation_ids']:
            raise ValueError('I02 endpoint used different acquired Support identities')
        if json.loads(metric['support_counts_json']) != episode['support_counts']:
            raise ValueError('I02 endpoint class counts differ from the post-ranking reveal')
        if int(float(metric.get('budget_labels_observed', episode['budget']))) != episode['budget']:
            raise ValueError('I02 acquired label cost differs from its fixed budget')
        if (metric['method'] == 'full_refit'
                and int(float(metric.get('labels_used_in_classifier_fit', episode['budget']))) != episode['budget']):
            raise ValueError('I02 classifier fit omitted acquired labels from its cost ledger')
        checks += 4
    for metric in guarded.to_dict('records'):
        budget = int(float(metric['budget']))
        present = int(float(metric['observed_support_class_count']))
        fallback = str(metric.get('guard_fallback_identity', '')).lower() in ('true', '1', '1.0')
        if fallback != (present < len(job['classes'])):
            raise ValueError('I02 predeclared missing-class fallback differs from observed coverage')
        if int(float(metric.get('labels_observed_for_guard', budget))) != budget:
            raise ValueError('I02 guard label cost was not counted')
        expected_fitted = 0 if fallback else budget
        if int(float(metric.get('labels_used_in_adapter_fit', expected_fitted))) != expected_fitted:
            raise ValueError('I02 fitted-adapter label usage differs')
        if fallback:
            rows = predictions[predictions.endpoint_id.eq(metric['endpoint_id'])]
            checks += _check_same_endpoint(identity, rows, 'I02 fallback is not the matched Source identity')
        counts = json.loads(metric['support_counts_json'])
        if sum(int(value) for value in counts.values()) != budget or sum(int(value) > 0 for value in counts.values()) != present:
            raise ValueError('I02 observed class coverage/counts differ from actual budget')
        checks += 4
    full_budget = len(pool)
    for method in ('full_refit', 'guarded_mean_anchor'):
        metrics_at_full = metrics[metrics.method.eq(method) & numeric(metrics.budget).eq(full_budget)]
        if len(metrics_at_full) != 5:
            raise ValueError('I02 full-pool policy coverage is incomplete')
        reference = predictions[predictions.endpoint_id.eq(metrics_at_full.iloc[0].endpoint_id)]
        for identifier in metrics_at_full.endpoint_id.iloc[1:]:
            checks += _check_same_endpoint(reference, predictions[predictions.endpoint_id.eq(identifier)],
                                           'I02 full-pool policies failed exact-objective reuse')
    return checks


def load_jobs(output):
    jobs, predictions, metrics, diagnostics, manifest, checks = shared.load_jobs(output)
    for field in ('budget', 'policy_index', 'nominal_shots_per_class', 'composition_index', 'draw_seed',
                  'effective_support_weight_ess', 'regularization', 'observed_support_class_count'):
        if field in metrics:
            metrics[field] = numeric(metrics[field])
    protocol = protocol_payload(output)
    contract = protocol.get('endpoint_contract', {})
    additional, selected = [], []
    for job in jobs:
        directory = Path(output) / 'jobs' / job['job_id']
        role_path = directory / 'split_ids.json'
        if role_path.is_file() and canonical_hash(json.loads(role_path.read_text())) != job['job_hash']:
            raise ValueError('Round-five frozen role definition differs from task hash')
        current = predictions[predictions.job_id.eq(job['job_id'])]
        current_metrics = metrics[metrics.job_id.eq(job['job_id'])]
        if job['job_id'] in contract and set(current.endpoint_id) != set(contract[job['job_id']]):
            raise ValueError('Round-five endpoint coverage differs from frozen protocol')
        for identifier, rows in current.groupby('endpoint_id', sort=False):
            if 'decision_rule' in rows and rows.decision_rule.eq('score_argmax').all():
                scores = np.asarray([json.loads(value) for value in rows.scores_json])
                expected = np.asarray(job['classes'])[scores.argmax(axis=1)]
                if not np.array_equal(expected, rows.pred_label.to_numpy()):
                    raise ValueError('Saved argmax predictions do not match saved class-aligned scores')
                checks += len(rows)
        if job['family'] == 'I01':
            count, choices = _verify_selector(directory, job, current)
            checks += count
            selected.append(choices)
        if job['family'] == 'I02':
            checks += _verify_acquisition(directory, job, current, current_metrics)
        for filename in ('score_geometry.csv', 'ess_geometry.csv', 'acquisition_coverage.csv', 'acquisition_diagnostics.csv',
                         'inner_fit_costs.csv', 'selection_diagnostics.csv'):
            candidate = directory / filename
            if not candidate.is_file():
                continue
            rows = _read_csv(candidate)
            if rows.empty:
                continue
            for key in ('family', 'job_id', 'experiment', 'target_origin', 'seed', 'ratio', 'coverage', 'fold'):
                rows[key] = job.get(key)
            if 'analysis' not in rows:
                rows['analysis'] = filename.removesuffix('.csv')
            additional.append(rows)
        inner_cost_path = directory / 'inner_model_costs.json'
        if inner_cost_path.is_file():
            frame = pd.DataFrame(json.loads(inner_cost_path.read_text()))
            if len(frame):
                frame['analysis'] = 'source_only_inner_classifier_fit_cost'
                frame['family'], frame['experiment'], frame['job_id'] = job['family'], job['experiment'], job['job_id']
                additional.append(frame)
        binary_path = directory / 'binary_equivalence.json'
        if binary_path.is_file():
            equivalence = json.loads(binary_path.read_text())
            if isinstance(equivalence, dict):
                equivalence = equivalence.get('comparisons', equivalence.get('records',
                                             equivalence.get('episodes', [equivalence])))
            frame = pd.DataFrame(equivalence)
            if len(frame):
                frame['analysis'] = 'saved_binary_feature_equivalence'
                frame['family'], frame['experiment'], frame['job_id'] = job['family'], job['experiment'], job['job_id']
                additional.append(frame)
    if selected:
        additional += selected
    if additional:
        diagnostics = pd.concat(([diagnostics] if not diagnostics.empty else []) + additional, ignore_index=True)
    if not predictions.empty:
        # Every experimental descriptor is constant within a saved endpoint.
        additions = [field for field in metrics if field not in predictions and field not in
                     ('accuracy', 'macro_f1', 'balanced_accuracy', 'n_query', 'n_predictions', 'n_physical_samples',
                      'correct', 'recall_json', 'class_recall_json', 'confusion_matrix_json', 'classes_json')]
        if additions:
            predictions = predictions.merge(metrics[['job_id', 'endpoint_id'] + additions],
                                            on=['job_id', 'endpoint_id'], validate='many_to_one')
        for field in ('budget', 'policy_index', 'nominal_shots_per_class', 'composition_index', 'draw_seed',
                      'effective_support_weight_ess', 'regularization', 'observed_support_class_count'):
            if field in predictions:
                predictions[field] = numeric(predictions[field])
    if len(manifest):
        job_map = {job['job_id']: job for job in jobs}
        for role in ('source', 'support', 'query'):
            manifest['frozen_' + role + '_count'] = manifest.job_id.map(
                lambda identifier: len(job_map[identifier][role + '_ids']))
        manifest['frozen_source_class_counts_json'] = manifest.job_id.map(
            lambda identifier: json.dumps(job_map[identifier].get('source_class_counts', {}), ensure_ascii=False, sort_keys=True))
    return jobs, predictions, metrics, diagnostics, manifest, checks


def comparison_tables(predictions, n_boot=1000):
    transitions, intervals = [], []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    def add(left, right, info, keys, repeat=False):
        if left.empty or right.empty:
            return
        common = set(left.job_id) & set(right.job_id)
        left, right = left[left.job_id.isin(common)], right[right.job_id.isin(common)]
        if left.empty:
            return
        if repeat:
            left = shared.repeat_baseline(left, right)
        rows = shared.pair(left, right, keys)
        for scope, block in [('ALL', rows)] + [(str(origin), block) for origin, block in rows.groupby('target_origin')]:
            record, interval = shared.transition_record(block, {**info, 'scope': scope}, n_boot)
            interval['interval_scope'] = LIMITS
            transitions.append(record); intervals.append(interval)

    keys = ['job_id', 'draw', 'target_origin', 'physical_sample_id']
    for experiment, rows in predictions[predictions.family.eq('I01')].groupby('experiment'):
        for k in KS:
            for method in ('bias', 'mean_anchor', 'full_refit', 'source_selected'):
                block = _method_rows(rows, method, k)
                if method == 'full_refit':
                    baseline = _method_rows(rows, 'svc_predict', 0)
                elif method == 'source_selected' and 'selected_method' in block:
                    native_jobs = set(block.loc[block.selected_method.eq('full_refit'), 'job_id'])
                    baseline = pd.concat([_method_rows(rows, 'svc_predict', 0).query('job_id in @native_jobs'),
                                          _method_rows(rows, 'identity', 0).query('job_id not in @native_jobs')], ignore_index=True)
                else:
                    baseline = _method_rows(rows, 'identity', 0)
                add(baseline, block, {'family': 'I01', 'experiment': experiment, 'method': method, 'k': k,
                                     'comparison': method + '_minus_matched_source_control'}, keys, repeat=True)
            for method in ('bias', 'mean_anchor', 'full_refit'):
                add(_method_rows(rows, method, k), _method_rows(rows, 'source_selected', k),
                    {'family': 'I01', 'experiment': experiment, 'k': k,
                     'comparison': 'source_selected_minus_' + method}, keys)

    for experiment, rows in predictions[predictions.family.eq('I02')].groupby('experiment'):
        if 'budget' not in rows or 'policy' not in rows:
            continue
        for (method, budget, policy), block in rows[rows.k.eq(-1)].groupby(['method', 'budget', 'policy']):
            baseline = _method_rows(rows, 'svc_predict' if method == 'full_refit' else 'identity', 0)
            add(baseline, block, {'family': 'I02', 'experiment': experiment, 'method': method, 'k': -1,
                                 'budget': int(budget), 'policy': policy,
                                 'comparison': method + '_minus_matched_source_control'}, keys, repeat=True)
        for budget, budget_rows in rows[rows.k.eq(-1)].groupby('budget'):
            for policy, block in budget_rows.groupby('policy'):
                add(_method_rows(block, 'guarded_mean_anchor'), _method_rows(block, 'full_refit'),
                    {'family': 'I02', 'experiment': experiment, 'budget': int(budget), 'policy': policy,
                     'comparison': 'full_refit_minus_guarded_mean_anchor'}, keys)
            randoms = [name for name in budget_rows.policy.unique() if str(name).startswith('random_')]
            for method in ('full_refit', 'guarded_mean_anchor'):
                for comparison_policy in ('uncertainty', 'diversity'):
                    for random_policy in sorted(randoms):
                        method_rows = _method_rows(budget_rows, method)
                        left = method_rows[method_rows.policy.eq(random_policy)]
                        right = method_rows[method_rows.policy.eq(comparison_policy)]
                        # Draw codes identify policies, so align by the immutable context/sample instead.
                        add(left, right, {'family': 'I02', 'experiment': experiment, 'method': method,
                                          'budget': int(budget), 'reference_policy': random_policy,
                                          'policy': comparison_policy, 'comparison': comparison_policy + '_minus_' + random_policy},
                            ['job_id', 'target_origin', 'physical_sample_id'])

    for experiment, rows in predictions[predictions.family.eq('I03')].groupby('experiment'):
        baseline = _method_rows(rows, 'identity', 0)
        for k in KS:
            for method in SCORE_METHODS:
                block = _method_rows(rows, method, k)
                add(baseline, block, {'family': 'I03', 'experiment': experiment, 'method': method, 'k': k,
                                     'comparison': method + '_minus_identity'}, keys, repeat=True)
            for earlier, later in (('raw_mean_anchor', 'ovr_trace1'), ('ovr_trace1', 'ovo_trace1'),
                                   ('ovo_trace1', 'ovo_pca_trace1')):
                add(_method_rows(rows, earlier, k), _method_rows(rows, later, k),
                    {'family': 'I03', 'experiment': experiment, 'k': k,
                     'comparison': later + '_minus_' + earlier}, keys)

    for experiment, rows in predictions[predictions.family.eq('I04')].groupby('experiment'):
        for composition, composition_rows in rows[rows.k.eq(-1)].groupby('composition_id'):
            baseline = _method_rows(rows, 'identity', 0)
            for method in ESS_METHODS:
                add(baseline, _method_rows(composition_rows, method),
                    {'family': 'I04', 'experiment': experiment, 'method': method, 'k': -1,
                     'composition_id': composition, 'comparison': method + '_minus_identity'}, keys, repeat=True)
            for earlier, later in (('weighted_lambda1', 'weighted_mean'), ('weighted_mean', 'weighted_ess')):
                add(_method_rows(composition_rows, earlier), _method_rows(composition_rows, later),
                    {'family': 'I04', 'experiment': experiment, 'k': -1, 'composition_id': composition,
                     'comparison': later + '_minus_' + earlier}, keys)
    return pd.DataFrame(transitions), pd.DataFrame(intervals)


def mechanism_table(predictions, metrics, diagnostics):
    records = diagnostics.to_dict('records') if not diagnostics.empty else []
    if not metrics.empty:
        acquisition = metrics[metrics.family.eq('I02') & metrics.k.eq(-1)]
        if len(acquisition):
            for (experiment, method, policy, budget), block in acquisition.groupby(['experiment', 'method', 'policy', 'budget']):
                record = {'analysis': 'acquisition_coverage_and_cost', 'family': 'I02', 'experiment': experiment,
                          'method': method, 'policy': policy, 'budget': int(budget), 'endpoint_rows': len(block),
                          'n_jobs': block.job_id.nunique(), 'mean_accuracy': float(block.accuracy.mean())}
                for name in ('observed_support_class_count', 'labels_observed_for_guard', 'budget_labels_observed',
                             'labels_used_in_adapter_fit', 'labels_used_in_classifier_fit'):
                    if name in block:
                        values = numeric(block[name]).dropna()
                        if len(values): record['mean_' + name] = float(values.mean())
                if 'guard_fallback_identity' in block:
                    record['fallback_endpoint_count'] = int(truthy(block.guard_fallback_identity).sum())
                records.append(record)
        ess = metrics[metrics.family.eq('I04') & metrics.k.eq(-1)]
        ess_groups = ess.groupby(['experiment', 'method', 'composition_id']) if len(ess) else []
        for (experiment, method, composition), block in ess_groups:
            record = {'analysis': 'ess_shrinkage_and_motion', 'family': 'I04', 'experiment': experiment,
                      'method': method, 'composition_id': composition, 'n_jobs': block.job_id.nunique(),
                      'mean_accuracy': float(block.accuracy.mean())}
            for name in ('effective_support_weight_ess', 'regularization', 'effective_df_per_output',
                         'score_moved_count', 'decision_changed_count', 'correction_frobenius'):
                if name in block:
                    values = numeric(block[name]).dropna()
                    if len(values): record['mean_' + name] = float(values.mean())
            records.append(record)
    if not predictions.empty:
        score_rows = predictions[predictions.family.eq('I03') & predictions.experiment.eq('E01')]
        for (job_id, k, draw), rows in score_rows[score_rows.k.gt(0)].groupby(['job_id', 'k', 'draw']):
            reference = _method_rows(rows, 'ovr_trace1').set_index('physical_sample_id')
            if reference.empty:
                continue
            a = np.asarray([json.loads(value) for value in reference.scores_json])
            for method in ('ovo_trace1', 'ovo_pca_trace1'):
                block = _method_rows(rows, method).set_index('physical_sample_id').loc[reference.index]
                b = np.asarray([json.loads(value) for value in block.scores_json])
                records.append({'analysis': 'independent_binary_score_equivalence', 'family': 'I03',
                                'experiment': 'E01', 'job_id': job_id, 'method': method, 'k': int(k), 'draw': int(draw),
                                'max_saved_score_difference': float(np.max(np.abs(a - b))),
                                'same_saved_predictions': bool(reference.pred_label.eq(block.pred_label).all()),
                                'interpretation': 'Algebraically equivalent normalized binary feature Grams are a numerical negative control, not independent method evidence'})
        for (family, experiment, method, origin), rows in predictions.groupby(
                ['family', 'experiment', 'method', 'origin_group_id']):
            records.append({'analysis': 'per_query_origin_performance', 'family': family, 'experiment': experiment,
                            'method': method, 'query_origin_group_id': origin, 'n_predictions': len(rows),
                            'n_physical_samples': rows.physical_sample_id.nunique(),
                            'accuracy': float(rows.true_label.eq(rows.pred_label).mean())})
    return pd.DataFrame(records)


def curves(metrics):
    if metrics.empty:
        return []
    records = []
    for (family, experiment, method, k), rows in metrics.groupby(['family', 'experiment', 'method', 'k']):
        groupings = ['policy', 'budget'] if family == 'I02' else ['composition_id'] if family == 'I04' else []
        blocks = rows.groupby(groupings, dropna=False) if groupings and all(field in rows for field in groupings) else [((), rows)]
        for names, block in blocks:
            names = names if isinstance(names, tuple) else (names,)
            records.append({'family': family, 'experiment': experiment, 'method': method, 'k': int(k),
                            **dict(zip(groupings, names)), 'accuracy': float(block.accuracy.mean()),
                            'macro_f1': float(block.macro_f1.mean()), 'balanced_accuracy': float(block.balanced_accuracy.mean()),
                            'endpoint_count': len(block)})
    return records


def make_figures(output, metrics):
    if metrics.empty:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(output) / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for family, methods in (('I01', ('bias', 'mean_anchor', 'full_refit', 'source_selected')),
                            ('I03', SCORE_METHODS)):
        for experiment, rows in metrics[metrics.family.eq(family)].groupby('experiment'):
            figure, axis = plt.subplots(figsize=(8.5, 4.7))
            for method in methods:
                line = rows[rows.method.eq(method) & rows.k.gt(0)].groupby('k').accuracy.mean().sort_index()
                if len(line): axis.plot(line.index, 100 * line, marker='o', label=method)
            baseline = rows[rows.method.eq('identity')].accuracy
            if len(baseline): axis.axhline(100 * baseline.mean(), color='black', linestyle='--', label='Source identity')
            axis.set(xlabel='Support labels per class', ylabel='Mean accuracy (%)', title=f'{family} / {experiment}: all prespecified methods')
            axis.set_xticks(KS); axis.grid(alpha=.2); axis.legend(fontsize=8)
            figure.tight_layout(); path = directory / f'{family.lower()}_{experiment.lower()}_curves.png'
            figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    acquisition = metrics[metrics.family.eq('I02') & metrics.k.eq(-1)]
    if len(acquisition):
        for experiment, rows in acquisition.groupby('experiment'):
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            for method, axis in zip(('full_refit', 'guarded_mean_anchor'), axes):
                for policy, block in rows[rows.method.eq(method)].groupby('policy'):
                    line = block.groupby('budget').accuracy.mean().sort_index()
                    axis.plot(line.index, line * 100, marker='o', label=policy)
                axis.set(xlabel='Acquired Support labels (total)', ylabel='Mean accuracy (%)', title=method)
                axis.grid(alpha=.2); axis.legend(fontsize=7)
            figure.suptitle(f'I02 / {experiment}: label-blind ranking within a previously stratified pool')
            figure.tight_layout(); path = directory / f'i02_{experiment.lower()}_acquisition.png'
            figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    for experiment, rows in metrics[metrics.family.eq('I04') & metrics.k.eq(-1)].groupby('experiment'):
        figure, axis = plt.subplots(figsize=(9, 4.8))
        names = sorted(rows.composition_id.unique())
        for method in ESS_METHODS:
            line = rows[rows.method.eq(method)].groupby('composition_id').accuracy.mean()
            axis.plot(np.arange(len(names)), [line.get(name, np.nan) * 100 for name in names], marker='o', label=method)
        baseline = metrics[metrics.family.eq('I04') & metrics.experiment.eq(experiment) & metrics.method.eq('identity')].accuracy
        if len(baseline): axis.axhline(100 * baseline.mean(), color='black', linestyle='--', label='Source identity')
        axis.set_xticks(np.arange(len(names)), names, rotation=20, ha='right')
        axis.set(xlabel='Fixed Support composition', ylabel='Mean accuracy (%)', title=f'I04 / {experiment}: complete shrinkage controls')
        axis.grid(alpha=.2); axis.legend(fontsize=8); figure.tight_layout()
        path = directory / f'i04_{experiment.lower()}_ess.png'
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    return paths


def report_text(summary, transitions, mechanisms):
    names = {'I01': '仅用Source内部验证选择校准策略', 'I02': '隐藏候选标签的Support获取顺序',
             'I03': '保留OVR输出锚的OVO信息校准', 'I04': '权重ESS与预设收缩'}
    text = ['# B方案第五轮：Source内策略、Support获取与分数信息', '',
            f"状态：**{summary['status']}**；已封存 **{summary['completed_jobs']}/{summary['expected_jobs']}** 个任务包。",
            '本报告只读取完整封存工件，不拟合或选择新模型，不修改前四轮输入、标签、协议、模型及结果。', '',
            f"新增分类器拟合{summary['new_model_fits']}次，校准拟合调用{summary['new_adapter_fits']}次；科研任务CPU合计{summary['cpu_seconds']:.6f}秒。",
            f"保存{summary['prediction_rows']}条相关外层Query预测，涉及{summary['physical_query_samples']}个唯一Query实物；控制、核验、失败及推送成本另见账本。", '',
            f"Source内部验证预测另计{summary['inner_validation_prediction_rows']}/{summary['planned_inner_validation_prediction_rows']}条（已封存/冻结计划），不计入外层Query预测数，也不增加独立验证实物。", '',
            '## 完成范围', '', '|块|问题|完成/计划|状态|', '|---|---|---:|---|']
    for family, count in summary['family_task_counts'].items():
        text.append(f"|{family}|{names[family]}|{count['completed']}/{count['planned']}|{count['status']}|")
    text += ['', '以下角色数直接来自冻结任务中的ID与source_class_counts，未由目标比例反推。', '',
             '|块|原任务|Source数|Source类别计数|候选Support池数|Query数|冻结上下文数|',
             '|---|---|---:|---|---:|---:|---:|']
    for layout in summary.get('frozen_role_layouts', []):
        text.append(f"|{layout['family']}|{layout['experiment']}|{layout['source_count']}|{layout['source_class_counts_json']}|{layout['support_pool_count']}|{layout['query_count']}|{layout['context_count']}|")
    text += ['', '## 设计与解释边界', '',
             'I01仅在每个外层Source内部构造三次角色划分，每次inner Source、Support和验证样品互斥，外层Support/Query值均不进入选择，外层结果在选择完成后才载入。预设Identity、Bias λ1、mean-Anchor λ=N和冻结坐标完整重训四个候选，按三次内验证平均Macro-F1及预先固定的简洁优先平局规则选择；随后一次应用到原冻结Source和外层Support。不能使用Source模型的训练内预测冒充内验证。最大内验证成绩是选择统计量，不是无偏效果；内层训练量更小，Source内域与外层Support来源也可能不同，策略迁移不保证成功。所有候选和选择后外层结果均报告。', '',
             'I02固定总标签预算，比较三个固定随机顺序、Source-margin不确定性和在Source坐标中持续保留Source质心锚的max-min覆盖。距离更新同时考虑这一预先存在的Source锚与已选候选点。排序接口不接收候选真标签、语义ID前缀或Query，只用候选谱、Source预测及不含语义的固定hash次序。候选池原本已按类别分层，故这是该池内的标签盲排序实验，并非完全无需标签的实际采集流程。选后揭示标签并记录覆盖；完整SVM仍包含全部Source类别，校准缺类时按预设规则回退Identity，不丢弃条件、不重抽至满意。不同政策同时改变入选实物、类别覆盖和谱位置，不能将差值唯一归因为多样性。回退前已观察的标签仍计入预算，不把回退称零标签成本。', '',
             'I03保持原OVR分数为输出锚；只把拟合残差的输入特征换为Source中心化并按全局row-L2均方根规范化的OVR、OVO margins和Source-only OVO PCA。所有基与归一化仅由Source得到，输出为原分数加拟合修正，K0决策保持不变；不对原始OVO列直接argmax。固定mean-loss单位先验（sum λ=N），不根据Query选强度。二分类规范化OVR/OVO Gram代数等价，三种嵌入的数值结果是负对照，不算三项独立方法证据；五分类PCA控制维度也改变保留子空间，不能唯一识别维度的单因素效应。', '',
             'I04完全复用上一轮Support构成与ID，将相同类别均衡权重的残差惩罚从λ1比较到λ=N及预设λ=N²/ESS。ESS=(Σw)²/Σw²是权重集中度，不是新增实物数；ESS公式是冻结的启发式收缩，不是由Query选出的最优λ或新算法。均衡时ESS=N，两个新条件等价并复用G01的mean-Anchor。减少误伤可能只是修正更小、更多保持Source，必须同时看分数运动、类别变化、纠错及净收益。', '',
             '## 全部冻结条件的描述性结果', '',
             '|块|原任务|方法|K/类|政策/预算/构成|Accuracy|Macro-F1|Balanced Accuracy|端点数|',
             '|---|---|---|---:|---|---:|---:|---:|---:|']
    for curve in summary['curves_macro_over_job_draws']:
        descriptor = ', '.join(f'{key}={curve[key]}' for key in ('policy', 'budget', 'composition_id') if key in curve and str(curve[key]) not in ('', 'nan'))
        shot = '不适用' if curve['k'] == -1 else curve['k']
        text.append(f"|{curve['family']}|{curve['experiment']}|{curve['method']}|{shot}|{descriptor}|{curve['accuracy']:.6f}|{curve['macro_f1']:.6f}|{curve['balanced_accuracy']:.6f}|{curve['endpoint_count']}|")
    text += ['', '## 配对变化与描述性不确定性', '',
             '所有正、负和零结果保留。配对以相同外层Query的physical_sample_id对齐，区间保留同实物相关预测；完整来源分层、Recall代价及纠错/误伤另见CSV。', '',
             '|块|原任务|比较|预算/政策/构成|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|',
             '|---|---|---|---|---:|---:|---:|']
    if not transitions.empty:
        for row in transitions[transitions.scope.eq('ALL')].to_dict('records'):
            descriptor = ', '.join(f'{key}={row[key]}' for key in ('k', 'budget', 'policy', 'reference_policy', 'composition_id') if key in row and pd.notna(row[key]) and str(row[key]) != '')
            low, high = row.get('ci_lower'), row.get('ci_upper')
            interval = f"[{100 * low:+.3f},{100 * high:+.3f}]" if low is not None and high is not None else '未计算'
            text.append(f"|{row['family']}|{row['experiment']}|{row['comparison']}|{descriptor}|{100 * row['delta_accuracy']:+.3f} {interval}|{row['wrong_to_correct']}/{row['correct_to_wrong']}|{row['n_predictions']}/{row['n_physical_samples']}|")
    text += ['', '## Source策略与获取成本审计', '',
             'source_policy.json、inner_plan.json与inner_validation_predictions.csv.gz记录每个Source内选择的依据；汇总再次核验内层角色与指标。Source选择器训练成本与选择后外层端点复用分开计数。acquisition_plan.json及机制表记录实际覆盖、资格回退、观察标签量与拟合用标签量；全池政策端点不重复训练，也不当作多个独立获取结果。', '',
             '三次Source内划分的实物重叠数、交叉角色复用数与逐次分类器拟合成本保存于mechanism_analysis.csv；它们不构成新增独立验证样品。']
    if len(mechanisms) and 'analysis' in mechanisms:
        choices = mechanisms[mechanisms.analysis.eq('source_policy_choice')]
        if len(choices):
            text += ['', '|原任务|K/类|Source内选中方法|上下文数|平均最大内验证Macro-F1（选择统计量）|',
                     '|---|---:|---|---:|---:|']
            for (experiment, k, method), block in choices.groupby(['experiment', 'k', 'selected_method']):
                text.append(f"|{experiment}|{int(k)}|{method}|{block.job_id.nunique()}|{numeric(block.mean_validation_macro_f1).mean():.6f}|")
    text += ['', '## 局限与未运行事项', '', LIMITS, '',
             '二分类和五来源的类别数、样品组成及标签预算不同，不能直接相减Accuracy解释类别数效应。构成、获取顺序、校准几何或收缩启发式均不依据本轮外层Query选择最终赢家。真实新批次评价继续 **NOT_RUN_NEEDS_NEW_BATCH_DATA**；真实batch_id、样品属性及独立验证不补造。', '',
             '## 复现', '', '```bash',
             'python scripts/run_b_mechanism_round5_20261111.py --resume',
             'python scripts/summarize_b_mechanism_round5_20261111.py --resume',
             'python scripts/summarize_b_mechanism_round5_20261111.py --verify',
             'python scripts/run_b_mechanism_round5_20261111.py --verify', '```', '',
             'PARTIAL只描述已完整封存子集；核验通过不会把未运行任务升级为DONE。', '']
    return '\n'.join(text)


def summarize(output, *, verify=False, n_boot=1000, report_path=None):
    output = Path(output)
    jobs, predictions, metrics, diagnostics, manifest, checks = load_jobs(output)
    if verify:
        root_metrics_path = output / 'metrics.csv'
        if not root_metrics_path.is_file():
            raise ValueError('No root metric table exists for independent verification')
        existing = _read_csv(root_metrics_path)
        required = {'job_id', 'endpoint_id', 'accuracy', 'macro_f1', 'balanced_accuracy',
                    'confusion_matrix_json', 'classes_json'}
        if not required.issubset(existing):
            raise ValueError('Root metric table lacks independent-verification fields')
        if len(existing) != len(metrics) or set(zip(existing.job_id, existing.endpoint_id)) != set(zip(metrics.job_id, metrics.endpoint_id)):
            raise ValueError('Root metrics do not cover exactly the sealed endpoints')
        old, new = existing.set_index(['job_id', 'endpoint_id']), metrics.set_index(['job_id', 'endpoint_id'])
        old = old.loc[new.index]
        for field in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            if not np.allclose(numeric(old[field]), numeric(new[field]), rtol=0, atol=1e-12):
                raise ValueError('Root independent metric mismatch: ' + field)
            checks += len(new)
        for field in ('confusion_matrix_json', 'classes_json'):
            if not all(json.loads(a) == json.loads(b) for a, b in zip(old[field], new[field])):
                raise ValueError('Root independent matrix/class vocabulary mismatch')
            checks += len(new)
        receipt = {'status': 'DONE', 'sealed_jobs': len(jobs), 'metric_rows': len(metrics), 'checks': checks,
                   'artifact_hashes_checked': True, 'independent_confusion_count_recomputation': True,
                   'source_only_policy_independently_checked': True, 'acquisition_guard_and_cost_checked': True,
                   'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        shared.write_json(output / 'summary_verification.json', receipt)
        return receipt
    transitions, intervals = comparison_tables(predictions, n_boot)
    mechanisms = mechanism_table(predictions, metrics, diagnostics)
    protocol = protocol_payload(output)
    design, _ = shared.frozen_design(output)
    counts = {}
    for family, fallback in FAMILIES.items():
        planned = sum(job['family'] == family for job in design) if design else fallback
        completed = sum(job['family'] == family for job in jobs)
        counts[family] = {'completed': completed, 'planned': planned,
                          'status': 'DONE' if completed == planned else 'PARTIAL' if completed else 'NOT_RUN'}
    expected = len(design) if design else 180
    layouts = {}
    for job in design or jobs:
        key = (job['family'], job['experiment'], len(job['source_ids']), len(job['support_ids']),
               len(job['query_ids']), json.dumps(job.get('source_class_counts', {}), ensure_ascii=False, sort_keys=True))
        layouts[key] = layouts.get(key, 0) + 1
    role_layouts = [{'family': key[0], 'experiment': key[1], 'source_count': key[2],
                     'support_pool_count': key[3], 'query_count': key[4],
                     'source_class_counts_json': key[5], 'context_count': count}
                    for key, count in sorted(layouts.items())]
    summary = {'status': 'DONE' if len(jobs) == expected else 'PARTIAL', 'completed_jobs': len(jobs),
               'expected_jobs': expected, 'family_task_counts': counts,
               'frozen_role_layouts': role_layouts,
               'new_model_fits': sum(job.get('new_model_fits', 0) for job in jobs),
               'new_adapter_fits': sum(job.get('new_adapter_fits', 0) for job in jobs),
               'planned_new_model_fits': protocol.get('planned_new_model_fits'),
               'planned_new_adapter_fits': protocol.get('planned_new_adapter_fits'),
               'cpu_seconds': sum(job.get('cpu_seconds', 0.) for job in jobs),
               'wall_seconds': sum(job.get('wall_seconds', 0.) for job in jobs),
               'prediction_rows': len(predictions), 'metric_rows': len(metrics),
               'inner_validation_prediction_rows': sum(job.get('inner_validation_prediction_rows', 0)
                                                       for job in jobs if job['family'] == 'I01'),
               'planned_inner_validation_prediction_rows': sum(
                   len(episode['inner_validation_ids']) * len(KS) * len(SELECTOR_CANDIDATES)
                   for job in design or jobs if job['family'] == 'I01'
                   for episode in job.get('inner_plan', {}).get('episodes', [])),
               'physical_query_samples': int(predictions.physical_sample_id.nunique()) if len(predictions) else 0,
               'e05_status': 'NOT_RUN_NEEDS_NEW_BATCH_DATA', 'previous_results_modified': False,
               'no_new_fits_by_summarizer': True, 'uncertainty': LIMITS,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'curves_macro_over_job_draws': curves(metrics)}
    if summary['status'] == 'DONE':
        for actual, planned in (('new_model_fits', 'planned_new_model_fits'), ('new_adapter_fits', 'planned_new_adapter_fits')):
            if summary[planned] is not None and summary[actual] != summary[planned]:
                raise ValueError('Actual fitting count differs from frozen plan: ' + actual)
    for name, frame in (('metrics.csv', metrics), ('run_manifest.csv', manifest), ('paired_transitions.csv', transitions),
                        ('uncertainty.csv', intervals), ('mechanism_analysis.csv', mechanisms)):
        shared.write_csv(output / name, frame)
    summary['figures'] = make_figures(output, metrics)
    shared.write_json(output / 'results_summary.json', summary)
    shared.write_json(output / 'status.json', {key: summary[key] for key in
                      ('status', 'completed_jobs', 'expected_jobs', 'family_task_counts', 'new_model_fits',
                       'new_adapter_fits', 'cpu_seconds', 'e05_status', 'timestamp_utc')})
    shared.atomic_text(report_path or REPORT, report_text(shared.clean(summary), transitions, mechanisms))
    return shared.clean(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--outdir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100')
    try:
        if args.dry_run:
            jobs, predictions, metrics, _, _, checks = load_jobs(args.output)
            design, _ = shared.frozen_design(args.output)
            print(json.dumps({'status': 'NOT_RUN', 'sealed_jobs': len(jobs), 'prediction_rows': len(predictions),
                              'endpoint_metrics': len(metrics), 'expected_jobs': len(design) if design else 180,
                              'independent_checks': checks, 'fits': 0}, ensure_ascii=False))
            return 0
        print(json.dumps(summarize(args.output, verify=args.verify, n_boot=args.bootstrap,
                                   report_path=args.report), ensure_ascii=False))
        return 0
    except (ValueError, AssertionError) as error:
        print('INTEGRITY_ERROR: ' + str(error), file=sys.stderr)
        return 42


if __name__ == '__main__':
    raise SystemExit(main())
