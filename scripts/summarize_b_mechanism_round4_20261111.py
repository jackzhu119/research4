#!/usr/bin/env python3
"""Independently summarize the fourth B study from sealed saved predictions.

This program never fits, selects or changes a model. Physical sample IDs, rather
than support redraws or repeated prediction rows, are the resampling units.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import summarize_b_mechanism_followup_20261110 as shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'reports/b_mechanism_round4_20261111'
REPORT = ROOT / 'docs/B_MECHANISM_ROUND4_RESULTS_20261111.md'
FAMILIES = {'H01': 45, 'H02': 240, 'H03': 45, 'H04': 45}
MASS_METHODS = ('source_mass_only', 'source_support_fixed_C', 'source_support_normalized_mass')
SUPPORT_METHODS = ('anchor_unweighted', 'anchor_class_balanced', 'bias_unweighted', 'bias_class_balanced')
C_LEVELS = ((.1, '0p1'), (1., '1'), (10., '10'))
LIMITS = (
    '本轮依据前三轮已曝光内部结果提出并冻结，仅使用master TRAIN，属于探索性机制补充。'
    '重复划分、Support重排、类别构成及参数条件不增加独立实物数。描述性95%区间以physical_sample_id聚类，'
    '保留同实物所有相关预测；不覆盖来源/批次总体、协议选择及长期开发偏倚。来源与批次仍混杂，'
    '不能据此宣称真实跨批次泛化、确认性因果机制或新的独立验证。'
)
METADATA = ('C', 'C_multiplier', 'class_weight_setting', 'labeled_training_n', 'statistics_training_n',
            'effective_loss_mass', 'C_times_labeled_n', 'support_labels_used', 'composition_index',
            'composition_id', 'support_counts_json', 'draw_seed', 'balanced_composition',
            'fixed_total_support', 'k_per_class_not_applicable', 'effective_support_weight_ess',
            'case_type', 'actual_noise_rate', 'noise_q_per_class', 'noise_seed')


def protocol_payload(output):
    path = Path(output) / 'frozen_protocol.json'
    return json.loads(path.read_text()) if path.exists() else {}


def truthy(values):
    return values.astype(str).str.lower().isin(('true', '1', '1.0'))


def _crossing(source, weak, blend=.05):
    """Independent first-boundary calculation; no true labels enter."""
    winner = source.argmax(axis=1)
    direction = (weak - source) / blend
    rows = np.arange(len(source))
    gaps = source[rows, winner, None] - source
    slopes = direction - direction[rows, winner, None]
    possible = slopes > 0
    possible[rows, winner] = False
    crossing = np.full(source.shape, np.inf)
    np.divide(gaps, slopes, out=crossing, where=possible)
    return crossing.min(axis=1)


def _verify_certificates(predictions, certificates):
    """Recompute the sufficient gap bound and first boundary from saved scores."""
    if certificates.empty:
        return 0
    needed = {'endpoint_id', 'physical_sample_id', 'source_top2_gap', 'actual_correction_linf',
              'strict_margin_certificate', 'decision_changed', 'first_crossing_blend'}
    if not needed.issubset(certificates):
        raise ValueError('Weak certificate fields missing: ' + str(needed - set(certificates)))
    if certificates.duplicated(['endpoint_id', 'physical_sample_id']).any():
        raise ValueError('Duplicate weak certificate for a physical sample/endpoint')
    baseline = predictions[predictions.method.eq('identity')].set_index('physical_sample_id')
    weak_rows = predictions[predictions.method.isin(('f01_weak', 'g03_weak'))]
    if set(weak_rows.endpoint_id) != set(certificates.endpoint_id):
        raise ValueError('Weak certificate endpoint coverage differs from predictions')
    checks = 0
    for identifier, frame in weak_rows.groupby('endpoint_id', sort=False):
        frame = frame.set_index('physical_sample_id')
        saved = certificates[certificates.endpoint_id.eq(identifier)].set_index('physical_sample_id')
        if set(frame.index) != set(saved.index):
            raise ValueError('Weak certificate physical sample coverage differs')
        saved = saved.loc[frame.index]
        reference = baseline.loc[frame.index]
        source = np.asarray([json.loads(value) for value in reference.scores_json])
        weak = np.asarray([json.loads(value) for value in frame.scores_json])
        gap = np.sort(source, axis=1)[:, -1] - np.sort(source, axis=1)[:, -2]
        movement = np.max(np.abs(weak - source), axis=1)
        certified = gap > 2 * movement
        changed = frame.pred_label.ne(reference.pred_label).to_numpy()
        if not np.allclose(gap, pd.to_numeric(saved.source_top2_gap), rtol=0, atol=1e-12):
            raise ValueError('Independent source top-two gap mismatch')
        if not np.allclose(movement, pd.to_numeric(saved.actual_correction_linf), rtol=0, atol=1e-12):
            raise ValueError('Independent weak score movement mismatch')
        if not np.array_equal(certified, truthy(saved.strict_margin_certificate).to_numpy()):
            raise ValueError('Independent strict margin certificate mismatch')
        if not np.array_equal(changed, truthy(saved.decision_changed).to_numpy()) or np.any(certified & changed):
            raise ValueError('Saved weak decision motion contradicts its gap certificate')
        crossing = _crossing(source, weak)
        existing = pd.to_numeric(saved.first_crossing_blend, errors='coerce').to_numpy()
        finite = np.isfinite(crossing)
        if 'first_crossing_finite' in saved:
            if not np.array_equal(finite, truthy(saved.first_crossing_finite).to_numpy()):
                raise ValueError('Independent boundary finiteness mismatch')
        if np.any(~finite & np.isfinite(existing)):
            raise ValueError('A noncrossing path was saved as a finite boundary')
        if not np.allclose(crossing[finite], existing[finite], rtol=1e-9, atol=1e-10):
            raise ValueError('Independent first argmax boundary mismatch')
        checks += 6 * len(frame)
    return checks


def load_jobs(output):
    jobs, predictions, metrics, diagnostics, manifest, checks = shared.load_jobs(output)
    extra, certificate_summaries = [], []
    for job in jobs:
        directory = Path(output) / 'jobs' / job['job_id']
        for name in ('C_constraint_mechanisms.csv', 'weak_mechanism.csv', 'support_geometry.csv'):
            candidate = directory / name
            if not candidate.is_file():
                continue
            try:
                rows = pd.read_csv(candidate, keep_default_na=False)
            except pd.errors.EmptyDataError:
                continue
            if rows.empty:
                continue
            for key in ('family', 'job_id', 'experiment', 'target_origin', 'seed', 'ratio', 'coverage', 'fold'):
                rows[key] = job.get(key)
            if 'analysis' not in rows:
                rows['analysis'] = name.removesuffix('.csv')
            extra.append(rows)
        certificate_path = directory / 'weak_certificates.csv'
        if certificate_path.is_file():
            certificates = pd.read_csv(certificate_path, keep_default_na=False)
            job_predictions = predictions[predictions.job_id.eq(job['job_id'])]
            checks += _verify_certificates(job_predictions, certificates)
            for (method, k), block in certificates.groupby(['method', 'k']):
                record = {'analysis': 'weak_certificate_job_summary', 'family': 'H03',
                          'experiment': job['experiment'], 'job_id': job['job_id'], 'method': method, 'k': int(k),
                          'n_predictions': len(block), 'n_physical_samples': int(block.physical_sample_id.nunique())}
                for column in ('score_changed', 'decision_changed', 'score_moved_decision_unchanged',
                               'strict_margin_certificate', 'certificate_violation', 'first_crossing_le_actual_blend',
                               'first_crossing_le_one', 'adapter_parameter_changed'):
                    if column in block:
                        record[column + '_count'] = int(truthy(block[column]).sum())
                crossings = pd.to_numeric(block.first_crossing_blend, errors='coerce')
                finite = crossings[np.isfinite(crossings)]
                record['minimum_finite_crossing_blend'] = float(finite.min()) if len(finite) else None
                record['median_finite_crossing_blend'] = float(finite.median()) if len(finite) else None
                certificate_summaries.append(record)
    if extra or certificate_summaries:
        frames = ([diagnostics] if not diagnostics.empty else []) + extra
        if certificate_summaries:
            frames.append(pd.DataFrame(certificate_summaries))
        diagnostics = pd.concat(frames, ignore_index=True)
    if not predictions.empty:
        additions = [name for name in METADATA if name in metrics and name not in predictions]
        if additions:
            predictions = predictions.merge(metrics[['job_id', 'endpoint_id'] + additions],
                                            on=['job_id', 'endpoint_id'], validate='many_to_one')
        for name in ('C', 'composition_index', 'draw_seed', 'fixed_total_support', 'actual_noise_rate'):
            if name in predictions:
                predictions[name] = pd.to_numeric(predictions[name], errors='coerce')
        # C is also identifiable in the frozen method names for native/interface diagnostics.
        if 'C' not in predictions:
            predictions['C'] = np.nan
        for value, token in C_LEVELS:
            selected = predictions.family.eq('H02') & predictions.method.str.startswith('C_' + token + '_')
            predictions.loc[selected, 'C'] = value
    return jobs, predictions, metrics, diagnostics, manifest, checks


def interval_delta(rows, n_boot):
    low, high = shared.cluster_interval(rows, n_boot=n_boot)
    return {'estimate': float(rows.delta.mean()), 'ci_lower': low, 'ci_upper': high,
            'n_predictions': len(rows), 'n_physical_samples': int(rows.physical_sample_id.nunique()),
            'confidence_level': .95, 'n_bootstrap': n_boot, 'resampling_unit': 'physical_sample_id',
            'interval_scope': LIMITS}


def comparison_tables(predictions, n_boot=1000):
    transitions, intervals, interactions = [], [], []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def add(left, right, info, keys, repeat=False, scopes=False):
        if left.empty or right.empty:
            return None
        # Keep entire matched Source contexts, rather than accidentally intersecting individual samples.
        if 'job_id' in keys:
            common = set(left.job_id) & set(right.job_id)
            left, right = left[left.job_id.isin(common)], right[right.job_id.isin(common)]
        else:
            context = [name for name in ('seed', 'target_origin') if name in keys]
            common = left[context].drop_duplicates().merge(right[context].drop_duplicates(), on=context)
            left = left.merge(common, on=context, validate='many_to_one')
            right = right.merge(common, on=context, validate='many_to_one')
        if left.empty or right.empty:
            return None
        if repeat:
            left = shared.repeat_baseline(left, right)
        paired = shared.pair(left, right, keys)
        scoped = [('ALL', paired)]
        if scopes and 'target_origin' in paired:
            scoped += [(str(origin), group) for origin, group in paired.groupby('target_origin')]
        for scope, group in scoped:
            record, ci = shared.transition_record(group, {**info, 'scope': scope}, n_boot)
            ci['interval_scope'] = LIMITS
            transitions.append(record); intervals.append(ci)
        return paired

    def interaction(first, second, info, keys):
        if first is None or second is None:
            return
        rows = first[keys + ['delta']].merge(second[keys + ['delta']], on=keys,
                                             suffixes=('_first', '_second'), how='outer',
                                             indicator=True, validate='one_to_one')
        if not rows._merge.eq('both').all():
            raise ValueError('Incomplete matched cells for a prespecified interaction')
        rows['delta'] = rows.delta_second - rows.delta_first
        item = {**info, 'scope': 'ALL', 'effect': 'difference_of_differences_accuracy',
                **interval_delta(rows, n_boot)}
        intervals.append(item); interactions.append({**item, 'analysis': 'factor_interaction'})

    keys = ['job_id', 'draw', 'target_origin', 'physical_sample_id']
    for experiment, rows in predictions[predictions.family.eq('H01')].groupby('experiment'):
        baseline = rows[rows.method.eq('identity') & rows.k.eq(0)]
        for k in (1, 3, 5, 10):
            blocks = {method: rows[rows.method.eq(method) & rows.k.eq(k)] for method in MASS_METHODS}
            for method, block in blocks.items():
                add(baseline, block, {'family': 'H01', 'experiment': experiment, 'method': method, 'k': k,
                                     'comparison': method + '_minus_source_C10'}, keys, repeat=True)
            for earlier, later, comparison in (
                    ('source_mass_only', 'source_support_fixed_C', 'additional_samples_minus_loss_mass_only'),
                    ('source_support_fixed_C', 'source_support_normalized_mass', 'normalized_mass_minus_fixed_C'),
                    ('source_mass_only', 'source_support_normalized_mass', 'normalized_additional_samples_minus_mass_only')):
                add(blocks[earlier], blocks[later], {'family': 'H01', 'experiment': experiment, 'k': k,
                                                    'comparison': comparison}, keys)

    rows = predictions[predictions.family.eq('H02')]
    for ratio in ('A', 'B', 'C'):
        for coverage in (0, 1):
            cell = rows[rows.ratio.eq(ratio) & rows.coverage.eq(coverage)]
            effects = {}
            for value, token in C_LEVELS:
                left = cell[cell.method.eq(f'C_{token}_balanced')]
                right = cell[cell.method.eq(f'C_{token}_none')]
                effects[value] = add(left, right, {'family': 'H02', 'experiment': 'E01', 'k': 0, 'C': value,
                                                  'ratio': ratio, 'coverage': coverage,
                                                  'comparison': 'none_minus_balanced'}, keys, scopes=True)
            for value in (.1, 1.):
                interaction(effects[10.], effects[value], {'family': 'H02', 'experiment': 'E01', 'k': 0,
                            'C': value, 'ratio': ratio, 'coverage': coverage,
                            'comparison': 'weight_effect_at_C_minus_weight_effect_at_C10'}, keys)
            for weight in ('balanced', 'none'):
                baseline = cell[cell.method.eq(f'C_10_{weight}')]
                for value, token in C_LEVELS[:2]:
                    add(baseline, cell[cell.method.eq(f'C_{token}_{weight}')],
                        {'family': 'H02', 'experiment': 'E01', 'k': 0, 'C': value,
                         'ratio': ratio, 'coverage': coverage, 'weight': weight,
                         'comparison': 'C_dose_minus_C10'}, keys, scopes=True)
    context_keys = ['seed', 'target_origin', 'physical_sample_id']
    for value, token in C_LEVELS:
        for weight in ('balanced', 'none'):
            data = rows[rows.method.eq(f'C_{token}_{weight}')]
            for ratio in ('A', 'B', 'C'):
                cell = data[data.ratio.eq(ratio)]
                add(cell[cell.coverage.eq(0)], cell[cell.coverage.eq(1)],
                    {'family': 'H02', 'experiment': 'E01', 'k': 0, 'C': value, 'weight': weight,
                     'ratio': ratio, 'comparison': 'coverage_1_minus_0'}, context_keys, scopes=True)
            for coverage in (0, 1):
                cell = data[data.coverage.eq(coverage)]
                for before, after in (('A', 'B'), ('A', 'C'), ('B', 'C')):
                    add(cell[cell.ratio.eq(before)], cell[cell.ratio.eq(after)],
                        {'family': 'H02', 'experiment': 'E01', 'k': 0, 'C': value, 'weight': weight,
                         'coverage': coverage, 'comparison': f'ratio_{after}_minus_{before}'}, context_keys)
            add(rows[rows.method.eq(f'C_{token}_{weight}')], rows[rows.method.eq(f'C_{token}_{weight}_argmax')],
                {'family': 'H02', 'experiment': 'E01', 'k': 0, 'C': value, 'weight': weight,
                 'comparison': 'score_argmax_minus_native'}, keys)

    for experiment, rows in predictions[predictions.family.eq('H03')].groupby('experiment'):
        baseline = rows[rows.method.eq('identity') & rows.k.eq(0)]
        for (method, k), block in rows[rows.method.isin(('f01_weak', 'g03_weak'))].groupby(['method', 'k']):
            add(baseline, block, {'family': 'H03', 'experiment': experiment, 'method': method, 'k': int(k),
                                 'comparison': 'saved_weak_minus_identity'}, keys, repeat=True)
        add(rows[rows.method.eq('svc_predict')], baseline,
            {'family': 'H03', 'experiment': experiment, 'k': 0, 'comparison': 'score_argmax_minus_native_zero_support'}, keys)

    for experiment, rows in predictions[predictions.family.eq('H04')].groupby('experiment'):
        baseline = rows[rows.method.eq('identity') & rows.k.eq(0)]
        adapted = rows[rows.k.eq(-1)]
        for composition, data in adapted.groupby('composition_id'):
            for method in SUPPORT_METHODS:
                block = data[data.method.eq(method)]
                add(baseline, block, {'family': 'H04', 'experiment': experiment, 'method': method,
                                     'k': -1, 'composition_id': composition,
                                     'comparison': method + '_minus_identity'}, keys, repeat=True)
            for kind in ('anchor', 'bias'):
                add(data[data.method.eq(kind + '_unweighted')], data[data.method.eq(kind + '_class_balanced')],
                    {'family': 'H04', 'experiment': experiment, 'k': -1, 'composition_id': composition,
                     'comparison': kind + '_class_balanced_minus_unweighted'}, keys)
            for weight in ('unweighted', 'class_balanced'):
                add(data[data.method.eq('bias_' + weight)], data[data.method.eq('anchor_' + weight)],
                    {'family': 'H04', 'experiment': experiment, 'k': -1, 'composition_id': composition,
                     'comparison': 'anchor_minus_bias_' + weight}, keys)
            for method in SUPPORT_METHODS:
                left = adapted[adapted.method.eq(method) & truthy(adapted.balanced_composition)].copy()
                right = data[data.method.eq(method)].copy()
                # Composition is a factor; align its own draw seed, not its encoded composition ID.
                if len(left) and len(right):
                    left['draw'] = pd.to_numeric(left.draw_seed).astype(int)
                    right['draw'] = pd.to_numeric(right.draw_seed).astype(int)
                    add(left, right, {'family': 'H04', 'experiment': experiment, 'method': method, 'k': -1,
                                     'composition_id': composition, 'comparison': 'composition_minus_balanced_support'}, keys)
    return pd.DataFrame(transitions), pd.DataFrame(intervals), pd.DataFrame(interactions)


def mechanism_table(predictions, metrics, diagnostics, interactions):
    records = diagnostics.to_dict('records') if not diagnostics.empty else []
    if not metrics.empty:
        for (family, experiment, method, k), group in metrics.groupby(['family', 'experiment', 'method', 'k']):
            records.append({'analysis': 'endpoint_repeat_description', 'family': family, 'experiment': experiment,
                            'method': method, 'k': int(k), 'endpoint_rows': len(group),
                            'n_jobs': group.job_id.nunique(), 'mean_accuracy': float(group.accuracy.mean()),
                            'minimum_accuracy': float(group.accuracy.min()), 'maximum_accuracy': float(group.accuracy.max()),
                            'interpretation': '所有冻结端点均保留；这些重复不构成独立实物'})
        h01 = metrics[metrics.family.eq('H01')]
        for (experiment, method, k), group in h01.groupby(['experiment', 'method', 'k']):
            item = {'analysis': 'loss_mass_design_summary', 'family': 'H01', 'experiment': experiment,
                    'method': method, 'k': int(k), 'n_jobs': group.job_id.nunique()}
            for name in ('C', 'C_multiplier', 'labeled_training_n', 'C_times_labeled_n', 'effective_loss_mass'):
                if name in group:
                    values = pd.to_numeric(group[name], errors='coerce').dropna()
                    if len(values):
                        item[name + '_min'] = float(values.min()); item[name + '_max'] = float(values.max())
            records.append(item)
        h02 = metrics[metrics.family.eq('H02') & ~metrics.method.str.endswith('_argmax')]
        for method, group in h02.groupby('method'):
            item = {'analysis': 'C_constraint_activation_summary', 'family': 'H02', 'method': method,
                    'n_contexts': group.job_id.nunique(), 'mean_accuracy': float(group.accuracy.mean())}
            for name in ('C', 'near_upper_bound_sv_count', 'n_support_vectors', 'max_alpha_over_C', 'source_hinge_loss_sum'):
                if name in group:
                    values = pd.to_numeric(group[name], errors='coerce').dropna()
                    if len(values):
                        item[name + '_mean'] = float(values.mean()); item[name + '_max'] = float(values.max())
            records.append(item)
    if not diagnostics.empty and 'analysis' in diagnostics:
        saved = diagnostics[diagnostics.analysis.eq('weak_certificate_job_summary')]
        for (experiment, method, k), group in saved.groupby(['experiment', 'method', 'k']):
            item = {'analysis': 'weak_certificate_total', 'family': 'H03', 'experiment': experiment,
                    'method': method, 'k': int(k), 'n_jobs': group.job_id.nunique(),
                    'n_predictions': int(pd.to_numeric(group.n_predictions).sum())}
            for name in group:
                if name.endswith('_count'):
                    item[name] = int(pd.to_numeric(group[name], errors='coerce').fillna(0).sum())
            item['interpretation'] = '非零分数修正可保持类别；足够间隔不变性是这些保存点的数值结论，不等于额外效用或新批次鲁棒性'
            records.append(item)
    if not predictions.empty:
        for (family, experiment, method, k, origin), group in predictions.groupby(
                ['family', 'experiment', 'method', 'k', 'origin_group_id']):
            records.append({'analysis': 'per_query_origin_performance', 'family': family,
                            'experiment': experiment, 'method': method, 'k': int(k),
                            'query_origin_group_id': origin, 'n_predictions': len(group),
                            'n_physical_samples': group.physical_sample_id.nunique(),
                            'accuracy': float(group.true_label.eq(group.pred_label).mean())})
    if not interactions.empty:
        records.extend(interactions.to_dict('records'))
    return pd.DataFrame(records)


def curves(metrics):
    if metrics.empty:
        return []
    rows = []
    for (family, experiment, method, k), group in metrics.groupby(['family', 'experiment', 'method', 'k']):
        grouping = group.groupby('composition_id') if family == 'H04' and 'composition_id' in group else [('', group)]
        for composition, block in grouping:
            rows.append({'family': family, 'experiment': experiment, 'method': method, 'k': int(k),
                         'composition_id': composition, 'accuracy': float(block.accuracy.mean()),
                         'macro_f1': float(block.macro_f1.mean()), 'balanced_accuracy': float(block.balanced_accuracy.mean()),
                         'endpoint_count': len(block)})
    return rows


def make_figures(output, metrics, diagnostics):
    if metrics.empty:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(output) / 'figures'; directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for experiment, rows in metrics[metrics.family.eq('H01')].groupby('experiment'):
        figure, axis = plt.subplots(figsize=(8.3, 4.8))
        for method in MASS_METHODS:
            line = rows[rows.method.eq(method)].groupby('k').accuracy.mean().sort_index()
            if len(line): axis.plot(line.index, line * 100, marker='o', label=method)
        baseline = rows[rows.method.eq('identity')].accuracy
        if len(baseline): axis.axhline(baseline.mean() * 100, color='black', linestyle='--', label='Source C10')
        axis.set(xlabel='Support labels per class', ylabel='Mean accuracy (%)', title=f'H01 / {experiment}: all loss-mass controls')
        axis.set_xticks([1, 3, 5, 10]); axis.grid(alpha=.2); axis.legend(fontsize=8)
        figure.tight_layout(); path = directory / f'h01_{experiment.lower()}_loss_mass.png'
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    rows = metrics[metrics.family.eq('H02')]
    if len(rows):
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        for weight in ('balanced', 'none'):
            values, bounds = [], []
            for _, token in C_LEVELS:
                group = rows[rows.method.eq(f'C_{token}_{weight}')]
                values.append(float(group.accuracy.mean()) * 100 if len(group) else np.nan)
                bounds.append(float(pd.to_numeric(group.near_upper_bound_sv_count, errors='coerce').mean())
                              if len(group) and 'near_upper_bound_sv_count' in group else np.nan)
            for axis, series in zip(axes, (values, bounds)):
                axis.plot([.1, 1, 10], series, marker='o', label=weight); axis.set_xscale('log'); axis.grid(alpha=.2)
        axes[0].set(xlabel='Prespecified C', ylabel='Mean accuracy (%)', title='H02: retain every C condition')
        axes[1].set(xlabel='Prespecified C', ylabel='Mean near-bound support vectors', title='H02: constraint activation')
        axes[0].legend(); axes[1].legend(); figure.tight_layout(); path = directory / 'h02_C_constraints.png'
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    if not diagnostics.empty and 'analysis' in diagnostics:
        rows = diagnostics[diagnostics.analysis.eq('weak_certificate_job_summary')]
        if len(rows):
            figure, axis = plt.subplots(figsize=(8.3, 4.6))
            labels, certified, moved = [], [], []
            for (experiment, method), group in rows.groupby(['experiment', 'method']):
                total = pd.to_numeric(group.n_predictions).sum()
                labels.append(experiment + '/' + method)
                certified.append(pd.to_numeric(group.strict_margin_certificate_count).sum() / total * 100)
                moved.append(pd.to_numeric(group.decision_changed_count).sum() / total * 100)
            x = np.arange(len(labels)); axis.bar(x - .18, certified, width=.36, label='strict gap certificate')
            axis.bar(x + .18, moved, width=.36, label='observed class change')
            axis.set_xticks(x, labels, rotation=20, ha='right'); axis.set(ylabel='Saved prediction rows (%)', title='H03: score motion versus decision motion')
            axis.legend(); figure.tight_layout(); path = directory / 'h03_weak_activation.png'
            figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    for experiment, rows in metrics[metrics.family.eq('H04') & metrics.k.eq(-1)].groupby('experiment'):
        figure, axis = plt.subplots(figsize=(9., 4.8))
        compositions = sorted(rows.composition_id.astype(str).unique())
        for method in SUPPORT_METHODS:
            line = rows[rows.method.eq(method)].groupby('composition_id').accuracy.mean()
            axis.plot(np.arange(len(compositions)), [line.get(name, np.nan) * 100 for name in compositions], marker='o', label=method)
        axis.set_xticks(np.arange(len(compositions)), compositions, rotation=20, ha='right')
        axis.set(ylabel='Mean accuracy (%)', xlabel='Frozen Support composition', title=f'H04 / {experiment}: fixed total Support labels')
        axis.grid(alpha=.2); axis.legend(fontsize=8); figure.tight_layout(); path = directory / f'h04_{experiment.lower()}_composition.png'
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(path.relative_to(output).as_posix())
    return paths


def report_text(summary, transitions, mechanisms):
    lines = ['# B方案第四轮：监督损失质量、C约束与Support构成', '',
             f"状态：**{summary['status']}**；已封存 **{summary['completed_jobs']}/{summary['expected_jobs']}** 个任务包。",
             '本报告只读取封存工件，不拟合或选择模型，不修改前三轮数据、标签、模型、协议和结果。', '',
             f"新增分类器拟合{summary['new_model_fits']}次、校准拟合调用{summary['new_adapter_fits']}次；科研任务CPU合计{summary['cpu_seconds']:.6f}秒。",
             f"保存{summary['prediction_rows']}条相关预测，涉及{summary['physical_query_samples']}个唯一Query实物；控制、核验和推送成本另见账本。", '',
             '## 完成范围', '', '|块|问题|完成/计划|状态|', '|---|---|---:|---|']
    labels = {'H01': '监督损失质量与新增样品分解', 'H02': 'C × 类别权重的约束激活',
              'H03': '弱校准分数移动与margin激活审计', 'H04': '固定Support总数的类别构成对照'}
    for family, count in summary['family_task_counts'].items():
        lines.append(f"|{family}|{labels[family]}|{count['completed']}/{count['planned']}|{count['status']}|")
    lines += ['', '## 机制定义与配对限制', '',
              'H01所有条件冻结原Source的Scaler和实际数值gamma。源模型C10为基线；仅提高Source的C到10×N_total/N_source，'
              '不加入Support谱或标签；新增Source+Support则比较固定C10和C=10×N_source/N_total。后者固定C×有标签样品数的总损失系数，'
              '旧固定C端点直接复用。libSVM实际训练one-vs-one子问题，即使输出OVR分数；本块Source各类均衡且每类添加相同K，'
              '因此全局N_total/N_source等于每个OVO类别对的样品数比例，损失质量对照在这个受限设计下成立。'
              '本轮描述固定损失质量下的增量监督收益，不能把新增谱值、标签、训练点和损失项的联合变化唯一归因于抽象标签信息。', '',
              'H02固定Source、Scaler、gamma、Query，预设C=0.1/1/10交叉balanced/None，保存原生SVC与argmax接口。'
              'C10复用；Source类别均衡时两种权重的有效训练目标相同，共享同一模型。C降低属于约束激活的机制干预，不是搜索最佳超参数；'
              '所有三个C均报告，Source训练内约束诊断不用于模型选择。', '',
              'H03完全不拟合：复用第二轮34个弱校准回合及第三轮16个正污染回合，排除已重复的零污染K10。'
              '每个保存点核验旧top2分数间隔g与实际修正无穷范数d，g>2d足以保持argmax；同时记录分数是否真的移动、类别是否改变。'
              '沿固定S(t)=S_source+t×Delta解析计算首次argmax边界，实际t=0.05，不做新blend网格，不依Query真值挑选t。'
              '首次边界≤0.05包含恰好相等的边界，argmax采用首索引平局规则，所以它与实际类别改变数分别保存，不能等同。'
              't>1仅是数学外推，不作为适配效用；未受害不等于有收益，分数移动但不跨margin也不等于完全没有计算。', '',
              'H04二分类Support总数固定10，类别构成为1:9/3:7/5:5/7:3/9:1；五分类总数25，比较5:5:5:5:5及各类轮流稀缺1份、其他类各6份。'
              'draw0/101/102在同一候选池取固定前缀；同一ID比较Anchor/Bias及普通/类别均衡残差拟合。'
              '类别权重N/(类别数×该类支持数)的总和为N，固定lambda1；均衡K5直接复用，权重等价端点共享结果。'
              'composition方法k=-1表示每类K不适用，不表示零支持，实际总量及类别数见support_counts_json。'
              '改变构成同时改变实际Support ID和各类覆盖量，不能称纯先验的单因素因果干预；同一构成内加权与普通拟合使用相同ID，是更直接的权重作用对照。', '',
              '所有Query和Source均来自master TRAIN。Support候选池原本按已知标签分层，构成对照是采样设计诊断，不是已知真实部署先验；'
              '残差分数不称后验概率。类别构成改变同时影响每类信息量和有效权重ESS，不能据此宣称跨批次先验修正。', '',
              '## 全部冻结条件的结果', '',
              '|块|原任务|方法|K/类|Support构成|Accuracy|Macro-F1|Balanced Accuracy|端点数|',
              '|---|---|---|---:|---|---:|---:|---:|---:|']
    for row in summary['curves_macro_over_job_draws']:
        k = '不适用' if row['k'] == -1 else str(row['k'])
        lines.append(f"|{row['family']}|{row['experiment']}|{row['method']}|{k}|{row['composition_id']}|"
                     f"{row['accuracy']:.6f}|{row['macro_f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['endpoint_count']}|")
    lines += ['', '## 配对效应与描述性95%区间', '',
              '保留全部正、负和零效应。以下区间以physical_sample_id聚类；每来源、每类Recall代价、纠错和误伤详见CSV。', '',
              '|块|原任务|比较|K/类|C|比例/覆盖/构成/来源范围|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|',
              '|---|---|---|---:|---:|---|---:|---:|---:|']
    if not transitions.empty:
        for _, row in transitions.iterrows():
            kval = row.get('k', 0); k = '不适用' if int(kval) == -1 else str(int(kval))
            c = '' if pd.isna(row.get('C')) else f"{float(row['C']):g}"
            context = ', '.join(f'{key}={row[key]}' for key in ('ratio', 'coverage', 'composition_id', 'scope')
                                if key in row and not pd.isna(row[key]))
            lines.append(f"|{row['family']}|{row.get('experiment', '')}|{row['comparison']}|{k}|{c}|{context}|"
                         f"{100*row.delta_accuracy:+.3f} [{100*row.ci_lower:+.3f},{100*row.ci_upper:+.3f}]|"
                         f"{int(row.wrong_to_correct)}/{int(row.correct_to_wrong)}|{int(row.n_predictions)}/{int(row.n_physical_samples)}|")
    lines += ['', '## 弱校准激活界限', '',
              '逐点证据在每任务weak_certificates.csv，聚合在mechanism_analysis.csv。严格间隔证书仅证明这些保存分数的类别不变性；'
              '证书未覆盖的点也可能保持类别。解析首次交界不使用Query标签做选择，不代表推荐提高blend；'
              '只有随后对齐真实标签的纠错/误伤才能描述当前查询上的有限效用。', '',
              '## 局限和未运行事项', '', LIMITS, '',
              '人工标签污染证据仅作为既有弱校准路径的诊断副本复用，不证明原始标签有错。'
              '二分类和五分类的类别数、样品组成及Support总预算不同，不直接相减解释类别数作用。'
              '本轮不根据Query选择最终C、Support构成或校准方法。真实新批次继续为 **NOT_RUN_NEEDS_NEW_BATCH_DATA**；'
              '未提供的真实batch_id、仪器及样品属性不推断或补造。', '', '## 复现', '', '```bash',
              'python scripts/run_b_mechanism_round4_20261111.py --resume',
              'python scripts/summarize_b_mechanism_round4_20261111.py --resume',
              'python scripts/summarize_b_mechanism_round4_20261111.py --verify',
              'python scripts/run_b_mechanism_round4_20261111.py --verify', '```', '',
              'PARTIAL报告仅反映当前完整封存子集；核验完成不把未运行任务升级为DONE。']
    return '\n'.join(lines) + '\n'


def summarize(output, *, verify=False, n_boot=1000, report_path=None):
    output = Path(output)
    jobs, predictions, metrics, diagnostics, manifest, checks = load_jobs(output)
    if verify:
        path = output / 'metrics.csv'
        if not path.is_file(): raise ValueError('Missing root metrics.csv')
        saved = pd.read_csv(path, keep_default_na=False)
        if len(saved) != len(metrics): raise ValueError('Root metric row count differs')
        if len(metrics):
            if saved.duplicated(['job_id', 'endpoint_id']).any(): raise ValueError('Duplicate root endpoint')
            before = saved.set_index(['job_id', 'endpoint_id']).sort_index()
            after = metrics.set_index(['job_id', 'endpoint_id']).sort_index()
            if not before.index.equals(after.index): raise ValueError('Root endpoint identity differs')
            for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                if not np.allclose(before[name].astype(float), after[name].astype(float), rtol=0, atol=1e-12):
                    raise ValueError('Root independent metric mismatch: ' + name)
                checks += len(after)
            for name in ('class_recall_json', 'confusion_matrix_json', 'classes_json'):
                if not all(json.loads(a) == json.loads(b) for a, b in zip(before[name], after[name])):
                    raise ValueError('Root independent confusion/vocabulary mismatch')
                checks += len(after)
        receipt = {'status': 'DONE', 'sealed_jobs': len(jobs), 'metric_rows': len(metrics), 'checks': checks,
                   'artifact_hashes_checked': True, 'independent_confusion_count_recomputation': True,
                   'weak_certificates_independently_checked': True,
                   'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        shared.write_json(output / 'summary_verification.json', receipt)
        return receipt
    transitions, intervals, interactions = comparison_tables(predictions, n_boot=n_boot)
    mechanisms = mechanism_table(predictions, metrics, diagnostics, interactions)
    protocol = protocol_payload(output); design, _ = shared.frozen_design(output)
    counts = {}
    for family, fallback in FAMILIES.items():
        planned = sum(job.get('family') == family for job in design) if design else fallback
        done = sum(job.get('family') == family for job in jobs)
        counts[family] = {'completed': done, 'planned': planned,
                          'status': 'DONE' if done == planned else ('PARTIAL' if done else 'NOT_RUN')}
    expected = len(design) if design else sum(FAMILIES.values())
    summary = {'status': 'DONE' if len(jobs) == expected else 'PARTIAL', 'completed_jobs': len(jobs),
               'expected_jobs': expected, 'family_task_counts': counts,
               'new_model_fits': sum(job.get('new_model_fits', 0) for job in jobs),
               'new_adapter_fits': sum(job.get('new_adapter_fits', 0) for job in jobs),
               'planned_new_model_fits': protocol.get('planned_new_model_fits'),
               'planned_new_adapter_fits': protocol.get('planned_new_adapter_fits'),
               'cpu_seconds': sum(job.get('cpu_seconds', 0.) for job in jobs),
               'wall_seconds': sum(job.get('wall_seconds', 0.) for job in jobs),
               'prediction_rows': len(predictions), 'metric_rows': len(metrics),
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
    summary['figures'] = make_figures(output, metrics, diagnostics)
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
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true'); parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap < 100: parser.error('--bootstrap must be at least100')
    try:
        if args.dry_run:
            jobs, predictions, metrics, _, _, checks = load_jobs(args.output); design, _ = shared.frozen_design(args.output)
            print(json.dumps({'status': 'NOT_RUN', 'sealed_jobs': len(jobs), 'prediction_rows': len(predictions),
                              'endpoint_metrics': len(metrics), 'expected_jobs': len(design) if design else 375,
                              'independent_checks': checks, 'fits': 0}, ensure_ascii=False)); return 0
        print(json.dumps(summarize(args.output, verify=args.verify, n_boot=args.bootstrap, report_path=args.report), ensure_ascii=False)); return 0
    except (ValueError, AssertionError) as error:
        print('INTEGRITY_ERROR: ' + str(error), file=sys.stderr); return 42


if __name__ == '__main__':
    raise SystemExit(main())
