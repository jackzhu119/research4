#!/usr/bin/env python3
"""Independently summarize sealed third-round B experiments without fitting.

Repeated seeds, support masks and endpoint rows remain clustered by physical
specimen. Artificial copied-label corruption is a diagnostic, never evidence
that any original annotation is wrong.
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
DEFAULT_OUTPUT = ROOT / 'reports/b_mechanism_round3_20261110'
REPORT = ROOT / 'docs/B_MECHANISM_ROUND3_RESULTS_20261110.md'
FAMILIES = {'G01': 45, 'G02': 45, 'G03': 45, 'G04': 240}
G01_METHODS = ('sum_penalized', 'mean_penalized', 'sum_free_intercept', 'mean_free_intercept')
G02_METHODS = ('unlabeled_statistics', 'true_support', 'cyclic_support')
G03_METHODS = ('raw_anchor_lambda1', 'weak_lambda1000_blend005', 'full_pipeline_fixedgamma')
LIMITS = (
    '本轮是依据已曝光内部结果设计的探索性机制补充，仅使用master TRAIN。重复划分、支持抽样和污染掩码'
    '不增加独立实物数；描述性95%区间按physical_sample_id聚类，保留同一实物全部相关预测，'
    '不覆盖来源/批次总体、协议选择和长期开发偏倚。当前来源与批次混杂，不能据此宣称真实跨批次泛化或确认性因果机制。'
)


def protocol_payload(output):
    path = Path(output) / 'frozen_protocol.json'
    return json.loads(path.read_text()) if path.exists() else {}


def load_jobs(output):
    jobs, predictions, metrics, diagnostics, manifest, checks = shared.load_jobs(output)
    extra = []
    for job in jobs:
        directory = Path(output) / 'jobs' / job['job_id']
        for name in ('margin_mechanism.csv', 'noise_diagnostics.csv', 'label_information.csv'):
            path = directory / name
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, keep_default_na=False)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            for key in ('family', 'job_id', 'experiment', 'target_origin', 'seed', 'ratio', 'coverage', 'fold'):
                frame[key] = job.get(key)
            if 'analysis' not in frame:
                frame['analysis'] = name.removesuffix('.csv')
            extra.append(frame)
    if extra:
        diagnostics = pd.concat(([diagnostics] if not diagnostics.empty else []) + extra, ignore_index=True)
    # Endpoint metadata lives once in metrics, not repeated on every prediction.
    if not predictions.empty:
        add_columns = [name for name in ('actual_noise_rate', 'noise_q_per_class', 'noise_seed',
                                         'diagnostic_labels_changed_count', 'objective_key', 'artificial_label_copy',
                                         'support_noise_fraction', 'total_labeled_training_noise_fraction')
                       if name in metrics and name not in predictions]
        if add_columns:
            predictions = predictions.merge(metrics[['job_id', 'endpoint_id'] + add_columns],
                                            on=['job_id', 'endpoint_id'], validate='many_to_one')
        for column in ('actual_noise_rate', 'noise_q_per_class', 'noise_seed'):
            if column in predictions:
                predictions[column] = pd.to_numeric(predictions[column], errors='coerce')
    return jobs, predictions, metrics, diagnostics, manifest, checks


def interval_for_delta(rows, n_boot):
    low, high = shared.cluster_interval(rows, n_boot=n_boot)
    return {'estimate': float(rows.delta.mean()), 'ci_lower': low, 'ci_upper': high,
            'n_predictions': len(rows), 'n_physical_samples': int(rows.physical_sample_id.nunique()),
            'confidence_level': .95, 'n_bootstrap': n_boot,
            'resampling_unit': 'physical_sample_id', 'interval_scope': LIMITS}


def comparison_tables(predictions, n_boot=1000):
    transitions, intervals, interactions = [], [], []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def add(before, after, info, keys, repeat=False):
        if before.empty or after.empty:
            return None
        shared_jobs = set(before.job_id) & set(after.job_id)
        before = before[before.job_id.isin(shared_jobs)]
        after = after[after.job_id.isin(shared_jobs)]
        if before.empty or after.empty:
            return None
        if repeat:
            before = shared.repeat_baseline(before, after)
        matched = shared.pair(before, after, keys)
        for scope, group in [('ALL', matched)]:
            record, uncertainty = shared.transition_record(group, {**info, 'scope': scope}, n_boot)
            uncertainty['interval_scope'] = LIMITS
            transitions.append(record)
            intervals.append(uncertainty)
        return matched

    def interaction(first, second, info, keys):
        if first is None or second is None:
            return
        rows = first[keys + ['delta']].merge(second[keys + ['delta']], on=keys,
                                             suffixes=('_first', '_second'), how='outer',
                                             indicator=True, validate='one_to_one')
        if not rows._merge.eq('both').all():
            raise ValueError('Incomplete matched cells in factor interaction')
        rows['delta'] = rows.delta_second - rows.delta_first
        value = {**info, 'scope': 'ALL', 'effect': 'difference_of_differences_accuracy',
                 **interval_for_delta(rows, n_boot)}
        intervals.append(value)
        interactions.append({**value, 'analysis': 'factor_interaction'})

    keys = ['job_id', 'draw', 'physical_sample_id']
    for experiment, data in predictions[predictions.family.eq('G01')].groupby('experiment'):
        baseline = data[data.method.eq('identity') & data.k.eq(0)]
        for k in (1, 3, 5, 10):
            for method in G01_METHODS:
                after = data[data.method.eq(method) & data.k.eq(k)]
                add(baseline, after, {'family': 'G01', 'experiment': experiment, 'method': method,
                                     'k': k, 'comparison': method + '_minus_identity'}, keys, repeat=True)
            methods = {method: data[data.method.eq(method) & data.k.eq(k)] for method in G01_METHODS}
            penalized_effect = add(methods['sum_penalized'], methods['mean_penalized'],
                                   {'family': 'G01', 'experiment': experiment, 'k': k,
                                    'comparison': 'mean_minus_sum_penalized_intercept'}, keys)
            free_effect = add(methods['sum_free_intercept'], methods['mean_free_intercept'],
                              {'family': 'G01', 'experiment': experiment, 'k': k,
                               'comparison': 'mean_minus_sum_free_intercept'}, keys)
            for loss in ('sum', 'mean'):
                add(methods[loss + '_penalized'], methods[loss + '_free_intercept'],
                    {'family': 'G01', 'experiment': experiment, 'k': k,
                     'comparison': loss + '_free_minus_penalized_intercept'}, keys)
            interaction(penalized_effect, free_effect,
                        {'family': 'G01', 'experiment': experiment, 'k': k,
                         'comparison': 'loss_normalization_x_free_intercept'}, keys)
        add(data[data.method.eq('svc_predict') & data.k.eq(0)], baseline,
            {'family': 'G01', 'experiment': experiment, 'k': 0,
             'comparison': 'score_argmax_minus_native_svc_zero_support'}, keys)

    for experiment, data in predictions[predictions.family.eq('G02')].groupby('experiment'):
        baseline = data[data.method.eq('identity') & data.k.eq(0)]
        for k in (1, 3, 5, 10):
            blocks = {name: data[data.method.eq(name) & data.k.eq(k)] for name in G02_METHODS}
            for method, after in blocks.items():
                add(baseline, after, {'family': 'G02', 'experiment': experiment, 'method': method,
                                     'k': k, 'comparison': method + '_minus_source_identity'}, keys, repeat=True)
            for first, second, comparison in (
                    ('unlabeled_statistics', 'true_support', 'true_labels_minus_unlabeled_statistics'),
                    ('true_support', 'cyclic_support', 'cyclic_labels_minus_true_labels'),
                    ('unlabeled_statistics', 'cyclic_support', 'cyclic_labels_minus_unlabeled_statistics')):
                add(blocks[first], blocks[second], {'family': 'G02', 'experiment': experiment,
                                                  'k': k, 'comparison': comparison}, keys)

    g03 = predictions[predictions.family.eq('G03')]
    if not g03.empty:
        if 'noise_q_per_class' not in g03:
            raise ValueError('G03 requires frozen actual noise counts in endpoint metadata')
        for experiment, data in g03.groupby('experiment'):
            identity = data[data.method.eq('identity') & data.k.eq(0)]
            for method in G03_METHODS:
                reference = (data[data.method.eq('svc_predict') & data.k.eq(0)]
                             if method == 'full_pipeline_fixedgamma' else identity)
                method_rows = data[data.method.eq(method) & data.k.eq(10)]
                zero = method_rows[method_rows.noise_q_per_class.eq(0)]
                for q, block in method_rows.groupby('noise_q_per_class'):
                    q = int(q)
                    info = {'family': 'G03', 'experiment': experiment, 'method': method, 'k': 10,
                            'noise_q_per_class': q, 'actual_noise_rate': q / 10}
                    add(reference, block, {**info, 'comparison': method + '_minus_matched_source_control'}, keys, repeat=True)
                    add(zero, block, {**info, 'comparison': method + '_noise_minus_true_support'}, keys, repeat=True)
            for q, group in data[data.k.eq(10)].groupby('noise_q_per_class'):
                for first, second in (('raw_anchor_lambda1', 'weak_lambda1000_blend005'),
                                      ('raw_anchor_lambda1', 'full_pipeline_fixedgamma'),
                                      ('weak_lambda1000_blend005', 'full_pipeline_fixedgamma')):
                    add(group[group.method.eq(first)], group[group.method.eq(second)],
                        {'family': 'G03', 'experiment': experiment, 'k': 10,
                         'noise_q_per_class': int(q), 'actual_noise_rate': int(q) / 10,
                         'comparison': second + '_minus_' + first}, keys)
            add(data[data.method.eq('svc_predict') & data.k.eq(0)], identity,
                {'family': 'G03', 'experiment': experiment, 'k': 0,
                 'comparison': 'score_argmax_minus_native_svc_zero_support'}, keys)

    g04 = predictions[predictions.family.eq('G04')]
    if not g04.empty:
        for ratio in sorted(g04.ratio.unique()):
            for coverage in (0, 1):
                data = g04[g04.ratio.eq(ratio) & g04.coverage.eq(coverage)]
                add(data[data.method.eq('class_weight_balanced')], data[data.method.eq('class_weight_none')],
                    {'family': 'G04', 'experiment': 'E01', 'k': 0, 'ratio': ratio,
                     'coverage': coverage, 'comparison': 'unweighted_minus_balanced'}, keys)
        for native in ('class_weight_balanced', 'class_weight_none'):
            add(g04[g04.method.eq(native)], g04[g04.method.eq(native + '_argmax')],
                {'family': 'G04', 'experiment': 'E01', 'k': 0,
                 'comparison': native + '_argmax_minus_native'}, keys)
    return pd.DataFrame(transitions), pd.DataFrame(intervals), pd.DataFrame(interactions)


def truthy(series):
    return series.astype(str).str.lower().isin(('true', '1', '1.0'))


def mechanism_table(predictions, metrics, diagnostics, interactions):
    records = diagnostics.to_dict('records') if not diagnostics.empty else []
    if not metrics.empty:
        for (family, experiment, method, k), group in metrics.groupby(['family', 'experiment', 'method', 'k']):
            by_job = group.groupby('job_id').accuracy.agg(['mean', 'std', 'count'])
            records.append({'analysis': 'within_job_repeat_variability', 'family': family,
                            'experiment': experiment, 'method': method, 'k': int(k),
                            'n_jobs': len(by_job), 'endpoint_rows': len(group),
                            'mean_accuracy_over_endpoints': float(group.accuracy.mean()),
                            'mean_within_job_sd': float(by_job['std'].mean()),
                            'minimum_accuracy': float(group.accuracy.min()),
                            'maximum_accuracy': float(group.accuracy.max()),
                            'interpretation': '重复抽样不等于新增实物；单一端点的SD不定义'})
    if not diagnostics.empty and 'analysis' in diagnostics:
        pair_rows = diagnostics[diagnostics.analysis.eq('weight_pair')]
        for (ratio, coverage), group in pair_rows.groupby(['ratio', 'coverage'], dropna=False):
            item = {'analysis': 'weight_pair_summary', 'family': 'G04', 'ratio': ratio,
                    'coverage': coverage, 'n_contexts': len(group)}
            for column in ('query_labels_identical', 'query_scores_exact_equal', 'scaler_equal',
                           'effective_gamma_equal', 'dual_coefficients_exact_equal', 'support_set_equal'):
                if column in group:
                    item[column + '_count'] = int(truthy(group[column]).sum())
            for column in ('max_query_score_difference', 'max_query_margin_difference',
                           'max_dual_coefficient_difference'):
                if column in group:
                    item[column + '_maximum'] = float(pd.to_numeric(group[column], errors='coerce').max())
            records.append(item)
        model_rows = diagnostics[diagnostics.analysis.eq('model_box_constraints')]
        if not model_rows.empty:
            for setting, group in model_rows.groupby('setting'):
                records.append({'analysis': 'box_constraint_summary', 'family': 'G04', 'setting': setting,
                                'n_contexts': len(group),
                                'near_upper_bound_sv_count_sum': float(pd.to_numeric(group.near_upper_bound_sv_count, errors='coerce').sum()),
                                'maximum_alpha_over_C': float(pd.to_numeric(group.max_alpha_over_C, errors='coerce').max()),
                                'interpretation': '接近边界的数值容差只用于诊断，不是训练终止条件；不外推类别权重普遍无效'})
    if not predictions.empty:
        for (family, experiment, method, k, origin), group in predictions.groupby(
                ['family', 'experiment', 'method', 'k', 'origin_group_id']):
            records.append({'analysis': 'per_query_origin_performance', 'family': family,
                            'experiment': experiment, 'method': method, 'k': int(k),
                            'query_origin_group_id': origin, 'n_predictions': len(group),
                            'n_physical_samples': int(group.physical_sample_id.nunique()),
                            'accuracy': float(group.true_label.eq(group.pred_label).mean())})
    if not interactions.empty:
        records.extend(interactions.to_dict('records'))
    return pd.DataFrame(records)


def make_figures(output, metrics, diagnostics):
    if metrics.empty:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(output) / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    generated = []
    for family, methods in (('G01', G01_METHODS), ('G02', G02_METHODS)):
        rows = metrics[metrics.family.eq(family)]
        for experiment, data in rows.groupby('experiment'):
            figure, axis = plt.subplots(figsize=(8.2, 4.8))
            for method in methods:
                curve = data[data.method.eq(method)].groupby('k').accuracy.mean().sort_index()
                if not curve.empty:
                    axis.plot(curve.index, curve.to_numpy() * 100, marker='o', label=method)
            baseline = data[data.method.eq('identity')].accuracy
            if len(baseline):
                axis.axhline(baseline.mean() * 100, color='black', linestyle='--', label='locked identity')
            axis.set(xlabel='Support labels per class (K)', ylabel='Mean accuracy (%)',
                     title=f'{family} / {experiment}: all prespecified methods')
            axis.set_xticks([1, 3, 5, 10]); axis.grid(alpha=.2); axis.legend(fontsize=8)
            figure.tight_layout()
            path = directory / f'{family.lower()}_{experiment.lower()}_curves.png'
            figure.savefig(path, dpi=160); plt.close(figure)
            generated.append(path.relative_to(output).as_posix())
    rows = metrics[metrics.family.eq('G03')]
    if 'actual_noise_rate' in rows:
        for experiment, data in rows.groupby('experiment'):
            figure, axis = plt.subplots(figsize=(8.2, 4.8))
            data = data.copy(); data['actual_noise_rate'] = pd.to_numeric(data.actual_noise_rate, errors='coerce')
            for method in G03_METHODS:
                curve = data[data.method.eq(method)].groupby('actual_noise_rate').accuracy.mean().sort_index()
                if not curve.empty:
                    axis.plot(curve.index * 100, curve.to_numpy() * 100, marker='o', label=method)
            axis.set(xlabel='Artificial support-label corruption (%)', ylabel='Mean accuracy (%)',
                     title=f'G03 / {experiment}: copied-label diagnostic')
            axis.set_xticks([0, 10, 20, 40, 100]); axis.grid(alpha=.2); axis.legend(fontsize=8)
            figure.tight_layout()
            path = directory / f'g03_{experiment.lower()}_noise_dose.png'
            figure.savefig(path, dpi=160); plt.close(figure)
            generated.append(path.relative_to(output).as_posix())
    if not diagnostics.empty and 'analysis' in diagnostics:
        pairs = diagnostics[diagnostics.analysis.eq('weight_pair')]
        if len(pairs) and 'max_query_margin_difference' in pairs:
            values = pd.to_numeric(pairs.max_query_margin_difference, errors='coerce').dropna()
            if not len(values) or np.ptp(values.to_numpy()) == 0:
                return generated
            figure, axis = plt.subplots(figsize=(7.4, 4.4))
            axis.hist(values, bins=20, color='#477fb3', edgecolor='white')
            axis.set(xlabel='Maximum absolute paired query-margin change', ylabel='Number of source contexts',
                     title='G04: paired query-margin difference audit')
            figure.tight_layout(); path = directory / 'g04_margin_change.png'
            figure.savefig(path, dpi=160); plt.close(figure)
            generated.append(path.relative_to(output).as_posix())
    return generated


def curves(metrics):
    if metrics.empty:
        return []
    result = []
    for (family, experiment, method, k), group in metrics.groupby(['family', 'experiment', 'method', 'k']):
        if family == 'G03' and 'actual_noise_rate' in group:
            for dose, block in group.groupby('actual_noise_rate', dropna=False):
                result.append({'family': family, 'experiment': experiment, 'method': method, 'k': int(k),
                               'actual_noise_rate': dose, 'accuracy': float(block.accuracy.mean()),
                               'macro_f1': float(block.macro_f1.mean()),
                               'balanced_accuracy': float(block.balanced_accuracy.mean()), 'endpoint_count': len(block)})
        else:
            result.append({'family': family, 'experiment': experiment, 'method': method, 'k': int(k),
                           'accuracy': float(group.accuracy.mean()), 'macro_f1': float(group.macro_f1.mean()),
                           'balanced_accuracy': float(group.balanced_accuracy.mean()), 'endpoint_count': len(group)})
    return result


def report_text(summary, transitions, mechanisms):
    lines = ['# B方案第三轮：正则化、增量监督与类别权重边界', '',
             f"状态：**{summary['status']}**；已封存 **{summary['completed_jobs']}/{summary['expected_jobs']}** 个任务包。",
             '本报告只读取完整封存工件，不拟合模型，不修改前两轮代码、模型、数据、标签或结果。', '',
             f"新增分类器拟合{summary['new_model_fits']}次、校准拟合调用{summary['new_adapter_fits']}次；"
             f"科研任务CPU合计{summary['cpu_seconds']:.6f}秒，控制/核验/推送成本另见后台账本。",
             f"已保存{summary['prediction_rows']}条预测，对应{summary['physical_query_samples']}个唯一Query实物；重复记录不是独立样本。", '',
             '## 完成范围', '', '|块|问题|完成/计划|状态|', '|---|---|---:|---|']
    labels = {'G01': '损失归一化 × 截距惩罚', 'G02': '增量监督训练与无标签统计量分解',
              'G03': '人工支持标签污染剂量', 'G04': '类别权重、margin与对偶约束的零拟合审计'}
    for family, counts in summary['family_task_counts'].items():
        lines.append(f"|{family}|{labels.get(family, family)}|{counts['completed']}/{counts['planned']}|{counts['status']}|")
    lines += ['', '## 冻结设计与解释边界', '',
              'G01在同一源分数、支持ID和Query上比较sum-loss λ1与mean-loss λ1（等价sum λ=N），交叉截距惩罚λ或0。'
              '原sum/惩罚截距端点复用第二轮Anchor；新增参数不依据Query选择。draw0/101/102在每类10份池内嵌套K1/3/5/10，'
              '真标签K10只保留一次。保存有效自由度、秩、条件数、跨类耦合和修正范数。', '',
              'G02保持数值gamma、C及支持预算固定。无标签统计条件只用Source+Support的谱拟合Scaler，再只用Source标签重训SVC；'
              '真实支持与循环错标条件使用完全相同的Source+Support谱和Scaler，只有Support标签副本不同。'
              '没有把新Scaler直接接到旧SVC上冒充一致的适配。真实标签重训端点优先复用。'
              'true_support相对unlabeled_statistics同时加入监督训练样品和损失项；固定C下样本数变化也改变总损失与正则项的相对关系。'
              '因此该差值描述增量监督训练，不唯一识别标签信息的单一因果作用。true_support与cyclic_support保持训练样本数、'
              'Scaler、gamma和标签直方图相同，是更直接的支持标签正确性对照。'
              'Support候选池原本由已知标签分层构造，所有条件共用该池；无标签统计条件指额外Support标签值不进入拟合，'
              '不能据此称实际采样流程完全无需标签或证明外部无监督适配。', '',
              'G03仅在每类10份Support上人工改变1/2/4/10份，实现精确10/20/40/100%污染。循环类标签保留直方图，'
              '原元数据及真值不变；100%完整污染端点只保留一次并复用G02模型，完全相同的掩码目标共享拟合。'
              '保留随机掩码的抽样权重，但不把重复当成新实物。它不是独立同分布现场误标，也不证明现有标签错误。', '',
              '污染百分比的分母仅为Support；Source标签保持正确。每类改变4/10份时，五来源为20/50份Support受扰，'
              '但占完整190+50训练集的20/240=8.33%；二分类为8/20份Support受扰，占240+20训练集的8/260=3.08%。'
              '各端点另存总有标签训练集污染比例，不能把40%Support污染表述为全部训练数据40%污染。', '',
              'G04只复用第二轮240个来源上下文的模型、原始预测和分数。类别决策一致时仍检查分数、排序、AUROC、'
              '支持向量和α/C约束。接近边界的数值容差仅是诊断，既不终止训练，也不作为通用类别权重无效的证明。', '',
              '## 全部方法的描述性结果', '',
              '以下端点按任务及合法draw等权汇总；K10、零污染及100%污染的单一端点不会被伪装为多次新实验。', '',
              '|块|原任务|方法|K/类|Support污染比例|Accuracy|Macro-F1|Balanced Accuracy|端点数|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['curves_macro_over_job_draws']:
        dose = row.get('actual_noise_rate')
        dose_text = '' if dose is None or not np.isfinite(float(dose)) else f'{float(dose):.0%}'
        lines.append(f"|{row['family']}|{row['experiment']}|{row['method']}|{row['k']}|{dose_text}|"
                     f"{row['accuracy']:.6f}|{row['macro_f1']:.6f}|{row['balanced_accuracy']:.6f}|{row['endpoint_count']}|")
    lines += ['', '## 配对变化与描述性区间', '',
              '全部方法、剂量、交互、逐类Recall变化及纠错/误伤在CSV中保留。下面列全体匹配Query的对照，'
              '按physical_sample_id聚类，不把支持draw、污染掩码或重复外折行当独立观察。', '',
              '|块|原任务|比较|K/类|Support污染比例|ΔAccuracy百分点及95%区间|纠错/误伤|预测行/实物|',
              '|---|---|---|---:|---:|---:|---:|---:|']
    if not transitions.empty:
        for _, row in transitions.iterrows():
            dose = row.get('actual_noise_rate')
            dose_text = '' if pd.isna(dose) else f'{float(dose):.0%}'
            lines.append(f"|{row['family']}|{row.get('experiment', '')}|{row['comparison']}|{int(row.get('k', 0))}|{dose_text}|"
                         f"{100 * row.delta_accuracy:+.3f} [{100 * row.ci_lower:+.3f},{100 * row.ci_upper:+.3f}]|"
                         f"{int(row.wrong_to_correct)}/{int(row.correct_to_wrong)}|{int(row.n_predictions)}/{int(row.n_physical_samples)}|")
    lines += ['', '## 权重与margin审计', '',
              '逐模型box约束、Source hinge loss、支持向量对应及成对分数结果见mechanism_analysis.csv与每任务margin_mechanism.csv。'
              '若两种权重下类别预测相同，只能先说明这些已保存Query上的决策一致；若对偶系数和截距一致，'
              '且两组箱约束均未激活，才支持当前Source上下文未激活权重差异的有限解释。'
              'A/C比例下箱约束上限C_i按设计不同，不能将它们说成约束相同；也不能据此概括所有来源或未来批次。', '',
              '## 局限与未运行事项', '', LIMITS, '',
              '二分类与五来源的样本组成、类别数、训练量及支持标签预算不同，不能直接相减解释为类别数的单因素作用。'
              'Source训练内诊断不被用于选择正则化、污染掩码或最终方法。所有正、负、零效应保留，不从多重比较挑单个正结果作确认性结论。', '',
              '真实新批次评价继续为 **NOT_RUN_NEEDS_NEW_BATCH_DATA**。尚未采集的批次、样品、仪器或物理属性不推断补填。', '',
              '## 复现', '', '```bash',
              'python scripts/run_b_mechanism_round3_20261110.py --resume',
              'python scripts/summarize_b_mechanism_round3_20261110.py --resume',
              'python scripts/summarize_b_mechanism_round3_20261110.py --verify',
              'python scripts/run_b_mechanism_round3_20261110.py --verify', '```', '',
              '训练中途PARTIAL仅表示已封存子集。summary_verification.json保存独立混淆计数复算和工件核验依据。']
    return '\n'.join(lines) + '\n'


def summarize(output, *, verify=False, n_boot=1000, report_path=None):
    output = Path(output)
    jobs, predictions, metrics, diagnostics, manifest, checks = load_jobs(output)
    if verify:
        path = output / 'metrics.csv'
        if not path.exists():
            raise ValueError('Missing root metrics.csv for independent verification')
        saved = pd.read_csv(path, keep_default_na=False)
        if len(saved) != len(metrics):
            raise ValueError('Root metric count differs from sealed endpoint count')
        if len(metrics):
            if saved.duplicated(['job_id', 'endpoint_id']).any():
                raise ValueError('Duplicate root endpoint metrics')
            before = saved.set_index(['job_id', 'endpoint_id']).sort_index()
            after = metrics.set_index(['job_id', 'endpoint_id']).sort_index()
            if not before.index.equals(after.index):
                raise ValueError('Root endpoint identity differs')
            for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                if not np.allclose(before[name].astype(float), after[name].astype(float), atol=1e-12, rtol=0):
                    raise ValueError('Root independent metric mismatch: ' + name)
                checks += len(after)
            for name in ('class_recall_json', 'confusion_matrix_json', 'classes_json'):
                if not all(json.loads(a) == json.loads(b) for a, b in zip(before[name], after[name])):
                    raise ValueError('Root independent matrix/vocabulary mismatch: ' + name)
                checks += len(after)
        receipt = {'status': 'DONE', 'sealed_jobs': len(jobs), 'metric_rows': len(metrics), 'checks': checks,
                   'artifact_hashes_checked': True, 'independent_confusion_count_recomputation': True,
                   'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        shared.write_json(output / 'summary_verification.json', receipt)
        return receipt
    transitions, intervals, interactions = comparison_tables(predictions, n_boot=n_boot)
    mechanisms = mechanism_table(predictions, metrics, diagnostics, interactions)
    protocol = protocol_payload(output)
    design, _ = shared.frozen_design(output)
    expected_by_family = {family: sum(job.get('family') == family for job in design) if design else count
                          for family, count in FAMILIES.items()}
    family_counts = {}
    for family, planned in expected_by_family.items():
        done = sum(job.get('family') == family for job in jobs)
        family_counts[family] = {'completed': done, 'planned': planned,
                                 'status': 'DONE' if done == planned else ('PARTIAL' if done else 'NOT_RUN')}
    expected = len(design) if design else sum(expected_by_family.values())
    summary = {'status': 'DONE' if len(jobs) == expected else 'PARTIAL',
               'completed_jobs': len(jobs), 'expected_jobs': expected, 'family_task_counts': family_counts,
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
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'curves_macro_over_job_draws': curves(metrics)}
    if summary['status'] == 'DONE':
        for actual, planned in (('new_model_fits', 'planned_new_model_fits'), ('new_adapter_fits', 'planned_new_adapter_fits')):
            if summary[planned] is not None and summary[actual] != summary[planned]:
                raise ValueError('Actual fit count differs from frozen count: ' + actual)
    shared.write_csv(output / 'metrics.csv', metrics)
    shared.write_csv(output / 'run_manifest.csv', manifest)
    shared.write_csv(output / 'paired_transitions.csv', transitions)
    shared.write_csv(output / 'uncertainty.csv', intervals)
    shared.write_csv(output / 'mechanism_analysis.csv', mechanisms)
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
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least100')
    try:
        if args.dry_run:
            jobs, predictions, metrics, _, _, checks = load_jobs(args.output)
            design, _ = shared.frozen_design(args.output)
            print(json.dumps({'status': 'NOT_RUN', 'sealed_jobs': len(jobs), 'prediction_rows': len(predictions),
                              'endpoint_metrics': len(metrics), 'expected_jobs': len(design) if design else 375,
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
