"""Summarize completed TRAIN-only background jobs without touching holdouts.

This program reads only artifacts below the background output directory. It
independently reconstructs confusion matrices from saved development-fold
predictions. Missing folds remain incomplete and cannot enter recommendations.
Repeated model seeds or support draws never increase unique specimen counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'data/research/species_background_20261108'
CLASSIC_SEEDS = (20261031, 20261108, 42)
NEURAL_SEEDS = (42, 43, 44)
FOLDS = frozenset(range(5))
CLASSIC_METHODS = ('R0', 'R2', 'fullSVM', 'PLSDA')
GROUP_FIELDS = ('family', 'task', 'representation', 'profile', 'pretrain_steps',
                'method', 'prediction_semantics', 'calibration', 'shots_per_class')
PAIR_FIELDS = ('family', 'task', 'representation', 'profile', 'pretrain_steps',
               'comparison', 'prediction_semantics', 'calibration', 'shots_per_class')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def atom(value):
    return sys.intern(str(value))


def integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} cannot be boolean')
    result = int(value)
    if str(value) not in (str(result), str(float(result))):
        raise ValueError(f'{name} must be integral: {value}')
    return result


def confusion_metrics(confusion, labels):
    """Compute metrics from counts, independently of worker metric helpers."""
    labels = tuple(str(label) for label in labels)
    total = sum(confusion.values())
    if total <= 0:
        raise ValueError('Metrics require nonempty predictions')
    if any(t not in labels or p not in labels for t, p in confusion):
        raise ValueError('Prediction or truth lies outside the fixed class order')
    correct = sum(confusion.get((label, label), 0) for label in labels)
    class_metrics = []
    for label in labels:
        tp = confusion.get((label, label), 0)
        support = sum(value for (truth, _), value in confusion.items() if truth == label)
        predicted = sum(value for (_, pred), value in confusion.items() if pred == label)
        denominator = support + predicted
        class_metrics.append({'label': label, 'support': support, 'predicted': predicted,
                              'correct': tp, 'recall': tp / support if support else None,
                              'f1': 2 * tp / denominator if denominator else 0.})
    recalls = [row['recall'] for row in class_metrics if row['recall'] is not None]
    return {'correct': correct, 'denominator': total, 'accuracy': correct / total,
            'macro_f1': sum(row['f1'] for row in class_metrics) / len(labels),
            'balanced_accuracy': sum(recalls) / len(recalls),
            'class_order': list(labels), 'per_class': class_metrics,
            'confusion_matrix': [[confusion.get((truth, pred), 0) for pred in labels] for truth in labels]}


@dataclass
class Replicate:
    labels: tuple
    expected_ids: frozenset
    confusion: Counter = field(default_factory=Counter)
    identifiers: set = field(default_factory=set)
    folds: set = field(default_factory=set)
    fold_sizes: dict = field(default_factory=dict)
    invalid_reasons: list = field(default_factory=list)

    def add(self, fold, predictions):
        if fold in self.folds:
            self.invalid_reasons.append(f'duplicate outer fold {fold}')
            return
        ids = set()
        for row in predictions:
            identifier = atom(row['record_id'])
            if identifier in ids or identifier in self.identifiers:
                self.invalid_reasons.append(f'repeated specimen {identifier} within one OOF replicate')
            if identifier not in self.expected_ids:
                self.invalid_reasons.append(f'non-TRAIN or unknown specimen {identifier}')
            ids.add(identifier)
            self.confusion[(atom(row['truth']), atom(row['prediction']))] += 1
        self.identifiers |= ids
        self.folds.add(fold)
        self.fold_sizes[str(fold)] = len(predictions)

    def result(self, replicate_key):
        complete = (self.folds == FOLDS and self.identifiers == set(self.expected_ids)
                    and not self.invalid_reasons)
        result = {'replicate': list(replicate_key), 'status': 'complete' if complete else 'incomplete',
                  'completed_folds': sorted(self.folds), 'expected_folds': sorted(FOLDS),
                  'fold_sizes': self.fold_sizes, 'unique_specimens': len(self.identifiers),
                  'expected_unique_specimens': len(self.expected_ids),
                  'missing_specimens': len(self.expected_ids - self.identifiers),
                  'invalid_reasons': self.invalid_reasons,
                  'metrics': confusion_metrics(self.confusion, self.labels)}
        return result


def summarize_replicates(replicates, expected_seeds):
    rows = [rep.result(key) for key, rep in sorted(replicates.items())]
    complete = [row for row in rows if row['status'] == 'complete']
    seeds = {row['replicate'][0] for row in complete}
    all_present_complete = len(complete) == len(rows)
    expected = set(expected_seeds)
    eligible = all_present_complete and seeds == expected
    expected_ids = {rep.expected_ids for rep in replicates.values()}
    if len(expected_ids) != 1:
        eligible = False
    result = {'status': 'complete' if eligible else 'incomplete',
              'eligible_for_development_selection': eligible,
              'complete_oof_replicates': len(complete), 'observed_oof_replicates': len(rows),
              'expected_model_or_split_seeds': sorted(expected),
              'complete_model_or_split_seeds': sorted(seeds),
              'unique_specimens': len(set().union(*(rep.identifiers for rep in replicates.values()))),
              'independent_sample_count_policy': 'Unique physical specimens only; seeds/support draws are repeated predictions.',
              'replicates': rows}
    if complete:
        result['complete_oof_metric_mean'] = {
            name: statistics.mean(row['metrics'][name] for row in complete)
            for name in ('accuracy', 'macro_f1', 'balanced_accuracy')}
        result['complete_oof_metric_range'] = {
            name: [min(row['metrics'][name] for row in complete),
                   max(row['metrics'][name] for row in complete)]
            for name in ('accuracy', 'macro_f1', 'balanced_accuracy')}
        result['repeated_prediction_correct'] = sum(row['metrics']['correct'] for row in complete)
        result['repeated_prediction_denominator'] = sum(row['metrics']['denominator'] for row in complete)
    return result


def check_receipt(directory):
    classic = (directory / 'job_manifest.json').is_file()
    configuration_name = 'job_manifest.json' if classic else 'protocol.json'
    required = (configuration_name, 'summary.json', 'validation_predictions.csv.gz', 'complete.json')
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return None, {'status': 'incomplete', 'missing_artifacts': missing}
    done = read_json(directory / 'complete.json')
    if done.get('status') != 'complete':
        return None, {'status': 'incomplete', 'reason': 'complete.json does not declare completion'}
    manifest = read_json(directory / configuration_name)
    manifest_hash = digest(directory / configuration_name)
    identity = (manifest_hash if classic else hashlib.sha256(json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest())
    receipt_identity = 'job_manifest_sha256' if classic else 'identity_sha256'
    if done.get(receipt_identity) != identity:
        raise ValueError('Frozen job identity disagrees with completion receipt')
    receipt = done.get('artifacts_sha256' if classic else 'artifacts', {})
    checked = [configuration_name, 'summary.json', 'validation_predictions.csv.gz']
    if not classic:
        checked += ['validation_truth.json', 'support_draws.json']
    for name in checked:
        if receipt.get(name) != digest(directory / name):
            raise ValueError(f'Completed artifact lacks a valid receipt hash: {name}')
    summary = read_json(directory / 'summary.json')
    if summary.get(receipt_identity) != identity:
        raise ValueError('Summary belongs to a different frozen job')
    return (manifest, summary), {'status': 'complete', 'job_identity_sha256': identity,
                                 'configuration_sha256': manifest_hash,
                                 'prediction_sha256': receipt['validation_predictions.csv.gz']}


def scope_check(manifest, summary):
    negative_flags = ('master_calibration_or_prediction_used',
                      'master_calibration_or_prediction_used_for_fit_or_selection')
    observed = []
    for payload in (manifest, summary):
        observed.extend(payload[name] for name in negative_flags if name in payload)
    if not observed or any(value is not False for value in observed):
        raise ValueError('A completed job must explicitly exclude master calibration/prediction use')
    expected = manifest.get('master_training_record_ids')
    if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
        raise ValueError('Job needs a unique, explicit master-TRAIN specimen allowlist')
    source = set(manifest.get('source_record_ids', []))
    valid = set(manifest.get('outer_validation_record_ids', []))
    allowed = set(expected)
    if not source or not valid or source & valid or not (source | valid) <= allowed:
        raise ValueError('Invalid or overlapping TRAIN-development source/validation identities')
    return frozenset(atom(value) for value in expected), frozenset(atom(value) for value in valid)


def csv_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def add_group(groups, metadata, seed, draw, fold, expected_ids, labels, predictions):
    key = tuple(metadata[field] for field in GROUP_FIELDS)
    replicate_key = (seed, draw)
    if key not in groups:
        groups[key] = {}
    if replicate_key not in groups[key]:
        groups[key][replicate_key] = Replicate(tuple(labels), expected_ids)
    replica = groups[key][replicate_key]
    if replica.expected_ids != expected_ids or replica.labels != tuple(labels):
        raise ValueError('OOF replicate changed class order or TRAIN population between folds')
    replica.add(fold, predictions)


def add_pairs(pairs, metadata, seed, draw, fold, expected_ids, labels, arms):
    """Pair predictions by actual record IDs before aggregating any effect."""
    for left, right in (('A', 'B'), ('Aextra', 'B'), ('matched_SVM', 'B')):
        if left not in arms or right not in arms:
            continue
        first = {row['record_id']: row for row in arms[left]}
        second = {row['record_id']: row for row in arms[right]}
        if len(first) != len(arms[left]) or len(second) != len(arms[right]) or first.keys() != second.keys():
            raise ValueError('Paired neural arms need identical nonduplicated validation specimens')
        meta = {**metadata, 'comparison': f'{right}-{left}'}
        key = tuple(meta[field] for field in PAIR_FIELDS)
        replica_key = (seed, draw)
        if key not in pairs:
            pairs[key] = {}
        if replica_key not in pairs[key]:
            pairs[key][replica_key] = {
                'left': Replicate(tuple(labels), expected_ids),
                'right': Replicate(tuple(labels), expected_ids), 'improved': 0, 'harmed': 0}
        item = pairs[key][replica_key]
        item['left'].add(fold, list(first.values()))
        item['right'].add(fold, list(second.values()))
        for identifier, left_row in first.items():
            right_row = second[identifier]
            if left_row['truth'] != right_row['truth']:
                raise ValueError('Paired arms disagree on validation truth')
            a = left_row['prediction'] == left_row['truth']
            b = right_row['prediction'] == right_row['truth']
            item['improved'] += int(b and not a)
            item['harmed'] += int(a and not b)


def consume_classic(directory, manifest, summary, groups, global_allowlist):
    expected, validation = scope_check(manifest, summary)
    if not expected <= global_allowlist:
        raise ValueError('Classic job contains IDs outside the frozen global TRAIN-only allowlist')
    labels = tuple(manifest['class_order'])
    fold = integer(manifest['outer_fold'], 'outer_fold')
    seed = integer(manifest['outer_split_seed'], 'outer_split_seed')
    if fold not in FOLDS or seed not in CLASSIC_SEEDS:
        raise ValueError('Classic fold or repeat seed lies outside the frozen matrix')
    rows = csv_rows(directory / 'validation_predictions.csv.gz')
    methods = {row['method'] for row in rows}
    if methods != set(CLASSIC_METHODS):
        raise ValueError('Classic predictions do not contain all four complete methods')
    for method in CLASSIC_METHODS:
        selected = [row for row in rows if row['method'] == method]
        if len(selected) != len(validation) or {row['record_id'] for row in selected} != set(validation):
            raise ValueError(f'Classic {method} does not cover the frozen outer validation set')
        if any(row['y_true'] not in labels or row['prediction_vote'] not in labels
               or row['prediction_score_argmax'] not in labels for row in selected):
            raise ValueError('Classic prediction/truth outside the fixed class order')
        for semantics, column in (('estimator_vote', 'prediction_vote'),
                                   ('score_argmax', 'prediction_score_argmax')):
            predictions = [{'record_id': row['record_id'], 'truth': row['y_true'],
                            'prediction': row[column]} for row in selected]
            metadata = {'family': 'classic', 'task': manifest['task'],
                        'representation': manifest['representation'], 'profile': '',
                        'pretrain_steps': 0, 'method': method, 'prediction_semantics': semantics,
                        'calibration': 'none', 'shots_per_class': 0}
            add_group(groups, metadata, seed, -1, fold, expected, labels, predictions)


def consume_neural(directory, manifest, summary, groups, pairs, global_allowlist):
    """Join only training-development truth locked by the neural worker."""
    if (manifest.get('no_master_calibration_or_prediction_used') is not True
            or summary.get('master_calibration_or_prediction_used') is not False
            or summary.get('validation_is_within_master_training') is not True):
        raise ValueError('Neural job lacks explicit TRAIN-only scope')
    source, support, validation = [frozenset(atom(x) for x in manifest[name])
                                   for name in ('source_ids', 'support_ids', 'validation_ids')]
    if not source or not support or not validation or source & support or source & validation or support & validation:
        raise ValueError('Neural source/support/validation identities overlap or are empty')
    expected = source | support | validation
    auxiliary = set().union(*(set(ids) for ids in manifest['auxiliary_ids'].values()))
    if not (expected | auxiliary) <= global_allowlist or expected & auxiliary:
        raise ValueError('Neural target/auxiliary IDs violate the frozen TRAIN-only allowlist')
    labels = tuple(summary['classes'])
    fold = integer(manifest['fold'], 'fold')
    seed = integer(manifest['seed'], 'seed')
    if fold not in FOLDS or seed not in NEURAL_SEEDS:
        raise ValueError('Neural fold or model seed lies outside frozen matrix')
    profile = manifest['profile']['name']
    steps = integer(manifest['pretrain_steps'], 'pretrain_steps')
    if summary.get('reference_same_source_and_128point_representation') is not True:
        raise ValueError('Neural SVM reference is not declared matched on source/representation')
    truth = read_json(directory / 'validation_truth.json')
    if set(truth) != set(validation) or any(label not in labels for label in truth.values()):
        raise ValueError('Neural development truth does not cover frozen validation identities/classes')
    draws = read_json(directory / 'support_draws.json')
    expected_draws = {(integer(d['repeat'], 'support repeat'), integer(d['shots'], 'shots')) for d in draws}
    repetitions = integer(manifest['support_repeats'], 'support_repeats')
    shots = tuple(integer(k, 'shots') for k in manifest['shots'])
    if expected_draws != {(repeat, k) for repeat in range(repetitions) for k in shots}:
        raise ValueError('Neural support draws do not cover the prespecified repeated K matrix')
    for draw in draws:
        ids = draw['support_ids']
        if len(ids) != len(set(ids)) or not set(ids) <= support:
            raise ValueError('A neural support draw contains non-support or duplicated identities')
    rows = csv_rows(directory / 'validation_predictions.csv.gz')
    settings = defaultdict(list)
    for row in rows:
        settings[(row['arm'], row['method'], integer(row['repeat'], 'repeat'),
                  integer(row['shots'], 'shots'))].append(row)
    arms = ('A', 'Aextra', 'B', 'SVM_reference')
    calibrators = tuple(candidate['id'] for candidate in manifest['calibrators'])
    required_settings = {(arm, 'source_only', -1, 0) for arm in arms}
    required_settings |= {(arm, method, repeat, k) for arm in arms for method in calibrators
                          for repeat, k in expected_draws}
    if set(settings) != required_settings:
        raise ValueError('Neural prediction settings do not exactly cover all arms, calibrators and support draws')
    paired = defaultdict(dict)
    task = {'suanzaoren': 'acid_origins', 'huangqi': 'huangqi_origins', 'tianma': 'tianma_origins'}[manifest['task']]
    for (arm, method, repeat, k), values in settings.items():
        if len(values) != len(validation) or {row['record_id'] for row in values} != set(validation):
            raise ValueError('Neural setting omits or duplicates validation specimens')
        predictions = [{'record_id': row['record_id'], 'truth': str(truth[row['record_id']]),
                        'prediction': row['predicted']} for row in values]
        if any(row['prediction'] not in labels for row in predictions):
            raise ValueError('Neural prediction outside fixed target classes')
        normalized_arm = 'matched_SVM' if arm == 'SVM_reference' else arm
        metadata = {'family': 'neural', 'task': task, 'representation': 'alpha_transfer_128',
                    'profile': profile, 'pretrain_steps': steps, 'method': normalized_arm,
                    'prediction_semantics': 'score_argmax', 'calibration': method,
                    'shots_per_class': k}
        add_group(groups, metadata, seed, repeat, fold, expected, labels, predictions)
        paired[(method, repeat, k)][normalized_arm] = predictions
    for (method, repeat, k), values in paired.items():
        metadata = {'family': 'neural', 'task': task, 'representation': 'alpha_transfer_128',
                    'profile': profile, 'pretrain_steps': steps,
                    'prediction_semantics': 'score_argmax', 'calibration': method, 'shots_per_class': k}
        add_pairs(pairs, metadata, seed, repeat, fold, expected, labels, values)


def merge_replicate(destination, incoming):
    if destination.expected_ids != incoming.expected_ids or destination.labels != incoming.labels:
        raise ValueError('Cross-fold class order or expected TRAIN population drift')
    if destination.folds & incoming.folds or destination.identifiers & incoming.identifiers:
        raise ValueError('Duplicate development fold or specimen across completed jobs')
    destination.confusion.update(incoming.confusion)
    destination.identifiers |= incoming.identifiers
    destination.folds |= incoming.folds
    destination.fold_sizes.update(incoming.fold_sizes)
    destination.invalid_reasons.extend(incoming.invalid_reasons)


def merge_job(groups, pairs, incoming_groups, incoming_pairs):
    # Preflight all joins before modifying aggregate state; invalid job outputs
    # never leave partial group contributions behind.
    for key, replicas in incoming_groups.items():
        for replica_key, incoming in replicas.items():
            existing = groups.get(key, {}).get(replica_key)
            if existing and (existing.expected_ids != incoming.expected_ids or existing.labels != incoming.labels
                             or existing.folds & incoming.folds or existing.identifiers & incoming.identifiers):
                raise ValueError('Repeated fold/specimen or changed population in background job')
    for key, replicas in incoming_pairs.items():
        for replica_key, incoming in replicas.items():
            existing = pairs.get(key, {}).get(replica_key)
            if existing:
                for side in ('left', 'right'):
                    a, b = existing[side], incoming[side]
                    if a.expected_ids != b.expected_ids or a.labels != b.labels or a.folds & b.folds or a.identifiers & b.identifiers:
                        raise ValueError('Repeated paired fold/specimen or changed population')
    for key, replicas in incoming_groups.items():
        for replica_key, incoming in replicas.items():
            existing = groups.setdefault(key, {}).get(replica_key)
            if existing is None:
                groups[key][replica_key] = incoming
            else:
                merge_replicate(existing, incoming)
    for key, replicas in incoming_pairs.items():
        for replica_key, incoming in replicas.items():
            existing = pairs.setdefault(key, {}).get(replica_key)
            if existing is None:
                pairs[key][replica_key] = incoming
            else:
                for side in ('left', 'right'):
                    merge_replicate(existing[side], incoming[side])
                existing['improved'] += incoming['improved']
                existing['harmed'] += incoming['harmed']


def pair_results(pairs):
    output = []
    for key, values in sorted(pairs.items()):
        meta = dict(zip(PAIR_FIELDS, key))
        rows = []
        for replica_key, item in sorted(values.items()):
            left, right = item['left'].result(replica_key), item['right'].result(replica_key)
            complete = left['status'] == right['status'] == 'complete'
            rows.append({'replicate': list(replica_key), 'status': 'complete' if complete else 'incomplete',
                         'left_metrics': left['metrics'], 'right_metrics': right['metrics'],
                         'completed_folds': left['completed_folds'],
                         'unique_specimens': left['unique_specimens'],
                         'improved': item['improved'], 'harmed': item['harmed'],
                         'accuracy_gain_percentage_points': 100 * (right['metrics']['accuracy'] - left['metrics']['accuracy']),
                         'macro_f1_gain': right['metrics']['macro_f1'] - left['metrics']['macro_f1']})
        complete = [row for row in rows if row['status'] == 'complete']
        eligible = (len(complete) == len(rows) and {row['replicate'][0] for row in complete} == set(NEURAL_SEEDS))
        result = {**meta, 'status': 'complete' if eligible else 'incomplete',
                  'eligible_for_development_selection': eligible,
                  'replicates': rows, 'complete_oof_replicates': len(complete),
                  'comparison_scope': 'Within-job matched validation specimens; no cross-family score subtraction',
                  'independence_note': 'Model seeds/support repetitions are repeated predictions on the same specimens.'}
        if complete:
            result['mean_accuracy_gain_percentage_points'] = statistics.mean(row['accuracy_gain_percentage_points'] for row in complete)
            result['mean_macro_f1_gain'] = statistics.mean(row['macro_f1_gain'] for row in complete)
            result['unique_specimens'] = max(row['unique_specimens'] for row in complete)
        output.append(result)
    return output


def select_development_candidates(results):
    """Rank only full-CV TRAIN development procedures, separately by family."""
    choices = defaultdict(list)
    for row in results:
        if not row['eligible_for_development_selection']:
            continue
        if row['family'] == 'classic' and row['prediction_semantics'] != 'estimator_vote':
            continue
        if row['family'] == 'neural' and row['method'] == 'matched_SVM':
            # This reference reuses the same source fit across model seeds,
            # profiles and pretraining budgets. It is retained for matched
            # comparisons, but is not a separately optimized neural profile.
            continue
        choices[(row['family'], row['task'])].append(row)
    selected = []
    for (family, task), values in sorted(choices.items()):
        ordered = sorted(values, key=lambda row: (
            -row['complete_oof_metric_mean']['macro_f1'],
            -row['complete_oof_metric_mean']['balanced_accuracy'],
            -row['complete_oof_metric_mean']['accuracy'],
            tuple(str(row[name]) for name in GROUP_FIELDS)))
        best = ordered[0]
        selected.append({'family': family, 'task': task,
                         'selected_group': {field: best[field] for field in GROUP_FIELDS},
                         'complete_oof_metric_mean': best['complete_oof_metric_mean'],
                         'unique_specimens': best['unique_specimens'],
                         'selection_rule': 'Complete repeated 5-fold TRAIN-only pooled OOF macroF1; then balanced accuracy and accuracy; stable metadata tie-break.',
                         'status': 'development_candidate_only_not_final_confirmation',
                         'compared_complete_groups': len(values)})
    return selected


def flattened_rows(results):
    rows = []
    for result in results:
        row = {field: result[field] for field in GROUP_FIELDS}
        row.update({name: result[name] for name in ('status', 'eligible_for_development_selection',
                                                  'complete_oof_replicates', 'observed_oof_replicates', 'unique_specimens')})
        for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            row[f'complete_oof_mean_{name}'] = result.get('complete_oof_metric_mean', {}).get(name, '')
        row['repeated_prediction_correct'] = result.get('repeated_prediction_correct', '')
        row['repeated_prediction_denominator'] = result.get('repeated_prediction_denominator', '')
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = list(rows[0]) if rows else [*GROUP_FIELDS, 'status', 'unique_specimens']
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render_markdown(payload):
    counts = payload['job_counts']
    lines = ['# 后台训练内开发结果', '',
             '本汇总只读取后台作业输出，不读取主校准集或主预测集。分数来自999份主训练样品内部的开发折；它们不是新的独立外部测试，也不能证明跨批次校准。', '',
             f"已完成并通过结果文件哈希核验：{counts['complete']}个作业；未启动：{counts['not_run']}；已启动未完成：{counts['incomplete']}；错误：{counts['invalid']}。", '',
             '每个重复先合并五个互斥开发折的逐样品预测，再重新计算正确数、总数、宏F1和平衡准确率。缺折结果只显示为未完成，不参与候选推荐。模型种子和支持集重抽样均不增加独立样品数。', '',
             '## 训练内候选', '']
    if payload['development_candidates']:
        lines += ['| 路线 | 任务 | 候选 | 训练内OOF准确率 | 宏F1 | 唯一样品数 |',
                  '| --- | --- | --- | ---: | ---: | ---: |']
        for row in payload['development_candidates']:
            group, metrics = row['selected_group'], row['complete_oof_metric_mean']
            candidate = '/'.join(str(group[name]) for name in ('representation', 'profile', 'pretrain_steps', 'method', 'calibration', 'shots_per_class') if str(group[name]))
            lines.append(f"| {row['family']} | {row['task']} | {candidate} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {row['unique_specimens']} |")
        lines += ['', '这些是当前已完成候选中的训练内结果；若预定矩阵尚有未运行或未完成部分，推荐是临时的。确认评价须另行锁定配置。不同数据量、频点数或划分的经典与神经路线不直接相减作为匹配提升；共享SVM缓存只是参考，不算新增独立重复。', '']
    else:
        lines += ['目前没有满足全部五折及规定重复种子的完整候选，尚不能推荐。', '']
    lines += ['## 跨物种配对对照', '']
    complete_pairs = [row for row in payload['paired_effects'] if row['status'] == 'complete']
    if complete_pairs:
        lines += ['| 任务 | 配置 | 比较 | 平均准确率差（百分点） | 宏F1差 |', '| --- | --- | --- | ---: | ---: |']
        for row in complete_pairs:
            config = f"{row['profile']}/{row['pretrain_steps']}步/{row['calibration']}/K{row['shots_per_class']}"
            lines.append(f"| {row['task']} | {config} | {row['comparison']} | {row['mean_accuracy_gain_percentage_points']:+.3f} | {row['mean_macro_f1_gain']:+.4f} |")
    else:
        lines += ['完整五折跨物种配对结果尚未形成；不能将已跑完的有利折当作正迁移证据。']
    lines += ['', '所有完整及不完整分组、正负增益、逐类混淆矩阵和重复统计保留在 `results_summary.json`；精简分组数值见 `results_summary.csv`。', '',
              '来源为各作业的 `job_manifest.json`、`complete.json`、`summary.json` 和 `validation_predictions.csv.gz`。本脚本不会改动工作器检查点或发布到网络。', '']
    return '\n'.join(lines)


def summarize(root, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / 'results'
    if not root.is_dir():
        raise ValueError('Background root does not exist')
    output.mkdir(parents=True, exist_ok=True)
    scope_path = root / ('protocol.json' if (root / 'protocol.json').is_file() else 'input_lock.json')
    scope = read_json(scope_path)
    allowlist = scope.get('training_allowlist')
    if not isinstance(allowlist, list) or not allowlist or len(allowlist) != len(set(allowlist)):
        raise ValueError('Frozen background protocol/input lock needs a unique TRAIN-only allowlist')
    global_allowlist = frozenset(atom(value) for value in allowlist)
    groups, pairs, jobs = {}, {}, []
    manifest_path = root / 'jobs.json'
    if manifest_path.is_file():
        planned = read_json(manifest_path)['jobs']
        experiment_jobs = [job for job in planned if job.get('kind') in ('classical', 'neural')]
        directories = [(Path(job['outputs'][0]).parent.resolve(), job['id'], job['kind'])
                       for job in experiment_jobs]
        if any(not path.is_relative_to(root) for path, _, _ in directories):
            raise ValueError('Planned output directory lies outside background root')
    else:
        directories = []
        for directory in sorted({path.parent for pattern in ('job_manifest.json', 'validation_predictions.csv.gz')
                                 for path in root.rglob(pattern) if not path.is_relative_to(output)}):
            directories.append((directory, str(directory.relative_to(root)), None))
    for directory, identifier, planned_kind in directories:
        job = {'job_id': identifier, 'path': str(directory.relative_to(root))}
        if planned_kind:
            job['family'] = 'classic' if planned_kind == 'classical' else 'neural'
        if not directory.is_dir():
            job['status'] = 'not_run'
            jobs.append(job)
            continue
        try:
            loaded, receipt = check_receipt(directory)
            job.update(receipt)
            if loaded is not None:
                manifest, summary = loaded
                protocol = str(manifest.get('worker_protocol', '')).lower()
                incoming_groups, incoming_pairs = {}, {}
                if 'classical' in protocol or ('representation' in manifest and 'outer_split_seed' in manifest):
                    consume_classic(directory, manifest, summary, incoming_groups, global_allowlist)
                    job['family'] = 'classic'
                else:
                    consume_neural(directory, manifest, summary, incoming_groups, incoming_pairs, global_allowlist)
                    job['family'] = 'neural'
                merge_job(groups, pairs, incoming_groups, incoming_pairs)
                job['cpu_seconds_reported'] = summary.get('elapsed_cpu_seconds_this_invocation', summary.get('process_cpu_seconds'))
        except (ValueError, KeyError, TypeError, OSError, EOFError, json.JSONDecodeError) as exc:
            job.update(status='invalid', reason=f'{type(exc).__name__}: {exc}')
        jobs.append(job)
    results = []
    for key, replicates in sorted(groups.items()):
        meta = dict(zip(GROUP_FIELDS, key))
        expected_seeds = CLASSIC_SEEDS if meta['family'] == 'classic' else NEURAL_SEEDS
        results.append({**meta, **summarize_replicates(replicates, expected_seeds)})
    payload = {'schema_version': 'species-background-summary-v1',
               'scope': 'TRAIN_only_development_not_master_calibration_or_prediction_not_cross_batch',
               'inputs_accessed': 'Only background-directory frozen protocol/input lock, job list, job manifests, completion receipts, summaries, development truth and prediction CSVs',
               'training_allowlist_records': len(global_allowlist),
               'training_scope_source': scope_path.name, 'training_scope_sha256': digest(scope_path),
               'job_counts': dict(Counter(row['status'] for row in jobs)),
               'known_frozen_matrix_expected_jobs': {'classic': 225, 'neural': 1080},
               'completion_policy': 'Only hashed complete jobs enter metrics; every seed/support replicate needs all five folds and exact TRAIN specimen coverage.',
               'metric_policy': 'Pool fold confusion counts before metrics, then average complete repeated OOF metrics. No fold-mean shortcut and no independent-N inflation.',
               'recommendation_policy': 'Only complete repeated 5-fold development procedures; keep classic estimator-vote semantics fixed; do not compare families as matched transfer gains.',
               'job_artifacts': jobs, 'results': results, 'paired_effects': pair_results(pairs),
               'development_candidates': select_development_candidates(results)}
    for status in ('complete', 'not_run', 'incomplete', 'invalid'):
        payload['job_counts'].setdefault(status, 0)
    atomic_json(output / 'results_summary.json', payload)
    write_csv(output / 'results_summary.csv', flattened_rows(results))
    report = output / 'BACKGROUND_RESULTS.md'
    temporary = report.with_name(report.name + f'.tmp.{os.getpid()}')
    temporary.write_text(render_markdown(payload), encoding='utf-8')
    os.replace(temporary, report)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.output)
    print(json.dumps({'jobs': result['job_counts'], 'groups': len(result['results']),
                      'development_candidates': len(result['development_candidates'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
