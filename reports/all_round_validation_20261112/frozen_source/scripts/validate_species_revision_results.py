"""Independently recount frozen species-revision predictions and pool audits.

No fitting or prediction helpers are imported. Metrics are calculated directly
from per-record confusion counts; truth is rejoined from the master split.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/research/species_revision_20261108'
METRICS = ('accuracy', 'macro_f1', 'balanced_accuracy')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recount(truth, predicted, classes):
    """Compute all three metrics from explicit integer confusion counts."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if not set(truth).issubset(set(classes)) or not set(predicted).issubset(set(classes)):
        raise ValueError('Unknown true or predicted class')
    matrix = np.asarray([[int(np.sum((truth == a) & (predicted == b)))
                          for b in classes] for a in classes], dtype=np.int64)
    diagonal = matrix.diagonal().astype(float)
    actual, called = matrix.sum(axis=1), matrix.sum(axis=0)
    f1 = np.divide(2 * diagonal, actual + called, out=np.zeros(len(classes)), where=(actual + called) > 0)
    recalls = np.divide(diagonal, actual, out=np.zeros(len(classes)), where=actual > 0)
    metrics = {'accuracy': float(diagonal.sum() / len(truth)),
               'macro_f1': float(np.mean(f1)),
               'balanced_accuracy': float(np.mean(recalls[actual > 0]))}
    return metrics, matrix.tolist(), {str(c): float(v) for c, v in zip(classes, recalls)}


def main():
    checks, differences, fail = [], [], []

    def check(name, passed, details=None):
        item = {'check': name, 'passed': bool(passed)}
        if details is not None:
            item['details'] = details
        checks.append(item)
        if not passed:
            fail.append(item)

    split = pd.read_csv(BASE / 'split_manifest.csv', dtype=str).fillna('')
    check('all_1672_record_and_physical_ids_unique_nonmissing',
          len(split) == 1672 and split.record_id.nunique() == 1672
          and split.physical_sample_id.nunique() == 1672 and split.physical_sample_id.ne('').all())
    master = split.set_index('record_id')
    pools = {role: set(split[split.role == role].physical_sample_id) for role in ('train', 'calibration', 'prediction')}
    for left, right in [('train', 'calibration'), ('train', 'prediction'), ('calibration', 'prediction')]:
        check('no_physical_overlap_' + left + '_' + right, not pools[left] & pools[right])
    old = json.loads((BASE / 'historical_split_plan.json').read_text())['records']
    check('all_855_historical_roles_and_physical_ids_preserved', len(old) == 855 and all(
        master.loc[r['record_id'], 'role'] == ('train' if r['role'] == 'training' else r['role'])
        and master.loc[r['record_id'], 'physical_sample_id'] == r['physical_sample_id'] for r in old))

    def ids_for(kind, role):
        rows = split[split.role == role]
        if kind is not None:
            rows = rows[rows.species.isin(kind if isinstance(kind, list) else [kind])]
        return set(rows.record_id)

    def verify_settings(name, frame, saved, classes, keys, truth_col, pred_col, expected, true_field):
        check(name + '_truth_matches_master_by_record', all(
            frame[truth_col].to_numpy() == master.loc[frame.record_id, true_field].to_numpy()))
        check(name + '_all_prediction_ids_match_locked_query', set(frame.record_id) == expected)
        groups = {key: value for key, value in frame.groupby(keys)}
        check(name + '_settings_count_matches_saved', len(groups) == len(saved))
        for row in saved:
            key = tuple(str(row[k]) for k in keys)
            group = groups[key]
            check(name + '_query_' + '|'.join(key), len(group) == len(expected)
                  and group.record_id.nunique() == len(expected) and set(group.record_id) == expected)
            measured, matrix, _ = recount(group[truth_col], group[pred_col], classes)
            reference = row.get('metrics', row)
            for metric in METRICS:
                difference = abs(measured[metric] - float(reference[metric]))
                differences.append(difference)
                check(name + '_' + metric + '_' + '|'.join(key), difference <= 1e-12)
            if 'confusion_matrix' in reference:
                check(name + '_confusion_' + '|'.join(key), matrix == reference['confusion_matrix'])
        return len(groups)

    classification = BASE / 'classification'
    frozen = json.loads((classification / 'frozen_config.json').read_text())
    for label, path, key in [
        ('script', ROOT / 'scripts/run_species_revision_classification.py', 'script_sha256'),
        ('features', BASE / 'features.npz', 'feature_sha256'),
        ('split', BASE / 'split_manifest.csv', 'split_sha256')]:
        check('classification_frozen_' + label + '_hash', sha(path) == frozen[key])
    task_kinds = {'binary_kinds': ['枣仁', '酸枣仁'], 'four_kinds': None,
                  'acid_origins': '酸枣仁', 'jujube_origins': '枣仁',
                  'huangqi_origins': '黄芪', 'tianma_origins': '天麻'}
    classification_settings = classification_rows = 0
    highlights = {}
    for task, kind in task_kinds.items():
        directory = classification / task
        saved = json.loads((directory / 'summary.json').read_text())
        complete = json.loads((directory / 'complete.json').read_text())
        for file, key in [('summary.json', 'summary_sha256'), ('predictions.csv.gz', 'predictions_sha256')]:
            check(task + '_' + key, sha(directory / file) == complete[key])
        frame = pd.read_csv(directory / 'predictions.csv.gz', dtype=str)
        classification_rows += len(frame)
        true_field = 'species' if task in ('binary_kinds', 'four_kinds') else 'origin_group_id'
        expected = ids_for(kind, 'prediction')
        classification_settings += verify_settings(task, frame, saved['all_results'], saved['classes'],
            ['model', 'method', 'shots_per_class', 'repeat'], 'y_true', 'y_predicted', expected, true_field)
        supports = json.loads((directory / 'support_draws.json').read_text())
        by_repeat = defaultdict(list)
        for draw in supports:
            support = set(draw['record_ids'])
            check(task + '_support_pool_' + str(draw['shots_per_class']) + '_' + str(draw['repeat']),
                  len(support) == len(draw['record_ids']) and support <= ids_for(kind, 'calibration') and not support & expected)
            if draw['shots_per_class'] != 'all':
                k = int(draw['shots_per_class'])
                check(task + '_support_class_counts_' + str(k) + '_' + str(draw['repeat']),
                      all(sum(master.loc[rid, true_field] == c for rid in support) == k for c in saved['classes']))
                by_repeat[draw['repeat']].append((k, support))
        for repeat, draws in by_repeat.items():
            draws.sort(key=lambda x: x[0])
            check(task + '_nested_support_' + str(repeat), all(a[1] <= b[1] for a, b in zip(draws, draws[1:])))
        for path in directory.glob('*_fit_receipt.json'):
            fit = json.loads(path.read_text())
            check(task + '_' + path.stem + '_fit_role_isolation',
                  set(fit['source_ids']) == ids_for(kind, 'train')
                  and set(fit['full_refit_ids']) == ids_for(kind, 'train') | ids_for(kind, 'calibration')
                  and not set(fit['full_refit_ids']) & expected)
        for path in directory.glob('*_inner_selection.json'):
            select = json.loads(path.read_text())
            check(task + '_' + path.stem + '_training_only', set(select['train_ids']) == ids_for(kind, 'train')
                  and all(set(f['train']) | set(f['validation']) == ids_for(kind, 'train')
                          and not set(f['train']) & set(f['validation']) for f in select['folds']))
        model = 'R2_reselected' if task in ('binary_kinds', 'acid_origins') else 'fixed_MC20'
        chosen = frame[(frame.model == model) & (frame.shots_per_class.isin(['0', 'all']))]
        highlights[task] = {}
        for method, rows in chosen.groupby('method'):
            metric, _, recall = recount(rows.y_true, rows.y_predicted, saved['classes'])
            highlights[task][method] = {**metric, 'correct': int((rows.y_true == rows.y_predicted).sum()),
                                       'n': len(rows), 'recall': recall}
        if task == 'four_kinds':
            comparison = chosen.pivot(index='record_id', columns='method', values='y_predicted')
            check('four_kind_weak_anchor_has_no_change_from_matched_argmax',
                  comparison['weak_anchor_lambda1000_blend005'].equals(comparison['source_score_argmax']))

    transfer = BASE / 'transfer'
    cfg = json.loads((transfer / 'protocol.json').read_text())
    lock = json.loads((transfer / 'prediction_lock.json').read_text())
    for label, path, expected in [
        ('script', ROOT / 'scripts/run_species_revision_transfer.py', cfg['code_sha256']),
        ('features', BASE / 'features.npz', cfg['features_sha256']),
        ('split', BASE / 'split_manifest.csv', cfg['split_sha256']),
        ('protocol', transfer / 'protocol.json', lock['protocol_sha256']),
        ('predictions', transfer / 'predictions_without_truth.csv.gz', lock['predictions_sha256'])]:
        check('transfer_frozen_' + label + '_hash', sha(path) == expected)
    for name, expected in cfg['reused_code_sha256'].items():
        check('transfer_historical_module_hash_' + name, sha(ROOT / 'src/herb_spectra' / name) == expected)
    frame = pd.read_csv(transfer / 'scored_predictions.csv.gz', dtype=str)
    saved = json.loads((transfer / 'metrics_by_setting.json').read_text())
    expected_query = ids_for('酸枣仁', 'prediction')
    transfer_settings = verify_settings('transfer', frame, saved, cfg['target_classes'],
        ['arm', 'seed', 'method', 'repeat', 'shots'], 'truth', 'predicted', expected_query, 'origin_group_id')
    check('transfer_all_prespecified_settings_present', transfer_settings == len(cfg['arms']) * len(cfg['seeds'])
          * (1 + len(cfg['calibration']['methods']) * len(cfg['calibration']['shots_per_class']) * cfg['calibration']['repeats']))
    draws = json.loads((transfer / 'support_draws.json').read_text())
    by_repeat = defaultdict(list)
    for draw in draws:
        support = set(draw['support_ids'])
        check('transfer_support_' + str(draw['repeat']) + '_' + str(draw['shots']),
              len(support) == len(draw['support_ids']) == 5 * draw['shots']
              and support <= ids_for('酸枣仁', 'calibration') and not support & expected_query
              and all(sum(master.loc[rid, 'origin_group_id'] == c for rid in support) == draw['shots'] for c in cfg['target_classes']))
        by_repeat[draw['repeat']].append((draw['shots'], support))
    for repeat, draws in by_repeat.items():
        draws.sort(key=lambda x: x[0])
        check('transfer_nested_support_' + str(repeat), all(a[1] <= b[1] for a, b in zip(draws, draws[1:])))
    training = json.loads((transfer / 'training_summary.json').read_text())
    for row in training:
        name = str(row['seed']) + '_' + row['arm']
        check('transfer_target_source_' + name, set(row['source_ids']) == ids_for('酸枣仁', 'train'))
        if row['arm'] == 'B':
            check('transfer_auxiliary_train_only_' + name,
                  all(set(row['auxiliary_ids'][kind]) == ids_for(kind, 'train') for kind in ('黄芪', '天麻')))
        audit = json.loads((transfer / 'models' / ('seed' + name) / 'pretrain_batch_ids.json').read_text())
        allowed = {'target': ids_for('酸枣仁', 'train'), '黄芪': ids_for('黄芪', 'train'), '天麻': ids_for('天麻', 'train')}
        check('transfer_actual_pretrain_batches_' + name,
              len(audit) == (0 if row['arm'] == 'A' else cfg['training']['pretrain_steps'])
              and all(set(batch['actual_ids']) <= allowed[batch['head']]
                      and len(set(batch['actual_ids'])) == 32 for batch in audit))
    for seed in cfg['seeds']:
        paired = [row for row in training if row['seed'] == seed]
        check('transfer_paired_init_and_target_orders_' + str(seed), len(paired) == 3 and all(
            len({row[key] for row in paired}) == 1 for key in ('initial_encoder_sha256', 'initial_target_head_sha256', 'target_order_sha256')))
    transfer_means = {}
    for arm, rows in frame[frame.method == 'source_only'].groupby('arm'):
        means = [recount(v.truth, v.predicted, cfg['target_classes'])[0] for _, v in rows.groupby('seed')]
        transfer_means[arm] = {metric: float(np.mean([m[metric] for m in means])) for metric in METRICS}
    result = {
        'date': '2026-11-08', 'validator': 'scripts/validate_species_revision_results.py',
        'validator_sha256': sha(__file__), 'status': 'passed' if not fail else 'failed',
        'checks_run': len(checks), 'checks_passed': len(checks) - len(fail), 'failures': fail,
        'classification': {'tasks': len(task_kinds), 'settings': classification_settings, 'prediction_rows': classification_rows},
        'transfer': {'settings': transfer_settings, 'prediction_rows': len(frame)},
        'metric_max_absolute_difference': max(differences),
        'independent_method': 'Integer confusion counts from per-record predictions; no project model/scoring helper or sklearn metric function imported.',
        'role_counts': split.role.value_counts().to_dict(),
        'classification_highlights': highlights, 'transfer_source_only_seed_means': transfer_means,
        'transfer_B_minus_A_accuracy_pp': 100 * (transfer_means['B']['accuracy'] - transfer_means['A']['accuracy']),
        'interpretation_checks': [
            'Four-kind weak anchor all-pool fixed_MC20 has the same 314/339 predictions as source score argmax; improvement over 313/339 voting is a score-rule effect, not demonstrated anchor gain.',
            'Jujube-origin fixed_MC20 accuracy 72/74 coexists with macroF1 about .597 and zero recall for ZR02/ZR03; do not claim reliable five-origin recognition.',
            'Acid-origin R2 87/100 to 93/100 uses complete train+calibration pipeline refitting; do not label this score-only calibration gain.',
            'R0 versus R2 on prediction must remain a comparison; do not choose the final method from the higher prediction score.',
            'Transfer B uses 300 labelled target-training specimens before K-shot calibration; K is additional score support, not the entire target-labelled budget.',
            'Historical ZY role inheritance does not remove prior dataset exposure or establish real cross-batch performance.'
        ],
        'static_code_review_scope': 'Training-only selection and preprocessing; fixed source class order and binary score conversion; held-out query labels used for scoring; source auxiliary tasks use train only; calibration supports nested and query shared. Model code was reviewed separately without modifying frozen entrypoints.',
        'limits': ['This validation verifies saved predictions, allocation audits, hashes and code data flow; it is not independent biological authentication or an external-batch validation.']}
    destination = BASE / 'independent_results_validation.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'checks_run', 'checks_passed', 'failures', 'classification', 'transfer', 'metric_max_absolute_difference')}, ensure_ascii=False, indent=2))
    return bool(fail)


if __name__ == '__main__':
    raise SystemExit(main())
