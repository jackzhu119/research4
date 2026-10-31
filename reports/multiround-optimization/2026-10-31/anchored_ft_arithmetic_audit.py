"""Read-only arithmetic audit of all saved source-anchored predictions."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/multiround-20261031/anchored-ft'
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1')
checks = 0
maximum = 0.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def close(a, b, message):
    global maximum
    difference = abs(float(a)-float(b))
    maximum = max(maximum, difference)
    check(difference < 2e-12, message)


def metrics(rows):
    truth = np.asarray([int(r['y_true'][2:])-1 for r in rows])
    prediction = np.asarray([int(r['y_predicted'][2:])-1 for r in rows])
    matrix = np.bincount(truth*7+prediction, minlength=49).reshape(7, 7)
    count = matrix.sum(1)
    diagonal = matrix.diagonal()
    recall = np.divide(diagonal, count, out=np.zeros(7), where=count > 0)
    f1 = np.divide(2*diagonal, count+matrix.sum(0), out=np.zeros(7),
                   where=count+matrix.sum(0) > 0)
    return {'accuracy': diagonal.sum()/len(rows), 'balanced_accuracy': recall[count > 0].mean(),
            'macro_f1': f1.mean(), 'confusion_matrix': matrix.tolist(),
            'class_recall': recall.tolist(), 'class_counts': count.tolist(),
            'correct': int(diagonal.sum()), 'total': len(rows)}, truth, prediction


def read_predictions(path):
    grouped = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            check(row['y_true'] == row['record_id'][:4], 'anonymous label/ID association')
            check(int(row['is_correct']) == int(row['y_true'] == row['y_predicted']), 'correctness flag')
            grouped[(int(row['repeat']), row['method'], row['candidate_id'])].append(row)
    for key, rows in grouped.items():
        check(len({r['record_id'] for r in rows}) == len(rows), 'duplicate query IDs in case')
    return grouped


manifest = json.loads((RUN/'run_manifest.json').read_text())
for rel, digest in manifest['source_sha256'].items():
    check(sha(ROOT/rel) == digest, 'executed source SHA '+rel)
check(sha(RUN/'protocol_snapshot.json') == manifest['protocol_sha256'], 'protocol snapshot')
lock = json.loads((RUN/'selection_lock.json').read_text())
development = []
development_prediction_rows = 0
for path in sorted((RUN/'development').glob('episode_*/seed_*/record.json')):
    record = json.loads(path.read_text())
    directory = path.parent
    check(sha(directory/'metrics.json') == record['metrics_sha256'], 'development metrics hash')
    check(sha(directory/'predictions.csv.gz') == record['prediction_sha256'], 'development prediction hash')
    check(record['source_fingerprint_before'] == record['source_fingerprint_after'], 'source frozen')
    sets = {k: set(v) for k, v in record['role_ids'].items()}
    check(not sets['source'] & sets['support_pool'], 'source/support isolated')
    check(not sets['source'] & sets['prediction'], 'source/query isolated')
    check(not sets['support_pool'] & sets['prediction'], 'support/query isolated')
    for draw in record['support_draws']:
        check(set(draw['ids']) <= sets['support_pool'], 'support draw membership')
        check(len(draw['ids']) == len(set(draw['ids'])) == 7, 'support independent IDs')
        check({x[:4] for x in draw['ids']} == {f'ZY{i:02d}' for i in range(1, 8)}, 'seven-class support')
    cases = read_predictions(directory/'predictions.csv.gz')
    development_prediction_rows += sum(len(v) for v in cases.values())
    source, _, _ = metrics(cases[(-1, 'source_zero', 'source_zero')])
    stored = json.loads((directory/'metrics.json').read_text())
    for row in stored:
        actual, _, _ = metrics(cases[(row['repeat'], 'adapted', row['candidate_id'])])
        matched, _, _ = metrics(cases[(-1, 'matched_zero', row['candidate_id'])])
        for name in METRICS:
            close(actual[name], row[name], 'development target metric')
            close(source[name], row['source_'+name], 'development source metric')
            close(matched[name], row['matched_'+name], 'development matched metric')
    development.extend(stored)
check(development == json.loads((RUN/'all_development_metrics.json').read_text()), 'all development rows')
check(len(development) == lock['development_evaluations'] == 2160, 'development count')
grouped = defaultdict(list)
for row in development:
    grouped[(row['candidate_id'], row['stage'])].append(row)
keys = METRICS+tuple('source_'+m for m in METRICS)+tuple('matched_'+m for m in METRICS)
means = {key: {**{m: float(np.mean([r[m] for r in rows])) for m in keys}, 'evaluations': len(rows)}
         for key, rows in grouped.items()}
grid = json.loads((RUN/'candidate_grid.json').read_text())
ranked = sorted(grid, key=lambda c: (-means[(c['id'], 'selection')]['accuracy'], c['learning_rate'],
    c['steps'], c['source_replay_weight'], c['proximal_lambda'], c['id']))


def passes(c, stage):
    v = means[(c['id'], stage)]
    return all(v['accuracy']-v[p+'accuracy'] > .001 and v['macro_f1']-v[p+'macro_f1'] >= -.005
        and v['balanced_accuracy']-v[p+'balanced_accuracy'] >= -.005 for p in ('source_', 'matched_'))


eligible = [c for c in ranked if passes(c, 'selection')]
diagnostic = eligible[0] if eligible else ranked[0]
selected = diagnostic if eligible and passes(diagnostic, 'gate') else None
check(lock['choice']['diagnostic'] == diagnostic, 'independent diagnostic choice')
check(lock['choice']['selected'] == selected, 'independent gated choice')
for stage in ('selection', 'gate'):
    for key, value in means[(diagnostic['id'], stage)].items():
        close(value, lock['choice'][stage][key], 'development lock aggregate')

final_prediction_rows = 0
final_count = 0
if (RUN/'final_metrics.json').exists():
    stored = json.loads((RUN/'final_metrics.json').read_text())
    final_count = len(stored)
    indexed = {(r['seed'], r['repeat'], r['method']): r for r in stored}
    actual_cases = []
    for record in json.loads((RUN/'final_metadata.json').read_text()):
        directory = RUN/'final'/f"seed_{record['seed']}"
        check(sha(directory/'predictions.csv.gz') == record['prediction_sha256'], 'final prediction hash')
        check(record['source_fingerprint_before'] == record['source_fingerprint_after'], 'final source frozen')
        cases = read_predictions(directory/'predictions.csv.gz')
        final_prediction_rows += sum(len(v) for v in cases.values())
        check(len(cases) == 500, 'final five methods times100draws')
        for (repeat, method, cid), rows in cases.items():
            row = indexed[(record['seed'], repeat, method)]
            check(row['candidate_id'] == cid, 'final candidate ID')
            values, truth, predicted = metrics(rows)
            source_rows = cases[(repeat, 'source_zero', 'source_zero')]
            source_values, source_truth, source_pred = metrics(source_rows)
            comparator_name = method+'_matched_zero' if method in ('selected', 'diagnostic') else method
            comparator = cases[(repeat, comparator_name, cid)] if method != 'source_zero' else source_rows
            matched_values, _, _ = metrics(comparator)
            for metric in METRICS:
                close(values[metric], row[metric], 'final arithmetic')
                close(values[metric]-source_values[metric], row[metric+'_gain_over_source'], 'final source gain')
                close(values[metric]-matched_values[metric], row[metric+'_gain_over_matched'], 'final matched gain')
            for key in ('confusion_matrix', 'class_recall', 'class_counts', 'correct', 'total'):
                check(values[key] == row[key], 'final detailed metric '+key)
            check(np.array_equal(truth, source_truth), 'same ordered prediction truth')
            check(row['corrected_errors'] == int(np.count_nonzero((source_pred != truth)&(predicted == truth))), 'corrected')
            check(row['new_errors'] == int(np.count_nonzero((source_pred == truth)&(predicted != truth))), 'new errors')
            actual_cases.append(row)
        for model in record['saved_adapted_first_draw_models']:
            check(sha(ROOT/model['path']) == model['sha256'], 'adapted weights hash')
    check(len(actual_cases) == len(stored) == 1500, 'final count')
    summary = json.loads((RUN/'summary.json').read_text())
    for row in summary:
        values = [r for r in stored if r['method'] == row['method']]
        for metric in METRICS:
            for name, operation in (('mean', np.mean), ('sd', np.std), ('min', np.min), ('max', np.max)):
                close(operation([r[metric] for r in values]), row[metric+'_'+name], 'summary '+name)
            for comparison in ('source', 'matched'):
                key = metric+'_gain_over_'+comparison
                close(np.mean([r[key] for r in values]), row[key+'_mean'], 'summary gain')
        check(sum(r['accuracy_gain_over_source'] > 0 for r in values) == row['positive_draws'], 'better count')
        check(sum(r['accuracy_gain_over_source'] == 0 for r in values) == row['unchanged_draws'], 'equal count')
        check(sum(r['accuracy_gain_over_source'] < 0 for r in values) == row['negative_draws'], 'worse count')

result = {'status': 'pass', 'scope': 'read-only arithmetic and saved lineage audit; no fitting',
    'checks': checks, 'maximum_numeric_difference': maximum, 'development_cases': len(development),
    'development_prediction_rows': development_prediction_rows, 'final_cases': final_count,
    'final_prediction_rows': final_prediction_rows, 'selected_candidate': selected,
    'diagnostic_candidate': diagnostic, 'selection_lock_sha256': sha(RUN/'selection_lock.json'),
    'audit_script_sha256': sha(Path(__file__))}
(RUN/'arithmetic_audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
