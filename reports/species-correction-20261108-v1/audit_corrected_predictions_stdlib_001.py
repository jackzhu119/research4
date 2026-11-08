"""Independent stdlib audit of fixed, already-published predictions; never fits."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math

ROOT = Path('/workspace/research4')
OUT = ROOT/'reports/species-correction-20261108-v1'
SNAP = OUT/'frozen-remote/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7'
BASE = SNAP/'data/research/species_revision_20261108'
N_CHECKS = 0
MAX_DIFF = 0.0


def check(ok, description):
    global N_CHECKS
    N_CHECKS += 1
    if not ok:
        raise AssertionError(description)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(a, b, description):
    global MAX_DIFF
    diff = abs(float(a)-float(b))
    MAX_DIFF = max(MAX_DIFF, diff)
    check(math.isfinite(diff) and diff < 5e-13, description+str((a, b, diff)))


def read_csv(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def shot(value):
    return int(value) if str(value).isdigit() else str(value)


def classification_key(row):
    return (row['model'], row['method'], shot(row['shots_per_class']), int(row['repeat']))


def calculate(rows, classes):
    order = {value: i for i, value in enumerate(classes)}
    cm = [[0]*len(classes) for _ in classes]
    for rid, truth, pred in rows:
        check(truth in order and pred in order, 'Class scope')
        cm[order[truth]][order[pred]] += 1
    support = [sum(row) for row in cm]
    predicted = [sum(row[j] for row in cm) for j in range(len(classes))]
    tp = [cm[j][j] for j in range(len(classes))]
    recall = [a/b if b else 0. for a, b in zip(tp, support)]
    precision = [a/b if b else 0. for a, b in zip(tp, predicted)]
    f1 = [2*a/(b+c) if b+c else 0. for a, b, c in zip(tp, support, predicted)]
    return dict(n=sum(support), correct=sum(tp), accuracy=sum(tp)/sum(support),
                macro_f1=math.fsum(f1)/len(classes),
                balanced_accuracy=math.fsum(recall)/len(classes),
                confusion_matrix=cm, class_order=classes,
                class_counts=dict(zip(classes, support)), class_recall=dict(zip(classes, recall)),
                class_precision=dict(zip(classes, precision)), class_f1=dict(zip(classes, f1)))


def compare(got, expected, description):
    for key in ['accuracy', 'macro_f1', 'balanced_accuracy']:
        numeric(got[key], expected[key], description+key)
    for key in ['confusion_matrix', 'class_order', 'class_counts']:
        if key in expected:
            check(got[key] == expected[key], description+key)
    if 'class_recall' in expected:
        check(set(got['class_recall']) == set(expected['class_recall']), 'Recall keys')
        for key, value in expected['class_recall'].items():
            numeric(got['class_recall'][key], value, description+' recall '+key)


def write_new(path, value):
    check(not path.exists(), 'Fresh output required: '+str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    artifacts = json.loads((SNAP/'REMOTE_RESULT_DOWNLOAD_MANIFEST.json').read_text())['artifacts']
    hashes = []
    for entry in artifacts:
        if entry.get('status') != 'DOWNLOADED_EXACT_FIXED_COMMIT':
            continue
        path = SNAP/entry['path']
        raw = path.read_bytes()
        check(len(raw) == entry['bytes'], 'Downloaded size')
        check(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Downloaded SHA256')
        check(hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest() == entry['git_blob_sha'], 'Downloaded Git blob')
        hashes.append({key: entry[key] for key in ['path', 'bytes', 'sha256', 'git_blob_sha']})
    master = read_csv(BASE/'split_manifest.csv')
    check(len(master) == len({r['record_id'] for r in master}) == 1672, 'Master unique IDs')
    check(len({r['physical_sample_id'] for r in master}) == 1672, 'Master physical IDs')
    check(Counter(r['role'] for r in master) == Counter(train=999, calibration=334, prediction=339), 'Master roles')
    kind_map = dict(acid_origins='酸枣仁', jujube_origins='枣仁', huangqi_origins='黄芪', tianma_origins='天麻')
    all_settings, task_results, main_results = [], {}, []
    computed_by_task, groups_by_task = {}, {}
    classification_rows = 0
    for task in ['binary_kinds', 'acid_origins', 'four_kinds', 'huangqi_origins', 'tianma_origins', 'jujube_origins']:
        folder = BASE/'classification'/task
        summary = json.loads((folder/'summary.json').read_text())
        complete = json.loads((folder/'complete.json').read_text())
        check(sha(folder/'summary.json') == complete['summary_sha256'], 'Summary complete receipt')
        check(sha(folder/'predictions.csv.gz') == complete['predictions_sha256'], 'Prediction complete receipt')
        expected = {r['record_id']: r['species' if task in ['binary_kinds', 'four_kinds'] else 'origin_group_id'] for r in master
                    if r['role'] == 'prediction' and (task == 'four_kinds' or
                    task == 'binary_kinds' and r['species'] in ['枣仁', '酸枣仁'] or
                    task in kind_map and r['species'] == kind_map[task])}
        classes = summary['classes']
        check(set(classes) == set(expected.values()), 'Canonical class set')
        groups = defaultdict(list)
        for row in read_csv(folder/'predictions.csv.gz'):
            classification_rows += 1
            rid = row['record_id']
            check(row['task'] == task, 'Prediction task')
            check(rid in expected and row['y_true'] == expected[rid], 'Canonical truth/role/kind/origin')
            check(row['is_correct'] == str(int(row['y_true'] == row['y_predicted'])), 'Correct flag')
            groups[classification_key(row)].append((rid, row['y_true'], row['y_predicted']))
        references = {classification_key(row): row for row in summary['all_results']}
        check(set(groups) == set(references), 'Complete setting coverage')
        computed = {}
        for key, rows in groups.items():
            check(len(rows) == len(expected) == len({x[0] for x in rows}) and {x[0] for x in rows} == set(expected), 'Same query IDs every setting')
            got = calculate(rows, classes)
            compare(got, references[key]['metrics'], task+str(key))
            computed[key] = got
            model, method, k, repeat = key
            all_settings.append(dict(study='classification', task=task, model_or_arm=model, seed='', method=method, shots=k, repeat=repeat,
                                     **{j: got[j] for j in ['n', 'correct', 'accuracy', 'macro_f1', 'balanced_accuracy']}))
            if method in ['source_only', 'train_plus_calibration_full_retrain']:
                main_results.append(dict(task=task, model=model, method=method, **got))
        for agg in summary['aggregates']:
            subset = [m for key, m in computed.items() if (key[0], key[1], key[2]) == (agg['model'], agg['method'], shot(agg['shots_per_class']))]
            check(len(subset) == agg['repeats'], 'Aggregate repeat count')
            for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                numeric(math.fsum(x[metric] for x in subset)/len(subset), agg[metric+'_mean'], 'Class aggregate')
            average = math.fsum(x['accuracy'] for x in subset)/len(subset)
            numeric(math.sqrt(math.fsum((x['accuracy']-average)**2 for x in subset)/len(subset)), agg['accuracy_sd_support_draws'], 'Class population SD')
        task_results[task] = dict(settings=len(groups), query_n=len(expected), classes=classes)
        computed_by_task[task], groups_by_task[task] = computed, groups

    tdir = BASE/'transfer'
    result = json.loads((tdir/'result.json').read_text())
    truth = {r['record_id']: r['origin_group_id'] for r in master if r['role'] == 'prediction' and r['species'] == '酸枣仁'}
    classes = sorted(set(truth.values()))
    tgroups = defaultdict(list)
    for row in read_csv(tdir/'scored_predictions.csv.gz'):
        rid = row['record_id']
        check(rid in truth and row['truth'] == truth[rid], 'Transfer canonical truth')
        check(row['correct'] in ['True', 'False'] and (row['correct'] == 'True') == (row['truth'] == row['predicted']), 'Transfer correct flag')
        scores = [float(row['score_'+label]) for label in classes]
        check(all(math.isfinite(x) for x in scores), 'Finite scores')
        check(classes[max(range(len(scores)), key=lambda j: scores[j])] == row['predicted'], 'Transfer argmax')
        tgroups[(row['arm'], int(row['seed']), row['method'], int(row['repeat']), int(row['shots']))].append((rid, row['truth'], row['predicted']))
    tc = {}
    for key, rows in tgroups.items():
        check(len(rows) == len(truth) == len({r[0] for r in rows}) and {r[0] for r in rows} == set(truth), 'Transfer same query')
        got = calculate(rows, classes)
        tc[key] = got
        arm, seed, method, repeat, k = key
        all_settings.append(dict(study='transfer', task='acid_origins', model_or_arm=arm, seed=seed, method=method, shots=k, repeat=repeat,
                                 **{j: got[j] for j in ['n', 'correct', 'accuracy', 'macro_f1', 'balanced_accuracy']}))
    for agg in result['summary']:
        vals = [m for key, m in tc.items() if (key[0], key[2], key[4]) == (agg['arm'], agg['method'], agg['shots'])]
        seeds = {key[1] for key in tc if (key[0], key[2], key[4]) == (agg['arm'], agg['method'], agg['shots'])}
        check(len(seeds) == agg['seed_count'] == 3 and len(vals) == 3*agg['support_repeats'], 'Transfer seeds/repeats')
        check(agg['query_samples'] == len(truth), 'Transfer sample count')
        for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
            numeric(math.fsum(v[metric] for v in vals)/len(vals), agg[metric+'_mean'], 'Transfer aggregate')
    for references in [result['paired_transfer'], read_csv(tdir/'paired_source_transfer.csv')]:
        for row in references:
            arm, seed = row['arm'], int(row['seed'])
            got, baseline = tc[(arm, seed, 'source_only', -1, 0)], tc[('A', seed, 'source_only', -1, 0)]
            check(got['correct'] == int(row['correct']) and got['n'] == int(row['n']), 'Transfer pair counts')
            for key in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                numeric(got[key], float(row[key]), 'Transfer pair')
            numeric(100*(got['accuracy']-baseline['accuracy']), float(row['accuracy_gain_vs_A_pp']), 'Transfer accuracy delta')
            numeric(got['macro_f1']-baseline['macro_f1'], float(row['macro_f1_gain_vs_A']), 'Transfer F1 delta')
    individual_path = tdir/'metrics_by_setting.json'
    raw = individual_path.read_bytes()
    check(hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest() == '294438cc208b3d927c7d83c2ea7e328449ed3abf', 'Individual transfer Git blob independently fetched')
    individual = json.loads(raw)
    check(len(individual) == len(tc), 'Transfer individual reference count')
    for reference in individual:
        key = (reference['arm'], reference['seed'], reference['method'], reference['repeat'], reference['shots'])
        got = tc[key]
        compare(got, reference, 'Transfer individual')
        check(got['correct'] == reference['correct'] and got['n'] == reference['n'], 'Transfer individual counts')
    hashes.append(dict(path=str(individual_path.relative_to(SNAP)), bytes=len(raw), sha256=sha(individual_path), git_blob_sha='294438cc208b3d927c7d83c2ea7e328449ed3abf'))
    paired = []
    for arm in ['A', 'Aextra', 'B']:
        values = [tc[(arm, seed, 'source_only', -1, 0)] for seed in [42, 43, 44]]
        paired.append(dict(arm=arm, seeds=[42, 43, 44], correct_by_seed=[x['correct'] for x in values],
                           **{key+'_mean': math.fsum(x[key] for x in values)/3 for key in ['accuracy', 'macro_f1', 'balanced_accuracy']}))
    claims = [('binary_kinds', 'R2_reselected', 174, 174), ('acid_origins', 'R2_reselected', 87, 100),
              ('four_kinds', 'full_svm', 331, 339), ('huangqi_origins', 'full_svm', 69, 82),
              ('tianma_origins', 'full_svm', 81, 83), ('jujube_origins', 'full_svm', 72, 74)]
    verified = []
    for task, model, correct, n in claims:
        got = computed_by_task[task][(model, 'source_only', 0, 0)]
        check(got['correct'] == correct and got['n'] == n, 'Published claim')
        verified.append(dict(task=task, model=model, **got))
    acid_refit = computed_by_task['acid_origins'][('R2_reselected', 'train_plus_calibration_full_retrain', 'all', 0)]
    check(acid_refit['correct'] == 93, 'Acid R2 full refit93')
    rare = computed_by_task['jujube_origins'][('full_svm', 'source_only', 0, 0)]
    check(rare['class_counts']['ZR02'] == rare['class_counts']['ZR03'] == 1 and rare['class_recall']['ZR02'] == rare['class_recall']['ZR03'] == 0, 'Rare jujube zero recalls')

    # New descriptive system audit; heads fixed from the published main table.
    route = {rid: pred for rid, truth, pred in groups_by_task['four_kinds'][('full_svm', 'source_only', 0, 0)]}
    heads = {'酸枣仁': ('acid_origins', 'R2_reselected'), '枣仁': ('jujube_origins', 'full_svm'),
             '黄芪': ('huangqi_origins', 'full_svm'), '天麻': ('tianma_origins', 'full_svm')}
    head_predictions = {kind: {rid: pred for rid, truth, pred in groups_by_task[task][(model, 'source_only', 0, 0)]} for kind, (task, model) in heads.items()}
    hierarchy = []
    for sample in master:
        if sample['role'] != 'prediction':
            continue
        rid, kind, origin = sample['record_id'], sample['species'], sample['origin_group_id']
        check(rid in route and rid in head_predictions[kind], 'Hierarchy exact query coverage')
        routed_kind = route[rid]
        route_correct = routed_kind == kind
        oracle_pred = head_predictions[kind][rid]
        joint_correct = route_correct and oracle_pred == origin
        hierarchy.append(dict(record_id=rid, true_kind=kind, true_origin_group=origin, predicted_kind=routed_kind,
                              route_correct=route_correct, predicted_origin_group=oracle_pred if route_correct else 'NOT_EVALUATED_ROUTING_ERROR',
                              joint_correct=joint_correct, oracle_true_kind_origin_prediction=oracle_pred,
                              oracle_origin_correct=oracle_pred == origin))
    check(len(hierarchy) == len(route) == 339, 'Hierarchy339')
    hierarchy_summary = dict(scope='POSTHOC_FIXED_HEAD_RETROSPECTIVE_SYSTEM_CORRECTNESS_AUDIT_NO_NEW_INFERENCE',
                             head_selection_basis='Frozen to source-only heads in already-published main table; not query-selected winners; R2 acid despite higher R0 query score.',
                             fixed_heads=heads, routing_head=['four_kinds', 'full_svm', 'source_only', 0, 0],
                             prediction_samples=339, routed_kind_correct=sum(x['route_correct'] for x in hierarchy),
                             joint_kind_and_origin_correct=sum(x['joint_correct'] for x in hierarchy),
                             joint_accuracy=sum(x['joint_correct'] for x in hierarchy)/339,
                             oracle_kind_origin_correct=sum(x['oracle_origin_correct'] for x in hierarchy),
                             oracle_kind_origin_accuracy=sum(x['oracle_origin_correct'] for x in hierarchy)/339,
                             no_accuracy_products_used=True,
                             wrong_route_origin_label_not_computed='A wrong kind is already a joint error regardless of the unavailable wrong-head output; do not fabricate its concrete origin or a full joint-F1.',
                             by_kind={k: dict(n=sum(x['true_kind'] == k for x in hierarchy),
                                              joint_correct=sum(x['joint_correct'] for x in hierarchy if x['true_kind'] == k),
                                              oracle_correct=sum(x['oracle_origin_correct'] for x in hierarchy if x['true_kind'] == k)) for k in heads})
    report = dict(schema='research4-corrected-results-independent-recount-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='PASS_ALL_DOWNLOADED_RAW_PREDICTIONS_AND_REFERENCE_METRICS',
                  commit='21d6edab24d198dc60b1ad4ef50e83602c9ecfa7', checks=N_CHECKS,
                  largest_absolute_metric_difference=MAX_DIFF, classification_prediction_rows=classification_rows,
                  classification_settings_recomputed_and_reference_verified=sum(v['settings'] for v in task_results.values()),
                  classification_tasks=task_results, transfer_prediction_rows=sum(len(v) for v in tgroups.values()),
                  transfer_settings_recomputed_and_individual_reference_verified=len(tc), main_results=verified,
                  acid_R2_full_calibration_refit=acid_refit, transfer_source_results=paired,
                  fixed_head_hierarchy=hierarchy_summary, verified_artifacts=hashes,
                  limitations=['Retrospective internal validation on previously exposed samples, not independent new batches.',
                               'R0 scored higher on the fixed query; it cannot become a preselected final method after observing it.',
                               '93/100 is full pipeline refit using100 additional labels, not1shot or frozen-only calibration.',
                               'Jujube72/74 masks two1sample origin groups with zero recall.',
                               'B79% versus A81% and Aextra80.33% is retained negative transfer.',
                               'Fixed-head hierarchy is posthoc arithmetic of existing predictions, not a freshly trained or confirmed system.'],
                  no_new_fit_inference_Q_Git_or_active_state_edit=True)
    write_new(OUT/'corrected-results-independent-recount-001.json', report)
    for path, rows in [(OUT/'corrected-results-independent-recount-001.csv', all_settings),
                       (OUT/'corrected-fixed-head-hierarchy-001.csv', hierarchy)]:
        check(not path.exists(), 'Fresh CSV')
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    md = ('# 种类修正结果独立逐预测复算\n\n固定提交`21d6edab24d198dc60b1ad4ef50e83602c9ecfa7`。只读原始压缩预测与主名单，'
          '未加载模型、训练、推断、量子计算或Git。\n\n分类%d设置、%d行预测，迁移%d设置、%d行预测，均按主预测ID、种类、来源和真值复算。'
          '所有设置的Accuracy、Macro-F1、Balanced Accuracy和混淆矩阵与下载的参考逐个一致。最大算术差%.3g；检查%d项。\n\n'
          '主结果：种类174/174；酸枣仁R2 87/100，全部校准池完整重训93/100；四药材331/339；黄芪69/82；天麻81/83；'
          '枣仁来源72/74，但辽宁和内蒙古各1份预测均错。迁移A81%%、Aextra80.33%%、B79%%。\n\n'
          '按已发表主表固定source-only层级头作事后系统审计：339份中种类与来源同时正确%d份，%.4f%%；'
          '真种类oracle路由为%d/339，%.4f%%。这是逐ID逻辑计错，没有把两层准确率相乘。'
          '路由错误不需要再推断错误物种的head即可确定joint错误；该head的具体来源不伪造，因此不报告完整joint-F1。\n\n'
          '93%%不等于少样本冻结校准；R0固定query更高不构成事后改选预选赢家的许可；'
          '全部仍为已曝光样品内部验证，不是新批次确认。完整证据、设置表和层级逐ID记录见同名JSON和CSV。\n') % (
              sum(v['settings'] for v in task_results.values()), classification_rows, len(tc), sum(len(v) for v in tgroups.values()),
              MAX_DIFF, N_CHECKS, hierarchy_summary['joint_kind_and_origin_correct'], 100*hierarchy_summary['joint_accuracy'],
              hierarchy_summary['oracle_kind_origin_correct'], 100*hierarchy_summary['oracle_kind_origin_accuracy'])
    path = OUT/'corrected-results-independent-recount-001.md'
    check(not path.exists(), 'Fresh MD')
    path.write_text(md)
    print(json.dumps(dict(checks=N_CHECKS, maxdiff=MAX_DIFF, class_settings=sum(v['settings'] for v in task_results.values()),
                          transfer_settings=len(tc), report_sha256=sha(OUT/'corrected-results-independent-recount-001.json'),
                          hierarchy=hierarchy_summary), ensure_ascii=False))


if __name__ == '__main__':
    main()
