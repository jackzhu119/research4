"""Independent, source-only audit; no fitting, native model replay, or query I/O."""
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

REPO = Path('/workspace/research4-supplemental')
RUN = REPO / 'reports/supplemental-spectra-benchmark-20261104-v1'
OUT = Path(__file__).resolve().parent
READS = {}
CHECKS = collections.Counter()
STATS = collections.Counter()
FITS = []
METRICS = []
SELECTED = {}
POLICIES = {}
MAX_DIFF = 0.
START = time.monotonic()


def check(condition, name):
    CHECKS[name] += 1
    if not condition:
        raise AssertionError(name)


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    READS[str(path.relative_to(REPO))] = h.hexdigest()
    return h.hexdigest()


def load(path):
    path = Path(path)
    digest(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def obj(value, ensure_ascii=True):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, ensure_ascii=ensure_ascii).encode()).hexdigest()


def arr(values):
    a = np.asarray(values)
    h = hashlib.sha256(json.dumps({'shape': a.shape, 'dtype': str(a.dtype)},
        sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def source_fp(x, y):
    return obj({'X': arr(x), 'y': arr(y)}, ensure_ascii=False)


def compare_float(a, b, name):
    global MAX_DIFF
    diff = abs(float(a) - float(b))
    MAX_DIFF = max(MAX_DIFF, diff)
    check(diff <= 2e-14, name)


def metric(y, pred, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for a, b in zip(y, pred, strict=True):
        cm[lookup[a], lookup[b]] += 1
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(len(classes)), where=den > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(len(classes)), where=cm.sum(1) > 0)
    return {'accuracy': float(tp.sum() / cm.sum()), 'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall[cm.sum(1) > 0].mean()),
        'confusion_matrix': cm.tolist(), 'total': int(cm.sum()), 'correct': int(tp.sum())}


protocol_path = REPO / 'data/splits/supplemental_spectra_20261104_v1/bound_protocol.json'
pr = load(protocol_path)
protocol_sha = digest(protocol_path)
check(digest(RUN / 'protocol_snapshot.json') == protocol_sha, 'protocol_snapshot_byte_exact')
lock = load(RUN / 'source_lock.json')
check(lock['protocol_sha256'] == protocol_sha, 'source_lock_bound_protocol')
check(lock['status'] == 'DONE_SOURCE_LOCKED_NO_QUERY_PREDICTIONS', 'complete_source_lock')
check(lock['query_truth_read'] is False, 'source_declares_no_query_truth')
check(pr['seeds'] == [42, 43, 44], 'exact_three_seeds')
refs = [{'path': p, 'sha256': s} for p, s in sorted(pr['code_files'].items())]
for ref in refs:
    check(digest(REPO / ref['path']) == ref['sha256'], 'code_inventory_byte_exact')
code = obj(refs)
check(code == lock['code_version'], 'code_inventory_aggregate')
for key in ['intake_manifest', 'preparation']:
    check(digest(REPO / pr[key]['path']) == pr[key]['sha256'], 'frozen_' + key)
release = load(RUN / 'release_snapshot.json')
check(release == load(REPO / 'data/research/supplemental_source_release_20261104.json'),
      'release_snapshot_exact')
check(release['protocol_sha256'] == protocol_sha and release['data_ready'] is True
      and release['allowed_stages'] == ['source'], 'source_release_correct_scope')
configs = [c for group in pr['candidates'].values() for c in group]
check(len(configs) == 21 and len({c['id'] for c in configs}) == 21, 'twenty_one_candidates')
check({k: len(v) for k, v in pr['candidates'].items()} ==
      {'mcuve_svm': 10, 'svm': 6, 'shrinkage_lda': 3, 'mlp': 2}, 'candidate_family_counts')
taskmap = {t['id']: t for t in pr['tasks']}
expected = {}
sources = {}
for t in pr['tasks']:
    tid = t['id']
    check(digest(REPO / t['source_pack']['path']) == t['source_pack']['sha256'], 'source_pack_sha')
    with np.load(REPO / t['source_pack']['path'], allow_pickle=False) as z:
        p = {k: z[k] for k in ['X', 'axis', 'record_ids', 'entity_ids', 'y']}
    p['lookup'] = {v: i for i, v in enumerate(p['record_ids'])}
    sources[tid] = p
    check(len(p['lookup']) == len(p['y']), 'unique_source_ids')
    check(len(set(p['entity_ids'])) == len(p['y']), 'unique_declared_source_entities')
    check(arr(p['axis']) == t['axis_sha256'], 'source_axis_sha')
    check(np.array_equal(np.unique(p['y']), np.array(t['classes'])), 'source_class_vocabulary')
    parent = t.get('inherit_selection_from')
    if parent:
        check(not t['inner_folds'] and not t['source_calibration_episodes'], 'no_singleton_cv_or_pseudoepisodes')
        check(lock['selection'][tid]['selected'] == lock['selection'][parent]['selected'], 'B_inherits_A_config_only')
        check(lock['selection'][tid]['parent'] == parent, 'B_inheritance_parent')
    else:
        summary = load(RUN / 'selection' / tid / 'summary.json')
        check(summary == lock['selection'][tid], 'selection_summary_locked_exact')
        foldids = []
        for fi, f in enumerate(t['inner_folds']):
            train, valid = set(f['train_ids']), set(f['valid_ids'])
            check(not train & valid and train | valid == set(p['record_ids']), 'inner_train_valid_partition')
            check(set(p['y'][[p['lookup'][v] for v in train]]) == set(t['classes']), 'inner_train_has_all_classes')
            foldids.extend(f['valid_ids'])
            for seed in pr['seeds']:
                seen = set()
                for c in configs:
                    trans = (c['preprocessing'], None if c['mc'] is None else
                             (c['mc']['runs'], c['mc']['components']))
                    fid = f"seed{seed}_fold{fi}_candidate{c['id']}"
                    expected[(tid, fid)] = (c, seed, f['train_ids'], 'inner', trans in seen)
                    seen.add(trans)
        check(len(foldids) == len(p['y']) and set(foldids) == set(p['record_ids']), 'OOF_once_per_source_entity')
        means = {}
        per_seed = {(r['candidate_id'], r['seed']): r for r in summary['per_seed_metrics']}
        check(len(per_seed) == 63 and len(summary['per_seed_metrics']) == 63, 'candidate_seed_metrics_complete')
        for c in configs:
            vals = []
            for seed in pr['seeds']:
                path = RUN / 'selection' / tid / f"{c['id']}_seed{seed}.npz"
                digest(path)
                with np.load(path, allow_pickle=False) as z:
                    o = {k: z[k] for k in z.files}
                check(np.array_equal(o['record_ids'], p['record_ids']), 'OOF_source_ID_order_exact')
                check(np.array_equal(o['classes'], np.array(t['classes'])), 'OOF_class_order')
                check(o['scores'].shape == (len(p['y']), len(t['classes']))
                      and np.isfinite(o['scores']).all(), 'OOF_scores_complete_finite')
                m = metric(p['y'], o['predictions'], t['classes'])
                r = per_seed[c['id'], seed]
                for name in ['accuracy', 'macro_f1', 'balanced_accuracy']:
                    compare_float(m[name], r[name], 'source_metric_independent_recompute')
                check(m['confusion_matrix'] == r['confusion_matrix'], 'source_confusion_exact')
                check(m['correct'] == r['correct'] and m['total'] == r['total'], 'source_denominator_exact')
                METRICS.append({'task': tid, 'candidate': c['id'], 'seed': seed, **m})
                vals.append(m)
            means[c['id']] = {k: float(np.mean([m[k] for m in vals]))
                for k in ['accuracy', 'macro_f1', 'balanced_accuracy']}
            for name in means[c['id']]:
                compare_float(means[c['id']][name], summary['means'][c['id']][name], 'mean_three_seed_metric')
        SELECTED[tid] = {}
        for family, candidates in pr['candidates'].items():
            i = min(range(len(candidates)), key=lambda j: (
                -means[candidates[j]['id']]['macro_f1'], -means[candidates[j]['id']]['accuracy'],
                -means[candidates[j]['id']]['balanced_accuracy'], j))
            choice = candidates[i]
            check(choice == summary['selected'][family], 'independent_family_selection')
            SELECTED[tid][family] = {'candidate': choice['id'], **means[choice['id']]}
    for family, c in lock['selection'][tid]['selected'].items():
        for seed in pr['seeds']:
            expected[tid, f"final_seed{seed}_candidate{c['id']}"] = (
                c, seed, p['record_ids'].tolist(), 'final', False)
    for epno, ep in enumerate(t.get('source_calibration_episodes', [])):
        role_sets = [set(ep[k]) for k in ['base_ids', 'support_ids', 'query_ids']]
        check(all(not role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)),
              'source_episode_disjoint_pseudoroles')
        check(set.union(*role_sets) == set(p['record_ids']), 'source_episode_full_source_partition')
        check(ep['stage'] == ('selection' if epno < 3 else 'gate'), 'three_selection_three_gate_episodes')
        check(ep['model_seed'] == 42, 'fixed_calibration_model_seed')
        for shot, ids in ep['shots'].items():
            check(set(ids) <= role_sets[1] and len(ids) == len(set(ids)), 'source_episode_shot_unique_subset')
            yy = p['y'][[p['lookup'][v] for v in ids]]
            check(all(np.sum(yy == c) == int(shot) for c in t['classes']), 'source_episode_shot_count')
        c = lock['selection'][tid]['selected']['mcuve_svm']
        expected[tid, f"pseudo{epno}_candidate{c['id']}"] = (c, 42, ep['base_ids'], 'pseudo', False)

# Every final family/seed, with fresh full-source context and source-only hashes.
model_keys = [(m['task'], m['family'], m['seed']) for m in lock['models']]
check(len(model_keys) == 48 and len(set(model_keys)) == 48, '48_unique_final_models')
check(set(model_keys) == {(t, f, s) for t in taskmap for f in pr['candidates'] for s in pr['seeds']},
      'all_tasks_families_three_final_seeds')
for m in lock['models']:
    check(m['config'] == lock['selection'][m['task']]['selected'][m['family']], 'final_config_matches_selection')
    check(m['source_pack_sha256'] == taskmap[m['task']]['source_pack']['sha256'], 'final_source_pack_bound')
    check(digest(RUN / m['path']) == m['sha256'], 'final_model_weight_hash_without_loading')
    check(digest(RUN / m['source_reference']['path']) == m['source_reference']['sha256'], 'source_reference_hash')

meta_files = sorted((RUN / 'fits').rglob('*.metadata.json.gz'))
trace_files = sorted((RUN / 'fits').rglob('*.jsonl.gz'))
check(len(expected) == len(meta_files) == len(trace_files) == 633, 'all_633_expected_fit_artifacts')
observed = set()
for path in meta_files:
    tid = path.parent.name
    fid = path.name.removesuffix('.metadata.json.gz')
    key = (tid, fid)
    check(key in expected and key not in observed, 'no_unregistered_or_duplicate_source_fit')
    observed.add(key)
    c, seed, ids, phase, reused = expected[key]
    p = sources[tid]
    idx = np.array([p['lookup'][v] for v in ids])
    m = load(path)
    check(m['source_ids'] == ids and m['allowed_fit_ids'] == ids, 'actual_and_allowed_IDs_exact_training_partition')
    check(m['config'] == c and m['seed'] == seed, 'fit_configuration_and_seed')
    check(m['source_fingerprint'] == source_fp(p['X'][idx], p['y'][idx].astype(str)), 'actual_raw_source_values_and_labels_fingerprint')
    check(m['axis_sha256'] == arr(p['axis']) and m['code_version'] == code, 'fit_axis_code_bound')
    check(m['class_order'] == taskmap[tid]['classes'], 'fit_all_declared_classes')
    check(m['transform_cache_reused'] is reused, 'cache_reuse_matches_prior_candidate_in_same_fold')
    trace = path.with_name(fid + '.jsonl.gz')
    digest(trace)
    starts = collections.Counter()
    stack = []
    coefs = []
    pls_ids = []
    events_n = 0
    with gzip.open(trace, 'rt') as f:
        for line in f:
            e = json.loads(line)
            events_n += 1
            STATS['event_lines'] += 1
            name, component = e['event'], e['component']
            expected_split = tid + '/' + fid.rsplit('_candidate', 1)[0]
            check(e['split_id'] == expected_split and e['code_version'] == code
                  and e['dataset_version'] == pr['dataset_version'], 'trace_context_exact')
            fitids = e.get('fit_sample_ids', [])
            check(bool(fitids) and len(fitids) == len(set(fitids)) and set(fitids) <= set(ids), 'instrumented_actual_IDs_within_training_only')
            if name == 'cache_reused':
                ident = e['cache_identity']
                want = {'source': m['source_fingerprint'], 'ids': arr(np.array(ids, dtype=p['record_ids'].dtype)),
                    'axis': arr(p['axis']), 'seed': seed, 'split_id': expected_split,
                    'dataset': pr['dataset_version'], 'code': code, 'physical_quantity': taskmap[tid]['quantity'],
                    'preprocessing': c['preprocessing'], 'mc': None if c['mc'] is None else
                    {k: c['mc'][k] for k in ['runs', 'components']}}
                check(reused and ident == want and fitids == ids, 'cache_actual_identity_exact_no_crossfold')
                STATS['cache_reuse_events'] += 1
                continue
            check(name in ['fit_start', 'fit_completed'], 'no_failed_rejected_or_unknown_fit_events')
            proof = {k: v for k, v in e.items() if k not in ['event', 'utc']}
            check(e['train_id_digest'] == obj(fitids, ensure_ascii=False), 'actual_train_ID_digest')
            if name == 'fit_start':
                stack.append(proof)
                starts[component] += 1
                STATS['fit_start_' + component] += 1
                if component == 'MC-UVE.coefficient_standardization':
                    coefs.append(fitids)
                elif component == 'MC-UVE.PLSRegression.fit':
                    pls_ids.append(fitids)
                else:
                    check(fitids == ids, 'scaler_selector_classifier_exact_training_rows')
            else:
                check(bool(stack) and stack.pop() == proof, 'nested_actual_fit_start_completed_pair')
    check(not stack, 'all_instrumented_operations_closed')
    check(starts[c['family']] == 1, 'exactly_one_classifier_fit_per_context')
    check(starts['StandardScaler'] == (0 if reused else 1), 'scaler_only_fit_when_not_cached')
    if c['mc'] is not None and not reused:
        check(starts['MC-UVE.original_ranking'] == 1, 'one_original_MC_ranking')
        _, labels = np.unique(p['y'][idx], return_inverse=True)
        rng = np.random.default_rng(seed)
        expected_mc = []
        for run in range(c['mc']['runs']):
            priorities = rng.random(len(ids))
            selected = []
            for label in np.unique(labels):
                candidates = np.flatnonzero(labels == label)
                n = min(len(candidates), max(1, int(np.ceil(len(candidates) * .8))))
                selected.extend(candidates[np.argsort(priorities[candidates], kind='stable')[:n]])
            expected_mc.append([ids[i] for i in np.sort(selected)])
        check(coefs == expected_mc, 'independent_MC_stratified_rng_subsets_exact')
        expected_pls = [row for row in expected_mc for _ in taskmap[tid]['classes']]
        check(pls_ids == expected_pls, 'every_legacy_PLS_fit_exact_MC_subset_and_class_count')
        STATS['uncached_MC_selectors'] += 1
        STATS['actual_MC_subsamples'] += len(coefs)
        STATS['actual_legacy_PLS_fits'] += len(pls_ids)
    else:
        check(not coefs and not pls_ids and not starts['MC-UVE.original_ranking'], 'no_unlogged_extra_MC_when_cached_or_none')
    STATS['fit_' + phase] += 1
    STATS['cache_reused_fits'] += int(reused)
    FITS.append({'task': tid, 'fit_id': fid, 'phase': phase, 'seed': seed,
                 'candidate': c['id'], 'source_n': len(ids), 'cache_reused': reused,
                 'events': events_n, 'PLS_fits': len(pls_ids),
                 'metadata_sha256': READS[str(path.relative_to(REPO))],
                 'trace_sha256': READS[str(trace.relative_to(REPO))]})
check(observed == set(expected), 'full_expected_fit_grid_closed')
check(lock['cost']['completed_source_model_fits'] == len(observed), 'ledger_fit_count_exact')

# Recompute policy selection/gating from recorded source-only evaluation rows.
grid = pr['calibration']['conservative']['candidate_grid']
for tid, t in taskmap.items():
    policy = lock['conservative_policies'][tid]
    episodes = t.get('source_calibration_episodes', [])
    if not episodes:
        check(policy['status'] == 'NOT_RUN' and policy['fallback_to_identity'] is True
              and policy['selected'] is None and t['calibration']['conservative_policy'] == 'global_no_op',
              'B_singleton_policy_global_identity')
        POLICIES[tid] = {'status': 'NOT_RUN_GLOBAL_IDENTITY', 'reason': policy['reason']}
        continue
    rows = load(RUN / 'calibration_selection' / tid / 'source_episode_results.json.gz')
    want_keys = {(i, int(k), c['id']) for i, ep in enumerate(episodes) for k in ep['shots'] for c in grid}
    got_keys = [(r['episode'], r['shot'], r['candidate_id']) for r in rows]
    check(len(got_keys) == len(set(got_keys)) and set(got_keys) == want_keys, 'source_policy_episode_shot_candidate_grid_complete')
    counts = collections.Counter()
    for r in rows:
        ep = episodes[r['episode']]
        check(r['stage'] == ep['stage'], 'source_calibration_stage_exact')
        c = next(c for c in grid if c['id'] == r['candidate_id'])
        check(r['candidate'] == c and r['configuration_sha256'] == obj(c), 'source_calibrator_config_hash')
        wanted_ids = ep['shots'][str(r['shot'])]
        stack = []
        for e in r['fit_events']:
            counts['adapter_event_lines'] += 1
            check(e['fit_sample_ids'] == wanted_ids, 'adapter_fits_only_current_source_support_shot')
            check(e['split_id'] == f"{tid}/pseudo{r['episode']}/K{r['shot']}/{c['id']}"
                  and e['code_version'] == code and e['dataset_version'] == pr['dataset_version'],
                  'source_adapter_context_bound')
            proof = {k: v for k, v in e.items() if k not in ['event', 'utc']}
            if e['event'] == 'fit_start':
                stack.append(proof)
                counts['adapter_fits'] += 1
            else:
                check(e['event'] == 'fit_completed' and bool(stack) and stack.pop() == proof,
                      'source_adapter_fit_pair_closed')
        check(not stack, 'source_adapter_all_operations_closed')
    aggregate = {}
    fields = ['accuracy', 'macro_f1', 'balanced_accuracy', 'accuracy_gain', 'macro_f1_gain',
              'balanced_accuracy_gain', 'corrected', 'harmed']
    for stage in ['selection', 'gate']:
        aggregate[stage] = {}
        for c in grid:
            subset = [r for r in rows if r['stage'] == stage and r['candidate_id'] == c['id']]
            values = {name: float(np.mean([r[name] for r in subset])) for name in fields}
            values['evaluations'] = len(subset)
            aggregate[stage][c['id']] = values
            for name, v in values.items():
                compare_float(v, policy['aggregate'][stage][c['id']][name], 'source_policy_aggregate_recomputed')
    def allowed(c, stage):
        s = aggregate[stage][c['id']]
        return s['accuracy_gain'] > .001 and s['macro_f1_gain'] > .001 and s['balanced_accuracy_gain'] >= -.005
    order = lambda c: (-aggregate['selection'][c['id']]['macro_f1_gain'], c['cap_fraction'],
                       -c['minimum_support_errors'], c['source_gate_quantile'], c['id'])
    eligible = [c for c in grid if allowed(c, 'selection')]
    proposed = min(eligible, key=order) if eligible else None
    selected = proposed if proposed is not None and allowed(proposed, 'gate') else None
    check(policy['proposed'] == proposed and policy['selected'] == selected
          and policy['diagnostic'] == min(grid, key=order)
          and policy['fallback_to_identity'] == (selected is None), 'independent_source_calibration_gate_decision')
    check(policy == load(RUN / 'calibration_selection' / tid / 'policy.json'), 'source_policy_file_locked_exact')
    POLICIES[tid] = {'episode_count': len(episodes), 'selection_episodes': 3, 'gate_episodes': 3,
        'evaluations': len(rows), 'selected': selected, 'fallback_to_identity': selected is None,
        'corrected_total_over_repeated_source_evaluations': sum(r['corrected'] for r in rows),
        'harmed_total_over_repeated_source_evaluations': sum(r['harmed'] for r in rows), **counts}
    STATS['source_calibration_evaluations'] += len(rows)
    STATS['source_calibration_adapter_fits'] += counts['adapter_fits']

check(all('/support.npz' not in p and '/query.npz' not in p and 'truth' not in p
          and '/predictions/' not in p for p in READS), 'audit_opens_no_target_support_query_truth_or_predictions')
receipt = {'status': 'PASS_SOURCE_ONLY_INDEPENDENT_AUDIT', 'utc': datetime.now(timezone.utc).isoformat(),
    'protocol_sha256': protocol_sha, 'source_lock_sha256': digest(RUN / 'source_lock.json'),
    'code_inventory_sha256': code, 'source_code_inventory_entries': len(refs),
    'counts': dict(STATS), 'checked_assertions': sum(CHECKS.values()), 'check_categories': dict(CHECKS),
    'maximum_recomputed_metric_difference': MAX_DIFF, 'selected_source_configurations': SELECTED,
    'conservative_source_policies': POLICIES, 'source_ledger_cost': lock['cost'],
    'read_file_count': len(READS), 'new_model_fits': 0, 'native_model_replays': 0,
    'quantum_calls': 0, 'target_support_or_query_pack_opened': False, 'target_truth_opened': False,
    'query_predictions_opened': False, 'historical_results_modified': False,
    'elapsed_wall_seconds': time.monotonic() - START,
    'limitations': ['Source OOF is a development selection result, not independent confirmation.',
        'Source calibration selection and gate episodes reuse the same source population; they are not independent batches.',
        'Instrumented trace ID/fingerprint consistency does not establish biological independence or intercept arbitrary uninstrumented code.',
        'No refitting, native-weight replay, or raw transformed PLS numerical recreation was undertaken.',
        'Calibration policy arithmetic was recomputed from source-only logged evaluation rows; no missing pseudoquery predictions were fabricated.']}
outputs = {'source_audit.json': receipt, 'source_fit_inventory.json': FITS,
           'source_oof_metrics_recomputed.json': METRICS, 'read_file_hashes.json': READS}
for name, value in outputs.items():
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
index = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in outputs}
index['audit_source.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (OUT / 'artifact_index.json').open('x') as f:
    json.dump(index, f, indent=2)
    f.write('\n')
print(json.dumps({'status': receipt['status'], 'counts': dict(STATS),
    'checks': receipt['checked_assertions'], 'max_metric_difference': MAX_DIFF,
    'audit_sha256': index['source_audit.json'], 'new_model_fits': 0,
    'target_truth_opened': False, 'elapsed_wall_seconds': receipt['elapsed_wall_seconds']}, indent=2))
