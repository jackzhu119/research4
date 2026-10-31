"""Boundary tests: synthetic numeric fixtures are not plant-model experiments."""
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from herb_spectra import public_fruit_regression as fruit
from herb_spectra.leakage_guard import LeakageError


class BatchDependentFixture:
    def predict(self, X):
        return np.repeat(X.mean(), len(X))


class MutatingFixture:
    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.zeros(len(X))


class PublicFruitRegressionTests(unittest.TestCase):
    def setUp(self):
        (fruit.ROOT / 'outputs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='fruit-unit-', dir=fruit.ROOT / 'outputs')
        self.out = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        rng = np.random.default_rng(174)
        source_ids = [f'S{i:02d}' for i in range(20)]
        query_ids = [f'Q{i:02d}' for i in range(12)]
        source_groups = {rid: f'SG{i // 4}' for i, rid in enumerate(source_ids)}
        query_groups = {rid: f'QG{i // 4}' for i, rid in enumerate(query_ids)}
        source_X, query_X = rng.normal(size=(20, 128)), rng.normal(size=(12, 128))
        source_y = 17 + 3 * source_X[:, 0] - source_X[:, 5]
        for name, X, ids in (('source', source_X, source_ids), ('query', query_X, query_ids)):
            pack = {'X': X, 'record_ids': np.array(ids), 'feature_index': np.arange(128)}
            if name == 'source':
                pack['source_targets'] = source_y
            np.savez_compressed(self.out / (name + '.npz'), **pack)
        context = {'split_id': 'unit.MA_Brix.outer0.inner0', 'task': 'MA_Brix', 'outer': 0,
                   'inner': 0, 'stage': 'inner', 'source_ids': source_ids, 'query_ids': query_ids,
                   'source_groups': sorted(set(source_groups.values())),
                   'query_groups': sorted(set(query_groups.values())),
                   'source_group_by_id': source_groups, 'query_group_by_id': query_groups,
                   'source_pack': 'source.npz', 'query_pack': 'query.npz'}
        for key in ('source_pack', 'query_pack'):
            context[key + '_sha256'] = fruit.sha(self.out / context[key])
        return context

    def test_grid_exact_budget_and_svr_inheritance(self):
        grid = fruit.r1_grid()
        self.assertEqual(len(grid), 34)
        self.assertEqual({k: sum(c['family'] == k for c in grid) for k in ('PLSR', 'full_SVR', 'MLP')},
                         {'PLSR': 6, 'full_SVR': 24, 'MLP': 4})
        source = next(c for c in grid if c['family'] == 'full_SVR' and c['preprocessing'] == 'snv')
        r2 = fruit.r2_grid(source)
        self.assertEqual(len(r2), 8)
        for c in r2:
            for key in ('C', 'gamma', 'epsilon', 'preprocessing'):
                self.assertEqual(c[key], source[key])
        self.assertEqual(4 * 5 * 3 * 3 * (len(grid) + len(r2)), 7560)
        self.assertEqual(fruit.default_protocol()['MC_PLScall_budget_with_transform_cache'], 10800)

    def test_fixed_block_means_and_pointwise_snv(self):
        X = np.vstack([np.arange(7000), np.arange(7000) + 100]).astype(float)
        result = fruit.block_means(X)
        self.assertEqual(result.shape, (2, 128))
        expected = np.array([np.arange(7000)[b].mean() for b in np.array_split(np.arange(7000), 128)])
        np.testing.assert_array_equal(result[0], expected)
        np.testing.assert_array_equal(fruit.block_means(X[::-1])[::-1], result)
        single = np.vstack([fruit.pointwise_preprocess(row[None], 'snv') for row in result])
        np.testing.assert_allclose(single, fruit.pointwise_preprocess(result, 'snv'))
        np.testing.assert_allclose(single.mean(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(single.std(axis=1), 1)
        with self.assertRaises(LeakageError):
            fruit.block_means(np.ones((3, 7001)))

    def test_all_persistent_fruits_grouped_without_y_argument(self):
        # Labels are deliberately absent; split construction only has groups/type.
        data = {'fruit_type': np.repeat(['MA', 'PM', 'PR'], 40),
                'groups': np.array([f'{kind}{g}' for kind in ('MA', 'PM', 'PR') for g in range(10) for _ in range(4)])}
        splits = fruit.split_indices(data)
        self.assertEqual(len(splits), 80)
        for task in fruit.TASKS:
            outer = [s for s in splits if s[0] == task and s[2] is None]
            self.assertEqual(len(outer), 5)
            self.assertEqual(len(set(np.concatenate([s[4] for s in outer]))), 40)
        for task, outer, inner, src, val in splits:
            self.assertFalse(set(data['groups'][src]) & set(data['groups'][val]))
            self.assertEqual((len(src), len(val)), (32, 8) if inner is None else ((20, 12) if len(val) == 12 else (24, 8)))

    def test_context_rejects_shared_fruit_even_different_record_ids(self):
        context = self.fixture()
        self.assertTrue(fruit.validate_context(context))
        poisoned = copy.deepcopy(context)
        for rid in poisoned['query_ids'][:4]:
            poisoned['query_group_by_id'][rid] = 'SG0'
        poisoned['query_groups'] = sorted(set(poisoned['query_group_by_id'].values()))
        with self.assertRaisesRegex(LeakageError, 'fruit crossed'):
            fruit.validate_context(poisoned)

    def test_strict_feature_pack_rejects_metadata_truth_wrong_axis_and_string_X(self):
        self.fixture()
        query = fruit.strict_pack(self.out / 'query.npz')
        for added in ('source_targets', 'fruit_type', 'groups', 'timestamp'):
            np.savez_compressed(self.out / (added + '.npz'), **query, **{added: np.zeros(12)})
            with self.assertRaises(LeakageError):
                fruit.strict_pack(self.out / (added + '.npz'))
        for key, value in [('X', query['X'].astype(str)), ('feature_index', np.arange(128) + 1),
                           ('record_ids', np.array(['duplicate'] * 12))]:
            broken = {**query, key: value}
            np.savez_compressed(self.out / 'broken.npz', **broken)
            with self.assertRaises(LeakageError):
                fruit.strict_pack(self.out / 'broken.npz')

    def test_real_synthetic_fit_logs_scalers_internal_MC_and_source_only(self):
        # Tiny synthetic guard test only: never loads the public fruit NPZ.
        context = self.fixture()
        source = fruit.strict_pack(self.out / 'source.npz', source=True)
        svr = next(c for c in fruit.r1_grid() if c['family'] == 'full_SVR')
        config = fruit.r2_grid(svr)[0]
        record, rows = fruit.trial(self.out, context, config, 42, {})
        self.assertEqual(len(rows), 12)
        self.assertEqual(record['retained_features'], 32)
        trace = [json.loads(line) for line in (self.out / 'fit_scope_events.jsonl').read_text().splitlines()]
        completed = [r for r in trace if r['event'] == 'fit_completed']
        for row in completed:
            self.assertTrue(set(row['fit_sample_ids']) <= set(context['source_ids']))
            self.assertFalse(set(row['fit_sample_ids']) & set(context['query_ids']))
        self.assertEqual(sum(r['component'] == 'MC-UVE.PLSRegression.fit' for r in completed), 10)
        self.assertEqual(sum(r['component'] == 'fruit.X_StandardScaler' for r in completed), 1)
        self.assertEqual(sum(r['component'] == 'fruit.y_StandardScaler' for r in completed), 1)
        metadata = fruit.read((self.out / record['estimator_path']).with_suffix('.json'))
        transform = fruit.load_guarded_cache(self.out / record['transform']['path'],
                                            fruit.CacheKey(record['transform']['cache_identity']))
        np.testing.assert_allclose(transform.y_scaler.mean_, np.mean(source['source_targets']))
        self.assertTrue(metadata['inference_audit']['state_unchanged'])
        self.assertFalse(metadata['outer_truth_parsed'])

    def test_source_cache_ratio_reuse_but_cross_fold_key_separation(self):
        context = self.fixture()
        source = fruit.strict_pack(self.out / 'source.npz', source=True)
        svr = next(c for c in fruit.r1_grid() if c['family'] == 'full_SVR')
        configs = fruit.r2_grid(svr)
        cache = {}
        first, ref1 = fruit.transform_source(source, configs[0], 42, context, self.out, cache)
        second, ref2 = fruit.transform_source(source, configs[1], 42, context, self.out, cache)
        self.assertIs(first, second)
        self.assertTrue(ref2['reused'])
        self.assertEqual(ref1['path'], ref2['path'])
        self.assertEqual(len(first.indices(.25)), 32)
        self.assertEqual(len(first.indices(.5)), 64)
        different = {**context, 'split_id': context['split_id'] + '_another'}
        _, ref3 = fruit.transform_source(source, configs[0], 42, different, self.out, cache)
        self.assertNotEqual(ref1['path'], ref3['path'])

    def test_changed_query_inputs_do_not_change_source_fitted_objects(self):
        context = self.fixture()
        config = next(c for c in fruit.r1_grid() if c['family'] == 'full_SVR')
        cache = {}
        record1, predictions1 = fruit.trial(self.out, context, config, 42, cache)
        query = fruit.strict_pack(self.out / 'query.npz')
        np.savez_compressed(self.out / 'changed_query.npz', **{**query, 'X': query['X'] + 5})
        changed = {**context, 'query_pack': 'changed_query.npz',
                   'query_pack_sha256': fruit.sha(self.out / 'changed_query.npz')}
        # Same source transform cache, distinct immutable estimator record name.
        record2, predictions2 = fruit.trial(self.out, changed, {**config, 'id': config['id']+'_fixture'}, 42, cache)
        self.assertEqual(record1['transform']['sha256'], record2['transform']['sha256'])
        self.assertEqual(joblib.hash(joblib.load(self.out / record1['estimator_path'])),
                         joblib.hash(joblib.load(self.out / record2['estimator_path'])))
        self.assertNotEqual([r['y_predicted'] for r in predictions1], [r['y_predicted'] for r in predictions2])

    def test_inference_rejects_batch_dependence_and_state_mutation(self):
        context = self.fixture()
        source = fruit.strict_pack(self.out / 'source.npz', source=True)
        config = next(c for c in fruit.r1_grid() if c['family'] == 'full_SVR')
        transform, _ = fruit.transform_source(source, config, 42, context, self.out, {})
        with self.assertRaisesRegex(LeakageError, 'batch or order'):
            fruit.inference_check(transform, BatchDependentFixture(), source['X'][:4], None)
        with self.assertRaisesRegex(LeakageError, 'changed learned state'):
            fruit.inference_check(transform, MutatingFixture(), source['X'][:4], None)

    def test_pooled_inner_RMSE_and_all_seeds_not_fold_mean(self):
        ids = np.array([f'I{i}' for i in range(32)])
        contexts = []
        for inner, indices in enumerate((np.arange(12), np.arange(12, 24), np.arange(24, 32))):
            path = self.out / f'truth{inner}.csv'
            fruit.csvwrite(path, [{'record_id': rid, 'target': float(i)} for rid, i in zip(ids[indices], indices)])
            contexts.append({'inner': inner, 'query_ids': ids[indices].tolist(),
                             'truth': path.name, 'truth_sha256': fruit.sha(path), 'split_id': f'inner{inner}'})
        def fake_trial(out, context, config, seed, cache):
            error = context['inner'] + 1 + (seed - 42)
            rows = [{'record_id': rid, 'seed': seed, 'y_predicted': float(int(rid[1:]) + error)} for rid in context['query_ids']]
            return {'synthetic_mock': True}, rows
        with patch.object(fruit, 'trial', side_effect=fake_trial):
            case = fruit.candidate_cases(self.out, fruit.r1_grid()[0], contexts, {})
        expected = np.mean([np.sqrt((12*(1+s)**2 + 12*(2+s)**2 + 8*(3+s)**2)/32) for s in range(3)])
        self.assertAlmostEqual(case['mean_seed_metrics']['rmse'], expected)
        self.assertEqual(case['cases'], 9)
        self.assertEqual(set(case['pooled_metrics_per_seed']), {'42', '43', '44'})
        self.assertEqual(fruit.choose([case])['config']['id'], case['config']['id'])

    def test_all_inner_truth_hashes_before_first_parse(self):
        contexts = []
        for inner in range(3):
            path = self.out / f'truth{inner}.csv'
            fruit.csvwrite(path, [{'record_id': f'I{inner}', 'target': float(inner)}])
            contexts.append({'inner': inner, 'query_ids': [f'I{inner}'], 'truth': path.name,
                             'truth_sha256': fruit.sha(path), 'split_id': f'inner{inner}'})
        (self.out / 'truth2.csv').write_text('tampered')
        with patch.object(fruit, 'trial', return_value=({}, [])), patch.object(fruit, 'load_truth') as parse:
            with self.assertRaisesRegex(LeakageError, 'Inner source truth changed'):
                fruit.candidate_cases(self.out, fruit.r1_grid()[0], contexts, {})
            parse.assert_not_called()

    def test_gate_rejection_before_any_source_fit(self):
        release = self.out / 'release.json'
        fruit.write(release, {'allow_source_training': False})
        with patch.object(fruit, 'verify_run', return_value=({}, {})), patch.object(fruit, 'candidate_cases') as train:
            with self.assertRaisesRegex(LeakageError, 'Root source release'):
                fruit.selection(self.out, release)
            train.assert_not_called()
        with self.assertRaises(LeakageError):
            fruit.local_directory(fruit.ROOT / 'reports' / 'raw-not-licensed')

    def test_every_outer_truth_hash_checked_before_first_semantic_parse(self):
        contexts = []
        for i in range(20):
            path = self.out / f'truth_outer{i}.csv'
            fruit.csvwrite(path, [{'record_id': f'R{i}', 'target': float(i)}])
            contexts.append({'stage': 'outer', 'truth': path.name, 'truth_sha256': fruit.sha(path)})
        for filename, value in (('protocol.json', {}), ('run_manifest.json', {}),
                                ('selection_lock.json', {}), ('outer_model_records.json', []),
                                ('outer_role_aliases.json', [])):
            fruit.write(self.out / filename, value)
        release = {'allow_outer_evaluation': True, 'created_at_utc': fruit.utc(),
                   'selection_lock_sha256': fruit.sha(self.out / 'selection_lock.json'),
                   'protocol_sha256': fruit.sha(self.out / 'protocol.json'),
                   'run_manifest_sha256': fruit.sha(self.out / 'run_manifest.json')}
        fruit.write(self.out / 'outer_release_snapshot.json', release)
        fruit.csvwrite(self.out / 'outer_predictions.csv.gz', [{'synthetic': 'not_parsed'}])
        lock = {'created_at_utc': fruit.utc(), 'outer_truth_parsed': False,
                'actual_outer_estimators': 0, 'prediction_rows': 2880}
        names = {'prediction_sha256': 'outer_predictions.csv.gz', 'outer_model_records_sha256': 'outer_model_records.json',
                 'outer_role_aliases_sha256': 'outer_role_aliases.json', 'selection_lock_sha256': 'selection_lock.json',
                 'root_outer_release_sha256': 'outer_release_snapshot.json', 'run_manifest_sha256': 'run_manifest.json',
                 'protocol_sha256': 'protocol.json'}
        lock.update({key: fruit.sha(self.out / name) for key, name in names.items()})
        fruit.write(self.out / 'prediction_lock.json', lock)
        (self.out / 'truth_outer19.csv').write_text('tampered late truth file')
        with patch.object(fruit, 'verify_selected', return_value=({'contexts': contexts}, {}, {})), patch.object(fruit, 'load_truth') as parse:
            with self.assertRaisesRegex(LeakageError, 'Frozen outer truth bytes changed'):
                fruit.score(self.out)
            parse.assert_not_called()

    def test_metrics_are_numeric_regression_not_accuracy(self):
        metrics = fruit.regression_metrics([1, 2, 3, 4], [1, 1, 4, 4])
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(.5))
        self.assertEqual(metrics['mae'], .5)
        self.assertAlmostEqual(metrics['r2'], .6)
        self.assertNotIn('accuracy', metrics)
        with self.assertRaises(LeakageError):
            fruit.regression_metrics([1, np.nan], [1, 2])

    def test_independent_score_joins_tasks_ids_and_keeps_all_seeds(self):
        contexts, predictions = [], []
        for task_index, task in enumerate(fruit.TASKS):
            for outer in range(5):
                ids = [f'{task}_F{2*outer+j//4}_T{j%4}' for j in range(8)]
                path = self.out / f'{task}_outer{outer}.csv'
                values = [10 + task_index + (outer * 8 + j) / 10 for j in range(8)]
                fruit.csvwrite(path, [{'record_id': rid, 'target': value} for rid, value in zip(ids, values)])
                context = {'task': task, 'outer': outer, 'stage': 'outer', 'query_ids': ids,
                           'split_id': f'{task}.outer{outer}.final', 'truth': path.name,
                           'query_group_by_id': {rid: f'{task}_F{2*outer+j//4}' for j, rid in enumerate(ids)}}
                contexts.append(context)
                for role_index, role in enumerate(fruit.ROLES):
                    for seed in fruit.SEEDS:
                        error = .1 * role_index + .01 * (seed - 42)
                        predictions.extend({'task': task, 'outer': outer, 'inner': None,
                                            'split_id': context['split_id'], 'candidate_id': 'synthetic_fixture',
                                            'seed': seed, 'record_id': rid, 'y_predicted': value + error,
                                            'role': role, 'role_candidate_id': 'synthetic_fixture'}
                                           for rid, value in zip(ids, values))
        fruit.csvwrite(self.out / 'outer_predictions.csv.gz', predictions)
        fruit.write(self.out / 'prediction_lock.json', {'synthetic_fixture_only': True})
        lock = {'prediction_sha256': fruit.sha(self.out / 'outer_predictions.csv.gz')}
        with patch.object(fruit, 'verify_predictions_before_truth', return_value=(contexts, lock)), patch.object(fruit, 'fit_estimator') as fit:
            result = fruit.score(self.out)
            fit.assert_not_called()
        self.assertEqual(result['cases'], 72)
        metrics = fruit.read(self.out / 'pooled_metrics_by_seed.json')
        self.assertEqual(len(metrics), 72)
        first = next(r for r in metrics if r['task'] == 'MA_Brix' and r['role'] == 'PLSR' and r['seed'] == 43)
        self.assertAlmostEqual(first['rmse'], .11)
        self.assertAlmostEqual(first['mae'], .11)
        self.assertEqual(first['observations'], 40)
        self.assertEqual(first['fruit_groups'], 10)
        self.assertEqual(len(fruit.read(self.out / 'fold_metrics.json')), 360)
        with self.assertRaises(FileExistsError):
            fruit.score(self.out)


if __name__ == '__main__':
    unittest.main()
