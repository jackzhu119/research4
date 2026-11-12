"""Independent scientific safeguards for the third B study; synthetic fits only."""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import joblib
import numpy as np
import pandas as pd

from herb_spectra.calibration import AnchorScoreCalibrator
import b_followup_common_20261110 as artifacts
import b_followup_models_20261110 as previous_models
import b_followup_calibration_20261110 as previous_calibration
import b_round3_common_20261110 as common
import b_round3_regularization_20261110 as regularization
import b_round3_labels_20261110 as label_models
import b_round3_margin_20261110 as margin_audit

OLDROOT = ROOT / 'reports/b_mechanism_20261110'
ROUND2ROOT = ROOT / 'reports/b_mechanism_followup_20261110'


def real_inputs():
    metadata = pd.read_csv(ROOT / 'data/research/species_revision_20261108/split_manifest.csv',
                           dtype=str, keep_default_na=False)
    jobs = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    phase2 = json.loads((ROUND2ROOT / 'frozen_protocol.json').read_text())['design']
    return metadata, jobs, phase2


def synthetic_fixture():
    """Small independent spectra with intentionally shifted unlabeled supports."""
    rng = np.random.default_rng(892)
    rows, roles = [], {name: [] for name in ('source', 'support', 'query')}
    for species, origin in (('枣仁', 'ZY01'), ('酸枣仁', 'ZY03')):
        for role, number in (('source', 9), ('support', 10), ('query', 3)):
            for i in range(number):
                identifier = f'{origin}-{role}-{i:03d}'
                rows.append({'record_id': identifier, 'physical_sample_id': identifier,
                             'origin_group_id': origin, 'species': species, 'role': 'train'})
                roles[role].append(identifier)
    metadata = pd.DataFrame(rows)
    X = rng.normal(size=(len(metadata), 10))
    X[metadata.species.eq('酸枣仁')] += np.linspace(-2, 2, X.shape[1])
    X[metadata.record_id.str.contains('support')] += np.sin(np.arange(X.shape[1])) * 2
    X[metadata.record_id.str.contains('query')] += np.cos(np.arange(X.shape[1])) * 7
    axis = np.linspace(.1, 1.5, X.shape[1])
    job = {'job_id': 'G02_synthetic', 'family': 'G02', 'parent_job_id': 'synthetic_parent',
           'phase2_job_id': 'F02_synthetic_parent', 'experiment': 'E01', 'target_origin': 'ZY03',
           'seed': 42, 'ratio': 'B', 'coverage': 0, 'fold': None,
           'classes': ['枣仁', '酸枣仁'], 'label_column': 'species',
           **{role + '_ids': values for role, values in roles.items()}}
    return metadata, X, axis, job


def take(metadata, X, ids):
    lookup = {record: i for i, record in enumerate(metadata.record_id)}
    return X[[lookup[record] for record in ids]]


class FrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.old_jobs, cls.phase2_jobs = real_inputs()
        cls.indexed = cls.metadata.set_index('record_id', drop=False)

    def test_current_id_kind_and_master_roles(self):
        self.assertEqual(len(self.metadata), 1672)
        self.assertTrue(self.metadata.record_id.is_unique)
        self.assertTrue(self.metadata.physical_sample_id.is_unique)
        self.assertEqual(self.metadata.role.value_counts().to_dict(),
                         {'train': 999, 'prediction': 339, 'calibration': 334})
        self.assertEqual(self.metadata.species.value_counts().to_dict(),
                         {'酸枣仁': 500, '天麻': 405, '黄芪': 403, '枣仁': 364})
        self.assertTrue(self.metadata.loc[self.metadata.origin_group_id.isin(['ZY01', 'ZY02']),
                                          'species'].eq('枣仁').all())

    def test_all375_jobs_are_frozen_train_only_and_parent_queries_match(self):
        runner = importlib.import_module('run_b_mechanism_round3_20261110')
        jobs = runner.build_design(self.metadata, self.old_jobs, self.phase2_jobs)
        self.assertEqual(len(jobs), 375)
        self.assertEqual(pd.Series([job['family'] for job in jobs]).value_counts().to_dict(),
                         {'G04': 240, 'G01': 45, 'G02': 45, 'G03': 45})
        self.assertEqual(len({job['job_id'] for job in jobs}), 375)
        old = {job['job_id']: job for job in self.old_jobs}
        phase2 = {job['job_id']: job for job in self.phase2_jobs}
        for job in jobs:
            artifacts.validate_roles(job, self.metadata)
            parent = phase2[job['phase2_job_id']] if job['family'] == 'G04' else old[job['parent_job_id']]
            for role in ('source', 'support', 'query'):
                self.assertEqual(job[role + '_ids'], parent[role + '_ids'])
            self.assertTrue(self.indexed.loc[job['source_ids'] + job['support_ids'] + job['query_ids'],
                                             'role'].eq('train').all())
        # Counts are computed from prespecified masks, not query performance.
        labels_module = importlib.import_module('b_round3_labels_20261110')
        g02 = [job for job in jobs if job['family'] == 'G02']
        g03 = [job for job in jobs if job['family'] == 'G03']
        self.assertEqual(sum(labels_module.fit_count(job) for job in g02), 360)
        self.assertLessEqual(sum(labels_module.fit_count(job) for job in g03), 675)
        self.assertGreater(sum(labels_module.fit_count(job) for job in g03), 0)
        self.assertEqual(sum(regularization.adapter_count(job) for job in jobs if job['family'] == 'G01'), 1350)

    def test_g01_support_episodes_are_nested_and_complete_pool_not_repeated(self):
        job = next(item for item in self.old_jobs if item['experiment'] == 'E03')
        episodes = regularization.episodes()
        self.assertEqual(len(episodes), 10)
        self.assertEqual([(draw, k) for draw, k in episodes if k == 10], [(0, 10)])
        for draw in (0, 101, 102):
            previous = set()
            for k in (1, 3, 5):
                ids = previous_calibration.support_plan(job, self.metadata, draw, k)
                self.assertTrue(previous.issubset(ids))
                self.assertEqual(len(ids), 5 * k)
                self.assertFalse(set(ids) & set(job['query_ids']))
                previous = set(ids)
        self.assertEqual(len(regularization.expected_endpoints(job)), 42)
        self.assertEqual(regularization.fit_count(job), 0)
        self.assertEqual(regularization.adapter_count(job), 30)

    def test_invalid_role_identity_and_kind_are_rejected(self):
        metadata, _, _, job = synthetic_fixture()
        for case in ('overlap', 'physical_duplicate', 'heldout', 'kind'):
            altered, frame = copy.deepcopy(job), metadata.copy()
            if case == 'overlap':
                altered['support_ids'][0] = altered['query_ids'][0]
            elif case == 'physical_duplicate':
                frame.loc[1, 'physical_sample_id'] = frame.loc[0, 'physical_sample_id']
            elif case == 'heldout':
                frame.loc[frame.record_id.eq(altered['support_ids'][0]), 'role'] = 'calibration'
            else:
                frame.loc[frame.origin_group_id.eq('ZY03'), 'species'] = '枣仁'
            with self.subTest(case=case), self.assertRaises(ValueError):
                artifacts.validate_roles(altered, frame)


class RidgeObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.classes = ['a', 'b', 'c']
        self.labels = np.tile(self.classes, 3)
        self.scores = np.random.default_rng(49).normal(size=(len(self.labels), 3)) + [1.2, -.8, .3]
        self.design = np.column_stack([self.scores, np.ones(len(self.labels))])
        self.target = np.eye(3)[[self.classes.index(label) for label in self.labels]]

    @staticmethod
    def delta(adapter):
        return np.vstack([np.asarray(adapter['coef']) - np.eye(len(adapter['intercept'])),
                          adapter['intercept']])

    def test_mean_loss_is_independently_equal_to_sum_lambda_n(self):
        adapted = regularization.fit_adapter(self.scores, self.labels, self.classes, loss='mean')
        reference = AnchorScoreCalibrator(self.classes, regularization=len(self.labels)).fit(
            self.scores, self.labels)
        np.testing.assert_allclose(adapted['coef'], reference.coef_, rtol=0, atol=1e-13)
        np.testing.assert_allclose(adapted['intercept'], reference.intercept_, rtol=0, atol=1e-13)
        self.assertEqual(adapted['effective_lambda'], len(self.labels))
        delta = self.delta(adapted)
        # Gradient of the mean loss + unit correction prior vanishes.
        gradient = self.design.T @ (self.design @ delta - (self.target - self.scores)) / len(self.labels) + delta
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-13)

    def test_sum_penalized_is_the_existing_anchor_estimator(self):
        adapted = regularization.fit_adapter(self.scores, self.labels, self.classes)
        reference = AnchorScoreCalibrator(self.classes, regularization=1).fit(self.scores, self.labels)
        np.testing.assert_allclose(adapted['coef'], reference.coef_, rtol=0, atol=1e-13)
        np.testing.assert_allclose(adapted['intercept'], reference.intercept_, rtol=0, atol=1e-13)

    def test_free_intercept_has_zero_penalty_and_zero_mean_residual(self):
        for loss in ('sum', 'mean'):
            adapted = regularization.fit_adapter(self.scores, self.labels, self.classes,
                                                  loss=loss, penalize_intercept=False)
            delta = self.delta(adapted)
            penalty = np.diag(adapted['penalty_diagonal'])
            self.assertEqual(penalty[-1, -1], 0)
            residual = self.design @ delta - (self.target - self.scores)
            np.testing.assert_allclose(residual.mean(axis=0), 0, rtol=0, atol=1e-13)
            np.testing.assert_allclose(self.design.T @ residual + penalty @ delta, 0, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(np.diag(penalty)[:-1], adapted['effective_lambda'])

    def test_geometry_df_is_hat_matrix_trace_not_nominal_parameter_count(self):
        adapted = regularization.fit_adapter(self.scores, self.labels, self.classes,
                                              loss='mean', penalize_intercept=False)
        geometry = regularization.adapter_geometry(self.scores, adapted, loss='mean', penalize_intercept=False)
        matrix = self.design.T @ self.design + np.diag(adapted['penalty_diagonal'])
        expected = np.trace(self.design @ np.linalg.solve(matrix, self.design.T))
        self.assertAlmostEqual(geometry['effective_df_per_output'], expected, places=12)
        self.assertLessEqual(expected, geometry['design_rank'] + 1e-12)
        self.assertLess(geometry['effective_df_all_outputs'], geometry['nominal_parameters'])

    def test_score_fit_rejects_invalid_or_missing_classes(self):
        with self.assertRaises(ValueError):
            regularization.fit_adapter(self.scores, ['a'] * len(self.scores), self.classes)
        bad = self.scores.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            regularization.fit_adapter(bad, self.labels, self.classes)
        self.assertFalse(any('query' in key.lower() for key in inspect.signature(regularization.fit_adapter).parameters))


class AtomicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata, self.X, self.axis, self.job = synthetic_fixture()

    def sealed(self, output=None):
        output = self.root / 'new' if output is None else output
        writer = artifacts.TaskWriter(self.job, self.X, self.metadata, self.axis, output, 'input', self.root / 'old')
        true = writer.labels(self.job['query_ids'])
        predicted = true.copy()
        predicted[[0, 4]] = np.where(predicted[[0, 4]] == self.job['classes'][0],
                                      self.job['classes'][1], self.job['classes'][0])
        scores = np.asarray([[2., -1.] if label == self.job['classes'][0] else [-1., 2.] for label in predicted])
        identifier = writer.record('identity', 0, scores)
        result = writer.finish([identifier], new_model_fits=0, new_adapter_fits=0)
        return output / 'jobs' / self.job['job_id'], result, identifier

    def test_metrics_independently_equal_hand_confusion(self):
        path, _, identifier = self.sealed()
        artifacts.verify_task(path, 'input', self.job, [identifier])
        measured = pd.read_csv(path / 'metrics.csv').iloc[0]
        self.assertEqual(json.loads(measured.confusion_matrix_json), [[2, 1], [1, 2]])
        for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            self.assertAlmostEqual(measured[name], 2 / 3)
        self.assertEqual(json.loads(measured.recall_json), {'枣仁': 2 / 3, '酸枣仁': 2 / 3})

    def test_recovery_reuses_exact_bytes_and_cpu_without_fitting(self):
        path, result, identifier = self.sealed()
        hashes = {p.name: artifacts.file_hash(p) for p in path.iterdir()}
        with mock.patch.object(previous_models, 'fit_model', side_effect=AssertionError('unexpected fitting')):
            first = artifacts.verify_task(path, 'input', self.job, [identifier])
            second = artifacts.verify_task(path, 'input', self.job, [identifier])
        self.assertEqual(first['cpu_seconds'], second['cpu_seconds'])
        self.assertEqual(result['cpu_seconds'], first['cpu_seconds'])
        self.assertEqual(hashes, {p.name: artifacts.file_hash(p) for p in path.iterdir()})
        with self.assertRaises(FileExistsError):
            artifacts.TaskWriter(self.job, self.X, self.metadata, self.axis, self.root / 'new', 'input', self.root / 'old')

    def test_input_hash_job_definition_and_endpoint_contract_guard_resume(self):
        path, _, identifier = self.sealed()
        with self.assertRaisesRegex(ValueError, 'input/code'):
            artifacts.verify_task(path, 'changed-input', self.job, [identifier])
        changed = copy.deepcopy(self.job)
        changed['source_ids'] = list(reversed(changed['source_ids']))
        with self.assertRaisesRegex(ValueError, 'frozen design'):
            artifacts.verify_task(path, 'input', changed, [identifier])
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            artifacts.verify_task(path, 'input', self.job, [identifier, 'missing'])

    def test_tampering_and_self_consistent_wrong_metric_are_rejected(self):
        path, result, identifier = self.sealed()
        measured = pd.read_csv(path / 'metrics.csv')
        measured.loc[0, 'accuracy'] = .99
        measured.to_csv(path / 'metrics.csv', index=False)
        result['artifact_hashes']['metrics.csv'] = artifacts.file_hash(path / 'metrics.csv')
        artifacts.write_json(path / 'result.json', result)
        seal = json.loads((path / 'COMPLETE.json').read_text())
        seal['result_sha256'] = artifacts.file_hash(path / 'result.json')
        artifacts.write_json(path / 'COMPLETE.json', seal)
        with self.assertRaisesRegex(ValueError, 'Independent metric mismatch'):
            artifacts.verify_task(path, 'input', self.job, [identifier])

    def test_second_round_evidence_requires_seal_and_matching_role_lock(self):
        output = self.root / 'phase2'
        path, result, _ = self.sealed(output)
        artifacts.write_json(output / 'frozen_protocol.json', {'input_hash': 'input', 'design': [self.job]})
        loaded, directory = common.load_round2_bundle(output, self.job['job_id'])
        self.assertEqual(loaded['job_hash'], result['job_hash'])
        self.assertEqual(path, directory)
        common.check_parent_roles(self.job, loaded)
        changed = copy.deepcopy(self.job)
        changed['query_ids'] = list(reversed(changed['query_ids']))
        with self.assertRaisesRegex(ValueError, 'sample/label'):
            common.check_parent_roles(changed, loaded)
        with (path / 'predictions.csv.gz').open('ab') as handle:
            handle.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            common.load_round2_bundle(output, self.job['job_id'])


class LabelInformationTests(unittest.TestCase):
    def setUp(self):
        self.metadata, self.X, self.axis, self.job = synthetic_fixture()
        self.job['job_id'] = 'G02_' + self.job['parent_job_id']
        self.indexed = self.metadata.set_index('record_id', drop=False)
        self.source_ids = self.job['source_ids']
        self.source_X = take(self.metadata, self.X, self.source_ids)
        self.source_y = self.indexed.loc[self.source_ids, 'species'].to_numpy()
        self.source = previous_models.fit_model(self.source_X, self.source_y, self.axis)

    def true_models(self):
        result = {}
        for k in label_models.K_VALUES:
            ids = self.source_ids + previous_models.nested_support(self.job, self.indexed, k)
            result[k] = previous_models.fit_model(take(self.metadata, self.X, ids),
                         self.indexed.loc[ids, 'species'], self.axis, reference=self.source,
                         scaler_mode='refit', gamma_mode='fixed')
        return result

    def execute_g02(self, output, *, inspect_calls=False):
        models = self.true_models()
        stats_calls, label_calls = [], []
        real_stats, real_model = label_models.fit_with_statistics, label_models.fit_model

        def inspected_stats(X_labeled, y_labeled, X_statistics, axis, reference):
            stats_calls.append((np.asarray(X_labeled).copy(), np.asarray(y_labeled).copy(),
                                np.asarray(X_statistics).copy()))
            return real_stats(X_labeled, y_labeled, X_statistics, axis, reference)

        def inspected_model(X_train, y_train, axis, **kwargs):
            label_calls.append((np.asarray(X_train).copy(), np.asarray(y_train).copy(), kwargs))
            return real_model(X_train, y_train, axis, **kwargs)

        with mock.patch.object(label_models, 'load_parent', return_value=({}, self.source, output)), \
             mock.patch.object(label_models, '_true_refit', side_effect=lambda job, k, *_: (models[k], {'fixture': k})), \
             mock.patch.object(label_models, 'fit_with_statistics', side_effect=inspected_stats), \
             mock.patch.object(label_models, 'fit_model', side_effect=inspected_model):
            result = label_models.execute(self.job, self.X, self.metadata, self.axis,
                                          output, 'input', output / 'old', output / 'phase2')
        if inspect_calls:
            return result, models, stats_calls, label_calls
        return result, models

    def test_unlabeled_statistics_use_source_labels_and_only_named_statistics(self):
        ids = self.source_ids + self.job['support_ids']
        statistics = take(self.metadata, self.X, ids)
        source_before = joblib.hash(self.source)
        model = label_models.fit_with_statistics(self.source_X, self.source_y, statistics, self.axis, self.source)
        expected_statistics = self.source.preprocessor_.transform(statistics)
        np.testing.assert_allclose(model.scaler_.mean_, expected_statistics.mean(axis=0), atol=1e-14, rtol=0)
        np.testing.assert_allclose(model.scaler_.var_, expected_statistics.var(axis=0), atol=1e-14, rtol=0)
        self.assertEqual(model.metadata['source_samples'], len(self.source_ids))
        self.assertEqual(model.metadata['support_labels_used'], 0)
        self.assertEqual(model.estimator_._gamma, self.source.estimator_._gamma)
        self.assertEqual(joblib.hash(self.source), source_before)
        # The far-shifted Query cannot contribute to the explicitly supplied statistics.
        forbidden_statistics = self.source.preprocessor_.transform(
            take(self.metadata, self.X, ids + self.job['query_ids']))
        self.assertFalse(np.allclose(model.scaler_.mean_, forbidden_statistics.mean(axis=0)))
        np.testing.assert_allclose(model._scaled(self.source_X),
                                   model.scaler_.transform(self.source.preprocessor_.transform(self.source_X)),
                                   atol=1e-14, rtol=0)

    def test_g02_fits_only_source_or_source_plus_support_and_reuses_true_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result, _, stats_calls, label_calls = self.execute_g02(output, inspect_calls=True)
            self.assertEqual(result['new_model_fits'], 8)
            self.assertEqual(len(stats_calls), 4)
            self.assertEqual(len(label_calls), 4)
            self.assertEqual(result['metric_rows'], 13)
            for k, (X_labeled, y_labeled, X_stats), (X_train, y_train, options) in zip(
                    label_models.K_VALUES, stats_calls, label_calls):
                support_ids = previous_models.nested_support(self.job, self.indexed, k)
                ids = self.source_ids + support_ids
                np.testing.assert_array_equal(X_labeled, self.source_X)
                np.testing.assert_array_equal(y_labeled, self.source_y)
                np.testing.assert_array_equal(X_stats, take(self.metadata, self.X, ids))
                np.testing.assert_array_equal(X_train, take(self.metadata, self.X, ids))
                np.testing.assert_array_equal(y_train[:len(self.source_ids)], self.source_y)
                copied = label_models.cyclic_labels(self.indexed.loc[support_ids, 'species'], self.job['classes'])
                np.testing.assert_array_equal(y_train[len(self.source_ids):], copied)
                self.assertFalse(set(ids) & set(self.job['query_ids']))
                self.assertEqual(options['gamma_mode'], 'fixed')
                self.assertEqual(options['scaler_mode'], 'refit')
            artifacts.verify_task(output / 'jobs' / self.job['job_id'], 'input', self.job,
                                  label_models.expected_endpoints(self.job))
            endpoints = pd.read_csv(output / 'jobs' / self.job['job_id'] / 'metrics.csv')
            self.assertTrue(endpoints.loc[endpoints.method.eq('true_support'), 'reused_model'].all())
            self.assertTrue(endpoints.loc[endpoints.method.eq('unlabeled_statistics'), 'support_labels_used'].eq(0).all())
            copied = endpoints.loc[endpoints.method.eq('cyclic_support')]
            self.assertTrue(copied.support_noise_fraction.eq(1).all())
            expected_fraction = copied.diagnostic_labels_changed_count / copied.labeled_training_n
            np.testing.assert_allclose(copied.total_labeled_training_noise_fraction,
                                       expected_fraction, atol=1e-14, rtol=0)
            self.assertTrue(copied.total_labeled_training_noise_fraction.lt(1).all())
            self.assertTrue(copied.source_training_labels_corrupted.eq(0).all())

    def test_noise_masks_realize_integer_doses_preserve_histograms_and_nest(self):
        original = self.metadata.copy(deep=True)
        job = {**self.job, 'family': 'G03'}
        plan = label_models.noise_plan(job, self.metadata)
        self.assertEqual(plan, label_models.noise_plan(job, self.metadata))
        self.assertEqual(len(plan['episodes']), 16)
        truth = np.asarray(plan['true_support_labels'])
        seen = {}
        for episode in plan['episodes']:
            copied = np.asarray(episode['copied_support_labels'])
            q, seed = episode['noise_q_per_class'], episode['noise_seed']
            self.assertEqual(sorted(copied.tolist()), sorted(truth.tolist()))
            self.assertEqual(np.count_nonzero(copied != truth), q * len(job['classes']))
            self.assertAlmostEqual(np.mean(copied != truth), episode['actual_noise_rate'])
            counts = self.indexed.loc[episode['changed_support_ids'], 'species'].value_counts()
            self.assertTrue(counts.eq(q).all())
            self.assertTrue(set(episode['changed_support_ids']).issubset(job['support_ids']))
            self.assertFalse(set(episode['changed_support_ids']) & set(job['query_ids']))
            if q < 10:
                prior = seen.get(seed, set())
                self.assertTrue(prior.issubset(episode['changed_support_ids']))
                seen[seed] = set(episode['changed_support_ids'])
        self.assertEqual([(e['noise_q_per_class'], e['noise_seed']) for e in plan['episodes'] if e['noise_q_per_class'] == 10],
                         [(10, 201)])
        pd.testing.assert_frame_equal(self.metadata, original)

    def test_cyclic_labels_are_copies_and_invalid_nonbalanced_masks_fail(self):
        ids = self.job['support_ids']
        labels = self.indexed.loc[ids, 'species'].to_numpy()
        original = labels.copy()
        copied = label_models.cyclic_labels(labels, self.job['classes'])
        np.testing.assert_array_equal(labels, original)
        self.assertFalse(np.shares_memory(labels, copied))
        self.assertEqual(sorted(labels.tolist()), sorted(copied.tolist()))
        self.assertTrue(np.all(labels != copied))
        with self.assertRaisesRegex(ValueError, 'equal counts'):
            label_models.cyclic_labels(labels, self.job['classes'], ids[:1], ids)
        with self.assertRaisesRegex(ValueError, 'outside'):
            label_models.cyclic_labels(labels, self.job['classes'], ['non-support'], ids)

    def test_noise_plan_does_not_use_query_labels_and_counts_unique_objectives(self):
        job = {**self.job, 'family': 'G03'}
        expected = label_models.noise_plan(job, self.metadata)
        altered = self.metadata.copy()
        altered.loc[altered.record_id.isin(job['query_ids']), 'species'] = 'unread query marker'
        self.assertEqual(expected, label_models.noise_plan(job, altered))
        partial = {e['objective_key'] for e in expected['episodes'] if e['noise_q_per_class'] < 10}
        all_keys = {e['objective_key'] for e in expected['episodes']}
        job['noise_plan'] = expected
        self.assertEqual(label_models.fit_count(job), len(partial))
        self.assertEqual(label_models.adapter_count(job), 2 * len(all_keys))
        self.assertEqual(len(label_models.expected_endpoints(job)), 53)

    def test_g03_reuses_full_cyclic_model_and_deduplicates_exact_noise_objectives(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _, true_models = self.execute_g02(output)
            job = {**copy.deepcopy(self.job), 'job_id': 'G03_' + self.job['parent_job_id'], 'family': 'G03'}
            plan = label_models.noise_plan(job, self.metadata)
            # Force two fixed Monte Carlo draws to share an objective. The
            # endpoint/multiplicity remains, but a second fit must not occur.
            first, repeated = plan['episodes'][0], plan['episodes'][1]
            for name in ('changed_support_ids', 'copied_support_labels', 'objective_key'):
                repeated[name] = copy.deepcopy(first[name])
            repeated['reused_from_draw'] = first['draw']
            plan['new_model_fits'] = len({e['objective_key'] for e in plan['episodes'] if e['noise_q_per_class'] < 10})
            plan['new_adapter_fits'] = 2 * len({e['objective_key'] for e in plan['episodes']})
            job['noise_plan'] = plan
            fit_calls, real_fit = [], label_models.fit_model

            def fitted(X_train, y_train, axis, **kwargs):
                fit_calls.append((np.asarray(X_train).copy(), np.asarray(y_train).copy(), kwargs))
                return real_fit(X_train, y_train, axis, **kwargs)

            def old_scores(_path, method, _query_ids):
                ids = plan['support_ids']
                scores, _ = label_models.calibrate_scores(method,
                    self.source.predict_scores(take(self.metadata, self.X, ids)),
                    self.indexed.loc[ids, 'species'].to_numpy(),
                    self.source.predict_scores(take(self.metadata, self.X, job['query_ids'])), job['classes'])
                return scores, np.asarray(job['classes'])[scores.argmax(axis=1)]

            source_before = joblib.hash(self.source)
            with mock.patch.object(label_models, 'load_parent', return_value=({}, self.source, output)), \
                 mock.patch.object(label_models, '_true_refit', return_value=(true_models[10], {'fixture': True})), \
                 mock.patch.object(label_models, '_old_scores', side_effect=old_scores), \
                 mock.patch.object(label_models, 'noise_plan', return_value=plan), \
                 mock.patch.object(label_models, 'fit_model', side_effect=fitted):
                result = label_models.execute(job, self.X, self.metadata, self.axis, output,
                                              'input', output / 'old', output / 'phase2')
            self.assertEqual(len(fit_calls), plan['new_model_fits'])
            self.assertEqual(result['new_model_fits'], plan['new_model_fits'])
            self.assertEqual(result['new_adapter_fits'], plan['new_adapter_fits'])
            self.assertEqual(result['metric_rows'], 53)
            self.assertEqual(joblib.hash(self.source), source_before)
            artifacts.verify_task(output / 'jobs' / job['job_id'], 'input', job,
                                  label_models.expected_endpoints(job))
            measured = pd.read_csv(output / 'jobs' / job['job_id'] / 'metrics.csv')
            full_corruption = measured.loc[measured.noise_q_per_class.eq(10) & measured.method.eq('full_pipeline_fixedgamma')]
            self.assertEqual(len(full_corruption), 1)
            self.assertFalse(bool(full_corruption.iloc[0]['new_model_fit']))
            self.assertTrue(bool(full_corruption.iloc[0]['reused_model']))
            noisy = measured.loc[measured.artificial_label_copy.eq(True)]
            expected = noisy.diagnostic_labels_changed_count / (len(job['source_ids']) + len(job['support_ids']))
            np.testing.assert_allclose(noisy.total_labeled_training_noise_fraction, expected,
                                       atol=1e-14, rtol=0)
            self.assertTrue(noisy.source_training_labels_corrupted.eq(0).all())


class MarginConstraintTests(unittest.TestCase):
    def setUp(self):
        self.metadata, self.X, self.axis, self.job = synthetic_fixture()
        frame = self.metadata.set_index('record_id')
        # Deliberately unequal class counts make C_i differ by class.
        self.ids = self.job['source_ids'][:9] + self.job['source_ids'][9:13]
        self.X_source = take(self.metadata, self.X, self.ids)
        self.y_source = frame.loc[self.ids, 'species'].to_numpy()
        self.X_query = take(self.metadata, self.X, self.job['query_ids'])
        self.y_query = frame.loc[self.job['query_ids'], 'species'].to_numpy()
        self.model = previous_models.fit_model(self.X_source, self.y_source, self.axis)

    def test_box_constraints_use_actual_class_specific_c_and_do_not_fit(self):
        before = joblib.hash(self.model)
        with mock.patch.object(type(self.model.estimator_), 'fit', side_effect=AssertionError('audit refitted')):
            audit = margin_audit.audit_model(self.model, self.X_source, self.y_source, self.ids,
                                              self.X_query, self.y_query, 'balanced')
        weights = dict(zip(self.model.classes_, self.model.estimator_.class_weight_))
        observed = pd.DataFrame(audit['samples'])
        expected_bounds = np.asarray([self.model.estimator_.C * weights[label] for label in self.y_source])
        np.testing.assert_allclose(observed.C_i, expected_bounds, rtol=0, atol=0)
        expected_alpha = np.zeros(len(self.ids))
        expected_alpha[self.model.estimator_.support_] = np.abs(self.model.estimator_.dual_coef_[0])
        np.testing.assert_allclose(observed.alpha, expected_alpha, rtol=0, atol=0)
        np.testing.assert_allclose(observed.alpha_over_C_i, expected_alpha / expected_bounds, rtol=0, atol=0)
        self.assertEqual(audit['summary']['max_alpha_over_C'], (expected_alpha / expected_bounds).max())
        self.assertEqual(audit['summary']['box_violation_count'], 0)
        self.assertEqual(joblib.hash(self.model), before)
        self.assertEqual(margin_audit.fit_count(self.job), 0)
        self.assertEqual(margin_audit.adapter_count(self.job), 0)

    def test_primal_dual_audit_matches_independent_rbf_norm_and_hinge(self):
        audit = margin_audit.audit_model(self.model, self.X_source, self.y_source, self.ids,
                                         self.X_query, self.y_query, 'balanced')
        estimator = self.model.estimator_
        points = estimator.support_vectors_
        distance = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        kernel = np.exp(-estimator._gamma * distance)
        beta = estimator.dual_coef_[0]
        norm_squared = float(beta @ kernel @ beta)
        self.assertAlmostEqual(audit['summary']['rkhs_norm_squared'], norm_squared, places=11)
        rows = pd.DataFrame(audit['samples'])
        primal = .5 * norm_squared + float((rows.C_i * rows.hinge_loss).sum())
        dual = float(rows.alpha.sum()) - .5 * norm_squared
        self.assertAlmostEqual(audit['summary']['primal_objective'], primal, places=11)
        self.assertAlmostEqual(audit['summary']['dual_objective'], dual, places=11)
        self.assertGreaterEqual(audit['summary']['primal_dual_gap'], -1e-9)

    def test_identical_model_is_zero_margin_and_zero_dual_change(self):
        audit = margin_audit.audit_model(self.model, self.X_source, self.y_source, self.ids,
                                         self.X_query, self.y_query, 'balanced')
        comparison = margin_audit.compare_models(self.model, self.model, audit, audit, self.y_query)
        self.assertTrue(comparison['query_labels_identical'])
        self.assertTrue(comparison['query_scores_exact_equal'])
        self.assertEqual(comparison['max_query_margin_difference'], 0)
        self.assertEqual(comparison['max_dual_coefficient_difference'], 0)
        self.assertEqual(comparison['wrong_to_correct'], 0)
        self.assertEqual(comparison['correct_to_wrong'], 0)
        self.assertEqual(comparison['query_auc_before'], comparison['query_auc_after'])
        self.assertEqual(len(margin_audit.expected_endpoints(self.job)), 4)

    def test_equal_dual_coefficients_with_shifted_intercept_are_not_same_solution(self):
        changed = copy.deepcopy(self.model)
        shift = .25
        # Binary sklearn/libsvm maintains public and internal opposite-sign
        # intercept arrays. Shift the actual decision function coherently.
        changed.estimator_.intercept_ = changed.estimator_.intercept_.copy() + shift
        changed.estimator_._intercept_ = changed.estimator_._intercept_.copy() - shift
        original_audit = margin_audit.audit_model(self.model, self.X_source, self.y_source, self.ids,
                                                   self.X_query, self.y_query, 'balanced')
        changed_audit = margin_audit.audit_model(changed, self.X_source, self.y_source, self.ids,
                                                  self.X_query, self.y_query, 'none')
        np.testing.assert_allclose(changed_audit['query_margin'] - original_audit['query_margin'],
                                   shift, rtol=0, atol=1e-12)
        comparison = margin_audit.compare_models(self.model, changed, original_audit,
                                                  changed_audit, self.y_query)
        self.assertTrue(comparison['kernel_coordinates_equal'])
        self.assertTrue(comparison['dual_coefficients_exact_equal'])
        self.assertTrue(comparison['both_weighted_optima_interior'])
        self.assertFalse(comparison['intercept_exact_equal'])
        self.assertFalse(comparison['inactive_bounds_same_solution'])
        self.assertGreater(comparison['max_query_margin_difference'], 0)


if __name__ == '__main__':
    unittest.main()
