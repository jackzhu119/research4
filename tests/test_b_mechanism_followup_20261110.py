"""Scientific isolation and recoverability checks using synthetic spectra only."""
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

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import joblib
import numpy as np
import pandas as pd

import b_followup_common_20261110 as common
import b_followup_calibration_20261110 as calibration
import b_followup_models_20261110 as models

OLDROOT = ROOT / 'reports/b_mechanism_20261110'
MANIFEST = ROOT / 'data/research/species_revision_20261108/split_manifest.csv'


def real_inputs():
    metadata = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    jobs = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    return metadata, jobs


def synthetic_fixture():
    rng = np.random.default_rng(20261110)
    rows, roles = [], {'source': [], 'support': [], 'query': []}
    for species, origin, n_source in (('枣仁', 'ZY01', 18), ('酸枣仁', 'ZY03', 6)):
        for role, count in (('source', n_source), ('support', 10), ('query', 3)):
            for number in range(count):
                identifier = f'{origin}-{role}-{number:03d}'
                rows.append({'record_id': identifier, 'physical_sample_id': identifier,
                             'origin_group_id': origin, 'species': species, 'role': 'train'})
                roles[role].append(identifier)
    metadata = pd.DataFrame(rows)
    X = rng.normal(size=(len(metadata), 12))
    pattern = np.linspace(-2, 2, 12)
    X[metadata.species.eq('酸枣仁')] += pattern
    X[metadata.record_id.str.contains('support')] += np.sin(np.arange(12)) * 3
    axis = np.linspace(.1, 1.5, 12)
    job = {'job_id': 'synthetic', 'family': 'F02', 'parent_job_id': 'parent',
           'experiment': 'E01', 'target_origin': 'ZY03', 'seed': 42,
           'coverage': 0, 'ratio': 'B', 'fold': None,
           'classes': ['枣仁', '酸枣仁'], 'label_column': 'species',
           **{name + '_ids': values for name, values in roles.items()}}
    return metadata, X, axis, job


def make_parent(root, metadata, X, axis, job):
    """Build a small self-contained old-style sealed fixture, never old data."""
    parent = {k: copy.deepcopy(v) for k, v in job.items() if k not in {'family', 'parent_job_id'}}
    parent['job_id'] = job['parent_job_id']
    directory = root / 'jobs' / parent['job_id']
    directory.mkdir(parents=True)
    frame = metadata.set_index('record_id')
    lookup = {value: index for index, value in enumerate(metadata.record_id)}
    take = lambda ids: X[[lookup[value] for value in ids]]
    model = models.fit_model(take(parent['source_ids']), frame.loc[parent['source_ids'], 'species'], axis)
    joblib.dump(model, directory / 'source_model.joblib')
    for k in models.K_VALUES:
        support = models.nested_support(parent, frame, k)
        fit_ids = parent['source_ids'] + support
        refit = models.fit_model(take(fit_ids), frame.loc[fit_ids, 'species'], axis)
        joblib.dump(refit, directory / f'retrain_k{k}.joblib')
    result = {**parent, 'input_hash': 'old-input', 'job_hash': common.canonical_hash(parent),
              'artifact_hashes': {p.name: common.file_hash(p) for p in directory.iterdir()}}
    common.write_json(directory / 'result.json', result)
    common.write_json(directory / 'COMPLETE.json', {'input_hash': 'old-input',
        'job_hash': result['job_hash'], 'result_sha256': common.file_hash(directory / 'result.json')})
    common.write_json(root / 'frozen_protocol.json', {'input_hash': 'old-input', 'design': [parent]})
    return directory


class FrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.old_jobs = real_inputs()
        cls.indexed = cls.metadata.set_index('record_id', drop=False)

    def test_current_metadata_identity_and_species(self):
        self.assertEqual(len(self.metadata), 1672)
        self.assertTrue(self.metadata.record_id.is_unique)
        self.assertTrue(self.metadata.physical_sample_id.is_unique)
        for group in ('ZY01', 'ZY02'):
            self.assertTrue(self.metadata.loc[self.metadata.origin_group_id.eq(group), 'species'].eq('枣仁').all())
        self.assertEqual(self.metadata.role.eq('train').sum(), 999)

    def test_full_frozen_design_counts_and_fit_accounting(self):
        runner = importlib.import_module('run_b_mechanism_followup_20261110')
        jobs = runner.build_design(self.metadata, self.old_jobs)
        self.assertEqual(len(jobs), 345)
        self.assertEqual(pd.Series([j['family'] for j in jobs]).value_counts().to_dict(),
                         {'F03': 240, 'F01': 45, 'F02': 45, 'F04': 15})
        self.assertEqual(len({j['job_id'] for j in jobs}), 345)
        self.assertEqual(sum(models.fit_count(j) for j in jobs if j['family'] != 'F01'), 723)
        self.assertEqual(sum(270 for j in jobs if j['family'] == 'F01'), 12150)
        for job in jobs:
            common.validate_roles(job, self.metadata)

    def test_old_seed_factorial_roles_are_exactly_reused(self):
        runner = importlib.import_module('run_b_mechanism_followup_20261110')
        jobs = runner.build_design(self.metadata, self.old_jobs)
        old = {job['job_id']: job for job in self.old_jobs}
        reused = [j for j in jobs if j['family'] == 'F03' and j.get('parent_job_id')]
        self.assertEqual(len(reused), 90)
        for job in reused:
            previous = old[job['parent_job_id']]
            for role in ('source', 'support', 'query'):
                self.assertEqual(job[role + '_ids'], previous[role + '_ids'])
            self.assertEqual(len(job['source_ids']), 240)
            acid, zaoren = {'A': (60, 180), 'B': (120, 120), 'C': (180, 60)}[job['ratio']]
            actual = self.indexed.loc[job['source_ids'], 'species'].value_counts().to_dict()
            self.assertEqual(actual, {'酸枣仁': acid, '枣仁': zaoren})
            self.assertEqual(self.indexed.loc[job['source_ids'], 'origin_group_id'].eq(job['target_origin']).sum(),
                             20 * job['coverage'])

    def test_roles_reject_overlap_duplicate_and_nontrain(self):
        metadata, _, _, job = synthetic_fixture()
        for change in ('overlap', 'duplicate', 'nontrain', 'kind'):
            frame, altered = metadata.copy(), copy.deepcopy(job)
            if change == 'overlap':
                altered['query_ids'][0] = altered['source_ids'][0]
            elif change == 'duplicate':
                frame.loc[1, 'physical_sample_id'] = frame.loc[0, 'physical_sample_id']
            elif change == 'nontrain':
                frame.loc[frame.record_id.eq(altered['query_ids'][0]), 'role'] = 'prediction'
            else:
                frame.loc[frame.origin_group_id.eq('ZY03'), 'species'] = '枣仁'
            with self.subTest(change=change), self.assertRaises(ValueError):
                common.validate_roles(altered, frame)

    def test_nested_support_is_deterministic_and_k10_is_one_pool(self):
        job = next(j for j in self.old_jobs if j['experiment'] == 'E03')
        for draw in calibration.DRAW_VALUES:
            prior = set()
            for k in (1, 3, 5, 10):
                ids = calibration.support_plan(job, self.metadata, draw, k)
                self.assertEqual(ids, calibration.support_plan(job, self.metadata, draw, k))
                self.assertTrue(prior.issubset(ids))
                self.assertEqual(len(ids), k * 5)
                self.assertTrue(set(ids).issubset(job['support_ids']))
                prior = set(ids)
            self.assertEqual(prior, set(job['support_ids']))
        endpoints = calibration.expected_endpoints(job)
        self.assertEqual(len(endpoints), 284)
        self.assertEqual(len(set(endpoints)), 284)
        for method in calibration.TRUE_METHODS:
            self.assertIn(common.endpoint_id(method, 10, 0), endpoints)
            for draw in calibration.DRAW_VALUES[1:]:
                self.assertNotIn(common.endpoint_id(method, 10, draw), endpoints)

    def test_learning_curve_ids_nested_balanced_and_reuses_endpoints(self):
        job = next(j for j in self.old_jobs if j['experiment'] == 'E03')
        prior = set()
        for budget in models.LEARNING_BUDGETS:
            ids = models.learning_ids(job, self.indexed, budget)
            self.assertEqual(len(ids), budget)
            self.assertEqual(len(set(ids)), budget)
            self.assertTrue(prior.issubset(ids))
            self.assertFalse(set(ids) & set(job['query_ids']))
            self.assertTrue(self.indexed.loc[ids].role.eq('train').all())
            self.assertTrue(self.indexed.loc[ids].origin_group_id.value_counts().eq(budget // 5).all())
            prior = set(ids)
        self.assertEqual(models.learning_ids(job, self.indexed, 190), job['source_ids'])
        self.assertEqual(set(models.learning_ids(job, self.indexed, 240)),
                         set(job['source_ids'] + job['support_ids']))
        self.assertEqual(models.fit_count({**job, 'family': 'F04'}), 3)


class CalibrationIsolationTests(unittest.TestCase):
    def test_score_unit_and_source_rms_transform_preserve_boundary(self):
        rng = np.random.default_rng(99)
        source, query = rng.normal(size=(31, 5)), rng.normal(size=(27, 5))
        transform, rms, effective = calibration.rowcenter_source_rms(source)
        self.assertGreater(rms, 0)
        self.assertEqual(rms, effective)
        self.assertTrue(np.array_equal(query.argmax(axis=1), (query @ transform).argmax(axis=1)))
        for scale in (.1, 10):
            self.assertTrue(np.array_equal(query.argmax(axis=1), (query * scale).argmax(axis=1)))
        query_shifted = query + rng.normal(size=(len(query), 1)) * 20
        np.testing.assert_allclose(query @ transform, query_shifted @ transform, atol=1e-12)

    def test_degenerate_source_rms_is_finite_without_query_fit(self):
        transform, rms, effective = calibration.rowcenter_source_rms(np.zeros((8, 5)))
        self.assertEqual(rms, 0)
        self.assertEqual(effective, 1)
        self.assertTrue(np.isfinite(transform).all())
        self.assertEqual(list(inspect.signature(calibration.rowcenter_source_rms).parameters), ['source_scores'])

    def test_shuffled_diagnostic_is_copy_with_unchanged_histogram(self):
        labels = np.array(['A'] * 4 + ['B'] * 4 + ['C'] * 4)
        original = labels.copy()
        shuffled = calibration.diagnostic_labels(labels, 101, 3)
        np.testing.assert_array_equal(labels, original)
        self.assertEqual(sorted(shuffled), sorted(labels))
        np.testing.assert_array_equal(shuffled, calibration.diagnostic_labels(labels, 101, 3))
        self.assertFalse(np.shares_memory(shuffled, labels))

    def test_diagonal_calibrator_has_no_cross_class_terms(self):
        rng = np.random.default_rng(8)
        scores = rng.normal(size=(15, 5))
        labels = np.tile(['a', 'b', 'c', 'd', 'e'], 3)
        adapter = calibration.fit_diagonal_anchor(scores, labels, ['a', 'b', 'c', 'd', 'e'])
        coefficient = np.asarray(adapter['coef'])
        np.testing.assert_array_equal(coefficient - np.diag(np.diag(coefficient)), np.zeros((5, 5)))
        self.assertTrue(np.isfinite(coefficient).all())
        parameters = inspect.signature(calibration.fit_score_method).parameters
        self.assertNotIn('query_labels', parameters)
        self.assertNotIn('y_query', parameters)


class ModelFactorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.X, cls.axis, cls.job = synthetic_fixture()
        cls.frame = cls.metadata.set_index('record_id')
        cls.lookup = {value: index for index, value in enumerate(cls.metadata.record_id)}
        cls.source_ids = cls.job['source_ids']
        cls.source = models.fit_model(cls.take(cls.source_ids), cls.frame.loc[cls.source_ids, 'species'], cls.axis)

    @classmethod
    def take(cls, ids):
        return cls.X[[cls.lookup[value] for value in ids]]

    def test_class_weights_are_actual_svc_weights(self):
        y = self.frame.loc[self.source_ids, 'species']
        plain = models.fit_model(self.take(self.source_ids), y, self.axis, class_weight=None)
        np.testing.assert_allclose(plain.estimator_.class_weight_, [1, 1])
        expected = [len(y) / (2 * int((y == cls).sum())) for cls in self.source.classes_]
        np.testing.assert_allclose(self.source.estimator_.class_weight_, expected)
        self.assertEqual(self.source.estimator_.C, 10)
        self.assertIsNone(plain.estimator_.class_weight)

    def test_scaler_gamma_factorial_freezes_numeric_gamma_independently(self):
        original = joblib.hash(self.source)
        ids = self.source_ids + models.nested_support(self.job, self.frame, 10)
        for _, scaler_mode, gamma_mode in models.SCALER_GAMMA_METHODS:
            fitted = models.fit_model(self.take(ids), self.frame.loc[ids, 'species'], self.axis,
                                      reference=self.source, scaler_mode=scaler_mode, gamma_mode=gamma_mode)
            with self.subTest(scaler=scaler_mode, gamma=gamma_mode):
                if scaler_mode == 'frozen':
                    self.assertEqual(joblib.hash(fitted.scaler_), joblib.hash(self.source.scaler_))
                else:
                    self.assertNotEqual(joblib.hash(fitted.scaler_), joblib.hash(self.source.scaler_))
                if gamma_mode == 'fixed':
                    self.assertEqual(fitted.estimator_._gamma, self.source.estimator_._gamma)
                else:
                    values = fitted._scaled(self.take(ids))
                    self.assertAlmostEqual(fitted.estimator_._gamma, 1 / (values.shape[1] * values.var()))
        self.assertEqual(joblib.hash(self.source), original)

    def test_complete_factorial_fits_only_source_plus_fixed_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            make_parent(path / 'old', self.metadata, self.X, self.axis, self.job)
            job = copy.deepcopy(self.job)
            job['exact_gamma_reuse_k'] = models.gamma_reuse_plan(job, path / 'old')
            real_fit = models.fit_model
            seen = []

            def inspected(X_train, y_train, axis, **kwargs):
                seen.append((np.array(X_train, copy=True), np.array(y_train, copy=True), kwargs))
                return real_fit(X_train, y_train, axis, **kwargs)

            with mock.patch.object(models, 'fit_model', side_effect=inspected):
                result = models.execute(job, self.X, self.metadata, self.axis,
                                        path / 'new', 'new-input', path / 'old')
            expected = [(k, scaler, gamma) for k in models.K_VALUES
                        for _, scaler, gamma in models.SCALER_GAMMA_METHODS
                        if not (scaler == 'refit' and (gamma == 'scale' or k in job['exact_gamma_reuse_k']))]
            self.assertEqual(result['new_model_fits'], len(expected))
            self.assertEqual(len(seen), len(expected))
            self.assertEqual(models.fit_count(job), len(expected))
            for (X_train, y_train, options), (k, scaler, gamma) in zip(seen, expected):
                ids = self.source_ids + models.nested_support(self.job, self.frame, k)
                np.testing.assert_array_equal(X_train, self.take(ids))
                np.testing.assert_array_equal(y_train, self.frame.loc[ids, 'species'])
                self.assertEqual(options['scaler_mode'], scaler)
                self.assertEqual(options['gamma_mode'], gamma)
                self.assertFalse(set(ids) & set(self.job['query_ids']))
            receipt = common.verify_task(path / 'new/jobs' / job['job_id'], 'new-input', job,
                                          models.expected_endpoints(job))
            self.assertEqual(receipt['metric_rows'], 17)

    def test_gamma_reuse_is_exact_parameter_equality_without_query_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / 'old'
            parent = make_parent(old, self.metadata, self.X, self.axis, self.job)
            reuse = models.gamma_reuse_plan(self.job, old)
            source = joblib.load(parent / 'source_model.joblib')
            exact = [k for k in models.K_VALUES if
                     joblib.load(parent / f'retrain_k{k}.joblib').estimator_._gamma == source.estimator_._gamma]
            self.assertEqual(reuse, exact)
            self.assertNotIn('query', ' '.join(inspect.signature(models.gamma_reuse_plan).parameters))

    def test_equal_class_weights_reuse_old_model_for_none_without_new_fit(self):
        metadata, X, axis, job = synthetic_fixture()
        job['family'] = 'F03'
        job['ratio'] = 'B'
        indexed = metadata.set_index('record_id')
        job['source_ids'] = (indexed.loc[job['source_ids']].query('species == "枣仁"').index[:6].tolist()
                             + indexed.loc[job['source_ids']].query('species == "酸枣仁"').index.tolist())
        self.assertTrue(indexed.loc[job['source_ids'], 'species'].value_counts().eq(6).all())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            parent = make_parent(path / 'old', metadata, X, axis, job)
            source = joblib.load(parent / 'source_model.joblib')
            np.testing.assert_array_equal(source.estimator_.class_weight_, [1., 1.])
            with mock.patch.object(models, 'fit_model', side_effect=AssertionError('duplicate fit')):
                result = models.execute(job, X, metadata, axis, path / 'new', 'new-input', path / 'old')
            self.assertEqual(result['new_model_fits'], 0)
            self.assertEqual(models.fit_count(job), 0)
            predictions = pd.read_csv(path / 'new/jobs' / job['job_id'] / 'predictions.csv.gz')
            balanced = predictions.loc[predictions.method.eq('class_weight_balanced')].sort_values('record_id')
            none = predictions.loc[predictions.method.eq('class_weight_none')].sort_values('record_id')
            np.testing.assert_array_equal(balanced.pred_label, none.pred_label)
            self.assertEqual(balanced.scores_json.tolist(), none.scores_json.tolist())


class AtomicArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.metadata, self.X, self.axis, self.job = synthetic_fixture()

    def sealed(self):
        writer = common.TaskWriter(self.job, self.X, self.metadata, self.axis, self.path / 'new', 'input', self.path / 'old')
        labels = writer.labels(self.job['query_ids'])
        prediction = labels.copy()
        prediction[[1, 4]] = np.where(prediction[[1, 4]] == self.job['classes'][0],
                                     self.job['classes'][1], self.job['classes'][0])
        scores = np.asarray([[2., -1.] if label == self.job['classes'][0] else [-1., 2.] for label in prediction])
        identifier = writer.record('identity', 0, scores)
        result = writer.finish([identifier], new_model_fits=0)
        return self.path / 'new/jobs' / self.job['job_id'], result, identifier

    def test_saved_metrics_independently_match_hand_confusion_matrix(self):
        path, result, endpoint = self.sealed()
        common.verify_task(path, 'input', self.job, [endpoint])
        metric = pd.read_csv(path / 'metrics.csv').iloc[0]
        self.assertAlmostEqual(metric.accuracy, 2 / 3)
        self.assertAlmostEqual(metric.macro_f1, 2 / 3)
        self.assertAlmostEqual(metric.balanced_accuracy, 2 / 3)
        self.assertEqual(json.loads(metric.confusion_matrix_json), [[2, 1], [1, 2]])
        self.assertEqual(json.loads(metric.recall_json), {'枣仁': 2 / 3, '酸枣仁': 2 / 3})
        self.assertEqual(result['new_model_fits'], 0)

    def test_changed_input_or_frozen_job_rejects_recovery(self):
        path, _, endpoint = self.sealed()
        with self.assertRaisesRegex(ValueError, 'input/code'):
            common.verify_task(path, 'different', self.job, [endpoint])
        altered = copy.deepcopy(self.job)
        altered['source_ids'] = list(reversed(altered['source_ids']))
        with self.assertRaisesRegex(ValueError, 'frozen design'):
            common.verify_task(path, 'input', altered, [endpoint])

    def test_resume_verification_never_fits_or_overwrites_complete_artifact(self):
        path, result, endpoint = self.sealed()
        before = {p.name: common.file_hash(p) for p in path.iterdir()}
        with mock.patch.object(models, 'fit_model', side_effect=AssertionError('unexpected fitting')):
            first = common.verify_task(path, 'input', self.job, [endpoint])
            second = common.verify_task(path, 'input', self.job, [endpoint])
        self.assertEqual(first['cpu_seconds'], second['cpu_seconds'])
        self.assertEqual(first['cpu_seconds'], result['cpu_seconds'])
        self.assertEqual(before, {p.name: common.file_hash(p) for p in path.iterdir()})
        with self.assertRaises(FileExistsError):
            common.TaskWriter(self.job, self.X, self.metadata, self.axis, self.path / 'new', 'input', self.path / 'old')

    def test_incomplete_endpoint_matrix_never_becomes_sealed(self):
        writer = common.TaskWriter(self.job, self.X, self.metadata, self.axis, self.path / 'new', 'input', self.path / 'old')
        scores = np.tile([1., 0.], (len(self.job['query_ids']), 1))
        identifier = writer.record('identity', 0, scores)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            writer.finish([identifier, common.endpoint_id('missing')])
        self.assertFalse(writer.destination.exists())

    def test_changed_prediction_bytes_are_rejected(self):
        path, _, endpoint = self.sealed()
        with (path / 'predictions.csv.gz').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            common.verify_task(path, 'input', self.job, [endpoint])

    def test_self_consistent_byte_seal_does_not_hide_wrong_metrics(self):
        path, result, endpoint = self.sealed()
        metric = pd.read_csv(path / 'metrics.csv')
        metric.loc[0, 'accuracy'] = .99
        metric.to_csv(path / 'metrics.csv', index=False)
        result['artifact_hashes']['metrics.csv'] = common.file_hash(path / 'metrics.csv')
        common.write_json(path / 'result.json', result)
        seal = json.loads((path / 'COMPLETE.json').read_text())
        seal['result_sha256'] = common.file_hash(path / 'result.json')
        common.write_json(path / 'COMPLETE.json', seal)
        with self.assertRaisesRegex(ValueError, 'Independent metric mismatch'):
            common.verify_task(path, 'input', self.job, [endpoint])

    def test_nonfinite_scores_and_outside_support_are_rejected(self):
        writer = common.TaskWriter(self.job, self.X, self.metadata, self.axis, self.path / 'new', 'input', self.path / 'old')
        scores = np.tile([1., 0.], (len(self.job['query_ids']), 1))
        scores[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'query scores'):
            writer.record('bad', 0, scores)
        scores[0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'outside'):
            writer.record('bad', 1, scores, support_ids=self.job['query_ids'][:1])

    def test_historical_parent_hash_and_sample_identity_checked_before_reuse(self):
        old = self.path / 'old'
        directory = make_parent(old, self.metadata, self.X, self.axis, self.job)
        common.load_parent(self.job, old)
        altered = copy.deepcopy(self.job)
        altered['query_ids'] = list(reversed(altered['query_ids']))
        with self.assertRaisesRegex(ValueError, 'definition differs'):
            common.load_parent(altered, old)
        with (directory / 'source_model.joblib').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            common.load_parent(self.job, old)


if __name__ == '__main__':
    unittest.main()
