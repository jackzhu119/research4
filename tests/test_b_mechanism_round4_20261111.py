"""Fourth-round scientific invariants, using synthetic fits and fixed ID metadata."""
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import joblib
import numpy as np
import pandas as pd

from herb_spectra.calibration import AnchorScoreCalibrator
import b_followup_common_20261110 as artifacts
import b_followup_models_20261110 as prior_models
import b_followup_calibration_20261110 as prior_calibration
import b_mechanism_engine_20261110 as prior_engine
import b_round4_common_20261111 as common
import b_round4_models_20261111 as models
import b_round4_support_20261111 as support
import b_round4_weak_20261111 as weak

OLDROOT = ROOT / 'reports/b_mechanism_20261110'
ROUND2ROOT = ROOT / 'reports/b_mechanism_followup_20261110'
ROUND3ROOT = ROOT / 'reports/b_mechanism_round3_20261110'


def real_inputs():
    metadata = pd.read_csv(ROOT / 'data/research/species_revision_20261108/split_manifest.csv',
                           dtype=str, keep_default_na=False)
    old = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    phase2 = json.loads((ROUND2ROOT / 'frozen_protocol.json').read_text())['design']
    return metadata, old, phase2


def synthetic_fixture(family='H01'):
    rng = np.random.default_rng(11592)
    rows, roles = [], {role: [] for role in ('source', 'support', 'query')}
    for species, origin in (('枣仁', 'ZY01'), ('酸枣仁', 'ZY03')):
        for role, number in (('source', 9), ('support', 10), ('query', 3)):
            for i in range(number):
                sample = f'{origin}-{role}-{i:03d}'
                rows.append({'record_id': sample, 'physical_sample_id': sample,
                             'origin_group_id': origin, 'species': species, 'role': 'train'})
                roles[role].append(sample)
    metadata = pd.DataFrame(rows)
    X = rng.normal(size=(len(rows), 10))
    X[metadata.species.eq('酸枣仁')] += np.linspace(-2, 2, X.shape[1])
    X[metadata.record_id.str.contains('support')] += np.sin(np.arange(10)) * 2
    X[metadata.record_id.str.contains('query')] += np.cos(np.arange(10)) * 9
    axis = np.linspace(.1, 1.5, X.shape[1])
    job = {'job_id': family + '_synthetic', 'family': family,
           'parent_job_id': 'synthetic_parent', 'phase2_job_id': 'F02_synthetic_parent',
           'experiment': 'E01', 'target_origin': 'ZY03', 'seed': 42,
           'ratio': 'B', 'coverage': 0, 'fold': None,
           'classes': ['枣仁', '酸枣仁'], 'label_column': 'species',
           **{role + '_ids': values for role, values in roles.items()}}
    return metadata, X, axis, job


def take(metadata, X, ids):
    lookup = {sample: i for i, sample in enumerate(metadata.record_id)}
    return X[[lookup[sample] for sample in ids]]


class FrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.old, cls.phase2 = real_inputs()
        cls.indexed = cls.metadata.set_index('record_id', drop=False)

    def test_all375_jobs_use_current_kind_and_mutually_exclusive_train_ids(self):
        runner = importlib.import_module('run_b_mechanism_round4_20261111')
        jobs = runner.build_design(self.metadata, self.old, self.phase2)
        self.assertEqual(len(jobs), 375)
        self.assertEqual(pd.Series([job['family'] for job in jobs]).value_counts().to_dict(),
                         {'H02': 240, 'H01': 45, 'H03': 45, 'H04': 45})
        self.assertEqual(len({job['job_id'] for job in jobs}), 375)
        parents = {job['job_id']: job for job in self.old}
        phase2 = {job['job_id']: job for job in self.phase2}
        for job in jobs:
            artifacts.validate_roles(job, self.metadata)
            parent = phase2[job['phase2_job_id']] if job['family'] == 'H02' else parents[job['parent_job_id']]
            for role in ('source', 'support', 'query'):
                self.assertEqual(job[role + '_ids'], parent[role + '_ids'])
            ids = job['source_ids'] + job['support_ids'] + job['query_ids']
            self.assertTrue(self.indexed.loc[ids, 'role'].eq('train').all())
        self.assertEqual(sum(models.fit_count(job) for job in jobs if job['family'] in ('H01', 'H02')), 1160)
        self.assertEqual(sum(support.adapter_count(job) for job in jobs if job['family'] == 'H04'), 2340)

    def test_h02_equivalent_weight_conditions_are_reused_at_all_fixed_cs(self):
        runner = importlib.import_module('run_b_mechanism_round4_20261111')
        jobs = runner.build_design(self.metadata, self.old, self.phase2)
        cases = [job for job in jobs if job['family'] == 'H02']
        self.assertEqual(len(cases), 240)
        self.assertEqual(sum(models.fit_count(job) for job in cases), 800)
        for job in cases:
            self.assertEqual(models.fit_count(job), 2 if job['ratio'] == 'B' else 4)
            self.assertEqual(len(models.expected_endpoints(job)), 12)
        selected = [job for job in jobs if job['family'] == 'H01']
        self.assertEqual(sum(models.fit_count(job) for job in selected), 360)
        self.assertTrue(all(len(models.expected_endpoints(job)) == 13 for job in selected))

    def test_weak_audit_reuses34_support_cases_and16_noisy_cases_without_fits(self):
        job = self.old[0]
        self.assertEqual(len(weak.expected_endpoints(job)), 52)
        self.assertEqual(len(set(weak.expected_endpoints(job))), 52)
        self.assertEqual(weak.fit_count(job), 0)
        self.assertEqual(weak.adapter_count(job), 0)

    def test_invalid_physical_overlap_heldout_and_kind_are_rejected(self):
        metadata, _, _, job = synthetic_fixture()
        for case in ('role_overlap', 'duplicate_physical', 'heldout', 'kind'):
            altered, frame = copy.deepcopy(job), metadata.copy()
            if case == 'role_overlap':
                altered['support_ids'][0] = altered['query_ids'][0]
            elif case == 'duplicate_physical':
                frame.loc[1, 'physical_sample_id'] = frame.loc[0, 'physical_sample_id']
            elif case == 'heldout':
                frame.loc[frame.record_id.eq(altered['source_ids'][0]), 'role'] = 'prediction'
            else:
                frame.loc[frame.origin_group_id.eq('ZY03'), 'species'] = '枣仁'
            with self.subTest(case=case), self.assertRaises(ValueError):
                artifacts.validate_roles(altered, frame)


class WeakCrossingTests(unittest.TestCase):
    def test_known_two_class_crossing_matches_closed_form_without_labels(self):
        old = np.asarray([[2., 0.], [.1, 0.]])
        new = np.asarray([[1.95, .1], [0., .2]])
        value = weak.crossing_diagnostics(old, new)
        np.testing.assert_allclose(value['first_crossing_blend'], [2 / 3, 1 / 60], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(value['decision_changed'], [False, True])
        np.testing.assert_array_equal(value['strict_margin_certificate'], [True, False])
        self.assertFalse(np.asarray(value['certificate_violation']).any())
        self.assertTrue(np.asarray(value['path_prediction_consistent']).all())
        self.assertFalse(any('label' in parameter.lower() or 'truth' in parameter.lower()
                             for parameter in inspect.signature(weak.crossing_diagnostics).parameters))

    def test_common_translation_moves_scores_but_cannot_change_argmax(self):
        old = np.asarray([[3., 1., -1.], [0., 2., 1.]])
        value = weak.crossing_diagnostics(old, old + 100)
        self.assertTrue(np.asarray(value['score_changed']).all())
        self.assertTrue(np.asarray(value['score_moved_decision_unchanged']).all())
        self.assertFalse(np.asarray(value['decision_changed']).any())
        self.assertTrue(np.isinf(value['first_crossing_blend']).all())
        # The gap bound is sufficient, not necessary: common-mode changes can
        # be much larger than the margin while leaving every difference fixed.
        self.assertFalse(np.asarray(value['strict_margin_certificate']).any())

    def test_first_class_tie_rule_and_crossing_boundary_are_explicit(self):
        old = np.asarray([[1., 1., 0.], [0., 1., -3.]])
        new = np.asarray([[1., 1.1, 0.], [.1, .95, -3.]])
        value = weak.crossing_diagnostics(old, new)
        self.assertEqual(int(value['source_winner_index'][0]), 0)
        self.assertEqual(float(value['first_crossing_blend'][0]), 0)
        self.assertFalse(bool(value['crossing_boundary_switches_at_tie'][0]))
        self.assertTrue(bool(value['crossing_boundary_switches_at_tie'][1]))
        self.assertAlmostEqual(value['first_crossing_blend'][1], 1 / 3)
        self.assertTrue(np.asarray(value['path_prediction_consistent']).all())

    def test_strict_margin_bound_never_certifies_a_changed_decision(self):
        rng = np.random.default_rng(314)
        old = rng.normal(size=(100, 5))
        new = old + rng.normal(scale=.03, size=old.shape)
        value = weak.crossing_diagnostics(old, new)
        certified = np.asarray(value['strict_margin_certificate'])
        self.assertTrue(certified.any())
        self.assertFalse(np.asarray(value['decision_changed'])[certified].any())
        self.assertFalse(np.asarray(value['certificate_violation']).any())
        identity = weak.crossing_diagnostics(old, old.copy())
        self.assertFalse(np.asarray(identity['score_changed']).any())
        self.assertTrue(np.isinf(identity['first_crossing_blend']).all())


class SupportCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.old, _ = real_inputs()
        cls.binary = next(job for job in cls.old if job['experiment'] == 'E01' and job['ratio'] == 'B')
        cls.pure = next(job for job in cls.old if job['experiment'] == 'E03')
        cls.indexed = cls.metadata.set_index('record_id', drop=False)

    def test_fixed_total_compositions_use_only_locked_support_pool(self):
        for job, count, total in ((self.binary, 5, 10), (self.pure, 6, 25)):
            plan = support.composition_plan(job, self.metadata)
            self.assertEqual(len(plan['episodes']), count * 3)
            self.assertEqual(plan['support_total'], total)
            self.assertEqual(plan, support.composition_plan(job, self.metadata))
            for episode in plan['episodes']:
                ids = episode['support_ids']
                self.assertEqual(len(ids), total)
                self.assertEqual(len(set(ids)), total)
                self.assertTrue(set(ids).issubset(job['support_ids']))
                self.assertFalse(set(ids) & set(job['query_ids']))
                self.assertFalse(set(ids) & set(job['source_ids']))
                actual = self.indexed.loc[ids, job['label_column']].value_counts()
                self.assertEqual([int(actual[label]) for label in job['classes']], episode['counts'])
            self.assertEqual(len(support.expected_endpoints(job)), 62 if total == 10 else 74)
            self.assertEqual(support.adapter_count(job), 48 if total == 10 else 60)

    def test_balanced_cells_match_historical_k5_ids_and_weights_exactly(self):
        for job in (self.binary, self.pure):
            plan = support.composition_plan(job, self.metadata)
            for episode in plan['episodes']:
                if not episode['balanced_composition']:
                    continue
                prior = prior_calibration.support_plan(job, self.metadata, episode['draw_seed'], 5)
                self.assertEqual(episode['support_ids'], prior)
                labels = self.indexed.loc[prior, job['label_column']].to_numpy()
                np.testing.assert_array_equal(support.class_balanced_weights(labels, job['classes']), 1.)

    def test_composition_plans_do_not_use_query_labels_or_query_spectra(self):
        original = support.composition_plan(self.pure, self.metadata)
        changed = self.metadata.copy()
        changed.loc[changed.record_id.isin(self.pure['query_ids']), 'origin_group_id'] = 'unread-query-marker'
        self.assertEqual(original, support.composition_plan(self.pure, changed))
        self.assertNotIn('X', inspect.signature(support.composition_plan).parameters)

    def test_class_balance_weights_preserve_total_mass_and_equalize_class_mass(self):
        classes = ['a', 'b', 'c']
        labels = np.asarray(['a', 'b', 'b', 'c', 'c', 'c'])
        weights = support.class_balanced_weights(labels, classes)
        self.assertAlmostEqual(weights.sum(), len(labels))
        for label in classes:
            self.assertAlmostEqual(weights[labels == label].sum(), len(labels) / len(classes))
        np.testing.assert_allclose(weights, [2., 1., 1., 2 / 3, 2 / 3, 2 / 3])

    def test_weighted_anchor_solves_independent_normal_equations(self):
        classes = ['a', 'b', 'c']
        y = np.asarray(['a', 'b', 'b', 'c', 'c', 'c'])
        scores = np.random.default_rng(91).normal(size=(len(y), len(classes)))
        adapter = support.fit_residual_adapter(scores, y, classes, kind='anchor', class_balanced=True)
        design = np.column_stack([scores, np.ones(len(y))])
        residual_target = np.eye(3)[[classes.index(label) for label in y]] - scores
        delta = np.vstack([np.asarray(adapter['coef']) - np.eye(3), adapter['intercept']])
        weights = np.asarray([2., 1., 1., 2 / 3, 2 / 3, 2 / 3])
        gradient = design.T @ (weights[:, None] * (design @ delta - residual_target)) + delta
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-12)
        self.assertFalse(any('query' in parameter.lower()
                             for parameter in inspect.signature(support.fit_residual_adapter).parameters))

    def test_weighted_bias_matches_penalized_weighted_residual_sum(self):
        classes = ['a', 'b']
        y = np.asarray(['a'] + ['b'] * 9)
        scores = np.random.default_rng(92).normal(size=(10, 2))
        adapter = support.fit_residual_adapter(scores, y, classes, kind='bias', class_balanced=True)
        weights = np.where(y == 'a', 5., 5 / 9)
        target = np.eye(2)[[classes.index(label) for label in y]]
        expected = (weights[:, None] * (target - scores)).sum(axis=0) / (len(y) + 1)
        np.testing.assert_allclose(adapter['coef'], np.eye(2), rtol=0, atol=0)
        np.testing.assert_allclose(adapter['intercept'], expected, rtol=0, atol=1e-13)

    def test_balanced_weighted_fit_is_existing_anchor_not_a_new_objective(self):
        classes = ['a', 'b', 'c']
        labels = np.tile(classes, 3)
        scores = np.random.default_rng(95).normal(size=(len(labels), len(classes)))
        adapted = support.fit_residual_adapter(scores, labels, classes, kind='anchor', class_balanced=True)
        prior = AnchorScoreCalibrator(classes, regularization=1).fit(scores, labels)
        np.testing.assert_allclose(adapted['coef'], prior.coef_, rtol=0, atol=1e-13)
        np.testing.assert_allclose(adapted['intercept'], prior.intercept_, rtol=0, atol=1e-13)

    def test_balanced_h04_endpoints_reuse_sealed_k5_without_any_adapter_refit(self):
        metadata, X, axis, job = synthetic_fixture('H04')
        indexed = metadata.set_index('record_id', drop=False)
        source = prior_models.fit_model(take(metadata, X, job['source_ids']),
                                        indexed.loc[job['source_ids'], 'species'], axis)
        query_scores = source.predict_scores(take(metadata, X, job['query_ids']))
        rows, parameters = [], []
        for draw in (0, 101, 102):
            ids = prior_calibration.support_plan(job, metadata, draw, 5)
            scores = source.predict_scores(take(metadata, X, ids))
            labels = indexed.loc[ids, 'species'].to_numpy()
            for raw, old_method in (('raw_anchor_lambda1', 'anchor_lambda1'), ('raw_bias_only', 'bias_only')):
                adapted, adapter = prior_engine.calibrate_scores(old_method, scores, labels,
                                                                 query_scores, job['classes'])
                endpoint = artifacts.endpoint_id(raw, 5, draw)
                parameters.append({'endpoint_id': endpoint, 'support_ids': ids,
                                   'fit_labels_diagnostic_copy': labels.tolist(),
                                   'diagnostic_label_copy': False, 'adapter': adapter})
                for i, sample in enumerate(job['query_ids']):
                    rows.append({'endpoint_id': endpoint, 'record_id': sample,
                                 'scores_json': json.dumps(adapted[i].tolist()),
                                 'pred_label': job['classes'][adapted[i].argmax()]})
        calls, original_fit = [], support.fit_residual_adapter

        def unbalanced_only(scores, y, classes, **kwargs):
            counts = np.unique(y, return_counts=True)[1]
            self.assertFalse(np.all(counts == counts[0]), 'a balanced K5 cell was needlessly refitted')
            calls.append((len(y), kwargs.get('kind'), kwargs.get('class_balanced')))
            return original_fit(scores, y, classes, **kwargs)

        frame_before = metadata.copy(deep=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / 'phase2'
            parent.mkdir()
            pd.DataFrame(rows).to_csv(parent / 'predictions.csv.gz', index=False)
            artifacts.write_json(parent / 'adapter_parameters.json', parameters)
            artifacts.write_json(parent / 'result.json', {'fixture': True})
            phase2 = {**job, 'artifact_hashes': {name: artifacts.file_hash(parent / name)
                       for name in ('predictions.csv.gz', 'adapter_parameters.json')}}
            with mock.patch.object(support, 'load_parent', return_value=({'source_hash': 'fixture'}, source, parent)), \
                 mock.patch.object(common, 'load_round2_bundle', return_value=(phase2, parent)), \
                 mock.patch.object(support, 'fit_residual_adapter', side_effect=unbalanced_only):
                result = support.execute(job, X, metadata, axis, root / 'new', 'input',
                                          root / 'old', parent, root / 'phase3')
            self.assertEqual(len(calls), 48)
            self.assertEqual(result['new_adapter_fits'], 48)
            self.assertEqual(result['new_model_fits'], 0)
            self.assertEqual(result['unique_historical_adapters_reused'], 6)
            path = root / 'new' / 'jobs' / job['job_id']
            artifacts.verify_task(path, 'input', job, support.expected_endpoints(job))
            measured = pd.read_csv(path / 'metrics.csv')
            adapted = measured.loc[measured.method.isin(support.METHODS)]
            self.assertTrue(adapted.k.eq(-1).all())
            self.assertTrue(adapted.total_support_labels.eq(10).all())
            self.assertTrue(adapted.k_per_class_not_applicable.all())
            balanced = adapted.loc[adapted.balanced_composition.eq(True)]
            self.assertTrue(balanced.adapter_reused.all())
            self.assertFalse(balanced.new_adapter_fit.any())
        pd.testing.assert_frame_equal(metadata, frame_before)


class LossMassAndCFactorTests(unittest.TestCase):
    def setUp(self):
        self.metadata, self.X, self.axis, self.job = synthetic_fixture()
        self.indexed = self.metadata.set_index('record_id', drop=False)
        self.source_ids = self.job['source_ids']
        self.source_X = take(self.metadata, self.X, self.source_ids)
        self.source_y = self.indexed.loc[self.source_ids, 'species'].to_numpy()
        self.source = prior_models.fit_model(self.source_X, self.source_y, self.axis)

    def test_C_formulas_hold_total_loss_mass_or_only_increase_source_mass(self):
        for ns, nk in ((190, 5), (190, 50), (240, 2), (240, 20)):
            source_up = models.loss_mass_C(ns, nk, 'source_mass_only')
            normalized = models.loss_mass_C(ns, nk, 'source_support_normalized_mass')
            self.assertAlmostEqual(source_up * ns, 10 * (ns + nk))
            self.assertAlmostEqual(normalized * (ns + nk), 10 * ns)
            self.assertEqual(models.loss_mass_C(ns, nk, 'source_support_fixed_C'), 10.)
        for counts in ((0, 2), (2, -1), (True, 2), (2.2, 1)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                models.loss_mass_C(*counts, 'source_mass_only')
        self.assertEqual(tuple(C for C, _ in models.C_CONDITIONS), (.1, 1., 10.))

    def test_new_C_fit_preserves_source_coordinates_gamma_and_actual_metadata(self):
        before = joblib.hash(self.source)
        points = self.source_ids + self.job['support_ids']
        model = models.fit_source_fixedcoords(take(self.metadata, self.X, points),
                    self.indexed.loc[points, 'species'], self.source, C=.1, class_weight=None)
        self.assertEqual(model.estimator_.C, .1)
        self.assertEqual(model.candidate['C'], .1)
        self.assertEqual(model.metadata['C'], .1)
        self.assertEqual(model.estimator_._gamma, self.source.estimator_._gamma)
        self.assertEqual(joblib.hash(model.scaler_), joblib.hash(self.source.scaler_))
        self.assertEqual(joblib.hash(model.preprocessor_), joblib.hash(self.source.preprocessor_))
        self.assertEqual(joblib.hash(self.source), before)
        np.testing.assert_array_equal(model.estimator_.class_weight_, 1.)
        self.assertFalse(any('query' in name.lower() for name in inspect.signature(models.fit_source_fixedcoords).parameters))

    def test_h01_source_mass_arm_never_reads_support_features_or_labels(self):
        reference_models = {}
        for k in models.K_VALUES:
            ids = self.source_ids + prior_models.nested_support(self.job, self.indexed, k)
            reference_models[k] = models.fit_source_fixedcoords(take(self.metadata, self.X, ids),
                self.indexed.loc[ids, 'species'], self.source, C=10., class_weight='balanced')
        calls = []
        original_fit = models.fit_source_fixedcoords

        def fitted(X_train, y_train, reference, *, C, class_weight='balanced'):
            calls.append((np.asarray(X_train).copy(), np.asarray(y_train).copy(), float(C)))
            return original_fit(X_train, y_train, reference, C=C, class_weight=class_weight)

        original_record = artifacts.TaskWriter.record

        def record_after_all_fits(writer, *args, **kwargs):
            self.assertEqual(len(calls), 8)
            return original_record(writer, *args, **kwargs)

        before = joblib.hash(self.source)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with mock.patch.object(models, 'load_parent', return_value=({}, self.source, output)), \
                 mock.patch.object(models, '_load_frozen_refit', side_effect=lambda job, k, *_: (reference_models[k], {'fixture': k})), \
                 mock.patch.object(models, 'fit_source_fixedcoords', side_effect=fitted), \
                 mock.patch.object(artifacts.TaskWriter, 'record', new=record_after_all_fits):
                result = models.execute(self.job, self.X, self.metadata, self.axis, output,
                                         'input', output / 'old', output / 'phase2', output / 'phase3')
            self.assertEqual(result['new_model_fits'], 8)
            self.assertEqual(result['metric_rows'], 13)
            for k, source_call, full_call in zip(models.K_VALUES, calls[::2], calls[1::2]):
                support_ids = prior_models.nested_support(self.job, self.indexed, k)
                full_ids = self.source_ids + support_ids
                np.testing.assert_array_equal(source_call[0], self.source_X)
                np.testing.assert_array_equal(source_call[1], self.source_y)
                self.assertGreater(source_call[2], 10)
                np.testing.assert_array_equal(full_call[0], take(self.metadata, self.X, full_ids))
                np.testing.assert_array_equal(full_call[1], self.indexed.loc[full_ids, 'species'])
                self.assertAlmostEqual(full_call[2] * len(full_ids), 10 * len(self.source_ids))
                self.assertFalse(set(full_ids) & set(self.job['query_ids']))
            artifacts.verify_task(output / 'jobs' / self.job['job_id'], 'input', self.job,
                                  models.expected_endpoints(self.job))
            measured = pd.read_csv(output / 'jobs' / self.job['job_id'] / 'metrics.csv')
            self.assertTrue(measured.loc[measured.method.eq('source_mass_only'), 'support_labels_used'].eq(0).all())
            self.assertTrue(measured.loc[measured.method.eq('source_support_fixed_C'), 'reused_model'].all())
        self.assertEqual(joblib.hash(self.source), before)

    def test_h02_balanced_alias_is_exact_at_each_C_but_new_C_still_refits(self):
        job = {**self.job, 'family': 'H02', 'job_id': 'H02_synthetic'}
        calls = []
        original = models.fit_source_fixedcoords

        def fitted(X_train, y_train, reference, *, C, class_weight='balanced'):
            calls.append((float(C), class_weight, np.asarray(X_train).copy(), np.asarray(y_train).copy()))
            return original(X_train, y_train, reference, C=C, class_weight=class_weight)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with mock.patch.object(models, '_load_weight', return_value=(self.source, {'fixture': True})), \
                 mock.patch.object(models, 'fit_source_fixedcoords', side_effect=fitted):
                result = models.execute(job, self.X, self.metadata, self.axis, output,
                                         'input', output / 'old', output / 'phase2', output / 'phase3')
            self.assertEqual([(C, weight) for C, weight, _, _ in calls], [(.1, 'balanced'), (1., 'balanced')])
            self.assertEqual(result['new_model_fits'], 2)
            self.assertEqual(result['metric_rows'], 12)
            for _, _, X, y in calls:
                np.testing.assert_array_equal(X, self.source_X)
                np.testing.assert_array_equal(y, self.source_y)
            frame = pd.read_csv(output / 'jobs' / job['job_id'] / 'predictions.csv.gz')
            for _, tag in models.C_CONDITIONS:
                first = frame.loc[frame.method.eq(f'C_{tag}_balanced')].set_index('record_id')
                second = frame.loc[frame.method.eq(f'C_{tag}_none')].set_index('record_id').loc[first.index]
                np.testing.assert_array_equal(first.pred_label, second.pred_label)
                np.testing.assert_array_equal(first.scores_json, second.scores_json)
            artifacts.verify_task(output / 'jobs' / job['job_id'], 'input', job, models.expected_endpoints(job))

    def test_historical_frozen_refit_rejects_wrong_C_even_with_valid_byte_hash(self):
        ids = self.source_ids + self.job['support_ids']
        model = models.fit_source_fixedcoords(take(self.metadata, self.X, ids),
                    self.indexed.loc[ids, 'species'], self.source, C=10.)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            name = 'scaler_frozen_gamma_fixed_k10.joblib'
            joblib.dump(model, directory / name)
            result = {**self.job, 'artifact_hashes': {name: artifacts.file_hash(directory / name)}}
            with mock.patch.object(common, 'load_round2_bundle', return_value=(result, directory)), \
                 mock.patch.object(common, 'load_parent', return_value=({}, self.source, directory)):
                loaded, _ = common.load_frozen_refit(self.job, 10, directory, directory)
                self.assertEqual(loaded.estimator_.C, 10)
                model.estimator_.C = 9.
                joblib.dump(model, directory / name)
                result['artifact_hashes'][name] = artifacts.file_hash(directory / name)
                with self.assertRaisesRegex(ValueError, 'C10'):
                    common.load_frozen_refit(self.job, 10, directory, directory)


class ArtifactAndReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.metadata, self.X, self.axis, self.job = synthetic_fixture()

    def seal(self):
        writer = artifacts.TaskWriter(self.job, self.X, self.metadata, self.axis,
                                       self.output, 'input', self.output / 'old')
        truth = writer.labels(self.job['query_ids'])
        predicted = truth.copy()
        predicted[[0, 4]] = np.where(predicted[[0, 4]] == self.job['classes'][0],
                                     self.job['classes'][1], self.job['classes'][0])
        scores = np.asarray([[2., -1.] if label == self.job['classes'][0] else [-1., 2.] for label in predicted])
        identifier = writer.record('identity', 0, scores)
        result = writer.finish([identifier], new_model_fits=0, new_adapter_fits=0)
        return self.output / 'jobs' / self.job['job_id'], result, identifier

    def test_saved_metrics_match_hand_confusion_recall_and_f1(self):
        path, _, identifier = self.seal()
        artifacts.verify_task(path, 'input', self.job, [identifier])
        row = pd.read_csv(path / 'metrics.csv').iloc[0]
        self.assertEqual(json.loads(row.confusion_matrix_json), [[2, 1], [1, 2]])
        for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            self.assertAlmostEqual(row[name], 2 / 3)
        self.assertEqual(json.loads(row.recall_json), {'枣仁': 2 / 3, '酸枣仁': 2 / 3})

    def test_resume_verification_never_refits_or_recounts_cpu(self):
        path, result, identifier = self.seal()
        before = {file.name: artifacts.file_hash(file) for file in path.iterdir()}
        with mock.patch.object(prior_models, 'fit_model', side_effect=AssertionError('unexpected fitting')):
            first = artifacts.verify_task(path, 'input', self.job, [identifier])
            second = artifacts.verify_task(path, 'input', self.job, [identifier])
        self.assertEqual(first['cpu_seconds'], second['cpu_seconds'])
        self.assertEqual(first['cpu_seconds'], result['cpu_seconds'])
        self.assertEqual(before, {file.name: artifacts.file_hash(file) for file in path.iterdir()})
        with self.assertRaises(FileExistsError):
            artifacts.TaskWriter(self.job, self.X, self.metadata, self.axis,
                                  self.output, 'input', self.output / 'old')

    def test_input_job_and_full_endpoint_contract_guard_incorrect_resume(self):
        path, _, identifier = self.seal()
        with self.assertRaisesRegex(ValueError, 'input/code'):
            artifacts.verify_task(path, 'other-input', self.job, [identifier])
        changed = copy.deepcopy(self.job)
        changed['query_ids'] = list(reversed(changed['query_ids']))
        with self.assertRaisesRegex(ValueError, 'frozen design'):
            artifacts.verify_task(path, 'input', changed, [identifier])
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            artifacts.verify_task(path, 'input', self.job, [identifier, 'missing'])

    def test_third_round_historical_evidence_requires_matching_seal_and_roles(self):
        path, result, _ = self.seal()
        artifacts.write_json(self.output / 'frozen_protocol.json', {'input_hash': 'input', 'design': [self.job]})
        loaded, loaded_path = common.load_round3_bundle(self.output, self.job['job_id'])
        self.assertEqual(loaded['job_hash'], result['job_hash'])
        self.assertEqual(path, loaded_path)
        with (path / 'predictions.csv.gz').open('ab') as handle:
            handle.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            common.load_round3_bundle(self.output, self.job['job_id'])

    def test_even_self_consistent_hashes_cannot_hide_wrong_metrics(self):
        path, result, identifier = self.seal()
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


if __name__ == '__main__':
    unittest.main()
