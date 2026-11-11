"""Fifth-round design, leakage, acquisition and score-geometry invariants."""
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

import b_followup_common_20261110 as artifacts
import b_followup_models_20261110 as prior_models
import b_round3_regularization_20261110 as prior_regularization
import b_round4_models_20261111 as prior_fixed
import b_round4_support_20261111 as prior_support
import b_round5_common_20261111 as common
import b_round5_selector_20261111 as selector
import b_round5_acquisition_20261111 as acquisition
import b_round5_scores_20261111 as scores
import summarize_b_mechanism_round5_20261111 as summary

OLDROOT = ROOT / 'reports/b_mechanism_20261110'
ROUND2ROOT = ROOT / 'reports/b_mechanism_followup_20261110'


def real_inputs(with_features=False):
    metadata = pd.read_csv(ROOT / 'data/research/species_revision_20261108/split_manifest.csv',
                           dtype=str, keep_default_na=False)
    old = json.loads((OLDROOT / 'frozen_protocol.json').read_text())['design']
    if not with_features:
        return metadata, old
    with np.load(OLDROOT / '.cache/features.npz', allow_pickle=False) as values:
        X, axis, ids = values['X_alpha'], values['axis'], values['record_ids'].astype(str)
    aligned = metadata.set_index('record_id', drop=False).loc[ids].reset_index(drop=True)
    return aligned, old, X, axis


def synthetic_fixture(family='I03', source_per_class=9, query_per_class=2, five=False):
    rng = np.random.default_rng(511711)
    rows, roles = [], {role: [] for role in ('source', 'support', 'query')}
    classes = [f'ZY0{i}' for i in range(3, 8)] if five else ['枣仁', '酸枣仁']
    for class_index, label in enumerate(classes):
        origin = label if five else ('ZY01' if class_index == 0 else 'ZY03')
        species = '酸枣仁' if five or class_index else '枣仁'
        for role, number in (('source', source_per_class), ('support', 10), ('query', query_per_class)):
            for index in range(number):
                sample = f'{origin}-{role}-{index:03d}'
                rows.append({'record_id': sample, 'physical_sample_id': sample,
                             'origin_group_id': origin, 'species': species, 'role': 'train',
                             'class_index': class_index})
                roles[role].append(sample)
    metadata = pd.DataFrame(rows)
    X = rng.normal(scale=.55, size=(len(rows), 9))
    for class_index in range(len(classes)):
        X[metadata.class_index.eq(class_index)] += 2.5 * np.sin(np.arange(9) * (.65 + .35 * class_index))
    X[metadata.record_id.str.contains('support')] += np.sin(np.arange(9)) * 1.1
    X[metadata.record_id.str.contains('query')] += np.cos(np.arange(9)) * 5
    axis = np.linspace(.1, 1.5, X.shape[1])
    job = {'job_id': family + '_synthetic', 'family': family,
           'parent_job_id': 'synthetic_parent', 'experiment': 'E03' if five else 'E01',
           'target_origin': 'ZY03-ZY07' if five else 'ZY03', 'seed': 42,
           'ratio': None if five else 'B', 'coverage': None if five else 0, 'fold': 0 if five else None,
           'classes': classes, 'label_column': 'origin_group_id' if five else 'species',
           **{role + '_ids': values for role, values in roles.items()}}
    return metadata, X, axis, job


def take(metadata, X, ids):
    lookup = {sample: index for index, sample in enumerate(metadata.record_id)}
    return X[[lookup[sample] for sample in ids]]


def source_model(metadata, X, axis, job):
    indexed = metadata.set_index('record_id', drop=False)
    return prior_models.fit_model(take(metadata, X, job['source_ids']),
                                  indexed.loc[job['source_ids'], job['label_column']], axis)


def seal_fixture(output, method_names=('identity',), family='I02'):
    metadata, X, axis, job = synthetic_fixture(family)
    writer = artifacts.TaskWriter(job, X, metadata, axis, output, 'synthetic-input', output)
    # Deliberately unequal precision/recall: confusion [[1,1],[0,2]].
    values = np.asarray([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    endpoints = []
    for method in method_names:
        endpoints.append(writer.record(method, 0, values))
    result = writer.finish(endpoints, new_model_fits=0, new_adapter_fits=0)
    path = Path(output) / 'jobs' / job['job_id']
    artifacts.write_json(Path(output) / 'frozen_protocol.json',
                         {'input_hash': 'synthetic-input', 'design': [job]})
    return job, result, path


def reseal(path):
    result = json.loads((path / 'result.json').read_text())
    for name in result['artifact_hashes']:
        result['artifact_hashes'][name] = artifacts.file_hash(path / name)
    artifacts.write_json(path / 'result.json', result)
    marker = json.loads((path / 'COMPLETE.json').read_text())
    marker['result_sha256'] = artifacts.file_hash(path / 'result.json')
    artifacts.write_json(path / 'COMPLETE.json', marker)


class FrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.old = real_inputs()
        cls.binary = next(job for job in cls.old if job['experiment'] == 'E01' and job['ratio'] == 'B')
        cls.pure = next(job for job in cls.old if job['experiment'] == 'E03')

    def test_all180_jobs_preserve_original_train_ids_and_kind_semantics(self):
        runner = importlib.import_module('run_b_mechanism_round5_20261111')
        metadata, old, X, axis = real_inputs(with_features=True)
        jobs = runner.build_design(metadata, X=X, axis=axis, old_jobs=old)
        self.assertEqual(len(jobs), 180)
        self.assertEqual(pd.Series([item['family'] for item in jobs]).value_counts().to_dict(),
                         {'I01': 45, 'I02': 45, 'I03': 45, 'I04': 45})
        self.assertEqual(len({job['job_id'] for job in jobs}), 180)
        originals = {job['job_id']: job for job in old}
        for job in jobs:
            artifacts.validate_roles(job, metadata)
            for role in ('source', 'support', 'query'):
                self.assertEqual(job[role + '_ids'], originals[job['parent_job_id']][role + '_ids'])
        selected = [job for job in jobs if job['family'] == 'I01']
        self.assertEqual(sum(selector.fit_count(job) for job in selected), 675)
        self.assertEqual(sum(selector.adapter_count(job) for job in selected), 1080)
        geometric = [job for job in jobs if job['family'] == 'I03']
        self.assertEqual(sum(scores.adapter_count(job) for job in geometric), 1350)
        ess = [job for job in jobs if job['family'] == 'I04']
        self.assertEqual(sum(scores.adapter_count(job) for job in ess), 1170)
        for job in (item for item in jobs if item['family'] == 'I02'):
            self.assertLessEqual(acquisition.fit_count(job), 15)
            self.assertLessEqual(acquisition.adapter_count(job), 15)

    def test_inner_roles_partition_source_and_exclude_outer_support_query(self):
        for old, expected_source in ((self.binary, 180), (self.pure, 100)):
            job = copy.deepcopy(old)
            job.update(family='I01', parent_job_id=old['job_id'], job_id='I01_' + old['job_id'])
            plan = selector.inner_plan(job, self.metadata)
            self.assertEqual(plan, selector.inner_plan(job, self.metadata))
            self.assertEqual(plan['episode_seeds'], [501, 502, 503])
            for episode in plan['episodes']:
                parts = [set(episode[key]) for key in ('inner_source_ids', 'inner_support_ids', 'inner_validation_ids')]
                self.assertEqual(set.union(*parts), set(old['source_ids']))
                self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
                self.assertEqual(len(parts[0]), expected_source)
                self.assertEqual(len(parts[2]), 40)
                self.assertFalse(set.union(*parts) & (set(old['support_ids']) | set(old['query_ids'])))
                indexed = self.metadata.set_index('record_id', drop=False)
                previous = set()
                for k in (1, 3, 5, 10):
                    chosen = selector.nested_inner_support(episode['inner_support_ids'], indexed,
                                                          old['label_column'], old['classes'], k)
                    self.assertEqual(len(chosen), k * len(old['classes']))
                    self.assertTrue(previous.issubset(chosen))
                    previous = set(chosen)

    def test_invalid_physical_role_and_kind_inputs_are_rejected(self):
        metadata, _, _, job = synthetic_fixture()
        for case in ('overlap', 'physical', 'heldout', 'kind'):
            changed, frame = copy.deepcopy(job), metadata.copy()
            if case == 'overlap':
                changed['support_ids'][0] = changed['query_ids'][0]
            elif case == 'physical':
                frame.loc[1, 'physical_sample_id'] = frame.loc[0, 'physical_sample_id']
            elif case == 'heldout':
                frame.loc[frame.record_id.eq(changed['source_ids'][0]), 'role'] = 'prediction'
            else:
                frame.loc[frame.origin_group_id.eq('ZY03'), 'species'] = '枣仁'
            with self.subTest(case=case), self.assertRaises(ValueError):
                artifacts.validate_roles(changed, frame)


class SourcePolicyTests(unittest.TestCase):
    def test_ties_choose_identity_and_means_not_best_single_episode(self):
        rows = [{'episode_seed': seed, 'k': k, 'method': method, 'macro_f1': .8}
                for seed in (501, 502, 503) for k in (1, 3, 5, 10)
                for method in selector.CANDIDATE_METHODS]
        rows = pd.DataFrame(rows)
        rows.loc[rows.k.eq(1) & rows.method.eq('full_refit'), 'macro_f1'] = [.99, .6, .6]
        rows.loc[rows.k.eq(3) & rows.method.eq('bias'), 'macro_f1'] = .8000000000001
        rows.loc[rows.k.eq(5) & rows.method.eq('mean_anchor'), 'macro_f1'] = .81
        choices = selector.select_policy(rows)
        self.assertEqual(choices['1']['selected_method'], 'identity')
        self.assertEqual(choices['3']['selected_method'], 'identity')
        self.assertEqual(choices['5']['selected_method'], 'mean_anchor')
        self.assertEqual(choices['10']['selected_method'], 'identity')
        self.assertEqual(list(inspect.signature(selector.select_policy).parameters), ['inner_metrics'])
        with self.assertRaises(ValueError):
            selector.select_policy(pd.concat([rows, rows.iloc[:1]], ignore_index=True))
        with self.assertRaises(ValueError):
            selector.select_policy(rows.loc[~rows.method.eq('bias')])

    def test_execute_fits_fresh_inner_source_only_then_freezes_policy_before_outer_reads(self):
        metadata, X, axis, job = synthetic_fixture('I01', source_per_class=120)
        indexed = metadata.set_index('record_id', drop=False)
        source = source_model(metadata, X, axis, job)
        original_hash = joblib.hash(source)
        plan = selector.inner_plan(job, metadata)
        job['inner_plan'] = plan
        row_ids = {row.tobytes(): identifier for identifier, row in zip(metadata.record_id, X)}
        outer_scores = source.predict_scores(take(metadata, X, job['query_ids']))
        full_models = {}
        for k in selector.K_VALUES:
            ids = prior_models.nested_support(job, indexed, k)
            all_ids = job['source_ids'] + ids
            full_models[k] = prior_fixed.fit_source_fixedcoords(
                take(metadata, X, all_ids), indexed.loc[all_ids, 'species'], source, C=10., class_weight='balanced')
        original_source_fit, original_refit = selector.fit_source_candidate, selector.fit_source_fixedcoords
        source_calls, head_calls, models_to_episode = [], [], {}
        policy_frozen = {'value': False}

        def fresh(candidate, train_X, train_y, frequency, seed=42):
            identifiers = [row_ids[row.tobytes()] for row in train_X]
            episode = plan['episodes'][len(source_calls)]
            self.assertEqual(identifiers, episode['inner_source_ids'])
            self.assertFalse(set(identifiers) & (set(job['support_ids']) | set(job['query_ids'])))
            model = original_source_fit(candidate, train_X, train_y, frequency, seed=seed)
            np.testing.assert_allclose(model.scaler_.mean_, model.preprocessor_.transform(train_X).mean(axis=0),
                                       rtol=0, atol=1e-13)
            self.assertFalse(np.array_equal(model.scaler_.mean_, source.scaler_.mean_))
            source_calls.append(identifiers)
            models_to_episode[id(model)] = episode
            return model

        def refit(train_X, train_y, reference, **kwargs):
            identifiers = [row_ids[row.tobytes()] for row in train_X]
            episode = models_to_episode[id(reference)]
            self.assertFalse(set(identifiers) & set(episode['inner_validation_ids']))
            self.assertFalse(set(identifiers) & (set(job['support_ids']) | set(job['query_ids'])))
            self.assertEqual(identifiers[:len(episode['inner_source_ids'])], episode['inner_source_ids'])
            head_calls.append(identifiers)
            return original_refit(train_X, train_y, reference, **kwargs)

        original_choice = selector.select_policy
        def choose(inner_metrics):
            value = original_choice(inner_metrics)
            policy_frozen['value'] = True
            return value

        def historical(root, parent, method, k, draw, query_ids):
            self.assertTrue(policy_frozen['value'])
            ids = prior_models.nested_support(job, indexed, k)
            y = indexed.loc[ids, 'species'].to_numpy()
            support_scores = source.predict_scores(take(metadata, X, ids))
            if method == 'raw_bias_only':
                value, adapter = selector.calibrate_scores('bias_only', support_scores, y, outer_scores, job['classes'])
            else:
                adapter = prior_regularization.fit_adapter(support_scores, y, job['classes'], loss='mean')
                value = outer_scores @ adapter['coef'] + adapter['intercept']
            return value, np.asarray(job['classes'])[value.argmax(axis=1)], {'support_ids': ids}, {'synthetic': True}

        def parent(*args):
            self.assertTrue(policy_frozen['value'])
            return {}, source, Path('/synthetic-verified-parent')

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(selector, 'fit_source_candidate', side_effect=fresh), \
                    mock.patch.object(selector, 'fit_source_fixedcoords', side_effect=refit), \
                    mock.patch.object(selector, 'select_policy', side_effect=choose), \
                    mock.patch.object(selector, 'load_parent', side_effect=parent), \
                    mock.patch.object(common, 'load_scores', side_effect=historical), \
                    mock.patch.object(common, 'load_frozen_refit', side_effect=lambda j,k,*a: (full_models[k], {'synthetic': True})):
                result = selector.execute(job, X, metadata, axis, directory, 'selector-input', directory,
                                          directory, directory, directory)
            path = Path(directory) / 'jobs' / job['job_id']
            artifacts.verify_task(path, 'selector-input', job, selector.expected_endpoints(job))
            self.assertEqual(result['new_model_fits'], 15)
            self.assertEqual(result['new_adapter_fits'], 24)
            self.assertEqual(len(source_calls), 3)
            self.assertEqual(len(head_calls), 12)
            predictions = pd.read_csv(path / 'predictions.csv.gz', keep_default_na=False)
            endpoints = {item['endpoint_id']: item for item in json.loads((path / 'endpoints.json').read_text())}
            choices = json.loads((path / 'source_policy.json').read_text())
            for k in selector.K_VALUES:
                choice = choices[str(k)]['selected_method']
                selected = predictions.loc[predictions.method.eq('source_selected') & predictions.k.eq(k)]
                baseline = predictions.loc[predictions.method.eq(choice) & predictions.k.eq(0 if choice == 'identity' else k)]
                self.assertEqual(selected.pred_label.tolist(), baseline.pred_label.tolist())
                endpoint = endpoints[artifacts.endpoint_id('source_selected', k)]
                self.assertEqual(endpoint['support_labels_used'], 0 if choice == 'identity' else 2 * k)
                self.assertEqual(endpoint['decision_rule'], 'SVC.predict' if choice == 'full_refit' else 'score_argmax')
        self.assertEqual(joblib.hash(source), original_hash)


class AcquisitionTests(unittest.TestCase):
    def test_ranking_is_permutation_equivariant_and_has_no_label_query_parameter(self):
        values = np.asarray([[0., 0.], [1., 0.], [2., 2.], [-1., 3.], [3., -3.]])
        score = np.asarray([[2., 0.], [.1, 0.], [4., 1.], [.4, .2], [.01, 0.]])
        keys = acquisition.opaque_keys(['ZY01-a', 'ZY03-b', 'ZY07-c', 'ZY04-d', 'ZY06-e'])
        permutation = np.asarray([3, 1, 4, 0, 2])
        for policy, seed in (('random', 601), ('uncertainty', 0), ('diversity', 0)):
            ordered = acquisition.label_blind_order(values, score, np.zeros(2), keys, policy, seed)
            changed = acquisition.label_blind_order(values[permutation], score[permutation], np.zeros(2),
                                                      np.asarray(keys)[permutation], policy, seed)
            self.assertEqual([keys[i] for i in ordered], [keys[permutation[i]] for i in changed])
            self.assertEqual(sorted(ordered), list(range(len(values))))
        uncertainty = acquisition.label_blind_order(values, score, np.zeros(2), keys, 'uncertainty')
        self.assertEqual(uncertainty[0], 4)
        self.assertTrue(all(len(key) == 64 for key in keys))
        parameters = inspect.signature(acquisition.label_blind_order).parameters
        self.assertFalse(any('label' in name or 'query' in name or 'record_id' in name for name in parameters))

    def test_ties_are_resolved_only_by_opaque_keys(self):
        values, score = np.zeros((4, 2)), np.zeros((4, 3))
        keys = ['b', 'd', 'a', 'c']
        for policy in ('uncertainty', 'diversity'):
            order = acquisition.label_blind_order(values, score, np.zeros(2), keys, policy)
            self.assertEqual([keys[i] for i in order], ['a', 'b', 'c', 'd'])

    def test_plan_is_nested_canonical_and_labels_cannot_change_ranked_identities(self):
        metadata, X, axis, job = synthetic_fixture('I02')
        source = source_model(metadata, X, axis, job)
        frame = metadata.copy()
        frame['hidden_label'] = frame['species']
        altered = copy.deepcopy(job)
        altered['label_column'] = 'hidden_label'
        parent = {'source_hash': 'synthetic-source-hash'}
        with mock.patch.object(acquisition, 'load_parent', return_value=(parent, source, Path('/synthetic'))):
            baseline = acquisition.acquisition_plan(altered, X, frame, axis, '.', '.', '.', '.')
            # Swap labels only after the already fitted source and opaque pool
            # exist; none of the acquisition rankings may consume this reveal.
            changed = frame.copy()
            pool = changed.record_id.isin(job['support_ids'])
            changed.loc[pool, 'hidden_label'] = changed.loc[pool, 'hidden_label'].map({'枣仁':'酸枣仁', '酸枣仁':'枣仁'})
            after = acquisition.acquisition_plan(altered, X, changed, axis, '.', '.', '.', '.')
        self.assertEqual([item['selected_ids'] for item in baseline['episodes']],
                         [item['selected_ids'] for item in after['episodes']])
        self.assertEqual(len(baseline['episodes']), 20)
        for policy in [value[0] for value in acquisition.POLICIES]:
            previous = set()
            episodes = [item for item in baseline['episodes'] if item['policy'] == policy]
            for item in episodes:
                self.assertEqual(len(item['selected_ids']), item['budget'])
                self.assertTrue(previous.issubset(item['selected_ids']))
                previous = set(item['selected_ids'])
                self.assertEqual(item['adaptation_ids'], [s for s in job['support_ids'] if s in previous])
            self.assertEqual(episodes[-1]['reuse_k'], 10)
            self.assertEqual(episodes[-1]['adaptation_ids'], job['support_ids'])
        self.assertEqual(len(acquisition.expected_endpoints(job)), 42)

    def test_missing_class_fallback_never_resamples_or_spends_unreported_labels(self):
        metadata, X, axis, job = synthetic_fixture('I02')
        indexed = metadata.set_index('record_id', drop=False)
        source = source_model(metadata, X, axis, job)
        source_scores = source.predict_scores(take(metadata, X, job['query_ids']))
        ids = job['support_ids']
        all_ids = job['source_ids'] + ids
        existing_full = prior_fixed.fit_source_fixedcoords(take(metadata, X, all_ids),
            indexed.loc[all_ids, 'species'], source, C=10., class_weight=None)
        support_scores = source.predict_scores(take(metadata, X, ids))
        mean = prior_regularization.fit_adapter(support_scores, indexed.loc[ids, 'species'].to_numpy(),
                                               job['classes'], loss='mean')
        mean_scores = source_scores @ mean['coef'] + mean['intercept']
        parent = {'source_hash': 'synthetic-source-hash'}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(acquisition, 'load_parent', return_value=(parent, source, Path('/synthetic'))), \
                    mock.patch.object(acquisition, 'label_blind_order', side_effect=lambda a,*args,**kwargs:list(range(len(a)))):
                plan = acquisition.acquisition_plan(job, X, metadata, axis, '.', '.', '.', '.')
                job['acquisition_plan'] = plan
                self.assertEqual(plan['new_model_fits'], 3)
                self.assertEqual(plan['new_adapter_fits'], 0)
                self.assertEqual(sum(ep['guard_fallback_identity'] for ep in plan['episodes']), 15)
                actual_refit = acquisition.fit_source_fixedcoords
                with mock.patch.object(acquisition, 'fit_source_fixedcoords', wraps=actual_refit) as fit, \
                        mock.patch.object(acquisition, 'fit_adapter', side_effect=AssertionError('No incomplete-class adapter fit')), \
                        mock.patch.object(acquisition, '_historical_mean', return_value=(mean_scores,
                            np.asarray(job['classes'])[mean_scores.argmax(axis=1)], mean, {'synthetic':True})), \
                        mock.patch.object(common, 'load_frozen_refit', return_value=(existing_full, {'synthetic':True})) as historical:
                    result = acquisition.execute(job, X, metadata, axis, directory, 'acq-input', directory,
                                                 directory, directory, directory)
                self.assertEqual(fit.call_count, 3)
                self.assertEqual(historical.call_count, 1)
            path = Path(directory) / 'jobs' / job['job_id']
            artifacts.verify_task(path, 'acq-input', job, acquisition.expected_endpoints(job))
            self.assertEqual(result['new_model_fits'], 3)
            self.assertEqual(result['new_adapter_fits'], 0)
            predictions = pd.read_csv(path / 'predictions.csv.gz', keep_default_na=False)
            base = predictions.loc[predictions.method.eq('identity')].pred_label.tolist()
            endpoints = json.loads((path / 'endpoints.json').read_text())
            fallback = [item for item in endpoints if item.get('guard_fallback_identity')]
            self.assertEqual(len(fallback), 15)
            for item in fallback:
                self.assertEqual(item['support_labels_used'], 0)
                self.assertEqual(item['labels_observed_for_guard'], item['budget'])
                self.assertEqual(item['budget_labels_observed'], item['budget'])
                self.assertEqual(item['fallback_reason'], 'FALLBACK_IDENTITY_MISSING_SUPPORT_CLASSES')
                self.assertEqual(predictions.loc[predictions.endpoint_id.eq(item['endpoint_id'])].pred_label.tolist(), base)


class ScoreGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = synthetic_fixture('I03')
        cls.pure = synthetic_fixture('I03', five=True)

    def test_source_only_bases_have_unit_trace_and_never_mutate_svc(self):
        metadata, X, axis, job = self.pure
        source = source_model(metadata, X, axis, job)
        original = joblib.hash(source)
        source_X = take(metadata, X, job['source_ids'])
        for mode in scores.FEATURE_METHODS:
            basis = scores.fit_feature_basis(source, source_X, mode)
            transformed = scores.transform_feature_basis(source, source_X, basis)
            np.testing.assert_allclose(transformed.mean(axis=0), 0, rtol=0, atol=1e-13)
            self.assertAlmostEqual(float(np.mean(np.sum(transformed ** 2, axis=1))), 1., places=12)
            self.assertEqual(basis['source_samples'], len(job['source_ids']))
            if basis['pca'] is not None:
                self.assertEqual(basis['pca'].n_samples_, len(job['source_ids']))
                self.assertEqual(basis['feature_count'], 5)
            # Transforming arbitrarily shifted Query values cannot refit the
            # Source center/PCA/scale or change the frozen source classifier.
            before = joblib.hash(basis)
            scores.transform_feature_basis(source, take(metadata, X, job['query_ids']) + 100, basis)
            self.assertEqual(joblib.hash(basis), before)
        self.assertEqual(joblib.hash(source), original)
        self.assertEqual(source.estimator_.decision_function_shape, 'ovr')
        self.assertNotIn('query', inspect.signature(scores.fit_feature_basis).parameters)

    def test_binary_normalized_ovr_ovo_and_full_rank_pca_grams_agree(self):
        metadata, X, axis, job = self.binary
        source = source_model(metadata, X, axis, job)
        source_X = take(metadata, X, job['source_ids'])
        blocks = []
        for mode in scores.FEATURE_METHODS:
            basis = scores.fit_feature_basis(source, source_X, mode)
            blocks.append(scores.transform_feature_basis(source, source_X, basis))
        reference = blocks[0] @ blocks[0].T
        for block in blocks[1:]:
            np.testing.assert_allclose(block @ block.T, reference, rtol=0, atol=1e-12)

    def test_feature_residual_gradient_uses_source_output_anchor_and_mean_prior(self):
        rng = np.random.default_rng(771)
        feature = rng.normal(size=(9, 5))
        base = rng.normal(size=(9, 3))
        classes, labels = ['a', 'b', 'c'], np.tile(['a', 'b', 'c'], 3)
        adapter = scores.fit_feature_residual(feature, base, labels, classes)
        design = np.column_stack([feature, np.ones(len(labels))])
        delta = np.vstack([adapter['feature_coef'], adapter['intercept']])
        residual = np.eye(3)[[classes.index(label) for label in labels]] - base
        np.testing.assert_allclose(design.T @ (design @ delta - residual) + len(labels) * delta,
                                   0, rtol=0, atol=1e-12)
        self.assertEqual(adapter['regularization'], len(labels))
        # Zero correction retains the ORIGINAL OVR anchor, irrespective of
        # whether the residual features contain OVO pairwise dimensions.
        np.testing.assert_array_equal(base + feature @ np.zeros((5, 3)), base)
        self.assertFalse(any('query' in name for name in inspect.signature(scores.fit_feature_residual).parameters))

    def test_score_and_ess_families_preserve_finite_endpoint_and_reuse_counts(self):
        for fixture in (self.binary, self.pure):
            _, _, _, job = fixture
            self.assertEqual(len(scores.expected_endpoints(job)), 42)
            self.assertEqual(scores.adapter_count(job), 30)
            self.assertEqual(scores.fit_count(job), 0)
            job = copy.deepcopy(job)
            job['family'] = 'I04'
            self.assertEqual(len(scores.expected_endpoints(job)), 47 if len(job['classes']) == 2 else 56)
            self.assertEqual(scores.adapter_count(job), 24 if len(job['classes']) == 2 else 30)
            methods = [identifier for identifier in scores.expected_endpoints(job) if 'weighted_' in identifier]
            self.assertTrue(all('__k-1__' in identifier for identifier in methods))

    def test_balanced_ess_penalty_is_the_existing_mean_loss_objective(self):
        for n in (10, 25):
            value = scores.ess_penalty(np.ones(n))
            self.assertEqual(value['ESS'], n)
            self.assertEqual(value['mean_lambda'], n)
            self.assertEqual(value['ess_lambda'], n)
        labels = np.asarray(['a'] + ['b'] * 9)
        weights = prior_support.class_balanced_weights(labels, ['a', 'b'])
        value = scores.ess_penalty(weights)
        self.assertAlmostEqual(value['ESS'], 3.6)
        self.assertAlmostEqual(value['ess_lambda'], 250 / 9)
        self.assertAlmostEqual(value['ess_lambda'], float(np.dot(weights, weights)))
        self.assertGreater(value['ess_lambda'], value['mean_lambda'])
        for invalid in ([0., 0.], [1., np.inf], [-1., 3.], [2., 2.]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scores.ess_penalty(invalid)

    def test_ess_weighted_fit_satisfies_fixed_same_id_penalized_objective(self):
        classes, labels = ['a', 'b'], np.asarray(['a'] + ['b'] * 9)
        values = np.random.default_rng(92).normal(size=(10, 2))
        weights = prior_support.class_balanced_weights(labels, classes)
        penalty = scores.ess_penalty(weights)['ess_lambda']
        adapter = prior_support.fit_residual_adapter(values, labels, classes, kind='anchor',
                                                    class_balanced=True, regularization=penalty)
        design = np.column_stack([values, np.ones(len(labels))])
        delta = np.vstack([np.asarray(adapter['coef']) - np.eye(2), adapter['intercept']])
        residual = np.eye(2)[[classes.index(label) for label in labels]] - values
        np.testing.assert_allclose(design.T @ (weights[:, None] * (design @ delta - residual)) + penalty * delta,
                                   0, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(adapter['weights'], weights)
        self.assertAlmostEqual(sum(adapter['weights']), len(labels))
        mean = prior_support.fit_residual_adapter(values, labels, classes, kind='anchor',
                                                  class_balanced=True, regularization=len(labels))
        mean_delta = np.vstack([np.asarray(mean['coef']) - np.eye(2), mean['intercept']])
        self.assertLess(np.linalg.norm(delta), np.linalg.norm(mean_delta))
        base = np.asarray([[3., 1.], [0., 2.], [1., 2.]])
        changed = np.asarray([[3., 1.], [1., 3.], [4., 2.]])
        movement = scores._motion(base, changed)
        self.assertEqual(movement['score_moved_count'], 2)
        self.assertEqual(movement['decision_changed_count'], 1)


class SummaryEvidenceTests(unittest.TestCase):
    def repeated_predictions(self):
        rows = []
        for identifier, label, before, after in (('p1', 'a', 'b', 'a'), ('p2', 'b', 'b', 'a')):
            base = {'job_id': 'I03_fixture', 'family': 'I03', 'experiment': 'E01',
                    'target_origin': 'ZY03', 'origin_group_id': 'ZY03',
                    'physical_sample_id': identifier, 'true_label': label}
            rows.append({**base, 'method': 'identity', 'k': 0, 'draw': 0, 'pred_label': before})
            for draw in (0, 101, 102):
                rows.append({**base, 'method': 'ovr_trace1', 'k': 1, 'draw': draw, 'pred_label': after})
        return pd.DataFrame(rows)

    def test_paired_summary_keeps_repeats_in_two_physical_clusters(self):
        predictions = self.repeated_predictions()
        transitions, intervals = summary.comparison_tables(predictions, n_boot=200)
        whole = transitions.loc[transitions.scope.eq('ALL')].iloc[0]
        self.assertEqual(whole['n_predictions'], 6)
        self.assertEqual(whole['n_physical_samples'], 2)
        self.assertEqual(whole['wrong_to_correct'], 3)
        self.assertEqual(whole['correct_to_wrong'], 3)
        self.assertEqual(whole['delta_accuracy'], 0)
        self.assertEqual(whole['ci_lower'], -1)
        self.assertEqual(whole['ci_upper'], 1)
        self.assertTrue(intervals.resampling_unit.eq('physical_sample_id').all())
        shuffled, _ = summary.comparison_tables(predictions.sample(frac=1, random_state=19), n_boot=200)
        columns = ['scope', 'delta_accuracy', 'ci_lower', 'ci_upper', 'n_physical_samples']
        pd.testing.assert_frame_equal(transitions[columns], shuffled[columns])
        earlier = predictions.loc[predictions.method.eq('ovr_trace1')].copy()
        earlier['method'] = 'raw_mean_anchor'
        earlier = earlier.drop(earlier.index[earlier.draw.eq(101)][0])
        missing = pd.concat([predictions, earlier], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'incomplete matched'):
            summary.comparison_tables(missing, n_boot=100)

    def test_exact_reuse_requires_both_paired_ids_and_saved_scores(self):
        left = pd.DataFrame({'physical_sample_id': ['p1', 'p2'], 'pred_label': ['a', 'b'],
                             'scores_json': ['[2,1]', '[0,1]']})
        right = left.iloc[::-1].copy()
        self.assertEqual(summary._check_same_endpoint(left, right, 'reuse'), 2)
        right.loc[right.physical_sample_id.eq('p1'), 'scores_json'] = '[3,1]'
        with self.assertRaisesRegex(ValueError, 'saved scores differ'):
            summary._check_same_endpoint(left, right, 'reuse')
        right = left.copy()
        right.loc[0, 'physical_sample_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'Query coverage differs'):
            summary._check_same_endpoint(left, right, 'reuse')

    def test_partial_summary_and_verification_never_fit_or_recount_task_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            job, result, path = seal_fixture(directory, family='I03')
            unrun = copy.deepcopy(job)
            unrun['job_id'] = 'I03_not_run'
            artifacts.write_json(Path(directory) / 'frozen_protocol.json', {
                'input_hash': 'synthetic-input', 'design': [job, unrun],
                'planned_new_model_fits': 0, 'planned_new_adapter_fits': 0,
                'endpoint_contract': {job['job_id']: result['expected_endpoints']}})
            before = {item.name: artifacts.file_hash(item) for item in path.iterdir() if item.is_file()}
            with mock.patch.object(prior_models, 'fit_model', side_effect=AssertionError('Summary cannot fit')), \
                    mock.patch.object(scores, 'fit_feature_residual', side_effect=AssertionError('Summary cannot adapt')), \
                    mock.patch.object(summary, 'make_figures', return_value=[]):
                value = summary.summarize(directory, n_boot=100, report_path=Path(directory) / 'report.md')
                receipt = summary.summarize(directory, verify=True)
            self.assertEqual(value['status'], 'PARTIAL')
            self.assertEqual(value['completed_jobs'], 1)
            self.assertEqual(value['expected_jobs'], 2)
            self.assertEqual(value['physical_query_samples'], 4)
            self.assertEqual(value['new_model_fits'], 0)
            self.assertEqual(value['new_adapter_fits'], 0)
            self.assertEqual(value['cpu_seconds'], result['cpu_seconds'])
            self.assertEqual(value['e05_status'], 'NOT_RUN_NEEDS_NEW_BATCH_DATA')
            self.assertEqual(receipt['sealed_jobs'], 1)
            self.assertEqual(before, {item.name: artifacts.file_hash(item) for item in path.iterdir() if item.is_file()})


class SealedEvidenceTests(unittest.TestCase):
    def test_saved_metrics_independently_match_hand_confusion_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            job, result, path = seal_fixture(directory)
            value = artifacts.verify_task(path, 'synthetic-input', job, result['expected_endpoints'])
            metric = pd.read_csv(path / 'metrics.csv').iloc[0]
            self.assertAlmostEqual(metric.accuracy, .75)
            self.assertAlmostEqual(metric.balanced_accuracy, .75)
            self.assertAlmostEqual(metric.macro_f1, (2 / 3 + .8) / 2)
            self.assertEqual(json.loads(metric.confusion_matrix_json), [[1, 1], [0, 2]])
            self.assertEqual(json.loads(metric.recall_json), {'枣仁': .5, '酸枣仁': 1.})
            self.assertEqual(value['new_model_fits'], 0)

    def test_resume_verification_does_not_fit_mutate_or_recount_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            job, result, path = seal_fixture(directory)
            before = {item.name: artifacts.file_hash(item) for item in path.iterdir() if item.is_file()}
            with mock.patch.object(prior_models, 'fit_model', side_effect=AssertionError('Resume cannot train')):
                first = artifacts.verify_task(path, 'synthetic-input', job)
                second = artifacts.verify_task(path, 'synthetic-input', job)
            self.assertEqual(first['cpu_seconds'], second['cpu_seconds'])
            self.assertEqual(first['cpu_seconds'], result['cpu_seconds'])
            self.assertEqual(before, {item.name: artifacts.file_hash(item) for item in path.iterdir() if item.is_file()})
            with self.assertRaises(ValueError):
                artifacts.verify_task(path, 'changed-input', job)
            changed = copy.deepcopy(job)
            changed['seed'] = 99
            with self.assertRaises(ValueError):
                artifacts.verify_task(path, 'synthetic-input', changed)

    def test_historical_reader_returns_scores_not_truth_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            job, result, path = seal_fixture(directory)
            values, predicted, metadata, provenance = common.load_scores(
                directory, job['job_id'], 'identity', 0, 0, job['query_ids'])
            self.assertEqual(values.shape, (4, 2))
            self.assertEqual(len(predicted), 4)
            self.assertNotIn('true_label', metadata)
            self.assertEqual(provenance['historical_input_hash'], 'synthetic-input')
            with (path / 'metrics.csv').open('a') as handle:
                handle.write('\nchanged')
            with self.assertRaises(ValueError):
                common.load_bundle(directory, job['job_id'])

    def test_self_consistent_forged_metric_still_fails_independent_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            job, _, path = seal_fixture(directory)
            values = pd.read_csv(path / 'metrics.csv')
            values.loc[0, 'macro_f1'] = .99
            values.to_csv(path / 'metrics.csv', index=False)
            reseal(path)
            with self.assertRaisesRegex(ValueError, 'Independent metric mismatch'):
                artifacts.verify_task(path, 'synthetic-input', job)

    def test_self_consistent_missing_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job, result, path = seal_fixture(directory, ('identity', 'other'))
            for name in ('metrics.csv', 'predictions.csv.gz'):
                frame = pd.read_csv(path / name)
                frame = frame.loc[frame.method.eq('identity')]
                frame.to_csv(path / name, index=False)
            reseal(path)
            with self.assertRaisesRegex(ValueError, 'Incomplete endpoint'):
                artifacts.verify_task(path, 'synthetic-input', job, result['expected_endpoints'])


if __name__ == '__main__':
    unittest.main()
