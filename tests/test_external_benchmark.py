import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification

from herb_spectra.benchmark import (PLSDAClassifier, benchmark_dataset,
                                    evaluation_metrics, model_pipeline,
                                    training_folds, validate_arrays)


class ExternalBenchmarkTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(314)
        X = rng.normal(size=(60, 9))
        X[:, 1] += np.repeat([-4., 0., 4.], 20)
        y = np.repeat(['origin-C', 'origin-A', 'origin-B'], 20)
        train = np.r_[np.arange(0, 15), np.arange(20, 35), np.arange(40, 55)]
        test = np.setdiff1d(np.arange(60), train)
        return {'X_train': X[train], 'y_train': y[train],
                'X_test': X[test], 'y_test': y[test]}

    def test_plsda_classes_are_training_labels_and_relabel_equivariant(self):
        arrays = self.data()
        model = PLSDAClassifier(n_components=2).fit(arrays['X_train'], arrays['y_train'])
        predicted = model.predict(arrays['X_test'])
        self.assertTrue(set(predicted) <= set(arrays['y_train']))
        mapping = {'origin-A': 90, 'origin-B': -3, 'origin-C': 12}
        inverse = {v: k for k, v in mapping.items()}
        relabelled = np.asarray([mapping[v] for v in arrays['y_train']])
        equivalent = clone(model).fit(arrays['X_train'], relabelled).predict(arrays['X_test'])
        np.testing.assert_array_equal(predicted, [inverse[v] for v in equivalent])

    def test_exact_cross_partition_duplicates_and_id_overlap_are_rejected(self):
        arrays = self.data()
        arrays['X_test'][0] = arrays['X_train'][3]
        with self.assertRaisesRegex(ValueError, 'Exact spectra overlap'):
            validate_arrays(arrays, 'classification')
        arrays = self.data()
        arrays['train_ids'] = np.asarray([f'sample-{i}' for i in range(45)])
        arrays['test_ids'] = np.asarray(['sample-0'] + [f'query-{i}' for i in range(14)])
        with self.assertRaisesRegex(ValueError, 'IDs overlap'):
            validate_arrays(arrays, 'classification')

    def test_training_duplicate_features_stay_in_one_fold(self):
        arrays = self.data()
        arrays['X_train'][1] = arrays['X_train'][0]
        X, y, _, _, _, _, hashes, audit = validate_arrays(arrays, 'classification')
        self.assertEqual(audit['train_duplicate_feature_rows'], 1)
        for train, valid in training_folds(X, y, 'classification', hashes):
            self.assertFalse(set(hashes[train]) & set(hashes[valid]))

    def test_metrics_preserve_reference_units_and_bias_direction(self):
        actual = np.asarray([1., 3., 5.])
        predicted = actual + 2
        metrics = evaluation_metrics(actual, predicted, 'regression')
        self.assertAlmostEqual(metrics['rmse'], 2.)
        self.assertAlmostEqual(metrics['mae'], 2.)
        self.assertAlmostEqual(metrics['bias_pred_minus_true'], 2.)
        self.assertAlmostEqual(metrics['r2'], -0.5)
        self.assertAlmostEqual(metrics['rpd'], 1.)
        labels = np.asarray(['a', 'a', 'b', 'b'])
        guessed = np.asarray(['a', 'a', 'a', 'b'])
        classified = evaluation_metrics(labels, guessed, 'classification', np.asarray(['a', 'b']))
        self.assertAlmostEqual(classified['accuracy'], .75)
        self.assertAlmostEqual(classified['balanced_accuracy'], .75)

    def test_whole_pipeline_refits_inside_cv_and_test_shift_never_changes_scaler(self):
        arrays = self.data()
        # The high query shift is deliberately absent from train. This is an
        # observable train/test boundary check, not a check of a copied formula.
        arrays['X_test'] += 1000
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'data.npz'
            np.savez(path, **arrays)
            results = benchmark_dataset({'id': 'fixture', 'task': 'classification',
                                         'data_path': str(path), 'source_url': 'synthetic-test',
                                         'preprocess_modes': ['raw']}, root/'run', ['svc'])
            self.assertEqual(results[0]['status'], 'pass')
            self.assertEqual(results[0]['candidate_count'], 4)
            folds = json.loads((root/'run/inner_folds.json').read_text())
            for fold in folds:
                self.assertFalse(set(fold['train_ids']) & set(fold['validation_ids']))
            pipe, _ = model_pipeline('svc', 'classification', 9, 30, ['raw'])
            pipe.fit(arrays['X_train'], arrays['y_train'])
            training_mean = pipe.named_steps['scale'].mean_.copy()
            pipe.predict(arrays['X_test'])
            np.testing.assert_array_equal(pipe.named_steps['scale'].mean_, training_mean)
            self.assertLess(np.max(np.abs(training_mean)), 10.)
            self.assertTrue((root/'run/svc/predictions.csv').is_file())

    def test_selector_is_inside_cloneable_pipeline_and_class_task_is_explicit(self):
        pipe, grid = model_pipeline('cars_svc', 'classification', 12, 30, ['raw'])
        selector = clone(pipe).named_steps['selector']
        self.assertEqual(selector.task, 'classification')
        self.assertEqual(selector.n_runs, 12)
        self.assertIn('model__C', grid)

    def test_prepared_data_hash_is_checked_before_model_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'data.npz'
            np.savez(path, **self.data())
            with self.assertRaisesRegex(ValueError, 'NPZ hash differs'):
                benchmark_dataset({'id': 'fixture', 'task': 'classification',
                                   'data_path': str(path), 'data_sha256': 'changed',
                                   'source_url': 'synthetic-test'}, root/'run', ['svc'])


if __name__ == '__main__':
    unittest.main()
