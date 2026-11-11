"""Check nested fit scope and independent Source+Support refitting."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parent),
               str(Path(__file__).resolve().parents[1] / 'scripts'),
               str(Path(__file__).resolve().parents[1] / 'src'),
               '/workspace/research4/src']

import numpy as np
import meta_strong_control as strong


class StrongControlTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(83)
        self.y = np.repeat(np.arange(5), 8)
        self.X = rng.normal(size=(40, 8))
        self.X[np.arange(40), self.y] += 2

    def test_inner_fits_are_source_only_and_each_ranking_reused(self):
        original = strong._ranking
        calls = []
        def observe(X, y):
            calls.append((X.copy(), y.copy()))
            return original(X, y)
        with patch.object(strong, '_ranking', side_effect=observe):
            model = strong.fit_source_control(self.X, self.y, seed=71)
        # Two independent inner rankings plus the one final Source ranking.
        self.assertEqual([len(X) for X, _ in calls], [20, 20, 40])
        self.assertEqual(len(model.metadata['inner_results']), 24)
        for partition, (matrix, _) in zip(model.metadata['inner_folds'], calls):
            fit = partition['fit_indices']
            validation = partition['validation_indices']
            self.assertFalse(set(fit) & set(validation))
            np.testing.assert_allclose(partition['scaler_mean'], self.X[fit].mean(axis=0))
            np.testing.assert_allclose(matrix.mean(axis=0), 0, atol=1e-12)
        self.assertEqual(model.scores(self.X[:7]).shape, (7, 5))
        self.assertEqual(model.predict(self.X[:7]).shape, (7,))

    def test_refit_freezes_source_choice_and_does_not_mutate_source(self):
        model = strong.fit_source_control(self.X, self.y, seed=71)
        metadata = copy.deepcopy(model.metadata)
        parameters = model.estimator.support_vectors_.copy()
        source_score = model.scores(self.X).copy()
        combined = np.concatenate([self.X, self.X[:5] + 0.5])
        combined_y = np.concatenate([self.y, self.y[:5]])
        with patch.object(strong, 'fit_source_control', side_effect=AssertionError('must not retune')):
            refitted = strong.refit_control(model, combined, combined_y)
        self.assertEqual(refitted.config, model.config)
        self.assertEqual(refitted.metadata['fit_records'], 45)
        self.assertEqual(model.metadata, metadata)
        np.testing.assert_array_equal(model.estimator.support_vectors_, parameters)
        np.testing.assert_array_equal(model.scores(self.X), source_score)
        self.assertIsNot(refitted.scaler, model.scaler)
        self.assertIsNot(refitted.selector, model.selector)

    def test_scores_never_estimate_query_statistics(self):
        model = strong.fit_source_control(self.X, self.y, seed=71)
        first = model.scores(self.X[:1])
        other = model.scores(np.concatenate([self.X[:1], np.full((3, 8), 1e5)]))
        np.testing.assert_allclose(first[0], other[0], atol=1e-12, rtol=0)
        with self.assertRaisesRegex(ValueError, 'five'):
            strong.fit_source_control(self.X[self.y != 4], self.y[self.y != 4])


if __name__ == '__main__':
    unittest.main()
