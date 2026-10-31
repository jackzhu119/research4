import unittest

import numpy as np
from sklearn.base import clone

from herb_spectra.stability import StableVariableSelector


class StabilityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(80, 12))
        self.y = -8 * self.X[:, 0] + 3 * self.X[:, 4] + rng.normal(scale=0.05, size=80)
        self.options = dict(n_features_to_select=2, n_resamples=6,
                            inner_runs=12, n_components=3)

    def test_negative_effect_survives_frequency_aggregation_and_fixed_seed(self):
        a = StableVariableSelector(**self.options).fit(self.X, self.y)
        b = clone(a).fit(self.X, self.y)
        self.assertEqual(set(a.selected_indices_), {0, 4})
        np.testing.assert_array_equal(a.frequency_, b.frequency_)
        np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)
        self.assertEqual(a.resample_supports_.shape, (6, 12))
        self.assertTrue(np.all((a.frequency_ >= 0) & (a.frequency_ <= 1)))
        self.assertEqual(a.training_unit_count_, 80)

    def test_unequal_repeat_scans_do_not_increase_sampling_weight(self):
        groups = np.arange(len(self.y))
        indices = np.r_[groups, np.repeat(0, 18), np.repeat(7, 11)]
        original = StableVariableSelector(**self.options).fit(self.X, self.y, groups=groups)
        repeats = StableVariableSelector(**self.options).fit(
            self.X[indices], self.y[indices], groups=groups[indices])
        self.assertEqual(repeats.training_unit_count_, len(self.y))
        self.assertEqual(repeats.sample_unit_, 'physical_sample_means')
        np.testing.assert_array_equal(original.frequency_, repeats.frequency_)
        np.testing.assert_allclose(original.mean_coefficient_importance_,
                                   repeats.mean_coefficient_importance_, rtol=1e-10)
        for first, second in zip(original.resample_unit_indices_, repeats.resample_unit_indices_):
            np.testing.assert_array_equal(first, second)
            self.assertLess(second.max(), 80)
            self.assertEqual(len(second), len(np.unique(second)))

    def test_multiclass_relabeling_is_invariant_for_both_bases(self):
        rng = np.random.default_rng(33)
        X = rng.normal(size=(60, 8))
        labels = np.repeat([0, 1, 2], 20)
        X[:, 0] += labels * 3
        remapped = np.choose(labels, [91, -4, 7])
        for base in ['mcuve', 'cars']:
            options = dict(base_selector=base, task='classification', n_features_to_select=3,
                           n_resamples=3, inner_runs=5, n_components=2, cv=2)
            a = StableVariableSelector(**options).fit(X, labels)
            b = StableVariableSelector(**options).fit(X, remapped)
            np.testing.assert_array_equal(a.frequency_, b.frequency_)
            np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)

    def test_query_transform_never_changes_selection_or_training_diagnostics(self):
        model = StableVariableSelector(**self.options).fit(self.X, self.y)
        before = model.frequency_.copy(), model.support_.copy(), model.resample_supports_.copy()
        query = model.transform(np.full((5, self.X.shape[1]), 1e6))
        self.assertEqual(query.shape, (5, 2))
        for expected, actual in zip(before, [model.frequency_, model.support_, model.resample_supports_]):
            np.testing.assert_array_equal(expected, actual)
        with self.assertRaises(ValueError):model.transform(np.ones((3, 11)))

    def test_count_is_capped_and_all_feature_selection_has_trivial_jaccard(self):
        model = StableVariableSelector(n_features_to_select=100, n_resamples=3,
                                       inner_runs=4).fit(self.X[:, :3], self.y)
        self.assertEqual(model.get_support().sum(), 3)
        np.testing.assert_array_equal(model.transform(self.X[:, :3]), self.X[:, :3])
        self.assertEqual(model.mean_pairwise_jaccard_, 1.)
        np.testing.assert_array_equal(model.frequency_, np.ones(3))

    def test_invalid_budgets_and_undersized_resamples_fail_explicitly(self):
        for option in [{'n_resamples': 1}, {'inner_runs': 1}, {'n_features_to_select': 0},
                       {'sample_fraction': np.nan}, {'sample_fraction': 0},
                       {'n_components': True}, {'base_selector': 'unknown'},
                       {'random_state': -1}, {'sample_fraction': 0.01}]:
            with self.assertRaises(ValueError):
                StableVariableSelector(**option).fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, 'per class'):
            StableVariableSelector(base_selector='cars', cv=3, sample_fraction=.5,
                                    task='classification').fit(self.X[:8], np.repeat([0,1],4))

    def test_conflicting_or_missing_group_identity_is_not_ignored(self):
        with self.assertRaisesRegex(ValueError, 'conflicting targets'):
            StableVariableSelector(**self.options).fit(self.X, self.y, groups=np.zeros(80))
        with self.assertRaisesRegex(ValueError, 'Missing group ID'):
            StableVariableSelector(**self.options).fit(self.X, self.y,
                                                       groups=np.asarray([None]+list(range(79))))
        with self.assertRaisesRegex(ValueError, 'four independent'):
            StableVariableSelector(**self.options).fit(self.X[:6], np.repeat([1.,2.,3.],2),
                                                       groups=np.repeat([0,1,2],2))


if __name__ == '__main__':unittest.main()
