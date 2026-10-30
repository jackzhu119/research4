import unittest

import numpy as np
from sklearn.base import clone

from herb_spectra.selection import BoundedIRIVInspiredSelector, CARSStyleSelector, MCUVESelector


class SelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(80, 12))
        self.y = -8 * self.X[:, 0] + 3 * self.X[:, 4] + rng.normal(scale=0.05, size=80)

    def test_mcuve_retains_strong_negative_effect_and_is_reproducible(self):
        a = MCUVESelector(n_runs=35, n_components=4, n_features_to_select=2).fit(self.X, self.y)
        b = clone(a).fit(self.X, self.y)
        self.assertIn(0, a.selected_indices_)
        self.assertIn(4, a.selected_indices_)
        self.assertLess(a.signed_stability_[0, 0], 0)
        np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)
        self.assertTrue(np.isfinite(a.importances_).all())

    def test_cars_selection_reproducible_and_transform_does_not_refit(self):
        model = CARSStyleSelector(n_runs=8, n_components=3).fit(self.X, self.y)
        copy = clone(model).fit(self.X, self.y)
        np.testing.assert_array_equal(model.selected_indices_, copy.selected_indices_)
        before = model.support_.copy()
        transformed = model.transform(np.full((5, 12), 1e6))
        self.assertEqual(transformed.shape, (5, len(model.selected_indices_)))
        np.testing.assert_array_equal(before, model.support_)
        self.assertGreater(len(model.selected_indices_), 0)
        self.assertTrue(np.isfinite(model.selection_mse_))

    def test_group_means_remove_unequal_repeat_scan_weight(self):
        groups = np.arange(len(self.y))
        indices = np.r_[np.arange(len(self.y)), np.repeat(0, 15)]
        original = MCUVESelector(n_runs=20, n_features_to_select=3).fit(self.X, self.y, groups=groups)
        repeated = MCUVESelector(n_runs=20, n_features_to_select=3).fit(
            self.X[indices], self.y[indices], groups=groups[indices])
        self.assertEqual(repeated.n_training_sample_units_, len(self.y))
        self.assertEqual(repeated.sample_unit_, "physical_sample_means")
        np.testing.assert_allclose(original.importances_, repeated.importances_, rtol=1e-11)
        with self.assertRaises(ValueError):
            MCUVESelector().fit(self.X[:6], np.arange(6), groups=[0, 0, 1, 1, 2, 2])

    def test_multiclass_label_permutation_is_not_ordinal_regression(self):
        rng = np.random.default_rng(33)
        X = rng.normal(size=(60, 8))
        labels = np.repeat([0, 1, 2], 20)
        X[:, 0] += labels * 3
        remap = np.choose(labels, [91, -4, 7])
        for cls, options in [(MCUVESelector, {"n_runs": 15, "n_features_to_select": 3}),
                             (CARSStyleSelector, {"n_runs": 5, "reweight": False})]:
            a = cls(task="classification", **options).fit(X, labels)
            b = cls(task="classification", **options).fit(X, remap)
            np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)

    def test_iriv_inspired_small_feature_search_terminates(self):
        for p in [1, 2, 3, 4]:
            X = self.X[:, :p]
            model = BoundedIRIVInspiredSelector(n_components=2, n_masks=3, max_rounds=1).fit(X, self.y)
            self.assertGreaterEqual(len(model.selected_indices_), 1)
            self.assertLessEqual(model.evaluation_count_, 1 + p * (p + 1) // 2)
            self.assertEqual(model.rounds_completed_, 0)

    def test_iriv_inspired_bounded_random_phase_is_reproducible(self):
        options = dict(n_components=2, n_masks=3, max_rounds=1, max_initial_features=6)
        a = BoundedIRIVInspiredSelector(**options).fit(self.X, self.y)
        b = BoundedIRIVInspiredSelector(**options).fit(self.X, self.y)
        np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)
        self.assertLessEqual(a.rounds_completed_, 1)
        self.assertTrue(np.isfinite(a.selection_mse_))


if __name__ == "__main__":
    unittest.main()
