import copy
import unittest

import joblib
import numpy as np

from herb_spectra.spectral_models import fit_source_candidate
from herb_spectra.support_svm_search import (build_context, candidate_grid,
    fit_weighted_support, mixed_scores, select_candidate)


class SupportSVMTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(97)
        self.x = rng.normal(size=(30, 20))
        self.y = np.repeat(["A", "B", "C"], 10)
        self.x[:, :4] += np.repeat([-2., 0., 2.], 10)[:, None]
        self.model = fit_source_candidate({"id": "fixture", "model": "svc", "preprocessing": "snv", "C": 10.},
            self.x, self.y, np.linspace(.1, 1.5, 20))
        self.z = self.model.transform(self.x)
        self.context = build_context(self.model, self.z, self.y)

    def test_zero_support_reproduces_scores_and_fixed_source_weights(self):
        zero, metadata = fit_weighted_support(self.context)
        np.testing.assert_allclose(zero.decision_function(self.z), self.model.predict_scores(self.x), rtol=0, atol=1e-12)
        self.assertEqual(metadata["support_labels_used"], 0)
        self.assertEqual(zero._gamma, self.model.estimator_._gamma)

    def test_target_statistics_do_not_change_gamma_or_class_weights_or_source(self):
        source_before, context_before = joblib.hash(self.model), joblib.hash(self.context)
        support = self.z[[0, 10, 20]]*50
        fitted, metadata = fit_weighted_support(self.context, support, np.array(["A", "B", "C"]), support_weight=16.)
        self.assertEqual(fitted._gamma, self.model.estimator_._gamma)
        np.testing.assert_array_equal(fitted.class_weight_, self.model.estimator_.class_weight_)
        self.assertEqual(metadata["support_labels_used"], 3)
        self.assertEqual(joblib.hash(self.model), source_before)
        self.assertEqual(joblib.hash(self.context), context_before)

    def test_incomplete_or_invalid_support_rejected(self):
        for weight in [0., -1., np.nan]:
            with self.assertRaises(ValueError):
                fit_weighted_support(self.context, support_weight=weight)
        with self.assertRaises(ValueError):
            fit_weighted_support(self.context, self.z[:2], np.array(["A", "B"]))
        with self.assertRaises(ValueError):
            fit_weighted_support(self.context, self.z[:2], None)
        with self.assertRaises(ValueError):
            build_context(self.model, self.z, np.repeat("A", 30))

    def test_mixing_endpoints_and_pointwise_prediction(self):
        zero, _ = fit_weighted_support(self.context)
        old = self.model.predict_scores(self.x)
        new = zero.decision_function(self.z)
        np.testing.assert_array_equal(mixed_scores(old, new, 0), old)
        np.testing.assert_array_equal(mixed_scores(old, new, 1), new)
        np.testing.assert_allclose(mixed_scores(old, new, .25)[:1], mixed_scores(old[:1], new[:1], .25))
        with self.assertRaises(ValueError):
            mixed_scores(old, new[:1], .5)

    def _aggregate(self):
        result = {"selection": {}, "gate": {}}
        for stage in result:
            for candidate in candidate_grid():
                result[stage][candidate["id"]] = {"accuracy": .8, "balanced_accuracy": .7,
                    "macro_f1": .7, "matched_accuracy": .8, "matched_balanced_accuracy": .7,
                    "matched_macro_f1": .7, "evaluations": 10, "failures": 0}
        return result

    def test_selection_requires_gain_on_both_stages_and_matched_control(self):
        grid = candidate_grid(); aggregate = self._aggregate(); winner = grid[1]
        aggregate["selection"][winner["id"]]["accuracy"] = .82
        self.assertEqual(select_candidate(aggregate, grid)["selected"], grid[0])
        aggregate["gate"][winner["id"]]["accuracy"] = .83
        self.assertEqual(select_candidate(aggregate, grid)["selected"], winner)
        aggregate["selection"][winner["id"]]["matched_accuracy"] = .82
        self.assertEqual(select_candidate(aggregate, grid)["selected"], grid[0])

    def test_grid_and_tie_rule_prefer_smaller_blend_then_weight(self):
        grid = candidate_grid(); self.assertEqual(len(grid), 21)
        self.assertEqual(len({c["id"] for c in grid}), 21)
        aggregate = self._aggregate()
        for stage in aggregate:
            for c in grid[1:]:
                aggregate[stage][c["id"]]["accuracy"] = .82
        self.assertEqual(select_candidate(aggregate, grid)["selected"], grid[1])
        aggregate["selection"][grid[1]["id"]]["macro_f1"] = .69
        self.assertNotEqual(select_candidate(aggregate, grid)["selected"], grid[1])


if __name__ == "__main__":
    unittest.main()
