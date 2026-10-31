"""Method isolation tests; these synthetic labels are not herbal evidence."""
from copy import deepcopy
import json
import unittest

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from herb_spectra.calibration import AnchorScoreCalibrator
from herb_spectra.conservative_calibration import (build_context, candidate_grid,
    fit_candidate, mistake_residual)


def fixture():
    rng = np.random.default_rng(83)
    classes = np.asarray(["C", "A", "B"])
    y = np.repeat(classes, 12)
    X = rng.normal(size=(36, 8)) + np.repeat([-2., 0., 2.], 12)[:, None]
    scores = rng.normal(scale=.15, size=(36, 3))
    for i, label in enumerate(classes):
        scores[y == label, i] += 3.
    context = build_context(X, scores, y, classes, seed=3)
    support = np.asarray([0, 12, 24])
    incorrect = scores[support].copy()
    incorrect[1] = [3., .2, .1]
    return context, X, scores, y, support, incorrect


class ConservativeCalibrationTests(unittest.TestCase):
    def test_grid_is_stable_unique_json_safe_and_includes_identity(self):
        grid = candidate_grid()
        self.assertEqual(grid, candidate_grid())
        self.assertEqual(len(grid), 116)
        self.assertEqual(len({c["id"] for c in grid}), 116)
        self.assertEqual(grid[0]["family"], "identity")
        self.assertEqual(grid[0]["complexity_order"], 0)
        json.dumps(grid, allow_nan=False)

    def test_all_candidates_preserve_shapes_class_order_and_source_context(self):
        context, X, scores, y, support, incorrect = fixture()
        before = joblib.hash(context)
        for candidate in candidate_grid():
            adapter = fit_candidate(context, candidate, X[support], incorrect, y[support])
            values = adapter.predict_scores(X[3:11], scores[3:11])
            self.assertEqual(values.shape, (8, 3), candidate)
            self.assertTrue(np.isfinite(values).all(), candidate)
            assert_array_equal(adapter.context.classes, ["C", "A", "B"])
            self.assertEqual(joblib.hash(context), before)
        self.assertFalse(context.projection_mean.flags.writeable)
        self.assertFalse(context.source_centroids.flags.writeable)

    def test_already_correct_support_means_exact_identity_for_mistake_methods(self):
        context, X, scores, y, support, _ = fixture()
        for candidate in candidate_grid():
            if candidate["family"].startswith("mistake_"):
                adapter = fit_candidate(context, candidate, X[support], scores[support], y[support])
                self.assertEqual(adapter.metadata["support_mistakes"], 0)
                assert_array_equal(adapter.predict_scores(X, scores), scores)
        residual = mistake_residual(scores[support], y[support], context.classes)
        assert_array_equal(residual, np.zeros((3, 3)))

    def test_mistake_residual_corrects_only_true_and_wrong_winner_without_label_encoding(self):
        context, _, _, y, support, incorrect = fixture()
        residual = mistake_residual(incorrect, y[support], context.classes)
        assert_array_equal(residual[0], [0, 0, 0])
        assert_array_equal(residual[2], [0, 0, 0])
        self.assertAlmostEqual(residual[1, 0], -1.45)
        self.assertAlmostEqual(residual[1, 1], 1.45)
        self.assertEqual(residual[1, 2], 0)
        self.assertEqual(np.argmax(incorrect[1] + residual[1]), 1)

    def test_query_batch_composition_and_order_do_not_fit_statistics(self):
        context, X, scores, y, support, incorrect = fixture()
        extra_X = np.full((2, X.shape[1]), 1000.)
        extra_scores = np.asarray([[30., 2., -1.], [-1., 20., 0.]])
        for family in {c["family"] for c in candidate_grid()}:
            candidate = next(c for c in candidate_grid() if c["family"] == family)
            adapter = fit_candidate(context, candidate, X[support], incorrect, y[support])
            before = joblib.hash(adapter)
            alone = adapter.predict_scores(X[4:5], scores[4:5])
            batched = adapter.predict_scores(np.vstack([extra_X, X[4:5]]), np.vstack([extra_scores, scores[4:5]]))
            assert_allclose(alone, batched[-1:], rtol=1e-10, atol=1e-10)
            self.assertEqual(joblib.hash(adapter), before)

    def test_support_permutation_leaves_each_mapping_unchanged(self):
        context, X, scores, y, support, incorrect = fixture()
        order = np.asarray([2, 0, 1])
        for candidate in candidate_grid():
            first = fit_candidate(context, candidate, X[support], incorrect, y[support])
            second = fit_candidate(context, candidate, X[support][order], incorrect[order], y[support][order])
            assert_allclose(first.predict_scores(X, scores), second.predict_scores(X, scores),
                            rtol=1e-10, atol=1e-10, err_msg=candidate["id"])

    def test_zero_support_matched_controls_do_not_claim_labels(self):
        context, X, scores, y, support, incorrect = fixture()
        for candidate in candidate_grid():
            zero = fit_candidate(context, candidate)
            self.assertFalse(zero.metadata["uses_support_labels"])
            self.assertEqual(zero.metadata["support_count"], 0)
            if candidate["family"] == "prototype_shrinkage":
                assert_array_equal(zero.centroids, context.source_centroids)
                with_support = fit_candidate(context, candidate, X[support] + 3, incorrect, y[support])
                self.assertFalse(np.allclose(with_support.centroids, zero.centroids))
                self.assertFalse(zero.metadata["scores_are_calibrated_probabilities"])
                self.assertEqual(zero.predict_scores(X, scores).shape, scores.shape)
            else:
                assert_array_equal(zero.predict_scores(X, scores), scores)

    def test_legacy_control_exactly_reproduces_existing_algorithm(self):
        context, X, scores, y, support, incorrect = fixture()
        candidate = next(c for c in candidate_grid() if c["family"] == "legacy_anchor")
        adapter = fit_candidate(context, candidate, X[support], incorrect, y[support])
        old = AnchorScoreCalibrator(context.classes, regularization=1.)
        old.fit(incorrect, y[support])
        assert_allclose(adapter.predict_scores(X, scores), old.transform(scores), rtol=1e-12, atol=1e-12)

    def test_binary_shapes_and_invalid_annotations_parameters_are_rejected(self):
        context, X, scores, y, support, incorrect = fixture()
        keep = y != "B"
        binary = build_context(X[keep], scores[keep, :2], y[keep], ["C", "A"])
        for family in {c["family"] for c in candidate_grid()}:
            candidate = next(c for c in candidate_grid() if c["family"] == family)
            adapter = fit_candidate(binary, candidate, X[[0, 12]], scores[[0, 12], :2], y[[0, 12]])
            self.assertEqual(adapter.predict_scores(X[:2], scores[:2, :2]).shape, (2, 2))
        candidate = next(c for c in candidate_grid() if c["family"] == "mistake_ridge")
        with self.assertRaisesRegex(ValueError, "every source class"):
            fit_candidate(context, candidate, X[support[:2]], incorrect[:2], y[support[:2]])
        wrong = y[support].copy(); wrong[0] = "?"
        with self.assertRaisesRegex(ValueError, "unknown class"):
            fit_candidate(context, candidate, X[support], incorrect, wrong)
        missing = y[support].astype(object); missing[0] = None
        with self.assertRaisesRegex(ValueError, "missing label"):
            fit_candidate(context, candidate, X[support], incorrect, missing)
        for value in [0, -1, np.nan, np.inf, True]:
            bad = deepcopy(candidate); bad["regularization"] = value
            with self.assertRaises(ValueError):
                fit_candidate(context, bad, X[support], incorrect, y[support])
        with self.assertRaisesRegex(ValueError, "all support"):
            fit_candidate(context, candidate, X[support])
        bad_features = X.copy(); bad_features[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite matrix"):
            build_context(bad_features, scores, y, context.classes)


if __name__ == "__main__":
    unittest.main()
