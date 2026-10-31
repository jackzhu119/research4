"""Synthetic tests of frozen latent adaptation; no herbal accuracy claims."""
from collections import Counter
import json
import unittest

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from herb_spectra.latent_adaptation import (anchored_ridge_delta,
    build_latent_context, fit_latent_adapter, latent_candidate_grid)


def fixture(binary=False):
    rng = np.random.default_rng(991)
    classes = np.asarray(["C", "A"] if binary else ["C", "A", "B"])
    counts = [20, 12] if binary else [20, 12, 8]
    labels = np.repeat(classes, counts)
    centers = rng.normal(size=(len(classes), 8)) * 1.5
    features = np.vstack([rng.normal(size=(count, 8)) + centers[i]
                          for i, count in enumerate(counts)])
    scores = rng.normal(scale=.2, size=(len(labels), len(classes)))
    for index, label in enumerate(classes):
        scores[labels == label, index] += 2.5
    support = np.asarray([sum(counts[:i]) for i in range(len(classes))])
    context = build_latent_context(features, scores, labels, classes, seed=42)
    return context, features, scores, labels, support


def representatives():
    return [next(c for c in latent_candidate_grid() if c["family"] == family)
            for family in ["identity", "anchored_ridge", "local_memory", "mahalanobis_centroid"]]


class LatentAdaptationTests(unittest.TestCase):
    def test_grid_is_stable_complete_unique_and_json_safe(self):
        grid = latent_candidate_grid()
        self.assertEqual(grid, latent_candidate_grid())
        self.assertEqual(len(grid), 60)
        self.assertEqual(len({item["id"] for item in grid}), 60)
        self.assertEqual(grid[0]["family"], "identity")
        self.assertEqual(Counter(item["family"] for item in grid),
                         {"identity": 1, "anchored_ridge": 27,
                          "local_memory": 27, "mahalanobis_centroid": 5})
        json.dumps(grid, allow_nan=False)

    def test_all_candidates_return_finite_aligned_scores_without_source_mutation(self):
        context, features, scores, labels, support = fixture()
        before = joblib.hash(context)
        for candidate in latent_candidate_grid():
            with self.subTest(candidate=candidate["id"]):
                adapter = fit_latent_adapter(context, candidate, features[support],
                                             scores[support], labels[support])
                predicted = adapter.predict_scores(features[3:12], scores[3:12])
                self.assertEqual(predicted.shape, (9, 3))
                self.assertTrue(np.isfinite(predicted).all())
                assert_array_equal(adapter.context.classes, ["C", "A", "B"])
                self.assertEqual(joblib.hash(context), before)

    def test_no_support_has_explicit_matching_control_and_identity_is_exact(self):
        context, features, scores, labels, support = fixture()
        for candidate in representatives():
            zero = fit_latent_adapter(context, candidate)
            self.assertFalse(zero.metadata["uses_support_labels"])
            self.assertEqual(zero.metadata["support_count"], 0)
            predicted = zero.predict_scores(features, scores)
            self.assertEqual(predicted.shape, scores.shape)
            if candidate["family"] == "identity":
                assert_array_equal(predicted, scores)
            else:
                repeated = fit_latent_adapter(context, candidate)
                assert_allclose(predicted, repeated.predict_scores(features, scores),
                                rtol=1e-12, atol=1e-12)
        ridge = next(c for c in latent_candidate_grid() if c["family"] == "anchored_ridge")
        zero_ridge = fit_latent_adapter(context, ridge)
        # Its zero-support reference is the source ridge head, not an OVR map.
        self.assertFalse(np.allclose(zero_ridge.predict_scores(features, scores), scores))
        assert_array_equal(zero_ridge.delta, np.zeros_like(context.source_head))
        assert_array_equal(zero_ridge.head_coefficients, context.source_head)
        assert_allclose(zero_ridge.predict_scores(features, scores),
                        context.design(features) @ context.source_head,
                        rtol=1e-12, atol=1e-12)

    def test_source_projection_statistics_are_consistent_in_high_dimension_and_rare_class(self):
        rng = np.random.default_rng(92)
        features = rng.normal(size=(41, 80))
        labels = np.repeat(["A", "B", "C"], [25, 1, 15])
        scores = rng.normal(size=(41, 3))
        context = build_latent_context(features, scores, labels, ["A", "B", "C"])
        assert_allclose(context.project(features), context.source_latent,
                        rtol=1e-12, atol=1e-12)
        assert_allclose(context.source_latent.mean(axis=0), 0., atol=1e-12)
        assert_allclose(context.source_latent.std(axis=0), 1., atol=1e-12)
        self.assertEqual(context.source_class_counts, (25, 1, 15))
        support = np.asarray([0, 25, 26])
        for candidate in latent_candidate_grid():
            fitted = fit_latent_adapter(context, candidate, features[support],
                                        scores[support], labels[support])
            self.assertTrue(np.isfinite(fitted.predict_scores(features[:4], scores[:4])).all())

    def test_anchor_loss_normalization_is_invariant_to_row_repetition_and_weight_mass(self):
        context, features, scores, labels, support = fixture()
        design = context.source_design
        support_design = context.design(features[support])
        onehot = np.eye(len(context.classes))
        kwargs = dict(source_anchor=1., ridge=.1, support_weight=5.)
        delta = anchored_ridge_delta(design, context.source_weights, support_design,
                                     onehot, context.source_head, **kwargs)
        source_repeated = anchored_ridge_delta(
            np.repeat(design, 2, axis=0), np.repeat(context.source_weights, 2),
            support_design, onehot, context.source_head, **kwargs)
        support_repeated = anchored_ridge_delta(
            design, context.source_weights, np.repeat(support_design, 3, axis=0),
            np.repeat(onehot, 3, axis=0), context.source_head, **kwargs)
        weights_rescaled = anchored_ridge_delta(design, context.source_weights * 123,
            support_design, onehot, context.source_head, **kwargs)
        assert_allclose(delta, source_repeated, rtol=1e-11, atol=1e-11)
        assert_allclose(delta, support_repeated, rtol=1e-11, atol=1e-11)
        assert_allclose(delta, weights_rescaled, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(float(context.source_weights.sum()), 1.)
        strong = anchored_ridge_delta(design, context.source_weights, support_design,
            onehot, context.source_head, source_anchor=1e6, ridge=.1, support_weight=5.)
        normal_source_change = np.sum(context.source_weights[:, None] * (design @ delta) ** 2)
        strong_source_change = np.sum(context.source_weights[:, None] * (design @ strong) ** 2)
        self.assertLess(strong_source_change, normal_source_change)

    def test_support_permutation_does_not_change_fitted_predictor(self):
        context, features, scores, labels, support = fixture()
        reverse = support[::-1]
        for candidate in latent_candidate_grid():
            first = fit_latent_adapter(context, candidate, features[support],
                                      scores[support], labels[support])
            permuted = fit_latent_adapter(context, candidate, features[reverse],
                                         scores[reverse], labels[reverse])
            assert_allclose(first.predict_scores(features[4:17], scores[4:17]),
                            permuted.predict_scores(features[4:17], scores[4:17]),
                            rtol=1e-9, atol=1e-9, err_msg=candidate["id"])

    def test_query_batch_composition_and_order_never_refit_source_statistics(self):
        context, features, scores, labels, support = fixture()
        extra_features = np.full((2, features.shape[1]), 1000.)
        extra_scores = np.asarray([[20., -3., 2.], [-4., 10., 1.]])
        for candidate in representatives():
            adapter = fit_latent_adapter(context, candidate, features[support],
                                         scores[support], labels[support])
            before_context = joblib.hash(context)
            before_adapter = joblib.hash(adapter)
            alone = adapter.predict_scores(features[7:8], scores[7:8])
            together = adapter.predict_scores(np.vstack([extra_features, features[7:8]]),
                                             np.vstack([extra_scores, scores[7:8]]))
            assert_allclose(alone, together[-1:], rtol=1e-9, atol=1e-9)
            normal = adapter.predict_scores(features[4:10], scores[4:10])
            reverse = adapter.predict_scores(features[4:10][::-1], scores[4:10][::-1])
            assert_allclose(normal, reverse[::-1], rtol=1e-9, atol=1e-9)
            self.assertEqual(joblib.hash(context), before_context)
            self.assertEqual(joblib.hash(adapter), before_adapter)

    def test_binary_class_order_and_all_candidates_are_supported(self):
        context, features, scores, labels, support = fixture(binary=True)
        for candidate in latent_candidate_grid():
            adapter = fit_latent_adapter(context, candidate, features[support],
                                         scores[support], labels[support])
            predicted = adapter.predict_scores(features, scores)
            self.assertEqual(predicted.shape, (len(labels), 2))
            self.assertTrue(np.isfinite(predicted).all())
            assert_array_equal(adapter.context.classes, ["C", "A"])

    def test_class_label_permutation_reorders_outputs_without_ordinal_encoding(self):
        context, features, scores, labels, support = fixture()
        order = np.asarray([2, 0, 1])
        reordered = build_latent_context(features, scores[:, order], labels,
                                          context.classes[order], seed=42)
        for candidate in representatives():
            first = fit_latent_adapter(context, candidate, features[support],
                                      scores[support], labels[support])
            second = fit_latent_adapter(reordered, candidate, features[support],
                                       scores[support][:, order], labels[support])
            assert_allclose(first.predict_scores(features, scores)[:, order],
                            second.predict_scores(features, scores[:, order]),
                            rtol=1e-9, atol=1e-9, err_msg=candidate["family"])

    def test_invalid_class_labels_partial_support_shapes_and_nonfinite_inputs_rejected(self):
        context, features, scores, labels, support = fixture()
        candidate = representatives()[1]
        unknown = labels[support].copy(); unknown[0] = "unknown"
        with self.assertRaises(ValueError):
            fit_latent_adapter(context, candidate, features[support], scores[support], unknown)
        with self.assertRaises(ValueError):
            fit_latent_adapter(context, candidate, features[support[:2]],
                               scores[support[:2]], labels[support[:2]])
        with self.assertRaises(ValueError):
            fit_latent_adapter(context, candidate, features[support], None, None)
        with self.assertRaises(ValueError):
            fit_latent_adapter(context, candidate, features[support, :-1],
                               scores[support], labels[support])
        bad_scores = scores[support].copy(); bad_scores[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_latent_adapter(context, candidate, features[support], bad_scores, labels[support])
        bad_features = features.copy(); bad_features[0, 0] = np.inf
        with self.assertRaises(ValueError):
            build_latent_context(bad_features, scores, labels, context.classes)
        with self.assertRaises(ValueError):
            build_latent_context(features, scores[:, :2], labels, context.classes)
        adapter = fit_latent_adapter(context, candidate)
        with self.assertRaises(ValueError):
            adapter.predict_scores(features[:2, :-1], scores[:2])
        with self.assertRaises(ValueError):
            adapter.predict_scores(features[:2], scores[:3])


if __name__ == "__main__":
    unittest.main()
