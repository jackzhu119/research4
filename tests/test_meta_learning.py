import json
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from herb_spectra.meta_learning import EpisodicEmbedding, prototype_probabilities


class MetaLearningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.X = rng.normal(size=(30, 4))
        self.y = np.repeat(np.arange(3), 10)

    def check_gradients(self, model, objective, keys):
        _, analytical = objective()
        epsilon = 1e-6
        for key in keys:
            for index in np.ndindex(model.params[key].shape):
                original = model.params[key][index]
                model.params[key][index] = original + epsilon
                high, _ = objective()
                model.params[key][index] = original - epsilon
                low, _ = objective()
                model.params[key][index] = original
                numerical = (high - low) / (2 * epsilon)
                self.assertAlmostEqual(numerical, analytical[key][index], delta=2e-7, msg=f"{key}{index}")

    def test_episode_gradients_include_support_path(self):
        model = EpisodicEmbedding(4, 3, 2, temperature=0.7, weight_decay=0.017)
        model.params["b"][:] = [0.12, -0.09]
        support = np.array([0, 1, 10, 11, 20, 21])
        query = np.array([2, 3, 4, 12, 13, 14, 22, 23, 24])
        self.check_gradients(
            model,
            lambda: model.episodic_loss_and_gradients(self.X[support], self.y[support], self.X[query], self.y[query]),
            ["W", "b"],
        )

    def test_supervised_gradients(self):
        model = EpisodicEmbedding(4, 3, 2, weight_decay=0.01)
        self.check_gradients(model, lambda: model.supervised_loss_and_gradients(self.X, self.y), ["W", "b", "V", "c"])

    def test_balanced_disjoint_episode_and_insufficient_data(self):
        model = EpisodicEmbedding(4, 3)
        for _ in range(20):
            episode = model.sample_episode(self.y, 3, 5)
            self.assertEqual(len(np.intersect1d(episode.support_indices, episode.query_indices)), 0)
            np.testing.assert_array_equal(np.bincount(self.y[episode.support_indices]), [3, 3, 3])
            np.testing.assert_array_equal(np.bincount(self.y[episode.query_indices]), [5, 5, 5])
        with self.assertRaisesRegex(ValueError, "enough distinct"):
            model.sample_episode(self.y, 6, 5)

    def test_exact_resume_in_both_training_modes(self):
        for mode in ("episodic", "supervised"):
            continuous = EpisodicEmbedding(4, 3, 5, random_state=17)
            interrupted = EpisodicEmbedding(4, 3, 5, random_state=17)
            continuous.train_chunk(self.X, self.y, 21, 2, 4, mode)
            interrupted.train_chunk(self.X, self.y, 8, 2, 4, mode)
            restored = EpisodicEmbedding.from_state_dict(json.loads(json.dumps(interrupted.state_dict())))
            restored.train_chunk(self.X, self.y, 13, 2, 4, mode)
            self.assertEqual(continuous.state_dict(), restored.state_dict())

    def test_modes_use_same_sample_draws_and_match_record_exposure(self):
        meta = EpisodicEmbedding(4, 3, 5, random_state=17)
        supervised = EpisodicEmbedding(4, 3, 5, random_state=17)
        meta_log = meta.train_chunk(self.X, self.y, 4, 2, 4, "episodic")
        supervised_log = supervised.train_chunk(self.X, self.y, 4, 2, 4, "supervised")
        self.assertEqual(meta_log["records_per_update"], supervised_log["records_per_update"])
        np.testing.assert_array_equal(meta.last_episode.support_indices, supervised.last_episode.support_indices)
        np.testing.assert_array_equal(meta.last_episode.query_indices, supervised.last_episode.query_indices)
        self.assertFalse(np.array_equal(meta.params["W"], supervised.params["W"]))

    def test_training_learns_separable_episode_task_and_predicts_from_support(self):
        rng = np.random.default_rng(123)
        # A large nuisance dimension hides the small class-specific coordinate.
        y = np.repeat(np.arange(3), 40)
        centers = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        X = np.column_stack([centers[y] + rng.normal(scale=0.15, size=(120, 2)), rng.normal(scale=1.0, size=(120, 2))])
        model = EpisodicEmbedding(4, 3, 6, learning_rate=0.01)
        support = np.r_[0:3, 40:43, 80:83]
        query = np.setdiff1d(np.arange(120), support)
        before, _ = model.episodic_loss_and_gradients(X[support], y[support], X[query], y[query])
        model.train_chunk(X, y, 120, 3, 5)
        after, _ = model.episodic_loss_and_gradients(X[support], y[support], X[query], y[query])
        self.assertLess(after, before)
        self.assertGreater((model.predict(X[query], X[support], y[support]) == y[query]).mean(), 0.95)

    def test_fixed_prototypes_use_explicit_class_order(self):
        support = np.array([[10.0], [0.0], [5.0]])
        labels = np.array([2, 0, 1])
        query = np.array([[0.1], [9.9], [5.1]])
        probability = prototype_probabilities(support, labels, query, 3)
        np.testing.assert_array_equal(probability.argmax(axis=1), [0, 2, 1])
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)

    def test_checkpoint_is_independent_and_rejects_corruption(self):
        model = EpisodicEmbedding(4, 3, 2)
        state = model.state_dict()
        state["params"]["W"][0][0] = 999.0
        self.assertNotEqual(model.params["W"][0, 0], 999.0)
        state["second_moment"]["W"][0][0] = -1
        with self.assertRaisesRegex(ValueError, "negative second moment"):
            EpisodicEmbedding.from_state_dict(state)

    def test_meta_model_cannot_silently_predict_with_untrained_linear_head(self):
        model = EpisodicEmbedding(4, 3, 2)
        model.train_chunk(self.X, self.y, 2, 1, 3, "episodic")
        with self.assertRaisesRegex(ValueError, "head has not been trained"):
            model.predict(self.X)
        model.train_chunk(self.X, self.y, 2, 1, 3, "supervised")
        self.assertEqual(model.predict(self.X).shape, (len(self.X),))

    def test_extreme_finite_gradient_cannot_poison_optimizer_checkpoint(self):
        model = EpisodicEmbedding(4, 3, 2)
        model.train_chunk(self.X, self.y, 3, 1, 3, "supervised")
        before = model.state_dict()
        # The first candidate update succeeds before the second overflows. Both
        # must remain uncommitted, including per-parameter and global step counts.
        gradients = {
            "b": np.ones_like(model.params["b"]),
            "W": np.full_like(model.params["W"], 1e308),
        }
        with self.assertRaisesRegex(FloatingPointError, "all existing optimizer state was preserved"):
            model._update(gradients)
        self.assertEqual(model.state_dict(), before)
        # The preserved state remains loadable and training can proceed.
        restored = EpisodicEmbedding.from_state_dict(json.loads(json.dumps(before)))
        restored.train_chunk(self.X, self.y, 1, 1, 3, "supervised")
        self.assertEqual(restored.step, model.step + 1)

    def test_empty_queries_return_empty_predictions(self):
        model = EpisodicEmbedding(4, 3, 2)
        empty = np.empty((0, 4))
        self.assertEqual(model.predict_proba(empty, self.X, self.y).shape, (0, 3))
        self.assertEqual(model.predict(empty, self.X, self.y).shape, (0,))
        model.train_chunk(self.X, self.y, 1, 1, 3, "supervised")
        self.assertEqual(model.predict_proba(empty).shape, (0, 3))
        self.assertEqual(model.predict(empty).shape, (0,))


if __name__ == "__main__":
    unittest.main()
