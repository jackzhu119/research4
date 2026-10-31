import unittest

import numpy as np
import torch

from herb_spectra.neural_base_search import candidate_grid, transform_features
from herb_spectra.paper_c_strong_mlp import METHODS, fit_source_pair, verify_release


class StrongMLPTests(unittest.TestCase):
    def test_release_binds_frozen_source_selection(self):
        expected = {"selection_lock_sha256": "a", "manifest_sha256": "b", "protocol_sha256": "c"}
        with self.assertRaises(ValueError):
            verify_release({"allow_outer_evaluation": True, "released_at_utc": "now", **expected, "manifest_sha256": "wrong"}, expected)
        verify_release({"allow_outer_evaluation": True, "released_at_utc": "now", **expected}, expected)

    def test_source_singleton_remains_in_supervision_but_not_episode_tasks(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(10)
        labels = np.repeat([f"ZY{i:02d}" for i in range(1, 8)], 2)
        keep = np.arange(len(labels)) != 2
        labels = labels[keep]
        values = rng.normal(size=(len(labels), 8))
        ids = np.array([f"sample{i}" for i in range(len(labels))])
        candidate = next(c for c in candidate_grid() if c["id"] == "b006_mlp")
        prep, models, meta = fit_source_pair(values, labels, ids, candidate, 42)
        self.assertEqual(meta["base_epochs"], 200)
        self.assertEqual(meta["continuation_updates_not_epochs"], 100)
        self.assertEqual(len(meta["base_losses"]), 200)
        self.assertEqual(len(meta["continuation"]["source_tasks"]), 100)
        self.assertEqual(meta["source_class_counts"]["ZY02"], 1)
        self.assertEqual(meta["base_fingerprint"], meta["base_fingerprint_after_continuation"])
        self.assertEqual(set(models), set(METHODS))
        # Class index1 is the singleton ZY02: episode tasks exclude it.
        for task in meta["continuation"]["source_tasks"]:
            self.assertNotIn(1, task["classes"])
            self.assertFalse(set(task["support_ids"]) & set(task["query_ids"]))
        self.assertEqual(prep["source_samples"], len(labels))
        self.assertEqual(transform_features(prep, values).shape, (len(labels), 8))


if __name__ == "__main__":
    unittest.main()
