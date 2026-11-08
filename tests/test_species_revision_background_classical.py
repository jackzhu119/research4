"""Guard the background worker's specimen and mixed-modality boundaries."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/species_revision_background_classical.py"
SPEC = importlib.util.spec_from_file_location("species_revision_background_classical_tested", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


class BackgroundClassicalBoundaries(unittest.TestCase):
    def test_mixed_modality_snv_uses_two_independent_blocks(self):
        axis = np.linspace(.1, 1.5, 9)
        rng = np.random.default_rng(19)
        n, alpha = rng.normal(size=(8, 9)), rng.normal(size=(8, 9))
        transform = worker.BlockPreprocessor("snv", axis, "n_alpha")
        result = transform.transform(np.concatenate((n, alpha), axis=1))
        changed_units = transform.transform(np.concatenate((n * 1000 + 72, alpha * .001 - 90), axis=1))
        np.testing.assert_allclose(result, changed_units, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(result[:, :9].mean(axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(result[:, 9:].std(axis=1, ddof=1), 1, atol=1e-14)

    def test_only_training_values_and_targets_are_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = np.asarray([f"R{i}" for i in range(6)])
            axis = np.linspace(.1, 1.5, 9)
            X = np.arange(54, dtype=float).reshape(6, 9)
            X[4:] = np.nan  # Withheld data must never enter a fit/finite check.
            np.savez(root / "features.npz", record_ids=ids, axis=axis,
                     X_alpha=X, X_n=X + 100, species=np.array(["DO_NOT_READ"] * 6))
            rows = [{"record_id": str(identifier), "physical_sample_id": str(identifier),
                     "role": "train" if i < 4 else ("calibration" if i == 4 else "prediction"),
                     "species": "酸枣仁" if i % 2 else "枣仁", "origin_group_id": str(i % 2)}
                    for i, identifier in enumerate(ids)]
            with (root / "split.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            observed, y, selected, physical, _ = worker.load_master_train(
                root / "features.npz", root / "split.csv", "binary_kinds", "alpha")
            np.testing.assert_array_equal(selected, ids[:4])
            np.testing.assert_array_equal(observed, X[:4])
            self.assertEqual(set(y), {"枣仁", "酸枣仁"})
            np.testing.assert_array_equal(physical, selected)

    def test_feature_annotations_keep_modalities_distinct(self):
        axis = np.asarray([.1, .2, .3])
        annotations = worker.feature_annotations([1, 4], axis, "n_alpha")
        self.assertEqual([row["modality"] for row in annotations], ["n", "alpha"])
        self.assertEqual([row["frequency_THz_export_label"] for row in annotations], [.2, .2])

    def test_atomic_frozen_configuration_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "manifest.json"
            worker.checked_freeze(target, {"source": "first"})
            worker.checked_freeze(target, {"source": "first"})
            with self.assertRaises(ValueError):
                worker.checked_freeze(target, {"source": "second"})
            self.assertFalse(list(Path(temporary).glob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
