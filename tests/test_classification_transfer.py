"""Protocol tests on synthetic domains, not medicinal-plant performance."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from herb_spectra.classification import aggregate_samples, build_pipeline, class_scores, representation
from herb_spectra.classification_transfer import (METHODS, TransferProtocolError, checked_partition,
    fit_adaptation, fit_source, fixed_partition, nested_support_draws, prepare_domains,
    run_classification_transfer, score_predictions)


def domain(role, count_per_class=12, shift=0., seed=1):
    rng = np.random.default_rng(seed)
    y = np.repeat(["A", "B"], count_per_class)
    X = rng.normal(size=(len(y), 8)) + (y == "B")[:, None] * np.linspace(.2, 2., 8) + shift
    ids = np.asarray([f"{role}-{i}" for i in range(len(y))])
    return {"X_alpha": X, "X_n": X / 3 + .5, "axis": np.linspace(.1, 1.5, 8),
            "record_ids": ids, "groups": ids.copy(), "y": y,
            "batch_ids": np.repeat(role + "-batch", len(y)), "labels_verified": np.ones(len(y), dtype=bool)}


class ClassificationTransferTests(unittest.TestCase):
    def test_unknown_batches_and_overlapping_physical_samples_are_rejected(self):
        source, dev, target = domain("s"), domain("d", seed=2), domain("t", seed=3)
        blank = deepcopy(target); blank["batch_ids"][:] = ""
        with self.assertRaisesRegex(TransferProtocolError, "lack verified batch"):
            prepare_domains(source, dev, blank)
        overlap = deepcopy(target); overlap["groups"][0] = source["groups"][0]
        with self.assertRaisesRegex(TransferProtocolError, "overlapping physical"):
            prepare_domains(source, dev, overlap)
        overlap = deepcopy(target); overlap["batch_ids"][:] = source["batch_ids"][0]
        with self.assertRaisesRegex(TransferProtocolError, "overlapping batch"):
            prepare_domains(source, dev, overlap)
        copied = deepcopy(target)
        copied["X_alpha"][0] = source["X_alpha"][0]
        copied["X_n"][0] = source["X_n"][0]
        with self.assertRaisesRegex(TransferProtocolError, "identical paired spectra"):
            prepare_domains(source, dev, copied)

    def test_frequency_grid_or_known_class_mismatch_is_rejected(self):
        source, dev, target = domain("s"), domain("d", seed=2), domain("t", seed=3)
        target["axis"] = target["axis"] + .01
        with self.assertRaisesRegex(TransferProtocolError, "frequency grid"):
            prepare_domains(source, dev, target)
        target = domain("t", seed=3); target["y"][target["y"] == "B"] = "C"
        with self.assertRaisesRegex(TransferProtocolError, "same known classes"):
            prepare_domains(source, dev, target)

    def test_fixed_queries_nested_supports_and_unavailable_k_do_not_shrink_query(self):
        data = domain("t", count_per_class=24)
        split = fixed_partition(data["y"], data["groups"], minimum_query_per_class=10, seed=5)
        draws, unavailable = nested_support_draws(data["y"], split, [1, 3, 10, 13], repeats=3, seed=6)
        self.assertEqual(set(unavailable), {13})
        for repeat in range(3):
            selected = {d["shots_per_class"]: d for d in draws if d["repeat"] == repeat}
            self.assertTrue(set(selected[1]["support"]) < set(selected[3]["support"]) < set(selected[10]["support"]))
            for draw in selected.values():
                assert_array_equal(draw["query"], split["query"])
                self.assertFalse(set(draw["support"]) & set(draw["query"]))
                self.assertEqual(len(draw["query"]), 24)
        invalid = {"query": np.array([0, 1]), "support_pool": np.arange(len(data["y"]))}
        with self.assertRaisesRegex(TransferProtocolError, "overlaps"):
            checked_partition(invalid, len(data["y"]))

    def test_binary_scores_and_complete_retrain_are_distinct_from_frozen_head(self):
        source = aggregate_samples(domain("s")); target = aggregate_samples(domain("t", shift=8., seed=3))
        X, blocks = representation(source)
        Xt, _ = representation(target)
        pipe = build_pipeline(block_sizes=blocks).fit(X, source["y"])
        score = class_scores(pipe, Xt[:4])
        assert_allclose(score[:, 0], -score[:, 1])
        source_mean = pipe.named_steps["scale"].mean_.copy()
        support = np.asarray([np.flatnonzero(target["y"] == label)[0] for label in np.unique(target["y"])])
        full, full_meta = fit_adaptation("source_plus_support_full_retrain", pipe, X, source["y"],
                                        Xt[support], target["y"][support])
        frozen, frozen_meta = fit_adaptation("frozen_representation_head", pipe, X, source["y"],
                                            Xt[support], target["y"][support])
        self.assertTrue(full_meta["transforms_refit"]); self.assertFalse(frozen_meta["transforms_refit"])
        self.assertEqual(full_meta["training_samples"], len(X) + 2)
        self.assertEqual(frozen_meta["training_samples"], 2)
        assert_allclose(pipe.named_steps["scale"].mean_, source_mean)
        self.assertFalse(np.allclose(full.named_steps["scale"].mean_, source_mean))
        self.assertIs(frozen.source_pipeline, pipe)

    def test_target_query_label_change_changes_scores_but_not_fit_or_predictions(self):
        source, dev, target = domain("s", seed=1), domain("d", shift=.3, seed=2), domain("t", shift=.6, seed=3)
        aggregated = aggregate_samples(target)
        frozen = fixed_partition(aggregated["y"], aggregated["sample_ids"], minimum_query_per_class=2, seed=7)
        kwargs = dict(shots=(1, 3), repeats=2, minimum_query_per_class=2, source_cv_splits=2,
                      regularizations=(.1, 10.), seed=42,
                      source_grid={"preprocess__mode": ["raw"], "model__C": [1.]}, target_partition=frozen)
        first = run_classification_transfer(source, dev, target, **kwargs)
        changed = deepcopy(target)
        query_ids = set(aggregated["sample_ids"][frozen["query"]].tolist())
        for i, identifier in enumerate(changed["groups"]):
            if identifier in query_ids:
                changed["y"][i] = "B" if changed["y"][i] == "A" else "A"
        second = run_classification_transfer(source, dev, changed, **kwargs)
        self.assertEqual(first["source"]["best_params"], second["source"]["best_params"])
        self.assertEqual(first["selected_anchor_regularization_by_shots"], second["selected_anchor_regularization_by_shots"])
        self.assertEqual(first["partitions"], second["partitions"])
        keys = ["method", "shots_per_class", "repeat", "sample_id", "y_predicted"]
        self.assertEqual([{k: row[k] for k in keys} for row in first["prediction_rows"]],
                         [{k: row[k] for k in keys} for row in second["prediction_rows"]])
        self.assertEqual({row["method"] for row in first["rows"] if row["status"] == "pass"}, set(METHODS))
        self.assertNotEqual([row["metrics"] for row in first["rows"]], [row["metrics"] for row in second["rows"]])

    def test_multiclass_source_predict_and_score_argmax_use_one_tie_policy(self):
        rng = np.random.default_rng(34)
        source = domain("s", count_per_class=18)
        source["y"] = np.repeat(["A", "B", "C"], 12)
        source["X_alpha"] = rng.normal(size=(36, 8))
        source["X_n"] = source["X_alpha"] / 3 + .5
        source = aggregate_samples(source)
        pipeline, metadata = fit_source(source, cv_splits=2,
            grid={"preprocess__mode": ["raw"], "model__C": [1.]})
        self.assertTrue(pipeline.get_params()["model__break_ties"])
        self.assertEqual(metadata["tie_policy"], "argmax_ovr_scores")
        query = rng.normal(size=(100, 8))
        assert_array_equal(pipeline.predict(query), score_predictions(pipeline, query))
        with patch.object(pipeline.named_steps["model"], "decision_function", return_value=np.zeros((2, 3))):
            exact_ties = np.zeros((2, 8))
            assert_array_equal(pipeline.predict(exact_ties), ["A", "A"])
            assert_array_equal(score_predictions(pipeline, exact_ties), ["A", "A"])

    def test_cli_preflight_writes_blocked_status_without_training(self):
        import json
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for role in ("source", "development", "target"):
                data = domain(role, seed=len(role)); data["batch_ids"][:] = ""
                np.savez_compressed(folder / (role + ".npz"), **data)
            result = subprocess.run([sys.executable, str(root / "scripts/run_classification_transfer.py"),
                "--source", str(folder / "source.npz"), "--development", str(folder / "development.npz"),
                "--target", str(folder / "target.npz"), "--output", str(folder / "result"), "--preflight-only"],
                cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            status = json.loads((folder / "result/status.json").read_text())
            self.assertEqual(status["status"], "blocked")
            self.assertIn("verified batch", status["reason"])
            self.assertFalse((folder / "result/results.json").exists())


if __name__ == "__main__":
    unittest.main()
