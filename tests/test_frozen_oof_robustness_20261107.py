"""Small synthetic, fit-free checks; never load real models or raw spectra."""
import importlib.util
import gzip
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_frozen_oof_robustness_20261107.py"
spec = importlib.util.spec_from_file_location("frozen_robustness", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FrozenRobustnessTests(unittest.TestCase):
    def test_finite_cases_and_all_stochastic_seeds(self):
        cases = m.conditions()
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({r["case"] for r in cases}), 18)
        for level in (.001, .005, .01):
            self.assertEqual([r["seed"] for r in cases if r["kind"] == "noise" and r["level"] == level], [42, 43, 44])
        self.assertEqual(len(m.conditions(False)), 16)
        self.assertFalse(any(r["kind"] == "axis_shift" for r in m.conditions(False)))

    def test_source_sd_uses_only_provided_source(self):
        source = np.array([[1., 10.], [3., 14.]])
        expected = np.sqrt((1. + 4.) / 2)
        self.assertEqual(m.source_sd(source), expected)
        query = np.ones((2, 2)) * 1e9
        query[:] = -1e10  # No query argument can affect the amplitude calculation.
        self.assertEqual(m.source_sd(source), expected)
        with self.assertRaises(m.Blocked):
            m.source_sd(np.ones((3, 2)))

    def test_identity_preserves_negative_alpha_and_input_bytes(self):
        x = np.array([[-5., 0., 3.], [4., -2., 7.]])
        saved = x.copy()
        y = m.corrupt(x, np.array([.1, .2, .3]), 2., m.conditions()[0], 0)
        np.testing.assert_array_equal(y, saved)
        np.testing.assert_array_equal(x, saved)
        self.assertIsNot(x, y)

    def test_noise_stream_is_seeded_fold_specific_and_paired_across_levels(self):
        x = np.zeros((4, 5)); axis = np.arange(5.)
        a = {"kind": "noise", "level": .001, "seed": 42}
        b = {**a, "level": .005}
        y = m.corrupt(x, axis, 3., a, 0)
        np.testing.assert_array_equal(y, m.corrupt(x, axis, 3., a, 0))
        np.testing.assert_allclose(m.corrupt(x, axis, 3., b, 0), y * 5, atol=1e-16)
        self.assertFalse(np.array_equal(y, m.corrupt(x, axis, 3., a, 1)))
        self.assertFalse(np.array_equal(y, m.corrupt(x, axis, 3., {**a, "seed": 43}, 0)))

    def test_gain_and_baseline_shape_formulas(self):
        x = np.array([[1., 2., 3.]])
        axis = np.array([.1, .2, .5])
        np.testing.assert_allclose(m.corrupt(x, axis, 2., {"kind": "gain", "level": .9}, 0), x * .9)
        np.testing.assert_allclose(m.corrupt(x, axis, 2., {"kind": "constant_baseline", "level": -.05}, 0), x - .1)
        expected = x + .1 * np.array([-1., -.5, 1.])
        np.testing.assert_allclose(m.corrupt(x, axis, 2., {"kind": "linear_baseline", "level": .05}, 0), expected)

    def test_axis_shift_sign_full_grid_and_endpoint_policy(self):
        axis = np.array([.1, .2, .3, .4])
        x = np.array([[1., 2., 3., 4.]])
        y = m.corrupt(x, axis, 9., {"kind": "axis_shift", "level": .05}, 0)
        np.testing.assert_allclose(y, [[1.5, 2.5, 3.5, 4.]])
        z = m.corrupt(x, axis, 9., {"kind": "axis_shift", "level": -.05}, 0)
        np.testing.assert_allclose(z, [[1., 1.5, 2.5, 3.5]])
        with self.assertRaises(m.Blocked):
            m.corrupt(x, axis, 2., {"kind": "unknown", "level": 1}, 0)

    def test_exact_replay_checks_identity_fold_predictions_and_scores(self):
        ref = [{"record_id": "a", "outer_fold": "0", "y_predicted": "A", "score_A": "1.25", "score_B": "0.1"}]
        pred = [{**ref[0], "outer_fold": 0, "score_A": 1.25, "score_B": .1}]
        self.assertEqual(m.verify_replay(pred, ref)["identical_predictions"], 1)
        for change in ({"record_id": "b"}, {"outer_fold": 1}, {"y_predicted": "B"}, {"score_A": 1.2501}, {"score_A": float("nan")}):
            with self.assertRaises(m.Blocked):
                m.verify_replay([{**pred[0], **change}], ref)
        with self.assertRaises(m.Blocked):
            m.verify_replay(pred * 2, ref)

    def test_confusion_independent_arithmetic_and_rare_class_denominator(self):
        result = m.metric_counts(["A", "A", "A", "B", "B"], ["A", "B", "A", "A", "B"], ["A", "B"])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 1]])
        self.assertEqual(result["accuracy"], .6)
        self.assertAlmostEqual(result["balanced_accuracy"], (2 / 3 + .5) / 2)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + .5) / 2)
        self.assertEqual(result["per_class"][1]["support"], 2)
        with self.assertRaises(m.Blocked):
            m.metric_counts(["A"], ["A"], ["A", "B"])
        with self.assertRaises(m.Blocked):
            m.metric_counts(["A", "B"], ["A", "C"], ["A", "B"])

    def test_split_identity_once_and_overlap_rejection(self):
        ids = [f"i{i}" for i in range(855)]
        splits = []
        for fold in range(5):
            query = ids[fold * 171:(fold + 1) * 171]
            splits.append({"fold": fold, "train_sample_ids": [i for i in ids if i not in query], "test_sample_ids": query})
        m.validate_splits(splits, ids)
        splits[0]["test_sample_ids"][0] = splits[0]["train_sample_ids"][0]
        with self.assertRaises(m.Blocked):
            m.validate_splits(splits, ids)

    def test_original_mask_then_rows_layout_and_training_fingerprint(self):
        full = np.arange(200., dtype=float).reshape(10, 20)
        mask = np.arange(20) % 2 == 0
        source = np.array([0, 2, 4, 6])
        historical = full[:, mask][source]
        actual = m.model_band(full[source], mask)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, historical)
        self.assertEqual(m.joblib.hash(actual), m.joblib.hash(historical))

    def test_inference_preserves_native_vote_instead_of_score_argmax(self):
        class FakeFrozenModel:
            classes_ = np.array([f"ZY{i:02d}" for i in range(1, 8)])
            def predict(self, x):
                return np.array(["ZY02"] * len(x))
            def predict_scores(self, x):
                return np.tile(np.arange(7., 0., -1.), (len(x), 1))
        rows = m.native_rows(FakeFrozenModel(), np.ones((2, 3)), ["i0", "i1"], 2)
        self.assertEqual([r["y_predicted"] for r in rows], ["ZY02", "ZY02"])
        self.assertEqual(rows[0]["score_ZY01"], 7.)
        self.assertNotIn("y_true", rows[0])

    def test_fit_blockers_reject_real_entry_points_without_fitting(self):
        from herb_spectra import staged_nested_search as staged
        with m.fit_blockers():
            with self.assertRaises(m.Blocked):
                staged.fit_source_candidate(None, None, None, None)
            with self.assertRaises(m.Blocked):
                staged.StandardScaler().fit(np.ones((2, 2)))
            with self.assertRaises(m.Blocked):
                staged.SVC().fit(np.ones((2, 2)), [0, 1])

    def test_model_provenance_allows_only_official_recipe_id_alias(self):
        from herb_spectra.staged_nested_search import StagedSVM
        source = np.array([[1., 2.], [3., 4.]])
        config = {"id": "role_A", "preprocessing": "snv", "mc": {"runs": 10}, "svm": {"C": 100.}}
        metadata = {"source_value_fingerprint": m.joblib.hash(source), "prediction_rule": "SVC standard vote, break_ties=False"}
        model = StagedSVM(config, None, None, metadata)
        m.validate_native_model(model, {**config, "id": "role_B"}, source)
        with self.assertRaises(m.Blocked):
            m.validate_native_model(model, {**config, "svm": {"C": 10.}}, source)
        with self.assertRaises(m.Blocked):
            m.validate_native_model(model, config, source[::-1])

    def test_release_and_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.py"; source.write_text("frozen")
            reference = root / "ref.csv"; reference.write_text("inference only")
            protocol = root / "protocol.json"
            m.write_json(protocol, {"schema": m.SCHEMA, "conditions": m.conditions(False),
                "axis_shift_status": "BLOCKED_UNKNOWN_FREQUENCY_UNIT", "input_code_hashes": {"source.py": m.sha(source)},
                "historical_versions": {}, "reference_file": "ref.csv", "reference_sha256": m.sha(reference)})
            release = root / "release.json"
            m.write_json(release, {"approved": False, "protocol_sha256": m.sha(protocol)})
            with self.assertRaises(m.Blocked):
                m.check_frozen(root, protocol, release)
            m.write_json(release, {"approved": True, "protocol_sha256": m.sha(protocol)})
            self.assertEqual(m.check_frozen(root, protocol, release)["schema"], m.SCHEMA)
            source.write_text("changed")
            with self.assertRaises(m.Blocked):
                m.check_frozen(root, protocol, release)

    def test_raw_inference_loader_needs_no_truth_or_label_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label_free.npz"
            ids = np.array([f"sample-{i:04d}" for i in range(855)][::-1])
            values = np.arange(855 * 3, dtype=float).reshape(855, 3)
            np.savez(path, X_alpha=values, axis=np.array([.1, .2, .3]), record_ids=ids, groups=ids)
            x, axis, ordered = m.raw_arrays(path)
            self.assertEqual(x.shape, (855, 3))
            self.assertEqual(ordered[0], "sample-0000")
            np.testing.assert_array_equal(x[0], values[-1])

    def test_prediction_only_reference_and_lock_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = root / "core"; core.mkdir()
            output = root / "results"; output.mkdir()
            (output / "predictions").mkdir()
            labels = ["ZY01"] * 350 + ["ZY02"] * 5
            for group in range(3, 8):
                labels.extend([f"ZY{group:02d}"] * 100)
            truth_rows, predictions = [], []
            for i, label in enumerate(labels):
                predicted = "ZY03" if i < 34 else label
                base = {"record_id": f"s{i:04d}", "outer_fold": i // 171, "y_predicted": predicted,
                        **{f"score_ZY{c:02d}": float(c) for c in range(1, 8)}}
                predictions.append(base)
                truth_rows.append({"method": "R2", **base, "y_true": label, "is_correct": int(label == predicted)})
            truth_path = core / "outer_predictions.csv.gz"
            m.write_csv(truth_path, truth_rows)
            ids = [r["record_id"] for r in predictions]
            splits = [{"fold": fold, "test_sample_ids": ids[fold * 171:(fold + 1) * 171],
                "train_sample_ids": ids[:fold * 171] + ids[(fold + 1) * 171:]} for fold in range(5)]
            m.write_json(core / "splits.json", splits)
            reference = m.baseline_reference(truth_path)
            self.assertEqual(len(reference), 855)
            self.assertFalse(any("y_true" in r or "is_correct" in r for r in reference))
            paths = []
            for case in m.conditions():
                path = output / "predictions" / (case["case"] + ".csv.gz")
                m.write_csv(path, predictions)
                paths.append(path)
            plan = {"historical_core": "core", "conditions": m.conditions(), "expected_historical_baseline_correct": 821}
            protocol_sha = "a" * 64
            with self.assertRaises(FileNotFoundError):
                m.score_locked(root, plan, output, protocol_sha)
            m.write_json(output / "baseline_replay.json", {"status": "PASS_EXACT_HISTORICAL_PREDICTIONS", "identical_predictions": 855, "scores_max_abs_difference": 0.})
            for name in ["source_fold_amplitudes.csv", "corruption_amplitude_diagnostics.csv", "axis_boundary_diagnostic.json"]:
                (output / name).write_text("synthetic audit placeholder")
            paths.extend(output / name for name in ["baseline_replay.json", "source_fold_amplitudes.csv", "corruption_amplitude_diagnostics.csv", "axis_boundary_diagnostic.json"])
            all_hashes = {str(p.relative_to(output)): m.sha(p) for p in paths}
            lock = {"status": m.LOCK_STATUS, "protocol_sha256": protocol_sha, "files_sha256": all_hashes}
            # Coverage must reject incomplete/empty mappings before any gzip
            # read, especially before the historical y_true file can open.
            for mapping in ({}, {k: v for k, v in all_hashes.items() if k != "predictions/baseline.csv.gz"}):
                m.write_json(output / "prediction_lock.json", {**lock, "files_sha256": mapping})
                with patch.object(m.gzip, "open", side_effect=AssertionError("No prediction/truth file should open")):
                    with self.assertRaises(m.Blocked):
                        m.score_locked(root, plan, output, protocol_sha)
            m.write_json(output / "prediction_lock.json", {**lock, "protocol_sha256": "b" * 64})
            with patch.object(m.gzip, "open", side_effect=AssertionError("No truth access with mismatched protocol")):
                with self.assertRaises(m.Blocked):
                    m.score_locked(root, plan, output, protocol_sha)
            m.write_json(output / "prediction_lock.json", lock)
            scored = m.score_locked(root, plan, output, protocol_sha)
            self.assertEqual(scored["historical_baseline_correct"], 821)
            self.assertEqual(scored["class_counts"]["ZY02"], 5)
            self.assertEqual(scored["prediction_rows_reused_samples"], 18 * 855)
            with (output / "metric_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(float(r["paired_baseline_change_accuracy"]) == 0. for r in rows))
            paths[0].write_bytes(b"mutated")
            with self.assertRaises(m.Blocked):
                m.score_locked(root, plan, output, protocol_sha)


if __name__ == "__main__":
    unittest.main()
