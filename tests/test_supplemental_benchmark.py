"""Synthetic fixtures only; no supplemental/historical scientific data is fit."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from herb_spectra.leakage_guard import LeakageError, array_fingerprint
from herb_spectra.paper_c_calibration import candidate_grid
from herb_spectra.selection import MCUVESelector
from herb_spectra.spectral_models import SpectralPreprocessor
from herb_spectra.supplemental_benchmark import (
    apply_changed_scores, calibrate_scores, fit_source_model, load_role_pack,
    predict_numeric, preflight_role_packs, score_predictions, validate_pack,
    write_new_json,
)


class SupplementalPreflightTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261104)
        self.axis = np.linspace(.1, 1.5, 18)

    def pack(self, role, number=9):
        result = {"X": self.rng.normal(size=(number, len(self.axis))), "axis": self.axis.copy(),
                  "record_ids": np.array([f"{role}-{i}" for i in range(number)], dtype="U80"),
                  "entity_ids": np.array([f"entity-{role}-{i}" for i in range(number)], dtype="U80"),
                  "batch_ids": np.array(["batch-user-declared"] * number),
                  "legacy_match_ids": np.array(["[]"] * number)}
        if role != "query":
            result["y"] = np.array(["A", "B", "C"] * (number // 3))
        return result

    def roles(self):
        return {r: self.pack(r) for r in ("source", "support", "query")}

    def test_valid_preflight_does_not_claim_confirmatory_evidence(self):
        summary = preflight_role_packs(self.roles())
        self.assertEqual(summary["classes"], ["A", "B", "C"])
        self.assertFalse(summary["confirmation_eligible"])
        self.assertEqual(summary["support_counts"], {"A": 3, "B": 3, "C": 3})

    def test_query_labels_or_nonfinite_features_are_rejected(self):
        packs = self.roles()
        packs["query"]["labels"] = np.array(["A"] * 9)
        with self.assertRaisesRegex(LeakageError, "truth labels"):
            preflight_role_packs(packs)
        packs = self.roles()
        packs["query"]["X"][0, 0] = np.nan
        with self.assertRaisesRegex(LeakageError, "finite"):
            preflight_role_packs(packs)

    def test_cross_role_entity_legacy_and_same_spectrum_rejected(self):
        for kind in ("entity_ids", "legacy_match_ids", "X"):
            with self.subTest(kind=kind):
                packs = self.roles()
                if kind == "legacy_match_ids":
                    packs["source"][kind] = np.array([json.dumps(["old-ZY01-1"])] + ["[]"] * 8)
                    packs["query"][kind] = np.array([json.dumps(["old-ZY01-1"])] + ["[]"] * 8)
                else:
                    packs["query"][kind][0] = packs["source"][kind][0]
                with self.assertRaisesRegex(LeakageError, "Repeated"):
                    preflight_role_packs(packs)

    def test_changed_axis_or_pack_bytes_and_object_pickle_rejected(self):
        packs = self.roles()
        packs["support"]["axis"][0] += .0001
        with self.assertRaisesRegex(LeakageError, "frequency axes"):
            preflight_role_packs(packs)
        with self.assertRaisesRegex(LeakageError, "Frequency axis"):
            validate_pack(self.pack("source"), role="source", expected_axis_sha256="wrong")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.npz"
            np.savez(path, **self.pack("source"))
            with self.assertRaisesRegex(LeakageError, "pack bytes"):
                load_role_pack(path, role="source", expected_sha256="wrong")
        pack = self.pack("source")
        pack["metadata"] = np.array([{"unsafe": True}], dtype=object)
        with self.assertRaisesRegex(LeakageError, "Object"):
            validate_pack(pack, role="source")

    def test_cross_namespace_physical_aliases_block_even_with_different_spectra(self):
        packs = self.roles()
        packs["source"]["record_ids"][0] = "OLD-HQ03-S0001"
        packs["source"]["entity_ids"][0] = "OLD-HQ03-S0001"
        packs["query"]["legacy_match_ids"] = np.array([json.dumps(["OLD-HQ03-S0001"])] + ["[]"] * 8)
        self.assertFalse(np.array_equal(packs["source"]["X"][0], packs["query"]["X"][0]))
        with self.assertRaisesRegex(LeakageError, "Repeated physical_alias"):
            preflight_role_packs(packs)
        packs = self.roles()
        packs["query"]["record_ids"][0] = packs["source"]["entity_ids"][0]
        with self.assertRaisesRegex(LeakageError, "Repeated physical_alias"):
            preflight_role_packs(packs)

    def test_alias_bridge_chain_blocks_and_same_row_self_alias_is_valid(self):
        packs = self.roles()
        source_id = str(packs["source"]["record_ids"][0])
        packs["source"]["legacy_match_ids"] = np.array([json.dumps([source_id])] + ["[]"] * 8)
        preflight_role_packs(packs)  # A same-row self-alias is not another sample.
        packs["source"]["legacy_match_ids"] = np.array([json.dumps(["bridge-A"])] + ["[]"] * 8)
        packs["support"]["legacy_match_ids"] = np.array([json.dumps(["bridge-A", "bridge-B"])] + ["[]"] * 8)
        packs["query"]["legacy_match_ids"] = np.array([json.dumps(["bridge-B"])] + ["[]"] * 8)
        with self.assertRaisesRegex(LeakageError, "Repeated physical_alias"):
            preflight_role_packs(packs)

    def test_no_overwrite_and_independent_truth_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sealed.json"
            write_new_json(path, {"test": 1})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"test": 2})
        prediction = {"record_ids": np.array(["q2", "q1", "q3"]),
                      "predictions": np.array(["B", "A", "A"]), "classes": np.array(["A", "B"])}
        scored = score_predictions(["q1", "q2", "q3"], ["A", "B", "B"], prediction,
                                   baseline=np.array(["A", "A", "B"]))
        self.assertEqual(scored["correct"], 2)
        self.assertEqual(scored["introduced_errors"], 1)
        self.assertEqual(scored["corrected_errors"], 1)
        with self.assertRaisesRegex(LeakageError, "Truth identity"):
            score_predictions(["q1", "q2", "wrong"], ["A", "B", "B"], prediction)
        duplicate = {**prediction, "record_ids": np.array(["q2", "q1", "q3", "q2"]),
                     "predictions": np.array(["B", "A", "A", "B"])}
        with self.assertRaisesRegex(LeakageError, "Truth identity"):
            score_predictions(["q1", "q2", "q3"], ["A", "B", "B"], duplicate)


class SupplementalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.classes = np.array(["A", "B", "C"])
        self.source = np.array([[2., 0, 0], [0, 2., 0], [0, 0, 2.]])
        self.query = np.array([[1.01, 1., 0], [3., 0, 0], [0, 1.01, 1.]])
        # Deliberate native-vote/OVR-argmax discrepancy in the first row.
        self.native = np.array(["B", "A", "B"])
        self.support = np.array([[0., 2., 0], [0, 0, 2.], [0, 0, 2.]])
        self.candidate = candidate_grid()[0]

    def apply(self, method, **kwargs):
        return calibrate_scores(method, self.source, self.classes, self.query, self.native, **kwargs)

    def test_all_zero_support_and_zero_label_controls_keep_native_votes(self):
        for method in ("none", "traditional", "conservative", "zero_label_control"):
            result = self.apply(method)
            np.testing.assert_array_equal(result["predictions"], self.native)
            np.testing.assert_array_equal(result["scores"], self.query)
        result = self.apply("zero_label_control", support_scores=self.support,
                            support_labels=self.classes, support_native=self.classes)
        np.testing.assert_array_equal(result["predictions"], self.native)

    def test_unaffected_rows_preserve_vote_even_when_argmax_differs(self):
        corrected = self.query.copy()
        corrected[2] = [0, 0, 2]
        prediction = apply_changed_scores(self.native, self.query, corrected, self.classes)
        np.testing.assert_array_equal(prediction, ["B", "A", "C"])
        for translation in (1e-12, 3.):
            shifted = apply_changed_scores(self.native, self.query, self.query + translation, self.classes)
            np.testing.assert_array_equal(shifted, self.native)
        changed_loser = self.query.copy()
        changed_loser[0, 2] -= 2
        np.testing.assert_array_equal(apply_changed_scores(self.native, self.query, changed_loser, self.classes), self.native)

    def test_missing_classes_and_insufficient_native_errors_are_noop(self):
        result = self.apply("traditional", support_scores=self.support[:2], support_labels=self.classes[:2])
        self.assertEqual(result["reason"], "incomplete_support_class_coverage")
        result = self.apply("conservative", support_scores=self.support, support_labels=self.classes,
                            support_native=self.classes, limited_candidate=self.candidate)
        self.assertEqual(result["reason"], "insufficient_native_support_error_evidence")
        np.testing.assert_array_equal(result["predictions"], self.native)

    def test_active_conservative_correction_bound_and_pointwise_behavior(self):
        kwargs = dict(support_scores=self.support, support_labels=self.classes,
                      support_native=np.array(["B", "C", "C"]), limited_candidate=self.candidate)
        result = self.apply("conservative", **kwargs)
        self.assertEqual(result["status"], "APPLIED")
        self.assertLessEqual(np.max(np.abs(result["scores"] - self.query)),
                             result["cap"] * self.candidate["blend"] + 1e-14)
        for i in range(len(self.query)):
            single = calibrate_scores("conservative", self.source, self.classes,
                                      self.query[i:i+1], self.native[i:i+1], **kwargs)
            np.testing.assert_allclose(single["scores"], result["scores"][i:i+1], atol=1e-14)
        np.testing.assert_array_equal(result["scores"][1], self.query[1])


class SupplementalSyntheticFitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(24, 18))
        self.y = np.array(["A"] * 8 + ["B"] * 8 + ["C"] * 8)
        self.X[self.y == "B", 3] += 2
        self.X[self.y == "C", 8] -= 2
        self.axis = np.linspace(.1, 1.5, 18)
        self.ids = np.array([f"synthetic-{i}" for i in range(24)])
        self.config = dict(family="mcuve_svm", preprocessing="snv", C=10., gamma="scale",
                           class_weight="balanced", mc=dict(runs=10, components=2, ratio=.2))

    def fit(self, config=None, **kwargs):
        return fit_source_model(self.config if config is None else config,
            self.X, self.y, self.axis, self.ids, seed=42, dataset_version="SYNTHETIC_ONLY",
            split_id="fixture-source", code_version="fixture-code", physical_quantity="synthetic_alpha", **kwargs)

    def test_legacy_selector_numerics_and_each_pls_fit_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            model = self.fit(log_path=path)
            z = StandardScaler().fit_transform(SpectralPreprocessor("snv", self.axis).transform(self.X))
            legacy = MCUVESelector(n_runs=10, n_components=2, n_features_to_select=18,
                                   sample_fraction=.8, task="classification", random_state=42).fit(z, self.y)
            np.testing.assert_array_equal(model.transform.selector.coefficients_, legacy.coefficients_)
            np.testing.assert_array_equal(model.transform.selector.ranking_, legacy.ranking_)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            completed = [e for e in events if e["event"] == "fit_completed"]
            pls = [e for e in completed if e["component"] == "MC-UVE.PLSRegression.fit"]
            self.assertEqual(len(pls), 30)  # Ten original MC draws, three PLS1 class targets.
            for event in completed:
                self.assertTrue(set(event["fit_sample_ids"]) <= set(self.ids))

    def test_query_mutation_has_no_fit_effect_and_cache_cannot_cross_axis(self):
        cache = {}
        model = self.fit(cache=cache)
        before = joblib.hash(model)
        query = self.X[:4] + 20
        first = predict_numeric(model, query, ["q0", "q1", "q2", "q3"])
        one = predict_numeric(model, query[:1], ["q0"])
        np.testing.assert_allclose(first["scores"][:1], one["scores"], atol=1e-10)
        self.assertEqual(before, joblib.hash(model))
        second_config = {**self.config, "C": 100.}
        reused = self.fit(config=second_config, cache=cache)
        self.assertTrue(reused.metadata["transform_cache_reused"])
        self.axis = self.axis + .001
        new_axis = self.fit(cache=cache)
        self.assertFalse(new_axis.metadata["transform_cache_reused"])
        self.assertEqual(len(cache), 2)

    def test_new_singleton_classes_are_kept_in_fixed_model_fit(self):
        # Mirrors enrollment feasibility only, never singleton CV evidence.
        rng = np.random.default_rng(20261104)
        x = np.r_[self.X, rng.normal(size=(3, 18))]
        y = np.r_[self.y, ["D", "E", "F"]]
        ids = np.r_[self.ids, ["synthetic-new-D", "synthetic-new-E", "synthetic-new-F"]]
        model = fit_source_model(self.config, x, y, self.axis, ids, seed=42,
            dataset_version="SYNTHETIC_ONLY", split_id="singleton-enrollment",
            code_version="fixture-code", physical_quantity="synthetic_alpha")
        np.testing.assert_array_equal(model.classes_, ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(model.scores(x[-3:]).shape, (3, 6))

    def test_non_selector_families_accept_explicit_null_mc(self):
        configs = [dict(family="svm", preprocessing="snv", mc=None, C=1., gamma="scale"),
                   dict(family="shrinkage_lda", preprocessing="snv", mc=None, shrinkage="auto"),
                   dict(family="mlp", preprocessing="snv", mc=None, hidden=[8], alpha=.001, max_iter=3)]
        for config in configs:
            with self.subTest(family=config["family"]):
                model = self.fit(config=config)
                result = predict_numeric(model, self.X[:2], ["fixture-q0", "fixture-q1"])
                self.assertEqual(result["scores"].shape, (2, 3))
                self.assertEqual(model.metadata["selected_features"], 18)


if __name__ == "__main__":
    unittest.main()
