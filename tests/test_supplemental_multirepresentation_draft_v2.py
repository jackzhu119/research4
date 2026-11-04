"""Pure standard-library tests; no real spectra, numerical imports or training."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/draft_supplemental_multirepresentation_20261104_v2.py"
SPEC = importlib.util.spec_from_file_location("supplemental_multirepresentation_draft", SOURCE)
DRAFT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRAFT)


def feature_pack(role="source", name="s", entity="entity-s"):
    metadata = {}
    for mode in DRAFT.FEATURES:
        time_axis = mode == "time"
        points = DRAFT.POINTS[mode]
        metadata[mode] = {
            "native_axis_fingerprint": "a" * 64,
            "native_axis_fingerprint_definition": "TEST_FIXTURE_ONLY_NOT_REAL_DATA",
            "points": points,
            "target_axis": [i / (points - 1) if time_axis else .21 + 1.28 * i / (points - 1) for i in range(points)],
            "axis_kind": "native_time_coordinate_UNIT_UNVERIFIED" if time_axis else "frequency_THz",
            "axis_unit": "UNVERIFIED" if time_axis else "THz_axis_label",
            "quantity": {"alpha": "absorption_coefficient", "alpha_long": "absorption_coefficient", "n": "refractive_index", "time": "time_domain_amplitude"}[mode],
            "value_unit": "UNVERIFIED_TEST_FIXTURE",
            "pointwise_transform": {"alpha": "snv_population", "alpha_long": "snv_population", "n": "identity", "time": "l2_no_center"}[mode],
            "resampling": {"method": "polyphase", "up": 512, "down": 3375, "window": ["kaiser", 5.0],
                           "padtype": "line", "native_points": 6750} if time_axis else {"method": "linear_no_extrapolation"},
            "derived_from": "registered_native_time6750" if time_axis else
                            "canonical_original_alpha612_not_alpha128" if mode == "alpha_long" else "registered_original",
            "pointwise_prepared": True}
    # Pure synthetic schema fixture, not a prepared acquisition or scientific result.
    endpoint = 89.9866638183594; step = endpoint / 6749
    axis_fixture = [i * step for i in range(6750)]; axis_fixture[-1] = endpoint
    uniform, audit = DRAFT.time_axis_profile(axis_fixture, synthetic=True)
    audit["native_axis_sha256_raw_float64"] = DRAFT.NATIVE_TIME_AXIS_SHA
    audit["synthetic_signal_control_not_real_specimens"] = False
    target = [i * step * 3375 / 512 for i in range(1024)]
    audit["output_axis_sha256_raw_float64"] = DRAFT.axis_float64_sha(target)
    metadata["time"].update(native_axis_fingerprint=DRAFT.NATIVE_TIME_AXIS_SHA, target_axis=target,
        resampling={**DRAFT.TIME_RESAMPLING_BASE, "axis_audit": audit})
    pack = {"schema": "research4-compact-multirepresentation-pack-v1", "task_id": "huangqi_legacy4",
            "role": role, "record_ids": [name], "entity_ids": [entity], "identity_aliases": [[name, entity]],
            "values": {mode: [[float(i) for i in range(DRAFT.POINTS[mode])]] for mode in DRAFT.FEATURES},
            "axis_metadata": metadata}
    if role == "source":
        pack["source_labels"] = ["TEST_CLASS"]
    return pack


def oof_fixture():
    # Synthetic software fixture, never saved as scientific predictions.
    ids = ["fixture-" + str(i) for i in range(241)]
    labels = {name: "class-" + str(i % 4) for i, name in enumerate(ids)}
    folds = [{"train_ids": [v for i, v in enumerate(ids) if i % 3 != f],
              "valid_ids": [v for i, v in enumerate(ids) if i % 3 == f]} for f in range(3)]
    rows = [{"epoch": epoch, "seed": seed, "fold": f, "record_id": name, "prediction": labels[name]}
            for epoch in DRAFT.EPOCHS for seed in DRAFT.SEEDS for f, fold in enumerate(folds)
            for name in fold["valid_ids"]]
    return labels, folds, rows


class DraftTests(unittest.TestCase):
    def test_explicit_axis_uniformization_profile_and_no_material_irregularity(self):
        axis = [i + .0002 * __import__("math").sin(i / 7.) for i in range(6750)]
        axis[0], axis[-1] = 0., 6749.
        uniform, audit = DRAFT.time_axis_profile(axis, synthetic=True)
        self.assertEqual(uniform, list(map(float, range(6750))))
        self.assertGreater(audit["max_native_uniform_coordinate_deviation"], 0.)
        self.assertLess(audit["max_coordinate_deviation_in_steps"], .001)
        self.assertTrue(audit["no_extrapolation"])
        linear = [2 * t + 1 for t in axis]
        result = DRAFT.resample_row(axis, linear, uniform)
        self.assertLess(max(abs(y - (2 * x + 1)) for x, y in zip(uniform, result)), 1e-10)
        for bad in ([*range(6749), 9000], [0.] * 6750, [float("nan")] * 6750):
            with self.assertRaises(ValueError): DRAFT.time_axis_profile(bad, synthetic=True)
        with self.assertRaises(ValueError): DRAFT.time_axis_profile(axis, synthetic=False)

    def test_two_step_axis_metadata_rejects_omission_altered_axes_and_extrapolation(self):
        pack = feature_pack("query", "q", "e")
        DRAFT.validate_feature_pack(pack, source=False)
        for change in ("method", "uniform_sha", "output_sha", "deviation", "bounds", "native_sha"):
            changed = deepcopy(pack); time = changed["axis_metadata"]["time"]; audit = time["resampling"]["axis_audit"]
            if change == "method": time["resampling"]["method"] = "polyphase"
            if change == "uniform_sha": audit["uniform_axis_sha256_raw_float64"] = "0" * 64
            if change == "output_sha": audit["output_axis_sha256_raw_float64"] = "0" * 64
            if change == "deviation": audit["max_coordinate_deviation_in_steps"] = .1
            if change == "bounds": audit["native_end"] = 1.
            if change == "native_sha": time["native_axis_fingerprint"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError): DRAFT.validate_feature_pack(changed, source=False)
        changed = deepcopy(pack); time = changed["axis_metadata"]["time"]; audit = time["resampling"]["axis_audit"]
        audit["uniform_step"] *= .99
        altered_uniform = [audit["native_start"] + i * audit["uniform_step"] for i in range(6750)]
        altered_uniform[-1] = audit["native_end"]
        time["target_axis"] = [audit["native_start"] + i * audit["uniform_step"] * 3375 / 512 for i in range(1024)]
        audit["uniform_axis_sha256_raw_float64"] = DRAFT.axis_float64_sha(altered_uniform)
        audit["output_axis_sha256_raw_float64"] = DRAFT.axis_float64_sha(time["target_axis"])
        with self.assertRaises(ValueError): DRAFT.validate_feature_pack(changed, source=False)

    def test_import_is_metadata_only(self):
        self.assertIsNone(DRAFT._RUNTIME)
        for name in ("torch", "numpy", "sklearn"):
            self.assertNotIn(name, sys.modules)
        tree = ast.parse(SOURCE.read_text())
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                self.assertFalse(any(n.name.split(".")[0] in {"torch", "numpy", "sklearn"} for n in statement.names))
            if isinstance(statement, ast.ImportFrom):
                self.assertNotIn((statement.module or "").split(".")[0], {"torch", "numpy", "sklearn", "herb_spectra"})

    def test_complete_three_arm_metadata_plan(self):
        protocol = json.loads((ROOT / "data/research/supplemental_multirepresentation_draft_20261104_v2.json").read_text())
        plan = DRAFT.metadata_plan(protocol)
        self.assertEqual((plan["expected_source_alpha_fits"], plan["expected_final_fits"], plan["fits_executed"]), (27, 36, 0))
        self.assertFalse(plan["runtime_imported"])
        for change in ("seed", "roles", "singleton", "architecture", "capacity"):
            mutated = deepcopy(protocol)
            if change == "seed": mutated["seeds"] = [42]
            if change == "roles": mutated["tasks"][0]["expected_roles"]["source"] -= 1
            if change == "singleton": mutated["tasks"][3]["source_cv_folds"] = 3
            if change == "architecture": mutated["architecture"]["width"] = 24
            if change == "capacity": mutated["arms"]["alpha_capacity3"]["branch_inputs"] = ["alpha"]
            with self.subTest(change=change), self.assertRaises(ValueError): DRAFT.metadata_plan(mutated)

    def test_capacity_control_has_independent_three_branches(self):
        self.assertEqual(DRAFT.arm_spec("alpha_capacity3")["branch_inputs"], ["alpha", "alpha", "alpha_long"])
        self.assertEqual(DRAFT.arm_spec("alpha_capacity3")["branch_points"], [128, 128, 1024])
        self.assertEqual(DRAFT.arm_spec("alpha_n_time")["branch_points"], [128, 128, 1024])
        self.assertEqual(DRAFT.arm_spec("alpha_n_time")["embedding_dimensions"], 48)
        self.assertFalse(DRAFT.arm_spec("alpha_capacity3")["shared_encoder_weights"])
        self.assertEqual(DRAFT.arm_spec("alpha_single")["embedding_dimensions"], 16)

    def test_metadata_rejects_signal_length_or_capacity_drift(self):
        protocol = json.loads((ROOT / "data/research/supplemental_multirepresentation_draft_20261104_v2.json").read_text())
        for field, value in (("branch_points", [128, 128, 128]), ("embedding_dimensions", 16),
                             ("encoders_share_weights", True)):
            mutated = deepcopy(protocol); mutated["arms"]["alpha_capacity3"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): DRAFT.metadata_plan(mutated)

    def test_metadata_rejects_alpha128_upsampling_recipe(self):
        protocol = json.loads((ROOT / "data/research/supplemental_multirepresentation_draft_20261104_v2.json").read_text())
        protocol["preprocessing"]["alpha_long"]["interpolation"] = "upsample_from_alpha128"
        with self.assertRaises(ValueError): DRAFT.metadata_plan(protocol)

    def test_roles_preserved_and_aliases_cannot_cross(self):
        rows = [{"record_id": name, "entity_id": "e-" + name, "role": role, "label": "C", "legacy_match_ids": []}
                for name, role in (("s", "source"), ("p", "support"), ("q", "query"))]
        expected = {"source": 1, "support": 1, "query": 1}
        self.assertEqual(DRAFT.validate_roles(rows, expected)["query"], ["q"])
        rows[2]["legacy_match_ids"] = ["e-s"]
        with self.assertRaises(ValueError): DRAFT.validate_roles(rows, expected)

    def test_alpha_fingerprint_cannot_cross_roles(self):
        rows = [{"record_id": n, "entity_id": n, "role": r, "label": "C", "alpha_fingerprint": "same"}
                for n, r in (("s", "source"), ("q", "query"), ("p", "support"))]
        with self.assertRaises(ValueError): DRAFT.validate_roles(rows, {"source": 1, "support": 1, "query": 1})

    def test_source_folds_exact_once(self):
        labels, folds, _ = oof_fixture()
        DRAFT.validate_folds(list(labels), folds)
        folds[1]["valid_ids"].append(folds[0]["valid_ids"][0])
        with self.assertRaises(ValueError): DRAFT.validate_folds(list(labels), folds)

    def test_metric_formula_known_confusion(self):
        metrics = DRAFT.classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
        self.assertAlmostEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 4 / 5) / 2)
        self.assertAlmostEqual(metrics["balanced_accuracy"], .75)

    def test_epoch_tie_prefers_earliest_without_query(self):
        labels, folds, rows = oof_fixture()
        result = DRAFT.select_common_epoch("huangqi_legacy4", labels, folds, rows)
        self.assertEqual(result["epoch"], 24)
        self.assertFalse(result["query_labels_used"])
        self.assertEqual(len(result["seed_metrics"]), 9)

    def test_epoch_grid_cannot_be_incomplete_or_contain_query_id(self):
        labels, folds, rows = oof_fixture()
        with self.assertRaises(ValueError): DRAFT.select_common_epoch("huangqi_legacy4", labels, folds, rows[:-1])
        rows[0]["record_id"] = "real-query-is-not-source"
        with self.assertRaises(ValueError): DRAFT.select_common_epoch("huangqi_legacy4", labels, folds, rows)

    def test_new_singleton_classes_inherit_only(self):
        labels, folds, rows = oof_fixture()
        with self.assertRaises(ValueError): DRAFT.select_common_epoch("zaoren_expanded10", labels, folds, rows)
        self.assertEqual(DRAFT.inherited_epoch("zaoren_expanded10", {
            "zaoren_legacy7": {"scope": "SOURCE_OOF_ONLY", "epoch": 48}}), 48)

    def test_linear_interpolation_and_no_extrapolation(self):
        self.assertEqual(DRAFT.resample_row([0, 2, 4], [0, 4, 8], [0, 1, 4]), [0., 2., 8.])
        with self.assertRaises(ValueError): DRAFT.resample_row([0, 1], [1, 2], [-.1, .5])
        with self.assertRaises(ValueError): DRAFT.resample_row([0, 0], [1, 2], [0])

    def test_n_mean_is_preserved(self):
        first = DRAFT.pointwise_row("n", [1., 2., 3.])
        second = DRAFT.pointwise_row("n", [6., 7., 8.])
        self.assertEqual(first, [1., 2., 3.])
        self.assertEqual([b - a for a, b in zip(first, second)], [5., 5., 5.])

    def test_alpha_snv_and_time_signed_l2(self):
        alpha = DRAFT.pointwise_row("alpha", [1., 2., 3.])
        self.assertAlmostEqual(sum(alpha), 0.)
        self.assertAlmostEqual(sum(v * v for v in alpha) / 3, 1.)
        self.assertEqual(DRAFT.pointwise_row("time", [-3., 4.]), [-.6, .8])
        for mode, row in (("alpha", [1., 1.]), ("time", [0., 0.]), ("n", [float("nan")])):
            with self.subTest(mode=mode), self.assertRaises(ValueError): DRAFT.pointwise_row(mode, row)

    def test_query_truth_is_rejected_before_runtime(self):
        pack = feature_pack("query", "q", "entity-q")
        DRAFT.validate_feature_pack(pack, source=False)
        for key in ("source_labels", "labels", "truth", "origin_group_ids", "y"):
            mutated = deepcopy(pack); mutated[key] = ["C"]
            with self.subTest(key=key), self.assertRaises(ValueError): DRAFT.validate_feature_pack(mutated, source=False)

    def test_parent_alias_and_modality_misalignment_block(self):
        source = feature_pack(); query = feature_pack("query", "q", "entity-q")
        DRAFT.assert_pack_separation(source, query)
        query["identity_aliases"][0].append("entity-s")
        with self.assertRaises(ValueError): DRAFT.assert_pack_separation(source, query)
        query = feature_pack("query", "q", "entity-q"); query["values"]["n"] = []
        with self.assertRaises(ValueError): DRAFT.assert_pack_separation(source, query)

    def test_time_cannot_be_relabelled_as_THz(self):
        pack = feature_pack("query", "q", "entity-q")
        pack["axis_metadata"]["time"]["axis_kind"] = "frequency_THz"
        with self.assertRaises(ValueError): DRAFT.validate_feature_pack(pack, source=False)

    def test_runtime_is_blocked_without_root_release(self):
        with self.assertRaises(ValueError): DRAFT.numerical_runtime({})
        self.assertIsNone(DRAFT._RUNTIME)

    def test_time_filter_is_not_run_in_metadata_phase(self):
        with self.assertRaises(ValueError): DRAFT.prepare_time_rows([], [], preparation_release={})
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("scipy", sys.modules)

    def test_no_linear_downsample_of_time_or_derived_alpha128_control(self):
        pack = feature_pack("query", "q", "entity-q")
        pack["axis_metadata"]["time"]["resampling"] = {"method": "linear_no_extrapolation"}
        with self.assertRaises(ValueError): DRAFT.validate_feature_pack(pack, source=False)
        pack = feature_pack("query", "q", "entity-q")
        pack["axis_metadata"]["alpha_long"]["derived_from"] = "upsampled_alpha128"
        with self.assertRaises(ValueError): DRAFT.validate_feature_pack(pack, source=False)

    def test_real_query_cannot_enter_source_fit(self):
        with mock.patch.object(DRAFT, "numerical_runtime", side_effect=AssertionError("runtime must stay cold")):
            with self.assertRaises(ValueError):
                DRAFT.fit_source_leaf(feature_pack(), validation=feature_pack("query", "q", "entity-q"),
                    arm="alpha_single", epochs=96, seed=42, directory="NOT_CREATED", split_id="test", release={},
                    budget_check=lambda: None, collect_source_epoch_predictions=True)
        self.assertIsNone(DRAFT._RUNTIME)

    def test_frozen_fit_context_checks_actual_ids_arm_seed(self):
        source = feature_pack(); validation = feature_pack("inner_validation", "v", "entity-v")
        release = {"fit_contexts": {"f": {"task_id": "huangqi_legacy4", "phase": "source_alpha_cv",
            "epochs": 96, "arm": "alpha_single", "seed": 42, "source_ids": ["s"], "inference_ids": ["v"]}}}
        DRAFT.validate_fit_context(release, "f", source, validation, "source_alpha_cv", 96, "alpha_single", 42)
        for key, value in (("source_ids", ["v"]), ("seed", 43), ("arm", "alpha_n_time")):
            changed = deepcopy(release); changed["fit_contexts"]["f"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                DRAFT.validate_fit_context(changed, "f", source, validation, "source_alpha_cv", 96, "alpha_single", 42)

    def test_capacity_pair_audit_rejects_different_initialization(self):
        capacity = {"arm": "alpha_capacity3", "task_id": "huangqi_legacy4", "seed": 42, "epochs": 48,
                    "source_ids": ["TEST"], "source_batch_order_sha256": "FIXTURE", "initial_state_sha256": "FIXTURE",
                    "initial_encoder_sha256": ["X", "Y", "Z"], "parameter_count": 123,
                    "query_features_received": False}
        multimodal = deepcopy(capacity); multimodal["arm"] = "alpha_n_time"
        self.assertEqual(DRAFT.audit_capacity_pair(capacity, multimodal)["status"], "TECHNICAL_METADATA_PAIRING_PASS")
        multimodal["initial_state_sha256"] = "DIFFERENT"
        with self.assertRaises(ValueError): DRAFT.audit_capacity_pair(capacity, multimodal)


if __name__ == "__main__":
    unittest.main()
