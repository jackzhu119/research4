"""Scientific invariants for the fixed B-mechanism experiment.

Run with ``python -m unittest discover -s tests -p
test_b_mechanism_gap_20261110.py``.  Execution tests use synthetic spectra and
temporary output directories; no historical result or production fit is used.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import b_mechanism_engine_20261110 as engine


class FrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(
            ROOT / "data/research/species_revision_20261108/split_manifest.csv",
            keep_default_na=False,
        )
        cls.jobs = engine.build_design(cls.metadata)
        cls.indexed = cls.metadata.set_index("record_id", drop=False)

    def test_record_and_physical_ids_are_unique(self):
        for column in ("record_id", "physical_sample_id"):
            self.assertEqual(len(self.metadata), self.metadata[column].nunique())
            invalid = self.metadata.copy()
            invalid.loc[1, column] = invalid.loc[0, column]
            with self.assertRaisesRegex(ValueError, column):
                engine.build_design(invalid)

    def test_source_support_query_disjoint_and_master_train_only(self):
        for job in self.jobs:
            with self.subTest(job=job["job_id"]):
                groups = [set(job[name + "_ids"]) for name in ("source", "support", "query")]
                self.assertFalse(groups[0] & groups[1])
                self.assertFalse(groups[0] & groups[2])
                self.assertFalse(groups[1] & groups[2])
                members = self.indexed.loc[list(set.union(*groups))]
                self.assertTrue(members.role.eq("train").all())
                self.assertTrue(members.physical_sample_id.is_unique)
        invalid = copy.deepcopy(self.jobs[0])
        invalid["query_ids"][0] = invalid["source_ids"][0]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            engine.validate_job(invalid, self.metadata)

    def test_nontrain_sample_cannot_enter_any_role(self):
        invalid = self.metadata.copy()
        sample = self.jobs[0]["source_ids"][0]
        invalid.loc[invalid.record_id.eq(sample), "role"] = "prediction"
        with self.assertRaisesRegex(ValueError, "master TRAIN"):
            engine.validate_job(self.jobs[0], invalid)

    def test_current_species_semantics_are_enforced(self):
        expected = {"ZY01": "枣仁", "ZY02": "枣仁", **{g: "酸枣仁" for g in engine.ACID_ORIGINS}}
        for group, species in expected.items():
            self.assertTrue(self.metadata.loc[self.metadata.origin_group_id.eq(group), "species"].eq(species).all())
        invalid = self.metadata.copy()
        invalid.loc[invalid.origin_group_id.eq("ZY01"), "species"] = "酸枣仁"
        with self.assertRaisesRegex(ValueError, "species semantics"):
            engine.build_design(invalid)

    def test_paired_conditions_have_identical_query_and_support_ids(self):
        binary = [j for j in self.jobs if j["experiment"] == "E01"]
        self.assertEqual(len(binary), 90)
        for target in engine.ACID_ORIGINS:
            for seed in (42, 43, 44):
                matched = [j for j in binary if j["target_origin"] == target and j["seed"] == seed]
                self.assertEqual(len(matched), 6)
                self.assertEqual(len({tuple(j["query_ids"]) for j in matched}), 1)
                self.assertEqual(len({tuple(j["support_ids"]) for j in matched}), 1)

    def test_fixed_source_counts_ratios_and_coverage(self):
        for job in (j for j in self.jobs if j["experiment"] == "E01"):
            with self.subTest(job=job["job_id"]):
                source = self.indexed.loc[job["source_ids"]]
                acid, zaoren = {"A": (60, 180), "B": (120, 120), "C": (180, 60)}[job["ratio"]]
                self.assertEqual(len(source), 240)
                self.assertEqual(source.species.value_counts().to_dict(), {"枣仁": zaoren, "酸枣仁": acid})
                self.assertEqual(int(source.origin_group_id.eq(job["target_origin"]).sum()), 20 * job["coverage"])
                self.assertTrue(source.loc[source.species.eq("枣仁"), "origin_group_id"].eq("ZY01").all())
        for left in (j for j in self.jobs if j["experiment"] == "E01" and j["coverage"] == 0):
            right = next(j for j in self.jobs if j["experiment"] == "E01" and j["coverage"] == 1
                         and (j["target_origin"], j["seed"], j["ratio"]) ==
                         (left["target_origin"], left["seed"], left["ratio"]))
            removed = set(left["source_ids"]) - set(right["source_ids"])
            added = set(right["source_ids"]) - set(left["source_ids"])
            self.assertEqual((len(removed), len(added)), (20, 20))
            self.assertTrue(self.indexed.loc[list(removed), "species"].eq("酸枣仁").all())
            self.assertTrue(self.indexed.loc[list(added), "origin_group_id"].eq(left["target_origin"]).all())

    def test_e03_repeats_cover_each_physical_sample_once_per_repeat(self):
        acid_train = set(self.metadata.loc[self.metadata.role.eq("train") & self.metadata.origin_group_id.isin(engine.ACID_ORIGINS), "record_id"])
        self.assertEqual(len(acid_train), 300)
        multiclass = [j for j in self.jobs if j["experiment"] == "E03"]
        self.assertEqual(len(multiclass), 15)
        for seed in (20261031, 20261108, 42):
            folds = [j for j in multiclass if j["seed"] == seed]
            queries = [s for j in folds for s in j["query_ids"]]
            self.assertEqual(len(queries), 300)
            self.assertEqual(set(queries), acid_train)
            self.assertEqual(len(set(queries)), len(queries))
            for job in folds:
                self.assertEqual(tuple(len(job[x + "_ids"]) for x in ("source", "support", "query")), (190, 50, 60))
                self.assertEqual(self.indexed.loc[job["source_ids"], "origin_group_id"].value_counts().to_dict(), {g: 38 for g in engine.ACID_ORIGINS})

    def test_nested_k_shot_and_sampling_determinism(self):
        self.assertEqual(self.jobs, engine.build_design(self.metadata.sample(frac=1, random_state=17)))
        for job in self.jobs:
            previous = set()
            for k in (0, 1, 3, 5, 10):
                selected = engine.nested_support(job, self.metadata, k)
                self.assertEqual(len(selected), k * len(job["classes"]))
                self.assertTrue(previous.issubset(selected))
                self.assertTrue(set(selected).issubset(job["support_ids"]))
                for label in job["classes"]:
                    self.assertEqual(sum(self.indexed.at[s, job["label_column"]] == label for s in selected), k)
                previous = set(selected)

    def test_infeasible_source_is_reported_without_redesign(self):
        row = self.metadata.loc[self.metadata.role.eq("train") & self.metadata.origin_group_id.eq("ZY01")].index[0]
        with self.assertRaisesRegex(ValueError, "INFEASIBLE"):
            engine.build_design(self.metadata.drop(row))


class ScoreRuleTests(unittest.TestCase):
    def test_zero_support_returns_exact_identity_for_all_frozen_calibrators(self):
        scores = np.array([[-2.0, 2.0], [0.0, 0.0], [0.6, -0.6]])
        for method in ("identity", "bias_only", "anchor_lambda1", "weak_lambda1000_blend005"):
            with self.subTest(method=method):
                result, state = engine.calibrate_scores(method, np.empty((0, 2)), [], scores, ["枣仁", "酸枣仁"])
                np.testing.assert_array_equal(result, scores)
                self.assertTrue(state["identity"])

    def test_bias_only_changes_intercept_and_preserves_binary_ranking(self):
        query = np.array([[3., -3.], [.5, -.5], [-.2, .2], [-2., 2.]])
        support = np.array([[2., -2.], [1., -1.]])
        calibrated, state = engine.calibrate_scores("bias_only", support, ["枣仁", "酸枣仁"], query, ["枣仁", "酸枣仁"])
        np.testing.assert_array_equal(np.asarray(state["coef"]), np.eye(2))
        np.testing.assert_allclose(calibrated - query, np.broadcast_to(np.asarray(state["intercept"]), query.shape))
        np.testing.assert_array_equal(np.argsort(query[:, 1] - query[:, 0]), np.argsort(calibrated[:, 1] - calibrated[:, 0]))

    def test_fitting_interface_has_no_query_argument(self):
        self.assertEqual(list(inspect.signature(engine.fit_source).parameters), ["X_source", "y_source", "axis", "seed"])
        self.assertEqual(list(inspect.signature(engine.calibrate_scores).parameters), ["method", "support_scores", "support_y", "query_scores", "classes"])


class AtomicExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(ROOT / "data/research/species_revision_20261108/split_manifest.csv", keep_default_na=False)
        cls.jobs = engine.build_design(cls.metadata)
        cls.binary = cls.jobs[0]
        cls.multiclass = next(j for j in cls.jobs if j["experiment"] == "E03")
        rng = np.random.default_rng(7293)
        cls.X = rng.normal(size=(len(cls.metadata), 12))
        cls.X[:, 0] = np.arange(len(cls.metadata), dtype=float) / 1000
        cls.axis = np.linspace(.1, 1.5, cls.X.shape[1])
        cls.temporary = tempfile.TemporaryDirectory(prefix="b-mechanism-tests-")
        cls.output = Path(cls.temporary.name)
        cls.fit_calls = []
        original = engine.fit_source

        def observe_fit(X_source, y_source, axis, seed=42):
            cls.fit_calls.append((np.asarray(X_source).copy(), np.asarray(y_source).copy()))
            return original(X_source, y_source, axis, seed=seed)

        with mock.patch.object(engine, "fit_source", side_effect=observe_fit):
            cls.result = engine.execute_job(cls.binary, cls.X, cls.metadata, cls.output, "synthetic-input-v1", axis=cls.axis)
        cls.multiresult = engine.execute_job(cls.multiclass, cls.X, cls.metadata, cls.output, "synthetic-input-v1", axis=cls.axis)
        cls.path = cls.output / "jobs" / cls.binary["job_id"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_source_fit_excludes_support_and_query_and_retraining_refits_pipeline(self):
        lookup = {s: i for i, s in enumerate(self.metadata.record_id)}
        expected = [self.binary["source_ids"]] + [self.binary["source_ids"] + engine.nested_support(self.binary, self.metadata, k) for k in (1, 3, 5, 10)]
        self.assertEqual(len(self.fit_calls), 5)
        for (observed_X, observed_y), expected_ids in zip(self.fit_calls, expected):
            expected_rows = [lookup[s] for s in expected_ids]
            np.testing.assert_array_equal(observed_X, self.X[expected_rows])
            np.testing.assert_array_equal(observed_y, self.metadata.iloc[expected_rows].species.to_numpy())
            self.assertFalse(set(expected_ids) & set(self.binary["query_ids"]))
        self.assertFalse(set(expected[0]) & set(self.binary["support_ids"]))
        self.assertTrue(self.result["source_model_unchanged"])
        self.assertEqual(self.result["scientific_model_fits"], 5)

    def test_zero_support_controls_match_their_decision_rules(self):
        for job in (self.binary, self.multiclass):
            predictions = pd.read_csv(self.output / "jobs" / job["job_id"] / "predictions.csv", keep_default_na=False)
            zero = predictions.loc[predictions.k.eq(0)]
            baseline = zero.loc[zero.method.eq("identity")].set_index("record_id").pred_label
            native = zero.loc[zero.method.eq("svc_predict")].set_index("record_id").pred_label
            for method in ("identity", "bias_only", "anchor_lambda1", "weak_lambda1000_blend005"):
                adapted = zero.loc[zero.method.eq(method)].set_index("record_id").pred_label
                pd.testing.assert_series_equal(adapted, baseline)
            retrain = zero.loc[zero.method.eq("full_pipeline_retrain")].set_index("record_id").pred_label
            pd.testing.assert_series_equal(retrain, native)
            for row in zero.itertuples():
                np.testing.assert_array_equal(json.loads(row.source_scores_json), json.loads(row.scores_json))

    def test_hash_change_blocks_resume(self):
        with self.assertRaisesRegex(ValueError, "Input hash changed"):
            engine.execute_job(self.binary, self.X, self.metadata, self.output, "changed-input", axis=self.axis)
        changed_job = copy.deepcopy(self.binary)
        changed_job["source_ids"] = list(reversed(changed_job["source_ids"]))
        with self.assertRaisesRegex(ValueError, "definition changed"):
            engine.execute_job(changed_job, self.X, self.metadata, self.output, "synthetic-input-v1", axis=self.axis)

    def test_resume_reuses_complete_artifact_without_fit_or_cpu_recount(self):
        before = {p.name: (p.stat().st_mtime_ns, engine.file_hash(p)) for p in self.path.iterdir()}
        with mock.patch.object(engine, "fit_source", side_effect=AssertionError("resume must not fit")):
            resumed = engine.execute_job(self.binary, self.X, self.metadata, self.output, "synthetic-input-v1", axis=self.axis)
        self.assertEqual(resumed["cpu_seconds"], self.result["cpu_seconds"])
        self.assertEqual(resumed["scientific_model_fits"], self.result["scientific_model_fits"])
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, engine.file_hash(p)) for p in self.path.iterdir()})
        self.assertEqual(len(list((self.output / "jobs").glob("E01*"))), 1)

    def test_all_metrics_recomputed_independently_from_sample_predictions(self):
        for job in (self.binary, self.multiclass):
            destination = self.output / "jobs" / job["job_id"]
            predictions = pd.read_csv(destination / "predictions.csv", keep_default_na=False)
            metrics = pd.read_csv(destination / "metrics.csv", keep_default_na=False)
            classes = job["classes"]
            for (method, k), rows in predictions.groupby(["method", "k"]):
                with self.subTest(job=job["job_id"], method=method, k=k):
                    saved = metrics.loc[metrics.method.eq(method) & metrics.k.eq(k)].iloc[0]
                    table = np.array([[sum((rows.true_label == truth) & (rows.pred_label == pred)) for pred in classes] for truth in classes])
                    self.assertEqual(table.sum(), len(rows))
                    recalls = np.diag(table) / table.sum(axis=1)
                    precisions = np.divide(np.diag(table), table.sum(axis=0), out=np.zeros(len(classes), dtype=float), where=table.sum(axis=0) > 0)
                    f1 = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros(len(classes), dtype=float), where=precisions + recalls > 0)
                    self.assertAlmostEqual(float(saved.accuracy), float(np.trace(table) / table.sum()), places=12)
                    self.assertAlmostEqual(float(saved.balanced_accuracy), float(recalls.mean()), places=12)
                    self.assertAlmostEqual(float(saved.macro_f1), float(f1.mean()), places=12)
                    self.assertEqual(json.loads(saved.confusion_matrix_json), table.tolist())
                    self.assertEqual(json.loads(saved.recall_json), dict(zip(classes, recalls.tolist())))
            engine.verify_job(destination, "synthetic-input-v1", expected_job=job)

    def test_artifact_tampering_is_detected(self):
        import shutil
        copied = self.output / "tampered"
        shutil.copytree(self.path, copied)
        with (copied / "predictions.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
            engine.verify_job(copied, "synthetic-input-v1")

    def test_self_consistent_but_incomplete_task_is_not_a_valid_checkpoint(self):
        """A writer's valid byte hashes cannot substitute for scientific coverage."""
        import shutil
        copied = self.output / "incomplete"
        shutil.copytree(self.path, copied)
        for filename in ("metrics.csv", "predictions.csv"):
            rows = pd.read_csv(copied / filename, keep_default_na=False)
            rows.loc[~(rows.method.eq("anchor_lambda1") & rows.k.eq(3))].to_csv(copied / filename, index=False)
        result = json.loads((copied / "result.json").read_text())
        result["metric_rows"] = len(pd.read_csv(copied / "metrics.csv"))
        result["prediction_rows"] = len(pd.read_csv(copied / "predictions.csv"))
        result["artifact_hashes"] = {name: engine.file_hash(copied / name) for name in result["artifact_hashes"]}
        engine.write_json(copied / "result.json", result)
        marker = json.loads((copied / "COMPLETE.json").read_text())
        marker["result_sha256"] = engine.file_hash(copied / "result.json")
        engine.write_json(copied / "COMPLETE.json", marker)
        with self.assertRaisesRegex(ValueError, "endpoints are incomplete"):
            engine.verify_job(copied, "synthetic-input-v1")


if __name__ == "__main__":
    unittest.main()
