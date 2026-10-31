import copy
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from herb_spectra import cross_plant_transfer as transfer
from herb_spectra.leakage_guard import LeakageError, array_fingerprint, audit_pointwise_inference
from herb_spectra.upgrade_neural import row_snv, state_hash


def fixture_pack(prefix, count, classes, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray([classes[i % len(classes)] for i in range(count)])
    x = rng.normal(size=(count, 128)) + np.linspace(0, 1, 128)[None, :] * np.arange(count)[:, None] / count
    return {"X": x, "record_ids": np.asarray([prefix + f"-{i:03d}" for i in range(count)]),
            "source_labels": labels, "axis": transfer.AXIS.copy()}


def small_profile():
    result = copy.deepcopy(transfer.protocol()["profiles"][0])
    result.update(id="synthetic2", epochs=2, pretrain_steps=4, width=4, blocks=1)
    return result


class PairedTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.source = fixture_pack("ZYsource", 42, transfer.CLASSES, 10)
        cls.auxiliary = {"ginseng": fixture_pack("GS", 42, ["0", "1"], 20),
                         "huangqi": fixture_pack("HQ", 48, ["HQ01", "HQ02", "HQ03", "HQ04"], 30),
                         "tianma": fixture_pack("TM", 48, ["TM01", "TM02", "TM03", "TM04"], 40)}
        query = fixture_pack("query", 7, transfer.CLASSES, 50)
        cls.query = {k: v for k, v in query.items() if k != "source_labels"}
        cls.profile = small_profile()
        cls.fitted = {arm: transfer.fit_pipeline(cls.source, cls.auxiliary, arm, cls.profile, 42,
                     "synthetic_source_only", cls.directory / arm) for arm in transfer.ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_interpolation_is_pointwise_and_forbids_extrapolation(self):
        axis = np.linspace(.2, 1.6, 65)
        x = np.stack([2 * axis + 1, axis - 2])
        expected = np.stack([2 * transfer.AXIS + 1, transfer.AXIS - 2])
        np.testing.assert_allclose(transfer.interpolate(x, axis), expected, atol=1e-14)
        with self.assertRaises(LeakageError):
            transfer.interpolate(x, axis, np.linspace(.1, 1.5, 128))
        with self.assertRaises(LeakageError):
            transfer.interpolate(x[:, ::-1], axis[::-1])

    def test_query_strict_schema_axis_and_numeric_identity(self):
        for changed in [dict(self.query, source_labels=np.asarray(transfer.CLASSES)),
                        dict(self.query, batch_id=np.zeros(7)),
                        dict(self.query, axis=transfer.AXIS + .001),
                        dict(self.query, X=self.query["X"].astype(str))]:
            with self.assertRaises(LeakageError):
                self.fitted["B"].predict_scores(changed)

    def test_current_source_and_external_identity_overlap_rejected_before_fit(self):
        bad = copy.deepcopy(self.auxiliary)
        bad["ginseng"]["record_ids"] = bad["ginseng"]["record_ids"].astype("U64")
        bad["ginseng"]["record_ids"][0] = self.source["record_ids"][0]
        with self.assertRaises(LeakageError):
            transfer.fit_pipeline(self.source, bad, "B", self.profile, 42, "invalid", self.directory / "invalid")
        self.assertFalse((self.directory / "invalid").exists())

    def test_paired_initial_head_encoder_optimizer_schedule_and_scalers(self):
        for key in ["initial_encoder_sha256", "initial_zy_head_sha256", "fresh_target_optimizer_initial_sha256", "target_order_sha256"]:
            self.assertEqual(len({f.metadata[key] for f in self.fitted.values()}), 1, key)
        for fitted in self.fitted.values():
            np.testing.assert_array_equal(fitted.scaler.mean_, self.fitted["A"].scaler.mean_)
            np.testing.assert_array_equal(fitted.scaler.var_, self.fitted["A"].scaler.var_)
            self.assertEqual(fitted.metadata["target_steps"], 2)
        self.assertEqual(self.fitted["A"].metadata["pretrain_steps"], 0)
        self.assertEqual(self.fitted["C"].metadata["pretrain_pool_steps"], {"huangqi": 2, "tianma": 1, "ginseng": 1})

    def test_separate_source_vocabularies_and_saved_pretrain_checkpoint(self):
        checkpoint = self.fitted["C"].pretrained_state
        self.assertEqual(set(checkpoint["classes"]), {"huangqi", "tianma", "ginseng"})
        for name, classes in checkpoint["classes"].items():
            self.assertEqual(checkpoint["heads"][name]["weight"].shape[0], len(classes))
            self.assertFalse(set(classes) & set(transfer.CLASSES))
        self.assertEqual(self.fitted["C"].model.head.out_features, 7)
        self.assertIn("encoder", self.fitted["A"].pretrained_state)
        self.assertIn("fresh_zy_head", checkpoint)

    def test_bn_buffers_reset_before_target_training(self):
        for fitted in self.fitted.values():
            for name, values in fitted.metadata["BN_after_reset"].items():
                expected = 1 if name.endswith("running_var") else 0
                self.assertTrue(np.all(np.asarray(values) == expected), name)
        before = self.fitted["C"].metadata["BN_before_reset"]
        self.assertTrue(any(np.any(np.asarray(v) != 0) for k, v in before.items() if k.endswith("running_mean")))

    def test_actual_batch_trace_reconstructs_current_source_values_and_labels(self):
        fitted = self.fitted["C"]
        with gzip.open(self.directory / "C" / "batch_trace.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        schedule = transfer.read(self.directory / "C" / "batch_schedule.json")
        self.assertEqual(len(rows), 6)
        for row, scheduled in zip(rows[:4], schedule["pretrain"]):
            name = row["head"]
            pool = self.auxiliary[name]
            selected = np.asarray(scheduled["indices"])
            self.assertEqual(len(selected), 32)
            self.assertEqual(len(set(selected)), 32)
            self.assertEqual(row["fit_sample_ids"], pool["record_ids"][selected].tolist())
            values = fitted.auxiliary_scalers[name].transform(row_snv(pool["X"])).astype(np.float32)
            classes = np.unique(pool["source_labels"]).tolist()
            target = np.asarray([classes.index(v) for v in pool["source_labels"]], dtype=np.int64)
            self.assertEqual(row["X_fingerprint"], array_fingerprint(values[selected]))
            self.assertEqual(row["y_fingerprint"], array_fingerprint(target[selected]))
        for row, order in zip(rows[4:], schedule["target_epochs"]):
            self.assertEqual(row["head"], "ZY")
            self.assertEqual(set(row["fit_sample_ids"]), set(self.source["record_ids"]))

    def test_pointwise_inference_no_query_batch_statistics_or_state_updates(self):
        fitted = self.fitted["B"]
        values = fitted.transform(self.query)
        result = audit_pointwise_inference(fitted.model, {"alpha": values}, batch_sizes=(1, 3), atol=1e-5, rtol=1e-5)
        self.assertTrue(result)
        before = state_hash(fitted.model)
        predicted = fitted.predict_scores(self.query)
        changed = copy.deepcopy(self.query)
        changed["X"][0, 0] += .3
        other = fitted.predict_scores(changed)
        np.testing.assert_allclose(predicted[1:], other[1:], rtol=1e-5, atol=1e-5)
        self.assertEqual(before, state_hash(fitted.model))

    def test_stored_checkpoint_and_trace_hashes_detect_tamper(self):
        directory = self.directory / "A"
        transfer.check_pipeline_evidence(directory)
        fitted = joblib.load(directory / "pipeline.joblib")
        np.testing.assert_array_equal(fitted.predict_scores(self.query), self.fitted["A"].predict_scores(self.query))
        path = directory / "batch_trace.jsonl.gz"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"bad")
            with self.assertRaises(LeakageError):
                transfer.check_pipeline_evidence(directory)
        finally:
            path.write_bytes(original)

    def test_named_truth_reads_and_writes_rejected_with_actual_hooks(self):
        path = self.directory / "truth.csv"
        path.write_text("record_id,label\nunknown,ZY01\n")
        before = path.read_bytes()
        with transfer.forbid_named_truth([path]) as attempts:
            for operation in [lambda: path.read_text(), lambda: open(path, "w"), lambda: os.open(path, os.O_RDONLY)]:
                with self.assertRaises(LeakageError):
                    operation()
        self.assertEqual(len(attempts), 3)
        self.assertEqual(path.read_bytes(), before)


class SourceChoiceTests(unittest.TestCase):
    def table(self):
        profiles = transfer.protocol()["profiles"]
        table = {(arm, p["id"]): {"arm": arm, "profile": p,
            "mean": {"macro_f1": .6 if p["id"] == "p48" else .5, "balanced_accuracy": .6, "accuracy": .7}}
            for arm in transfer.ARMS for p in profiles}
        return table

    def test_scratch_alone_selects_shared_epoch_and_guard_falls_back(self):
        table = self.table()
        table["C", "p96"]["mean"] = {"macro_f1": .99, "balanced_accuracy": .99, "accuracy": .99}
        choice = transfer.select_shared_profile(table)
        self.assertEqual(choice["shared_profile"]["id"], "p48")
        self.assertEqual(choice["policy_arm"], "A")
        self.assertTrue(choice["profile_chosen_only_from_A"])

    def test_policy_requires_f1_margin_and_no_ba_accuracy_regression(self):
        table = self.table()
        table["Aextra", "p48"]["mean"]["macro_f1"] = .604
        table["B", "p48"]["mean"] = {"macro_f1": .7, "balanced_accuracy": .59, "accuracy": .8}
        table["C", "p48"]["mean"] = {"macro_f1": .61, "balanced_accuracy": .61, "accuracy": .71}
        self.assertEqual(transfer.select_shared_profile(table)["policy_arm"], "C")
        table["Aextra", "p48"]["mean"] = dict(table["C", "p48"]["mean"])
        self.assertEqual(transfer.select_shared_profile(table)["policy_arm"], "Aextra")

    def test_source_gate_binds_all_code_and_data_before_learning(self):
        with patch.object(transfer, "verify_run", return_value={}), patch.object(transfer, "read", return_value={"protocol_sha256": "p", "preparation_sha256": "d", "source_sha256": {"code": "s"}}):
            with self.assertRaises(LeakageError):
                transfer._check_source_release(Path("dummy"), {"allow_source_training": True, "created_at_utc": "now", "protocol_sha256": "p", "preparation_sha256": "d", "source_sha256": {"code": "wrong"}})

    def test_prediction_truth_byte_guard_precedes_all_truth_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            transfer.write(out / "prediction_lock.json", {"prediction_sha256": "wrong"})
            with patch.object(transfer, "verify_selected", return_value=({"contexts": []}, {})), patch.object(transfer, "_truth", side_effect=AssertionError("truth parsed")), patch.object(transfer, "sha", return_value="actual"):
                with self.assertRaises(LeakageError):
                    transfer.verify_predictions_before_truth(out)


if __name__ == "__main__":
    unittest.main()
