"""Boundary tests; no scientific training, model fit, Git or background process."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research4_representation_transfer_validation_20261108.py"
SPEC = importlib.util.spec_from_file_location("representation_transfer_worker_test", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


class RepresentationTransferBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(192)
        self.rows = []
        for species, prefix, groups, count in [("酸枣仁", "ZY", range(3, 8), 60), ("黄芪", "HQ", range(1, 5), 12),
                                                ("天麻", "TM", range(1, 5), 12)]:
            for group in groups:
                for number in range(count):
                    rid = f"{prefix}{group:02d}-{number:03d}"
                    self.rows.append({"record_id": rid, "physical_sample_id": rid, "species": species,
                                      "origin_group_id": f"{prefix}{group:02d}", "role": "train"})
        for index in range(4):
            rid = "FORBIDDEN" + str(index)
            self.rows.append({"record_id": rid, "physical_sample_id": rid, "species": "酸枣仁",
                              "origin_group_id": "FORBIDDEN", "role": "calibration" if index % 2 else "prediction"})
        self.axis = np.linspace(.1, 1.5, 25)
        self.values = rng.normal(size=(len(self.rows), len(self.axis)))
        self.values[-4:] = np.nan
        self.transfer_axis = np.linspace(.21, 1.49, 128)
        self.transfer = np.stack([np.interp(self.transfer_axis, self.axis, row) for row in self.values])
        self.features = self.root / "features.npz"
        self.split = self.root / "split.csv"
        np.savez(self.features, record_ids=np.asarray([r["record_id"] for r in self.rows]),
                 X_alpha=self.values, axis=self.axis, X_alpha_transfer=self.transfer, axis_transfer=self.transfer_axis)
        with self.split.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
    def tearDown(self):
        self.temp.cleanup()
    def loaded(self, representation="alpha128"):
        return worker.load_training(self.features, self.split, representation)
    def test_holdout_poison_cannot_enter_loaded_or_normalized_arrays(self):
        for representation, count in [("alpha128", 128), ("alpha512", 512)]:
            store, species, axis, metadata = self.loaded(representation)
            self.assertEqual(store["X"].shape, (len(self.rows)-4, count))
            self.assertTrue(np.isfinite(store["X"]).all())
            self.assertFalse(any(str(x).startswith("FORBIDDEN") for x in store["record_ids"]))
            self.assertFalse(metadata["master_CAL_PRED_exported_or_used"])
    def test_resolution_same_frequency_endpoints_and_same_sample_order(self):
        first, s1, a1, _ = self.loaded("alpha128")
        second, s2, a2, _ = self.loaded("alpha512")
        np.testing.assert_array_equal(first["record_ids"], second["record_ids"])
        np.testing.assert_array_equal(s1, s2)
        self.assertEqual((a1[0], a1[-1]), (a2[0], a2[-1]))
        self.assertEqual((a2[0], a2[-1]), (.21, 1.49))
    def test_alpha128_is_exact_historical_array_not_reinterpolation(self):
        store, _, _, _ = self.loaded()
        lookup = {r["record_id"]: index for index, r in enumerate(self.rows)}
        np.testing.assert_array_equal(store["X"], self.transfer[[lookup[x] for x in store["record_ids"]]])
    def test_partition190_exact_and_query_truth_separated(self):
        store, species, _, _ = self.loaded()
        seen = []
        for fold in range(5):
            source, support, query, truth, auxiliary, classes = worker.partition(store, species, fold, 190, "hq_tm")
            self.assertEqual((len(source["X"]), len(support["X"]), len(query["X"])), (190, 50, 60))
            self.assertNotIn("labels", query)
            self.assertFalse(set(source["record_ids"]) & set(query["record_ids"]))
            self.assertFalse(set(support["record_ids"]) & set(query["record_ids"]))
            self.assertEqual(set(classes), set(truth))
            seen.extend(query["record_ids"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), 300)
    def test_240_uses_only_original_source_plus_support_and_same_queries(self):
        store, species, _, _ = self.loaded()
        a = worker.partition(store, species, 2, 190, "hq_tm")
        b = worker.partition(store, species, 2, 240, "hq_tm")
        self.assertEqual(set(b[0]["record_ids"]), set(a[0]["record_ids"]) | set(a[1]["record_ids"]))
        self.assertEqual(len(b[0]["X"]), 240)
        self.assertEqual(len(b[1]["X"]), 0)
        np.testing.assert_array_equal(a[2]["record_ids"], b[2]["record_ids"])
        np.testing.assert_array_equal(a[3], b[3])
    def test_auxiliary_species_are_separate_tasks_and_labels(self):
        store, species, _, _ = self.loaded()
        for name, expected in [("target_only", set()), ("target_extra", set()), ("hq_only", {"huangqi"}),
                               ("tm_only", {"tianma"}), ("hq_tm", {"huangqi", "tianma"})]:
            source, _, query, _, auxiliary, classes = worker.partition(store, species, 0, 190, name)
            self.assertEqual(set(auxiliary), expected)
            for pack in auxiliary.values():
                self.assertFalse(set(pack["labels"]) & set(classes))
                self.assertFalse(set(pack["record_ids"]) & set(source["record_ids"]))
                self.assertFalse(set(pack["record_ids"]) & set(query["record_ids"]))
    def test_exact_17_new_conditions_reference_duplicates_rejected(self):
        allowed, references = 0, 0
        for rep in worker.REPRESENTATIONS:
            for source_count in (190, 240):
                for auxiliary in worker.AUXILIARIES:
                    try:
                        worker.validate_case(rep, source_count, auxiliary); allowed += 1
                    except ValueError as error:
                        self.assertIn("REFERENCE_ALREADY_RUN", str(error)); references += 1
        self.assertEqual((allowed, references), (17, 3))
    def test_fixed_profile_and_cache_identity_include_representation_axis(self):
        store, species, axis, _ = self.loaded("alpha512")
        source, _, _, _, auxiliary, classes = worker.partition(store, species, 0, 190, "hq_only")
        args = SimpleNamespace(seed=42, representation="alpha512", source_count=190, auxiliary="hq_only")
        config = worker.configuration(args, source, auxiliary, classes, axis)
        self.assertEqual(config["profile"]["learning_rate"], .0005)
        self.assertEqual(config["profile"]["epochs"], 192)
        self.assertEqual(config["profile"]["pretrain_steps"], 480)
        self.assertNotIn("query", config)
        self.assertNotIn("support", config)
        different = worker.configuration(args, source, auxiliary, classes, np.linspace(.21, 1.49, 128))
        self.assertNotEqual(config["axis_sha256"], different["axis_sha256"])
    def test_fit_interface_receives_only_source_and_selected_auxiliary(self):
        old = worker.old_worker()
        store, species, axis, _ = self.loaded()
        source, support, query, truth, auxiliary, classes = worker.partition(store, species, 0, 190, "hq_only")
        args = SimpleNamespace(seed=42, representation="alpha128", source_count=190, auxiliary="hq_only")
        config = worker.configuration(args, source, auxiliary, classes, axis)
        calls = []
        def fake_neural(cache, actual_source, actual_auxiliary, arm, config, actual_classes):
            calls.append((actual_source, actual_auxiliary, arm))
            self.assertEqual(arm, "B")
            self.assertIs(actual_source, source)
            self.assertIs(actual_auxiliary, auxiliary)
            self.assertFalse(set(actual_source["record_ids"]) & set(query["record_ids"]))
            self.assertFalse(set(actual_source["record_ids"]) & set(support["record_ids"]))
            return object(), {"source_ids": source["record_ids"].tolist()}, {"new_model_fits": 0}
        with patch.object(old, "cached_neural", side_effect=fake_neural), \
             patch.object(old, "cached_reference", return_value=(object(), {"new_model_fits": 0})):
            worker.fit_pair(self.root / "cache", source, auxiliary, config, classes)
        self.assertEqual(len(calls), 1)
    def test_predictions_written_without_query_truth_and_deterministic_gzip(self):
        rows = [{"model": "neural", "setting": "source_only", "repeat": -1, "shots": 0,
                 "predicted": np.asarray(["ZY03", "ZY04"]), "scores": np.asarray([[1., 0.], [0., 1.]])}]
        first, second = self.root / "first.gz", self.root / "second.gz"
        worker.write_predictions(first, rows, ["a", "b"])
        worker.write_predictions(second, rows, ["a", "b"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with __import__("gzip").open(first, "rt") as f:
            values = list(csv.DictReader(f))
        self.assertNotIn("y_true", values[0])
        self.assertEqual(values[0]["predicted"], "ZY03")
    def test_completion_hash_detects_artifact_tamper(self):
        output = self.root / "out";output.mkdir()
        artifact = output / "summary.json";artifact.write_text('{}')
        worker.atomic_json(output / "complete.json", {"status": "complete", "job_manifest_sha256": "identity",
                           "artifacts_sha256": {"summary.json": worker.sha(artifact)}})
        self.assertTrue(worker.verified_complete(output, "identity"))
        artifact.write_text('{"changed":true}')
        with self.assertRaises(ValueError):worker.verified_complete(output, "identity")
    def test_240source_cannot_claim_support_calibration(self):
        args = SimpleNamespace(representation="alpha512", source_count=240,
                               auxiliary="target_only", with_calibration=True)
        with self.assertRaisesRegex(ValueError, "No support calibration"):
            worker.run_job(args)
    def test_complete_model_checkpoint_hash_is_also_reverified(self):
        output = self.root / "done";output.mkdir()
        cache = self.root / "model-cache";cache.mkdir()
        artifact = cache / "weights.pt";artifact.write_bytes(b"tiny-not-a-model")
        worker.atomic_json(cache / "complete.json", {"artifacts": {"weights.pt": worker.sha(artifact)}})
        worker.atomic_json(output / "complete.json", {"status": "complete", "job_manifest_sha256": "identity",
            "artifacts_sha256": {}, "model_receipts": {"neural": {"cache_path": str(cache),
              "complete_sha256": worker.sha(cache / "complete.json")}}})
        self.assertTrue(worker.verified_complete(output, "identity"))
        artifact.write_bytes(b"tampered")
        with self.assertRaises(ValueError):worker.verified_complete(output, "identity")
    def frozen_inputs(self):
        folder = self.root / "frozen";folder.mkdir()
        ids = [f"R{i:04d}" for i in range(999)]
        features, split = folder / "features.npz", folder / "split.csv"
        np.savez(features, record_ids=np.asarray(ids))
        with split.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["record_id", "role"]);w.writeheader()
            w.writerows({"record_id": rid, "role": "train"} for rid in ids)
        lock = {"schema": "research4-evidence-validation-TRAIN-only-v3", "run_id": "evidence-validation-20261108-v3",
                "training_records": 999, "no_CAL_or_PRED_exported": True, "training_allowlist": ids,
                "training_feature_sha256": worker.sha(features), "training_split_sha256": worker.sha(split)}
        worker.atomic_json(folder / "input_lock.json", lock)
        return features, split, lock
    def test_frozen_input_bytes_and_exact_allowlist_are_checked(self):
        features, split, lock = self.frozen_inputs()
        self.assertTrue(worker.validate_frozen_inputs(features, split)["exact_allowlist_verified"])
        with features.open("ab") as f:f.write(b"drift")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            worker.validate_frozen_inputs(features, split)
    def test_same_count_changed_ID_cannot_replace_an_allowed_specimen(self):
        features, split, lock = self.frozen_inputs()
        ids = lock["training_allowlist"].copy();ids[0] = "UNAUTHORIZED"
        np.savez(features, record_ids=np.asarray(ids))
        lock["training_feature_sha256"] = worker.sha(features)
        worker.atomic_json(features.parent / "input_lock.json", lock)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            worker.validate_frozen_inputs(features, split)
    def test_exact_four_matched_update_conditions_and_new_identity(self):
        allowed = []
        for representation in worker.REPRESENTATIONS:
            for count in (190, 240):
                for auxiliary in worker.AUXILIARIES:
                    try:
                        worker.validate_case(representation, count, auxiliary, 144)
                    except ValueError:continue
                    allowed.append((representation, count, auxiliary))
        self.assertEqual(len(allowed), 4)
        self.assertTrue(all(count == 240 and aux in ("target_only", "hq_tm") for _, count, aux in allowed))
        store, species, axis, _ = self.loaded()
        source, _, _, _, auxiliary, classes = worker.partition(store, species, 0, 240, "hq_tm")
        args = SimpleNamespace(seed=42, representation="alpha128", source_count=240, auxiliary="hq_tm", target_epochs=144)
        short = worker.configuration(args, source, auxiliary, classes, axis)
        args.target_epochs = 192
        long = worker.configuration(args, source, auxiliary, classes, axis)
        self.assertEqual(short["profile"]["epochs"] * int(np.ceil(240 / 64)), 192 * int(np.ceil(190 / 64)))
        self.assertNotEqual(worker.objhash(short), worker.objhash(long))


if __name__ == "__main__":
    unittest.main()
