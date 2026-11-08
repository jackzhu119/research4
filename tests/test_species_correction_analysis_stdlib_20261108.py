"""Synthetic metadata/metric/stream tests; no spectra, ML, or quantum jobs."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("species_correction_unit_worker", ROOT / "scripts/research4_species_correction_analysis_20261108.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fake_legacy():
    legacy, groups = {}, {}
    for index, count in enumerate((350, 5, 100, 100, 100, 100, 100), 1):
        group = "ZY%02d" % index
        kind = "枣仁" if index < 3 else "酸枣仁"
        groups[group] = {"species": kind, "origin": "Origin%d" % index}
        for n in range(count):
            key = group + "-%03d" % (n + 1)
            legacy[key] = {"record_id": key, "origin_group_id": group, "species": kind,
                           "origin": "Origin%d" % index, "role": "prediction"}
    return legacy, groups


def prediction_rows(legacy):
    return [{"record_id": key, "outer_fold": "0", "y_true": value["origin_group_id"],
             "y_predicted": value["origin_group_id"]} for key, value in legacy.items()]


class Metrics(unittest.TestCase):
    def test_01_perfect(self):
        m = M.metric(["a", "b"], ["a", "b"], ["a", "b"])
        self.assertEqual((m["accuracy"], m["macro_f1"], m["balanced_accuracy"]), (1, 1, 1))

    def test_02_wrong(self):
        m = M.metric(["a", "b"], ["b", "a"], ["a", "b"])
        self.assertEqual((m["correct"], m["macro_f1"]), (0, 0))

    def test_03_precision_not_accuracy(self):
        m = M.metric(["a", "a", "b"], ["a", "a", "a"], ["a", "b"])
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertAlmostEqual(m["macro_f1"], .4)
        self.assertEqual(m["balanced_accuracy"], .5)

    def test_04_outside_prediction_is_not_removed(self):
        m = M.metric(["a", "b"], [M.OUTSIDE, "b"], ["a", "b"])
        self.assertEqual(m["samples"], 2)
        self.assertEqual(m["accuracy"], .5)
        self.assertIn(M.OUTSIDE, m["confusion_axis"])

    def test_05_empty_rejected(self):
        with self.assertRaises(ValueError):
            M.metric([], [], ["a"])

    def test_06_unknown_truth_rejected(self):
        with self.assertRaises(ValueError):
            M.metric(["z"], ["a"], ["a"])

    def test_07_paired(self):
        self.assertEqual(M.paired_counts([True, False, False], [False, True, True]),
                         {"rescued": 2, "harmed": 1, "net_correct": 1})

    def test_08_bad_pair_length(self):
        with self.assertRaises(ValueError):
            M.paired_counts([True], [])


class Population(unittest.TestCase):
    def setUp(self):
        self.legacy, self.groups = fake_legacy()
        self.rows = prediction_rows(self.legacy)

    def test_09_map_all_ids(self):
        result = M.map_rows(self.rows, self.legacy, self.groups)
        self.assertEqual(len(result), 855)
        self.assertTrue(all(r["old_group_correct"] for r in result))

    def test_10_duplicate_ids_rejected(self):
        self.rows[-1] = self.rows[0]
        with self.assertRaises(ValueError):
            M.map_rows(self.rows, self.legacy, self.groups)

    def test_11_unknown_predicted_group_rejected(self):
        self.rows[0]["y_predicted"] = "ZR01"
        with self.assertRaises(ValueError):
            M.map_rows(self.rows, self.legacy, self.groups)

    def test_12_wrong_old_truth_rejected(self):
        self.rows[0]["y_true"] = "ZY07"
        with self.assertRaises(ValueError):
            M.map_rows(self.rows, self.legacy, self.groups)

    def test_13_invalid_fold_rejected(self):
        self.rows[0]["outer_fold"] = "5"
        with self.assertRaises(ValueError):
            M.map_rows(self.rows, self.legacy, self.groups)

    def test_14_cross_species_errors_kept(self):
        self.rows[0]["y_predicted"] = "ZY03"
        result = M.subsets(M.map_rows(self.rows, self.legacy, self.groups))
        self.assertEqual(result["枣仁"]["samples"], 355)
        self.assertEqual(result["枣仁"]["correct"], 354)
        self.assertIn(M.OUTSIDE, result["枣仁"]["confusion_axis"])
        self.assertEqual(result["ALL"]["species_task"]["correct"], 854)

    def test_15_within_species_origin_error(self):
        self.rows[0]["y_predicted"] = "ZY02"
        result = M.subsets(M.map_rows(self.rows, self.legacy, self.groups))
        self.assertEqual(result["ALL"]["species_task"]["correct"], 855)
        self.assertEqual(result["枣仁"]["correct"], 354)

    def test_16_source_only_counts(self):
        zaoren = [k for k, v in self.legacy.items() if v["species"] == "枣仁"]
        acid = [k for k, v in self.legacy.items() if v["species"] == "酸枣仁"]
        for k in zaoren[:213] + acid[:300]:
            self.legacy[k]["role"] = "train"
        selected = M.selected_source_ids(self.legacy, list(self.legacy))
        self.assertEqual({k: len(v) for k, v in selected.items()}, {"枣仁": 213, "酸枣仁": 300})
        self.assertTrue(all(self.legacy[k]["role"] == "train" for v in selected.values() for k in v))

    def test_17_wrong_source_counts_rejected(self):
        with self.assertRaises(ValueError):
            M.selected_source_ids(self.legacy, list(self.legacy))

    def test_18_wrong_candidate_ids_rejected(self):
        ids = list(self.legacy); ids[-1] = "NEW-ID"
        with self.assertRaises(ValueError):
            M.selected_source_ids(self.legacy, ids)


class RowStream(unittest.TestCase):
    class Stream:
        def __init__(self, short=False):
            self.seeks, self.reads, self.short = [], [], short
        def seek(self, offset):
            self.seeks.append(offset)
        def read(self, size):
            self.reads.append(size)
            return b"x" * (size - 1 if self.short else size)

    def test_19_only_selected_rows_are_read(self):
        stream = self.Stream()
        result = M.selected_row_bytes(stream, 128, (855, 612), 8, [100, 1])
        self.assertEqual(stream.seeks, [128 + 612 * 8, 128 + 100 * 612 * 8])
        self.assertEqual(stream.reads, [612 * 8, 612 * 8])
        self.assertEqual(sorted(result), [1, 100])

    def test_20_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError):
            M.selected_row_bytes(self.Stream(), 128, (855, 612), 8, [1, 1])

    def test_21_shape_rejected(self):
        with self.assertRaises(ValueError):
            M.selected_row_bytes(self.Stream(), 128, (854, 612), 8, [1])

    def test_22_dtype_rejected(self):
        with self.assertRaises(ValueError):
            M.selected_row_bytes(self.Stream(), 128, (855, 612), 4, [1])

    def test_23_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError):
            M.selected_row_bytes(self.Stream(), 128, (855, 612), 8, [855])

    def test_24_truncated_row_rejected(self):
        with self.assertRaises(ValueError):
            M.selected_row_bytes(self.Stream(True), 128, (855, 612), 8, [0])


class HashBoundary(unittest.TestCase):
    def test_25_parent_traversal_rejected(self):
        with self.assertRaises(ValueError):
            M.path("reports/../data/x")

    def test_26_hash_change_rejected(self):
        with tempfile.TemporaryDirectory(dir=M.PREFIX) as tmp:
            p = Path(tmp) / "input.json"; p.write_text("{}")
            record = M.ref(p); p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                M.checked(record)

    def test_27_metadata_save_cannot_overwrite(self):
        with tempfile.TemporaryDirectory(dir=M.PREFIX) as tmp:
            p = Path(tmp) / "output.json"; M.save(p, {"a": 1})
            with self.assertRaises(ValueError):
                M.save(p, {"a": 2})

    def test_28_empty_outside_class_still_hurts_recall(self):
        result = M.metric(["a", "a"], ["a", M.OUTSIDE], ["a"])
        self.assertEqual(result["per_class"][0]["support"], 2)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
