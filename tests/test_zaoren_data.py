import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from herb_spectra.dataset_registry import file_sha256
from herb_spectra.zaoren_data import prepare_zaoren_dataset


def fixture(repo, specifications):
    entries, metadata = [], []
    for number, spec in enumerate(specifications, 1):
        group = f"ZY{number:02d}"
        alpha = np.asarray(spec["alpha"], dtype=float)
        n = np.asarray(spec.get("n", alpha + 2), dtype=float)
        count, points = alpha.shape
        identifiers = [f"{group}-{i + 1:03d}" for i in range(count)]
        n_identifiers = spec.get("n_ids", identifiers)
        axis = np.asarray(spec.get("axis", np.linspace(.1, 1.4, points)))
        n_axis = np.asarray(spec.get("n_axis", axis))
        path = repo / f"data/processed/zaoren_20261031/样本{number}_编号版.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        wb = Workbook(); wb.remove(wb.active)
        mapping = []
        for title, values, ids, coordinate in [("吸收系数", alpha, identifiers, axis),
                                               ("折射率", n, n_identifiers, n_axis)]:
            ws = wb.create_sheet(title); ws.append(["Thz", *ids])
            for p in range(points):
                ws.append([float(coordinate[p]), *[float(v) if np.isfinite(v) else None for v in values[:, p]]])
            for i, identifier in enumerate(ids):
                mapping.append({"sheet": title, "cell": f"{chr(66 + i)}1",
                                "new_id": identifier, "new_header": identifier,
                                "old_header": f"样品{i + 1}", "preserved_suffix": ""})
        for i, identifier in enumerate(identifiers):
            mapping.append({"sheet": "时域", "cell": f"{chr(67 + i)}1", "new_id": identifier,
                            "new_header": identifier + "_1.5mm", "old_header": f"样品{i + 1}_1.5mm",
                            "preserved_suffix": "_1.5mm"})
        wb.save(path)
        mapping_path = path.with_name(path.stem.replace("_编号版", "_编号对照") + ".csv")
        with mapping_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(mapping[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(mapping)
        entries.append({"path": str(path.relative_to(repo)), "sha256": file_sha256(path),
                        "status": "active", "dataset": "zaoren", "record_count": count,
                        "origin_group_id": group})
        for i, identifier in enumerate(identifiers):
            metadata.append({"source_workbook": str(path.relative_to(repo)), "sheet": "吸收系数",
                             "excel_column": chr(66 + i), "record_id": identifier,
                             "origin_group_id": group,
                             "physical_sample_id": spec.get("physical", identifiers)[i],
                             "batch_id": spec.get("batch", ""), "origin": "",
                             "origin_group_verified": spec.get("verified", "true"),
                             "origin_verified": spec.get("legacy_verified", ""),
                             "species_verified": ""})
    (repo / "data/active_workbooks.json").write_text(json.dumps({"version": "fixture",
        "effective_date": "2026-10-31", "workbooks": entries}), encoding="utf-8")
    with (repo / "data/metadata_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(metadata)


class ZaorenPreparationTests(unittest.TestCase):
    def run_preparation(self, root):
        return prepare_zaoren_dataset(repo=root, output=root / "result", near_duplicate_limit=3)

    def test_pairs_by_identity_not_position_and_writes_pickle_free_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, [{"alpha": [[1, 2, 4], [2, 4, 5]],
                            "n": [[12, 14, 15], [11, 12, 14]],
                            "n_ids": ["ZY01-002", "ZY01-001"]}])
            report = self.run_preparation(root)
            self.assertEqual(report["quality_status"], "passed")
            self.assertTrue(report["readiness"]["within_dataset_ready"])
            self.assertFalse(report["readiness"]["cross_batch_ready"])
            with np.load(root / "result/dataset.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["X_n"], [[11, 12, 14], [12, 14, 15]])
                np.testing.assert_array_equal(data["groups"], data["record_ids"])
                self.assertEqual(data["labels_verified"].dtype, np.dtype(bool))
                self.assertTrue(data["labels_verified"].all())
                self.assertFalse(data["species_verified"].any())
                for key in data.files:
                    self.assertNotEqual(data[key].dtype, np.dtype(object))

    def test_missing_sample_identity_stays_missing_and_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, [{"alpha": [[1, 2, 4], [2, 4, 5]], "physical": ["", ""], "verified": ""}])
            report = self.run_preparation(root)
            self.assertTrue(report["npz_written"])
            self.assertFalse(report["readiness"]["within_dataset_ready"])
            with np.load(root / "result/dataset.npz", allow_pickle=False) as data:
                self.assertFalse(data["labels_verified"].any())
                self.assertTrue((data["groups"] == "").all())

    def test_explicit_new_false_does_not_inherit_old_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, [{"alpha": [[1, 2, 4], [2, 4, 5]], "verified": "false", "legacy_verified": "true"}])
            report = self.run_preparation(root)
            self.assertEqual(report["readiness"]["verified_origin_group_labels"], 0)

    def test_axis_mismatch_is_reported_and_no_npz_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, [{"alpha": [[1, 2, 4], [2, 4, 5]], "n_axis": [.1, .8, 1.5]}])
            report = self.run_preparation(root)
            self.assertEqual(report["quality_status"], "blocked")
            self.assertFalse(report["npz_written"])
            self.assertIn("frequency axes differ", report["errors"][0]["error"])
            self.assertTrue((root / "result/quality_report.json").exists())

    def test_mismatched_record_ids_cannot_be_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, [{"alpha": [[1, 2, 4], [2, 4, 5]], "n_ids": ["ZY01-001", "ZY01-003"]}])
            report = self.run_preparation(root)
            self.assertFalse(report["npz_written"])
            self.assertIn("record-ID sets differ", report["errors"][0]["error"])

    def test_nonfinite_values_are_reported_without_dropping_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, [{"alpha": [[1, np.nan, 4], [2, 4, 5]], "n": [[3, 4, 6], [4, 6, 7]]}])
            report = self.run_preparation(root)
            self.assertEqual(report["record_count"], 2)
            self.assertEqual(report["sources"][0]["alpha"]["nonfinite_cells"], 1)
            self.assertFalse(report["npz_written"])
            self.assertTrue(report["no_records_removed"])

    def test_conflicting_physical_labels_or_joint_duplicate_labels_block(self):
        for shared_identity in [True, False]:
            with self.subTest(shared_identity=shared_identity), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a = {"alpha": [[1, 2, 4]], "physical": ["shared"] if shared_identity else ["A"]}
                b = {"alpha": [[2, 5, 9]] if shared_identity else [[1, 2, 4]],
                     "physical": ["shared"] if shared_identity else ["B"]}
                fixture(root, [a, b])
                report = self.run_preparation(root)
                self.assertFalse(report["npz_written"])
                self.assertTrue(any("conflicting origin labels" in row["error"] for row in report["errors"]))


if __name__ == "__main__":
    unittest.main()
