import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import Workbook

from herb_spectra.dataset_registry import file_sha256, select_workbooks


REPO = Path(__file__).resolve().parents[1]


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.registry = self.repo / "data/active_workbooks.json"
        self.huangqi = self.workbook("data/raw/huangqi/hq.xlsx", ["样品1", "样品2"])
        self.zaoren = self.workbook("data/processed/current/样本1_编号版.xlsx",
                                    ["ZY01-001", "ZY01-002"])
        self.payload = {"version": "2026-10-31-test", "effective_date": "2026-10-31",
                        "workbooks": [self.entry(self.huangqi, "huangqi"),
                                      self.entry(self.zaoren, "zaoren", "ZY01")]}
        self.save_registry()

    def workbook(self, relative, headers):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        ws = wb.active
        ws.title = "吸收系数"
        ws.append(["THz", *headers])
        ws.append([.1, *range(1, len(headers) + 1)])
        ws.append([.2, *range(2, len(headers) + 2)])
        wb.save(path)
        return path

    def entry(self, path, dataset, group=None):
        result = {"path": path.relative_to(self.repo).as_posix(), "sha256": file_sha256(path),
                  "dataset": dataset, "record_count": 2, "status": "active"}
        if group:
            result["origin_group_id"] = group
        return result

    def save_registry(self):
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        self.registry.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_routes_only_registered_inputs_and_preserves_unknown_scientific_fields(self):
        self.workbook("data/archive/legacy.xlsx", ["旧样品1", "旧样品2"])
        self.workbook("data/processed/current/编号对照表.xlsx", ["说明1", "说明2"])
        self.assertEqual([s.path for s in select_workbooks(repo=self.repo)],
                         [self.huangqi, self.zaoren])
        spec = importlib.util.spec_from_file_location("metadata_cli", REPO / "scripts/build_metadata_template.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = self.repo / "annotations.csv"
        with patch.object(module, "REPO", self.repo), patch("sys.argv", ["metadata", "--output", str(output)]):
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        with output.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["record_id"] for r in rows if r["origin_group_id"]}, {"ZY01-001", "ZY01-002"})
        self.assertEqual({r["dataset_version"] for r in rows}, {"2026-10-31-test"})
        self.assertEqual(rows[0]["scan_id"], hashlib.sha256("data/raw/huangqi/hq.xlsx|吸收系数|2".encode()).hexdigest()[:20])
        for row in rows:
            for key in ["physical_sample_id", "origin", "origin_verified", "batch_id", "assay_value"]:
                self.assertEqual(row[key], "")

    def test_missing_registry_refuses_raw_directory_fallback(self):
        self.registry.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no raw-directory fallback"):
            select_workbooks(repo=self.repo)

    def test_changed_workbook_bytes_are_rejected_before_use(self):
        with self.zaoren.open("ab") as handle:
            handle.write(b"unexpected change")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            select_workbooks(repo=self.repo)

    def test_duplicate_registry_input_is_rejected(self):
        self.payload["workbooks"].append(self.payload["workbooks"][0].copy())
        self.save_registry()
        with self.assertRaisesRegex(ValueError, "duplicate workbook"):
            select_workbooks(repo=self.repo)

    def test_explicit_directory_override_is_labelled_and_cannot_combine_with_manifest(self):
        with self.assertWarnsRegex(UserWarning, "override"):
            sources = select_workbooks(repo=self.repo, data_root="data/raw")
        self.assertEqual([s.path for s in sources], [self.huangqi])
        self.assertEqual(sources[0].selection_mode, "explicit_data_root_override")
        self.assertEqual(sources[0].dataset_version, "explicit-data-root-override")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            select_workbooks(repo=self.repo, manifest=self.registry, data_root="data/raw")


if __name__ == "__main__":
    unittest.main()
