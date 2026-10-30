import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from herb_spectra.io import read_spectra
from herb_spectra.validation import group_folds, make_pipeline, nested_evaluate, support_query_split


class SpectralFoundationTests(unittest.TestCase):
    def test_underreported_excel_dimensions_and_reference_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.xlsx"
            wb = Workbook(); ws = wb.active; ws.title = "吸收系数"
            ws.append(["THz", "参考值", "样品1", "样品2"])
            ws.append([0.1, 99, 2, 3]); ws.append([0.2, 98, 4, 5]); wb.save(source)
            broken = Path(tmp) / "dimension_A1.xlsx"
            with zipfile.ZipFile(source) as src, zipfile.ZipFile(broken, "w") as dst:
                for member in src.infolist():
                    data = src.read(member.filename)
                    if member.filename == "xl/worksheets/sheet1.xml":
                        data = data.replace(b'ref="A1:D3"', b'ref="A1:A1"')
                    dst.writestr(member, data)
            result = read_spectra(broken)
            np.testing.assert_array_equal(result.values, [[2, 4], [3, 5]])
            self.assertEqual(result.column_numbers, (3, 4))

    def test_empty_sheet_cannot_be_silently_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.xlsx"
            wb = Workbook(); wb.active.title = "折射率"; wb.save(p)
            with self.assertRaises(ValueError): read_spectra(p, "折射率")

    def test_repeat_scans_never_cross_folds(self):
        y = np.repeat([0, 1], 24)
        groups = np.repeat(np.arange(24), 2)
        X = np.arange(48 * 8).reshape(48, 8)
        for train, test in group_folds(X, y, groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_scaling_uses_training_data_only(self):
        X = np.array([[1., 2., 3.], [2., 4., 2.], [3., 2., 1.], [4., 1., 2.]])
        model = make_pipeline("svc").fit(X, [0, 0, 1, 1])
        original = model.named_steps["scale"].mean_.copy()
        model.predict(X + 100000)
        np.testing.assert_array_equal(model.named_steps["scale"].mean_, original)
        np.testing.assert_array_equal(original, X.mean(axis=0))

    def test_few_shot_counts_physical_samples_and_reserves_queries(self):
        groups = np.repeat(np.arange(12), 3)
        y = np.repeat([0, 1], 18)
        support, query = support_query_split(y, groups, shots_per_class=2)
        self.assertEqual(len(np.unique(groups[support])), 4)
        self.assertFalse(set(groups[support]) & set(groups[query]))
        with self.assertRaises(ValueError): support_query_split(y, groups, 6)

    def test_conflicting_sample_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            group_folds(np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]), n_splits=2)

    def test_nested_classification_and_regression_on_synthetic_data(self):
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(24), 2)
        labels = np.repeat([0, 1], 24)
        X = rng.normal(size=(48, 8)) + labels[:, None]
        result = nested_evaluate(X, labels, groups, inner_splits=2)
        self.assertEqual(len(result["predictions"]), 48)
        self.assertIn("sample_macro_f1", result["metrics"])
        for fold in result["folds"]:
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
        values = np.repeat(rng.normal(size=24), 2)
        result = nested_evaluate(X, values, groups, model="pls", inner_splits=2)
        self.assertTrue(np.isfinite(result["metrics"]["sample_rmse"]))
        result = nested_evaluate(X, values, groups, model="svr", inner_splits=2)
        self.assertTrue(np.isfinite(result["metrics"]["sample_rmse"]))


if __name__ == "__main__":
    unittest.main()
