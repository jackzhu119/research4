import unittest
from pathlib import Path
import tempfile

import numpy as np

from herb_spectra.paper_c_external import (
    ExternalTask, interpolate_within_measured_support, require_transfer_compatibility,
    load_ginseng_alpha_pellets,
)


def task(**kwargs):
    values = dict(task_id="test", axis=np.array([0.2, 0.4, 0.6]),
                  spectra=np.array([[1., 2., 3.], [4., 5., 6.]]),
                  record_ids=("r1", "r2"), labels=("a", "b"), physical_ids=("p1", "p2"),
                  axis_unit="THz", quantity="absorption_coefficient", value_unit="cm^-1",
                  independent_units_verified=True, labels_verified=True, provenance={})
    values.update(kwargs)
    return ExternalTask(**values)


class ExternalTaskTests(unittest.TestCase):
    def test_missing_physical_identity_blocks_independent_evaluation(self):
        with self.assertRaisesRegex(ValueError, "independence"):
            task(physical_ids=("", ""), independent_units_verified=False).require_independent_evaluation()

    def test_filename_groups_do_not_satisfy_missing_scientific_labels(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            task(labels=None, labels_verified=False).require_independent_evaluation()

    def test_unknown_absorption_unit_blocks_transfer(self):
        with self.assertRaisesRegex(ValueError, "units"):
            require_transfer_compatibility(task(value_unit=None), task())

    def test_different_physical_quantity_blocks_transfer(self):
        with self.assertRaisesRegex(ValueError, "quantities"):
            require_transfer_compatibility(task(quantity="refractive_index", value_unit="RIU"), task())

    def test_interpolation_has_no_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "Extrapolation"):
            interpolate_within_measured_support(task(), [0.1, 0.3])

    def test_interpolation_preserves_rows_and_measured_values(self):
        result = interpolate_within_measured_support(task(), [0.2, 0.3, 0.4, 0.6])
        np.testing.assert_allclose(result, [[1, 1.5, 2, 3], [4, 4.5, 5, 6]])

    def test_duplicate_record_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            task(record_ids=("r1", "r1")).validate()

    def test_label_and_spectrum_misalignment_rejected(self):
        with self.assertRaisesRegex(ValueError, "Labels"):
            task(labels=("a",)).validate()

    def test_verified_separate_tasks_can_share_coordinates_without_labels_merging(self):
        a, b = task(), task(labels=("ZY01", "ZY02"))
        self.assertTrue(require_transfer_compatibility(a, b))
        self.assertEqual(a.labels, ("a", "b"))
        self.assertEqual(b.labels, ("ZY01", "ZY02"))

    def test_uploaded_bom_labels_are_read_without_dropping_first_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data/external/ginseng_hvbp5mdtxx_v2/original"
            root.mkdir(parents=True)
            for group in range(1, 7):
                (root / f"data_a{group}.csv").write_text(
                    "Frequency (THz),,\n0.2,1,2\n0.4,3,4\n0.6,5,6\n", encoding="utf-8-sig")
                (root / f"label_{group}.csv").write_text("1,0\n", encoding="utf-8-sig")
            value = load_ginseng_alpha_pellets(temp)
            self.assertEqual(value.spectra.shape, (12, 3))
            self.assertEqual(value.labels[:2], ("Panax_ginseng", "Panax_quinquefolius"))
            self.assertFalse(value.independent_units_verified)


if __name__ == "__main__":
    unittest.main()
