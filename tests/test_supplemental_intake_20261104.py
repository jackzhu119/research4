"""Important identity/format guards, independent of medicinal-plant labels."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
from intake_supplemental_workbooks_20261104 import fingerprint,base_id,species
from resolve_supplemental_legacy_formatting_20261104 import verify_format_equivalence


class IdentityGuards(unittest.TestCase):
    def test_axis_is_part_of_measurement_identity(self):
        x=np.array([1.,2.,3.]);a=np.array([.1,.2,.3])
        self.assertNotEqual(fingerprint(a,x),fingerprint(a+.01,x))

    def test_values_are_part_of_measurement_identity(self):
        a=np.array([.1,.2,.3]);x=np.array([1.,2.,3.]);y=x.copy();y[1]+=1e-8
        self.assertNotEqual(fingerprint(a,x),fingerprint(a,y))

    def test_fixed_window_identity_does_not_modify_values(self):
        a=np.array([.1,.2,2.]);x=np.array([1.,2.,3.]);y=x.copy();y[-1]=10.;keep=a<=1.5
        before=x.copy();self.assertEqual(fingerprint(a,x,keep),fingerprint(a,y,keep));self.assertTrue(np.array_equal(x,before))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):fingerprint(np.arange(3),np.arange(4))

    def test_header_identity_and_anonymous_legacy(self):
        self.assertEqual(base_id("样品005_3mm"),"5");self.assertEqual(base_id("ZY03-001_吸收系数"),"ZY03-001")

    def test_unknown_header_does_not_become_a_column_label(self):
        with self.assertRaises(ValueError):base_id("mean of scans")

    def test_species_tasks_are_separate(self):
        self.assertEqual(species("天麻云南.xlsx"),"tianma");self.assertEqual(species("黄芪曲靖.xlsx"),"huangqi");self.assertEqual(species("枣仁.xlsx"),"zaoren")

    def test_tiny_formatting_changes_pass_fixed_absolute_tolerance(self):
        a=np.array([[1.,2.,3.]]);b=a.copy();b[0,1]+=1e-14
        d=verify_format_equivalence(a,b);self.assertGreater(d[0],0);self.assertLess(d[0],1e-12)

    def test_material_difference_is_not_a_legacy_alias(self):
        a=np.array([[1.,2.,3.]]);b=a.copy();b[0,1]+=1e-6
        with self.assertRaises(ValueError):verify_format_equivalence(a,b)

    def test_nonfinite_changes_cannot_be_hidden_as_formatting(self):
        with self.assertRaises(ValueError):verify_format_equivalence(np.array([[1.,np.nan]]),np.array([[1.,2.]]))
        with self.assertRaises(ValueError):verify_format_equivalence(np.array([[1.,np.nan]]),np.array([[1.,np.nan]]))


if __name__=="__main__":unittest.main()
