"""Source-peak unit fixtures are not real chemical spectra or study samples."""
import inspect
import unittest

import numpy as np

from herb_spectra.experimental_peaks import analyze_source_peaks, cm1_to_thz, thz_to_cm1
from herb_spectra.leakage_guard import FitScope, LeakageError


class ExperimentalPeakTests(unittest.TestCase):
    def test_coordinate_conversion_roundtrip(self):
        values=np.array([0.,.1,1.,1.5])
        self.assertAlmostEqual(float(thz_to_cm1(1.)),33.3564095198,places=10)
        np.testing.assert_allclose(cm1_to_thz(thz_to_cm1(values)),values,rtol=1e-15,atol=1e-15)

    def test_single_constant_source_records_no_peaks_without_numerical_artifacts(self):
        X=np.ones((1,51));axis=np.linspace(.1,1.5,51)
        scope=FitScope(["s0"],"outer0.source","fixture")
        result=analyze_source_peaks(X,axis,["s0"],scope)
        self.assertEqual(result["status"],"NO_PEAKS")
        self.assertEqual(result["candidates"],[])
        np.testing.assert_array_equal(X,np.ones((1,51)))

    def test_fixed_rule_detects_known_fixture_positions(self):
        axis=np.linspace(.1,1.5,401)
        signal=.2+np.exp(-((axis-.4)/.035)**2)+.7*np.exp(-((axis-1.)/.04)**2)
        X=np.stack([signal,signal*1.1]);scope=FitScope(["s0","s1"],"outer0.source","fixture")
        result=analyze_source_peaks(X,axis,["s0","s1"],scope)
        self.assertEqual(result["candidate_count"],2)
        np.testing.assert_allclose([p["frequency_thz"] for p in result["candidates"]],[.4,1.],atol=.004)
        self.assertTrue(all(p["chemical_assignment"]=="" for p in result["candidates"]))

    def test_forbidden_source_scope_and_label_arguments_cannot_enter_analysis(self):
        X=np.ones((2,51));axis=np.linspace(.1,1.5,51);scope=FitScope(["s0","s1"],"outer0.source","fixture")
        with self.assertRaises(LeakageError):
            analyze_source_peaks(X,axis,["s0","test0"],scope)
        self.assertFalse({"y","query","X_test","test_labels"}&set(inspect.signature(analyze_source_peaks).parameters))
        with self.assertRaises(TypeError):
            analyze_source_peaks(X,axis,["s0","s1"],scope,y=np.array([0,1]))


if __name__=="__main__":
    unittest.main()
