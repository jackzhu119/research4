import unittest

import numpy as np

from herb_spectra.paper_c_calibration import candidate_grid, fit_limited


class LimitedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.classes=np.array(["a","b","c"])
        self.source=np.array([[2.,0,0],[0,2.,0],[0,0,2.]])
        self.support=np.array([[0.,2.,0],[0,2.,0],[0,0,2.]])
        self.query=np.array([[1.01,1.,0],[2.,0,0],[0,1.01,1.]])
        self.candidate=candidate_grid()[0]

    def test_zero_support_and_same_structure_without_labels_are_exact_identity(self):
        for adapter in (fit_limited(self.source,self.classes,self.candidate),
                        fit_limited(self.source,self.classes,self.candidate,self.support,None)):
            np.testing.assert_array_equal(adapter.predict_scores(self.query),self.query)

    def test_no_support_mistakes_prevents_any_change(self):
        adapter=fit_limited(self.source,self.classes,self.candidate,self.source,self.classes)
        self.assertEqual(adapter.rejection_reason,"insufficient_support_error_evidence")
        np.testing.assert_array_equal(adapter.predict_scores(self.query),self.query)

    def test_correction_is_bounded_and_pointwise_without_query_statistics(self):
        adapter=fit_limited(self.source,self.classes,self.candidate,self.support,self.classes)
        result=adapter.predict_scores(self.query)
        self.assertLessEqual(np.max(np.abs(result-self.query)),adapter.cap*.25+1e-14)
        for i in range(len(self.query)):
            np.testing.assert_allclose(adapter.predict_scores(self.query[i:i+1]),result[i:i+1],atol=1e-14)
        np.testing.assert_array_equal(adapter.predict_scores(np.array([[5.,0.,0.]])),[[5.,0.,0.]])

    def test_minimum_two_errors_gates_one_error(self):
        adapter=fit_limited(self.source,self.classes,candidate_grid()[1],self.support,self.classes)
        self.assertEqual(adapter.support_error_count,1)
        self.assertIsNone(adapter.ridge)

    def test_incomplete_or_nonfinite_support_rejected(self):
        with self.assertRaises(ValueError):
            fit_limited(self.source,self.classes,self.candidate,self.support[:2],self.classes[:2])
        with self.assertRaises(ValueError):
            fit_limited(self.source,self.classes,self.candidate,np.full((3,3),np.nan),self.classes)


if __name__=="__main__":
    unittest.main()
