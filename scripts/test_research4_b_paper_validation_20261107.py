"""Numerical boundary tests for the new marginal interval/replay diagnostics."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("paper_validation", Path(__file__).with_name("research4_b_paper_validation_20261107.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PaperValidationTests(unittest.TestCase):
    def test_zero_and_all_success_boundaries(self):
        lower, upper = M.recall_interval(0, 5)
        self.assertEqual(lower, 0)
        self.assertAlmostEqual(upper, 1-.025**.2, places=13)
        lower, upper = M.recall_interval(5, 5)
        self.assertAlmostEqual(lower, .025**.2, places=13)
        self.assertEqual(upper, 1)

    def test_minority_interval_reference_and_complement(self):
        lower, upper = M.recall_interval(3, 5)
        self.assertAlmostEqual(lower, .14663279963467313, places=12)
        self.assertAlmostEqual(upper, .9472550494736831, places=12)
        low2, high2 = M.recall_interval(2, 5)
        self.assertAlmostEqual(lower, 1-high2, places=13)
        self.assertAlmostEqual(upper, 1-low2, places=13)

    def test_mcnemar_small_and_holm_monotone(self):
        self.assertEqual(M.exact_mcnemar(0, 0), 1)
        self.assertAlmostEqual(M.exact_mcnemar(5, 10), .3017578125, places=13)
        self.assertEqual(M.holm([.04, .01, .02]), [.04, .03, .04])

    def test_invalid_binomial_support_rejected(self):
        for k, n in [(-1, 5), (6, 5), (0, 0), (1.5, 5)]:
            with self.assertRaises(ValueError):
                M.recall_interval(k, n)


if __name__ == "__main__":
    unittest.main()
