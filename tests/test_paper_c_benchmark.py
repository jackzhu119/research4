import copy
import unittest

import numpy as np
from herb_spectra.paper_c_benchmark import controls, fit_candidate, grids, random_indices, select_candidate


class PaperCBenchmarkTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(92)
        self.X = rng.normal(size=(36, 18))
        self.y = np.array(["A"]*18+["B"]*10+["C"]*8)
        self.axis = np.linspace(.1, 1.5, 18)

    def test_complete_prespecified_grids(self):
        self.assertEqual({k:len(v) for k,v in grids().items()}, {"full_svm":96,"shrinkage_lda":10,"mlp":8})

    def test_random_control_label_blind_same_budget_and_deterministic(self):
        a = random_indices(612, 92, 42)
        self.assertEqual(len(set(a)), 92)
        np.testing.assert_array_equal(a, random_indices(612, 92, 42))
        self.assertTrue(np.all(np.diff(a)>0))

    def test_query_does_not_change_training_statistics(self):
        candidate = grids()["full_svm"][32]
        model = fit_candidate(candidate, self.X, self.y, self.axis)
        before = copy.deepcopy(model.metadata)
        model.predict(self.X[:3]+500)
        self.assertEqual(model.metadata, before)
        self.assertEqual(model.metadata["source_samples"], 36)
        self.assertEqual(model.metadata["selected_features"], 18)

    def test_controls_keep_other_source_selected_factors(self):
        selected = {"mc":{"runs":50,"components":1,"ratio":.2},
                    "svm":{"C":1000.,"gamma":"scale_x0.1","class_weight":"power_0.5"}}
        c = controls(selected)
        self.assertEqual(c["R2_random_K"]["random_K"], 123)
        self.assertNotIn("mc", c["R2_full_features"])
        self.assertEqual(c["R2_no_C_gamma_search"]["class_weight"], "power_0.5")
        self.assertEqual(c["R2_no_class_weight"]["C"], 1000.)
        self.assertEqual(c["R2_fixed_MC20_LV2_K62"]["mc"], {"runs":20,"components":2,"ratio":.1})

    def test_selection_uses_macro_f1_before_accuracy_and_excludes_failures(self):
        candidates = [{"id":"A"}, {"id":"B"}, {"id":"C"}]
        rows = []
        for i in range(2):
            rows += [{"candidate_id":"A", "status":"ok", "metrics":{"macro_f1":.7,"balanced_accuracy":.7,"accuracy":.99}},
                     {"candidate_id":"B", "status":"ok", "metrics":{"macro_f1":.8,"balanced_accuracy":.8,"accuracy":.8}},
                     {"candidate_id":"C", "status":"failed"}]
        chosen = select_candidate(rows, candidates)
        self.assertEqual(chosen["selected"]["id"], "B")
        self.assertFalse(chosen["selected_using_outer_test"])


if __name__ == "__main__":
    unittest.main()
