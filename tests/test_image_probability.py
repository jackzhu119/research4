import unittest
import numpy as np
import torch

from herb_spectra.image_probability import (softmax_probabilities,probability_metrics,fit_temperature,
    singleton_anchor_folds,fit_coordinate_bias,fit_decision,apply_decision,supervised_continuation)
from herb_spectra.neural_base_search import DenseClassifier
from herb_spectra.paper_fewshot_classification import state_fingerprint
from herb_spectra.meta_refinement_search import train_meta_pair


class ProbabilityMethodTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_metrics_use_multiclass_brier_sum_not_divided_by_classes(self):
        p=np.full((2,2),.5);m=probability_metrics(p,np.array([0,1]))
        self.assertAlmostEqual(m["nll"],np.log(2));self.assertAlmostEqual(m["brier"],.5)
        self.assertEqual(m["ece10"],0.)

    def test_temperature_preserves_argmax_and_pointwise_prediction(self):
        scores=np.random.default_rng(3).normal(size=(20,7))*20
        for temperature in (.05,1.,20.):
            p=softmax_probabilities(scores,temperature)
            np.testing.assert_array_equal(p.argmax(1),scores.argmax(1))
            np.testing.assert_allclose(p[3:4],softmax_probabilities(scores[3:4],temperature))
        with self.assertRaises(ValueError):softmax_probabilities(scores,0.)

    def test_temperature_fit_can_reduce_overconfident_nll(self):
        scores=np.array([[10.,0.],[10.,0.],[10.,0.],[0.,10.]])
        y=np.array([0,0,1,1]);t,record=fit_temperature(scores,y)
        self.assertGreater(t,1.);self.assertTrue(record["argmax_unchanged"])
        self.assertLess(record["temperature_oof"]["nll"],record["raw_oof"]["nll"])

    def test_anchor_fold_never_validates_singleton_or_overlaps_ids(self):
        y=np.repeat(np.arange(7),[4,1,4,4,4,4,4]);ids=np.array([f"id{i}" for i in range(len(y))])
        folds,policy=singleton_anchor_folds(y,ids)
        rare=np.flatnonzero(y==1)[0];seen=[]
        for fold in folds:
            self.assertIn(rare,fold["train"]);self.assertNotIn(rare,fold["validation"])
            self.assertFalse(set(fold["train"])&set(fold["validation"]));seen.extend(fold["validation"])
        self.assertEqual(len(seen),24);self.assertEqual(len(set(seen)),24)
        self.assertEqual(policy["excluded_oof_ids"],[ids[rare]])

    def test_normal_three_per_class_has_complete_oof(self):
        y=np.repeat(np.arange(7),3);folds,policy=singleton_anchor_folds(y,np.arange(21))
        self.assertEqual(policy["excluded_oof_indices"],[])
        self.assertEqual(sorted(np.concatenate([f["validation"] for f in folds]).tolist()),list(range(21)))

    def test_coordinate_bias_fixes_reference_and_singleton(self):
        source=np.repeat(np.arange(7),[4,1,4,4,4,4,4]);y=source[source!=1]
        p=softmax_probabilities(np.random.default_rng(4).normal(size=(len(y),7)))
        bias,record=fit_coordinate_bias(p,y,source)
        self.assertEqual(bias[0],0.);self.assertEqual(bias[1],0.)
        self.assertLessEqual(abs(bias).max(),1.);self.assertIn(1,record["fixed_bias_classes"])

    def test_decision_bias_does_not_change_reported_probability(self):
        scores=np.tile(np.array([[1.,.9]]),(8,1));y=np.array([0,1]*4);source=np.array([0]*7+[1])
        fitted=fit_decision({"kind":"prior_bias","beta":1.},scores,y,source,1.)
        pred,p=apply_decision(fitted,scores)
        np.testing.assert_allclose(p,softmax_probabilities(scores))
        self.assertTrue(np.all(pred==1));self.assertTrue(np.all(p.argmax(1)==0))

    def test_source_continuation_exactly_matches_preexisting_supervised_arm(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(21,4)).astype(np.float32);y=np.repeat(np.arange(7),3)
        torch.manual_seed(5);initial=DenseClassifier(4,(5,),7);before=state_fingerprint(initial)
        actual,record=supervised_continuation(initial,x,y,np.arange(21),seed=17,updates=3)
        reference,_=train_meta_pair(initial,x,y,np.arange(21),{"updates":3,"inner_learning_rate":.001,
            "outer_learning_rate":.0001,"source_anchor_weight":1.},seed=17,class_weight="none")
        self.assertEqual(state_fingerprint(actual),state_fingerprint(reference["supervised"]))
        self.assertEqual(before,state_fingerprint(initial));self.assertEqual(len(record["source_tasks"]),3)


if __name__=="__main__":unittest.main()
