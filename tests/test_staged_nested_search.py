import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from herb_spectra.selection import MCUVESelector
from herb_spectra.classification import build_pipeline
from herb_spectra.staged_nested_search import (
    _prefix_selector, class_weight_spec, gamma_spec, retained_count,
    mc_grid, svm_grid, historical_grid, fit_preprocess_bundle,
    fit_source_candidate, choose_candidate, write_json,
)


class StagedNestedTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.y=np.array(["A"]*18+["B"]*10+["C"]*8)
        self.X=rng.normal(size=(36,18))
        self.X[self.y=="B",3]+=2;self.X[self.y=="C",8]-=3
        self.axis=np.linspace(.1,1.5,18)
        self.config={"id":"fixture","preprocessing":"snv",
            "mc":{"runs":10,"components":2,"ratio":.2},
            "svm":{"C":10.,"gamma":"scale","class_weight":"balanced"}}

    def test_ceil_ratios_and_complete_grids(self):
        self.assertEqual([retained_count(612,f) for f in (.05,.1,.15,.2)],[31,62,92,123])
        self.assertEqual(len(mc_grid()),36);self.assertEqual(len(svm_grid()),96)
        with self.assertRaises(ValueError):retained_count(612,0)

    def test_prefix_stream_exactly_matches_independent_mc(self):
        z=StandardScaler().fit_transform(self.X);cache={"__coefficient_budget__":50}
        for runs in (10,20,50):
            for lv in (1,2,3):
                config={"runs":runs,"components":lv,"ratio":.2}
                prefix=_prefix_selector(z,self.y,config,42,cache)
                actual=MCUVESelector(n_runs=runs,n_components=lv,n_features_to_select=4,
                    sample_fraction=.8,task="classification",random_state=42).fit(z,self.y)
                np.testing.assert_array_equal(prefix.coefficients_,actual.coefficients_)
                np.testing.assert_array_equal(prefix.importances_,actual.importances_)
                np.testing.assert_array_equal(prefix.selected_indices_,actual.selected_indices_)
                self.assertEqual(prefix.computed_stream_runs_,50)

    def test_power_class_weights_source_mean_one(self):
        for mode in ("power_0.5","power_1.5"):
            value,record=class_weight_spec(self.y,mode)
            self.assertAlmostEqual(np.mean([value[v] for v in self.y]),1.)
            self.assertEqual(value,record);self.assertGreater(value["C"],value["A"])
        value,record=class_weight_spec(self.y,"balanced")
        self.assertEqual(value,"balanced");self.assertEqual(record["C"],36/(3*8))
        self.assertIsNone(class_weight_spec(self.y,"none")[0])

    def test_gamma_alias_and_relative_rules(self):
        z=StandardScaler().fit_transform(self.X)[:,[1,3,8,9]]
        self.assertAlmostEqual(gamma_spec(z,"scale")[1],gamma_spec(z,"auto")[1])
        self.assertAlmostEqual(gamma_spec(z,"scale_x0.1")[0],.1*gamma_spec(z,"scale")[1])
        self.assertAlmostEqual(gamma_spec(z,"scale_x10")[0],10*gamma_spec(z,"scale")[1])

    def test_msc_reference_fits_source_only(self):
        # Positive curved reference avoids near-zero slope fixtures.
        source=np.exp(self.X*.1)+np.linspace(1,3,18)
        prep,scale,a,b=fit_preprocess_bundle(source,source[:3]+4,self.axis,"msc")
        np.testing.assert_array_equal(prep.reference_,source.mean(0))
        first=scale.mean_.copy();prep.transform(source[:2]*10+100)
        np.testing.assert_array_equal(scale.mean_,first)
        self.assertEqual(a.shape,(36,18));self.assertEqual(b.shape,(3,18))

    def test_prediction_pointwise_and_source_immutable(self):
        model=fit_source_candidate(self.config,self.X,self.y,self.axis)
        before=joblib.hash(model)
        alone=model.predict(self.X[:1]);batch=model.predict(np.r_[self.X[:1],self.X[8:]*50])
        np.testing.assert_array_equal(alone,batch[:1]);self.assertEqual(before,joblib.hash(model))
        self.assertFalse(model.estimator.break_ties)
        np.testing.assert_array_equal(model.predict(self.X),model.estimator.predict(model.bundle.transform(self.X)))
        self.assertIn("vote",model.metadata["prediction_rule"])

    def test_model_roundtrip_stable_prediction(self):
        model=fit_source_candidate(self.config,self.X,self.y,self.axis)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"model.joblib";joblib.dump(model,p);loaded=joblib.load(p)
            np.testing.assert_array_equal(model.predict(self.X),loaded.predict(self.X))
            np.testing.assert_allclose(model.predict_scores(self.X),loaded.predict_scores(self.X),atol=0,rtol=0)

    def test_historical_grid_matches_parametergrid_order(self):
        grid=historical_grid();self.assertEqual(len(grid),8)
        self.assertEqual([(c["svm"]["C"],c["preprocessing"]) for c in grid],
            [(c,p) for c in (.1,1.,10.,100.) for p in ("raw","snv")])

    def test_historical_pipeline_exact_predictions_and_scores(self):
        x=np.tile(self.X,(1,4));axis=np.linspace(.1,1.5,x.shape[1])
        config={"id":"legacy","preprocessing":"snv","mc":{"runs":20,"components":2,"ratio":.1},
                "svm":{"C":10.,"gamma":"scale","class_weight":"balanced"}}
        new=fit_source_candidate(config,x,self.y,axis)
        old=build_pipeline("svc","mcuve20",(x.shape[1],),42).set_params(preprocess__mode="snv",model__C=10.)
        old.fit(x,self.y)
        np.testing.assert_array_equal(new.predict(x),old.predict(x))
        np.testing.assert_allclose(new.predict_scores(x),old.decision_function(x),atol=0,rtol=0)
        np.testing.assert_array_equal(new.bundle.selector.selected_indices_,old.named_steps["selector"].selected_indices_)

    def test_selection_uses_macro_then_balanced_not_outer_labels(self):
        a={**self.config,"id":"a"};b={**self.config,"id":"b"}
        rows=[]
        for fold in (0,1):
            rows.extend([{"candidate_id":"a","metrics":{"macro_f1":.8,"balanced_accuracy":.8,"accuracy":.95}},
                         {"candidate_id":"b","metrics":{"macro_f1":.81,"balanced_accuracy":.79,"accuracy":.90}}])
        choice=choose_candidate(rows,[a,b]);self.assertEqual(choice["selected"]["id"],"b")
        self.assertFalse(choice["selected_using_outer_test"])


if __name__=="__main__":unittest.main()
