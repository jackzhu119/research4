import unittest

import joblib
import numpy as np

from herb_spectra.image_feature_models import (ScatterPreprocessor, anchored_folds,
    candidate_grid, cars_path, fit_candidate, noise_uve_selection, spa_chain)


class ImageFeatureModelsTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(8)
        y=np.asarray(["A"]*12+["B"]*12+["C"])
        X=rng.normal(size=(len(y),20))
        X[y=="B",:4]+=2
        return X,y,np.linspace(.1,1.5,20)

    def test_grid_is_40_distinct_and_both_scatter_modes(self):
        grid=candidate_grid()
        self.assertEqual(len(grid),40)
        self.assertEqual(len(set(c["id"] for c in grid)),40)
        self.assertEqual({c["preprocessing"] for c in grid},{"snv","msc"})

    def test_msc_reference_source_only_and_affine_scatter(self):
        r=np.linspace(1,3,20)**2
        source=np.vstack([r,2*r+3,.5*r-1])
        prep=ScatterPreprocessor("msc").fit(source)
        reference=prep.reference_.copy(); row=4*r+8
        a=prep.transform(row[None])
        b=prep.transform(np.vstack([row,-7*r+1]))[:1]
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-13)
        np.testing.assert_allclose(a[0],reference,atol=1e-12)
        np.testing.assert_array_equal(prep.reference_,reference)
        with self.assertRaisesRegex(ValueError,"near zero"):
            prep.transform(np.ones((1,20)))

    def test_anchored_folds_never_validate_singleton(self):
        _,y,_=self.fixture(); folds,policy=anchored_folds(y)
        self.assertEqual(policy["anchor_classes"],["C"])
        validation=[]
        for train,valid in folds:
            self.assertIn(24,train);self.assertNotIn(24,valid)
            self.assertFalse(set(train)&set(valid));validation.extend(valid)
        self.assertEqual(sorted(validation),list(range(24)))

    def test_spa_rejects_redundant_column_and_stops_at_rank(self):
        r=np.linspace(-1,1,30)
        X=np.column_stack([r,2*r,-3*r])
        indices,details=spa_chain(X,maximum=10)
        self.assertEqual(indices.tolist(),[2])
        self.assertTrue(details["numerical_rank_stop"])

    def test_noise_uve_fallback_explicit_and_threshold(self):
        indices,details=noise_uve_selection(np.asarray([.1,.5,.2]),np.ones(10),.95)
        self.assertEqual(indices.tolist(),[1]);self.assertTrue(details["empty_set_fallback_used"])
        indices,details=noise_uve_selection(np.asarray([2.,.5,3.]),np.ones(10),.95)
        self.assertEqual(indices.tolist(),[0,2]);self.assertFalse(details["empty_set_fallback_used"])

    def test_source_freeze_pointwise_predictions_and_cache_guard(self):
        X,y,axis=self.fixture();cache={}
        c=next(c for c in candidate_grid() if c["preprocessing"]=="snv" and c["family"]=="full_svm")
        model=fit_candidate(c,X,y,axis,cache=cache)
        before=joblib.hash(model)
        first=model.predict_scores(X[:1]);together=model.predict_scores(X[:5])[:1]
        np.testing.assert_allclose(first,together,rtol=0,atol=1e-12)
        self.assertEqual(before,joblib.hash(model))
        with self.assertRaisesRegex(ValueError,"cannot cross"):
            fit_candidate(c,X+.1,y,axis,cache=cache)

    def test_cars_anchored_classification_path_feasible(self):
        X,y,_=self.fixture();indices,details=cars_path(X,y)
        self.assertGreater(len(indices),0)
        self.assertEqual(len(details["path"]),16)
        self.assertEqual(details["internal_cv"]["anchor_classes"],[2])
        self.assertTrue(np.isfinite([r["inner_onehot_pls_mse"] for r in details["path"]]).all())


if __name__=="__main__":unittest.main()
