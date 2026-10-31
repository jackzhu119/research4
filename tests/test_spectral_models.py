"""Synthetic tests of train-only source models, not herbal performance claims."""
from copy import deepcopy
import json
import unittest
import warnings

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import savgol_filter

from herb_spectra.spectral_models import (SpectralPreprocessor, fit_source_candidate,
                                          source_candidate_grid)


def fixture(counts=(14, 14, 14), seed=3):
    rng = np.random.default_rng(seed)
    axis = np.linspace(.1, 1.5, 31)
    y = np.repeat(["ZY01", "ZY05", "ZY07"], counts)
    X = rng.normal(scale=.1, size=(len(y), len(axis)))
    for i, label in enumerate(np.unique(y)):
        X[y == label] += np.sin(axis*(i+1)*3) + .2*i*axis**2
    return X, y, axis


class SpectralModelTests(unittest.TestCase):
    def test_grid_stable_complete_and_json_safe(self):
        grid = source_candidate_grid()
        self.assertEqual(grid, source_candidate_grid())
        self.assertEqual(len(grid), 54)
        self.assertEqual(len({c["id"] for c in grid}), 54)
        self.assertEqual(len({c["preprocessing"] for c in grid}), 6)
        json.dumps(grid, allow_nan=False)

    def test_all_sources_have_finite_aligned_scores_and_frozen_embeddings(self):
        X, y, axis = fixture()
        for candidate in source_candidate_grid():
            model = fit_source_candidate(candidate, X, y, axis)
            before = joblib.hash(model)
            scores, latent = model.predict_scores(X[:4]), model.transform(X[:4])
            self.assertEqual(scores.shape, (4, 3), candidate)
            self.assertEqual(len(latent), 4)
            self.assertTrue(np.isfinite(latent).all())
            assert_array_equal(model.predict(X[:4]), model.classes_[np.argmax(scores, axis=1)])
            assert_array_equal(model.classes_, ["ZY01", "ZY05", "ZY07"])
            self.assertEqual(joblib.hash(model), before)

    def test_sg_matches_frequency_delta_calculation_and_rejects_irregular_grid(self):
        X, _, axis = fixture()
        for derivative in (0,1,2):
            processor = SpectralPreprocessor(f"sg{derivative}_snv", axis)
            filtered = savgol_filter(X[:2], 15, 3, deriv=derivative,
                                     delta=np.mean(np.diff(axis)), axis=1, mode="interp")
            expected = (filtered-filtered.mean(axis=1, keepdims=True))/filtered.std(axis=1,ddof=1,keepdims=True)
            assert_allclose(processor.transform(X[:2]), expected, atol=1e-12)
            self.assertAlmostEqual(processor.derivative_delta_, .04666666666666666)
        irregular = axis.copy();irregular[10] += .002
        with self.assertRaisesRegex(ValueError, "uniform"):
            SpectralPreprocessor("sg1_snv", irregular)
        # Raw spectra do not require equidistant coordinates.
        SpectralPreprocessor("raw", irregular).transform(X[:2])
        with self.assertRaisesRegex(ValueError, "15"):
            SpectralPreprocessor("sg0_snv", axis[:10])

    def test_queries_never_refit_scaler_pca_or_per_row_operations(self):
        X, y, axis = fixture()
        candidate = next(c for c in source_candidate_grid()
                         if c["model"] == "pca_svc" and c["preprocessing"] == "snv")
        model = fit_source_candidate(candidate, X, y, axis)
        snv = (X-X.mean(axis=1,keepdims=True))/X.std(axis=1,ddof=1,keepdims=True)
        assert_allclose(model.scaler_.mean_, snv.mean(axis=0))
        before = joblib.hash(model)
        extra = np.full((2, X.shape[1]), 1000.) + np.linspace(-10,20,X.shape[1])
        single = model.predict_scores(X[0:1])
        together = model.predict_scores(np.vstack([extra, X[0:1]]))
        assert_allclose(single, together[-1:], atol=1e-11)
        assert_allclose(model.transform(X[0:1]),model.transform(np.vstack([extra,X[0:1]]))[-1:],atol=1e-11)
        self.assertEqual(joblib.hash(model), before)

    def test_binary_decision_margins_become_two_class_aligned_columns(self):
        X, y, axis = fixture(counts=(12,12,12))
        keep = y != "ZY05"
        for kind in ("svc", "pca_svc", "shrinkage_lda", "ridge"):
            candidate = next(c for c in source_candidate_grid() if c["model"] == kind)
            model = fit_source_candidate(candidate, X[keep], y[keep], axis)
            scores = model.predict_scores(X[:3])
            self.assertEqual(scores.shape, (3,2))
            assert_allclose(scores[:,0],-scores[:,1])
            assert_array_equal(model.classes_, ["ZY01","ZY07"])

    def test_pls_targets_are_one_hot_and_invariant_to_unordered_label_names(self):
        X, y, axis = fixture()
        candidate = next(c for c in source_candidate_grid() if c["model"] == "plsda")
        first = fit_source_candidate(candidate,X,y,axis)
        rename = {"ZY01":"zeta","ZY05":"alpha","ZY07":"middle"}
        changed = np.array([rename[v] for v in y])
        second = fit_source_candidate(candidate,X,changed,axis)
        # A vocabulary permutation changes score column order, not numeric label distances.
        order = [list(second.classes_).index(rename[v]) for v in first.classes_]
        # NIPALS terminates at the existing library tolerance and can differ
        # slightly after response-column permutations; the class predictions
        # and one-hot target meaning must remain unchanged.
        assert_allclose(first.predict_scores(X),second.predict_scores(X)[:,order],rtol=0,atol=1e-5)
        assert_array_equal(np.asarray([rename[v] for v in first.predict(X)]),second.predict(X))
        self.assertEqual(first.transform(X[:2]).shape,(2,5))
        self.assertEqual(first.estimator_.model_._y_mean.shape,(3,))

    def test_one_source_specimen_of_rare_class_remains_in_every_classifier(self):
        X, y, axis = fixture(counts=(15,1,15))
        for kind in ("svc","pca_svc","plsda","shrinkage_lda","ridge"):
            candidate=next(c for c in source_candidate_grid() if c["model"]==kind)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",UserWarning)
                model=fit_source_candidate(candidate,X,y,axis)
            assert_array_equal(model.classes_,["ZY01","ZY05","ZY07"])
            self.assertTrue(np.isfinite(model.predict_scores(X[:2])).all())
            self.assertEqual(model.metadata["class_counts"]["ZY05"],1)

    def test_invalid_shapes_labels_axis_components_and_constant_snv_fail(self):
        X,y,axis=fixture()
        c=source_candidate_grid()[0]
        for bad_axis in (axis[::-1],axis[:-1],np.full(len(axis),np.nan)):
            with self.assertRaisesRegex(ValueError,"axis"):
                fit_source_candidate(c,X,y,bad_axis)
        with self.assertRaisesRegex(ValueError,"missing"):
            fit_source_candidate(c,X,np.array(["" if i==0 else v for i,v in enumerate(y)]),axis)
        with self.assertRaisesRegex(ValueError,"two classes"):
            fit_source_candidate(c,X,np.repeat("same",len(y)),axis)
        wrong=deepcopy(next(c for c in source_candidate_grid() if c["model"]=="plsda"));wrong["n_components"]=False
        with self.assertRaisesRegex(ValueError,"component"):
            fit_source_candidate(wrong,X,y,axis)
        for invalid in (None,False,-1.,np.nan,"bad"):
            wrong=deepcopy(c);wrong["C"]=invalid
            with self.assertRaisesRegex(ValueError,"positive"):
                fit_source_candidate(wrong,X,y,axis)
        with self.assertRaisesRegex(ValueError,"constant"):
            SpectralPreprocessor("snv",axis).transform(np.ones((2,len(axis))))
        model=fit_source_candidate(c,X,y,axis)
        with self.assertRaisesRegex(ValueError,"matrix"):
            model.predict_scores(X[:2,:-1])


if __name__ == "__main__":
    unittest.main()
