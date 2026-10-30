import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel
from herb_spectra.algorithms import (ImagePCA, LSSVMRegressor, glcm_texture,
                                    make_bp_classifier, smooth_spectra, sobel_edges, stack_scan_rows)


class AlgorithmReplacementTests(unittest.TestCase):
    def test_lssvm_satisfies_independently_constructed_block_system(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(12, 3)); y = np.sin(X[:, 0]) + 0.3 * X[:, 1] + 4
        model = LSSVMRegressor(C=7.0, kernel_gamma=0.8).fit(X, y)
        K = rbf_kernel(X, gamma=0.8)
        assert_allclose(model.dual_coef_.sum(), 0, atol=1e-12)
        assert_allclose(K @ model.dual_coef_ + model.intercept_ + model.dual_coef_ / 7, y, atol=1e-12)
        self.assertLess(model.system_residual_max_, 1e-12)

    def test_lssvm_intercept_is_unpenalized_and_inference_is_independent(self):
        rng = np.random.default_rng(3); X = rng.normal(size=(20, 2)); y = X[:, 0] ** 2
        a = LSSVMRegressor().fit(X, y); b = LSSVMRegressor().fit(X, y + 17)
        query = rng.normal(size=(6, 2))
        assert_allclose(a.dual_coef_, b.dual_coef_, atol=1e-12)
        assert_allclose(b.predict(query) - a.predict(query), 17, atol=1e-12)
        assert_allclose(a.predict(query[:1]), a.predict(query)[:1])
        with self.assertRaises(ValueError): a.predict(np.ones((2, 3)))
        with self.assertRaises(NotFittedError): LSSVMRegressor().predict(query)
        with self.assertRaises(ValueError): LSSVMRegressor(C=0).fit(X, y)

    def test_image_pca_applies_training_mean_and_preserves_shape(self):
        rng = np.random.default_rng(2); image = rng.normal(size=(9, 11, 4)) + 10
        model = ImagePCA(4).fit(image); scores = model.transform(image)
        self.assertEqual(scores.shape, (9, 11, 4))
        assert_allclose(scores.reshape(-1, 4).mean(axis=0), 0, atol=1e-14)
        assert_allclose(model.inverse_transform(scores), image, atol=1e-13)
        query = rng.normal(size=(3, 7, 4)) + 20
        expected = (query.reshape(-1, 4) - model.pca_.mean_) @ model.pca_.components_.T
        assert_allclose(model.transform(query).reshape(-1, 4), expected)
        with self.assertRaises(ValueError): model.transform(np.ones((4, 4, 3)))

    def test_texture_uses_image_values_and_normalizes_actual_neighbours(self):
        rng = np.random.default_rng(4)
        constant = glcm_texture(np.zeros((16, 16), dtype=np.uint8))
        random = glcm_texture(rng.integers(0, 256, size=(16, 16), dtype=np.uint8))
        self.assertFalse(np.allclose(constant.features, random.features))
        assert_allclose(constant.features, [1, 0, 0, 0, 0, 0, 1, 0])
        assert_allclose(random.matrices.sum(axis=(1, 2)), 1)
        assert_allclose(random.matrices, random.matrices.transpose(0, 2, 1))
        self.assertTrue(np.isfinite(random.features).all())
        self.assertTrue((np.abs(random.per_direction[:, 3]) <= 1).all())

    def test_texture_directional_contrast_on_checkerboard_and_fixed_range(self):
        image = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)
        result = glcm_texture(image, levels=2, value_range=(0, 1))
        assert_allclose(result.per_direction[:, 2], [1, 0, 1, 0])
        assert_allclose(result.per_direction[:, 3], [-1, 1, -1, 1])
        with self.assertRaises(ValueError): glcm_texture(np.array([[0, 256], [0, 1]]))
        with self.assertRaises(ValueError): glcm_texture(np.ones((1, 4)))

    def test_sobel_matches_known_single_impulse_response(self):
        image = np.array([[1.]])
        expected = np.array([[np.sqrt(2), 2, np.sqrt(2)], [2, 0, 2], [np.sqrt(2), 2, np.sqrt(2)]])
        assert_allclose(sobel_edges(image), expected)
        self.assertEqual(sobel_edges(np.zeros((8, 7))).shape, (10, 9))

    def test_sg_preserves_quadratics_and_filters_only_frequency_axis(self):
        x = np.arange(31, dtype=float); signal = 2 * x ** 2 + 3 * x + 1
        values = np.stack([signal, signal + 100])
        assert_allclose(smooth_spectra(values, 7, 2), values, atol=1e-10)
        with self.assertRaises(ValueError): smooth_spectra(values, 8, 2)

    def test_bp_preserves_string_class_labels_and_training_scaler(self):
        rng = np.random.default_rng(9)
        X = np.r_[rng.normal(-2, 0.2, (15, 3)), rng.normal(2, 0.2, (15, 3))]
        y = np.repeat(["origin-A", "origin-B"], 15)
        model = make_bp_classifier(hidden_layer_sizes=(5,), max_iter=100).fit(X, y)
        before = model.named_steps['standardscaler'].mean_.copy()
        self.assertEqual(set(model.classes_), set(y))
        self.assertEqual(set(model.predict(X)), set(y))
        model.predict(X + 1000)
        assert_allclose(model.named_steps['standardscaler'].mean_, before)
        assert_allclose(model.predict_proba(X).sum(axis=1), 1)

    def test_scan_row_adapter_preserves_explicit_selection_and_rejects_mismatch(self):
        arrays = [np.arange(300 * 92).reshape(300, 92) + i for i in range(91)]
        stacked = stack_scan_rows(arrays, 299, slice(1, 92))
        self.assertEqual(stacked.shape, (91, 91))
        assert_allclose(stacked[0], arrays[0][299, 1:92])
        assert_allclose(stacked[-1], arrays[-1][299, 1:92])
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            stack_scan_rows([np.zeros((1, 90)), np.zeros((1, 91))], 0)
        with self.assertRaises(ValueError): stack_scan_rows(arrays, -1)


if __name__ == '__main__':
    unittest.main()
