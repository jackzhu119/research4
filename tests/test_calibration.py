import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from herb_spectra.calibration import AnchorScoreCalibrator, ShrinkageAffineCalibrator


class SmallSupportCalibrationTests(unittest.TestCase):
    def test_affine_known_shift_and_scale_recovers_unseen_values(self):
        support = np.asarray([-4., -2., 1., 3., 6.])
        target = 1.4 * support + 7
        support_before, target_before = support.copy(), target.copy()
        model = ShrinkageAffineCalibrator(regularization=0, source_y_scale=4).fit(support, target)
        query = np.asarray([-10., 0., 12.])
        assert_allclose(model.predict(query), 1.4 * query + 7, atol=1e-12)
        self.assertAlmostEqual(model.slope_, 1.4)
        self.assertAlmostEqual(model.intercept_, 7.)
        assert_array_equal(support, support_before); assert_array_equal(target, target_before)

    def test_identity_prior_and_unit_equivariance(self):
        support = np.asarray([1., 2., 5., 8.])
        target = support * 2 + 10
        weak = ShrinkageAffineCalibrator(.1, source_y_scale=3).fit(support, target)
        strong = ShrinkageAffineCalibrator(1e12, source_y_scale=3).fit(support, target)
        assert_allclose(strong.predict(support), support, atol=1e-9)
        self.assertLess(np.mean((weak.predict(support)-target)**2), np.mean((support-target)**2))
        changed_units = ShrinkageAffineCalibrator(.1, source_y_scale=3000).fit(1000*support, 1000*target)
        assert_allclose(changed_units.predict(1000*support), 1000*weak.predict(support), rtol=1e-12)

    def test_one_shot_and_constant_predictions_use_bias_only(self):
        model = ShrinkageAffineCalibrator(regularization=1, source_y_scale=20).fit([3], [11])
        self.assertEqual(model.fit_mode_, 'bias_only_one_shot')
        self.assertEqual(model.slope_, 1)
        self.assertEqual(model.support_count_, 1)
        assert_allclose(model.predict([0, 10]), [4, 14])
        constant = ShrinkageAffineCalibrator(0).fit([3,3,3], [5,7,9])
        self.assertEqual(constant.fit_mode_, 'bias_only_constant_prediction')
        assert_allclose(constant.predict([3]), [7])

    def test_regression_query_batches_do_not_change_fit_or_use_query_statistics(self):
        model = ShrinkageAffineCalibrator(.2).fit([1, 3, 5], [4, 6, 8])
        state = (model.slope_, model.intercept_, model.source_y_scale_, model.support_count_)
        alone = model.predict([4])
        together = model.predict([4, 1e10, -1e10])
        assert_allclose(alone, together[:1])
        self.assertEqual(state, (model.slope_, model.intercept_, model.source_y_scale_, model.support_count_))
        self.assertEqual(set(model.__dict__) & {'X_query_', 'query_mean_', 'y_query_'}, set())

    def test_class_score_correction_can_fix_support_domain_offset(self):
        classes = np.asarray(['west', 'east', 'north'])
        support = np.tile(np.eye(3), (3,1))
        support[:,0] += 1.4
        target = np.tile(classes, 3)
        before = support.copy()
        model = AnchorScoreCalibrator(classes, regularization=.01).fit(support,target)
        query = np.eye(3) + np.asarray([1.4,0,0])
        assert_array_equal(model.predict(query), classes)
        self.assertEqual(model.support_count_,9)
        assert_array_equal(support,before)
        strong = AnchorScoreCalibrator(classes,regularization=1e12).fit(support,target)
        assert_allclose(strong.transform(query),query,atol=1e-10)

    def test_score_vocabulary_permutation_and_query_independence(self):
        rng = np.random.default_rng(11)
        classes = np.asarray(['origin-C','origin-A','origin-B'])
        support = np.tile(np.eye(3),(4,1)) + rng.normal(0,.08,(12,3))
        target = np.tile(classes,4)
        query = rng.normal(size=(7,3))
        model = AnchorScoreCalibrator(classes,.7).fit(support,target)
        permutation = np.asarray([2,0,1])
        equivalent = AnchorScoreCalibrator(classes[permutation],.7).fit(support[:,permutation],target)
        assert_array_equal(model.predict(query),equivalent.predict(query[:,permutation]))
        state = model.coef_.copy(),model.intercept_.copy()
        assert_allclose(model.transform(query[:1]),model.transform(np.r_[query[:1],np.ones((3,3))*1e8])[:1])
        assert_array_equal(model.coef_,state[0]); assert_array_equal(model.intercept_,state[1])
        clone(model)  # Constructors preserve sklearn cloning semantics.

    def test_bad_shapes_labels_scales_and_unfitted_use_fail_explicitly(self):
        with self.assertRaises(NotFittedError):ShrinkageAffineCalibrator().predict([1])
        with self.assertRaises(NotFittedError):AnchorScoreCalibrator(['a','b']).predict([[1,0]])
        for model in [ShrinkageAffineCalibrator(-1), ShrinkageAffineCalibrator(source_y_scale=0),
                      ShrinkageAffineCalibrator(mode='unknown')]:
            with self.assertRaises(ValueError):model.fit([1,2],[2,3])
        with self.assertRaises(ValueError):ShrinkageAffineCalibrator().fit([np.nan],[1])
        with self.assertRaises(ValueError):ShrinkageAffineCalibrator().fit([1,2],[1])
        for model, scores, target in [
            (AnchorScoreCalibrator(['a','b']), [[1,0]], ['a']),
            (AnchorScoreCalibrator(['a','b']), [[1,0],[0,1]], ['a','c']),
            (AnchorScoreCalibrator(['a','a']), [[1,0],[0,1]], ['a','a']),
            (AnchorScoreCalibrator(['a','b'],0), [[1,0],[0,1]], ['a','b']),
            (AnchorScoreCalibrator(['a','b']), [[1,np.inf],[0,1]], ['a','b']),
        ]:
            with self.assertRaises(ValueError):model.fit(scores,target)
        explicit_partial = AnchorScoreCalibrator(['a','b'], require_all_classes=False).fit([[1,0]],['a'])
        self.assertEqual(explicit_partial.missing_support_classes_,['b'])
        with self.assertRaises(ValueError):explicit_partial.transform([[1,2,3]])


if __name__=='__main__':unittest.main()
