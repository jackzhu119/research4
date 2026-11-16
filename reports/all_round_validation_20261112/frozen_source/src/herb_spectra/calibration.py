"""Small-support calibration candidates with explicit source-model anchoring.

These are independent engineering implementations of standard affine and ridge
ideas, not claimed new algorithms. Fit uses only labelled target support data.
The caller must keep support and final query physical samples disjoint, choose
regularization without final-query labels, and preserve the source model. No
query-dependent means, scales, labels, or distribution alignment are fitted.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _positive_regularization(value, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("regularization must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regularization must be a finite real number") from exc
    if not np.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError("regularization must be finite and " + ("nonnegative" if allow_zero else "positive"))
    return number


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _scores(values, n_classes):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not len(array) or array.shape[1] != n_classes or not np.isfinite(array).all():
        raise ValueError("scores must be a nonempty finite matrix with one column per source class")
    return array


def _missing_label(value):
    return value is None or str(value).strip().lower() in {"", "nan", "none"}


class ShrinkageAffineCalibrator(RegressorMixin, BaseEstimator):
    """Adapt source predictions using a small labelled target support set.

    Solve ``sum((y-p)/s - Z @ delta)**2 + lambda*||delta||²`` where
    ``Z = [(p-mean_support_prediction)/s, 1]``. Prediction is
    ``p + s * Z @ delta``; delta=0 is the identity map. Thus the ridge prior
    shrinks the slope toward 1 and the correction at the support centre toward
    0. The penalty uses a *sum* loss, so more support observations outweigh the
    same fixed prior. ``regularization=0`` is the unregularized affine baseline.

    Scale ``s`` is an explicitly supplied positive source-target scale, or the
    population standard deviation of support predictions (fallback 1 if zero).
    It is never obtained from query data. With one support observation, or
    constant support predictions, only a bias is identifiable; slope stays 1.
    ``mode='bias'`` always uses this bias-only option. This is point calibration,
    not prediction-interval construction or chemical-reference validation.
    """

    def __init__(self, regularization=1.0, source_y_scale=None, mode="affine"):
        self.regularization = regularization
        self.source_y_scale = source_y_scale
        self.mode = mode

    def fit(self, pred_support, y_support):
        prediction = _vector(pred_support, "support predictions")
        target = _vector(y_support, "support targets")
        if len(prediction) != len(target):
            raise ValueError("Support prediction and target lengths differ")
        penalty = _positive_regularization(self.regularization, allow_zero=True)
        if self.mode not in {"affine", "bias"}:
            raise ValueError("mode must be affine or bias")
        centre = float(np.mean(prediction))
        prediction_scale = float(np.std(prediction))
        if self.source_y_scale is None:
            scale = prediction_scale if prediction_scale > np.finfo(float).eps else 1.0
        else:
            if isinstance(self.source_y_scale, (bool, np.bool_)):
                raise ValueError("source_y_scale must be finite and positive")
            scale = float(self.source_y_scale)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("source_y_scale must be finite and positive")
        if self.mode == "bias":
            fit_mode = "bias_only_requested"
        elif len(prediction) == 1:
            fit_mode = "bias_only_one_shot"
        elif prediction_scale <= np.finfo(float).eps * max(1.0, abs(centre)):
            fit_mode = "bias_only_constant_prediction"
        else:
            fit_mode = "affine"
        residual = (target - prediction) / scale
        if fit_mode != "affine":
            delta = np.asarray([0.0, residual.sum() / (len(target) + penalty)])
        else:
            design = np.column_stack([(prediction - centre) / scale, np.ones(len(prediction))])
            if penalty == 0:
                delta = np.linalg.lstsq(design, residual, rcond=None)[0]
            else:
                delta = np.linalg.solve(design.T @ design + penalty * np.eye(2), design.T @ residual)
        slope = 1.0 + float(delta[0])
        intercept = scale * float(delta[1]) - centre * float(delta[0])
        if not np.isfinite([slope, intercept]).all():
            raise ValueError("Nonfinite affine correction; inspect scales and support values")
        self.support_count_ = len(prediction)
        self.support_prediction_mean_ = centre
        self.source_y_scale_ = scale
        self.scale_origin_ = "explicit_source_scale" if self.source_y_scale is not None else "support_predictions_only"
        self.fit_mode_ = fit_mode
        self.correction_coefficients_ = delta.copy()
        self.slope_ = slope
        self.intercept_ = intercept
        self.n_features_in_ = 1
        return self

    def predict(self, pred_query):
        check_is_fitted(self, ["slope_", "intercept_"])
        prediction = _vector(pred_query, "query predictions")
        calibrated = self.slope_ * prediction + self.intercept_
        if not np.isfinite(calibrated).all():
            raise ValueError("Affine prediction overflow; inspect query and correction scales")
        return calibrated


class AnchorScoreCalibrator(ClassifierMixin, BaseEstimator):
    """Fit a source-anchored linear score correction on labelled support only.

    The caller supplies a fixed source class vocabulary and N x K decision or
    PLS-DA scores in precisely that order. Fit solves ridge residual regression
    from support scores (plus an intercept) to ``onehot(y)-source_scores``.
    Calibrated query scores are ``source_scores + [source_scores,1] @ delta``.
    The prior delta=0 is the source identity map, including zero added bias.
    All weights and the intercept are penalized. Regularization must be positive
    because K-shot support can be rank deficient. No probabilities are claimed;
    these scores are used only for argmax classification.

    By default all source classes must occur in support. Disabling that check
    is explicit and does not establish reliable calibration of missing classes.
    Raw score units affect the ridge fit; fix the source score definition and
    choose regularization in source/validation episodes, not on final query.
    """

    def __init__(self, classes, regularization=1.0, require_all_classes=True):
        self.classes = classes
        self.regularization = regularization
        self.require_all_classes = require_all_classes

    def fit(self, score_support, y_support):
        penalty = _positive_regularization(self.regularization)
        classes = np.asarray(self.classes)
        if classes.ndim != 1 or len(classes) < 2 or any(_missing_label(v) for v in classes):
            raise ValueError("Provide at least two nonmissing source classes in score-column order")
        if len(set(classes.tolist())) != len(classes):
            raise ValueError("Source classes must be unique")
        if not isinstance(self.require_all_classes, (bool, np.bool_)):
            raise ValueError("require_all_classes must be boolean")
        scores = _scores(score_support, len(classes))
        target = np.asarray(y_support)
        if target.ndim != 1 or len(target) != len(scores) or any(_missing_label(v) for v in target):
            raise ValueError("One nonmissing support label is required per score row")
        vocabulary = {v: k for k, v in enumerate(classes.tolist())}
        if any(v not in vocabulary for v in target.tolist()):
            raise ValueError("Support labels include a class absent from the source vocabulary")
        present = set(target.tolist())
        if self.require_all_classes and present != set(classes.tolist()):
            raise ValueError("Support is missing a source class")
        encoded = np.asarray([vocabulary[v] for v in target.tolist()])
        one_hot = np.eye(len(classes))[encoded]
        design = np.column_stack([scores, np.ones(len(scores))])
        delta = np.linalg.solve(design.T @ design + penalty * np.eye(len(classes) + 1),
                                design.T @ (one_hot - scores))
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite score correction; inspect support score scale")
        self.classes_ = classes.copy()
        self.correction_coefficients_ = delta.copy()
        self.coef_ = np.eye(len(classes)) + delta[:-1]
        self.intercept_ = delta[-1].copy()
        self.support_count_ = len(scores)
        self.support_counts_by_class_ = {str(v): int(np.count_nonzero(target == v)) for v in classes}
        self.n_features_in_ = len(classes)
        self.missing_support_classes_ = [v for v in classes.tolist() if v not in present]
        return self

    def transform(self, score_query):
        check_is_fitted(self, ["coef_", "intercept_", "classes_"])
        scores = _scores(score_query, len(self.classes_))
        calibrated = scores @ self.coef_ + self.intercept_
        if not np.isfinite(calibrated).all():
            raise ValueError("Score correction overflow; inspect query and source score scales")
        return calibrated

    def decision_function(self, score_query):
        return self.transform(score_query)

    def predict(self, score_query):
        check_is_fitted(self, ["coef_", "intercept_", "classes_"])
        return self.classes_[np.argmax(self.transform(score_query), axis=1)]
