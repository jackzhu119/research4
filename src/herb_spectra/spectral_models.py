"""Train-only spectral preprocessing and source classifiers.

These are source-model comparisons, not gains attributed to few-shot labels.
The fitted object's prediction and embedding methods never estimate query
statistics. Spectral identifiers, workbook names and origin metadata are not
accepted as numerical features.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend, savgol_filter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .benchmark import PLSDAClassifier


PREPROCESSORS = ("raw", "snv", "sg0_snv", "sg1_snv", "sg2_snv", "linear_detrend_snv")


def _matrix(values, columns=None):
    result = np.asarray(values, dtype=float)
    if (result.ndim != 2 or not len(result) or result.shape[1] < 2
            or not np.isfinite(result).all()
            or (columns is not None and result.shape[1] != columns)):
        raise ValueError("Spectra must be a nonempty finite samples-by-frequency matrix")
    return result


def _axis(values, columns):
    result = np.asarray(values, dtype=float)
    if (result.shape != (columns,) or not np.isfinite(result).all()
            or not np.all(np.diff(result) > 0)):
        raise ValueError("Frequency axis must be finite, increasing and match the spectral columns")
    result = result.copy()
    result.setflags(write=False)
    return result


def _snv(values):
    deviation = values.std(axis=1, ddof=1, keepdims=True)
    if not np.isfinite(deviation).all() or np.any(deviation <= np.finfo(float).eps):
        raise ValueError("SNV is undefined for a constant preprocessed spectrum")
    return (values - values.mean(axis=1, keepdims=True)) / deviation


class SpectralPreprocessor:
    """Per-row spectral operations with a fixed, verified frequency grid."""
    def __init__(self, mode, axis):
        if mode not in PREPROCESSORS:
            raise ValueError("Unknown spectral preprocessing mode")
        self.mode = mode
        values = np.asarray(axis, dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("Frequency axis must be a one-dimensional vector with at least two points")
        self.axis_ = _axis(values, len(values))
        self.n_features_in_ = len(values)
        self.derivative_delta_ = float(np.mean(np.diff(self.axis_)))
        if mode.startswith("sg"):
            if self.n_features_in_ < 15:
                raise ValueError("The fixed Savitzky-Golay window requires at least 15 frequencies")
            steps = np.diff(self.axis_)
            if not np.allclose(steps, self.derivative_delta_, rtol=1e-7,
                               atol=max(1e-12, abs(self.derivative_delta_)*1e-9)):
                raise ValueError("Savitzky-Golay operations require a uniform frequency grid")

    def transform(self, X):
        values = _matrix(X, self.n_features_in_).copy()
        if self.mode == "raw":
            return values
        if self.mode.startswith("sg"):
            derivative = int(self.mode[2])
            values = savgol_filter(values, window_length=15, polyorder=3,
                                   deriv=derivative, delta=self.derivative_delta_,
                                   axis=1, mode="interp")
        elif self.mode == "linear_detrend_snv":
            values = detrend(values, axis=1, type="linear")
        return _snv(values)


def source_candidate_grid():
    """The stable 54-case main source grid, independent of prediction labels."""
    result = []
    models = [
        {"model": "svc", "C": 1.}, {"model": "svc", "C": 10.},
        {"model": "svc", "C": 100.}, {"model": "pca_svc", "C": 10., "n_components": 20},
        {"model": "plsda", "n_components": 5}, {"model": "plsda", "n_components": 10},
        {"model": "shrinkage_lda", "priors": "uniform"},
        {"model": "ridge", "alpha": 1.}, {"model": "ridge", "alpha": 10.},
    ]
    for mode in PREPROCESSORS:
        for model in models:
            result.append({"id": f"s{len(result):03d}_{mode}_{model['model']}",
                           "preprocessing": mode, **model})
    return result


def _candidate(candidate):
    result = dict(candidate)
    if result.get("preprocessing") not in PREPROCESSORS:
        raise ValueError("Unknown source preprocessing")
    if result.get("model") not in {"svc", "pca_svc", "plsda", "shrinkage_lda", "ridge"}:
        raise ValueError("Unknown source classifier")
    name = "C" if result["model"] in {"svc", "pca_svc"} else "alpha"
    if result["model"] in {"svc", "pca_svc", "ridge"}:
        value = result.get(name)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Source {name} must be positive and finite") from exc
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"Source {name} must be positive and finite")
        result[name] = numeric
    if result["model"] in {"pca_svc", "plsda"}:
        value = result.get("n_components")
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or value < 1):
            raise ValueError("Source component count must be a positive integer")
    if result["model"] == "shrinkage_lda" and result.get("priors", "uniform") != "uniform":
        raise ValueError("The declared shrinkage-LDA source uses uniform class priors")
    return result


@dataclass
class FittedSpectralSource:
    candidate: dict
    preprocessor_: SpectralPreprocessor
    scaler_: StandardScaler
    estimator_: object
    classes_: np.ndarray
    metadata: dict
    reducer_: object = None

    def _scaled(self, X):
        return self.scaler_.transform(self.preprocessor_.transform(X))

    def transform(self, X):
        """Return a source-fitted frozen representation, never a query fit."""
        values = self._scaled(X)
        if self.reducer_ is not None:
            return self.reducer_.transform(values)
        if self.candidate["model"] == "plsda":
            return self.estimator_.model_.transform(values)
        return values

    def predict_scores(self, X):
        values = self._scaled(X)
        if self.reducer_ is not None:
            values = self.reducer_.transform(values)
        scores = np.asarray(self.estimator_.decision_function(values), dtype=float)
        if scores.ndim == 1:
            if len(self.classes_) != 2:
                raise ValueError("One-dimensional source scores require exactly two classes")
            scores = np.column_stack([-scores, scores])
        if scores.shape != (len(values), len(self.classes_)) or not np.isfinite(scores).all():
            raise ValueError("Source classifier returned invalid class-aligned scores")
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_scores(X), axis=1)]


def fit_source_candidate(candidate, X_train, y_train, axis, seed=42):
    """Fit all source transforms and one classifier using training arrays only."""
    candidate = _candidate(candidate)
    X = _matrix(X_train)
    if len(X) < 3:
        raise ValueError("At least three source specimens are required")
    supplied = np.asarray(y_train)
    if supplied.ndim != 1 or len(supplied) != len(X):
        raise ValueError("Provide one source class label per specimen")
    if any(value is None or str(value).strip().lower() in {"", "none", "nan", "null"}
           for value in supplied):
        raise ValueError("Source labels contain a missing value")
    y = np.asarray([str(value).strip() for value in supplied])
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Source classification requires at least two classes")
    frequency = _axis(axis, X.shape[1])
    preprocessor = SpectralPreprocessor(candidate["preprocessing"], frequency)
    preprocessed = preprocessor.transform(X)
    scaler = StandardScaler()
    values = scaler.fit_transform(preprocessed)
    kind = candidate["model"]
    reducer, components = None, None
    if kind in {"pca_svc", "plsda"}:
        components = min(int(candidate["n_components"]), len(values)-1, values.shape[1])
    if kind == "pca_svc":
        reducer = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        values = reducer.fit_transform(values)
    if kind in {"svc", "pca_svc"}:
        estimator = SVC(C=float(candidate["C"]), kernel="rbf", gamma="scale",
                        class_weight="balanced", break_ties=True,
                        decision_function_shape="ovr", random_state=seed)
    elif kind == "plsda":
        estimator = PLSDAClassifier(n_components=components)
    elif kind == "shrinkage_lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                               priors=np.full(len(classes), 1/len(classes)))
    else:
        estimator = RidgeClassifier(alpha=float(candidate["alpha"]), class_weight="balanced")
    estimator.fit(values, y)
    if not np.array_equal(estimator.classes_, classes):
        raise AssertionError("Source classifier class order differs from the fitted vocabulary")
    frozen_classes = classes.copy(); frozen_classes.setflags(write=False)
    embedding = ("source_pca" if kind == "pca_svc" else "source_pls_x_scores"
                 if kind == "plsda" else "source_standardized_spectral_features")
    metadata = {"candidate": candidate.copy(), "fit_scope": "source_training_only",
                "source_samples": len(values), "input_features": X.shape[1],
                "class_order": classes.tolist(),
                "class_counts": {str(c): int(np.count_nonzero(y == c)) for c in classes},
                "frequency_range": [float(frequency[0]), float(frequency[-1])],
                "frequency_step": preprocessor.derivative_delta_,
                "SG_window": 15 if candidate["preprocessing"].startswith("sg") else None,
                "SG_polyorder": 3 if candidate["preprocessing"].startswith("sg") else None,
                "requested_components": candidate.get("n_components"),
                "effective_components": components, "embedding": embedding,
                "lda_priors": "uniform" if kind == "shrinkage_lda" else None,
                "query_statistics_fitted": False, "scores_are_probabilities": False,
                "prediction_rule": "argmax_class_aligned_scores; first source class on exact ties"}
    return FittedSpectralSource(candidate, preprocessor, scaler, estimator,
                                frozen_classes, metadata, reducer)
