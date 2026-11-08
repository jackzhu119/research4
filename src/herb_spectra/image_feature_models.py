"""Source-only feature-selection experiments inspired by the supplied figure.

These bounded Python CARS/MC-UVE/noise-UVE/SPA implementations are explicitly
classification adaptations, not verbatim reproductions of archived MATLAB.
"""
from dataclasses import dataclass
import warnings

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .selection import MCUVESelector, _coefficients, _subsample


def candidate_grid():
    settings = ([{"family": "full_svm", "C": c} for c in (1., 10., 100.)]
        + [{"family": "cars_svm", "C": 10.}]
        + [{"family": "mcuve_svm", "C": 10., "features": k} for k in (20, 50, 100)]
        + [{"family": "noise_uve_svm", "C": 10., "noise_quantile": q} for q in (.95, .99)]
        + [{"family": "spa_svm", "C": 10., "features": k} for k in (10, 20, 50)]
        + [{"family": "lda", "priors": p} for p in ("uniform", "empirical")]
        + [{"family": "pca_lda", "components": k, "priors": p}
           for k in (10, 20, 40) for p in ("uniform", "empirical")])
    result = []
    for prep in ("snv", "msc"):
        for settings_ in settings:
            result.append({"id": f"f{len(result):03d}_{prep}_{settings_['family']}",
                           "preprocessing": prep, **settings_})
    return result


def matrix(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not len(x) or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty samples-by-frequency spectra")
    return x


class ScatterPreprocessor:
    """SNV or MSC with its reference fitted exclusively on source spectra."""
    def __init__(self, mode):
        if mode not in ("snv", "msc"):
            raise ValueError("Unknown scatter preprocessing")
        self.mode = mode

    def fit(self, X):
        x = matrix(X)
        self.n_features_in_ = x.shape[1]
        self.reference_ = x.mean(0).copy() if self.mode == "msc" else None
        if self.reference_ is not None:
            centered = self.reference_-self.reference_.mean()
            if np.dot(centered, centered) <= np.finfo(float).eps:
                raise ValueError("MSC source mean reference is constant")
        return self

    def transform(self, X):
        x = matrix(X)
        if not hasattr(self, "n_features_in_") or x.shape[1] != self.n_features_in_:
            raise ValueError("Preprocessor has no matching source frequency grid")
        if self.mode == "snv":
            scale = x.std(1, ddof=1, keepdims=True)
            if np.any(scale <= np.finfo(float).eps):
                raise ValueError("SNV does not accept constant spectra")
            return (x-x.mean(1, keepdims=True))/scale
        ref = self.reference_; centered = ref-ref.mean()
        slope = (x-x.mean(1, keepdims=True))@centered/np.dot(centered, centered)
        if np.any(~np.isfinite(slope)) or np.any(np.abs(slope) < 1e-10):
            raise ValueError("MSC row slope is near zero; refusing unstable correction")
        intercept = x.mean(1)-slope*ref.mean()
        result = (x-intercept[:, None])/slope[:, None]
        if not np.isfinite(result).all():
            raise ValueError("MSC correction produced nonfinite values")
        return result


def anchored_folds(labels, n_splits=3, seed=42):
    """Singleton/rare classes stay in every training fold, never validation.

    This source-only inner criterion does not estimate rare-class generalization.
    The outer development query retains all seven classes.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    anchor_classes = classes[counts < n_splits]
    anchors = np.flatnonzero(np.isin(labels, anchor_classes))
    eligible = np.flatnonzero(~np.isin(labels, anchor_classes))
    if len(np.unique(labels[eligible])) < 2:
        raise ValueError("Need two sufficiently represented classes for inner subset search")
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = [(np.sort(np.r_[eligible[t], anchors]), eligible[v])
             for t,v in splitter.split(eligible, labels[eligible])]
    return folds, {"inner_splits": n_splits, "anchor_classes": anchor_classes.tolist(),
        "anchor_source_indices": anchors.tolist(), "validation_excludes_anchor_classes": True,
        "inner_mse_is_subset_search_only": True}


def _subset_mse(X, Y, indices, folds):
    errors = []
    for train, valid in folds:
        scaler = StandardScaler()
        a = scaler.fit_transform(X[train][:, indices]); b = scaler.transform(X[valid][:, indices])
        model = PLSRegression(n_components=min(3, len(indices), len(train)-1), scale=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y residual is constant.*")
            model.fit(a, Y[train])
        errors.append(float(np.mean((model.predict(b)-Y[valid])**2)))
    return float(np.mean(errors))


def cars_path(X, labels, seed=42):
    x = matrix(X); labels = np.asarray(labels)
    classes, local = np.unique(labels, return_inverse=True); Y = np.eye(len(classes))[local]
    folds, policy = anchored_folds(local, 3, seed)
    rng = np.random.default_rng(seed); active = np.arange(x.shape[1]); path = []
    runs, minimum = 16, min(10, x.shape[1])
    for iteration in range(runs):
        active_before=active.copy()
        sample = _subsample(local, len(x), .8, rng)
        coefs = _coefficients(x[sample][:, active], Y[sample], 3)
        importance = np.linalg.norm(coefs, axis=0)
        wanted = min(len(active), max(minimum, int(round(x.shape[1]*(minimum/x.shape[1])**(iteration/(runs-1))))))
        rank = np.argsort(-importance, kind="stable")
        retained = rank[:wanted]
        if wanted > minimum and importance[retained].sum() > 0:
            probabilities = importance[retained]/importance[retained].sum()
            retained = np.unique(rng.choice(retained, size=len(active), replace=True, p=probabilities))
            if len(retained) < minimum:
                retained = np.unique(np.r_[retained, rank[:minimum]])
        active = np.sort(active[retained])
        path.append({"iteration": iteration, "selected_indices": active.tolist(),
                     "coefficient_indices":active_before.tolist(),"coefficient_norm_importance":importance.tolist(),
                     "inner_onehot_pls_mse": _subset_mse(x, Y, active, folds)})
    best = min(range(runs), key=lambda i: (path[i]["inner_onehot_pls_mse"], len(path[i]["selected_indices"])))
    return np.asarray(path[best]["selected_indices"]), {"method": "bounded classification CARS-style",
        "runs": runs, "components": 3, "best_iteration": best, "path": path,
        "internal_cv": policy, "exact_archived_matlab_reproduction": False}


def noise_uve_statistics(X, labels, seed=42):
    x = matrix(X); classes, local = np.unique(labels, return_inverse=True)
    Y = np.eye(len(classes))[local]; rng = np.random.default_rng(seed)
    noise = rng.normal(size=x.shape)
    # Noise is fitted once to source-only standard-deviation units.
    noise = StandardScaler().fit_transform(noise)
    augmented = np.column_stack([x, noise]); coefficients = []
    for _ in range(30):
        sample = _subsample(local, len(x), .8, rng)
        coefficients.append(_coefficients(augmented[sample], Y[sample], 3))
    coefficients = np.asarray(coefficients)
    deviation = coefficients.std(0, ddof=1)
    floor = max(1., float(deviation.max()))*1e-12
    stability = np.linalg.norm(np.abs(coefficients.mean(0)/np.maximum(deviation, floor)), axis=0)
    return stability[:x.shape[1]], stability[x.shape[1]:]


def noise_uve_selection(real, noise, quantile):
    threshold = float(np.quantile(noise, quantile))
    indices = np.flatnonzero(np.asarray(real) > threshold)
    empty = len(indices) == 0
    if empty:
        indices = np.asarray([int(np.argmax(real))])
    return indices, {"method": "noise-threshold MC-UVE-style", "runs": 30,
        "components": 3, "noise_quantile": quantile, "noise_threshold": threshold,
        "real_stability": np.asarray(real).tolist(), "noise_stability": np.asarray(noise).tolist(),
        "empty_set_fallback_used": empty, "fallback_rule": "single largest real stability",
        "exact_archived_matlab_reproduction": False}


def spa_chain(X, maximum=50, initial_scores=None):
    """A deterministic single-chain successive-orthogonal-projection selector."""
    x = matrix(X); residual = x-x.mean(0); initial_norm = np.sum(residual**2, axis=0)
    scores = initial_norm if initial_scores is None else np.asarray(initial_scores, dtype=float)
    if scores.shape != (x.shape[1],) or not np.isfinite(scores).all():
        raise ValueError("SPA source-only initial scores have wrong shape")
    chosen = []; norms = []; available = np.ones(x.shape[1], dtype=bool)
    relative_floor = max(float(initial_norm.max()), 1.)*1e-12
    for step in range(min(maximum, len(x)-1, x.shape[1])):
        lengths = np.sum(residual**2, axis=0)
        if step == 0:
            index = int(np.argmax(np.where(available, scores, -np.inf)))
        else:
            index = int(np.argmax(np.where(available, lengths, -np.inf)))
        norm = float(lengths[index])
        if norm <= relative_floor:
            break
        direction = residual[:, index]/np.sqrt(norm)
        residual = residual-direction[:,None]*(direction@residual)[None,:]
        # Reorthogonalize once against the latest direction to limit drift.
        residual = residual-direction[:,None]*(direction@residual)[None,:]
        chosen.append(index); norms.append(norm); available[index] = False
    if not chosen:
        raise ValueError("SPA found no nonconstant source variable")
    return np.asarray(chosen), {"method": "deterministic single-chain successive projections",
        "initial_rule": "largest source pre-scaling column variance; first index on exact ties",
        "projection_space": "source standardized centered spectra", "projection_residual_norms": norms,
        "numerical_rank_stop": len(chosen) < min(maximum, len(x)-1, x.shape[1]),
        "rank_stop_relative_squared_norm": 1e-12, "full_multistart_spa_reproduction": False}


@dataclass
class FittedImageFeatureModel:
    candidate: dict
    preprocessor: ScatterPreprocessor
    scaler: StandardScaler
    selected_indices: np.ndarray
    reducer: object
    estimator: object
    classes_: np.ndarray
    metadata: dict

    def transform(self, X):
        z = self.scaler.transform(self.preprocessor.transform(X))[:, self.selected_indices]
        return self.reducer.transform(z) if self.reducer is not None else z

    def predict_scores(self, X):
        scores = np.asarray(self.estimator.decision_function(self.transform(X)), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        if scores.shape != (len(X), len(self.classes_)) or not np.isfinite(scores).all():
            raise ValueError("Invalid source class-aligned scores")
        return scores

    def predict(self, X):
        return self.classes_[self.predict_scores(X).argmax(1)]


def fit_candidate(candidate, X, y, axis, *, seed=42, cache=None):
    """No evaluation arrays or labels enter this source-only fitting API."""
    candidate = dict(candidate); x = matrix(X); labels = np.asarray(y)
    classes = np.unique(labels); axis = np.asarray(axis, dtype=float)
    if labels.shape != (len(x),) or len(classes) < 2 or axis.shape != (x.shape[1],) or not np.all(np.diff(axis)>0):
        raise ValueError("Source labels or frequency grid are invalid")
    if candidate not in candidate_grid():
        raise ValueError("Unknown frozen feature candidate")
    cache = {} if cache is None else cache
    data_hash = joblib.hash((x, labels, axis, seed))
    if cache.get("source_hash", data_hash) != data_hash:
        raise ValueError("Feature cache cannot cross source datasets or seeds")
    cache["source_hash"] = data_hash
    mode = candidate["preprocessing"]
    if mode not in cache:
        prep = ScatterPreprocessor(mode).fit(x); before = prep.transform(x)
        scaler = StandardScaler().fit(before); z = scaler.transform(before)
        cache[mode] = {"preprocessor": prep, "scaler": scaler, "X": z,
                       "initial_variance": before.var(0), "selectors": {}, "reducers": {}}
    stored = cache[mode]; z = stored["X"]; selectors = stored["selectors"]
    family = candidate["family"]; selected = np.arange(x.shape[1]); details = {"method": "all exported candidate-band variables"}
    if family == "cars_svm":
        if "cars" not in selectors:
            selectors["cars"] = cars_path(z, labels, seed)
        selected, details = selectors["cars"]
    elif family == "mcuve_svm":
        if "mcuve" not in selectors:
            selector = MCUVESelector(n_runs=30, n_components=3, n_features_to_select=100,
                sample_fraction=.8, task="classification", random_state=seed).fit(z, labels)
            selectors["mcuve"] = selector
        selector = selectors["mcuve"]
        selected = np.sort(selector.ranking_[:candidate["features"]])
        details = {"method": "MC-UVE absolute onehot coefficient stability topK", "runs": 30,
            "components": 3, "stability_importance": selector.importances_.tolist(),
            "exact_archived_matlab_reproduction": False}
    elif family == "noise_uve_svm":
        if "noise_uve" not in selectors:
            selectors["noise_uve"] = noise_uve_statistics(z, labels, seed)
        selected, details = noise_uve_selection(*selectors["noise_uve"], candidate["noise_quantile"])
    elif family == "spa_svm":
        if "spa" not in selectors:
            selectors["spa"] = spa_chain(z, maximum=50, initial_scores=stored["initial_variance"])
        chain, details = selectors["spa"]
        selected = np.sort(chain[:candidate["features"]])
    reducer = None; values = z[:, selected]
    if family == "pca_lda":
        count = min(candidate["components"], len(x)-1, len(selected))
        if count not in stored["reducers"]:
            stored["reducers"][count] = PCA(n_components=count, svd_solver="randomized", random_state=seed).fit(values)
        reducer = stored["reducers"][count]; values = reducer.transform(values)
    if family.endswith("svm"):
        estimator = SVC(C=candidate["C"], gamma="scale", class_weight="balanced", break_ties=True,
                        decision_function_shape="ovr", random_state=seed)
    else:
        priors = np.full(len(classes), 1/len(classes)) if candidate["priors"] == "uniform" else None
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(values, labels)
    if not np.array_equal(estimator.classes_, classes):
        raise AssertionError("Source fitted class ordering changed")
    metadata = {"candidate": candidate, "seed": seed, "fit_scope": "source_only", "source_samples": len(x),
        "source_class_counts": {str(c): int(np.sum(labels == c)) for c in classes}, "class_order": classes.tolist(),
        "selected_indices": selected.tolist(), "selected_frequencies": axis[selected].tolist(),
        "selected_features": len(selected), "classifier_features": values.shape[1], "selector": details,
        "msc_reference": stored["preprocessor"].reference_.tolist() if mode == "msc" else None,
        "source_only_scaler_mean": stored["scaler"].mean_.tolist(),
        "source_only_scaler_variance": stored["scaler"].var_.tolist(),
        "lda_low_dimensional_transform_claimed": False, "lda_solver": "lsqr" if not family.endswith("svm") else None,
        "query_statistics_fitted": False, "scores_are_probabilities": False,
        "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in captured]}
    if mode == "msc":
        ref=stored["preprocessor"].reference_;centered=ref-ref.mean()
        slopes=(x-x.mean(1,keepdims=True))@centered/np.dot(centered,centered)
        metadata["msc_source_slope_min"]=float(slopes.min());metadata["msc_source_slope_max"]=float(slopes.max())
        metadata["msc_source_negative_slopes"]=int(np.sum(slopes<0))
    return FittedImageFeatureModel(candidate, stored["preprocessor"], stored["scaler"], selected.copy(),
                                  reducer, estimator, classes.copy(), metadata)
