"""Independent, bounded PLS-based variable-selection implementations.

These utilities are inspired by the archived CARS/MC-UVE/IRIV methods. They do
not copy the archived implementations and are not exact reproductions of them.
CARS: Li et al., Analytica Chimica Acta 648 (2009), 77-84. IRIV: Yun et al.,
Analytica Chimica Acta (2014), doi:10.1016/j.aca.2013.11.032.

Selection must be refitted inside each outer evaluation training partition.
Any internal selection MSE is a search diagnostic, not an independent estimate
of predictive performance. No query or test arrays are accepted by fit().
"""
from numbers import Integral
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


def _positive_int(value, name, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prepare(X, y, task, groups):
    X = check_array(X, dtype=float, ensure_min_samples=4)
    y = np.asarray(y)
    if task not in {"regression", "classification"}:
        raise ValueError("task must be explicitly regression or classification")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("One target per training scan is required")
    if task == "regression":
        y = y.astype(float)
        if not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite measured values")
    elif any(v is None or str(v).strip().lower() in {"", "nan", "none"} for v in y):
        raise ValueError("Missing classification label")
    if groups is None:
        grouped = X, y
        sample_unit = "rows_assumed_independent"
    else:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(X):
            raise ValueError("One physical-sample group ID per training scan is required")
        if any(v is None or str(v).strip().lower() in {"", "nan", "none"} for v in groups):
            raise ValueError("Missing group ID")
        group_X, group_y = [], []
        for group in np.unique(groups):
            indices = groups == group
            targets = y[indices]
            if task == "classification":
                consistent = len(np.unique(targets)) == 1
            else:
                consistent = np.allclose(targets, targets[0], rtol=1e-10, atol=1e-12)
            if not consistent:
                raise ValueError("Repeat scans of one sample have conflicting targets")
            group_X.append(X[indices].mean(axis=0))
            group_y.append(targets[0])
        grouped = np.asarray(group_X), np.asarray(group_y)
        sample_unit = "physical_sample_means"
    X_fit, target = grouped
    if len(X_fit) < 4:
        raise ValueError("At least four independent training sample units are required")
    if task == "classification":
        classes, encoded = np.unique(target, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Classification selection requires at least two classes")
        Y = np.eye(len(classes))[encoded]
        labels = encoded
    else:
        if np.std(target) <= np.finfo(float).eps:
            raise ValueError("Regression target has no variation")
        classes = None
        labels = None
        Y = target[:, None]
    return X_fit, Y, labels, classes, sample_unit


def _subsample(labels, n, fraction, rng):
    if not 0 < fraction <= 1:
        raise ValueError("sample_fraction must be in (0,1]")
    if labels is None:
        size = min(n, max(3, int(np.ceil(n * fraction))))
        return np.sort(rng.choice(n, size=size, replace=False))
    selected = []
    priorities = rng.random(n)
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        size = min(len(candidates), max(1, int(np.ceil(len(candidates) * fraction))))
        selected.extend(candidates[np.argsort(priorities[candidates], kind="stable")[:size]])
    return np.sort(np.asarray(selected, dtype=int))


def _coefficients(X, Y, components):
    """Standardized OVR PLS1 coefficients, so label permutations preserve norms."""
    scale = X.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1
    Z = (X - X.mean(axis=0)) / scale
    coefficients = []
    n_components = min(components, X.shape[1], len(X) - 1)
    for column in Y.T:
        target_sd = column.std()
        if target_sd <= np.finfo(float).eps:
            coefficients.append(np.zeros(X.shape[1]))
            continue
        target = (column - column.mean()) / target_sd
        model = PLSRegression(n_components=n_components, scale=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y residual is constant.*")
            model.fit(Z, target)
        coefficients.append(np.asarray(model.coef_).reshape(-1))
    result = np.asarray(coefficients)
    if not np.isfinite(result).all():
        raise ValueError("PLS coefficients became non-finite; inspect rank/constant features")
    return result


def _folds(X, Y, labels, cv, seed):
    cv = _positive_int(cv, "cv", 2)
    if len(X) < cv:
        raise ValueError("Not enough independent sample units for selection CV")
    if labels is None:
        splitter = KFold(cv, shuffle=True, random_state=seed)
        return list(splitter.split(X))
    if min(np.bincount(labels)) < cv:
        raise ValueError("Not enough independent samples per class for selection CV")
    return list(StratifiedKFold(cv, shuffle=True, random_state=seed).split(X, labels))


def _selection_mse(X, Y, selected, folds, components):
    if len(selected) == 0:
        return float("inf")
    errors = []
    for train, valid in folds:
        n_components = min(components, len(selected), len(train) - 1)
        model = make_pipeline(StandardScaler(), PLSRegression(n_components=n_components, scale=False))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y residual is constant.*")
            model.fit(X[train][:, selected], Y[train])
        predicted = np.asarray(model.predict(X[valid][:, selected])).reshape(len(valid), -1)
        errors.append(float(np.mean((predicted - Y[valid]) ** 2)))
    return float(np.mean(errors))


class _SelectorBase(SelectorMixin, BaseEstimator):
    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def _finish(self, X, selected):
        self.n_features_in_ = X.shape[1]
        self.selected_indices_ = np.sort(np.asarray(selected, dtype=int))
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selected_indices_] = True
        return self


class CARSStyleSelector(_SelectorBase):
    """CARS-style Monte Carlo PLS coefficient pruning and internal subset search.

    Choices: absolute standardized coefficient norms, exponential feature-count
    reduction, optional weighted resampling, fixed PLS component count, and raw
    unscaled selection MSE. This is neither archived CARS version verbatim.
    Classification uses OVR one-hot columns; integer class IDs are never used as
    ordered concentration values when task='classification'.
    """
    def __init__(self, n_runs=20, n_components=3, min_features=2,
                 sample_fraction=0.8, cv=3, reweight=True, task="regression", random_state=42):
        self.n_runs = n_runs
        self.n_components = n_components
        self.min_features = min_features
        self.sample_fraction = sample_fraction
        self.cv = cv
        self.reweight = reweight
        self.task = task
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        runs = _positive_int(self.n_runs, "n_runs", 2)
        components = _positive_int(self.n_components, "n_components")
        X_fit, Y, labels, self.classes_, self.sample_unit_ = _prepare(X, y, self.task, groups)
        self.n_training_sample_units_ = len(X_fit)
        minimum = min(_positive_int(self.min_features, "min_features"), X_fit.shape[1])
        folds = _folds(X_fit, Y, labels, self.cv, self.random_state)
        rng = np.random.default_rng(self.random_state)
        active = np.arange(X_fit.shape[1])
        self.path_ = []
        self.selection_mse_path_ = []
        for iteration in range(runs):
            sample = _subsample(labels, len(X_fit), self.sample_fraction, rng)
            coefficients = _coefficients(X_fit[sample][:, active], Y[sample], components)
            importance = np.linalg.norm(coefficients, axis=0)
            wanted = max(minimum, int(round(X_fit.shape[1] * (minimum / X_fit.shape[1]) ** (iteration / (runs - 1)))))
            wanted = min(wanted, len(active))
            rank = np.argsort(-importance, kind="stable")
            retained = rank[:wanted]
            if self.reweight and wanted > minimum and importance[retained].sum() > 0:
                weights = importance[retained] / importance[retained].sum()
                chosen = np.unique(rng.choice(retained, size=len(active), replace=True, p=weights))
                if len(chosen) < minimum:
                    chosen = np.unique(np.r_[chosen, rank[:minimum]])
                retained = chosen
            active = np.sort(active[retained])
            self.path_.append(active.copy())
            self.selection_mse_path_.append(_selection_mse(X_fit, Y, active, folds, components))
        best = min(range(runs), key=lambda i: (self.selection_mse_path_[i], len(self.path_[i])))
        self.best_iteration_ = best
        self.selection_mse_ = float(self.selection_mse_path_[best])
        return self._finish(X_fit, self.path_[best])


class MCUVESelector(_SelectorBase):
    """Absolute MC-UVE coefficient stability, with explicit retained feature count.

    Classification aggregates per-class absolute stability by Euclidean norm.
    Small coefficient standard deviations are floored; zero coefficients have
    zero stability, not NaN. n_features_to_select is capped at available columns.
    """
    def __init__(self, n_runs=50, n_components=3, n_features_to_select=20,
                 sample_fraction=0.8, task="regression", random_state=42):
        self.n_runs = n_runs
        self.n_components = n_components
        self.n_features_to_select = n_features_to_select
        self.sample_fraction = sample_fraction
        self.task = task
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        runs = _positive_int(self.n_runs, "n_runs", 2)
        components = _positive_int(self.n_components, "n_components")
        X_fit, Y, labels, self.classes_, self.sample_unit_ = _prepare(X, y, self.task, groups)
        self.n_training_sample_units_ = len(X_fit)
        n_select = min(_positive_int(self.n_features_to_select, "n_features_to_select"), X_fit.shape[1])
        rng = np.random.default_rng(self.random_state)
        self.coefficients_ = np.asarray([_coefficients(X_fit[sample], Y[sample], components)
                                        for sample in (_subsample(labels, len(X_fit), self.sample_fraction, rng)
                                                       for _ in range(runs))])
        mean = self.coefficients_.mean(axis=0)
        sd = self.coefficients_.std(axis=0, ddof=1)
        floor = max(1.0, float(sd.max())) * 1e-12
        self.signed_stability_ = mean / np.maximum(sd, floor)
        self.importances_ = np.linalg.norm(np.abs(self.signed_stability_), axis=0)
        self.ranking_ = np.argsort(-self.importances_, kind="stable")
        return self._finish(X_fit, self.ranking_[:n_select])


class BoundedIRIVInspiredSelector(_SelectorBase):
    """Bounded paired inclusion/exclusion search followed by backward elimination.

    This independent IRIV-inspired replacement deliberately differs from Yun et
    al.'s IRIV: no Mann-Whitney category claims, no balanced-column rejection
    sampler, optional MC-UVE prescreening, a hard round limit, and immediate
    backward elimination at <=4 variables. Scores are internal search MSEs.
    """
    def __init__(self, n_components=3, cv=3, n_masks=8, max_rounds=3,
                 max_initial_features=15, task="regression", random_state=42):
        self.n_components = n_components
        self.cv = cv
        self.n_masks = n_masks
        self.max_rounds = max_rounds
        self.max_initial_features = max_initial_features
        self.task = task
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        components = _positive_int(self.n_components, "n_components")
        masks = _positive_int(self.n_masks, "n_masks", 2)
        max_rounds = _positive_int(self.max_rounds, "max_rounds")
        initial_limit = _positive_int(self.max_initial_features, "max_initial_features")
        X_fit, Y, labels, self.classes_, self.sample_unit_ = _prepare(X, y, self.task, groups)
        self.n_training_sample_units_ = len(X_fit)
        folds = _folds(X_fit, Y, labels, self.cv, self.random_state)
        rng = np.random.default_rng(self.random_state)
        active = np.arange(X_fit.shape[1])
        if len(active) > initial_limit:
            prescreen = MCUVESelector(n_runs=20, n_components=components, n_features_to_select=initial_limit,
                                      task=self.task, random_state=self.random_state)
            targets = labels if self.task == "classification" else Y.ravel()
            prescreen.fit(X_fit, targets)
            active = prescreen.selected_indices_.copy()
        cache = {}
        def error(indices):
            key = tuple(sorted(map(int, indices)))
            if key not in cache:
                cache[key] = _selection_mse(X_fit, Y, np.asarray(key, dtype=int), folds, components)
            return cache[key]
        self.rounds_completed_ = 0
        self.path_ = [active.copy()]
        for _ in range(max_rounds):
            if len(active) <= 4:
                break
            matrix = rng.random((masks, len(active))) < 0.5
            for row in matrix:
                if not row.any(): row[rng.integers(len(row))] = True
            strengths = np.zeros(len(active))
            for j in range(len(active)):
                differences = []
                for row in matrix:
                    included = row.copy(); excluded = row.copy()
                    included[j] = True; excluded[j] = False
                    # Empty exclusion is not a usable competing model.
                    if not excluded.any(): continue
                    differences.append(error(active[excluded]) - error(active[included]))
                strengths[j] = np.mean(differences) if differences else 0.0
            keep = strengths > 0
            if not keep.any(): keep[np.argmax(strengths)] = True
            self.rounds_completed_ += 1
            if keep.all(): break
            active = active[keep]
            self.path_.append(active.copy())
        while len(active) > 1:
            current = error(active)
            candidates = [np.delete(active, i) for i in range(len(active))]
            scores = [error(candidate) for candidate in candidates]
            best = int(np.argmin(scores))
            if scores[best] > current + 1e-12:
                break
            active = candidates[best]
            self.path_.append(active.copy())
        self.selection_mse_ = float(error(active))
        self.evaluation_count_ = len(cache)
        return self._finish(X_fit, active)
