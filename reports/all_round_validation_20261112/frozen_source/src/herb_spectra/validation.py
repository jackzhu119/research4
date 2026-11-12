"""Group-aware evaluation helpers. These are baselines, not a claimed new method."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.validation import check_array


class RowPreprocessor(TransformerMixin, BaseEstimator):
    """Per-scan SNV; no statistics from other scans are used."""
    def __init__(self, mode="raw"):
        self.mode = mode

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        if self.mode not in {"raw", "snv"}:
            raise ValueError("mode must be raw or snv")
        return self

    def transform(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature grid differs from training")
        if self.mode == "raw":
            return X.copy()
        sd = X.std(axis=1, ddof=1, keepdims=True)
        if np.any(sd <= np.finfo(float).eps) or not np.isfinite(sd).all():
            raise ValueError("SNV is undefined for constant spectra")
        return (X - X.mean(axis=1, keepdims=True)) / sd


def check_groups(groups, y=None):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("One physical-sample group ID is required for every scan")
    if any(g is None or str(g).strip().lower() in {"", "nan", "none"} for g in groups):
        raise ValueError("Missing physical-sample group ID")
    if y is not None:
        if np.asarray(y).ndim != 1 or len(y) != len(groups):
            raise ValueError("One class label is required per scan group ID")
        for group in np.unique(groups):
            if len(np.unique(np.asarray(y)[groups == group])) != 1:
                raise ValueError("One physical sample has conflicting class labels")
    return groups


def group_folds(X, y, groups, task="classification", n_splits=3, seed=42):
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown task")
    groups = check_groups(groups, y if task == "classification" else None)
    if len(groups) != len(y) or len(X) != len(y):
        raise ValueError("X, y and groups must have the same number of scans")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough independent samples for the requested folds")
    if task == "classification":
        if len(np.unique(y)) < 2:
            raise ValueError("Classification requires at least two classes")
        for label in np.unique(y):
            if len(np.unique(groups[np.asarray(y) == label])) < n_splits:
                raise ValueError("Not enough independent samples per class")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(X, y, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Physical sample overlaps training and evaluation")
        if task == "classification":
            if set(np.unique(np.asarray(y)[train])) != set(np.unique(y)):
                raise ValueError("A training fold is missing a class")
            if set(np.unique(np.asarray(y)[test])) != set(np.unique(y)):
                raise ValueError("An evaluation fold is missing a class; reduce folds or collect samples")
    return folds


def make_pipeline(model="svc"):
    estimators = {"svc": SVC(class_weight="balanced"), "svr": SVR(),
                  "pls": PLSRegression(scale=False)}
    if model not in estimators:
        raise ValueError("model must be svc, svr or pls")
    return Pipeline([("preprocess", RowPreprocessor()),
                     ("scale", StandardScaler()), ("model", estimators[model])])


def nested_evaluate(X, y, groups, model="svc", outer_splits=3, inner_splits=3, seed=42):
    """Return out-of-fold predictions; tune only within each outer training fold.

    Return physical-sample metrics as primary results, with scan-level metrics
    as diagnostics. Confidence intervals still require sample/batch resampling.
    """
    if model not in {"svc", "svr", "pls"}:
        raise ValueError("Unknown model")
    X = check_array(X)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("One target per scan is required")
    task = "classification" if model == "svc" else "regression"
    if task == "regression" and not np.isfinite(y.astype(float)).all():
        raise ValueError("Regression requires finite independently measured targets")
    groups = check_groups(groups, y if task == "classification" else None)
    if task == "regression":
        for group in np.unique(groups):
            values = y[groups == group].astype(float)
            if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
                raise ValueError("Repeat scans of one sample have conflicting assay targets")
    outer = group_folds(X, y, groups, task, outer_splits, seed)
    predictions = np.empty(len(y), dtype=y.dtype if task == "classification" else float)
    fold_records = []
    for number, (train, test) in enumerate(outer):
        inner = group_folds(X[train], y[train], groups[train], task, inner_splits, seed)
        grid = {"preprocess__mode": ["raw", "snv"]}
        if model == "pls":
            max_components = min(X.shape[1], min(len(t) - 1 for t, _ in inner))
            grid["model__n_components"] = [n for n in [1, 2, 3, 5] if n <= max_components]
        else:
            grid.update({"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale"]})
        search = GridSearchCV(make_pipeline(model), grid, cv=inner, n_jobs=1,
                              scoring="balanced_accuracy" if task == "classification" else "neg_mean_squared_error",
                              error_score="raise")
        search.fit(X[train], y[train])
        predictions[test] = np.asarray(search.predict(X[test])).reshape(-1)
        fold_records.append({"fold": number, "train_groups": sorted(set(map(str, groups[train]))),
                             "test_groups": sorted(set(map(str, groups[test]))),
                             "best_params": search.best_params_})
    sample_true, sample_predicted = [], []
    for group in np.unique(groups):
        mask = groups == group
        sample_true.append(y[mask][0])
        if task == "classification":
            labels, counts = np.unique(predictions[mask], return_counts=True)
            sample_predicted.append(labels[np.argmax(counts)])
        else:
            sample_predicted.append(float(np.mean(predictions[mask])))
    if task == "classification":
        metrics = {"scan_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
                   "scan_macro_f1": float(f1_score(y, predictions, average="macro")),
                   "sample_balanced_accuracy": float(balanced_accuracy_score(sample_true, sample_predicted)),
                   "sample_macro_f1": float(f1_score(sample_true, sample_predicted, average="macro"))}
    else:
        metrics = {"scan_rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                   "scan_r2": float(r2_score(y, predictions)),
                   "sample_rmse": float(np.sqrt(mean_squared_error(sample_true, sample_predicted))),
                   "sample_r2": float(r2_score(sample_true, sample_predicted))}
    return {"metrics": metrics, "predictions": predictions, "folds": fold_records}


def support_query_split(y, groups, shots_per_class=1, seed=42):
    """Partition a labeled target-domain pool by physical sample, not by scan.

    This models a deliberately balanced labeled calibration pool, not an
    unlabeled deployment stream. It does not train a transfer-learning model.
    """
    if not isinstance(shots_per_class, (int, np.integer)) or shots_per_class < 1:
        raise ValueError("shots_per_class must be a positive integer")
    y = np.asarray(y)
    groups = check_groups(groups, y)
    if y.ndim != 1 or len(y) != len(groups):
        raise ValueError("One class and group per scan is required")
    rng = np.random.default_rng(seed)
    support_groups = []
    for label in np.unique(y):
        candidates = np.unique(groups[y == label])
        if len(candidates) <= shots_per_class:
            raise ValueError("Each class needs unused query samples after calibration")
        support_groups.extend(rng.choice(candidates, shots_per_class, replace=False))
    mask = np.isin(groups, support_groups)
    return np.flatnonzero(mask), np.flatnonzero(~mask)
