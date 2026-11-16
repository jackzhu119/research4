"""Training-only repeated variable-selection aggregation.

This is an engineering research baseline around this repository's independent
CARS-style/MC-UVE selectors, not an exact reproduction of the archived code or
a claim of a new statistical stability-selection method. Frequencies describe
the specified resampling/algorithm budget; they are not p-values, controlled
false-discovery rates, or evidence of robustness to a new batch/instrument.

Refit this complete selector within every training partition used for model
selection. With ``groups`` supplied, repeat scans are averaged by physical
sample before *any* subsampling or base selection. Without groups, rows are
assumed independent. Indices always address the original feature grid; no
physical wavenumber mapping or neighborhood smoothing is inferred.
"""
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .selection import (CARSStyleSelector, MCUVESelector, _coefficients,
                        _positive_int, _prepare, _subsample)


class StableVariableSelector(SelectorMixin, BaseEstimator):
    """Aggregate repeated training-only selections, then retain a fixed count.

    Samples are drawn without replacement. Classification draws independently
    within each class using the same per-row priorities, which preserves label
    permutation invariance. Frequency is the primary rank; mean normalized
    standardized OVR-PLS coefficient norm breaks frequency ties. Original index
    only resolves exact ties in both quantities.

    ``inner_runs=None`` means 20 MC-UVE or 12 CARS-style iterations. The final
    count is capped at the feature count. For CARS, individual resamples may
    choose different subset sizes; ``base_selected_feature_counts_`` records
    that fact. The final count remains fixed for matched-dimensional comparisons.

    ``resample_unit_indices_`` indexes the aggregated training units, not input
    scans when groups are supplied. Large Jaccard values alone do not establish
    useful selection: retaining every variable trivially gives Jaccard=1.
    """
    def __init__(self, base_selector="mcuve", n_features_to_select=20,
                 n_resamples=10, sample_fraction=0.8, inner_runs=None,
                 n_components=2, cv=3, task="regression", random_state=42):
        self.base_selector = base_selector
        self.n_features_to_select = n_features_to_select
        self.n_resamples = n_resamples
        self.sample_fraction = sample_fraction
        self.inner_runs = inner_runs
        self.n_components = n_components
        self.cv = cv
        self.task = task
        self.random_state = random_state

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def fit(self, X, y, groups=None):
        if self.base_selector not in {"mcuve", "cars"}:
            raise ValueError("base_selector must be mcuve or cars")
        repeats = _positive_int(self.n_resamples, "n_resamples", 2)
        count = _positive_int(self.n_features_to_select, "n_features_to_select")
        components = _positive_int(self.n_components, "n_components")
        cv = _positive_int(self.cv, "cv", 2)
        if (not isinstance(self.sample_fraction, Real) or isinstance(self.sample_fraction, bool)
                or not np.isfinite(self.sample_fraction) or not 0 < self.sample_fraction <= 1):
            raise ValueError("sample_fraction must be finite and in (0, 1]")
        if self.random_state is not None and (not isinstance(self.random_state, Integral)
                                              or isinstance(self.random_state, bool)
                                              or self.random_state < 0):
            raise ValueError("random_state must be a nonnegative integer or None")
        runs = (20 if self.base_selector == "mcuve" else 12) if self.inner_runs is None else self.inner_runs
        runs = _positive_int(runs, "inner_runs", 2)
        X_fit, Y, labels, self.classes_, self.sample_unit_ = _prepare(X, y, self.task, groups)
        target = labels if self.task == "classification" else Y[:, 0]
        self.training_unit_count_ = len(X_fit)
        self.n_training_sample_units_ = len(X_fit)
        self.training_unit_ids_ = (np.unique(np.asarray(groups)) if groups is not None
                                   else np.arange(len(X_fit)))
        self.n_features_in_ = X_fit.shape[1]
        count = min(count, self.n_features_in_)
        if labels is None:
            sample_size = max(3, int(np.ceil(len(X_fit) * self.sample_fraction)))
            if sample_size < 4:
                raise ValueError("sample_fraction leaves fewer than four independent training units")
        else:
            counts = np.bincount(labels)
            selected_counts = np.maximum(1, np.ceil(counts * self.sample_fraction).astype(int))
            if selected_counts.sum() < 4:
                raise ValueError("sample_fraction leaves fewer than four independent training units")
            if self.base_selector == "cars" and selected_counts.min() < cv:
                raise ValueError("Each classification resample needs at least cv independent units per class")
        if self.base_selector == "cars" and labels is None and sample_size < cv:
            raise ValueError("CARS resamples have fewer independent units than cv")

        rng = np.random.default_rng(self.random_state)
        supports, importance, selections, seeds = [], [], [], []
        for _ in range(repeats):
            selected = _subsample(labels, len(X_fit), self.sample_fraction, rng)
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            if self.base_selector == "mcuve":
                base = MCUVESelector(n_runs=runs, n_components=components,
                                     n_features_to_select=count, sample_fraction=0.8,
                                     task=self.task, random_state=seed)
            else:
                base = CARSStyleSelector(n_runs=runs, n_components=components,
                                         min_features=min(2, self.n_features_in_),
                                         sample_fraction=0.8, cv=cv,
                                         task=self.task, random_state=seed)
            base.fit(X_fit[selected], target[selected])
            supports.append(base.get_support())
            coefficients = _coefficients(X_fit[selected], Y[selected], components)
            magnitude = np.linalg.norm(coefficients, axis=0)
            total = magnitude.sum()
            importance.append(magnitude / total if total > 0 else np.zeros_like(magnitude))
            selections.append(selected)
            seeds.append(seed)

        self.resample_supports_ = np.asarray(supports, dtype=bool)
        self.frequency_ = self.resample_supports_.mean(axis=0)
        self.mean_coefficient_importance_ = np.mean(importance, axis=0)
        self.ranking_ = np.lexsort((np.arange(self.n_features_in_),
                                   -self.mean_coefficient_importance_, -self.frequency_))
        self.selected_indices_ = np.sort(self.ranking_[:count])
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_indices_] = True
        self.resample_unit_indices_ = selections
        self.resample_seeds_ = np.asarray(seeds)
        self.base_selected_feature_counts_ = self.resample_supports_.sum(axis=1)
        self.inner_runs_ = runs
        self.base_method_ = self.base_selector
        pairwise = []
        for first in range(repeats - 1):
            for second in range(first + 1, repeats):
                a, b = self.resample_supports_[first], self.resample_supports_[second]
                pairwise.append(np.count_nonzero(a & b) / np.count_nonzero(a | b))
        self.pairwise_jaccard_ = np.asarray(pairwise)
        self.mean_pairwise_jaccard_ = float(self.pairwise_jaccard_.mean())
        return self
