"""Source-only nested MC-UVE20 control for the matched episodic experiment.

The paper's historical 93.4444% used a different 240-row outer training role.
This module refits that method family within E03's 190-row Source and never
imports a global ranking or accepts the outer Query for model selection.
Inputs have already undergone per-spectrum (and per-modality) SNV.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from herb_spectra.selection import MCUVESelector
from herb_spectra.staged_nested_search import gamma_spec


K_VALUES = (31, 62, 123, 184, 245)
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMAS = ('scale', 'scale_x0.1', 'scale_x10')
CLASSES = np.arange(5)


def _inputs(X, y=None, columns=None):
    X = np.asarray(X, dtype=np.float64)
    if (X.ndim != 2 or not len(X) or X.shape[1] < 2
            or not np.isfinite(X).all()
            or (columns is not None and X.shape[1] != columns)):
        raise ValueError('A finite nonempty samples-by-frequency matrix is required')
    if y is None:
        return X
    y = np.asarray(y)
    if (y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer)
            or not np.array_equal(np.unique(y), CLASSES)):
        raise ValueError('Source labels must contain all five explicit integer classes 0..4')
    return X, y.astype(np.int64, copy=False)


def _ranking(X_scaled, y):
    selector = MCUVESelector(n_runs=20, n_components=2,
        n_features_to_select=X_scaled.shape[1], sample_fraction=0.8,
        task='classification', random_state=42)
    selector.fit(X_scaled, y)
    return selector


def _estimator(config, X, y):
    gamma, numeric_gamma = gamma_spec(X, config['gamma'])
    estimator = SVC(C=config['C'], gamma=gamma, kernel='rbf',
        class_weight='balanced', break_ties=False, decision_function_shape='ovr',
        random_state=42)
    estimator.fit(X, y)
    if not np.array_equal(estimator.classes_, CLASSES):
        raise ValueError('SVC class-column order changed')
    return estimator, float(numeric_gamma)


@dataclass
class StrongControl:
    scaler: StandardScaler
    selector: MCUVESelector
    selected_indices: np.ndarray
    estimator: SVC
    config: dict
    metadata: dict

    def transform(self, X):
        X = _inputs(X, columns=self.scaler.n_features_in_)
        return self.scaler.transform(X)[:, self.selected_indices]

    def scores(self, X):
        result = self.estimator.decision_function(self.transform(X))
        if result.shape != (len(X), 5) or not np.isfinite(result).all():
            raise ValueError('Invalid explicit-five-class SVC scores')
        return result

    def predict(self, X):
        """Native OVO vote; intentionally distinct from OVR-score argmax."""
        return self.estimator.predict(self.transform(X))


def _fit(config, X, y, *, extra_metadata):
    started_cpu, started_wall = time.process_time(), time.monotonic()
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    selector = _ranking(scaled, y)
    chosen = np.sort(selector.ranking_[:config['k']]).astype(np.int64)
    estimator, numeric_gamma = _estimator(config, scaled[:, chosen], y)
    metadata = {
        'algorithm': 'SNV_MC-UVE20_RBF_SVM_source_selected',
        'selected_config': dict(config),
        'selected_feature_indices': chosen.tolist(),
        'input_feature_count': X.shape[1], 'fit_records': len(X),
        'class_order': CLASSES.tolist(),
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'selector_parameters': {'n_runs': 20, 'n_components': 2,
                                'sample_fraction': 0.8, 'random_state': 42},
        'numeric_gamma': numeric_gamma,
        'prediction_rule': 'native_OVO_vote_break_ties_false; OVR_argmax_available_separately',
        'query_used_for_fit_or_selection': False,
        'fit_cpu_seconds': time.process_time() - started_cpu,
        'fit_wall_seconds': time.monotonic() - started_wall,
        **extra_metadata,
    }
    return StrongControl(scaler, selector, chosen, estimator, dict(config), metadata)


def fit_source_control(X_source_snv, y_source, seed=20261031):
    """Choose K/C/gamma using only two Source-internal folds, then fit Source.

    Each inner-fit computes exactly one MC20 ranking. Prefix reuse changes no
    ranking and avoids fitting an identical selector for every K/C/gamma.
    """
    started_cpu, started_wall = time.process_time(), time.monotonic()
    X, y = _inputs(X_source_snv, y_source)
    if min(np.bincount(y, minlength=5)) < 2:
        raise ValueError('Two Source-internal folds require at least two records per class')
    counts = sorted({min(k, X.shape[1]) for k in K_VALUES})
    configurations = [{'k': k, 'C': C, 'gamma': gamma}
                      for k in counts for C in C_VALUES for gamma in GAMMAS]
    rows = []
    partitions = []
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(seed))
    for fold, (fit, validation) in enumerate(splitter.split(X, y)):
        scaler = StandardScaler().fit(X[fit])
        fit_scaled, validation_scaled = scaler.transform(X[fit]), scaler.transform(X[validation])
        selector = _ranking(fit_scaled, y[fit])
        partitions.append({'inner_fold': fold, 'fit_indices': fit.tolist(),
                           'validation_indices': validation.tolist(),
                           'scaler_mean': scaler.mean_.tolist(),
                           'ranking': selector.ranking_.tolist()})
        selected = {k: np.sort(selector.ranking_[:k]) for k in counts}
        for config in configurations:
            index = selected[config['k']]
            estimator, numeric_gamma = _estimator(config, fit_scaled[:, index], y[fit])
            predicted = estimator.predict(validation_scaled[:, index])
            rows.append({'inner_fold': fold, 'config': dict(config),
                'accuracy': float(accuracy_score(y[validation], predicted)),
                'balanced_accuracy': float(balanced_accuracy_score(y[validation], predicted)),
                'macro_f1': float(f1_score(y[validation], predicted, labels=CLASSES,
                                          average='macro', zero_division=0)),
                'numeric_gamma': numeric_gamma})
    summarized = []
    for config in configurations:
        group = [row for row in rows if row['config'] == config]
        summarized.append({'config': dict(config),
            **{metric: float(np.mean([row[metric] for row in group]))
               for metric in ('macro_f1', 'balanced_accuracy', 'accuracy')}})
    def ordering(row):
        config = row['config']
        return (-row['macro_f1'], -row['balanced_accuracy'], -row['accuracy'],
                config['k'], config['C'], GAMMAS.index(config['gamma']))
    selected = min(summarized, key=ordering)
    result = _fit(selected['config'], X, y, extra_metadata={
        'fit_scope': 'current_E03_Source_only', 'selection_scope': 'Source_internal_2fold_only',
        'inner_split_seed': int(seed), 'inner_folds': partitions,
        'inner_results': rows, 'inner_config_means': summarized,
        'selection_tie_order': 'macroF1,balanced_accuracy,accuracy,lower_K,lower_C,declared_gamma_order',
        'historical_score_reused': False,
    })
    result.metadata['total_cpu_seconds'] = time.process_time() - started_cpu
    result.metadata['total_wall_seconds'] = time.monotonic() - started_wall
    return result


def refit_control(control, X_combined_snv, y_combined):
    """Refit the Source-chosen pipeline with labeled support, without reselecting."""
    X, y = _inputs(X_combined_snv, y_combined,
                   columns=control.scaler.n_features_in_)
    return _fit(dict(control.config), X, y, extra_metadata={
        'fit_scope': 'Source_plus_labeled_Support',
        'selection_scope': 'frozen_Source_choice; no new hyperparameter search',
        'parent_source_config': dict(control.config),
        'parent_source_fit_records': control.metadata['fit_records'],
    })
