"""Source-frozen, small-support score corrections for development experiments.

No method receives query labels or estimates a statistic from the query batch.
These are engineering combinations of ridge, prototypes and kernel residuals,
not claimed original algorithms or verified probability calibration methods.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.decomposition import PCA

from .calibration import AnchorScoreCalibrator


FAMILIES = ("identity", "legacy_anchor", "damped_anchor", "mistake_bias",
            "mistake_ridge", "mistake_kernel", "prototype_shrinkage")


def _matrix(values, name, columns=None):
    result = np.asarray(values, dtype=float)
    if (result.ndim != 2 or not len(result) or not result.shape[1]
            or not np.isfinite(result).all()
            or (columns is not None and result.shape[1] != columns)):
        raise ValueError(f"{name} needs a nonempty finite matrix with the expected columns")
    return result


def _labels(values, classes, n, name, require_all=True):
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != n:
        raise ValueError(f"{name} needs one label per row")
    if any(v is None or str(v).strip().lower() in {"", "none", "nan", "null"} for v in result):
        raise ValueError(f"{name} contains a missing label")
    known = set(classes.tolist())
    if not set(result.tolist()).issubset(known):
        raise ValueError(f"{name} contains an unknown class")
    if require_all and set(result.tolist()) != known:
        raise ValueError(f"{name} must cover every source class")
    return result


def _number(value, name, *, positive=False, unit=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite numeric")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if (not np.isfinite(result) or (positive and result <= 0)
            or (unit and not 0 <= result <= 1)):
        raise ValueError(f"Invalid {name}")
    return result


def _readonly(values):
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _squared_distances(left, right):
    result = ((left * left).sum(axis=1)[:, None]
              + (right * right).sum(axis=1)[None, :] - 2 * left @ right.T)
    return np.maximum(result, 0.)


def _softmax(scores):
    values = np.exp(scores - scores.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def _gate(scores, threshold):
    if threshold is None:
        return np.ones(len(scores), dtype=bool)
    ordered = np.partition(scores, -2, axis=1)
    return ordered[:, -1] - ordered[:, -2] <= threshold


@dataclass(frozen=True)
class SourceContext:
    classes: np.ndarray
    n_features: int
    projection_mean: np.ndarray
    projection_components: np.ndarray
    source_centroids: np.ndarray
    bandwidth_squared: float
    source_count: int

    def project(self, features):
        features = _matrix(features, "features", self.n_features)
        return (features - self.projection_mean) @ self.projection_components.T


def build_context(source_features, source_scores, source_y, classes, seed=42):
    """Fit PCA and a distance scale exclusively on frozen source features."""
    features = _matrix(source_features, "source_features")
    classes = np.asarray(classes)
    if (classes.ndim != 1 or len(classes) < 2 or len(set(classes.tolist())) != len(classes)
            or any(v is None or str(v).strip().lower() in {"", "none", "nan", "null"} for v in classes)):
        raise ValueError("Provide a unique nonmissing source class order")
    scores = _matrix(source_scores, "source_scores", len(classes))
    if len(scores) != len(features) or len(features) < 2:
        raise ValueError("Source feature and score rows must agree and contain at least two specimens")
    labels = _labels(source_y, classes, len(features), "source_y")
    dimensions = min(20, len(features) - 1, features.shape[1])
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed)
    pca.fit(features)
    projected = pca.transform(features)
    centroids = np.asarray([projected[labels == label].mean(axis=0) for label in classes])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(features), size=min(256, len(features)), replace=False)
    pairwise = _squared_distances(projected[chosen], projected[chosen])
    positive = pairwise[np.triu_indices(len(chosen), 1)]
    positive = positive[positive > np.finfo(float).eps]
    bandwidth = float(np.median(positive)) if len(positive) else 1.
    return SourceContext(_readonly(classes), features.shape[1], _readonly(pca.mean_),
                         _readonly(pca.components_), _readonly(centroids), bandwidth, len(features))


def candidate_grid():
    """Return the stable predeclared grid; null gate means all margins."""
    result = []

    def add(family, **parameters):
        index = len(result)
        result.append({"id": f"c{index:03d}_{family}", "family": family,
                       "complexity_order": index, **parameters})

    add("identity")
    add("legacy_anchor", regularization=1.)
    for penalty, blend in product([1., 10., 100., 1000.], [.05, .1, .25, .5, 1.]):
        add("damped_anchor", regularization=penalty, blend=blend)
    for strength, gate in product([.1, .25, .5, 1.], [1., None]):
        add("mistake_bias", strength=strength, gate_margin=gate)
    for penalty, blend, gate in product([1., 10., 100., 1000.], [.1, .25, .5, 1.], [1., None]):
        add("mistake_ridge", regularization=penalty, blend=blend, gate_margin=gate)
    for penalty, blend, bandwidth in product([.1, 1., 10.], [.1, .25, .5, 1.], [.5, 1., 2.]):
        add("mistake_kernel", regularization=penalty, blend=blend, bandwidth_multiplier=bandwidth)
    for pseudocount, blend, gate in product([1., 5., 20.], [.1, .25, .5], [1., None]):
        add("prototype_shrinkage", pseudocount=pseudocount, blend=blend, gate_margin=gate)
    return result


def _checked_candidate(candidate):
    result = dict(candidate)
    family = result.get("family")
    if family not in FAMILIES:
        raise ValueError("Unknown calibration family")
    if family in {"legacy_anchor", "damped_anchor", "mistake_ridge", "mistake_kernel"}:
        result["regularization"] = _number(result.get("regularization"), "regularization", positive=True)
    if family in {"damped_anchor", "mistake_ridge", "mistake_kernel", "prototype_shrinkage"}:
        result["blend"] = _number(result.get("blend"), "blend", unit=True)
    if family == "mistake_bias":
        result["strength"] = _number(result.get("strength"), "strength", unit=True)
    if family == "mistake_kernel":
        result["bandwidth_multiplier"] = _number(result.get("bandwidth_multiplier"), "bandwidth_multiplier", positive=True)
    if family == "prototype_shrinkage":
        result["pseudocount"] = _number(result.get("pseudocount"), "pseudocount", positive=True)
    if family in {"mistake_bias", "mistake_ridge", "prototype_shrinkage"}:
        threshold = result.get("gate_margin")
        if threshold is not None:
            threshold = _number(threshold, "gate_margin")
            if threshold < 0:
                raise ValueError("gate_margin must be nonnegative or null")
        result["gate_margin"] = threshold
    return result


def mistake_residual(scores, labels, classes, margin=.1):
    """Leave already-correct rows untouched; shift true and winner on mistakes."""
    scores = _matrix(scores, "support_scores", len(classes))
    labels = _labels(labels, np.asarray(classes), len(scores), "support_y")
    margin = _number(margin, "mistake_margin", positive=True)
    vocabulary = {label: i for i, label in enumerate(classes)}
    encoded = np.asarray([vocabulary[label] for label in labels])
    winner = np.argmax(scores, axis=1)
    wrong = np.flatnonzero(winner != encoded)
    residual = np.zeros_like(scores)
    amount = (scores[wrong, winner[wrong]] - scores[wrong, encoded[wrong]] + margin) / 2
    residual[wrong, encoded[wrong]] = amount
    residual[wrong, winner[wrong]] = -amount
    return residual


@dataclass
class FittedCalibration:
    context: SourceContext
    candidate: dict
    metadata: dict
    weights: np.ndarray | None = None
    intercept: np.ndarray | None = None
    support_projection: np.ndarray | None = None
    centroids: np.ndarray | None = None
    identity: bool = False

    def predict_scores(self, query_features, query_scores):
        features = _matrix(query_features, "query_features", self.context.n_features)
        scores = _matrix(query_scores, "query_scores", len(self.context.classes))
        if len(features) != len(scores):
            raise ValueError("Query features and scores need the same rows")
        if self.identity:
            return scores.copy()
        family = self.candidate["family"]
        gate = _gate(scores, self.candidate.get("gate_margin"))
        if family == "prototype_shrinkage":
            projected = self.context.project(features)
            distances = _squared_distances(projected, self.centroids)
            similarities = _softmax(-distances / (2 * self.context.bandwidth_squared))
            source = _softmax(scores)
            result = source.copy()
            blend = self.candidate["blend"]
            result[gate] = (1 - blend) * source[gate] + blend * similarities[gate]
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite prototype scores")
            return result
        if family == "mistake_kernel":
            projected = self.context.project(features)
            scale = self.context.bandwidth_squared * self.candidate["bandwidth_multiplier"] ** 2
            kernel = np.exp(-_squared_distances(projected, self.support_projection) / (2 * scale))
            correction = kernel @ self.weights
        elif family == "mistake_bias":
            correction = np.broadcast_to(self.intercept, scores.shape)
        else:
            correction = scores @ self.weights + self.intercept
        result = scores.copy()
        result[gate] += correction[gate]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite calibrated scores")
        return result


def fit_candidate(context, candidate, support_features=None, support_scores=None, support_y=None):
    """Fit a correction with no query input; absent support gives a matched control.

    For prototype fusion, zero support retains source centroids and the same
    blend/gate. Every other zero-support control is the original source map.
    """
    candidate = _checked_candidate(candidate)
    family = candidate["family"]
    absent = (support_features is None, support_scores is None, support_y is None)
    if any(absent) and not all(absent):
        raise ValueError("Supply all support arrays, or none for a zero-support control")
    n_support = 0
    if not all(absent):
        features = _matrix(support_features, "support_features", context.n_features)
        scores = _matrix(support_scores, "support_scores", len(context.classes))
        if len(features) != len(scores):
            raise ValueError("Support feature and score rows differ")
        labels = _labels(support_y, context.classes, len(features), "support_y")
        n_support = len(features)
    adapter = FittedCalibration(context, candidate, {
        "candidate_id": candidate.get("id", family), "family": family,
        "support_count": n_support,
        "uses_support_labels": bool(n_support and family != "identity"),
        "zero_support_control": not bool(n_support),
        "scores_are_calibrated_probabilities": False,
        "context_fit_scope": "source_only", "query_statistics_used": False})
    if family == "identity" or (not n_support and family != "prototype_shrinkage"):
        adapter.identity = True
        return adapter
    if family == "prototype_shrinkage":
        centroids = context.source_centroids.copy()
        if n_support:
            projected = context.project(features)
            pseudo = candidate["pseudocount"]
            for i, label in enumerate(context.classes):
                group = projected[labels == label]
                centroids[i] = (pseudo * centroids[i] + group.sum(axis=0)) / (pseudo + len(group))
        adapter.centroids = _readonly(centroids)
        return adapter
    if family in {"legacy_anchor", "damped_anchor"}:
        legacy = AnchorScoreCalibrator(context.classes, regularization=candidate["regularization"])
        legacy.fit(scores, labels)
        blend = candidate.get("blend", 1.)
        adapter.weights = _readonly(blend * (legacy.coef_ - np.eye(len(context.classes))))
        adapter.intercept = _readonly(blend * legacy.intercept_)
        return adapter
    residual = mistake_residual(scores, labels, context.classes)
    adapter.metadata["support_mistakes"] = int(np.count_nonzero(np.any(residual != 0, axis=1)))
    if not np.any(residual):
        adapter.identity = True
        return adapter
    if family == "mistake_bias":
        adapter.intercept = _readonly(candidate["strength"] * residual.mean(axis=0))
        return adapter
    if family == "mistake_ridge":
        design = np.column_stack([scores, np.ones(len(scores))])
        delta = np.linalg.solve(design.T @ design + candidate["regularization"] * np.eye(len(context.classes) + 1),
                                design.T @ residual)
        adapter.weights = _readonly(candidate["blend"] * delta[:-1])
        adapter.intercept = _readonly(candidate["blend"] * delta[-1])
        return adapter
    projected = context.project(features)
    scale = context.bandwidth_squared * candidate["bandwidth_multiplier"] ** 2
    kernel = np.exp(-_squared_distances(projected, projected) / (2 * scale))
    adapter.weights = _readonly(candidate["blend"] * np.linalg.solve(
        kernel + candidate["regularization"] * np.eye(len(scores)), residual))
    adapter.support_projection = _readonly(projected)
    return adapter
