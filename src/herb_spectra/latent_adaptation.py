"""Few-shot readout adaptation in a source-frozen spectral representation.

These are engineering implementations of anchored ridge, local memory and
shrunken class centres, not claims of new algorithms or calibrated probability.
Every learned statistic comes from source or explicitly labelled support data.
Query rows are only transformed and predicted independently.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FAMILIES = ("identity", "anchored_ridge", "local_memory", "mahalanobis_centroid")
SOURCE_HEAD_RIDGE = .01
DISTANCE_EPSILON = 1e-8


def _matrix(values, name, columns=None):
    array = np.asarray(values, dtype=float)
    if (array.ndim != 2 or not len(array) or not array.shape[1]
            or not np.isfinite(array).all()
            or (columns is not None and array.shape[1] != columns)):
        raise ValueError(f"{name} requires a nonempty finite matrix with the expected columns")
    return array


def _classes(values):
    array = np.asarray(values)
    if (array.ndim != 1 or len(array) < 2 or len(set(array.tolist())) != len(array)
            or any(value is None or str(value).strip().lower() in
                   {"", "nan", "none", "null"} for value in array)):
        raise ValueError("Provide at least two unique nonmissing source classes")
    return array


def _encoded(values, classes, size, name):
    labels = np.asarray(values)
    if (labels.ndim != 1 or len(labels) != size
            or any(value is None or str(value).strip().lower() in
                   {"", "nan", "none", "null"} for value in labels)):
        raise ValueError(f"{name} requires one nonmissing label per row")
    vocabulary = {value: index for index, value in enumerate(classes)}
    if not set(labels.tolist()).issubset(vocabulary):
        raise ValueError(f"{name} contains an unknown class")
    if set(labels.tolist()) != set(classes.tolist()):
        raise ValueError(f"{name} must cover every source class")
    return np.asarray([vocabulary[value] for value in labels], dtype=int)


def _positive(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite numeric")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not np.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def _readonly(values):
    array = np.asarray(values).copy()
    array.setflags(write=False)
    return array


def _augment(projected):
    return np.column_stack([projected, np.ones(len(projected))])


def _squared_distances(left, right):
    return np.maximum((left*left).sum(axis=1)[:, None]
                      + (right*right).sum(axis=1)[None, :] - 2*left@right.T, 0.)


def _softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class LatentContext:
    classes: np.ndarray
    n_features: int
    source_count: int
    projection_mean: np.ndarray
    projection_components: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray
    source_latent: np.ndarray
    source_encoded: np.ndarray
    source_weights: np.ndarray
    source_design: np.ndarray
    source_gram: np.ndarray
    source_head: np.ndarray
    source_centroids: np.ndarray
    source_precision: np.ndarray
    source_score_scale: float
    source_class_counts: tuple

    def project(self, features):
        values = _matrix(features, "features", self.n_features)
        projected = (values - self.projection_mean) @ self.projection_components.T
        result = (projected-self.latent_mean)/self.latent_scale
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen latent projection")
        return result

    def design(self, features):
        return _augment(self.project(features))

    def source_distances(self, features):
        """A pure pointwise distance calculation, suitable for runner memoization."""
        return _squared_distances(self.project(features), self.source_latent)


def build_latent_context(source_features, source_scores, source_y, classes, seed=42):
    """Fit the projection, source head and covariance using source rows only."""
    features = _matrix(source_features, "source_features")
    classes = _classes(classes)
    scores = _matrix(source_scores, "source_scores", len(classes))
    if len(features) != len(scores) or len(features) < 2:
        raise ValueError("Source features and scores require matching rows and at least two specimens")
    labels = _encoded(source_y, classes, len(features), "source_y")
    dimensions = min(20, len(features)-1, features.shape[1])
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed)
    pca.fit(features)
    # Use the same transform for source and future rows. Randomized PCA's
    # fit_transform can return an approximate U*S that differs from X@V.
    projected = pca.transform(features)
    scaler = StandardScaler().fit(projected)
    latent = scaler.transform(projected)
    counts = np.bincount(labels, minlength=len(classes))
    weights = 1. / (len(classes)*counts[labels])
    design = _augment(latent)
    gram = design.T @ (weights[:, None]*design) / weights.sum()
    targets = np.eye(len(classes))[labels]
    head = np.linalg.solve(gram + SOURCE_HEAD_RIDGE*np.eye(design.shape[1]),
                           design.T @ (weights[:, None]*targets) / weights.sum())
    centroids = np.asarray([latent[labels == index].mean(axis=0)
                            for index in range(len(classes))])
    residuals = latent-centroids[labels]
    covariance = LedoitWolf(assume_centered=True).fit(residuals)
    centered_scores = scores-scores.mean(axis=1, keepdims=True)
    score_scale = float(np.sqrt(np.mean(centered_scores**2)))
    if not np.isfinite(score_scale) or score_scale <= np.finfo(float).eps:
        score_scale = 1.
    for name, value in [("latent", latent), ("source head", head),
                        ("precision", covariance.precision_)]:
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name}; inspect source feature conditioning")
    return LatentContext(_readonly(classes), features.shape[1], len(features),
        _readonly(pca.mean_), _readonly(pca.components_), _readonly(scaler.mean_),
        _readonly(scaler.scale_), _readonly(latent), _readonly(labels),
        _readonly(weights), _readonly(design), _readonly(gram), _readonly(head),
        _readonly(centroids), _readonly(covariance.precision_), score_scale,
        tuple(int(value) for value in counts))


def latent_candidate_grid():
    """Return the stable 60-configuration latent adaptation grid."""
    result = []

    def add(family, **parameters):
        result.append({"id": f"l{len(result):03d}_{family}",
                       "family": family, **parameters})

    add("identity")
    for anchor, ridge, weight in product([.1, 1., 10.], [.1, 1., 10.], [1., 5., 20.]):
        add("anchored_ridge", source_anchor=anchor, ridge=ridge, support_weight=weight)
    for neighbors, boost, blend in product([3, 7, 15], [1., 5., 20.], [.1, .25, .5]):
        add("local_memory", neighbors=neighbors, support_boost=boost, blend=blend)
    for pseudocount in [1., 3., 10., 30., 100.]:
        add("mahalanobis_centroid", source_pseudocount=pseudocount)
    return result


def _candidate(values):
    result = dict(values)
    family = result.get("family")
    if family not in FAMILIES:
        raise ValueError("Unknown latent adaptation family")
    if family == "anchored_ridge":
        for name in ("source_anchor", "ridge", "support_weight"):
            result[name] = _positive(result.get(name), name)
    elif family == "local_memory":
        neighbors = result.get("neighbors")
        if (isinstance(neighbors, (bool, np.bool_))
                or not isinstance(neighbors, (int, np.integer)) or neighbors < 1):
            raise ValueError("neighbors must be a positive integer")
        result["neighbors"] = int(neighbors)
        result["support_boost"] = _positive(result.get("support_boost"), "support_boost")
        blend = _positive(result.get("blend"), "blend", allow_zero=True)
        if blend > 1:
            raise ValueError("blend must lie between zero and one")
        result["blend"] = blend
    elif family == "mahalanobis_centroid":
        result["source_pseudocount"] = _positive(result.get("source_pseudocount"), "source_pseudocount")
    return result


def anchored_ridge_delta(source_design, source_weights, support_design, support_onehot,
                         source_head, source_anchor, ridge, support_weight):
    """Solve a normalized mean-loss source-anchored residual update.

    Both source anchor and target loss are divided by their weight mass, so
    duplicating rows does not silently change effective regularization. Bias is
    represented by the last design column and is penalized with other weights.
    """
    source = _matrix(source_design, "source_design")
    support = _matrix(support_design, "support_design", source.shape[1])
    targets = _matrix(support_onehot, "support_onehot")
    head = _matrix(source_head, "source_head", targets.shape[1])
    weights = np.asarray(source_weights, dtype=float)
    if (weights.shape != (len(source),) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or head.shape[0] != source.shape[1]
            or len(targets) != len(support)):
        raise ValueError("Source weights, head and support dimensions do not agree")
    if (not np.all((targets == 0) | (targets == 1))
            or not np.all(targets.sum(axis=1) == 1)):
        raise ValueError("support_onehot requires a one-hot target per support row")
    anchor = _positive(source_anchor, "source_anchor")
    penalty = _positive(ridge, "ridge")
    target_weight = _positive(support_weight, "support_weight")
    source_gram = source.T @ (weights[:, None]*source) / weights.sum()
    target_gram = support.T @ support / len(support)
    left = anchor*source_gram + target_weight*target_gram + penalty*np.eye(source.shape[1])
    right = target_weight*support.T @ (targets-support@head) / len(support)
    delta = np.linalg.solve(left, right)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite source-anchored update")
    return delta


@dataclass
class FittedLatentAdapter:
    context: LatentContext
    candidate: dict
    metadata: dict
    delta: np.ndarray | None = None
    head_coefficients: np.ndarray | None = None
    centroids: np.ndarray | None = None
    support_latent: np.ndarray | None = None
    support_encoded: np.ndarray | None = None

    def predict_scores(self, query_features, query_scores):
        features = _matrix(query_features, "query_features", self.context.n_features)
        scores = _matrix(query_scores, "query_scores", len(self.context.classes))
        if len(features) != len(scores):
            raise ValueError("Query feature and score rows must agree")
        family = self.candidate["family"]
        if family == "identity":
            return scores.copy()
        if family == "anchored_ridge":
            result = _augment(self.context.project(features)) @ self.head_coefficients
        elif family == "mahalanobis_centroid":
            latent = self.context.project(features)
            difference = latent[:, None, :]-self.centroids[None, :, :]
            result = -.5*np.einsum("nki,ij,nkj->nk", difference,
                                    self.context.source_precision, difference)
        else:
            latent = self.context.project(features)
            distances = self.context.source_distances(features)
            encoded = self.context.source_encoded
            weights = self.context.source_weights*self.context.source_count
            if self.support_latent is not None:
                distances = np.column_stack([distances, _squared_distances(latent, self.support_latent)])
                encoded = np.concatenate([encoded, self.support_encoded])
                weights = np.concatenate([weights, np.full(len(self.support_latent),
                                                           self.candidate["support_boost"])])
            k = min(self.candidate["neighbors"], distances.shape[1])
            cutoff = np.partition(distances, k-1, axis=1)[:, k-1:k]
            # Include all exact cutoff ties: results do not depend on support
            # order when equal-distance points straddle the kth position.
            selected = distances <= cutoff
            votes = np.where(selected, weights[None, :]/(np.sqrt(distances)+DISTANCE_EPSILON), 0.)
            votes = votes @ np.eye(len(self.context.classes))[encoded]
            votes /= votes.sum(axis=1, keepdims=True)
            source = _softmax(scores/self.context.source_score_scale)
            blend = self.candidate["blend"]
            result = (1-blend)*source + blend*votes
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite latent-adapter output")
        return result


def fit_latent_adapter(context, candidate, support_features=None, support_scores=None, support_y=None):
    """Fit support-only adaptation, or a matched zero-support readout.

    A no-support ridge, memory or centroid readout can differ from the original
    classifier. Its exact zero-support control must be reported separately;
    that base-readout improvement is not a gain from support labels.
    """
    candidate = _candidate(candidate)
    absent = (support_features is None, support_scores is None, support_y is None)
    if any(absent) and not all(absent):
        raise ValueError("Supply all support arrays, or none for the matched control")
    family = candidate["family"]
    support_count = 0
    latent = None
    labels = None
    if not all(absent):
        features = _matrix(support_features, "support_features", context.n_features)
        scores = _matrix(support_scores, "support_scores", len(context.classes))
        if len(features) != len(scores):
            raise ValueError("Support feature and score rows must agree")
        labels = _encoded(support_y, context.classes, len(features), "support_y")
        support_count = len(features)
        latent = context.project(features)
    adapter = FittedLatentAdapter(context, candidate, {
        "candidate_id": candidate.get("id", family), "family": family,
        "support_count": support_count,
        "uses_support_labels": bool(support_count and family != "identity"),
        "zero_support_control": support_count == 0,
        "representation_fit_scope": "source_only", "query_statistics_fitted": False,
        "scores_are_calibrated_probabilities": False})
    if family == "identity":
        return adapter
    if family == "anchored_ridge":
        delta = np.zeros_like(context.source_head)
        if support_count:
            delta = anchored_ridge_delta(context.source_design, context.source_weights,
                _augment(latent), np.eye(len(context.classes))[labels], context.source_head,
                candidate["source_anchor"], candidate["ridge"], candidate["support_weight"])
        adapter.delta = _readonly(delta)
        adapter.head_coefficients = _readonly(context.source_head + delta)
        adapter.metadata.update(source_head_ridge=SOURCE_HEAD_RIDGE,
            losses_normalized_by_weight_mass=True, bias_penalized=True)
    elif family == "mahalanobis_centroid":
        centroids = context.source_centroids.copy()
        if support_count:
            tau = candidate["source_pseudocount"]
            for index in range(len(context.classes)):
                rows = latent[labels == index]
                centroids[index] = (tau*centroids[index]+rows.sum(axis=0))/(tau+len(rows))
        adapter.centroids = _readonly(centroids)
        adapter.metadata.update(covariance_fit_scope="source_within_class_residuals_only",
                                covariance_shrinkage="LedoitWolf", class_priors="uniform")
    else:
        if support_count:
            adapter.support_latent = _readonly(latent)
            adapter.support_encoded = _readonly(labels)
        adapter.metadata.update(distance_weight="inverse_euclidean_plus_1e-8",
            distance_ties="include_all_at_kth_distance", source_memory_class_balanced=True,
            source_score_scale=context.source_score_scale,
            score_scale_fit_scope="source_only")
    return adapter
