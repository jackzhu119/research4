"""Small, resumable NumPy ProtoNet and matched supervised embedding control.

This module does not split data, choose checkpoints, or inspect validation scores.
Callers must provide training-only arrays and keep evaluation query samples out of
training. Labels are integer encoded in the fixed range ``0 .. n_classes - 1``.
Episodes are class balanced, with disjoint support and query record indices.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax_loss(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    normalizer = np.exp(shifted).sum(axis=1, keepdims=True)
    log_prob = shifted - np.log(normalizer)
    loss = -float(log_prob[np.arange(len(y)), y].mean())
    grad = np.exp(log_prob)
    grad[np.arange(len(y)), y] -= 1.0
    grad /= len(y)
    return loss, grad


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prototype_probabilities(
    support: np.ndarray,
    support_y: np.ndarray,
    query: np.ndarray,
    n_classes: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """Fixed Euclidean prototypes, also usable as an unlearned baseline.

    Inputs may be raw standardized features or an explicitly chosen embedding.
    Every class needs at least one support record. Probability columns use the
    explicit integer class order; they are distance-softmax scores, not calibrated
    probabilities of correctness.
    """
    support = np.asarray(support, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    support_y = np.asarray(support_y)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query must be two-dimensional with equal feature widths")
    if support_y.shape != (len(support),) or not np.issubdtype(support_y.dtype, np.integer):
        raise ValueError("support_y must contain one integer class per support record")
    if np.any((support_y < 0) | (support_y >= n_classes)):
        raise ValueError("support labels are outside the fixed class range")
    if not np.isfinite(support).all() or not np.isfinite(query).all():
        raise ValueError("prototype inputs must be finite")
    if any(not np.any(support_y == label) for label in range(n_classes)):
        raise ValueError("support must include every fixed class")
    prototypes = np.stack([support[support_y == label].mean(axis=0) for label in range(n_classes)])
    distance = ((query[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return _probabilities(-distance / temperature)


@dataclass(frozen=True)
class Episode:
    support_indices: np.ndarray
    query_indices: np.ndarray


class EpisodicEmbedding:
    """One-layer tanh embedding with Adam and a supervised linear-head control.

    ``mode='episodic'`` backpropagates the query cross-entropy through BOTH query
    embeddings and the means of support embeddings. ``mode='supervised'`` uses
    the same embedding and a linear head on exactly the same number of sampled
    records per update. It therefore matches update count and record exposure,
    without claiming identical floating-point operations or loss definitions.

    A chunk always completes its requested number of updates unless input or
    numerical validity fails. There is no metric, time, or patience stopping rule.
    Serialization includes Adam moments, optimizer step, and the RNG state.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        embedding_dim: int = 16,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if any(not isinstance(value, (int, np.integer)) or value <= 0 for value in (n_features, n_classes, embedding_dim)):
            raise ValueError("feature, class, and embedding dimensions must be positive integers")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.embedding_dim = int(embedding_dim)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.params = {
            "W": self.rng.normal(0.0, 1.0 / np.sqrt(self.n_features), (self.n_features, self.embedding_dim)),
            "b": np.zeros(self.embedding_dim, dtype=np.float64),
            "V": self.rng.normal(0.0, 1.0 / np.sqrt(self.embedding_dim), (self.embedding_dim, self.n_classes)),
            "c": np.zeros(self.n_classes, dtype=np.float64),
        }
        self.first_moment = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second_moment = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.optimizer_steps = {key: 0 for key in self.params}
        self.step = 0
        self.head_trained = False
        self.last_episode: Episode | None = None

    def _features(self, X: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features or (not allow_empty and len(X) == 0):
            raise ValueError("X has an invalid shape or no records")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite")
        return X

    def _labels(self, y: np.ndarray, length: int, *, all_classes: bool = False) -> np.ndarray:
        y = np.asarray(y)
        if y.shape != (length,) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain one integer class per record")
        if np.any((y < 0) | (y >= self.n_classes)):
            raise ValueError("labels are outside the fixed class range")
        if all_classes and not np.array_equal(np.unique(y), np.arange(self.n_classes)):
            raise ValueError("training/support records must contain every fixed class")
        return y.astype(np.int64, copy=False)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._features(X, allow_empty=True)
        return np.tanh(X @ self.params["W"] + self.params["b"])

    def sample_episode(self, y: np.ndarray, support_per_class: int, query_per_class: int) -> Episode:
        y = self._labels(y, len(y), all_classes=True)
        for value in (support_per_class, query_per_class):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError("support/query sizes must be positive integers")
        total = int(support_per_class + query_per_class)
        groups = [np.flatnonzero(y == label) for label in range(self.n_classes)]
        if any(len(indices) < total for indices in groups):
            raise ValueError("each class needs enough distinct records for disjoint support and query")
        draws = [self.rng.choice(indices, total, replace=False) for indices in groups]
        return Episode(
            np.concatenate([draw[:support_per_class] for draw in draws]),
            np.concatenate([draw[support_per_class:] for draw in draws]),
        )

    def episodic_loss_and_gradients(
        self, X_support: np.ndarray, y_support: np.ndarray, X_query: np.ndarray, y_query: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Public differentiable objective, suitable for numerical verification."""
        X_support = self._features(X_support)
        X_query = self._features(X_query)
        y_support = self._labels(y_support, len(X_support), all_classes=True)
        y_query = self._labels(y_query, len(X_query))
        support = self.transform(X_support)
        query = self.transform(X_query)
        counts = np.bincount(y_support, minlength=self.n_classes)
        prototypes = np.stack([support[y_support == label].mean(axis=0) for label in range(self.n_classes)])
        difference = query[:, None, :] - prototypes[None, :, :]
        logits = -(difference ** 2).sum(axis=2) / self.temperature
        loss, grad_logits = _softmax_loss(logits, y_query)
        grad_query = (-2.0 / self.temperature) * (grad_logits[:, :, None] * difference).sum(axis=1)
        grad_prototypes = (2.0 / self.temperature) * (grad_logits[:, :, None] * difference).sum(axis=0)
        grad_support = grad_prototypes[y_support] / counts[y_support, None]
        grad_support_pre = grad_support * (1.0 - support ** 2)
        grad_query_pre = grad_query * (1.0 - query ** 2)
        gradients = {
            "W": X_support.T @ grad_support_pre + X_query.T @ grad_query_pre + self.weight_decay * self.params["W"],
            "b": grad_support_pre.sum(axis=0) + grad_query_pre.sum(axis=0),
        }
        loss += 0.5 * self.weight_decay * float((self.params["W"] ** 2).sum())
        return loss, gradients

    def supervised_loss_and_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        X = self._features(X)
        y = self._labels(y, len(X))
        embedded = self.transform(X)
        logits = embedded @ self.params["V"] + self.params["c"]
        loss, grad_logits = _softmax_loss(logits, y)
        grad_pre = (grad_logits @ self.params["V"].T) * (1.0 - embedded ** 2)
        gradients = {
            "W": X.T @ grad_pre + self.weight_decay * self.params["W"],
            "b": grad_pre.sum(axis=0),
            "V": embedded.T @ grad_logits + self.weight_decay * self.params["V"],
            "c": grad_logits.sum(axis=0),
        }
        loss += 0.5 * self.weight_decay * float((self.params["W"] ** 2).sum() + (self.params["V"] ** 2).sum())
        return loss, gradients

    def _update(self, gradients: dict[str, np.ndarray]) -> None:
        next_step = self.step + 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        updates = {}
        for key, grad in gradients.items():
            if not np.isfinite(grad).all():
                raise FloatingPointError("non-finite gradient; existing parameters were preserved")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    moment1 = beta1 * self.first_moment[key] + (1.0 - beta1) * grad
                    moment2 = beta2 * self.second_moment[key] + (1.0 - beta2) * grad ** 2
                    parameter_step = self.optimizer_steps[key] + 1
                    corrected1 = moment1 / (1.0 - beta1 ** parameter_step)
                    corrected2 = moment2 / (1.0 - beta2 ** parameter_step)
                    param = self.params[key] - self.learning_rate * corrected1 / (np.sqrt(corrected2) + epsilon)
            except FloatingPointError as exc:
                raise FloatingPointError("invalid optimizer arithmetic; all existing optimizer state was preserved") from exc
            if not all(np.isfinite(value).all() for value in (moment1, moment2, param)):
                raise FloatingPointError("non-finite update/moment; all existing optimizer state was preserved")
            updates[key] = (param, moment1, moment2, parameter_step)
        for key, (param, moment1, moment2, parameter_step) in updates.items():
            self.params[key] = param
            self.first_moment[key] = moment1
            self.second_moment[key] = moment2
            self.optimizer_steps[key] = parameter_step
        self.step = next_step

    def train_chunk(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps: int,
        support_per_class: int = 1,
        query_per_class: int = 5,
        mode: str = "episodic",
    ) -> dict[str, Any]:
        """Train a fixed-size chunk; caller checkpoints and schedules next chunks."""
        X = self._features(X)
        y = self._labels(y, len(X), all_classes=True)
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if mode not in {"episodic", "supervised"}:
            raise ValueError("mode must be episodic or supervised")
        losses = []
        for _ in range(steps):
            episode = self.sample_episode(y, support_per_class, query_per_class)
            support, query = episode.support_indices, episode.query_indices
            if mode == "episodic":
                loss, gradients = self.episodic_loss_and_gradients(X[support], y[support], X[query], y[query])
            else:
                indices = np.concatenate([support, query])
                loss, gradients = self.supervised_loss_and_gradients(X[indices], y[indices])
            if not np.isfinite(loss):
                raise FloatingPointError("non-finite training loss; update was not applied")
            self._update(gradients)
            if mode == "supervised":
                self.head_trained = True
            self.last_episode = episode
            losses.append(loss)
        return {
            "mode": mode,
            "step": self.step,
            "chunk_steps": int(steps),
            "mean_training_loss": float(np.mean(losses)),
            "last_training_loss": float(losses[-1]),
            "records_per_update": self.n_classes * (int(support_per_class) + int(query_per_class)),
            "support_per_class": int(support_per_class),
            "query_per_class": int(query_per_class),
        }

    def predict_proba(self, X: np.ndarray, X_support: np.ndarray | None = None, y_support: np.ndarray | None = None) -> np.ndarray:
        """Predict with prototypes when support is supplied; otherwise linear head."""
        embedded = self.transform(X)
        if X_support is None and y_support is None:
            if not self.head_trained:
                raise ValueError("the linear head has not been trained; supply labeled support for prototype prediction")
            return _probabilities(embedded @ self.params["V"] + self.params["c"])
        if X_support is None or y_support is None:
            raise ValueError("both X_support and y_support must be supplied")
        support = self.transform(X_support)
        labels = self._labels(y_support, len(support), all_classes=True)
        return prototype_probabilities(support, labels, embedded, self.n_classes, self.temperature)

    def predict(self, X: np.ndarray, X_support: np.ndarray | None = None, y_support: np.ndarray | None = None) -> np.ndarray:
        return self.predict_proba(X, X_support, y_support).argmax(axis=1)

    def state_dict(self) -> dict[str, Any]:
        """Return independent, JSON-serializable state, including exact RNG state."""
        return {
            "state_version": self.STATE_VERSION,
            "config": {
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "embedding_dim": self.embedding_dim,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "weight_decay": self.weight_decay,
                "random_state": self.random_state,
            },
            "params": {key: value.tolist() for key, value in self.params.items()},
            "first_moment": {key: value.tolist() for key, value in self.first_moment.items()},
            "second_moment": {key: value.tolist() for key, value in self.second_moment.items()},
            "optimizer_steps": dict(self.optimizer_steps),
            "step": self.step,
            "head_trained": self.head_trained,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "EpisodicEmbedding":
        if state.get("state_version") != cls.STATE_VERSION:
            raise ValueError("unsupported embedding checkpoint version")
        model = cls(**state["config"])
        for field in ("params", "first_moment", "second_moment"):
            if set(state[field]) != set(model.params):
                raise ValueError("checkpoint parameter keys do not match")
            restored = {}
            for key, initial in model.params.items():
                value = np.asarray(state[field][key], dtype=np.float64)
                if value.shape != initial.shape or not np.isfinite(value).all():
                    raise ValueError("checkpoint contains invalid parameter/moment arrays")
                restored[key] = value.copy()
            setattr(model, field, restored)
        if not isinstance(state["step"], int) or state["step"] < 0:
            raise ValueError("checkpoint optimizer step is invalid")
        if any((value < 0).any() for value in model.second_moment.values()):
            raise ValueError("checkpoint contains a negative second moment")
        model.step = state["step"]
        if set(state["optimizer_steps"]) != set(model.params) or any(
            not isinstance(count, int) or count < 0 or count > model.step
            for count in state["optimizer_steps"].values()
        ):
            raise ValueError("checkpoint per-parameter optimizer steps are invalid")
        model.optimizer_steps = dict(state["optimizer_steps"])
        if not isinstance(state["head_trained"], bool):
            raise ValueError("checkpoint supervised-head flag is invalid")
        model.head_trained = state["head_trained"]
        model.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return model
