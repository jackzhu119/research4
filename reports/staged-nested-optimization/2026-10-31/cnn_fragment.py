"""Scratch, source-only CNN helper for integration in the nested R4 runner.

No real-data training runs on import. The executable self-check uses a clearly
labelled synthetic fixture and is not a research accuracy estimate.
"""
import io

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from herb_spectra.paper_fewshot_classification import state_fingerprint


def _cnn_tensor(values):
    """Create a finite CPU float32 batch with the complete ordered 612 points."""
    if torch.is_tensor(values):
        tensor = values.detach().to(device="cpu", dtype=torch.float32)
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("CNN input must contain finite numerical source-scaled spectra")
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    if tensor.ndim != 3 or tensor.shape[1:] != (1, 612) or tensor.shape[0] < 1:
        raise ValueError("CNN requires a nonempty [N,612] or [N,1,612] batch")
    if not torch.isfinite(tensor).all():
        raise ValueError("CNN input has nonfinite values")
    return tensor.contiguous()


class SmallNestedCNN(nn.Module):
    """The fixed full-spectrum 8→16-channel CNN and 32-coordinate embedding."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.embedding = nn.Linear(256, 32)
        self.head = nn.Linear(32, 7)

    def encode(self, values):
        if (not torch.is_tensor(values) or values.ndim != 3
                or values.shape[1:] != (1, 612)):
            raise ValueError("SmallNestedCNN.encode expects a [N,1,612] tensor")
        return F.relu(self.embedding(self.features(values).flatten(1)))

    def forward(self, values):
        return self.head(self.encode(values))


def predict_logits(network, values):
    """Pointwise logits only; no fitting, gradients, query statistics or labels."""
    network.eval()
    with torch.no_grad():
        scores = network(_cnn_tensor(values)).cpu().numpy().copy()
    if scores.shape != (len(values), 7) or not np.isfinite(scores).all():
        raise FloatingPointError("CNN returned invalid seven-class logits")
    return scores


def encode_source(network, values):
    """Frozen 32-dimensional embedding, using the same fitted source network."""
    network.eval()
    with torch.no_grad():
        embedding = network.encode(_cnn_tensor(values)).cpu().numpy().copy()
    if embedding.shape != (len(values), 32) or not np.isfinite(embedding).all():
        raise FloatingPointError("CNN returned invalid embeddings")
    return embedding


def fit_cnn_source(Zsource, ysource_encoded, *, seed=42, epochs=200,
                   batch_size=64, learning_rate=.001, weight_decay=.001):
    """Fit exclusively on caller-source data, with no validation/test API.

    The caller must fit its own preprocessing and full-612-point StandardScaler
    on the same source partition. Thread control belongs to the caller.
    """
    features = _cnn_tensor(Zsource)
    encoded = np.asarray(ysource_encoded)
    if (encoded.shape != (len(features),) or encoded.dtype.kind not in "iu"
            or set(encoded.tolist()) != set(range(7))):
        raise ValueError("Source training requires each of the seven integer classes 0..6")
    if (not isinstance(epochs, int) or epochs < 1
            or not isinstance(batch_size, int) or batch_size < 1
            or not np.isfinite([learning_rate, weight_decay]).all()
            or learning_rate <= 0 or weight_decay < 0):
        raise ValueError("Invalid fixed source-training budget")
    labels = torch.as_tensor(encoded, dtype=torch.long)
    counts = torch.bincount(labels, minlength=7).float()
    weights = len(features)/(7*counts)
    torch.manual_seed(seed)
    network = SmallNestedCNN()
    initial = state_fingerprint(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for epoch in range(epochs):
        network.train()
        total_loss = 0.
        for indices in torch.randperm(len(features), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(network(features[indices]), labels[indices], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weighted source CNN cross-entropy")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        entry = {"epoch": epoch+1,
                 "mean_batch_weighted_cross_entropy": total_loss/len(features)}
        if epoch == 0 or (epoch+1) % 20 == 0 or epoch+1 == epochs:
            predictions = predict_logits(network, features).argmax(1)
            confusion = np.bincount(encoded*7+predictions, minlength=49).reshape(7, 7)
            entry.update(source_accuracy=float(np.trace(confusion)/len(encoded)),
                         source_confusion_matrix=confusion.tolist())
        losses.append(entry)
    network.eval()
    metadata = {
        "fit_scope": "caller_source_training_only",
        "source_samples": len(features),
        "input_features": 612,
        "input": "caller-source fitted preprocessing and StandardScaler; no MC feature cut",
        "source_class_counts": counts.to(dtype=torch.int64).tolist(),
        "balanced_class_weights": weights.tolist(),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "loss_semantics": "PyTorch weighted CE normalized by weights present in each mini-batch; epoch report is batch-size weighted mean, not full-source global weighted CE",
        "validation_or_test_labels_accepted": False,
        "early_stopping": False,
        "input_batch_statistics_fitted": False,
        "network_normalization": "per-instance GroupNorm, no BatchNorm",
        "embedding_dimensions": 32,
        "head_classes": 7,
        "dropout": 0,
        "prediction": "softmax argmax equals logit argmax; returned scores are logits",
        "initial_fingerprint": initial,
        "final_fingerprint": state_fingerprint(network),
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "losses": losses,
    }
    return network, metadata


def _synthetic_selfcheck():
    """Synthetic engineering assertions, never real-research accuracy."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(94)
    features = rng.normal(size=(21, 612)).astype(np.float32)
    labels = np.repeat(np.arange(7), 3)
    features_before = features.copy()
    network, metadata = fit_cnn_source(features, labels, epochs=2)
    fingerprint = state_fingerprint(network)
    logits = predict_logits(network, features)
    embedding = encode_source(network, features)
    np.testing.assert_allclose(predict_logits(network, features[:1]), logits[:1],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(encode_source(network, features[:1]), embedding[:1],
                               rtol=1e-5, atol=1e-6)
    assert fingerprint == state_fingerprint(network)
    np.testing.assert_array_equal(features, features_before)
    assert logits.shape == (21, 7) and embedding.shape == (21, 32)
    assert len(metadata["losses"]) == 2 and not network.training
    buffer = io.BytesIO()
    torch.save(network.state_dict(), buffer)
    buffer.seek(0)
    restored = SmallNestedCNN()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    restored.eval()
    np.testing.assert_array_equal(logits, predict_logits(restored, features))
    assert fingerprint == state_fingerprint(restored)
    again, _ = fit_cnn_source(features, labels, epochs=2)
    assert fingerprint == state_fingerprint(again)
    return {"synthetic_fixture_only": True, "source_samples": 21,
            "epochs": 2, "parameters": metadata["parameter_count"],
            "shape_and_pointwise_checks": "pass", "source_frozen_checks": "pass",
            "state_roundtrip": "pass", "repeat_seed_determinism": "pass"}


if __name__ == "__main__":
    import json
    print(json.dumps(_synthetic_selfcheck()))
