"""Read-only CNN source-state audit for ten nested selection callers.

Only caller training IDs determine the spectra/labels used below. No validation
or outer-test prediction files or test scores are read, and no fit is invoked.
"""
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/staged-nested-20261031"
RUN = BASE / "combinations"
CHECKS = 0
MAXIMUM_DIFFERENCE = 0.


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(first, second, message, *, atol=1e-10, rtol=1e-10):
    global MAXIMUM_DIFFERENCE
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    difference = float(np.max(np.abs(first-second)))
    MAXIMUM_DIFFERENCE = max(MAXIMUM_DIFFERENCE, difference)
    check(np.allclose(first, second, atol=atol, rtol=rtol), message)
    return difference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_fingerprint(network):
    """Independent tensor-value implementation, without tensor pickle IDs."""
    value = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def model_fingerprint(model):
    return joblib.hash({
        "family": model.family, "inherited": model.inherited,
        "preprocessor": model.preprocessor, "scaler": model.scaler,
        "indices": model.indices, "classes": model.classes_,
        "estimator": model.estimator, "embedding_scaler": model.embedding_scaler,
        "network_state": state_fingerprint(model.network),
        "network_training": bool(model.network.training), "metadata": model.metadata,
    })


def inspect_architecture(network):
    convolutions = [m for m in network.modules() if isinstance(m, nn.Conv1d)]
    groups = [m for m in network.modules() if isinstance(m, nn.GroupNorm)]
    pools = [m for m in network.modules() if isinstance(m, nn.AvgPool1d)]
    adaptive = [m for m in network.modules() if isinstance(m, nn.AdaptiveAvgPool1d)]
    check([(m.in_channels, m.out_channels, m.kernel_size, m.padding) for m in convolutions]
          == [(1, 8, (7,), (3,)), (8, 16, (5,), (2,))], "Fixed convolution architecture")
    check([(m.num_groups, m.num_channels) for m in groups] == [(2, 8), (4, 16)], "Fixed GroupNorm architecture")
    check(len(pools) == 1 and pools[0].kernel_size == (2,) and pools[0].stride == (2,), "Fixed average pool")
    check(len(adaptive) == 1 and adaptive[0].output_size == 16, "Fixed adaptive pool")
    check(network.embedding.in_features == 256 and network.embedding.out_features == 32,
          "Fixed 32-dimensional embedding")
    check(network.head.in_features == 32 and network.head.out_features == 7, "Fixed seven-class head")
    check(sum(p.numel() for p in network.parameters()) == 9223, "Exactly 9223 network parameters")
    check(not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in network.modules()), "No BatchNorm")
    check(not any(isinstance(m, nn.modules.dropout._DropoutNd) for m in network.modules()), "No dropout")
    check(not network.training, "Persisted network is in evaluation mode")


def main():
    started = time.monotonic()
    directories = sorted((RUN / "selection").glob("outer_*/inner_*"))
    complete = [p for p in directories if (p / "source_checkpoint.json").exists()]
    if len(complete) != 10:
        print(json.dumps({"status": "waiting_for_all_ten_source_callers", "complete": len(complete)}))
        return None
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    input_path = ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"
    protocol_path = ROOT / "data/research/staged_nested_combinations_protocol_20261031.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    check(sha(input_path) == protocol["input_sha256"], "Frozen input SHA")
    check(manifest["protocol_sha256"] == sha(protocol_path), "Frozen R4 protocol SHA")
    for relative, digest in manifest["source_sha256"].items():
        check(sha(ROOT / relative) == digest, "Frozen source SHA: " + relative)
    splits = json.loads((BASE / "core/splits.json").read_text())
    with np.load(input_path, allow_pickle=False) as data:
        identifiers = data["record_ids"]
        features = data["X_alpha"]
        labels = data["y"]
    check(features.shape == (855, 612), "Full ordered 612-feature input")
    lookup = {str(identifier): index for index, identifier in enumerate(identifiers)}
    check(len(lookup) == 855, "Unique record IDs")
    caller_results = []
    observed_fit_invocations = 0
    source_predictions_checked = 0
    maximum_pointwise_difference = 0.
    for directory in complete:
        checkpoint = json.loads((directory / "source_checkpoint.json").read_text())
        common = json.loads((directory / "caller_common.json").read_text())
        expected = checkpoint["expected"]
        fold = int(directory.parent.name.rsplit("_", 1)[1])
        inner = int(directory.name.rsplit("_", 1)[1])
        source_ids = expected["source_ids"]
        check(source_ids == splits[fold]["inner_folds"][inner]["train_sample_ids"], "Caller source IDs match fixed inner training partition")
        check(not set(source_ids) & set(splits[fold]["test_sample_ids"]), "No outer-test ID in CNN source")
        check(not set(source_ids) & set(splits[fold]["inner_folds"][inner]["validation_sample_ids"]), "No inner-validation ID in CNN source")
        check(len(source_ids) == 342 and len(set(source_ids)) == 342, "342 unique caller source samples")
        source_indices = np.asarray([lookup[str(identifier)] for identifier in source_ids])
        source = features[source_indices]
        truth = labels[source_indices]
        check(joblib.hash(source) == expected["source_value_hash"], "Source feature value fingerprint")
        check(joblib.hash(truth) == expected["source_label_hash"], "Source label fingerprint")
        check(common["source_ids"] == source_ids and common["source_samples"] == 342, "Common source identity")
        check(sha(directory / "caller_common.json") == checkpoint["common_sha256"], "Common metadata SHA")
        check(expected["seed"] == 42 and expected["cnn_epochs"] == 200, "Fixed seed and CNN budget")
        check(checkpoint["cnn_source_training_invocations"] == 1, "One shared CNN source fitting invocation")
        observed_fit_invocations += checkpoint["cnn_source_training_invocations"]
        neural = common["neural_training"]
        check(neural["seed"] == 42 and neural["epochs"] == 200 and neural["batch_size"] == 64, "Persisted CNN training settings")
        close(neural["learning_rate"], .001, "Fixed learning rate")
        close(neural["weight_decay"], .001, "Fixed weight decay")
        check(neural["optimizer"] == "Adam" and not neural["early_stopping"], "Fixed optimizer, no early stopping")
        check(neural["parameter_count"] == 9223, "Training metadata parameter count")
        check(len(neural["losses"]) == 200, "All 200 training epochs recorded")
        check([r["epoch"] for r in neural["losses"]] == list(range(1, 201)), "Consecutive loss epochs")
        check(np.isfinite([r["mean_batch_weighted_cross_entropy"] for r in neural["losses"]]).all(), "Finite source losses")
        classes, counts = np.unique(truth, return_counts=True)
        check(len(classes) == 7 and counts.min() == 2, "Seven classes including two rare source specimens")
        check(neural["source_class_counts"] == counts.tolist(), "Source class counts")
        close(neural["balanced_class_weights"], len(source)/(7*counts), "Source balanced CE weights", atol=1e-6, rtol=1e-6)
        check(sha(directory / "cnn_source_state.pt") == checkpoint["cnn_state_sha256"], "Standalone CNN state SHA")
        state = torch.load(directory / "cnn_source_state.pt", map_location="cpu", weights_only=True)
        models = {}
        family_results = []
        for family in ["cnn_softmax", "cnn_embedding_svm"]:
            artifact = checkpoint["artifacts"][family]
            path = directory / artifact["file"]
            check(sha(path) == artifact["sha256"], "CNN model artifact SHA")
            model = joblib.load(path)
            check(model.family == family, "Stored model family")
            check(model_fingerprint(model) == artifact["numeric_fingerprint"], "Numerical source model fingerprint")
            inspect_architecture(model.network)
            check(np.array_equal(model.classes_, classes), "Known source class order")
            check(np.array_equal(model.indices, np.arange(612)), "CNN has no MC variable cut")
            check(model.metadata["selected_features"] == 612 and not model.metadata["feature_selection_applied"], "Full-spectrum CNN metadata")
            check(model.metadata["same_encoder_shared_between_CNN_head_and_SVM"] and not model.metadata["old_weights_reused"], "Same newly fitted encoder declared")
            fingerprint = state_fingerprint(model.network)
            check(fingerprint == checkpoint["shared_encoder_fingerprint"] == neural["final_fingerprint"], "Shared learned encoder fingerprint")
            check(set(state) == set(model.network.state_dict()), "Standalone state keys")
            for name, values in state.items():
                check(torch.equal(values, model.network.state_dict()[name]), "Standalone CNN state tensor equality")
            prepared = model.preprocessor.transform(source)
            close(model.scaler.mean_, prepared.mean(0), "Source-only scaler mean")
            close(model.scaler.var_, prepared.var(0), "Source-only scaler population variance")
            close(common["source_scaler_mean"], prepared.mean(0), "Recorded source-only scaler mean")
            close(common["source_scaler_variance"], prepared.var(0), "Recorded source-only scaler variance")
            if expected["inherited"]["preprocessing"] == "msc":
                close(model.preprocessor.reference_, source.mean(0), "MSC reference equals source mean only")
            model_before = model_fingerprint(model)
            predicted = model.predict(source)
            accuracy = float(np.mean(predicted == truth))
            close(artifact["source_accuracy"], accuracy, "Independent source-training accuracy")
            source_predictions_checked += len(source)
            scores_alone = model.predict_scores(source[:1])
            scores_together = model.predict_scores(source[:7])[:1]
            pointwise_difference = close(scores_alone, scores_together, "Pointwise scores unaffected by other source batch members", atol=1e-4, rtol=1e-4)
            maximum_pointwise_difference = max(maximum_pointwise_difference, pointwise_difference)
            check(model_before == model_fingerprint(model), "Inference left source model and statistics unchanged")
            if family == "cnn_softmax":
                numeric = np.searchsorted(classes, predicted)
                encoded = np.searchsorted(classes, truth)
                cm = np.bincount(encoded*7+numeric, minlength=49).reshape(7, 7)
                check(cm.tolist() == neural["losses"][-1]["source_confusion_matrix"], "Final recorded source confusion matrix")
                close(accuracy, neural["losses"][-1]["source_accuracy"], "Final logged source CNN accuracy")
            else:
                standardized = model.scaler.transform(prepared)
                with torch.no_grad():
                    embedding = model.network.encode(torch.from_numpy(np.ascontiguousarray(standardized, dtype=np.float32))[:, None, :]).cpu().numpy()
                check(embedding.shape == (342, 32), "Source embedding dimension")
                close(model.embedding_scaler.mean_, embedding.astype(float).mean(0), "Embedding scaler fitted only on source", atol=1e-7, rtol=1e-6)
                close(model.embedding_scaler.var_, embedding.astype(float).var(0), "Embedding scaler source variance", atol=1e-7, rtol=1e-6)
                check(model.estimator.break_ties is False, "Embedding SVM standard vote prediction")
            models[family] = model
            family_results.append({"family": family, "source_accuracy": accuracy,
                                   "source_roundtrip_equal": artifact["source_roundtrip_predictions_equal"],
                                   "source_pointwise_max_absolute_difference": pointwise_difference,
                                   "numeric_fingerprint": model_before})
        check(state_fingerprint(models["cnn_softmax"].network) == state_fingerprint(models["cnn_embedding_svm"].network), "Both heads use identical learned tensors")
        caller_results.append({"source_directory": str(directory.relative_to(ROOT)),
                               "outer_fold": fold, "inner_fold": inner, "source_samples": 342,
                               "shared_encoder_fingerprint": checkpoint["shared_encoder_fingerprint"],
                               "families": family_results})
    result = {"status": "pass", "scope": "Read-only ten selection callers, source training predictions only",
              "checks_evaluated": CHECKS, "maximum_numeric_difference": MAXIMUM_DIFFERENCE,
              "maximum_pointwise_score_difference": maximum_pointwise_difference,
              "caller_sources": len(caller_results), "stored_network_parameter_count": 9223,
              "CNN_source_training_invocations_recorded": observed_fit_invocations,
              "shared_head_pairs_checked": len(caller_results),
              "source_training_predictions_recomputed": source_predictions_checked,
              "classifier_or_network_fits_in_audit": 0,
              "outer_test_or_validation_predictions_read": 0,
              "input_sha256": sha(input_path), "audit_script_sha256": sha(__file__),
              "source_fingerprints_verified": True, "results": caller_results,
              "seconds": time.monotonic()-started,
              "limits": ["Source accuracy is a training-fit diagnostic, not held-out performance.",
                         "Saved identities/hashes do not independently attest to every execution event.",
                         "The CNN softmax and embedding-SVM share an encoder, while their final classifiers differ."]}
    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as exc:
        result = {"status": "fail", "checks_evaluated": CHECKS,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "classifier_or_network_fits_in_audit": 0,
                  "outer_test_or_validation_predictions_read": 0,
                  "audit_script_sha256": sha(__file__)}
        (BASE / "cnn_source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        raise
    if result is not None:
        (BASE / "cnn_source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, ensure_ascii=False))
