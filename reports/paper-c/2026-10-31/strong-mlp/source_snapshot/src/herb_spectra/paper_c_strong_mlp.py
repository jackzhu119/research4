"""Same-fold validation of the previously locked 64--32 PyTorch MLP.

This fixed, previously selected architecture is not the new sklearn MLP grid.
The paired 100-update continuation uses source labels only, not calibration.
"""
from datetime import datetime, timezone
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import torch

from .calibration_exploration import encoded
from .classification import classification_metrics
from .image_probability import supervised_continuation
from .neural_base_search import DenseClassifier, fit_model, fit_preprocessor, model_scores, transform_features
from .paper_fewshot_classification import state_fingerprint
from .staged_nested_search import digest, prepare_protocol_roles, write_json


METHODS = ("strong_mlp_200", "strong_mlp_200_plus100_source_updates")
SOURCE_FILES = (
    "src/herb_spectra/paper_c_strong_mlp.py", "scripts/run_paper_c_strong_mlp.py",
    "src/herb_spectra/neural_base_search.py", "src/herb_spectra/image_probability.py",
    "src/herb_spectra/paper_fewshot_classification.py", "src/herb_spectra/calibration_exploration.py",
    "src/herb_spectra/staged_nested_search.py", "src/herb_spectra/classification.py",
    "src/herb_spectra/spectral_models.py", "src/herb_spectra/selection.py",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def event(output, name, **fields):
    with (Path(output) / "stage_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": name, "utc": utc(), **fields}) + "\n")


def metric_record(truth, predicted, classes):
    value = classification_metrics(truth, predicted, classes)
    matrix = np.asarray(value["confusion_matrix"], dtype=int)
    tp = matrix.diagonal(); actual = matrix.sum(axis=1); found = matrix.sum(axis=0)
    value["per_class"] = {str(c): {"precision": float(t / p) if p else 0., "recall": float(t / n) if n else 0.,
        "f1": float(2 * t / (p + n)) if p + n else 0., "support": int(n)}
        for c, t, p, n in zip(classes, tp, found, actual, strict=True)}
    value.update(correct=int(np.count_nonzero(np.asarray(truth) == np.asarray(predicted))), total=len(truth))
    return value


def fit_source_pair(X, y, identifiers, candidate, seed):
    """Fit the original source-only 200-epoch model, then its paired control."""
    x, y, identifiers = np.asarray(X), np.asarray(y), np.asarray(identifiers)
    if len(set(identifiers.tolist())) != len(x) or y.shape != (len(x),):
        raise ValueError("Each source specimen needs a unique ID and one class")
    classes = np.unique(y)
    if len(classes) != 7:
        raise ValueError("All seven source classes must remain in supervision")
    start = time.perf_counter()
    preprocessor = fit_preprocessor(x, candidate)
    features = transform_features(preprocessor, x)
    preprocessing_seconds = time.perf_counter() - start
    start = time.perf_counter()
    base, losses = fit_model(features, encoded(y, classes), candidate, seed=seed)
    base_seconds = time.perf_counter() - start
    base_before = state_fingerprint(base)
    start = time.perf_counter()
    continued, continuation = supervised_continuation(base, features, encoded(y, classes), identifiers,
                                                      seed=seed + 1600, updates=100)
    continuation_seconds = time.perf_counter() - start
    if state_fingerprint(base) != base_before:
        raise AssertionError("Continuation changed the paired base model")
    metadata = {"candidate": candidate, "model_seed": seed, "continuation_seed": seed + 1600,
        "source_ids": identifiers.tolist(), "source_samples": len(x), "input_features": x.shape[1],
        "class_order": classes.tolist(), "source_fingerprint": joblib.hash(x), "label_fingerprint": joblib.hash(y),
        "source_class_counts": {str(c): int(np.count_nonzero(y == c)) for c in classes},
        "preprocessing": preprocessor["normalization"], "scaler_mean": preprocessor["scaler"].mean_.tolist(),
        "scaler_variance": preprocessor["scaler"].var_.tolist(), "scaler_scale": preprocessor["scaler"].scale_.tolist(),
        "preprocessing_seconds": preprocessing_seconds, "base_fit_seconds": base_seconds,
        "continuation_seconds": continuation_seconds, "base_epochs": candidate["epochs"],
        "continuation_updates_not_epochs": 100, "base_optimizer": "Adam lr0.001 weight_decay0.001 minibatch64",
        "continuation_optimizer": "new Adam lr0.0001 weight_decay0; 100 source-task query CE + full-source CE updates",
        "initial_fingerprint": base._source_initial_fingerprint, "base_fingerprint": base_before,
        "base_fingerprint_after_continuation": state_fingerprint(base), "continued_fingerprint": state_fingerprint(continued),
        "base_losses": losses, "continuation": continuation,
        "parameter_count": sum(p.numel() for p in base.parameters()), "query_statistics_fitted": False,
        "target_support_labels_used": 0, "source_only": True, "early_stopping": False}
    return preprocessor, {METHODS[0]: base, METHODS[1]: continued}, metadata


def save_context(directory, preprocessor, models, metadata):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    prep_path = directory / "preprocessor.joblib"
    joblib.dump(preprocessor, prep_path, compress=3)
    paths["preprocessor.joblib"] = digest(prep_path)
    for method, model in models.items():
        path = directory / (method + ".pt")
        torch.save(model.state_dict(), path)
        paths[path.name] = digest(path)
    metadata["artifact_sha256"] = paths
    write_json(directory / "training.json", metadata)


def load_context(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "training.json").read_text())
    for name, expected in metadata["artifact_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Saved MLP context bytes changed")
    prep = joblib.load(directory / "preprocessor.joblib")
    models = {}
    for method, fingerprint in zip(METHODS, (metadata["base_fingerprint"], metadata["continued_fingerprint"]), strict=True):
        model = DenseClassifier(metadata["input_features"], metadata["candidate"]["hidden_sizes"], 7)
        model.load_state_dict(torch.load(directory / (method + ".pt"), map_location="cpu", weights_only=True))
        model.eval()
        if state_fingerprint(model) != fingerprint:
            raise ValueError("Loaded MLP context fingerprint changed")
        models[method] = model
    return prep, models, metadata


def verify_release(release, expected):
    if release.get("allow_outer_evaluation") is not True or not release.get("released_at_utc"):
        raise ValueError("Explicit dated outer evaluation release required")
    for key, value in expected.items():
        if release.get(key) != value:
            raise ValueError("Outer release hash mismatch: " + key)


def _manifest(root, protocol_path, protocol):
    return {"input_sha256": digest(root / protocol["input"]), "protocol_sha256": digest(protocol_path),
            "historical_protocol_sha256": digest(root / protocol["historical_protocol"]),
            "old_architecture_selection_lock_sha256": digest(root / protocol["old_architecture_selection_lock"]),
            "source_sha256": {p: digest(root / p) for p in SOURCE_FILES}, "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch")},
            "threads": torch.get_num_threads()}


def _predict_pair(preprocessor, models, values, truth, ids, classes, common, writer, records):
    features = transform_features(preprocessor, values)
    result = []
    for method, model in models.items():
        before = time.perf_counter()
        scores = model_scores(model, features)
        prediction = classes[scores.argmax(axis=1)]
        result.append({**common, "method": method, "metrics": metric_record(truth, prediction, classes),
                       "prediction_seconds": time.perf_counter() - before})
        for rid, label, predicted, logits in zip(ids, truth, prediction, scores, strict=True):
            row = {**common, "method": method, "record_id": str(rid), "y_true": str(label),
                   "y_predicted": str(predicted), "is_correct": int(label == predicted),
                   **{"score_" + str(c): float(s) for c, s in zip(classes, logits, strict=True)}}
            writer.writerow(row)
            records.append(row)
    return result


def run(protocol_path, output, *, stage="selection", release_path=None):
    root = Path(__file__).resolve().parents[2]
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text())
    torch.set_num_threads(1)
    manifest = _manifest(root, protocol_path, protocol)
    for key in ("input_sha256", "historical_protocol_sha256", "old_architecture_selection_lock_sha256"):
        if manifest[key] != protocol[key]:
            raise ValueError("Frozen protocol input changed: " + key)
    old = json.loads((root / protocol["old_architecture_selection_lock"]).read_text())
    if old["choice"]["selected"] != protocol["candidate"]:
        raise ValueError("Original source-selected strong MLP architecture changed")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Strong MLP source, protocol or environment changed")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
        (output / "protocol_snapshot.json").write_bytes(protocol_path.read_bytes())
        for path in SOURCE_FILES:
            target = output / "source_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / path).read_bytes())
    data, specs = prepare_protocol_roles(root / protocol["input"], root / protocol["historical_protocol"], output)
    classes = np.unique(data["y"])
    lock_path = output / "selection_lock.json"
    event(output, "stage_start", stage=stage, manifest_sha256=digest(manifest_path))
    start = time.perf_counter()
    shared_fields = ["outer_fold", "seed", "inner_fold"] if stage == "selection" else ["outer_fold", "seed"]
    fields = shared_fields + ["method", "record_id", "y_true", "y_predicted", "is_correct"] + ["score_" + str(c) for c in classes]
    if stage == "selection":
        if lock_path.exists() or (output / "inner_predictions.csv.gz").exists():
            raise ValueError("Completed or partial source run cannot be overwritten")
        rows, predictions = [], []
        with gzip.open(output / "inner_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
            for spec in specs:
                for inner_id, inner in enumerate(spec["inner_folds"]):
                    train, valid = inner["train_indices"], inner["validation_indices"]
                    assert not (set(train) & set(valid) or set(train) & set(spec["test_indices"]) or set(valid) & set(spec["test_indices"]))
                    for seed in protocol["seeds"]:
                        directory = output / "selection" / f"outer_{spec['outer_fold']:02d}" / f"inner_{inner_id}" / f"seed_{seed}"
                        prep, models, metadata = fit_source_pair(data["X_alpha"][train], data["y"][train], data["groups"][train], protocol["candidate"], seed)
                        metadata.update(outer_fold=spec["outer_fold"], inner_fold=inner_id, validation_ids=data["groups"][valid].tolist())
                        save_context(directory, prep, models, metadata)
                        rows += _predict_pair(prep, models, data["X_alpha"][valid], data["y"][valid], data["groups"][valid], classes,
                                              {"outer_fold": spec["outer_fold"], "inner_fold": inner_id, "seed": seed}, writer, predictions)
                    write_json(output / "status.json", {"status": "selection", "completed_method_cases": len(rows), "outer_test_scored": False})
                print(json.dumps({"stage": "selection", "outer_fold": spec["outer_fold"], "method_cases": len(rows)}), flush=True)
        write_json(output / "inner_metrics.json", rows)
        evidence = {str(path.relative_to(output)): digest(path) for path in (output / "selection").rglob("*") if path.is_file()}
        lock = {"created_at_utc": utc(), "manifest_sha256": digest(manifest_path), "protocol_sha256": digest(protocol_path),
                "inner_metrics_sha256": digest(output / "inner_metrics.json"), "inner_predictions_sha256": digest(output / "inner_predictions.csv.gz"),
                "source_contexts": 30, "method_cases": len(rows), "inner_prediction_rows": len(predictions),
                "fixed_candidate": protocol["candidate"], "fixed_methods": list(METHODS), "all_seeds": protocol["seeds"],
                "primary_seed": protocol["primary_seed"], "no_architecture_or_seed_selected": True, "outer_test_used_to_select": False,
                "source_artifact_sha256": evidence}
        write_json(lock_path, lock)
        write_json(output / "status.json", {"status": "selection_completed", "seconds": time.perf_counter() - start,
                    "source_contexts": 30, "method_cases": len(rows), "inner_prediction_rows": len(predictions), "outer_test_scored": False,
                    "selection_lock_sha256": digest(lock_path)})
        event(output, "selection_lock_written", selection_lock_sha256=digest(lock_path))
        return json.loads((output / "status.json").read_text())
    if stage != "evaluation" or release_path is None:
        raise ValueError("Evaluation requires an explicit outer release")
    lock = json.loads(lock_path.read_text())
    release = json.loads(Path(release_path).read_text())
    verify_release(release, {"selection_lock_sha256": digest(lock_path), "manifest_sha256": digest(manifest_path), "protocol_sha256": digest(protocol_path)})
    for name, expected in lock["source_artifact_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError("Frozen inner source artifact changed")
    for name, key in (("inner_metrics.json", "inner_metrics_sha256"), ("inner_predictions.csv.gz", "inner_predictions_sha256")):
        if digest(output / name) != lock[key]:
            raise ValueError("Frozen inner metrics or predictions changed")
    if (output / "outer_predictions.csv.gz").exists():
        raise ValueError("Outer predictions already exist")
    (output / "release_snapshot.json").write_bytes(Path(release_path).read_bytes())
    rows, predictions, contexts = [], [], []
    with gzip.open(output / "outer_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for spec in specs:
            train, test = spec["train_indices"], spec["test_indices"]
            for seed in protocol["seeds"]:
                directory = output / "models" / f"outer_{spec['outer_fold']:02d}" / f"seed_{seed}"
                prep, models, metadata = fit_source_pair(data["X_alpha"][train], data["y"][train], data["groups"][train], protocol["candidate"], seed)
                metadata.update(outer_fold=spec["outer_fold"], heldout_ids=data["groups"][test].tolist())
                save_context(directory, prep, models, metadata)
                contexts.append({"outer_fold": spec["outer_fold"], "seed": seed, "directory": str(directory),
                                 "preprocessing_seconds": metadata["preprocessing_seconds"], "base_fit_seconds": metadata["base_fit_seconds"],
                                 "continuation_seconds": metadata["continuation_seconds"], "parameter_count": metadata["parameter_count"],
                                 "training_record_sha256": digest(directory / "training.json")})
                rows += _predict_pair(prep, models, data["X_alpha"][test], data["y"][test], data["groups"][test], classes,
                                      {"outer_fold": spec["outer_fold"], "seed": seed}, writer, predictions)
            print(json.dumps({"stage": "evaluation", "outer_fold": spec["outer_fold"], "method_cases": len(rows)}), flush=True)
    pooled, summary = {}, {}
    for method in METHODS:
        pooled[method] = {}
        for seed in protocol["seeds"]:
            found = [row for row in predictions if row["method"] == method and row["seed"] == seed]
            assert len(found) == len({row["record_id"] for row in found}) == 855
            pooled[method][str(seed)] = metric_record(np.asarray([row["y_true"] for row in found]), np.asarray([row["y_predicted"] for row in found]), classes)
            selected_contexts = [context for context in contexts if context["seed"] == seed]
            pooled[method][str(seed)]["cost"] = {"source_preprocessing_seconds": sum(c["preprocessing_seconds"] for c in selected_contexts),
                "source_base_fit_seconds": sum(c["base_fit_seconds"] for c in selected_contexts),
                "additional_continuation_seconds": 0. if method == METHODS[0] else sum(c["continuation_seconds"] for c in selected_contexts),
                "prediction_seconds": sum(row["prediction_seconds"] for row in rows if row["seed"] == seed and row["method"] == method),
                "parameter_count_each_model": selected_contexts[0]["parameter_count"], "input_features": 612, "latent_features": 32}
        summary[method] = {metric: {"mean": float(np.mean([pooled[method][str(s)][metric] for s in protocol["seeds"]])),
                                   "sample_std": float(np.std([pooled[method][str(s)][metric] for s in protocol["seeds"]], ddof=1))}
                           for metric in ("accuracy", "macro_f1", "balanced_accuracy")}
    write_json(output / "pooled_metrics_by_seed.json", pooled)
    write_json(output / "seed_sensitivity.json", summary)
    write_json(output / "primary_seed42_metrics.json", {method: pooled[method]["42"] for method in METHODS})
    write_json(output / "outer_metrics.json", rows)
    write_json(output / "model_contexts.json", contexts)
    write_json(output / "status.json", {"status": "completed", "seconds": time.perf_counter() - start, "outer_source_contexts": 15,
                "outer_method_cases": len(rows), "outer_prediction_rows": len(predictions), "all_seeds_reported": True,
                "primary_seed": 42, "selection_lock_sha256": digest(lock_path)})
    event(output, "evaluation_completed", outer_predictions_sha256=digest(output / "outer_predictions.csv.gz"))
    return json.loads((output / "status.json").read_text())
