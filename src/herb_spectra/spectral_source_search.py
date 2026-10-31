"""Source-only search over fixed spectral pipelines, before any final scoring.

This stage never asks a fitted classifier to score the outer prediction role.
Source-model gains are distinct from gains attributed to few-shot labels.
"""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np

from .calibration_exploration import development_episodes, encoded, fast_metrics
from .spectral_models import fit_source_candidate, source_candidate_grid
from .within_batch_holdout import prepare_holdout


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
PREPROCESS_ORDER = {"raw": 0, "snv": 1, "sg0_snv": 2,
                    "linear_detrend_snv": 3, "sg1_snv": 4, "sg2_snv": 5}
MODEL_ORDER = {"ridge": 0, "plsda": 1, "svc": 2, "pca_svc": 3,
               "shrinkage_lda": 4}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_model_path(output, candidate_id, episode):
    return Path(output) / "development" / candidate_id / f"episode_{episode:02d}.joblib"


def source_metadata_path(output, candidate_id, episode):
    return source_model_path(output, candidate_id, episode).with_suffix(".json")


def source_reference(candidates):
    return next(c for c in candidates if c.get("preprocessing") == "snv"
                and c.get("model") == "svc" and c.get("C") == 10.)


def source_complexity(candidate, reference_id):
    """Fixed tie policy; source class labels and final scores are not inputs."""
    return (candidate["id"] != reference_id,
            MODEL_ORDER[candidate["model"]], PREPROCESS_ORDER[candidate["preprocessing"]],
            candidate.get("n_components", 0), -candidate.get("alpha", 0.),
            candidate.get("C", 0.), candidate["id"])


def select_sources(aggregate, candidates, *, minimum_accuracy_gain=.001,
                   max_macro_f1_loss=.005, max_balanced_accuracy_loss=.005,
                   tie_tolerance=.001):
    """Select source pipelines on developer labels, then apply a fixed gate."""
    lookup = {c["id"]: c for c in candidates}
    reference = source_reference(candidates)
    reference_stats = aggregate[reference["id"]]
    if any(reference_stats[stage]["failures"] for stage in ("selection", "gate")):
        raise ValueError("Reference source pipeline failed a development task")

    def allowed(identifier, stage):
        value, baseline = aggregate[identifier][stage], reference_stats[stage]
        return (not value["failures"] and value["evaluations"] == baseline["evaluations"]
                and value["accuracy"]-baseline["accuracy"] > minimum_accuracy_gain
                and value["macro_f1"]-baseline["macro_f1"] >= -max_macro_f1_loss
                and value["balanced_accuracy"]-baseline["balanced_accuracy"] >= -max_balanced_accuracy_loss)

    def best(identifiers):
        maximum = max(aggregate[i]["selection"]["accuracy"] for i in identifiers)
        near = [lookup[i] for i in identifiers
                if aggregate[i]["selection"]["accuracy"] >= maximum-tie_tolerance]
        return min(near, key=lambda c: source_complexity(c, reference["id"]))

    eligible = [c["id"] for c in candidates if allowed(c["id"], "selection")]
    proposed = best(eligible) if eligible else reference
    selected = proposed if proposed["id"] != reference["id"] and allowed(proposed["id"], "gate") else reference
    families = {}
    for name, models in [("plsda", {"plsda"}), ("linear", {"ridge", "shrinkage_lda"})]:
        available = [c["id"] for c in candidates if c["model"] in models
                     and not any(aggregate[c["id"]][stage]["failures"]
                                 for stage in ("selection", "gate"))]
        winner = best(available) if available else reference
        families[name] = {"candidate": winner, "fallback_to_reference": not bool(available),
                          "selection": aggregate[winner["id"]]["selection"],
                          "gate": aggregate[winner["id"]]["gate"],
                          "passes_source_gate": allowed(winner["id"], "gate")}
    adaptation, seen = [], set()
    for candidate in [reference, selected, families["plsda"]["candidate"], families["linear"]["candidate"]]:
        if candidate["id"] not in seen:
            adaptation.append(candidate); seen.add(candidate["id"])
    return {"reference": reference, "proposed": proposed, "selected": selected,
            "fallback_to_reference": selected["id"] == reference["id"],
            "eligible_selection_candidates": len(eligible), "family_best": families,
            "candidates_for_adaptation": adaptation,
            "reference_selection": reference_stats["selection"],
            "reference_gate": reference_stats["gate"],
            "selected_selection": aggregate[selected["id"]]["selection"],
            "selected_gate": aggregate[selected["id"]]["gate"],
            "selection_scope": "original training role only; outer query labels never scored"}


def _source_code_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ["src/herb_spectra/spectral_source_search.py", "src/herb_spectra/spectral_models.py",
             "src/herb_spectra/calibration_exploration.py", "src/herb_spectra/within_batch_holdout.py",
             "src/herb_spectra/classification.py", "src/herb_spectra/benchmark.py",
             "scripts/run_spectral_source_search.py"]
    return {name: file_digest(root / name) for name in names}


def source_manifest(input_path, plan_path, protocol_path, protocol, candidates):
    return {"protocol_id": protocol["protocol_id"], "input_sha256": file_digest(input_path),
            "plan_sha256": file_digest(plan_path), "protocol_sha256": file_digest(protocol_path),
            "source_grid_sha256": object_digest(candidates), "source_code_sha256": _source_code_hashes(),
            "python": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ["numpy", "scipy", "scikit-learn", "joblib"]}}


def _settings(protocol):
    config = protocol["source_search"]
    if not 0 < config["selection_episodes"] < config["episodes"]:
        raise ValueError("Separate source selection and gate tasks are required")
    return config


def evaluate_source_episode(training, partition, candidate, output, episode,
                            selection_episodes, model_seed):
    """Fit one source pipeline on its source role, and preserve every prediction."""
    model_path = source_model_path(output, candidate["id"], episode)
    meta_path = model_path.with_suffix(".json")
    prediction_path = model_path.with_suffix(".predictions.csv")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    source, query = partition["source"], partition["query"]
    start = time.monotonic()
    meta = {"candidate": candidate, "episode": episode,
            "stage": "selection" if episode < selection_episodes else "gate",
            "source_fit_calls": 1, "status": "pass",
            "partition": {role: training["record_ids"][rows].tolist()
                          for role, rows in partition.items()}, "warnings": []}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = fit_source_candidate(candidate, training["X_alpha"][source],
                                         training["y"][source], training["axis"], model_seed)
            fingerprint = joblib.hash(model)
            scores = model.predict_scores(training["X_alpha"][query])
            classes = np.asarray(model.classes_)
            predictions = np.argmax(scores, axis=1)
            truth = encoded(training["y"][query], classes)
            metrics = fast_metrics(truth, predictions, len(classes), detailed=True)
            if joblib.hash(model) != fingerprint:
                raise AssertionError("Source model changed during developer prediction")
            temporary = model_path.with_name(model_path.name + ".partial")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(model_path)
            with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["record_id", "y_true", "y_predicted", "is_correct"],
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows({"record_id": str(identifier), "y_true": str(classes[actual]),
                                  "y_predicted": str(classes[predicted]), "is_correct": int(actual == predicted)}
                                 for identifier, actual, predicted in
                                 zip(training["record_ids"][query], truth, predictions))
            meta.update(metrics=metrics, class_order=classes.tolist(),
                        model_metadata=model.metadata, source_fingerprint_before=fingerprint,
                        source_fingerprint_after=joblib.hash(model), model_sha256=file_digest(model_path),
                        model_file=str(model_path), predictions_file=str(prediction_path),
                        predictions_sha256=file_digest(prediction_path),
                        query_labels_used_for_fit=False, final_query_scored=False)
            meta["warnings"] = sorted(set(str(item.message) for item in caught))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        meta.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        model_path.with_name(model_path.name + ".partial").unlink(missing_ok=True)
    meta["seconds"] = float(time.monotonic()-start)
    write_json(meta_path, meta)
    return meta


def aggregate_source_results(metadata, candidates, selection_episodes):
    result = {}
    for candidate in candidates:
        result[candidate["id"]] = {}
        for stage in ("selection", "gate"):
            rows = [m for m in metadata if m["candidate"]["id"] == candidate["id"]
                    and m["stage"] == stage]
            passed = [m for m in rows if m["status"] == "pass"]
            result[candidate["id"]][stage] = {
                **{metric: float(np.mean([m["metrics"][metric] for m in passed]))
                   if passed else 0. for metric in METRICS},
                "evaluations": len(passed), "failures": len(rows)-len(passed)}
    return result


def load_source_model(output, candidate_id, episode):
    """Only load locally generated cache artifacts after their recorded SHA check."""
    meta = json.loads(source_metadata_path(output, candidate_id, episode).read_text(encoding="utf-8"))
    path = source_model_path(output, candidate_id, episode)
    if meta["status"] != "pass" or file_digest(path) != meta["model_sha256"]:
        raise ValueError("Source cache artifact is incomplete or changed")
    model = joblib.load(path)
    if joblib.hash(model) != meta["source_fingerprint_before"]:
        raise ValueError("Source cached fitted-state fingerprint differs")
    return model


def run_source_search(input_path, plan_path, protocol_path, output, *, resume=False):
    """Evaluate fixed source candidates and freeze selections, without final scoring."""
    output = Path(output)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    config = _settings(protocol)
    candidates = source_candidate_grid()
    candidate_config = protocol.get("candidates", {})
    expected_grid = candidate_config.get("source_grid_sha256")
    if expected_grid is not None and expected_grid != object_digest(candidates):
        raise ValueError("Source grid differs from frozen protocol")
    if candidate_config.get("source_count", len(candidates)) != len(candidates):
        raise ValueError("Source candidate count differs from frozen protocol")
    manifest = source_manifest(input_path, plan_path, protocol_path, protocol, candidates)
    manifest_path = output / "source_run_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError("Use a fresh source-search directory, or resume its verified checkpoints")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume rejected: source inputs, code, protocol or runtime changed")
        if (output / "source_status.json").exists():
            status = json.loads((output / "source_status.json").read_text(encoding="utf-8"))
            if status.get("status") == "completed":
                print(json.dumps({"stage": "source_search_already_completed"}), flush=True)
                return json.loads((output / "source_selection_lock.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    write_json(output / "source_candidate_grid.json", candidates)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if str(np.asarray(arrays.get("dataset_version", ""))) != protocol["dataset_version"]:
        raise ValueError("Input dataset version differs from protocol")
    roles, _, preparation = prepare_holdout(arrays, plan,
        frequency_min=protocol["frequency"]["lower_inclusive"],
        frequency_max=protocol["frequency"]["upper_inclusive"])
    if preparation["feature_count"] != protocol["frequency"]["expected_points"]:
        raise ValueError("Candidate frequency-grid point count differs from protocol")
    expected_counts = protocol.get("fixed_role_counts")
    if expected_counts and {name: len(item["record_ids"]) for name,item in roles.items()} != expected_counts:
        raise ValueError("Fixed research-role counts differ from protocol")
    training = roles["training"]
    write_json(output / "source_preparation.json", preparation)
    partitions = development_episodes(training, config["episodes"], config["split_seed"])
    partition_records = [{role: training["record_ids"][rows].tolist() for role,rows in partition.items()}
                         for partition in partitions]
    partition_path = output / "development_partitions.json"
    if partition_path.exists() and json.loads(partition_path.read_text(encoding="utf-8")) != partition_records:
        raise ValueError("Source development partitions changed")
    write_json(partition_path, partition_records)
    all_metadata, fits_now, reused = [], 0, 0
    for episode, partition in enumerate(partitions):
        for candidate in candidates:
            meta_path = source_metadata_path(output, candidate["id"], episode)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["candidate"] != candidate or meta["partition"] != partition_records[episode]:
                    raise ValueError("Source checkpoint assignment or candidate changed")
                if meta["status"] == "pass":
                    if (file_digest(source_model_path(output, candidate["id"], episode)) != meta["model_sha256"]
                            or file_digest(source_model_path(output, candidate["id"], episode).with_suffix(".predictions.csv")) != meta["predictions_sha256"]):
                        raise ValueError("Source checkpoint hash mismatch")
                reused += 1
            else:
                meta = evaluate_source_episode(training, partition, candidate, output, episode,
                    config["selection_episodes"], config["model_seed"])
                fits_now += meta["source_fit_calls"]
            all_metadata.append(meta)
        print(json.dumps({"stage": "source_development", "completed_episode": episode+1,
                          "episodes": len(partitions), "candidate_count": len(candidates),
                          "failures_so_far": sum(m["status"] != "pass" for m in all_metadata)}), flush=True)
    aggregate = aggregate_source_results(all_metadata, candidates, config["selection_episodes"])
    write_json(output / "source_development_aggregate.json", aggregate)
    selection_settings = {key: config[key] for key in
        ("minimum_accuracy_gain", "max_macro_f1_loss", "max_balanced_accuracy_loss", "tie_tolerance")}
    selected = select_sources(aggregate, candidates, **selection_settings)
    lock = {**selected, "source_run_manifest_sha256": file_digest(manifest_path),
            "source_grid_sha256": object_digest(candidates),
            "partitions_sha256": file_digest(partition_path),
            "development_aggregate_sha256": file_digest(output / "source_development_aggregate.json"),
            "source_fits_recorded": sum(m["source_fit_calls"] for m in all_metadata),
            "source_successes": sum(m["status"] == "pass" for m in all_metadata),
            "source_failures": sum(m["status"] != "pass" for m in all_metadata),
            "created_before_final_query_scoring": True,
            "final_query_scored": False}
    lock_path = output / "source_selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Source selection lock changed")
    write_json(lock_path, lock)
    write_json(output / "source_selection_lock_sha256.json", {"sha256": file_digest(lock_path)})
    write_json(output / "source_status.json", {"status": "completed", "source_fits_recorded": lock["source_fits_recorded"],
        "source_fits_this_invocation": fits_now, "checkpoints_reused": reused,
        "source_successes": lock["source_successes"], "source_failures": lock["source_failures"],
        "final_query_scored": False, "source_selection_lock_sha256": file_digest(lock_path)})
    print(json.dumps({"stage": "source_selection_locked", "selected": lock["selected"],
                      "adapter_source_ids": [c["id"] for c in lock["candidates_for_adaptation"]],
                      "sha256": file_digest(lock_path)}), flush=True)
    return lock
