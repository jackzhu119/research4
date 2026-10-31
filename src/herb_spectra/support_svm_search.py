"""Source-frozen preprocessing with support-weighted SVM parameter adaptation.

This is a newly fitted SVM on a copy, not a score-only frozen-classifier
calibrator. Source gamma, class weights and preprocessing stay fixed; a
zero-support refit is required to reproduce the original SVM scores.
"""
from collections import defaultdict
import csv
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from sklearn.svm import SVC

from .calibration_exploration import development_episodes, encoded, fast_metrics, nested_draws
from .latent_calibration_search import validate_source_run
from .spectral_source_search import load_source_model, source_model_path
from .within_batch_holdout import prepare_holdout


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
SOURCE_ID = "s010_snv_svc"
PREDICTION_FIELDS = ["episode", "stage", "repeat", "method", "candidate_id", "record_id",
                     "y_true", "y_predicted", "is_correct", "support_labels_used"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def candidate_grid():
    candidates = [{"id": "w000_identity", "family": "identity", "support_weight": 0., "blend": 0.}]
    for weight in (.25, 1., 4., 16., 64.):
        for blend in (.1, .25, .5, 1.):
            candidates.append({"id": f"w{len(candidates):03d}_support_svm", "family": "support_svm",
                               "support_weight": weight, "blend": blend})
    return candidates


@dataclass(frozen=True)
class RefitContext:
    source_features: np.ndarray
    source_labels: np.ndarray
    classes: np.ndarray
    parameters: dict


def _matrix(values, columns=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or not len(values) or not np.isfinite(values).all()
            or (columns is not None and values.shape[1] != columns)):
        raise ValueError("A finite nonempty source-transformed feature matrix is required")
    return values


def build_context(source_model, features, labels):
    """Source arrays alone determine hyperparameters and fixed class weights."""
    values = _matrix(features).copy()
    y = np.asarray(labels).copy()
    classes = np.asarray(source_model.classes_).copy()
    estimator = source_model.estimator_
    if not isinstance(estimator, SVC) or estimator.kernel != "rbf":
        raise ValueError("Support refit needs the declared RBF-SVC source")
    if y.shape != (len(values),) or set(y.tolist()) != set(classes.tolist()):
        raise ValueError("One known source class per record, covering all source classes, is required")
    class_weights = {str(c): float(len(y)/(len(classes)*np.count_nonzero(y == c))) for c in classes}
    if not np.allclose([class_weights[str(c)] for c in classes], estimator.class_weight_, rtol=0, atol=1e-14):
        raise ValueError("Source labels do not reproduce the fitted balanced class weights")
    parameters = {"C": float(estimator.C), "kernel": "rbf", "gamma": float(estimator._gamma),
        "class_weight": class_weights, "break_ties": bool(estimator.break_ties),
        "decision_function_shape": estimator.decision_function_shape, "tol": float(estimator.tol),
        "shrinking": bool(estimator.shrinking), "cache_size": float(estimator.cache_size),
        "max_iter": int(estimator.max_iter), "random_state": estimator.random_state,
        "probability": False}
    for array in (values, y, classes):
        array.setflags(write=False)
    return RefitContext(values, y, classes, parameters)


def fit_weighted_support(context, support_features=None, support_labels=None, *, support_weight=1.):
    """Fit a new SVM; query features, labels and IDs are not accepted."""
    if not np.isfinite(support_weight) or support_weight <= 0:
        raise ValueError("The support sample weight must be positive and finite")
    features, labels = context.source_features, context.source_labels
    weights = np.ones(len(features))
    count = 0
    if support_features is not None or support_labels is not None:
        if support_features is None or support_labels is None:
            raise ValueError("Both support features and labels are required")
        target = _matrix(support_features, features.shape[1])
        target_labels = np.asarray(support_labels)
        if (target_labels.shape != (len(target),)
                or set(target_labels.tolist()) != set(context.classes.tolist())):
            raise ValueError("Support requires one known label per row and all seven source classes")
        features = np.concatenate([features, target])
        labels = np.concatenate([labels, target_labels])
        weights = np.concatenate([weights, np.full(len(target), float(support_weight))])
        count = len(target)
    model = SVC(**context.parameters)
    model.fit(features, labels, sample_weight=weights)
    if not np.array_equal(model.classes_, context.classes):
        raise AssertionError("Refit changed the source class score order")
    return model, {"source_sample_weight": 1., "support_sample_weight": float(support_weight),
        "support_labels_used": count, "fixed_gamma": context.parameters["gamma"],
        "fixed_source_class_weight": context.parameters["class_weight"],
        "new_parameter_fit": True, "query_labels_used_for_fit": False}


def mixed_scores(original_scores, refitted_scores, blend):
    original, refitted = np.asarray(original_scores), np.asarray(refitted_scores)
    if (original.ndim != 2 or original.shape != refitted.shape or not np.isfinite(original).all()
            or not np.isfinite(refitted).all() or not np.isfinite(blend) or not 0 <= blend <= 1):
        raise ValueError("Finite class-aligned scores and a blend in[0,1] are required")
    return (1-float(blend))*original + float(blend)*refitted


def _complexity(candidate):
    return (candidate["blend"], candidate["support_weight"], candidate["id"])


def select_candidate(aggregate, candidates, *, minimum_accuracy_gain=.001,
                     max_macro_f1_loss=.005, max_balanced_accuracy_loss=.005,
                     tie_tolerance=.001):
    lookup = {c["id"]: c for c in candidates}
    identity = candidates[0]

    def passes(identifier, stage):
        value, base = aggregate[stage][identifier], aggregate[stage][identity["id"]]
        return (value["evaluations"] == base["evaluations"] and not value["failures"]
            and value["accuracy"]-base["accuracy"] > minimum_accuracy_gain
            and value["macro_f1"]-base["macro_f1"] >= -max_macro_f1_loss
            and value["balanced_accuracy"]-base["balanced_accuracy"] >= -max_balanced_accuracy_loss
            and value["accuracy"]-value["matched_accuracy"] > minimum_accuracy_gain
            and value["macro_f1"]-value["matched_macro_f1"] >= -max_macro_f1_loss
            and value["balanced_accuracy"]-value["matched_balanced_accuracy"] >= -max_balanced_accuracy_loss)

    def best(ids):
        top = max(aggregate["selection"][i]["accuracy"] for i in ids)
        return min((lookup[i] for i in ids if aggregate["selection"][i]["accuracy"] >= top-tie_tolerance),
                   key=_complexity)

    available = [c["id"] for c in candidates if c["family"] != "identity"
                 and not aggregate["selection"][c["id"]]["failures"]]
    eligible = [c["id"] for c in candidates if passes(c["id"], "selection")]
    proposed = best(eligible) if eligible else identity
    selected = proposed if proposed["family"] != "identity" and passes(proposed["id"], "gate") else identity
    diagnostic = best(available) if available else identity
    return {"proposed": proposed, "selected": selected, "diagnostic": diagnostic,
        "eligible_selection_candidates": len(eligible), "fallback_to_identity": selected["family"] == "identity",
        "proposed_selection": aggregate["selection"][proposed["id"]],
        "proposed_gate": aggregate["gate"][proposed["id"]],
        "diagnostic_selection": aggregate["selection"][diagnostic["id"]],
        "diagnostic_gate": aggregate["gate"][diagnostic["id"]],
        "selected_using_final_query": False}


def aggregate_development(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["candidate_id"])].append(row)
    result = {"selection": {}, "gate": {}}
    for (stage, identifier), values in grouped.items():
        result[stage][identifier] = {key: float(np.mean([r[key] for r in values]))
            for key in (*METRICS, *("matched_"+m for m in METRICS))}
        result[stage][identifier].update(evaluations=len(values), failures=0)
    return result


def _record_predictions(writer, identifiers, truth, predicted, classes, *, episode, stage,
                        repeat, method, candidate, support_count):
    for identifier, actual, guessed in zip(identifiers, truth, predicted):
        writer.writerow({"episode": episode, "stage": stage, "repeat": repeat,
            "method": method, "candidate_id": candidate["id"], "record_id": str(identifier),
            "y_true": str(classes[actual]), "y_predicted": str(classes[guessed]),
            "is_correct": int(actual == guessed), "support_labels_used": support_count})


def evaluate_episode(source_model, source, support_pool, prediction, truth, candidates,
                     *, episode, stage, repeats, support_seed, directory, methods=None,
                     save_first_models=False):
    """No target truth reaches transforms or refitting; it is joined for scoring."""
    if "y" in prediction:
        raise ValueError("Prediction payload must not contain target labels")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fingerprint = joblib.hash(source_model)
    source_z = source_model.transform(source["X_alpha"])
    support_z = source_model.transform(support_pool["X_alpha"])
    query_z = source_model.transform(prediction["X_alpha"])
    original_scores = source_model.predict_scores(prediction["X_alpha"])
    context = build_context(source_model, source_z, source["y"])
    context_before = joblib.hash(context)
    zero, zero_meta = fit_weighted_support(context)
    zero_scores = zero.decision_function(query_z)
    max_diff = float(np.max(np.abs(zero_scores-original_scores)))
    if not np.allclose(zero_scores, original_scores, rtol=0, atol=1e-9):
        raise AssertionError(f"Zero-support refit did not reproduce original scores: {max_diff}")
    classes = context.classes
    identifiers = prediction["record_ids"]
    y = encoded([truth[str(i)] for i in identifiers], classes)
    base_pred = np.argmax(original_scores, axis=1)
    base_metrics = fast_metrics(y, base_pred, len(classes), detailed=True)
    zero_pred = np.argmax(zero_scores, axis=1)
    zero_metrics = fast_metrics(y, zero_pred, len(classes), detailed=True)
    draws = nested_draws(support_pool["y"], classes, [1], repeats, support_seed)
    metrics, support_records, artifacts = [], [], []
    refit_count = 1
    if methods is None:
        methods = {c["id"]: c for c in candidates}
    needed = sorted({c["support_weight"] for c in methods.values() if c["family"] != "identity"})
    path = directory/"predictions.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n"); writer.writeheader()
        for draw in draws:
            indices = draw["indices"]
            support_records.append({"repeat": draw["repeat"], "support_ids": support_pool["record_ids"][indices].tolist()})
            fitted_scores = {}
            for weight in needed:
                model, _ = fit_weighted_support(context, support_z[indices], support_pool["y"][indices], support_weight=weight)
                fitted_scores[weight] = model.decision_function(query_z)
                refit_count += 1
                if save_first_models and draw["repeat"] == 0:
                    artifact = directory/f"adapted_draw000_weight{weight:g}.joblib"
                    joblib.dump(model, artifact, compress=3)
                    artifacts.append({"path": str(artifact), "sha256": digest(artifact), "support_weight": weight})
            for method, candidate in methods.items():
                is_zero = method == "matched_zero"
                is_source = candidate["family"] == "identity"
                scores = (zero_scores if is_zero else original_scores if is_source else
                    mixed_scores(original_scores, fitted_scores[candidate["support_weight"]], candidate["blend"]))
                pred = np.argmax(scores, axis=1)
                value = fast_metrics(y, pred, len(classes), detailed=stage == "final")
                count = 0 if is_zero or is_source else len(indices)
                metrics.append({"episode": episode, "stage": stage, "repeat": draw["repeat"],
                    "method": method, "candidate_id": candidate["id"], "support_labels_used": count,
                    **value, **{"matched_"+m: zero_metrics[m] for m in METRICS},
                    **{m+"_gain_vs_source": value[m]-base_metrics[m] for m in METRICS},
                    "accuracy_gain_vs_matched": value["accuracy"]-zero_metrics["accuracy"],
                    "corrected_errors": int(np.count_nonzero((base_pred != y) & (pred == y))),
                    "new_errors": int(np.count_nonzero((base_pred == y) & (pred != y)))})
                _record_predictions(writer, identifiers, y, pred, classes, episode=episode, stage=stage,
                    repeat=draw["repeat"], method=method, candidate=candidate, support_count=count)
    if joblib.hash(source_model) != fingerprint or joblib.hash(context) != context_before:
        raise AssertionError("Original source model or frozen refit context changed")
    metadata = {"episode": episode, "stage": stage, "class_order": classes.tolist(),
        "source_ids": source["record_ids"].tolist(), "support_pool_ids": support_pool["record_ids"].tolist(),
        "query_ids": identifiers.tolist(), "draws": support_records,
        "source_fingerprint_before": fingerprint, "source_fingerprint_after": joblib.hash(source_model),
        "context_fingerprint_before": context_before, "context_fingerprint_after": joblib.hash(context),
        "zero_refit_max_score_difference": max_diff, "matched_zero_metadata": zero_meta,
        "source_baseline_metrics": base_metrics, "matched_zero_metrics": zero_metrics,
        "fixed_parameters": context.parameters, "new_source_preprocessing_fits": 0,
        "new_parameter_refits": refit_count, "predictions_sha256": digest(path),
        "final_query_labels_used_for_fit": False, "final_query_statistics_fitted": False,
        "saved_first_draw_adapted_models": artifacts}
    write_json(directory/"metrics.json", metrics)
    metadata["metrics_sha256"] = digest(directory/"metrics.json")
    write_json(directory/"metadata.json", metadata)
    return metrics, metadata


def summarize_final(cases):
    grouped = defaultdict(list)
    for row in cases:
        grouped[row["method"]].append(row)
    result = []
    for method, rows in grouped.items():
        summary = {"method": method, "candidate_id": rows[0]["candidate_id"], "evaluations": len(rows)}
        for key in METRICS:
            values = np.asarray([r[key] for r in rows])
            summary.update({key+"_"+stat: float(operation(values)) for stat, operation in
                [("mean", np.mean), ("sd", np.std), ("min", np.min), ("max", np.max)]})
            summary[key+"_gain_vs_source_mean"] = float(np.mean([r[key+"_gain_vs_source"] for r in rows]))
        summary.update(accuracy_gain_vs_matched_mean=float(np.mean([r["accuracy_gain_vs_matched"] for r in rows])),
            positive_draws=sum(r["accuracy_gain_vs_source"] > 0 for r in rows),
            unchanged_draws=sum(r["accuracy_gain_vs_source"] == 0 for r in rows),
            negative_draws=sum(r["accuracy_gain_vs_source"] < 0 for r in rows),
            corrected_mean=float(np.mean([r["corrected_errors"] for r in rows])),
            new_errors_mean=float(np.mean([r["new_errors"] for r in rows])))
        result.append(summary)
    return result


def make_manifest(input_path, plan_path, protocol_path, source_run, final_source):
    root = Path(__file__).resolve().parents[2]
    code = ["src/herb_spectra/support_svm_search.py", "scripts/run_support_svm_search.py",
            "src/herb_spectra/spectral_models.py", "src/herb_spectra/spectral_source_search.py",
            "src/herb_spectra/latent_calibration_search.py",
            "src/herb_spectra/calibration_exploration.py", "src/herb_spectra/within_batch_holdout.py"]
    return {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
        "protocol_sha256": digest(protocol_path), "grid_sha256": object_digest(candidate_grid()),
        "source_manifest_sha256": digest(Path(source_run)/"source_run_manifest.json"),
        "source_selection_lock_sha256": digest(Path(source_run)/"source_selection_lock.json"),
        "final_source_sha256": digest(final_source), "source_sha256": {p: digest(root/p) for p in code},
        "python": platform.python_version(), "versions": {n: importlib.metadata.version(n) for n in
            ["numpy", "scikit-learn", "scipy", "joblib"]}}


def run_support_svm_search(input_path, plan_path, protocol_path, source_run, final_source, output, *,
                           resume=False, stage="development"):
    if stage not in {"development", "final"}:
        raise ValueError("Run development first, then explicitly request final after all round locks")
    start = time.monotonic()
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    grid = candidate_grid()
    if protocol["grid_sha256"] != object_digest(grid) or protocol["candidate_count"] != len(grid):
        raise ValueError("Candidate grid differs from the frozen protocol")
    validate_source_run(source_run, input_path, plan_path, protocol["source_protocol"])
    output = Path(output)
    manifest = make_manifest(input_path, plan_path, protocol_path, source_run, final_source)
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"run_manifest.json").exists():
            raise FileExistsError("Use a fresh output or verified --resume")
        if json.loads((output/"run_manifest.json").read_text()) != manifest:
            raise ValueError("Resume rejected changed input, protocol, code, cache or runtime")
        if (output/"status.json").exists() and json.loads((output/"status.json").read_text())["status"] == "completed":
            status = json.loads((output/"status.json").read_text())
            if any(digest(output/p) != sha for p, sha in status["result_sha256"].items()):
                raise ValueError("Completed evidence changed")
            print(json.dumps({"stage": "support_svm_already_completed"}), flush=True)
            return json.loads((output/"summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if not (output/"run_manifest.json").exists():
        write_json(output/"run_manifest.json", manifest)
    write_json(output/"candidate_grid.json", grid)
    write_json(output/"protocol_snapshot.json", protocol)
    with np.load(input_path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    roles, outer_truth, preparation = prepare_holdout(arrays, plan, frequency_min=.1, frequency_max=1.5)
    if (preparation["feature_count"] != 612 or str(arrays["dataset_version"]) != protocol["dataset_version"]
            or {k: len(v["record_ids"]) for k, v in roles.items()} != protocol["role_counts"]):
        raise ValueError("Source cohort, fixed band or role counts changed")
    write_json(output/"preparation.json", preparation)
    training = roles["training"]
    partitions = development_episodes(training, protocol["development"]["episodes"], protocol["development"]["split_seed"])
    cache_records = json.loads((Path(source_run)/"development_partitions.json").read_text())
    expected_records = [{key: training["record_ids"][i].tolist() for key, i in p.items()} for p in partitions]
    if expected_records != cache_records:
        raise ValueError("Source-cache development partitions differ")
    write_json(output/"development_partitions.json", expected_records)
    all_development, metadata = [], []
    config = protocol["development"]
    for episode, partition in enumerate(partitions):
        directory = output/"development"/f"episode_{episode:02d}"
        if resume and (directory/"metadata.json").exists():
            meta = json.loads((directory/"metadata.json").read_text())
            if (digest(directory/"predictions.csv.gz") != meta["predictions_sha256"]
                    or digest(directory/"metrics.json") != meta["metrics_sha256"]):
                raise ValueError("Resumed prediction evidence changed")
            cases = json.loads((directory/"metrics.json").read_text())
        else:
            source_model = load_source_model(source_run, SOURCE_ID, episode)
            source = {key: training[key][partition["source"]] for key in ("X_alpha", "y", "record_ids")}
            pool = {key: training[key][partition["support_pool"]] for key in ("X_alpha", "y", "record_ids")}
            prediction = {key: training[key][partition["query"]] for key in ("X_alpha", "record_ids")}
            truth = dict(zip(training["record_ids"][partition["query"]], training["y"][partition["query"]]))
            cases, meta = evaluate_episode(source_model, source, pool, prediction, truth, grid,
                episode=episode, stage="selection" if episode < config["selection_episodes"] else "gate",
                repeats=config["support_draws"], support_seed=config["support_seed"]+episode, directory=directory)
            meta["source_model_sha256"] = digest(source_model_path(source_run, SOURCE_ID, episode))
            write_json(directory/"metadata.json", meta)
        all_development.extend(cases); metadata.append(meta)
        print(json.dumps({"stage": "support_svm_development", "episode": episode,
            "evaluations": len(cases), "new_parameter_refits": meta["new_parameter_refits"]}), flush=True)
    aggregate = aggregate_development(all_development)
    choice = select_candidate(aggregate, grid, **protocol["selection"])
    lock = {"choice": choice, "source_only_development": True,
        "development_evaluations": len(all_development), "manifest_sha256": digest(output/"run_manifest.json"),
        "grid_sha256": object_digest(grid), "final_query_not_scored_at_lock": True}
    write_json(output/"all_development_metrics.json", all_development)
    write_json(output/"development_aggregate.json", aggregate)
    if (output/"selection_lock.json").exists() and json.loads((output/"selection_lock.json").read_text()) != lock:
        raise ValueError("Resumed development selection lock changed")
    write_json(output/"selection_lock.json", lock)
    lock_digest = digest(output/"selection_lock.json")
    print(json.dumps({"stage": "support_svm_choice_locked", "sha256": lock_digest, "choice": choice["selected"]}), flush=True)
    if stage == "development":
        write_json(output/"status.json", {"status": "development_completed", "final_query_scored": False,
            "development_evaluations": len(all_development), "selection_lock_sha256": lock_digest,
            "development_prediction_rows": len(all_development)*103,
            "new_source_preprocessing_fits": 0, "new_source_model_fits": 0,
            "new_parameter_refits": sum(m["new_parameter_refits"] for m in metadata),
            "candidate_failures": 0, "seconds": time.monotonic()-start})
        return {"stage": "development_completed", "choice": choice, "selection_lock_sha256": lock_digest}
    final_model = joblib.load(final_source)
    original_meta = json.loads(Path(final_source).with_name("metadata.json").read_text())
    if (digest(final_source) != original_meta["source_model_sha256"]
            or original_meta["source_ids"] != training["record_ids"].tolist()):
        raise ValueError("Final source model provenance differs from513 training role")
    artifact = output/"final"/"source_model.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        artifact.write_bytes(Path(final_source).read_bytes())
    if digest(artifact) != digest(final_source):
        raise ValueError("Archived final source differs")
    methods = {"source_original": grid[0], "matched_zero": grid[0],
        "development_selected": choice["selected"], "family_support_svm": choice["diagnostic"]}
    cases, final_meta = evaluate_episode(final_model, roles["training"], roles["calibration"], roles["prediction"],
        outer_truth, grid, episode=-1, stage="final", repeats=protocol["final"]["support_draws"],
        support_seed=protocol["final"]["support_seed"], directory=output/"final", methods=methods, save_first_models=True)
    final_meta.update(source_model_sha256=digest(artifact), selection_lock_sha256=lock_digest)
    write_json(output/"final"/"metadata.json", final_meta)
    summary = summarize_final(cases)
    write_json(output/"summary.json", summary)
    rng = np.random.default_rng(protocol["final"]["random_query_seed"])
    random_ids = rng.choice(np.asarray(sorted(outer_truth)), protocol["final"]["random_query_examples"], replace=False).tolist()
    write_json(output/"random_query_ids.json", random_ids)
    with gzip.open(output/"final"/"predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        random_rows = [r for r in csv.DictReader(handle) if int(r["repeat"]) == 0 and r["record_id"] in random_ids]
    with (output/"random_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(random_rows)
    write_json(output/"status.json", {"status": "completed", "development_evaluations": len(all_development),
        "final_evaluations": len(cases), "development_prediction_rows": len(all_development)*103,
        "final_prediction_rows": len(cases)*171, "development_source_cache_loads": len(partitions),
        "new_source_preprocessing_fits": 0, "new_source_model_fits": 0,
        "new_parameter_refits": sum(m["new_parameter_refits"] for m in metadata)+final_meta["new_parameter_refits"],
        "candidate_failures": 0, "selection_lock_sha256": lock_digest,
        "seconds_this_invocation": time.monotonic()-start, "retrospective_same_batch": True,
        "result_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p != output/"status.json" and p.suffix != ".partial"}})
    return summary
