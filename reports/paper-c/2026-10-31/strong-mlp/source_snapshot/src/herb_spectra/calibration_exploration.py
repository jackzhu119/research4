"""Source-only development and locked retrospective small-support evaluation.

The final prediction labels never enter source fitting, adapter fitting, or
method selection. Repeated development splits are overlapping engineering
tasks, not new physical batches or independent scientific replications.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
from itertools import permutations
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np

from .classification import build_pipeline, class_scores
from .conservative_calibration import build_context, candidate_grid, fit_candidate
from .within_batch_holdout import prepare_holdout


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
DEV_METRICS = METRICS + tuple("matched_" + name for name in METRICS)
FAMILY_ORDER = {"identity": 0, "mistake_bias": 1, "mistake_kernel": 2,
                "mistake_ridge": 3, "prototype_shrinkage": 4,
                "damped_anchor": 5, "legacy_anchor": 6}
DEV_FIELDS = ["scope", "selector", "episode", "stage", "shots_per_class", "repeat",
              "candidate_id", "family", *DEV_METRICS, "status", "error"]
FINAL_FIELDS = ["scope", "selector", "shots_per_class", "repeat", "method",
                "candidate_id", "record_id", "y_true", "y_predicted", "is_correct"]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def fast_metrics(encoded_truth, encoded_prediction, n_classes, detailed=False):
    """Confusion-matrix formulas, avoiding repeated estimator metric overhead."""
    truth, prediction = np.asarray(encoded_truth, dtype=int), np.asarray(encoded_prediction, dtype=int)
    if (truth.ndim != 1 or truth.shape != prediction.shape or not len(truth)
            or np.any(truth < 0) or np.any(truth >= n_classes)
            or np.any(prediction < 0) or np.any(prediction >= n_classes)):
        raise ValueError("Invalid class-index vectors for scoring")
    cm = np.bincount(truth*n_classes + prediction,
                     minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    true_count, predicted_count = cm.sum(axis=1), cm.sum(axis=0)
    diagonal = cm.diagonal()
    recall = np.divide(diagonal, true_count, out=np.zeros(n_classes), where=true_count > 0)
    f1 = np.divide(2*diagonal, true_count+predicted_count, out=np.zeros(n_classes),
                   where=(true_count+predicted_count) > 0)
    result = {"accuracy": float(diagonal.sum()/len(truth)),
              "balanced_accuracy": float(recall[true_count > 0].mean()),
              "macro_f1": float(f1.mean())}
    if detailed:
        result.update(confusion_matrix=cm.tolist(), class_recall=recall.tolist(),
                      class_counts=true_count.tolist(), correct=int(diagonal.sum()), total=len(truth))
    return result


def encoded(labels, classes):
    vocabulary = {str(value): index for index, value in enumerate(classes)}
    try:
        return np.asarray([vocabulary[str(value)] for value in labels], dtype=int)
    except KeyError as exc:
        raise ValueError("Unknown class in scoring vocabulary") from exc


def development_episodes(training, count=30, seed=20261103):
    """Freeze source/support/query roles using only the outer training group."""
    labels, identifiers = np.asarray(training["y"]), np.asarray(training["record_ids"])
    classes = np.unique(labels)
    if count < 1 or len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Development needs unique physical records and a positive episode count")
    episodes = []
    six_orders = list(permutations(range(3)))
    for repeat in range(count):
        rng = np.random.default_rng(seed + repeat)
        roles = {name: [] for name in ("source", "support_pool", "query")}
        for label in classes:
            rows = np.flatnonzero(labels == label)
            rows = rows[np.argsort(identifiers[rows])]
            if len(rows) < 3:
                raise ValueError("Each class needs source, support and development query specimens")
            order = rows[list(six_orders[repeat % 6])] if len(rows) == 3 else rng.permutation(rows)
            n_source, n_support = max(1, int(.6*len(rows))), max(1, int(.2*len(rows)))
            if n_source+n_support >= len(rows):
                raise ValueError("Development split leaves no per-class query")
            roles["source"].extend(order[:n_source])
            roles["support_pool"].extend(order[n_source:n_source+n_support])
            roles["query"].extend(order[n_source+n_support:])
        episodes.append({key: np.asarray(sorted(rows), dtype=int) for key, rows in roles.items()})
    return episodes


def nested_draws(labels, classes, shots, repeats=20, seed=20261104):
    labels = np.asarray(labels)
    if repeats < 1 or not shots or len(set(shots)) != len(shots) or min(shots) < 1:
        raise ValueError("Positive repeated shot counts are required")
    pools = {str(label): np.flatnonzero(labels == label) for label in classes}
    if any(len(values) < max(shots) for values in pools.values()):
        raise ValueError("Requested K is not feasible for every class")
    rng = np.random.default_rng(seed)
    result = []
    for repeat in range(repeats):
        ordered = {label: rng.permutation(values) for label, values in pools.items()}
        for k in sorted(shots):
            result.append({"repeat": repeat, "shots_per_class": k,
                           "indices": np.concatenate([ordered[str(label)][:k] for label in classes])})
    return result


class ProjectionCache:
    """Cache pure source-fitted PCA transforms, retaining exact array identities."""
    def __init__(self, context):
        self.context = context
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.context, name)

    def project(self, values):
        key = id(values)
        cached = self.cache.get(key)
        if cached is not None and cached[0] is values:
            return cached[1]
        result = self.context.project(values)
        self.cache[key] = (values, result)
        return result


def fit_fixed_source(features, labels, selector, seed=42):
    pipe = build_pipeline(model="svc", selector=selector,
                          block_sizes=(features.shape[1],), random_state=seed)
    pipe.set_params(preprocess__mode="snv", model__C=10., model__break_ties=True,
                    model__decision_function_shape="ovr")
    pipe.fit(features, labels)
    return pipe


def feature_payload(pipe, features):
    return pipe[:-1].transform(features), class_scores(pipe, features)


def evaluate_development_episode(training, partition, selector, candidates, shots,
                                 repeats, model_seed, support_seed, destination,
                                 scope_id, episode, selection_episodes):
    """One complete checkpoint; fitting is limited to source and support roles."""
    source, pool, query = (partition[name] for name in ("source", "support_pool", "query"))
    pipe = fit_fixed_source(training["X_alpha"][source], training["y"][source], selector, model_seed)
    fingerprint = joblib.hash(pipe)
    source_features, source_scores = feature_payload(pipe, training["X_alpha"][source])
    pool_features, pool_scores = feature_payload(pipe, training["X_alpha"][pool])
    query_features, query_scores = feature_payload(pipe, training["X_alpha"][query])
    classes = np.asarray(pipe.classes_)
    context = ProjectionCache(build_context(source_features, source_scores, training["y"][source],
                                           classes, seed=model_seed))
    draws = nested_draws(training["y"][pool], classes, shots, repeats, support_seed)
    truth = encoded(training["y"][query], classes)
    controls = {}
    for candidate in candidates:
        control = fit_candidate(context, candidate)
        predicted = np.argmax(control.predict_scores(query_features, query_scores), axis=1)
        controls[candidate["id"]] = fast_metrics(truth, predicted, len(classes))
    output = Path(destination)
    temporary = output.with_name(output.name + ".partial")
    failures, calls = [], 0
    started = time.monotonic()
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for draw in draws:
            indices = draw["indices"]
            xs, ss, ys = pool_features[indices], pool_scores[indices], training["y"][pool][indices]
            for candidate in candidates:
                row = {"scope": scope_id, "selector": selector, "episode": episode,
                       "stage": "selection" if episode < selection_episodes else "gate",
                       "shots_per_class": draw["shots_per_class"], "repeat": draw["repeat"],
                       "candidate_id": candidate["id"], "family": candidate["family"],
                       "status": "pass", "error": "",
                       **{"matched_" + name: controls[candidate["id"]][name] for name in METRICS}}
                try:
                    fitted = fit_candidate(context, candidate, xs, ss, ys)
                    prediction = np.argmax(fitted.predict_scores(query_features, query_scores), axis=1)
                    row.update(fast_metrics(truth, prediction, len(classes)))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    row.update({key: "" for key in METRICS})
                    row.update(status="fail", error=f"{type(exc).__name__}: {exc}")
                    failures.append({key: row[key] for key in ("candidate_id", "repeat", "shots_per_class", "error")})
                writer.writerow(row)
                calls += 1
    if joblib.hash(pipe) != fingerprint:
        raise AssertionError("Source pipeline changed during development adaptation")
    temporary.replace(output)
    return {"source_classifier_fits": 1, "candidate_fit_calls": calls,
            "seconds": time.monotonic()-started, "failures": failures,
            "source_fingerprint_unchanged": fingerprint,
            "source_params": {"C": 10., "preprocessing": "snv", "fit_scope": "episode_source_only"},
            "partition": {name: training["record_ids"][rows].tolist() for name, rows in partition.items()},
            "draws": [{"repeat": d["repeat"], "shots_per_class": d["shots_per_class"],
                       "support_ids": training["record_ids"][pool][d["indices"]].tolist()} for d in draws],
            "metrics_file_sha256": digest(output)}


def complexity(candidate):
    return (FAMILY_ORDER[candidate["family"]],
            candidate.get("blend", candidate.get("strength", 0.)),
            -candidate.get("regularization", 0.), -candidate.get("pseudocount", 0.),
            candidate["id"])


def choose_with_gate(stage_stats, candidates, *, minimum_accuracy_gain=.001,
                     max_macro_f1_loss=.005, max_balanced_accuracy_loss=.005,
                     tie_tolerance=.001):
    """Select from development metrics only, then apply a distinct dev gate."""
    lookup = {c["id"]: c for c in candidates}
    identity = next(c for c in candidates if c["family"] == "identity")
    baseline = stage_stats["selection"][identity["id"]]
    gate_baseline = stage_stats["gate"][identity["id"]]

    def allowed(candidate_id, stage):
        value = stage_stats[stage][candidate_id]
        reference = baseline if stage == "selection" else gate_baseline
        return (value.get("failures", 0) == 0 and value["accuracy"]-reference["accuracy"] > minimum_accuracy_gain
                and value["macro_f1"]-reference["macro_f1"] >= -max_macro_f1_loss
                and value["balanced_accuracy"]-reference["balanced_accuracy"] >= -max_balanced_accuracy_loss
                and value["accuracy"]-value.get("matched_accuracy", reference["accuracy"]) > minimum_accuracy_gain
                and value["macro_f1"]-value.get("matched_macro_f1", reference["macro_f1"]) >= -max_macro_f1_loss
                and value["balanced_accuracy"]-value.get("matched_balanced_accuracy", reference["balanced_accuracy"]) >= -max_balanced_accuracy_loss)

    def best(ids):
        top = max(stage_stats["selection"][key]["accuracy"] for key in ids)
        near = [lookup[key] for key in ids if stage_stats["selection"][key]["accuracy"] >= top-tie_tolerance]
        return min(near, key=complexity)

    eligible = [c["id"] for c in candidates if allowed(c["id"], "selection")]
    proposed = best(eligible) if eligible else identity
    deployed = proposed if proposed["family"] != "identity" and allowed(proposed["id"], "gate") else identity
    family_winners = {}
    for family in sorted({c["family"] for c in candidates}):
        available = [c["id"] for c in candidates if c["family"] == family
                     and stage_stats["selection"][c["id"]].get("failures", 0) == 0]
        if not available:
            family_winners[family] = {"status": "failed", "candidate": identity, "passes_gate": False}
            continue
        winner = best(available)
        family_winners[family] = {"status": "selected_on_development", "candidate": winner,
            "passes_gate": allowed(winner["id"], "gate"),
            "selection": stage_stats["selection"][winner["id"]],
            "gate": stage_stats["gate"][winner["id"]]}
    return {"selected": deployed, "proposed": proposed,
            "fallback_to_identity": deployed["family"] == "identity",
            "selection_metrics": stage_stats["selection"][proposed["id"]],
            "gate_metrics": stage_stats["gate"][proposed["id"]],
            "baseline_selection": baseline, "baseline_gate": gate_baseline,
            "family_winners": family_winners,
            "selection_scope": "outer_training_development_only; no final prediction labels"}


def aggregate_development(paths):
    accumulator = defaultdict(lambda: {"sums": np.zeros(len(DEV_METRICS)), "count": 0, "failures": 0})
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["shots_per_class"]), row["stage"], row["candidate_id"])
                entry = accumulator[key]
                if row["status"] == "pass":
                    entry["sums"] += np.asarray([float(row[name]) for name in DEV_METRICS])
                    entry["count"] += 1
                else:
                    entry["failures"] += 1
    result = defaultdict(lambda: {"selection": {}, "gate": {}})
    for (k, stage, candidate), values in accumulator.items():
        means = values["sums"] / max(1, values["count"])
        result[k][stage][candidate] = {**dict(zip(DEV_METRICS, means.tolist())),
            "evaluations": values["count"], "failures": values["failures"]}
    return dict(result)


def filter_scope(roles, truth, scope):
    excluded = set(scope.get("exclude_groups", []))
    result = {}
    for role, arrays in roles.items():
        labels = arrays["y"] if role != "prediction" else np.asarray([truth[str(v)] for v in arrays["record_ids"]])
        keep = np.flatnonzero([str(value) not in excluded for value in labels])
        result[role] = {key: (value[keep].copy() if key != "axis" else value.copy())
                        for key, value in arrays.items()}
    ids = result["prediction"]["record_ids"].tolist()
    return result, {key: truth[key] for key in ids}


def final_predict_draw(context, candidate, calibration_features, calibration_scores,
                       calibration_y, query_features, query_scores, indices=None):
    """This prediction boundary has no query-label or query-ID input."""
    if indices is None:
        fitted = fit_candidate(context, candidate)
    else:
        fitted = fit_candidate(context, candidate, calibration_features[indices],
                               calibration_scores[indices], calibration_y[indices])
    scores = fitted.predict_scores(query_features, query_scores)
    return np.argmax(scores, axis=1), fitted.metadata


def validate_protocol(protocol, candidates):
    """Require agreement with the recorded, finite exploration budget."""
    if (protocol["candidates"]["count"] != len(candidates)
            or protocol["candidates"]["grid_sha256"] != object_digest(candidates)):
        raise ValueError("Candidate grid differs from the recorded protocol")
    if protocol["selection"]["family_rank"] != FAMILY_ORDER:
        raise ValueError("Candidate complexity order differs from protocol")
    source = protocol["source_model"]
    if source["preprocessing"] != "snv" or source["C"] != 10 or protocol["model_seed"] != 42:
        raise ValueError("Source settings differ from the fixed engineering models")
    dev = protocol["development"]
    if dev["selection_episodes"] + dev["gate_episodes"] != dev["episodes"]:
        raise ValueError("Development selection/gate episode counts disagree")
    if not 0 < dev["selection_episodes"] < dev["episodes"]:
        raise ValueError("Separate selection and gating episodes are required")


def _source_code_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "scripts/run_calibration_optimization.py",
             Path(__file__).with_name("conservative_calibration.py"),
             Path(__file__).with_name("classification.py"),
             Path(__file__).with_name("selection.py"),
             Path(__file__).with_name("benchmark.py"),
             Path(__file__).with_name("calibration.py"),
             Path(__file__).with_name("within_batch_holdout.py")]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def verify_existing_training_plan(plan_path, reference_plan_path):
    """A pretrained source artifact can only be reused for its original cohort."""
    if digest(plan_path) != digest(reference_plan_path):
        raise ValueError("Frozen source training cohort differs from the original recorded split plan")
    return digest(reference_plan_path)


def make_manifest(input_path, plan_path, protocol_path, protocol, candidates):
    repo = Path(__file__).resolve().parents[2]
    model_manifest = repo / protocol["source_model"]["existing_seven_class_models"]
    model_entries = json.loads(model_manifest.read_text())["models"]
    model_hashes = {entry["path"]: digest(repo / entry["path"]) for entry in model_entries}
    if any(model_hashes[entry["path"]] != entry["sha256"] for entry in model_entries):
        raise ValueError("Existing frozen source model SHA mismatch")
    reference = repo / protocol["source_model"].get("training_plan_reference",
                        "reports/zaoren-threeway/2026-10-31/split_plan_snapshot.json")
    cohort_sha = verify_existing_training_plan(plan_path, reference)
    return {"protocol_id": protocol["protocol_id"],
            "input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
            "protocol_sha256": digest(protocol_path), "grid_sha256": object_digest(candidates),
            "python": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ["numpy", "scipy", "scikit-learn", "joblib"]},
            "frozen_source_manifest_sha256": digest(model_manifest),
            "frozen_source_artifact_sha256": model_hashes,
            "frozen_source_training_plan_sha256": cohort_sha,
            "source_code_sha256": _source_code_hashes()}


def _summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scope"], row["selector"], row["shots_per_class"], row["method"])].append(row)
    result = []
    for (scope, selector, k, method), cases in sorted(grouped.items()):
        value = {"scope": scope, "selector": selector, "shots_per_class": k,
                 "method": method, "candidate_id": cases[0]["candidate_id"], "evaluations": len(cases)}
        for metric in METRICS:
            numbers = np.asarray([row[metric] for row in cases])
            gains = np.asarray([row[metric + "_gain"] for row in cases])
            for name, operation in [("mean", np.mean), ("sd", np.std), ("min", np.min), ("max", np.max)]:
                value[f"{metric}_{name}"] = float(operation(numbers))
            value[metric + "_gain_mean"] = float(gains.mean())
        value["positive_accuracy_draws"] = sum(row["accuracy_gain"] > 0 for row in cases)
        value["negative_accuracy_draws"] = sum(row["accuracy_gain"] < 0 for row in cases)
        value["zero_accuracy_draws"] = sum(row["accuracy_gain"] == 0 for row in cases)
        value["paired_corrected_mean"] = float(np.mean([row["corrected_errors"] for row in cases]))
        value["paired_harmed_mean"] = float(np.mean([row["new_errors"] for row in cases]))
        value["gain_vs_matched_zero_support_mean"] = float(np.mean([row["gain_vs_matched_control"] for row in cases]))
        result.append(value)
    return result


def run_final_scope(roles, truth, scope, selector, choices, candidates, protocol,
                    output, random_ids, repo):
    """Use already locked configurations; join truth only after each prediction."""
    training, calibration, query = (roles[key] for key in ("training", "calibration", "prediction"))
    if "y" in query:
        raise ValueError("Final prediction payload cannot contain truth")
    directory = Path(output) / "final" / scope["id"] / selector
    directory.mkdir(parents=True, exist_ok=True)
    artifact_info = {}
    if not scope.get("exclude_groups"):
        manifest_path = Path(repo) / protocol["source_model"]["existing_seven_class_models"]
        manifest = json.loads(manifest_path.read_text())
        entry = next(value for value in manifest["models"] if value["selector"] == selector)
        artifact = Path(repo) / entry["path"]
        if digest(artifact) != entry["sha256"]:
            raise ValueError("Existing frozen model artifact hash mismatch")
        pipe = joblib.load(artifact)
        artifact_info = {"path": entry["path"], "sha256": entry["sha256"], "new_source_fits": 0}
    else:
        pipe = fit_fixed_source(training["X_alpha"], training["y"], selector, protocol["model_seed"])
        artifact = directory / "source_model.joblib"
        joblib.dump(pipe, artifact, compress=3)
        artifact_info = {"path": str(artifact), "sha256": digest(artifact), "new_source_fits": 1}
    before = joblib.hash(pipe)
    classes = np.asarray(pipe.classes_)
    if set(classes.tolist()) != set(training["y"].tolist()):
        raise ValueError("Final source class vocabulary mismatch")
    zsource, ssource = feature_payload(pipe, training["X_alpha"])
    zcal, scal = feature_payload(pipe, calibration["X_alpha"])
    zquery, squery = feature_payload(pipe, query["X_alpha"])
    context = ProjectionCache(build_context(zsource, ssource, training["y"], classes, seed=protocol["model_seed"]))
    baseline_prediction = np.argmax(squery, axis=1)
    query_ids = query["record_ids"].tolist()
    # Truth is used below only by the independent metric/CSV recording boundary.
    labels = encoded([truth[str(value)] for value in query_ids], classes)
    baseline_metrics = fast_metrics(labels, baseline_prediction, len(classes), detailed=True)
    draws = nested_draws(calibration["y"], classes, scope["shots"],
                         protocol["final"]["support_draws"], protocol["final"]["support_seed"])
    lookup = {candidate["id"]: candidate for candidate in candidates}
    identity = next(c for c in candidates if c["family"] == "identity")
    legacy = next(c for c in candidates if c["family"] == "legacy_anchor")
    case_rows, random_rows, draws_manifest = [], [], []
    all_ids = set(random_ids)
    handles, writers = {}, {}
    for k in scope["shots"]:
        path = directory / f"predictions_k{k}.csv.gz"
        handles[k] = gzip.open(path, "wt", encoding="utf-8", newline="")
        writers[k] = csv.DictWriter(handles[k], fieldnames=FINAL_FIELDS, lineterminator="\n")
        writers[k].writeheader()
    try:
        for draw in draws:
            k, repeat, selected = draw["shots_per_class"], draw["repeat"], draw["indices"]
            choice = choices[str(k)]
            global_candidate = choice["selected"]
            comparison = {"source_identity": identity, "legacy_lambda1": legacy,
                          "development_selected": global_candidate}
            for family, winner in choice["family_winners"].items():
                if family not in {"identity", "legacy_anchor"}:
                    comparison["family_" + family] = winner["candidate"]
            prototype_controls = {candidate["id"]: candidate for candidate in comparison.values()
                                  if candidate["family"] == "prototype_shrinkage"}
            draws_manifest.append({"repeat": repeat, "shots_per_class": k,
                "support_ids": calibration["record_ids"][selected].tolist()})
            prediction_cache, control_cache = {}, {}
            for candidate in {value["id"]: value for value in comparison.values()}.values():
                prediction_cache[candidate["id"]] = final_predict_draw(context, candidate, zcal, scal,
                    calibration["y"], zquery, squery, selected)[0]
                if candidate["family"] == "prototype_shrinkage":
                    control_cache[candidate["id"]] = final_predict_draw(context, candidate, zcal, scal,
                        calibration["y"], zquery, squery, None)[0]
            for candidate_id, candidate in prototype_controls.items():
                comparison["matched_zero_support_" + candidate_id] = candidate
            for method, candidate in comparison.items():
                prediction = (control_cache[candidate["id"]] if method.startswith("matched_zero_support_")
                              else prediction_cache[candidate["id"]])
                metrics = fast_metrics(labels, prediction, len(classes), detailed=True)
                matched = control_cache.get(candidate["id"], baseline_prediction)
                if method.startswith("matched_zero_support_"):
                    matched = prediction
                row = {"scope": scope["id"], "selector": selector, "shots_per_class": k,
                    "repeat": repeat, "method": method, "candidate_id": candidate["id"],
                    **metrics, **{name + "_gain": metrics[name]-baseline_metrics[name] for name in METRICS},
                    "corrected_errors": int(np.count_nonzero((baseline_prediction != labels) & (prediction == labels))),
                    "new_errors": int(np.count_nonzero((baseline_prediction == labels) & (prediction != labels))),
                    "gain_vs_matched_control": float(np.mean(prediction == labels)-np.mean(matched == labels))}
                case_rows.append(row)
                for identifier, actual, predicted in zip(query_ids, labels, prediction):
                    sample = {"scope": scope["id"], "selector": selector, "shots_per_class": k,
                        "repeat": repeat, "method": method, "candidate_id": candidate["id"],
                        "record_id": identifier, "y_true": str(classes[actual]),
                        "y_predicted": str(classes[predicted]), "is_correct": int(actual == predicted)}
                    writers[k].writerow(sample)
                    if identifier in all_ids and method in {"source_identity", "development_selected"} and repeat == 0:
                        random_rows.append(sample)
    finally:
        for handle in handles.values():
            handle.close()
    if joblib.hash(pipe) != before:
        raise AssertionError("Final source model mutated during calibration")
    metadata = {"scope": scope["id"], "selector": selector, "class_order": classes.tolist(),
        "source_sample_ids": training["record_ids"].tolist(), "query_ids": query_ids,
        "source_fingerprint_before": before, "source_fingerprint_after": joblib.hash(pipe),
        "source_artifact": artifact_info, "baseline_metrics": baseline_metrics,
        "draws": draws_manifest, "evaluations": len(case_rows),
        "query_labels_used_to_fit": False, "query_statistics_fitted": False,
        "predictions_files": {f"predictions_k{k}.csv.gz": digest(directory / f"predictions_k{k}.csv.gz") for k in scope["shots"]}}
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "metrics.json", case_rows)
    write_json(directory / "summary.json", _summary_rows(case_rows))
    return case_rows, random_rows, metadata


def run_exploration(input_path, plan_path, protocol_path, output, *, resume=False, repo=None):
    """Finite, checkpointed development; all final choices lock before scoring."""
    output = Path(output)
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    protocol = json.loads(Path(protocol_path).read_text())
    candidates = candidate_grid()
    validate_protocol(protocol, candidates)
    current = make_manifest(input_path, plan_path, protocol_path, protocol, candidates)
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError("Choose a fresh directory or resume a verified run")
        if json.loads(manifest_path.read_text()) != current:
            raise ValueError("Resume refused: input, protocol, grid or source code hashes changed")
        if (output / "status.json").is_file() and json.loads((output / "status.json").read_text()).get("status") == "completed":
            print(json.dumps({"status": "already_completed", "output": str(output)}), flush=True)
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        write_json(manifest_path, current)
    write_json(output / "candidate_grid.json", candidates)
    plan = json.loads(Path(plan_path).read_text())
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    roles, truth, preparation = prepare_holdout(arrays, plan,
        frequency_min=protocol["frequency"]["lower_inclusive"],
        frequency_max=protocol["frequency"]["upper_inclusive"])
    if preparation["feature_count"] != protocol["frequency"]["expected_points"]:
        raise ValueError("Frequency grid differs from recorded protocol")
    write_json(output / "preparation.json", preparation)
    scoped, scoped_truth, examples = {}, {}, {}
    for scope in protocol["scopes"]:
        scoped[scope["id"]], scoped_truth[scope["id"]] = filter_scope(roles, truth, scope)
        if {name: len(value["record_ids"]) for name, value in scoped[scope["id"]].items()} != scope["role_counts"]:
            raise ValueError("Scope record counts differ from recorded protocol")
        ids = np.asarray(sorted(scoped_truth[scope["id"]]))
        rng = np.random.default_rng(protocol["final"]["random_query_seed"])
        examples[scope["id"]] = rng.choice(ids, min(len(ids), protocol["final"]["random_query_examples"]), replace=False).tolist()
    examples_path = output / "random_query_ids.json"
    if examples_path.exists() and json.loads(examples_path.read_text()) != examples:
        raise ValueError("Precommitted random query IDs changed")
    write_json(examples_path, examples)
    choices, aggregate_rows = {}, []
    dev = protocol["development"]
    model_fits, candidate_calls = 0, 0
    for scope in protocol["scopes"]:
        scope_id = scope["id"]
        training = scoped[scope_id]["training"]
        episodes = development_episodes(training, dev["episodes"], dev["split_seed"])
        choices[scope_id] = {}
        for selector in protocol["selectors"]:
            directory = output / "development" / scope_id / selector
            directory.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, partition in enumerate(episodes):
                path = directory / f"episode_{i:02d}.csv.gz"
                meta_path = directory / f"episode_{i:02d}.json"
                if path.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta["metrics_file_sha256"] != digest(path):
                        raise ValueError("Completed development checkpoint hash changed")
                else:
                    meta = evaluate_development_episode(training, partition, selector, candidates,
                        scope["shots"], dev["support_draws"], protocol["model_seed"],
                        dev["support_seed"]+i, path, scope_id, i, dev["selection_episodes"])
                    write_json(meta_path, meta)
                paths.append(path)
                model_fits += meta["source_classifier_fits"]
                candidate_calls += meta["candidate_fit_calls"]
                print(json.dumps({"stage": "development", "scope": scope_id, "selector": selector,
                                  "completed_episode": i+1, "episodes": len(episodes)}, ensure_ascii=False), flush=True)
            stats = aggregate_development(paths)
            choices[scope_id][selector] = {}
            for k, stages in stats.items():
                config = {key: protocol["selection"][key] for key in
                          ("minimum_accuracy_gain", "max_macro_f1_loss", "max_balanced_accuracy_loss", "tie_tolerance")}
                choices[scope_id][selector][str(k)] = choose_with_gate(stages, candidates, **config)
                for stage, values in stages.items():
                    for candidate_id, value in values.items():
                        aggregate_rows.append({"scope": scope_id, "selector": selector,
                            "shots_per_class": k, "stage": stage, "candidate_id": candidate_id, **value})
            print(json.dumps({"stage": "development_model_completed", "scope": scope_id,
                              "selector": selector}, ensure_ascii=False), flush=True)
    write_json(output / "development_aggregate.json", aggregate_rows)
    locked = {"run_manifest_sha256": digest(manifest_path), "candidate_grid_sha256": object_digest(candidates),
        "selection_scope": "source training role only; final labels not accessed by selector",
        "development_source_classifier_fits": model_fits,
        "development_candidate_fit_calls": candidate_calls, "choices": choices}
    lock_path = output / "selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != locked:
        raise ValueError("Locked selections disagree with recomputed development choices")
    write_json(lock_path, locked)
    lock_sha = digest(lock_path)
    write_json(output / "selection_lock_sha256.json", {"sha256": lock_sha,
               "created_before_final_scoring": True})
    print(json.dumps({"stage": "all_choices_locked", "sha256": lock_sha}, ensure_ascii=False), flush=True)
    final_rows, random_rows, final_metadata = [], [], []
    for scope in protocol["scopes"]:
        for selector in protocol["selectors"]:
            if digest(lock_path) != lock_sha:
                raise ValueError("Selection lock changed before final evaluation")
            rows, examples_rows, metadata = run_final_scope(scoped[scope["id"]], scoped_truth[scope["id"]],
                scope, selector, choices[scope["id"]][selector], candidates, protocol, output,
                examples[scope["id"]], repo)
            final_rows.extend(rows); random_rows.extend(examples_rows); final_metadata.append(metadata)
            print(json.dumps({"stage": "final_model_completed", "scope": scope["id"],
                              "selector": selector, "evaluations": len(rows)}, ensure_ascii=False), flush=True)
    summary = _summary_rows(final_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "all_final_metrics.json", final_rows)
    with (output / "random_query_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(random_rows)
    write_json(output / "final_metadata.json", final_metadata)
    write_json(output / "status.json", {"status": "completed", "scope": protocol["scope"],
        "new_development_source_classifier_fits": model_fits,
        "new_final_source_classifier_fits": sum(m["source_artifact"]["new_source_fits"] for m in final_metadata),
        "development_candidate_fit_calls": candidate_calls,
        "final_evaluations": len(final_rows),
        "prediction_rows": sum(row["total"] for row in final_rows),
        "selection_lock_sha256": lock_sha,
        "final_query_used_for_selection": False,
        "repeats_are_independent_batches": False})
    return summary
