"""Independent numerical and provenance audit; never fits a research model."""
from collections import defaultdict
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "outputs/image-guided-20261031"
INPUT = ROOT / "outputs/approved-zaoren-20261031/data-preparation/dataset.npz"
PLAN = ROOT / "data/processed/zaoren_threeway_20261031_v1/split_plan.json"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.max_difference = 0.
        self.notes = []

    def require(self, condition, description):
        self.checks += 1
        if not condition:
            raise AssertionError(description)

    def compare(self, actual, expected, description, tolerance=1e-10):
        if isinstance(actual, dict) and isinstance(expected, dict):
            self.require(set(actual) == set(expected), description + " keys")
            for key in actual:
                self.compare(actual[key], expected[key], description + ":" + str(key), tolerance)
        elif isinstance(actual, (list, tuple, np.ndarray)) and isinstance(expected, (list, tuple, np.ndarray)):
            self.require(len(actual) == len(expected), description + " length")
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.compare(a, e, description + ":" + str(i), tolerance)
        elif isinstance(actual, (int, float, np.integer, np.floating)) and isinstance(expected, (int, float, np.integer, np.floating)):
            delta = abs(float(actual) - float(expected))
            self.max_difference = max(self.max_difference, delta)
            self.require(np.isfinite(delta) and delta <= tolerance, (description, actual, expected))
        else:
            self.require(actual == expected, (description, actual, expected))


def independent_metrics(truth, prediction, classes=7):
    truth, prediction = np.asarray(truth, int), np.asarray(prediction, int)
    cm = np.zeros((classes, classes), dtype=np.int64)
    for actual, guessed in zip(truth, prediction):
        cm[actual, guessed] += 1
    actual_count, guessed_count = cm.sum(1), cm.sum(0)
    correct = cm.diagonal()
    recall = np.divide(correct, actual_count, out=np.zeros(classes), where=actual_count > 0)
    f1 = np.divide(2 * correct, actual_count + guessed_count, out=np.zeros(classes), where=actual_count + guessed_count > 0)
    return {"accuracy": float(correct.sum() / len(truth)),
        "balanced_accuracy": float(recall[actual_count > 0].mean()),
        "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
        "class_recall": recall.tolist(), "class_counts": actual_count.tolist(),
        "correct": int(correct.sum()), "total": len(truth)}


def independent_probability_metrics(probability, truth):
    p = np.asarray(probability, float); truth = np.asarray(truth, int)
    n, classes = p.shape
    target = np.zeros((n, classes)); target[np.arange(n), truth] = 1
    nll = float(np.mean([-np.log(max(p[i, label], 1e-15)) for i, label in enumerate(truth)]))
    brier = float(np.sum((p-target)**2) / n)
    confidence = p.max(1); predicted = p.argmax(1); ece = 0.
    for bucket in range(10):
        lower, upper = bucket/10, (bucket+1)/10
        rows = (confidence >= lower) & (confidence < upper if bucket < 9 else confidence <= upper)
        if rows.any():
            ece += rows.sum()/n * abs(float(np.mean(predicted[rows] == truth[rows])) - float(confidence[rows].mean()))
    return {"nll": nll, "brier": brier, "ece10": float(ece)}


def independent_softmax(scores, temperature=1.):
    values = np.asarray(scores, float)/temperature
    values -= values.max(1, keepdims=True)
    exp = np.exp(values)
    return exp/exp.sum(1, keepdims=True)


def independent_state_fingerprint(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(value.dtype).encode())
        result.update(np.asarray(value.shape, dtype="<i8").tobytes())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def datasets(audit):
    with np.load(INPUT, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    plan = read(PLAN)
    identifiers = arrays["record_ids"].astype(str)
    labels = arrays["y"].astype(str)
    audit.require(len(identifiers) == 855 and len(set(identifiers)) == 855, "855 unique specimens")
    truth = dict(zip(identifiers, labels))
    roles = {r["record_id"]: r["role"] for r in plan["records"]}
    audit.require(set(roles) == set(truth), "split covers input records")
    for r in plan["records"]:
        audit.require(truth[r["record_id"]] == r["origin_group_id"], "plan label agrees with input")
    role_ids = {name: identifiers[[roles[i] == name for i in identifiers]].tolist()
                for name in ("training", "calibration", "prediction")}
    audit.compare([len(role_ids[k]) for k in role_ids], [513, 171, 171], "fixed outer counts")
    training_ids = np.asarray(role_ids["training"]); y = np.asarray([truth[i] for i in training_ids])
    orders = list(permutations(range(3))); partitions = []
    for episode in range(6):
        rng = np.random.default_rng(20261103 + episode)
        part = {k: [] for k in ("source", "support_pool", "query")}
        for label in sorted(set(y)):
            rows = np.flatnonzero(y == label)
            rows = rows[np.argsort(training_ids[rows])]
            order = rows[list(orders[episode])] if len(rows) == 3 else rng.permutation(rows)
            ns, npool = max(1, int(.6*len(rows))), max(1, int(.2*len(rows)))
            part["source"].extend(order[:ns]); part["support_pool"].extend(order[ns:ns+npool]); part["query"].extend(order[ns+npool:])
        partitions.append({k: training_ids[sorted(v)].tolist() for k, v in part.items()})
    axis = arrays["axis"]
    selected = (axis >= .1) & (axis <= 1.5)
    audit.require(int(selected.sum()) == 612, "612 source frequency points")
    return arrays, truth, role_ids, partitions, np.asarray(axis[selected], float)


def check_manifest(audit, path):
    manifest = read(path)
    audit.compare(sha(INPUT), manifest["input_sha256"], str(path) + " input SHA")
    split_digest = manifest.get("plan_sha256", manifest.get("split_plan_sha256"))
    audit.compare(sha(PLAN), split_digest, str(path) + " partition SHA")
    for key, value in manifest.items():
        if isinstance(value, dict):
            for relative, digest in value.items():
                if isinstance(relative, str) and relative.endswith(".py") and isinstance(digest, str) and len(digest) == 64:
                    audit.compare(sha(ROOT / relative), digest, "executed source hash " + relative)
    return manifest


def csv_rows(audit, path, truth, classes):
    mapping = {str(c): i for i, c in enumerate(classes)}
    handle = gzip.open(path, "rt", encoding="utf-8-sig", newline="") if str(path).endswith(".gz") else Path(path).open(encoding="utf-8-sig", newline="")
    with handle:
        result = list(csv.DictReader(handle))
    for row in result:
        identifier = row["record_id"]
        audit.require(identifier in truth, "known input record")
        audit.compare(row["y_true"], truth[identifier], "raw CSV ground truth")
        audit.require(row["y_predicted"] in mapping, "known predicted class")
        audit.compare(int(row["is_correct"]), int(row["y_true"] == row["y_predicted"]), "raw CSV correctness")
    return result


def check_saved_metric(audit, row, expected, description):
    for key, value in expected.items():
        if key in row:
            audit.compare(row[key], value, description + ":" + key)


def audit_feature(audit, truth, roles, parts, axis, final=False):
    run = RUN / ("features-v2" if (RUN / "features-v2").exists() else "features")
    if not (run / "selection_lock.json").exists():
        return {"status": "not_yet_locked"}
    import joblib
    manifest = check_manifest(audit, run / "run_manifest.json")
    audit.compare(sha(run / "protocol_snapshot.json"), manifest["protocol_sha256"], "feature protocol bytes")
    protocol = read(run / "protocol_snapshot.json"); grid = read(run / "candidate_grid.json")
    audit.require(len(grid) == 40, "40 feature candidates")
    audit.compare(read(run / "development_partitions.json"), parts, "feature PCG source partitions")
    classes = sorted(set(truth.values())); class_index = {c: i for i, c in enumerate(classes)}
    development = read(run / "all_development_metrics.json")
    raw_metric_lookup = {}; raw_rows = 0; warnings_count = 0; replayed = 0
    lock = read(run / "selection_lock.json"); selected_id = lock["choice"]["selected"]["id"]
    with np.load(INPUT, allow_pickle=False) as a:
        original_ids = a["record_ids"].astype(str); original_X = a["X_alpha"][:, (a["axis"] >= .1) & (a["axis"] <= 1.5)]
    raw_lookup = {i: row for row, i in enumerate(original_ids)}
    for episode in range(6):
        for seed in (42, 43, 44):
            directory = run / "development" / f"episode_{episode:02d}" / f"seed_{seed}"
            context = read(directory / "context.json")
            audit.compare(context["source_ids"], parts[episode]["source"], "feature context source IDs")
            audit.compare(context["query_ids"], parts[episode]["query"], "feature context query IDs")
            audit.compare(sha(directory / "predictions.csv.gz"), context["predictions_sha256"], "feature raw SHA")
            rows = csv_rows(audit, directory / "predictions.csv.gz", truth, classes); raw_rows += len(rows)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["candidate_id"]].append(row)
            sx = original_X[[raw_lookup[i] for i in context["source_ids"]]]
            for candidate in grid:
                identifier = candidate["id"]
                values = grouped[identifier]
                audit.compare([r["record_id"] for r in values], context["query_ids"], "feature candidate query order")
                expected = independent_metrics([class_index[r["y_true"]] for r in values], [class_index[r["y_predicted"]] for r in values])
                raw_metric_lookup[(episode, seed, identifier)] = expected
                # Check source statistics and selected variables from the exact saved model.
                metadata_path = directory / identifier / "metadata.json"
                if not metadata_path.exists():
                    metadata_path = directory / f"{identifier}.json"
                model_path = directory / identifier / "model.joblib"
                if not model_path.exists():
                    model_path = directory / f"{identifier}.joblib"
                metadata = read(metadata_path)
                audit.compare(sha(model_path), metadata["model_sha256"], "feature checkpoint SHA")
                model = joblib.load(model_path)
                audit.compare(joblib.hash(model), metadata["model_fingerprint"], "feature state fingerprint")
                info = metadata["fit"]
                indices = np.asarray(info["selected_indices"], int)
                audit.require(len(indices) > 0 and len(set(indices.tolist())) == len(indices)
                    and np.all(indices >= 0) and np.all(indices < 612), "valid selected feature indices")
                audit.compare(info["selected_frequencies"], axis[indices], "selected physical frequency coordinates")
                audit.compare(info["selected_features"], len(indices), "effective selected feature count")
                audit.compare(info["source_samples"], len(sx), "feature source sample count")
                audit.compare(info["class_order"], classes, "feature class order")
                audit.compare(model.selected_indices.tolist(), indices.tolist(), "saved selector indices")
                if candidate["preprocessing"] == "msc":
                    reference = sx.mean(0)
                    audit.compare(model.preprocessor.reference_, reference, "MSC source mean reference")
                    centered = reference-reference.mean()
                    slope = ((sx-sx.mean(1, keepdims=True)) @ centered) / (centered @ centered)
                    intercept = sx.mean(1)-slope*reference.mean()
                    transformed = (sx-intercept[:, None])/slope[:, None]
                else:
                    transformed = (sx-sx.mean(1, keepdims=True))/sx.std(1, ddof=1, keepdims=True)
                audit.compare(model.scaler.mean_, transformed.mean(0), "source-only scaler mean")
                audit.compare(model.scaler.var_, transformed.var(0), "source-only scaler variance")
                if candidate["family"] == "cars_svm":
                    policy = info["selector"]["internal_cv"]
                    audit.require(policy["validation_excludes_anchor_classes"], "CARS excludes singleton from inner validation")
                    audit.compare(policy["anchor_classes"], [1], "CARS explicit rare-class index")
                    audit.compare(policy["anchor_source_indices"], [i for i, rid in enumerate(context["source_ids"]) if truth[rid] == "ZY02"], "CARS singleton source anchor")
                warnings_count += len(info.get("warnings", []))
                if identifier in {selected_id, "f001_snv_full_svm", "f021_msc_full_svm"}:
                    query_x = original_X[[raw_lookup[i] for i in context["query_ids"]]]
                    before = joblib.hash(model)
                    guessed = model.predict(query_x)
                    audit.compare(guessed.tolist(), [r["y_predicted"] for r in values], "feature prediction replay")
                    audit.compare(joblib.hash(model), before, "inference keeps source frozen")
                    replayed += 1
    for row in development:
        check_saved_metric(audit, row, raw_metric_lookup[(row["episode"], row["seed"], row["candidate_id"])], "feature development case")
    # Independent Macro-F1-first aggregation and source family selection.
    means = {}
    for candidate in grid:
        for stage in ("selection", "gate"):
            matches = [r for r in development if r["candidate_id"] == candidate["id"] and r["stage"] == stage]
            means[(candidate["id"], stage)] = {"evaluations": len(matches), "failures": 0,
                "selected_features_mean": float(np.mean([r["selected_features"] for r in matches])),
                **{m: float(np.mean([r[m] for r in matches])) for m in METRICS}}
    reference = next(c for c in grid if c["preprocessing"] == "snv" and c["family"] == "full_svm" and c["C"] == 10)
    policy = protocol["selection"]
    def passes(candidate, stage):
        a, b = means[(candidate["id"], stage)], means[(reference["id"], stage)]
        return a["macro_f1"]-b["macro_f1"] > policy["minimum_macro_f1_gain"] and all(
            a[k]-b[k] >= -policy["maximum_accuracy_or_balanced_loss"] for k in ("accuracy", "balanced_accuracy"))
    def rank(options):
        if not options:
            return reference
        maximum = max(means[(c["id"], "selection")]["macro_f1"] for c in options)
        options = [c for c in options if means[(c["id"], "selection")]["macro_f1"] >= maximum-policy["macro_f1_tie_tolerance"]]
        return min(options, key=lambda c: (-means[(c["id"], "selection")]["accuracy"],
            means[(c["id"], "selection")]["selected_features_mean"], c.get("components", 0), c.get("C", 0), c["id"]))
    eligible = [c for c in grid if passes(c, "selection")]
    proposed = rank(eligible) if eligible else reference
    chosen = proposed if eligible and passes(proposed, "gate") else reference
    audit.compare(lock["choice"]["selected"], chosen, "independent feature selected source")
    audit.compare(lock["choice"]["proposed"], proposed, "independent feature proposed source")
    audit.compare(lock["choice"]["eligible_selection_candidates"], len(eligible), "independent feature selection eligibility")
    family_choices = {family: rank([c for c in grid if c["family"] == family]) for family in sorted({c["family"] for c in grid})}
    audit.compare(lock["choice"]["family_diagnostics"], family_choices, "independent feature family choices")
    for row in lock["choice"]["means"]:
        check_saved_metric(audit, row, means[(row["candidate_id"], row["stage"])], "feature mean")
    final_count = 0
    old_run = RUN / "features-v1-checkpoint-issue"
    version_prediction_comparisons = 0
    if run.name == "features-v2" and (old_run / "selection_lock.json").exists():
        audit.compare(read(old_run / "candidate_grid.json"), grid, "v1/v2 source hypothesis grid unchanged")
        audit.compare(read(old_run / "selection_lock.json")["choice"], lock["choice"], "v1/v2 choice unchanged after checkpoint fix")
        for episode in range(6):
            for seed in (42, 43, 44):
                relative = Path("development") / f"episode_{episode:02d}" / f"seed_{seed}"
                with gzip.open(old_run / relative / "predictions.csv.gz", "rb") as h:
                    old_bytes = h.read()
                with gzip.open(run / relative / "predictions.csv.gz", "rb") as h:
                    new_bytes = h.read()
                audit.compare(hashlib.sha256(new_bytes).hexdigest(), hashlib.sha256(old_bytes).hexdigest(), "v1/v2 all source predictions byte-equivalent after decompression")
                for candidate in grid:
                    old = read(old_run / relative / (candidate["id"]+".json"))
                    new = read(run / relative / (candidate["id"]+".json"))
                    audit.compare(new["fit"]["selected_indices"], old["fit"]["selected_indices"], "v1/v2 selected variables unchanged")
                    audit.compare(new["fit"]["selected_frequencies"], old["fit"]["selected_frequencies"], "v1/v2 frequency coordinates unchanged")
                    version_prediction_comparisons += 1
    if final:
        final_lookup = {}
        for seed in (42, 43, 44):
            directory = run / "final" / f"seed_{seed}"
            context = read(directory / "context.json")
            audit.compare(context["source_ids"], roles["training"], "feature final 513 source IDs")
            audit.compare(context["query_ids"], roles["prediction"], "feature final 171 query IDs")
            rows = csv_rows(audit, directory / "predictions.csv.gz", truth, classes); final_count += len(rows)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["candidate_id"]].append(row)
            for candidate_id, values in grouped.items():
                final_lookup[(seed, candidate_id)] = independent_metrics([class_index[r["y_true"]] for r in values], [class_index[r["y_predicted"]] for r in values])
        final_rows = read(run / "final_metrics.json")
        for row in final_rows:
            check_saved_metric(audit, row, final_lookup[(row["seed"], row["candidate_id"])], "feature final case")
        for row in read(run / "summary.json"):
            values = [r for r in final_rows if r["method"] == row["method"]]
            for m in METRICS:
                audit.compare(row[m+"_mean"], float(np.mean([v[m] for v in values])), "feature final summary")
    return {"status": "pass", "development_prediction_rows": raw_rows, "development_metric_cases": len(development),
        "source_models_checked": len(development), "source_inference_replays": replayed,
        "fit_warning_records": warnings_count, "selected": chosen["id"], "eligible_selection_candidates": len(eligible),
        "v1_v2_predictor_comparisons": version_prediction_comparisons,
        "final_prediction_rows": final_count}


def probability_context(audit, directory, truth, source_ids, query_ids, raw_X, raw_lookup, final):
    """Verify and replay source-only folds without any .fit or optimization call."""
    import joblib
    import torch
    from herb_spectra.neural_base_search import DenseClassifier
    directory = Path(directory); meta = read(directory / "context.json")
    audit.compare(meta["source_ids"], source_ids, "probability source role IDs")
    audit.compare(meta["prediction_ids"], query_ids, "probability query role IDs")
    audit.compare(meta["source_fingerprints"], meta["source_fingerprints_after"], "outer source state unchanged")
    audit.compare(meta["target_support_labels_used"], 0, "no target calibration labels used")
    audit.require(not meta["prediction_labels_used_for_fit"] and not meta["query_statistics_fitted"], "source-only probability fitting flags")
    audit.compare(sha(directory / "scores.npz"), meta["scores_npz_sha256"], "source score cache SHA")
    with np.load(directory / "scores.npz", allow_pickle=False) as h:
        values = {k: h[k] for k in h.files}
    classes = values["classes"].astype(str).tolist(); class_index = {c: i for i, c in enumerate(classes)}
    audit.compare(values["source_ids"].astype(str).tolist(), source_ids, "cache source order")
    audit.compare(values["prediction_ids"].astype(str).tolist(), query_ids, "cache query order")
    source_y = np.asarray([class_index[truth[i]] for i in source_ids])
    audit.compare(values["source_y"], source_y, "source labels align with raw uploaded input")
    audit.require(("prediction_y" not in values) if final else "prediction_y" in values, "final score payload does not contain truth")
    if not final:
        audit.compare(values["prediction_y"], [class_index[truth[i]] for i in query_ids], "development scoring truth")
    counts = np.bincount(source_y, minlength=7)
    excluded = np.flatnonzero(counts[source_y] < 3); eligible = np.flatnonzero(counts[source_y] >= 3)
    audit.compare(meta["oof_policy"]["excluded_oof_ids"], [source_ids[i] for i in excluded], "rare source OOF exclusion")
    audit.compare(values["oof_ids"].astype(str).tolist(), [source_ids[i] for i in eligible], "genuine heldout source OOF IDs")
    audit.compare(values["oof_y"], source_y[eligible], "genuine heldout source OOF truth")
    audit.compare(len(eligible), 513 if final else 306, "source OOF row count")
    rng = np.random.default_rng(meta["oof_policy"]["seed"])
    validation = [[] for _ in range(3)]
    for label in range(7):
        if counts[label] >= 3:
            indices = np.flatnonzero(source_y == label)
            for fold, group in enumerate(np.array_split(rng.permutation(indices), 3)):
                validation[fold].extend(group.tolist())
    oof_index = {i: n for n, i in enumerate(values["oof_ids"].astype(str))}
    model_states = 0
    for record in meta["folds"]:
        fold = record["fold"]; case = directory / "oof" / f"fold_{fold:02d}"
        valid = np.asarray(sorted(validation[fold]), int)
        train = np.asarray(sorted(set(range(len(source_ids))) - set(valid.tolist())), int)
        expected_train = [source_ids[i] for i in train]; expected_valid = [source_ids[i] for i in valid]
        audit.compare(record["source_train_ids"], expected_train, "independent source OOF training PCG split")
        audit.compare(record["source_validation_ids"], expected_valid, "independent source OOF heldout PCG split")
        audit.require(not set(expected_train) & set(expected_valid), "source OOF physical train/validation isolation")
        audit.require(set([source_ids[i] for i in excluded]) <= set(expected_train), "singleton retained in every OOF fit")
        audit.compare(record, read(case / "training.json"), "fold metadata exact copy")
        for name, digest in record["artifacts"].items():
            audit.compare(sha(case / name), digest, "source OOF artifact SHA")
        pre = joblib.load(case / "preprocessor.joblib")
        sx = raw_X[[raw_lookup[i] for i in expected_train]]
        qx = raw_X[[raw_lookup[i] for i in expected_valid]]
        snv_source = (sx-sx.mean(1, keepdims=True))/sx.std(1, ddof=1, keepdims=True)
        snv_query = (qx-qx.mean(1, keepdims=True))/qx.std(1, ddof=1, keepdims=True)
        audit.compare(pre["source_samples"], len(sx), "OOF neural scaler fit sample count")
        audit.require(pre["pca"] is None, "locked b006 source OOF uses no PCA")
        audit.compare(pre["scaler"].mean_, snv_source.mean(0), "OOF scaler mean fitted on fold training only")
        audit.compare(pre["scaler"].var_, snv_source.var(0), "OOF scaler variance fitted on fold training only")
        transformed = np.ascontiguousarray(pre["scaler"].transform(snv_query), dtype=np.float32)
        locations = np.asarray([oof_index[i] for i in expected_valid])
        svm = joblib.load(case / "svm.joblib")
        audit.compare(joblib.hash(svm), record["fingerprints"]["svm"], "OOF SVM state fingerprint")
        audit.compare(svm.scaler_.mean_, snv_source.mean(0), "OOF SVM scaler source-only mean")
        audit.compare(svm.classes_.astype(str).tolist(), classes, "OOF SVM fixed seven-class score order")
        audit.compare(svm.predict_scores(qx), values["oof_scores_svm"][locations], "OOF SVM raw score replay")
        for provider in ("mlp", "mlp_sup100"):
            state = torch.load(case / (provider + ".pt"), map_location="cpu", weights_only=True)
            audit.compare(independent_state_fingerprint(state), record["fingerprints"][provider], "OOF neural state fingerprint")
            model = DenseClassifier(612, (64, 32), 7); model.load_state_dict(state); model.eval()
            with torch.no_grad():
                score = model(torch.from_numpy(transformed)).numpy()
            audit.compare(score, values["oof_scores_" + provider][locations], "OOF neural raw score replay")
            model_states += 1
    for provider, artifact in meta["source_artifacts"].items():
        audit.compare(sha(ROOT / artifact["path"]), artifact["sha256"], "outer source artifact SHA")
        if provider == "svm":
            model = joblib.load(ROOT / artifact["path"])
            audit.compare(joblib.hash(model), meta["source_fingerprints"][provider], "outer cached SVM state")
            qx = raw_X[[raw_lookup[i] for i in query_ids]]
            audit.compare(model.predict_scores(qx), values["prediction_scores_"+provider], "outer cached SVM score replay")
        else:
            state = torch.load(ROOT / artifact["path"], map_location="cpu", weights_only=True)
            audit.compare(independent_state_fingerprint(state), meta["source_fingerprints"][provider], "outer cached MLP state")
            base_artifact = Path(meta["source_artifacts"]["mlp"]["path"])
            pre = joblib.load(ROOT / base_artifact.parent / "preprocessor.joblib")
            sx = raw_X[[raw_lookup[i] for i in source_ids]]
            qx = raw_X[[raw_lookup[i] for i in query_ids]]
            snv_source = (sx-sx.mean(1, keepdims=True))/sx.std(1, ddof=1, keepdims=True)
            snv_query = (qx-qx.mean(1, keepdims=True))/qx.std(1, ddof=1, keepdims=True)
            audit.compare(pre["scaler"].mean_, snv_source.mean(0), "outer source-only neural scale mean")
            audit.compare(pre["scaler"].var_, snv_source.var(0), "outer source-only neural scale variance")
            model = DenseClassifier(612, (64, 32), 7); model.load_state_dict(state); model.eval()
            with torch.no_grad():
                scores = model(torch.from_numpy(np.ascontiguousarray(pre["scaler"].transform(snv_query), dtype=np.float32))).numpy()
            audit.compare(scores, values["prediction_scores_"+provider], "outer source neural score replay")
            model_states += 1
    for provider in ("svm", "mlp", "mlp_sup100"):
        score = values["oof_scores_" + provider]; temp = meta["temperatures"][provider]
        fit = meta["temperature_fits"][provider]
        p = independent_softmax(score, temp); raw = independent_softmax(score)
        audit.require(np.array_equal(score.argmax(1), p.argmax(1)), "positive temperature preserves hard classification")
        audit.compare(fit["raw_oof"], independent_probability_metrics(raw, values["oof_y"]), "uncalibrated OOF probability diagnostic")
        audit.compare(fit["temperature_oof"], independent_probability_metrics(p, values["oof_y"]), "source OOF calibrated probability diagnostic")
        log_temp = np.log(temp)
        objective = independent_probability_metrics(p, values["oof_y"])["nll"] + fit["regularization"]*log_temp**2
        audit.compare(fit["objective"], objective, "temperature source objective")
        for offset in (-.001, .001):
            nearby = np.clip(log_temp + offset, -3, 3)
            alternative = independent_probability_metrics(independent_softmax(score, np.exp(nearby)), values["oof_y"])["nll"] + fit["regularization"]*nearby**2
            audit.require(objective <= alternative + 1e-10, "temperature objective local optimality arithmetic")
    return meta, values, model_states


def independent_decision_fit(audit, candidate, context, cache, fitted):
    provider = candidate["provider"]; kind = candidate["kind"]
    temperature = 1. if kind == "raw" else context["temperatures"][provider]
    bias = np.zeros(7); source_y = cache["source_y"]
    if kind == "prior_bias":
        counts = np.bincount(source_y, minlength=7)
        bias = -candidate["beta"]*np.log(counts/counts.sum()); bias -= bias[0]
    elif kind == "coordinate_bias":
        p = independent_softmax(cache["oof_scores_"+provider], temperature)
        logp = np.log(np.maximum(p, 1e-15)); truth = cache["oof_y"]
        fixed = set([0]) | set(np.flatnonzero(np.bincount(source_y, minlength=7) < 3).tolist())
        trace = []
        for sweep in range(2):
            for c in range(7):
                if c in fixed:
                    continue
                choices = []
                for value in (-1., -.5, -.25, 0., .25, .5, 1.):
                    alternative = bias.copy(); alternative[c] = value
                    objective = independent_metrics(truth, (logp+alternative).argmax(1))["macro_f1"] - candidate["penalty"]*float(np.mean(alternative**2))
                    choices.append((objective, -abs(value), -value, value))
                chosen = max(choices); bias[c] = chosen[3]
                trace.append({"sweep": sweep, "class_index": c, "bias": float(bias[c]), "penalized_macro_f1": float(chosen[0])})
        audit.compare(fitted["fit_record"]["trace"], trace, "coordinate decision trace deterministic arithmetic")
        audit.compare(fitted["fit_record"]["fixed_bias_classes"], sorted(fixed), "singleton/reference coordinate bias remains fixed")
    audit.compare(fitted["temperature"], temperature, "source-only decision temperature")
    audit.compare(fitted["bias"], bias, "source-only decision bias")
    return temperature, bias


def audit_probability(audit, truth, roles, parts, final=False):
    run = RUN / "probability"
    if not (run / "selection_lock.json").exists():
        return {"status": "not_yet_locked"}
    manifest = check_manifest(audit, run / "run_manifest.json")
    audit.compare(sha(run / "protocol_snapshot.json"), manifest["protocol_sha256"], "probability protocol bytes")
    protocol = read(run / "protocol_snapshot.json")
    lock = read(run / "selection_lock.json")
    grid = []
    for provider in ("svm", "mlp", "mlp_sup100"):
        variants = [{"kind": "raw"}, {"kind": "temperature"}]
        variants += [{"kind": "prior_bias", "beta": beta} for beta in (.25, .5, 1.)]
        variants += [{"kind": "coordinate_bias", "penalty": penalty} for penalty in (.001, .01)]
        for variant in variants:
            grid.append({"id": f"p{len(grid):03d}_{provider}_{variant['kind']}", "provider": provider, **variant})
    with np.load(INPUT, allow_pickle=False) as a:
        ids = a["record_ids"].astype(str); raw_X = a["X_alpha"][:, (a["axis"] >= .1) & (a["axis"] <= 1.5)]
    lookup = {i: n for n, i in enumerate(ids)}
    all_rows = read(run / "all_development_metrics.json")
    saved_lookup = {(r["episode"], r["seed"], r["candidate_id"]): r for r in all_rows}
    counts = {"development_prediction_rows": 0, "final_prediction_rows": 0, "oof_contexts": 0, "oof_neural_states_replayed": 0}
    scopes = ("development", "final") if final else ("development",)
    for scope in scopes:
        for relative in read(run / (scope+"_contexts.json")):
            directory = run / relative; record = read(directory / "context.json")
            episode, seed = record["episode"], record["seed"]
            source_ids = roles["training"] if scope == "final" else parts[episode]["source"]
            query_ids = roles["prediction"] if scope == "final" else parts[episode]["query"]
            context, cache, states = probability_context(audit, directory, truth, source_ids, query_ids, raw_X, lookup, scope == "final")
            counts["oof_contexts"] += 1; counts["oof_neural_states_replayed"] += states
            classes = cache["classes"].astype(str).tolist(); ci = {c: i for i, c in enumerate(classes)}
            actual = np.asarray([ci[truth[i]] for i in query_ids])
            values = csv_rows(audit, directory / "predictions.csv.gz", truth, classes)
            counts[scope+"_prediction_rows"] += len(values)
            groups = defaultdict(list)
            for r in values:
                groups[r["candidate_id"]].append(r)
            fits = {f["candidate"]["id"]: f for f in read(directory / "decision_fits.json")}
            metrics = read(directory / "metrics.json")
            audit.require(len(metrics) == len(grid), "all 21 probability candidates preserved")
            by_id = {r["candidate_id"]: r for r in metrics}
            for candidate in grid:
                key = candidate["id"]; rows = groups[key]
                audit.compare([r["record_id"] for r in rows], query_ids, "probability candidate sample order")
                temp, bias = independent_decision_fit(audit, candidate, context, cache, fits[key])
                p = independent_softmax(cache["prediction_scores_"+candidate["provider"]], temp)
                guessed = (np.log(np.maximum(p, 1e-15))+bias).argmax(1)
                recorded_p = np.asarray([[float(r[f"p{c}"]) for c in range(7)] for r in rows])
                audit.compare(recorded_p, p, "raw CSV class probabilities")
                audit.compare([r["y_predicted"] for r in rows], [classes[c] for c in guessed], "multiclass decision prediction replay")
                audit.compare([r["probability_argmax"] for r in rows], [classes[c] for c in p.argmax(1)], "probability argmax distinct from bias decision")
                audit.compare([float(r["confidence"]) for r in rows], p.max(1), "probability confidence")
                expected = independent_metrics(actual, guessed) | independent_probability_metrics(p, actual)
                check_saved_metric(audit, by_id[key], expected, "probability candidate metric")
                if scope == "development":
                    check_saved_metric(audit, saved_lookup[(episode, seed, key)], expected, "combined development probability metric")
                audit.require(by_id[key]["proper_probability_metrics_exclude_decision_bias"], "proper probabilities not replaced by decision bias")
    means = {c["id"]: {stage: {m: float(np.mean([r[m] for r in all_rows if r["candidate_id"] == c["id"] and r["stage"] == stage]))
                    for m in (*METRICS, "nll", "brier", "ece10")} for stage in ("selection", "gate")} for c in grid}
    audit.compare(lock["means"], means, "independent probability candidate aggregation")
    choices = {}; policy = protocol["selection"]
    for provider in ("svm", "mlp", "mlp_sup100"):
        options = [c for c in grid if c["provider"] == provider]
        raw = next(c for c in options if c["kind"] == "raw"); temp = next(c for c in options if c["kind"] == "temperature")
        ranked = sorted(options, key=lambda c: (-means[c["id"]]["selection"]["macro_f1"], -means[c["id"]]["selection"]["accuracy"], 0 if c["kind"] == "raw" else 1, c["id"]))
        def passes(candidate, stage):
            a, b = means[candidate["id"]][stage], means[raw["id"]][stage]
            return a["macro_f1"]-b["macro_f1"] > policy["minimum_macro_f1_gain"] and a["accuracy"]-b["accuracy"] >= -policy["maximum_accuracy_loss"] and a["balanced_accuracy"]-b["balanced_accuracy"] >= -policy["maximum_balanced_accuracy_loss"]
        eligible = [c for c in ranked if passes(c, "selection")]
        proposed = eligible[0] if eligible else raw
        selected = proposed if eligible and passes(proposed, "gate") else raw
        def probability_pass(stage):
            a, b = means[temp["id"]][stage], means[raw["id"]][stage]
            return b["nll"]-a["nll"] > policy["minimum_nll_gain"] and a["brier"]-b["brier"] <= policy["maximum_brier_loss"]
        expected = {"raw": raw, "selected": selected, "proposed": proposed, "diagnostic": ranked[0],
            "probability_selected": temp if probability_pass("selection") and probability_pass("gate") else raw,
            "eligible_classification_candidates": len(eligible), "fallback_to_raw": selected["kind"] == "raw", "selected_using_final_query": False}
        audit.compare(lock["choices"][provider], expected, "independent classification and proper-probability choices")
        choices[provider] = {k: expected[k]["id"] for k in ("selected", "probability_selected", "diagnostic")}
    if final:
        final_rows = read(run / "final_metrics.json")
        for row in read(run / "summary.json"):
            cases = [r for r in final_rows if r["candidate_id"] == row["id"]]
            for metric in (*METRICS, "nll", "brier", "ece10"):
                audit.compare(row[metric+"_mean"], float(np.mean([r[metric] for r in cases])), "probability final summary")
    return {"status": "pass", **counts, "development_metric_cases": len(all_rows), "choices": choices}


def audit_fusion(audit, truth, roles, parts, final=False):
    run = RUN / "fusion"
    if not (run / "selection_lock.json").exists():
        return {"status": "not_yet_locked"}
    manifest = check_manifest(audit, run / "run_manifest.json")
    audit.compare(sha(run / "protocol_snapshot.json"), manifest["protocol_sha256"], "fusion protocol bytes")
    audit.compare(sha(RUN / "probability/run_manifest.json"), manifest["score_run_manifest_sha256"], "fusion source score manifest")
    audit.compare(sha(RUN / "probability/selection_lock.json"), manifest["score_selection_lock_sha256"], "fusion source score lock")
    protocol = read(run / "protocol_snapshot.json"); lock = read(run / "selection_lock.json")
    grid = []
    for provider in ("mlp", "mlp_sup100"):
        for kind in ("raw_softmax", "source_oof_temperature"):
            for w in (0., .25, .5, .75, 1.):
                grid.append({"id": f"f{len(grid):03d}_fusion", "mlp_provider": provider, "probability": kind, "svm_weight": w})
    audit.compare(read(run / "candidate_grid.json"), grid, "independent 20-candidate fusion grid")
    candidates = {c["id"]: c for c in grid}
    development = read(run / "all_development_metrics.json")
    all_dev_bykey = {(r["episode"], r["seed"], r["candidate_id"]): r for r in development}
    counts = {"development_prediction_rows": 0, "final_prediction_rows": 0, "complementarity_records": 0}
    scopes = ("development", "final") if final else ("development",)
    for scope in scopes:
        paths = sorted((run / scope).glob("**/evaluation.json"))
        audit.compare(len(paths), 3 if scope == "final" else 18, "all prespecified fusion contexts")
        for path in paths:
            record = read(path); directory = path.parent
            source_dir = ROOT / record["lineage"]["directory"]
            source_meta = read(source_dir / "context.json")
            episode, seed = source_meta["episode"], source_meta["seed"]
            expected_source = roles["training"] if scope == "final" else parts[episode]["source"]
            expected_query = roles["prediction"] if scope == "final" else parts[episode]["query"]
            audit.compare(record["source_ids"], expected_source, "fusion source IDs")
            audit.compare(record["query_ids"], expected_query, "fusion query IDs")
            audit.compare(record["context_fingerprint_before"], record["context_fingerprint_after"], "fusion leaves shared source context frozen")
            audit.compare(record["lineage"]["context_metadata_sha256"], sha(source_dir / "context.json"), "fusion source metadata byte lineage")
            audit.compare(record["lineage"]["cache_npz_sha256"], sha(source_dir / "scores.npz"), "fusion source raw scores byte lineage")
            audit.compare(record["predictions_sha256"], sha(directory / "predictions.csv.gz"), "fusion prediction bytes")
            audit.compare(record["target_support_labels_used"], 0, "fusion uses no target calibration labels")
            audit.compare(record["source_models_fitted_by_fusion"], 0, "fusion reuses sources without fitting")
            with np.load(source_dir / "scores.npz", allow_pickle=False) as h:
                cache = {k: h[k] for k in h.files}
            classes = cache["classes"].astype(str).tolist(); ci = {c: i for i, c in enumerate(classes)}
            audit.compare(record["class_order"], classes, "fusion seven-column class order")
            actual = np.asarray([ci[truth[i]] for i in expected_query])
            raw = csv_rows(audit, directory / "predictions.csv.gz", truth, classes)
            counts[scope+"_prediction_rows"] += len(raw)
            grouped = defaultdict(list)
            for row in raw:
                grouped[row["method"]].append(row)
            for row in record["rows"]:
                values = grouped[row["method"]]; candidate = candidates[row["candidate_id"]]
                audit.compare([v["record_id"] for v in values], expected_query, "fusion candidate specimen order")
                temp = source_meta["temperatures"] if candidate["probability"] == "source_oof_temperature" else {p: 1. for p in ("svm", "mlp", "mlp_sup100")}
                first = independent_softmax(cache["prediction_scores_svm"], temp["svm"])
                second = independent_softmax(cache["prediction_scores_"+candidate["mlp_provider"]], temp[candidate["mlp_provider"]])
                probability = candidate["svm_weight"]*first + (1-candidate["svm_weight"])*second
                recorded = np.asarray([[float(v["p_"+c]) for c in classes] for v in values])
                audit.compare(recorded, probability, "independent weighted probability fusion")
                predicted = probability.argmax(1)
                audit.compare([v["y_predicted"] for v in values], [classes[c] for c in predicted], "fusion hard-class predictions")
                expected = independent_metrics(actual, predicted)
                proper = independent_probability_metrics(probability, actual); proper["ece"] = proper.pop("ece10")
                expected.update(proper)
                check_saved_metric(audit, row, expected, "fusion raw metrics")
                if scope == "development":
                    check_saved_metric(audit, all_dev_bykey[(episode, seed, candidate["id"])], expected, "fusion aggregate development row")
            for row in record["complementarity"]:
                first = cache["prediction_scores_svm"].argmax(1)
                second = cache["prediction_scores_"+row["mlp_provider"]].argmax(1)
                a, b = first == actual, second == actual
                expected = {"n": len(actual), "both_correct": int(np.sum(a & b)), "svm_only_correct": int(np.sum(a & ~b)),
                    "mlp_only_correct": int(np.sum(~a & b)), "both_wrong": int(np.sum(~a & ~b)),
                    "different_predictions": int(np.sum(first != second)),
                    "both_wrong_same_prediction": int(np.sum(~a & ~b & (first == second))),
                    "oracle_accuracy_upper_bound": float(np.mean(a | b)), "oracle_is_truth_based_unavailable_predictor": True}
                check_saved_metric(audit, row, expected, "truth-labelled diagnostic complementarity")
                counts["complementarity_records"] += 1
    aggregate = {stage: {c["id"]: {**{m: float(np.mean([r[m] for r in development if r["stage"] == stage and r["candidate_id"] == c["id"]]))
                                for m in (*METRICS, "nll", "brier", "ece")},
                            "evaluations": len([r for r in development if r["stage"] == stage and r["candidate_id"] == c["id"]])}
                        for c in grid} for stage in ("selection", "gate")}
    audit.compare(read(run / "development_aggregate.json"), aggregate, "independent fusion stage aggregation")
    policy = protocol["selection"]
    interiors = [c for c in grid if 0 < c["svm_weight"] < 1]
    def endpoint(candidate, w):
        return next(c for c in grid if c["mlp_provider"] == candidate["mlp_provider"] and c["probability"] == candidate["probability"] and c["svm_weight"] == w)
    def passes(candidate, stage):
        a = aggregate[stage][candidate["id"]]
        for w in (0., 1.):
            b = aggregate[stage][endpoint(candidate, w)["id"]]
            if not (a["macro_f1"]-b["macro_f1"] > policy["minimum_macro_f1_gain"] and a["accuracy"]-b["accuracy"] >= -policy["max_accuracy_loss"] and a["balanced_accuracy"]-b["balanced_accuracy"] >= -policy["max_balanced_accuracy_loss"]):
                return False
        return True
    def rank(options):
        top = max(aggregate["selection"][c["id"]]["macro_f1"] for c in options)
        choices = [c for c in options if aggregate["selection"][c["id"]]["macro_f1"] >= top-policy["tie_tolerance"]]
        return min(choices, key=lambda c: (-aggregate["selection"][c["id"]]["accuracy"], abs(c["svm_weight"]-.5), c["mlp_provider"], c["probability"], c["id"]))
    eligible = [c for c in interiors if passes(c, "selection")]; proposed = rank(eligible) if eligible else None
    passed = proposed is not None and passes(proposed, "gate")
    fallback = next(c for c in grid if c["mlp_provider"] == "mlp_sup100" and c["probability"] == "source_oof_temperature" and c["svm_weight"] == 0)
    expected_choice = {"selected": proposed if passed else fallback, "proposed": proposed, "diagnostic": rank(interiors),
        "eligible_selection_candidates": len(eligible), "passed_gate": passed,
        "fallback_to_prespecified_mlp_sup100": not passed, "selected_using_final_query": False}
    audit.compare(lock["choice"], expected_choice, "independent fusion selected/proposed/diagnostic decision")
    if final:
        rows = read(run / "final_metrics.json")
        for row in read(run / "summary.json"):
            cases = [r for r in rows if r["method"] == row["method"]]
            for m in (*METRICS, "nll", "brier", "ece"):
                values = np.asarray([r[m] for r in cases])
                audit.compare(row[m+"_mean"], float(values.mean()), "fusion final summary mean")
                audit.compare(row[m+"_sd"], float(values.std(ddof=1)), "fusion final summary deviation")
    return {"status": "pass", **counts, "development_metric_cases": len(development),
        "selected": expected_choice["selected"]["id"], "diagnostic": expected_choice["diagnostic"]["id"], "eligible_candidates": len(eligible)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    audit = Audit(); started = time.monotonic()
    arrays, truth, roles, parts, axis = datasets(audit)
    features = audit_feature(audit, truth, roles, parts, axis, final=args.phase == "final")
    probability = audit_probability(audit, truth, roles, parts, final=args.phase == "final")
    fusion = audit_fusion(audit, truth, roles, parts, final=args.phase == "final")
    result = {"status": "pass" if features["status"] == probability["status"] == fusion["status"] == "pass" else "partial_pass", "phase": args.phase, "new_fits": 0, "features": features, "probability": probability, "fusion": fusion,
        "checks_evaluated": audit.checks, "maximum_numeric_difference": audit.max_difference,
        "seconds": time.monotonic()-started, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha(__file__)}
    destination = RUN / ("independent_audit.json" if args.phase == "final" else "independent_development_audit.json" if result["status"] == "pass" else "independent_development_partial.json")
    if args.phase == "final" and (RUN / "all_development_locks.json").exists():
        result["global_release_sha256"] = sha(RUN / "all_development_locks.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
