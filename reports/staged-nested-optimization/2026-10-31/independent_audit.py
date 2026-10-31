"""Independent staged nested-CV audit; does not fit research models."""
from collections import defaultdict
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "outputs/staged-nested-20261031"
INPUT = ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"
HISTORICAL_SPLITS = ROOT / "outputs/approved-zaoren-20261031/experiments/alpha_primary/splits.json"
HISTORICAL_PREDICTIONS = ROOT / "outputs/approved-zaoren-20261031/experiments/alpha_primary/alpha_mcuve20_svc/predictions.csv"
CLASS_METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
PREPROCESSORS = ("raw", "snv", "sg0_snv", "sg1_snv", "sg2_snv", "msc", "linear_detrend_snv")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.maximum_numeric_difference = 0.

    def require(self, passed, message):
        self.checks += 1
        if not passed:
            raise AssertionError(message)

    def same(self, actual, expected, message, tolerance=1e-10):
        if isinstance(actual, dict) and isinstance(expected, dict):
            self.require(set(actual) == set(expected), message + " keys")
            for key in actual:
                self.same(actual[key], expected[key], message + ":" + str(key), tolerance)
        elif isinstance(actual, (list, tuple, np.ndarray)) and isinstance(expected, (list, tuple, np.ndarray)):
            try:
                a, e = np.asarray(actual), np.asarray(expected)
            except (TypeError, ValueError):
                a = e = None
            if a is not None and a.dtype.kind in "bifu" and e.dtype.kind in "bifu" and a.shape == e.shape:
                difference = np.abs(a.astype(float)-e.astype(float))
                maximum = float(difference.max()) if difference.size else 0.
                self.checks += int(difference.size)
                self.maximum_numeric_difference = max(self.maximum_numeric_difference, maximum)
                self.require(np.isfinite(difference).all() and np.all(difference <= tolerance), (message, "maximum difference", maximum))
                return
            self.require(len(actual) == len(expected), message + " length")
            for n, (a, e) in enumerate(zip(actual, expected)):
                self.same(a, e, message + ":" + str(n), tolerance)
        elif isinstance(actual, (int, float, np.integer, np.floating)) and isinstance(expected, (int, float, np.integer, np.floating)):
            delta = abs(float(actual) - float(expected))
            self.maximum_numeric_difference = max(self.maximum_numeric_difference, delta)
            self.require(np.isfinite(delta) and delta <= tolerance, (message, actual, expected))
        else:
            self.require(actual == expected, (message, actual, expected))


def metrics(truth, predicted, classes):
    vocab = {str(c): n for n, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), np.int64)
    for actual, guessed in zip(truth, predicted):
        cm[vocab[str(actual)], vocab[str(guessed)]] += 1
    total = int(cm.sum()); correct = cm.diagonal()
    actual, guessed = cm.sum(1), cm.sum(0)
    recall = np.divide(correct, actual, out=np.zeros(len(classes)), where=actual > 0)
    precision = np.divide(correct, guessed, out=np.zeros(len(classes)), where=guessed > 0)
    f1 = np.divide(2*correct, actual+guessed, out=np.zeros(len(classes)), where=actual+guessed > 0)
    return {"accuracy": float(correct.sum()/total), "balanced_accuracy": float(recall[actual > 0].mean()),
        "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
        "class_recall": dict(zip(classes, recall.tolist())), "class_precision": dict(zip(classes, precision.tolist())), "class_f1": dict(zip(classes, f1.tolist())),
        "class_counts": dict(zip(classes, actual.tolist())), "class_order": list(classes), "correct": int(correct.sum()), "total": total}


def check_metrics(audit, actual, expected, description):
    for key, value in expected.items():
        if key in actual:
            audit.same(actual[key], value, description + ":" + key)


def load_data(audit):
    with np.load(INPUT, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    groups = arrays["groups"].astype(str)
    audit.require(len(groups) == 855 and len(np.unique(groups)) == 855, "855 user-declared independent sample groups")
    order = np.argsort(groups); ids = groups[order]; y = arrays["y"][order].astype(str)
    band = (arrays["axis"] >= .1) & (arrays["axis"] <= 1.5)
    X = arrays["X_alpha"][order][:, band]; axis = arrays["axis"][band]
    audit.same(X.shape, (855, 612), "fixed current spectra shape")
    classes = sorted(set(y)); truth = dict(zip(ids, y)); lookup = {i: n for n, i in enumerate(ids)}
    audit.same({c: int(np.sum(y == c)) for c in classes}, {"ZY01": 350, "ZY02": 5, **{f"ZY0{i}": 100 for i in range(3, 8)}}, "source class counts")
    outer = list(StratifiedKFold(5, shuffle=True, random_state=20261031).split(np.zeros((len(y), 1)), y))
    records = []
    for number, (train, test) in enumerate(outer):
        inner = list(StratifiedKFold(2, shuffle=True, random_state=20261031).split(np.zeros((len(train), 1)), y[train]))
        records.append({"fold": number, "train_sample_ids": ids[train].tolist(), "test_sample_ids": ids[test].tolist(),
            "inner_folds": [{"train_sample_ids": ids[train[t]].tolist(), "validation_sample_ids": ids[train[v]].tolist()} for t, v in inner]})
        audit.require(not set(ids[train]) & set(ids[test]), "outer physical sample isolation")
        for t, v in inner:
            audit.require(not set(ids[train[t]]) & set(ids[train[v]]), "inner physical sample isolation")
            audit.same(int(np.sum(y[train[t]] == "ZY02")), 2, "inner source contains two rare specimens")
    audit.same(records, read(HISTORICAL_SPLITS), "exactly reproduce historical five/two folds")
    return {"ids": ids, "X": X, "y": y, "axis": axis, "classes": classes, "truth": truth, "lookup": lookup, "splits": records}


def check_manifest(audit, directory):
    directory = Path(directory); manifest = read(directory / "run_manifest.json")
    audit.same(sha(INPUT), manifest["input_sha256"], "actual input SHA")
    audit.same(sha(directory / "protocol_snapshot.json"), manifest["protocol_sha256"], "executed protocol bytes SHA")
    for key, value in manifest.items():
        if isinstance(value, dict):
            for relative, expected in value.items():
                if isinstance(relative, str) and relative.endswith(".py") and isinstance(expected, str) and len(expected) == 64:
                    audit.same(sha(ROOT / relative), expected, "executed algorithm source SHA")
    return manifest


def read_predictions(audit, path, data):
    handle = gzip.open(path, "rt", encoding="utf-8-sig", newline="") if str(path).endswith(".gz") else Path(path).open(encoding="utf-8-sig", newline="")
    with handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = row.get("record_id", row.get("sample_id"))
        audit.require(key in data["truth"], "known original physical sample ID")
        audit.same(row["y_true"], data["truth"][key], "ground truth agrees with uploaded input")
        audit.require(row["y_predicted"] in data["classes"], "known predicted anonymous origin")
        if "is_correct" in row:
            audit.same(int(row["is_correct"]), int(row["y_true"] == row["y_predicted"]), "per-sample correct/incorrect flag")
    return rows


def independent_preprocessing(X, mode, axis):
    from scipy.signal import detrend, savgol_filter
    x = np.asarray(X, float)
    if mode == "raw":
        return x.copy(), None
    if mode == "msc":
        reference = x.mean(0); centered = reference-reference.mean()
        slope = ((x-x.mean(1, keepdims=True)) @ centered)/(centered @ centered)
        intercept = x.mean(1)-slope*reference.mean()
        return (x-intercept[:, None])/slope[:, None], reference
    if mode.startswith("sg"):
        x = savgol_filter(x, 15, 3, deriv=int(mode[2]), delta=float(np.diff(axis).mean()), axis=1, mode="interp")
    elif mode == "linear_detrend_snv":
        x = detrend(x, axis=1, type="linear")
    return (x-x.mean(1, keepdims=True))/x.std(1, ddof=1, keepdims=True), None


def independent_weights(y, mode):
    classes, counts = np.unique(y, return_counts=True)
    if mode == "none":
        weights = np.ones(len(classes))
    else:
        weights = len(y)/(len(classes)*counts.astype(float))
        if mode.startswith("power_"):
            weights = weights ** float(mode.split("_")[1])
            weights /= (weights*counts).sum()/len(y)
    return {str(c): float(w) for c, w in zip(classes, weights)}


def check_transform(audit, reference, expected_ids, config, data, cache):
    import joblib
    key = reference["metadata"]
    if key not in cache:
        metadata = read(ROOT / key)
        audit.same(sha(ROOT / key), reference["metadata_sha256"], "inner source transformation metadata SHA")
        for field, hash_field in (("statistics", "statistics_sha256"), ("coefficients", "coefficients_sha256")):
            audit.same(sha(ROOT / reference[field]), reference[hash_field], "actual saved source statistics bytes")
        with np.load(ROOT / reference["statistics"], allow_pickle=False) as p:
            stats = {k: p[k] for k in p.files}
        with np.load(ROOT / reference["coefficients"], allow_pickle=False) as p:
            coefficients = p["coefficients"]
        X = data["X"][[data["lookup"][i] for i in expected_ids]]
        y = np.asarray([data["truth"][i] for i in expected_ids])
        audit.same(metadata["source_ids"], expected_ids, "source transform fitted physical sample IDs")
        audit.same(metadata["source_samples"], len(expected_ids), "source transform fit count")
        audit.same(metadata["source_value_fingerprint"], joblib.hash(X), "source transform numerical input fingerprint")
        audit.same(metadata["source_label_fingerprint"], joblib.hash(y), "source transform category fingerprint")
        prepared, msc_ref = independent_preprocessing(X, metadata["preprocessing"], data["axis"])
        audit.same(stats["scaler_mean"], prepared.mean(0), "source-only preprocessor/scaler mean")
        audit.same(stats["scaler_var"], prepared.var(0), "source-only preprocessor/scaler variance")
        audit.same(metadata["scaler_mean"], stats["scaler_mean"], "readable scaler mean binds binary statistics")
        audit.same(metadata["scaler_variance"], stats["scaler_var"], "readable scaler variance binds binary statistics")
        audit.same(metadata["msc_source_reference"], None if msc_ref is None else msc_ref.tolist(), "MSC reference only current inner source")
        audit.same(coefficients.shape, (metadata["mc"]["runs"], 7, 612), "saved classification onehot MC coefficient dimensions")
        mean = coefficients.mean(0); sd = coefficients.std(0, ddof=1)
        signed = mean/np.maximum(sd, max(1., float(sd.max()))*1e-12)
        importance = np.linalg.norm(np.abs(signed), axis=0)
        ranked = np.argsort(-importance, kind="stable")
        count = int(np.ceil(612*metadata["mc"]["ratio"]))
        selected = np.sort(ranked[:count])
        audit.same(stats["signed_stability"], signed, "independent MC stability mean/ddof1 SD/floor")
        audit.same(stats["importance"], importance, "independent onehot coefficient norm")
        audit.same(stats["ranking"], ranked, "independent stable importance ordering")
        audit.same(stats["selected_indices"], selected, "independent ratio/ceil feature selection")
        audit.same(metadata["selected_indices"], selected.tolist(), "readable selected indices")
        audit.same(metadata["selected_frequencies"], data["axis"][selected].tolist(), "selected original frequency coordinates")
        audit.same(metadata["selected_features"], count, "selected source feature count")
        scale = np.sqrt(stats["scaler_var"]); scale[scale == 0] = 1.
        z = ((prepared-stats["scaler_mean"])/scale)[:, selected]
        cache[key] = {"meta": metadata, "stats": stats, "coefficients": coefficients, "z": z, "source_y": y}
    entry = cache[key]
    audit.same(entry["meta"]["preprocessing"], config["preprocessing"], "candidate source preprocessing inheritance")
    audit.same(entry["meta"]["mc"], config["mc"], "candidate source MC inheritance")
    return entry


def candidates_for(stage, previous):
    if stage == "R0":
        return [{"id": f"hist{i:03d}", "preprocessing": prep, "mc": {"runs": 20, "components": 2, "ratio": .1},
                 "svm": {"C": C, "gamma": "scale", "class_weight": "balanced"}}
                for i, (C, prep) in enumerate((c, p) for c in (.1, 1., 10., 100.) for p in ("raw", "snv"))]
    if stage == "R1":
        return [{"id": f"mc{i:03d}", "preprocessing": "snv", "mc": {"runs": runs, "components": lv, "ratio": fraction},
                 "svm": {"C": 10., "gamma": "scale", "class_weight": "balanced"}}
                for i, (runs, lv, fraction) in enumerate((r, l, f) for r in (10, 20, 50) for l in (1, 2, 3) for f in (.05, .1, .15, .2))]
    if stage == "R2":
        return [{"id": f"sv{i:03d}", "preprocessing": "snv", "mc": previous["R1"]["selected"]["mc"],
                 "svm": {"C": C, "gamma": gamma, "class_weight": weight}}
                for i, (C, gamma, weight) in enumerate((c, g, w) for c in (.01, .1, 1., 10., 100., 1000.)
                     for g in ("scale", "auto", "scale_x0.1", "scale_x10") for w in ("none", "balanced", "power_0.5", "power_1.5"))]
    return [{"id": f"prep{i:02d}_{prep}", "preprocessing": prep, "mc": previous["R1"]["selected"]["mc"],
             "svm": previous["R2"]["selected"]["svm"]} for i, prep in enumerate(PREPROCESSORS)]


def independent_choice(rows, candidates, historical=False):
    means = {candidate["id"]: {metric: float(np.mean([r["metrics"][metric] for r in rows if r["candidate_id"] == candidate["id"]]))
                              for metric in CLASS_METRICS} for candidate in candidates}
    if historical:
        selected = candidates[int(np.argmax([means[c["id"]]["macro_f1"] for c in candidates]))]
    else:
        best = max(v["macro_f1"] for v in means.values())
        near = [c for c in candidates if means[c["id"]]["macro_f1"] >= best-1e-12]
        selected = min(near, key=lambda c: (-means[c["id"]]["balanced_accuracy"], -means[c["id"]]["accuracy"],
            c["mc"]["ratio"], c["mc"]["components"], c["mc"]["runs"], c["svm"]["C"], PREPROCESSORS.index(c["preprocessing"]), c["id"]))
    return {"selected": selected, "mean_inner_metrics": means[selected["id"]], "all_candidate_means": means,
        "selected_using_outer_test": False, "historical_first_tie_policy": historical}


def audit_core_selection(audit, data):
    directory = RUN / "core"
    if not (directory / "selection_lock.json").exists():
        return {"status": "not_yet_locked"}
    check_manifest(audit, directory)
    audit.same(read(directory / "splits.json"), data["splits"], "core uses exact historical split IDs")
    lock = read(directory / "selection_lock.json"); all_inner = []; total_raw = 0
    transform_cache = {}; coefficient_prefixes = {}; alias_rows = 0
    for fold in lock["outer_folds"]:
        outer = fold["outer_fold"]; expected_split = data["splits"][outer]; previous = {}
        for stage in ("R0", "R1", "R2", "R3"):
            candidates = candidates_for(stage, previous)
            case = directory / "selection" / f"outer_{outer:02d}" / stage
            checkpoint = read(case / "checkpoint.json")
            audit.same(checkpoint["candidates"], candidates, "outer-specific stage candidate grid")
            audit.same(sha(case / "metrics.json"), checkpoint["metrics_sha256"], "inner metrics byte SHA")
            audit.same(sha(case / "predictions.csv.gz"), checkpoint["predictions_sha256"], "inner predictions byte SHA")
            rows = read(case / "metrics.json"); raw = read_predictions(audit, case / "predictions.csv.gz", data)
            total_raw += len(raw); grouped = defaultdict(list)
            for r in raw:
                grouped[(int(r["inner_fold"]), r["candidate_id"])].append(r)
            audit.same(len(rows), 2*len(candidates), "complete two-inner-fold grid")
            by_id = {c["id"]: c for c in candidates}
            for row in rows:
                inner = row["inner_fold"]; config = by_id[row["candidate_id"]]
                audit.same(row["candidate"], config, "inner candidate parameters")
                source_ids = expected_split["inner_folds"][inner]["train_sample_ids"]
                validation_ids = expected_split["inner_folds"][inner]["validation_sample_ids"]
                audit.same(row["source_ids"], source_ids, "inner current source IDs")
                audit.same(row["validation_ids"], validation_ids, "inner current validation IDs")
                values = grouped[(inner, config["id"])]
                audit.same([r["record_id"] for r in values], validation_ids, "every validation row predicted exactly once per candidate")
                expected = metrics([r["y_true"] for r in values], [r["y_predicted"] for r in values], data["classes"])
                check_metrics(audit, row["metrics"], expected, "independent source-inner metrics")
                scores = np.asarray([[float(v["score_"+c]) for c in data["classes"]] for v in values])
                vote_argmax_diff = sum(v["y_predicted"] != data["classes"][i] for v, i in zip(values, scores.argmax(1)))
                audit.same(row["vote_argmax_disagreements"], vote_argmax_diff, "standard SVC voting distinct from OVR argmax")
                saved = check_transform(audit, row["transform_diagnostic"], source_ids, config, data, transform_cache)
                audit.same(row["selected_indices"], saved["meta"]["selected_indices"], "inner selected source frequencies bind candidate")
                z = saved["z"]; scale = 1/(z.shape[1]*z.var()) if z.var() else 1.
                auto = 1/z.shape[1]
                numeric = {"scale": scale, "auto": auto, "scale_x0.1": .1*scale, "scale_x10": 10*scale}[config["svm"]["gamma"]]
                audit.same(row["effective_gamma"], numeric, "SVM numeric gamma only current selected source values")
                audit.same(row["auto_gamma"], auto, "SVM auto gamma source K")
                audit.same(row["scale_gamma"], scale, "SVM scale gamma source variance")
                audit.same(row["training_class_weights"], independent_weights(saved["source_y"], config["svm"]["class_weight"]), "current-inner source class weights")
                if np.isclose(scale, auto, atol=1e-14, rtol=1e-12):
                    alias_rows += 1
                if stage == "R1":
                    key = (outer, inner, config["mc"]["components"], config["mc"]["runs"])
                    coefficient = saved["coefficients"]
                    if key in coefficient_prefixes:
                        audit.same(coefficient, coefficient_prefixes[key], "ratio variants reuse identical MC stream")
                    else:
                        coefficient_prefixes[key] = coefficient
            expected_choice = independent_choice(rows, candidates, historical=stage == "R0")
            audit.same(fold["choices"][stage], expected_choice, "independent per-outer sequential choice " + stage)
            previous[stage] = expected_choice
            all_inner.extend(rows)
    for outer in range(5):
        for inner in range(2):
            for lv in (1, 2, 3):
                longest = coefficient_prefixes[(outer, inner, lv, 50)]
                for runs in (10, 20):
                    audit.same(coefficient_prefixes[(outer, inner, lv, runs)], longest[:runs], "exact saved MC stream prefix equivalence")
    audit.same(read(directory / "all_inner_metrics.json"), all_inner, "all-inner table exact per-outer composition")
    return {"status": "pass", "inner_metric_cases": len(all_inner), "inner_prediction_rows": total_raw,
        "unique_inner_transform_states": len(transform_cache), "mc_prefix_comparisons": 60,
        "scale_auto_alias_inner_rows": alias_rows,
        "per_outer_choices": [{"outer_fold": f["outer_fold"], "selected": {s: v["selected"] for s, v in f["choices"].items()}} for f in lock["outer_folds"]]}


FAMILIES = ("ordinary_mcuve", "budget_mcuve", "stable_mcuve", "stable_cars", "stable_joint", "cnn_softmax", "cnn_embedding_svm")


def tensor_fingerprint(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(tensor.dtype).encode())
        result.update(np.asarray(tensor.shape, dtype="<i8").tobytes()); result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def numeric_fingerprint(model):
    import joblib
    return joblib.hash({"family": model.family, "inherited": model.inherited,
        "preprocessor": model.preprocessor, "scaler": model.scaler, "indices": model.indices,
        "classes": model.classes_, "estimator": model.estimator, "embedding_scaler": model.embedding_scaler,
        "network_state": tensor_fingerprint(model.network.state_dict()) if model.network is not None else None,
        "network_training": bool(model.network.training) if model.network is not None else None, "metadata": model.metadata})


def inherited_for(fold):
    choices = fold["choices"]
    return {"preprocessing": choices["R3"]["selected"]["preprocessing"],
        "mc": choices["R1"]["selected"]["mc"], "svm": choices["R2"]["selected"]["svm"]}


def topk(frequency, ranking, count):
    position = np.empty(len(ranking), int); position[np.asarray(ranking, int)] = np.arange(len(ranking))
    return np.sort(np.lexsort((position, -np.asarray(frequency)))[:count])


def check_stability(audit, common, source_ids, inherited, data):
    source_y = np.asarray([data["truth"][i] for i in source_ids]); k = int(np.ceil(612*inherited["mc"]["ratio"]))
    stable = common["stability"]
    audit.same(stable["resample_count"], 10, "ten source-only stability resamples")
    audit.require(not stable["resamples_are_independent_batches"], "resamples not claimed independent batches")
    counts_mc = np.zeros(612); counts_cars = np.zeros(612); sets_mc = []; sets_cars = []; joint_sets = []
    rng = np.random.default_rng(42+9000); classes = sorted(set(source_y))
    ordinary_ranking = np.asarray(common["ordinary_ranking"], int)
    audit.same(stable["ordinary_full_source_ranking"], ordinary_ranking, "shared no-extra-fit tie ranking")
    for repeat, record in enumerate(stable["source_resamples"]):
        chosen = []
        for label in classes:
            indices = np.flatnonzero(source_y == label)
            count = min(len(indices), max(2, int(np.ceil(.8*len(indices)))))
            chosen.extend(rng.permutation(indices)[:count])
        chosen = np.asarray(sorted(chosen), int)
        audit.same(record["source_indices"], chosen, "independent PCG stratified stability sample")
        audit.same(record["source_ids"], [source_ids[i] for i in chosen], "source stability physical IDs")
        audit.require(len(set(record["source_ids"])) == len(chosen) and set(record["source_ids"]) <= set(source_ids), "without-replacement source-only sampling")
        actual_counts = {c: int(np.sum(source_y[chosen] == c)) for c in classes}
        audit.same(record["class_counts"], actual_counts, "source bootstrap class counts")
        audit.require(min(actual_counts.values()) >= 2, "two-fold internal CARS feasible for each resample class")
        audit.same(record["seed"], 42+1000+repeat, "fixed per-resample fit seed")
        mc = np.asarray(record["mcuve_indices"], int); cars = np.asarray(record["cars_indices"], int)
        audit.same(len(mc), k, "MC per-resample feature budget")
        for indices in (mc, cars):
            audit.require(len(indices) == len(set(indices.tolist())) and np.all((indices >= 0) & (indices < 612)), "valid source selected variables")
        path = record["cars_path_indices"]; mse = record["cars_inner_mse_path"]
        best = min(range(len(path)), key=lambda i: (mse[i], len(path[i])))
        audit.same(record["cars_best_iteration"], best, "CARS selected by own source-internal MSE")
        audit.same(cars.tolist(), path[best], "CARS returned source path subset")
        counts_mc[mc] += 1; counts_cars[cars] += 1; sets_mc.append(set(mc)); sets_cars.append(set(cars))
        frequency = np.zeros(612); frequency[mc] += .5; frequency[cars] += .5
        joint_sets.append(set(topk(frequency, ordinary_ranking, k)))
    frequencies = {"stable_mcuve": counts_mc/10, "stable_cars": counts_cars/10, "stable_joint": .5*(counts_mc/10+counts_cars/10)}
    for family, frequency in frequencies.items():
        audit.same(stable["selection_frequencies"][family], frequency, "independent selected-frequency tally")
        selected = topk(frequency, ordinary_ranking, k)
        audit.same(stable["selected_indices"][family], selected, "fixed-K stable frequency ranking")
        audit.same(stable["supplementation"][family]["positive_frequency_variables"], int(np.sum(frequency > 0)), "positive stability variables")
        audit.same(stable["supplementation"][family]["zero_frequency_supplement_count"], int(np.sum(frequency[selected] <= 0)), "zero-frequency supplementation disclosure")
    for family, sets in (("mcuve", sets_mc), ("cars", sets_cars), ("joint_resample_diagnostic", joint_sets)):
        pairs = []
        for first in range(10):
            for second in range(first+1, 10):
                intersection, union = len(sets[first] & sets[second]), len(sets[first] | sets[second])
                pairs.append({"first": first, "second": second, "intersection": intersection, "union": union, "jaccard": intersection/union if union else 1.})
        audit.same(stable["pairwise_jaccard"][family], {"pairs": pairs, "mean": float(np.mean([r["jaccard"] for r in pairs]))}, "all 45 source resampling Jaccards")
    runs = int(inherited["mc"]["runs"]); budget = stable["fit_budget"]
    expected_budget = {"ordinary_mcuve_coefficient_rounds": runs, "ordinary_ovr_pls_fits": 7*runs,
        "budget_mcuve_coefficient_rounds": 10*runs, "budget_ovr_pls_fits": 10*7*runs,
        "stable_mcuve_coefficient_rounds": 10*runs, "stable_mcuve_ovr_pls_fits": 10*7*runs,
        "stable_cars_coefficient_rounds": 120, "stable_cars_ovr_pls_fits": 120*7,
        "stable_cars_multivariate_internal_cv_pls_fits": 240, "tie_breaker_extra_pls_fits": 0,
        "tie_breaker_reuses_ordinary_full_source_ranking": True, "budget_control_matches_MC_rounds_not_walltime_or_FLOPs": True,
        "stable_joint_reuses_already_fitted_MC_and_CARS": True}
    audit.same(budget, expected_budget, "explicit MC/CARS computational budget")
    return {family: np.asarray(stable["selected_indices"][family], int) for family in frequencies}


def check_r4_caller(audit, directory, source_ids, query_ids, inherited, data):
    import joblib
    import torch
    directory = Path(directory); common = read(directory / "caller_common.json"); checkpoint = read(directory / "source_checkpoint.json")
    audit.same(sha(directory / "caller_common.json"), checkpoint["common_sha256"], "R4 common source state bytes")
    audit.same(common["source_ids"], source_ids, "R4 source physical sample IDs")
    audit.same(common["inherited"], inherited, "R4 current-outer inherited configuration")
    X = data["X"][[data["lookup"][i] for i in source_ids]]; y = np.asarray([data["truth"][i] for i in source_ids])
    qX = data["X"][[data["lookup"][i] for i in query_ids]]
    audit.same(common["source_value_hash"], joblib.hash(X), "R4 exact source spectrum values")
    audit.same(common["source_label_hash"], joblib.hash(y), "R4 exact source class labels")
    prepared, reference = independent_preprocessing(X, inherited["preprocessing"], data["axis"])
    audit.same(common["source_scaler_mean"], prepared.mean(0), "R4 source-only full spectrum scale mean")
    audit.same(common["source_scaler_variance"], prepared.var(0), "R4 source-only full spectrum scale variance")
    audit.same(common["msc_source_reference"], None if reference is None else reference.tolist(), "R4 source-only MSC reference")
    stable_indices = check_stability(audit, common, source_ids, inherited, data)
    k = int(np.ceil(612*inherited["mc"]["ratio"]))
    audit.same(common["ordinary_selected_indices"], np.sort(np.argsort(-np.asarray(common["ordinary_importance"]), kind="stable")[:k]), "ordinary MC matched K")
    audit.same(common["budget_selected_indices"], np.sort(np.argsort(-np.asarray(common["budget_importance"]), kind="stable")[:k]), "budget MC matched K")
    saved_state = torch.load(directory / "cnn_source_state.pt", map_location="cpu", weights_only=True)
    audit.same(sha(directory / "cnn_source_state.pt"), checkpoint["cnn_state_sha256"], "shared CNN source state bytes")
    audit.same(tensor_fingerprint(saved_state), checkpoint["shared_encoder_fingerprint"], "shared CNN source numerical state")
    audit.same(checkpoint["cnn_source_training_invocations"], 1, "one shared encoder training per caller source")
    models = {}; raw_predictions = {}
    for family in FAMILIES:
        record = checkpoint["artifacts"][family]; path = directory / record["file"]
        audit.same(sha(path), record["sha256"], "R4 source checkpoint SHA")
        model = joblib.load(path)
        audit.same(numeric_fingerprint(model), record["numeric_fingerprint"], "R4 source canonical numeric fingerprint")
        audit.same(model.metadata, record["metadata"], "R4 readable metadata bound to loaded state")
        audit.require(record["source_roundtrip_predictions_equal"], "R4 serialization preserves source class decisions")
        audit.same(model.inherited, inherited, "R4 source model current-fold inheritance")
        audit.same(model.metadata["source_ids"], source_ids, "R4 classifier source physical IDs")
        if family.startswith("cnn"):
            state = model.network.state_dict()
            audit.same(tensor_fingerprint(state), checkpoint["shared_encoder_fingerprint"], "CNNhead/SVM identical learned encoder")
            audit.same(model.metadata["full_input_features"], 612, "CNN independent full-spectrum input")
            audit.same(model.metadata["embedding_features"], 32, "CNN independent 32-dimensional representation")
            audit.require(not model.metadata["old_weights_reused"], "new source-only CNN, no prior855 weights")
        else:
            expected_indices = (np.asarray(common[family.split('_')[0]+"_selected_indices"], int) if family in ("ordinary_mcuve", "budget_mcuve") else stable_indices[family])
            audit.same(model.indices, expected_indices, "R4 source-only selector mask")
            audit.same(len(model.indices), k, "five selector classifiers matched K")
        before = numeric_fingerprint(model)
        predicted = model.predict(qX); scores = model.predict_scores(qX)
        audit.same(numeric_fingerprint(model), before, "R4 prediction preserves source state")
        raw_predictions[family] = (predicted, scores)
        if model.estimator is not None:
            z = model.scaler.transform(model.preprocessor.transform(X))
            if family == "cnn_embedding_svm":
                with torch.no_grad():
                    embedding = model.network.encode(torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))[:, None, :]).numpy()
                audit.same(model.embedding_scaler.mean_, embedding.mean(0, dtype=np.float64), "CNN embedding scaler only source features")
                audit.same(model.embedding_scaler.var_, embedding.astype(float).var(0), "CNN embedding variance only source features")
                z = model.embedding_scaler.transform(embedding)
            else:
                z = z[:, model.indices]
            numeric_values = np.asarray(z, dtype=float)
            scale = 1/(numeric_values.shape[1]*numeric_values.var()) if numeric_values.var() else 1.
            effective = {"scale": scale, "auto": 1/numeric_values.shape[1], "scale_x0.1": .1*scale, "scale_x10": 10*scale}[inherited["svm"]["gamma"]]
            audit.same(model.estimator._gamma, effective, "R4 gamma on actual source classifier inputs")
            expected_weights = independent_weights(y, inherited["svm"]["class_weight"])
            audit.same(model.metadata["training_class_weights"], expected_weights, "R4 current-source SVM class weights")
            audit.same(model.estimator.class_weight_, [expected_weights[c] for c in model.classes_], "R4 actual fitted SVM class weights")
            audit.same(model.estimator.C, inherited["svm"]["C"], "R4 inherited source C")
            audit.require(not model.estimator.break_ties, "R4 standard SVC vote retained")
        models[family] = model
    neural = common["neural_training"]
    audit.same(neural["source_samples"], len(source_ids), "CNN source-training sample count")
    audit.same(neural["epochs"], 200, "CNN fixed source epoch budget")
    audit.require(not neural["early_stopping"] and not neural["validation_or_test_labels_accepted"], "no outer/inner validation early stopping")
    counts = np.asarray([np.sum(y == c) for c in data["classes"]])
    # Torch's Python-scalar / tensor dispatch computes a float32 reciprocal
    # followed by scalar multiplication; direct float64 division then casting
    # can differ by one float32 ULP. Reproduce the declared arithmetic exactly.
    expected_weights = np.float32(len(y))*np.reciprocal(np.asarray(7*counts, dtype=np.float32))
    audit.same(neural["balanced_class_weights"], expected_weights.tolist(), "CNN float32 source category weights")
    return common, raw_predictions


def independent_family_choice(rows):
    means = {family: {metric: float(np.mean([r["metrics"][metric] for r in rows if r["family"] == family])) for metric in CLASS_METRICS} for family in FAMILIES}
    best = max(v["macro_f1"] for v in means.values()); near = [f for f in FAMILIES if means[f]["macro_f1"] >= best-1e-12]
    chosen = min(near, key=lambda f: (-means[f]["balanced_accuracy"], -means[f]["accuracy"], FAMILIES.index(f)))
    return {"selected": {"id": chosen, "family": chosen}, "all_candidate_means": means, "mean_inner_metrics": means[chosen],
        "selected_using_outer_test": False, "tie_rule": "macroF1 within1e-12, balanced_accuracy, accuracy, predeclared simpler family order"}


def audit_combinations_selection(audit, data):
    run = RUN / "combinations"
    if not (run / "selection_lock.json").exists():
        return {"status": "not_yet_locked"}
    manifest = check_manifest(audit, run)
    audit.same(manifest["core_selection_lock_sha256"], sha(RUN / "core/selection_lock.json"), "R4 frozen upstream source choices")
    lock = read(run / "selection_lock.json"); core = read(RUN / "core/selection_lock.json")
    all_rows = []; total_raw = 0
    for outer in range(5):
        inherited = inherited_for(core["outer_folds"][outer]); recorded = lock["outer_folds"][outer]
        audit.same(recorded["inherited_config"], inherited, "R4 inherits only same outer-fold prior stages")
        current = []
        for inner in range(2):
            directory = run / "selection" / f"outer_{outer:02d}" / f"inner_{inner:02d}"
            source_ids = data["splits"][outer]["inner_folds"][inner]["train_sample_ids"]
            query_ids = data["splits"][outer]["inner_folds"][inner]["validation_sample_ids"]
            checkpoint = read(directory / "evaluation_checkpoint.json")
            audit.same(sha(directory / "metrics.json"), checkpoint["metrics_sha256"], "R4 inner metrics SHA")
            audit.same(sha(directory / "predictions.csv.gz"), checkpoint["predictions_sha256"], "R4 inner raw predictions SHA")
            common, predicted = check_r4_caller(audit, directory, source_ids, query_ids, inherited, data)
            rows = read(directory / "metrics.json"); raw = read_predictions(audit, directory / "predictions.csv.gz", data)
            total_raw += len(raw); grouped = defaultdict(list)
            for r in raw:
                grouped[r["family"]].append(r)
            audit.same(len(rows), 7, "all seven source family controls present")
            for row in rows:
                family = row["family"]; values = grouped[family]
                audit.same([v["record_id"] for v in values], query_ids, "R4 family validation order")
                audit.same([v["y_predicted"] for v in values], predicted[family][0].tolist(), "R4 source model independent inference replay")
                scores = np.asarray([[float(v["score_"+c]) for c in data["classes"]] for v in values])
                audit.same(scores, predicted[family][1], "R4 saved source validation scores replay")
                expected = metrics([v["y_true"] for v in values], [v["y_predicted"] for v in values], data["classes"])
                check_metrics(audit, row["metrics"], expected, "R4 independent inner classification metrics")
            current.extend(rows); all_rows.extend(rows)
        audit.same(recorded["choice"], independent_family_choice(current), "R4 per-outer independent source family selection")
    audit.same(read(run / "all_inner_metrics.json"), all_rows, "R4 combined inner table exact source composition")
    return {"status": "pass", "inner_metric_cases": len(all_rows), "inner_prediction_rows": total_raw,
        "shared_CNN_source_contexts": 10, "source_models_replayed": 70,
        "per_outer_choices": [{"outer_fold": r["outer_fold"], "family": r["choice"]["selected"]["family"]} for r in lock["outer_folds"]]}


def check_core_outer_model(audit, model, config, source_ids, data):
    import joblib
    semantic = lambda c: {k: v for k, v in c.items() if k != "id"}
    audit.same(semantic(model.config), semantic(config), "outer source model exactly released configuration")
    X = data["X"][[data["lookup"][i] for i in source_ids]]; y = np.asarray([data["truth"][i] for i in source_ids])
    prepared, reference = independent_preprocessing(X, config["preprocessing"], data["axis"])
    audit.same(model.bundle.scaler.mean_, prepared.mean(0), "outer preprocessing/scaler only current684 source")
    audit.same(model.bundle.scaler.var_, prepared.var(0), "outer scaler variance only current684 source")
    if reference is not None:
        audit.same(model.bundle.preprocessor.reference_, reference, "outer MSC current source reference")
    selector = model.bundle.selector; coefficient = selector.coefficients_
    audit.same(coefficient.shape, (config["mc"]["runs"], 7, 612), "outer MC actual run count and onehot classes")
    mean, deviation = coefficient.mean(0), coefficient.std(0, ddof=1)
    stability = mean/np.maximum(deviation, max(1., float(deviation.max()))*1e-12)
    importance = np.linalg.norm(np.abs(stability), axis=0)
    ranking = np.argsort(-importance, kind="stable"); count = int(np.ceil(config["mc"]["ratio"]*612))
    indices = np.sort(ranking[:count])
    audit.same(selector.signed_stability_, stability, "outer MC stability independently recomputed")
    audit.same(selector.importances_, importance, "outer MC importance independently recomputed")
    audit.same(selector.selected_indices_, indices, "outer MC fixed-ratio variable mask")
    audit.same(model.metadata["selected_frequencies"], data["axis"][indices], "outer retained original frequencies")
    z = model.bundle.scaler.transform(prepared)[:, indices]
    scale = 1/(z.shape[1]*z.var()) if z.var() else 1.
    gamma = {"scale": scale, "auto": 1/z.shape[1], "scale_x0.1": .1*scale, "scale_x10": 10*scale}[config["svm"]["gamma"]]
    audit.same(model.estimator._gamma, gamma, "outer fitted SVC gamma on current selected source values")
    weights = independent_weights(y, config["svm"]["class_weight"])
    audit.same(model.estimator.class_weight_, [weights[c] for c in model.classes_], "outer fitted SVC source class weights")
    audit.require(not model.estimator.break_ties and model.estimator.decision_function_shape == "ovr", "outer standard SVC voting with separate OVR diagnostics")
    audit.same(model.estimator.C, config["svm"]["C"], "outer fitted source C")
    audit.same(model.metadata["source_value_fingerprint"], joblib.hash(X), "outer exact source spectral values")
    audit.same(model.metadata["source_label_fingerprint"], joblib.hash(y), "outer exact source class labels")


def audit_oof_table(audit, rows, summaries, fold_rows, data, description):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary_by_method = {r["method"]: r for r in summaries}
    audit.same(set(groups), set(summary_by_method), description + " all method summaries present")
    result = {}
    for method, values in groups.items():
        audit.same(len(values), 855, description + " 855 predictions per method")
        audit.same(sorted(r["record_id"] for r in values), data["ids"].tolist(), description + " exactly one OOF per physical sample")
        expected = metrics([r["y_true"] for r in values], [r["y_predicted"] for r in values], data["classes"])
        check_metrics(audit, summary_by_method[method], expected, description + " pooled 855 OOF metrics")
        fold_values = [r for r in fold_rows if r["method"] == method]
        audit.same(len(fold_values), 5, description + " all five outer folds")
        if "fold_macro_f1_sd" in summary_by_method[method]:
            audit.same(summary_by_method[method]["fold_macro_f1_sd"], float(np.std([r["macro_f1"] for r in fold_values], ddof=1)), "descriptive fold macro-F1 SD")
        if "fold_mean" in summary_by_method[method]:
            audit.same(summary_by_method[method]["fold_mean"], {k: float(np.mean([r[k] for r in fold_values])) for k in CLASS_METRICS}, "fold means separate from pooled F1")
            audit.same(summary_by_method[method]["fold_sd"], {k: float(np.std([r[k] for r in fold_values], ddof=1)) for k in CLASS_METRICS}, "descriptive fold SD, not batch CI")
        result[method] = {k: expected[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "correct", "total")}
    return result


def audit_core_evaluation(audit, data):
    import joblib
    run = RUN / "core"; raw = read_predictions(audit, run / "outer_predictions.csv.gz", data)
    rows = read(run / "outer_metrics.json"); lock = read(run / "selection_lock.json")
    grouped = defaultdict(list)
    for r in raw:
        grouped[(int(r["outer_fold"]), r["method"])].append(r)
    model_cache = {}; replay_count = 0; r0_predictions = {}
    for row in rows:
        outer, method = row["outer_fold"], row["method"]
        source_ids = data["splits"][outer]["train_sample_ids"]; test_ids = data["splits"][outer]["test_sample_ids"]
        values = grouped[(outer, method)]; choices = lock["outer_folds"][outer]["choices"]
        audit.same([v["record_id"] for v in values], test_ids, "each outer model scores its own171heldout specimens")
        if method in ("R0", "R1", "R2", "R3"):
            expected_config = choices[method]["selected"]
        elif method.startswith("R3_diagnostic_"):
            prep = method.removeprefix("R3_diagnostic_")
            expected_config = {**choices["R3"]["selected"], "id": f"prep{PREPROCESSORS.index(prep):02d}_{prep}", "preprocessing": prep}
        else:
            audit.same(method, "fixed_snv_C10_MC20", "only prespecified core final diagnostic role")
            expected_config = {"id": "fixed_snv_C10_MC20", "preprocessing": "snv", "mc": {"runs": 20, "components": 2, "ratio": .1},
                "svm": {"C": 10., "gamma": "scale", "class_weight": "balanced"}}
        audit.same(row["config"], expected_config, "outer method bound to its own inner selected parameters")
        key = row["model_path"]; path = ROOT / key
        audit.same(sha(path), row["model_sha256"], "core outer trained model bytes")
        if key not in model_cache:
            model = joblib.load(path)
            check_core_outer_model(audit, model, row["config"], source_ids, data)
            model_cache[key] = model
        model = model_cache[key]; Xtest = data["X"][[data["lookup"][i] for i in test_ids]]
        before = joblib.hash(model); predicted = model.predict(Xtest); scores = model.predict_scores(Xtest)
        audit.same(joblib.hash(model), before, "outer core source frozen during inference")
        audit.same(predicted.tolist(), [v["y_predicted"] for v in values], "outer core hard prediction replay")
        recorded = np.asarray([[float(v["score_"+c]) for c in data["classes"]] for v in values])
        audit.same(recorded, scores, "outer core OVR score replay")
        audit.same(row["selected_indices"], model.bundle.selector.selected_indices_, "outer selected-variable metadata binding")
        check_metrics(audit, row, metrics([v["y_true"] for v in values], [v["y_predicted"] for v in values], data["classes"]), "outer core fold metrics")
        audit.same(row["vote_argmax_disagreements"], int(np.sum(predicted != np.asarray(data["classes"])[scores.argmax(1)])), "outer standard vote/score argmax mismatch count")
        if method == "R0":
            r0_predictions.update({v["record_id"]: v for v in values})
        replay_count += 1
    legacy = read_predictions(audit, HISTORICAL_PREDICTIONS, data)
    for row in legacy:
        identifier = row["sample_id"]
        audit.same(r0_predictions[identifier]["y_predicted"], row["y_predicted"], "historical exact R0 prediction reproduction")
        audit.same(int(r0_predictions[identifier]["outer_fold"]), int(row["outer_fold"]), "historical exact fold assignment reproduction")
    audit.same(sum(v["y_true"] == v["y_predicted"] for v in r0_predictions.values()), 773, "historical773of855 baseline exactly reproduced")
    score_argmax_correct = sum(row["y_true"] == data["classes"][int(np.argmax([float(row["score_"+c]) for c in data["classes"]]))]
                               for row in r0_predictions.values())
    score_vote_differences = sum(row["y_predicted"] != data["classes"][int(np.argmax([float(row["score_"+c]) for c in data["classes"]]))]
                                 for row in r0_predictions.values())
    audit.same(score_argmax_correct, 772, "OVR argmax is a separate lower-correct baseline diagnostic")
    summaries = audit_oof_table(audit, raw, read(run / "summary.json"), rows, data, "core")
    return {"status": "pass", "outer_prediction_rows": len(raw), "outer_method_fold_cases": len(rows),
        "unique_outer_source_models": len(model_cache), "source_prediction_replays": replay_count,
        "historical_R0_predictions_identical": 855, "historical_standard_vote_correct": 773,
        "historical_OVR_argmax_correct": score_argmax_correct, "historical_vote_argmax_different_predictions": score_vote_differences,
        "pooled_oof_results": summaries}, raw


def audit_combinations_evaluation(audit, data, core_raw):
    run = RUN / "combinations"; raw = read_predictions(audit, run / "outer_predictions.csv.gz", data)
    rows = read(run / "outer_metrics.json"); lock = read(run / "selection_lock.json")
    core_lock = read(RUN / "core/selection_lock.json"); grouped = defaultdict(list)
    for row in raw:
        grouped[(int(row["outer_fold"]), row["method"])].append(row)
    source_models = 0
    for outer in range(5):
        inherited = inherited_for(core_lock["outer_folds"][outer])
        source_ids = data["splits"][outer]["train_sample_ids"]; test_ids = data["splits"][outer]["test_sample_ids"]
        common, predicted = check_r4_caller(audit, run / "evaluation" / f"outer_{outer:02d}", source_ids, test_ids, inherited, data)
        source_models += 7
        for row in [r for r in rows if r["outer_fold"] == outer]:
            family = row["family"]; method = row["method"]
            expected_family = lock["outer_folds"][outer]["choice"]["selected"]["family"] if method == "R4" else method.removeprefix("R4_diagnostic_")
            audit.same(family, expected_family, "R4 outer role only its own source inner family choice")
            audit.same(row["config"], inherited, "R4 outer source inherited stage1/2/3 parameters")
            values = grouped[(outer, method)]
            audit.same([v["record_id"] for v in values], test_ids, "R4 outer171heldout sample order")
            audit.same([v["y_predicted"] for v in values], predicted[family][0].tolist(), "R4 outer source model class replay")
            scores = np.asarray([[float(v["score_"+c]) for c in data["classes"]] for v in values])
            audit.same(scores, predicted[family][1], "R4 outer source model score replay")
            check_metrics(audit, row, metrics([v["y_true"] for v in values], [v["y_predicted"] for v in values], data["classes"]), "R4 outer fold metrics")
    core_reference = {r["record_id"]: r for r in core_raw if r["method"] == "R3"}
    ordinary = {r["record_id"]: r for r in raw if r["method"] == "R4_diagnostic_ordinary_mcuve"}
    for identifier in data["ids"]:
        audit.same(ordinary[identifier]["y_predicted"], core_reference[identifier]["y_predicted"], "R4 ordinary control exactly matches inherited R3 predictor")
        for c in data["classes"]:
            audit.same(float(ordinary[identifier]["score_"+c]), float(core_reference[identifier]["score_"+c]), "R4 ordinary matched control OVR score equality")
    summaries = audit_oof_table(audit, raw, read(run / "summary.json"), rows, data, "R4")
    return {"status": "pass", "outer_prediction_rows": len(raw), "outer_method_fold_cases": len(rows),
        "newly_fitted_outer_CNN_contexts": 5, "source_models_replayed": source_models,
        "ordinary_control_predictions_identical_to_R3": 855, "pooled_oof_results": summaries}, raw


def audit_release(audit):
    path = RUN / "outer_evaluation_release.json"; release = read(path)
    moment = release.get("released_at_utc", release.get("released_at"))
    audit.require(isinstance(moment, str), "actual outer-evaluation release UTC")
    released = datetime.fromisoformat(moment.replace("Z", "+00:00")); records = []
    for item in release["rounds"]:
        lock = ROOT / item["selection_lock"]; run = lock.parent
        audit.same(sha(lock), item["selection_lock_sha256"], "global released source choice lock")
        audit.same(sha(ROOT / item["protocol"]), item["protocol_sha256"], "global released source protocol")
        audit.same(sha(run / "run_manifest.json"), item["run_manifest_sha256"], "global released run manifest")
        events = [json.loads(line) for line in (run / "stage_events.jsonl").read_text().splitlines() if line.strip()]
        selection = [e for e in events if e.get("stage") == "selection" and e.get("event") in ("start", "stage_start")]
        evaluation = [e for e in events if e.get("stage") == "evaluation" and e.get("event") in ("start", "stage_start")]
        audit.require(bool(selection) and bool(evaluation), "recorded selection and outer-evaluation actual starts")
        audit.require(datetime.fromisoformat(selection[0]["utc"].replace("Z", "+00:00")) <= released, "source selection starts before global release")
        for event in evaluation:
            audit.require(datetime.fromisoformat(event["utc"].replace("Z", "+00:00")) >= released, "every outer-evaluation start after all5fold locks release")
        records.append({"name": item.get("name"), "selection_start_utc": selection[0]["utc"], "outer_evaluation_start_utc": evaluation[0]["utc"]})
    return {"status": "pass", "release_sha256": sha(path), "released_at_utc": moment, "rounds": records,
        "execution_record_not_external_attestation": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("selection", "final"), default="selection")
    args = parser.parse_args(); started = time.monotonic(); audit = Audit()
    data = load_data(audit)
    core = audit_core_selection(audit, data)
    combinations = audit_combinations_selection(audit, data)
    outer = {}
    if args.phase == "final":
        outer["core"], core_predictions = audit_core_evaluation(audit, data)
        outer["combinations"], combination_predictions = audit_combinations_evaluation(audit, data, core_predictions)
        outer["release"] = audit_release(audit)
    result = {"status": "pass" if core["status"] == combinations["status"] == "pass" else "partial_pass", "phase": args.phase, "core": core, "combinations": combinations, "checks_evaluated": audit.checks,
        "maximum_numeric_difference": audit.maximum_numeric_difference, "new_research_model_fits": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.monotonic()-started,
        "audit_script_sha256": sha(__file__)}
    if outer:
        result["outer_evaluation"] = outer
        result["outer_prediction_rows"] = outer["core"]["outer_prediction_rows"] + outer["combinations"]["outer_prediction_rows"]
    result["inner_prediction_rows"] = core.get("inner_prediction_rows", 0) + combinations.get("inner_prediction_rows", 0)
    result["limits"] = ["No new research fitting, source optimization or outer-score-based parameter selection in the independent audit.",
        "MC prefixes are checked from actual saved coefficient streams; independent .fit equivalence is separately covered by frozen implementation tests.",
        "Ordinary/budget/stability feature selectors match K; CNN controls have independent full612input/32dimembedding architecture and loss.",
        "Source resampling stability is not independent batch validation; MC-round budget controls do not imply equal FLOPs.",
        "Every method's855OOF records are pooled across5outerfolds; fixed171holdout results are not ranked as the same test set.",
        "All855samples and legacy outer results have historical exposure; retrospective single-batch anonymous-origin internal validation only.",
        "Run events and source/state hashes are recorded evidence, not external execution attestation."]
    (RUN / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
