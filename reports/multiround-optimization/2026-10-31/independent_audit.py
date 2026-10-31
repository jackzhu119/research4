"""Independent arithmetic/lineage audit of the four locked research rounds.

No project fitting, metric or selection helper is imported. This script reads
saved records, computes seven-class confusion counts directly and checks the
recorded choices. Saved execution assertions are not external attestation.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from itertools import permutations
import argparse
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/multiround-20261031"
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
CHECKS = 0
MAX_DIFFERENCE = 0.0


def check(condition, detail):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(detail)


def close(actual, expected, detail, atol=2e-12):
    global MAX_DIFFERENCE
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    MAX_DIFFERENCE = max(MAX_DIFFERENCE, difference)
    check(difference <= atol, (detail, actual, expected, difference))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    confusion = np.bincount(actual * 7 + predicted, minlength=49).reshape(7, 7)
    true_counts = confusion.sum(axis=1)
    prediction_counts = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, true_counts, out=np.zeros(7, dtype=float), where=true_counts > 0)
    f1 = np.divide(2 * diagonal, true_counts + prediction_counts,
                   out=np.zeros(7, dtype=float), where=(true_counts + prediction_counts) > 0)
    return {"accuracy": float(diagonal.sum() / len(actual)),
            "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
            "confusion_matrix": confusion.tolist(), "correct": int(diagonal.sum()),
            "total": int(len(actual)), "recall": dict(zip(CLASSES, recall.tolist()))}


def prediction_cases(path, truth_lookup, allowed_ids):
    """Return cases keyed only by recorded non-feature evaluation dimensions."""
    cases = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        dimensions = [field for field in ("episode", "stage", "seed", "strategy", "repeat", "method", "candidate_id")
                      if field in reader.fieldnames]
        for row in reader:
            identifier = row["record_id"]
            actual, predicted = row["y_true"], row["y_predicted"]
            check(identifier in allowed_ids, (str(path), "unallowed prediction ID", identifier))
            check(actual == truth_lookup[identifier], (str(path), "wrong recorded truth", row))
            check(predicted in CLASS_INDEX, (str(path), "unknown prediction", row))
            check(int(row["is_correct"]) == int(actual == predicted), (str(path), "wrong correct flag", row))
            key = tuple(row[field] for field in dimensions)
            cases[key].append((identifier, CLASS_INDEX[actual], CLASS_INDEX[predicted]))
    result = {}
    for key, records in cases.items():
        records.sort()
        identifiers = [r[0] for r in records]
        check(len(identifiers) == len(set(identifiers)), (str(path), "duplicate ID in case", key))
        check(set(identifiers) == set(allowed_ids), (str(path), "incomplete prediction case", key))
        actual, predicted = [r[1] for r in records], [r[2] for r in records]
        result[key] = {"dimensions": dict(zip(dimensions, key)), "ids": identifiers,
                       "actual": actual, "predicted": predicted, "metrics": direct_metrics(actual, predicted)}
    return result


def verify_case_metrics(path, cases, rows):
    lookup = {tuple((k, v) for k, v in value["dimensions"].items()): value for value in cases.values()}
    for row in rows:
        dimensions = next(iter(cases.values()))["dimensions"]
        if all(name in row for name in dimensions):
            candidate = lookup.get(tuple((name, str(row[name])) for name in dimensions))
            matches = [candidate] if candidate is not None else []
        else:
            matches = [value for value in cases.values()
                       if all(str(row[name]) == value["dimensions"][name]
                              for name in value["dimensions"] if name in row)]
        check(len(matches) == 1, (str(path), "metric row must match one saved case", row, len(matches)))
        independently = matches[0]["metrics"]
        for metric in METRICS:
            close(row[metric], independently[metric], (str(path), metric, matches[0]["dimensions"]))
        for metric in ("correct", "total", "confusion_matrix"):
            if metric in row:
                check(row[metric] == independently[metric], (str(path), metric, row[metric], independently[metric]))


def audit_provenance(directory):
    manifest = read_json(directory / "run_manifest.json")
    for field in ("source_sha256", "source_code_sha256"):
        for relative, digest in manifest.get(field, {}).items():
            check(sha256(ROOT / relative) == digest, (str(directory), "changed algorithm source", relative))
    check(sha256(ROOT / "data/processed/zaoren_threeway_20261031_v1/split_plan.json")
          == manifest.get("plan_sha256", manifest.get("split_plan_sha256")), (str(directory), "split hash"))
    check(sha256(ROOT / "outputs/approved-zaoren-20261031/data-preparation/dataset.npz")
          == manifest["input_sha256"], (str(directory), "input hash"))
    check(sha256(directory / "protocol_snapshot.json") == manifest["protocol_sha256"],
          (str(directory), "protocol snapshot hash"))
    return manifest


def _source_rank(candidates, means, tolerance):
    peak = max(means[c["id"]]["accuracy"] for c in candidates)
    eligible = [c for c in candidates if peak - means[c["id"]]["accuracy"] <= tolerance]
    return min(eligible, key=lambda c: (-means[c["id"]]["macro_f1"],
        -means[c["id"]]["balanced_accuracy"], len(c["hidden_sizes"]), c["epochs"],
        c["pca_components"] or 10**6, int(c["class_weight"] == "balanced"), c["id"]))


def audit_source_selection(directory):
    protocol = read_json(directory / "protocol_snapshot.json")
    grid = read_json(directory / "candidate_grid.json")
    rows = read_json(directory / "all_development_metrics.json")
    recorded = read_json(directory / "selection_lock.json")["choice"]
    means = {stage: {} for stage in ("selection", "gate")}
    for stage in means:
        for candidate in grid:
            group = [r for r in rows if r["stage"] == stage and r["candidate_id"] == candidate["id"]]
            check(len(group) == (4 if stage == "selection" else 2)*3, ("source selection group", stage, candidate["id"]))
            means[stage][candidate["id"]] = {m: float(np.mean([r[m] for r in group])) for m in METRICS}
            for metric in METRICS:
                close(recorded["means"][stage][candidate["id"]][metric], means[stage][candidate["id"]][metric],
                      ("source dev aggregation", stage, candidate["id"], metric))
    settings = protocol["selection"]
    proposed = _source_rank(grid, means["selection"], settings["accuracy_tolerance"])
    reference = next(c for c in grid if c["id"] == settings["reference_candidate_id"])
    def passes(stage):
        candidate, baseline = means[stage][proposed["id"]], means[stage][reference["id"]]
        return (candidate["accuracy"] - baseline["accuracy"] > settings["minimum_accuracy_gain"]
            and candidate["macro_f1"] - baseline["macro_f1"] >= -settings["max_macro_f1_loss"]
            and candidate["balanced_accuracy"] - baseline["balanced_accuracy"] >= -settings["max_balanced_accuracy_loss"])
    passed = passes("selection") and passes("gate")
    selected = proposed if passed else reference
    check(recorded["proposed"] == proposed, "source proposed candidate")
    check(recorded["selected"] == selected, "source selected candidate")
    check(recorded["passed_selection_and_gate"] == passed, "source gate decision")
    for family in ("none", "balanced"):
        family_choice = _source_rank([c for c in grid if c["class_weight"] == family], means["selection"], settings["accuracy_tolerance"])
        check(recorded["family_diagnostics"][family] == family_choice, ("source family", family))
    return {"selected": selected["id"], "proposed": proposed["id"], "passed": passed,
            "development_rows": len(rows)}


def independent_partitions(training_ids, truth, count, seed=20261103):
    """Reconstruct fixed source/support/query partitions without project code."""
    ids = np.asarray(training_ids)
    labels = np.asarray([truth[i] for i in ids])
    small_orders = list(permutations(range(3)))
    result = []
    for episode in range(count):
        rng = np.random.default_rng(seed + episode)
        current = {role: [] for role in ("source", "support_pool", "query")}
        for label in CLASSES:
            rows = np.flatnonzero(labels == label)
            rows = rows[np.argsort(ids[rows])]
            order = rows[list(small_orders[episode % 6])] if len(rows) == 3 else rng.permutation(rows)
            source_count, pool_count = max(1, int(.6*len(rows))), max(1, int(.2*len(rows)))
            current["source"].extend(order[:source_count])
            current["support_pool"].extend(order[source_count:source_count+pool_count])
            current["query"].extend(order[source_count+pool_count:])
        result.append({name: ids[np.asarray(sorted(values), dtype=int)].tolist() for name, values in current.items()})
    return result


def check_draws(draws, pool_ids, truth, repeats, seed):
    rng = np.random.default_rng(seed)
    pools = {label: np.asarray([i for i, identifier in enumerate(pool_ids) if truth[identifier] == label])
             for label in CLASSES}
    check(len(draws) == repeats, "support draw count")
    for repeat, draw in enumerate(draws):
        expected = [pool_ids[int(rng.permutation(pools[label])[0])] for label in CLASSES]
        identifiers = draw.get("support_ids", draw.get("ids"))
        check(draw["repeat"] == repeat, "support draw repeat")
        check(identifiers == expected, ("fixed PCG support draw", repeat))
        check(len(set(identifiers)) == 7 and {truth[i] for i in identifiers} == set(CLASSES), "seven distinct known support specimens")


def audit_svm_selection(directory):
    protocol = read_json(directory / "protocol_snapshot.json")
    grid = read_json(directory / "candidate_grid.json")
    rows = read_json(directory / "all_development_metrics.json")
    saved = read_json(directory / "development_aggregate.json")
    aggregate = {stage: {} for stage in ("selection", "gate")}
    for stage in aggregate:
        for candidate in grid:
            group = [r for r in rows if r["stage"] == stage and r["candidate_id"] == candidate["id"]]
            check(len(group) == (24 if stage == "selection" else 6)*20, ("SVM group length", stage, candidate["id"]))
            aggregate[stage][candidate["id"]] = {metric: float(np.mean([r[metric] for r in group]))
                for metric in (*METRICS, *("matched_"+m for m in METRICS))}
            for metric, value in aggregate[stage][candidate["id"]].items():
                close(saved[stage][candidate["id"]][metric], value, ("SVM aggregate", stage, candidate["id"], metric))
    settings = protocol["selection"]
    identity = grid[0]
    def passes(candidate, stage):
        value, baseline = aggregate[stage][candidate["id"]], aggregate[stage][identity["id"]]
        return all(value["accuracy"] - comparator["accuracy"] > settings["minimum_accuracy_gain"]
                   and value["macro_f1"] - comparator["macro_f1"] >= -settings["max_macro_f1_loss"]
                   and value["balanced_accuracy"] - comparator["balanced_accuracy"] >= -settings["max_balanced_accuracy_loss"]
                   for comparator in (baseline, {m: value["matched_"+m] for m in METRICS}))
    def best(candidates):
        peak = max(aggregate["selection"][c["id"]]["accuracy"] for c in candidates)
        return min((c for c in candidates if aggregate["selection"][c["id"]]["accuracy"] >= peak-settings["tie_tolerance"]),
                   key=lambda c: (c["blend"], c["support_weight"], c["id"]))
    eligible = [c for c in grid if passes(c, "selection")]
    proposed = best(eligible) if eligible else identity
    selected = proposed if proposed != identity and passes(proposed, "gate") else identity
    diagnostic = best(grid[1:])
    recorded = read_json(directory / "selection_lock.json")["choice"]
    for name, value in (("selected", selected), ("proposed", proposed), ("diagnostic", diagnostic)):
        check(recorded[name] == value, ("SVM independent choice", name))
    check(recorded["eligible_selection_candidates"] == len(eligible), "SVM eligible count")
    return {"selected": selected["id"], "proposed": proposed["id"], "diagnostic": diagnostic["id"],
            "eligible_selection_candidates": len(eligible), "development_rows": len(rows)}


def audit_svm_development(directory, truth, training_ids):
    import joblib
    audit_provenance(directory)
    parts = independent_partitions(training_ids, truth, 30)
    check(read_json(directory / "development_partitions.json") == parts, "SVM development partition regeneration")
    rows = 0
    for episode, partition in enumerate(parts):
        current = directory / "development" / f"episode_{episode:02d}"
        meta = read_json(current / "metadata.json")
        old_source = ROOT/"outputs/representation-optimization-20261031/source/development/s010_snv_svc"/f"episode_{episode:02d}.joblib"
        check(sha256(old_source) == meta["source_model_sha256"], "actually reused SVM source cache bytes")
        original_model = joblib.load(old_source)
        check(joblib.hash(original_model) == meta["source_fingerprint_before"], "actually reused SVM source object fingerprint")
        close(meta["fixed_parameters"]["gamma"], original_model.estimator_._gamma, "frozen source numeric SVM gamma")
        check(meta["fixed_parameters"]["C"] == 10., "fixed source SVM C")
        check(meta["source_ids"] == partition["source"], ("SVM source IDs", episode))
        check(meta["support_pool_ids"] == partition["support_pool"], ("SVM support pool IDs", episode))
        check(meta["query_ids"] == partition["query"], ("SVM query IDs", episode))
        for label in CLASSES:
            close(meta["fixed_parameters"]["class_weight"][label],
                  len(partition["source"])/(7*sum(truth[i] == label for i in partition["source"])),
                  "SVM class weights use source counts only")
        check(meta["source_fingerprint_before"] == meta["source_fingerprint_after"], ("SVM source unchanged", episode))
        check(meta["context_fingerprint_before"] == meta["context_fingerprint_after"], ("SVM context unchanged", episode))
        close(meta["zero_refit_max_score_difference"], 0., ("SVM zero score difference", episode), atol=1e-9)
        check_draws(meta["draws"], partition["support_pool"], truth, 20, 20261104+episode)
        prediction_path = current / "predictions.csv.gz"
        check(sha256(prediction_path) == meta["predictions_sha256"], ("SVM prediction SHA", episode))
        cases = prediction_cases(prediction_path, truth, partition["query"])
        metrics = read_json(current / "metrics.json")
        check(len(metrics) == 420 and len(cases) == 420, ("SVM case count", episode))
        verify_case_metrics(prediction_path, cases, metrics)
        identities = {case["dimensions"]["repeat"]: case for case in cases.values()
                      if case["dimensions"]["candidate_id"] == "w000_identity"}
        for row in metrics:
            own = identities[str(row["repeat"])]
            case = next(case for case in cases.values()
                        if case["dimensions"]["repeat"] == str(row["repeat"])
                        and case["dimensions"]["candidate_id"] == row["candidate_id"])
            for metric in METRICS:
                close(row["matched_"+metric], own["metrics"][metric], ("SVM zero control", episode, metric))
                close(row[metric+"_gain_vs_source"], case["metrics"][metric]-own["metrics"][metric], ("SVM source gain", episode, metric))
            actual, predicted, source_predicted = np.asarray(case["actual"]), np.asarray(case["predicted"]), np.asarray(own["predicted"])
            check(row["corrected_errors"] == int(np.sum((source_predicted != actual)&(predicted == actual))), "SVM corrected count")
            check(row["new_errors"] == int(np.sum((source_predicted == actual)&(predicted != actual))), "SVM new error count")
        rows += sum(len(case["ids"]) for case in cases.values())
    return {"prediction_rows": rows, "cases": 12600, "choice": audit_svm_selection(directory)}


def direct_state_fingerprint(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(tensor.dtype).encode())
        result.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def dense_scores_from_saved_state(state, features):
    """Replay dense inference directly from tensors, without project classes."""
    import torch
    torch.set_num_threads(1)
    tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    layers = sorted({int(name.split(".")[1]) for name in state if name.startswith("encoder.")})
    with torch.no_grad():
        for layer in layers:
            tensor = torch.relu(torch.nn.functional.linear(tensor, state[f"encoder.{layer}.weight"], state[f"encoder.{layer}.bias"]))
        return torch.nn.functional.linear(tensor, state["head.weight"], state["head.bias"]).numpy()


def dense_prediction_from_saved_state(state, features):
    return dense_scores_from_saved_state(state, features).argmax(1)


def direct_transform(preprocessor, raw):
    snv = (raw-raw.mean(axis=1, keepdims=True))/raw.std(axis=1, ddof=1, keepdims=True)
    transformed = (snv-preprocessor["scaler"].mean_)/preprocessor["scaler"].scale_
    if preprocessor["pca"] is not None:
        transformed = (transformed-preprocessor["pca"].mean_) @ preprocessor["pca"].components_.T
    return np.asarray(transformed, dtype=np.float32)


def audit_neural_development(directory, truth, training_ids):
    import joblib
    import torch
    audit_provenance(directory)
    protocol = read_json(directory / "protocol_snapshot.json")
    parts = independent_partitions(training_ids, truth, 6)
    check(read_json(directory / "development_partitions.json") == parts, "MLP independent development partitions")
    grid = read_json(directory / "candidate_grid.json")
    selected = read_json(directory / "selection_lock.json")["choice"]["selected"]["id"]
    with np.load(ROOT/"outputs/approved-zaoren-20261031/data-preparation/dataset.npz", allow_pickle=False) as data:
        row_lookup = {str(i): index for index, i in enumerate(data["record_ids"])}
        mask = (data["axis"] >= .1)&(data["axis"] <= 1.5)
        check(int(mask.sum()) == 612, "MLP fixed frequency count")
        raw = data["X_alpha"][:, mask]
    prediction_rows = 0
    source_replays = 0
    for episode, partition in enumerate(parts):
        source_raw = raw[[row_lookup[i] for i in partition["source"]]]
        query_raw = raw[[row_lookup[i] for i in partition["query"]]]
        source_snv = (source_raw-source_raw.mean(axis=1,keepdims=True))/source_raw.std(axis=1,ddof=1,keepdims=True)
        source_truth = [CLASS_INDEX[truth[i]] for i in partition["source"]]
        for seed in protocol["seeds"]:
            for candidate in grid:
                current = directory/"development"/f"episode_{episode:02d}"/f"seed_{seed}"/candidate["id"]
                meta = read_json(current/"training.json")
                check(meta["source_ids"] == partition["source"], ("MLP training IDs", episode, seed, candidate["id"]))
                check(meta["candidate"] == candidate and meta["seed"] == seed, "MLP candidate identity")
                check(meta["classes"] == list(CLASSES), "MLP all-seven-class vocabulary")
                check(sha256(current/"model.pt") == meta["model_sha256"], "MLP saved state SHA")
                check(sha256(current/"preprocessor.joblib") == meta["preprocessor_sha256"], "MLP preprocessing SHA")
                state = torch.load(current/"model.pt", map_location="cpu", weights_only=True)
                check(direct_state_fingerprint(state) == meta["fingerprint"], "MLP independent state fingerprint")
                check(meta["source_samples"] == 307 and len(meta["losses"]) == candidate["epochs"], "MLP source size and epoch count")
                preprocessor = joblib.load(current/"preprocessor.joblib")
                check(preprocessor["scaler"].n_samples_seen_ == 307, "MLP scaler source row count")
                close(preprocessor["scaler"].mean_, source_snv.mean(axis=0), "MLP source-only scaler mean")
                close(preprocessor["scaler"].var_, source_snv.var(axis=0), "MLP source-only scaler variance")
                if preprocessor["pca"] is not None:
                    pca = preprocessor["pca"]
                    check(pca.n_samples_ == 307 and pca.n_components_ == candidate["pca_components"], "MLP source-only PCA shape")
                    close(pca.components_@pca.components_.T, np.eye(pca.n_components_), "MLP PCA orthonormal basis")
                prediction_path = current/"predictions.csv.gz"
                saved = read_json(current/"metrics.json")
                check(sha256(prediction_path) == saved["prediction_sha256"], "MLP prediction SHA")
                cases = prediction_cases(prediction_path, truth, partition["query"])
                check(len(cases) == 1, "MLP one developer prediction case per fit")
                verify_case_metrics(prediction_path, cases, [saved])
                prediction_rows += 103
                if candidate["id"] == selected:
                    source_prediction = dense_prediction_from_saved_state(state, direct_transform(preprocessor, source_raw))
                    metrics = direct_metrics(source_truth, source_prediction)
                    for metric in METRICS:
                        close(meta["source_metrics"][metric], metrics[metric], ("MLP source inference replay", metric))
                    query_prediction = dense_prediction_from_saved_state(state, direct_transform(preprocessor, query_raw))
                    case = next(iter(cases.values()))
                    observed = dict(zip(partition["query"], query_prediction.tolist()))
                    check([observed[i] for i in case["ids"]] == case["predicted"], "MLP selected-source query inference replay")
                    source_replays += 1
    return {"prediction_rows": prediction_rows, "source_models_hashed": 432,
            "source_inference_replays_without_fit": source_replays, "cases": 432,
            "choice": audit_source_selection(directory)}


def audit_anchored_selection(directory):
    protocol = read_json(directory / "protocol_snapshot.json")
    grid = read_json(directory / "candidate_grid.json")
    rows = read_json(directory / "all_development_metrics.json")
    recorded = read_json(directory / "selection_lock.json")["choice"]
    means = {}
    for candidate in grid:
        for stage in ("selection", "gate"):
            group = [r for r in rows if r["candidate_id"] == candidate["id"] and r["stage"] == stage]
            check(len(group) == (4 if stage == "selection" else 2)*3*5, "anchored development group size")
            means[(candidate["id"], stage)] = {metric: float(np.mean([r[metric] for r in group]))
                for metric in (*METRICS, *("source_"+m for m in METRICS), *("matched_"+m for m in METRICS))}
    ranked = sorted(grid, key=lambda c: (-means[(c["id"], "selection")]["accuracy"],
        c["learning_rate"], c["steps"], c["source_replay_weight"], c["proximal_lambda"], c["id"]))
    policy = protocol["selection"]
    def passes(candidate, stage):
        value = means[(candidate["id"], stage)]
        return all(value["accuracy"]-value[prefix+"accuracy"] > policy["minimum_accuracy_gain"]
                   and value["macro_f1"]-value[prefix+"macro_f1"] >= -policy["max_macro_f1_loss"]
                   and value["balanced_accuracy"]-value[prefix+"balanced_accuracy"] >= -policy["max_balanced_accuracy_loss"]
                   for prefix in ("source_", "matched_"))
    eligible = [c for c in ranked if passes(c, "selection")]
    diagnostic = eligible[0] if eligible else ranked[0]
    selected = diagnostic if eligible and passes(diagnostic, "gate") else None
    check(recorded["selected"] == selected, "anchored independently selected candidate")
    check(recorded["diagnostic"] == diagnostic, "anchored independently diagnostic candidate")
    check(recorded["selection_eligible_candidates"] == len(eligible), "anchored eligible count")
    for stage in ("selection", "gate"):
        for metric, value in means[(diagnostic["id"], stage)].items():
            close(recorded[stage][metric], value, ("anchored diagnostic aggregation", stage, metric))
    return {"selected": None if selected is None else selected["id"], "diagnostic": diagnostic["id"],
            "eligible_candidates": len(eligible), "development_rows": len(rows)}


def audit_anchored_development(directory, truth, training_ids):
    audit_provenance(directory)
    parts = independent_partitions(training_ids, truth, 6)
    base_choice = read_json(RUN/"neural-base/selection_lock.json")["choice"]["selected"]["id"]
    rows = 0; count = 0
    for episode, partition in enumerate(parts):
        for seed in (42, 43, 44):
            current = directory/"development"/f"episode_{episode:02d}"/f"seed_{seed}"
            record = read_json(current/"record.json")
            for name, source_name in (("source", "source"), ("support_pool", "support_pool"), ("prediction", "query")):
                check(record["role_ids"][name] == partition[source_name], ("anchored role IDs", episode, seed, name))
            check(record["source_fingerprint_before"] == record["source_fingerprint_after"], "anchored source unchanged")
            original = read_json(RUN/"neural-base/development"/f"episode_{episode:02d}"/f"seed_{seed}"/base_choice/"training.json")
            check(record["source_fingerprint_before"] == original["fingerprint"], "anchored actual selected source fingerprint")
            check_draws(record["support_draws"], partition["support_pool"], truth, 5, 20261104+episode)
            check(sha256(current/"metrics.json") == record["metrics_sha256"], "anchored developer metric SHA")
            check(sha256(current/"predictions.csv.gz") == record["prediction_sha256"], "anchored developer prediction SHA")
            cases = prediction_cases(current/"predictions.csv.gz", truth, partition["query"])
            metrics = read_json(current/"metrics.json")
            check(len(metrics) == 120 and len(cases) == 145, "anchored developer cases include all controls")
            verify_case_metrics(current/"predictions.csv.gz", cases, metrics)
            zero = next(c for c in cases.values() if c["dimensions"]["method"] == "source_zero")
            matched = {c["dimensions"]["candidate_id"]: c for c in cases.values() if c["dimensions"]["method"] == "matched_zero"}
            for row in metrics:
                for metric in METRICS:
                    close(row["source_"+metric], zero["metrics"][metric], "anchored source baseline arithmetic")
                    close(row["matched_"+metric], matched[row["candidate_id"]]["metrics"][metric], "anchored matched control arithmetic")
            rows += sum(len(c["ids"]) for c in cases.values()); count += len(metrics)
    return {"prediction_rows": rows, "adapted_cases": count, "developer_contexts": 18,
            "choice": audit_anchored_selection(directory)}


def audit_meta_selection(directory):
    protocol = read_json(directory/"protocol_snapshot.json")
    grid = read_json(directory/"candidate_grid.json")
    rows = read_json(directory/"all_development_metrics.json")
    recorded = read_json(directory/"selection_lock.json")
    metas = {f"m{i:03d}": {"updates": updates, "inner_learning_rate": rate}
             for i, (updates, rate) in enumerate((u, r) for u in (100, 500) for r in (.001, .01))}
    means = {}
    for strategy in ("fomaml", "supervised"):
        for candidate in grid:
            for stage in ("selection", "gate"):
                group = [r for r in rows if r["strategy"] == strategy and r["candidate_id"] == candidate["id"] and r["stage"] == stage]
                check(len(group) == (4 if stage == "selection" else 2)*3*5, "meta developer group length")
                means[(strategy, candidate["id"], stage)] = {metric: float(np.mean([r[metric] for r in group]))
                    for metric in (*METRICS, *("source_"+m for m in METRICS), *("matched_"+m for m in METRICS))}
    for saved in recorded["means"]:
        values = means[(saved["strategy"], saved["candidate_id"], saved["stage"])]
        for metric, value in values.items():
            close(saved[metric], value, ("meta independent dev aggregation", saved["strategy"], saved["candidate_id"], saved["stage"], metric))
    def rank(strategy):
        return sorted(grid, key=lambda c: (-means[(strategy, c["id"], "selection")]["accuracy"],
            metas[c["meta_id"]]["updates"], metas[c["meta_id"]]["inner_learning_rate"],
            c["learning_rate"], c["steps"], c["id"]))
    policy = protocol["selection"]
    def passes(candidate, stage):
        value = means[("fomaml", candidate["id"], stage)]
        return all(value["accuracy"]-value[prefix+"accuracy"] > policy["minimum_accuracy_gain"]
                   and value["macro_f1"]-value[prefix+"macro_f1"] >= -policy["max_macro_f1_loss"]
                   and value["balanced_accuracy"]-value[prefix+"balanced_accuracy"] >= -policy["max_balanced_accuracy_loss"]
                   for prefix in ("source_", "matched_"))
    ranked = rank("fomaml")
    eligible = [c for c in ranked if passes(c, "selection")]
    proposed = eligible[0] if eligible else ranked[0]
    selected = proposed if eligible and passes(proposed, "gate") else None
    expected = {"selected": selected, "proposed": proposed, "diagnostic": ranked[0], "supervised_diagnostic": rank("supervised")[0]}
    for field, value in expected.items():
        check(recorded[field] == value, ("meta independent selection", field))
    check(recorded["eligible_selection_candidates"] == len(eligible), "meta independently eligible count")
    return {name: None if value is None else value["id"] for name, value in expected.items()} | {
        "eligible_candidates": len(eligible), "development_rows": len(rows)}


def verify_meta_training(directory, partition, seed, raw, row_lookup, truth, preprocessor, expected_initializer, final=False):
    import torch
    count, tasks = 0, 0
    source_raw = raw[[row_lookup[i] for i in partition["source"]]]
    features = direct_transform(preprocessor, source_raw)
    source_truth = np.asarray([CLASS_INDEX[truth[i]] for i in partition["source"]])
    source_ids = np.asarray(partition["source"])
    eligible = [c for c in range(7) if int(np.count_nonzero(source_truth == c)) >= 2]
    for current in sorted(directory.glob("m*/training.json")):
        metadata = read_json(current)
        record = metadata["training"]
        check(metadata["expected"]["source_ids"] == partition["source"], "meta source initializer ID set")
        check(metadata["expected"]["initial_fingerprint"] == expected_initializer, "meta source initializer fingerprint")
        config = record["config"]
        check(record["initializer_fingerprint"] == expected_initializer, "meta initializer remained source model")
        expected_seed = seed+1600 if final else seed+1000+100*int(directory.parent.name.split("_")[-1])
        check(record["seed"] == expected_seed, "meta context seed")
        check(record["policy"]["eligible_classes"] == eligible, "meta scarce class eligibility")
        check(record["source_only_training"] and record["same_task_ids_and_update_counts"] and not record["equal_FLOPs"] and record["first_order"], "meta declared method scope")
        check(len(record["source_tasks"]) == config["updates"], "meta update manifest count")
        rng = np.random.default_rng(expected_seed)
        pools = {c: np.flatnonzero(source_truth == c) for c in range(7)}
        for index, task in enumerate(record["source_tasks"]):
            categories = rng.choice(eligible, 3, replace=False)
            support, query = [], []
            for category in categories:
                order = rng.permutation(pools[int(category)])
                support.append(str(source_ids[order[0]])); query.append(str(source_ids[order[1]]))
            check(task["episode"] == index, "meta source task index")
            check(task["classes"] == categories.tolist(), "meta source class sampling")
            check(task["support_ids"] == support and task["query_ids"] == query, "meta disjoint source task IDs regenerated")
            check(not set(support)&set(query), "meta source support/query distinct physical IDs")
            tasks += 1
        for strategy in ("fomaml", "supervised"):
            path = current.parent/(strategy+".pt")
            artifact = metadata["artifacts"][strategy]
            check(sha256(path) == artifact["sha256"], "meta trained state hash")
            state = torch.load(path, map_location="cpu", weights_only=True)
            check(direct_state_fingerprint(state) == artifact["fingerprint"], "meta independent trained state fingerprint")
            check(record["models"][strategy]["fingerprint"] == artifact["fingerprint"], "meta paired metadata state consistency")
            check(len(record["models"][strategy]["losses"]) == config["updates"], "meta optimizer outer updates")
            source_prediction = dense_prediction_from_saved_state(state, features)
            independently = direct_metrics(source_truth, source_prediction)
            for metric in METRICS:
                close(record["models"][strategy]["source_metrics"][metric], independently[metric], "meta source inference replay without fitting")
            count += 1
    return count, tasks


def audit_meta_development(directory, truth, training_ids):
    import joblib
    audit_provenance(directory)
    parts = independent_partitions(training_ids, truth, 6)
    base_choice = read_json(RUN/"neural-base/selection_lock.json")["choice"]["selected"]["id"]
    with np.load(ROOT/"outputs/approved-zaoren-20261031/data-preparation/dataset.npz", allow_pickle=False) as data:
        row_lookup = {str(i): index for index, i in enumerate(data["record_ids"])}
        mask = (data["axis"] >= .1)&(data["axis"] <= 1.5)
        raw = data["X_alpha"][:, mask]
    rows = count = saved_models = task_manifests = 0
    for episode, partition in enumerate(parts):
        for seed in (42, 43, 44):
            current = directory/"development"/f"episode_{episode:02d}"/f"seed_{seed}"
            record = read_json(current/"metrics.json")
            for field, role in (("source_ids", "source"), ("pool_ids", "support_pool"), ("query_ids", "query")):
                check(record[field] == partition[role], ("meta developer role IDs", field, episode, seed))
            check_draws(record["draws"], partition["support_pool"], truth, 5, 20261104+episode)
            check(sha256(current/"predictions.csv.gz") == record["predictions_sha256"], "meta developer prediction hash")
            cases = prediction_cases(current/"predictions.csv.gz", truth, partition["query"])
            metrics = record["rows"]
            check(len(cases) == 160 and len(metrics) == 160, "meta two strategies all sixteen candidates and five draws")
            verify_case_metrics(current/"predictions.csv.gz", cases, metrics)
            ordered_truth = np.asarray([CLASS_INDEX[truth[i]] for i in record["query_ids"]])
            for row in metrics:
                control = record["controls"][row["strategy"]+":"+row["candidate_id"]]
                for prefix, name in (("source_", "source_prediction"), ("matched_", "matched_prediction")):
                    independently = direct_metrics(ordered_truth, control[name])
                    for metric in METRICS:
                        close(row[prefix+metric], independently[metric], "meta source/matched control arithmetic")
                case = next(c for c in cases.values() if c["dimensions"]["candidate_id"] == row["candidate_id"] and c["dimensions"]["strategy"] == row["strategy"] and c["dimensions"]["repeat"] == str(row["repeat"]))
                predictions = dict(zip(case["ids"], case["predicted"]))
                target = np.asarray([predictions[i] for i in record["query_ids"]])
                source = np.asarray(control["source_prediction"])
                check(row["corrected"] == int(np.sum((source != ordered_truth)&(target == ordered_truth))), "meta paired corrected developer count")
                check(row["new_errors"] == int(np.sum((source == ordered_truth)&(target != ordered_truth))), "meta paired new errors developer count")
            initial_dir = RUN/"neural-base/development"/f"episode_{episode:02d}"/f"seed_{seed}"/base_choice
            initial = read_json(initial_dir/"training.json")
            check(record["source_fingerprint"] == initial["fingerprint"], "meta developer initial model hash")
            models, tasks = verify_meta_training(current, partition, seed, raw, row_lookup, truth,
                joblib.load(initial_dir/"preprocessor.joblib"), initial["fingerprint"])
            saved_models += models; task_manifests += tasks
            rows += sum(len(c["ids"]) for c in cases.values()); count += len(metrics)
    return {"prediction_rows": rows, "cases": count, "source_models_hashed_and_replayed": saved_models,
            "shared_source_task_manifests_independently_regenerated": task_manifests,
            "choice": audit_meta_selection(directory)}


def audit_release_barrier():
    from datetime import datetime
    barrier = read_json(RUN/"all_development_locks.json")
    check(barrier["status"] == "released_for_final_evaluation", "global final release")
    release = datetime.fromisoformat(barrier["released_at_utc"])
    for item in barrier["rounds"]:
        check(not item["final_directory_existed_at_release"], "final directory absent at release")
        check(sha256(ROOT/item["selection_lock"]) == item["selection_lock_sha256"], "all-round source-only choice stayed locked")
        check(sha256(ROOT/item["protocol"]) == item["protocol_sha256"], "all-round protocol stayed fixed")
        check(sha256(RUN/item["name"]/"run_manifest.json") == item["run_manifest_sha256"], "all-round run manifest stayed fixed")
    reference = barrier["independent_development_audit"]
    check(sha256(ROOT/reference["file"]) == reference["sha256"], "recorded development audit preserved")
    timestamps = []
    for path in sorted(RUN.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            argv = row.get("argv", [])
            final = row.get("stage") in {"final", "final_resume"} or ("--stage" in argv and argv[argv.index("--stage")+1] == "final")
            event = row.get("event", "")
            final = final or event == "final_start"
            if (event == "start" or event.endswith("_start")) and final:
                moment = datetime.fromisoformat(row["utc"])
                check(moment >= release, ("final invocation after all-round release", str(path)))
                timestamps.append({"file": str(path.relative_to(ROOT)), "utc": row["utc"]})
    coverage = [name for name in ("neural-base", "support-svm", "anchored-ft", "meta-refinement")
                if any(name in item["file"] for item in timestamps)]
    return {"release_sha256": sha256(RUN/"all_development_locks.json"),
            "released_at_utc": barrier["released_at_utc"], "checked_final_starts": timestamps,
            "rounds_with_explicit_final_start_timestamps": coverage,
            "other_chronology_evidence": "The release records final directory absence and all four locked hashes. Any round without an explicit start event is not assigned an invented timestamp.",
            "scope": "saved execution chronology and hashes, not external attestation"}


def audit_final_numeric(directory, name, truth, query_ids):
    paths = sorted((directory/"final").rglob("predictions.csv.gz"))
    check(bool(paths), (name, "final prediction evidence exists"))
    cases = {}
    prediction_rows = 0
    for path in paths:
        current = prediction_cases(path, truth, query_ids)
        for case in current.values():
            dimensions = tuple(sorted(case["dimensions"].items()))
            check(dimensions not in cases, (name, "unique final case key", dimensions))
            cases[dimensions] = case
            prediction_rows += len(case["ids"])
    metric_path = directory/("final/metrics.json" if name == "support-svm" else "final_metrics.json")
    rows = read_json(metric_path)
    check(len(rows) == len(cases), (name, "all final cases have one metrics record"))
    verify_case_metrics(metric_path, cases, rows)
    lookup = {(case["dimensions"].get("seed", ""), case["dimensions"].get("repeat", ""), case["dimensions"].get("method", "")): case
              for case in cases.values()}
    for row in rows:
        if name == "neural-base":
            continue
        seed, repeat, method = str(row.get("seed", "")), str(row.get("repeat", "")), row["method"]
        case = lookup[(seed, repeat, method)]
        if name == "support-svm":
            source = lookup[(seed, repeat, "source_original")]
            matched = lookup[(seed, repeat, "matched_zero")]
        elif name == "anchored-ft":
            source = lookup[(seed, repeat, "source_zero")]
            matched_name = "source_zero" if method == "source_zero" else method if method.endswith("_matched_zero") else method+"_matched_zero"
            matched = lookup[(seed, repeat, matched_name)]
        else:
            prefix = method.removesuffix("_source_zero").removesuffix("_matched_zero")
            source = lookup[(seed, repeat, prefix+"_source_zero")]
            matched = lookup[(seed, repeat, prefix+"_matched_zero")]
        actual, prediction, baseline = np.asarray(case["actual"]), np.asarray(case["predicted"]), np.asarray(source["predicted"])
        for metric in METRICS:
            if name == "support-svm":
                close(row["matched_"+metric], matched["metrics"][metric], (name, "matched final baseline", metric))
                close(row[metric+"_gain_vs_source"], case["metrics"][metric]-source["metrics"][metric], (name, "final source gain", metric))
            elif name == "anchored-ft":
                close(row[metric+"_gain_over_source"], case["metrics"][metric]-source["metrics"][metric], (name, "final source gain", metric))
                close(row[metric+"_gain_over_matched"], case["metrics"][metric]-matched["metrics"][metric], (name, "final matched gain", metric))
            else:
                close(row["source_"+metric], source["metrics"][metric], (name, "final own source baseline", metric))
                close(row["matched_"+metric], matched["metrics"][metric], (name, "final matched baseline", metric))
        check(row["corrected" if name == "meta-refinement" else "corrected_errors"] == int(np.sum((baseline != actual)&(prediction == actual))), (name, "final corrected count"))
        check(row["new_errors"] == int(np.sum((baseline == actual)&(prediction != actual))), (name, "final new errors"))
    summary = read_json(directory/"summary.json")
    if name == "neural-base":
        for metric in METRICS:
            close(summary[metric+"_mean"], np.mean([r[metric] for r in rows]), (name, "summary", metric))
        check(summary["calibration_labels_used"] == 0, "MLP base model gain is not calibration")
    else:
        for entry in summary:
            group = [r for r in rows if r["method"] == entry["method"]]
            check(len(group) == entry["evaluations"], (name, "final summary count", entry["method"]))
            for metric in METRICS:
                values = [r[metric] for r in group]
                close(entry[metric+"_mean"], np.mean(values), (name, "final summary mean", metric))
                close(entry[metric+"_sd"], np.std(values, ddof=1 if name == "meta-refinement" else 0), (name, "final summary sd", metric))
                if name == "support-svm":
                    close(entry[metric+"_gain_vs_source_mean"], np.mean([r[metric+"_gain_vs_source"] for r in group]), "SVM summary source gain")
                elif name == "anchored-ft":
                    for comparator in ("source", "matched"):
                        close(entry[metric+"_gain_over_"+comparator+"_mean"], np.mean([r[metric+"_gain_over_"+comparator] for r in group]), "anchored summary gain")
                else:
                    for comparator in ("source", "matched"):
                        close(entry[metric+"_"+comparator+"_gain"], np.mean([r[metric]-r[comparator+"_"+metric] for r in group]), "meta summary gain")
            source_key = "accuracy_gain_vs_source" if name == "support-svm" else "accuracy_gain_over_source" if name == "anchored-ft" else None
            differences = [r[source_key] if source_key else r["accuracy"]-r["source_accuracy"] for r in group]
            cutoff = 1e-12 if name == "meta-refinement" else 0
            check(entry["positive_draws"] == sum(d > cutoff for d in differences), "summary positive scenarios")
            check(entry["unchanged_draws"] == sum(abs(d) <= cutoff for d in differences), "summary unchanged scenarios")
            check(entry["negative_draws"] == sum(d < -cutoff for d in differences), "summary negative scenarios")
    examples_path = directory/"random_predictions.csv"
    examples_count = incorrect_count = 0
    if examples_path.exists():
        with examples_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                check(row["y_true"] == truth[row["record_id"]], "random example truth")
                check(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "random example correctness")
                examples_count += 1; incorrect_count += int(not int(row["is_correct"]))
    return {"prediction_rows": prediction_rows, "cases": len(cases), "summary_rows": 1 if isinstance(summary, dict) else len(summary),
            "random_examples_preserved": examples_count, "random_incorrect_examples_preserved": incorrect_count,
            "summary": summary}


def audit_final_states(truth, training_ids, pool_ids, query_ids):
    import joblib
    import torch
    with np.load(ROOT/"outputs/approved-zaoren-20261031/data-preparation/dataset.npz", allow_pickle=False) as data:
        row_lookup = {str(i): index for index, i in enumerate(data["record_ids"])}
        mask = (data["axis"] >= .1)&(data["axis"] <= 1.5)
        raw = data["X_alpha"][:, mask]
    source_raw = raw[[row_lookup[i] for i in training_ids]]
    query_raw = raw[[row_lookup[i] for i in query_ids]]
    source_snv = (source_raw-source_raw.mean(axis=1,keepdims=True))/source_raw.std(axis=1,ddof=1,keepdims=True)
    source_truth = [CLASS_INDEX[truth[i]] for i in training_ids]
    choice = read_json(RUN/"neural-base/selection_lock.json")["choice"]["selected"]
    count = meta_tasks = 0
    dense_logits = {}
    anchored_records = {r["seed"]: r for r in read_json(RUN/"anchored-ft/final_metadata.json")}
    for seed in (42, 43, 44):
        base = RUN/"neural-base/final"/f"seed_{seed}"/choice["id"]
        metadata = read_json(base/"training.json")
        check(metadata["source_ids"] == training_ids and metadata["source_samples"] == 513, "final neural source IDs")
        check(sha256(base/"model.pt") == metadata["model_sha256"], "final neural source state hash")
        check(sha256(base/"preprocessor.joblib") == metadata["preprocessor_sha256"], "final neural preprocessor hash")
        state = torch.load(base/"model.pt", map_location="cpu", weights_only=True)
        check(direct_state_fingerprint(state) == metadata["fingerprint"], "final neural tensor fingerprint")
        preprocessor = joblib.load(base/"preprocessor.joblib")
        close(preprocessor["scaler"].mean_, source_snv.mean(axis=0), "final source-only scaler mean")
        close(preprocessor["scaler"].var_, source_snv.var(axis=0), "final source-only scaler variance")
        check(preprocessor["scaler"].n_samples_seen_ == 513, "final scaler used only513 source")
        source_pred = dense_prediction_from_saved_state(state, direct_transform(preprocessor, source_raw))
        for metric in METRICS:
            close(metadata["source_metrics"][metric], direct_metrics(source_truth, source_pred)[metric], "final MLP source metrics replay")
        features = direct_transform(preprocessor, query_raw)
        logits = dense_scores_from_saved_state(state, features)
        dense_logits[(seed, "base_b006")] = logits
        check(len(prediction_cases(base/"predictions.csv.gz", truth, query_ids)) == 1, "final base one prediction per seed")
        with gzip.open(base/"predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            saved = {r["record_id"]: CLASS_INDEX[r["y_predicted"]] for r in csv.DictReader(handle)}
        check([saved[i] for i in query_ids] == logits.argmax(1).tolist(), "final base predictions replay")
        count += 1
        anchored = anchored_records[seed]
        check(anchored["source_ids"] == training_ids and anchored["query_ids"] == query_ids, "anchored final source/query IDs")
        check(anchored["source_fingerprint_before"] == anchored["source_fingerprint_after"] == metadata["fingerprint"], "anchored final source retained")
        check_draws(anchored["support_draws"], pool_ids, truth, 100, 20261105)
        check(sha256(RUN/"anchored-ft/final"/f"seed_{seed}"/"predictions.csv.gz") == anchored["prediction_sha256"], "anchored final predictions hash")
        for artifact in anchored["saved_adapted_first_draw_models"]:
            path = ROOT/artifact["path"]
            check(sha256(path) == artifact["sha256"], "anchored adapted first draw state hash")
            adapted = torch.load(path, map_location="cpu", weights_only=True)
            check(direct_state_fingerprint(adapted) == artifact["fingerprint"], "anchored adapted first draw fingerprint")
            guesses = dense_prediction_from_saved_state(adapted, features)
            with gzip.open(RUN/"anchored-ft/final"/f"seed_{seed}"/"predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                recorded = {r["record_id"]: CLASS_INDEX[r["y_predicted"]] for r in csv.DictReader(handle)
                            if int(r["repeat"]) == 0 and r["candidate_id"] == artifact["candidate_id"] and r["method"] in {"selected", "diagnostic"}}
            check([recorded[i] for i in query_ids] == guesses.tolist(), "anchored first support adaptation inference replay")
            count += 1
        final = RUN/"meta-refinement/final"/f"seed_{seed}"
        evaluation = read_json(final/"evaluation.json")
        check(evaluation["source_ids"] == training_ids and evaluation["pool_ids"] == pool_ids and evaluation["query_ids"] == query_ids, "meta final physical specimen roles")
        check_draws(evaluation["draws"], pool_ids, truth, 100, 20261105)
        check(sha256(final/"predictions.csv.gz") == evaluation["predictions_sha256"], "meta final predictions hash")
        source_count, shared_tasks = verify_meta_training(final, {"source": training_ids}, seed, raw, row_lookup,
            truth, preprocessor, metadata["fingerprint"], final=True)
        count += source_count; meta_tasks += shared_tasks
        for meta_id in ("m000", "m002"):
            for strategy in ("fomaml", "supervised"):
                state = torch.load(final/meta_id/(strategy+".pt"), map_location="cpu", weights_only=True)
                dense_logits[(seed, meta_id+"_"+strategy)] = dense_scores_from_saved_state(state, features)
    svm = RUN/"support-svm/final"
    record = read_json(svm/"metadata.json")
    check(record["source_ids"] == training_ids and record["support_pool_ids"] == pool_ids and record["query_ids"] == query_ids, "final SVM physical specimen roles")
    check_draws(record["draws"], pool_ids, truth, 100, 20261105)
    check(record["source_fingerprint_before"] == record["source_fingerprint_after"], "final SVM source unchanged")
    check(record["context_fingerprint_before"] == record["context_fingerprint_after"], "final SVM context unchanged")
    check(sha256(svm/"source_model.joblib") == record["source_model_sha256"], "final SVM source artifact hash")
    check(sha256(svm/"predictions.csv.gz") == record["predictions_sha256"], "final SVM predictions hash")
    for artifact in record["saved_first_draw_adapted_models"]:
        check(sha256(ROOT/artifact["path"]) == artifact["sha256"], "final SVM adapted first draw hash")
        count += 1
    count += 1
    return {"final_saved_states_hashed": count, "neural_final_states_inference_replayed": count-2,
            "final_shared_meta_tasks_regenerated": meta_tasks,
            "final_support_draws_independently_regenerated": 700}, dense_logits


def audit_source_control_diagnostic(truth, query_ids, logits):
    from itertools import combinations
    directory = RUN/"meta-refinement"
    record = read_json(directory/"source_control_diagnostic.json")
    path = directory/"source_control_diagnostic.csv"
    check(sha256(path) == record["csv_sha256"], "extra source-control CSV hash")
    check(sha256(directory/"source_control_diagnostic.py") == record["script_sha256"], "extra source-control script hash")
    check(record["new_fits"] == 0 and record["no_support_labels_used"] and record["model_states_unchanged"], "source-control diagnostic scope")
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["source_model"])
            groups[key].append(row)
            check(row["y_true"] == truth[row["record_id"]], "source-control diagnostic truth")
            check(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "source-control diagnostic correct flag")
    independently = {}
    for key, rows in groups.items():
        by_id = {r["record_id"]: r for r in rows}
        check(len(rows) == len(by_id) == 171 and set(by_id) == set(query_ids), "source-control unique fixed predictions")
        recorded_logits = np.asarray([[float(by_id[i]["logit_"+c]) for c in CLASSES] for i in query_ids])
        close(recorded_logits, logits[key], ("source-control logits independently replayed", key), atol=1e-6)
        check([CLASS_INDEX[by_id[i]["y_predicted"]] for i in query_ids] == logits[key].argmax(1).tolist(), "source-control source class labels replayed")
        independently[key] = direct_metrics([CLASS_INDEX[truth[i]] for i in query_ids], logits[key].argmax(1))
    for saved in record["source_model_metrics"]:
        metrics = independently[(saved["seed"], saved["source_model"])]
        for metric in METRICS:
            close(saved[metric], metrics[metric], "source-control source metrics")
    for saved in record["source_model_means"]:
        for metric in METRICS:
            close(saved[metric+"_mean"], np.mean([independently[(seed, saved["source_model"])][metric] for seed in (42, 43, 44)]), "source-control three-seed means")
    for saved in record["pairwise_disagreement"]:
        first = logits[(saved["seed"], saved["first"])]; second = logits[(saved["seed"], saved["second"])]
        first_pred, second_pred = first.argmax(1), second.argmax(1)
        actual = np.asarray([CLASS_INDEX[truth[i]] for i in query_ids])
        check(saved["different_predictions"] == int(np.sum(first_pred != second_pred)), "source-control pairwise label differences")
        check(saved["second_corrected_first"] == int(np.sum((first_pred != actual)&(second_pred == actual))), "source-control pairwise corrected")
        check(saved["second_new_errors"] == int(np.sum((first_pred == actual)&(second_pred != actual))), "source-control pairwise new errors")
        close(saved["max_absolute_logit_difference"], np.max(np.abs(first-second)), "source-control logit differences", atol=1e-6)
    paired = []
    for seed in (42, 43, 44):
        for meta in ("m000", "m002"):
            first, second = logits[(seed,meta+"_fomaml")], logits[(seed,meta+"_supervised")]
            paired.append({"seed": seed, "meta_id": meta, "different_labels": int(np.sum(first.argmax(1) != second.argmax(1))),
                           "same_labels": int(np.sum(first.argmax(1) == second.argmax(1))),
                           "maximum_logit_difference": float(np.max(np.abs(first-second)))})
    return {"prediction_rows": 2565, "source_model_cases": 15, "pairwise_checks": len(record["pairwise_disagreement"]),
            "fomaml_vs_same_budget_supervised": paired,
            "interpretation": "All six paired 171-label vectors are identical; differing logits do not establish FO-specific classification gains. This is an additional read-only control, not a new selected model."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--round", choices=["support-svm", "neural-base", "anchored-ft", "meta-refinement", "all"], default="all")
    args = parser.parse_args()
    plan = read_json(ROOT / "data/processed/zaoren_threeway_20261031_v1/split_plan.json")
    truth = {r["record_id"]: r["origin_group_id"] for r in plan["records"]}
    training_ids = [r["record_id"] for r in plan["records"] if r["role"] == "training"]
    query_ids = [r["record_id"] for r in plan["records"] if r["role"] == "prediction"]
    pool_ids = [r["record_id"] for r in plan["records"] if r["role"] == "calibration"]
    with np.load(ROOT/"outputs/approved-zaoren-20261031/data-preparation/dataset.npz", allow_pickle=False) as data:
        check(len(data["record_ids"]) == 855, "fixed canonical sample count")
        for identifier, label in zip(data["record_ids"], data["y"]):
            check(truth[str(identifier)] == str(label), "canonical NPZ label agrees with numbered split truth")
    started = time.monotonic()
    result = {}
    if args.round in {"all", "support-svm"}:
        result["support-svm"] = audit_svm_development(RUN / "support-svm", truth, training_ids)
    if args.round in {"all", "neural-base"}:
        result["neural-base"] = audit_neural_development(RUN / "neural-base", truth, training_ids)
    if args.round in {"all", "anchored-ft"}:
        result["anchored-ft"] = audit_anchored_development(RUN / "anchored-ft", truth, training_ids)
    if args.round in {"all", "meta-refinement"}:
        result["meta-refinement"] = audit_meta_development(RUN / "meta-refinement", truth, training_ids)
    barrier = None
    final_states = source_controls = None
    if not args.development_only:
        barrier = audit_release_barrier()
        for name in result:
            result[name]["final"] = audit_final_numeric(RUN/name, name, truth, query_ids)
        if args.round == "all":
            final_states, logits = audit_final_states(truth, training_ids, pool_ids, query_ids)
            source_controls = audit_source_control_diagnostic(truth, query_ids, logits)
    output = {"status": "pass", "scope": "development-only independent audit" if args.development_only else "independent four-round development and final arithmetic/lineage audit",
              "checks_evaluated": CHECKS, "maximum_numeric_difference": MAX_DIFFERENCE,
              "rounds": result, "seconds": time.monotonic()-started, "new_fits": 0,
              "barrier": barrier, "final_states": final_states, "extra_source_control": source_controls,
              "total_development_prediction_rows": sum(r["prediction_rows"] for r in result.values()),
              "total_final_prediction_rows": sum(r.get("final",{}).get("prediction_rows",0) for r in result.values()),
              "total_development_metric_cases": sum(r.get("cases",r.get("adapted_cases",0)) for r in result.values()),
              "total_final_metric_cases": sum(r.get("final",{}).get("cases",0) for r in result.values()),
              "audit_script_sha256": sha256(__file__),
              "limits": ["Real uploaded spectra; anonymous labels are not independently verified named geographic origins.",
                         "Same batch and previously exposed query records; repeats are not independent external batches.",
                         "Recorded before/after hashes and chronology are not external execution attestation.",
                         "No project training, metrics, aggregation, or selection helper imported."]}
    filename = "independent_development_audit.json" if args.development_only else "independent_audit.json"
    (RUN / filename).write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
