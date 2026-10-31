"""Source-replay and parameter-anchored adaptation on locked dense models.

Target query labels are deliberately absent from gradient routines. Each target
adaptation is compared with both its untouched source and the identical update
path without target supervision. All current tasks are retrospective same-batch
experiments, not independent acquisition domains.
"""
from collections import defaultdict
from copy import deepcopy
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from .calibration_exploration import digest, encoded, fast_metrics, nested_draws, object_digest, write_json
from .paper_fewshot_classification import state_fingerprint


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def candidate_grid():
    result = []
    for learning_rate in (.0001, .001):
        for steps in (5, 20):
            for replay in (0., 1., 4.):
                for proximal in (0., .1):
                    result.append({"id": f"a{len(result):03d}", "learning_rate": learning_rate,
                        "steps": steps, "source_replay_weight": replay,
                        "proximal_lambda": proximal})
    return result


def _tensor(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite, nonempty specimens-by-source-features array")
    return torch.as_tensor(values)


def _labels(values, n, classes):
    values = np.asarray(values)
    if values.shape != (n,) or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Expected one known class index per specimen")
    return torch.as_tensor(values, dtype=torch.long)


def index_labels(values, classes):
    values = np.asarray(values)
    if values.dtype.kind in "iu":
        if np.any(values < 0) or np.any(values >= len(classes)):
            raise ValueError("Encoded category outside the fixed vocabulary")
        return values.astype(int)
    return encoded(values, classes)


def source_class_weights(labels, n_classes, mode):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=n_classes)
    if mode not in {"none", "balanced"} or np.any(counts <= 0):
        raise ValueError("Source classes and source weighting policy must be explicit")
    weights = np.ones(n_classes) if mode == "none" else len(labels)/(n_classes*counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def predict_scores(model, values):
    model.eval()
    with torch.no_grad():
        return model(_tensor(values)).cpu().numpy()


def validate_candidate(config):
    expected = {c["id"]: c for c in candidate_grid()}
    if expected.get(config.get("id")) != config:
        raise ValueError("Adaptation configuration is outside the frozen24-grid")


def adapt_paths(initial, source_x, source_y, prediction_x, candidates, *,
                support_x=None, support_y=None, class_weight_mode="balanced",
                capture_models=False):
    """Predict each requested path checkpoint without prediction truth or IDs.

    With no support, the same replay, proximal penalty, learning rate and steps
    are retained. A no-support/no-replay path is exactly the source predictor.
    """
    if not candidates:
        return {}, {}, {}
    for candidate in candidates:
        validate_candidate(candidate)
    source_x, prediction_x = _tensor(source_x), _tensor(prediction_x)
    if source_x.shape[1] != prediction_x.shape[1]:
        raise ValueError("Source and prediction features must share the fitted representation")
    n_classes = initial.head.out_features
    source_y = _labels(source_y, len(source_x), n_classes)
    weights = source_class_weights(source_y.numpy(), n_classes, class_weight_mode)
    if (support_x is None) != (support_y is None):
        raise ValueError("Support spectra and labels must be supplied together")
    if support_x is not None:
        support_x = _tensor(support_x)
        support_y = _labels(support_y, len(support_x), n_classes)
        if support_x.shape[1] != source_x.shape[1] or set(support_y.tolist()) != set(range(n_classes)):
            raise ValueError("Labelled support must cover every known class in the same feature space")
    before = state_fingerprint(initial)
    groups = defaultdict(list)
    for c in candidates:
        groups[(c["learning_rate"], c["source_replay_weight"], c["proximal_lambda"])].append(c)
    predictions, traces, fitted_models = {}, {}, {}
    for (rate, replay, proximal), configs in groups.items():
        model = deepcopy(initial)
        origin = {name: value.detach().clone() for name, value in model.named_parameters()}
        checkpoints = {c["steps"]: c for c in configs}
        if support_x is None and replay == 0:
            # The proximal penalty has its minimum at the unchanged source.
            scores = predict_scores(model, prediction_x.numpy())
            for c in configs:
                predictions[c["id"]] = scores.copy()
                traces[c["id"]] = {"updates": 0, "specified_steps": c["steps"],
                    "support_loss": 0., "source_loss": None, "proximal_sum": 0.,
                    "exact_identity_no_support_no_replay": True}
                if capture_models:
                    fitted_models[c["id"]] = deepcopy(model)
            continue
        optimizer = torch.optim.SGD(model.parameters(), lr=rate)
        model.train()
        for step in range(1, max(checkpoints)+1):
            optimizer.zero_grad(set_to_none=True)
            support_loss = (F.cross_entropy(model(support_x), support_y)
                            if support_x is not None else source_x.new_tensor(0.))
            if replay:
                losses = F.cross_entropy(model(source_x), source_y, reduction="none")
                sample_weights = weights[source_y]
                source_loss = (losses*sample_weights).sum()/sample_weights.sum()
            else:
                source_loss = source_x.new_tensor(0.)
            proximal_sum = sum((value-origin[name]).square().sum()
                              for name, value in model.named_parameters())
            loss = support_loss + replay*source_loss + proximal*proximal_sum
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite source-anchored target loss")
            loss.backward(); optimizer.step()
            if step in checkpoints:
                c = checkpoints[step]
                predictions[c["id"]] = predict_scores(model, prediction_x.numpy())
                traces[c["id"]] = {"updates": step, "support_loss": float(support_loss.detach()),
                    "source_loss": float(source_loss.detach()), "proximal_sum": float(proximal_sum.detach()),
                    "total_loss_before_last_update": float(loss.detach()),
                    "target_labels_used": support_x is not None}
                if capture_models:
                    fitted_models[c["id"]] = deepcopy(model)
                model.train()
    if state_fingerprint(initial) != before:
        raise AssertionError("Adaptation mutated its saved source model")
    return predictions, traces, fitted_models


def verify_context(context, *, final=False):
    """Validate physical record isolation before any gradient routine runs."""
    role_sets = {}
    for name in ("source", "support_pool", "prediction"):
        role = context[name]
        identifiers = np.asarray(role["record_ids"]).tolist()
        if len(set(identifiers)) != len(identifiers) or len(role["X"]) != len(identifiers):
            raise ValueError("A role contains duplicate or misaligned physical specimens")
        role_sets[name] = set(identifiers)
    for a, b in (("source", "support_pool"), ("source", "prediction"), ("support_pool", "prediction")):
        if role_sets[a] & role_sets[b]:
            raise ValueError("Source/support/prediction physical specimens overlap")
    if final and "y" in context["prediction"]:
        raise ValueError("Final prediction truth must be separated at the scoring boundary")
    return {name: len(values) for name, values in role_sets.items()}


def verify_context_collection(contexts, *, final=False):
    expected = {(0 if final else episode, seed) for episode in range(1 if final else 6) for seed in (42, 43, 44)}
    observed = {(0 if final else c["episode"], c["seed"]) for c in contexts}
    if observed != expected or len(contexts) != len(expected):
        raise ValueError("Locked source contexts do not cover every required task and seed")
    expected_counts = {"source": 513 if final else 307, "support_pool": 171 if final else 103,
                       "prediction": 171 if final else 103}
    for context in contexts:
        if verify_context(context, final=final) != expected_counts:
            raise ValueError("Locked source contexts disagree with the fixed physical role counts")
        if list(context["classes"]) != [f"ZY{i:02d}" for i in range(1, 8)]:
            raise ValueError("Locked source category order differs from the fixed anonymous groups")
        if context.get("fingerprint") != state_fingerprint(context["model"]):
            raise ValueError("Loaded source model differs from its declared fingerprint")


def _mean_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate_id"], row["stage"])].append(row)
    keys = METRICS + tuple("source_"+k for k in METRICS) + tuple("matched_"+k for k in METRICS)
    return {key: {**{m: float(np.mean([r[m] for r in cases])) for m in keys},
                  "evaluations": len(cases)} for key, cases in grouped.items()}


def choose_candidate(rows, candidates, policy):
    means = _mean_rows(rows)
    ranked = sorted(candidates, key=lambda c: (-means[(c["id"], "selection")]["accuracy"],
        c["learning_rate"], c["steps"], c["source_replay_weight"], c["proximal_lambda"], c["id"]))
    def passes(candidate, stage):
        values = means[(candidate["id"], stage)]
        for comparator in ("source_", "matched_"):
            if not (values["accuracy"]-values[comparator+"accuracy"] > policy["minimum_accuracy_gain"]
                and values["macro_f1"]-values[comparator+"macro_f1"] >= -policy["max_macro_f1_loss"]
                and values["balanced_accuracy"]-values[comparator+"balanced_accuracy"] >= -policy["max_balanced_accuracy_loss"]):
                return False
        return True
    eligible = [c for c in ranked if passes(c, "selection")]
    diagnostic = eligible[0] if eligible else ranked[0]
    selected = diagnostic if eligible and passes(diagnostic, "gate") else None
    return {"selected": selected, "diagnostic": diagnostic, "fallback_to_source": selected is None,
        "selection": means[(diagnostic["id"], "selection")], "gate": means[(diagnostic["id"], "gate")],
        "selection_eligible_candidates": len(eligible), "final_query_used_for_selection": False}


def _weight_mode(candidate):
    if "class_weight" not in candidate:
        raise ValueError("Locked source candidate must record its class_weight policy")
    return candidate["class_weight"]


def _provenance(input_path, plan_path, protocol_path, source_run):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root/"scripts/run_anchored_neural_search.py",
        Path(__file__).with_name("neural_base_search.py"),
        Path(__file__).with_name("paper_fewshot_classification.py"),
        Path(__file__).with_name("calibration_exploration.py")]
    source_run = Path(source_run)
    return {"input_sha256": digest(input_path), "split_plan_sha256": digest(plan_path),
        "protocol_sha256": digest(protocol_path), "grid_sha256": object_digest(candidate_grid()),
        "source_run_manifest_sha256": digest(source_run/"run_manifest.json"),
        "source_selection_lock_sha256": digest(source_run/"selection_lock.json"),
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        "python": platform.python_version(), "versions": {n: importlib.metadata.version(n)
            for n in ("torch", "numpy", "scikit-learn")}}


def validate_protocol(protocol):
    if protocol["candidate_grid_sha256"] != object_digest(candidate_grid()):
        raise ValueError("Protocol does not lock the implemented24-grid")
    if protocol["seeds"] != [42, 43, 44] or protocol["development"] != {
        "episodes": 6, "selection_episodes": 4, "support_draws": 5,
        "support_seed": 20261104}:
        raise ValueError("Expected the fixed three-seed six-task development design")
    if protocol["final"] != {"support_draws": 100, "support_seed": 20261105,
        "random_query_seed": 20261102, "random_query_examples": 20}:
        raise ValueError("Expected the fixed100-support-draw final design")


def _writer(path):
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    fields = ["episode", "seed", "repeat", "method", "candidate_id", "record_id", "y_true", "y_predicted", "is_correct"]
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return handle, writer


def _verify_locked_development(output, protocol):
    lock = json.loads((output/"selection_lock.json").read_text())
    if lock["run_manifest_sha256"] != digest(output/"run_manifest.json"):
        raise ValueError("Development lock belongs to a different manifest")
    all_rows = []
    for episode in range(6):
        for seed in protocol["seeds"]:
            directory = output/"development"/f"episode_{episode:02d}"/f"seed_{seed}"
            record = json.loads((directory/"record.json").read_text())
            if (digest(directory/"metrics.json") != record["metrics_sha256"]
                    or digest(directory/"predictions.csv.gz") != record["prediction_sha256"]):
                raise ValueError("Locked development evidence changed")
            all_rows.extend(json.loads((directory/"metrics.json").read_text()))
    if (choose_candidate(all_rows, candidate_grid(), protocol["selection"]) != lock["choice"]
            or len(all_rows) != lock["development_evaluations"]
            or json.loads((output/"all_development_metrics.json").read_text()) != all_rows):
        raise ValueError("Development lock no longer matches its saved cases")
    return lock


def _write_predictions(writer, context, truth, predicted, *, repeat, method, candidate_id):
    classes = np.asarray(context["classes"])
    result = []
    for identifier, actual, prediction in zip(context["prediction"]["record_ids"], truth, predicted):
        row = {"episode": context["episode"], "seed": context["seed"], "repeat": repeat,
            "method": method, "candidate_id": candidate_id, "record_id": str(identifier),
            "y_true": str(classes[actual]), "y_predicted": str(classes[prediction]),
            "is_correct": int(actual == prediction)}
        writer.writerow(row); result.append(row)
    return result


def run_search(input_path, plan_path, protocol_path, source_run, output, *, stage, resume=False):
    """Run development or, only after external round gates, final evaluation."""
    if stage not in {"development", "final"}:
        raise ValueError("A development/final stage is required")
    from .neural_base_search import load_locked_contexts
    protocol = json.loads(Path(protocol_path).read_text())
    validate_protocol(protocol)
    output = Path(output); source_run = Path(source_run)
    provenance = _provenance(input_path, plan_path, protocol_path, source_run)
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"run_manifest.json").exists():
            raise FileExistsError("Use a fresh run or verified --resume")
        if json.loads((output/"run_manifest.json").read_text()) != provenance:
            raise ValueError("Resume refused: code/input/protocol/source lock changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/"run_manifest.json", provenance)
        write_json(output/"candidate_grid.json", candidate_grid())
        (output/"protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
    if digest(output/"protocol_snapshot.json") != provenance["protocol_sha256"]:
        raise ValueError("Recorded protocol snapshot changed")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    if stage == "development":
        if (output/"selection_lock.json").exists():
            if not resume:
                raise FileExistsError("Development lock already exists")
            return _verify_locked_development(output, protocol)
        contexts, _ = load_locked_contexts(input_path, plan_path, source_run, scope="development")
        verify_context_collection(contexts)
        candidates = candidate_grid(); all_rows, records = [], []
        for context in contexts:
            episode, seed = context["episode"], context["seed"]
            directory = output/"development"/f"episode_{episode:02d}"/f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            metric_path = directory/"metrics.json"; record_path = directory/"record.json"
            if resume and record_path.exists():
                record = json.loads(record_path.read_text())
                if digest(metric_path) != record["metrics_sha256"] or digest(directory/"predictions.csv.gz") != record["prediction_sha256"]:
                    raise ValueError("Development evidence changed")
                all_rows.extend(json.loads(metric_path.read_text())); records.append(record)
                continue
            role_counts = verify_context(context)
            classes = np.asarray(context["classes"])
            source, pool, query = context["source"], context["support_pool"], context["prediction"]
            source_y, pool_y, truth = (index_labels(role["y"], classes) for role in (source, pool, query))
            initial = context["model"]; before = state_fingerprint(initial)
            mode = _weight_mode(context["candidate"])
            zero = np.argmax(predict_scores(initial, query["X"]), axis=1)
            source_metrics = fast_metrics(truth, zero, len(classes))
            matched_scores, matched_traces, _ = adapt_paths(initial, source["X"], source_y, query["X"], candidates,
                class_weight_mode=mode)
            matched = {key: np.argmax(values, axis=1) for key, values in matched_scores.items()}
            matched_metrics = {key: fast_metrics(truth, values, len(classes)) for key, values in matched.items()}
            draws = nested_draws(pool_y, np.arange(len(classes)), [1], protocol["development"]["support_draws"],
                protocol["development"]["support_seed"]+episode)
            handle, writer = _writer(directory/"predictions.csv.gz")
            rows = []
            with handle:
                _write_predictions(writer, context, truth, zero, repeat=-1, method="source_zero", candidate_id="source_zero")
                for candidate in candidates:
                    _write_predictions(writer, context, truth, matched[candidate["id"]], repeat=-1,
                        method="matched_zero", candidate_id=candidate["id"])
                for draw in draws:
                    si = draw["indices"]
                    scores, _, _ = adapt_paths(initial, source["X"], source_y, query["X"], candidates,
                        support_x=np.asarray(pool["X"])[si], support_y=pool_y[si], class_weight_mode=mode)
                    for candidate in candidates:
                        key = candidate["id"]; predicted = np.argmax(scores[key], axis=1)
                        metrics = fast_metrics(truth, predicted, len(classes))
                        rows.append({"episode": episode, "seed": seed, "repeat": draw["repeat"],
                            "stage": "selection" if episode < protocol["development"]["selection_episodes"] else "gate",
                            "candidate_id": key, **metrics,
                            **{"source_"+m: source_metrics[m] for m in METRICS},
                            **{"matched_"+m: matched_metrics[key][m] for m in METRICS}})
                        _write_predictions(writer, context, truth, predicted, repeat=draw["repeat"],
                            method="adapted", candidate_id=key)
            if state_fingerprint(initial) != before:
                raise AssertionError("Development adaptation changed a source model")
            record = {"episode": episode, "seed": seed, "role_counts": role_counts,
                "role_ids": {name: np.asarray(context[name]["record_ids"]).tolist()
                    for name in ("source", "support_pool", "prediction")},
                "source_fingerprint_before": before, "source_fingerprint_after": state_fingerprint(initial),
                "source_preprocessor_sha256": context["preprocessor_sha256"],
                "source_candidate": context["candidate"], "source_class_weight": mode,
                "support_draws": [{"repeat": d["repeat"], "ids": np.asarray(pool["record_ids"])[d["indices"]].tolist()} for d in draws],
                "matched_control_last_losses": matched_traces, "evaluations": len(rows),
                "prediction_sha256": digest(directory/"predictions.csv.gz")}
            write_json(metric_path, rows); record["metrics_sha256"] = digest(metric_path)
            write_json(record_path, record); all_rows.extend(rows); records.append(record)
            print(json.dumps({"stage": "development", "episode": episode, "seed": seed,
                "candidate_evaluations": len(rows)}), flush=True)
        lock = {"choice": choose_candidate(all_rows, candidates, protocol["selection"]),
            "source_only_development": True, "source_selection_lock_sha256": provenance["source_selection_lock_sha256"],
            "run_manifest_sha256": digest(output/"run_manifest.json"), "development_evaluations": len(all_rows)}
        write_json(output/"all_development_metrics.json", all_rows)
        write_json(output/"development_metadata.json", records)
        write_json(output/"selection_lock.json", lock)
        write_json(output/"status.json", {"status": "development_locked", "seconds": time.monotonic()-started,
            "development_evaluations": len(all_rows), "selection_lock_sha256": digest(output/"selection_lock.json"),
            "final_query_scored": False})
        return lock
    if not (output/"selection_lock.json").exists():
        raise ValueError("Final stage requires a completed development lock")
    _verify_locked_development(output, protocol)
    if (output/"summary.json").exists() and resume:
        status = json.loads((output/"status.json").read_text())
        if (status.get("status") != "completed"
                or status["final_metrics_sha256"] != digest(output/"final_metrics.json")
                or status["summary_sha256"] != digest(output/"summary.json")):
            raise ValueError("Final metrics or summary changed")
        for record in json.loads((output/"final_metadata.json").read_text()):
            directory = output/"final"/f"seed_{record['seed']}"
            if digest(directory/"predictions.csv.gz") != record["prediction_sha256"]:
                raise ValueError("Final per-specimen predictions changed")
            for model in record["saved_adapted_first_draw_models"]:
                if digest(model["path"]) != model["sha256"]:
                    raise ValueError("Saved adapted model changed")
        return json.loads((output/"summary.json").read_text())
    lock_path = output/"selection_lock.json"; lock = json.loads(lock_path.read_text()); lock_sha = digest(lock_path)
    if lock["run_manifest_sha256"] != digest(output/"run_manifest.json"):
        raise ValueError("Development lock no longer belongs to this run")
    contexts, final_truth = load_locked_contexts(input_path, plan_path, source_run, scope="final")
    verify_context_collection(contexts, final=True)
    choice = lock["choice"]; configs = {c["id"]: c for c in (choice["selected"], choice["diagnostic"]) if c is not None}
    all_rows, records, random_rows = [], [], []
    random_ids = None
    for context in contexts:
        verify_context(context, final=True)
        classes = np.asarray(context["classes"])
        source, pool, query = context["source"], context["support_pool"], context["prediction"]
        source_y, pool_y = (index_labels(role["y"], classes) for role in (source, pool))
        # Target labels join only after all choices and gradient inputs are fixed.
        truth = index_labels([final_truth[str(i)] for i in query["record_ids"]], classes)
        seed = context["seed"]; initial = context["model"]; before = state_fingerprint(initial)
        if random_ids is None:
            random_ids = np.random.default_rng(protocol["final"]["random_query_seed"]).choice(
                np.asarray(sorted(query["record_ids"])), protocol["final"]["random_query_examples"], replace=False).tolist()
            write_json(output/"random_query_ids.json", random_ids)
        mode = _weight_mode(context["candidate"])
        zero = np.argmax(predict_scores(initial, query["X"]), axis=1)
        zero_metrics = fast_metrics(truth, zero, len(classes))
        matched_scores, matched_traces, _ = adapt_paths(initial, source["X"], source_y, query["X"], list(configs.values()), class_weight_mode=mode)
        matched = {key: np.argmax(values, axis=1) for key, values in matched_scores.items()}
        draws = nested_draws(pool_y, np.arange(len(classes)), [1], protocol["final"]["support_draws"], protocol["final"]["support_seed"])
        directory = output/"final"/f"seed_{seed}"; directory.mkdir(parents=True, exist_ok=True)
        handle, writer = _writer(directory/"predictions.csv.gz")
        saved_models = []
        with handle:
            for draw in draws:
                if digest(lock_path) != lock_sha:
                    raise ValueError("Choice lock changed during final scoring")
                si = draw["indices"]
                scores, _, models = adapt_paths(initial, source["X"], source_y, query["X"], list(configs.values()),
                    support_x=np.asarray(pool["X"])[si], support_y=pool_y[si], class_weight_mode=mode,
                    capture_models=draw["repeat"] == 0)
                if draw["repeat"] == 0:
                    for key, model in models.items():
                        path = directory/(key+"_draw000.pt"); torch.save(model.state_dict(), path)
                        saved_models.append({"candidate_id": key, "repeat": 0, "path": str(path),
                            "sha256": digest(path), "fingerprint": state_fingerprint(model)})
                for method in ("source_zero", "selected", "selected_matched_zero", "diagnostic", "diagnostic_matched_zero"):
                    config = None if method == "source_zero" else choice[method.split("_matched")[0]]
                    key = "source_zero" if config is None else config["id"]
                    predicted = zero if config is None else np.argmax(scores[key], axis=1) if "matched_zero" not in method else matched[key]
                    comparator = zero if config is None else matched[key]
                    comparator_metrics = fast_metrics(truth, comparator, len(classes))
                    metrics = fast_metrics(truth, predicted, len(classes), detailed=True)
                    all_rows.append({"seed": seed, "repeat": draw["repeat"], "method": method, "candidate_id": key,
                        **metrics, **{m+"_gain_over_source": metrics[m]-zero_metrics[m] for m in METRICS},
                        **{m+"_gain_over_matched": metrics[m]-comparator_metrics[m] for m in METRICS},
                        "corrected_errors": int(np.count_nonzero((zero != truth)&(predicted == truth))),
                        "new_errors": int(np.count_nonzero((zero == truth)&(predicted != truth)))})
                    examples = _write_predictions(writer, {**context, "episode": "final"}, truth, predicted,
                        repeat=draw["repeat"], method=method, candidate_id=key)
                    if draw["repeat"] == 0:
                        random_rows.extend(r for r in examples if r["record_id"] in random_ids)
        if state_fingerprint(initial) != before:
            raise AssertionError("Final adaptation mutated the saved source")
        records.append({"seed": seed, "source_candidate": context["candidate"], "source_class_weight": mode,
            "source_preprocessor_sha256": context["preprocessor_sha256"],
            "source_ids": np.asarray(source["record_ids"]).tolist(), "query_ids": np.asarray(query["record_ids"]).tolist(),
            "source_fingerprint_before": before, "source_fingerprint_after": state_fingerprint(initial),
            "support_draws": [{"repeat": d["repeat"], "ids": np.asarray(pool["record_ids"])[d["indices"]].tolist()} for d in draws],
            "matched_control_last_losses": matched_traces, "saved_adapted_first_draw_models": saved_models,
            "prediction_sha256": digest(directory/"predictions.csv.gz")})
        print(json.dumps({"stage": "final", "seed": seed, "evaluations": 5*len(draws)}), flush=True)
    summaries = []
    for method in ("source_zero", "selected", "selected_matched_zero", "diagnostic", "diagnostic_matched_zero"):
        rows = [r for r in all_rows if r["method"] == method]
        result = {"method": method, "candidate_id": rows[0]["candidate_id"], "evaluations": len(rows), "seeds": protocol["seeds"]}
        for m in METRICS:
            for name, operation in (("mean", np.mean), ("sd", np.std), ("min", np.min), ("max", np.max)):
                result[m+"_"+name] = float(operation([r[m] for r in rows]))
            result[m+"_gain_over_source_mean"] = float(np.mean([r[m+"_gain_over_source"] for r in rows]))
            result[m+"_gain_over_matched_mean"] = float(np.mean([r[m+"_gain_over_matched"] for r in rows]))
        result.update(positive_draws=sum(r["accuracy_gain_over_source"] > 0 for r in rows),
            unchanged_draws=sum(r["accuracy_gain_over_source"] == 0 for r in rows),
            negative_draws=sum(r["accuracy_gain_over_source"] < 0 for r in rows))
        summaries.append(result)
    write_json(output/"final_metrics.json", all_rows); write_json(output/"summary.json", summaries)
    write_json(output/"final_metadata.json", records)
    with (output/"random_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(random_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(random_rows)
    write_json(output/"status.json", {"status": "completed", "final_seconds": time.monotonic()-started,
        "development_evaluations": lock["development_evaluations"], "final_evaluations": len(all_rows),
        "prediction_rows": sum(r["total"] for r in all_rows), "selection_lock_sha256": lock_sha,
        "final_metrics_sha256": digest(output/"final_metrics.json"), "summary_sha256": digest(output/"summary.json"),
        "source_objects_preserved": True, "query_labels_used_to_fit": False,
        "same_batch_retrospective": True, "query_draws_independent_batches": False})
    return summaries
