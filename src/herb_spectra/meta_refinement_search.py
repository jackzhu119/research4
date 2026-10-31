"""Predeclared source-anchored first-order MAML refinement on locked MLPs.

This is a classification engineering experiment, not the cited regression
paper's exact reproduction. Final target truth never enters source training,
target adaptation, or method selection. All model changes operate on copies.
"""
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call

from .calibration_exploration import encoded, fast_metrics, nested_draws
from .paper_fewshot_classification import sample_meta_episodes, state_fingerprint

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def meta_grid():
    return [{"id": f"m{i:03d}", "updates": updates, "inner_learning_rate": inner,
             "outer_learning_rate": .0001, "source_anchor_weight": 1.}
            for i, (updates, inner) in enumerate((u, rate) for u in (100, 500) for rate in (.001, .01))]


def candidate_grid():
    result = []
    for meta in meta_grid():
        for learning_rate in (.001, .01):
            for steps in (5, 20):
                result.append({"id": f"r{len(result):03d}_meta_ft", "meta_id": meta["id"],
                               "learning_rate": learning_rate, "steps": steps,
                               "source_replay_weight": 1., "proximal_lambda": .1})
    return result


def context_meta_seed(context):
    """Final source uses a distinct predeclared offset, not a fabricated task."""
    episode = context["episode"]
    return int(context["seed"]) + (1600 if episode is None else 1000+100*int(episode))


def _x(values):
    result = (values.detach().to(dtype=torch.float32, device="cpu") if torch.is_tensor(values)
              else torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)))
    if result.ndim != 2 or not len(result) or not torch.isfinite(result).all():
        raise ValueError("Expected finite, nonempty source-transformed feature matrix")
    return result


def _y(values, n, classes):
    values = np.asarray(values)
    if values.shape != (n,) or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Expected known integer class per physical specimen")
    return torch.as_tensor(values, dtype=torch.long)


def class_weights(source_y, classes, mode):
    if mode not in {"none", "balanced"}:
        raise ValueError("Unknown locked source class-weight policy")
    counts = torch.bincount(source_y, minlength=classes).float()
    if (counts == 0).any():
        raise ValueError("All known classes must remain in full-source supervision")
    return None if mode == "none" else len(source_y)/(classes*counts)


def model_scores(model, values):
    model.eval()
    with torch.no_grad():
        result = model(_x(values)).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite classification score")
    return result


def train_meta_pair(initial, source_x, source_y, source_ids, config, *, seed=42, class_weight="none"):
    """Train genuine FO-MAML and a same-task supervised continuation control.

    FO support gradients form detached fast parameters. Source-episode query
    gradients on those parameters are copied to original parameters, then a
    full-source supervised anchor gradient is added. No target query is used.
    """
    x = _x(source_x); y = _y(source_y, len(x), initial.head.out_features)
    source_ids = np.asarray(source_ids)
    if source_ids.shape != (len(x),) or len(set(source_ids.tolist())) != len(x):
        raise ValueError("Source records must have unique physical IDs")
    if config["updates"] < 1 or config["inner_learning_rate"] <= 0 or config["outer_learning_rate"] <= 0:
        raise ValueError("Positive meta update budget and learning rates are required")
    before = state_fingerprint(initial)
    tasks, policy = sample_meta_episodes(source_y, source_ids, episodes=config["updates"], seed=seed,
                                        way=3, support=1, query=1)
    weights = class_weights(y, initial.head.out_features, class_weight)
    models, records = {}, {}
    manifests = [{"episode": t["episode"], "classes": t["classes"],
                  "support_ids": source_ids[t["support"]].tolist(), "query_ids": source_ids[t["query"]].tolist()}
                 for t in tasks]
    for strategy in ("fomaml", "supervised"):
        model = deepcopy(initial); model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["outer_learning_rate"])
        losses = []
        for task in tasks:
            si, qi = task["support"], task["query"]
            optimizer.zero_grad(set_to_none=True)
            parameters = dict(model.named_parameters())
            if strategy == "fomaml":
                support_loss = F.cross_entropy(model(x[si]), y[si])
                support_gradients = torch.autograd.grad(support_loss, tuple(parameters.values()))
                adapted = {name: (parameter-config["inner_learning_rate"]*gradient).detach().requires_grad_(True)
                           for (name, parameter), gradient in zip(parameters.items(), support_gradients)}
                episodic_loss = F.cross_entropy(functional_call(model, adapted, (x[qi],)), y[qi])
                episodic_gradients = torch.autograd.grad(episodic_loss, tuple(adapted.values()))
                anchor_loss = F.cross_entropy(model(x), y, weight=weights)
                anchor_gradients = torch.autograd.grad(anchor_loss, tuple(parameters.values()))
                for parameter, meta_gradient, anchor_gradient in zip(parameters.values(), episodic_gradients, anchor_gradients):
                    parameter.grad = (meta_gradient + config["source_anchor_weight"]*anchor_gradient).detach().clone()
            else:
                episodic_loss = F.cross_entropy(model(x[qi]), y[qi])
                anchor_loss = F.cross_entropy(model(x), y, weight=weights)
                (episodic_loss + config["source_anchor_weight"]*anchor_loss).backward()
            if (not torch.isfinite(episodic_loss) or not torch.isfinite(anchor_loss)
                    or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values())):
                raise FloatingPointError("Nonfinite source meta gradient/loss")
            optimizer.step()
            losses.append({"update": task["episode"]+1, "source_episode_query_ce": float(episodic_loss.detach()),
                           "full_source_anchor_ce": float(anchor_loss.detach())})
        model.eval(); models[strategy] = model
        records[strategy] = {"losses": losses, "fingerprint": state_fingerprint(model),
                             "source_metrics": fast_metrics(np.asarray(source_y), model_scores(model, x).argmax(1),
                                                            initial.head.out_features, detailed=True)}
    if state_fingerprint(initial) != before:
        raise AssertionError("Source meta training mutated shared initializer")
    return models, {"config": config, "seed": seed, "class_weight": class_weight,
                    "initializer_fingerprint": before, "source_ids": source_ids.tolist(),
                    "policy": policy, "source_tasks": manifests, "models": records,
                    "same_task_ids_and_update_counts": True, "equal_FLOPs": False,
                    "first_order": True, "source_only_training": True}


def anchored_adaptation(initial, source_x, source_y, support_x, support_y, query_x, configs,
                        *, class_weight="none", target_loss=True):
    """Return exact SGD path checkpoints; no query labels or batch statistics.

    The matched no-support control uses identical source replay/proximal steps
    with target CE omitted. A copy is always used, including source-only steps.
    """
    if not configs:
        return {}
    before = state_fingerprint(initial)
    x = _x(source_x); y = _y(source_y, len(x), initial.head.out_features); q = _x(query_x)
    weights = class_weights(y, initial.head.out_features, class_weight)
    if target_loss:
        if support_x is None or support_y is None:
            raise ValueError("Labelled support is required for target adaptation")
        tx = _x(support_x); ty = _y(support_y, len(tx), initial.head.out_features)
        if set(ty.tolist()) != set(range(initial.head.out_features)):
            raise ValueError("Seven-class target adaptation needs every known class")
    groups = defaultdict(list)
    for config in configs:
        if config["learning_rate"] <= 0 or config["steps"] < 1 or config["source_replay_weight"] < 0 or config["proximal_lambda"] < 0:
            raise ValueError("Invalid target adaptation configuration")
        groups[(config["learning_rate"], config["source_replay_weight"], config["proximal_lambda"])].append(config)
    outputs = {}
    for (rate, replay, proximal), candidates in groups.items():
        model = deepcopy(initial); model.train()
        origin = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=rate)
        steps = defaultdict(list)
        for config in candidates:
            steps[config["steps"]].append(config["id"])
        for step in range(1, max(steps)+1):
            optimizer.zero_grad(set_to_none=True)
            loss = sum((p-origin[name]).square().sum()*proximal for name, p in model.named_parameters())
            if replay:
                loss = loss + replay*F.cross_entropy(model(x), y, weight=weights)
            if target_loss:
                loss = loss + F.cross_entropy(model(tx), ty)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite target adaptation loss")
            loss.backward(); optimizer.step()
            if step in steps:
                prediction = model_scores(model, q).argmax(1)
                for identifier in steps[step]:
                    outputs[identifier] = prediction.copy()
        model.train(False)
    if state_fingerprint(initial) != before:
        raise AssertionError("Target/source-only adaptation mutated frozen source model")
    return outputs


def choose_configuration(rows, candidates, selection):
    """Select only from development records, with own/source-matched controls."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["candidate_id"], row["stage"])].append(row)
    means = {}
    for key, cases in groups.items():
        means[key] = {name: float(np.mean([r[name] for r in cases]))
                      for name in (*METRICS, *("source_"+m for m in METRICS), *("matched_"+m for m in METRICS))}
        means[key]["evaluations"] = len(cases)
    configs = {c["id"]: c for c in candidates}; metas = {m["id"]: m for m in meta_grid()}
    def rank(strategy):
        return sorted(candidates, key=lambda c: (-means[(strategy, c["id"], "selection")]["accuracy"],
            metas[c["meta_id"]]["updates"], metas[c["meta_id"]]["inner_learning_rate"],
            c["learning_rate"], c["steps"], c["id"]))
    def passes(config, stage):
        values = means[("fomaml", config["id"], stage)]
        return all(values["accuracy"]-values[prefix+"accuracy"] > selection["minimum_accuracy_gain"]
                   and values["macro_f1"]-values[prefix+"macro_f1"] >= -selection["max_macro_f1_loss"]
                   and values["balanced_accuracy"]-values[prefix+"balanced_accuracy"] >= -selection["max_balanced_accuracy_loss"]
                   for prefix in ("source_", "matched_"))
    ranked = rank("fomaml")
    eligible = [c for c in ranked if passes(c, "selection")]
    proposed = eligible[0] if eligible else ranked[0]
    passed = bool(eligible) and passes(proposed, "gate")
    serialized_means = [{"strategy": k[0], "candidate_id": k[1], "stage": k[2], **v}
                        for k, v in sorted(means.items())]
    return {"selected": proposed if passed else None, "proposed": proposed, "diagnostic": ranked[0],
            "supervised_diagnostic": rank("supervised")[0], "fallback_to_reference_source": not passed,
            "reference_meta_id": selection["reference_meta_id"], "selected_using_final_query": False,
            "eligible_selection_candidates": len(eligible), "passed_gate": passed,
            "means": serialized_means}


def _pair(context, meta, directory, *, resume=False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory/"training.json"
    expected = {"config": meta, "initial_fingerprint": context["fingerprint"],
                "source_ids": context["source"]["record_ids"].tolist(), "seed": context["seed"],
                "preprocessor_sha256": context["preprocessor_sha256"],
                "base_selection_lock_sha256": context["selection_lock_sha256"]}
    if resume and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["expected"] != expected:
            raise ValueError("Cached source meta pair differs from locked source context")
        models = {}
        for strategy in ("fomaml", "supervised"):
            path = directory/(strategy+".pt")
            if digest(path) != metadata["artifacts"][strategy]["sha256"]:
                raise ValueError("Cached meta source weight hash changed")
            model = deepcopy(context["model"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); model.eval()
            if state_fingerprint(model) != metadata["artifacts"][strategy]["fingerprint"]:
                raise ValueError("Cached meta source state fingerprint changed")
            models[strategy] = model
        return models, metadata
    models, record = train_meta_pair(context["model"], context["source"]["X"], context["source"]["y"],
        context["source"]["record_ids"], meta, seed=context_meta_seed(context),
        class_weight=context["candidate"]["class_weight"])
    artifacts = {}
    for strategy, model in models.items():
        path = directory/(strategy+".pt"); torch.save(model.state_dict(), path)
        artifacts[strategy] = {"sha256": digest(path), "fingerprint": state_fingerprint(model)}
    metadata = {"expected": expected, "training": record, "artifacts": artifacts}
    write_json(metadata_path, metadata)
    return models, metadata


def _score_case(truth, predicted, source_predicted, matched_predicted, classes):
    result = fast_metrics(truth, predicted, classes, detailed=True)
    for prefix, values in (("source_", source_predicted), ("matched_", matched_predicted)):
        for name, value in fast_metrics(truth, values, classes, detailed=True).items():
            result[prefix+name] = value
    result["corrected"] = int(np.sum((source_predicted != truth) & (predicted == truth)))
    result["new_errors"] = int(np.sum((source_predicted == truth) & (predicted != truth)))
    return result


def evaluate_development(context, protocol, output, *, resume=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metrics_path = output/"metrics.json"; prediction_path = output/"predictions.csv.gz"
    if resume and metrics_path.exists():
        record = json.loads(metrics_path.read_text())
        if record["source_fingerprint"] != context["fingerprint"] or digest(prediction_path) != record["predictions_sha256"]:
            raise ValueError("Development checkpoint changed")
        for meta in meta_grid():
            _pair(context, meta, output/meta["id"], resume=True)
        return record["rows"]
    classes = np.asarray(context["classes"]); source = context["source"]; pool = context["support_pool"]
    query = context["prediction"]; truth = np.asarray(query["y"])
    if set(source["record_ids"]) & set(pool["record_ids"]) or set(source["record_ids"]) & set(query["record_ids"]) or set(pool["record_ids"]) & set(query["record_ids"]):
        raise ValueError("Source, support pool, and query physical IDs overlap")
    draws = nested_draws(pool["y"], np.arange(len(classes)), [1],
                         protocol["development"]["support_draws"], protocol["development"]["support_seed"]+context["episode"])
    rows, controls, manifests = [], {}, []
    temporary = prediction_path.with_name(prediction_path.name+".partial")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "seed", "strategy", "candidate_id", "repeat", "record_id", "y_true", "y_predicted", "is_correct"], lineterminator="\n")
        writer.writeheader()
        for meta in meta_grid():
            models, _ = _pair(context, meta, output/meta["id"], resume=resume)
            candidates = [c for c in candidate_grid() if c["meta_id"] == meta["id"]]
            for strategy, model in models.items():
                source_prediction = model_scores(model, query["X"]).argmax(1)
                matched = anchored_adaptation(model, source["X"], source["y"], None, None,
                    query["X"], candidates, class_weight=context["candidate"]["class_weight"], target_loss=False)
                for candidate in candidates:
                    controls[strategy+":"+candidate["id"]] = {"source_prediction": source_prediction.tolist(), "matched_prediction": matched[candidate["id"]].tolist()}
                for draw in draws:
                    indices = draw["indices"]
                    predictions = anchored_adaptation(model, source["X"], source["y"], pool["X"][indices], pool["y"][indices],
                        query["X"], candidates, class_weight=context["candidate"]["class_weight"])
                    for candidate in candidates:
                        prediction = predictions[candidate["id"]]
                        row = {"episode": context["episode"], "seed": context["seed"], "strategy": strategy,
                            "candidate_id": candidate["id"], "stage": "selection" if context["episode"] < protocol["development"]["selection_episodes"] else "gate",
                            "repeat": draw["repeat"], **_score_case(truth, prediction, source_prediction, matched[candidate["id"]], len(classes))}
                        rows.append(row)
                        for identifier, label, pred in zip(query["record_ids"], truth, prediction):
                            writer.writerow({key: row[key] for key in ("episode", "seed", "strategy", "candidate_id", "repeat")} | {
                                "record_id": identifier, "y_true": classes[label], "y_predicted": classes[pred], "is_correct": int(label == pred)})
            manifests.append({"meta_id": meta["id"], "source_fingerprints": {name: state_fingerprint(model) for name, model in models.items()}})
    temporary.replace(prediction_path)
    if state_fingerprint(context["model"]) != context["fingerprint"]:
        raise AssertionError("Development changed locked neural source")
    write_json(metrics_path, {"source_fingerprint": context["fingerprint"], "rows": rows, "controls": controls,
        "meta_sources": manifests, "source_ids": source["record_ids"].tolist(), "pool_ids": pool["record_ids"].tolist(),
        "query_ids": query["record_ids"].tolist(), "classes": classes.tolist(),
        "draws": [{"repeat": d["repeat"], "support_ids": pool["record_ids"][d["indices"]].tolist()} for d in draws],
        "predictions_sha256": digest(prediction_path)})
    return rows


def _final_specs(lock):
    reference = {"id": "reference_source", "meta_id": lock["reference_meta_id"]}
    selected = lock["selected"] or reference
    return [("selected", "fomaml", selected), ("diagnostic", "fomaml", lock["diagnostic"]),
            ("supervised_diagnostic", "supervised", lock["supervised_diagnostic"])]


def evaluate_final(context, truth_by_id, lock, protocol, output, *, resume=False):
    """Fit/adapt without final truth; join labels only at the scoring boundary."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output/"predictions.csv.gz"; checkpoint = output/"evaluation.json"
    if resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["source_fingerprint"] != context["fingerprint"] or digest(path) != saved["predictions_sha256"]:
            raise ValueError("Final checkpoint changed")
        for meta in meta_grid():
            if (output/meta["id"] / "training.json").exists():
                _pair(context, meta, output/meta["id"], resume=True)
        return saved["rows"]
    source, pool, query = context["source"], context["support_pool"], context["prediction"]
    if "y" in query:
        raise ValueError("Final prediction payload must not expose labels")
    classes = np.asarray(context["classes"])
    if set(source["record_ids"]) & set(pool["record_ids"]) or set(source["record_ids"]) & set(query["record_ids"]) or set(pool["record_ids"]) & set(query["record_ids"]):
        raise ValueError("Final physical specimen roles overlap")
    draws = nested_draws(pool["y"], np.arange(len(classes)), [1], protocol["final"]["support_draws"], protocol["final"]["support_seed"])
    pairs = {}; meta_by_id = {m["id"]: m for m in meta_grid()}
    specs = _final_specs(lock)
    for meta_id in sorted(set(spec[2]["meta_id"] for spec in specs)):
        pairs[meta_id], _ = _pair(context, meta_by_id[meta_id], output/meta_id, resume=resume)
    plans = []
    for label, strategy, candidate in specs:
        model = pairs[candidate["meta_id"]][strategy]
        source_prediction = model_scores(model, query["X"]).argmax(1)
        matched = source_prediction if candidate["id"] == "reference_source" else anchored_adaptation(model,
            source["X"], source["y"], None, None, query["X"], [candidate],
            class_weight=context["candidate"]["class_weight"], target_loss=False)[candidate["id"]]
        plans.append((label, strategy, candidate, model, source_prediction, matched))
    temporary = path.with_name(path.name+".partial"); rows = []; all_support = []
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "repeat", "method", "strategy", "candidate_id", "record_id", "y_true", "y_predicted", "is_correct"], lineterminator="\n"); writer.writeheader()
        for draw in draws:
            indices = draw["indices"]; all_support.append({"repeat": draw["repeat"], "support_ids": pool["record_ids"][indices].tolist()})
            predictions = []
            for label, strategy, candidate, model, source_pred, matched_pred in plans:
                prediction = source_pred if candidate["id"] == "reference_source" else anchored_adaptation(model,
                    source["X"], source["y"], pool["X"][indices], pool["y"][indices], query["X"], [candidate],
                    class_weight=context["candidate"]["class_weight"])[candidate["id"]]
                predictions.extend([(label, strategy, candidate, prediction, source_pred, matched_pred),
                    (label+"_source_zero", strategy, candidate, source_pred, source_pred, matched_pred),
                    (label+"_matched_zero", strategy, candidate, matched_pred, source_pred, matched_pred)])
            # Only this scoring block receives the separate final truth mapping.
            truth = encoded([truth_by_id[str(identifier)] for identifier in query["record_ids"]], classes)
            for label, strategy, candidate, predicted, source_pred, matched_pred in predictions:
                row = {"seed": context["seed"], "repeat": draw["repeat"], "method": label,
                       "strategy": strategy, "candidate_id": candidate["id"],
                       **_score_case(truth, predicted, source_pred, matched_pred, len(classes))}
                rows.append(row)
                for identifier, actual, pred in zip(query["record_ids"], truth, predicted):
                    writer.writerow({k: row[k] for k in ("seed", "repeat", "method", "strategy", "candidate_id")} | {
                        "record_id": identifier, "y_true": classes[actual], "y_predicted": classes[pred], "is_correct": int(actual == pred)})
    temporary.replace(path)
    if state_fingerprint(context["model"]) != context["fingerprint"]:
        raise AssertionError("Final meta training changed locked source initializer")
    write_json(checkpoint, {"source_fingerprint": context["fingerprint"], "rows": rows, "draws": all_support,
        "source_ids": source["record_ids"].tolist(), "pool_ids": pool["record_ids"].tolist(),
        "query_ids": query["record_ids"].tolist(), "predictions_sha256": digest(path)})
    return rows


def summarize_final(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    result = []
    for method, cases in groups.items():
        entry = {"method": method, "evaluations": len(cases), "seeds": sorted(set(r["seed"] for r in cases)),
                 "candidate_id": cases[0]["candidate_id"], "strategy": cases[0]["strategy"]}
        for metric in METRICS:
            values = np.asarray([r[metric] for r in cases]); own = np.asarray([r["source_"+metric] for r in cases]); matched = np.asarray([r["matched_"+metric] for r in cases])
            entry[metric+"_mean"] = float(values.mean()); entry[metric+"_sd"] = float(values.std(ddof=1))
            entry[metric+"_source_gain"] = float((values-own).mean()); entry[metric+"_matched_gain"] = float((values-matched).mean())
        delta = np.asarray([r["accuracy"]-r["source_accuracy"] for r in cases])
        entry.update(positive_draws=int(np.sum(delta > 1e-12)), unchanged_draws=int(np.sum(np.abs(delta) <= 1e-12)),
                     negative_draws=int(np.sum(delta < -1e-12)), mean_corrected=float(np.mean([r["corrected"] for r in cases])),
                     mean_new_errors=float(np.mean([r["new_errors"] for r in cases])))
        result.append(entry)
    return result


def _provenance(input_path, plan_path, protocol_path, base_run):
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/"scripts/run_meta_refinement_search.py", Path(__file__).with_name("neural_base_search.py"),
             Path(__file__).with_name("paper_fewshot_classification.py"), Path(__file__).with_name("calibration_exploration.py")]
    return {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path), "protocol_sha256": digest(protocol_path),
            "grid_sha256": object_digest(candidate_grid()), "base_selection_lock_sha256": digest(Path(base_run)/"selection_lock.json"),
            "base_run_manifest_sha256": digest(Path(base_run)/"run_manifest.json"),
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in files}, "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scikit-learn", "joblib")}}


def run_search(input_path, plan_path, protocol_path, base_run, output, *, stage="development", resume=False):
    from .neural_base_search import load_locked_contexts
    if stage not in {"development", "final"}:
        raise ValueError("Expected development or final stage")
    torch.set_num_threads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol["candidate_grid_sha256"] != object_digest(candidate_grid()) or protocol["candidate_configurations"] != len(candidate_grid()):
        raise ValueError("Frozen meta candidate grid differs")
    provenance = _provenance(input_path, plan_path, protocol_path, base_run)
    manifest = output/"run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Source code, input, base source lock or dependency provenance changed")
    write_json(manifest, provenance); write_json(output/"candidate_grid.json", candidate_grid())
    snapshot = output/"protocol_snapshot.json"
    if snapshot.exists() and digest(snapshot) != provenance["protocol_sha256"]:
        raise ValueError("Archived frozen protocol snapshot changed")
    shutil.copyfile(protocol_path, snapshot)
    with (output/"stage_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "stage_start", "stage": stage, "resume": resume,
            "utc": datetime.now(timezone.utc).isoformat(), "provenance_sha256": object_digest(provenance)})+"\n")
    started = time.monotonic()
    if stage == "development":
        lock_path = output/"selection_lock.json"
        if resume and lock_path.exists():
            lock = json.loads(lock_path.read_text())
            if lock["provenance_sha256"] != object_digest(provenance):
                raise ValueError("Completed development lock provenance differs")
            contexts, truth = load_locked_contexts(input_path, plan_path, base_run, scope="development")
            if truth:
                raise ValueError("Development resume must not load final target truth")
            for context in contexts:
                directory = output/"development"/f"episode_{context['episode']:02d}"/f"seed_{context['seed']}"
                if not (directory/"metrics.json").exists() or any(not (directory/m["id"]/"training.json").exists() for m in meta_grid()):
                    raise ValueError("Completed development artifacts are missing; refuse implicit refit")
                evaluate_development(context, protocol, directory, resume=True)
            return {"status": "development_already_completed", "lock": lock}
        contexts, truth = load_locked_contexts(input_path, plan_path, base_run, scope="development")
        if truth:
            raise ValueError("Development context loading unexpectedly returned final truth")
        if len(contexts) != protocol["development"]["episodes"]*len(protocol["seeds"]):
            raise ValueError("Unexpected development context count")
        rows = []
        for context in contexts:
            directory = output/"development"/f"episode_{context['episode']:02d}"/f"seed_{context['seed']}"
            rows.extend(evaluate_development(context, protocol, directory, resume=resume))
            write_json(output/"status.json", {"stage": stage, "completed_contexts": len(rows)//(len(candidate_grid())*2*protocol["development"]["support_draws"]), "development_rows": len(rows)})
        lock = choose_configuration(rows, candidate_grid(), protocol["selection"])
        lock.update(provenance_sha256=object_digest(provenance), source_only_development=True, development_evaluations=len(rows))
        write_json(output/"all_development_metrics.json", rows); write_json(lock_path, lock)
        write_json(output/"selection_lock_sha256.json", {"sha256": digest(lock_path)})
        status = {"stage": "development_complete_final_not_scored", "development_rows": len(rows),
                  "seconds": time.monotonic()-started, "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output/"status.json", status); write_json(output/"development_status.json", status)
        return {"status": "development_complete_final_not_scored", "lock": lock}
    lock = json.loads((output/"selection_lock.json").read_text())
    if lock["provenance_sha256"] != object_digest(provenance):
        raise ValueError("Final evaluation requires unchanged locked development provenance")
    contexts, truth = load_locked_contexts(input_path, plan_path, base_run, scope="final")
    if len(contexts) != len(protocol["seeds"]) or len(truth) != protocol["final"]["query_samples"]:
        raise ValueError("Unexpected final source/truth shape")
    rows = []
    for context in contexts:
        rows.extend(evaluate_final(context, truth, lock, protocol, output/"final"/f"seed_{context['seed']}", resume=resume))
    summary = summarize_final(rows)
    write_json(output/"final_metrics.json", rows); write_json(output/"summary.json", summary)
    random_ids = np.asarray(sorted(truth))[np.random.default_rng(protocol["final"]["random_query_seed"]).choice(len(truth), protocol["final"]["random_query_examples"], replace=False)].tolist()
    write_json(output/"random_query_ids.json", {"record_ids": random_ids, "selection_seed": protocol["final"]["random_query_seed"], "chosen_without_labels": True})
    examples = []
    for context in contexts:
        with gzip.open(output/"final"/f"seed_{context['seed']}"/"predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            examples.extend(r for r in csv.DictReader(handle) if r["record_id"] in set(random_ids) and int(r["repeat"]) == 0)
    with (output/"random_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(examples[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(examples)
    status = {"stage": "final_complete", "final_evaluations": len(rows), "prediction_rows": sum(r["total"] for r in rows),
        "lock_sha256": digest(output/"selection_lock.json"), "seconds": time.monotonic()-started, "same_batch_retrospective": True,
        "named_origins_verified": False, "new_independent_batches": 0, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"status.json", status); write_json(output/"final_status.json", status)
    return {"status": "final_complete", "summary": summary}
