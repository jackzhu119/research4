"""Locked latent-readout adaptation after source-only spectral model search.

Development uses only the original training role. Source improvements and
support-label improvements are reported separately. The query label join is a
scoring operation, never an argument to source or adapter fitting.
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np

from .calibration_exploration import (development_episodes, encoded, fast_metrics,
                                    nested_draws)
from .latent_adaptation import (build_latent_context, fit_latent_adapter,
                               latent_candidate_grid)
from .spectral_models import fit_source_candidate, source_candidate_grid
from .spectral_source_search import (load_source_model, source_metadata_path,
                                    source_model_path)
from .within_batch_holdout import prepare_holdout


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
DEV_METRICS = METRICS + tuple("matched_" + key for key in METRICS)
FAMILY_RANK = {"identity": 0, "mahalanobis_centroid": 1,
               "local_memory": 2, "anchored_ridge": 3}
DEV_FIELDS = ["source_id", "episode", "stage", "repeat", "candidate_id", "family",
              *DEV_METRICS, "status", "error"]
PREDICTION_FIELDS = ["source_id", "method", "candidate_id", "repeat", "record_id",
                     "y_true", "y_predicted", "is_correct", "support_labels_used"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _code_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ["src/herb_spectra/latent_calibration_search.py",
             "scripts/run_latent_calibration_search.py",
             "src/herb_spectra/latent_adaptation.py",
             "src/herb_spectra/spectral_models.py",
             "src/herb_spectra/spectral_source_search.py",
             "src/herb_spectra/calibration_exploration.py",
             "src/herb_spectra/within_batch_holdout.py",
             "src/herb_spectra/classification.py",
             "src/herb_spectra/benchmark.py"]
    return {name: digest(root / name) for name in paths}


def validate_source_run(source_run, input_path, plan_path, protocol_path):
    """Reject stale/cohort-incompatible source caches before any joblib load."""
    root = Path(source_run)
    required = ["source_status.json", "source_run_manifest.json",
                "source_selection_lock.json", "source_selection_lock_sha256.json",
                "source_candidate_grid.json", "development_partitions.json",
                "source_development_aggregate.json"]
    if any(not (root / name).is_file() for name in required):
        raise ValueError("A complete, verified Stage A source search is required")
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    status, manifest, lock = (read(name) for name in required[:3])
    if status.get("status") != "completed" or status.get("final_query_scored") is not False:
        raise ValueError("Stage A must be complete without outer-query scoring")
    expected = {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
                "protocol_sha256": digest(protocol_path)}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage A input/cohort/protocol hashes disagree")
    runtime = {name: importlib.metadata.version(name) for name in
               ["numpy", "scipy", "scikit-learn", "joblib"]}
    if manifest.get("python") != platform.python_version() or manifest.get("dependencies") != runtime:
        raise ValueError("Stage A runtime differs from current source-cache consumers")
    if digest(root / "source_selection_lock.json") != status.get("source_selection_lock_sha256"):
        raise ValueError("Stage A selection lock hash differs from completed status")
    if read("source_selection_lock_sha256.json").get("sha256") != digest(root / "source_selection_lock.json"):
        raise ValueError("Stage A selection lock checksum differs")
    checks = [("source_run_manifest_sha256", "source_run_manifest.json"),
              ("partitions_sha256", "development_partitions.json"),
              ("development_aggregate_sha256", "source_development_aggregate.json")]
    if any(lock.get(key) != digest(root / filename) for key, filename in checks):
        raise ValueError("Stage A source evidence hash changed")
    grid = read("source_candidate_grid.json")
    if object_digest(grid) != lock.get("source_grid_sha256"):
        raise ValueError("Stage A source grid changed")
    repo = Path(__file__).resolve().parents[2]
    if any(digest(repo / name) != sha for name, sha in manifest["source_code_sha256"].items()):
        raise ValueError("Stage A executed source code differs from the current code")
    chosen = lock.get("candidates_for_adaptation", [])
    if not chosen or len({c["id"] for c in chosen}) != len(chosen):
        raise ValueError("Stage A must lock unique adaptation source candidates")
    known = {c["id"]: c for c in grid}
    if any(known.get(c["id"]) != c for c in chosen):
        raise ValueError("Adaptation source candidate is outside the locked source grid")
    return lock, manifest, read("development_partitions.json")


def make_manifest(input_path, plan_path, protocol_path, source_run, candidates):
    lock, source_manifest, _ = validate_source_run(source_run, input_path, plan_path, protocol_path)
    return {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
            "protocol_sha256": digest(protocol_path),
            "source_run_manifest_sha256": digest(Path(source_run) / "source_run_manifest.json"),
            "source_selection_lock_sha256": digest(Path(source_run) / "source_selection_lock.json"),
            "source_candidate_ids": [c["id"] for c in lock["candidates_for_adaptation"]],
            "latent_grid_sha256": object_digest(candidates), "source_code_sha256": _code_hashes(),
            "python": platform.python_version(), "source_runtime": {
                key: source_manifest[key] for key in ("python", "dependencies")},
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ["numpy", "scipy", "scikit-learn", "joblib"]}}


class LatentTransformCache:
    """Memoize pure pointwise frozen transforms; no query statistic is fitted."""
    def __init__(self, context):
        self.context = context
        self.projection_cache = {}
        self.distance_cache = {}

    def __getattr__(self, key):
        return getattr(self.context, key)

    def project(self, features):
        key = id(features)
        cached = self.projection_cache.get(key)
        if cached is None or cached[0] is not features:
            cached = (features, self.context.project(features))
            self.projection_cache[key] = cached
        return cached[1]

    def design(self, features):
        projected = self.project(features)
        return np.column_stack([projected, np.ones(len(projected))])

    def source_distances(self, features):
        key = id(features)
        cached = self.distance_cache.get(key)
        if cached is None or cached[0] is not features:
            left, right = self.project(features), self.context.source_latent
            distances = np.maximum((left*left).sum(axis=1)[:, None]
                + (right*right).sum(axis=1)[None, :] - 2*left@right.T, 0.)
            cached = (features, distances)
            self.distance_cache[key] = cached
        return cached[1]


def latent_complexity(candidate):
    return (FAMILY_RANK[candidate["family"]], candidate.get("blend", 0.),
            candidate.get("support_boost", 0.), candidate.get("support_weight", 0.),
            -candidate.get("source_anchor", 0.), -candidate.get("ridge", 0.),
            -candidate.get("source_pseudocount", 0.), -candidate.get("neighbors", 0),
            candidate["id"])


def choose_latent_with_gate(stats, candidates, *, minimum_accuracy_gain=.001,
        max_macro_f1_loss=.005, max_balanced_accuracy_loss=.005, tie_tolerance=.001):
    """Choose on the first development stage, inspect only that winner at gate."""
    lookup = {c["id"]: c for c in candidates}
    identity = next(c for c in candidates if c["family"] == "identity")

    def passes(identifier, stage):
        value, base = stats[stage][identifier], stats[stage][identity["id"]]
        return (not value["failures"] and value["evaluations"] == base["evaluations"]
                and value["accuracy"]-base["accuracy"] > minimum_accuracy_gain
                and value["macro_f1"]-base["macro_f1"] >= -max_macro_f1_loss
                and value["balanced_accuracy"]-base["balanced_accuracy"] >= -max_balanced_accuracy_loss
                and value["accuracy"]-value["matched_accuracy"] > minimum_accuracy_gain
                and value["macro_f1"]-value["matched_macro_f1"] >= -max_macro_f1_loss
                and value["balanced_accuracy"]-value["matched_balanced_accuracy"] >= -max_balanced_accuracy_loss)

    def best(ids):
        top = max(stats["selection"][key]["accuracy"] for key in ids)
        return min((lookup[key] for key in ids
                    if stats["selection"][key]["accuracy"] >= top-tie_tolerance),
                   key=latent_complexity)

    eligible = [c["id"] for c in candidates if passes(c["id"], "selection")]
    proposed = best(eligible) if eligible else identity
    selected = proposed if proposed["family"] != "identity" and passes(proposed["id"], "gate") else identity
    families = {}
    for family in sorted({c["family"] for c in candidates}):
        available = [c["id"] for c in candidates if c["family"] == family
                     and not stats["selection"][c["id"]]["failures"]]
        winner = best(available) if available else identity
        families[family] = {"candidate": winner, "status": "development_selected" if available else "failed",
            "passes_gate": passes(winner["id"], "gate"),
            "selection": stats["selection"][winner["id"]], "gate": stats["gate"][winner["id"]]}
    return {"proposed": proposed, "selected": selected,
            "fallback_to_identity": selected["family"] == "identity",
            "selection": stats["selection"][proposed["id"]],
            "gate": stats["gate"][proposed["id"]],
            "identity_selection": stats["selection"][identity["id"]],
            "identity_gate": stats["gate"][identity["id"]],
            "family_winners": families,
            "selection_scope": "original training role development only; never outer query labels"}


def _partition_indices(training, records):
    lookup = {str(value): i for i, value in enumerate(training["record_ids"])}
    result = {}
    for role in ("source", "support_pool", "query"):
        try:
            result[role] = np.asarray([lookup[str(value)] for value in records[role]], dtype=int)
        except KeyError as exc:
            raise ValueError("Development partition contains an ID outside original training") from exc
    all_indices = np.concatenate(list(result.values()))
    if len(all_indices) != len(training["record_ids"]) or len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Development source/support/query must be complete and disjoint")
    return result


def prediction_only(context, candidate, support_features, support_scores, support_y,
                    query_features, query_scores, indices=None):
    """A pure prediction boundary; it has no query truth or query ID argument."""
    adapter = (fit_latent_adapter(context, candidate) if indices is None else
        fit_latent_adapter(context, candidate, support_features[indices],
                           support_scores[indices], support_y[indices]))
    return np.argmax(adapter.predict_scores(query_features, query_scores), axis=1), adapter.metadata


def evaluate_latent_episode(training, partition, source_candidate, source_run,
        candidates, config, episode, destination):
    model = load_source_model(source_run, source_candidate["id"], episode)
    fingerprint = joblib.hash(model)
    source, pool, query = (partition[key] for key in ("source", "support_pool", "query"))
    X = training["X_alpha"]
    zsource, ssource = model.transform(X[source]), model.predict_scores(X[source])
    zpool, spool = model.transform(X[pool]), model.predict_scores(X[pool])
    zquery, squery = model.transform(X[query]), model.predict_scores(X[query])
    classes = np.asarray(model.classes_)
    context_raw = build_latent_context(zsource, ssource, training["y"][source], classes,
                                       seed=config["model_seed"])
    context_before = joblib.hash(context_raw)
    context = LatentTransformCache(context_raw)
    truth = encoded(training["y"][query], classes)
    controls, control_errors = {}, {}
    for candidate in candidates:
        try:
            pred, _ = prediction_only(context, candidate, zpool, spool, training["y"][pool],
                                      zquery, squery, None)
            controls[candidate["id"]] = fast_metrics(truth, pred, len(classes))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            control_errors[candidate["id"]] = f"{type(exc).__name__}: {exc}"
    draws = nested_draws(training["y"][pool], classes, [1], config["support_draws"],
                         config["support_seed"]+episode)
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".partial")
    failures, calls = [], 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for draw in draws:
            for candidate in candidates:
                row = {"source_id": source_candidate["id"], "episode": episode,
                    "stage": "selection" if episode < config["selection_episodes"] else "gate",
                    "repeat": draw["repeat"], "candidate_id": candidate["id"],
                    "family": candidate["family"], "status": "pass", "error": ""}
                try:
                    if candidate["id"] in control_errors:
                        raise ValueError("Matched control failed: " + control_errors[candidate["id"]])
                    row.update({"matched_" + key: controls[candidate["id"]][key] for key in METRICS})
                    pred, _ = prediction_only(context, candidate, zpool, spool, training["y"][pool],
                                              zquery, squery, draw["indices"])
                    row.update(fast_metrics(truth, pred, len(classes)))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    row.update({key: "" for key in DEV_METRICS})
                    row.update(status="fail", error=f"{type(exc).__name__}: {exc}")
                    failures.append({key: row[key] for key in ("candidate_id", "repeat", "error")})
                writer.writerow(row); calls += 1
    if joblib.hash(model) != fingerprint or joblib.hash(context_raw) != context_before:
        raise AssertionError("Frozen source model or latent context changed during development")
    temporary.replace(destination)
    return {"source_id": source_candidate["id"], "episode": episode,
        "new_source_fit_calls": 0, "candidate_fit_calls": calls, "failures": failures,
        "source_model_sha256": digest(source_model_path(source_run, source_candidate["id"], episode)),
        "source_fingerprint_before": fingerprint, "source_fingerprint_after": joblib.hash(model),
        "context_fingerprint_before": context_before, "context_fingerprint_after": joblib.hash(context_raw),
        "partition": {key: training["record_ids"][values].tolist() for key, values in partition.items()},
        "draws": [{"repeat": d["repeat"],
                   "support_ids": training["record_ids"][pool][d["indices"]].tolist()} for d in draws],
        "metrics_file_sha256": digest(destination)}


def aggregate_latent(paths):
    accum = defaultdict(lambda: {"sums": np.zeros(len(DEV_METRICS)), "evaluations": 0, "failures": 0})
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                entry = accum[(row["stage"], row["candidate_id"])]
                if row["status"] == "pass":
                    entry["sums"] += np.asarray([float(row[key]) for key in DEV_METRICS])
                    entry["evaluations"] += 1
                else:
                    entry["failures"] += 1
    result = {"selection": {}, "gate": {}}
    for (stage, identifier), value in accum.items():
        result[stage][identifier] = {**dict(zip(DEV_METRICS,
                (value["sums"]/max(1,value["evaluations"])).tolist())),
                "evaluations": value["evaluations"], "failures": value["failures"]}
    return result


def _summaries(cases):
    groups = defaultdict(list)
    for row in cases:
        groups[(row["source_id"], row["method"])].append(row)
    result = []
    for (source_id, method), rows in sorted(groups.items()):
        value = {"source_id": source_id, "method": method,
                 "candidate_id": rows[0]["candidate_id"], "evaluations": len(rows)}
        for metric in METRICS:
            numbers = np.asarray([r[metric] for r in rows])
            for label, operation in [("mean",np.mean),("sd",np.std),("min",np.min),("max",np.max)]:
                value[f"{metric}_{label}"] = float(operation(numbers))
            value[metric+"_gain_vs_own_source_mean"] = float(np.mean([
                r[metric+"_gain_vs_own_source"] for r in rows]))
        value.update(accuracy_gain_vs_reference_mean=float(np.mean([r["accuracy_gain_vs_reference"] for r in rows])),
            accuracy_gain_vs_matched_mean=float(np.mean([r["accuracy_gain_vs_matched"] for r in rows])),
            positive_draws=sum(r["accuracy_gain_vs_own_source"]>0 for r in rows),
            negative_draws=sum(r["accuracy_gain_vs_own_source"]<0 for r in rows),
            unchanged_draws=sum(r["accuracy_gain_vs_own_source"]==0 for r in rows),
            corrected_mean=float(np.mean([r["corrected_errors"] for r in rows])),
            harmed_mean=float(np.mean([r["new_errors"] for r in rows])))
        result.append(value)
    return result


def evaluate_final_source(training, calibration, prediction, truth, source_candidate,
        choice, candidates, config, output, reference_prediction=None, random_ids=()):
    if "y" in prediction:
        raise ValueError("Final prediction features must not contain truth labels")
    directory = Path(output) / "final" / source_candidate["id"]
    directory.mkdir(parents=True, exist_ok=True)
    model = fit_source_candidate(source_candidate, training["X_alpha"], training["y"],
                                 training["axis"], config["model_seed"])
    before = joblib.hash(model)
    artifact = directory / "source_model.joblib"
    joblib.dump(model, artifact, compress=3)
    zs, ss = model.transform(training["X_alpha"]), model.predict_scores(training["X_alpha"])
    zc, sc = model.transform(calibration["X_alpha"]), model.predict_scores(calibration["X_alpha"])
    zq, sq = model.transform(prediction["X_alpha"]), model.predict_scores(prediction["X_alpha"])
    classes = np.asarray(model.classes_)
    if set(classes.tolist()) != set(training["y"].tolist()):
        raise ValueError("Final source classes differ from fixed training labels")
    context_raw = build_latent_context(zs, ss, training["y"], classes, seed=config["model_seed"])
    context_before = joblib.hash(context_raw)
    context = LatentTransformCache(context_raw)
    base_pred = np.argmax(sq, axis=1)
    reference_prediction = base_pred if reference_prediction is None else np.asarray(reference_prediction)
    # Truth enters only the scoring boundary. prediction_only has no such input.
    ids = prediction["record_ids"].tolist()
    y = encoded([truth[str(identifier)] for identifier in ids], classes)
    base_metrics = fast_metrics(y, base_pred, len(classes), detailed=True)
    reference_accuracy = float(np.mean(reference_prediction == y))
    identity = next(c for c in candidates if c["family"] == "identity")
    methods = {"source_original": identity, "development_selected": choice["selected"]}
    for family, value in choice["family_winners"].items():
        if family != "identity":
            methods["family_"+family] = value["candidate"]
    unique = {c["id"]: c for c in methods.values()}
    controls = {identifier: prediction_only(context,candidate,zc,sc,calibration["y"],zq,sq)[0]
                for identifier,candidate in unique.items()}
    for identifier,candidate in unique.items():
        if candidate["family"] != "identity":
            methods["matched_zero_"+identifier] = candidate
    draws = nested_draws(calibration["y"],classes,[1],config["support_draws"],config["support_seed"])
    path = directory / "predictions.csv.gz"
    cases, random_rows, draw_records = [], [], []
    with gzip.open(path,"wt",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=PREDICTION_FIELDS,lineterminator="\n")
        writer.writeheader()
        for draw in draws:
            draw_records.append({"repeat":draw["repeat"],
                "support_ids":calibration["record_ids"][draw["indices"]].tolist()})
            cache = {identifier: prediction_only(context,candidate,zc,sc,calibration["y"],zq,sq,
                                                 draw["indices"])[0]
                     for identifier,candidate in unique.items()}
            for method,candidate in methods.items():
                matched_method = method.startswith("matched_zero_")
                pred = controls[candidate["id"]] if matched_method else cache[candidate["id"]]
                matched = pred if matched_method else controls[candidate["id"]]
                metrics = fast_metrics(y,pred,len(classes),detailed=True)
                count = 0 if matched_method or candidate["family"]=="identity" else len(draw["indices"])
                cases.append({"source_id":source_candidate["id"],"method":method,
                    "candidate_id":candidate["id"],"repeat":draw["repeat"],"support_labels_used":count,
                    **metrics,**{m+"_gain_vs_own_source":metrics[m]-base_metrics[m] for m in METRICS},
                    "accuracy_gain_vs_reference":metrics["accuracy"]-reference_accuracy,
                    "accuracy_gain_vs_matched":metrics["accuracy"]-float(np.mean(matched==y)),
                    "corrected_errors":int(np.count_nonzero((base_pred!=y)&(pred==y))),
                    "new_errors":int(np.count_nonzero((base_pred==y)&(pred!=y)))})
                for identifier,actual,guessed in zip(ids,y,pred):
                    row={"source_id":source_candidate["id"],"method":method,
                        "candidate_id":candidate["id"],"repeat":draw["repeat"],"record_id":identifier,
                        "y_true":str(classes[actual]),"y_predicted":str(classes[guessed]),
                        "is_correct":int(actual==guessed),"support_labels_used":count}
                    writer.writerow(row)
                    if draw["repeat"]==0 and identifier in random_ids and method in {"source_original","development_selected"}:
                        random_rows.append(row)
    if joblib.hash(model)!=before or joblib.hash(context_raw)!=context_before:
        raise AssertionError("Final source or latent context changed during support adaptation")
    metadata={"source_candidate":source_candidate,"class_order":classes.tolist(),
        "source_ids":training["record_ids"].tolist(),"query_ids":ids,"draws":draw_records,
        "source_model_sha256":digest(artifact),"source_model_fingerprint_before":before,
        "source_model_fingerprint_after":joblib.hash(model),"context_fingerprint_before":context_before,
        "context_fingerprint_after":joblib.hash(context_raw),"source_model_metadata":model.metadata,
        "source_baseline_metrics":base_metrics,"source_improvement_vs_reference":base_metrics["accuracy"]-reference_accuracy,
        "predictions_sha256":digest(path),"new_source_fit_calls":1,
        "final_query_labels_used_for_fit":False,"final_query_statistics_fitted":False}
    write_json(directory/"metadata.json",metadata)
    write_json(directory/"metrics.json",cases)
    return cases,random_rows,metadata,base_pred


def full_range_sensitivity(arrays, plan, truth, config, output, reference_accuracy):
    """A separate source-only exported-range diagnostic, never a candidate winner."""
    lookup={str(v):i for i,v in enumerate(arrays["record_ids"])}
    source_ids=[r["record_id"] for r in plan["records"] if r["role"]=="training"]
    query_ids=sorted(truth)
    s=np.asarray([lookup[v] for v in source_ids]);q=np.asarray([lookup[v] for v in query_ids])
    candidates=[c for c in source_candidate_grid() if c["preprocessing"] in {"raw","snv"}
                and ((c["model"]=="svc" and c.get("C")==10.) or c["model"]=="plsda")]
    directory=Path(output)/"full_range_sensitivity";directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for candidate in candidates:
        model=fit_source_candidate(candidate,arrays["X_alpha"][s],arrays["y"][s],arrays["axis"],config["model_seed"])
        fingerprint=joblib.hash(model);scores=model.predict_scores(arrays["X_alpha"][q])
        classes=np.asarray(model.classes_);pred=np.argmax(scores,axis=1)
        y=encoded([truth[v] for v in query_ids],classes)
        metrics=fast_metrics(y,pred,len(classes),detailed=True)
        if joblib.hash(model)!=fingerprint:raise AssertionError("Sensitivity source model changed")
        path=directory/(candidate["id"]+".predictions.csv")
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["record_id","y_true","y_predicted","is_correct"],lineterminator="\n")
            writer.writeheader();writer.writerows({"record_id":v,"y_true":str(classes[a]),
                "y_predicted":str(classes[p]),"is_correct":int(a==p)} for v,a,p in zip(query_ids,y,pred))
        rows.append({"candidate":candidate,"scope":"full_export_source_only_sensitivity_not_primary_or_calibration",
            "feature_count":len(arrays["axis"]),"frequency_range":[float(arrays["axis"][0]),float(arrays["axis"][-1])],
            "source_count":len(s),"query_count":len(q),"support_labels_used":0,**metrics,
            "accuracy_gain_vs_reference":metrics["accuracy"]-reference_accuracy,
            "source_fingerprint_before":fingerprint,"source_fingerprint_after":joblib.hash(model),
            "predictions_sha256":digest(path),"negative_absorption_not_removed":True,
            "physical_effective_range_verified":False})
    write_json(directory/"results.json",rows)
    return rows


def run_latent_search(input_path, plan_path, protocol_path, source_run, output, *, resume=False):
    protocol=json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    candidates=latent_candidate_grid();configuration=protocol["calibration_search"]
    expected=protocol.get("candidates",{})
    if expected.get("latent_count",len(candidates))!=len(candidates) or expected.get("latent_grid_sha256",object_digest(candidates))!=object_digest(candidates):
        raise ValueError("Latent candidate grid differs from the frozen protocol")
    if configuration.get("shots",[1])!=[1] or configuration.get("shots_per_class",1)!=1:
        raise ValueError("The seven-class fixed pool supports only uniform one-shot")
    if protocol["final"].get("shots_per_class",1)!=1:
        raise ValueError("Final seven-class comparison requires uniform one-shot")
    source_lock,_,partition_records=validate_source_run(source_run,input_path,plan_path,protocol_path)
    if (configuration["episodes"]!=protocol["source_search"]["episodes"]
            or configuration["selection_episodes"]!=protocol["source_search"]["selection_episodes"]
            or len(partition_records)!=configuration["episodes"]):
        raise ValueError("Source and latent stages need identical development tasks and selection boundaries")
    output=Path(output);manifest=make_manifest(input_path,plan_path,protocol_path,source_run,candidates)
    manifest_path=output/"latent_run_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():raise FileExistsError("Use a fresh latent-search directory or verified resume")
        if json.loads(manifest_path.read_text(encoding="utf-8"))!=manifest:raise ValueError("Resume rejected: latent inputs, source lock, code or runtime changed")
        if (output/"status.json").is_file() and json.loads((output/"status.json").read_text())["status"]=="completed":
            print(json.dumps({"stage":"latent_search_already_completed"}),flush=True)
            return json.loads((output/"summary.json").read_text())
    output.mkdir(parents=True,exist_ok=True)
    if not manifest_path.exists():write_json(manifest_path,manifest)
    write_json(output/"latent_candidate_grid.json",candidates)
    plan=json.loads(Path(plan_path).read_text(encoding="utf-8"))
    with np.load(input_path,allow_pickle=False) as f:arrays={key:f[key] for key in f.files}
    if str(np.asarray(arrays.get("dataset_version","")))!=protocol["dataset_version"]:
        raise ValueError("Dataset version differs from the recorded protocol")
    roles,truth,preparation=prepare_holdout(arrays,plan,
        frequency_min=protocol["frequency"]["lower_inclusive"],frequency_max=protocol["frequency"]["upper_inclusive"])
    if preparation["feature_count"]!=protocol["frequency"]["expected_points"]:raise ValueError("Fixed spectral interval differs")
    if protocol.get("fixed_role_counts") and {k:len(v["record_ids"]) for k,v in roles.items()}!=protocol["fixed_role_counts"]:
        raise ValueError("Outer role counts differ from frozen protocol")
    training=roles["training"]
    regenerated=development_episodes(training,configuration["episodes"],protocol["source_search"]["split_seed"])
    expected_partitions=[{key:training["record_ids"][v].tolist() for key,v in p.items()} for p in regenerated]
    if expected_partitions!=partition_records:raise ValueError("Source-cache partitions disagree with original training cohort")
    partitions=[_partition_indices(training,p) for p in partition_records]
    final_config={**protocol["final"],"model_seed":configuration["model_seed"]}
    rng=np.random.default_rng(final_config["random_query_seed"])
    examples=rng.choice(np.asarray(sorted(truth)),min(len(truth),final_config["random_query_examples"]),replace=False).tolist()
    write_json(output/"random_query_ids.json",examples)
    write_json(output/"preparation.json",preparation)
    choices,aggregates={},{}
    total_calls,reused=0,0
    for source_candidate in source_lock["candidates_for_adaptation"]:
        identifier=source_candidate["id"];directory=output/"development"/identifier
        directory.mkdir(parents=True,exist_ok=True);paths=[]
        for episode,partition in enumerate(partitions):
            path=directory/f"episode_{episode:02d}.csv.gz";meta_path=directory/f"episode_{episode:02d}.json"
            if path.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text(encoding="utf-8"))
                if digest(path)!=meta["metrics_file_sha256"] or meta["partition"]!=partition_records[episode]:
                    raise ValueError("Latent checkpoint changed or uses a different partition")
                if digest(source_model_path(source_run,identifier,episode))!=meta["source_model_sha256"]:
                    raise ValueError("Latent checkpoint source model changed")
                reused+=1
            else:
                meta=evaluate_latent_episode(training,partition,source_candidate,source_run,candidates,
                                             configuration,episode,path)
                write_json(meta_path,meta)
            paths.append(path);total_calls+=meta["candidate_fit_calls"]
            print(json.dumps({"stage":"latent_development","source_id":identifier,
                "completed_episode":episode+1,"episodes":len(partitions)}),flush=True)
        stats=aggregate_latent(paths);aggregates[identifier]=stats
        settings={key:configuration[key] for key in ["minimum_accuracy_gain","max_macro_f1_loss",
                                                    "max_balanced_accuracy_loss","tie_tolerance"]}
        choices[identifier]=choose_latent_with_gate(stats,candidates,**settings)
    write_json(output/"latent_development_aggregate.json",aggregates)
    lock={"source_selection_lock_sha256":digest(Path(source_run)/"source_selection_lock.json"),
        "latent_run_manifest_sha256":digest(manifest_path),"latent_grid_sha256":object_digest(candidates),
        "development_candidate_fit_calls":total_calls,"choices":choices,
        "selection_scope":"original training role only; no outer prediction labels", "created_before_final_query_scoring":True}
    lock_path=output/"latent_selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8"))!=lock:raise ValueError("Locked latent selections changed")
    write_json(lock_path,lock);lock_sha=digest(lock_path)
    write_json(output/"latent_selection_lock_sha256.json",{"sha256":lock_sha})
    print(json.dumps({"stage":"all_latent_choices_locked","sha256":lock_sha}),flush=True)
    all_cases,all_random,all_metadata=[],[],[];reference_prediction=None
    for source_candidate in source_lock["candidates_for_adaptation"]:
        if digest(lock_path)!=lock_sha:raise ValueError("Latent selection lock changed before final scoring")
        cases,random_rows,metadata,pred=evaluate_final_source(roles["training"],roles["calibration"],
            roles["prediction"],truth,source_candidate,choices[source_candidate["id"]],candidates,
            final_config,output,reference_prediction,examples)
        if reference_prediction is None:
            if source_candidate["id"]!=source_lock["reference"]["id"]:raise ValueError("Reference source must be evaluated first")
            reference_prediction=pred
        all_cases.extend(cases);all_random.extend(random_rows);all_metadata.append(metadata)
        print(json.dumps({"stage":"latent_final_source_completed","source_id":source_candidate["id"],
                           "evaluations":len(cases)}),flush=True)
    reference_accuracy=all_metadata[0]["source_baseline_metrics"]["accuracy"]
    sensitivity=[]
    sensitivity_settings=protocol.get("full_export_sensitivity",protocol.get("full_range_sensitivity",{}))
    if sensitivity_settings.get("enabled",True):
        if digest(lock_path)!=lock_sha:raise ValueError("Latent selection lock changed before sensitivity scoring")
        sensitivity=full_range_sensitivity(arrays,plan,truth,final_config,output,reference_accuracy)
    summary=_summaries(all_cases)
    write_json(output/"summary.json",summary);write_json(output/"all_final_metrics.json",all_cases)
    write_json(output/"final_metadata.json",all_metadata)
    with (output/"random_query_predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=PREDICTION_FIELDS,lineterminator="\n");writer.writeheader();writer.writerows(all_random)
    write_json(output/"status.json",{"status":"completed","scope":protocol.get("scope","retrospective within-batch"),
        "selected_source_count":len(all_metadata),"new_development_source_fits":0,
        "development_candidate_fit_calls":total_calls,"new_final_source_fits":len(all_metadata),
        "sensitivity_source_fits":len(sensitivity),"final_evaluations":len(all_cases),
        "prediction_rows":sum(row["total"] for row in all_cases),"checkpoints_reused_this_invocation":reused,
        "source_selection_lock_sha256":lock["source_selection_lock_sha256"],"latent_selection_lock_sha256":lock_sha,
        "final_query_used_for_selection":False,"source_gains_are_not_calibration_gains":True})
    return summary
