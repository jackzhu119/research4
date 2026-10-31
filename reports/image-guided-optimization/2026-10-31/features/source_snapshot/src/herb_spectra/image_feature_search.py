"""Two-stage, Macro-F1-first source feature search for the user's figure."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np

from .calibration_exploration import development_episodes, encoded, fast_metrics
from .image_feature_models import candidate_grid, fit_candidate
from .within_batch_holdout import prepare_holdout


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def reference_candidate():
    return next(c for c in candidate_grid() if c["preprocessing"] == "snv" and c["family"] == "full_svm" and c["C"] == 10)


def summarize_development(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate_id"], row["stage"])].append(row)
    result = {}
    for (identifier, stage), values in groups.items():
        valid = [r for r in values if r["status"] == "ok"]
        result[(identifier, stage)] = {"evaluations": len(values), "failures": len(values)-len(valid),
            "selected_features_mean": float(np.mean([r["selected_features"] for r in valid])) if valid else None,
            **{k: float(np.mean([r[k] for r in valid])) if valid else None for k in METRICS}}
    return result


def choose_candidate(rows, candidates, policy):
    means = summarize_development(rows); reference = reference_candidate(); ref_id = reference["id"]
    minimum, drop, tolerance = policy["minimum_macro_f1_gain"], policy["maximum_accuracy_or_balanced_loss"], policy["macro_f1_tie_tolerance"]
    def valid(c, stage):
        a = means.get((c["id"], stage))
        return a is not None and not a["failures"] and all(a[k] is not None for k in METRICS)
    def passes(c, stage):
        if not valid(c, stage) or not valid(reference, stage):
            return False
        a,b = means[(c["id"], stage)],means[(ref_id, stage)]
        return (a["macro_f1"]-b["macro_f1"] > minimum
            and a["accuracy"]-b["accuracy"] >= -drop
            and a["balanced_accuracy"]-b["balanced_accuracy"] >= -drop)
    def rank(options):
        options = [c for c in options if valid(c, "selection")]
        if not options:
            return reference
        best = max(means[(c["id"], "selection")]["macro_f1"] for c in options)
        tied = [c for c in options if means[(c["id"], "selection")]["macro_f1"] >= best-tolerance]
        return min(tied, key=lambda c: (-means[(c["id"], "selection")]["accuracy"],
            means[(c["id"], "selection")]["selected_features_mean"],
            c.get("components", 0), c.get("C", 0), c["id"]))
    eligible = [c for c in candidates if passes(c, "selection")]
    proposed = rank(eligible) if eligible else reference
    selected = proposed if eligible and passes(proposed, "gate") else reference
    families = {family: rank([c for c in candidates if c["family"] == family])
                for family in sorted(set(c["family"] for c in candidates))}
    return {"reference": reference, "proposed": proposed, "selected": selected,
        "fallback_to_reference": selected == reference, "eligible_selection_candidates": len(eligible),
        "proposed_passed_gate": bool(eligible) and passes(proposed, "gate"),
        "family_diagnostics": families, "selected_using_final_query": False,
        "means": [{"candidate_id": key[0], "stage": key[1], **value} for key,value in sorted(means.items())]}


def _subset(training, rows):
    return {"X": training["X_alpha"][rows], "y": training["y"][rows], "record_ids": training["record_ids"][rows]}


def _provenance(input_path, plan_path, protocol_path):
    root = Path(__file__).resolve().parents[2]
    names = ["src/herb_spectra/image_feature_models.py", "src/herb_spectra/image_feature_search.py",
             "scripts/run_image_feature_search.py", "src/herb_spectra/selection.py",
             "src/herb_spectra/calibration_exploration.py", "src/herb_spectra/within_batch_holdout.py"]
    return {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
        "protocol_sha256": digest(protocol_path), "grid_sha256": object_digest(candidate_grid()),
        "source_sha256": {n: digest(root/n) for n in names}, "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}}


def verify_release(output, protocol_path):
    output=Path(output); release_path=output.parent/"all_development_locks.json"
    if not release_path.exists():
        raise ValueError("Global image-guided final release is required")
    release=json.loads(release_path.read_text())
    matches=[r for r in release.get("rounds", [])
             if Path(r["selection_lock"]).resolve() == (output/"selection_lock.json").resolve()]
    if len(matches)!=1:
        raise ValueError("Global release does not identify exactly one feature selection lock")
    record=matches[0]
    expected={"selection_lock_sha256":digest(output/"selection_lock.json"),
              "protocol_sha256":digest(protocol_path),
              "run_manifest_sha256":digest(output/"run_manifest.json")}
    if any(record.get(key)!=value for key,value in expected.items()):
        raise ValueError("Global feature lock/protocol/manifest hashes changed")
    return release_path


def _cached_fit(directory, candidate, source, axis, seed, *, cache, resume):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(candidate["id"]+".joblib")
    meta_path = directory/(candidate["id"]+".json")
    expected = {"candidate": candidate, "seed": seed, "source_ids": source["record_ids"].tolist()}
    if resume and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["expected"] != expected or digest(path) != metadata["model_sha256"]:
            raise ValueError("Feature source checkpoint identity/hash changed")
        model = joblib.load(path)
        if joblib.hash(model) != metadata["model_fingerprint"]:
            raise ValueError("Feature source checkpoint fingerprint changed")
        return model, metadata
    model = fit_candidate(candidate, source["X"], source["y"], axis, seed=seed, cache=cache)
    in_memory_fingerprint=joblib.hash(model)
    source_scores=model.predict_scores(source["X"])
    joblib.dump(model, path, compress=3)
    # PCA.components_ may be a noncontiguous view before serialization. joblib
    # hashes memory layout, while restoring the same numbers as C-contiguous.
    # Record the actual persisted/reloaded state, never a pre-save view layout.
    canonical=joblib.load(path)
    restored_scores=canonical.predict_scores(source["X"])
    maximum_difference=float(np.max(np.abs(source_scores-restored_scores)))
    if (not np.allclose(source_scores,restored_scores,rtol=1e-10,atol=1e-10)
            or not np.array_equal(source_scores.argmax(1),restored_scores.argmax(1))):
        raise AssertionError("Saving/restoring feature model changed source inference")
    metadata = {"expected": expected, "model_sha256": digest(path),
        "model_fingerprint": joblib.hash(canonical), "fit": canonical.metadata,
        "pre_serialization_fingerprint":in_memory_fingerprint,
        "checkpoint_fingerprint_policy":"saved-and-reloaded canonical object",
        "source_dump_load_max_score_difference":maximum_difference,
        "source_dump_load_labels_equal":True}
    write_json(meta_path, metadata)
    return canonical, metadata


def _score_context(source, query, truth, axis, candidates, classes, directory, *, episode, seed, stage, resume):
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory/"context.json"
    if resume and completion.exists():
        saved = json.loads(completion.read_text())
        if (saved["source_ids"] != source["record_ids"].tolist() or saved["query_ids"] != query["record_ids"].tolist()
                or saved["candidate_ids"] != [c["id"] for c in candidates]
                or digest(directory/"predictions.csv.gz") != saved["predictions_sha256"]):
            raise ValueError("Completed feature context changed")
        for candidate in candidates:
            _cached_fit(directory, candidate, source, axis, seed, cache={}, resume=True)
        return saved["rows"]
    cache = {}; rows = []; actual = encoded(truth, classes)
    path = directory/"predictions.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        fields = ["episode", "seed", "candidate_id", "record_id", "y_true", "y_predicted", "is_correct"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for candidate in candidates:
            model, metadata = _cached_fit(directory, candidate, source, axis, seed, cache=cache, resume=resume)
            before = joblib.hash(model)
            predicted = encoded(model.predict(query["X"]), classes)
            if joblib.hash(model) != before:
                raise AssertionError("Prediction modified source feature model")
            row = {"episode": episode, "seed": seed, "candidate_id": candidate["id"],
                "stage": stage, "status": "ok", "selected_features": metadata["fit"]["selected_features"],
                **fast_metrics(actual, predicted, len(classes), detailed=True),
                "model_sha256": metadata["model_sha256"], "model_fingerprint": before}
            rows.append(row)
            for identifier,a,p in zip(query["record_ids"],actual,predicted):
                writer.writerow({"episode":episode,"seed":seed,"candidate_id":candidate["id"],
                    "record_id":str(identifier),"y_true":str(classes[a]),"y_predicted":str(classes[p]),"is_correct":int(a==p)})
    write_json(completion, {"source_ids": source["record_ids"].tolist(), "query_ids": query["record_ids"].tolist(),
        "candidate_ids": [c["id"] for c in candidates], "rows": rows, "predictions_sha256": digest(path)})
    return rows


def run_search(input_path, plan_path, protocol_path, output, *, stage="development", resume=False):
    if stage not in ("development", "final"):
        raise ValueError("Expected development or final stage")
    protocol = json.loads(Path(protocol_path).read_text()); candidates = candidate_grid()
    if protocol["candidate_grid_sha256"] != object_digest(candidates) or protocol["candidate_configurations"] != len(candidates):
        raise ValueError("Frozen feature grid changed")
    output = Path(output); provenance = _provenance(input_path,plan_path,protocol_path)
    if output.exists() and any(output.iterdir()):
        if not resume or json.loads((output/"run_manifest.json").read_text()) != provenance:
            raise ValueError("Use new output or verified unchanged resume")
        if (output/"status.json").exists():
            status = json.loads((output/"status.json").read_text())
            if status["status"] == "completed":
                return json.loads((output/"summary.json").read_text())
            if stage == "development" and status["status"] == "development_completed":
                return json.loads((output/"selection_lock.json").read_text())
    if stage == "final":
        if not (output/"selection_lock.json").exists():
            raise ValueError("Final requires completed development")
        release=verify_release(output,protocol_path)
        (output/"final_release_snapshot.json").write_bytes(release.read_bytes())
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"run_manifest.json",provenance)
    write_json(output/"candidate_grid.json",candidates)
    (output/"protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
    with np.load(input_path,allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in archive.files}
    plan = json.loads(Path(plan_path).read_text())
    roles,truth,preparation = prepare_holdout(arrays,plan,frequency_min=.1,frequency_max=1.5)
    if preparation["feature_count"] != 612:
        raise ValueError("Unexpected fixed feature band")
    write_json(output/"preparation.json",preparation)
    training = roles["training"]; classes = np.unique(training["y"]); axis=training["axis"]
    started=time.monotonic()
    event={"stage":stage,"event":"start","utc":datetime.now(timezone.utc).isoformat(),"resume":resume}
    with (output/"stage_events.jsonl").open("a") as stream:stream.write(json.dumps(event)+"\n")
    if stage == "development":
        partitions=development_episodes(training,6,protocol["development"]["split_seed"])
        write_json(output/"development_partitions.json",[{k:training["record_ids"][v].tolist() for k,v in part.items()} for part in partitions])
        rows=[]
        for episode,part in enumerate(partitions):
            source=_subset(training,part["source"]); query=_subset(training,part["query"])
            for seed in protocol["seeds"]:
                rows.extend(_score_context(source,query,query["y"],axis,candidates,classes,
                    output/"development"/f"episode_{episode:02d}"/f"seed_{seed}",episode=episode,seed=seed,
                    stage="selection" if episode<4 else "gate",resume=resume))
            write_json(output/"status.json",{"status":"development","completed_episodes":episode+1,"development_evaluations":len(rows)})
            print(json.dumps({"stage":"feature_development","episode":episode+1,"evaluations":len(rows)}),flush=True)
        choice=choose_candidate(rows,candidates,protocol["selection"])
        lock={"choice":choice,"source_only_development":True,"development_evaluations":len(rows),"provenance_sha256":object_digest(provenance)}
        write_json(output/"all_development_metrics.json",rows)
        write_json(output/"selection_lock.json",lock)
        write_json(output/"status.json",{"status":"development_completed","seconds":time.monotonic()-started,
            "development_evaluations":len(rows),"development_prediction_rows":sum(r["total"] for r in rows),
            "selection_lock_sha256":digest(output/"selection_lock.json"),"final_query_scored":False})
        return lock
    lock_path=output/"selection_lock.json";lock=json.loads(lock_path.read_text());lock_sha=digest(lock_path)
    if choose_candidate(json.loads((output/"all_development_metrics.json").read_text()),candidates,protocol["selection"]) != lock["choice"]:
        raise ValueError("Development selection no longer matches saved source metrics")
    choice=lock["choice"]
    methods={"reference":choice["reference"],"selected":choice["selected"],
             **{"family_"+f:c for f,c in choice["family_diagnostics"].items()}}
    final_candidates=list({c["id"]:c for c in methods.values()}.values())
    source={"X":training["X_alpha"],"y":training["y"],"record_ids":training["record_ids"]}
    prediction=roles["prediction"]
    if "y" in prediction:
        raise AssertionError("Final feature payload contains truth")
    query={"X":prediction["X_alpha"],"record_ids":prediction["record_ids"]}
    rows=[]
    for seed in protocol["seeds"]:
        actual=[truth[str(i)] for i in query["record_ids"]]
        current=_score_context(source,query,actual,axis,final_candidates,classes,output/"final"/f"seed_{seed}",
            episode=-1,seed=seed,stage="final",resume=resume)
        by_id={r["candidate_id"]:r for r in current}
        for method,candidate in methods.items():
            rows.append({"method":method,**by_id[candidate["id"]]})
    if digest(lock_path)!=lock_sha:
        raise ValueError("Feature lock changed during final")
    summary=[]
    for method,candidate in methods.items():
        values=[r for r in rows if r["method"]==method]
        summary.append({"method":method,"candidate":candidate,"evaluations":len(values),"seeds":protocol["seeds"],
            **{k+"_mean":float(np.mean([r[k] for r in values])) for k in METRICS},
            "selected_features_mean":float(np.mean([r["selected_features"] for r in values]))})
    write_json(output/"final_metrics.json",rows);write_json(output/"summary.json",summary)
    write_json(output/"status.json",{"status":"completed","final_seconds":time.monotonic()-started,
        "final_method_evaluations":len(rows),"final_unique_candidate_evaluations":len(current)*len(protocol["seeds"]),
        "final_prediction_rows":len(current)*len(protocol["seeds"])*len(query["record_ids"]),
        "selection_lock_sha256":lock_sha,"source_feature_selection_not_calibration":True,
        "same_batch_retrospective":True,"new_independent_batches":0})
    return summary
