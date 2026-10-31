"""Source-only limited score adaptation for retrospective seven-class studies.

This is an engineering control around the existing anchored ridge score map,
not a new algorithm or a real cross-batch calibration claim. No adaptation API
accepts query labels or learns a statistic from the query batch.
"""
from collections import defaultdict
from dataclasses import dataclass
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

from .calibration import AnchorScoreCalibrator
from .calibration_exploration import (development_episodes, encoded, fast_metrics,
                                      feature_payload, fit_fixed_source, nested_draws)
from .within_batch_holdout import prepare_holdout


METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def candidate_grid():
    return [{"id": f"cap{cap:g}_q{q:g}_m{mistakes}", "cap_fraction": cap,
             "source_gate_quantile": q, "minimum_support_errors": mistakes,
             "blend": .25}
            for cap in (.05, .1, .25) for q in (.5, .75) for mistakes in (1, 2)]


def _scores(values, classes):
    value = np.asarray(values, dtype=float)
    if value.ndim != 2 or not len(value) or value.shape[1] != len(classes) or not np.isfinite(value).all():
        raise ValueError("Expected nonempty finite scores aligned to source classes")
    return value


def margins(scores):
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


@dataclass
class LimitedScoreAdapter:
    classes: np.ndarray
    candidate: dict
    cap: float
    gate_margin: float
    ridge: object | None
    support_error_count: int
    rejection_reason: str

    def predict_scores(self, query_scores):
        scores = _scores(query_scores, self.classes)
        if self.ridge is None:
            return scores.copy()
        correction = self.ridge.transform(scores) - scores
        correction = np.clip(correction, -self.cap, self.cap) * self.candidate["blend"]
        correction[margins(scores) > self.gate_margin] = 0.
        return scores + correction


def fit_limited(source_scores, classes, candidate, support_scores=None, support_y=None):
    classes = np.asarray(classes)
    if classes.ndim != 1 or len(classes) < 2 or len(np.unique(classes)) != len(classes):
        raise ValueError("Unique source class vocabulary required")
    source = _scores(source_scores, classes)
    if candidate not in candidate_grid():
        raise ValueError("Candidate must belong to the frozen finite grid")
    source_margins = margins(source)
    cap = candidate["cap_fraction"] * max(float(np.median(source_margins)), np.finfo(float).eps)
    gate = float(np.quantile(source_margins, candidate["source_gate_quantile"]))
    if support_scores is None:
        if support_y is not None:
            raise ValueError("Support scores are required with labels")
        return LimitedScoreAdapter(classes.copy(), dict(candidate), cap, gate, None, 0, "zero_support")
    support = _scores(support_scores, classes)
    if support_y is None:
        return LimitedScoreAdapter(classes.copy(), dict(candidate), cap, gate, None, 0, "no_support_labels")
    labels = np.asarray(support_y)
    if labels.ndim != 1 or len(labels) != len(support) or set(labels.tolist()) != set(classes.tolist()):
        raise ValueError("Labelled support must cover every source class")
    errors = int(np.count_nonzero(classes[np.argmax(support, axis=1)] != labels))
    if errors < candidate["minimum_support_errors"]:
        return LimitedScoreAdapter(classes.copy(), dict(candidate), cap, gate, None, errors,
                                   "insufficient_support_error_evidence")
    ridge = AnchorScoreCalibrator(classes, regularization=1.).fit(support, labels)
    return LimitedScoreAdapter(classes.copy(), dict(candidate), cap, gate, ridge, errors, "")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv_file(path, rows):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _record(truth, pred, baseline, classes):
    stats = fast_metrics(truth, pred, len(classes), detailed=True)
    baseline_stats = fast_metrics(truth, baseline, len(classes))
    stats.update({name+"_gain": stats[name]-baseline_stats[name] for name in METRICS})
    stats["corrected"] = int(np.count_nonzero((baseline != truth) & (pred == truth)))
    stats["harmed"] = int(np.count_nonzero((baseline == truth) & (pred != truth)))
    return stats


def choose_development(rows, candidates):
    aggregate = {}
    for stage in ("selection", "gate"):
        aggregate[stage] = {}
        for candidate in candidates:
            subset = [r for r in rows if r["stage"] == stage and r["candidate_id"] == candidate["id"]]
            aggregate[stage][candidate["id"]] = {
                name: float(np.mean([r[name] for r in subset]))
                for name in (*METRICS, *(name+"_gain" for name in METRICS), "corrected", "harmed")}
            aggregate[stage][candidate["id"]]["evaluations"] = len(subset)
    def allowed(candidate, stage):
        stats = aggregate[stage][candidate["id"]]
        return stats["accuracy_gain"] > .001 and stats["macro_f1_gain"] > .001 and stats["balanced_accuracy_gain"] >= -.005
    eligible = [c for c in candidates if allowed(c, "selection")]
    order = lambda c: (-aggregate["selection"][c["id"]]["macro_f1_gain"], c["cap_fraction"],
                       -c["minimum_support_errors"], c["source_gate_quantile"], c["id"])
    diagnostic = min(candidates, key=order)
    proposed = min(eligible, key=order) if eligible else None
    selected = proposed if proposed is not None and allowed(proposed, "gate") else None
    return {"selected":selected,"proposed":proposed,"diagnostic":diagnostic,
            "fallback_to_identity":selected is None,"aggregate":aggregate,
            "gate_rule":"accuracy gain >.001; macro-F1 gain >.001; balanced-accuracy gain >=-.005 in selection and gate"}


def _load(repo, protocol):
    with np.load(repo/protocol["input"], allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in archive.files}
    plan = json.loads((repo/protocol["split_plan"]).read_text())
    return prepare_holdout(arrays, plan)


def _manifest(repo, protocol_path, protocol):
    sources = [Path(__file__), repo/"scripts/run_paper_c_calibration.py",
               repo/"src/herb_spectra/calibration.py", repo/"src/herb_spectra/calibration_exploration.py",
               repo/"src/herb_spectra/classification.py", repo/"src/herb_spectra/selection.py",
               repo/"src/herb_spectra/within_batch_holdout.py"]
    frozen = json.loads((repo/protocol["frozen_source_manifest"]).read_text())
    for model in frozen["models"]:
        assert digest(repo/model["path"]) == model["sha256"]
    return {"protocol_sha256":digest(protocol_path),"input_sha256":digest(repo/protocol["input"]),
            "split_sha256":digest(repo/protocol["split_plan"]),"source_code_sha256":{str(p.relative_to(repo)):digest(p) for p in sources},
            "frozen_sources":frozen,"python":platform.python_version(),
            "dependencies":{k:importlib.metadata.version(k) for k in ("numpy","scipy","scikit-learn","joblib")}}


def select(repo, protocol_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Select into a fresh directory")
    protocol = json.loads(protocol_path.read_text())
    assert protocol["candidates"] == candidate_grid()
    manifest = _manifest(repo, protocol_path, protocol)
    write_json(destination/"run_manifest.json", manifest)
    write_json(destination/"protocol_snapshot.json", protocol)
    events = [{"event":"selection_started","utc":datetime.now(timezone.utc).isoformat()}]
    roles, _, preparation = _load(repo, protocol)
    training = roles["training"]
    write_json(destination/"preparation.json", preparation)
    partitions = development_episodes(training, 30, 20261103)
    candidates = candidate_grid()
    choices, budget = {}, []
    for selector in protocol["selectors"]:
        directory = destination/"development"/selector
        directory.mkdir(parents=True)
        rows = []
        for episode, partition in enumerate(partitions):
            start = time.monotonic()
            source, pool, query = (partition[k] for k in ("source","support_pool","query"))
            assert not set(source)&set(pool) and not set(source)&set(query) and not set(pool)&set(query)
            pipe = fit_fixed_source(training["X_alpha"][source], training["y"][source], selector, 42)
            before = joblib.hash(pipe)
            classes = pipe.classes_
            ss = feature_payload(pipe, training["X_alpha"][source])[1]
            sp = feature_payload(pipe, training["X_alpha"][pool])[1]
            sq = feature_payload(pipe, training["X_alpha"][query])[1]
            np.savez_compressed(directory/f"episode_{episode:02d}_scores.npz", source_scores=ss,pool_scores=sp,query_scores=sq,
                                source_ids=training["record_ids"][source],pool_ids=training["record_ids"][pool],query_ids=training["record_ids"][query],
                                pool_y=training["y"][pool],query_y=training["y"][query],classes=classes)
            truth = encoded(training["y"][query], classes)
            baseline = np.argmax(sq, axis=1)
            draws = nested_draws(training["y"][pool], classes, [1],20,20261104+episode)
            predictions = []
            for draw in draws:
                chosen = draw["indices"]
                for candidate in candidates:
                    adapter = fit_limited(ss, classes, candidate, sp[chosen],training["y"][pool][chosen])
                    pred = np.argmax(adapter.predict_scores(sq),axis=1)
                    row = {"selector":selector,"episode":episode,"stage":"selection" if episode <24 else "gate",
                           "repeat":draw["repeat"],"candidate_id":candidate["id"],
                           **_record(truth,pred,baseline,classes),"support_rejected":bool(adapter.rejection_reason),
                           "support_error_count":adapter.support_error_count,"cap":adapter.cap,"gate_margin":adapter.gate_margin}
                    rows.append(row)
                    for identifier, actual, predicted in zip(training["record_ids"][query],truth,pred):
                        predictions.append({"episode":episode,"repeat":draw["repeat"],"candidate_id":candidate["id"],
                                            "record_id":str(identifier),"y_true":str(classes[actual]),"y_predicted":str(classes[predicted])})
            csv_file(directory/f"episode_{episode:02d}_predictions.csv.gz", predictions)
            assert joblib.hash(pipe) == before
            budget.append({"selector":selector,"episode":episode,"source_samples":len(source),"source_fit_count":1,
                           "seconds":time.monotonic()-start,"source_fingerprint_unchanged":before,
                           "source_ids":training["record_ids"][source].tolist(),"support_pool_ids":training["record_ids"][pool].tolist(),
                           "query_ids":training["record_ids"][query].tolist(),
                           "draws":[{"repeat":d["repeat"],"ids":training["record_ids"][pool][d["indices"]].tolist()} for d in draws]})
            print(json.dumps({"stage":"development","selector":selector,"episode":episode+1}),flush=True)
        csv_file(directory/"metrics.csv.gz",[{**r,"confusion_matrix":json.dumps(r["confusion_matrix"]),
                 "class_recall":json.dumps(r["class_recall"]),"class_counts":json.dumps(r["class_counts"])} for r in rows])
        choices[selector] = choose_development(rows,candidates)
    lock = {"protocol_id":protocol["protocol_id"],"run_manifest_sha256":digest(destination/"run_manifest.json"),
            "selected_at_utc":datetime.now(timezone.utc).isoformat(),"choices":choices,
            "final_query_scored_during_selection":False,"source_fits":len(budget),"candidate_evaluations":len(budget)*20*12}
    write_json(destination/"development_budget.json",budget)
    write_json(destination/"selection_lock.json",lock)
    events.append({"event":"selection_locked","utc":lock["selected_at_utc"],"sha256":digest(destination/"selection_lock.json")})
    write_json(destination/"selection_events.json",events)
    return lock


def evaluate(repo, protocol_path, destination, release_path):
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((destination/"run_manifest.json").read_text())
    assert manifest == _manifest(repo,protocol_path,protocol),"Frozen manifest changed"
    release = json.loads(release_path.read_text())
    assert release["selection_lock_sha256"] == digest(destination/"selection_lock.json")
    if (destination/"final_metrics.json").exists():
        raise FileExistsError("Final evaluation already recorded")
    lock = json.loads((destination/"selection_lock.json").read_text())
    roles, truth_map, _ = _load(repo,protocol)
    source,pool,query = (roles[k] for k in ("training","calibration","prediction"))
    assert "y" not in query
    rows,all_predictions,metadata = [],[],[]
    for selector in protocol["selectors"]:
        start = time.monotonic()
        entry = next(m for m in manifest["frozen_sources"]["models"] if m["selector"]==selector)
        assert digest(repo/entry["path"]) == entry["sha256"]
        pipe = joblib.load(repo/entry["path"])
        before = joblib.hash(pipe)
        classes=pipe.classes_
        ss=feature_payload(pipe,source["X_alpha"])[1]
        sp=feature_payload(pipe,pool["X_alpha"])[1]
        sq=feature_payload(pipe,query["X_alpha"])[1]
        baseline=np.argmax(sq,axis=1)
        selected=lock["choices"][selector]["selected"]
        diagnostic=lock["choices"][selector]["diagnostic"]
        chosen_config=selected if selected is not None else diagnostic
        draws=nested_draws(pool["y"],classes,[1],100,20261104)
        # Only scoring below receives query truth; all prediction functions are label-free.
        truth=encoded([truth_map[str(v)] for v in query["record_ids"]],classes)
        for draw in draws:
            chosen=draw["indices"]
            conservative=fit_limited(ss,classes,chosen_config,sp[chosen],pool["y"][chosen])
            zero=fit_limited(ss,classes,chosen_config)
            nolabel=fit_limited(ss,classes,chosen_config,sp[chosen],None)
            legacy=AnchorScoreCalibrator(classes,regularization=1.).fit(sp[chosen],pool["y"][chosen])
            diagnostic_adapter=fit_limited(ss,classes,diagnostic,sp[chosen],pool["y"][chosen])
            predictions={"no_calibration":baseline,"traditional_lambda1":np.argmax(legacy.transform(sq),axis=1),
                         "conservative_selected":baseline.copy() if selected is None else np.argmax(conservative.predict_scores(sq),axis=1),
                         "same_structure_zero_support":np.argmax(zero.predict_scores(sq),axis=1),
                         "same_structure_no_labels":np.argmax(nolabel.predict_scores(sq),axis=1),
                         "prespecified_development_diagnostic":np.argmax(diagnostic_adapter.predict_scores(sq),axis=1)}
            for method,pred in predictions.items():
                row={"selector":selector,"repeat":draw["repeat"],"method":method,
                     "candidate_id":"identity" if method=="conservative_selected" and selected is None else chosen_config["id"],
                     **_record(truth,pred,baseline,classes),"support_evidence_rejected":bool(conservative.rejection_reason),
                     "development_gate_fallback":selected is None,"support_error_count":conservative.support_error_count,
                     "unchanged_query_predictions":int(np.count_nonzero(pred==baseline))}
                rows.append(row)
                for identifier,actual,predicted in zip(query["record_ids"],truth,pred):
                    all_predictions.append({"selector":selector,"repeat":draw["repeat"],"method":method,"record_id":str(identifier),
                                            "y_true":str(classes[actual]),"y_predicted":str(classes[predicted]),"is_correct":int(actual==predicted)})
        assert before==joblib.hash(pipe)
        metadata.append({"selector":selector,"source_artifact":entry,"fingerprint_before_after":before,
                         "new_source_fits":0,"source_ids":source["record_ids"].tolist(),"query_ids":query["record_ids"].tolist(),
                         "draws":[{"repeat":d["repeat"],"support_ids":pool["record_ids"][d["indices"]].tolist()} for d in draws],
                         "elapsed_seconds":time.monotonic()-start,"query_statistics_fitted":False,"query_labels_used_to_fit":False})
    summary=[]
    for selector in protocol["selectors"]:
        for method in sorted({r["method"] for r in rows}):
            subset=[r for r in rows if r["selector"]==selector and r["method"]==method]
            row={"selector":selector,"method":method,"support_draws":len(subset)}
            for name in (*METRICS,*(k+"_gain" for k in METRICS),"corrected","harmed","unchanged_query_predictions"):
                values=np.asarray([r[name] for r in subset])
                row.update({name+"_mean":float(values.mean()),name+"_sd":float(values.std()),
                            name+"_min":float(values.min()),name+"_max":float(values.max())})
            row.update(positive_draws=sum(r["accuracy_gain"]>0 for r in subset),negative_draws=sum(r["accuracy_gain"]<0 for r in subset),
                       support_evidence_rejection_rate=float(np.mean([r["support_evidence_rejected"] for r in subset])),
                       development_gate_fallback=subset[0]["development_gate_fallback"])
            summary.append(row)
    write_json(destination/"final_metrics.json",rows)
    write_json(destination/"final_metadata.json",metadata)
    write_json(destination/"summary.json",summary)
    csv_file(destination/"final_predictions.csv.gz",all_predictions)
    csv_file(destination/"summary.csv",summary)
    write_json(destination/"completed.json",{"utc":datetime.now(timezone.utc).isoformat(),"selection_lock_sha256":digest(destination/"selection_lock.json"),
               "release_sha256":digest(release_path),"evaluations":len(rows),"predictions":len(all_predictions),"new_final_source_fits":0,
               "scope":"retrospective within-batch, 513/171/171; not R2 OOF or real cross-batch", "support_draws_are_new_independent_batches":False})
    return summary
