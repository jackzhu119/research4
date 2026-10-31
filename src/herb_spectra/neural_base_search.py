"""Source-only, predeclared BP/MLP representation search for seven-class spectra.

This is an independently implemented PyTorch dense baseline, not MATLAB newff
reproduction. Neither class names nor physical identifiers are numeric features.
Final prediction truth is separated from every fitting and context API.
"""
from collections import defaultdict
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .calibration_exploration import development_episodes, encoded, fast_metrics
from .paper_fewshot_classification import state_fingerprint
from .within_batch_holdout import prepare_holdout

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def candidate_grid():
    result = []
    for components in (None, 20, 40):
        for hidden in ((64,), (64, 32)):
            for epochs in (80, 200):
                for balance in ("none", "balanced"):
                    result.append({"id": f"b{len(result):03d}_mlp", "pca_components": components,
                                   "hidden_sizes": list(hidden), "epochs": epochs, "class_weight": balance,
                                   "learning_rate": .001, "weight_decay": .001, "batch_size": 64})
    return result


def reference_candidate():
    return next(c for c in candidate_grid() if c["pca_components"] == 20 and c["hidden_sizes"] == [64, 32]
                and c["epochs"] == 200 and c["class_weight"] == "balanced")


def _snv(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Finite specimens-by-frequency spectra are required")
    std = values.std(axis=1, ddof=1, keepdims=True)
    if np.any(std <= np.finfo(float).eps):
        raise ValueError("SNV cannot normalize constant spectra")
    return (values-values.mean(axis=1, keepdims=True))/std


def fit_preprocessor(source_X, candidate):
    x = _snv(source_X)
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    components = candidate.get("pca_components")
    if components is not None and (components < 1 or components >= min(x.shape)):
        raise ValueError("PCA needs fewer components than source samples and features")
    reducer = None if components is None else PCA(n_components=components, svd_solver="full").fit(standardized)
    return {"scaler": scaler, "pca": reducer, "pca_components": components,
            "input_features": int(x.shape[1]), "source_samples": int(len(x)),
            "normalization": "per-spectrum SNV ddof1, then source-only StandardScaler, optional unwhitened PCA"}


def transform_features(preprocessor, X):
    x = _snv(X)
    if x.shape[1] != preprocessor["input_features"]:
        raise ValueError("Prediction frequency count differs from source preprocessing")
    x = preprocessor["scaler"].transform(x)
    if preprocessor["pca"] is not None:
        x = preprocessor["pca"].transform(x)
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite transformed features")
    return np.ascontiguousarray(x, dtype=np.float32)


class DenseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(64, 32), n_classes=7):
        super().__init__()
        if input_dim < 1 or n_classes < 2 or not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError("Positive dense dimensions and at least two classes are required")
        modules = []; previous = input_dim
        for width in hidden_sizes:
            modules.extend([nn.Linear(previous, width), nn.ReLU()]); previous = width
        self.encoder = nn.Sequential(*modules)
        self.head = nn.Linear(previous, n_classes)

    def encode(self, values):
        return self.encoder(values)

    def forward(self, values):
        return self.head(self.encode(values))


def _features_tensor(values):
    if torch.is_tensor(values):
        result = values.detach().to(dtype=torch.float32, device="cpu")
    else:
        result = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    if result.ndim != 2 or not len(result) or not torch.isfinite(result).all():
        raise ValueError("Finite nonempty specimens-by-features tensors are required")
    return result


def fit_model(Z, y, candidate, seed=42):
    x = _features_tensor(Z)
    labels = np.asarray(y)
    if labels.shape != (len(x),) or labels.dtype.kind not in "iu" or set(labels.tolist()) != set(range(7)):
        raise ValueError("Source fit requires all seven encoded integer classes")
    torch.manual_seed(seed)
    model = DenseClassifier(x.shape[1], candidate["hidden_sizes"], 7)
    initial = state_fingerprint(model)
    truth = torch.as_tensor(labels, dtype=torch.long)
    if candidate["class_weight"] not in {"none", "balanced"}:
        raise ValueError("Unknown source class weighting")
    counts = torch.bincount(truth, minlength=7).float()
    weights = None if candidate["class_weight"] == "none" else len(truth)/(7*counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for epoch in range(candidate["epochs"]):
        model.train(); total = 0.; samples = 0
        for indices in torch.randperm(len(x), generator=generator).split(candidate["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices]), truth[indices], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite source cross-entropy")
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(indices); samples += len(indices)
        record = {"epoch": epoch+1, "mean_batch_cross_entropy": total/samples}
        if epoch == 0 or (epoch+1) % 20 == 0 or epoch+1 == candidate["epochs"]:
            predicted = model_scores(model, x).argmax(1)
            record.update({"source_"+k: v for k,v in fast_metrics(labels, predicted, 7).items()})
        losses.append(record)
    model.eval()
    model._source_initial_fingerprint = initial
    return model, losses


def model_scores(model, Z):
    x = _features_tensor(Z); model.eval()
    with torch.no_grad():
        scores = model(x).cpu().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite source neural scores")
    return scores


def _rank(candidates, means, tolerance=.001):
    best_accuracy = max(means[c["id"]]["accuracy"] for c in candidates)
    near = [c for c in candidates if best_accuracy-means[c["id"]]["accuracy"] <= tolerance]
    # Metric ranking within the accuracy equivalence window is predeclared.
    return sorted(near, key=lambda c: (-means[c["id"]]["macro_f1"], -means[c["id"]]["balanced_accuracy"],
        c["pca_components"] or 10**6, len(c["hidden_sizes"]), c["epochs"],
        0 if c["class_weight"] == "none" else 1, c["id"]))[0]


def choose_source(rows, candidates, selection):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate_id"], row["stage"])].append(row)
    means = {stage: {c["id"]: {k: float(np.mean([r[k] for r in grouped[(c["id"], stage)]]))
                              for k in METRICS} for c in candidates} for stage in ("selection", "gate")}
    reference = next(c for c in candidates if c["id"] == selection["reference_candidate_id"])
    def passes(candidate, stage):
        a,b = means[stage][candidate["id"]], means[stage][reference["id"]]
        return (a["accuracy"]-b["accuracy"] > selection["minimum_accuracy_gain"]
                and a["macro_f1"]-b["macro_f1"] >= -selection["max_macro_f1_loss"]
                and a["balanced_accuracy"]-b["balanced_accuracy"] >= -selection["max_balanced_accuracy_loss"])
    eligible = [c for c in candidates if passes(c,"selection")]
    proposed = _rank(eligible if eligible else candidates, means["selection"], selection["accuracy_tolerance"])
    passed = bool(eligible) and passes(proposed,"gate")
    families = {balance: _rank([c for c in candidates if c["class_weight"] == balance], means["selection"],
                              selection["accuracy_tolerance"]) for balance in ("none", "balanced")}
    return {"selected": proposed if passed else reference, "proposed": proposed, "reference": reference,
            "passed_selection_and_gate": passed, "fallback_to_reference": not passed,
            "eligible_selection_candidates":len(eligible),
            "family_diagnostics": families, "means": means, "selected_using_final_query": False}


def _subset(training, indices):
    return {"X_alpha": training["X_alpha"][indices], "y": training["y"][indices],
            "record_ids": training["record_ids"][indices]}


def _provenance(input_path, plan_path, protocol_path):
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/"scripts/run_neural_base_search.py",
             Path(__file__).with_name("calibration_exploration.py"),
             Path(__file__).with_name("within_batch_holdout.py"),
             Path(__file__).with_name("paper_fewshot_classification.py")]
    return {"input_sha256": digest(input_path), "plan_sha256": digest(plan_path),
            "protocol_sha256": digest(protocol_path), "candidate_grid_sha256": object_digest(candidate_grid()),
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in files},
            "python": platform.python_version(),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "scikit-learn", "joblib")}}


def _train_cached(directory, source, classes, candidate, seed, preprocessor, features, *, resume=False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory/"training.json"
    if resume and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["source_ids"] != source["record_ids"].tolist() or meta["candidate"] != candidate or meta["seed"] != seed:
            raise ValueError("Cached neural source context differs")
        if digest(directory/"model.pt") != meta["model_sha256"] or digest(directory/"preprocessor.joblib") != meta["preprocessor_sha256"]:
            raise ValueError("Cached source artifacts changed")
        model = DenseClassifier(features.shape[1], candidate["hidden_sizes"], len(classes))
        model.load_state_dict(torch.load(directory/"model.pt", map_location="cpu", weights_only=True)); model.eval()
        if state_fingerprint(model) != meta["fingerprint"]:
            raise ValueError("Cached source fingerprint changed")
        return model,meta
    model,losses = fit_model(features, encoded(source["y"],classes), candidate, seed)
    torch.save(model.state_dict(), directory/"model.pt"); joblib.dump(preprocessor, directory/"preprocessor.joblib")
    prediction = model_scores(model, features).argmax(1)
    meta = {"candidate": candidate, "seed": seed, "classes": classes.tolist(), "source_ids": source["record_ids"].tolist(),
            "source_samples": len(features), "transformed_features": int(features.shape[1]),
            "fingerprint": state_fingerprint(model), "initial_fingerprint": model._source_initial_fingerprint,
            "model_sha256": digest(directory/"model.pt"), "preprocessor_sha256": digest(directory/"preprocessor.joblib"),
            "source_metrics": fast_metrics(encoded(source["y"], classes), prediction, len(classes), detailed=True),
            "losses": losses, "class_weight_semantics": "none or source n/(7*class_count) in torch cross_entropy, mean reduction within each minibatch; no balanced resampling",
            "parameter_count": sum(p.numel() for p in model.parameters()), "source_only": True}
    write_json(meta_path, meta)
    return model,meta


def _load_roles(input_path, plan_path, protocol):
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    roles,truth,preparation = prepare_holdout(arrays, json.loads(Path(plan_path).read_text()),
        frequency_min=protocol["frequency"]["lower_inclusive"], frequency_max=protocol["frequency"]["upper_inclusive"])
    if preparation["feature_count"] != protocol["frequency"]["expected_points"]:
        raise ValueError("Frequency features differ from frozen protocol")
    if str(np.asarray(arrays["dataset_version"])) != protocol["dataset_version"]:
        raise ValueError("Input version differs from neural protocol")
    return roles,truth,preparation


def load_locked_contexts(input_path, plan_path, run, scope="development"):
    """Load only the source-selected context; final prediction has no y field.

    Returns (contexts, truth). Every role's X is already transformed exclusively
    by its own source-fitted preprocessor, and y uses the fixed class-index order.
    The separate final truth dictionary maps physical record ID to string class.
    No context loading operation fits or adapts a model.
    """
    run = Path(run)
    manifest = json.loads((run/"run_manifest.json").read_text())
    if digest(input_path) != manifest["input_sha256"] or digest(plan_path) != manifest["plan_sha256"]:
        raise ValueError("Context input/partition differs from recorded source run")
    protocol = json.loads((run/"protocol_snapshot.json").read_text())
    if digest(run/"protocol_snapshot.json") != manifest["protocol_sha256"]:
        raise ValueError("Cached protocol snapshot differs")
    lock = json.loads((run/"selection_lock.json").read_text())
    choice = lock["choice"]["selected"]
    roles,truth,_ = _load_roles(input_path, plan_path, protocol)
    training = roles["training"]; classes = np.unique(training["y"])
    if scope == "development":
        parts = development_episodes(training, protocol["development"]["episodes"], protocol["development"]["split_seed"])
        specs = [(episode, _subset(training,p["source"]), _subset(training,p["support_pool"]),
                  _subset(training,p["query"]), run/"development"/f"episode_{episode:02d}") for episode,p in enumerate(parts)]
    elif scope == "final":
        specs = [(None, training, roles["calibration"], roles["prediction"], run/"final")]
    else:
        raise ValueError("Context scope must be development or final")
    contexts = []
    for episode,source,pool,prediction,base in specs:
        for seed in protocol["seeds"]:
            directory = base/f"seed_{seed}"/choice["id"]
            meta = json.loads((directory/"training.json").read_text())
            if (meta["candidate"] != choice or meta["source_ids"] != source["record_ids"].tolist()
                    or digest(directory/"model.pt") != meta["model_sha256"]
                    or digest(directory/"preprocessor.joblib") != meta["preprocessor_sha256"]):
                raise ValueError("Source context identity or artifact hash changed")
            preprocessor = joblib.load(directory/"preprocessor.joblib")
            model = DenseClassifier(meta["transformed_features"], choice["hidden_sizes"], len(classes))
            model.load_state_dict(torch.load(directory/"model.pt", map_location="cpu", weights_only=True)); model.eval()
            if state_fingerprint(model) != meta["fingerprint"]:
                raise ValueError("Source context model state differs")
            payloads = []
            for item in (source,pool,prediction):
                payload = {"X": transform_features(preprocessor,item["X_alpha"]), "record_ids": item["record_ids"].copy()}
                if "y" in item:
                    payload["y"] = encoded(item["y"],classes)
                payloads.append(payload)
            if scope == "final" and "y" in payloads[2]:
                raise AssertionError("Final context leaked prediction truth")
            contexts.append({"episode": episode, "seed": seed, "candidate": choice, "classes": classes.copy(),
                "model": model, "preprocessor": preprocessor, "source": payloads[0], "support_pool": payloads[1],
                "prediction": payloads[2], "directory": str(directory), "fingerprint": meta["fingerprint"],
                "preprocessor_sha256": meta["preprocessor_sha256"], "selection_lock_sha256": digest(run/"selection_lock.json")})
    return contexts,truth if scope == "final" else {}


def run_search(input_path, plan_path, protocol_path, output, *, resume=False, stage="development"):
    if stage not in {"development", "final"}:
        raise ValueError("Stage must be development or final")
    output = Path(output); protocol = json.loads(Path(protocol_path).read_text()); candidates = candidate_grid()
    if protocol.get("candidate_grid_sha256") != object_digest(candidates) or protocol["candidate_configurations"] != len(candidates):
        raise ValueError("Frozen protocol does not match implemented grid")
    if protocol["selection"]["reference_candidate_id"] != reference_candidate()["id"]:
        raise ValueError("Reference must be the predeclared PCA20/64-32/200/balanced candidate")
    provenance = _provenance(input_path,plan_path,protocol_path)
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"run_manifest.json").exists():
            raise FileExistsError("Use a new output directory or verified resume")
        if json.loads((output/"run_manifest.json").read_text()) != provenance:
            raise ValueError("Resume refused because source code/input/protocol/runtime changed")
        if (output/"status.json").exists():
            status = json.loads((output/"status.json").read_text())
            if status["status"] == "completed":
                return json.loads((output/"summary.json").read_text())
            if stage == "development" and status["status"] == "development_completed":
                lock = json.loads((output/"selection_lock.json").read_text())
                return {"stage":"development_completed","candidate":lock["choice"]["selected"],
                        "selection_lock_sha256":digest(output/"selection_lock.json")}
    if stage == "final" and not (output/"selection_lock.json").exists():
        raise ValueError("Final requires a completed source-only development lock")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"run_manifest.json",provenance); write_json(output/"candidate_grid.json",candidates)
    (output/"protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    roles,truth,preparation = _load_roles(input_path,plan_path,protocol)
    write_json(output/"preparation.json",preparation)
    training = roles["training"]; classes = np.unique(training["y"])
    development = protocol["development"]
    partitions = development_episodes(training,development["episodes"],development["split_seed"])
    write_json(output/"development_partitions.json",[{key: training["record_ids"][values].tolist() for key,values in part.items()} for part in partitions])
    started = time.monotonic()
    previous_development_seconds = 0.
    if stage == "final":
        status = json.loads((output/"status.json").read_text())
        if status["status"] != "development_completed":
            raise ValueError("Final requires completed development status")
        previous_development_seconds = status["seconds"]
        all_rows = json.loads((output/"all_development_metrics.json").read_text())
        if digest(output/"selection_lock.json") != status["selection_lock_sha256"]:
            raise ValueError("Development source lock changed")
        pending = []
    else:
        all_rows = []; pending = list(enumerate(partitions))
    for episode,part in pending:
        source = _subset(training,part["source"]); query = _subset(training,part["query"])
        y_query = encoded(query["y"],classes)
        preprocessors = {}; transformed = {}
        for candidate in candidates:
            pc = candidate["pca_components"]
            if pc not in preprocessors:
                preprocessors[pc] = fit_preprocessor(source["X_alpha"],candidate)
                transformed[pc] = (transform_features(preprocessors[pc],source["X_alpha"]),
                                   transform_features(preprocessors[pc],query["X_alpha"]))
            source_x,query_x = transformed[pc]
            for seed in protocol["seeds"]:
                directory = output/"development"/f"episode_{episode:02d}"/f"seed_{seed}"/candidate["id"]
                model,metadata = _train_cached(directory,source,classes,candidate,seed,preprocessors[pc],source_x,resume=resume)
                scores = model_scores(model,query_x); predicted = scores.argmax(1)
                metrics = fast_metrics(y_query,predicted,len(classes),detailed=True)
                path = directory/"predictions.csv.gz"
                with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
                    writer = csv.DictWriter(f,fieldnames=["candidate_id","episode","seed","record_id","y_true","y_predicted","is_correct"],lineterminator="\n")
                    writer.writeheader();writer.writerows({"candidate_id":candidate["id"],"episode":episode,"seed":seed,
                        "record_id":str(i),"y_true":str(classes[t]),"y_predicted":str(classes[p]),"is_correct":int(t==p)}
                        for i,t,p in zip(query["record_ids"],y_query,predicted))
                row = {"candidate_id":candidate["id"],"episode":episode,"seed":seed,
                       "stage":"selection" if episode < development["selection_episodes"] else "gate",
                       **metrics,"model_fingerprint":metadata["fingerprint"],"prediction_sha256":digest(path)}
                write_json(directory/"metrics.json",row); all_rows.append(row)
        write_json(output/"status.json",{"status":"development","episodes_completed":episode+1,"evaluations":len(all_rows)})
        print(json.dumps({"stage":"neural_source_development","episode":episode+1,"evaluations":len(all_rows)}),flush=True)
    choice = choose_source(all_rows,candidates,protocol["selection"])
    lock = {"choice":choice,"source_only_development":True,"provenance_sha256":object_digest(provenance),"development_evaluations":len(all_rows)}
    lock_path = output/"selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Cached source choice differs")
    write_json(output/"all_development_metrics.json",all_rows);write_json(lock_path,lock);lock_sha=digest(lock_path)
    print(json.dumps({"stage":"neural_source_choice_locked","candidate":choice["selected"]["id"],"sha256":lock_sha}),flush=True)
    if stage == "development":
        write_json(output/"status.json",{"status":"development_completed","seconds":time.monotonic()-started,
            "development_evaluations":len(all_rows),"development_prediction_rows":sum(r["total"] for r in all_rows),
            "saved_source_models":len(all_rows),"selection_lock_sha256":lock_sha,"final_prediction_scored":False})
        return {"stage":"development_completed","candidate":choice["selected"],"selection_lock_sha256":lock_sha}
    # This boundary follows a source-only choice; final truth is joined only for scoring.
    selected = choice["selected"]; preprocessor = fit_preprocessor(training["X_alpha"],selected)
    source_x = transform_features(preprocessor,training["X_alpha"])
    query_x = transform_features(preprocessor,roles["prediction"]["X_alpha"])
    query_ids = roles["prediction"]["record_ids"]
    final_rows = []
    for seed in protocol["seeds"]:
        directory = output/"final"/f"seed_{seed}"/selected["id"]
        model,metadata = _train_cached(directory,training,classes,selected,seed,preprocessor,source_x,resume=resume)
        predicted = model_scores(model,query_x).argmax(1)
        if digest(lock_path) != lock_sha:
            raise ValueError("Source lock changed before final scoring")
        actual = encoded([truth[str(i)] for i in query_ids],classes)
        metrics = fast_metrics(actual,predicted,len(classes),detailed=True)
        path = directory/"predictions.csv.gz"
        with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["candidate_id","seed","record_id","y_true","y_predicted","is_correct"],lineterminator="\n")
            writer.writeheader();writer.writerows({"candidate_id":selected["id"],"seed":seed,"record_id":str(i),"y_true":str(classes[t]),
                "y_predicted":str(classes[p]),"is_correct":int(t==p)} for i,t,p in zip(query_ids,actual,predicted))
        final_rows.append({"candidate_id":selected["id"],"seed":seed,**metrics,"source_metrics":metadata["source_metrics"],
                           "fingerprint":metadata["fingerprint"],"prediction_sha256":digest(path)})
    summary={"candidate":selected,"seeds":protocol["seeds"],"final_seed_evaluations":len(final_rows),
             **{k+"_mean":float(np.mean([r[k] for r in final_rows])) for k in METRICS},
             "per_seed":final_rows,"calibration_labels_used":0,"source_model_improvement_not_calibration":True,
             "same_batch_retrospective":True}
    write_json(output/"final_metrics.json",final_rows);write_json(output/"summary.json",summary)
    write_json(output/"status.json",{"status":"completed","seconds":previous_development_seconds+time.monotonic()-started,
        "development_evaluations":len(all_rows),"development_prediction_rows":sum(r["total"] for r in all_rows),
        "final_evaluations":len(final_rows),"final_prediction_rows":sum(r["total"] for r in final_rows),
        "saved_source_models":len(all_rows)+len(final_rows),"selection_lock_sha256":lock_sha,
        "query_labels_used_to_fit":False,"cross_batch_validation":False,"newff_exact_reproduction":False})
    return summary
