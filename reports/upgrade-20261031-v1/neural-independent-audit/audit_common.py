"""Independent arithmetic and saved-state checks; no fit entry is called."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/upgrade-20261031-v1/neural"
DEST = Path(__file__).resolve().parent
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value):
    value = np.asarray(value)
    data = json.dumps(value.tolist(), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() if value.dtype.kind == "O" else np.ascontiguousarray(value).tobytes()
    result = hashlib.sha256(json.dumps({"shape":value.shape, "dtype":str(value.dtype)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
    result.update(data)
    return result.hexdigest()


def source_sha(X, y=None):
    return object_sha({"X":array_sha(X), "y":None if y is None else array_sha(y)})


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def predictions(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len({r["record_id"] for r in rows}):
        raise AssertionError("Repeated truth record")
    return {r["record_id"]:r["label"] for r in rows}


def metrics(rows, mapping):
    if len(rows) != len({r["record_id"] for r in rows}) or {r["record_id"] for r in rows} != set(mapping):
        raise AssertionError("Duplicated or incomplete independent denominator")
    matrix = np.zeros((7, 7), dtype=int)
    for row in rows:
        matrix[CLASSES.index(mapping[row["record_id"]]), CLASSES.index(row["y_predicted"])] += 1
    tp = matrix.diagonal(); actual = matrix.sum(1); found = matrix.sum(0)
    recall = np.divide(tp, actual, out=np.zeros(7), where=actual > 0)
    precision = np.divide(tp, found, out=np.zeros(7), where=found > 0)
    f1 = np.divide(2 * tp, actual + found, out=np.zeros(7), where=actual + found > 0)
    return {"accuracy":float(tp.sum()/matrix.sum()), "macro_f1":float(f1.mean()),
        "balanced_accuracy":float(recall[actual > 0].mean()), "correct":int(tp.sum()),
        "total":int(matrix.sum()), "confusion_matrix":matrix.tolist(),
        "per_class":{c:{"precision":float(p), "recall":float(r), "f1":float(f), "support":int(n)}
            for c,p,r,f,n in zip(CLASSES,precision,recall,f1,actual,strict=True)}}


class Auditor:
    def __init__(self):
        self.checks = 0
        self.max_difference = 0.
        self.source_cache = {}
        self.initial_cache = {}
        self.fit_expectations = []
        self.models_replayed = 0
        self.mc_selectors_recomputed = 0

    def require(self, condition, detail):
        self.checks += 1
        if not condition:
            raise AssertionError(detail)

    def close(self, actual, expected, detail, atol=1e-10, rtol=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.require(actual.shape == expected.shape, detail + " shape")
        difference = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        self.max_difference = max(self.max_difference, difference)
        self.require(np.allclose(actual, expected, atol=atol, rtol=rtol), detail)

    def metric_match(self, actual, expected, detail):
        for name in ("accuracy", "macro_f1", "balanced_accuracy"):
            self.close(actual[name], expected[name], detail+" "+name, atol=1e-14, rtol=1e-13)
        for name in ("correct", "total", "confusion_matrix"):
            self.require(actual[name] == expected[name], detail+" "+name)
        for cls in CLASSES:
            for name in ("precision", "recall", "f1", "support"):
                self.close(actual["per_class"][cls][name], expected["per_class"][cls][name], detail+" "+cls+" "+name)

    def source(self, context):
        key = context["split_id"]
        if key not in self.source_cache:
            self.require(sha(ROOT/context["source_pack"]) == context["source_pack_sha256"], key+" source pack bytes")
            self.require(sha(ROOT/context["inference_pack"]) == context["inference_pack_sha256"], key+" inference pack bytes")
            source, query = npz(ROOT/context["source_pack"]), npz(ROOT/context["inference_pack"])
            self.require(source["record_ids"].tolist() == context["source_ids"], key+" source IDs")
            self.require(query["record_ids"].tolist() == context["inference_ids"], key+" inference IDs")
            self.require(not set(context["source_ids"]) & set(context["inference_ids"]), key+" disjoint")
            self.require(set(query) == {"record_ids","X_alpha","X_n","X_time","X_frequency"}, key+" labels excluded")
            self.source_cache[key] = source, query, truth(ROOT/context["scoring_truth"])
        return self.source_cache[key]

    def model(self, trial, context, rows):
        from herb_spectra.upgrade_neural import load_fitted, UpgradeNet, state_hash
        directory = ROOT / trial["model_directory"]
        source, query, target = self.source(context)
        metadata = load(directory/"training.json")
        self.require(sha(directory/"training.json") == trial["model_metadata_sha256"], "Training metadata bytes")
        self.require(sha(directory/"predictions.csv.gz") == trial["predictions_sha256"], "Trial predictions bytes")
        fitted = load_fitted(directory, expected_split_id=context["split_id"], expected_source_ids=context["source_ids"])
        self.require(metadata["seed"] == trial["seed"] and metadata["family"] == trial["family"], "Fitted role/seed")
        self.require(metadata["class_order"] == list(CLASSES), "Fitted classes")
        self.require(metadata["source_ids"] == context["source_ids"], "Model source scope")
        self.require(metadata["pretrained_weights"] is None and metadata["optimizer_initialized_fresh"] is True, "No reused weights/optimizer")
        self.require(metadata["target_labels_received_by_training"] is False and metadata["early_stopping"] is False, "No target supervision/early stopping")
        y = source["source_labels"]
        counts = np.asarray([np.count_nonzero(y == c) for c in CLASSES])
        self.require(metadata["source_class_counts"] == dict(zip(CLASSES,counts.tolist())), "Source class counts")
        expected_weight = len(y)/(7*counts) if metadata["profile"]["class_weight"] == "balanced" else np.ones(7)
        self.close(metadata["class_weights"], expected_weight, "Source-only class weights")
        source_features, source_64, scaler_fingerprints = {}, {}, {}
        for mode, scaler in fitted.scalers.items():
            raw = np.asarray(source["X_"+mode],float)
            values = (raw-raw.mean(1,keepdims=True))/raw.std(1,ddof=1,keepdims=True) if mode=="alpha" else raw
            self.close(scaler.mean_, values.mean(0), "Source-only scaler mean")
            self.close(scaler.var_, values.var(0), "Source-only scaler variance")
            self.close(metadata["scaler_statistics"][mode]["mean"], scaler.mean_, "Saved mean")
            self.close(metadata["scaler_statistics"][mode]["variance"], scaler.var_, "Saved variance")
            self.require(joblib.hash(source["X_"+mode]) == metadata["raw_source_modality_fingerprints"][mode], "Raw source value fingerprint")
            source_64[mode] = scaler.transform(values)
            source_features[mode] = source_64[mode].astype(np.float32)
            scaler_fingerprints["scaler_"+mode] = source_sha(values)
        if fitted.selector is not None:
            selector = fitted.selector
            coefficients = np.asarray(selector.coefficients)
            runs, classes, width = coefficients.shape
            self.require((runs,classes,width)==(metadata["profile"]["mc_runs"],7,612), "MC coefficients budget/shape")
            mean, sd = coefficients.mean(0), coefficients.std(0,ddof=1)
            floor = max(1.,float(sd.max()))*1e-12
            importance = np.linalg.norm(np.abs(mean/np.maximum(sd,floor)),axis=0)
            ranking = np.argsort(-importance,kind="stable")
            K = int(np.ceil(metadata["profile"]["mc_ratio"]*612))
            self.close(selector.importance, importance, "MC importance from saved actual coefficients", atol=0., rtol=0.)
            self.require(np.array_equal(selector.ranking,ranking), "MC ranking")
            self.require(np.array_equal(selector.selected_indices,np.sort(ranking[:K])), "MC feature set")
            self.require(selector.metadata["source_ids"] == context["source_ids"], "MC source IDs")
            source_features["mcuve"] = source_features["alpha"][:,selector.selected_indices]
            self.mc_selectors_recomputed += 1
        features_fp = object_sha({str(k):array_sha(v) for k,v in sorted(source_features.items())})
        encoded = np.asarray([CLASSES.index(c) for c in y])
        proof = fitted._leakage_fit_provenance
        self.require(proof["data_fingerprint"] == object_sha({"X":features_fp,"y":array_sha(encoded)}), "Actual neural fit source values/labels")
        self.require(proof["fit_sample_ids"] == context["source_ids"] and proof["split_id"] == context["split_id"], "Neural cache training provenance")
        identity = metadata["cache_identity"]
        self.require(identity["train_ids"] == context["source_ids"] and identity["split_id"] == context["split_id"], "Guarded cache context")
        self.require(identity["data_fingerprint"] == proof["data_fingerprint"], "Cache actual source fingerprint")
        init_key = (metadata["family"],metadata["profile"]["id"],metadata["seed"])
        if init_key not in self.initial_cache:
            torch.manual_seed(metadata["seed"])
            model = UpgradeNet(metadata["family"],metadata["profile"],{k:v.shape[1] for k,v in source_features.items()})
            self.initial_cache[init_key] = state_hash(model)
        self.require(metadata["initial_state_sha256"] == self.initial_cache[init_key], "Independent seed initialization")
        self.require(len(metadata["loss_curve"]) == metadata["epochs"] == metadata["profile"]["epochs"], "Fixed epoch budget")
        self.require(np.isfinite(metadata["loss_curve"]).all(), "Finite training loss")
        batches = metadata["epochs"]*int(np.ceil(len(y)/64))
        for module in fitted.model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                self.require(module.track_running_stats, "No query-batch BN statistics")
                self.require(int(module.num_batches_tracked) == batches, "BN only updated on source minibatches")
        inference = load(directory/"inference_audit.json")
        self.require(inference["parameters_and_buffers_unchanged"] is True, "BN/state inference audit")
        self.require(inference["samples_checked"] == (171 if context["stage"]=="outer" else 7), "Pointwise scope")
        self.require([r["record_id"] for r in rows] == context["inference_ids"], "Prediction row binding")
        before = state_hash(fitted.model)
        score = fitted.predict_scores({k[2:]:v for k,v in query.items() if k.startswith("X_")})
        stored = np.asarray([[float(r["score_"+c]) for c in CLASSES] for r in rows])
        self.close(score,stored,"Loaded model score replay",atol=1e-5,rtol=1e-5)
        self.require([CLASSES[i] for i in score.argmax(1)] == [r["y_predicted"] for r in rows], "Loaded model class replay")
        self.require(state_hash(fitted.model) == before == metadata["fitted_state_sha256"], "Inference state unchanged")
        self.models_replayed += 1
        self.fit_expectations.append({"split_id":context["split_id"], "source_ids":context["source_ids"],
            "scaler_fingerprints":scaler_fingerprints, "torch_fingerprint":proof["data_fingerprint"],
            "source_labels":y, "mc_X":source_64.get("alpha") if fitted.selector is not None else None,
            "profile":metadata["profile"], "seed":metadata["seed"], "family":metadata["family"],
            "code_version":metadata["code_version"]})
        return metrics(rows,target)

    def scope_log(self, path, trials):
        """Check actual fit identities and transformed values, including each PLS."""
        self.require(len(self.fit_expectations) == len(trials), "All saved models have fit expectations")
        trial_index, stack, run_index, target_index = 0, [], 0, 0
        coefficient, components, lines = None, Counter(), 0
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                lines += 1
                row = json.loads(line)
                self.require(trial_index < len(trials), "Extra source fit after expected trial coverage")
                expected = self.fit_expectations[trial_index]
                ids = row["fit_sample_ids"]
                self.require(row["split_id"] == expected["split_id"], "Fit context changed")
                self.require(row["code_version"] == expected["code_version"], "Fit source code version changed")
                self.require(len(set(ids)) == len(ids) and set(ids) <= set(expected["source_ids"]), "Actual fit outside source scope")
                self.require(row["train_id_digest"] == object_sha(ids), "Fit ID digest")
                component, event = row["component"], row["event"]
                self.require(event in {"fit_start","fit_completed"}, "Failed/rejected fit in completed search")
                if event == "fit_start":
                    if component in expected["scaler_fingerprints"]:
                        self.require(ids == expected["source_ids"], "Scaler did not fit the current complete source")
                        self.require(row["data_fingerprint"] == expected["scaler_fingerprints"][component], "Scaler fit values")
                    elif component == "mcuve_selector":
                        self.require(ids == expected["source_ids"] and expected["mc_X"] is not None, "MC source context")
                        self.require(row["data_fingerprint"] == source_sha(expected["mc_X"], expected["source_labels"]), "MC actual supervised source")
                        order = np.argsort(expected["source_ids"])
                        mc_ids = np.asarray(expected["source_ids"])[order]
                        mc_labels = expected["source_labels"][order]
                        mc_x = expected["mc_X"][order]
                        rng = np.random.default_rng(expected["seed"])
                        draws = []
                        for _ in range(expected["profile"]["mc_runs"]):
                            priority = rng.random(len(mc_ids)); indices = []
                            for label in CLASSES:
                                members = np.flatnonzero(mc_labels == label)
                                k = min(len(members),max(1,int(np.ceil(len(members)*.8))))
                                indices.extend(members[np.argsort(priority[members],kind="stable")[:k]])
                            draws.append(np.sort(indices))
                        coefficient = {"ids":mc_ids,"labels":mc_labels,"X":mc_x,"draws":draws}
                        run_index = 0
                    elif component == "MC-UVE.coefficient_standardization":
                        self.require(coefficient is not None and run_index < len(coefficient["draws"]), "Unbound coefficient fit")
                        positions = coefficient["draws"][run_index]
                        selected_ids = coefficient["ids"][positions].tolist()
                        self.require(ids == selected_ids, "Actual Monte Carlo subsample differs from fixed seed/source")
                        x = coefficient["X"][positions]
                        target = np.eye(7)[[CLASSES.index(c) for c in coefficient["labels"][positions]]]
                        self.require(row["data_fingerprint"] == source_sha(x,target), "MC coefficient standardization actual values")
                        sd = x.std(0); sd[sd <= np.finfo(float).eps] = 1
                        coefficient["current_Z"] = (x-x.mean(0))/sd
                        coefficient["current_Y"] = target
                        coefficient["current_ids"] = selected_ids
                        target_index = 0
                    elif component == "MC-UVE.PLSRegression.fit":
                        self.require(coefficient is not None and target_index < 7, "Unbound PLS target")
                        self.require(ids == coefficient["current_ids"], "PLS source IDs differ from MC subset")
                        column = coefficient["current_Y"][:,target_index]
                        target = (column-column.mean())/column.std()
                        self.require(row["data_fingerprint"] == source_sha(coefficient["current_Z"],target), "PLS actual standardized X/y")
                        target_index += 1
                    elif component == "torch_training":
                        self.require(ids == expected["source_ids"], "Torch source scope")
                        self.require(row["data_fingerprint"] == expected["torch_fingerprint"], "Torch actual source features/labels")
                    else:
                        self.require(False,"Unknown learned component "+component)
                    stack.append((component,ids,row["data_fingerprint"]))
                else:
                    self.require(stack and stack.pop() == (component,ids,row["data_fingerprint"]), "Fit start/end provenance mismatch")
                    components[component] += 1
                    if component == "MC-UVE.coefficient_standardization":
                        self.require(target_index == 7, "All OVR PLS targets traced")
                        run_index += 1
                    if component == "mcuve_selector":
                        self.require(run_index == expected["profile"]["mc_runs"], "All MC iterations traced")
                    if component == "torch_training":
                        self.require(not stack, "Nested source operation left unfinished")
                        trial_index += 1
                        coefficient = None
        self.require(not stack and trial_index == len(trials), "Fit coverage incomplete")
        return {"jsonl_rows":lines,"model_fit_contexts":trial_index,"actual_fit_components":dict(components),
                "actual_fits_total":sum(components.values()),"all_fit_ids_in_current_source":True,
                "source_values_and_labels_bound_independently":True,"MC_subsamples_and_each_PLS_target_verified":True}
