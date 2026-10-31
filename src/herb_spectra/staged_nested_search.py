"""Four-round, source-only nested tuning; outer results never select parameters.

MC-UVE uses the existing one-hot PLS1 implementation. Prefix reuse computes the
same random coefficient stream as independent n_runs calls, not a new selector.
Standard SVC vote prediction intentionally matches the historical 90.41% run.
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
import warnings

import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .classification import (aggregate_samples, load_reviewed_dataset,
                             sample_folds, classification_metrics)
from .selection import MCUVESelector, _prepare, _subsample, _coefficients
from .spectral_models import SpectralPreprocessor
from .image_feature_models import ScatterPreprocessor


METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
PREPROCESSORS = ("raw", "snv", "sg0_snv", "sg1_snv", "sg2_snv", "msc", "linear_detrend_snv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def retained_count(features, fraction):
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Retained ratio must be in (0,1]")
    return max(1, min(features, int(np.ceil(features*fraction))))


def class_weight_spec(y, mode):
    classes, counts = np.unique(y, return_counts=True)
    if mode == "none":
        return None, {str(c): 1. for c in classes}
    base = len(y)/(len(classes)*counts.astype(float))
    if mode == "balanced":
        return "balanced", {str(c):float(w) for c,w in zip(classes,base)}
    powers = {"power_0.5": .5, "power_1.5": 1.5}
    if mode not in powers:
        raise ValueError("Unknown class-weight rule")
    weights = base**powers[mode]
    weights /= np.sum(weights*counts)/len(y)
    result = {c.item() if isinstance(c,np.generic) else c:float(w) for c,w in zip(classes,weights)}
    return result, {str(c):float(w) for c,w in zip(classes,weights)}


def gamma_spec(values, rule):
    values = np.asarray(values, dtype=float)
    variance = float(values.var())
    scale = 1/(values.shape[1]*variance) if variance else 1.
    if rule == "scale": return "scale", scale
    if rule == "auto": return "auto", 1/values.shape[1]
    if rule == "scale_x0.1": return .1*scale, .1*scale
    if rule == "scale_x10": return 10*scale, 10*scale
    raise ValueError("Unknown gamma rule")


def mc_grid():
    return [{"id":f"mc{i:03d}", "runs":runs,"components":lv,"ratio":ratio}
            for i,(runs,lv,ratio) in enumerate((r,l,f) for r in (10,20,50)
                for l in (1,2,3) for f in (.05,.10,.15,.20))]


def svm_grid():
    return [{"id":f"sv{i:03d}","C":c,"gamma":gamma,"class_weight":weight}
            for i,(c,gamma,weight) in enumerate((c,g,w) for c in (.01,.1,1.,10.,100.,1000.)
                for g in ("scale","auto","scale_x0.1","scale_x10")
                for w in ("none","balanced","power_0.5","power_1.5"))]


def historical_grid():
    # ParameterGrid sorts keys: C first, then preprocessing, matching sklearn.
    return [{"id":f"hist{i:03d}","preprocessing":p["preprocess__mode"],
             "mc":{"runs":20,"components":2,"ratio":.1},
             "svm":{"C":p["model__C"],"gamma":"scale","class_weight":"balanced"}}
            for i,p in enumerate(ParameterGrid({"preprocess__mode":["raw","snv"],
                                                "model__C":[.1,1.,10.,100.]}))]


def _prefix_selector(values, y, mc, seed, cache):
    runs, components = int(mc["runs"]), int(mc["components"])
    count = retained_count(values.shape[1],float(mc["ratio"]))
    if runs not in (10,20,50):
        # Public helper also allows R4's computation-budget controls.
        selector = MCUVESelector(n_runs=runs,n_components=components,
            n_features_to_select=count,sample_fraction=.8,task="classification",random_state=seed)
        return selector.fit(values,y)
    key = ("coefficients",joblib.hash(values),joblib.hash(y),components,seed)
    budget=max(runs,int(cache.get("__coefficient_budget__",runs)))
    if key not in cache or len(cache[key][0]) < budget:
        x,Y,labels,classes,unit = _prepare(values,y,"classification",None)
        rng=np.random.default_rng(seed)
        coefficients=np.asarray([_coefficients(x[s],Y[s],components)
            for s in (_subsample(labels,len(x),.8,rng) for _ in range(budget))])
        cache[key]=(coefficients,classes,unit)
    full,classes,unit=cache[key];coefficients=full[:runs]
    mean=coefficients.mean(0);sd=coefficients.std(0,ddof=1)
    floor=max(1.,float(sd.max()))*1e-12
    stability=mean/np.maximum(sd,floor)
    importance=np.linalg.norm(np.abs(stability),axis=0)
    selector=MCUVESelector(n_runs=runs,n_components=components,n_features_to_select=count,
        sample_fraction=.8,task="classification",random_state=seed)
    selector.classes_=classes;selector.sample_unit_=unit
    selector.n_training_sample_units_=len(values);selector.coefficients_=coefficients.copy()
    selector.signed_stability_=stability;selector.importances_=importance
    selector.ranking_=np.argsort(-importance,kind="stable")
    selector.computed_stream_runs_=len(full)
    selector._finish(values,selector.ranking_[:count])
    return selector


@dataclass
class TransformBundle:
    preprocessor: object
    scaler: StandardScaler
    selector: MCUVESelector
    metadata: dict

    def transform(self, values):
        return self.selector.transform(self.scaler.transform(self.preprocessor.transform(values)))


def fit_preprocess_bundle(Xsource,Xquery,axis,preprocessing):
    """Full612 preprocessing/scale for independent CNN controls; no MC-UVE."""
    x=np.asarray(Xsource,dtype=float);query=np.asarray(Xquery,dtype=float)
    prep=(ScatterPreprocessor("msc").fit(x) if preprocessing=="msc"
          else SpectralPreprocessor(preprocessing,np.asarray(axis)))
    scaler=StandardScaler();a=scaler.fit_transform(prep.transform(x))
    return prep,scaler,a,scaler.transform(prep.transform(query))


def fit_transform_bundle(Xtrain,ytrain,Xquery,axis,preprocessing,mc,seed=42,cache=None):
    """No validation targets or outer test arrays enter fitted transformations."""
    cache={} if cache is None else cache
    x=np.asarray(Xtrain,dtype=float);q=np.asarray(Xquery,dtype=float);y=np.asarray(ytrain)
    if x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1] or len(y)!=len(x):
        raise ValueError("Training/query dimensions disagree")
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite spectrum")
    key=("transform",joblib.hash(x),joblib.hash(y),preprocessing,object_digest(mc),seed)
    if key not in cache:
        prep=(ScatterPreprocessor("msc").fit(x) if preprocessing=="msc"
              else SpectralPreprocessor(preprocessing,np.asarray(axis)))
        prepared=prep.transform(x);scaler=StandardScaler();scaled=scaler.fit_transform(prepared)
        selector=_prefix_selector(scaled,y,mc,seed,cache)
        selected=selector.transform(scaled)
        meta={"preprocessing":preprocessing,"mc":dict(mc),"source_samples":len(x),
            "source_value_fingerprint":joblib.hash(x),"source_label_fingerprint":joblib.hash(y),
            "selected_indices":selector.selected_indices_.tolist(),
            "selected_frequencies":np.asarray(axis)[selector.selected_indices_].tolist(),
            "selected_features":len(selector.selected_indices_),"prefix_stream_computed_runs":selector.computed_stream_runs_ if hasattr(selector,"computed_stream_runs_") else mc["runs"],
            "sample_fraction":.8,"scaler_mean":scaler.mean_.tolist(),"scaler_variance":scaler.var_.tolist(),
            "msc_source_reference":prep.reference_.tolist() if preprocessing=="msc" else None,
            "fit_scope":"current_training_partition_only"}
        cache[key]=(TransformBundle(prep,scaler,selector,meta),selected)
    bundle,selected=cache[key]
    return bundle,selected,bundle.transform(q)


@dataclass
class StagedSVM:
    config: dict
    bundle: TransformBundle
    estimator: SVC
    metadata: dict

    @property
    def classes_(self): return self.estimator.classes_

    def predict(self, X): return self.estimator.predict(self.bundle.transform(X))

    def predict_scores(self, X): return self.estimator.decision_function(self.bundle.transform(X))


def fit_source_candidate(config,X,y,axis,seed=42,cache=None):
    bundle,z,_=fit_transform_bundle(X,y,np.asarray(X)[:1],axis,config["preprocessing"],config["mc"],seed,cache)
    svm=config["svm"];weight,weights=class_weight_spec(y,svm["class_weight"])
    gamma,numeric=gamma_spec(z,svm["gamma"])
    estimator=SVC(C=svm["C"],kernel="rbf",gamma=gamma,class_weight=weight,
        break_ties=False,decision_function_shape="ovr",random_state=seed)
    estimator.fit(z,y)
    metadata={**bundle.metadata,"svm":dict(svm),"effective_gamma":float(estimator._gamma),
        "declared_gamma_numeric":numeric,"auto_gamma_numeric":1/z.shape[1],
        "scale_gamma_numeric":1/(z.shape[1]*z.var()) if z.var() else 1.,
        "training_class_weights":weights,"prediction_rule":"SVC standard vote, break_ties=False",
        "scores_are_probabilities":False,"class_order":estimator.classes_.tolist()}
    return StagedSVM(dict(config),bundle,estimator,metadata)


def prepare_protocol_roles(input_path,protocol_path,run=None):
    protocol=json.loads(Path(protocol_path).read_text())
    data=aggregate_samples(load_reviewed_dataset(input_path))
    mask=(data["axis"]>=protocol["frequency"]["lower_inclusive"])&(data["axis"]<=protocol["frequency"]["upper_inclusive"])
    data["axis"]=data["axis"][mask]
    data["X_alpha"]=data["X_alpha"][:,mask];data["X_n"]=data["X_n"][:,mask]
    if data["X_alpha"].shape!=(855,612):raise ValueError("Expected declared855specimens and612features")
    y,groups=data["y"],data["groups"]
    outer=sample_folds(y,groups,protocol["outer_splits"],protocol["split_seed"])
    specs=[];record=[]
    for fold,(train,test) in enumerate(outer):
        inner=sample_folds(y[train],groups[train],protocol["inner_splits"],protocol["split_seed"])
        specs.append({"outer_fold":fold,"train_indices":train,"test_indices":test,
            "inner_folds":[{"train_indices":train[t],"validation_indices":train[v]} for t,v in inner]})
        record.append({"fold":fold,"train_sample_ids":groups[train].tolist(),"test_sample_ids":groups[test].tolist(),
            "inner_folds":[{"train_sample_ids":groups[train][t].tolist(),"validation_sample_ids":groups[train][v].tolist()} for t,v in inner]})
    old=Path(protocol["historical_split_file"])
    if record!=json.loads(old.read_text()):raise ValueError("Nested folds differ from original855sample benchmark")
    if run is not None:
        write_json(Path(run)/"splits.json",record)
        write_json(Path(run)/"data_summary.json",{"samples":len(y),"features":len(data["axis"]),
            "class_counts":{str(c):int(np.sum(y==c)) for c in np.unique(y)},
            "historical_split_sha256":digest(old),"split_records_equal_original":True})
    return data,specs


def load_selection_contexts(input_path,protocol_path,run):
    run=Path(run);data,specs=prepare_protocol_roles(input_path,protocol_path)
    manifest=json.loads((run/"run_manifest.json").read_text())
    if manifest["input_sha256"]!=digest(input_path) or manifest["protocol_sha256"]!=digest(protocol_path):
        raise ValueError("Core context provenance changed")
    locks=json.loads((run/"selection_lock.json").read_text())
    contexts=[]
    for spec,lock in zip(specs,locks["outer_folds"]):
        if spec["outer_fold"]!=lock["outer_fold"]:raise ValueError("Fold ordering changed")
        contexts.append({**spec,"choices":lock["choices"],"axis":data["axis"],
            "source":{"X":data["X_alpha"][spec["train_indices"]],"y":data["y"][spec["train_indices"]],
                      "record_ids":data["groups"][spec["train_indices"]]},
            "source_global_indices":spec["train_indices"],"data":data})
    return contexts


def _metric_rows(truth,predicted,classes):
    return classification_metrics(truth,predicted,classes)


def _save_transform_diagnostic(directory,inner_no,candidate,model,source_ids):
    """Keep actual inner-source statistics and MC coefficients for no-fit audit."""
    directory=Path(directory)/"transforms"/f"inner_{inner_no:02d}"
    directory.mkdir(parents=True,exist_ok=True)
    selector=model.bundle.selector
    coeff=np.ascontiguousarray(selector.coefficients_)
    coeff_key=hashlib.sha256(coeff.tobytes()).hexdigest()
    coefficient_file=directory/("coefficients_"+coeff_key+".npz")
    if not coefficient_file.exists():np.savez_compressed(coefficient_file,coefficients=coeff)
    transform_key=object_digest({"preprocessing":candidate["preprocessing"],"mc":candidate["mc"]})
    stats=directory/(transform_key+".npz");metadata=directory/(transform_key+".json")
    if not stats.exists():
        np.savez_compressed(stats,scaler_mean=model.bundle.scaler.mean_,scaler_var=model.bundle.scaler.var_,
            selected_indices=selector.selected_indices_,importance=selector.importances_,
            signed_stability=selector.signed_stability_,ranking=selector.ranking_)
        write_json(metadata,{"source_ids":np.asarray(source_ids).tolist(),**model.bundle.metadata,
            "coefficient_file":str(coefficient_file),"coefficient_sha256":digest(coefficient_file),
            "statistics_sha256":digest(stats)})
    return {"metadata":str(metadata),"metadata_sha256":digest(metadata),
            "statistics":str(stats),"statistics_sha256":digest(stats),
            "coefficients":str(coefficient_file),"coefficients_sha256":digest(coefficient_file)}


def _verify_transform_diagnostics(rows):
    seen=set()
    for row in rows:
        info=row["transform_diagnostic"]
        for name in ("metadata","statistics","coefficients"):
            pair=(info[name],info[name+"_sha256"])
            if pair not in seen:
                if digest(pair[0])!=pair[1]:raise ValueError("Saved source transform evidence changed")
                seen.add(pair)


def _inner_evaluate(data,spec,stage,candidates,output,seed,*,resume=False):
    directory=Path(output)/"selection"/f"outer_{spec['outer_fold']:02d}"/stage
    directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/"checkpoint.json"
    if resume and checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved["candidates"]!=candidates or digest(directory/"metrics.json")!=saved["metrics_sha256"] or digest(directory/"predictions.csv.gz")!=saved["predictions_sha256"]:
            raise ValueError("Inner checkpoint or candidate grid changed")
        rows=json.loads((directory/"metrics.json").read_text());_verify_transform_diagnostics(rows)
        return rows
    rows=[];classes=np.unique(data["y"][spec["train_indices"]])
    with gzip.open(directory/"predictions.csv.gz","wt",encoding="utf-8",newline="") as handle:
        fields=["outer_fold","stage","inner_fold","candidate_id","record_id","y_true","y_predicted","is_correct",*["score_"+str(c) for c in classes]]
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for inner_no,inner in enumerate(spec["inner_folds"]):
            train,valid=inner["train_indices"],inner["validation_indices"]
            if set(data["groups"][train])&set(data["groups"][valid]) or set(train)&set(spec["test_indices"]) or set(valid)&set(spec["test_indices"]):
                raise AssertionError("Sample crossed nested roles")
            cache={"__coefficient_budget__":max(c["mc"]["runs"] for c in candidates)}
            # Cache coefficient streams/transform bundles across all C/ratio settings.
            for candidate in candidates:
                before=time.monotonic()
                model=fit_source_candidate(candidate,data["X_alpha"][train],data["y"][train],data["axis"],seed,cache)
                diagnostic=_save_transform_diagnostic(directory,inner_no,candidate,model,data["groups"][train])
                pred=model.predict(data["X_alpha"][valid]);scores=model.predict_scores(data["X_alpha"][valid])
                metric=_metric_rows(data["y"][valid],pred,classes)
                row={"outer_fold":spec["outer_fold"],"stage":stage,"inner_fold":inner_no,
                    "candidate_id":candidate["id"],"candidate":candidate,"metrics":metric,
                    "source_ids":data["groups"][train].tolist(),"validation_ids":data["groups"][valid].tolist(),
                    "selected_indices":model.bundle.selector.selected_indices_.tolist(),
                    "effective_gamma":model.metadata["effective_gamma"],"auto_gamma":model.metadata["auto_gamma_numeric"],
                    "scale_gamma":model.metadata["scale_gamma_numeric"],"training_class_weights":model.metadata["training_class_weights"],
                    "vote_argmax_disagreements":int(np.count_nonzero(pred!=classes[np.argmax(scores,axis=1)])),
                    "transform_diagnostic":diagnostic,
                    "seconds":time.monotonic()-before}
                rows.append(row)
                for i,t,predicted,score in zip(data["groups"][valid],data["y"][valid],pred,scores):
                    writer.writerow({"outer_fold":spec["outer_fold"],"stage":stage,"inner_fold":inner_no,
                        "candidate_id":candidate["id"],"record_id":str(i),"y_true":str(t),"y_predicted":str(predicted),
                        "is_correct":int(t==predicted),**{"score_"+str(c):float(v) for c,v in zip(classes,score)}})
    write_json(directory/"metrics.json",rows)
    write_json(checkpoint,{"candidates":candidates,"metrics_sha256":digest(directory/"metrics.json"),
        "predictions_sha256":digest(directory/"predictions.csv.gz"),"outer_test_scored":False})
    return rows


def choose_candidate(rows,candidates,*,historical=False,tolerance=1e-12):
    groups=defaultdict(list)
    for row in rows:groups[row["candidate_id"]].append(row["metrics"])
    means={c["id"]:{m:float(np.mean([r[m] for r in groups[c["id"]]])) for m in METRICS} for c in candidates}
    if any(len(groups[c["id"]])!=2 for c in candidates):raise ValueError("Expected exactly two complete innerfolds")
    if historical:
        chosen=candidates[int(np.argmax([means[c["id"]]["macro_f1"] for c in candidates]))]
    else:
        maximum=max(v["macro_f1"] for v in means.values())
        near=[c for c in candidates if means[c["id"]]["macro_f1"]>=maximum-tolerance]
        chosen=min(near,key=lambda c:(-means[c["id"]]["balanced_accuracy"],-means[c["id"]]["accuracy"],
            c["mc"]["ratio"],c["mc"]["components"],c["mc"]["runs"],c["svm"]["C"],
            PREPROCESSORS.index(c["preprocessing"]),c["id"]))
    return {"selected":chosen,"mean_inner_metrics":means[chosen["id"]],"all_candidate_means":means,
            "selected_using_outer_test":False,"historical_first_tie_policy":historical}


def _manifest(input_path,protocol_path):
    root=Path(__file__).resolve().parents[2]
    sources=["src/herb_spectra/staged_nested_search.py","scripts/run_staged_nested_search.py",
        "src/herb_spectra/classification.py","src/herb_spectra/selection.py",
        "src/herb_spectra/spectral_models.py","src/herb_spectra/image_feature_models.py"]
    return {"input_sha256":digest(input_path),"protocol_sha256":digest(protocol_path),
        "source_sha256":{p:digest(root/p) for p in sources},"python":platform.python_version(),
        "versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-learn","joblib")}}


def verify_selection_checkpoint(output):
    """Read/check only; completed selection resume cannot silently accept edits."""
    output=Path(output);lock=json.loads((output/"selection_lock.json").read_text())
    if lock["run_manifest_sha256"]!=digest(output/"run_manifest.json"):
        raise ValueError("Selection manifest changed")
    collected=[]
    for fold in lock["outer_folds"]:
        for stage,choice in fold["choices"].items():
            directory=output/"selection"/f"outer_{fold['outer_fold']:02d}"/stage
            checkpoint=json.loads((directory/"checkpoint.json").read_text())
            for name,key in (("metrics.json","metrics_sha256"),("predictions.csv.gz","predictions_sha256")):
                if digest(directory/name)!=checkpoint[key]:raise ValueError("Saved inner evidence changed")
            rows=json.loads((directory/"metrics.json").read_text())
            _verify_transform_diagnostics(rows)
            if choose_candidate(rows,checkpoint["candidates"],historical=stage=="R0")!=choice:
                raise ValueError("Inner selection no longer matches locked choice")
            collected.extend(rows)
    if collected!=json.loads((output/"all_inner_metrics.json").read_text()) or len(collected)!=lock["inner_evaluations"]:
        raise ValueError("Combined inner metrics changed")
    return lock


def run_search(input_path,protocol_path,output,*,stage="selection",resume=False,release_path=None):
    if stage not in ("selection","evaluation"):raise ValueError("Unknown execution stage")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(protocol_path).read_text());manifest=_manifest(input_path,protocol_path)
    mp=output/"run_manifest.json"
    if mp.exists():
        if not resume or json.loads(mp.read_text())!=manifest:raise ValueError("Existing run differs or resume not requested")
    else:
        write_json(mp,manifest);(output/"protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
    with (output/"stage_events.jsonl").open("a",encoding="utf-8") as event:
        event.write(json.dumps({"event":"stage_start","stage":stage,"resume":resume,
            "utc":datetime.now(timezone.utc).isoformat(),"manifest_sha256":digest(mp)})+"\n")
    data,specs=prepare_protocol_roles(input_path,protocol_path,output)
    started=time.monotonic();lockpath=output/"selection_lock.json"
    if stage=="selection":
        if resume and lockpath.exists():
            verify_selection_checkpoint(output)
            return {"status":"selection_already_completed","selection_lock_sha256":digest(lockpath)}
        all_rows=[];foldlocks=[]
        for spec in specs:
            stages={};choices={}
            candidates=historical_grid()
            stages["R0"]=_inner_evaluate(data,spec,"R0",candidates,output,protocol["model_seed"],resume=resume)
            choices["R0"]=choose_candidate(stages["R0"],candidates,historical=True)
            candidates=[{"id":m["id"],"preprocessing":"snv","mc":{k:m[k] for k in ("runs","components","ratio")},
                "svm":{"C":10.,"gamma":"scale","class_weight":"balanced"}} for m in mc_grid()]
            stages["R1"]=_inner_evaluate(data,spec,"R1",candidates,output,protocol["model_seed"],resume=resume)
            choices["R1"]=choose_candidate(stages["R1"],candidates)
            mc=choices["R1"]["selected"]["mc"]
            candidates=[{"id":s["id"],"preprocessing":"snv","mc":mc,
                "svm":{k:s[k] for k in ("C","gamma","class_weight")}} for s in svm_grid()]
            stages["R2"]=_inner_evaluate(data,spec,"R2",candidates,output,protocol["model_seed"],resume=resume)
            choices["R2"]=choose_candidate(stages["R2"],candidates)
            svm=choices["R2"]["selected"]["svm"]
            candidates=[{"id":f"prep{i:02d}_{prep}","preprocessing":prep,"mc":mc,"svm":svm} for i,prep in enumerate(PREPROCESSORS)]
            stages["R3"]=_inner_evaluate(data,spec,"R3",candidates,output,protocol["model_seed"],resume=resume)
            choices["R3"]=choose_candidate(stages["R3"],candidates)
            for rows in stages.values():all_rows.extend(rows)
            foldlocks.append({"outer_fold":spec["outer_fold"],"choices":choices})
            write_json(output/"status.json",{"status":"selection","completed_outer_folds":spec["outer_fold"]+1,
                "inner_evaluations":len(all_rows),"outer_test_scored":False})
            print(json.dumps({"stage":"core_selection","outer_fold":spec["outer_fold"],
                "inner_evaluations":len(all_rows),"R1":choices["R1"]["selected"]["mc"],
                "R2":svm,"R3":choices["R3"]["selected"]["preprocessing"]}),flush=True)
        write_json(output/"all_inner_metrics.json",all_rows)
        write_json(lockpath,{"outer_folds":foldlocks,"outer_scores_used_to_select":False,
            "inner_evaluations":len(all_rows),"run_manifest_sha256":digest(mp)})
        write_json(output/"status.json",{"status":"selection_completed","seconds":time.monotonic()-started,
            "inner_evaluations":len(all_rows),"selection_lock_sha256":digest(lockpath),"outer_test_scored":False})
        with (output/"stage_events.jsonl").open("a",encoding="utf-8") as event:
            event.write(json.dumps({"event":"selection_lock_written","utc":datetime.now(timezone.utc).isoformat(),
                "selection_lock_sha256":digest(lockpath)})+"\n")
        return json.loads((output/"status.json").read_text())
    if not lockpath.exists():raise ValueError("Complete inner selection first")
    verify_selection_checkpoint(output)
    if release_path is None:raise ValueError("Global selection-lock release required")
    release=Path(release_path);gate=json.loads(release.read_text())
    matches=[r for r in gate["rounds"] if Path(r["selection_lock"]).resolve()==lockpath.resolve()]
    if len(matches)!=1:raise ValueError("Core lock missing in global release")
    rec=matches[0]
    if digest(lockpath)!=rec["selection_lock_sha256"] or digest(mp)!=rec["run_manifest_sha256"] or digest(protocol_path)!=rec["protocol_sha256"]:
        raise ValueError("Released core hashes changed")
    (output/"release_snapshot.json").write_bytes(release.read_bytes())
    if resume and (output/"summary.json").exists():
        status=json.loads((output/"status.json").read_text())
        for rel,sha in status["result_sha256"].items():
            if digest(output/rel)!=sha:raise ValueError("Completed result changed")
        return json.loads((output/"summary.json").read_text())
    lock=json.loads(lockpath.read_text());rows=[];all_predictions=[];classes=np.unique(data["y"])
    for spec,flock in zip(specs,lock["outer_folds"]):
        roles={name:choice["selected"] for name,choice in flock["choices"].items()}
        for i,prep in enumerate(PREPROCESSORS):
            roles["R3_diagnostic_"+prep]={**flock["choices"]["R3"]["selected"],"id":f"prep{i:02d}_{prep}","preprocessing":prep}
        roles["fixed_snv_C10_MC20"]={"id":"fixed_snv_C10_MC20","preprocessing":"snv", "mc":{"runs":20,"components":2,"ratio":.1},"svm":{"C":10.,"gamma":"scale","class_weight":"balanced"}}
        directory=output/"evaluation"/f"outer_{spec['outer_fold']:02d}";directory.mkdir(parents=True,exist_ok=True)
        train,test=spec["train_indices"],spec["test_indices"];cache={};fitted={}
        for method,config in roles.items():
            key=object_digest({k:v for k,v in config.items() if k!="id"})
            if key not in fitted:
                model=fit_source_candidate(config,data["X_alpha"][train],data["y"][train],data["axis"],protocol["model_seed"],cache)
                artifact=directory/(key+".joblib");joblib.dump(model,artifact,compress=3)
                loaded=joblib.load(artifact)
                if not np.array_equal(loaded.predict(data["X_alpha"][train]),model.predict(data["X_alpha"][train])):
                    raise AssertionError("Saved source model changed training predictions")
                fitted[key]=(loaded,artifact)
            model,artifact=fitted[key]
            predicted=model.predict(data["X_alpha"][test]);scores=model.predict_scores(data["X_alpha"][test])
            row={"method":method,"outer_fold":spec["outer_fold"],"config":config,
                **_metric_rows(data["y"][test],predicted,classes),"selected_features":model.metadata["selected_features"],
                "selected_indices":model.bundle.selector.selected_indices_.tolist(),"selected_frequencies":model.metadata["selected_frequencies"],
                "effective_gamma":model.metadata["effective_gamma"],"model_path":str(artifact),"model_sha256":digest(artifact),
                "vote_argmax_disagreements":int(np.count_nonzero(predicted!=classes[np.argmax(scores,axis=1)]))}
            rows.append(row)
            for identifier,t,p,score in zip(data["groups"][test],data["y"][test],predicted,scores):
                all_predictions.append({"method":method,"outer_fold":spec["outer_fold"],"record_id":str(identifier),
                    "y_true":str(t),"y_predicted":str(p),"is_correct":int(t==p),
                    **{"score_"+str(c):float(v) for c,v in zip(classes,score)}})
        print(json.dumps({"stage":"core_outer_evaluation","outer_fold":spec["outer_fold"],"method_roles":len(roles)}),flush=True)
    p=output/"outer_predictions.csv.gz"
    with gzip.open(p,"wt",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_predictions[0]),lineterminator="\n");writer.writeheader();writer.writerows(all_predictions)
    summary=[]
    for method in sorted(set(r["method"] for r in all_predictions)):
        values=[r for r in all_predictions if r["method"]==method]
        if len(values)!=855 or len(set(r["record_id"] for r in values))!=855:raise AssertionError("Eachmethod must OOF every specimen exactlyonce")
        truth=np.asarray([r["y_true"] for r in values]);pred=np.asarray([r["y_predicted"] for r in values])
        summary.append({"method":method,"scope":"855sample 5outer OOF retrospective withinbatch",
            **_metric_rows(truth,pred,classes),"fold_macro_f1_sd":float(np.std([r["macro_f1"] for r in rows if r["method"]==method],ddof=1)),
            "mean_selected_features":float(np.mean([r["selected_features"] for r in rows if r["method"]==method]))})
    write_json(output/"outer_metrics.json",rows);write_json(output/"summary.json",summary)
    write_json(output/"status.json",{"status":"completed","evaluation_seconds":time.monotonic()-started,
        "outer_method_fold_cases":len(rows),"outer_prediction_rows":len(all_predictions),
        "selection_lock_sha256":digest(lockpath),"outer_scores_used_to_select":False,
        "result_sha256":{name:digest(output/name) for name in ("outer_metrics.json","outer_predictions.csv.gz","summary.json")}})
    with (output/"stage_events.jsonl").open("a",encoding="utf-8") as event:
        event.write(json.dumps({"event":"outer_evaluation_complete","utc":datetime.now(timezone.utc).isoformat(),
            "selection_lock_sha256":digest(lockpath)})+"\n")
    return summary
