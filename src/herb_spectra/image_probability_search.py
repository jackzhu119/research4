"""Image-guided source-OOF probability and multiclass decision experiments."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import torch

from .calibration_exploration import development_episodes, encoded, fast_metrics
from .image_probability import (PROVIDERS, apply_decision, candidate_grid, fit_decision,
    fit_temperature, probability_metrics, singleton_anchor_folds, supervised_continuation)
from .neural_base_search import (digest, object_digest, write_json, fit_preprocessor,
    transform_features, fit_model, model_scores, load_locked_contexts)
from .paper_fewshot_classification import state_fingerprint
from .spectral_models import fit_source_candidate
from .spectral_source_search import load_source_model, source_metadata_path, source_model_path
from .within_batch_holdout import prepare_holdout

METRICS=("accuracy","balanced_accuracy","macro_f1","nll","brier","ece10")
SVM_CANDIDATE={"id":"s010_snv_svc","preprocessing":"snv","model":"svc","C":10.}
PREDICTION_FIELDS=["episode","seed","candidate_id","provider","kind","record_id","y_true","y_predicted",
    "is_correct","probability_argmax","confidence",*[f"p{i}" for i in range(7)]]


def _event(output,event,stage,**fields):
    with (Path(output)/"stage_events.jsonl").open("a",encoding="utf-8") as handle:
        handle.write(json.dumps({"event":event,"stage":stage,"utc":datetime.now(timezone.utc).isoformat(),**fields})+"\n")


def _read_evaluation(directory,context):
    directory=Path(directory);saved=json.loads((directory/"evaluation.json").read_text())
    for name,key in [("metrics.json","metrics_sha256"),("predictions.csv.gz","predictions_sha256"),("decision_fits.json","decision_fits_sha256")]:
        if digest(directory/name)!=saved[key]:raise ValueError("Probability evaluation cache changed")
    if saved["source_context_metadata_sha256"]!=context["lineage"]["context_metadata_sha256"]:
        raise ValueError("Probability evaluation context changed")
    return json.loads((directory/"metrics.json").read_text())


def _roles(input_path,plan_path,protocol):
    with np.load(input_path,allow_pickle=False) as handle:arrays={k:handle[k] for k in handle.files}
    roles,truth,preparation=prepare_holdout(arrays,json.loads(Path(plan_path).read_text()),frequency_min=.1,frequency_max=1.5)
    if preparation["feature_count"]!=612 or str(np.asarray(arrays["dataset_version"]))!=protocol["dataset_version"]:
        raise ValueError("Dataset version or fixed frequency band differs")
    mask=(arrays["axis"]>=.1)&(arrays["axis"]<=1.5)
    return roles,truth,np.asarray(arrays["axis"])[mask],preparation


def _subset(training,indices):
    return {"X_alpha":training["X_alpha"][indices],"y":training["y"][indices],"record_ids":training["record_ids"][indices]}


def _source_artifacts(raw,base_context,episode,protocol):
    classes=base_context["classes"];seed=base_context["seed"]
    if raw["record_ids"].tolist()!=base_context["source"]["record_ids"].tolist():
        raise ValueError("Cached MLP source physical IDs differ")
    source_run=Path(protocol["source_runs"]["svm"])
    if episode is None:
        svm_path=Path(protocol["source_runs"]["svm_final"])
        meta=json.loads(Path(protocol["source_runs"]["svm_final_metadata"]).read_text())
        if digest(svm_path)!=meta["source_model_sha256"] or meta["source_ids"]!=raw["record_ids"].tolist():
            raise ValueError("Final cached SVM source identity changed")
        svm=joblib.load(svm_path)
    else:
        svm_path=source_model_path(source_run,"s010_snv_svc",episode)
        meta=json.loads(source_metadata_path(source_run,"s010_snv_svc",episode).read_text())
        if meta["partition"]["source"]!=raw["record_ids"].tolist():raise ValueError("Cached SVM source IDs differ")
        svm=load_source_model(source_run,"s010_snv_svc",episode)
    if not np.array_equal(svm.classes_,classes):raise ValueError("SVM score class order differs")
    base=base_context["model"]
    meta_root=Path(protocol["source_runs"]["supervised_continuation"])
    directory=(meta_root/"final"/f"seed_{seed}" if episode is None else
               meta_root/"development"/f"episode_{episode:02d}"/f"seed_{seed}")/"m000"
    metadata=json.loads((directory/"training.json").read_text());path=directory/"supervised.pt"
    if (metadata["expected"]["source_ids"]!=raw["record_ids"].tolist()
            or metadata["expected"]["initial_fingerprint"]!=base_context["fingerprint"]
            or digest(path)!=metadata["artifacts"]["supervised"]["sha256"]):
        raise ValueError("Cached supervised continuation lineage changed")
    continued=deepcopy(base);continued.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));continued.eval()
    if state_fingerprint(continued)!=metadata["artifacts"]["supervised"]["fingerprint"]:
        raise ValueError("Cached supervised continuation state changed")
    return {"svm":svm,"mlp":base,"mlp_sup100":continued},{
        "svm":{"path":str(svm_path),"sha256":digest(svm_path)},
        "mlp":{"path":str(Path(base_context["directory"])/"model.pt"),"sha256":digest(Path(base_context["directory"])/"model.pt")},
        "mlp_sup100":{"path":str(path),"sha256":digest(path)}}


def _fingerprints(models):
    return {p:joblib.hash(m) if p=="svm" else state_fingerprint(m) for p,m in models.items()}


def build_context(raw_source,raw_prediction,axis,base_context,episode,protocol,directory):
    """Fit source-only OOF models; original outer models remain unchanged."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    classes=np.asarray(base_context["classes"]);source_y=encoded(raw_source["y"],classes)
    seed=base_context["seed"];episode_number=6 if episode is None else episode
    models,artifacts=_source_artifacts(raw_source,base_context,episode,protocol);before=_fingerprints(models)
    source_scores={};query_scores={}
    source_nn=base_context["source"]["X"];query_nn=transform_features(base_context["preprocessor"],raw_prediction["X_alpha"])
    for provider,model in models.items():
        source_scores[provider]=(model.predict_scores(raw_source["X_alpha"]) if provider=="svm" else model_scores(model,source_nn))
        query_scores[provider]=(model.predict_scores(raw_prediction["X_alpha"]) if provider=="svm" else model_scores(model,query_nn))
    folds,policy=singleton_anchor_folds(source_y,raw_source["record_ids"],folds=3,
                                       seed=protocol["oof"]["split_seed"]+100*episode_number)
    oof={p:np.full((len(source_y),7),np.nan) for p in PROVIDERS};fold_records=[]
    for fold in folds:
        train,valid=fold["train"],fold["validation"];f=fold["fold"]
        case=directory/"oof"/f"fold_{f:02d}";case.mkdir(parents=True,exist_ok=True)
        svm=fit_source_candidate(SVM_CANDIDATE,raw_source["X_alpha"][train],raw_source["y"][train],axis,seed=seed)
        preprocessor=fit_preprocessor(raw_source["X_alpha"][train],base_context["candidate"])
        x=transform_features(preprocessor,raw_source["X_alpha"][train]);q=transform_features(preprocessor,raw_source["X_alpha"][valid])
        model,losses=fit_model(x,source_y[train],base_context["candidate"],seed=seed)
        continuation_seed=seed+1000+100*episode_number+17*f
        continued,continuation_record=supervised_continuation(model,x,source_y[train],raw_source["record_ids"][train],continuation_seed)
        oof["svm"][valid]=svm.predict_scores(raw_source["X_alpha"][valid])
        oof["mlp"][valid]=model_scores(model,q);oof["mlp_sup100"][valid]=model_scores(continued,q)
        joblib.dump(svm,case/"svm.joblib",compress=3);joblib.dump(preprocessor,case/"preprocessor.joblib",compress=3)
        torch.save(model.state_dict(),case/"mlp.pt");torch.save(continued.state_dict(),case/"mlp_sup100.pt")
        record={"fold":f,"source_train_ids":raw_source["record_ids"][train].tolist(),
            "source_validation_ids":raw_source["record_ids"][valid].tolist(),"source_fit_rows":len(train),"oof_validation_rows":len(valid),
            "base_fit_seed":seed,"continuation_seed":continuation_seed,
            "preprocessor_fit_source_rows":preprocessor["source_samples"],"source_losses":losses,
            "supervised_continuation":continuation_record,"fingerprints":{
                "svm":joblib.hash(svm),"mlp":state_fingerprint(model),"mlp_sup100":state_fingerprint(continued)},
            "artifacts":{name:digest(case/name) for name in ("svm.joblib","preprocessor.joblib","mlp.pt","mlp_sup100.pt")}}
        write_json(case/"training.json",record);fold_records.append(record)
    eligible=np.asarray(sorted(set(range(len(source_y)))-set(policy["excluded_oof_indices"])),dtype=int)
    temperatures={};temperature_records={}
    for provider in PROVIDERS:
        if not np.isfinite(oof[provider][eligible]).all():raise FloatingPointError("Source OOF scores incomplete")
        temperatures[provider],temperature_records[provider]=fit_temperature(oof[provider][eligible],source_y[eligible],protocol["temperature"]["regularization"])
    payload={"source_ids":raw_source["record_ids"],"source_y":source_y,"prediction_ids":raw_prediction["record_ids"],
             "oof_ids":raw_source["record_ids"][eligible],"oof_y":source_y[eligible],"classes":classes}
    if "y" in raw_prediction:payload["prediction_y"]=encoded(raw_prediction["y"],classes)
    for provider in PROVIDERS:
        payload["source_scores_"+provider]=source_scores[provider];payload["prediction_scores_"+provider]=query_scores[provider]
        payload["oof_scores_"+provider]=oof[provider][eligible]
    np.savez_compressed(directory/"scores.npz",**payload)
    if _fingerprints(models)!=before:raise AssertionError("Original source models changed while collecting probabilities")
    record={"episode":episode,"seed":seed,"classes":classes.tolist(),"source_ids":raw_source["record_ids"].tolist(),
        "prediction_ids":raw_prediction["record_ids"].tolist(),"source_artifacts":artifacts,
        "source_fingerprints":before,"source_fingerprints_after":_fingerprints(models),"temperatures":temperatures,
        "temperature_fits":temperature_records,"oof_policy":policy,"folds":fold_records,
        "scores_npz_sha256":digest(directory/"scores.npz"),"target_support_labels_used":0,
        "prediction_labels_used_for_fit":False,"query_statistics_fitted":False,"new_oof_model_optimizations":9,
        "source_feature_preprocessors_refitted_within_each_oof_training_fold":True}
    write_json(directory/"context.json",record)
    return record


def _load_context(directory):
    directory=Path(directory);meta=json.loads((directory/"context.json").read_text())
    if digest(directory/"scores.npz")!=meta["scores_npz_sha256"]:raise ValueError("Score cache bytes changed")
    for fold in meta["folds"]:
        for name,expected in fold["artifacts"].items():
            if digest(directory/"oof"/f"fold_{fold['fold']:02d}"/name)!=expected:
                raise ValueError("Source OOF weight or preprocessing bytes changed")
    for artifact in meta["source_artifacts"].values():
        if digest(artifact["path"])!=artifact["sha256"]:raise ValueError("Outer source weight bytes changed")
    with np.load(directory/"scores.npz",allow_pickle=False) as npz:values={k:npz[k] for k in npz.files}
    prediction={"record_ids":values["prediction_ids"],"scores_by_provider":{p:values["prediction_scores_"+p] for p in PROVIDERS}}
    if "prediction_y" in values:prediction["y"]=values["prediction_y"]
    return {"episode":meta["episode"],"seed":meta["seed"],"classes":values["classes"],
        "source_ids":values["source_ids"],"source_y":values["source_y"],
        "source_scores_by_provider":{p:values["source_scores_"+p] for p in PROVIDERS},"prediction":prediction,
        "oof":{"record_ids":values["oof_ids"],"y":values["oof_y"],
            "scores_by_provider":{p:values["oof_scores_"+p] for p in PROVIDERS},"folds":meta["folds"],"policy":meta["oof_policy"]},
        "temperatures":meta["temperatures"],"source_fingerprints":meta["source_fingerprints"],
        "lineage":{"source_artifacts":meta["source_artifacts"],"cache_npz_sha256":meta["scores_npz_sha256"],
            "context_metadata_sha256":digest(directory/"context.json"),"directory":str(directory)},"target_support_labels_used":0}


def get_score_contexts(input_path,plan_path,run,scope="development"):
    """Read source-OOF caches. Final prediction labels are returned separately."""
    run=Path(run);manifest=json.loads((run/"run_manifest.json").read_text())
    root=Path(__file__).resolve().parents[2]
    for path,expected in manifest.get("source_sha256",{}).items():
        if digest(root/path)!=expected:raise ValueError("Probability cache source code changed")
    if digest(input_path)!=manifest["input_sha256"] or digest(plan_path)!=manifest["plan_sha256"]:
        raise ValueError("Input or split plan differs from probability source contexts")
    if digest(run/"protocol_snapshot.json")!=manifest["protocol_sha256"]:
        raise ValueError("Probability protocol snapshot changed")
    if scope not in {"development","final"}:raise ValueError("Unknown score scope")
    index=json.loads((run/(scope+"_contexts.json")).read_text())
    contexts=[_load_context(run/path) for path in index]
    if scope=="final" and any("y" in c["prediction"] for c in contexts):raise AssertionError("Final score context leaked labels")
    if scope=="final":
        protocol=json.loads((run/"protocol_snapshot.json").read_text());_,truth,_,_=_roles(input_path,plan_path,protocol)
    else:truth={}
    return contexts,truth


def evaluate_context(context,truth,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    classes=context["classes"];query=context["prediction"]
    y=query["y"] if "y" in query else encoded([truth[str(i)] for i in query["record_ids"]],classes)
    metrics=[];fits=[]
    with gzip.open(output/"predictions.csv.gz","wt",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=PREDICTION_FIELDS,lineterminator="\n");writer.writeheader()
        for candidate in candidate_grid():
            provider=candidate["provider"]
            fitted=fit_decision(candidate,context["oof"]["scores_by_provider"][provider],context["oof"]["y"],
                                context["source_y"],context["temperatures"][provider]);fits.append(fitted)
            pred,probability=apply_decision(fitted,query["scores_by_provider"][provider])
            value={"episode":context["episode"],"seed":context["seed"],"candidate_id":candidate["id"],
                "provider":provider,"kind":candidate["kind"],"stage":"final" if context["episode"] is None else
                "selection" if context["episode"]<4 else "gate",**fast_metrics(y,pred,7,detailed=True),
                **probability_metrics(probability,y),"target_support_labels_used":0,
                "proper_probability_metrics_exclude_decision_bias":True}
            metrics.append(value)
            for n,(identifier,actual,predicted) in enumerate(zip(query["record_ids"],y,pred)):
                writer.writerow({k:value[k] for k in ("episode","seed","candidate_id","provider","kind")} | {
                    "record_id":str(identifier),"y_true":str(classes[actual]),"y_predicted":str(classes[predicted]),
                    "is_correct":int(actual==predicted),"probability_argmax":str(classes[probability[n].argmax()]),
                    "confidence":float(probability[n].max()),**{f"p{c}":float(probability[n,c]) for c in range(7)}})
    write_json(output/"metrics.json",metrics);write_json(output/"decision_fits.json",fits)
    write_json(output/"evaluation.json",{"metrics_sha256":digest(output/"metrics.json"),
        "predictions_sha256":digest(output/"predictions.csv.gz"),"decision_fits_sha256":digest(output/"decision_fits.json"),
        "source_context_metadata_sha256":context["lineage"]["context_metadata_sha256"]})
    return metrics


def select_methods(rows,policy):
    grouped=defaultdict(list)
    for row in rows:grouped[(row["candidate_id"],row["stage"])].append(row)
    means={candidate["id"]:{stage:{k:float(np.mean([r[k] for r in grouped[(candidate["id"],stage)]])) for k in METRICS}
                            for stage in ("selection","gate")} for candidate in candidate_grid()}
    choices={}
    for provider in PROVIDERS:
        candidates=[c for c in candidate_grid() if c["provider"]==provider];raw=next(c for c in candidates if c["kind"]=="raw")
        temp=next(c for c in candidates if c["kind"]=="temperature")
        def passes(c,stage):
            a,b=means[c["id"]][stage],means[raw["id"]][stage]
            return (a["macro_f1"]-b["macro_f1"]>policy["minimum_macro_f1_gain"] and
                a["accuracy"]-b["accuracy"]>=-policy["maximum_accuracy_loss"] and
                a["balanced_accuracy"]-b["balanced_accuracy"]>=-policy["maximum_balanced_accuracy_loss"])
        ranked=sorted(candidates,key=lambda c:(-means[c["id"]]["selection"]["macro_f1"],
            -means[c["id"]]["selection"]["accuracy"],0 if c["kind"]=="raw" else 1,c["id"]))
        eligible=[c for c in ranked if passes(c,"selection")];proposed=eligible[0] if eligible else raw
        selected=proposed if eligible and passes(proposed,"gate") else raw
        def probability_pass(stage):
            a,b=means[temp["id"]][stage],means[raw["id"]][stage]
            return b["nll"]-a["nll"]>policy["minimum_nll_gain"] and a["brier"]-b["brier"]<=policy["maximum_brier_loss"]
        choices[provider]={"raw":raw,"selected":selected,"proposed":proposed,"diagnostic":ranked[0],
            "probability_selected":temp if probability_pass("selection") and probability_pass("gate") else raw,
            "eligible_classification_candidates":len(eligible),"fallback_to_raw":selected["kind"]=="raw",
            "selected_using_final_query":False}
    return {"choices":choices,"means":means,"source_only_development":True,"development_evaluations":len(rows)}


def manifest(input_path,plan_path,protocol_path,protocol):
    root=Path(__file__).resolve().parents[2]
    code=["src/herb_spectra/image_probability.py","src/herb_spectra/image_probability_search.py","scripts/run_image_probability_search.py",
          "src/herb_spectra/neural_base_search.py","src/herb_spectra/spectral_models.py","src/herb_spectra/spectral_source_search.py",
          "src/herb_spectra/paper_fewshot_classification.py","src/herb_spectra/calibration_exploration.py","src/herb_spectra/within_batch_holdout.py"]
    return {"input_sha256":digest(input_path),"plan_sha256":digest(plan_path),"protocol_sha256":digest(protocol_path),
        "candidate_grid_sha256":object_digest(candidate_grid()),"source_sha256":{p:digest(root/p) for p in code},
        "source_run_locks":{str(Path(protocol["source_runs"][name])/lock):digest(Path(protocol["source_runs"][name])/lock)
                            for name,lock in [("mlp","selection_lock.json"),("supervised_continuation","selection_lock.json"),("svm","source_selection_lock.json")]},
        "python":platform.python_version(),"versions":{n:importlib.metadata.version(n) for n in ("numpy","scipy","torch","scikit-learn","joblib")}}


def run_search(input_path,plan_path,protocol_path,output,stage="development",resume=False):
    torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(protocol_path).read_text());provenance=manifest(input_path,plan_path,protocol_path,protocol)
    if object_digest(candidate_grid())!=protocol["candidate_grid_sha256"]:raise ValueError("Candidate grid changed")
    if (output/"run_manifest.json").exists():
        if not resume or json.loads((output/"run_manifest.json").read_text())!=provenance:raise ValueError("Probability provenance differs; use a fresh output")
    else:
        write_json(output/"run_manifest.json",provenance);(output/"protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
    if stage not in {"development","final"}:raise ValueError("Unknown stage")
    _event(output,"stage_start",stage,resume=resume,run_manifest_sha256=digest(output/"run_manifest.json"))
    if stage=="final":
        gate=output.parent/"all_development_locks.json"
        if not gate.exists():raise ValueError("Three-track global release absent")
        if not (output/"selection_lock.json").exists():raise ValueError("Development selection absent")
        release=json.loads(gate.read_text());records=[r for r in release["rounds"] if r["name"]=="probability"]
        if len(records)!=1:raise ValueError("Global release lacks unique probability track")
        record=records[0]
        if (digest(output/"selection_lock.json")!=record["selection_lock_sha256"]
                or digest(protocol_path)!=record["protocol_sha256"]
                or digest(output/"run_manifest.json")!=record["run_manifest_sha256"]
                or Path(record["selection_lock"]).resolve()!=(output/"selection_lock.json").resolve()):
            raise ValueError("Probability development/protocol/provenance differs from global release")
        (output/"final_gate_snapshot.json").write_bytes(gate.read_bytes())
    if resume and (output/("development_completed.json" if stage=="development" else "final_completed.json")).exists():
        if stage=="development":
            contexts,_=get_score_contexts(input_path,plan_path,output,"development")
            rows=json.loads((output/"all_development_metrics.json").read_text())
            verified=[]
            for context in contexts:verified.extend(_read_evaluation(context["lineage"]["directory"],context))
            if verified!=rows:raise ValueError("Development aggregate differs from cached cases")
            if select_methods(rows,protocol["selection"])!=json.loads((output/"selection_lock.json").read_text()):raise ValueError("Development choice changed")
            _event(output,"resume_completed",stage,new_fits=0)
            return {"status":"development_already_completed","contexts":len(contexts)}
        contexts,_=get_score_contexts(input_path,plan_path,output,"final");verified=[]
        for context in contexts:verified.extend(_read_evaluation(context["lineage"]["directory"],context))
        if verified!=json.loads((output/"final_metrics.json").read_text()):raise ValueError("Final aggregate differs from cached cases")
        status=json.loads((output/"final_completed.json").read_text())
        for name,expected in status["result_sha256"].items():
            if digest(output/name)!=expected:raise ValueError("Final summary/metrics bytes changed")
        _event(output,"resume_completed",stage,new_fits=0)
        return json.loads((output/"summary.json").read_text())
    started=time.monotonic();roles,truth,axis,preparation=_roles(input_path,plan_path,protocol)
    training=roles["training"];base_contexts,_=load_locked_contexts(input_path,plan_path,protocol["source_runs"]["mlp"],scope=stage)
    if stage=="development":
        parts=development_episodes(training,6,20261103)
        specs=[(i,_subset(training,p["source"]),_subset(training,p["query"])) for i,p in enumerate(parts)]
    else:specs=[(None,training,roles["prediction"])]
    rows=[];index=[]
    for episode,source,prediction in specs:
        for seed in protocol["seeds"]:
            context_base=next(c for c in base_contexts if c["episode"]==episode and c["seed"]==seed)
            relative=Path(stage)/("source" if episode is None else f"episode_{episode:02d}")/f"seed_{seed}"
            directory=output/relative
            if resume and (directory/"context.json").exists():context=_load_context(directory)
            else:
                build_context(source,prediction,axis,context_base,episode,protocol,directory)
                context=_load_context(directory)
            if stage=="final" and "y" in context["prediction"]:raise AssertionError("Prediction label leak")
            evaluation=directory/"evaluation.json"
            if resume and evaluation.exists():
                values=_read_evaluation(directory,context)
            else:values=evaluate_context(context,truth,directory)
            rows.extend(values);index.append(str(relative))
            write_json(output/"status.json",{"status":stage,"contexts_completed":len(index),"evaluations":len(rows)})
            print(json.dumps({"stage":"probability_"+stage,"contexts":len(index),"evaluations":len(rows)}),flush=True)
    write_json(output/(stage+"_contexts.json"),index)
    if stage=="development":
        write_json(output/"all_development_metrics.json",rows);choice=select_methods(rows,protocol["selection"])
        write_json(output/"selection_lock.json",choice)
        status={"status":"development_completed","seconds":time.monotonic()-started,"development_evaluations":len(rows),
            "development_prediction_rows":sum(r["total"] for r in rows),"contexts":len(index),"oof_optimizations":9*len(index),
            "selection_lock_sha256":digest(output/"selection_lock.json"),"final_query_scored":False,"target_support_labels_used":0}
        write_json(output/"development_completed.json",status);write_json(output/"status.json",status)
        _event(output,"stage_completed",stage,evaluations=len(rows),selection_lock_sha256=status["selection_lock_sha256"])
        return status
    lock=json.loads((output/"selection_lock.json").read_text());groups=defaultdict(list)
    for row in rows:groups[row["candidate_id"]].append(row)
    summaries=[]
    for candidate in candidate_grid():
        values=groups[candidate["id"]]
        summaries.append({**candidate,"evaluations":len(values),"seeds":protocol["seeds"],
            **{k+"_mean":float(np.mean([v[k] for v in values])) for k in METRICS},
            "classification_selected":lock["choices"][candidate["provider"]]["selected"]["id"]==candidate["id"],
            "probability_selected":lock["choices"][candidate["provider"]]["probability_selected"]["id"]==candidate["id"],"target_support_labels_used":0})
    write_json(output/"final_metrics.json",rows);write_json(output/"summary.json",summaries)
    status={"status":"completed","seconds":time.monotonic()-started,"final_evaluations":len(rows),
        "final_prediction_rows":sum(r["total"] for r in rows),"contexts":len(index),"new_oof_optimizations":9*len(index),
        "selection_lock_sha256":digest(output/"selection_lock.json"),"target_support_labels_used":0,
        "same_batch_retrospective":True,"query_labels_used_for_fit":False,
        "result_sha256":{name:digest(output/name) for name in ("final_metrics.json","summary.json","final_contexts.json")}}
    write_json(output/"final_completed.json",status);write_json(output/"status.json",status)
    _event(output,"stage_completed",stage,evaluations=len(rows),selection_lock_sha256=status["selection_lock_sha256"])
    return summaries
