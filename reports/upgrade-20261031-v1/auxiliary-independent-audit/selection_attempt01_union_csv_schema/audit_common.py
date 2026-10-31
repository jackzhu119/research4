"""Independent auxiliary arithmetic/state/provenance checks; never calls fit."""
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
RUN = ROOT/"outputs/upgrade-20261031-v1/auxiliary/v3"
DEST = Path(__file__).resolve().parent
PROTOCOL = ROOT/"data/research/auxiliary_multiplant_protocol_20261031.json"
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def load(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def array_sha(value):
    value = np.asarray(value)
    data = json.dumps(value.tolist(),sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode() if value.dtype.kind=="O" else np.ascontiguousarray(value).tobytes()
    out = hashlib.sha256(json.dumps({"shape":value.shape,"dtype":str(value.dtype)},sort_keys=True,ensure_ascii=False,separators=(",",":")).encode())
    out.update(data);return out.hexdigest()


def source_sha(X,y=None):return object_sha({"X":array_sha(X),"y":None if y is None else array_sha(y)})


def npz(path):
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k] for k in archive.files}


def predictions(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as handle:return list(csv.DictReader(handle))


def truth(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle))
    if len(rows)!=len({r["record_id"] for r in rows}):raise AssertionError("Repeated truth specimen")
    return {r["record_id"]:r["label"] for r in rows}


def metrics(rows,mapping,classes):
    if len(rows)!=len({r["record_id"] for r in rows}) or {r["record_id"] for r in rows}!=set(mapping):raise AssertionError("Incorrect denominator")
    matrix=np.zeros((len(classes),len(classes)),dtype=int)
    for row in rows:matrix[classes.index(mapping[row["record_id"]]),classes.index(row["y_predicted"])] += 1
    tp=matrix.diagonal();actual=matrix.sum(1);found=matrix.sum(0)
    precision=np.divide(tp,found,out=np.zeros(len(classes)),where=found>0)
    recall=np.divide(tp,actual,out=np.zeros(len(classes)),where=actual>0)
    f1=np.divide(2*tp,actual+found,out=np.zeros(len(classes)),where=actual+found>0)
    return {"accuracy":float(tp.sum()/matrix.sum()),"macro_f1":float(f1.mean()),"balanced_accuracy":float(recall[actual>0].mean()),
        "correct":int(tp.sum()),"total":int(matrix.sum()),"confusion_matrix":matrix.tolist(),
        "per_class":{c:{"precision":float(p),"recall":float(r),"f1":float(f),"support":int(n)} for c,p,r,f,n in zip(classes,precision,recall,f1,actual,strict=True)}}


class Auditor:
    def __init__(self):
        self.checks=0;self.max_difference=0.;self.source_cache={};self.preprocessed_cache={};self.initial_cache={};self.fit_expectations=[]
        self.models_replayed=0;self.svm_replayed=0;self.neural_replayed=0;self.mc_selectors_recomputed=0

    def require(self,condition,detail):
        self.checks += 1
        if not condition:raise AssertionError(detail)

    def close(self,actual,expected,detail,atol=1e-10,rtol=1e-9):
        actual,expected=np.asarray(actual),np.asarray(expected)
        self.require(actual.shape==expected.shape,detail+" shape")
        difference=float(np.max(np.abs(actual-expected))) if actual.size else 0.
        self.max_difference=max(self.max_difference,difference)
        self.require(np.allclose(actual,expected,atol=atol,rtol=rtol),detail)

    def metric_match(self,actual,expected,detail):
        for name in ("accuracy","macro_f1","balanced_accuracy"):
            self.close(actual[name],expected[name],detail+" "+name,atol=1e-14,rtol=1e-13)
        for name in ("correct","total","confusion_matrix"):self.require(actual[name]==expected[name],detail+" "+name)
        for cls in actual["per_class"]:
            for name in ("precision","recall","f1","support"):
                self.close(actual["per_class"][cls][name],expected["per_class"][cls][name],detail+" "+cls+" "+name,atol=1e-14,rtol=1e-13)

    def source(self,context):
        key=context["split_id"]
        if key not in self.source_cache:
            self.require(sha(ROOT/context["source_pack"])==context["source_pack_sha256"],"Source pack changed")
            self.require(sha(ROOT/context["inference_pack"])==context["inference_pack_sha256"],"Inference pack changed")
            self.require(sha(ROOT/context["scoring_truth"])==context["scoring_truth_sha256"],"Scoring truth changed")
            source,query=npz(ROOT/context["source_pack"]),npz(ROOT/context["inference_pack"])
            self.require(source["record_ids"].tolist()==context["source_ids"] and query["record_ids"].tolist()==context["inference_ids"],"Pack IDs changed")
            self.require(not set(context["source_ids"])&set(context["inference_ids"]),"Specimen split leakage")
            self.require(set(query)=={k for k in source if k!="source_labels"},"Inference/source schema mismatch")
            self.source_cache[key]=source,query,truth(ROOT/context["scoring_truth"])
        return self.source_cache[key]

    def model(self,trial,context,rows):
        from herb_spectra.auxiliary_multiplant import load_fitted,AuxiliaryNet
        from herb_spectra.upgrade_neural import state_hash
        source,query,target=self.source(context);path=ROOT/trial["pipeline"]
        metadata=load(path.with_suffix(".json"));candidate=trial["candidate"];classes=tuple(metadata["classes"])
        self.require(sha(path)==trial["pipeline_sha256"],"Pipeline byte binding")
        self.require(sha(ROOT/trial["pipeline_metadata"])==trial["pipeline_metadata_sha256"],"Pipeline metadata bytes changed")
        self.require(metadata["code_version"]==object_sha(load(RUN/"run_manifest.json")["source_sha256"]),"Model code identity does not bind all implementation dependencies")
        self.require(all(trial[k]==context[k] for k in ("source_pack_sha256","inference_pack_sha256")),"Model prepared pack bindings changed")
        self.require(sha(ROOT/trial["predictions"])==trial["predictions_sha256"],"Prediction byte binding")
        self.require(all(metadata[k]==v for k,v in trial["metadata"].items()),"Saved metadata differs from trial record")
        self.require(metadata["candidate"]==candidate and metadata["source_ids"]==context["source_ids"],"Model candidate/source context")
        self.require(metadata["seed"]==trial["seed"] and metadata["dataset_id"]==context["dataset_id"],"Model seed/species")
        self.require(metadata["query_truth_received"] is False,"Model used inference truth")
        self.require(classes==tuple(("HQ" if context["dataset_id"]=="huangqi" else "TM")+f"{i:02d}" for i in range(1,5)),"Class spaces mixed")
        fitted=load_fitted(path,expected_split=context["split_id"],expected_ids=context["source_ids"])
        y=source["source_labels"];counts=np.asarray([np.count_nonzero(y==c) for c in classes])
        self.require(metadata["source_class_counts"]==dict(zip(classes,counts.tolist())),"Current source counts")
        standardized,scaler_fps={},{}
        for mode,scaler in fitted.scalers.items():
            cache_key=(context["split_id"],mode)
            if cache_key not in self.preprocessed_cache:
                raw=np.asarray(source["X_"+mode],float)
                values=(raw-raw.mean(1,keepdims=True))/raw.std(1,ddof=1,keepdims=True) if mode=="alpha" else raw
                self.preprocessed_cache[cache_key]=(values,scaler.transform(values))
            values,z=self.preprocessed_cache[cache_key]
            self.close(scaler.mean_,values.mean(0),"Source-only scaler mean")
            self.close(scaler.var_,values.var(0),"Source-only scaler variance")
            expected_scale=np.sqrt(values.var(0));expected_scale[expected_scale==0]=1
            self.close(scaler.scale_,expected_scale,"Independent source-only scaler scale")
            self.close(metadata["scaler_statistics"][mode]["mean"],scaler.mean_,"Saved scaler mean")
            self.close(metadata["scaler_statistics"][mode]["variance"],scaler.var_,"Saved scaler variance")
            self.require(joblib.hash(source["X_"+mode])==metadata["raw_source_fingerprints"][mode],"Raw source numeric fingerprint")
            self.close(scaler.transform(values),z,"Same context/scaler transform")
            standardized[mode]=z;scaler_fps["source_scaler_"+mode]=source_sha(values)
        full_alpha=standardized.get("alpha")
        if fitted.selector is not None:
            sel=fitted.selector;coefficients=np.asarray(sel.coefficients_)
            self.require(coefficients.shape==(candidate["mc_runs"],4,612),"MC coefficient shape")
            mean,sd=coefficients.mean(0),coefficients.std(0,ddof=1);floor=max(1.,float(sd.max()))*1e-12
            importance=np.linalg.norm(np.abs(mean/np.maximum(sd,floor)),axis=0);rank=np.argsort(-importance,kind="stable")
            selected=np.sort(rank[:int(np.ceil(candidate["mc_ratio"]*612))])
            self.close(sel.importances_,importance,"MC independent stability",atol=0.,rtol=0.)
            self.require(np.array_equal(sel.ranking_,rank) and np.array_equal(sel.selected_indices_,selected),"MC feature selection")
            self.require(metadata["selected_indices"]==selected.tolist(),"MC saved feature indices")
            standardized["alpha"]=full_alpha[:,selected];self.mc_selectors_recomputed += 1
        proof=fitted._leakage_fit_provenance
        self.require(proof["fit_sample_ids"]==context["source_ids"] and proof["split_id"]==context["split_id"],"Actual fitted source provenance")
        self.require(proof["dataset_version"]==context["dataset_id"]+"-user-declaration-20261031-v1","Dataset version mixed")
        identity=metadata["cache_identity"]
        self.require(identity["train_ids"]==context["source_ids"] and identity["split_id"]==context["split_id"],"Cache cross-fold identity")
        self.require(identity["code_version"]==metadata["code_version"] and identity["seed"]==trial["seed"] and identity["selector"]==candidate,"Cache parameters/code identity")
        if candidate["kind"] in ("full_svm","mcuve_svm"):
            z=standardized["alpha"];est=fitted.estimator
            expected_fp=source_sha(z,y);final_component="SVC.fit"
            self.require(np.array_equal(est.classes_,np.asarray(classes)),"SVC class order")
            self.close(est.class_weight_,len(y)/(4*counts),"Source-only balanced SVC weights")
            self.close(metadata["source_class_weights"],est.class_weight_,"Saved SVC weights")
            gamma=(1. if candidate["gamma_rule"]=="scale" else .1)/(z.shape[1]*z.var())
            self.close(est._gamma,gamma,"Source-only effective gamma")
            self.close(metadata["effective_gamma"],gamma,"Saved effective gamma")
            self.close(est.support_vectors_,z[est.support_],"Support vectors belong to current source",atol=0.,rtol=0.)
            self.require(est.C==candidate["C"] and est.kernel=="rbf" and est.probability is False and est.break_ties is False,"SVC parameters or vote rule changed")
            self.require(metadata["standard_vote_preserved"] is True and metadata["selected_features"]==z.shape[1],"SVC feature/vote metadata")
            self.require(est._leakage_fit_provenance==proof,"Embedded SVC provenance differs from cache proof")
            before=joblib.hash(est);self.svm_replayed += 1
        else:
            encoded=np.asarray([classes.index(c) for c in y]);features32={k:v.astype(np.float32) for k,v in standardized.items()}
            expected_fp=object_sha({"X":object_sha({str(k):array_sha(v) for k,v in sorted(features32.items())}),"y":array_sha(encoded)})
            final_component="torch_training"
            weights=len(y)/(4*counts) if candidate["class_weight"]=="balanced" else np.ones(4)
            self.close(metadata["class_weights"],weights,"Source-only neural class weights")
            self.require(metadata["optimizer_initialized_fresh"] is True and metadata["early_stopping"] is False,"Optimizer reuse or validation feedback")
            init=(context["dataset_id"],trial["family"],candidate["id"],trial["seed"])
            if init not in self.initial_cache:
                torch.manual_seed(trial["seed"])
                initial=AuxiliaryNet(candidate["kind"],candidate,tuple(standardized),4)
                self.initial_cache[init]=state_hash(initial)
            self.require(metadata["initial_state_sha256"]==self.initial_cache[init],"Independent neural initial state")
            self.require(len(metadata["loss_curve"])==metadata["epochs"]==candidate["epochs"] and np.isfinite(metadata["loss_curve"]).all(),"Neural fixed training budget")
            self.require(metadata["parameters"]==sum(v.numel() for v in fitted.estimator.parameters()),"Neural parameter count")
            for module in fitted.estimator.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                    self.require(module.track_running_stats and int(module.num_batches_tracked)==candidate["epochs"]*int(np.ceil(len(y)/64)),"BN source minibatch state")
            before=state_hash(fitted.estimator);self.neural_replayed += 1
        self.require(proof["data_fingerprint"]==expected_fp==identity["data_fingerprint"],"Actual estimator values/labels differ from allowed source")
        query_features={k[2:]:v for k,v in query.items() if k.startswith("X_")}
        score=fitted.predict_scores(query_features);found=fitted.predict(query_features)
        stored=np.asarray([[float(r["score_"+c]) for c in classes] for r in rows])
        self.close(score,stored,"Reloaded model score replay",atol=1e-5 if final_component=="torch_training" else 1e-12,rtol=1e-5 if final_component=="torch_training" else 1e-12)
        self.require(found.tolist()==[r["y_predicted"] for r in rows],"Reloaded standard vote/neural prediction")
        self.require([r["record_id"] for r in rows]==context["inference_ids"],"Prediction specimen order")
        audit=load(path.parent/"inference_audit.json")
        self.require(audit==trial["inference_audit"],"Saved inference audit differs from trial")
        self.require(audit["samples_checked"]==(len(context["inference_ids"]) if context["stage"]=="outer" else min(7,len(context["inference_ids"]))),"Pointwise audit population")
        if final_component=="torch_training":
            self.require(state_hash(fitted.estimator)==before==metadata["fitted_state_sha256"] and audit["parameters_and_buffers_unchanged"] is True,"BN/weight state changed during inference")
        else:
            self.require(joblib.hash(fitted.estimator)==before and audit["state_unchanged"] is True,"SVC state changed during inference")
            self.require(set(audit["batch_maximum_difference"])=={"1","2","3","7"} and all(v<=1e-12 for v in audit["batch_maximum_difference"].values()),"SVC tail-safe batch audit coverage")
        self.models_replayed += 1
        self.fit_expectations.append({"source_ids":context["source_ids"],"source_labels":y,"split_id":context["split_id"],
            "dataset_id":context["dataset_id"],"classes":classes,"candidate":candidate,"seed":trial["seed"],
            "code_version":metadata["code_version"],"scaler_fingerprints":scaler_fps,"mc_X":full_alpha if fitted.selector is not None else None,
            "final_component":final_component,"final_fingerprint":expected_fp})
        return metrics(rows,target,classes)

    def scope_log(self,path,trials):
        self.require(len(self.fit_expectations)==len(trials),"Incomplete model fit expectations")
        index,stack,run,target_index=0,[],0,0;coefficient=None;components=Counter();lines=0
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                lines += 1;row=json.loads(line);self.require(index<len(trials),"Extra learned operation")
                expected=self.fit_expectations[index];ids=row["fit_sample_ids"];classes=expected["classes"]
                self.require(row["split_id"]==expected["split_id"] and row["code_version"]==expected["code_version"],"Fit code/context changed")
                self.require(row["dataset_version"]==expected["dataset_id"]+"-user-declaration-20261031-v1","Fit plant version changed")
                self.require(len(ids)==len(set(ids)) and set(ids)<=set(expected["source_ids"]),"Actual learned operation outside current source")
                self.require(row["train_id_digest"]==object_sha(ids),"Fit specimen order digest")
                component,event=row["component"],row["event"]
                self.require(event in {"fit_start","fit_completed"},"Failed/rejected learned operation")
                if event=="fit_start":
                    if component in expected["scaler_fingerprints"]:
                        self.require(ids==expected["source_ids"] and row["data_fingerprint"]==expected["scaler_fingerprints"][component],"Scaler actual source data")
                    elif component=="MCUVESelector":
                        self.require(ids==expected["source_ids"] and expected["mc_X"] is not None,"MC source scope")
                        self.require(row["data_fingerprint"]==source_sha(expected["mc_X"],expected["source_labels"]),"MC actual source values/labels")
                        order=np.argsort(expected["source_ids"]);mcids=np.asarray(expected["source_ids"])[order];labels=expected["source_labels"][order];X=expected["mc_X"][order]
                        rng=np.random.default_rng(expected["seed"]);draws=[]
                        for _ in range(expected["candidate"]["mc_runs"]):
                            priority=rng.random(len(mcids));selected=[]
                            for cls in classes:
                                members=np.flatnonzero(labels==cls);n=min(len(members),max(1,int(np.ceil(len(members)*.8))))
                                selected.extend(members[np.argsort(priority[members],kind="stable")[:n]])
                            draws.append(np.sort(selected))
                        coefficient={"ids":mcids,"labels":labels,"X":X,"draws":draws};run=0
                    elif component=="MC-UVE.coefficient_standardization":
                        self.require(coefficient is not None and run<len(coefficient["draws"]),"Unexpected MC standardization")
                        pos=coefficient["draws"][run];x=coefficient["X"][pos];wanted=coefficient["ids"][pos].tolist();y=np.eye(4)[[classes.index(c) for c in coefficient["labels"][pos]]]
                        self.require(ids==wanted and row["data_fingerprint"]==source_sha(x,y),"Actual MC subsample/target differs from fixed RNG/source")
                        sd=x.std(0);sd[sd<=np.finfo(float).eps]=1;coefficient.update(Z=(x-x.mean(0))/sd,Y=y,current_ids=wanted);target_index=0
                    elif component=="MC-UVE.PLSRegression.fit":
                        self.require(coefficient is not None and target_index<4,"Unexpected PLS target")
                        y=coefficient["Y"][:,target_index];target=(y-y.mean())/y.std()
                        self.require(ids==coefficient["current_ids"] and row["data_fingerprint"]==source_sha(coefficient["Z"],target),"Individual PLS actual X/y binding")
                        target_index += 1
                    elif component==expected["final_component"]:
                        self.require(ids==expected["source_ids"] and row["data_fingerprint"]==expected["final_fingerprint"],"Actual final estimator source values/labels")
                    else:self.require(False,"Unknown learned component "+component)
                    stack.append((component,ids,row["data_fingerprint"]))
                else:
                    self.require(stack and stack.pop()==(component,ids,row["data_fingerprint"]),"Unpaired learned operation")
                    components[component] += 1
                    if component=="MC-UVE.coefficient_standardization":self.require(target_index==4,"PLS OVR class coverage");run += 1
                    if component=="MCUVESelector":self.require(run==expected["candidate"]["mc_runs"],"MC iteration coverage")
                    if component==expected["final_component"]:
                        self.require(not stack,"Incomplete nested learned operation");index += 1;coefficient=None
        self.require(not stack and index==len(trials),"Actual fit scope coverage")
        return {"jsonl_rows":lines,"model_fit_contexts":index,"actual_fit_components":dict(components),"actual_fits_total":sum(components.values()),
            "all_fit_ids_in_current_source":True,"source_values_and_labels_bound_independently":True,"MC_subsamples_and_each_PLS_target_verified":True}
