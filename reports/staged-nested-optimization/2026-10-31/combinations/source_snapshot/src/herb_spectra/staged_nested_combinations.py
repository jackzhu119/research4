"""Per-outer-fold R4 stable-selector and new-source CNN comparisons.

Every inherited parameter is selected using that same outer training partition.
Stable resampling is source-only search, not independent batch validation.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F

from .classification import classification_metrics
from .paper_fewshot_classification import state_fingerprint
from .selection import CARSStyleSelector, MCUVESelector
from .staged_nested_search import (class_weight_spec, gamma_spec,
    load_selection_contexts, prepare_protocol_roles, retained_count)


def _cnn_tensor(values):
    """Create a finite CPU float32 batch with the complete ordered 612 points."""
    if torch.is_tensor(values):
        tensor = values.detach().to(device="cpu", dtype=torch.float32)
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("CNN input must contain finite numerical source-scaled spectra")
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    if tensor.ndim != 3 or tensor.shape[1:] != (1, 612) or tensor.shape[0] < 1:
        raise ValueError("CNN requires a nonempty [N,612] or [N,1,612] batch")
    if not torch.isfinite(tensor).all():
        raise ValueError("CNN input has nonfinite values")
    return tensor.contiguous()


class SmallNestedCNN(nn.Module):
    """The fixed full-spectrum 8→16-channel CNN and 32-coordinate embedding."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.embedding = nn.Linear(256, 32)
        self.head = nn.Linear(32, 7)

    def encode(self, values):
        if (not torch.is_tensor(values) or values.ndim != 3
                or values.shape[1:] != (1, 612)):
            raise ValueError("SmallNestedCNN.encode expects a [N,1,612] tensor")
        return F.relu(self.embedding(self.features(values).flatten(1)))

    def forward(self, values):
        return self.head(self.encode(values))


def cnn_logits(network, values):
    """Pointwise logits only; no fitting, gradients, query statistics or labels."""
    network.eval()
    with torch.no_grad():
        scores = network(_cnn_tensor(values)).cpu().numpy().copy()
    if scores.shape != (len(values), 7) or not np.isfinite(scores).all():
        raise FloatingPointError("CNN returned invalid seven-class logits")
    return scores


def cnn_encode(network, values):
    """Frozen 32-dimensional embedding, using the same fitted source network."""
    network.eval()
    with torch.no_grad():
        embedding = network.encode(_cnn_tensor(values)).cpu().numpy().copy()
    if embedding.shape != (len(values), 32) or not np.isfinite(embedding).all():
        raise FloatingPointError("CNN returned invalid embeddings")
    return embedding


def fit_cnn_source(Zsource, ysource_encoded, *, seed=42, epochs=200,
                   batch_size=64, learning_rate=.001, weight_decay=.001):
    """Fit exclusively on caller-source data, with no validation/test API.

    The caller must fit its own preprocessing and full-612-point StandardScaler
    on the same source partition. Thread control belongs to the caller.
    """
    features = _cnn_tensor(Zsource)
    encoded = np.asarray(ysource_encoded)
    if (encoded.shape != (len(features),) or encoded.dtype.kind not in "iu"
            or set(encoded.tolist()) != set(range(7))):
        raise ValueError("Source training requires each of the seven integer classes 0..6")
    if (not isinstance(epochs, int) or epochs < 1
            or not isinstance(batch_size, int) or batch_size < 1
            or not np.isfinite([learning_rate, weight_decay]).all()
            or learning_rate <= 0 or weight_decay < 0):
        raise ValueError("Invalid fixed source-training budget")
    labels = torch.as_tensor(encoded, dtype=torch.long)
    counts = torch.bincount(labels, minlength=7).float()
    weights = len(features)/(7*counts)
    torch.manual_seed(seed)
    network = SmallNestedCNN()
    initial = state_fingerprint(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for epoch in range(epochs):
        network.train()
        total_loss = 0.
        for indices in torch.randperm(len(features), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(network(features[indices]), labels[indices], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weighted source CNN cross-entropy")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        entry = {"epoch": epoch+1,
                 "mean_batch_weighted_cross_entropy": total_loss/len(features)}
        if epoch == 0 or (epoch+1) % 20 == 0 or epoch+1 == epochs:
            predictions = cnn_logits(network, features).argmax(1)
            confusion = np.bincount(encoded*7+predictions, minlength=49).reshape(7, 7)
            entry.update(source_accuracy=float(np.trace(confusion)/len(encoded)),
                         source_confusion_matrix=confusion.tolist())
        losses.append(entry)
    network.eval()
    metadata = {
        "fit_scope": "caller_source_training_only",
        "source_samples": len(features),
        "input_features": 612,
        "input": "caller-source fitted preprocessing and StandardScaler; no MC feature cut",
        "source_class_counts": counts.to(dtype=torch.int64).tolist(),
        "balanced_class_weights": weights.tolist(),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "loss_semantics": "PyTorch weighted CE normalized by weights present in each mini-batch; epoch report is batch-size weighted mean, not full-source global weighted CE",
        "validation_or_test_labels_accepted": False,
        "early_stopping": False,
        "input_batch_statistics_fitted": False,
        "network_normalization": "per-instance GroupNorm, no BatchNorm",
        "embedding_dimensions": 32,
        "head_classes": 7,
        "dropout": 0,
        "prediction": "softmax argmax equals logit argmax; returned scores are logits",
        "initial_fingerprint": initial,
        "final_fingerprint": state_fingerprint(network),
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "losses": losses,
    }
    return network, metadata



FAMILIES=("ordinary_mcuve","budget_mcuve","stable_mcuve","stable_cars",
          "stable_joint","cnn_softmax","cnn_embedding_svm")
METRICS=("macro_f1","balanced_accuracy","accuracy")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def inherited_config(context):
    choices=context["choices"]
    mc=deepcopy(choices["R1"]["selected"]["mc"])
    svm=deepcopy(choices["R2"]["selected"]["svm"])
    prep=choices["R3"]["selected"]["preprocessing"]
    r3=choices["R3"]["selected"]
    if r3["mc"]!=mc or r3["svm"]!=svm:
        raise ValueError("R4 must inherit this same outer fold's sequential choices")
    return {"preprocessing":prep,"mc":mc,"svm":svm}


def source_resamples(labels,identifiers,count=10,fraction=.8,seed=42):
    labels=np.asarray(labels);identifiers=np.asarray(identifiers)
    if labels.shape!=identifiers.shape or len(set(identifiers.tolist()))!=len(labels):
        raise ValueError("Need unique source physical sample IDs")
    if count<2 or not 0<fraction<=1:
        raise ValueError("Invalid stable-selection resampling budget")
    groups={c:np.flatnonzero(labels==c) for c in np.unique(labels)}
    if min(map(len,groups.values()))<2:
        raise ValueError("Stable CARS internalCV2 needs at least2 source samples in each class")
    rng=np.random.default_rng(seed+9000);result=[]
    for repeat in range(count):
        selected=[]
        for c,indices in groups.items():
            take=min(len(indices),max(2,int(np.ceil(fraction*len(indices)))))
            selected.extend(rng.permutation(indices)[:take].tolist())
        indices=np.asarray(sorted(selected),dtype=int)
        result.append({"repeat":repeat,"indices":indices,"source_ids":identifiers[indices].tolist(),
            "class_counts":{str(c):int(np.sum(labels[indices]==c)) for c in groups}})
    return result


def ranked_topk(frequency,tie_ranking,count):
    frequency=np.asarray(frequency,dtype=float);tie_ranking=np.asarray(tie_ranking,dtype=int)
    if (frequency.ndim!=1 or tie_ranking.shape!=frequency.shape or not np.isfinite(frequency).all()
            or set(tie_ranking.tolist())!=set(range(len(frequency))) or not 1<=count<=len(frequency)):
        raise ValueError("Invalid source frequency/tie ranking/K")
    tie_position=np.empty(len(frequency),dtype=int);tie_position[tie_ranking]=np.arange(len(frequency))
    ordered=np.lexsort((tie_position,-frequency));chosen=np.sort(ordered[:count])
    return chosen,{"positive_frequency_variables":int(np.sum(frequency>0)),
        "zero_frequency_supplement_count":int(np.sum(frequency[chosen]<=0)),
        "tie_rule":"caller full-source ordinary MC-UVE ranking, then its stable index order"}


def pairwise_jaccard(sets):
    values=[]
    for i,j in itertools.combinations(range(len(sets)),2):
        left,right=set(map(int,sets[i])),set(map(int,sets[j]));union=left|right
        values.append({"first":i,"second":j,"intersection":len(left&right),"union":len(union),
            "jaccard":len(left&right)/len(union) if union else 1.})
    return {"pairs":values,"mean":float(np.mean([r["jaccard"] for r in values])) if values else None}


def fit_mcuve(values,labels,mc,*,runs=None,seed=42):
    return MCUVESelector(n_runs=int(mc["runs"] if runs is None else runs),n_components=int(mc["components"]),
        n_features_to_select=retained_count(values.shape[1],float(mc["ratio"])),sample_fraction=.8,
        task="classification",random_state=seed).fit(values,labels)


def fit_stable_selectors(values,labels,identifiers,mc,ordinary,*,seed=42,progress=None):
    x=np.asarray(values,dtype=float);y=np.asarray(labels);features=x.shape[1]
    count=retained_count(features,mc["ratio"]);parts=source_resamples(y,identifiers,seed=seed)
    mc_sets=[];cars_sets=[];records=[];mc_counts=np.zeros(features);cars_counts=np.zeros(features)
    for part in parts:
        indices=part["indices"];local_seed=seed+1000+part["repeat"]
        before=time.monotonic()
        selected_mc=fit_mcuve(x[indices],y[indices],mc,seed=local_seed)
        selected_cars=CARSStyleSelector(n_runs=12,n_components=int(mc["components"]),min_features=2,
            sample_fraction=.8,cv=2,reweight=True,task="classification",random_state=local_seed).fit(x[indices],y[indices])
        a=selected_mc.selected_indices_.copy();b=selected_cars.selected_indices_.copy()
        mc_sets.append(a);cars_sets.append(b);mc_counts[a]+=1;cars_counts[b]+=1
        records.append({k:v for k,v in part.items() if k!="indices"}|{
            "source_indices":indices.tolist(),"seed":local_seed,"mcuve_indices":a.tolist(),"cars_indices":b.tolist(),
            "cars_path_indices":[v.tolist() for v in selected_cars.path_],
            "cars_inner_mse_path":selected_cars.selection_mse_path_,"cars_best_iteration":selected_cars.best_iteration_,
            "seconds":time.monotonic()-before})
        if progress:progress({"event":"stable_resample_completed","repeat":part["repeat"],"samples":len(indices)})
    mc_frequency=mc_counts/len(parts);cars_frequency=cars_counts/len(parts)
    frequencies={"stable_mcuve":mc_frequency,"stable_cars":cars_frequency,
                 "stable_joint":.5*mc_frequency+.5*cars_frequency}
    selected={};supplements={}
    for family,frequency in frequencies.items():
        selected[family],supplements[family]=ranked_topk(frequency,ordinary.ranking_,count)
    joint_sets=[]
    for a,b in zip(mc_sets,cars_sets):
        indicator=np.zeros(features);indicator[a]+=.5;indicator[b]+=.5
        joint_sets.append(ranked_topk(indicator,ordinary.ranking_,count)[0])
    classes=len(np.unique(y));runs=int(mc["runs"])
    budget={"ordinary_mcuve_coefficient_rounds":runs,"ordinary_ovr_pls_fits":classes*runs,
        "budget_mcuve_coefficient_rounds":10*runs,"budget_ovr_pls_fits":10*classes*runs,
        "stable_mcuve_coefficient_rounds":10*runs,"stable_mcuve_ovr_pls_fits":10*classes*runs,
        "stable_cars_coefficient_rounds":10*12,"stable_cars_ovr_pls_fits":10*12*classes,
        "stable_cars_multivariate_internal_cv_pls_fits":10*12*2,
        "tie_breaker_extra_pls_fits":0,"tie_breaker_reuses_ordinary_full_source_ranking":True,
        "budget_control_matches_MC_rounds_not_walltime_or_FLOPs":True,
        "stable_joint_reuses_already_fitted_MC_and_CARS":True}
    metadata={"source_resamples":records,"resample_count":10,"sample_fraction":.8,
        "without_replacement":True,"minimum_samples_per_class":2,"resamples_are_independent_batches":False,
        "preprocessing_and_scaling_scope":"full caller source before resampling; internal source search only",
        "selection_frequencies":{k:v.tolist() for k,v in frequencies.items()},
        "selected_indices":{k:v.tolist() for k,v in selected.items()},"supplementation":supplements,
        "pairwise_jaccard":{"mcuve":pairwise_jaccard(mc_sets),"cars":pairwise_jaccard(cars_sets),
            "joint_resample_diagnostic":pairwise_jaccard(joint_sets)},
        "joint_resample_diagnostic_rule":"per-resample indicator mean with same full-source MC tie rank; global fit uses global averaged frequencies",
        "ordinary_full_source_ranking":ordinary.ranking_.tolist(),"fit_budget":budget}
    return selected,metadata


@dataclass
class CombinationModel:
    family:str
    inherited:dict
    preprocessor:object
    scaler:StandardScaler
    indices:np.ndarray
    classes_:np.ndarray
    metadata:dict
    estimator:object=None
    network:object=None
    embedding_scaler:object=None

    def _full_values(self,X):
        return self.scaler.transform(self.preprocessor.transform(np.asarray(X,dtype=float)))

    def _classifier_values(self,X):
        values=self._full_values(X)
        if self.family=="cnn_embedding_svm":
            return self.embedding_scaler.transform(cnn_encode(self.network,values))
        return values[:,self.indices]

    def predict_scores(self,X):
        if self.family=="cnn_softmax":
            return cnn_logits(self.network,self._full_values(X))
        scores=np.asarray(self.estimator.decision_function(self._classifier_values(X)),dtype=float)
        return np.column_stack([-scores,scores]) if scores.ndim==1 else scores

    def predict(self,X):
        if self.family=="cnn_softmax":
            return self.classes_[np.argmax(self.predict_scores(X),axis=1)]
        return self.estimator.predict(self._classifier_values(X))


def numeric_fingerprint(model):
    # Tensor pickle IDs/storage allocation are not numerical model state.
    return joblib.hash({"family":model.family,"inherited":model.inherited,
        "preprocessor":model.preprocessor,"scaler":model.scaler,"indices":model.indices,
        "classes":model.classes_,"estimator":model.estimator,"embedding_scaler":model.embedding_scaler,
        "network_state":state_fingerprint(model.network) if model.network is not None else None,
        "network_training":bool(model.network.training) if model.network is not None else None,
        "metadata":model.metadata})


def fit_classifier(values,labels,svm,seed=42):
    weights,weight_values=class_weight_spec(labels,svm["class_weight"])
    gamma,numeric=gamma_spec(values,svm["gamma"])
    estimator=SVC(C=float(svm["C"]),kernel="rbf",gamma=gamma,class_weight=weights,
        decision_function_shape="ovr",break_ties=False,random_state=seed).fit(values,labels)
    return estimator,{"svm":deepcopy(svm),"effective_gamma":float(estimator._gamma),
        "declared_gamma_numeric":numeric,"training_class_weights":weight_values,
        "classifier_features":int(values.shape[1]),"classifier_input_variance":float(np.asarray(values,dtype=float).var()),
        "prediction_rule":"SVC.predict standard vote, break_ties=False","scores_are_probabilities":False}


def fit_caller_models(X,y,axis,identifiers,inherited,*,seed=42,progress=None,cnn_epochs=200):
    """Fit all seven families on exactly one caller source partition."""
    from .staged_nested_search import fit_preprocess_bundle
    X=np.asarray(X,dtype=float);y=np.asarray(y);identifiers=np.asarray(identifiers);axis=np.asarray(axis)
    if (X.ndim!=2 or X.shape[1]!=612 or len(y)!=len(X) or axis.shape!=(612,)
            or len(np.unique(y))!=7 or len(set(identifiers.tolist()))!=len(X)):
        raise ValueError("R4 requires full612features and seven known source classes with unique IDs")
    start=time.monotonic();prep,scaler,z,_=fit_preprocess_bundle(X,X[:1],axis,inherited["preprocessing"])
    classes=np.unique(y);mc=inherited["mc"];keep=retained_count(612,mc["ratio"])
    ordinary=fit_mcuve(z,y,mc,seed=seed)
    if progress:progress({"event":"ordinary_mcuve_completed","requested_runs":int(mc["runs"])})
    budget=fit_mcuve(z,y,mc,runs=10*int(mc["runs"]),seed=seed)
    if progress:progress({"event":"budget_mcuve_completed","requested_runs":10*int(mc["runs"])})
    stable,stability=fit_stable_selectors(z,y,identifiers,mc,ordinary,seed=seed,progress=progress)
    indices={"ordinary_mcuve":ordinary.selected_indices_.copy(),"budget_mcuve":budget.selected_indices_.copy(),**stable}
    common={"inherited":deepcopy(inherited),"source_ids":identifiers.tolist(),"source_samples":len(X),
        "source_counts":{str(c):int(np.sum(y==c)) for c in classes},"source_value_hash":joblib.hash(X),
        "source_label_hash":joblib.hash(y),"input_features":612,"requested_selected_features":keep,
        "source_scaler_mean":scaler.mean_.tolist(),"source_scaler_variance":scaler.var_.tolist(),
        "msc_source_reference":prep.reference_.tolist() if inherited["preprocessing"]=="msc" else None,
        "ordinary_importance":ordinary.importances_.tolist(),"ordinary_ranking":ordinary.ranking_.tolist(),
        "ordinary_selected_indices":ordinary.selected_indices_.tolist(),
        "budget_importance":budget.importances_.tolist(),"budget_selected_indices":budget.selected_indices_.tolist(),
        "stability":stability,"model_seed":seed,"outer_test_values_or_labels_used_for_fit":False,
        "source_objects_reused_only_within_this_caller":True}
    models={}
    for family,selected in indices.items():
        estimator,meta=fit_classifier(z[:,selected],y,inherited["svm"],seed)
        details={"family":family,"source_ids":identifiers.tolist(),"source_samples":len(X),
            "selected_indices":selected.tolist(),"selected_frequencies":axis[selected].tolist(),
            "selected_features":len(selected),"full_input_features":612,"fit_budget":stability["fit_budget"],
            "source_preprocessing":inherited["preprocessing"],"feature_selection_applied":True,**meta}
        models[family]=CombinationModel(family,deepcopy(inherited),prep,scaler,selected.copy(),classes.copy(),details,estimator=estimator)
    encoded=np.searchsorted(classes,y)
    network,neural=fit_cnn_source(z,encoded,seed=seed,epochs=cnn_epochs)
    embedding=cnn_encode(network,z);embedding_scaler=StandardScaler().fit(embedding)
    standardized=embedding_scaler.transform(embedding)
    estimator,meta=fit_classifier(standardized,y,inherited["svm"],seed)
    neural_common={"source_ids":identifiers.tolist(),"source_samples":len(X),"full_input_features":612,
        "selected_indices":list(range(612)),"selected_frequencies":axis.tolist(),"selected_features":612,
        "feature_selection_applied":False,"embedding_features":32,"neural_training":neural,
        "source_preprocessing":inherited["preprocessing"],"cnn_encoder_fingerprint":state_fingerprint(network),
        "same_encoder_shared_between_CNN_head_and_SVM":True,"old_weights_reused":False,
        "input_and_learning_objective_differ_from_selected_variable_SVM":True}
    models["cnn_softmax"]=CombinationModel("cnn_softmax",deepcopy(inherited),prep,scaler,np.arange(612),
        classes.copy(),{**neural_common,"family":"cnn_softmax","classifier_features":32,
            "prediction_rule":"uncalibrated CNN softmax/logit argmax","scores_are_probabilities":False},network=network)
    models["cnn_embedding_svm"]=CombinationModel("cnn_embedding_svm",deepcopy(inherited),prep,scaler,np.arange(612),
        classes.copy(),{**neural_common,"family":"cnn_embedding_svm",**meta,
            "embedding_scaler_mean":embedding_scaler.mean_.tolist(),"embedding_scaler_variance":embedding_scaler.var_.tolist(),
            "embedding_constant_features":int(np.sum(embedding_scaler.var_==0))},estimator=estimator,
        network=network,embedding_scaler=embedding_scaler)
    common["neural_training"]=neural;common["source_fit_seconds"]=time.monotonic()-start
    if progress:progress({"event":"caller_source_models_completed","families":len(models),"seconds":common["source_fit_seconds"]})
    return models,common


def choose_family(rows,tolerance=1e-12):
    grouped=defaultdict(list)
    for row in rows:grouped[row["family"]].append(row["metrics"])
    if set(grouped)!=set(FAMILIES) or any(len(v)!=2 for v in grouped.values()):
        raise ValueError("R4 needs exactly two inner evaluations for each predeclared family")
    means={family:{m:float(np.mean([r[m] for r in grouped[family]])) for m in METRICS} for family in FAMILIES}
    maximum=max(v["macro_f1"] for v in means.values())
    eligible=[family for family in FAMILIES if means[family]["macro_f1"]>=maximum-tolerance]
    chosen=min(eligible,key=lambda f:(-means[f]["balanced_accuracy"],-means[f]["accuracy"],FAMILIES.index(f)))
    return {"selected":{"id":chosen,"family":chosen},"all_candidate_means":means,
        "mean_inner_metrics":means[chosen],"selected_using_outer_test":False,
        "tie_rule":"macroF1 within1e-12, balanced_accuracy, accuracy, predeclared simpler family order"}


def _manifest(input_path,protocol_path,core_protocol,core_run):
    root=Path(__file__).resolve().parents[2]
    paths=["src/herb_spectra/staged_nested_combinations.py","scripts/run_staged_nested_combinations.py",
        "src/herb_spectra/staged_nested_search.py","src/herb_spectra/selection.py",
        "src/herb_spectra/spectral_models.py","src/herb_spectra/image_feature_models.py",
        "src/herb_spectra/classification.py","src/herb_spectra/paper_fewshot_classification.py"]
    return {"input_sha256":digest(input_path),"protocol_sha256":digest(protocol_path),
        "core_protocol_sha256":digest(core_protocol),"core_run_manifest_sha256":digest(Path(core_run)/"run_manifest.json"),
        "core_selection_lock_sha256":digest(Path(core_run)/"selection_lock.json"),
        "source_sha256":{p:digest(root/p) for p in paths},"python":platform.python_version(),
        "versions":{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","torch","joblib")}}


def _source_expected(X,y,identifiers,inherited,seed,cnn_epochs):
    return {"source_ids":np.asarray(identifiers).tolist(),"source_value_hash":joblib.hash(np.asarray(X)),
        "source_label_hash":joblib.hash(np.asarray(y)),"inherited":inherited,"seed":seed,"cnn_epochs":cnn_epochs}


def fit_cached_caller(directory,X,y,axis,identifiers,inherited,*,seed=42,cnn_epochs=200,resume=False,progress=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    expected=_source_expected(X,y,identifiers,inherited,seed,cnn_epochs)
    checkpoint=directory/"source_checkpoint.json"
    if resume and checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved["expected"]!=expected or digest(directory/"caller_common.json")!=saved["common_sha256"]:
            raise ValueError("Cached R4 caller source identity or metadata changed")
        models={}
        for family in FAMILIES:
            row=saved["artifacts"][family];path=directory/row["file"]
            if digest(path)!=row["sha256"]:raise ValueError("Cached R4 source binary changed")
            model=joblib.load(path)
            if numeric_fingerprint(model)!=row["numeric_fingerprint"]:
                raise ValueError("Cached R4 numerical model state changed")
            models[family]=model
        if digest(directory/"cnn_source_state.pt")!=saved["cnn_state_sha256"]:
            raise ValueError("Cached shared CNN source state changed")
        return models,json.loads((directory/"caller_common.json").read_text())
    models,common=fit_caller_models(X,y,axis,identifiers,inherited,seed=seed,progress=progress,cnn_epochs=cnn_epochs)
    write_json(directory/"caller_common.json",common)
    artifacts={};loaded={}
    for family,model in models.items():
        path=directory/(family+".joblib")
        original_prediction=model.predict(X);original_scores=model.predict_scores(X)
        joblib.dump(model,path,compress=3);restored=joblib.load(path)
        restored_prediction=restored.predict(X);restored_scores=restored.predict_scores(X)
        if (not np.array_equal(original_prediction,restored_prediction)
                or not np.allclose(original_scores,restored_scores,rtol=1e-8,atol=1e-8)):
            raise AssertionError("R4 source checkpoint changed inference")
        artifacts[family]={"file":path.name,"sha256":digest(path),"numeric_fingerprint":numeric_fingerprint(restored),
            "source_roundtrip_predictions_equal":True,
            "source_roundtrip_max_score_difference":float(np.max(np.abs(original_scores-restored_scores))),
            "source_accuracy":float(np.mean(restored_prediction==np.asarray(y))),
            "metadata":restored.metadata}
        loaded[family]=restored
    torch.save(models["cnn_softmax"].network.state_dict(),directory/"cnn_source_state.pt")
    encoder=state_fingerprint(loaded["cnn_softmax"].network)
    if encoder!=state_fingerprint(loaded["cnn_embedding_svm"].network):
        raise AssertionError("CNN head and embedding SVM do not share the same learned encoder")
    write_json(checkpoint,{"expected":expected,"artifacts":artifacts,
        "common_sha256":digest(directory/"caller_common.json"),"cnn_state_sha256":digest(directory/"cnn_source_state.pt"),
        "shared_encoder_fingerprint":encoder,"cnn_source_training_invocations":1})
    return loaded,common


def _check_inner_roles(context,train,valid):
    outer_train=set(context["train_indices"].tolist());outer_test=set(context["test_indices"].tolist())
    if (set(train)&set(valid) or not set(train)<=outer_train or not set(valid)<=outer_train
            or (set(train)|set(valid))&outer_test or set(train)|set(valid)!=outer_train):
        raise ValueError("R4 inner source/validation roles violate outer partition isolation")


def _source_progress(output,context_id,event):
    record={"context":context_id,"utc":datetime.now(timezone.utc).isoformat(),**event}
    with (output/"source_fit_events.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
    print(json.dumps(record),flush=True)


def _verify_caller_files(directory):
    directory=Path(directory);source=json.loads((directory/"source_checkpoint.json").read_text())
    if digest(directory/"caller_common.json")!=source["common_sha256"]:
        raise ValueError("Source common metadata changed")
    for row in source["artifacts"].values():
        if digest(directory/row["file"])!=row["sha256"]:
            raise ValueError("Source artifact changed")
    if digest(directory/"cnn_source_state.pt")!=source["cnn_state_sha256"]:
        raise ValueError("Source CNN artifact changed")
    metric=directory/"metrics.json";pred=directory/"predictions.csv.gz"
    if (directory/"evaluation_checkpoint.json").exists():
        record=json.loads((directory/"evaluation_checkpoint.json").read_text())
        if digest(metric)!=record["metrics_sha256"] or digest(pred)!=record["predictions_sha256"]:
            raise ValueError("R4 saved prediction or metrics changed")


def _inner_evaluate(data,context,inherited,inner_no,fold,output,seed,*,resume=False):
    train=np.asarray(fold["train_indices"]);valid=np.asarray(fold["validation_indices"])
    _check_inner_roles(context,train,valid)
    outer=int(context["outer_fold"]);directory=output/"selection"/f"outer_{outer:02d}"/f"inner_{inner_no:02d}"
    checkpoint=directory/"evaluation_checkpoint.json"
    if resume and checkpoint.exists():
        _verify_caller_files(directory)
        previous=json.loads(checkpoint.read_text())
        if previous["source_global_indices"]!=train.tolist() or previous["validation_global_indices"]!=valid.tolist():
            raise ValueError("R4 cached inner roles changed")
        return json.loads((directory/"metrics.json").read_text())
    models,common=fit_cached_caller(directory,data["X_alpha"][train],data["y"][train],data["axis"],
        data["groups"][train],inherited,seed=seed,resume=resume,
        progress=lambda event:_source_progress(output,f"outer{outer}_inner{inner_no}",event))
    rows=[];classes=np.unique(data["y"][train]);predictions=[]
    for family,model in models.items():
        before=numeric_fingerprint(model);pred=model.predict(data["X_alpha"][valid]);scores=model.predict_scores(data["X_alpha"][valid])
        if before!=numeric_fingerprint(model):raise AssertionError("R4 validation prediction changed source model")
        rows.append({"outer_fold":outer,"inner_fold":inner_no,"stage":"R4","candidate_id":family,"family":family,
            "config":inherited,"metrics":classification_metrics(data["y"][valid],pred,classes),
            "validation_samples":len(valid),
            "selected_features":model.metadata["selected_features"],"classifier_features":model.metadata["classifier_features"],
            "feature_selection_applied":model.metadata["feature_selection_applied"],
            "embedding_features":model.metadata.get("embedding_features"),
            "cnn_encoder_fingerprint":model.metadata.get("cnn_encoder_fingerprint"),
            "model_path":str(directory/(family+".joblib")),"model_sha256":digest(directory/(family+".joblib")),
            "vote_argmax_disagreements":int(np.sum(pred!=classes[np.argmax(scores,axis=1)]))})
        for identifier,t,p,s in zip(data["groups"][valid],data["y"][valid],pred,scores):
            predictions.append({"outer_fold":outer,"inner_fold":inner_no,"stage":"R4","candidate_id":family,
                "family":family,"record_id":str(identifier),"y_true":str(t),"y_predicted":str(p),"is_correct":int(t==p),
                **{"score_"+str(c):float(v) for c,v in zip(classes,s)}})
    with gzip.open(directory/"predictions.csv.gz","wt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]),lineterminator="\n");writer.writeheader();writer.writerows(predictions)
    write_json(directory/"metrics.json",rows)
    write_json(checkpoint,{"source_global_indices":train.tolist(),"validation_global_indices":valid.tolist(),
        "metrics_sha256":digest(directory/"metrics.json"),"predictions_sha256":digest(directory/"predictions.csv.gz"),
        "outer_test_scored":False})
    return rows


def _check_release(output,protocol_path,release_path):
    if release_path is None:raise ValueError("Global nested selection release required")
    output=Path(output);release=Path(release_path);gate=json.loads(release.read_text());lock=output/"selection_lock.json"
    matches=[r for r in gate["rounds"] if Path(r["selection_lock"]).resolve()==lock.resolve()]
    if len(matches)!=1:raise ValueError("R4 selection lock missing from global release")
    record=matches[0]
    for key,path in [("selection_lock_sha256",lock),("protocol_sha256",protocol_path),("run_manifest_sha256",output/"run_manifest.json")]:
        if record.get(key)!=digest(path):raise ValueError("R4 released hashes changed")
    (output/"release_snapshot.json").write_bytes(release.read_bytes())
    return release


def _evaluation(data,contexts,lock,output,protocol,*,resume=False):
    rows=[];all_predictions=[];classes=np.unique(data["y"])
    for context,foldlock in zip(contexts,lock["outer_folds"]):
        outer=int(context["outer_fold"]);train=context["train_indices"];test=context["test_indices"]
        if outer!=foldlock["outer_fold"] or inherited_config(context)!=foldlock["inherited_config"]:
            raise ValueError("R4 outer inheritance no longer matches its inner selection")
        inherited=foldlock["inherited_config"]
        directory=output/"evaluation"/f"outer_{outer:02d}"
        models,common=fit_cached_caller(directory,data["X_alpha"][train],data["y"][train],data["axis"],
            data["groups"][train],inherited,seed=protocol["model_seed"],resume=resume,
            progress=lambda event:_source_progress(output,f"outer{outer}_fullsource",event))
        roles={"R4":foldlock["choice"]["selected"]["family"],
               **{"R4_diagnostic_"+family:family for family in FAMILIES}}
        predicted_cache={}
        for family,model in models.items():
            before=numeric_fingerprint(model);pred=model.predict(data["X_alpha"][test]);scores=model.predict_scores(data["X_alpha"][test])
            if before!=numeric_fingerprint(model):raise AssertionError("R4 outer prediction changed source model")
            predicted_cache[family]=(pred,scores)
        for method,family in roles.items():
            model=models[family];pred,scores=predicted_cache[family]
            rows.append({"method":method,"outer_fold":outer,"family":family,"config":inherited,
                **classification_metrics(data["y"][test],pred,classes),
                "test_samples":len(test),
                "selected_features":model.metadata["selected_features"],"classifier_features":model.metadata["classifier_features"],
                "feature_selection_applied":model.metadata["feature_selection_applied"],
                "embedding_features":model.metadata.get("embedding_features"),
                "cnn_encoder_fingerprint":model.metadata.get("cnn_encoder_fingerprint"),
                "selected_indices":model.indices.tolist(),"selected_frequencies":data["axis"][model.indices].tolist(),
                "model_path":str(directory/(family+".joblib")),"model_sha256":digest(directory/(family+".joblib")),
                "vote_argmax_disagreements":int(np.sum(pred!=classes[np.argmax(scores,axis=1)]))})
            for identifier,t,p,s in zip(data["groups"][test],data["y"][test],pred,scores):
                all_predictions.append({"method":method,"outer_fold":outer,"family":family,"record_id":str(identifier),
                    "y_true":str(t),"y_predicted":str(p),"is_correct":int(t==p),
                    **{"score_"+str(c):float(v) for c,v in zip(classes,s)}})
        print(json.dumps({"stage":"R4_outer_evaluation","outer_fold":outer,"families":len(models),"method_roles":len(roles)}),flush=True)
    with gzip.open(output/"outer_predictions.csv.gz","wt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_predictions[0]),lineterminator="\n");writer.writeheader();writer.writerows(all_predictions)
    summaries=[]
    for method in sorted(set(r["method"] for r in all_predictions)):
        values=[r for r in all_predictions if r["method"]==method]
        if len(values)!=855 or len(set(r["record_id"] for r in values))!=855:
            raise AssertionError("R4 OOF method must predict every original specimen exactlyonce")
        folds=[r for r in rows if r["method"]==method]
        summaries.append({"method":method,"samples":len(values),
            **classification_metrics([r["y_true"] for r in values],[r["y_predicted"] for r in values],classes),
            "fold_mean":{m:float(np.mean([r[m] for r in folds])) for m in METRICS},
            "fold_sd":{m:float(np.std([r[m] for r in folds],ddof=1)) for m in METRICS},
            "same_batch_retrospective":True,"outer_scores_used_to_select":False})
    write_json(output/"outer_metrics.json",rows);write_json(output/"summary.json",summaries)
    return summaries


def run_search(input_path,protocol_path,core_protocol,core_run,output,*,stage="selection",resume=False,release_path=None):
    if stage not in ("selection","evaluation"):raise ValueError("Unknown R4 execution stage")
    output=Path(output);output.mkdir(parents=True,exist_ok=True);protocol=json.loads(Path(protocol_path).read_text())
    if protocol["families"]!=list(FAMILIES) or protocol["cnn"]["epochs"]!=200 or protocol["model_seed"]!=42:
        raise ValueError("R4 implementation differs from frozen family/architecture budget")
    manifest=_manifest(input_path,protocol_path,core_protocol,core_run);mp=output/"run_manifest.json"
    if mp.exists():
        if not resume or json.loads(mp.read_text())!=manifest:raise ValueError("R4 existing provenance differs")
    else:
        write_json(mp,manifest);(output/"protocol_snapshot.json").write_bytes(Path(protocol_path).read_bytes())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contexts=load_selection_contexts(input_path,core_protocol,core_run)
    if len(contexts)!=5:raise ValueError("Expected five independently inherited outer contexts")
    data=contexts[0]["data"];started=time.monotonic();lockpath=output/"selection_lock.json"
    with (output/"stage_events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event":"start","stage":stage,"resume":resume,"utc":datetime.now(timezone.utc).isoformat()})+"\n")
    if stage=="selection":
        if resume and lockpath.exists():
            lock=json.loads(lockpath.read_text())
            for context in contexts:
                for inner_no in range(2):_verify_caller_files(output/"selection"/f"outer_{context['outer_fold']:02d}"/f"inner_{inner_no:02d}")
            rows=json.loads((output/"all_inner_metrics.json").read_text())
            for item in lock["outer_folds"]:
                if choose_family([r for r in rows if r["outer_fold"]==item["outer_fold"]])!=item["choice"]:
                    raise ValueError("R4 saved choice no longer matches inner metrics")
            return {"status":"selection_already_completed","selection_lock_sha256":digest(lockpath)}
        all_rows=[];locks=[]
        for context in contexts:
            inherited=inherited_config(context);rows=[]
            for inner_no,fold in enumerate(context["inner_folds"]):
                rows.extend(_inner_evaluate(data,context,inherited,inner_no,fold,output,protocol["model_seed"],resume=resume))
            choice=choose_family(rows)
            locks.append({"outer_fold":context["outer_fold"],"inherited_config":inherited,"choice":choice})
            all_rows.extend(rows)
            write_json(output/"status.json",{"status":"selection","completed_outer_folds":len(locks),"inner_evaluations":len(all_rows),"outer_test_scored":False})
            print(json.dumps({"stage":"R4_selection","outer_fold":context["outer_fold"],"choice":choice["selected"]}),flush=True)
        write_json(output/"all_inner_metrics.json",all_rows)
        lock={"outer_folds":locks,"core_selection_lock_sha256":manifest["core_selection_lock_sha256"],
            "inner_evaluations":len(all_rows),"outer_scores_used_to_select":False,"run_manifest_sha256":digest(mp)}
        write_json(lockpath,lock)
        write_json(output/"status.json",{"status":"selection_completed","seconds":time.monotonic()-started,
            "inner_evaluations":len(all_rows),"inner_prediction_rows":sum(r["validation_samples"] for r in all_rows),
            "selection_lock_sha256":digest(lockpath),"outer_test_scored":False})
        return json.loads((output/"status.json").read_text())
    if not lockpath.exists():raise ValueError("Complete all five R4 inner selections before outer evaluation")
    _check_release(output,protocol_path,release_path)
    if resume and (output/"summary.json").exists():
        status=json.loads((output/"status.json").read_text())
        for p,sha in status["result_sha256"].items():
            if digest(output/p)!=sha:raise ValueError("Completed R4 evaluation result changed")
        return json.loads((output/"summary.json").read_text())
    lock=json.loads(lockpath.read_text())
    for context,item in zip(contexts,lock["outer_folds"]):
        if inherited_config(context)!=item["inherited_config"]:
            raise ValueError("R4 inheritance changed after selection lock")
    summaries=_evaluation(data,contexts,lock,output,protocol,resume=resume)
    result_files={str(p.relative_to(output)):digest(p) for p in output.rglob('*')
        if p.is_file() and p!=output/"status.json" and p.suffix!='.partial' and p.name!='stage_events.jsonl'}
    write_json(output/"status.json",{"status":"completed","evaluation_seconds":time.monotonic()-started,
        "outer_evaluations":40,"outer_prediction_rows":6840,"selection_lock_sha256":digest(lockpath),
        "new_independent_batches":0,"target_calibration_labels":0,"result_sha256":result_files})
    return summaries
