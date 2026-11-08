"""Small, explicitly guarded inductive neural comparisons for exposed THz data.

Preparation, training, inference and scoring are separate entry points. Source
packs carry only current-source labels; inference packs carry none. The file
interfaces and guards are auditable boundaries, not an OS security sandbox.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .leakage_guard import (FitScope, LeakageError, audit_pointwise_inference,
    assert_modalities_aligned, guarded_cache_key, save_guarded_cache,
    load_guarded_cache, validate_fixed_partitions, validate_lineage_partitions, instrument_mcuve_pls)
from .selection import MCUVESelector
from .classification import classification_metrics
from .multiplant_database import validate_zaoren_row_bindings


CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
FAMILIES = ("cnn_alpha", "resnet_alpha", "se_alpha", "dual_alpha_mcuve",
            "fusion_alpha_n", "cnn_n", "fusion_time_frequency", "fusion_all_four")
MODALITIES = {"cnn_alpha":("alpha",), "resnet_alpha":("alpha",), "se_alpha":("alpha",),
    "dual_alpha_mcuve":("alpha",), "fusion_alpha_n":("alpha","n"), "cnn_n":("n",),
    "fusion_time_frequency":("time","frequency"), "fusion_all_four":("alpha","n","time","frequency"),
    "mcuve_only":("alpha",), "dual_alpha_mcuve_se":("alpha",)}
PROFILES = (
    {"id":"small12", "width":8, "blocks":1, "epochs":12, "dropout":.15,
     "learning_rate":.001, "class_weight":"none", "mc_runs":10, "mc_components":1, "mc_ratio":.1},
    {"id":"regularized24", "width":12, "blocks":2, "epochs":24, "dropout":.35,
     "learning_rate":.0005, "class_weight":"balanced", "mc_runs":20, "mc_components":2, "mc_ratio":.2},
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def state_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def fixed_blocks(values, axis, maximum):
    """Consecutive block means from one spectrum; no population statistics."""
    values,axis=np.asarray(values,float),np.asarray(axis,float)
    if values.ndim!=2 or values.shape[1]!=len(axis) or not np.isfinite(values).all() or not np.all(np.diff(axis)>0):
        raise ValueError("Finite increasing coordinate with paired spectra required")
    blocks=np.array_split(np.arange(len(axis)),min(int(maximum),len(axis)))
    return (np.stack([values[:,b].mean(1) for b in blocks],axis=1),
            np.asarray([axis[b].mean() for b in blocks]),
            [[int(b[0]),int(b[-1])+1] for b in blocks])


def row_snv(values):
    values=np.asarray(values,float);sd=values.std(1,ddof=1,keepdims=True)
    if np.any(sd<=np.finfo(float).eps):raise ValueError("Constant spectrum cannot receive SNV")
    return (values-values.mean(1,keepdims=True))/sd


class SE(nn.Module):
    def __init__(self,channels):
        super().__init__();self.net=nn.Sequential(nn.Linear(channels,max(2,channels//4)),nn.ReLU(),nn.Linear(max(2,channels//4),channels),nn.Sigmoid())
    def forward(self,x):return x*self.net(x.mean(2)).unsqueeze(2)


class Residual(nn.Module):
    def __init__(self,width,attention):
        super().__init__();self.path=nn.Sequential(nn.Conv1d(width,width,3,padding=1,bias=False),nn.BatchNorm1d(width),nn.ReLU(),nn.Conv1d(width,width,3,padding=1,bias=False),nn.BatchNorm1d(width));self.attention=SE(width) if attention else nn.Identity()
    def forward(self,x):return torch.relu(x+self.attention(self.path(x)))


class Encoder(nn.Module):
    def __init__(self,width,blocks,dropout,residual=False,attention=False):
        super().__init__();layers=[nn.Conv1d(1,width,7,stride=4,padding=3,bias=False),nn.BatchNorm1d(width),nn.ReLU()]
        for _ in range(blocks):
            layers.append(Residual(width,attention) if residual else nn.Sequential(nn.Conv1d(width,width,5,stride=2,padding=2,bias=False),nn.BatchNorm1d(width),nn.ReLU()))
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(8),nn.Flatten(),nn.Dropout(dropout),nn.Linear(width*8,16),nn.ReLU())
    def forward(self,x):return self.net(x.unsqueeze(1))


class UpgradeNet(nn.Module):
    def __init__(self,family,profile,dimensions):
        super().__init__();self.family=family
        chosen=() if family=="mcuve_only" else MODALITIES[family]
        residual=family in {"resnet_alpha","se_alpha","dual_alpha_mcuve_se"}
        attention=family in {"se_alpha","dual_alpha_mcuve_se"}
        self.encoders=nn.ModuleDict({m:Encoder(profile["width"],profile["blocks"],profile["dropout"],residual,attention) for m in chosen})
        use_mc=family in {"dual_alpha_mcuve","mcuve_only","dual_alpha_mcuve_se"}
        self.mc=nn.Sequential(nn.Linear(dimensions["mcuve"],32),nn.ReLU(),nn.Dropout(profile["dropout"]),nn.Linear(32,16),nn.ReLU()) if use_mc else None
        self.head=nn.Linear(16*(len(chosen)+int(use_mc)),7)
    def forward(self,inputs):
        branches=[net(inputs[name]) for name,net in self.encoders.items()]
        if self.mc is not None:branches.append(self.mc(inputs["mcuve"]))
        return self.head(torch.cat(branches,1))


@dataclass
class GuardedMC:
    selected_indices: np.ndarray
    coefficients: np.ndarray
    importance: np.ndarray
    ranking: np.ndarray
    metadata: dict


def fit_guarded_mc(values,labels,ids,profile,seed,scope):
    """Original selector; instrument every internal PLS fit without changing math."""
    values=np.asarray(values,float);labels=np.asarray(labels);ids=np.asarray(ids)
    if tuple(np.unique(labels))!=CLASSES:raise LeakageError("All seven source classes must remain")
    k=int(np.ceil(profile["mc_ratio"]*values.shape[1]))
    selector=MCUVESelector(n_runs=profile["mc_runs"],n_components=profile["mc_components"],
        n_features_to_select=k,sample_fraction=.8,task="classification",random_state=seed)
    with instrument_mcuve_pls(scope,values,ids):
        scope.fit(selector,values,labels,sample_ids=ids,component="mcuve_selector",groups=ids)
    return GuardedMC(selector.selected_indices_.copy(),selector.coefficients_.copy(),
        selector.importances_.copy(),selector.ranking_.copy(),
        {"runs":profile["mc_runs"],"components":profile["mc_components"],"ratio":profile["mc_ratio"],
         "K":k,"source_ids":ids.tolist(),"selection_scope":"current_inner_or_outer_training_only",
         "mathematical_implementation":"unchanged original MCUVESelector"})


@dataclass
class FittedUpgrade:
    family: str
    profile: dict
    scalers: dict
    selector: object
    model: object
    metadata: dict
    def transform(self,modalities):
        result={}
        for name,scaler in self.scalers.items():
            x=row_snv(modalities[name]) if name=="alpha" else np.asarray(modalities[name],float)
            result[name]=scaler.transform(x).astype(np.float32)
        if self.selector is not None:result["mcuve"]=result["alpha"][:,self.selector.selected_indices]
        return result
    def predict_scores(self,modalities):
        values=self.transform(modalities);self.model.eval()
        before=state_hash(self.model)
        with torch.no_grad():out=self.model({k:torch.as_tensor(v,dtype=torch.float32) for k,v in values.items()}).cpu().numpy()
        if state_hash(self.model)!=before:raise LeakageError("Inference changed model state")
        return out


def fit_source(modalities,labels,ids,family,profile,seed,split_id,scope_log,code_version):
    """No validation/test arrays or labels are accepted by this training API."""
    ids=np.asarray(ids);labels=np.asarray(labels)
    if tuple(np.unique(labels))!=CLASSES or len(ids)!=len(labels):raise LeakageError("Seven-class unique source ID mapping required")
    scope=FitScope(ids,split_id,"2026-10-31-zaoren-numbered-v1",scope_log,code_version=code_version)
    scalers={};features={};standardized64={};start=time.perf_counter()
    for name in MODALITIES[family]:
        x=row_snv(modalities[name]) if name=="alpha" else np.asarray(modalities[name],float)
        if len(x)!=len(ids) or not np.isfinite(x).all():raise LeakageError("Source modality rows must be finite and paired")
        scaler=scope.fit(StandardScaler(),x,sample_ids=ids,component="scaler_"+name)
        scalers[name]=scaler;standardized64[name]=scaler.transform(x);features[name]=standardized64[name].astype(np.float32)
    selector=None
    if family in {"dual_alpha_mcuve","mcuve_only","dual_alpha_mcuve_se"}:
        selector=fit_guarded_mc(standardized64["alpha"],labels,ids,profile,seed,scope)
        features["mcuve"]=features["alpha"][:,selector.selected_indices]
    transform_seconds=time.perf_counter()-start
    torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=UpgradeNet(family,profile,{k:v.shape[1] for k,v in features.items()})
    initial=state_hash(model);optimizer=torch.optim.Adam(model.parameters(),lr=profile["learning_rate"],weight_decay=.001)
    encoded=np.asarray([CLASSES.index(v) for v in labels]);counts=np.bincount(encoded,minlength=7)
    weight=len(labels)/(7*counts) if profile["class_weight"]=="balanced" else np.ones(7)
    tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in features.items()};target=torch.as_tensor(encoded,dtype=torch.long)
    criterion=nn.CrossEntropyLoss(weight=torch.as_tensor(weight,dtype=torch.float32));generator=torch.Generator().manual_seed(seed)
    losses=[];start=time.perf_counter()
    with scope.operation("torch_training",ids,X=features,y=encoded) as proof:
        for epoch in range(profile["epochs"]):
            model.train();order=torch.randperm(len(ids),generator=generator);total=0.
            for left in range(0,len(ids),64):
                chosen=order[left:left+64];optimizer.zero_grad(set_to_none=True)
                loss=criterion(model({k:v[chosen] for k,v in tensors.items()}),target[chosen]);loss.backward();optimizer.step();total+=float(loss.detach())*len(chosen)
            losses.append(total/len(ids))
    model.eval();train_seconds=time.perf_counter()-start
    metadata={"family":family,"profile":profile,"seed":seed,"source_ids":ids.tolist(),"source_class_counts":dict(zip(CLASSES,counts.tolist())),
        "split_id":split_id,"code_version":code_version,"initial_state_sha256":initial,"fitted_state_sha256":state_hash(model),
        "class_order":list(CLASSES),"parameters":sum(p.numel() for p in model.parameters()),"epochs":profile["epochs"],"loss_curve":losses,
        "training_seconds":train_seconds,"transform_fit_seconds":transform_seconds,"class_weights":weight.tolist(),
        "selected_indices":None if selector is None else selector.selected_indices.tolist(),"fit_scope_entries":len(scope.entries),
        "raw_source_modality_fingerprints":{k:joblib.hash(modalities[k]) for k in scalers},
        "MC_numerical_precision":"float64; network inputs float32",
        "scaler_statistics":{k:{"mean":v.mean_.tolist(),"variance":v.var_.tolist()} for k,v in scalers.items()},
        "target_labels_received_by_training":False,"optimizer_initialized_fresh":True,"early_stopping":False,"pretrained_weights":None}
    fitted=FittedUpgrade(family,dict(profile),scalers,selector,model,metadata);fitted._leakage_fit_provenance=proof
    return fitted


def cache_identity(fitted):
    m=fitted.metadata;proof=fitted._leakage_fit_provenance
    return dict(dataset_version=proof["dataset_version"],train_ids=m["source_ids"],split_id=m["split_id"],
        preprocessing={"alpha":"per-spectrum SNV ddof1 then source scaler","others":"source scaler", "raw_source_modality_fingerprints":m["raw_source_modality_fingerprints"]},
        selector={"family":m["family"],"profile":m["profile"]},code_version=m["code_version"],seed=m["seed"],data_fingerprint=proof["data_fingerprint"])


def save_fitted(fitted,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=cache_identity(fitted);key=guarded_cache_key(**identity)
    path=directory/"pipeline.joblib";save_guarded_cache(path,fitted,key)
    json_write(directory/"training.json",{**fitted.metadata,"cache_identity":identity,"cache_key":str(key),"pipeline_sha256":digest(path)})


def load_fitted(directory,*,expected_split_id=None,expected_source_ids=None):
    directory=Path(directory);metadata=json.loads((directory/"training.json").read_text());identity=metadata["cache_identity"]
    if expected_split_id is not None and identity["split_id"]!=expected_split_id:raise LeakageError("Cross-fold neural weight rejected")
    if expected_source_ids is not None and list(expected_source_ids)!=identity["train_ids"]:raise LeakageError("Changed source identity rejected")
    if digest(directory/"pipeline.joblib")!=metadata["pipeline_sha256"]:raise LeakageError("Neural artifact bytes changed")
    fitted=load_guarded_cache(directory/"pipeline.joblib",guarded_cache_key(**identity))
    if state_hash(fitted.model)!=metadata["fitted_state_sha256"]:raise LeakageError("Stored network state mismatch")
    return fitted


def read_pack(path,source=False):
    with np.load(path,allow_pickle=False) as a:result={k:a[k] for k in a.files}
    allowed={"record_ids","X_alpha","X_n","X_time","X_frequency"}
    if source:allowed.add("source_labels")
    unexpected=set(result)-allowed
    if unexpected:raise LeakageError("Pack may not contain truth or unknown fields: "+str(sorted(unexpected)))
    identifiers=np.asarray(result.get("record_ids",[]))
    if identifiers.ndim!=1 or not len(identifiers) or len(set(identifiers.tolist()))!=len(identifiers):raise LeakageError("Pack requires unique one-dimensional record IDs")
    for key,value in result.items():
        if key.startswith("X_") and (value.ndim!=2 or len(value)!=len(identifiers) or not np.issubdtype(value.dtype,np.number) or not np.isfinite(value).all()):
            raise LeakageError("Spectral features must be paired finite numeric matrices, never IDs or paths")
    if source:
        if "source_labels" not in result:raise LeakageError("Source labels missing")
    return result


def pack_modalities(pack):return {k[2:]:v for k,v in pack.items() if k.startswith("X_")}


def metric_record(truth,predicted):
    value=classification_metrics(np.asarray(truth),np.asarray(predicted),np.asarray(CLASSES))
    matrix=np.asarray(value["confusion_matrix"]);tp=matrix.diagonal();actual=matrix.sum(1);found=matrix.sum(0)
    value["per_class"]={c:{"precision":float(t/p) if p else 0.,"recall":float(t/n) if n else 0.,"f1":float(2*t/(p+n)) if p+n else 0.,"support":int(n)} for c,t,p,n in zip(CLASSES,tp,found,actual)}
    value.update(correct=int(tp.sum()),total=int(matrix.sum()));return value


def prediction_rows(fitted,pack,common):
    scores=fitted.predict_scores(pack_modalities(pack));pred=np.asarray(CLASSES)[scores.argmax(1)]
    return [{**common,"record_id":str(rid),"y_predicted":str(p),**{"score_"+c:float(v) for c,v in zip(CLASSES,score)}} for rid,p,score in zip(pack["record_ids"],pred,scores)]


def write_predictions(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def read_predictions(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as h:return list(csv.DictReader(h))


def truth_map(path):
    with Path(path).open(encoding="utf-8",newline="") as h:return {r["record_id"]:r["label"] for r in csv.DictReader(h)}


def score_predictions(rows,truth):
    identifiers=[r["record_id"] for r in rows]
    if len(set(identifiers))!=len(rows) or set(identifiers)!=set(truth):raise LeakageError("Duplicate or mismatched OOF denominator")
    return metric_record([truth[r] for r in identifiers],[r["y_predicted"] for r in rows])


def select_neural_strategy(family_choices):
    """Select architecture/attention/fusion using inner means only, before test."""
    candidates=[]
    for family,choice in family_choices.items():
        if choice.get("outer_test_used_to_select") is not False:raise LeakageError("Architecture choice may not use outer results")
        for row in choice["means"]:
            metrics=row["mean_inner_metrics"]
            if not all(np.isfinite(metrics[k]) for k in ("macro_f1","balanced_accuracy","accuracy")):
                raise LeakageError("Nonfinite inner metric cannot select architecture")
            candidates.append({"selected_family":family,"selected_profile":row["profile"],
                "mean_inner_metrics":dict(metrics),"selection_case_count":row["selection_case_count"]})
    if not candidates:raise LeakageError("No completed inner architecture candidates")
    selected=min(candidates,key=lambda c:(-c["mean_inner_metrics"]["macro_f1"],-c["mean_inner_metrics"]["balanced_accuracy"],
        -c["mean_inner_metrics"]["accuracy"],c["selected_family"],c["selected_profile"]["id"]))
    return {**selected,"all_architecture_candidate_means":candidates,"outer_test_used_to_select":False,
        "prediction_role":"inner_selected_neural","new_outer_model_fits":0,"weights_are_alias_of_selected_family":True}


def prepare(protocol_path,output):
    """Independent data-preparation role; labels never enter inference packs."""
    root=Path(__file__).resolve().parents[2];protocol=json.loads(Path(protocol_path).read_text());output=Path(output)
    if (output/"preparation.json").exists():raise FileExistsError("Prepared run already exists")
    if digest(root/protocol["input"])!=protocol["input_sha256"]:raise LeakageError("Database values or global pairing changed after source audit")
    if digest(root/protocol["alignment"])!=protocol["alignment_sha256"]:raise LeakageError("Original modality pairing evidence changed")
    with np.load(root/protocol["input"],allow_pickle=False) as a:data={k:a[k] for k in a.files}
    ids=np.asarray(data["record_ids"]).astype(str);labels=np.asarray(data["y"]).astype(str)
    if len(ids)!=855 or len(set(ids))!=855 or tuple(np.unique(labels))!=CLASSES:raise LeakageError("Unexpected seven-class855 database")
    aligned=json.loads((root/protocol["alignment"]).read_text())
    binding_path=root/aligned["original_sheet_value_binding_file"]
    if digest(binding_path)!=aligned["original_sheet_value_binding_sha256"]:raise LeakageError("Original modality value binding changed")
    value_bindings=validate_zaoren_row_bindings(data,binding_path)
    with (root/protocol["lineage"]).open(encoding="utf-8-sig",newline="") as h:lineage=[r for r in csv.DictReader(h) if r["dataset_id"]=="zaoren"]
    splits=json.loads((root/protocol["historical_splits"]).read_text());index={str(r):i for i,r in enumerate(ids)}
    modalities={};axes={};transforms={}
    for name,maximum in protocol["maximum_points"].items():
        if "X_"+name not in data:continue
        x=data["X_"+name];axis=data["axis_"+name]
        if len(x)!=855:raise LeakageError("A modality lost source records")
        assert_modalities_aligned(ids,{name:ids})
        if maximum is not None:x,axis,blocks=fixed_blocks(x,axis,maximum)
        else:blocks=[[j,j+1] for j in range(len(axis))]
        modalities[name]=x;axes[name]=axis;transforms[name]={"blocks_start_stop":blocks,"source_points":data["X_"+name].shape[1],"output_points":x.shape[1],"fit_statistics":False,"operation":"per-spectrum consecutive block means" if maximum else "unchanged coordinate samples"}
    eligible=[f for f in protocol["families"] if set(MODALITIES[f])<=set(modalities)]
    blocked={f:"missing complete paired modality; no sample deletion" for f in protocol["families"] if f not in eligible}
    if len(eligible)!=len(protocol["families"]):raise LeakageError("Predeclared family unavailable; revise before any fit: "+str(blocked))
    output.mkdir(parents=True,exist_ok=True);contexts=[]
    def pack(path,identifiers,source):
        positions=np.asarray([index[r] for r in identifiers]);payload={"record_ids":ids[positions],**{"X_"+m:x[positions] for m,x in modalities.items()}}
        if source:payload["source_labels"]=labels[positions]
        np.savez_compressed(path,**payload)
        return digest(path)
    def truth(path,identifiers):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=["record_id","label"],lineterminator="\n");w.writeheader();w.writerows({"record_id":r,"label":labels[index[r]]} for r in identifiers)
    for outer in splits:
        fold=outer["fold"]
        roles=[("inner",i,r["train_sample_ids"],r["validation_sample_ids"]) for i,r in enumerate(outer["inner_folds"])]
        roles.append(("outer",None,outer["train_sample_ids"],outer["test_sample_ids"]))
        for stage,inner,source_ids,inference_ids in roles:
            split_id=f"outer{fold:02d}."+(f"inner{inner:02d}" if stage=="inner" else "final")
            partitions={"source":source_ids,"inference":inference_ids}
            if stage=="inner":partitions["forbidden_outer_test"]=outer["test_sample_ids"]
            validate_fixed_partitions(partitions,expected_ids=ids)
            proof=validate_lineage_partitions(lineage,partitions,allow_user_declared=True)
            if proof["status"]=="BLOCKED":raise LeakageError("Unresolved source lineage blocks neural run")
            context=output/"inputs"/split_id;context.mkdir(parents=True)
            sp=context/"source.npz";ip=context/"inference.npz";tp=output/"scoring_truth"/(split_id+".csv")
            sh=pack(sp,source_ids,True);ih=pack(ip,inference_ids,False);truth(tp,inference_ids)
            contexts.append({"split_id":split_id,"stage":stage,"outer_fold":fold,"inner_fold":inner,
                "source_pack":str(sp),"source_pack_sha256":sh,"inference_pack":str(ip),"inference_pack_sha256":ih,
                "scoring_truth":str(tp),"source_ids":source_ids,"inference_ids":inference_ids,"lineage_status":proof["status"]})
    value={"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_id":protocol["protocol_id"],
        "input_database_sha256":digest(root/protocol["input"]),"input_alignment_sha256":digest(root/protocol["alignment"]),
        "lineage_sha256":digest(root/protocol["lineage"]),"historical_split_sha256":digest(root/protocol["historical_splits"]),
        "source_role":"separate preparation process may read labels for fixed split packaging", "contexts":contexts,
        "families":eligible,"blocked":blocked,"transforms":transforms,"axes":{k:v.tolist() for k,v in axes.items()},
        "labels_excluded_from_inference_packs":True,"study_status":"EXPLORATORY", "os_enforced_file_access_isolation":False,
        "original_modality_values_and_ids_revalidated":value_bindings,
        "labels_and_source_features_never_fed_as_filename_or_ID":True,"alignment_evidence_path":protocol["alignment"]}
    json_write(output/"preparation.json",value);return {"contexts":len(contexts),"families":len(eligible),"status":"EXPLORATORY"}


def manifest(protocol_path,output):
    root=Path(__file__).resolve().parents[2];prep=json.loads((Path(output)/"preparation.json").read_text())
    names=["src/herb_spectra/upgrade_neural.py","scripts/run_upgrade_neural.py","src/herb_spectra/leakage_guard.py","src/herb_spectra/selection.py","src/herb_spectra/classification.py","src/herb_spectra/multiplant_database.py"]
    return {"protocol_sha256":digest(protocol_path),"preparation_sha256":digest(Path(output)/"preparation.json"),
        "prepared_input_database_sha256":prep["input_database_sha256"],"source_sha256":{n:digest(root/n) for n in names},
        "versions":{k:importlib.metadata.version(k) for k in ("numpy","scipy","scikit-learn","torch","joblib")},
        "training_has_no_full_database_reader":True,"OS_enforced_label_access_prohibition":False}


def check_manifest(protocol_path,output):
    saved=json.loads((Path(output)/"run_manifest.json").read_text())
    if saved!=manifest(protocol_path,output):raise LeakageError("Frozen source/protocol/preparation changed")
    return saved


def event(output,name,**values):
    with (Path(output)/"stage_events.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps({"event":name,"utc":datetime.now(timezone.utc).isoformat(),**values})+"\n")


def trial_fit(context,family,profile,seed,output,code_version,outer=False):
    source=read_pack(context["source_pack"],source=True);inference=read_pack(context["inference_pack"])
    if digest(context["source_pack"])!=context["source_pack_sha256"]:raise LeakageError("Source pack changed")
    if digest(context["inference_pack"])!=context["inference_pack_sha256"]:raise LeakageError("Inference pack changed")
    directory=Path(output)/("evaluation" if outer else "selection")/context["split_id"]/family/profile["id"]/f"seed{seed}"
    if directory.exists():raise FileExistsError("Do not overwrite a trial; use another protocol")
    directory.mkdir(parents=True);start=time.perf_counter()
    fitted=fit_source(pack_modalities(source),source["source_labels"],source["record_ids"],family,profile,seed,
        context["split_id"],Path(output)/"fit_scope_events.jsonl",code_version)
    features=fitted.transform(pack_modalities(inference));n=len(inference["record_ids"]) if outer else min(7,len(inference["record_ids"]))
    pointwise=audit_pointwise_inference(fitted.model,{k:v[:n] for k,v in features.items()},batch_sizes=(1,3),atol=1e-5,rtol=1e-5)
    save_fitted(fitted,directory)
    common={"outer_fold":context["outer_fold"],"family":family,"architecture_family":family,"profile":profile["id"],"seed":seed}
    if not outer:common["inner_fold"]=context["inner_fold"]
    rows=prediction_rows(fitted,inference,common);write_predictions(directory/"predictions.csv.gz",rows)
    json_write(directory/"inference_audit.json",pointwise)
    json_write(directory/"context.json",{"split_id":context["split_id"],"source_pack_sha256":context["source_pack_sha256"],"inference_pack_sha256":context["inference_pack_sha256"],"prediction_file_sha256":digest(directory/"predictions.csv.gz"),"elapsed_seconds":time.perf_counter()-start,"outer_truth_read":False})
    result={**common,"split_id":context["split_id"],"model_directory":str(directory),"pointwise_audit":pointwise,
        "source_ids":source["record_ids"].tolist(),"inference_ids":inference["record_ids"].tolist(),
        "parameters":fitted.metadata["parameters"],"epochs":profile["epochs"],"elapsed_seconds":time.perf_counter()-start,
        "model_metadata_sha256":digest(directory/"training.json"),"predictions_sha256":digest(directory/"predictions.csv.gz")}
    return result,rows


def select(protocol_path,output,source_release_path):
    output=Path(output);protocol=json.loads(Path(protocol_path).read_text());prep=json.loads((output/"preparation.json").read_text())
    if (output/"selection_lock.json").exists():raise FileExistsError("Completed selection is immutable")
    current=manifest(protocol_path,output);release=json.loads(Path(source_release_path).read_text())
    if not release.get("allow_source_training") or release.get("protocol_sha256")!=digest(protocol_path):raise LeakageError("Root source-training authorization required")
    if release.get("preparation_sha256")!=digest(output/"preparation.json"):raise LeakageError("Source authorization must bind prepared contexts")
    if release.get("source_sha256")!=current["source_sha256"]:raise LeakageError("Source authorization must bind all implementation hashes")
    json_write(output/"run_manifest.json",current);json_write(output/"source_training_release_snapshot.json",release)
    event(output,"selection_started");rows=[];trials=[];choices=[];code_version=current["source_sha256"]["src/herb_spectra/upgrade_neural.py"]
    for context in [c for c in prep["contexts"] if c["stage"]=="inner"]:
        for family in prep["families"]:
            for profile in protocol["profiles"]:
                for seed in protocol["seeds"]:
                    trial,pred=trial_fit(context,family,profile,seed,output,code_version)
                    # Inner validation truth is allowed for source-only selection.
                    trial["metrics"]=score_predictions(pred,truth_map(context["scoring_truth"]));trials.append(trial);rows.extend(pred)
        json_write(output/"selection_progress.json",{"completed_inner_contexts":1+sum(c["stage"]=="inner" and c["split_id"]<context["split_id"] for c in prep["contexts"]),"trials":len(trials),"outer_scored":False})
        print(json.dumps({"stage":"selection","context":context["split_id"],"trials":len(trials)}),flush=True)
    for fold in range(5):
        choice={}
        for family in prep["families"]:
            means=[]
            for profile in protocol["profiles"]:
                subset=[r for r in trials if r["outer_fold"]==fold and r["family"]==family and r["profile"]==profile["id"]]
                if len(subset)!=2*len(protocol["seeds"]):raise LeakageError("Incomplete inner seed/fold coverage")
                means.append({"profile":profile,"selection_case_count":len(subset),"mean_inner_metrics":{k:float(np.mean([r["metrics"][k] for r in subset])) for k in ("macro_f1","balanced_accuracy","accuracy")}})
            selected=min(means,key=lambda r:(-r["mean_inner_metrics"]["macro_f1"],-r["mean_inner_metrics"]["balanced_accuracy"],-r["mean_inner_metrics"]["accuracy"],r["profile"]["id"]))
            choice[family]={"selected":selected["profile"],"means":means,"outer_test_used_to_select":False}
        choices.append({"outer_fold":fold,"choices":choice,"neural_strategy":select_neural_strategy(choice)})
    check_manifest(protocol_path,output)
    json_write(output/"inner_metrics.json",trials);write_predictions(output/"inner_predictions.csv.gz",rows)
    lock={"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(protocol_path),"manifest_sha256":digest(output/"run_manifest.json"),
        "inner_metrics_sha256":digest(output/"inner_metrics.json"),"inner_predictions_sha256":digest(output/"inner_predictions.csv.gz"),
        "outer_folds":choices,"candidate_profiles":protocol["profiles"],"seeds":protocol["seeds"],"primary_seed":42,"outer_test_used_to_select":False,"inner_cases":len(trials)}
    json_write(output/"selection_lock.json",lock);event(output,"selection_locked",selection_lock_sha256=digest(output/"selection_lock.json"))
    return {"inner_cases":len(trials),"inner_predictions":len(rows),"selection_lock_sha256":digest(output/"selection_lock.json"),"outer_scored":False}


def evaluate(protocol_path,output,release_path):
    """Fit permitted sources and save predictions; this phase never opens truth."""
    output=Path(output);protocol=json.loads(Path(protocol_path).read_text());prep=json.loads((output/"preparation.json").read_text());manifest_value=check_manifest(protocol_path,output)
    release=json.loads(Path(release_path).read_text());lock=json.loads((output/"selection_lock.json").read_text())
    for key,value in {"selection_lock_sha256":digest(output/"selection_lock.json"),"manifest_sha256":digest(output/"run_manifest.json"),"protocol_sha256":digest(protocol_path)}.items():
        if release.get(key)!=value:raise LeakageError("Outer authorization mismatch: "+key)
    if release.get("allow_outer_evaluation") is not True or not release.get("released_at_utc"):raise LeakageError("Dated external outer release required")
    if (output/"outer_predictions.csv.gz").exists():raise FileExistsError("Do not overwrite outer predictions")
    json_write(output/"outer_release_snapshot.json",release);event(output,"outer_inference_started");rows=[];trials=[]
    for context in [c for c in prep["contexts"] if c["stage"]=="outer"]:
        fold_lock=lock["outer_folds"][context["outer_fold"]];choices=fold_lock["choices"];policy=fold_lock["neural_strategy"]
        for family in prep["families"]:
            for seed in protocol["seeds"]:
                trial,pred=trial_fit(context,family,choices[family]["selected"],seed,output,manifest_value["source_sha256"]["src/herb_spectra/upgrade_neural.py"],True)
                trials.append(trial);rows.extend(pred)
                if family==policy["selected_family"]:
                    if choices[family]["selected"]!=policy["selected_profile"]:raise LeakageError("Locked neural policy does not alias the family-selected profile")
                    rows.extend({**r,"family":"inner_selected_neural"} for r in pred)
        print(json.dumps({"stage":"outer_inference","fold":context["outer_fold"],"method_seed_cases":len(trials)}),flush=True)
    check_manifest(protocol_path,output)
    json_write(output/"outer_inference_manifest.json",trials);write_predictions(output/"outer_predictions.csv.gz",rows)
    json_write(output/"prediction_lock.json",{"created_at_utc":datetime.now(timezone.utc).isoformat(),"outer_predictions_sha256":digest(output/"outer_predictions.csv.gz"),"actual_model_seed_fold_cases":len(trials),"aliased_policy_seed_fold_cases":15,"prediction_rows":len(rows),"truth_read_during_inference":False})
    event(output,"outer_predictions_locked");return {"cases":len(trials),"rows":len(rows),"truth_read":False}


def score(protocol_path,output):
    """Separate scorer joins true labels only after predictions are immutable."""
    output=Path(output);protocol=json.loads(Path(protocol_path).read_text());prep=json.loads((output/"preparation.json").read_text())
    lock=json.loads((output/"prediction_lock.json").read_text());assert digest(output/"outer_predictions.csv.gz")==lock["outer_predictions_sha256"]
    rows=read_predictions(output/"outer_predictions.csv.gz");all_truth={}
    for context in prep["contexts"]:
        if context["stage"]=="outer":
            t=truth_map(context["scoring_truth"])
            if set(all_truth)&set(t):raise LeakageError("A specimen has two outer truth records")
            all_truth.update(t)
    pooled={};folds=[];averages={}
    for family in [*prep["families"],"inner_selected_neural"]:
        pooled[family]={}
        for seed in protocol["seeds"]:
            selected=[r for r in rows if r["family"]==family and int(r["seed"])==seed]
            if len(selected)!=855:raise LeakageError("Each family/seed requires exactly855 OOF predictions")
            pooled[family][str(seed)]=score_predictions(selected,all_truth)
            for fold in range(5):
                found=[r for r in selected if int(r["outer_fold"])==fold];truth={r["record_id"]:all_truth[r["record_id"]] for r in found}
                folds.append({"family":family,"seed":seed,"outer_fold":fold,"metrics":score_predictions(found,truth)})
        averages[family]={metric:{"mean":float(np.mean([pooled[family][str(s)][metric] for s in protocol["seeds"]])),"sample_sd":float(np.std([pooled[family][str(s)][metric] for s in protocol["seeds"]],ddof=1)),"seeds":protocol["seeds"]} for metric in ("accuracy","macro_f1","balanced_accuracy")}
    json_write(output/"pooled_metrics_by_seed.json",pooled);json_write(output/"fold_metrics.json",folds);json_write(output/"seed_sensitivity.json",averages)
    json_write(output/"primary_seed42_metrics.json",{f:pooled[f]["42"] for f in pooled})
    json_write(output/"scoring_manifest.json",{"scored_at_utc":datetime.now(timezone.utc).isoformat(),"prediction_lock_sha256":digest(output/"prediction_lock.json"),"truth_paths":[c["scoring_truth"] for c in prep["contexts"] if c["stage"]=="outer"],"n_real_specimens":855,"seeds_are_independent_specimens":False,"scope":"EXPLORATORY exposed single-batch internal development","new_model_fits":0})
    event(output,"independent_scoring_completed");return {"primary_seed42":{f:{k:v for k,v in pooled[f]["42"].items() if k in ("accuracy","macro_f1","balanced_accuracy","correct")} for f in pooled},"seed_sensitivity":averages}
