"""Post-scoring source-only input-times-gradient descriptions of frozen models."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from herb_spectra.leakage_guard import FitScope,LeakageError
from herb_spectra.upgrade_neural import load_fitted,pack_modalities,read_pack,state_hash


ROOT=Path(__file__).resolve().parents[1]
FAMILIES=("cnn_alpha","se_alpha","fusion_all_four")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def csv_write(path,rows,fields):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def select_source_indices(record_ids,source_labels,allowed_source_ids,per_class=3):
    """Balanced ID rule, not a selection by correctness, loss or attribution."""
    ids=np.asarray(record_ids).astype(str);labels=np.asarray(source_labels).astype(str)
    if len(ids)!=len(labels) or len(set(ids.tolist()))!=len(ids) or set(ids)!=set(allowed_source_ids):
        raise LeakageError("Source identity/label mapping must match the frozen outer source")
    indices=[]
    for label in sorted(set(labels)):
        group=sorted(np.flatnonzero(labels==label).tolist(),key=lambda j:ids[j])
        if len(group)<per_class:
            raise LeakageError("Insufficient source specimens for the locked balanced description")
        indices.extend(group[:per_class])
    return np.asarray(indices,dtype=int)


def source_alpha_attribution(model,features,sample_ids,scope):
    """Explain each predicted logit; no truth/query/optimizer enters this API."""
    if "alpha" not in features:
        raise LeakageError("Alpha branch is required for this input attribution")
    arrays={str(k):np.asarray(v,dtype=np.float32) for k,v in features.items()}
    if any(v.ndim!=2 or not np.isfinite(v).all() for v in arrays.values()):
        raise LeakageError("Finite aligned standardized input matrices required")
    with scope.operation("posthoc_source_alpha_input_gradient",sample_ids,X=arrays) as proof:
        model.eval()
        before=state_hash(model)
        params=list(model.parameters())
        gradients_before=[None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
        device=next(model.parameters(),torch.empty(0)).device
        tensors={k:torch.as_tensor(v,dtype=torch.float32,device=device).detach().clone() for k,v in arrays.items()}
        tensors["alpha"].requires_grad_(True)
        with torch.enable_grad():
            logits=model(tensors)
            if logits.ndim!=2 or len(logits)!=len(sample_ids):
                raise LeakageError("Attribution requires aligned class logits")
            predicted=logits.detach().argmax(dim=1)
            chosen=logits[torch.arange(len(sample_ids),device=device),predicted]
            gradient=torch.autograd.grad(chosen.sum(),tensors["alpha"],create_graph=False,retain_graph=False)[0]
        alpha=tensors["alpha"].detach().cpu().numpy()
        signed=gradient.detach().cpu().numpy()
        attribution=np.abs(alpha*signed)
        denominators=attribution.sum(axis=1,keepdims=True)
        normalized=np.divide(attribution,denominators,out=np.zeros_like(attribution),where=denominators>0)
        if not np.isfinite(signed).all() or not np.isfinite(attribution).all():
            raise LeakageError("Nonfinite attribution cannot be reported")
        if state_hash(model)!=before:
            raise LeakageError("Source interpretation changed parameters or BatchNorm buffers")
        for parameter,prior in zip(params,gradients_before,strict=True):
            if prior is None:
                if parameter.grad is not None:
                    raise LeakageError("Input attribution accumulated a new parameter gradient")
            elif parameter.grad is None or not torch.equal(parameter.grad.detach().cpu(),prior):
                raise LeakageError("Input attribution changed an existing parameter gradient")
        if model.training or any(module.training for module in model.modules()):
            raise LeakageError("Source attribution must keep every module in eval state")
    return {"alpha_standardized":alpha,"input_gradient":signed,"absolute_input_times_gradient":attribution,
            "row_l1_normalized_attribution":normalized,"predicted_class_indices":predicted.cpu().numpy(),
            "predicted_logits":chosen.detach().cpu().numpy(),"fit_provenance":dict(proof),
            "state_sha256_before":before,"state_sha256_after":state_hash(model),
            "parameter_gradients_unchanged":True,"parameter_gradient_non_none_count_before":sum(g is not None for g in gradients_before),
            "optimizer_created":False,"true_label_used_for_gradient":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-run",type=Path,default=ROOT/"outputs/upgrade-20261031-v1/neural")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Source descriptions must use a fresh directory")
    root=args.neural_run
    lock=json.loads((root/"selection_lock.json").read_text())
    choice=lock["outer_folds"][0]
    assert choice["outer_fold"]==0 and choice["neural_strategy"]["selected_family"]=="fusion_all_four"
    preparation=json.loads((root/"preparation.json").read_text())
    context=next(c for c in preparation["contexts"] if c["split_id"]=="outer00.final")
    source_path=Path(context["source_pack"])
    assert sha(source_path)==context["source_pack_sha256"]
    source=read_pack(source_path,source=True);source_ids=source["record_ids"].astype(str)
    assert source_ids.tolist()==context["source_ids"] and len(source_ids)==684
    old_splits=json.loads((ROOT/"reports/staged-nested-optimization/2026-10-31/core/splits.json").read_text())
    assert source_ids.tolist()==old_splits[0]["train_sample_ids"]
    chosen=select_source_indices(source_ids,source["source_labels"],context["source_ids"],3)
    chosen_ids=source_ids[chosen].tolist();assert len(chosen_ids)==21
    assert not set(chosen_ids)&set(old_splits[0]["test_sample_ids"])
    alpha_axis=np.asarray(preparation["axes"]["alpha"],dtype=float);assert alpha_axis.shape==(612,)
    source_paths=[Path(__file__),ROOT/"src/herb_spectra/upgrade_neural.py",ROOT/"src/herb_spectra/leakage_guard.py"]
    code={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    model_entries=[]
    for family in FAMILIES:
        profile=choice["choices"][family]["selected"]["id"]
        directory=root/"evaluation/outer00.final"/family/profile/"seed42"
        model_entries.append({"family":family,"profile":profile,"directory":str(directory),
                              "pipeline_sha256":sha(directory/"pipeline.joblib"),"training_sha256":sha(directory/"training.json")})
    args.output.mkdir(parents=True,exist_ok=True)
    protocol={"protocol_id":"upgrade-source-alpha-attribution-20261031-v1","recorded_at_utc":datetime.now(timezone.utc).isoformat(),
              "timing":"post-scoring source-only description; never fed back to this model version",
              "families":list(FAMILIES),"family_basis":"prespecified CNN/SE plus inner-selected fusion_all_four, not outer-score ranking",
              "seed":42,"outer_fold":0,"source_population":684,"source_selected":21,
              "sampling_rule":"three lexicographically first source record IDs per source label; no correctness/score filtering",
              "target":"each model's own predicted class logit","attribution":"absolute standardized-alpha times input gradient",
              "normalization":"optional per-row L1 for locations only, not between-model scale or modality importance",
              "source_pack":str(source_path),"source_pack_sha256":sha(source_path),"selection_lock_sha256":sha(root/"selection_lock.json"),
              "preparation_sha256":sha(root/"preparation.json"),"source_sha256":code,"models":model_entries,
              "true_labels_used_only_for_balanced_source_sampling":True,"query_values_or_labels_read":False,
              "new_model_fits":0,"model_parameters_updated":False,"DFT_candidates":"NOT_RUN; no chemical assignment"}
    write(args.output/"protocol_lock.json",protocol)
    for path in source_paths:
        target=args.output/"source_snapshot"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    selection=[{"record_id":source_ids[j],"source_group_for_sampling_only":str(source["source_labels"][j]),
                "source_population_index":int(j),"outer_fold":0,"never_used_as_query":True} for j in chosen]
    csv_write(args.output/"balanced_source_selection.csv",selection,list(selection[0]))
    raw_modalities=pack_modalities(source);summary=[]
    fig,axes=plt.subplots(3,1,figsize=(9.2,9.2),sharex=True)
    for axis_plot,entry in zip(axes,model_entries,strict=True):
        family=entry["family"];directory=Path(entry["directory"])
        fitted=load_fitted(directory,expected_split_id="outer00.final",expected_source_ids=source_ids)
        for name,expected in fitted.metadata["raw_source_modality_fingerprints"].items():
            assert joblib.hash(raw_modalities[name])==expected,"Frozen source values changed"
        features=fitted.transform({k:v[chosen] for k,v in raw_modalities.items()})
        scope=FitScope(context["source_ids"],"outer00.source_attribution","2026-10-31-zaoren-numbered-v1",
                       args.output/"fit_scope_manifest.jsonl",code_version=code[str(Path(__file__).relative_to(ROOT))])
        result=source_alpha_attribution(fitted.model,features,chosen_ids,scope)
        assert result["state_sha256_after"]==fitted.metadata["fitted_state_sha256"]
        assert sha(directory/"pipeline.joblib")==entry["pipeline_sha256"]
        target=args.output/family;target.mkdir()
        array_keys=[k for k,v in result.items() if isinstance(v,np.ndarray)]
        np.savez_compressed(target/"input_gradient_arrays.npz",**{k:result[k] for k in array_keys})
        data_rows=[]
        for i,rid in enumerate(chosen_ids):
            row={"record_id":rid,"predicted_class":fitted.metadata["class_order"][int(result["predicted_class_indices"][i])],
                 "predicted_logit":float(result["predicted_logits"][i])}
            row.update({f"abs_input_gradient_{j:03d}":float(v) for j,v in enumerate(result["absolute_input_times_gradient"][i])})
            data_rows.append(row)
        csv_write(target/"sample_attribution_profiles.csv",data_rows,list(data_rows[0]))
        means=result["absolute_input_times_gradient"].mean(axis=0)
        normalized=result["row_l1_normalized_attribution"].mean(axis=0)
        mean_rows=[{"frequency_index":j,"frequency_thz":float(f),"mean_absolute_input_times_gradient":float(a),
                    "mean_row_l1_normalized_attribution":float(n)} for j,(f,a,n) in enumerate(zip(alpha_axis,means,normalized,strict=True))]
        csv_write(target/"mean_alpha_attribution_612.csv",mean_rows,list(mean_rows[0]))
        metadata={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        metadata.update(family=family,profile=entry["profile"],source_sample_ids=chosen_ids,source_count=21,
                        source_population_ids=context["source_ids"],model_pipeline_sha256=entry["pipeline_sha256"],
                        no_classifier_selection_feedback=True,other_modalities_fixed=family=="fusion_all_four")
        write(target/"attribution_state_audit.json",metadata)
        axis_plot.plot(alpha_axis,normalized,color={"cnn_alpha":"#155E75","se_alpha":"#9A3412","fusion_all_four":"#4D7C0F"}[family],lw=1.2)
        axis_plot.set(title=f"{family}: 21 balanced source spectra, predicted-logit saliency",ylabel="Mean row-normalized\n|input x gradient|")
        axis_plot.set_ylim(bottom=0);axis_plot.grid(alpha=.15)
        summary.append({"family":family,"source_count":21,"frequency_points":612,"parameter_gradients_unchanged":True,
                        "model_state_unchanged":True,"pipeline_sha256":entry["pipeline_sha256"],
                        "attribution_raw_sum_mean":float(result["absolute_input_times_gradient"].sum(1).mean()),
                        "zero_attribution_rows":int(np.count_nonzero(result["absolute_input_times_gradient"].sum(1)==0))})
    axes[-1].set_xlabel("Frequency (THz, inherited axis)")
    fig.suptitle("Post-scoring SOURCE descriptions; no chemical assignment or model feedback",fontsize=11)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(args.output/"source_alpha_attributions.png",dpi=220)
    fig.savefig(args.output/"source_alpha_attributions.pdf");plt.close(fig)
    for p,h in code.items():assert sha(ROOT/p)==h
    for entry in model_entries:assert sha(Path(entry["directory"])/"pipeline.joblib")==entry["pipeline_sha256"]
    write(args.output/"completed.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),
          "status":"EXPLORATORY_POSTHOC_SOURCE_DESCRIPTION","families":summary,"unique_selected_source_records":21,
          "true_labels_used_for_gradient":False,"new_model_fits":0,"trained_parameter_updates":0,
          "theoretical_assignments":0,"current_model_feedback":False})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
