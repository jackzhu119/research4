"""Read-only direct-autograd and table replay; no optimizers or model fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from herb_spectra.upgrade_neural import load_fitted,pack_modalities,read_pack,state_hash


ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent


def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    protocol=json.loads((DEST/"protocol_lock.json").read_text());checks=0;maximum=0.
    source=read_pack(protocol["source_pack"],source=True);source_ids=source["record_ids"].astype(str)
    chosen=rows(DEST/"balanced_source_selection.csv");chosen_ids=[r["record_id"] for r in chosen]
    expected=[]
    for group in sorted(set(source["source_labels"].astype(str))):
        expected.extend(sorted(source_ids[source["source_labels"]==group].tolist())[:3])
    assert chosen_ids==expected and len(set(chosen_ids))==21;checks+=2
    index={rid:i for i,rid in enumerate(source_ids)};positions=np.asarray([index[rid] for rid in chosen_ids])
    split=json.loads((ROOT/"reports/staged-nested-optimization/2026-10-31/core/splits.json").read_text())[0]
    assert set(chosen_ids)<=set(split["train_sample_ids"]) and not set(chosen_ids)&set(split["test_sample_ids"]);checks+=1
    raw=pack_modalities(source)
    for entry in protocol["models"]:
        directory=Path(entry["directory"])
        assert hashlib.sha256((directory/"pipeline.joblib").read_bytes()).hexdigest()==entry["pipeline_sha256"];checks+=1
        pipe=load_fitted(directory,expected_split_id="outer00.final",expected_source_ids=source_ids)
        features=pipe.transform({k:v[positions] for k,v in raw.items()});pipe.model.eval()
        before=state_hash(pipe.model)
        param_grad=[None if p.grad is None else p.grad.detach().cpu().clone() for p in pipe.model.parameters()]
        inputs={k:torch.tensor(v,dtype=torch.float32) for k,v in features.items()};inputs["alpha"].requires_grad_(True)
        logits=pipe.model(inputs);predicted=logits.detach().argmax(1)
        target=logits[torch.arange(21),predicted]
        gradient=torch.autograd.grad(target.sum(),inputs["alpha"])[0].detach().numpy()
        absolute=np.abs(features["alpha"]*gradient);sums=absolute.sum(1,keepdims=True)
        normalized=np.divide(absolute,sums,out=np.zeros_like(absolute),where=sums>0)
        folder=DEST/entry["family"]
        with np.load(folder/"input_gradient_arrays.npz",allow_pickle=False) as archive:
            for key,expected in [("alpha_standardized",features["alpha"]),("input_gradient",gradient),
                                 ("absolute_input_times_gradient",absolute),("row_l1_normalized_attribution",normalized),
                                 ("predicted_class_indices",predicted.numpy()),("predicted_logits",target.detach().numpy())]:
                found=archive[key];difference=float(np.max(np.abs(found-expected)))
                maximum=max(maximum,difference);np.testing.assert_allclose(found,expected,atol=1e-7,rtol=1e-5);checks+=found.size
        profiles=rows(folder/"sample_attribution_profiles.csv")
        assert [r["record_id"] for r in profiles]==chosen_ids;checks+=1
        csv_abs=np.asarray([[float(row[f"abs_input_gradient_{j:03d}"]) for j in range(612)] for row in profiles])
        np.testing.assert_array_equal(csv_abs,absolute.astype(float));checks+=absolute.size
        mean_rows=rows(folder/"mean_alpha_attribution_612.csv")
        for key,expected in [("mean_absolute_input_times_gradient",absolute.mean(0)),
                             ("mean_row_l1_normalized_attribution",normalized.mean(0))]:
            found=np.asarray([float(row[key]) for row in mean_rows]);np.testing.assert_array_equal(found,expected.astype(float));checks+=len(found)
        assert state_hash(pipe.model)==before;checks+=1
        for p,prior in zip(pipe.model.parameters(),param_grad,strict=True):
            assert p.grad is None if prior is None else p.grad is not None and torch.equal(p.grad.detach().cpu(),prior)
            checks+=1
    report={"status":"TECHNICAL_CHECK_PASS_FOR_POSTHOC_SOURCE_DESCRIPTION","checks":checks,
            "maximum_gradient_numeric_difference":maximum,"models":3,"unique_source_samples":21,
            "source_gradient_profiles":63,"new_model_fits":0,"parameter_updates":0,
            "parameter_gradients_and_batchnorm_state_unchanged":True,"test_samples_read":False,
            "chemical_assignment_established":False,"feedback_to_model_selection":False,
            "audit_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (DEST/"independent_attribution_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
