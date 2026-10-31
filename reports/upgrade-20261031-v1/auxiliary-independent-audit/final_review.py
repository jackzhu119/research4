"""Replay frozen auxiliary outer models and independently recompute all metrics."""
from collections import defaultdict
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_common import Auditor,ROOT,RUN,DEST,PROTOCOL,load,sha,predictions,metrics


def case_key(row):return row["dataset_id"],int(row["outer_fold"]),row["family"],int(row["seed"])


def utc(value):return datetime.fromisoformat(value.replace("Z","+00:00"))


def main():
    target=DEST/"final_review.json"
    if target.exists():raise FileExistsError("Preserve completed review")
    a=Auditor();prior=load(DEST/"selection_review.json")
    protocol,prep,manifest,selection=(load(p) for p in (PROTOCOL,RUN/"preparation.json",RUN/"run_manifest.json",RUN/"selection_lock.json"))
    a.require(prior["allow_outer_evaluation"] is True,"Inner review missing")
    for path,key in ((PROTOCOL,"protocol_sha256"),(RUN/"selection_lock.json","selection_lock_sha256"),(RUN/"run_manifest.json","run_manifest_sha256"),(DEST/"audit_common.py","audit_common_sha256")):
        a.require(sha(path)==prior[key],"Reviewed immutable evidence changed: "+str(path))
    for relative,expected in manifest["source_sha256"].items():a.require(sha(ROOT/relative)==expected,"Source implementation changed "+relative)
    release=load(RUN/"outer_release_snapshot.json")
    a.require(release["allow_outer_evaluation"] is True and release["selection_lock_sha256"]==sha(RUN/"selection_lock.json") and release["protocol_sha256"]==sha(PROTOCOL) and release["manifest_sha256"]==sha(RUN/"run_manifest.json"),"Outer release binding")
    lock=load(RUN/"prediction_lock.json")
    for path,key in ((RUN/"outer_predictions.csv.gz","prediction_sha256"),(RUN/"outer_model_records.json","outer_model_records_sha256"),(RUN/"selection_lock.json","selection_lock_sha256"),(RUN/"run_manifest.json","manifest_sha256"),(RUN/"preparation.json","preparation_sha256"),(RUN/"outer_release_snapshot.json","root_outer_release_sha256"),(PROTOCOL,"protocol_sha256")):
        a.require(sha(path)==lock[key],"Prediction lock changed: "+str(path))
    a.require(lock["outer_truth_parsed"] is False and lock["source_sha256"]==manifest["source_sha256"],"Inference permitted true-label/code changes")
    events=[json.loads(line) for line in (RUN/"stage_events.jsonl").read_text().splitlines()]
    times={r["event"]:utc(r["utc"]) for r in events}
    a.require(times["selection_locked"] < utc(release["released_at_utc"]) <= times["outer_inference_started"] < utc(lock["created_at_utc"]) <= times["outer_inference_completed"] < times["independent_scoring_completed"],"Selection/release/prediction/score order")
    contexts={c["split_id"]:c for c in prep["contexts"] if c["stage"]=="outer"}
    records=load(RUN/"outer_model_records.json");rows=predictions(RUN/"outer_predictions.csv.gz")
    a.require(len(records)==len({case_key(r) for r in records})==180,"Actual model budget")
    a.require(len(rows)==16968,"Prediction role/seed denominator")
    a.require(lock["prediction_rows"]==len(rows) and lock["fitted_model_count"]==len(records),"Saved role counts")
    grouped=defaultdict(list)
    for row in rows:
        a.require(not set(row)&{"y","label","y_true","truth","is_correct","source_labels"},"Unscored prediction file carries truth")
        grouped[case_key(row)].append(row)
    a.require(len(grouped)==210,"Six actual families and one policy alias per seed/fold/plant")
    truths={d:{} for d in protocol["datasets"]};fold_truth={}
    for c in contexts.values():
        _,_,mapping=a.source(c);d=c["dataset_id"]
        a.require(not set(mapping)&set(truths[d]),"Repeated OOF specimen")
        truths[d].update(mapping);fold_truth[(d,c["outer_fold"])]=mapping
    a.require({d:len(v) for d,v in truths.items()}==protocol["sample_counts"],"Independent plant denominators")
    for i,r in enumerate(records):
        c=contexts[r["metadata"]["split_id"]]
        choice=next(v for v in selection["outer_folds"] if v["dataset_id"]==r["dataset_id"] and v["outer_fold"]==r["outer_fold"])
        a.require(r["candidate"]==choice["choices"][r["family"]]["selected"],"Outer model differs from source-selected candidate")
        a.require(r["source_ids"]==c["source_ids"] and r["inference_ids"]==c["inference_ids"],"Outer source/inference identifiers changed")
        own=predictions(ROOT/r["predictions"])
        other="TM" if r["dataset_id"]=="huangqi" else "HQ"
        extra={"score_"+other+f"{j:02d}" for j in range(1,5)}
        a.require(all(not (set(row)&extra) for row in own),"Per-task trial contains other-plant scores")
        normalized=[{**row,"architecture_family":r["family"],**{k:"" for k in extra}} for row in own]
        a.require(normalized==grouped[case_key(r)],"Combined actual model predictions differ from saved trial")
        a.model(r,c,own)
        if (i+1)%30==0:print(json.dumps({"independent_outer_models_replayed":i+1}),flush=True)
    for choice in selection["outer_folds"]:
        d,fold=choice["dataset_id"],choice["outer_fold"];selected=choice["inner_selected_neural"]
        for seed in protocol["seeds"]:
            expected=[{**r,"family":"inner_selected_neural","architecture_family":selected["family"]} for r in grouped[(d,fold,selected["family"],seed)]]
            a.require(grouped[(d,fold,"inner_selected_neural",seed)]==expected,"Policy alias differs from frozen selected architecture")
    saved=load(RUN/"pooled_metrics.json");seed_summary=load(RUN/"seed_summary.json");primary=load(RUN/"primary_seed42_metrics.json")
    independent={};table=[];class_table=[]
    for d in protocol["datasets"]:
        independent[d]={};classes=tuple(protocol["class_orders"][d])
        for family in [*protocol["candidates"],"inner_selected_neural"]:
            independent[d][family]={}
            for seed in protocol["seeds"]:
                own=[r for r in rows if r["dataset_id"]==d and r["family"]==family and int(r["seed"])==seed]
                value=metrics(own,truths[d],classes);a.metric_match(value,saved[d][family][str(seed)],"Independent pooled confusion counts")
                independent[d][family][str(seed)]=value
                table.append({"dataset_id":d,"family":family,"seed":seed,"correct":value["correct"],"samples":value["total"],"accuracy":value["accuracy"],"macro_f1":value["macro_f1"],"balanced_accuracy":value["balanced_accuracy"]})
                for cls in classes:class_table.append({"dataset_id":d,"family":family,"seed":seed,"class":cls,**value["per_class"][cls]})
                if seed==42:a.metric_match(value,primary[d][family],"Primary seed42 table")
            summary=next(r for r in seed_summary if r["dataset_id"]==d and r["family"]==family)
            a.require(summary["samples_per_seed"]==protocol["sample_counts"][d] and summary["seeds"]==protocol["seeds"],"Seed summary denominator")
            for m in ("accuracy","macro_f1","balanced_accuracy"):
                values=[independent[d][family][str(s)][m] for s in protocol["seeds"]]
                a.close(summary[m+"_mean"],np.mean(values),"Three-seed descriptive mean",atol=1e-14,rtol=1e-13)
                a.close(summary[m+"_sample_sd"],np.std(values,ddof=1),"Three-seed descriptive SD",atol=1e-14,rtol=1e-13)
    scored=predictions(RUN/"scored_outer_predictions.csv.gz")
    a.require(len(scored)==len(rows),"Scored row count")
    for r,original in zip(scored,rows,strict=True):
        expected={**original,"y_true":truths[original["dataset_id"]][original["record_id"]],"is_correct":str(int(original["y_predicted"]==truths[original["dataset_id"]][original["record_id"]]))}
        a.require(r==expected,"Scored prediction correctness flag/true label/order")
    scope=(RUN/"fit_scope_events.jsonl").read_bytes();prefix=scope[:prior["scope_prefix_bytes"]]
    a.require(hashlib.sha256(prefix).hexdigest()==prior["scope_prefix_sha256"],"Original source scope prefix changed")
    suffix=scope[prior["scope_prefix_bytes"]:]
    with gzip.open(DEST/"outer_fit_scope_suffix.jsonl.gz","wb") as handle:handle.write(suffix)
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as handle:
        handle.write(suffix);handle.flush();scope_summary=a.scope_log(Path(handle.name),records)
    for name,values in (("model_metrics_by_seed.csv",table),("class_metrics_by_seed.csv",class_table)):
        with (DEST/name).open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]),lineterminator="\n");writer.writeheader();writer.writerows(values)
    result={"status":"TECHNICAL_AUXILIARY_OUTER_REPLAY_PASS_EXPLORATORY_DATA","completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "checks":a.checks,"maximum_numeric_difference":a.max_difference,"actual_outer_models_replayed":a.models_replayed,
        "SVC_models_replayed":a.svm_replayed,"neural_models_replayed":a.neural_replayed,"MC_selectors_recomputed_from_saved_coefficients":a.mc_selectors_recomputed,
        "actual_model_roles":180,"aliased_policy_roles":30,"raw_prediction_rows":len(rows),"pooled_plant_method_seed_cases":42,
        "real_specimens_per_plant_method_seed":protocol["sample_counts"],"seeds_expand_real_sample_count":False,"inner_scope_prefix_unchanged":True,
        "outer_fit_scope":scope_summary,"new_estimator_fits":0,"new_calibrator_fits":0,"independent_metrics":independent,
        "selection_lock_sha256":sha(RUN/"selection_lock.json"),"prediction_lock_sha256":sha(RUN/"prediction_lock.json"),
        "outer_predictions_sha256":sha(RUN/"outer_predictions.csv.gz"),"outer_release_sha256":sha(RUN/"outer_release_snapshot.json"),
        "outer_model_records_sha256":sha(RUN/"outer_model_records.json"),"pooled_metrics_sha256":sha(RUN/"pooled_metrics.json"),
        "scored_outer_predictions_sha256":sha(RUN/"scored_outer_predictions.csv.gz"),"stage_events_sha256":sha(RUN/"stage_events.jsonl"),
        "run_manifest_sha256":sha(RUN/"run_manifest.json"),"protocol_sha256":sha(PROTOCOL),"preparation_sha256":sha(RUN/"preparation.json"),
        "selection_review_sha256":sha(DEST/"selection_review.json"),"source_review_v3_sha256":sha(DEST/"source_review_v3.json"),
        "audit_script_sha256":sha(Path(__file__)),"audit_common_sha256":sha(DEST/"audit_common.py"),
        "limits":["Exploratory source-file-group classification, not authenticated growth origin confirmation","User declared separate samples; unknown shared upstream material and batch identities","Independent classifier tasks do not establish cross-plant transfer","Different modality pairs and neural budgets limit causal attribution","Saved/API checks are not arbitrary-code or OS isolation proof","SVC outputs are decision scores with standard vote, not calibrated probabilities"]}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},ensure_ascii=False))


if __name__=="__main__":main()
