"""Replay all auxiliary source models and independently lock inner choices."""
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_common import Auditor,ROOT,RUN,DEST,PROTOCOL,load,sha,predictions


def case_key(row):
    return row["dataset_id"],int(row["outer_fold"]),int(row["inner_fold"]),row["family"],row["candidate_id"],int(row["seed"])


def main():
    target=DEST/"selection_review.json"
    if target.exists():raise FileExistsError("Preserve completed review")
    a=Auditor();source_review=load(DEST/"source_review_v3.json")
    a.require(source_review["allow_source_training"] is True,"Source review missing")
    protocol,prep,manifest,lock=(load(p) for p in (PROTOCOL,RUN/"preparation.json",RUN/"run_manifest.json",RUN/"selection_lock.json"))
    a.require(sha(PROTOCOL)==source_review["protocol_sha256"]==manifest["protocol_sha256"],"Protocol changed")
    a.require(sha(RUN/"preparation.json")==source_review["preparation_sha256"]==manifest["preparation_sha256"],"Input preparation changed")
    a.require(sha(RUN/"candidate_manifest.json")==source_review["candidate_manifest_sha256"],"Candidate manifest changed")
    a.require(manifest==load(RUN/"candidate_manifest.json"),"Frozen run/source candidates differ")
    for relative,expected in manifest["source_sha256"].items():a.require(sha(ROOT/relative)==expected,"Source implementation changed "+relative)
    release=load(RUN/"source_release_snapshot.json")
    a.require(release["allow_source_training"] is True and all(release[k]==manifest[k] for k in ("protocol_sha256","preparation_sha256","source_sha256")),"Source release binding")
    a.require(not (RUN/"outer_predictions.csv.gz").exists() and not list(RUN.glob("evaluation/**/pipeline.joblib")),"Inner review must precede actual outer models/predictions")
    a.require(sha(RUN/"run_manifest.json")==lock["manifest_sha256"] and sha(PROTOCOL)==lock["protocol_sha256"],"Selection lock context")
    a.require(sha(RUN/"inner_metrics.json")==lock["inner_metrics_sha256"] and sha(RUN/"inner_predictions.csv.gz")==lock["inner_predictions_sha256"],"Inner immutable evidence")
    a.require(lock["outer_test_used_to_select"] is False and lock["inner_trials"]==1200,"Outer selection contamination or changed budget")
    records=load(RUN/"inner_metrics.json");rows=predictions(RUN/"inner_predictions.csv.gz")
    a.require(len(records)==len({case_key(r) for r in records})==1200,"Actual source trial coverage")
    a.require(len(rows)==193920,"Inner prediction count")
    grouped=defaultdict(list)
    for row in rows:
        a.require(not set(row)&{"y","label","y_true","truth","is_correct"},"Inner inference file carries truth")
        grouped[case_key(row)].append(row)
    contexts={c["split_id"]:c for c in prep["contexts"] if c["stage"]=="inner"}
    independent=[]
    for i,record in enumerate(records):
        c=contexts[record["metadata"]["split_id"]]
        a.require(record["source_ids"]==c["source_ids"] and record["inference_ids"]==c["inference_ids"],"Trial role identifiers changed")
        a.require(record["candidate"] in protocol["candidates"][record["family"]] and record["seed"] in protocol["seeds"],"Unregistered model candidate/seed")
        own=predictions(ROOT/record["predictions"])
        other="TM" if record["dataset_id"]=="huangqi" else "HQ"
        extra={"score_"+other+f"{j:02d}" for j in range(1,5)}
        a.require(all(not (set(row)&extra) for row in own),"Per-task trial contains other-plant scores")
        normalized=[{**row,**{k:"" for k in extra}} for row in own]
        a.require(normalized==grouped[case_key(record)],"Combined inner file differs from saved trial plus empty other-task CSV columns")
        value=a.model(record,c,own);a.metric_match(value,record["metrics"],"Independent inner metrics")
        independent.append({**{k:record[k] for k in ("dataset_id","outer_fold","inner_fold","family","candidate_id","seed")},"metrics":value})
        if (i+1)%120==0:print(json.dumps({"independent_source_models_replayed":i+1}),flush=True)
    expected_cases={(d,f,j,family,c["id"],seed) for d in protocol["datasets"] for f in range(5) for j in range(2) for family,candidates in protocol["candidates"].items() for c in candidates for seed in protocol["seeds"]}
    a.require(set(grouped)==expected_cases,"Missing or extra candidate/seed/fold roles")
    checked_choices=0;primary=[]
    for choice in lock["outer_folds"]:
        d,fold=choice["dataset_id"],choice["outer_fold"];means={}
        for family,candidates in protocol["candidates"].items():
            means[family]=[]
            for candidate in candidates:
                subset=[r for r in independent if r["dataset_id"]==d and r["outer_fold"]==fold and r["family"]==family and r["candidate_id"]==candidate["id"]]
                a.require(len(subset)==6 and {(r["inner_fold"],r["seed"]) for r in subset}=={(j,s) for j in range(2) for s in protocol["seeds"]},"Inner aggregation lacks fixed fold/seed")
                means[family].append({"candidate":candidate,"cases":6,"mean_inner_metrics":{m:float(np.mean([r["metrics"][m] for r in subset])) for m in ("macro_f1","balanced_accuracy","accuracy")}})
            saved=choice["choices"][family]
            a.require(saved["means"]==means[family],"Independent candidate inner means")
            best=min(means[family],key=lambda r:(-r["mean_inner_metrics"]["macro_f1"],-r["mean_inner_metrics"]["balanced_accuracy"],-r["mean_inner_metrics"]["accuracy"],r["candidate"]["id"]))
            a.require(saved["selected"]==best["candidate"] and saved["mean_inner_metrics"]==best["mean_inner_metrics"] and saved["outer_test_used_to_select"] is False,"Independent family selection")
            checked_choices += 1
        best=min(({"family":family,**row} for family in ("cnn","resnet","se","fusion") for row in means[family]),key=lambda r:(-r["mean_inner_metrics"]["macro_f1"],-r["mean_inner_metrics"]["balanced_accuracy"],-r["mean_inner_metrics"]["accuracy"],r["family"],r["candidate"]["id"]))
        selected=choice["inner_selected_neural"]
        a.require(selected["family"]==best["family"] and selected["candidate"]==best["candidate"] and selected["mean_inner_metrics"]==best["mean_inner_metrics"] and selected["outer_test_used_to_select"] is False,"Independent overall neural strategy")
        primary.append({"dataset_id":d,"outer_fold":fold,"family":best["family"],"candidate_id":best["candidate"]["id"]})
    a.require(len(lock["outer_folds"])==len({(c["dataset_id"],c["outer_fold"]) for c in lock["outer_folds"]})==10,"Missing outer decisions")
    scope_path=RUN/"fit_scope_events.jsonl";scope=a.scope_log(scope_path,records);scope_bytes=scope_path.read_bytes()
    with gzip.open(DEST/"selection_fit_scope_prefix.jsonl.gz","wb") as handle:handle.write(scope_bytes)
    result={"status":"TECHNICAL_AUXILIARY_INNER_REVIEW_PASS_EXPLORATORY_DATA","allow_outer_evaluation":True,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":a.checks,"maximum_numeric_difference":a.max_difference,
        "models_replayed":a.models_replayed,"SVC_models_replayed":a.svm_replayed,"neural_models_replayed":a.neural_replayed,
        "MC_selectors_recomputed_from_saved_coefficients":a.mc_selectors_recomputed,"inner_prediction_rows":len(rows),
        "family_candidate_choices_checked":checked_choices,"overall_neural_policy_choices_checked":len(primary),"primary_policy":primary,
        "fit_scope":scope,"new_estimator_fits":0,"new_calibrator_fits":0,"outer_predictions_scored":False,
        "selection_lock_sha256":sha(RUN/"selection_lock.json"),"run_manifest_sha256":sha(RUN/"run_manifest.json"),"protocol_sha256":sha(PROTOCOL),
        "scope_prefix_bytes":len(scope_bytes),"scope_prefix_sha256":hashlib.sha256(scope_bytes).hexdigest(),"scope_prefix_gzip_sha256":sha(DEST/"selection_fit_scope_prefix.jsonl.gz"),
        "audit_script_sha256":sha(Path(__file__)),"audit_common_sha256":sha(DEST/"audit_common.py"),
        "limits":["User-declared independent column specimens; higher biological ancestors and batches unknown","Anonymous source-file-group labels, named origin not authenticated","Source/selection locks protect declared paths, not arbitrary OS file access","Previously exposed source cohorts are not independent confirmation","No cross-plant weights or chemistry features used"]}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},ensure_ascii=False))


if __name__=="__main__":main()
