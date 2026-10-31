"""Review the isolated v3 inference-check fix against the full v2 source audit."""
import ast
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from source_review import ROOT,DEST,PROTOCOL,sha,load,npz


RUN=ROOT/"outputs/upgrade-20261031-v1/auxiliary/v3"
PREVIOUS=RUN.parent/"v2"
ARCHIVE=ROOT/"reports/upgrade-20261031-v1/auxiliary/source_inference_failure_v2"
checks=0;cells=0


def require(value,detail):
    global checks
    checks += 1
    if not value:raise AssertionError(detail)


def normalized_definitions(text):
    result={}
    tree=ast.parse(text)
    result["_top_level"]=[ast.dump(node,include_attributes=False) for node in tree.body if not isinstance(node,(ast.FunctionDef,ast.ClassDef))]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            if node.name=="default_protocol":continue
            if node.name=="trial":
                for j,statement in enumerate(node.body):
                    if isinstance(statement,ast.If) and ast.unparse(statement.test)=="family in NEURAL":
                        node.body[j]=ast.Pass()
            result[node.name]=ast.dump(node,include_attributes=False)
    return result


def main():
    global cells
    target=DEST/"source_review_v3.json"
    if target.exists():raise FileExistsError("Preserve completed v3 source review")
    prior=load(DEST/"source_review.json")
    require(sha(DEST/"source_review.json")=="c75e3ce3bc8f6c76b6220deaac344302c64c32202cf71c1d252f8ee44a89529a" and prior["allow_source_training"] is True,"Full source binding review changed")
    candidate,prep,protocol=(load(p) for p in (RUN/"candidate_manifest.json",RUN/"preparation.json",PROTOCOL))
    require(sha(PROTOCOL)==candidate["protocol_sha256"] and sha(RUN/"preparation.json")==candidate["preparation_sha256"],"V3 protocol/preparation binding")
    require(protocol==load(RUN/"protocol_snapshot.json"),"Protocol snapshot differs")
    require(candidate["protocol_snapshot_sha256"]==sha(RUN/"protocol_snapshot.json") and candidate["split_manifest_sha256"]==sha(RUN/"split_manifest.json"),"Additional manifest bindings")
    require(not list(RUN.glob("selection/**/pipeline.joblib")) and not (RUN/"selection_lock.json").exists() and not (RUN/"fit_scope_events.jsonl").exists() and not (RUN/"outer_predictions.csv.gz").exists(),"V3 review must precede real source fits")
    for relative,expected in candidate["source_sha256"].items():require(sha(ROOT/relative)==expected,"Frozen V3 dependency changed "+relative)
    previous_manifest=load(PREVIOUS/"candidate_manifest.json")
    require(sha(PREVIOUS/"candidate_manifest.json")==prior["candidate_manifest_sha256"],"Reviewed v2 candidates changed")
    for relative,expected in previous_manifest["source_sha256"].items():
        if relative!="src/herb_spectra/auxiliary_multiplant.py":require(candidate["source_sha256"][relative]==expected,"Unrelated frozen implementation changed")
    old_protocol=load(PREVIOUS/"protocol_snapshot.json")
    require(all(protocol[k]==v for k,v in old_protocol.items() if k not in {"protocol_id","boundary_revision"}),"V3 changed mathematics/data/selection protocol")
    old_source=(ARCHIVE/"src/herb_spectra/auxiliary_multiplant.py").read_text();new_source=(ROOT/"src/herb_spectra/auxiliary_multiplant.py").read_text()
    require(sha(ARCHIVE/"src/herb_spectra/auxiliary_multiplant.py")==previous_manifest["source_sha256"]["src/herb_spectra/auxiliary_multiplant.py"],"Archived v2 code changed")
    require(normalized_definitions(old_source)==normalized_definitions(new_source),"V3 changed a learned operation, inference rule or scorer outside diagnostics")
    require("v[i:min(i+size,n)]" in new_source and "for size in (1,2,3,7)" in new_source,"Tail-check clamp missing")
    # Independently exercise the revised slice arithmetic with no estimator.
    tail_cases=0
    for actual in (1,2,3,7,13,80,81,161,162):
        for outer in (False,True):
            n=actual if outer else min(7,actual)
            for size in (1,2,3,7):
                found=np.concatenate([np.arange(actual)[i:min(i+size,n)] for i in range(0,n,size)])
                require(np.array_equal(found,np.arange(n)),"Tail-check diagnostic population differs from requested prefix")
                tail_cases += 1
    boundary=load(RUN/"boundary_test_provenance.json")
    require(boundary["source_sha256"]==candidate["source_sha256"] and boundary["test_file_sha256"]==sha(ROOT/"tests/test_auxiliary_multiplant.py"),"V3 test source binding")
    require(boundary["candidate_manifest_sha256"]==sha(RUN/"candidate_manifest.json") and all(boundary[k]==candidate[k] for k in ("protocol_sha256","preparation_sha256")),"V3 test input binding")
    require(boundary["returncode"]==0 and boundary["tests_run"]==13 and boundary["production_data_fits"]==0,"V3 boundary suite status")
    require(sha(ROOT/boundary["log"])==boundary["log_sha256"] and (ROOT/boundary["log"]).read_text().rstrip().endswith("OK"),"V3 recorded test execution")
    require("test_svm_trial_tail_batches_use_exact_inference_population" in (ROOT/boundary["log"]).read_text(),"Tail regression test was not executed")
    oldprep=load(PREVIOUS/"preparation.json");oldcontexts={c["split_id"]:c for c in oldprep["contexts"]}
    require(sha(PREVIOUS/"preparation.json")==prior["preparation_sha256"],"Previously bound input contexts changed")
    require(len(prep["contexts"])==len({c["split_id"] for c in prep["contexts"]})==30 and set(c["split_id"] for c in prep["contexts"])==set(oldcontexts),"V3 role set differs")
    for c in prep["contexts"]:
        old=oldcontexts[c["split_id"]]
        for field in ("dataset_id","stage","outer_fold","inner_fold","source_ids","inference_ids","lineage_status"):
            require(c[field]==old[field],"V3 split scope changed "+field)
        for kind in ("source_pack","inference_pack","scoring_truth"):
            require(c[kind+"_sha256"]==old[kind+"_sha256"]==sha(ROOT/c[kind])==sha(ROOT/old[kind]),"V3 raw prepared bytes/labels differ "+kind)
            if kind!="scoring_truth":
                left,right=npz(ROOT/c[kind]),npz(ROOT/old[kind]);require(set(left)==set(right),"V3 pack schema changed")
                for key in left:
                    require(np.array_equal(left[key],right[key]),"V3 prepared values/ID binding changed")
                    if key.startswith("X_"):cells += left[key].size
    current_splits,old_splits=load(RUN/"split_manifest.json"),load(PREVIOUS/"split_manifest.json")
    require(all(current_splits[k]==v for k,v in old_splits.items() if k!="protocol_id"),"V3 split metadata changed")
    require(prep["transforms"]==oldprep["transforms"],"V3 spectral axis/representation changed")
    for key in ("overlay_manifest","overlay_lineage","annotation"):
        require(sha(ROOT/protocol[key])==protocol[key+"_sha256"],"User declaration changed")
    for entries in protocol["inputs"].values():
        for entry in entries.values():require(sha(ROOT/entry["path"])==entry["sha256"],"Original DB modality changed")
    overlay=load(ROOT/protocol["overlay_manifest"])
    require(sha(ROOT/overlay["base_lineage"])==prior["historical_lineage_sha256"],"Historical lineage rewritten")
    secondary=load(ROOT/"data/multiplant_database/data_quality/secondary_read_evidence/manifest.json")
    for workbook in secondary["source_workbooks"]:
        if workbook["dataset"] in protocol["datasets"]:require(sha(ROOT/workbook["source_file"])==workbook["sha256"],"Original workbook bytes changed since full binding audit")
    failure=load(ARCHIVE/"failure_manifest.json")
    require(failure["completed_model_fits"]==1 and failure["completed_trial_predictions"]==0 and failure["inner_truth_parsed_by_failed_trial"] is False and failure["outer_evaluation_run"] is False,"V2 failure incorrectly counted")
    scope=[json.loads(line) for line in (PREVIOUS/"fit_scope_events.jsonl").read_text().splitlines()]
    require(Counter(r["component"] for r in scope if r["event"]=="fit_completed")==Counter({"source_scaler_alpha":1,"SVC.fit":1}),"V2 failed-run fit inventory")
    require(not (PREVIOUS/"inner_metrics.json").exists() and not (PREVIOUS/"outer_predictions.csv.gz").exists(),"V2 failed run contains unacknowledged scored results")
    result={"status":"TECHNICAL_AUXILIARY_V3_SOURCE_REVIEW_PASS_EXPLORATORY_DATA","allow_source_training":True,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"numeric_cells_compared_v2_v3":cells,
        "prior_full_numeric_cells_compared":prior["numeric_cells_compared"],"prior_full_source_review_sha256":sha(DEST/"source_review.json"),
        "split_contexts":30,"fixed_outer_folds":5,"fixed_inner_folds":2,"planned_source_candidates":1200,"boundary_tests":13,"independent_tail_index_cases":tail_cases,
        "v2_failed_classifier_fits_preserved":1,"v2_completed_scaler_fits_preserved":1,"v2_trial_scores_or_outer_results":0,
        "new_estimator_fits_in_this_audit":0,"v3_research_fits_before_review":0,"outer_predictions_scored":False,
        "protocol_sha256":sha(PROTOCOL),"candidate_manifest_sha256":sha(RUN/"candidate_manifest.json"),"preparation_sha256":sha(RUN/"preparation.json"),
        "boundary_test_provenance_sha256":sha(RUN/"boundary_test_provenance.json"),"v2_failure_archive_manifest_sha256":sha(ARCHIVE/"archive_manifest.json"),
        "review_script_sha256":sha(Path(__file__)),"source_sha256":candidate["source_sha256"],
        "scope":["Only diagnostic tail batching and version metadata changed","Model fitting, prediction/vote, source selection and scoring definitions AST-identical","V3 numeric packs, canonical IDs, labels and thirty split contexts exactly equal the full-audited v2 inputs","Fresh V3 source training allowed; outer requires separate post-selection review"],
        "limits":prior["limits"]}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},ensure_ascii=False))


if __name__=="__main__":main()
