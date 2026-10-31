"""Read-only final recomputation; no classifier or calibrator fits."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/upgrade-20261031-v1/adversarial-audit"
REPORT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks=0; maxdiff=0.; protocol=json.loads((REPORT/"protocol.json").read_text())
    audit=json.loads((REPORT/"leakage_audit.json").read_text())
    scorer=OUT/"scorer_only" if (OUT/"scorer_only").exists() else REPORT/"scorer_evidence"
    original=json.loads((scorer/"labels_original.json").read_text())
    shuffled=json.loads((scorer/"labels_shuffled.json").read_text())
    legacy=json.loads((ROOT/"reports/staged-nested-optimization/2026-10-31/core/splits.json").read_text())[0]
    allowed={"outer0.full_source":set(legacy["train_sample_ids"]),
             **{f"outer0.inner{i}":set(legacy["inner_folds"][i]["train_sample_ids"]) for i in range(2)}}
    workers=OUT/"workers" if (OUT/"workers").exists() else REPORT/"worker_evidence"
    starts=Counter(); completed=Counter(); isolation=[]; score_rows=[]
    for folder in sorted(workers.iterdir()):
        name=folder.name
        log_path=folder/"fit_scope_manifest.jsonl"
        text=log_path.read_text() if log_path.exists() else gzip.decompress(log_path.with_suffix('.jsonl.gz').read_bytes()).decode()
        logs=[json.loads(line) for line in text.splitlines()]
        for row in logs:
            assert row["code_version"]==protocol["script_sha256"]
            ids=row["fit_sample_ids"]
            assert len(ids)==len(set(ids)) and set(ids)<=allowed[row["split_id"]]
            assert not set(ids)&set(original)
            checks+=3
            if row["event"]=="fit_start":starts[row["component"]]+=1
            elif row["event"]=="fit_completed":completed[row["component"]]+=1
            else:raise AssertionError("Unexpected failure/rejection in allowed actual fits")
        completed_here=sum(row["event"]=="fit_completed" for row in logs)
        assert completed_here==221;checks+=1
        state=json.loads((folder/"worker_isolation.json").read_text())
        assert state["intentional_truth_probe_blocked"] and state["post_inference_state_unchanged"]
        assert state["pointwise_score_max_difference"]<=1e-10
        isolation.append({"variant":name,**state});checks+=3
        with np.load(folder/"predictions.npz",allow_pickle=False) as v:
            p={k:v[k] for k in v.files}
        assert len(p["record_ids"])==len(set(p["record_ids"].tolist()))==171
        assert set(p["record_ids"])==set(original);checks+=2
        labelmap=shuffled if name=="shuffled_test_truth" else original
        truth=np.asarray([labelmap[str(i)] for i in p["record_ids"]])
        stats={"accuracy":float(accuracy_score(truth,p["predicted"])),
               "macro_f1":float(f1_score(truth,p["predicted"],labels=p["class_order"],average="macro",zero_division=0)),
               "balanced_accuracy":float(balanced_accuracy_score(truth,p["predicted"]))}
        if name=="baseline":expected=audit["original_score"]
        elif name=="shuffled_test_truth":expected=audit["shuffled_test_truth_score"]
        elif name.startswith("source_permutation_"):
            expected=next(r for r in audit["permutation_results"] if r["seed"]==int(name.rsplit("_",1)[1]))
        else:expected=None
        if expected is not None:
            for key,value in stats.items():
                delta=abs(value-expected[key]);assert delta<1e-12;maxdiff=max(maxdiff,delta);checks+=1
            assert confusion_matrix(truth,p["predicted"],labels=p["class_order"]).tolist()==expected["confusion_matrix"]
            checks+=1
        score_rows.append({"variant":name,**stats})
    assert starts==completed and sum(completed.values())==2873;checks+=2
    invariant_files=[workers/name/"model.joblib" for name in ("baseline","shuffled_test_truth","changed_test_input")]
    assert len({sha(p) for p in invariant_files})==1;checks+=1
    events=json.loads((REPORT/"command_invocations.json").read_text())
    release=json.loads((REPORT/"minimal_run_release.json").read_text())
    assert all(r["returncode"]==0 and r["started_at_utc"]>release["released_at_utc"] for r in events);checks+=1
    value={"status":"PASS_FOR_DECLARED_TECHNICAL_CHECKS_DATA_EXPLORATORY","checks":checks,
        "maximum_metric_difference":maxdiff,"worker_paths":len(events),"final_prediction_rows":13*171,
        "fit_calls_completed":sum(completed.values()),"fit_component_counts":dict(completed),
        "all_actual_fit_ids_in_current_source":True,"model_bytes_equal_C_D":sha(invariant_files[0]),
        "all_truth_probe_attempts_blocked":True,"scores":score_rows,
        "new_classifier_fits":0,"new_calibrator_solutions":0,
        "coverage":"one fixed legacy outer fold and thirteen prespecified retrainings; no OS sandbox or universal zero-leakage proof"}
    (REPORT/"independent_recompute.json").write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in value.items() if k not in ('scores','fit_component_counts')},ensure_ascii=False))


if __name__=="__main__":main()
