"""Read-only independent formulas for limited-score development/final evidence."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
RUN=Path(__file__).resolve().parent/"run"
PROTOCOL=json.loads((RUN/"protocol_snapshot.json").read_text())
MANIFEST=json.loads((RUN/"run_manifest.json").read_text())
CHECKS=0
MAX=0.


def check(value):
    global CHECKS
    CHECKS+=1
    assert value


def close(a,b,tol=1e-12):
    global MAX
    a,b=np.asarray(a),np.asarray(b)
    diff=float(np.max(np.abs(a-b))) if a.size else 0.
    MAX=max(MAX,diff)
    check(diff <= tol)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(truth,pred):
    cm=np.zeros((7,7),dtype=int)
    np.add.at(cm,(truth,pred),1)
    support=cm.sum(1); found=cm.sum(0); tp=cm.diagonal()
    recall=np.divide(tp,support,out=np.zeros(7),where=support>0)
    f1=np.divide(2*tp,support+found,out=np.zeros(7),where=support+found>0)
    return {"accuracy":float(tp.sum()/cm.sum()),"macro_f1":float(f1.mean()),"balanced_accuracy":float(recall.mean())}


def margin(s):
    z=np.sort(s,axis=1)
    return z[:,-1]-z[:,-2]


def recreate(ss,sp,sq,ys,classes,candidate):
    errors=int(np.count_nonzero(classes[np.argmax(sp,axis=1)] != ys))
    if errors < candidate["minimum_support_errors"]:
        return sq.copy()
    design=np.column_stack([sp,np.ones(len(sp))])
    target=np.eye(7)[np.array([list(classes).index(v) for v in ys])]
    delta=np.linalg.solve(design.T@design+np.eye(8),design.T@(target-sp))
    raw=np.column_stack([sq,np.ones(len(sq))])@delta
    source_gap=margin(ss)
    cap=candidate["cap_fraction"]*max(float(np.median(source_gap)),np.finfo(float).eps)
    change=np.clip(raw,-cap,cap)*candidate["blend"]
    change[margin(sq)>np.quantile(source_gap,candidate["source_gate_quantile"])]=0.
    close(np.maximum(np.abs(change)-cap*candidate["blend"],0),np.zeros_like(change))
    return sq+change


def main():
    check(sha(ROOT/PROTOCOL["input"])==MANIFEST["input_sha256"])
    check(sha(ROOT/PROTOCOL["split_plan"])==MANIFEST["split_sha256"])
    for p,h in MANIFEST["source_code_sha256"].items():
        check(sha(ROOT/p)==h)
    plan=json.loads((ROOT/PROTOCOL["split_plan"]).read_text())
    train={r["record_id"] for r in plan["records"] if r["role"]=="training"}
    all_truth={r["record_id"]:r["origin_group_id"] for r in plan["records"]}
    ids_to_role={r["record_id"]:r["role"] for r in plan["records"]}
    budget=json.loads((RUN/"development_budget.json").read_text())
    metric_count=0
    pred_count=0
    gain_max=-1
    for selector in PROTOCOL["selectors"]:
        directory=RUN/"development"/selector
        with gzip.open(directory/"metrics.csv.gz","rt",encoding="utf-8") as f:
            metric={(int(r["episode"]),int(r["repeat"]),r["candidate_id"]):r for r in csv.DictReader(f)}
        for episode in range(30):
            entry=next(x for x in budget if x["selector"]==selector and x["episode"]==episode)
            with np.load(directory/f"episode_{episode:02d}_scores.npz",allow_pickle=False) as f:
                z={k:f[k] for k in f.files}
            sets=[set(z[k+"_ids"].tolist()) for k in ("source","pool","query")]
            check([len(s) for s in sets]==[307,103,103])
            check(set.union(*sets)==train)
            check(not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2])
            for s in sets:
                check(sum(all_truth[v]=="ZY02" for v in s)==1)
                check(all(ids_to_role[v]=="training" for v in s))
            check(z["classes"].tolist()==[f"ZY{i:02d}" for i in range(1,8)])
            close(z["query_scores"].shape,[103,7],0)
            classes=z["classes"]
            truth=np.array([list(classes).index(all_truth[str(v)]) for v in z["query_ids"]])
            baseline=np.argmax(z["query_scores"],axis=1)
            base=stats(truth,baseline)
            saved=defaultdict(list)
            with gzip.open(directory/f"episode_{episode:02d}_predictions.csv.gz","rt",encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    saved[(int(row["repeat"]),row["candidate_id"])].append(row)
            for draw in entry["draws"]:
                support=draw["ids"]
                check(len(support)==len(set(support))==7)
                check(set(support).issubset(sets[1]))
                check({all_truth[v] for v in support}==set(classes))
                idx=np.array([list(z["pool_ids"]).index(v) for v in support])
                for candidate in PROTOCOL["candidates"]:
                    pred=np.argmax(recreate(z["source_scores"],z["pool_scores"][idx],z["query_scores"],z["pool_y"][idx],classes,candidate),axis=1)
                    values=stats(truth,pred)
                    key=(draw["repeat"],candidate["id"])
                    records=saved[key]
                    check(len(records)==103)
                    check([r["record_id"] for r in records]==z["query_ids"].tolist())
                    check([r["y_true"] for r in records]==classes[truth].tolist())
                    check([r["y_predicted"] for r in records]==classes[pred].tolist())
                    m=metric[(episode,*key)]
                    for k,v in values.items():
                        close(float(m[k]),v)
                        close(float(m[k+"_gain"]),v-base[k])
                    gain_max=max(gain_max,values["accuracy"]-base["accuracy"])
                    check(int(m["corrected"])==int(np.count_nonzero((baseline!=truth)&(pred==truth))))
                    check(int(m["harmed"])==int(np.count_nonzero((baseline==truth)&(pred!=truth))))
                    metric_count+=1; pred_count+=len(pred)
    result={"status":"pass","scope":"development only; no new source fits","checks":CHECKS,
            "maximum_numeric_difference":MAX,"source_fit_records_checked":len(budget),
            "candidate_draw_cases":metric_count,"prediction_rows":pred_count,
            "maximum_candidate_accuracy_gain":gain_max,"new_model_fits":0,
            "selection_lock_sha256":sha(RUN/"selection_lock.json"),"final_evaluation_exists":(RUN/"final_metrics.json").exists()}
    final=RUN/"final_predictions.csv.gz"
    if final.exists():
        groups=defaultdict(list)
        with gzip.open(final,"rt",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                groups[(row["selector"],int(row["repeat"]),row["method"])].append(row)
        final_metrics=json.loads((RUN/"final_metrics.json").read_text())
        finals={(r["selector"],r["repeat"],r["method"]):r for r in final_metrics}
        expected_query={r["record_id"] for r in plan["records"] if r["role"]=="prediction"}
        for key,rows in groups.items():
            check(len(rows)==171 and {r["record_id"] for r in rows}==expected_query)
            check(all(r["y_true"]==all_truth[r["record_id"]] for r in rows))
            truth=np.array([int(r["y_true"][-2:])-1 for r in rows])
            pred=np.array([int(r["y_predicted"][-2:])-1 for r in rows])
            for k,v in stats(truth,pred).items():
                close(v,finals[key][k])
            for row in rows:
                check(int(row["is_correct"])==int(row["y_true"]==row["y_predicted"]))
            baseline=groups[(key[0],key[1],"no_calibration")]
            basepred=np.array([int(r["y_predicted"][-2:])-1 for r in baseline])
            check(finals[key]["corrected"]==int(np.count_nonzero((basepred!=truth)&(pred==truth))))
            check(finals[key]["harmed"]==int(np.count_nonzero((basepred==truth)&(pred!=truth))))
            if key[2] in {"same_structure_zero_support","same_structure_no_labels","conservative_selected"}:
                check(np.array_equal(basepred,pred))
        result.update(scope="development and final; read-only reconstruction",final_prediction_rows=sum(len(v) for v in groups.values()),
                      final_cases=len(groups),checks=CHECKS,maximum_numeric_difference=MAX)
    target=RUN.parent/"recompute_audit.json"
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
