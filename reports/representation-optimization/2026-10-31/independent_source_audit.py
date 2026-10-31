"""Independent Stage-A lineage, prediction, metric and source-choice audit.

No source-search aggregation/selection/model-fitting function is imported.
Representative model replay performs prediction only after SHA validation.
"""
from collections import Counter
from itertools import permutations
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import joblib
import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/representation-optimization-20261031/source"
OUTPUT = ROOT.parent / "independent_source_audit.json"
START = time.monotonic()
checks, maximum = 0, 0.
failures = []


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        raise AssertionError(message)


def near(actual, expected, message):
    global maximum
    difference = abs(float(actual)-float(expected))
    maximum = max(maximum, difference)
    check(difference <= 1e-12, message)


def readj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def metrics_from_rows(rows, classes, truth):
    vocabulary = {label:i for i,label in enumerate(classes)}
    cm = [[0]*len(classes) for _ in classes]
    for row in rows:
        check(row["y_true"] == truth[row["record_id"]], "Canonical NPZ truth mismatch")
        check(row["y_predicted"] in vocabulary, "Unknown predicted class")
        check(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "Incorrect correctness flag")
        cm[vocabulary[row["y_true"]]][vocabulary[row["y_predicted"]]] += 1
    true_counts = [sum(row) for row in cm]
    predicted_counts = [sum(row[i] for row in cm) for i in range(len(classes))]
    diagonal = [cm[i][i] for i in range(len(classes))]
    recall = [d/n if n else 0. for d,n in zip(diagonal,true_counts)]
    f1 = [2*d/(a+b) if a+b else 0. for d,a,b in zip(diagonal,true_counts,predicted_counts)]
    return {"accuracy":sum(diagonal)/len(rows), "balanced_accuracy":math.fsum(recall)/len(classes),
            "macro_f1":math.fsum(f1)/len(classes), "confusion_matrix":cm,
            "class_recall":recall, "class_counts":true_counts,
            "correct":sum(diagonal), "total":len(rows)}


def main():
    manifest = readj(ROOT / "source_run_manifest.json")
    protocol_path = REPO / "data/research/representation_calibration_protocol_20261031.json"
    protocol = readj(protocol_path)
    config = protocol["source_search"]
    input_path = REPO / "outputs/approved-zaoren-20261031/data-preparation/dataset.npz"
    plan_path = REPO / protocol["fixed_split"]
    plan = readj(plan_path)
    for path,key in [(input_path,"input_sha256"),(plan_path,"plan_sha256"),(protocol_path,"protocol_sha256")]:
        check(sha(path) == manifest[key], "Input/plan/protocol SHA mismatch")
    for name,value in manifest["source_code_sha256"].items():
        check(sha(REPO/name) == value, "Source code SHA mismatch: "+name)
    check(platform.python_version() == manifest["python"], "Python version mismatch")
    for name,version in manifest["dependencies"].items():
        check(importlib.metadata.version(name) == version, "Dependency version mismatch: "+name)
    with np.load(input_path, allow_pickle=False) as data:
        record_ids = data["record_ids"].copy()
        labels = data["y"].copy()
        groups = data["groups"].copy()
        axis = data["axis"].copy()
        alpha = data["X_alpha"].copy()
    truth = dict(zip(record_ids.tolist(),labels.tolist()))
    physical = dict(zip(record_ids.tolist(),groups.tolist()))
    check(len(truth) == 855 and len(set(groups.tolist())) == 855, "Canonical sample identity count")
    all_roles = {role:{row["record_id"] for row in plan["records"] if row["role"] == role}
                 for role in ["training","calibration","prediction"]}
    check([len(all_roles[k]) for k in all_roles] == [513,171,171], "Outer role counts")
    check(set.union(*all_roles.values()) == set(truth), "Outer role coverage")
    check(all(not all_roles[a]&all_roles[b] for a,b in [("training","calibration"),("training","prediction"),("calibration","prediction")]), "Outer role overlap")
    for row in plan["records"]:
        check(row["physical_sample_id"] == physical[row["record_id"]] and row["origin_group_id"] == truth[row["record_id"]], "Plan annotation mismatch")
    training_ids = record_ids[np.asarray([str(v) in all_roles["training"] for v in record_ids])]
    training_y = np.asarray([truth[str(v)] for v in training_ids])
    classes = sorted(set(training_y.tolist()))
    check(classes == [f"ZY{i:02d}" for i in range(1,8)], "Class vocabulary")
    recorded_partitions = readj(ROOT / "development_partitions.json")
    expected_partitions = []
    orders = list(permutations(range(3)))
    for episode in range(config["episodes"]):
        rng = np.random.default_rng(config["split_seed"]+episode)
        positions = {name:[] for name in ["source","support_pool","query"]}
        for label in classes:
            members = np.flatnonzero(training_y == label)
            members = members[np.argsort(training_ids[members])]
            order = members[list(orders[episode%6])] if len(members) == 3 else rng.permutation(members)
            first,second = max(1,int(.6*len(members))),max(1,int(.2*len(members)))
            positions["source"].extend(order[:first]); positions["support_pool"].extend(order[first:first+second]); positions["query"].extend(order[first+second:])
        expected = {role:training_ids[sorted(indices)].tolist() for role,indices in positions.items()}
        check(expected == recorded_partitions[episode], "Seeded development partition mismatch")
        sets = [set(expected[name]) for name in expected]
        check([len(v) for v in sets] == [307,103,103], "Development role counts")
        check(set.union(*sets) == all_roles["training"], "Development coverage escapes original training")
        check(all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)), "Development role overlap")
        expected_partitions.append(expected)
    preprocs = ["raw","snv","sg0_snv","sg1_snv","sg2_snv","linear_detrend_snv"]
    configs = [{"model":"svc","C":1.},{"model":"svc","C":10.},{"model":"svc","C":100.},
               {"model":"pca_svc","C":10.,"n_components":20},{"model":"plsda","n_components":5},
               {"model":"plsda","n_components":10},{"model":"shrinkage_lda","priors":"uniform"},
               {"model":"ridge","alpha":1.},{"model":"ridge","alpha":10.}]
    candidates=[]
    for preprocessor in preprocs:
        for candidate in configs:
            candidates.append({"id":f"s{len(candidates):03d}_{preprocessor}_{candidate['model']}","preprocessing":preprocessor,**candidate})
    check(candidates == readj(ROOT/"source_candidate_grid.json"), "Independent source candidate grid mismatch")
    check(objsha(candidates) == manifest["source_grid_sha256"] == protocol["candidates"]["source_grid_sha256"], "Source grid SHA")
    metadata_by_case, recomputed, warning_count = {}, {}, Counter()
    row_count = 0
    for candidate in candidates:
        values={"selection":[],"gate":[]}
        for episode,partition in enumerate(expected_partitions):
            stem=ROOT/"development"/candidate["id"]/f"episode_{episode:02d}"
            meta=readj(stem.with_suffix(".json"))
            check(meta["status"] == "pass" and meta["source_fit_calls"] == 1, "Source episode status")
            check(meta["candidate"] == candidate and meta["partition"] == partition, "Candidate/partition metadata mismatch")
            stage="selection" if episode < config["selection_episodes"] else "gate"
            check(meta["stage"] == stage and meta["episode"] == episode, "Episode stage mismatch")
            check(meta["class_order"] == classes, "Source class order mismatch")
            check(meta["source_fingerprint_before"] == meta["source_fingerprint_after"], "Recorded source-freeze assertion")
            check(meta["query_labels_used_for_fit"] is False and meta["final_query_scored"] is False, "Metadata query-use assertion")
            check(meta["model_metadata"]["fit_scope"] == "source_training_only" and meta["model_metadata"]["source_samples"] == 307, "Fit scope/count assertion")
            expected_counts=dict(Counter(truth[v] for v in partition["source"]))
            check(meta["model_metadata"]["class_counts"] == expected_counts, "Source class counts mismatch")
            model_path=stem.with_suffix(".joblib"); prediction_path=stem.with_suffix(".predictions.csv")
            check(sha(model_path) == meta["model_sha256"], "Source model artifact SHA mismatch")
            check(sha(prediction_path) == meta["predictions_sha256"], "Saved prediction SHA mismatch")
            with prediction_path.open(encoding="utf-8-sig",newline="") as handle: rows=list(csv.DictReader(handle))
            check([r["record_id"] for r in rows] == partition["query"], "Prediction record order/membership mismatch")
            metric=metrics_from_rows(rows,classes,truth); saved=meta["metrics"]
            for key in ["confusion_matrix","class_counts","correct","total"]:check(metric[key] == saved[key], "Saved detailed metric mismatch: "+key)
            for key in ["accuracy","balanced_accuracy","macro_f1"]:near(metric[key],saved[key],"Saved scalar metric mismatch: "+key)
            for left,right in zip(metric["class_recall"],saved["class_recall"]):near(left,right,"Saved per-class recall mismatch")
            warning_count.update(meta.get("warnings",[])); row_count += len(rows)
            values[stage].append(metric); metadata_by_case[(candidate["id"],episode)]=meta
        recomputed[candidate["id"]]={stage:{**{key:math.fsum(v[key] for v in entries)/len(entries) for key in ["accuracy","balanced_accuracy","macro_f1"]},"evaluations":len(entries),"failures":0} for stage,entries in values.items()}
    recorded=readj(ROOT/"source_development_aggregate.json")
    for identifier,stages in recomputed.items():
        for stage,value in stages.items():
            for key in ["accuracy","balanced_accuracy","macro_f1"]:near(value[key],recorded[identifier][stage][key],"Source aggregate mismatch")
            for key in ["evaluations","failures"]:check(value[key] == recorded[identifier][stage][key], "Source aggregate count mismatch")
    lookup={c["id"]:c for c in candidates}; ref=config["reference_id"]
    check(lookup[ref] == {"id":"s010_snv_svc","preprocessing":"snv","model":"svc","C":10.}, "Reference candidate identity")
    pre_order={"raw":0,"snv":1,"sg0_snv":2,"linear_detrend_snv":3,"sg1_snv":4,"sg2_snv":5}
    model_order={"ridge":0,"plsda":1,"svc":2,"pca_svc":3,"shrinkage_lda":4}
    def complexity(c):return(c["id"] != ref,model_order[c["model"]],pre_order[c["preprocessing"]],c.get("n_components",0),-c.get("alpha",0),c.get("C",0),c["id"])
    def permitted(identifier,stage):
        v,b=recomputed[identifier][stage],recomputed[ref][stage]
        return(not v["failures"] and v["evaluations"] == b["evaluations"] and v["accuracy"]-b["accuracy"] > config["minimum_accuracy_gain"] and v["macro_f1"]-b["macro_f1"] >= -config["max_macro_f1_loss"] and v["balanced_accuracy"]-b["balanced_accuracy"] >= -config["max_balanced_accuracy_loss"])
    def best(ids):
        highest=max(recomputed[i]["selection"]["accuracy"] for i in ids)
        return min([lookup[i] for i in ids if recomputed[i]["selection"]["accuracy"] >= highest-config["tie_tolerance"]],key=complexity)
    eligible=[c["id"] for c in candidates if permitted(c["id"],"selection")]
    proposed=best(eligible) if eligible else lookup[ref]
    selected=proposed if proposed["id"] != ref and permitted(proposed["id"],"gate") else lookup[ref]
    lock=readj(ROOT/"source_selection_lock.json")
    check(lock["reference"] == lookup[ref] and lock["proposed"] == proposed and lock["selected"] == selected,"Independent source choice mismatch")
    check(lock["eligible_selection_candidates"] == len(eligible),"Eligibility candidate count")
    family=[]
    for name,kinds in [("plsda",{"plsda"}),("linear",{"ridge","shrinkage_lda"})]:
        ids=[c["id"] for c in candidates if c["model"] in kinds and all(recomputed[c["id"]][stage]["failures"] == 0 for stage in ["selection","gate"])]
        chosen=best(ids) if ids else lookup[ref]
        check(lock["family_best"][name]["candidate"] == chosen,"Family source choice mismatch")
        check(lock["family_best"][name]["passes_source_gate"] == permitted(chosen["id"],"gate"),"Family source gate mismatch")
        for stage in ["selection","gate"]:
            for key in ["accuracy","balanced_accuracy","macro_f1"]:near(lock["family_best"][name][stage][key],recomputed[chosen["id"]][stage][key],"Family source metrics mismatch")
        family.append(chosen)
    expected_adaptation=[]
    for candidate in [lookup[ref],selected,*family]:
        if candidate not in expected_adaptation:expected_adaptation.append(candidate)
    check(expected_adaptation == lock["candidates_for_adaptation"],"Adaptation source cohort mismatch")
    for filename,key in [("source_run_manifest.json","source_run_manifest_sha256"),("development_partitions.json","partitions_sha256"),("source_development_aggregate.json","development_aggregate_sha256")]:check(sha(ROOT/filename) == lock[key],"Source-lock provenance SHA mismatch")
    lock_sha=sha(ROOT/"source_selection_lock.json")
    check(lock_sha == readj(ROOT/"source_selection_lock_sha256.json")["sha256"] == readj(ROOT/"source_status.json")["source_selection_lock_sha256"],"Source selection lock SHA mismatch")
    check(lock["source_fits_recorded"] == 1620 and lock["source_successes"] == 1620 and lock["source_failures"] == 0,"Recorded source-fit counts")
    replay_count=0
    frequency_mask=(axis >= protocol["frequency"]["lower_inclusive"]) & (axis <= protocol["frequency"]["upper_inclusive"])
    all_lookup={str(v):i for i,v in enumerate(record_ids)}
    for candidate in expected_adaptation:
        for episode in [0,29]:
            meta=metadata_by_case[(candidate["id"],episode)]
            model_path=ROOT/"development"/candidate["id"]/f"episode_{episode:02d}.joblib"
            model=joblib.load(model_path)
            check(joblib.hash(model) == meta["source_fingerprint_before"],"Representative fitted-model fingerprint mismatch")
            query_ids=meta["partition"]["query"]
            predicted=model.predict(alpha[[all_lookup[v] for v in query_ids]][:,frequency_mask]).tolist()
            with model_path.with_suffix(".predictions.csv").open(encoding="utf-8-sig",newline="") as handle:saved=[r["y_predicted"] for r in csv.DictReader(handle)]
            check(predicted == saved,"Representative saved prediction replay mismatch")
            check(joblib.hash(model) == meta["source_fingerprint_before"],"Representative prediction mutated model")
            replay_count += 1
    return {"status":"pass","auditor":"independent Stage-A audit; no project aggregation/selection function imported", "checks_evaluated":checks,"failed_checks":failures,"source_candidates":len(candidates),"source_episodes":config["episodes"],"source_fits":1620,"prediction_rows":row_count,"fixed_training_records":513,"outer_calibration_or_prediction_records_entering_development":0,"source_development_role_counts":{"source":307,"support_pool":103,"query":103},"maximum_metric_or_mean_difference":maximum,"selected_source":selected,"reference":lookup[ref],"adaptation_sources":expected_adaptation,"selection_eligible_candidate_count":len(eligible),"source_selection_lock_sha256":lock_sha,"artifact_hashes_checked":1620,"representative_model_replays":replay_count,"warnings_by_message":dict(warning_count),"new_model_fits_in_audit":0,"seconds":time.monotonic()-START,"limits":["All model file hashes checked; fitted-state fingerprints and prediction replays independently checked for six representative selected/reference/PLS caches, not all 1620 unpickled objects.","Other per-episode freeze and fit-scope assertions are checked against saved metadata and source code hashes; this is not external execution attestation.","Origin labels are user-declared anonymous workbook groups, not independently verified geographic origins; the dataset remains retrospective and within-batch."]}


if __name__ == "__main__":
    try:
        result=main()
    except Exception as exc:
        result={"status":"fail","checks_evaluated":checks,"failed_checks":failures,"error":repr(exc),"maximum_metric_or_mean_difference":maximum,"seconds":time.monotonic()-START}
        OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        raise
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","checks_evaluated","source_fits","prediction_rows","maximum_metric_or_mean_difference","representative_model_replays","seconds"]},ensure_ascii=False))
