"""Independent final read-only replay of Paper-C experiments.

No estimator is fitted. The200 frozen-support anchored-ridge solutions are
reconstructed explicitly with8x8 algebra and counted, never hidden as zero fits.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from herb_spectra.neural_base_search import DenseClassifier, model_scores, transform_features
from herb_spectra.paper_fewshot_classification import state_fingerprint


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REPORT = ROOT / "reports/paper-c/2026-10-31"
BASE = ROOT / "outputs/paper-c-20261031"
CLASSES = [f"ZY{i:02d}" for i in range(1,8)]
VOCAB = {c:i for i,c in enumerate(CLASSES)}
METRICS = ("accuracy","macro_f1","balanced_accuracy")
checks = 0
maximum_difference = 0.0
calibrator_solutions = 0


def require(value,message):
    global checks
    checks += 1
    if not value:raise AssertionError(message)


def close(actual,expected,message,tolerance=1e-12):
    global maximum_difference
    difference=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    maximum_difference=max(maximum_difference,difference)
    require(difference <= tolerance,f"{message}: {difference}")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,"rt",encoding="utf-8-sig") as handle:return list(csv.DictReader(handle))


def compute(records):
    matrix=np.zeros((7,7),dtype=int)
    for row in records:matrix[VOCAB[row["y_true"]],VOCAB[row["y_predicted"]]]+=1
    actual,found,tp=matrix.sum(1),matrix.sum(0),matrix.diagonal()
    precision=np.divide(tp,found,out=np.zeros(7),where=found!=0)
    recall=np.divide(tp,actual,out=np.zeros(7),where=actual!=0)
    f1=np.divide(2*tp,actual+found,out=np.zeros(7),where=actual+found!=0)
    return {"accuracy":float(tp.sum()/matrix.sum()),"macro_f1":float(f1.mean()),"balanced_accuracy":float(recall.mean()),
        "correct":int(tp.sum()),"total":int(matrix.sum()),"confusion_matrix":matrix.tolist(),
        "precision":precision.tolist(),"recall":recall.tolist(),"f1":f1.tolist(),"support":actual.tolist(),"predicted_count":found.tolist()}


def compare_metrics(value,expected,label):
    for name in METRICS:close(value[name],expected[name],label+name)
    require(value["confusion_matrix"]==expected["confusion_matrix"],label+"confusion")
    if "correct" in expected:require(value["correct"]==expected["correct"],label+"correct")
    if "per_class" in expected:
        for i,c in enumerate(CLASSES):
            for name in ["precision","recall","f1"]:close(value[name][i],expected["per_class"][c][name],label+c+name)
            require(value["support"][i]==expected["per_class"][c]["support"],label+c+"support")


def check_rows(records,truth,query_ids=None):
    for row in records:
        require(row["y_true"]==truth[row["record_id"]],"Canonical raw query label")
        require(int(row["is_correct"])==int(row["y_true"]==row["y_predicted"]),"Raw correctness flag")
    if query_ids is not None:
        require(len(records)==len({r["record_id"] for r in records})==len(query_ids) and {r["record_id"] for r in records}==set(query_ids),"All intended query samples exactly once")


def main():
    global calibrator_solutions
    started=datetime.now(timezone.utc).isoformat();torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    release_path=REPORT/"evaluation_release.json";release=read(release_path)
    require(sha(release_path)=="fa016aa83fb6a4bdc36dc2545dcac16284587bd79252f06ef516b735dc05d410","Original actual global release SHA")
    require(release["all_selections_frozen_before_any_new_evaluation"] and release["allow_outer_evaluation"],"Global release explicit")
    released=datetime.fromisoformat(release["released_at_utc"])
    for path,expected in release["development_reviews_sha256"].items():require(sha(ROOT/path)==expected,"Earlier independent review not overwritten")
    invocations=[json.loads(line) for line in (REPORT/"command_invocations.jsonl").read_text().splitlines()]
    execution=[]
    for experiment in release["experiments"]:
        name=experiment["name"];run=ROOT/experiment["run_directory"]
        manifest=read(run/"run_manifest.json");lock=read(run/"selection_lock.json")
        for path,key in [(run/"run_manifest.json","manifest_sha256"),(run/"selection_lock.json","selection_lock_sha256"),(ROOT/experiment["protocol"],"protocol_sha256")]:
            require(sha(path)==experiment[key],"Released source/protocol/selection unchanged")
        sources=manifest.get("source_sha256",manifest.get("source_code_sha256"))
        for path,expected in sources.items():require(sha(ROOT/path)==expected,"Executed ML source unchanged after evaluation")
        require(experiment["new_evaluation_artifact_existed_at_release"] is False,"New output absent at actual release")
        individual=read(REPORT/"evaluation-releases"/(name+".json"))
        require(individual["global_release_sha256"]==sha(release_path) and individual["selection_lock_sha256"]==experiment["selection_lock_sha256"] and individual["manifest_sha256"]==experiment["manifest_sha256"],"Per-run release bound to global release")
        label=name+"-evaluation"
        starts=[e for e in invocations if e["label"]==label and e["event"]=="command_start"]
        ends=[e for e in invocations if e["label"]==label and e["event"]=="command_completed"]
        require(len(starts)==len(ends)==1 and ends[0]["returncode"]==0,"One successful actual evaluation invocation")
        require(released<datetime.fromisoformat(starts[0]["utc"])<datetime.fromisoformat(ends[0]["utc"]),"Actual evaluation begins after all locks/reviews release")
        execution.append({"name":name,"start_utc":starts[0]["utc"],"completed_utc":ends[0]["utc"],"selection_lock_sha256":experiment["selection_lock_sha256"],"source_hashes_verified":len(sources)})
    historical=ROOT/"reports/staged-nested-optimization/2026-10-31"
    for directory in [historical,ROOT/"models/staged_nested_20261031"]:
        for relative,expected in read(directory/"files_sha256.json").items():require(sha(directory/relative)==expected,"Historical immutable evidence preserved")
    data=np.load(ROOT/"outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz",allow_pickle=False)
    X,y,ids,axis=(data[k] for k in ["X_alpha","y","record_ids","axis"])
    truth=dict(zip(ids.tolist(),y.tolist(),strict=True));id_index={name:i for i,name in enumerate(ids)}
    splits=read(historical/"core/splits.json")
    source_cache={}
    def source_info(outer):
        if outer not in source_cache:
            names=splits[outer]["train_sample_ids"];indices=[id_index[name] for name in names];values=X[indices]
            snv=(values-values.mean(1,keepdims=True))/values.std(1,ddof=1,keepdims=True)
            source_cache[outer]=(names,indices,values,y[indices],snv.mean(0),snv.var(0))
        return source_cache[outer]
    all_metrics=[];class_metrics=[];model_replays=[];fit_costs=[]
    benchmark=BASE/"fair-benchmark";require(read(benchmark/"status.json")["status"]=="completed","Benchmark actually completed")
    benchmark_rows=rows(benchmark/"outer_predictions.csv.gz")
    require(len(benchmark_rows)==7695,"Nine benchmark roles x855")
    grouped=defaultdict(list)
    for row in benchmark_rows:grouped[(row["method"],int(row["outer_fold"]))].append(row)
    by_method=defaultdict(list)
    for key,records in grouped.items():by_method[key[0]].extend(records)
    outer=read(benchmark/"outer_metrics.json");pooled=read(benchmark/"pooled_metrics.json")
    require(len(outer)==len(grouped)==45,"All45 source-trained benchmark pipelines")
    before_hashes={}
    for entry in outer:
        method,fold=entry["method"],entry["outer_fold"];records=grouped[(method,fold)]
        names,source_indices,source_values,source_labels,mean,variance=source_info(fold)
        require(entry["source_ids"]==names and entry["test_ids"]==splits[fold]["test_sample_ids"],"Benchmark actual source/query IDs")
        check_rows(records,truth,entry["test_ids"]);computed=compute(records);compare_metrics(computed,entry["metrics"],"Benchmark outer fold ")
        path=Path(entry["model_path"]);path=path if path.is_absolute() else ROOT/path
        require(sha(path)==entry["model_sha256"],"New evaluated benchmark model hash")
        before_hashes[path]=sha(path);model=joblib.load(path);meta=model.metadata
        require(meta==entry["model_metadata"] and meta["source_samples"]==684 and meta["model_seed"]==42,"Benchmark actual model metadata")
        require(meta["source_fingerprint"]==joblib.hash(source_values) and meta["source_label_fingerprint"]==joblib.hash(source_labels),"Benchmark source data fingerprint")
        close(model.scaler.mean_,mean,"Benchmark actual source scaler mean");close(model.scaler.var_,variance,"Benchmark actual source scaler variance")
        require(np.array_equal(model.indices,np.asarray(meta["selected_indices"])) and len(set(model.indices.tolist()))==len(model.indices),"Benchmark unique selected indices")
        close(axis[model.indices],meta["selected_frequencies"],"Benchmark selected physical frequencies")
        ordered=sorted(records,key=lambda r:r["record_id"]);query=X[[id_index[r["record_id"]] for r in ordered]]
        require(np.array_equal(model.predict(query),[r["y_predicted"] for r in ordered]),"Benchmark frozen pipeline label replay")
        scores=model.scores(query);saved=np.asarray([[float(r["score_"+c]) for c in CLASSES] for r in ordered])
        close(scores,saved,"Benchmark frozen pipeline scores")
        if hasattr(model.estimator,"_gamma"):
            require(model.estimator.break_ties is False and model.estimator.random_state==42,"Benchmark original SVC vote/seed")
            z=model.transform(source_values);variance_z=z.var();scale=1/(z.shape[1]*variance_z) if variance_z else 1.
            gammas={"scale":scale,"auto":1/z.shape[1],"scale_x0.1":.1*scale,"scale_x10":10*scale}
            close(model.estimator._gamma,gammas[entry["candidate"]["gamma"]],"Actual source-only gamma")
            classes,counts=np.unique(source_labels,return_counts=True);mode=entry["candidate"]["class_weight"]
            if mode=="none":weights=np.ones(len(classes))
            else:
                weights=len(source_labels)/(len(classes)*counts.astype(float))
                if mode.startswith("power_"):
                    weights=weights**float(mode.split("_")[1]);weights/=np.sum(weights*counts)/len(source_labels)
            close([meta["class_weights"][c] for c in classes],weights,"Actual source-only class weights")
        if entry["candidate"]["family"]=="mlp":require(model.estimator.n_iter_==200 and not model.estimator.early_stopping,"Fixed200 epoch MLP no early stop")
        model_replays.append({"experiment":"fair-benchmark","method":method,"outer_fold":fold,"source":684,"query":171,"path":str(path.relative_to(ROOT)),"sha256":sha(path),"labels_and_scores_replayed":True})
        fit_costs.append({"method":method,"outer_fold":fold,"transform_seconds":meta["transform_seconds"],"estimator_fit_seconds":meta["estimator_fit_seconds"],"prediction_seconds":entry["prediction_seconds"],"bytes":path.stat().st_size,"features":len(model.indices)})
    for method,records in by_method.items():
        check_rows(records,truth,ids);computed=compute(records);compare_metrics(computed,pooled[method],"Pooled benchmark ")
        cost=pooled[method]["cost"];subset=[r for r in fit_costs if r["method"]==method]
        for field,key in [("transform_seconds","outer_transform_seconds"),("estimator_fit_seconds","outer_estimator_fit_seconds"),("prediction_seconds","outer_predict_seconds"),("bytes","model_bytes_sum")]:close(sum(r[field] for r in subset),cost[key],"Reported measured outer cost sum")
        close(float(np.mean([r["features"] for r in subset])),cost["mean_selected_features"],"Reported feature mean")
        all_metrics.append({"experiment":"fair-benchmark","method":method,"seed":42,**computed,"mean_features":cost["mean_selected_features"]})
    historical_r2={r["record_id"]:r for r in rows(historical/"core/outer_predictions.csv.gz") if r["method"]=="R2"}
    for row in by_method["R2_reproduction"]:
        old=historical_r2[row["record_id"]]
        require(row["y_predicted"]==old["y_predicted"] and row["outer_fold"]==old["outer_fold"],"Fresh R2 exact historical label/fold reproduction")
        close([float(row["score_"+c]) for c in CLASSES],[float(old["score_"+c]) for c in CLASSES],"Fresh R2 exact scores")

    strong=BASE/"strong-mlp";require(read(strong/"status.json")["status"]=="completed","StrongMLP actually completed")
    strong_rows=rows(strong/"outer_predictions.csv.gz");require(len(strong_rows)==5130,"Both strongMLP methods x3seeds x855")
    strong_groups=defaultdict(list)
    for row in strong_rows:strong_groups[(row["method"],int(row["seed"]),int(row["outer_fold"]))].append(row)
    strong_pooled=read(strong/"pooled_metrics_by_seed.json");strong_outer={(r["method"],r["seed"],r["outer_fold"]):r for r in read(strong/"outer_metrics.json")}
    strong_methods=list(strong_pooled)
    for fold in range(5):
        names,indices,values,source_labels,mean,variance=source_info(fold)
        for seed in [42,43,44]:
            directory=strong/"models"/f"outer_{fold:02d}"/f"seed_{seed}";meta=read(directory/"training.json")
            require(meta["source_ids"]==names and meta["heldout_ids"]==splits[fold]["test_sample_ids"],"StrongMLP actual disjoint source/query IDs")
            require(meta["source_samples"]==684 and meta["model_seed"]==seed and meta["continuation_seed"]==seed+1600,"StrongMLP actual source sizes/seeds")
            require(meta["source_fingerprint"]==joblib.hash(values) and meta["label_fingerprint"]==joblib.hash(source_labels),"StrongMLP actual source fingerprints")
            require(meta["base_fingerprint"]==meta["base_fingerprint_after_continuation"] and meta["base_epochs"]==200 and meta["continuation_updates_not_epochs"]==100,"StrongMLP paired source training budget")
            require(meta["parameter_count"]==41543 and meta["target_support_labels_used"]==0 and meta["source_only"],"StrongMLP architecture/source scope")
            prep=joblib.load(directory/"preprocessor.joblib");close(prep["scaler"].mean_,mean,"StrongMLP source mean");close(prep["scaler"].var_,variance,"StrongMLP source variance")
            for relative,expected in meta["artifact_sha256"].items():
                path=directory/relative;require(sha(path)==expected,"StrongMLP artifact frozen");before_hashes[path]=sha(path)
            for task in meta["continuation"]["source_tasks"]:require(not set(task["support_ids"])&set(task["query_ids"]) and (set(task["support_ids"])|set(task["query_ids"]))<=set(names),"StrongMLP continuation source-only episode")
            for method,fingerprint in zip(strong_methods,[meta["base_fingerprint"],meta["continued_fingerprint"]],strict=True):
                records=strong_groups[(method,seed,fold)];check_rows(records,truth,splits[fold]["test_sample_ids"]);compare_metrics(compute(records),strong_outer[(method,seed,fold)]["metrics"],"StrongMLP fold ")
                model=DenseClassifier(612,[64,32],7);model.load_state_dict(torch.load(directory/(method+".pt"),map_location="cpu",weights_only=True));model.eval()
                require(state_fingerprint(model)==fingerprint,"StrongMLP exact loaded weights")
                ordered=sorted(records,key=lambda r:r["record_id"]);query=X[[id_index[r["record_id"]] for r in ordered]]
                scores=model_scores(model,transform_features(prep,query));saved=np.asarray([[float(r["score_"+c]) for c in CLASSES] for r in ordered])
                close(scores,saved,"StrongMLP frozen outer score replay",tolerance=1e-4)
                require(np.array_equal(np.asarray(CLASSES)[scores.argmax(1)],[r["y_predicted"] for r in ordered]),"StrongMLP frozen outer label replay")
                model_replays.append({"experiment":"strong-mlp","method":method,"seed":seed,"outer_fold":fold,"source":684,"query":171,"labels_and_scores_replayed":True})
    for method in strong_methods:
        for seed in [42,43,44]:
            records=[r for r in strong_rows if r["method"]==method and int(r["seed"])==seed];check_rows(records,truth,ids)
            computed=compute(records);compare_metrics(computed,strong_pooled[method][str(seed)],"StrongMLP pooled seed ")
            all_metrics.append({"experiment":"strong-mlp","method":method,"seed":seed,**computed,"mean_features":612})
    sensitivity=read(strong/"seed_sensitivity.json")
    for method in strong_methods:
        for name in METRICS:
            values=[r[name] for r in all_metrics if r["experiment"]=="strong-mlp" and r["method"]==method]
            close(np.mean(values),sensitivity[method][name]["mean"],"Allseed sensitivity mean")
            close(np.std(values,ddof=1),sensitivity[method][name]["sample_std"],"Allseed sensitivity sampleSD")

    # Frozen171-source calibration: independent8x8 replay from actual support
    # IDs, reusing each solution for legacy and capped diagnostic predictions.
    cal=BASE/"calibration/run";completed=read(cal/"completed.json")
    require(completed["predictions"]==205200 and completed["evaluations"]==1200,"Calibration completed real row counts")
    cal_rows=rows(cal/"final_predictions.csv.gz");require(len(cal_rows)==205200,"All calibration raw rows")
    cal_groups=defaultdict(list)
    for row in cal_rows:cal_groups[(row["selector"],int(row["repeat"]),row["method"])].append(row)
    cal_metrics={(r["selector"],r["repeat"],r["method"]):r for r in read(cal/"final_metrics.json")}
    require(len(cal_groups)==len(cal_metrics)==1200,"All calibration source/draw/method roles")
    cp=read(ROOT/"data/research/paper_c_calibration_protocol_20261031.json");lock=read(cal/"selection_lock.json")
    plan=read(ROOT/cp["split_plan"]);role_ids={role:sorted(r["record_id"] for r in plan["records"] if r["role"]==role) for role in ["training","calibration","prediction"]}
    require([len(role_ids[role]) for role in role_ids]==[513,171,171],"Unchanged513/171/171 population")
    source_names,pool_names,query_names=(role_ids[k] for k in ["training","calibration","prediction"])
    require(not set(source_names)&set(pool_names) and not set(source_names)&set(query_names) and not set(pool_names)&set(query_names),"Actual calibration roles disjoint")
    summary=read(cal/"summary.json");metadata=read(cal/"final_metadata.json");cal_audited=[];geometry=[]
    for entry in metadata:
        selector=entry["selector"];source_artifact=entry["source_artifact"];path=ROOT/source_artifact["path"]
        require(sha(path)==source_artifact["sha256"],"Original frozen171 source model bytes")
        before_hashes[path]=sha(path);pipe=joblib.load(path)
        require(pipe[-1].break_ties is True and pipe[-1].decision_function_shape=="ovr","Calibration oldargmax score rule retained")
        require(entry["source_ids"]==source_names and entry["query_ids"]==query_names and entry["new_source_fits"]==0,"Actual old source/query identities")
        require(joblib.hash(pipe)==entry["fingerprint_before_after"],"Actual original source fingerprint preserved")
        source_X=X[[id_index[name] for name in source_names]];pool_X=X[[id_index[name] for name in pool_names]];query_X=X[[id_index[name] for name in query_names]]
        prep_source=pipe.named_steps["preprocess"].transform(source_X);scaler=pipe.named_steps["scale"]
        close(scaler.mean_,prep_source.mean(0),"Frozen171 source scaler mean");close(scaler.var_,prep_source.var(0),"Frozen171 source scaler variance")
        require(int(scaler.n_samples_seen_)==513,"Frozen171 source actually fitted513")
        source_scores=pipe.decision_function(source_X);pool_scores=pipe.decision_function(pool_X);query_scores=pipe.decision_function(query_X)
        classes=pipe.classes_;require(classes.tolist()==CLASSES,"Frozen171 class score columns")
        baseline=query_scores.argmax(1);require(np.array_equal(pipe.predict(query_X),classes[baseline]),"Actual oldSVC argmax and predict agree")
        source_order=np.sort(source_scores,axis=1);source_margin=source_order[:,-1]-source_order[:,-2]
        query_order=np.argsort(query_scores,axis=1);query_margin=np.sort(query_scores,axis=1)[:,-1]-np.sort(query_scores,axis=1)[:,-2]
        selected=lock["choices"][selector]["selected"];diagnostic=lock["choices"][selector]["diagnostic"]
        require(selected is None and lock["choices"][selector]["fallback_to_identity"],"Conservative primary keeps source-development identity")
        pool_index={name:i for i,name in enumerate(pool_names)};query_truth=np.asarray([VOCAB[truth[name]] for name in query_names])
        unchanged_predictions={method:0 for method in ["conservative_selected","same_structure_zero_support","same_structure_no_labels","prespecified_development_diagnostic"]}
        for draw in entry["draws"]:
            repeat=draw["repeat"];support_ids=draw["support_ids"]
            require(len(support_ids)==len(set(support_ids))==7 and set(support_ids)<=set(pool_names) and set(truth[name] for name in support_ids)==set(CLASSES),"Actual oneperclass7 support specimens")
            support=pool_scores[[pool_index[name] for name in support_ids]];labels=np.asarray([VOCAB[truth[name]] for name in support_ids])
            design=np.column_stack([support,np.ones(7)]);query_design=np.column_stack([query_scores,np.ones(171)])
            delta=np.linalg.solve(design.T@design+np.eye(8),design.T@(np.eye(7)[labels]-support));calibrator_solutions+=1
            transformed=query_scores+query_design@delta
            errors=int(np.count_nonzero(support.argmax(1)!=labels))
            cap=diagnostic["cap_fraction"]*max(float(np.median(source_margin)),np.finfo(float).eps)
            gate=float(np.quantile(source_margin,diagnostic["source_gate_quantile"]))
            correction=np.clip(transformed-query_scores,-cap,cap)*diagnostic["blend"]
            correction[query_margin>gate]=0.
            rejected=errors<diagnostic["minimum_support_errors"]
            diagnostic_scores=query_scores if rejected else query_scores+correction
            expected_predictions={"no_calibration":baseline,"traditional_lambda1":transformed.argmax(1),"conservative_selected":baseline,
                "same_structure_zero_support":baseline,"same_structure_no_labels":baseline,"prespecified_development_diagnostic":diagnostic_scores.argmax(1)}
            if not rejected:
                top=query_order[:,-1];second=query_order[:,-2];eligible=(query_margin<=2*diagnostic["blend"]*cap)&(query_margin<=gate)
                geometry.append({"selector":selector,"repeat":repeat,"eligible_rows":int(eligible.sum()),
                    "equal_top_two_clipped_shift_rows":int(np.count_nonzero(eligible & (correction[np.arange(171),top]==correction[np.arange(171),second]))),
                    "diagnostic_flips":int(np.count_nonzero(expected_predictions["prespecified_development_diagnostic"]!=baseline))})
            for method,prediction in expected_predictions.items():
                key=(selector,repeat,method);records=cal_groups[key];check_rows(records,truth,query_names)
                saved={r["record_id"]:r for r in records};require(np.array_equal(classes[prediction],[saved[name]["y_predicted"] for name in query_names]),"Independent anchored/capped/zero exact prediction replay")
                computed=compute(records);compare_metrics(computed,cal_metrics[key],"Calibration perdraw ")
                corrected=int(np.count_nonzero((baseline!=query_truth)&(prediction==query_truth)));harmed=int(np.count_nonzero((baseline==query_truth)&(prediction!=query_truth)))
                require(corrected==cal_metrics[key]["corrected"] and harmed==cal_metrics[key]["harmed"],"Calibration corrected/harmed recomputed from true labels")
                baseline_records=cal_groups[(selector,repeat,"no_calibration")];base_stats=compute(baseline_records)
                for name in METRICS:close(computed[name]-base_stats[name],cal_metrics[key][name+"_gain"],"Calibration paired metric gain")
                require(cal_metrics[key]["support_error_count"]==errors and cal_metrics[key]["support_evidence_rejected"]==rejected,"Calibration support evidence thresholds")
                unchanged=int(np.count_nonzero(prediction==baseline));require(unchanged==cal_metrics[key]["unchanged_query_predictions"],"Calibration unchanged sample count")
                if method in unchanged_predictions:unchanged_predictions[method]+=unchanged
                cal_audited.append({"selector":selector,"repeat":repeat,"method":method,**{k:computed[k] for k in [*METRICS,"correct","total"]},
                    "corrected":corrected,"harmed":harmed,"unchanged":unchanged})
        require(all(v==17100 for v in unchanged_predictions.values()),"All identity and cappeddiagnostic label controls unchanged")
    for record in summary:
        subset=[r for r in cal_audited if r["selector"]==record["selector"] and r["method"]==record["method"]]
        require(len(subset)==record["support_draws"]==100,"Calibration summary all100 draws")
        for name in (*METRICS,"corrected","harmed"):
            values=[r[name] for r in subset]
            for suffix,value in [("mean",np.mean(values)),("sd",np.std(values)),("min",min(values)),("max",max(values))]:close(value,record[name+"_"+suffix],"Calibration support-draw summary")
    require(calibrator_solutions==200,"Exactly200 disclosed8x8support reconstruction solves")
    for path,digest in before_hashes.items():require(sha(path)==digest,"Read-only audit preserved original/new model bytes")
    for record in all_metrics:
        for i,c in enumerate(CLASSES):class_metrics.append({"experiment":record["experiment"],"method":record["method"],"seed":record["seed"],"class":c,**{k:record[k][i] for k in ["precision","recall","f1","support","predicted_count"]}})
    best=max([r for r in all_metrics if r["seed"]==42],key=lambda r:r["accuracy"])
    result={"status":"pass","started_at_utc":started,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,
        "maximum_numeric_difference":maximum_difference,"new_estimator_fits":0,"new_calibrator_solutions":calibrator_solutions,
        "calibrator_solution_meaning":"200 deterministic8x8anchored-ridge systems reconstructed on recorded7sample supports, reused for traditional and capped diagnostic; no tuning/new classifiers.",
        "global_release_sha256":sha(release_path),"actual_execution":execution,"benchmark_prediction_rows":len(benchmark_rows),
        "strong_mlp_prediction_rows":len(strong_rows),"calibration_prediction_rows":len(cal_rows),"total_new_final_prediction_rows":len(benchmark_rows)+len(strong_rows)+len(cal_rows),
        "benchmark_model_roles_replayed":45,"strong_mlp_model_roles_replayed":30,"frozen_calibration_sources_replayed":2,
        "model_replays":model_replays,"same855_metrics":all_metrics,"source_cost_records":fit_costs,
        "calibration_draw_cases":cal_audited,"calibration_capped_geometry":geometry,"current_best_prespecified_seed42":{k:best[k] for k in ["experiment","method","seed",*METRICS,"correct","total"]},
        "findings":["Fresh R2 reproduces every855 historical label and all7scores exactly:821correct/96.0234%, MacroF1 .9196846, BalancedAccuracy .8991837.",
                    "Same855 fullSVM, shrinkageLDA, sklearnMLP and exact oldstrongMLP controls all report every prescribed seed; none replaces main R2 from a better cherry-picked seed.",
                    "Traditional calibration and main conservative identity/zero/no-label/capped diagnostic roles independently replayed on205200 raw rows; corrected/harmed counts are actual paired counts.",
                    "Primary conservative methods fell back to identity before final scores. No accuracy gain claimed for safety/no-op; equal clipped shifts provide a narrow mechanism diagnostic, not evidence against all calibration.",
                    "200 support ridge solutions disclosed separately from zero new estimator fits.",
                    "Source/query isolation, exact source statistics, frozen source/protocol/locks, original historical file indices, actual postrelease execution all pass.",
                    "Entire855 and old171queries historically exposed; oldb006 architecture also historically selected from513overlapping newfold queries. Results remain retrospective development controls, not external confirmation."]}
    (OUT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    def write_csv(path,records):
        with path.open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]),lineterminator="\n");writer.writeheader();writer.writerows(records)
    write_csv(OUT/"same855_model_metrics.csv",[{k:r[k] for k in ["experiment","method","seed","correct","total",*METRICS,"mean_features"]} for r in all_metrics])
    write_csv(OUT/"same855_class_metrics.csv",class_metrics)
    write_csv(OUT/"calibration_per_draw.csv",cal_audited)
    write_csv(OUT/"calibration_capped_geometry.csv",geometry)
    table=["| 同855方法 | 种子 | 正确数 | Accuracy | Macro-F1 | Balanced Accuracy |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in all_metrics:table.append(f"| {r['method']} | {r['seed']} | {r['correct']} | {100*r['accuracy']:.4f}% | {r['macro_f1']:.4f} | {100*r['balanced_accuracy']:.4f}% |")
    text="\n".join(["# 本轮最终结果独立审计","",f"{checks}项检查通过，最大数值差{maximum_difference:.3g}。重新加载45个公平对照管线、30个强MLP权重角色及2个冻结171协议源模型；复算7695＋5130＋205200＝218025条本轮最终预测。无新增分类器拟合，独立重构200个8×8支持集ridge解，已显式记录。","",*table,"","原R2的855个标签及7列分数精确复现，仍为821份正确。统一对照和所有种子均保留；支持校准205200条记录的纠正／伤害、零支持与同结构无标签身份均逐项复算。旧96.02%继续是当前预设seed42候选中最高的总体准确率。","","所有实际最终调用晚于统一放行；三个选择锁、协议、源码和旧历史SHA索引未变。每外折源684／外测171互斥，所有缩放统计来自源；原171校准模型仍是513源训练，不冒充R2。","","保守方法在开发阶段已选identity，最终保持不动不是准确率提升。所有结果是已曝光数据上的回顾性同批验证；旧强MLP架构的历史513选择与新855查询有历史重叠，不能将其架构选择称全新无偏嵌套确认。","","[完整审计](audit.json) · [同855模型表](same855_model_metrics.csv) · [逐类指标](same855_class_metrics.csv) · [校准逐抽样](calibration_per_draw.csv)","","重现：`.runtime/bin/python reports/paper-c/2026-10-31/final-audit/audit.py`。只读模型重放和支持集代数重构，不重新训练分类器或调参。"])
    (OUT/"README.md").write_text(text+"\n")
    print(json.dumps({"status":"pass","checks":checks,"maximum_difference":maximum_difference,"new_estimator_fits":0,
        "new_calibrator_solutions":calibrator_solutions,"final_prediction_rows":218025,"current_best":result["current_best_prespecified_seed42"]},ensure_ascii=False))


if __name__=="__main__":main()
