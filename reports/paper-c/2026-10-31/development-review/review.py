"""Read-only pre-evaluation review: no estimator/calibrator fit, no outer scoring."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
CLASSES = [f"ZY{i:02d}" for i in range(1, 8)]
VOCAB = {c: i for i, c in enumerate(CLASSES)}
checks = 0
maximum_difference = 0.0


def require(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def close(actual, expected, label, tolerance=1e-12):
    global maximum_difference
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    maximum_difference = max(maximum_difference, difference)
    require(difference <= tolerance, f"{label}: {difference}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metric(matrix):
    tp, source, found = matrix.diagonal(), matrix.sum(1), matrix.sum(0)
    recall = np.divide(tp, source, out=np.zeros(7), where=source != 0)
    f1 = np.divide(2 * tp, source + found, out=np.zeros(7), where=source + found != 0)
    return {"accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "correct": int(tp.sum())}


def main():
    started = datetime.now(timezone.utc).isoformat()
    data = np.load(ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz", allow_pickle=False)
    X, y, ids = (data[k] for k in ["X_alpha", "y", "record_ids"])
    truth = dict(zip(ids.tolist(), y.tolist(), strict=True))
    id_index = {name: i for i, name in enumerate(ids)}
    old = ROOT / "reports/staged-nested-optimization/2026-10-31/core"
    splits = read(old / "splits.json")
    historical = read(old / "selection_lock.json")
    baseline_state = {}
    benchmark = ROOT / "outputs/paper-c-20261031/fair-benchmark"
    bp = ROOT / "data/research/paper_c_benchmark_protocol_20261031.json"
    protocol, manifest, lock = (read(p) for p in [bp, benchmark / "run_manifest.json", benchmark / "selection_lock.json"])
    require(not (benchmark / "outer_predictions.csv.gz").exists(), "Benchmark outer predictions absent during review")
    require(read(benchmark / "splits.json") == splits, "New benchmark uses exact old outer and inner folds")
    require(manifest["protocol_sha256"] == sha(bp), "Benchmark frozen protocol")
    require(manifest["input_sha256"] == protocol["input_sha256"] == sha(ROOT / protocol["input"]), "Benchmark frozen input")
    require(sha(ROOT / protocol["historical_protocol"]) == protocol["historical_protocol_sha256"], "Historical protocol actual hash")
    require(manifest["historical_selection_lock_sha256"] == protocol["historical_selection_lock_sha256"] == sha(old / "selection_lock.json"), "Historical R2 lock bound")
    for path, expected in manifest["source_sha256"].items():
        require(sha(ROOT / path) == expected, "Benchmark code unchanged")
    require(lock["manifest_sha256"] == sha(benchmark / "run_manifest.json"), "Benchmark lock/manifest")
    require(lock["inner_metrics_sha256"] == sha(benchmark / "all_inner_metrics.json"), "Benchmark raw metrics frozen")
    require(lock["inner_predictions_sha256"] == sha(benchmark / "inner_predictions.csv.gz"), "Benchmark raw predictions frozen")
    require(lock["outer_test_used_to_select"] is False, "Benchmark outer labels not declared in selection")
    inner_metrics = read(benchmark / "all_inner_metrics.json")
    require(len(inner_metrics) == lock["inner_evaluations"] == 1140, "Benchmark1140 inner cases")
    expected = {(r["outer_fold"], r["inner_fold"], r["family"], r["candidate_id"]): r for r in inner_metrics}
    require(len(expected) == 1140, "Unique benchmark case identities")
    matrices, query_ids = {}, defaultdict(set)
    with gzip.open(benchmark / "inner_predictions.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["outer_fold"]), int(row["inner_fold"]), row["family"], row["candidate_id"])
            require(row["y_true"] == truth[row["record_id"]], "Benchmark raw labels")
            require(row["record_id"] not in query_ids[key], "Benchmark duplicate prediction")
            query_ids[key].add(row["record_id"])
            matrices.setdefault(key, np.zeros((7, 7), dtype=int))[VOCAB[row["y_true"]], VOCAB[row["y_predicted"]]] += 1
    require(set(matrices) == set(expected), "Every benchmark case has raw predictions")
    for key, record in expected.items():
        outer, inner = key[:2]
        source_ids = splits[outer]["inner_folds"][inner]["train_sample_ids"]
        valid_ids = splits[outer]["inner_folds"][inner]["validation_sample_ids"]
        require(record["status"] == "ok" and record["source_ids"] == source_ids and record["validation_ids"] == valid_ids, "Benchmark source membership")
        require(query_ids[key] == set(valid_ids) and int(matrices[key].sum()) == 342, "Benchmark complete disjoint validation set")
        for name in METRICS:
            close(metric(matrices[key])[name], record["metrics"][name], "Benchmark inner metric raw recomputation")
        require(matrices[key].tolist() == record["metrics"]["confusion_matrix"], "Benchmark inner confusion matrix")
        meta = record["model_metadata"]
        require(meta["model_seed"] == 42 and meta["source_samples"] == 342 and meta["fit_scope"] == "current_training_partition_only", "Benchmark fit scope")
        if (outer, inner) not in baseline_state:
            indices = [id_index[name] for name in source_ids]
            values = X[indices]
            snv = (values - values.mean(1, keepdims=True)) / values.std(1, ddof=1, keepdims=True)
            baseline_state[(outer, inner)] = (joblib.hash(values), joblib.hash(y[indices]), snv.mean(0), snv.var(0))
        fingerprint, labels_hash, mean, variance = baseline_state[(outer, inner)]
        require(meta["source_fingerprint"] == fingerprint and meta["source_label_fingerprint"] == labels_hash, "Benchmark actual training numeric/label fingerprints")
        close(meta["scaler_mean"], mean, "Benchmark source-only SNV mean")
        close(meta["scaler_variance"], variance, "Benchmark source-only SNV variance")
    selected = []
    for fold in lock["outer_folds"]:
        outer = fold["outer_fold"]
        for family, choice in fold["choices"].items():
            means = {}
            for candidate in protocol["candidate_grids"][family]:
                means[candidate["id"]] = {m:float(np.mean([metric(matrices[(outer, i, family, candidate["id"])])[m] for i in [0, 1]])) for m in METRICS}
            for cid, values in means.items():
                for name in METRICS:
                    close(values[name], choice["all_candidate_means"][cid][name], "Benchmark means from raw predictions")
            winner = min(means, key=lambda cid: (-means[cid]["macro_f1"], -means[cid]["balanced_accuracy"], -means[cid]["accuracy"], cid))
            require(winner == choice["selected"]["id"] and choice["selected_using_outer_test"] is False, "Benchmark winner source-only exact rule")
            selected.append({"outer_fold": outer, "family": family, "selected": choice["selected"]})
        old_r2 = historical["outer_folds"][outer]["choices"]["R2"]["selected"]
        controls = fold["conditional_controls"]
        original = {"family":"conditional_svm", **old_r2["svm"], "mc":old_r2["mc"]}
        for name, candidate in controls.items():
            expected_control = dict(original, id=name)
            if name == "R2_full_features": expected_control.pop("mc")
            if name == "R2_random_K":
                expected_control.pop("mc")
                expected_control["random_K"] = int(np.ceil(612 * old_r2["mc"]["ratio"]))
            if name == "R2_no_C_gamma_search": expected_control.update(C=10., gamma="scale")
            if name == "R2_no_class_weight": expected_control["class_weight"] = "none"
            if name == "R2_fixed_MC20_LV2_K62": expected_control["mc"] = {"runs":20, "components":2, "ratio":.1}
            require(candidate == expected_control, "Conditional control inheritance exact")
    benchmark_result = {"status":"pass", "allow_outer_evaluation":True, "new_model_fits":0,
                        "inner_cases":len(expected),"inner_predictions":sum(int(m.sum()) for m in matrices.values()),
                        "selection_lock_sha256":sha(benchmark / "selection_lock.json"), "run_manifest_sha256":sha(benchmark / "run_manifest.json"),
                        "protocol_sha256":sha(bp),"selected":selected,
                        "limitations":["Different family search budgets96/10/8 and MC additional36 disclosed.",
                                       "Conditional ablations inherit train-internal historical R2 parameters; not factorial causal main effects.",
                                       "R2 and SVM controls must preserve SVC.predict voting; score argmax is a different rule.",
                                       "Historical protocol actual hash verified by review although runtime does not independently enforce the declared historical_protocol_sha256 field.",
                                       "All855 already exposed; no independent-confirmation claim."]}

    # Calibration source development is entirely within the old513 training
    # samples. Only saved development scores and predictions are inspected.
    calibration = ROOT / "outputs/paper-c-20261031/calibration/run"
    cp = ROOT / "data/research/paper_c_calibration_protocol_20261031.json"
    p, manifest, lock = (read(path) for path in [cp, calibration / "run_manifest.json", calibration / "selection_lock.json"])
    require(not (calibration / "final_predictions.csv.gz").exists(), "Calibration final predictions absent during review")
    require(manifest["protocol_sha256"] == sha(cp), "Calibration protocol frozen")
    require(manifest["input_sha256"] == sha(ROOT / p["input"]) and manifest["split_sha256"] == sha(ROOT / p["split_plan"]), "Calibration input/split frozen")
    require(lock["run_manifest_sha256"] == sha(calibration / "run_manifest.json"), "Calibration lock/manifest frozen")
    for path, expected_hash in manifest["source_code_sha256"].items():
        require(sha(ROOT / path) == expected_hash, "Calibration executed source unchanged")
    frozen_source_flags = []
    for source in manifest["frozen_sources"]["models"]:
        require(sha(ROOT / source["path"]) == source["sha256"], "Original frozen source weights unchanged")
        pipe = joblib.load(ROOT / source["path"])
        require(pipe[-1].break_ties is True and pipe[-1].decision_function_shape == "ovr", "Calibration source argmax/OVR alignment")
        frozen_source_flags.append({"selector":source["selector"], "break_ties":pipe[-1].break_ties,
                                    "decision_function_shape":pipe[-1].decision_function_shape,"source_samples":source["training_sample_count"]})
    split = read(ROOT / p["split_plan"])
    training_ids = {r["record_id"] for r in split["records"] if r["role"] == "training"}
    final_ids = {r["record_id"] for r in split["records"] if r["role"] != "training"}
    budget = read(calibration / "development_budget.json")
    require(len(budget) == lock["source_fits"] == 60 and lock["candidate_evaluations"] == 14400, "Calibration real source/evaluation counts")
    draw_map, margin_details, metrics_by_case = {}, [], {}
    for entry in budget:
        selector, episode = entry["selector"], entry["episode"]
        source, pool, query = [set(entry[k]) for k in ["source_ids","support_pool_ids","query_ids"]]
        require((len(source),len(pool),len(query)) == (307,103,103), "Development actual partition sizes")
        require(not source & pool and not source & query and not pool & query and source | pool | query == training_ids, "Development roles disjoint within513")
        require(not (source | pool | query) & final_ids, "No fixed support/query leakage into development")
        score_file = calibration / "development" / selector / f"episode_{episode:02d}_scores.npz"
        with np.load(score_file, allow_pickle=False) as scores:
            require(set(scores["source_ids"]) == source and set(scores["pool_ids"]) == pool and set(scores["query_ids"]) == query, "Saved score identities")
            require(scores["classes"].tolist() == CLASSES, "Score class vocabulary")
            for rid, label in zip(scores["query_ids"], scores["query_y"], strict=True): require(truth[rid] == label, "Development query labels")
            ordered_source = np.sort(scores["source_scores"], axis=1)
            source_margin = ordered_source[:,-1] - ordered_source[:,-2]
            ordered_query = np.sort(scores["query_scores"], axis=1)
            query_margin = ordered_query[:,-1] - ordered_query[:,-2]
            baseline = np.argmax(scores["query_scores"], axis=1)
            baseline_truth = np.asarray([VOCAB[truth[rid]] for rid in scores["query_ids"]])
            cm = np.zeros((7,7),dtype=int)
            for actual,predicted in zip(baseline_truth,baseline,strict=True):cm[actual,predicted] += 1
            baseline_stats = metric(cm)
            for candidate in p["candidates"]:
                cap = candidate["cap_fraction"] * max(float(np.median(source_margin)), np.finfo(float).eps)
                gate = float(np.quantile(source_margin,candidate["source_gate_quantile"]))
                bound = 2 * candidate["blend"] * cap
                potentially_changeable = (query_margin <= bound) & (query_margin <= gate)
                margin_details.append({"selector":selector,"episode":episode,"candidate_id":candidate["id"],
                    "cap":cap,"gate_margin":gate,"maximum_pairwise_score_change":bound,"query_minimum_margin":float(query_margin.min()),
                    "query_rows":len(query_margin),"potentially_changeable_by_bound":int(potentially_changeable.sum())})
            for draw in entry["draws"]:
                support = draw["ids"]
                require(len(support) == len(set(support)) == 7 and set(support) <= pool, "Actual support7 distinct pool specimens")
                require(set(truth[rid] for rid in support) == set(CLASSES), "Actual one support per class")
                draw_map[(selector,episode,draw["repeat"])] = support
            with gzip.open(calibration / "development" / selector / f"episode_{episode:02d}_predictions.csv.gz", "rt") as handle:
                groups = defaultdict(lambda:np.zeros((7,7),dtype=int))
                identities = defaultdict(set)
                for row in csv.DictReader(handle):
                    key = (selector,int(row["episode"]),int(row["repeat"]),row["candidate_id"])
                    require(row["record_id"] in query and row["y_true"] == truth[row["record_id"]], "Calibration raw prediction source-only label")
                    require(row["record_id"] not in identities[key], "Calibration repeated row")
                    identities[key].add(row["record_id"])
                    groups[key][VOCAB[row["y_true"]],VOCAB[row["y_predicted"]]] += 1
                for key,matrix in groups.items():
                    require(identities[key] == query and int(matrix.sum()) == 103, "Calibration all103 development query rows")
                    values = metric(matrix)
                    metrics_by_case[key] = {**values, **{name+"_gain":values[name]-baseline_stats[name] for name in METRICS}}
    total_metrics = 0
    candidate_behavior = []
    for selector in p["selectors"]:
        with gzip.open(calibration / "development" / selector / "metrics.csv.gz", "rt") as handle:
            records = list(csv.DictReader(handle))
        require(len(records) == 7200, "Calibration candidate count per source")
        for record in records:
            key = (selector,int(record["episode"]),int(record["repeat"]),record["candidate_id"])
            for name in (*METRICS,*(name+"_gain" for name in METRICS)):
                close(metrics_by_case[key][name],float(record[name]),"Calibration metric reconstructed from raw predictions")
            require(int(record["corrected"]) == int(record["harmed"]) == 0, "Saved calibration inert result explicitly retained")
            total_metrics += 1
        choice = lock["choices"][selector]
        for stage in ["selection","gate"]:
            for candidate in p["candidates"]:
                subset = [r for r in records if r["candidate_id"] == candidate["id"] and r["stage"] == stage]
                for name in (*METRICS,*(name+"_gain" for name in METRICS),"corrected","harmed"):
                    close(float(np.mean([float(r[name]) for r in subset])),choice["aggregate"][stage][candidate["id"]][name],"Calibration selection aggregates")
                require(len(subset) == choice["aggregate"][stage][candidate["id"]]["evaluations"], "Calibration selection/gate counts")
        require(choice["selected"] is None and choice["proposed"] is None and choice["fallback_to_identity"], "No positive selection gain; exact fallback")
        require(all(float(r["accuracy_gain"]) == float(r["macro_f1_gain"]) == float(r["balanced_accuracy_gain"]) == 0 for r in records), "All calibration candidates no observed classification effect")
        candidate_behavior.append({"selector":selector,"candidate_evaluations":len(records),
                                   "support_rejected":sum(r["support_rejected"] == "True" for r in records),
                                   "corrected":0,"harmed":0,"selected":None,"fallback_to_identity":True})
    largest = [r for r in margin_details if r["candidate_id"] == "cap0.25_q0.75_m1"]
    margin_summary = [{"selector":selector,"unique_development_query_occurrences":sum(r["query_rows"] for r in largest if r["selector"] == selector),
                       "potentially_changeable_by_maximum_bound":sum(r["potentially_changeable_by_bound"] for r in largest if r["selector"] == selector)} for selector in p["selectors"]]
    calibration_result = {"status":"pass_with_inert_method_limitation","allow_final_evaluation":True,"new_model_fits":0,
        "new_calibrator_fits":0,"final_query_scored":False,"source_fits_in_original_selection":len(budget),
        "development_cases":total_metrics,"development_prediction_rows":sum(int(r["correct"]*0+103) for r in metrics_by_case.values()),
        "selection_lock_sha256":sha(calibration / "selection_lock.json"),"run_manifest_sha256":sha(calibration / "run_manifest.json"),
        "protocol_sha256":sha(cp),"frozen_source_prediction_flags":frozen_source_flags,"candidate_behavior":candidate_behavior,
        "margin_summary":margin_summary,
        "interpretation":["Both source tasks must remain identity after development; no claim of successful calibration or accuracy improvement.",
            "Small cap does not mathematically prohibit every possible flip: some development score gaps lie within the2*blend*cap bound.",
            "Componentwise clipping of raw anchored OVR-to-onehot residuals can saturate top two components at the same negative bound, producing an equal translation that cannot alter ranking. This is a mechanistic hypothesis here, not independently refitted/calculated.",
            "No new adapter fits performed under read-only review. Any quantitative saturation proof requires a separately authorized replay from saved source/support scores.",
            "Zero-support/no-label branches return an exact copy because ridge=None; they are identity controls, not positive calibration evidence.",
            "100 support draws overlap and share the same soleZY02 calibration specimen; no3/5/10shots or cross-batch claim.",
            "Calibration uses old fixed171 argmax/OVR rule with both savedSVC break_ties=True, distinct from R0/R2 standard vote.",
            "Runtime evaluation checks frozen manifest/current sources and release lock SHA, but external reviewer must additionally bind actual manifest hash and explicit dated allow flag; this audit verifies the actual current lock-manifest link."]}
    result = {"status":"pass_for_prespecified_evaluation_with_limitations","started_at_utc":started,
              "completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"maximum_numeric_difference":maximum_difference,
              "new_estimator_fits":0,"new_calibrator_fits":0,"outer_or_final_query_scored":False,
              "benchmark":benchmark_result,"calibration":calibration_result}
    (OUT / "review.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    with (OUT / "calibration_margin_bounds.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(margin_details[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(margin_details)
    text = f"""# 新开发锁的外测前独立审查

审查完成UTC：{result['completed_at_utc']}。{checks}项检查通过，最大数值差{maximum_difference:.3g}。新增分类器拟合0、校准器拟合0、外层／最终预测集评分0。

公平对照：1140内层候选评价、389880条原始预测全部复算；每个源342／验证342及原外测171互斥，源数值和标签指纹、SNV后缩放统计、15个逐折分类器赢家及30个条件消融设置均通过。按实际冻结hash允许预设外层评价。各家族搜索预算及特征维数不相等，条件消融不是全因子因果主效应。

校准：原60个源模型任务、14400候选评价、1483200条开发原始预测复算通过，全部只在旧513训练记录内进行；源／支持池／查询307／103／103互斥。两个源的所有候选均无改对／改错且开发回退identity。原冻结SVC均break_ties=True，argmax规则与旧171协议一致；不替换855 OOF的标准投票。

保守机制当前没有分类收益，不能把零损害称为校准成功。最大cap下，full的3090个开发查询出现次数中23个、MC20中33个满足可能改判的margin上界，故“cap小到所有行都不可能改判”不成立。原始OVR到onehot残差可能同时在前两类负向饱和，产生等量平移，是值得核验的结构原因；本次遵照只读要求未重新拟合ridge，未把该解释伪装成定量证明。详见[分数差界限](calibration_margin_bounds.csv)。

允许按冻结协议评价传统校准与预设诊断，同时主保守方法保持开发已选identity。不得根据最终结果重选cap或把诊断替换为主方法。100个重叠支持抽样仍共享唯一ZY02支持样品；不构成更多独立样品或批次。

两个运行的hash和当前源码一致。公平运行未在执行器里另行强制校验historical_protocol_sha256字段，本审查已经核对其真实SHA；校准执行器的release条件较宽，父任务应使用显式日期／allow标记并同时绑定本审查给出的manifest、协议及锁SHA。当前实际锁均与各自manifest一致，未发现需阻塞本次预定评价的数据泄漏问题。

[完整审查及锁SHA](review.json)。本审查仅涉及开发数据和锁，未查看任何本次新外层分数。
"""
    (OUT / "README.md").write_text(text)
    print(json.dumps({"status":result["status"],"checks":checks,"max_difference":maximum_difference,
                      "benchmark_cases":len(expected),"calibration_cases":total_metrics,"margin_summary":margin_summary},ensure_ascii=False))


if __name__ == "__main__":
    main()
