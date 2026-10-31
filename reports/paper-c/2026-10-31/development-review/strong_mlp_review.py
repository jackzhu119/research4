"""Pre-outer review of fixed historical strong MLP; no fitting or outer scoring."""
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
RUN = ROOT / "outputs/paper-c-20261031/strong-mlp"
PROTOCOL = ROOT / "data/research/paper_c_strong_mlp_protocol_20261031.json"
CLASSES = [f"ZY{i:02d}" for i in range(1, 8)]
VOCAB = {name: i for i, name in enumerate(CLASSES)}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
METHODS = ["strong_mlp_200", "strong_mlp_200_plus100_source_updates"]
checks = 0
maximum_score_difference = 0.0


def require(value, message):
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-12):
    require(float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))) <= tolerance, message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metrics(matrix):
    tp, actual, found = matrix.diagonal(), matrix.sum(1), matrix.sum(0)
    recall = np.divide(tp, actual, out=np.zeros(7), where=actual != 0)
    f1 = np.divide(2 * tp, actual + found, out=np.zeros(7), where=actual + found != 0)
    return {"accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean())}


def main():
    global maximum_score_difference
    started = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    require(not (RUN / "outer_predictions.csv.gz").exists(), "No new outer scores before review")
    protocol = read(PROTOCOL)
    manifest, lock = read(RUN / "run_manifest.json"), read(RUN / "selection_lock.json")
    require(manifest["protocol_sha256"] == lock["protocol_sha256"] == sha(PROTOCOL), "StrongMLP protocol frozen")
    require(manifest["input_sha256"] == protocol["input_sha256"] == sha(ROOT / protocol["input"]), "StrongMLP input frozen")
    require(manifest["historical_protocol_sha256"] == protocol["historical_protocol_sha256"] == sha(ROOT / protocol["historical_protocol"]), "Exact historical split protocol")
    require(manifest["old_architecture_selection_lock_sha256"] == protocol["old_architecture_selection_lock_sha256"] == sha(ROOT / protocol["old_architecture_selection_lock"]), "Historical architecture lock")
    require(read(ROOT / protocol["old_architecture_selection_lock"])["choice"]["selected"] == protocol["candidate"] == lock["fixed_candidate"], "Exact oldb006 reused")
    require(protocol["candidate"]["hidden_sizes"] == [64, 32] and protocol["candidate"]["epochs"] == 200 and protocol["candidate"]["pca_components"] is None and protocol["candidate"]["class_weight"] == "none", "Original architecture and base budget")
    require(protocol["continuation"]["updates_not_epochs"] == 100 and protocol["continuation"]["helper"] == "image_probability.supervised_continuation", "Exact old supervised helper")
    require(lock["all_seeds"] == protocol["seeds"] == [42, 43, 44] and lock["primary_seed"] == protocol["primary_seed"] == 42, "All prespecified seeds and primary42")
    require(lock["fixed_methods"] == protocol["methods"] == METHODS and lock["no_architecture_or_seed_selected"] and lock["outer_test_used_to_select"] is False, "No new architecture or seed selection")
    require(lock["manifest_sha256"] == sha(RUN / "run_manifest.json") and lock["inner_metrics_sha256"] == sha(RUN / "inner_metrics.json") and lock["inner_predictions_sha256"] == sha(RUN / "inner_predictions.csv.gz"), "Lock binds all source evidence")
    for path, expected in manifest["source_sha256"].items():
        require(sha(ROOT / path) == expected and sha(RUN / "source_snapshot" / path) == expected, "Executed source snapshot exact")
    for relative, expected in lock["source_artifact_sha256"].items():
        require(sha(RUN / relative) == expected, "All saved source artifacts frozen")
    require(len(lock["source_artifact_sha256"]) == 120, "Thirty source contexts each four original files")
    splits = read(ROOT / "reports/staged-nested-optimization/2026-10-31/core/splits.json")
    require(read(RUN / "splits.json") == splits, "Exact old5outer/2inner folds")
    dataset = np.load(ROOT / protocol["input"], allow_pickle=False)
    X, y, ids = (dataset[key] for key in ["X_alpha", "y", "record_ids"])
    index = {name: i for i, name in enumerate(ids)}
    truth = dict(zip(ids.tolist(), y.tolist(), strict=True))
    with gzip.open(RUN / "inner_predictions.csv.gz", "rt") as handle:
        raw = list(csv.DictReader(handle))
    require(len(raw) == lock["inner_prediction_rows"] == 20520, "Thirty contexts x two methods x342 validation rows")
    groups = defaultdict(list)
    for row in raw:
        key = (int(row["outer_fold"]), int(row["inner_fold"]), int(row["seed"]), row["method"])
        require(row["y_true"] == truth[row["record_id"]] and int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "True canonical validation labels")
        groups[key].append(row)
    expected_metrics = {(row["outer_fold"], row["inner_fold"], row["seed"], row["method"]): row for row in read(RUN / "inner_metrics.json")}
    require(len(expected_metrics) == len(groups) == lock["method_cases"] == 60 and lock["source_contexts"] == 30, "Complete inner method roles")
    source_contexts, source_task_total = [], 0
    for outer in range(5):
        fold = splits[outer]
        for inner in range(2):
            source_ids = fold["inner_folds"][inner]["train_sample_ids"]
            valid_ids = fold["inner_folds"][inner]["validation_sample_ids"]
            require(not set(source_ids) & set(valid_ids) and not (set(source_ids) | set(valid_ids)) & set(fold["test_sample_ids"]), "Source/inner query/outer query disjoint")
            indices = [index[name] for name in source_ids]
            source_values = X[indices]
            snv = (source_values - source_values.mean(1, keepdims=True)) / source_values.std(1, ddof=1, keepdims=True)
            counts = {c: int(np.count_nonzero(y[indices] == c)) for c in CLASSES}
            require(counts["ZY02"] == 2, "ZY02 two actual inner-source specimens")
            for seed in [42, 43, 44]:
                directory = RUN / "selection" / f"outer_{outer:02d}" / f"inner_{inner}" / f"seed_{seed}"
                meta = read(directory / "training.json")
                require(meta["candidate"] == protocol["candidate"] and meta["source_ids"] == source_ids and meta["validation_ids"] == valid_ids, "Source-specific candidate and identities")
                require(meta["source_samples"] == 342 and meta["input_features"] == 612 and meta["model_seed"] == seed and meta["continuation_seed"] == seed + 1600, "Actual source count and both seeds")
                require(meta["source_fingerprint"] == joblib.hash(source_values) and meta["label_fingerprint"] == joblib.hash(y[indices]), "Actual source numeric and label fingerprints")
                require(meta["parameter_count"] == 41543 and meta["base_epochs"] == 200 and meta["continuation_updates_not_epochs"] == 100, "Actual architecture and distinct update units")
                require(meta["source_only"] and not meta["query_statistics_fitted"] and meta["target_support_labels_used"] == 0 and not meta["early_stopping"], "Source-only fixed training")
                require(meta["base_fingerprint"] == meta["base_fingerprint_after_continuation"], "Continuation does not mutate base")
                require(len(meta["base_losses"]) == 200 and len(meta["continuation"]["losses"]) == 100 and meta["continuation"]["updates"] == 100, "All200 base epochs and100 continued updates logged")
                require(all(np.isfinite(row["mean_batch_cross_entropy"]) for row in meta["base_losses"]), "Finite actual base losses")
                continuation = meta["continuation"]
                require(continuation["source_only"] and continuation["seed"] == seed + 1600 and len(continuation["source_tasks"]) == 100, "Actual source continuation tasks")
                for task in continuation["source_tasks"]:
                    si, qi = task["support_ids"], task["query_ids"]
                    require(len(si) == len(set(si)) == len(qi) == len(set(qi)) == 3 and not set(si) & set(qi), "Disjoint3way1support1query source task")
                    require(set(si) | set(qi) <= set(source_ids), "Source episode never uses validation or outer query")
                    require(set(truth[name] for name in si) == set(truth[name] for name in qi) and len(set(truth[name] for name in si)) == 3, "Source episodic class coverage")
                    source_task_total += 1
                prep = joblib.load(directory / "preprocessor.joblib")
                require(prep["source_samples"] == 342 and prep["pca"] is None, "Original noPCA source preprocessor")
                close(prep["scaler"].mean_, snv.mean(0), "Actual source-only scaler mean")
                close(prep["scaler"].var_, snv.var(0), "Actual source-only scaler variance")
                close(meta["scaler_mean"], prep["scaler"].mean_, "Recorded source scaler mean")
                close(meta["scaler_variance"], prep["scaler"].var_, "Recorded source scaler variance")
                for name, expected in meta["artifact_sha256"].items(): require(sha(directory / name) == expected, "Paired source model/preprocessor bytes")
                torch.manual_seed(seed)
                initializer = DenseClassifier(612, [64, 32], 7)
                require(state_fingerprint(initializer) == meta["initial_fingerprint"], "Exact seeded base initializer")
                for method, fingerprint in zip(METHODS, [meta["base_fingerprint"], meta["continued_fingerprint"]], strict=True):
                    rows = groups[(outer, inner, seed, method)]
                    require(len(rows) == len({row["record_id"] for row in rows}) == 342 and set(row["record_id"] for row in rows) == set(valid_ids), "Exactly one prediction per inner validation specimen")
                    model = DenseClassifier(612, [64, 32], 7)
                    model.load_state_dict(torch.load(directory / (method + ".pt"), map_location="cpu", weights_only=True))
                    model.eval()
                    require(state_fingerprint(model) == fingerprint, "Exact loaded paired model state")
                    ordered = sorted(rows, key=lambda row: row["record_id"])
                    features = transform_features(prep, X[[index[row["record_id"]] for row in ordered]])
                    scores = model_scores(model, features)
                    expected_scores = np.asarray([[float(row["score_" + c]) for c in CLASSES] for row in ordered])
                    difference = float(np.max(np.abs(scores - expected_scores)))
                    maximum_score_difference = max(maximum_score_difference, difference)
                    require(difference < 1e-4, "Loaded model float32 validation score replay")
                    prediction = np.asarray(CLASSES)[scores.argmax(1)]
                    require(np.array_equal(prediction, [row["y_predicted"] for row in ordered]), "Exact paired validation label replay")
                    matrix = np.zeros((7, 7), dtype=int)
                    for row in rows: matrix[VOCAB[row["y_true"]], VOCAB[row["y_predicted"]]] += 1
                    expected = expected_metrics[(outer, inner, seed, method)]["metrics"]
                    require(matrix.tolist() == expected["confusion_matrix"], "Raw strongMLP validation confusion matrix")
                    for name in METRICS: close(metrics(matrix)[name], expected[name], "Raw strongMLP validation metric")
                source_contexts.append({"outer_fold": outer, "inner_fold": inner, "seed": seed,
                    "continuation_seed": seed + 1600, "source_samples": 342,"validation_samples":342,
                    "parameter_count":41543,"base_epochs":200,"continuation_updates":100,
                    "initial_seed_verified":True,"base_unchanged":True,"two_model_labels_replayed":True})
    require(len(source_contexts) == 30 and source_task_total == 3000, "Complete30 source contexts and3000 source continuation tasks")
    require(not (RUN / "outer_predictions.csv.gz").exists(), "Outer evaluation still absent at review completion")
    result = {"status":"pass_for_fixed_historical_architecture_retrospective_control","allow_outer_evaluation":True,
        "started_at_utc":started,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,
        "new_estimator_fits":0,"new_calibrator_fits":0,"outer_query_scored":False,
        "selection_lock_sha256":sha(RUN / "selection_lock.json"),"manifest_sha256":sha(RUN / "run_manifest.json"),
        "protocol_sha256":sha(PROTOCOL),"source_contexts":source_contexts,"inner_method_cases":60,
        "inner_predictions":len(raw),"source_continuation_tasks":source_task_total,"maximum_float32_score_replay_difference":maximum_score_difference,
        "limitations":["b006 was historically selected on the old513 source subset, which overlaps the new855-fold queries; fixed-architecture reuse is a retrospective development control, not fully nested unbiased architecture selection.",
                       "New actual source weights, scalers and continuation tasks use only the current342/684source partition.",
                       "No new architecture, checkpoint, method or best seed is selected from these inner or outer results. Two methods remain separate prespecified roles; seed42 is primary and all43/44 results must remain visible.",
                       "200base epochs plus100source updates is not300epochs, not target calibration, and not a MAML-specific benefit.",
                       "The old171 accuracy and new855OOF remain separate evaluation populations; equal architecture is not equality of earlier accuracy."]}
    (OUT / "strong_mlp_review.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    text = f"""# 旧强MLP同折重训：第三协议外测前审查

UTC：{result['completed_at_utc']}。{checks}项检查通过；新增分类器／校准器拟合0，外层评分0。允许按冻结hash进行预设外层评价。

核验30个内层源上下文、60个配对方法评价、20520条验证预测及3000个源内continuation任务。三个种子42／43／44完整保留，原41543参数的612→64→32→7结构、SNV与源缩放、200个base epoch及100次追加源监督更新均与旧实现一致。30份源预处理器、60份权重均重载并在当前内层验证谱上逐标签重放通过；float32最大分数差{maximum_score_difference:.3g}。

**历史架构来源须单列**：b006来自旧513源组的历史选择，该513包含新855外折中的部分查询样品。本轮是固定历史架构的回顾性同折重训，不能宣称架构选择也完成了全新的无偏嵌套验证。当前权重、缩放与源任务训练仍严格只使用各自当前源342／684份，未发现新的直接外测拟合。

主表固定seed42；43／44及均值／样本标准差只是种子敏感性，不选最好种子。200epoch＋100updates不能写成300epoch，也不是使用目标支持样品校准；不得根据最终外层结果将两个预设角色中的较好者改称内层选择。

[完整审查及冻结hash](strong_mlp_review.json)。原两协议review.json保持不变。
"""
    (OUT / "STRONG_MLP_REVIEW.md").write_text(text)
    print(json.dumps({"status":result["status"],"checks":checks,"inner_cases":60,"inner_rows":len(raw),
                      "source_contexts":len(source_contexts),"maximum_float32_score_difference":maximum_score_difference},ensure_ascii=False))


if __name__ == "__main__":
    main()
