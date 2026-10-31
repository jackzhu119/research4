"""Index and replay the five archived R2 fold models without fitting."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_paper_c_candidate import predict_records, sha256, spectrum_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"
DEFAULT_OUTPUT = ROOT / "reports/paper-c/2026-10-31/best-candidate"
EVIDENCE = ROOT / "reports/staged-nested-optimization/2026-10-31/core"
MODEL_ROOT = ROOT / "models/staged_nested_20261031/core"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Candidate package output must be empty; preserve earlier packages")
    original_manifest = json.loads((MODEL_ROOT / "manifest.json").read_text())
    original_run = json.loads((EVIDENCE / "run_manifest.json").read_text())
    split = json.loads((EVIDENCE / "splits.json").read_text())
    lock = json.loads((EVIDENCE / "selection_lock.json").read_text())
    assert sha256(args.input) == original_run["input_sha256"]
    for relative, expected in original_run["source_sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    with np.load(args.input, allow_pickle=False) as data:
        x, axis, ids, y = data["X_alpha"], data["axis"], data["record_ids"], data["y"]
        version = str(data["dataset_version"])
    assert x.shape == (855, 612)
    classes = sorted(set(y.tolist()))
    index = {"schema": "research4_r2_fold_candidate_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "candidate": "R2: SNV + MC-UVE + RBF-SVM; fold-specific inner selection",
             "dataset_version": version, "input_sha256": sha256(args.input),
             "model_seed": 42, "split_seed": 20261031, "outer_folds": 5, "inner_folds": 2,
             "frequency_axis": axis.tolist(), "frequency_axis_unit": "THz",
             "physical_quantity": "absorption coefficient in original numerical export; absolute unit not independently verified",
             "classes": classes, "source_code_sha256": original_run["source_sha256"],
             "historical_record_value_sha256": {str(rid): spectrum_sha256(values) for rid, values in zip(ids, x, strict=True)},
             "original_environment": {"python": original_run["python"], **original_run["versions"]},
             "packaging_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                       "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
             "model_kind": "five research fold models, each trained on 684 source specimens; no full-855 deployment model",
             "future_ensemble_external_accuracy": None, "folds": []}
    for entry in original_manifest["models"]:
        if "R2" not in entry["method_roles"]:
            continue
        path = MODEL_ROOT / entry["path"]
        assert sha256(path) == entry["sha256"]
        meta = json.loads(path.with_suffix(".json").read_text())
        fold_id = meta["outer_fold"]
        model = joblib.load(path)
        selected = lock["outer_folds"][fold_id]["choices"]["R2"]["selected"]
        assert model.config == selected
        assert model.config["preprocessing"] == "snv"
        assert meta["source_ids"] == split[fold_id]["train_sample_ids"]
        assert meta["outer_query_ids"] == split[fold_id]["test_sample_ids"]
        assert set(meta["source_ids"]).isdisjoint(meta["outer_query_ids"])
        assert len(meta["source_ids"]) == 684 and len(meta["outer_query_ids"]) == 171
        selection = model.bundle.selector.selected_indices_
        assert np.array_equal(selection, meta["model_metadata"]["selected_indices"])
        assert np.array_equal(axis[selection], meta["model_metadata"]["selected_frequencies"])
        index["folds"].append({"outer_fold": fold_id, "model_path": str(path.relative_to(ROOT)),
                               "model_sha256": entry["sha256"], "metadata_path": str(path.with_suffix(".json").relative_to(ROOT)),
                               "metadata_sha256": sha256(path.with_suffix(".json")), "source_record_ids": meta["source_ids"],
                               "heldout_record_ids": meta["outer_query_ids"], "config": selected,
                               "selected_feature_indices_zero_based": selection.tolist(),
                               "selected_frequencies": axis[selection].tolist(),
                               "fitted_preprocessor": "row-wise SNV; stored in original joblib pipeline",
                               "fitted_scaler": {"type": "StandardScaler before MC-UVE", "mean": model.bundle.scaler.mean_.tolist(),
                                                 "variance": model.bundle.scaler.var_.tolist(), "scale": model.bundle.scaler.scale_.tolist(),
                                                 "fitted_samples": int(model.bundle.scaler.n_samples_seen_)},
                               "effective_gamma": float(model.estimator._gamma),
                               "training_class_weights": model.metadata["training_class_weights"],
                               "prediction_rule": "SVC standard vote, break_ties=False",
                               "scores_are_probabilities": False})
    index["folds"].sort(key=lambda fold: fold["outer_fold"])
    assert len(index["folds"]) == 5
    replay = predict_records(index, x, axis, ids, mode="heldout-replay")
    with gzip.open(EVIDENCE / "outer_predictions.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        old = {row["record_id"]: row for row in csv.DictReader(handle) if row["method"] == "R2"}
    for row in replay:
        assert row["predicted_origin_group"] == old[row["record_id"]]["y_predicted"]
        assert row["outer_fold"] == int(old[row["record_id"]]["outer_fold"])
    prediction = [row["predicted_origin_group"] for row in replay]
    correct = sum(y == prediction)
    assert correct == 821
    metrics = {"correct": int(correct), "total": len(y), "accuracy": accuracy_score(y, prediction),
               "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
               "balanced_accuracy": balanced_accuracy_score(y, prediction),
               "class_order": classes, "confusion_matrix": confusion_matrix(y, prediction, labels=classes).tolist(),
               "classification_report": classification_report(y, prediction, labels=classes, output_dict=True, zero_division=0)}
    for name, value in [("candidate_index.json", index), ("replay_metrics.json", metrics)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "heldout_replay.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(replay)
    readme = """# R2最佳历史候选：可复现索引包

本包引用现有五个R2完整管线，不重复复制权重。每个模型只训练684份源样品；855份OOF由各自171份留出拼接。本次**未训练新模型**，855个标签逐项重放与冻结记录相同：821份正确，Accuracy 96.0234%，Macro-F1 0.9197，Balanced Accuracy 89.9184%。这不是新增独立确认成绩。

`candidate_index.json`保存每折源／留出编号、完整612点频轴、MC次数／潜变量／保留比例、实际频点索引与频率、C／gamma／类别权重、源StandardScaler统计、原环境和SHA。原权重路径从仓库根解析，原模型与历史结果不改动。`replay_metrics.json`含逐类指标和混淆矩阵；`heldout_replay.csv`是实际重放结果。

## 重现与预测命令

在仓库根、已安装本项目的Python环境运行：

```bash
.runtime/bin/python scripts/package_paper_c_best_candidate.py --output outputs/paper-c-r2-package-new
.runtime/bin/python scripts/predict_paper_c_candidate.py --input outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz --mode heldout-replay --output outputs/paper-c-r2-heldout-new.csv
.runtime/bin/python scripts/predict_paper_c_candidate.py --input new_spectra.npz --mode future-ensemble --confirm-compatible-measurement --output outputs/paper-c-new-spectra.csv
```

当前打包命令读取云端原始冻结NPZ并检查其文件SHA；原缓存不在时，按[数据准备和频率裁剪](DATA_PREPARATION_LINK)重建，再用heldout预测入口验证相同数值。重新运行内层选择可用原入口（完整重跑需要原分区记录及匹配输入，评价阶段还须新的选择锁和放行记录，不应复用旧放行）：

```bash
.runtime/bin/python scripts/run_staged_nested_search.py --input outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz --protocol data/research/staged_nested_protocol_20261031.json --output outputs/paper-c-nested-reproduction-new --stage selection
```

实际冻结证据见[原核心证据](../../../../staged-nested-optimization/2026-10-31/core/README.md)，实验定义见[四轮协议](../../../../../docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md)。本次未重新拟合，上述完整选择重跑命令未在本打包步骤再次执行。

输入NPZ必须含`X_alpha`（N×612）、`axis`（原顺序的612点）和唯一`record_ids`。原始数值应为同一定义、单位和测量条件下的吸收系数；训练导出文件的绝对吸收单位尚未独立核实，不能自动接收其他仪器的不同物理量。未来模式要求显式确认兼容性，拒绝旧855编号；不得将旧样品改名后当作新样品。

默认heldout模式只调用该编号原留出折，避免对见过样品的模型打分。future模式对真正未知、同轴兼容样品输出五折类别、投票和分歧；平票按固定类别顺序处理并标记。五折集成目前**没有独立外部准确率**，投票比例不是校准概率。所有预测仅在ZY01—ZY07匿名已知类别间作强制分类；真实产地名称、未知产地检测和跨批次适用性仍未验证。96.02%不可标到该未来集成上。

频轴不符、非有限输入、历史ID在future集成中出现、模型／冻结源码SHA不符均拒绝处理。原855份谱的逐行数值hash也保存在索引中，heldout模式核对编号与原谱数值对应，future模式拒绝将已知原谱改名当作新样品。预测和打包输出不覆盖已有文件／非空目录。
"""
    readme = readme.replace("../../../../staged-nested-optimization/2026-10-31/core/README.md",
                            os.path.relpath(EVIDENCE / "README.md", args.output))
    readme = readme.replace("../../../../../docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md",
                            os.path.relpath(ROOT / "docs/STAGED_NESTED_OPTIMIZATION_PROTOCOL.md", args.output))
    readme = readme.replace("DATA_PREPARATION_LINK",
                            os.path.relpath(ROOT / "docs/ZAOREN_APPROVED_EXECUTION_REPORT.md", args.output))
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    validation = {"status": "pass", "replayed_records": 855, "exact_historical_label_matches": 855,
                  "source_heldout_disjoint_folds": 5, "model_files_referenced": 5, "new_model_fits": 0,
                  "delivery_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                  (Path(__file__), ROOT / "scripts/predict_paper_c_candidate.py", ROOT / "tests/test_paper_c_candidate.py")},
                  "metrics": metrics, "original_evidence_sha256": {str((EVIDENCE / name).relative_to(ROOT)): sha256(EVIDENCE / name)
                  for name in ("selection_lock.json", "splits.json", "run_manifest.json", "outer_predictions.csv.gz")},
                  "package_files_sha256": {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}}
    (args.output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "replayed_records": 855, "correct": int(correct), "accuracy": metrics["accuracy"],
                      "new_model_fits": 0, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
