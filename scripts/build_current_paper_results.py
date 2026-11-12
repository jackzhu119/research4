#!/usr/bin/env python3
"""Build or verify a publication view containing corrected-kind results only.

Sources are immutable, allowlisted Git objects already present locally. This
script performs no network access, training, model selection or scientific fit.
Large predictions/models remain at their pinned source commits.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/current"
BASE_URL = "https://github.com/jackzhu119/research4/blob/"
CORRECTION = "0156e2189d13456d1faf05ebfd53d86eeeb4f5e7"
PAPER = "b051733989161517a68c4a60cce75766880c95f5"
TAIL = "df3e12656e2177cfcf0a298e66bdedc5c524cb74"
VERIFIED = "516de760b3bf72143269f76c904cc35df7b69f76"
FIXED = "data/research/species_revision_20261108"
BUNDLE = "reports/paper-evidence-20261109"
V4 = "data/research/evidence_validation_20261108_v4"
B_ROUNDS = [
    ("B1", "76966534df314e5bcb108a1fdccd611d1ad52910", "b_mechanism_20261110", 105),
    ("B2", "ad21deecbb53371f9b4547646a2aa985d8f617fb", "b_mechanism_followup_20261110", 345),
    ("B3", "69354eb7e2ec726ac5f5568fdac1035afff36da6", "b_mechanism_round3_20261110", 375),
    ("B4", "bbab5a9de3f3e79fe8b12dd5e06c29547e3b9778", "b_mechanism_round4_20261111", 375),
    ("B5", "647f069c1ed4d56b098689571f43074b581a0cc0", "b_mechanism_round5_20261111", 180),
]
TASKS = {
    "binary_kinds": "枣仁与酸枣仁二分类",
    "acid_origins": "纯酸枣仁：ZY03—ZY07五来源",
    "jujube_origins": "纯枣仁：ZY01、ZY02、ZR01—ZR03五来源",
    "four_kinds": "枣仁、酸枣仁、黄芪、天麻四种类",
    "huangqi_origins": "黄芪四来源",
    "tianma_origins": "天麻四来源",
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_bytes(sha, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", sha + ":" + path])


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def csv_bytes(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def current_path(relative):
    path = OUT / relative
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("Unsafe output path")
    return path


def allowed_fixed_sources():
    """Keep manifest edits from admitting withdrawn sources into this view."""
    sources = {(CORRECTION, path) for path in (
        "data/active_workbooks.json", "data/metadata_template.csv",
        "data/annotations/origin_kind_confirmation_20261108.json",
        "data/annotations/specimen_independence_user_declaration_20261108.json",
        FIXED + "/split_manifest.csv", FIXED + "/protocol.json",
        FIXED + "/feature_audit.json", FIXED + "/classification/frozen_config.json",
        FIXED + "/classification/summary.json", FIXED + "/classification/metrics_overview.csv",
        FIXED + "/transfer/protocol.json", FIXED + "/transfer/result.json",
        FIXED + "/transfer/summary.csv", FIXED + "/transfer/paired_source_transfer.csv",
        FIXED + "/transfer/metrics_by_setting.json", FIXED + "/independent_results_validation.json",
    )}
    for name in ("summary.json", "metrics.csv", "complete.json"):
        sources.add((TAIL, "reports/00480/" + name))
        sources.add((PAPER, BUNDLE + "/sources/current_v4/complete.json" if name == "complete.json"
                     else V4 + "/snapshots/00330/" + name))
    for name in ("all_final_model_settings.csv", "current_classical_relative_to_raw.csv",
                 "current_confusion_counts_by_repeat.csv", "current_per_class_by_repeat.csv",
                 "feature_and_parameter_selection.csv", "paired_classical_effects.csv",
                 "scientific_job_register.csv", "model_protocol_register.csv"):
        sources.add((PAPER, BUNDLE + "/tables/" + name))
    for stem in ("01_classical_primary_comparison", "02_feature_budget_ablation",
                 "03_primary_confusion_by_repeat", "04_transfer_representation_budget"):
        for extension in ("pdf", "png", "svg"):
            sources.add((PAPER, BUNDLE + "/figures/" + stem + "." + extension))
    for _, sha, directory, _ in B_ROUNDS:
        for name in ("results_summary.json", "artifact_verification.json", "status.json",
                     "frozen_protocol.json", "metrics.csv", "mechanism_analysis.csv",
                     "uncertainty.csv", "paired_transitions.csv"):
            sources.add((sha, "reports/" + directory + "/" + name))
    for name in ("summary.json", "species_all_predictions.json"):
        sources.add((VERIFIED, "reports/all_round_validation_20261112/ml/" + name))
    return sources


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []

    def add(destination, sha, path, transformation="byte_copy"):
        raw = source_bytes(sha, path)
        result = raw
        if transformation == "gzip_mtime_zero":
            result = gzip.compress(raw, mtime=0)
        elif transformation == "protocol_without_design":
            protocol = json.loads(raw)
            design = protocol.pop("design")
            protocol["design_not_copied"] = {
                "task_count": len(design),
                "complete_source_sha256": digest(raw),
                "source_url": BASE_URL + sha + "/" + path,
                "scope": "Compact display only; this derivative is not the original frozen protocol.",
            }
            result = encoded(protocol)
        elif transformation == "corrected_protocol_rows":
            rows = [row for row in csv_rows(raw)
                    if row["experiment_id"].startswith(("V4-", "OLD-", "FIXED-"))]
            if len(rows) != 57:
                raise ValueError("Corrected protocol table changed")
            for row in rows:
                original = row["experiment_id"]
                row["current_scope"] = ("corrected_1305" if original.startswith("OLD-") else
                                        "corrected_v4_330" if original.startswith("V4-") else
                                        "corrected_fixed_holdout")
                row["display_id"] = original.replace("OLD-", "C1305-", 1)
            result = csv_bytes(rows)
        elif transformation == "task_scope_rows":
            rows = []
            for task in json.loads(raw)["tasks"]:
                roles = task["roles"]
                rows.append({"task": task["task"], "scope": TASKS[task["task"]],
                             "classes": " / ".join(task["classes"]),
                             "train": roles["train"]["count"],
                             "calibration": roles["calibration"]["count"],
                             "prediction": roles["prediction"]["count"],
                             "tiny_class_warning": task.get("tiny_class_warning") or "",
                             "validation_scope": task["scope"]})
            result = csv_bytes(rows)
        target = current_path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result)
        records.append({
            "path": destination, "bytes": len(result), "sha256": digest(result),
            "source_commit": sha, "source_path": path,
            "source_sha256": digest(raw), "source_bytes": len(raw),
            "source_url": BASE_URL + sha + "/" + path,
            "transformation": transformation,
        })

    for path, destination in [
        ("data/active_workbooks.json", "inputs/active_workbooks.json"),
        ("data/metadata_template.csv", "inputs/metadata.csv"),
        ("data/annotations/origin_kind_confirmation_20261108.json", "inputs/origin_kind_confirmation.json"),
        ("data/annotations/specimen_independence_user_declaration_20261108.json", "inputs/specimen_independence.json"),
        (FIXED + "/split_manifest.csv", "inputs/split_manifest.csv"),
        (FIXED + "/protocol.json", "inputs/split_protocol.json"),
        (FIXED + "/feature_audit.json", "inputs/feature_audit.json"),
        (FIXED + "/classification/frozen_config.json", "protocols/fixed_classification.json"),
        (FIXED + "/classification/summary.json", "results/fixed_holdout/classification_summary.json"),
        (FIXED + "/classification/metrics_overview.csv", "tables/fixed_classification.csv"),
        (FIXED + "/transfer/protocol.json", "protocols/fixed_transfer.json"),
        (FIXED + "/transfer/result.json", "results/fixed_holdout/transfer_result.json"),
        (FIXED + "/transfer/summary.csv", "tables/fixed_transfer.csv"),
        (FIXED + "/transfer/paired_source_transfer.csv", "tables/fixed_transfer_paired.csv"),
        (FIXED + "/transfer/metrics_by_setting.json", "results/fixed_holdout/transfer_settings.json"),
        (FIXED + "/independent_results_validation.json", "acceptance/fixed_results_source_receipt.json"),
    ]:
        add(destination, CORRECTION, path)
    add("tables/task_scope.csv", CORRECTION, FIXED + "/classification/summary.json", "task_scope_rows")

    for name in ("summary.json", "metrics.csv", "complete.json"):
        add("results/corrected_1305/" + name, TAIL, "reports/00480/" + name)
        v4_path = (BUNDLE + "/sources/current_v4/complete.json" if name == "complete.json"
                   else V4 + "/snapshots/00330/" + name)
        add("results/v4_330/" + name, PAPER, v4_path)
    for name in ("all_final_model_settings.csv", "current_classical_relative_to_raw.csv",
                 "current_confusion_counts_by_repeat.csv", "current_per_class_by_repeat.csv",
                 "feature_and_parameter_selection.csv", "paired_classical_effects.csv",
                 "scientific_job_register.csv"):
        add("tables/" + name, PAPER, BUNDLE + "/tables/" + name)
    add("tables/corrected_protocol_register.csv", PAPER,
        BUNDLE + "/tables/model_protocol_register.csv", "corrected_protocol_rows")
    for stem in ("01_classical_primary_comparison", "02_feature_budget_ablation",
                 "03_primary_confusion_by_repeat", "04_transfer_representation_budget"):
        for extension in ("pdf", "png", "svg"):
            name = stem + "." + extension
            add("figures/" + name, PAPER, BUNDLE + "/figures/" + name)

    for id, sha, directory, _ in B_ROUNDS:
        source = "reports/" + directory
        for name in ("results_summary.json", "artifact_verification.json", "status.json"):
            add("results/" + id + "/" + name, sha, source + "/" + name)
        add("protocols/" + id + ".json", sha, source + "/frozen_protocol.json", "protocol_without_design")
        for name in ("metrics.csv", "mechanism_analysis.csv", "uncertainty.csv", "paired_transitions.csv"):
            add("tables/" + id + "/" + name + ".gz", sha, source + "/" + name, "gzip_mtime_zero")
    for name in ("summary.json", "species_all_predictions.json"):
        add("acceptance/all_round_" + name, VERIFIED,
            "reports/all_round_validation_20261112/ml/" + name)

    overview = {
        "schema": "research4-corrected-paper-view-v1",
        "dataset_version": "2026-11-08-confirmed-origin-kind-v1",
        "dataset_records": 1672, "workbooks": 18,
        "species_counts": {"枣仁": 364, "酸枣仁": 500, "黄芪": 403, "天麻": 405},
        "task_definitions": TASKS,
        "corrected_matrix_jobs": {"original_corrected": 1305, "v4_additional": 330},
        "b_round_jobs": {id: count for id, _, _, count in B_ROUNDS},
        "b_total_unique_task_bundles": 1380,
        "old_mixed_kind_metrics_included": False,
        "new_batch_evaluation": "NOT_RUN_NEEDS_NEW_BATCH_DATA",
        "continuous_meta_learning_final_results": "NOT_INCLUDED_NOT_FINISHED",
        "quantum_results": "NOT_INCLUDED_IN_THIS_CLASSIFICATION_VIEW",
        "new_model_fits_by_builder": 0,
        "scope": "All corrected positive and negative result settings are retained; internal exploration, not independent new-batch confirmation.",
        "large_outputs": "Weights, raw predictions and compressed restoration archives remain at their immutable corrected source commits; this view contains paper tables, summaries, protocols and validation provenance.",
        "aliases": {"legacy_TRAIN_* / OLD-* / P-CV-OLD": "Corrected-kind original 1305 matrix from 2026-11-08, not pre-correction mixed-kind results."},
    }
    generated = {
        "overview.json": encoded(overview),
        "README.md": README.encode(),
    }
    for name, raw in generated.items():
        current_path(name).write_bytes(raw)
        records.append({"path": name, "bytes": len(raw), "sha256": digest(raw),
                        "transformation": "generated_current_scope_guidance"})
    manifest = {
        "schema": "research4-corrected-paper-manifest-v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Explicit allowlist of corrected-kind results only. No recursive source discovery or old mixed-kind metrics.",
        "files": sorted(records, key=lambda row: row["path"]),
        "excluded_from_paper": ["pre-correction seven-group mixed-kind results", "historical_relabel_audit.json",
                                "historical_quantum_experiment_register.csv", "sources/history_quantum",
                                "parallel_b_*.csv", "old bundle PASS/manifest certificates", "unfinished continuous meta-learning"],
        "verification_receipts": "Copied receipts describe original immutable corrected sources; they are not a certificate for this newly built view.",
    }
    current_path("manifest.json").write_bytes(encoded(manifest))
    return verify()


def verify():
    manifest = json.loads(current_path("manifest.json").read_text())
    allowed = allowed_fixed_sources()
    paths = set()
    verified_sources = 0
    for row in manifest["files"]:
        if row["path"] in paths:
            raise ValueError("Duplicate manifest path")
        paths.add(row["path"])
        raw = current_path(row["path"]).read_bytes()
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError("Current file checksum differs: " + row["path"])
        if "source_commit" in row:
            if (row["source_commit"], row["source_path"]) not in allowed:
                raise ValueError("Source outside corrected-results allowlist: " + row["path"])
            original = source_bytes(row["source_commit"], row["source_path"])
            if digest(original) != row["source_sha256"] or len(original) != row["source_bytes"]:
                raise ValueError("Fixed source object checksum differs: " + row["path"])
            if row["transformation"] == "byte_copy" and raw != original:
                raise ValueError("Byte-preserved source differs: " + row["path"])
            verified_sources += 1
        elif row["path"] not in {"README.md", "overview.json"}:
            raise ValueError("Unapproved generated paper file: " + row["path"])
        if row["transformation"] == "gzip_mtime_zero" and digest(gzip.decompress(raw)) != row["source_sha256"]:
            raise ValueError("Lossless table transformation differs")
    observed = {str(path.relative_to(OUT)) for path in OUT.rglob("*") if path.is_file()}
    if observed != paths | {"manifest.json", "verification.json"} and observed != paths | {"manifest.json"}:
        raise ValueError("Unexpected or missing files in current paper view")
    registry = json.loads(current_path("inputs/active_workbooks.json").read_text())
    protocol = json.loads(current_path("inputs/split_protocol.json").read_text())
    if digest(current_path("inputs/active_workbooks.json").read_bytes()) != protocol["active_registry_sha256"]:
        raise ValueError("Registry not equal to corrected frozen input")
    if digest(current_path("inputs/metadata.csv").read_bytes()) != protocol["metadata_sha256"]:
        raise ValueError("Metadata not equal to corrected frozen input")
    if digest(current_path("inputs/split_manifest.csv").read_bytes()) != protocol["manifest_sha256"]:
        raise ValueError("Role manifest not equal to corrected frozen input")
    frame = csv_rows(current_path("inputs/split_manifest.csv").read_bytes())
    counts = Counter(row["species"] for row in frame)
    expected = {"枣仁": 364, "酸枣仁": 500, "黄芪": 403, "天麻": 405}
    if counts != expected or len({row["record_id"] for row in frame}) != 1672:
        raise ValueError("Corrected specimen counts differ")
    if Counter(row["role"] for row in frame) != {"train": 999, "calibration": 334, "prediction": 339}:
        raise ValueError("Frozen roles differ")
    if len(registry["workbooks"]) != 18:
        raise ValueError("Require eighteen active workbooks")
    for workbook in registry["workbooks"]:
        if digest((ROOT / workbook["path"]).read_bytes()) != workbook["sha256"]:
            raise ValueError("An active original spectrum changed: " + workbook["path"])
    for group in ("ZY01", "ZY02"):
        if {r["species"] for r in frame if r["origin_group_id"] == group} != {"枣仁"}:
            raise ValueError("Uncorrected ZY01/ZY02 kind")
    for group in ("ZY03", "ZY04", "ZY05", "ZY06", "ZY07"):
        if {r["species"] for r in frame if r["origin_group_id"] == group} != {"酸枣仁"}:
            raise ValueError("Uncorrected acid-jujube group")
    classified = json.loads(current_path("results/fixed_holdout/classification_summary.json").read_text())
    tasks = {row["task"]: row for row in classified["tasks"]}
    if set(tasks) != set(TASKS) or set(tasks["acid_origins"]["classes"]) != {"ZY03", "ZY04", "ZY05", "ZY06", "ZY07"}:
        raise ValueError("Corrected task vocabulary differs")
    if set(tasks["binary_kinds"]["classes"]) != {"枣仁", "酸枣仁"}:
        raise ValueError("Binary kind labels differ")
    if set(tasks["jujube_origins"]["classes"]) != {"ZY01", "ZY02", "ZR01", "ZR02", "ZR03"}:
        raise ValueError("Pure jujube origin labels differ")
    if set(tasks["four_kinds"]["classes"]) != set(expected):
        raise ValueError("Four-kind label vocabulary differs")
    jobs = csv_rows(current_path("tables/scientific_job_register.csv").read_bytes())
    if Counter(row["queue"] for row in jobs) != {"species_relaunch_v2": 1305, "evidence_validation_v4": 330}:
        raise ValueError("Corrected 1305/330 count differs")
    if not all(row["status"] == "completed" and row["receipt_hash_verified"] == "True" for row in jobs):
        raise ValueError("An unfinished job entered the paper table")
    for id, _, _, expected_jobs in B_ROUNDS:
        result = json.loads(current_path("results/" + id + "/results_summary.json").read_text())
        if (result["status"], result["completed_jobs"], result["expected_jobs"]) != ("DONE", expected_jobs, expected_jobs):
            raise ValueError("Incomplete B mechanism round")
        if result["e05_status"] != "NOT_RUN_NEEDS_NEW_BATCH_DATA":
            raise ValueError("New-batch scope unexpectedly changed")
    result = {"schema": "research4-current-paper-verification-v1", "status": "PASS",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "manifest_sha256": digest(current_path("manifest.json").read_bytes()),
              "checked_files": len(paths), "corrected_records": 1672,
              "verified_fixed_source_objects": verified_sources, "unchanged_active_workbooks": 18,
              "corrected_jobs": 1635, "b_unique_task_bundles": 1380,
              "no_new_fit": True,
              "scope": "Current-view bytes, immutable Git source objects, active workbook hashes and corrected labels/counts; not a rerun of all original models or external validation."}
    current_path("verification.json").write_bytes(encoded(result))
    print(json.dumps(result, ensure_ascii=False))
    return 0


README = """# 当前论文结果入口：种类与产地修正后

本目录只收录2026-11-08确认种类和产地之后的分类、迁移及机制结果。
原855条旧混种任务的成绩、旧结果重贴标签得到的分析、旧论文汇编的历史章节均不在这里。

|入口|内容与适用范围|
|---|---|
|[任务范围](tables/task_scope.csv)／[六项固定留出](tables/fixed_classification.csv)|枣仁/酸枣仁二分类、四种类分类，以及酸枣仁、枣仁、黄芪、天麻各自的来源分类；训练/校准/预测数量分别登记|
|[固定留出迁移](tables/fixed_transfer_paired.csv)|三种子目标单训、额外目标预训、黄芪/天麻跨种类预训；负迁移保留|
|[主模型表](tables/all_final_model_settings.csv)|修正后原1305矩阵与新增V4 330项的已保存摘要；不同划分/预算分开解释|
|[协议目录](tables/corrected_protocol_register.csv)|仅修正后主线57项；display_id中的C1305对应原来源的OLD命名|
|[科学任务登记](tables/scientific_job_register.csv)|1305＋330已完成任务，备份与续算不重复计数|
|[B1](results/B1/results_summary.json)／[B2](results/B2/results_summary.json)／[B3](results/B3/results_summary.json)／[B4](results/B4/results_summary.json)／[B5](results/B5/results_summary.json)|五轮1380任务包的全部正负结果汇总；各轮指标/机制/不确定性明细以tables/B*/下CSV.gz保存|
|[四张主图](figures/)|仅从修正后的V4结果生成，提供PNG、SVG及PDF|
|[输入与协议](inputs/)|18表有效登记、1672行修正元数据、固定999/334/339角色；原始光谱仍使用仓库有效清单位置|
|[来源清单](manifest.json)／[当前包校验](verification.json)|每个文件的固定Git来源、原件SHA-256、变换方式与当前文件校验值|

## 任务边界

- 纯酸枣仁来源只包含ZY03—ZY07；纯枣仁来源包含ZY01、ZY02、ZR01—ZR03，种类标签分开。
- 枣仁新增三个来源各只有3份实物，每个角色1份；相关比率是小样本描述，不应与酸枣仁五来源300TRAIN三次五折成绩直接排名。
- `legacy_TRAIN_*`、`OLD-*`及`P-CV-OLD`出现在来源表中时，指修正种类后的原1305矩阵，不是修正前旧855混种结果；算法名`legacy_lambda1`也不是数据版本。
- 原1305最终轻量摘要仅汇总源模型预测，不覆盖全部校准明细；完整预测和校准输出保留于固定的纠正来源Git归档。本目录不复制大型预测包或模型权重，保留其已完成复核的来源链。
- 本目录完整保留所收录协议的负结果，不按准确率挑选设置。B1—5仍属已曝光样品上的内部探索，E05真实新批次评价为`NOT_RUN_NEEDS_NEW_BATCH_DATA`。
- 元学习尚未完成的最终结果和量子计算不进入这个分类结果入口；量子资格应由对应分子、数值模型和验收证据单独判断。
- `acceptance/`里的回执描述原固定提交的核验范围；新生成的`verification.json`只认证本目录文件、标签、计数与来源，不能当作全部模型重训或外部新批次认证。

维护命令：`python scripts/build_current_paper_results.py --build` 按固定白名单生成；`python scripts/build_current_paper_results.py --verify` 校验本地文件、固定Git原件与18份有效原谱。两者均不联网、不训练模型。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        return build() if args.build else verify()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print("Current paper view failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
