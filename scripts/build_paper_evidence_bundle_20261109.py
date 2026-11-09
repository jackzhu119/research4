#!/usr/bin/env python3
"""Assemble reviewed experiment-family inventories without training any model.

Historical evidence is read from fixed Git objects. Small referenced sources are
copied verbatim for offline inspection; original paths and immutable URLs remain
the primary provenance. No live producer, dataset, model, or ledger is modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/paper_bundle_20261109"
DOCS = ROOT / "docs/paper_bundle_20261109"
MODULES = [
    ("current_species", "03_CURRENT_DATA_AND_SPECIES.md"),
    ("historical_ml", "02_HISTORICAL_ML.md"),
    ("external_robustness", "04_EXTERNAL_TRANSFER_AND_ROBUSTNESS.md"),
    ("worker_b_ml", "06_WORKER_B_DIAGNOSTICS.md"),
    ("quantum", "05_QUANTUM_AND_MECHANISM.md"),
]
MAX_COPY_BYTES = 5 * 1024**2
TEXT_SUFFIXES = {".md", ".json", ".csv", ".txt", ".tsv"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_key(source):
    return (source.get("repository", "jackzhu119/research4"),
            source.get("commit") or "LOCAL_SNAPSHOT", source["path"], source["sha256"])


def scalar_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def csv_write(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_text(row.get(key, "")) for key in columns})


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, prefix + ("." if prefix else "") + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from flatten(child, prefix + f"[{i}]")
    else:
        yield prefix, value


def read_source(source):
    path = source["path"]
    commit = source.get("commit")
    if commit:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("A historical source needs a complete commit SHA: " + str(source))
        try:
            size = int(git("cat-file", "-s", f"{commit}:{path}"))
            raw = git("show", f"{commit}:{path}")
            if len(raw) != size:
                raise ValueError("Git object length changed: " + path)
        except subprocess.CalledProcessError:
            # A single-branch clone may lack other research histories. The
            # previously verified verbatim copy provides an offline handoff.
            archived = EVIDENCE / "reviewed_sources" / (source["sha256"] + Path(path).suffix.lower())
            if not archived.is_file():
                raise
            raw = archived.read_bytes()
    else:
        target = (ROOT / path).resolve()
        if not target.is_relative_to(EVIDENCE.resolve()) or "live_snapshots" not in target.parts:
            raise ValueError("Local evidence must be an immutable live_snapshots copy: " + path)
        raw = target.read_bytes()
    if digest(raw) != source["sha256"]:
        raise ValueError("Source SHA-256 does not match reviewed bytes: " + str(source))
    return raw


def build():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    experiments, modules, sources = [], [], {}
    for module, document in MODULES:
        path = EVIDENCE / (module + ".json")
        payload = json.loads(path.read_text())
        rows = payload["experiments"]
        modules.append({"module": module, "inventory_path": str(path.relative_to(ROOT)),
                        "inventory_sha256": digest(path.read_bytes()),
                        "document_path": str((DOCS / document).relative_to(ROOT)),
                        "experiment_families": len(rows), "cutoff_utc": payload.get("cutoff_utc")})
        for row in rows:
            for required in ("experiment_id", "title", "status", "dataset_scope", "design",
                             "metrics", "conclusion", "paper_use", "limitations", "sources"):
                if required not in row:
                    raise ValueError(f"{module}: {row.get('experiment_id')} missing {required}")
            record = dict(row, module=module, document_path=modules[-1]["document_path"])
            record["source_ids"] = []
            for source in row["sources"]:
                key = source_key(source)
                if key not in sources:
                    sources[key] = dict(source, source_id=f"S{len(sources)+1:03d}", used_by=[])
                entry = sources[key]
                entry["used_by"].append(row["experiment_id"])
                record["source_ids"].append(entry["source_id"])
            experiments.append(record)
    ids = [row["experiment_id"] for row in experiments]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment-family IDs must be globally unique")

    snapshot_dir = EVIDENCE / "reviewed_sources"
    snapshot_dir.mkdir(exist_ok=True)
    total_bytes = 0
    for source in sources.values():
        raw = read_source(source)
        source["verified_bytes"] = len(raw)
        source["sha256_verified"] = True
        source["used_by"] = sorted(set(source["used_by"]))
        if source.get("commit"):
            source["immutable_url"] = "https://github.com/" + source.get("repository", "jackzhu119/research4") + "/blob/" + source["commit"] + "/" + quote(source["path"], safe="/")
        suffix = Path(source["path"]).suffix.lower()
        if len(raw) <= MAX_COPY_BYTES and suffix in TEXT_SUFFIXES:
            target = snapshot_dir / (source["sha256"] + suffix)
            if not target.exists():
                target.write_bytes(raw)
                total_bytes += len(raw)
            elif target.read_bytes() != raw:
                raise ValueError("Snapshot content collision")
            source["verbatim_snapshot_path"] = str(target.relative_to(ROOT))
        else:
            source["verbatim_snapshot_path"] = None
            source["snapshot_note"] = "Linked and SHA-256 verified; not copied because of size or binary format."

    now = datetime.now(timezone.utc).isoformat()
    registry = {"schema": "research4-paper-experiment-family-registry-v1", "built_at_utc": now,
                "scope": "Reviewed experiment families and frozen protocols, not independent studies or individual fit counts.",
                "all_background_producers_untouched": True, "modules": modules, "experiments": experiments}
    write_json(EVIDENCE / "experiment_registry.json", registry)
    write_json(EVIDENCE / "sources.json", {"schema": "research4-paper-reviewed-source-registry-v1", "built_at_utc": now, "sources": list(sources.values())})
    columns = ["experiment_id", "module", "title", "status", "dataset_scope", "design", "metrics_location", "metric_value_count", "conclusion", "paper_use", "limitations", "source_ids", "document_path"]
    # Excel truncates fields beyond 32,767 characters. Complete nested results
    # belong in the JSON registry and normalized long CSV, not one giant cell.
    csv_records = [dict(row, metrics_location="metric_values.csv; filter experiment_id=" + row["experiment_id"],
                        metric_value_count=sum(1 for _ in flatten(row["metrics"]))) for row in experiments]
    csv_write(EVIDENCE / "experiment_registry.csv", columns, csv_records)
    metrics = []
    for row in experiments:
        for metric_path, value in flatten(row["metrics"]):
            metrics.append({"experiment_id": row["experiment_id"], "metric_path": metric_path,
                            "value": value, "dataset_scope": row["dataset_scope"], "source_ids": row["source_ids"]})
    csv_write(EVIDENCE / "metric_values.csv", ["experiment_id", "metric_path", "value", "dataset_scope", "source_ids"], metrics)
    csv_write(EVIDENCE / "sources.csv", ["source_id", "repository", "commit", "path", "sha256", "verified_bytes", "immutable_url", "verbatim_snapshot_path", "used_by"], list(sources.values()))

    current = {x["experiment_id"]: x for x in experiments if x["module"] == "current_species"}
    tables = EVIDENCE / "tables"
    tables.mkdir(exist_ok=True)
    primary = current["CUR-014"]["metrics"]["primary_results"]
    primary_rows = [{"method": x["method"], "selection_scope": x["selection_scope"],
                     "representation": x["representation"], "k_rule": x["k"],
                     "unique_specimens": x["unique_specimens"], "source_count_each_outer_fold": x["source_count"],
                     "complete_partition_repeats": x["complete_repeats"],
                     **x["mean_metrics"], "experiment_id": "CUR-014"} for x in primary]
    csv_write(tables / "current_acid_primary_models.csv", ["method", "selection_scope", "representation", "k_rule", "unique_specimens", "source_count_each_outer_fold", "complete_partition_repeats", "accuracy", "macro_f1", "balanced_accuracy", "experiment_id"], primary_rows)
    holdout_rows = [{"task": x["task"], "model": x["model"], "method": x["method"],
                     "prediction_N": x["prediction_N"], **{key: x["metrics"][key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
                     "experiment_id": "CUR-004"} for x in current["CUR-004"]["metrics"]["all_source_and_full_retrain_results"]]
    csv_write(tables / "corrected_fixed_holdout_all_results.csv", ["task", "model", "method", "prediction_N", "accuracy", "macro_f1", "balanced_accuracy", "experiment_id"], holdout_rows)
    roles = current["CUR-002"]["metrics"]["kind_roles"]
    csv_write(tables / "current_task_roles.csv", ["task", "train", "calibration", "prediction"], [{"task": key, **value} for key, value in roles.items()])

    def cell(value):
        return scalar_text(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# 全部实验族台账", "", "由已审查的分模块台账生成。每行代表一个实验族或冻结协议，不能把行数当独立研究数量。数据范围不同的成绩不在此横向排序。", "",
             "完整参数与指标见[JSON台账](../../evidence/paper_bundle_20261109/experiment_registry.json)、[Excel可读CSV](../../evidence/paper_bundle_20261109/experiment_registry.csv)和[长表指标](../../evidence/paper_bundle_20261109/metric_values.csv)。", "",
             "| ID | 实验族 | 状态 | 结论与用途 | 分模块记录 |", "| --- | --- | --- | --- | --- |"]
    for row in experiments:
        target = Path(row["document_path"]).name
        lines.append("| " + " | ".join([row["experiment_id"], cell(row["title"]), cell(row["status"]), cell(row["conclusion"]), f"[设计、指标与证据]({target})"]) + " |")
    (DOCS / "08_EXPERIMENT_REGISTER.md").write_text("\n".join(lines) + "\n")
    evidence_lines = ["# 核验过的来源", "", "源文件按固定Git提交核对SHA-256；小型文本原样保存在reviewed_sources。原样快照内的历史相对链接请通过原始提交打开。实时结果引用的是带观察时间的独立副本，不是对进行中任务完成的承诺。", "", "| 来源 | 原路径与固定提交 | SHA-256 | 本地核验副本 |", "| --- | --- | --- | --- |"]
    for source in sources.values():
        original = source.get("immutable_url")
        label = source["path"]
        link = f"[{cell(label)}]({original})" if original else cell(label) + "（运行快照）"
        copy = source.get("verbatim_snapshot_path")
        local = f"[原样副本](../../{copy})" if copy else "保留原提交链接"
        evidence_lines.append(f"| {source['source_id']} | {link} | `{source['sha256']}` | {local} |")
    (DOCS / "09_REVIEWED_SOURCES.md").write_text("\n".join(evidence_lines) + "\n")
    report = {"status": "ALL_REFERENCED_SOURCE_HASHES_VERIFIED", "built_at_utc": now,
              "experiment_families": len(experiments), "referenced_sources": len(sources),
              "metric_values": len(metrics), "module_counts": {x["module"]: x["experiment_families"] for x in modules},
              "source_copy_bytes_added": total_bytes, "new_models_trained": 0, "background_tasks_modified": False}
    write_json(EVIDENCE / "build_validation.json", report)
    return report


if __name__ == "__main__":
    try:
        os.nice(5)
    except OSError:
        pass
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))
