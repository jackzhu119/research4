#!/usr/bin/env python3
"""Guarded historical-prediction reanalysis and TRAIN-only descriptive peaks.

No fit, inference, new quantum calculation, task relabeling in place, or Git.
--prepare-config writes metadata/hash references only; actual numerical work
requires a hash-bound root release and the existing sole 72h guardian.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PREFIX = ROOT / "reports/species-correction-20261108-v1"
REMOTE = PREFIX / "frozen-remote/21d6edab24d198dc60b1ad4ef50e83602c9ecfa7"
GUARD = ROOT / "scripts/run_research4_research72h_budget_20261107.py"
GUARD_SHA = "643a431b7229eca94ac50645d467e6c3472f976b8ac48cc2f36be68241319e68"
SCHEMA = "research4-species-correction-postq-config-v1"
RESULT_SCHEMA = "research4-species-correction-postq-result-v1"
RELEASE_SCHEMA = "research4-species-correction-postq-root-release-v1"
COMMIT = "21d6edab24d198dc60b1ad4ef50e83602c9ecfa7"
KINDS = ("枣仁", "酸枣仁")
OUTSIDE = "__OTHER_SPECIES__"
SCOPE = "HISTORICAL_MIXED_TAXON_TRAINED_MODEL_RETROSPECTIVE_SUBSET_NOT_PURE_RETRAINING"
PEAK_PARAMETERS = {"sg_window": 15, "sg_polyorder": 3, "prominence_fraction_of_raw_mean_ptp": .05,
                   "minimum_frequency_distance_thz": .03, "axis_uniform_rtol": 1e-5, "axis_uniform_atol": 1e-10}


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    raw = Path(value)
    candidate = raw if raw.is_absolute() else ROOT / raw
    require(".." not in raw.parts and not any(p.is_symlink() for p in (candidate, *candidate.parents)), "Unsafe path")
    resolved = candidate.resolve()
    require(resolved.is_relative_to(ROOT), "Path escapes repository")
    return resolved


def sha(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def ref(file):
    file = path(file)
    return {"path": str(file), "sha256": sha(file)}


def checked(record):
    require(set(record) == {"path", "sha256"}, "Exact path/hash reference required")
    p = path(record["path"])
    require(p.is_file() and sha(p) == record["sha256"], "Input hash differs: " + str(p))
    return p


def read(file):
    p = path(file)
    require(p.stat().st_size <= 16 * 1024**2, "Oversized JSON metadata")
    return json.loads(p.read_text())


def save(file, value):
    p = path(file)
    require(p.is_relative_to(PREFIX) and not p.exists(), "Fresh correction output required")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_csv(file):
    p = path(file)
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) <= 100000, "Bounded prediction/metadata rows required")
    return rows


def write_csv(file, rows, fields):
    p = path(file)
    require(p.is_relative_to(PREFIX) and not p.exists(), "Fresh CSV output required")
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def proposed_config():
    """Hash existing artifacts; never materialize numerical NPZ arrays."""
    core = ROOT / "reports/staged-nested-optimization/2026-10-31/core/outer_predictions.csv.gz"
    combos = ROOT / "reports/staged-nested-optimization/2026-10-31/combinations/outer_predictions.csv.gz"
    sources = [{"key": "historical_" + method, "family": "nested_old_7_group", "ref": ref(core),
                "method_filter": method, "paired_baseline": "historical_R2"}
               for method in ("R0", "R1", "R2", "R3")]
    sources.append({"key": "historical_R4", "family": "nested_old_7_group", "ref": ref(combos),
                    "method_filter": "R4", "paired_baseline": "historical_R2"})
    robust = ROOT / "reports/research72h-20261107-v1/frozen-robustness/results-v1/predictions"
    for file in sorted(robust.glob("*.csv.gz")):
        sources.append({"key": "robustness::" + file.name.removesuffix(".csv.gz"), "family": "frozen_old_R2_perturbation",
                        "ref": ref(file), "method_filter": None, "paired_baseline": "robustness::baseline"})
    aug = ROOT / "reports/research72h-20261107-v1/autonomous-main-001/jobs/source-shift-augmentation/attempt-001/predictions"
    for file in sorted(aug.glob("*.csv.gz")):
        case = file.name.removesuffix(".csv.gz").split("__", 1)[1]
        base = {"clean": "baseline", "axis_minus_0p002": "axis_shift_-0.002_THz", "axis_plus_0p002": "axis_shift_0.002_THz"}[case]
        sources.append({"key": "augmentation::" + file.name.removesuffix(".csv.gz"), "family": "old_source_augmented_SVC",
                        "ref": ref(file), "method_filter": None, "paired_baseline": "robustness::" + base})
    require(len(sources) == 29, "Expected 5 nested +18 robustness +6 augmentation prediction groups")
    return {"schema": SCHEMA, "worker": ref(SELF), "guard": ref(GUARD), "remote_commit": COMMIT,
            "remote_snapshot": ref(REMOTE / "REMOTE_SNAPSHOT.json"),
            "canonical_registry": ref(REMOTE / "data/active_workbooks.json"),
            "canonical_split": ref(REMOTE / "data/research/species_revision_20261108/split_manifest.csv"),
            "candidate": ref(ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"),
            "peak_module": ref(ROOT / "src/herb_spectra/experimental_peaks.py"),
            "leakage_guard_module": ref(ROOT / "src/herb_spectra/leakage_guard.py"),
            "prediction_sources": sources, "scope": SCOPE, "expected_source_counts": {"枣仁": 213, "酸枣仁": 300},
            "peak_parameters": dict(PEAK_PARAMETERS),
            "fits": 0, "inference": 0, "quantum": 0}


def verify_snapshot(config):
    p = checked(config["remote_snapshot"])
    snapshot = read(p)
    require(config["remote_commit"] == COMMIT and snapshot["source"]["commit"] == COMMIT
            and snapshot.get("raw_utf8_bytes_git_blob_verified") is True, "Frozen verified remote commit required")
    for row in snapshot["files"]:
        file = path(p.parent / row["path"])
        require(file.is_relative_to(p.parent) and file.is_file(), "Remote path escapes snapshot")
        raw = file.read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
                and hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == row["git_blob_sha"],
                "Frozen remote Git blob/hash differs")
    entries = {str(path(p.parent / row["path"])): row for row in snapshot["files"]}
    for key in ("canonical_registry", "canonical_split"):
        file = checked(config[key])
        require(str(file) in entries and entries[str(file)]["sha256"] == config[key]["sha256"], "Canonical ref absent from Git snapshot")
    return snapshot


def taxonomy(registry, split_rows):
    require(len(split_rows) == 1672 and len({r["record_id"] for r in split_rows}) == 1672, "Unique1672 canonical rows required")
    by_id = {r["record_id"]: r for r in split_rows}
    groups = {}
    for entry in registry["workbooks"]:
        group = entry.get("origin_group_id")
        if group and group.startswith("ZY"):
            require(entry.get("species_verified") is True and entry.get("origin_verified") is True, "User-confirmed category/origin required")
            value = {"species": entry["species"], "origin": entry["origin"], "source_workbook": entry["path"], "workbook_sha256": entry["sha256"]}
            require(group not in groups or groups[group] == value, "Conflicting group metadata")
            groups[group] = value
    require(set(groups) == {"ZY%02d" % i for i in range(1, 8)}, "Seven canonical historical ZY groups required")
    legacy = {k: v for k, v in by_id.items() if v["origin_group_id"] in {"ZY%02d" % i for i in range(1, 8)}}
    require(len(legacy) == 855, "Canonical historical855 identities required")
    for row in legacy.values():
        group = groups[row["origin_group_id"]]
        require(row["species"] == group["species"] and row["origin"] == group["origin"], "Canonical record/group disagreement")
    require(sum(v["species"] == "枣仁" for v in legacy.values()) == 355
            and sum(v["species"] == "酸枣仁" for v in legacy.values()) == 500, "Historical355/500 category counts changed")
    return by_id, groups, legacy


def metric(truth, predicted, labels):
    require(len(truth) == len(predicted) and len(truth) > 0 and len(labels) == len(set(labels)), "Invalid metric population")
    require(set(truth).issubset(labels), "Truth outside declared labels")
    axis = list(labels) + sorted(set(predicted) - set(labels))
    cm = {t: {p: 0 for p in axis} for t in axis}
    for t, p in zip(truth, predicted):
        cm[t][p] += 1
    per = []
    for label in labels:
        support = sum(cm[label].values()); total = sum(cm[t][label] for t in axis); tp = cm[label][label]
        precision = tp / total if total else 0.0; recall = tp / support if support else 0.0
        per.append({"label": label, "support": support, "correct": tp, "predicted": total, "precision": precision,
                    "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    correct = sum(t == p for t, p in zip(truth, predicted))
    return {"samples": len(truth), "correct": correct, "accuracy": correct / len(truth),
            "macro_f1": sum(v["f1"] for v in per) / len(labels),
            "balanced_accuracy": sum(v["recall"] for v in per) / len(labels), "labels_for_macro": labels,
            "confusion_axis": axis, "confusion": cm, "per_class": per}


def paired_counts(before, after):
    require(len(before) == len(after), "Paired populations differ")
    return {"rescued": sum(not b and a for b, a in zip(before, after)), "harmed": sum(b and not a for b, a in zip(before, after)),
            "net_correct": sum(after) - sum(before)}


def map_rows(rows, legacy, groups):
    ids = [v["record_id"] for v in rows]
    require(len(ids) == 855 and len(set(ids)) == 855 and set(ids) == set(legacy), "Exact855 prediction population required")
    mapped = []
    for v in sorted(rows, key=lambda x: x["record_id"]):
        info = legacy[v["record_id"]]; t = info["origin_group_id"]; p = v["y_predicted"]
        require(p in groups and p in {"ZY%02d" % i for i in range(1, 8)}, "Prediction outside historical7 groups")
        if "y_true" in v:
            require(v["y_true"] == t, "Old true group does not match canonical ID group")
        require(v["outer_fold"] in {str(i) for i in range(5)}, "Historical outer fold invalid")
        tg, pg = groups[t], groups[p]
        mapped.append({"record_id": v["record_id"], "outer_fold": v["outer_fold"], "historical_true_group": t,
                       "historical_predicted_group": p, "true_species": tg["species"], "predicted_species": pg["species"],
                       "true_origin": tg["origin"], "predicted_origin": pg["origin"],
                       "old_group_correct": t == p, "species_correct": tg["species"] == pg["species"],
                       "joint_species_origin_correct": (tg["species"], tg["origin"]) == (pg["species"], pg["origin"]),
                       "canonical_role": info["role"]})
    return mapped


def subsets(mapped):
    result = {}
    for category in ("ALL", *KINDS):
        rows = mapped if category == "ALL" else [r for r in mapped if r["true_species"] == category]
        labels = sorted({r["true_species"] + "|" + r["true_origin"] for r in rows})
        truth = [r["true_species"] + "|" + r["true_origin"] for r in rows]
        predicted = [r["predicted_species"] + "|" + r["predicted_origin"]
                     if category == "ALL" or r["predicted_species"] == category else OUTSIDE for r in rows]
        metrics = metric(truth, predicted, labels)
        metrics.update(old_group_correct=sum(r["old_group_correct"] for r in rows),
                       species_correct=sum(r["species_correct"] for r in rows),
                       species_accuracy=sum(r["species_correct"] for r in rows) / len(rows), scope=SCOPE)
        if category == "ALL":
            metrics["species_task"] = metric([r["true_species"] for r in rows], [r["predicted_species"] for r in rows], list(KINDS))
        result[category] = metrics
    return result


def selected_source_ids(legacy, candidate_ids):
    require(len(candidate_ids) == 855 and len(set(candidate_ids)) == 855 and set(candidate_ids) == set(legacy), "Candidate IDs differ from historical855")
    chosen = {kind: sorted(k for k, v in legacy.items() if v["species"] == kind and v["role"] == "train") for kind in KINDS}
    require({kind: len(ids) for kind, ids in chosen.items()} == {"枣仁": 213, "酸枣仁": 300}, "Expected canonical TRAIN213/300 only")
    require(all(legacy[r]["role"] == "train" for ids in chosen.values() for r in ids), "Non-TRAIN source feature access")
    return chosen


def selected_row_bytes(stream, header_end, shape, itemsize, indices):
    require(len(shape) == 2 and shape == (855, 612) and itemsize == 8, "Exact historical float64 candidate matrix required")
    require(len(indices) == len(set(indices)) and all(type(i) is int and 0 <= i < shape[0] for i in indices), "Unique in-range source rows")
    result = {}
    width = shape[1] * itemsize
    for index in sorted(indices):
        stream.seek(header_end + index * width)
        raw = stream.read(width)
        require(len(raw) == width, "Truncated selected source row")
        result[index] = raw
    return result


def peaks(config, legacy, groups, out, check_budget):
    # Numerical imports and archive decoding occur only after actual live supervision.
    import numpy as np
    peak_file = checked(config["peak_module"]); guard_file = checked(config["leakage_guard_module"])
    require(peak_file == ROOT / "src/herb_spectra/experimental_peaks.py"
            and guard_file == ROOT / "src/herb_spectra/leakage_guard.py", "Canonical original peak and fit-scope modules required")
    sys.path.insert(0, str(ROOT / "src"))
    from herb_spectra import experimental_peaks, leakage_guard
    require(Path(experimental_peaks.__file__).resolve() == peak_file
            and Path(leakage_guard.__file__).resolve() == guard_file, "Unexpected imported module location")
    require(asdict(experimental_peaks.LOCKED_PEAK_PROTOCOL) == config["peak_parameters"] == PEAK_PARAMETERS,
            "Exact original locked peak parameters required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    candidate = checked(config["candidate"])
    with np.load(candidate, allow_pickle=False) as archive:
        ids = archive["record_ids"].astype(str).tolist()
        axis = np.asarray(archive["axis"], dtype=float)
    require(axis.shape == (612,) and np.isfinite(axis).all() and (np.diff(axis) > 0).all()
            and float(axis.min()) >= .1 and float(axis.max()) <= 1.5, "Finite registered612point THz axis required")
    chosen = selected_source_ids(legacy, ids); positions = {v: i for i, v in enumerate(ids)}
    select = sorted({positions[v] for vv in chosen.values() for v in vv})
    with zipfile.ZipFile(candidate) as z, z.open("X_alpha.npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Supported row-major NPY header required")
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                else np.lib.format.read_array_header_2_0(stream))
        require(fortran is False and dtype.kind == "f" and dtype.itemsize == 8, "Row-major finite float64 source decoder required")
        raw = selected_row_bytes(stream, stream.tell(), shape, dtype.itemsize, select)
    source = {i: np.frombuffer(raw[i], dtype=dtype).copy() for i in select}
    require(all(np.isfinite(v).all() for v in source.values()), "Nonfinite selected TRAIN spectrum")
    source_rows, candidate_rows, summaries = [], [], []
    for number, kind in enumerate(KINDS):
        check_budget(); selected = chosen[kind]
        values = np.stack([source[positions[v]] for v in selected])
        scope = leakage_guard.FitScope(selected, "species-correction-TRAIN-" + kind,
            "2026-11-08-confirmed-origin-kind-v1", out / "peak_fit_scope_manifest.jsonl",
            code_version=config["peak_module"]["sha256"])
        result = experimental_peaks.analyze_source_peaks(values, axis, selected, scope)
        mean, smooth = result["raw_source_mean"], result["smoothed_source_mean"]
        prominence = result["required_prominence"]
        indices = np.asarray([v["frequency_index"] for v in result["candidates"]], dtype=int)
        save(out / ("train_peak_scope_audit_%d.json" % number),
             {k: v for k, v in result.items() if k not in ("raw_source_mean", "smoothed_source_mean")})
        for rid in selected:
            group = groups[legacy[rid]["origin_group_id"]]
            source_rows.append({"record_id": rid, "species": kind, "origin": group["origin"], "role": "train",
                                "source_workbook": group["source_workbook"], "source_workbook_sha256": group["workbook_sha256"],
                                "candidate_row_index": positions[rid]})
        mean_rows = [{"frequency_THz": float(x), "raw_train_mean": float(y), "SG15_poly3_train_mean": float(s)}
                     for x, y, s in zip(axis, mean, smooth)]
        write_csv(out / ("train_mean_%d.csv" % number), mean_rows, list(mean_rows[0]))
        for candidate_row in result["candidates"]:
            candidate_rows.append({"species": kind, **candidate_row, "source_count": len(selected),
                                   "required_prominence": prominence})
        fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(axis, mean, alpha=.6, label="Raw TRAIN mean")
        ax.plot(axis, smooth, label="SG15/poly3 TRAIN mean"); ax.plot(axis[indices], smooth[indices], "o", label="Candidate local maxima")
        ax.set(xlabel="Frequency (THz)", ylabel="Measured alpha (original instrument units)", title=("Suanzaoren" if kind == "酸枣仁" else "Zaoren historical subset") + " / TRAIN n=%d" % len(selected))
        ax.legend(); fig.tight_layout()
        for suffix in ("png", "pdf"):
            file = out / ("train_mean_%d.%s" % (number, suffix)); require(not file.exists(), "Preserve existing plot"); fig.savefig(file, dpi=180)
        plt.close(fig)
        summaries.append({"species": kind, "selected_train_ids": selected, "source_count": len(selected),
                          "peak_count": len(indices), "prominence_threshold": prominence,
                          "raw_source_mean_ptp": result["raw_source_mean_ptp"],
                          "minimum_distance_samples": result["minimum_distance_samples"],
                          "median_axis_step_thz": result["median_axis_step_thz"],
                          "source_data_fingerprint": result["source_data_fingerprint"],
                          "fit_provenance": result["fit_provenance"],
                          "exclude_new_ZR01_02_03_not_present_in_old_candidate": True})
    write_csv(out / "peak_source_ids.csv", source_rows, list(source_rows[0]))
    fields = ["species", "candidate_number", "frequency_index", "frequency_thz", "wavenumber_cm1",
              "smoothed_source_mean_height", "raw_source_mean_height", "prominence", "left_base_frequency_thz",
              "right_base_frequency_thz", "chemical_assignment", "theoretical_peak_cm1", "classification_prior_used",
              "source_count", "required_prominence"]
    write_csv(out / "train_only_candidate_peaks.csv", candidate_rows, fields)
    checked(config["peak_module"]); checked(config["leakage_guard_module"])
    return {"groups": summaries, "source_IDs": ref(out / "peak_source_ids.csv"), "candidate_input": config["candidate"],
            "source_rows_decoded": len(select), "CAL_or_QUERY_rows_decoded_as_feature_arrays": 0, "candidate_y_read": False,
            "access_boundary": "NPZ metadata axis/IDs read; X_alpha member streamed and only selected TRAIN rows decoded. ZIP implementation may decompress skipped bytes; no OS-isolation claim.",
            "peaks_are_measured_training_mean_descriptions_not_theoretical_IR_or_chemical_assignments": True,
            "classification_priors_created": False}


def execute(args):
    config_file = checked({"path": str(args.config), "sha256": args.config_sha256})
    config = read(config_file)
    require(config.get("schema") == SCHEMA and checked(config["worker"]) == SELF and config["worker"]["sha256"] == sha(SELF)
            and checked(config["guard"]) == GUARD and config["guard"]["sha256"] == GUARD_SHA, "Frozen worker and guard required")
    release_file = checked({"path": str(args.root_release), "sha256": args.root_release_sha256}); release = read(release_file)
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == "ROOT_APPROVED_FINITE_POST_Q_ANALYSIS"
            and release.get("config") == ref(config_file) and release.get("worker") == ref(SELF), "Exact root release required")
    spec = importlib.util.spec_from_file_location("species_correction_actual72hguard", GUARD); guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    supervision = guard.verify_resource_supervision(expected_threads=1)
    ledger = guard.read(guard.root_path(supervision["proof"]["ledger_path"])); guard.validate_ledger(ledger, allow_running=True)
    running = [v for v in ledger["phases"] if v["status"] == "RUNNING"]
    require(len(running) == 1 and running[0]["phase"] == supervision["proof"]["phase"], "Exactly this guarded analysis, no parallel Q")
    command = running[0]["command"]; output = path(args.output)
    require(str(SELF) in command and "--output" in command and path(command[command.index("--output") + 1]) == output, "Live worker/output binding differs")
    require(output.is_relative_to(PREFIX) and not output.exists(), "Fresh distinct correction output required")
    output.mkdir(parents=True); started = time.process_time()
    def check_budget():
        require(time.process_time() < supervision["proof"]["actual_cpu_stop_threshold_seconds"], "Analysis CPU cleanup threshold reached")
        require(datetime.now(timezone.utc) < datetime.fromisoformat(ledger["effective_run_deadline_at_utc"]), "Authorized deadline reached")
    verify_snapshot(config)
    require(config.get("scope") == SCOPE and config.get("fits") == 0 and config.get("inference") == 0 and config.get("quantum") == 0, "Analysis-only scope required")
    by_id, groups, legacy = taxonomy(read(checked(config["canonical_registry"])), read_csv(checked(config["canonical_split"])))
    require(config["expected_source_counts"] == {"枣仁": 213, "酸枣仁": 300}, "Predeclared TRAIN counts differ")
    require(len(config["prediction_sources"]) == 29, "Finite29 prediction groups required")
    datasets, file_cache = {}, {}
    for item in config["prediction_sources"]:
        check_budget(); file = checked(item["ref"])
        if str(file) not in file_cache:
            file_cache[str(file)] = read_csv(file)
        rows = file_cache[str(file)]
        if item["method_filter"] is not None:
            rows = [r for r in rows if r.get("method") == item["method_filter"]]
        require(item["key"] not in datasets, "Duplicate prediction group")
        datasets[item["key"]] = {"mapped": map_rows(rows, legacy, groups), "spec": item}
    require(set(datasets) >= {"historical_" + m for m in ("R0", "R1", "R2", "R3", "R4")} | {"robustness::baseline"}, "Missing old comparison families")
    require([r["historical_predicted_group"] for r in datasets["historical_R2"]["mapped"]]
            == [r["historical_predicted_group"] for r in datasets["robustness::baseline"]["mapped"]], "Frozen baseline no longer equals old R2")
    comparisons, summaries, mapped_rows = {}, [], []
    for key, value in datasets.items():
        check_budget(); mapped = value["mapped"]; baseline = datasets[value["spec"]["paired_baseline"]]["mapped"]
        result = subsets(mapped); comparisons[key] = result
        for category, metrics in result.items():
            selected = mapped if category == "ALL" else [r for r in mapped if r["true_species"] == category]
            before = baseline if category == "ALL" else [r for r in baseline if r["true_species"] == category]
            require([r["record_id"] for r in selected] == [r["record_id"] for r in before], "Paired subset IDs differ")
            pair = {name: paired_counts([r[name] for r in before], [r[name] for r in selected])
                    for name in ("old_group_correct", "species_correct", "joint_species_origin_correct")}
            metrics["paired_changes"] = pair
            summaries.append({"dataset": key, "family": value["spec"]["family"], "subset_true_species": category,
                              "samples": metrics["samples"], "joint_species_origin_correct": metrics["correct"],
                              "joint_accuracy": metrics["accuracy"], "origin_macro_f1": metrics["macro_f1"],
                              "origin_balanced_accuracy": metrics["balanced_accuracy"], "species_accuracy": metrics["species_accuracy"],
                              "old_group_correct": metrics["old_group_correct"], "baseline_dataset": value["spec"]["paired_baseline"],
                              "joint_rescued": pair["joint_species_origin_correct"]["rescued"], "joint_harmed": pair["joint_species_origin_correct"]["harmed"], "scope": SCOPE})
        mapped_rows.extend({"dataset": key, **row} for row in mapped)
    write_csv(output / "historical_corrected_semantics_metrics.csv", summaries, list(summaries[0]))
    write_csv(output / "historical_prediction_species_origin_mapping.csv.gz", mapped_rows, list(mapped_rows[0]))
    save(output / "confusions_and_metrics.json", comparisons)
    peak_result = peaks(config, legacy, groups, output, check_budget)
    for key in ("canonical_registry", "canonical_split", "candidate", "remote_snapshot", "peak_module", "leakage_guard_module"):
        checked(config[key])
    for item in config["prediction_sources"]:
        checked(item["ref"])
    require(sha(config_file) == args.config_sha256 and sha(release_file) == args.root_release_sha256, "Config/release changed during analysis")
    save(output / "README.json", {"scope": SCOPE, "labels": "user-confirmed kinds/origins, not independent taxonomic laboratory authentication",
         "origin_metric": "Joint species+origin; cross-species predictions remain errors. Per-species macro averages only canonical true origin classes, confusion retains OTHER_SPECIES.",
         "old_predictions": "Reused without model fitting/inference; they were generated by mixed-taxon training. Not pure-species retraining or new confirmation.",
         "source_peaks": "Only canonical TRAIN rows available in the historical855 file: acid300 and oldZaoren213. The nine added Zaoren records are not included.",
         "future_use": "Descriptive candidate peaks; no model prior, chemical attribution, theory or IR intensity computed."})
    artifacts = {str(p.relative_to(output)): ref(p) for p in sorted(output.iterdir()) if p.is_file()}
    result = {"schema": RESULT_SCHEMA, "status": "DONE_SPECIES_CORRECTION_POSTQ_ANALYSIS", "completed_at_utc": utc(),
              "source_worker": ref(SELF), "resource_supervision": supervision, "output_directory": str(output),
              "config": ref(config_file), "root_release": ref(release_file), "historical_prediction_groups": len(datasets),
              "historical_prediction_rows": len(mapped_rows), "new_model_fits": 0, "new_model_inferences": 0,
              "new_quantum_calculations": 0, "canonical_species_retraining_performed": False, "scope": SCOPE,
              "training_peak_analysis": peak_result, "artifacts": artifacts, "CPU_seconds": time.process_time() - started}
    save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path); parser.add_argument("--config-sha256")
    parser.add_argument("--root-release", type=Path); parser.add_argument("--root-release-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_config is not None:
        require(not args.execute, "Preparation and execution are separate")
        save(args.prepare_config, proposed_config())
        print(json.dumps({"status": "CONFIG_PREPARED_METADATA_ONLY_NOT_RUN", "config": ref(args.prepare_config)})); return
    require(args.execute and all((args.config, args.config_sha256, args.root_release, args.root_release_sha256, args.output)), "Explicit execution/config/root release/output required")
    result = execute(args)
    print(json.dumps({"status": result["status"], "output": result["output_directory"]}))


if __name__ == "__main__":
    main()
