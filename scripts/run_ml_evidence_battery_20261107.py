#!/usr/bin/env python3
"""Additive post-hoc diagnostics of frozen, same-855 OOF predictions.

This script never fits, loads, predicts with, or selects a learned model. The
2026-11-06 battery already covered R0--R4 uncertainty and feature stability;
this battery covers their missing strong controls and rare-class sensitivity.
Freeze input/source hashes first, then run into a fresh, separately named
directory. Results are conditional development diagnostics, not confirmation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time


ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path("reports/research72h-20261107-v1")
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
TEST = Path("tests/test_ml_evidence_battery_20261107.py")
FAIR = Path("reports/paper-c/2026-10-31/fair-benchmark")
STRONG = Path("reports/paper-c/2026-10-31/strong-mlp")
DATA = Path("outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz")
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
COUNTS = {"ZY01": 350, "ZY02": 5, **{f"ZY{i:02d}": 100 for i in range(3, 8)}}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
FAIR_METHODS = (
    "R2_reproduction", "full_svm", "shrinkage_lda", "mlp",
    "R2_full_features", "R2_random_K", "R2_no_C_gamma_search",
    "R2_no_class_weight", "R2_fixed_MC20_LV2_K62",
)
STRONG_METHODS = ("strong_mlp_200", "strong_mlp_200_plus100_source_updates")
METHODS = FAIR_METHODS + STRONG_METHODS
PAIRS = tuple(("R2_reproduction", m) for m in METHODS[1:]) + ((STRONG_METHODS[1], STRONG_METHODS[0]),)
REPLICATES = 2000
SEED = 20261107
INPUTS = (
    DATA, Path("data/active_workbooks.json"),
    Path("data/annotations/zaoren_user_declaration_20261031.json"),
    FAIR / "outer_predictions.csv.gz", FAIR / "pooled_metrics.json",
    FAIR / "splits.json", FAIR / "selection_lock.json", FAIR / "run_manifest.json",
    FAIR / "release_snapshot.json", STRONG / "outer_predictions.csv.gz",
    STRONG / "primary_seed42_metrics.json", STRONG / "selection_lock.json",
    STRONG / "run_manifest.json", Path("reports/paper-c/2026-10-31/history-audit/main_results.csv"),
    Path("reports/cloud-followup-20261106-v1/model-evidence/summary.json"),
)
LIMITATIONS = (
    "All 855 specimens and their old test scores have been exposed historically; this is retrospective development analysis.",
    "Each anonymous origin group has only one user-declared source lot; no actual cross-batch or named-origin validation is created.",
    "Bootstrap intervals condition on saved OOF predictions and observed class counts; training, tuning, architecture, and seed uncertainty are excluded.",
    "OOF models share training specimens. Conditional percentile ranges are not a validated nested-CV significance test or population/batch confidence guarantee.",
    "ZY02 has only five user-declared independent specimens. Deletion is fixed-prediction influence analysis, not retraining or biological replication.",
    "Strong MLP uses the predeclared primary seed 42 only; seeds 43/44 were not used to select this seed or pooled as additional specimens.",
    "Source-update continuation is 200 epochs plus 100 source gradient updates, not 300 training epochs.",
    "All controls share IDs and folds, but candidate grids and compute differ; no equal-compute or universal superiority claim is made.",
    "The 171-record, external-plant, and quantum experiments are deliberately not mixed with this same-855 OOF analysis.",
    "Existing 2026-11-06 R0--R4 bootstrap and feature stability are cited, not repeated or relabelled as newly independent evidence.",
    "No new model, feature, seed, or hyperparameter may be chosen from these diagnostics. No new training, inference, label permutation, or quantum task is performed.",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing empty evidence table")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fresh_output(path, root=None):
    path, allowed = Path(path).resolve(), (Path(ROOT if root is None else root) / PREFIX).resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("Output must be a fresh child of the additive 72h report prefix")
    if path.exists():
        raise ValueError("Fresh output required; refusing to overwrite")
    return path


def freeze(output):
    output = fresh_output(output)
    artifacts = {str(p): {"sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size}
                 for p in (*INPUTS, SCRIPT, TEST)}
    protocol = {
        "protocol_id": "same855-fixed-oof-strong-controls-20261107-v1",
        "frozen_at_utc": now(), "status": "FROZEN_NOT_RUN", "new_model_fits": 0,
        "methods": list(METHODS), "pairs": [list(p) for p in PAIRS],
        "primary_strong_mlp_seed": 42, "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED, "bootstrap_unit": "user-declared physical specimen, within true class",
        "metric_order": list(METRICS), "rare_class": "ZY02", "rare_class_deletions": "each of five single specimens separately",
        "fold_sensitivity": "omit each of five saved outer test folds, without refitting",
        "multiplicity": "descriptive conditional intervals only; no p-values or significance decisions",
        "bounds": {"cpu_seconds": 1800, "wall_seconds": 1800, "threads": 1,
                   "bootstrap_chunk_size": 100, "expected_disk_bytes_below": 5000000},
        "limitations": list(LIMITATIONS), "artifacts": artifacts,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(sha(output / "protocol.json") + "\n", encoding="ascii")
    print(json.dumps({"status": "FROZEN_NOT_RUN", "protocol": str(output / "protocol.json"),
                      "sha256": sha(output / "protocol.json")}, ensure_ascii=False))


def verify_protocol(path):
    path = Path(path)
    expected = path.with_name("protocol.sha256").read_text(encoding="ascii").strip()
    if sha(path) != expected:
        raise ValueError("Frozen protocol hash changed")
    protocol = read_json(path)
    if (protocol.get("methods") != list(METHODS) or protocol.get("pairs") != [list(p) for p in PAIRS]
            or protocol.get("bootstrap_replicates") != REPLICATES or protocol.get("bootstrap_seed") != SEED
            or protocol.get("limitations") != list(LIMITATIONS)
            or protocol.get("bounds") != {"cpu_seconds": 1800, "wall_seconds": 1800, "threads": 1,
                                          "bootstrap_chunk_size": 100, "expected_disk_bytes_below": 5000000}):
        raise ValueError("Frozen scientific settings do not match reviewed source")
    if set(protocol["artifacts"]) != {str(p) for p in (*INPUTS, SCRIPT, TEST)}:
        raise ValueError("Frozen input/source closure differs")
    for relative, entry in protocol["artifacts"].items():
        if sha(ROOT / relative) != entry["sha256"] or (ROOT / relative).stat().st_size != entry["bytes"]:
            raise ValueError(f"Frozen artifact changed: {relative}")
    return protocol


def metric_array(confusion):
    import numpy as np
    cm = np.asarray(confusion)
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    if np.any(support <= 0):
        raise ValueError("Every metric stratum must retain every true class")
    recall = diagonal / support
    f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros_like(diagonal, dtype=float),
                   where=support + predicted > 0)
    return np.stack([diagonal.sum(axis=-1) / support.sum(axis=-1), f1.mean(axis=-1), recall.mean(axis=-1)], axis=-1)


def confusion(y, pred, n_classes):
    import numpy as np
    return np.bincount(np.asarray(y) * n_classes + np.asarray(pred), minlength=n_classes**2).reshape(n_classes, n_classes)


def align_rows(rows, methods, truth, membership, seed=None):
    """Use declared IDs and folds, never CSV row order, to form paired arrays."""
    import numpy as np
    by_method = {method: {} for method in methods}
    for row in rows:
        if seed is not None and int(row["seed"]) != seed:
            continue
        method, name = row["method"], row["record_id"]
        if method not in by_method:
            raise ValueError(f"Unexpected method: {method}")
        if name not in truth or name in by_method[method]:
            raise ValueError("Unknown or duplicate method/specimen ID")
        if row["y_true"] != truth[name] or int(row["outer_fold"]) != membership[name]:
            raise ValueError("Truth or outer-fold identity mismatch")
        if row["y_predicted"] not in CLASSES:
            raise ValueError("Unknown predicted class")
        if "is_correct" in row and int(row["is_correct"]) != int(row["y_true"] == row["y_predicted"]):
            raise ValueError("Saved correctness flag disagrees with labels")
        by_method[method][name] = row["y_predicted"]
    names = sorted(truth)
    vocabulary = {label: index for index, label in enumerate(CLASSES)}
    result = {}
    for method, predictions in by_method.items():
        if set(predictions) != set(truth):
            raise ValueError(f"Incomplete method specimen set: {method}")
        result[method] = np.asarray([vocabulary[predictions[name]] for name in names], dtype=int)
    return result


def validate_splits(splits, truth):
    membership = {}
    if len(splits) != 5 or {int(x["fold"]) for x in splits} != set(range(5)):
        raise ValueError("Exactly the original five outer folds are required")
    for record in splits:
        train_list, test_list = record["train_sample_ids"], record["test_sample_ids"]
        train, test = set(train_list), set(test_list)
        if len(train) != len(train_list) or len(test) != len(test_list) or train & test or train | test != set(truth):
            raise ValueError("Invalid source/query specimen partition")
        if len(train) != 684 or len(test) != 171:
            raise ValueError("Historical fold sizes changed")
        for name in test:
            if name in membership:
                raise ValueError("Duplicate outer query specimen")
            membership[name] = int(record["fold"])
    if set(membership) != set(truth):
        raise ValueError("Missing outer query specimen")
    return membership


def bootstrap_metrics(y, predictions, replicates=REPLICATES, seed=SEED, chunk_size=100):
    import numpy as np
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y))
    if set(np.unique(y)) != set(range(n_classes)):
        raise ValueError("Class indices must be contiguous")
    pools = [np.flatnonzero(y == label) for label in range(n_classes)]
    result = {method: np.empty((replicates, len(METRICS))) for method in predictions}
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = np.concatenate([rng.choice(pool, size=(count, len(pool)), replace=True) for pool in pools], axis=1)
        offsets = np.arange(count)[:, None] * n_classes**2
        for method, prediction in predictions.items():
            encoded = y[indices] * n_classes + prediction[indices] + offsets
            cms = np.bincount(encoded.ravel(), minlength=count * n_classes**2).reshape(count, n_classes, n_classes)
            result[method][start:start+count] = metric_array(cms)
    return result


def diagnostic_tables(names, y, folds, predictions, replicates=REPLICATES, seed=SEED):
    import numpy as np
    cms = {m: confusion(y, p, len(CLASSES)) for m, p in predictions.items()}
    points = {m: metric_array(cm) for m, cm in cms.items()}
    boot = bootstrap_metrics(y, predictions, replicates, seed)
    metrics, paired, per_class, deletion, omitted_folds = [], [], [], [], []
    for method in predictions:
        for j, metric in enumerate(METRICS):
            low, high = np.quantile(boot[method][:, j], [.025, .975])
            metrics.append({"method": method, "metric": metric, "estimate": float(points[method][j]),
                            "conditional_percentile_low": float(low), "conditional_percentile_high": float(high),
                            "n_specimens": len(y), "bootstrap_replicates": replicates})
    for method, baseline in PAIRS:
        for j, metric in enumerate(METRICS):
            delta = boot[method][:, j] - boot[baseline][:, j]
            low, high = np.quantile(delta, [.025, .975])
            paired.append({"method": method, "baseline": baseline, "metric": metric,
                           "delta_percentage_points": float(100 * (points[method][j] - points[baseline][j])),
                           "conditional_low_percentage_points": float(100 * low),
                           "conditional_high_percentage_points": float(100 * high),
                           "bootstrap_replicates": replicates})
        for label, class_name in enumerate(CLASSES):
            mask = y == label
            a, b = predictions[method][mask] == label, predictions[baseline][mask] == label
            per_class.append({"method": method, "baseline": baseline, "true_class": class_name,
                              "support": int(mask.sum()), "correct_method": int(a.sum()), "correct_baseline": int(b.sum()),
                              "rescued": int((a & ~b).sum()), "harmed": int((b & ~a).sum()),
                              "recall_delta_percentage_points": float(100 * (a.mean() - b.mean()))})
    omissions = [("single_ZY02_specimen", name, np.arange(len(y)) != i)
                 for i, name in enumerate(names) if y[i] == 1]
    omissions += [("outer_query_fold", str(fold), folds != fold) for fold in sorted(set(folds))]
    for kind, value, mask in omissions:
        subset = {m: metric_array(confusion(y[mask], p[mask], len(CLASSES))) for m, p in predictions.items()}
        target = deletion if kind == "single_ZY02_specimen" else omitted_folds
        for method, baseline in PAIRS:
            for j, metric in enumerate(METRICS):
                target.append({"omission_type": kind, "omitted": value, "method": method, "baseline": baseline,
                               "metric": metric, "remaining_specimens": int(mask.sum()),
                               "method_metric": float(subset[method][j]), "baseline_metric": float(subset[baseline][j]),
                               "delta_percentage_points": float(100 * (subset[method][j] - subset[baseline][j])),
                               "change_from_full_delta_percentage_points": float(100 * ((subset[method][j] - subset[baseline][j])
                                                                                      - (points[method][j] - points[baseline][j])))})
    return {"method_metrics.csv": metrics, "paired_conditional_ranges.csv": paired,
            "paired_per_class_changes.csv": per_class, "rare_class_single_deletion.csv": deletion,
            "outer_fold_omission.csv": omitted_folds}, points


def apply_local_resource_limits():
    """Keep the stricter inherited supervisor limit; never raise its hard cap."""
    _, inherited_hard = resource.getrlimit(resource.RLIMIT_CPU)
    hard = 1801 if inherited_hard == resource.RLIM_INFINITY else min(1801, inherited_hard)
    soft = min(1800, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    signal.alarm(1800)


def run(protocol_path, output):
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    apply_local_resource_limits()
    import numpy as np
    started, cpu, wall = now(), time.process_time(), time.monotonic()
    protocol = verify_protocol(protocol_path)
    output = fresh_output(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol_snapshot.json", protocol)
    try:
        with np.load(ROOT / DATA, allow_pickle=False) as data:
            ids, targets, groups = [data[key].astype(str) for key in ("record_ids", "y", "groups")]
        truth = dict(zip(ids.tolist(), targets.tolist(), strict=True))
        if len(ids) != 855 or len(truth) != 855 or not np.array_equal(ids, groups):
            raise ValueError("Current user-declared physical specimen identities changed")
        if {c: int(np.count_nonzero(targets == c)) for c in np.unique(targets)} != COUNTS:
            raise ValueError("Expected seven-group counts changed")
        declaration = read_json(ROOT / INPUTS[2])
        if declaration.get("cross_batch_calibration_ready") is not False:
            raise ValueError("Dataset scope changed; separately review a new protocol")
        membership = validate_splits(read_json(ROOT / FAIR / "splits.json"), truth)
        predictions = align_rows(read_csv(ROOT / FAIR / "outer_predictions.csv.gz"), FAIR_METHODS, truth, membership)
        predictions.update(align_rows(read_csv(ROOT / STRONG / "outer_predictions.csv.gz"), STRONG_METHODS, truth, membership, seed=42))
        for directory in (FAIR, STRONG):
            lock, manifest = read_json(ROOT / directory / "selection_lock.json"), read_json(ROOT / directory / "run_manifest.json")
            if lock["outer_test_used_to_select"] is not False or manifest["input_sha256"] != sha(ROOT / DATA):
                raise ValueError("Historical selection or dataset hash mismatch")
        names = sorted(truth)
        y = np.asarray([CLASSES.index(truth[name]) for name in names], dtype=int)
        folds = np.asarray([membership[name] for name in names], dtype=int)
        expected = read_json(ROOT / FAIR / "pooled_metrics.json")
        expected.update(read_json(ROOT / STRONG / "primary_seed42_metrics.json"))
        for method, prediction in predictions.items():
            calculated = metric_array(confusion(y, prediction, len(CLASSES)))
            if any(abs(calculated[j] - expected[method][metric]) > 1e-12 for j, metric in enumerate(METRICS)):
                raise ValueError(f"Independent metric reconstruction differs: {method}")
        if np.count_nonzero(predictions["R2_reproduction"] == y) != 821:
            raise ValueError("Historical R2 result changed")
        tables, points = diagnostic_tables(names, y, folds, predictions)
        for filename, rows in tables.items():
            write_csv(output / filename, rows)
        summary = {
            "status": "COMPLETE", "started_at_utc": started, "completed_at_utc": now(),
            "cpu_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall,
            "new_model_fits": 0, "new_model_inferences": 0, "new_quantum_calculations": 0,
            "specimens": len(y), "class_counts": COUNTS, "methods": list(METHODS),
            "paired_comparisons": len(PAIRS), "bootstrap_replicates": REPLICATES, "seed": SEED,
            "protocol_sha256": sha(protocol_path), "numpy_version": np.__version__,
            "strong_mlp_primary_seed": 42, "limitations": list(LIMITATIONS),
            "point_metrics": {m: {metric: float(points[m][j]) for j, metric in enumerate(METRICS)} for m in METHODS},
            "table_row_counts": {filename: len(rows) for filename, rows in tables.items()},
            "inputs_sha256_unchanged_after_analysis": all(sha(ROOT / p) == value["sha256"] for p, value in protocol["artifacts"].items()),
        }
        if not summary["inputs_sha256_unchanged_after_analysis"]:
            raise ValueError("An input changed during analysis; results are not qualified")
        write_json(output / "summary.json", summary)
        write_json(output / "artifact_index.json", {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                                     for p in sorted(output.iterdir()) if p.is_file()})
        print(json.dumps({k: summary[k] for k in ("status", "cpu_seconds", "wall_seconds", "new_model_fits", "paired_comparisons")}, ensure_ascii=False))
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "at_utc": now(), "error_type": type(error).__name__,
                                             "error": str(error), "cpu_seconds": time.process_time()-cpu,
                                             "new_model_fits": 0})
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declaration = commands.add_parser("freeze", help="Write fixed protocol and source/input hash closure; no analysis")
    declaration.add_argument("--output", type=Path, required=True)
    analysis = commands.add_parser("run", help="Run only the frozen prediction diagnostics, without fitting")
    analysis.add_argument("--protocol", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
