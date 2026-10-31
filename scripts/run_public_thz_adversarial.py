#!/usr/bin/env python3
"""Separately gated G1 diagnostics; never part of main model selection.

prepare/fixture do no research fits. run requires a hash-bound root release.
Truth hooks cover named Python file entry points, not an OS security sandbox.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import csv
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from herb_spectra.leakage_guard import LeakageError

PROTOCOL = "public-thz-ginseng-G1-adversarial-20261031-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol():
    return {
        "protocol_id": PROTOCOL,
        "outer_query_export_group": "G1",
        "model_seed": 42,
        "source_observations": 35,
        "query_observations": 7,
        "inner": "leave each of the five source export groups out, fixed members",
        "representation": "alpha",
        "preprocessing": "snv",
        "mc": {"runs": 10, "components": 1, "ratio": .2},
        "expected_retained_features": 13,
        "candidates": [candidate(c) for c in (1., 10.)],
        "selection": ["mean inner Macro-F1 descending", "mean inner Balanced Accuracy descending", "mean inner Accuracy descending", "candidate ID ascending"],
        "null_permutations": 10,
        "null_permutation_seeds": list(range(1000, 1010)),
        "null_permutation_unit": "labels shuffled within each fixed source export group; membership and class counts fixed",
        "label_invariance": "repeat real source selection and final fit while external query truth CSV is permuted",
        "single_query": "modify only alpha[0,0] by an explicitly fixed offset; model stays fitted; other six scores/predictions must stay unchanged",
        "query_perturbation": {"row": 0, "feature": 0, "offset": .01},
        "classifier_fit_budget": {"per_run": 11, "original_plus_truth_permuted": 22, "ten_null_runs": 110, "total": 132},
        "MC_PLS_fit_budget": {"per_run": 60, "total": 720},
        "majority_baseline": "class chosen using the 35 source labels only; evaluate actual G1 class prevalence",
        "null_pvalue": "(1 + count(null_metric >= observed_metric)) / 11; resolution 1/11; exploratory diagnostic only",
        "outer_diagnostics_cannot_select_main_model": True,
        "no_main_source_budget_change": True,
        "unknown_biological_ancestry": True,
        "export_groups_are_not_biological_batches": True,
        "fresh_independent_confirmation": False,
        "fit_release_required": True,
        "truth_hooks": "named builtins.open/io.open/os.open reads/writes; no claim of OS isolation or arbitrary native-code interception",
    }


def candidate(c):
    return {"id": f"ADV_C{int(c):02d}", "representation": "alpha", "preprocessing": "snv",
            "kind": "svm", "mc": {"runs": 10, "components": 1, "ratio": .2},
            "C": c, "gamma": "scale", "class_weight": "balanced"}


@contextmanager
def truth_file_boundary(paths, events, *, scoring_read=False):
    """Actually reject protected truth reads in fit stages and all writes.

    This intentionally narrow process-local diagnostic is not a security sandbox.
    Existing integer file descriptors and native extension I/O are out of scope.
    """
    protected = {str(Path(p).resolve()) for p in paths}
    old_builtin, old_io, old_os = builtins.open, io.open, os.open

    def check(file, write_mode, route):
        if isinstance(file, int):
            return
        try:
            name = str(Path(os.fsdecode(file)).resolve())
        except TypeError:
            return
        if name not in protected:
            return
        if write_mode or not scoring_read:
            events.append({"event": "TRUTH_ACCESS_REJECTED", "path": name,
                           "route": route, "write": bool(write_mode), "scoring_read": scoring_read})
            raise LeakageError("Scoring truth is isolated from fitting and permanently read-only in named hooks")
        events.append({"event": "TRUTH_SCORING_READ", "path": name, "route": route})

    def hooked_builtin(file, mode="r", *args, **kwargs):
        check(file, any(c in str(mode) for c in "wax+"), "builtins.open")
        return old_builtin(file, mode, *args, **kwargs)

    def hooked_io(file, mode="r", *args, **kwargs):
        check(file, any(c in str(mode) for c in "wax+"), "io.open/Path.open")
        return old_io(file, mode, *args, **kwargs)

    def hooked_os(file, flags, *args, **kwargs):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        check(file, flags & write_flags, "os.open")
        return old_os(file, flags, *args, **kwargs)

    builtins.open, io.open, os.open = hooked_builtin, hooked_io, hooked_os
    try:
        yield
    finally:
        builtins.open, io.open, os.open = old_builtin, old_io, old_os


def query_schema(pack):
    if set(pack) != {"record_ids", "X_alpha", "X_n", "axis"}:
        raise LeakageError("Query pack may contain only numerical spectra, axis and routing IDs")
    n = len(pack["record_ids"])
    if np.asarray(pack["record_ids"]).dtype.kind not in "US" or len(set(pack["record_ids"])) != n:
        raise LeakageError("Query routing IDs must be unique strings")
    axis = np.asarray(pack["axis"])
    if axis.ndim != 1 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise LeakageError("Physical axis must be finite and increasing")
    for key in ("X_alpha", "X_n"):
        x = np.asarray(pack[key])
        if x.dtype.kind not in "iuf" or x.shape != (n, len(axis)) or not np.isfinite(x).all():
            raise LeakageError("Query metadata/labels cannot be numeric spectral features")
    return pack


def shuffle_source_labels(y, groups, seed):
    y, groups = np.asarray(y), np.asarray(groups)
    result = y.copy()
    rng = np.random.default_rng(seed)
    for group in sorted(set(groups)):
        ix = np.flatnonzero(groups == group)
        result[ix] = rng.permutation(y[ix])
        if sorted(result[ix]) != sorted(y[ix]):
            raise LeakageError("Permutation changed group label counts")
    return result


def frozen_paths():
    # Delayed import lets no-fit boundary fixtures run while the core is built.
    from herb_spectra import public_thz_benchmark as core
    names = list(core.SOURCE_FILES) + ["scripts/run_public_thz_adversarial.py", "tests/test_public_thz_adversarial.py"]
    return {name: digest(ROOT / name) for name in names}


def prepare(out):
    from herb_spectra.public_thz_data import load_ginseng
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Fresh adversarial preparation directory required")
    out.mkdir(parents=True, exist_ok=True)
    data = load_ginseng()
    source = data["export_groups"] != "G1"
    query = ~source
    assert int(source.sum()) == 35 and int(query.sum()) == 7
    sp = {k: data[k][source] for k in ("record_ids", "X_alpha", "X_n")}
    sp.update(axis=data["axis"], source_labels=data["y"][source], source_groups=data["export_groups"][source])
    qp = {k: data[k][query] for k in ("record_ids", "X_alpha", "X_n")}
    qp["axis"] = data["axis"]
    query_schema(qp)
    np.savez_compressed(out / "source.npz", **sp)
    np.savez_compressed(out / "query.npz", **qp)
    truth_dir = out / "scoring_truth"
    truth_dir.mkdir()
    truth_path = truth_dir / "outer_G1.csv"
    with truth_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "label"])
        writer.writeheader()
        writer.writerows({"record_id": str(rid), "label": str(y)} for rid, y in zip(data["record_ids"][query], data["y"][query], strict=True))
    # A changed-label file is a trap input, never usable as fitted supervision.
    perm = np.random.default_rng(4242).permutation(data["y"][query])
    if np.array_equal(perm, data["y"][query]):
        raise LeakageError("The fixed query-truth permutation must actually change labels; no fit may start")
    perm_path = truth_dir / "outer_G1_permuted_trap.csv"
    with perm_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "label"])
        writer.writeheader()
        writer.writerows({"record_id": str(rid), "label": str(y)} for rid, y in zip(data["record_ids"][query], perm, strict=True))
    truth_path.chmod(0o444)
    perm_path.chmod(0o444)
    write(out / "plan.json", protocol())
    write(out / "split_manifest.json", {"source_ids": sp["record_ids"].tolist(), "query_ids": qp["record_ids"].tolist(),
        "source_groups": sp["source_groups"].tolist(), "query_group": "G1", "no_shared_record_ids": not bool(set(sp["record_ids"]) & set(qp["record_ids"]))})
    files = ["source.npz", "query.npz", "plan.json", "split_manifest.json", "scoring_truth/outer_G1.csv", "scoring_truth/outer_G1_permuted_trap.csv"]
    manifest = {"protocol_id": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {name: digest(out / name) for name in files}, "source_sha256": frozen_paths(),
        "environment": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "requested_native_threads_during_run": 1, "research_fits": 0,
        "release_required": True, "main_benchmark_source_budget_modified": False}
    write(out / "frozen_manifest.json", manifest)
    return {"status": "PREPARED_NOT_TRAINED", "manifest_sha256": digest(out / "frozen_manifest.json"), "plan_sha256": digest(out / "plan.json")}


def check_release(out, gate_path):
    out = Path(out)
    manifest = read(out / "frozen_manifest.json")
    gate = read(gate_path)
    if not gate.get("allow_adversarial_source_training") or gate.get("protocol_id") != PROTOCOL or gate.get("manifest_sha256") != digest(out / "frozen_manifest.json"):
        raise LeakageError("A hash-bound root diagnostic fit release is required")
    if manifest["source_sha256"] != frozen_paths():
        raise LeakageError("Frozen core, diagnostic source or test code changed")
    if any(digest(out / key) != value for key, value in manifest["files_sha256"].items()) or read(out / "plan.json") != protocol():
        raise LeakageError("Frozen protocol, group members or source/query inputs changed")
    return gate


def fit_one_run(out, src, query, run_name, truth_paths, blocked_truth_file):
    """Source-only full 2-candidate selection then final fit; no outer truth API."""
    from herb_spectra import public_thz_benchmark as core
    from herb_spectra.classification import classification_metrics
    directory = Path(out) / "fits" / run_name
    directory.mkdir(parents=True, exist_ok=False)
    events = []
    cases, fitted = [], {}
    with truth_file_boundary(truth_paths, events):
        # Real protected-file read is attempted and rejected before fits. The
        # observed/repeat calls probe different external label files. Neither
        # file is passed as supervision to source transform/estimator APIs.
        try:
            Path(blocked_truth_file).read_text()
        except LeakageError:
            pass
        else:
            raise LeakageError("Expected external query truth read rejection")
        for config in protocol()["candidates"]:
            scores = []
            records = []
            for group in sorted(set(src["source_groups"])):
                train = src["source_groups"] != group
                val = ~train
                sp = {k: v[train] if k != "axis" else v for k, v in src.items() if k != "source_groups"}
                vp = {k: src[k][val] if k != "axis" else src[k] for k in ("record_ids", "X_alpha", "X_n", "axis")}
                context = {"split_id": f"adversarial.outer_G1.inner_{group}", "source_ids": sp["record_ids"].tolist()}
                # Same source ranking is reused across C values within this run only.
                key = str(group)
                if key not in fitted:
                    bundle, ref = core.transform_source(sp, config, 42, context, directory, {})
                    fitted[key] = (bundle, ref)
                bundle, ref = fitted[key]
                z = bundle.matrix(core.features(sp), .2)
                est = core.fit_estimator(z, sp["source_labels"], sp["record_ids"], config, 42, context, directory)
                pred, raw, audit = core.inference_check(bundle, est, query_schema(vp), .2)
                metric = classification_metrics(src["source_labels"][val], pred, core.CLASSES)
                scores.append(metric)
                records.append({"held_source_group": str(group), "source_ids": sp["record_ids"].tolist(), "validation_ids": vp["record_ids"].tolist(),
                    "prediction": pred.tolist(), "scores": raw.tolist(), "metrics": metric, "retained_indices": bundle.selected(.2).tolist(),
                    "transform": ref, "inference_audit": audit})
            mean = {key: float(np.mean([m[key] for m in scores])) for key in core.METRICS}
            cases.append({"config": config, "mean_inner_metrics": mean, "cases": records})
        winner = sorted(cases, key=lambda c: tuple(-c["mean_inner_metrics"][key] for key in core.METRICS) + (c["config"]["id"],))[0]
        config = winner["config"]
        write(directory / "source_selection_lock.json", {"config": config, "inner_candidates": cases, "outer_truth_read": False})
        sp = {k: v for k, v in src.items() if k != "source_groups"}
        context = {"split_id": "adversarial.outer_G1.final", "source_ids": sp["record_ids"].tolist()}
        bundle, reference = core.transform_source(sp, config, 42, context, directory, {})
        z = bundle.matrix(core.features(sp), .2)
        est = core.fit_estimator(z, sp["source_labels"], sp["record_ids"], config, 42, context, directory)
        pred, raw, audit = core.inference_check(bundle, est, query_schema(query), .2)
        if len(bundle.selected(.2)) != 13:
            raise LeakageError("Prelocked 13 retained features required")
        state = joblib.hash((bundle, est))
        model_path = directory / "model.joblib"
        joblib.dump({"bundle": bundle, "estimator": est}, model_path, compress=3)
        result = {"run": run_name, "chosen_config": config, "retained_indices": bundle.selected(.2).tolist(),
            "training_state_hash": state, "model_sha256": digest(model_path), "model_path": str(model_path.relative_to(out)),
            "source_selection_lock_sha256": digest(directory / "source_selection_lock.json"), "source_labels": src["source_labels"].tolist(),
            "query_ids": query["record_ids"].tolist(), "predictions": pred.tolist(), "scores": raw.tolist(), "inference_audit": audit,
            "classifier_fits": 11, "internal_PLS_fits_expected": 60, "truth_access_events": events,
            "outer_truth_read": False, "outer_metrics_present": False,
            "external_truth_probe": str(Path(blocked_truth_file).relative_to(out))}
        write(directory / "predictions_before_scoring.json", result)
    return result, bundle, est


def run(out, gate_path):
    out = Path(out)
    gate = check_release(out, gate_path)
    write(out / "fit_release_snapshot.json", gate)
    with np.load(out / "source.npz", allow_pickle=False) as pack:
        src = {k: pack[k] for k in pack.files}
    with np.load(out / "query.npz", allow_pickle=False) as pack:
        query = query_schema({k: pack[k] for k in pack.files})
    truth_paths = sorted((out / "scoring_truth").glob("*.csv"))
    original_truth = out / "scoring_truth/outer_G1.csv"
    changed_truth = out / "scoring_truth/outer_G1_permuted_trap.csv"
    observed, bundle, est = fit_one_run(out, src, query, "observed", truth_paths, original_truth)
    repeated, _, _ = fit_one_run(out, src, query, "permuted_query_truth_trap", truth_paths, changed_truth)
    same = {key: observed[key] == repeated[key] for key in ("chosen_config", "retained_indices", "training_state_hash", "predictions", "scores")}
    if not all(same.values()):
        raise LeakageError("External query truth changed source-trained objects")
    events = []
    with truth_file_boundary(truth_paths, events):
        from herb_spectra import public_thz_benchmark as core
        changed = {k: v.copy() for k, v in query.items()}
        changed["X_alpha"][0, 0] += .01
        before = joblib.hash((bundle, est))
        pred, raw, audit = core.inference_check(bundle, est, changed, .2)
        unchanged = before == joblib.hash((bundle, est))
        others_scores = bool(np.allclose(raw[1:], np.array(observed["scores"])[1:], atol=1e-10, rtol=1e-10))
        others_preds = pred[1:].tolist() == observed["predictions"][1:]
        if not unchanged or not others_scores or not others_preds:
            raise LeakageError("One changed query affected fitted state or other query predictions")
    nulls = []
    for i, seed in enumerate(protocol()["null_permutation_seeds"]):
        permuted = dict(src)
        permuted["source_labels"] = shuffle_source_labels(src["source_labels"], src["source_groups"], seed)
        null, _, _ = fit_one_run(out, permuted, query, f"null_{i:02d}_seed{seed}", truth_paths, original_truth)
        nulls.append({"permutation_seed": seed, **null})
    # All 12 source selections, models and predictions are saved before truth read.
    write(out / "diagnostic_prediction_lock.json", {"observed": observed, "query_label_permutation_invariants": same,
        "single_query": {"fitted_state_unchanged": unchanged, "other_six_scores_unchanged": others_scores,
            "other_six_predictions_unchanged": others_preds, "inference_audit": audit}, "nulls": nulls,
        "main_model_selected_from_these_diagnostics": False, "classifier_fits": 132, "outer_truth_read": False})
    from herb_spectra.classification import classification_metrics
    with truth_file_boundary(truth_paths, events, scoring_read=True):
        with (out / "scoring_truth/outer_G1.csv").open(newline="", encoding="utf-8") as stream:
            lookup = {row["record_id"]: row["label"] for row in csv.DictReader(stream)}
        y = np.array([lookup[rid] for rid in query["record_ids"]])
    metric = classification_metrics(y, np.array(observed["predictions"]), ("0", "1"))
    for item in nulls:
        item["metrics"] = classification_metrics(y, np.array(item["predictions"]), ("0", "1"))
    counts = {key: int(np.sum(src["source_labels"] == key)) for key in ("0", "1")}
    majority = sorted(counts, key=lambda key: (-counts[key], key))[0]
    majority_metrics = classification_metrics(y, np.full(len(y), majority), ("0", "1"))
    null_summary = {key: {"observed": metric[key], "null_values": [item["metrics"][key] for item in nulls],
        "empirical_pvalue_plus_one": (1 + sum(item["metrics"][key] >= metric[key] for item in nulls)) / 11,
        "null_mean": float(np.mean([item["metrics"][key] for item in nulls]))} for key in ("accuracy", "macro_f1", "balanced_accuracy")}
    result = {"protocol_id": PROTOCOL, "status": "EXPLORATORY_DIAGNOSTIC_NOT_CONFIRMATION", "observed_metrics": metric,
        "null_summary": null_summary, "majority_source_class_counts": counts, "majority_chosen_class": majority,
        "majority_metrics": majority_metrics, "query_label_permutation_invariants": same,
        "single_query_other_six_unchanged": bool(unchanged and others_scores and others_preds),
        "query_truth_hooks_events": events, "classifier_fits": 132,
        "main_model_selection_or_source_budget_changed": False, "prediction_lock_sha256": digest(out / "diagnostic_prediction_lock.json")}
    write(out / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.action == "run" and args.release is None:
        parser.error("run requires --release from the root after independent review")
    if args.action == "prepare":
        result = prepare(args.out)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result = run(args.out, args.release)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
