"""Finite corrected-species representation/auxiliary ablation, TRAIN rows only.

Every fit accepts source data and selected auxiliary tasks, never support/query
truth. The inherited scientific encoder, losses, optimizers and C10 reference
are reused unchanged. New axes/source scopes have new cache identities.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/research/evidence_validation_20261108_v3"
OLD_SCRIPT = ROOT / "scripts/species_revision_background_neural.py"
OLD_SHA = "d776d682aae3b5606bedaf3ef36b04664e8fd5ec87317a413c2e012a8c13aa4c"
REPRESENTATIONS = ("alpha128", "alpha512")
AUXILIARIES = ("target_only", "target_extra", "hq_only", "tm_only", "hq_tm")
CLASS_ORDER = ["ZY03", "ZY04", "ZY05", "ZY06", "ZY07"]
SCHEMA = "research4-representation-transfer-v3"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(path, value):
    if Path(path).exists():
        need(read(path) == value, "Existing job identity differs; preserve this attempt")
    else:
        atomic_json(path, value)


def old_worker():
    need(sha(OLD_SCRIPT) == OLD_SHA, "Inherited scientific worker source changed")
    name = "research4_representation_inherited_neural"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, OLD_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def validate_case(representation, source_count, auxiliary, target_epochs=192):
    need(representation in REPRESENTATIONS and source_count in (190, 240)
         and auxiliary in AUXILIARIES, "Unregistered finite ablation condition")
    need(target_epochs in (144, 192), "Unregistered target-epoch count")
    if target_epochs == 144:
        need(source_count == 240 and auxiliary in ("target_only", "hq_tm"),
             "Matched576update144epoch control only permits240source target_only orhq_tm")
    reference = representation == "alpha128" and source_count == 190 and auxiliary in ("target_only", "target_extra", "hq_tm")
    need(not reference, "REFERENCE_ALREADY_RUN: use the original complete OOF group, do not refit it as new")


def validate_frozen_inputs(features_path, split_path):
    """Verify the exact999-row exported pack before opening a fit scope."""
    features_path, split_path = Path(features_path).resolve(), Path(split_path).resolve()
    lock_path = features_path.parent / "input_lock.json"
    need(lock_path.is_file(), "Frozen TRAIN input_lock.json required")
    lock = read(lock_path)
    need(lock.get("schema") == "research4-evidence-validation-TRAIN-only-v3"
         and lock.get("run_id") == "evidence-validation-20261108-v3"
         and lock.get("training_records") == 999 and lock.get("no_CAL_or_PRED_exported") is True,
         "Exact new-run TRAIN-only lock required")
    allowlist = lock.get("training_allowlist", [])
    need(len(allowlist) == len(set(allowlist)) == 999, "Lock requires999 unique allowed IDs")
    need(sha(features_path) == lock["training_feature_sha256"]
         and sha(split_path) == lock["training_split_sha256"], "Frozen feature/split bytes changed")
    with np.load(features_path, allow_pickle=False) as data:
        ids = data["record_ids"].astype(str).tolist()
    need(len(ids) == len(set(ids)) == 999 and set(ids) == set(allowlist), "NPZ differs from exact TRAIN allowlist")
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    need(len(rows) == 999 and all(row["role"] == "train" for row in rows)
         and len({row["record_id"] for row in rows}) == 999
         and {row["record_id"] for row in rows} == set(allowlist), "CSV differs from exact TRAIN allowlist")
    return {"path": str(lock_path), "sha256": sha(lock_path), "training_records": 999,
            "exact_allowlist_verified": True, "no_master_CAL_PRED_exported": True}


def load_training(features_path, split_path, representation):
    """Narrow master TRAIN before finite checks, interpolation or normalization."""
    need(representation in REPRESENTATIONS, "Unknown spectral representation")
    with Path(split_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [{key: row[key] for key in ("record_id", "physical_sample_id", "species", "origin_group_id")}
                for row in csv.DictReader(handle) if row["role"] == "train"]
    rows.sort(key=lambda row: row["record_id"])
    need(rows and len({r["record_id"] for r in rows}) == len(rows), "Unique TRAIN IDs required")
    need(all(r["physical_sample_id"] for r in rows)
         and len({r["physical_sample_id"] for r in rows}) == len(rows), "Unique independent TRAIN physical IDs required")
    with np.load(features_path, allow_pickle=False) as data:
        all_ids = data["record_ids"].astype(str)
        lookup = {rid: i for i, rid in enumerate(all_ids)}
        need(len(lookup) == len(all_ids) and all(r["record_id"] in lookup for r in rows), "TRAIN feature/ID mismatch")
        indices = np.asarray([lookup[r["record_id"]] for r in rows], dtype=int)
        axis = np.linspace(.21, 1.49, 128 if representation == "alpha128" else 512)
        if representation == "alpha128":
            historical_axis = data["axis_transfer"].astype(float, copy=True)
            need(historical_axis.shape == axis.shape and np.allclose(historical_axis, axis, rtol=0, atol=1e-12),
                 "Historical128 frequency axis differs")
            axis = historical_axis
            values = data["X_alpha_transfer"][indices].astype(float, copy=True)
            mode = "exact historical TRAIN alpha128 cache, no resampling"
        else:
            axis_key = "axis_main" if "axis_main" in data.files else "axis"
            alpha_key = "X_alpha_main" if "X_alpha_main" in data.files else "X_alpha"
            original_axis = data[axis_key].astype(float, copy=True)
            original = data[alpha_key][indices].astype(float, copy=True)
            need(original_axis.ndim == 1 and np.isfinite(original_axis).all()
                 and np.all(np.diff(original_axis) > 0), "Increasing finite native axis required")
            need(original.shape == (len(rows), len(original_axis)) and np.isfinite(original).all(),
                 "Finite native TRAIN samples-by-frequency matrix required")
            need(original_axis[0] <= axis[0] and original_axis[-1] >= axis[-1], "Interpolation must not extrapolate")
            values = np.stack([np.interp(axis, original_axis, row) for row in original])
            mode = "per-spectrum linear interpolation of TRAIN native alpha; same0.21–1.49THz range as128"
    need(values.shape == (len(rows), len(axis)) and np.isfinite(values).all(), "Finite TRAIN representation required")
    store = {"record_ids": np.asarray([r["record_id"] for r in rows]),
             "sample_ids": np.asarray([r["physical_sample_id"] for r in rows]),
             "labels": np.asarray([r["origin_group_id"] for r in rows]), "X": values}
    species = np.asarray([r["species"] for r in rows])
    return store, species, axis, {"representation": representation, "points": len(axis),
                                "range_THz": [float(axis[0]), float(axis[-1])], "method": mode,
                                "frequency_units": "workbook THz label, metrology not independently verified",
                                "master_CAL_PRED_exported_or_used": False}


def partition(store, species, fold, source_count, auxiliary):
    old = old_worker()
    source, support, query, truth, pools, classes = old.development_partition(store, species, "suanzaoren", fold)
    need(classes == CLASS_ORDER and len(source["record_ids"]) == 190
         and len(support["record_ids"]) == 50 and len(query["record_ids"]) == 60,
         "Canonical acid190/50/60 partition required")
    if source_count == 240:
        combined = {key: np.concatenate((source[key], support[key]), axis=0) for key in source}
        order = np.argsort(combined["record_ids"])
        source = {key: value[order].copy() for key, value in combined.items()}
        support = {key: value[:0].copy() for key, value in support.items()}
    else:
        need(source_count == 190, "Only190 or240 source specimens are registered")
    names = {"target_only": (), "target_extra": (), "hq_only": ("huangqi",),
             "tm_only": ("tianma",), "hq_tm": ("huangqi", "tianma")}[auxiliary]
    pools = {name: pools[name] for name in names}
    source_ids, support_ids, query_ids = [set(p["sample_ids"].tolist()) for p in (source, support, query)]
    need(not source_ids & support_ids and not source_ids & query_ids and not support_ids & query_ids,
         "Source/support/query samples overlap")
    need("labels" not in query, "Query truth must remain separate from inference input")
    for name, pool in pools.items():
        need(not source_ids & set(pool["sample_ids"]) and not query_ids & set(pool["sample_ids"]), "Auxiliary target overlap")
        need(not set(pool["labels"]) & set(classes), "Auxiliary labels cannot become acid origin labels")
    return source, support, query, truth, pools, classes


def configuration(args, source, auxiliary, classes, axis):
    old = old_worker()
    profile = dict(old.PROFILES["p24_e192_full"], name="p24_e192_full", blocks=2, dropout=.35,
                   residual=True, attention=False, learning_rate=.0005, weight_decay=.001,
                   target_batch_size=64, pretrain_batch_size=32, pretrain_steps=480,
                   class_weight="source balanced")
    profile["epochs"] = getattr(args, "target_epochs", 192)
    if profile["epochs"] == 144:
        profile["name"] = "p24_e144_full_matched_updates"
    files = ["upgrade_neural.py", "conservative_calibration.py", "calibration.py"]
    return {"schema": SCHEMA, "profile": profile, "seed": args.seed,
            "source_fingerprint": old.pack_hash(source),
            "auxiliary_fingerprints": {name: old.pack_hash(pack) for name, pack in auxiliary.items()},
            "axis_sha256": hashlib.sha256(np.ascontiguousarray(axis).tobytes()).hexdigest(),
            "worker_sha256": sha(__file__),
            "reused_code_sha256": {**{name: sha(ROOT / "src/herb_spectra" / name) for name in files},
                                     "inherited_worker": sha(OLD_SCRIPT)},
            "classes": classes, "representation": args.representation, "source_count": args.source_count,
            "auxiliary_selection": args.auxiliary, "target_epochs": profile["epochs"]}


def fit_pair(cache, source, auxiliary, config, classes):
    """No query/support arrays or truth are accepted by this fit interface."""
    old = old_worker()
    arm = {"target_only": "A", "target_extra": "Aextra", "hq_only": "B", "tm_only": "B", "hq_tm": "B"}[config["auxiliary_selection"]]
    neural, metadata, receipt = old.cached_neural(cache / "neural", source, auxiliary, arm, config, classes)
    reference, ref_receipt = old.cached_reference(cache / "reference", source, config, classes)
    need(metadata["source_ids"] == source["record_ids"].tolist(), "Neural training source receipt differs")
    return {"neural": neural, "matched_SVM": reference}, {"neural": receipt, "matched_SVM": ref_receipt}, metadata


def metric_record(truth, predicted, classes):
    precision, recall, f1, counts = precision_recall_fscore_support(truth, predicted, labels=classes, zero_division=0)
    return {"correct": int(np.sum(np.asarray(truth) == np.asarray(predicted))), "denominator": len(truth),
            "accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)), "class_order": classes,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
            "per_class": [{"label": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
                          for label, p, r, f, n in zip(classes, precision, recall, f1, counts)]}


def score_pair(scorers, source, support, query, classes, with_calibration):
    """Pointwise predictions have no access to query truth or query statistics."""
    old = old_worker()
    result = []
    for name, scorer in scorers.items():
        query_features, scores = scorer(query["X"])
        need(scores.shape == (len(query["record_ids"]), len(classes)) and np.isfinite(scores).all(), "Invalid class scores")
        base = np.asarray(classes)[scores.argmax(axis=1)]
        result.append({"model": name, "setting": "source_only", "repeat": -1, "shots": 0,
                       "predicted": base, "scores": scores})
        if with_calibration:
            need(len(support["record_ids"]) == 50, "240source has no independent support pool; calibration disabled")
            source_features, source_scores = scorer(source["X"])
            support_features, support_scores = scorer(support["X"])
            context = old.build_context(source_features, source_scores, source["labels"], np.asarray(classes), seed=42)
            zero = old.fit_candidate(context, {"id": "identity_zero_support", "family": "identity"})
            result.append({"model": name, "setting": "identity_zero_support", "repeat": -1, "shots": 0,
                           "predicted": np.asarray(classes)[zero.predict_scores(query_features, scores).argmax(axis=1)], "scores": None})
            for draw in old.support_draws(support, classes):
                indices = np.asarray(draw["indices"], dtype=int)
                for candidate in old.CALIBRATORS:
                    fitted = old.fit_candidate(context, candidate, support_features[indices], support_scores[indices], support["labels"][indices])
                    calibrated = fitted.predict_scores(query_features, scores)
                    result.append({"model": name, "setting": candidate["id"], "repeat": draw["repeat"], "shots": draw["shots"],
                                   "predicted": np.asarray(classes)[calibrated.argmax(axis=1)], "scores": None})
    return result


def write_predictions(path, predictions, query_ids):
    # Deterministic gzip bytes allow safe retry verification after a model-cache boundary.
    with Path(path).open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "setting", "repeat", "shots", "record_id", "predicted", "scores_json"], lineterminator="\n")
            writer.writeheader()
            for block in predictions:
                for i, rid in enumerate(query_ids):
                    writer.writerow({"model": block["model"], "setting": block["setting"], "repeat": block["repeat"], "shots": block["shots"],
                                     "record_id": str(rid), "predicted": str(block["predicted"][i]),
                                     "scores_json": "" if block["scores"] is None else json.dumps(block["scores"][i].tolist(), separators=(",", ":"), allow_nan=False)})


def verified_complete(output, manifest_sha):
    p = output / "complete.json"
    if not p.exists():
        return False
    done = read(p)
    need(done["job_manifest_sha256"] == manifest_sha and done["status"] == "complete", "Completion identity differs")
    for name, checksum in done["artifacts_sha256"].items():
        need((output / name).is_file() and sha(output / name) == checksum, "Completed artifact seal differs")
    for reference in done.get("model_receipts", {}).values():
        cache = Path(reference["cache_path"])
        receipt_path = cache / "complete.json"
        need(sha(receipt_path) == reference["complete_sha256"], "Completed model receipt seal differs")
        old_worker().validate_receipt(cache, read(receipt_path))
    return True


def run_job(args):
    target_epochs = getattr(args, "target_epochs", 192)
    validate_case(args.representation, args.source_count, args.auxiliary, target_epochs)
    need(not args.with_calibration or args.source_count == 190, "No support calibration for240source")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    input_lock = validate_frozen_inputs(args.features, args.split)
    store, species, axis, representation = load_training(args.features, args.split, args.representation)
    need(len(store["record_ids"]) == 999 and
         {kind: int(np.sum(species == kind)) for kind in ("酸枣仁", "枣仁", "黄芪", "天麻")} ==
         {"酸枣仁": 300, "枣仁": 216, "黄芪": 241, "天麻": 242},
         "The frozen canonical999TRAIN corrected-kind pool is required")
    source, support, query, truth, auxiliary, classes = partition(store, species, args.outer_fold, args.source_count, args.auxiliary)
    config = configuration(args, source, auxiliary, classes, axis)
    manifest = {"schema": SCHEMA, "task": "acid_origins", "fold": args.outer_fold, "seed": args.seed,
                "config": config, "source_ids": source["record_ids"].tolist(), "support_ids": support["record_ids"].tolist(),
                "query_ids": query["record_ids"].tolist(), "auxiliary_ids": {name: pack["record_ids"].tolist() for name, pack in auxiliary.items()},
                "features_sha256": sha(args.features), "split_sha256": sha(args.split), "representation": representation,
                "input_lock": input_lock, "target_epochs": target_epochs,
                "with_calibration": args.with_calibration, "query_truth_accepted_by_fit": False,
                "scope": "TRAIN_only_exploratory_internal_OOF_not_new_test_not_cross_batch"}
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(out / "job_manifest.json", manifest)
        identity = objhash(manifest)
        if verified_complete(out, identity):
            print(json.dumps({"status": "already_complete", "new_model_fits": 0, "output": str(out)})); return
        attempt = out / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        cache = Path(args.cache_root).resolve() if args.cache_root else BASE / "cache/representation_transfer"
        scorers, model_receipts, training = fit_pair(cache, source, auxiliary, config, classes)
        atomic_json(attempt / "fit_lock.json", {"job_manifest_sha256": identity, "model_receipts": model_receipts,
                    "all_fits_closed_before_query_scoring": True, "no_support_or_query_accepted_by_fit": True})
        predictions = score_pair(scorers, source, support, query, classes, args.with_calibration)
        write_predictions(attempt / "predictions.csv.gz", predictions, query["record_ids"])
        atomic_json(attempt / "prediction_lock.json", {"prediction_sha256": sha(attempt / "predictions.csv.gz"),
                    "query_ids": query["record_ids"].tolist(), "query_truth_used_to_generate_predictions": False})
        # Truth is first consumed by this scoring stage, after prediction bytes are fixed.
        bases = {p["model"]: p["predicted"] for p in predictions if p["setting"] == "source_only"}
        metrics = []
        for p in predictions:
            before = bases[p["model"]] == truth
            after = p["predicted"] == truth
            metrics.append({"model": p["model"], "setting": p["setting"], "repeat": p["repeat"], "shots": p["shots"],
                            **metric_record(truth, p["predicted"], classes),
                            "corrected": int(np.sum(~before & after)), "damaged": int(np.sum(before & ~after))})
        atomic_json(attempt / "query_truth.json", {str(rid): str(y) for rid, y in zip(query["record_ids"], truth)})
        summary = {"schema": SCHEMA, "status": "complete", "task": "acid_origins", "fold": args.outer_fold,
                   "seed": args.seed, "representation": args.representation, "source_count": args.source_count,
                   "target_epochs": target_epochs,
                   "support_count": len(support["record_ids"]), "query_count": len(query["record_ids"]),
                   "auxiliary": args.auxiliary, "profile": config["profile"], "job_manifest_sha256": identity,
                   "model_receipts": model_receipts, "metrics": metrics, "fit_source_ids": training["source_ids"],
                   "fit_auxiliary_ids": training["auxiliary_ids"], "initial_encoder_sha256": training["initial_encoder_sha256"],
                   "initial_head_sha256": training["initial_head_sha256"], "target_order_sha256": training["target_order_sha256"],
                   "pretrain_steps_actual": training["pretrain_steps_actual"],
                   "target_epochs_actual": training["target_epochs_actual"],
                   "target_optimizer_steps_actual": int(np.ceil(args.source_count / 64)) * training["target_epochs_actual"],
                   "new_model_fits": sum(r["new_model_fits"] for r in model_receipts.values()),
                   "elapsed_cpu_seconds": time.process_time() - start_cpu, "elapsed_wall_seconds": time.monotonic() - start_wall,
                   "no_master_CAL_PRED_used": True, "scope": manifest["scope"],
                   "interpretation": "Resolution also changes physical receptive field in the inherited CNN; no isolated chemistry claim."}
        atomic_json(attempt / "summary.json", summary)
        names = ["summary.json", "predictions.csv.gz", "fit_lock.json", "prediction_lock.json", "query_truth.json"]
        partial = [out / name for name in names if (out / name).exists()]
        if partial:
            preserved = out / "preserved_partial_publication" / uuid.uuid4().hex
            preserved.mkdir(parents=True)
            for path in partial:
                path.rename(preserved / path.name)
        for name in names:
            shutil.copyfile(attempt / name, out / name)
        atomic_json(out / "complete.json", {"status": "complete", "schema": SCHEMA,
                    "job_manifest_sha256": identity, "attempt": str(attempt.relative_to(out)),
                    "artifacts_sha256": {name: sha(out / name) for name in ["job_manifest.json", *names]},
                    "model_receipts": model_receipts, "checkpoint_scope": "exact-identity complete fitted-model boundaries, no arbitrary mid-epoch resume"})
        print(json.dumps({"status": "complete", "output": str(out), "new_model_fits": summary["new_model_fits"],
                          "base_metrics": [{"model": m["model"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                                           for m in metrics if m["setting"] == "source_only"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--source-count", type=int, choices=(190, 240), required=True)
    parser.add_argument("--target-epochs", type=int, choices=(144, 192), default=192)
    parser.add_argument("--auxiliary", choices=AUXILIARIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--with-calibration", action="store_true")
    run_job(parser.parse_args())


if __name__ == "__main__":
    main()
