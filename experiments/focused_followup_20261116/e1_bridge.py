#!/usr/bin/env python3
"""Exact-ID old-V4 full-alpha SNV SVM bridge; no feature ranking or NN fitting.

The archived V4 input NPZ is not in Git. ``freeze-source`` reconstructs it with
the pinned original decoder and accepts it only when its original SHA256 is
reproduced. Outer and inner partitions are read from archived job manifests.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("FOCUSED_SOURCE_ROOT", ROOT / ".cache/focused_followup_20261116"))
V4 = SOURCE / "v4"
RESULTS = Path(os.environ.get("FOCUSED_RESULT_ROOT", ROOT / "reports/focused_followup_20261116"))
OUT = RESULTS / "e1"
COMMIT = "71a303e089c7c7b7b2557d62372aaceda44e2f91"
ARCHIVE = "data/research/evidence_validation_20261108_v4/"
SEEDS = (42, 20261031, 20261108)
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMAS = ("scale", "scale_x0.1", "scale_x10")
CLASSES = np.asarray(["ZY03", "ZY04", "ZY05", "ZY06", "ZY07"])
EXPECTED_TRAIN = "33540088d94293a87501b9a995d3002689214c5a8ca6694398818b5af0fb9f13"
EXPECTED_MASTER = "40a42f933c0b92843e9261b76b2892fc32772605bb66cb11b83ed1456e75b670"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def obj_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return obj_sha({"shape": list(a.shape), "dtype": a.dtype.str,
                    "payload": hashlib.sha256(a.tobytes()).hexdigest()})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def need(test, message):
    if not test:
        raise ValueError(message)


def receipt(status, **details):
    task = os.environ.get("FOCUSED_TASK_ID", "E1")
    atomic(RESULTS / "task_results" / (task + ".json"), {"status": status, "module": "E1", **details})


def download(row):
    path = row["path"]
    relative = path[len(ARCHIVE):] if path.startswith(ARCHIVE) else "source/" + path
    target = V4 / relative
    if target.exists():
        data = target.read_bytes()
    else:
        url = "https://raw.githubusercontent.com/jackzhu119/research4/" + COMMIT + "/" + urllib.parse.quote(path)
        error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research4-focused-followup/1"}), timeout=120) as response:
                    data = response.read()
                break
            except (OSError, TimeoutError) as exc:
                error = exc
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(str(error))
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".partial")
        temp.write_bytes(data)
        temp.replace(target)
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    need(blob == row["git_blob_sha1"], "Pinned Git blob differs: " + path)
    return {"path": relative, "source_path": path, "source_commit": COMMIT,
            "git_blob_sha1": blob, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_inputs():
    """Run only the sealed value decoder, then reproduce the isolated NPZ hash."""
    output = V4 / "train_only_features.npz"
    lock = read(V4 / "input_lock.json")
    if output.is_file():
        need(sha(output) == lock["training_feature_sha256"], "Reconstructed TRAIN input changed")
        return {"status": "REUSED_VERIFIED", "training_feature_sha256": sha(output)}
    registry = read(V4 / "source/data/active_workbooks.json")
    current = read(ROOT / "data/active_workbooks.json")
    current_by_group = {row["origin_group_id"]: row for row in current["workbooks"]}
    shadow = V4 / "decoder_root"
    (shadow / "data").mkdir(parents=True, exist_ok=True)
    for name in ("active_workbooks.json", "metadata_template.csv"):
        target = shadow / "data" / name
        if not target.exists():
            target.write_bytes((V4 / "source/data" / name).read_bytes())
    checked = []
    for entry in registry["workbooks"]:
        approved = current_by_group.get(entry["origin_group_id"])
        need(approved and approved["sha256"] == entry["sha256"]
             and approved["species"] == entry["species"], "Current authorized workbook does not match pinned decoder source")
        source = ROOT / approved["path"]
        need(source.is_file() and sha(source) == entry["sha256"], "Authorized source workbook hash differs: " + str(source))
        target = shadow / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
        checked.append({"origin_group_id": entry["origin_group_id"], "path": approved["path"], "sha256": entry["sha256"]})
    module_path = V4 / "source/scripts/prepare_species_revision_features.py"
    spec = importlib.util.spec_from_file_location("focused_original_v4_decoder", module_path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    master_dir = V4 / "reconstructed_master"
    decoder.prepare(shadow, master_dir)
    master = master_dir / "features.npz"
    master_sha = sha(master)
    need(master_sha == EXPECTED_MASTER, "Original master NPZ hash not reproduced; bridge remains blocked, do not substitute")
    with np.load(master, allow_pickle=False) as pack:
        data = {key: pack[key].copy() for key in pack.files}
    identifiers = data["record_ids"].astype(str)
    lookup = {rid: i for i, rid in enumerate(identifiers)}
    sorted_ids = lock["training_allowlist"]
    need(len(sorted_ids) == 999 and set(sorted_ids).issubset(lookup), "TRAIN allowlist invalid")
    original_ids = [rid for rid in identifiers if rid in set(sorted_ids)]
    matched = None
    for order_name, ids in (("frozen_allowlist_order", sorted_ids), ("master_order", original_ids)):
        indices = np.asarray([lookup[rid] for rid in ids])
        arrays = {key: value[indices] if value.ndim and value.shape[0] == 1672 else value
                  for key, value in data.items()}
        for keys_name, keys in (("original_key_order", list(arrays)), ("sorted_key_order", sorted(arrays))):
            temp = V4 / "candidate_train.npz"
            np.savez_compressed(temp, **{key: arrays[key] for key in keys})
            if sha(temp) == EXPECTED_TRAIN:
                temp.replace(output)
                matched = {"record_order": order_name, "key_order": keys_name}
                break
            temp.unlink()
        if matched:
            break
    need(matched is not None, "Original isolated TRAIN NPZ hash not reproduced; do not silently replace original data pack")
    return {"status": "COMPLETED", "master_sha256": master_sha, "training_feature_sha256": sha(output),
            "reconstruction": matched, "workbooks_verified": checked,
            "note": "Original sealed decoder; byte-identical original master and isolated TRAIN NPZ; no fitted preprocessing."}


def audit_contexts():
    rows = csv_rows(V4 / "train_only_split.csv")
    lookup = {row["record_id"]: row for row in rows}
    need(len(rows) == len(lookup) == 999 and all(row["role"] == "train" for row in rows), "TRAIN CSV role/count changed")
    need(len({row["physical_sample_id"] for row in rows}) == 999
         and all(row["physical_sample_id"] for row in rows), "Declared physical specimen identity differs")
    contexts = []
    expected = {rid for rid, row in lookup.items() if row["species"] == "酸枣仁"}
    need(len(expected) == 300 and {lookup[rid]["origin_group_id"] for rid in expected} == set(CLASSES), "Pure-acid TRAIN pool changed")
    for seed in SEEDS:
        seen = []
        for fold in range(5):
            folder = V4 / "runs" / ("matched_r%d_f%d" % (seed, fold))
            done, manifest = read(folder / "complete.json"), read(folder / "job_manifest.json")
            need(done["status"] == "complete", "Archived fold is not complete")
            for artifact in ("job_manifest.json", "summary.json", "validation_predictions.csv.gz", "selected_features.csv.gz", "inner_search.csv.gz"):
                need(sha(folder / artifact) == done["artifacts"][artifact], "Archived completed artifact hash differs: " + artifact)
            mlp_checkpoint = folder / "models/fullalpha_MLP.joblib"
            need(mlp_checkpoint.is_file()
                 and sha(mlp_checkpoint) == done["artifacts"]["models/fullalpha_MLP.joblib"],
                 "Archived MLP checkpoint hash differs")
            train, test = manifest["source_record_ids"], manifest["outer_validation_record_ids"]
            need(manifest["repeat_seed"] == seed and manifest["outer_fold"] == fold, "Archived split context differs")
            need(set(manifest["master_TRAIN_acid_ids"]) == expected, "Archived master TRAIN IDs differ")
            need(len(train) == len(set(train)) == 240 and len(test) == len(set(test)) == 60,
                 "Outer split counts differ")
            need(not set(train).intersection(test) and set(train).union(test) == expected, "Outer entity leakage or missing specimen")
            physical_train = {lookup[rid]["physical_sample_id"] for rid in train}
            physical_test = {lookup[rid]["physical_sample_id"] for rid in test}
            need(not physical_train.intersection(physical_test), "Physical entity overlap")
            for inner in manifest["inner_partitions"]:
                fit, valid = inner["fit_record_ids"], inner["validation_record_ids"]
                need(len(fit) == len(set(fit)) == len(valid) == len(set(valid)) == 120,
                     "Actual inner partition counts differ")
                need(not set(fit).intersection(valid) and set(fit).union(valid) == set(train), "Actual inner isolation differs")
            need({x["inner_fold"] for x in manifest["inner_partitions"]} == {0, 1}, "Exactly two original inner partitions required")
            seen.extend(test)
            contexts.append({"index": len(contexts), "repeat_seed": seed, "outer_fold": fold,
                             "folder": str(folder.relative_to(V4)), "job_manifest_sha256": sha(folder / "job_manifest.json"),
                             "outer_train_ids_sha256": obj_sha(train), "outer_test_ids_sha256": obj_sha(test),
                             "inner_partitions_sha256": obj_sha(manifest["inner_partitions"]),
                             "source_record_ids": train, "outer_validation_record_ids": test,
                             "inner_partitions": manifest["inner_partitions"]})
        need(len(seen) == len(set(seen)) == 300 and set(seen) == expected, "Each original repeat needs exactly one OOF prediction per specimen")
    return contexts


def freeze_source():
    inventory = read(Path(__file__).with_name("e1_source_inventory.json"))
    need(inventory["commit"] == COMMIT, "Source inventory commit differs")
    files = [download(row) for row in inventory["files"]]
    context = audit_contexts()
    if (OUT / "frozen_sources.json").is_file():
        frozen = read(OUT / "frozen_sources.json")
        need(frozen["files"] == files and frozen["contexts"] == context,
             "Previously frozen original sources or actual split IDs changed")
        if not (V4 / "train_only_features.npz").exists():
            reconstruct_inputs()
        need(sha(V4 / "train_only_features.npz") == EXPECTED_TRAIN,
             "Previously frozen TRAIN NPZ changed; refuse to substitute")
        receipt("REUSED_VERIFIED", original_split_contexts=15, actual_new_classifier_fits=0,
                training_feature_sha256=EXPECTED_TRAIN)
        return
    rebuilt = reconstruct_inputs()
    with np.load(V4 / "train_only_features.npz", allow_pickle=False) as data:
        axis = data["axis"]
        need(axis.shape == (612,), "Exactly 612 original paired frequencies required")
        for item in context:
            manifest = read(V4 / item["folder"] / "job_manifest.json")
            need(array_sha(axis) == manifest["axis_sha256"], "Actual old-V4 axis identity differs")
    frozen = {"schema": "focused-old-v4-input-freeze-v1", "status": "COMPLETED", "source_commit": COMMIT,
              "files": files, "reconstruction": rebuilt, "contexts": context,
              "training_feature_sha256": EXPECTED_TRAIN, "axis_sha256": array_sha(axis),
              "candidates": [{"C": c, "gamma": g} for c in C_VALUES for g in GAMMAS],
              "SNV": "per spectrum over all612 alpha points, sample std ddof1; StandardScaler fit only current fitting partition",
              "selection": "mean inner macro-F1, balanced accuracy, accuracy; lower C, declared gamma order",
              "original_runtime_versions": read(V4 / "protocol.json").get("runtime_versions")}
    atomic(OUT / "frozen_sources.json", frozen)
    write_csv(OUT / "source_registry.csv", files)
    receipt("COMPLETED", original_split_contexts=15, actual_new_classifier_fits=0,
            training_feature_sha256=EXPECTED_TRAIN, original_npz_reconstructed_byte_identically=True)


def snv(values):
    deviation = values.std(axis=1, ddof=1, keepdims=True)
    need(np.isfinite(deviation).all() and np.all(deviation > np.finfo(float).eps), "SNV undefined")
    return (values - values.mean(axis=1, keepdims=True)) / deviation


def metrics(y, pred):
    conf = confusion_matrix(y, pred, labels=CLASSES)
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)),
            "confusion_matrix": conf.tolist(), "class_order": CLASSES.tolist(),
            "per_source_recall": {str(c): float(conf[i, i] / conf[i].sum()) for i, c in enumerate(CLASSES)}}


def fit(config, z, y):
    variance = float(z.var())
    scale = 1 / (z.shape[1] * variance) if variance else 1.0
    multiplier = {"scale": 1.0, "scale_x0.1": 0.1, "scale_x10": 10.0}[config["gamma"]]
    gamma = "scale" if config["gamma"] == "scale" else scale * multiplier
    model = SVC(C=config["C"], kernel="rbf", gamma=gamma, class_weight="balanced",
                decision_function_shape="ovr", break_ties=False, random_state=42)
    model.fit(z, y)
    need(np.isclose(model._gamma, scale * multiplier, rtol=1e-14, atol=0), "Resolved gamma differs")
    return model, float(model._gamma)


def run_fold(index):
    start_cpu, start_wall = time.process_time(), time.monotonic()
    invocation = uuid.uuid4().hex
    attempt_path = OUT / "attempt_logs" / ("fold_%02d_%s.jsonl" % (index, invocation))
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    def event(value):
        with attempt_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"context": index, "invocation": invocation,
                                     "process_cpu_seconds": time.process_time() - start_cpu, **value},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    def recorded_fit(config, z, y, stage, fitting_ids):
        fit_identity = uuid.uuid4().hex
        common = {"fit_identity": fit_identity, "config": config, "stage": stage,
                  "fitting_ids_sha256": obj_sha(fitting_ids), "samples": len(y), "dimension": z.shape[1]}
        event({"status": "FIT_STARTED", **common})
        try:
            result = fit(config, z, y)
            event({"status": "FIT_COMPLETED", "numeric_gamma": result[1], **common})
            return result
        except BaseException as error:
            event({"status": "FIT_FAILED", "error_type": type(error).__name__, "error": str(error), **common})
            raise
    frozen = read(OUT / "frozen_sources.json")
    need(sha(V4 / "train_only_features.npz") == EXPECTED_TRAIN, "TRAIN pack changed after freeze")
    item = frozen["contexts"][index]
    manifest_path = V4 / item["folder"] / "job_manifest.json"
    need(sha(manifest_path) == item["job_manifest_sha256"], "Actual original partition manifest changed")
    final_path = OUT / "folds" / ("fold_%02d.json" % index)
    if final_path.exists():
        done = read(final_path)
        need(done["source_freeze_sha256"] == sha(OUT / "frozen_sources.json"), "Existing fold source freeze differs")
        for path, expected in done["artifacts_sha256"].items():
            need(sha(OUT / path) == expected, "Sealed bridge artifact changed")
        receipt("REUSED_VERIFIED", context=index, actual_new_classifier_fits=0)
        return
    with np.load(V4 / "train_only_features.npz", allow_pickle=False) as data:
        ids = data["record_ids"].astype(str)
        positions = {rid: i for i, rid in enumerate(ids)}
        alpha = data["X_alpha"]
        need(alpha.shape == (999, 612), "Full-alpha dimensionality changed")
    metadata = {row["record_id"]: row for row in csv_rows(V4 / "train_only_split.csv")}
    truth = {rid: row["origin_group_id"] for rid, row in metadata.items()}
    contexts = item["inner_partitions"]
    inner_rows, stages = [], []
    # Row SNV does not estimate any between-specimen quantity. Scalers below
    # are always fitted on the exact saved fitting-ID partition.
    normalized = snv(alpha)
    stages.append({"stage": "loading_and_per_spectrum_SNV", "cpu_seconds": time.process_time() - start_cpu,
                   "wall_seconds": time.monotonic() - start_wall, "classifier_fits": 0,
                   "input_dimension": 612, "between_specimen_statistics": False})
    def matrix(records):
        return normalized[[positions[rid] for rid in records]], np.asarray([truth[rid] for rid in records])
    candidates = [{"C": c, "gamma": g} for c in C_VALUES for g in GAMMAS]
    fits = 0
    with threadpool_limits(limits=1):
        for inner in contexts:
            stage_cpu, stage_wall = time.process_time(), time.monotonic()
            z, y = matrix(inner["fit_record_ids"])
            q, yq = matrix(inner["validation_record_ids"])
            scaler = StandardScaler()
            z, q = scaler.fit_transform(z), scaler.transform(q)
            for config in candidates:
                fit_cpu, fit_wall = time.process_time(), time.monotonic()
                model, numeric_gamma = recorded_fit(config, z, y, "inner_%d" % inner["inner_fold"], inner["fit_record_ids"])
                fits += 1
                pred = model.predict(q)
                score = metrics(yq, pred)
                inner_rows.append({"inner_fold": inner["inner_fold"], "config": config,
                                   "metrics": score, "numeric_gamma": numeric_gamma,
                                   "scaler_fit_ids_sha256": obj_sha(inner["fit_record_ids"]),
                                   "validation_ids_sha256": obj_sha(inner["validation_record_ids"]),
                                   "cpu_seconds": time.process_time() - fit_cpu,
                                   "wall_seconds": time.monotonic() - fit_wall})
            stages.append({"stage": "inner_search", "inner_fold": inner["inner_fold"],
                           "cpu_seconds": time.process_time() - stage_cpu,
                           "wall_seconds": time.monotonic() - stage_wall, "classifier_fits": 12})
        scores = []
        for config in candidates:
            rows = [row for row in inner_rows if row["config"] == config]
            scores.append({"config": config, "mean_inner_metrics": {
                key: float(np.mean([row["metrics"][key] for row in rows]))
                for key in ("macro_f1", "balanced_accuracy", "accuracy")}})
        def order(row):
            m, c = row["mean_inner_metrics"], row["config"]
            return (-m["macro_f1"], -m["balanced_accuracy"], -m["accuracy"], c["C"], GAMMAS.index(c["gamma"]))
        chosen = min(scores, key=order)
        # Lock the selection before accessing any outer labels for evaluation.
        atomic(OUT / "folds" / ("fold_%02d_selection.json" % index), {
            "chosen": chosen, "candidates": scores, "outer_labels_used": False,
            "inner_partitions_sha256": item["inner_partitions_sha256"]})
        stage_cpu, stage_wall = time.process_time(), time.monotonic()
        z, y = matrix(item["source_record_ids"])
        scaler = StandardScaler()
        z = scaler.fit_transform(z)
        model, gamma = recorded_fit(chosen["config"], z, y, "outer_final", item["source_record_ids"])
        fits += 1
        q, yq = matrix(item["outer_validation_record_ids"])
        inference_cpu = time.process_time()
        predictions = model.predict(scaler.transform(q))
        inference_cpu = time.process_time() - inference_cpu
        stages.append({"stage": "outer_fit_and_inference", "cpu_seconds": time.process_time() - stage_cpu,
                       "wall_seconds": time.monotonic() - stage_wall, "classifier_fits": 1,
                       "inference_cpu_seconds": inference_cpu})
    need(fits == 25, "Unexpected bridge fit count")
    model_path = OUT / "models" / ("fold_%02d.joblib" % index)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "SNV_ddof": 1, "SNV_input_dimension": 612,
                 "scaler_fit_ids": item["source_record_ids"], "selected_indices": list(range(612)),
                 "numeric_gamma": gamma, "chosen": chosen}, model_path, compress=3)
    prediction_path = OUT / "predictions" / ("fold_%02d.csv.gz" % index)
    write_csv(prediction_path, [{"record_id": rid, "physical_sample_id": metadata[rid]["physical_sample_id"], "y_true": str(label),
                                "prediction": str(pred), "repeat_seed": item["repeat_seed"],
                                "outer_fold": item["outer_fold"], "method": "fullalpha_snv_svm_bridge"}
                               for rid, label, pred in zip(item["outer_validation_record_ids"], yq, predictions)])
    inner_path = OUT / "folds" / ("fold_%02d_inner.json" % index)
    atomic(inner_path, inner_rows)
    done = {"status": "COMPLETED", "context": index, "repeat_seed": item["repeat_seed"],
            "outer_fold": item["outer_fold"], "source_freeze_sha256": sha(OUT / "frozen_sources.json"),
            "actual_new_classifier_fits": fits, "actual_new_pls_fits": 0, "chosen": chosen,
            "numeric_gamma": gamma, "outer_metrics": metrics(yq, predictions), "stages": stages,
            "cpu_seconds_this_attempt": time.process_time() - start_cpu,
            "wall_seconds_this_attempt": time.monotonic() - start_wall,
            "training_dimension": 612, "SNV_input_dependency": 612,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "model_bytes": model_path.stat().st_size,
            "support_vectors": int(model.n_support_.sum()),
            "outer_train_ids_sha256": item["outer_train_ids_sha256"],
            "outer_test_ids_sha256": item["outer_test_ids_sha256"],
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
            "artifacts_sha256": {str(path.relative_to(OUT)): sha(path) for path in (
                model_path, prediction_path, inner_path, OUT / "folds" / ("fold_%02d_selection.json" % index))}}
    atomic(final_path, done)
    event({"status": "INVOCATION_COMPLETED", "classifier_fits": fits})
    receipt("COMPLETED", context=index, actual_new_classifier_fits=fits, actual_new_pls_fits=0,
            cpu_seconds_this_attempt=done["cpu_seconds_this_attempt"])


def summarize():
    frozen = read(OUT / "frozen_sources.json")
    folds, missing, all_predictions = [], [], []
    for index in range(15):
        path = OUT / "folds" / ("fold_%02d.json" % index)
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        need(result["source_freeze_sha256"] == sha(OUT / "frozen_sources.json"), "Bridge fold source freeze changed")
        folds.append(result)
        all_predictions.extend(csv_rows(OUT / "predictions" / ("fold_%02d.csv.gz" % index)))
    comparisons, fold_effects, errors = [], [], []
    for method in ("alpha_snv20", "alpha_raw20", "fullalpha_shrinkLDA"):
        repeat_effects = []
        for seed in SEEDS:
            bridge = [row for row in all_predictions if int(row["repeat_seed"]) == seed]
            if len(bridge) != 300:
                continue
            need(len({row["record_id"] for row in bridge}) == 300, "Bridge repeat duplicated specimen")
            archived = []
            for item in frozen["contexts"]:
                if item["repeat_seed"] == seed:
                    archived.extend(row for row in csv_rows(V4 / item["folder"] / "validation_predictions.csv.gz")
                                    if row["family"] == method and row["is_primary_k"] in ("1", "True", "true"))
            old = {row["record_id"]: row for row in archived}
            need(len(archived) == len(old) == 300 and set(old) == {row["record_id"] for row in bridge}, "Original V4 comparator pairing differs")
            new = {row["record_id"]: row for row in bridge}
            ids = sorted(old)
            truth = [new[rid]["y_true"] for rid in ids]
            need(all(old[rid]["y_true"] == new[rid]["y_true"] for rid in ids), "Comparator truth differs")
            a, b = metrics(truth, [new[rid]["prediction"] for rid in ids]), metrics(truth, [old[rid]["prediction"] for rid in ids])
            repeat_effects.append({"repeat_seed": seed, "bridge": a, "archived": b,
                                   "difference_pp": {key: 100 * (a[key] - b[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1")}})
            for fold in range(5):
                records = [rid for rid in ids if int(new[rid]["outer_fold"]) == fold]
                aa = metrics([new[rid]["y_true"] for rid in records], [new[rid]["prediction"] for rid in records])
                bb = metrics([old[rid]["y_true"] for rid in records], [old[rid]["prediction"] for rid in records])
                fold_effects.append({"comparator": method, "repeat_seed": seed, "outer_fold": fold,
                                     **{key + "_difference_pp": 100 * (aa[key] - bb[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1")}})
            for rid in ids:
                errors.append({"comparator": method, "repeat_seed": seed, "record_id": rid,
                               "y_true": new[rid]["y_true"], "bridge_prediction": new[rid]["prediction"],
                               "archived_prediction": old[rid]["prediction"],
                               "bridge_correct": int(new[rid]["prediction"] == new[rid]["y_true"]),
                               "archived_correct": int(old[rid]["prediction"] == old[rid]["y_true"])})
        method_fold_effects = [row["accuracy_difference_pp"] for row in fold_effects if row["comparator"] == method]
        comparisons.append({"comparator": method, "complete_repeats": len(repeat_effects), "repeats": repeat_effects,
                            "split_accuracy_win_tie_loss": {
                                "win": sum(delta > 1e-12 for delta in method_fold_effects),
                                "tie": sum(abs(delta) <= 1e-12 for delta in method_fold_effects),
                                "loss": sum(delta < -1e-12 for delta in method_fold_effects)},
                            "mean_difference_pp": {key: float(np.mean([r["difference_pp"][key] for r in repeat_effects]))
                                                   for key in ("accuracy", "balanced_accuracy", "macro_f1")} if repeat_effects else None})
    attempts = []
    for path in sorted((OUT / "attempt_logs").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    attempts.append(json.loads(line))
                except json.JSONDecodeError:
                    attempts.append({"status": "TRUNCATED_INTERRUPTED_EVENT", "path": str(path.relative_to(OUT))})
    attempted = sum(row["status"] == "FIT_STARTED" for row in attempts)
    completed = sum(row["status"] == "FIT_COMPLETED" for row in attempts)
    failed = sum(row["status"] == "FIT_FAILED" for row in attempts)
    status = "COMPLETED" if not missing else "BLOCKED_MISSING_ARTIFACT"
    atomic(OUT / "summary.json", {"status": status, "missing_fold_indices": missing, "protocol": "oldV4-3x5-inner2",
                                  "actual_new_classifier_fits": completed,
                                  "classifier_fit_attempts": attempted, "failed_classifier_fit_attempts": failed,
                                  "interrupted_fit_attempts_without_completion": attempted - completed - failed,
                                  "completed_final_protocol_classifier_fits": sum(row["actual_new_classifier_fits"] for row in folds),
                                  "expected_classifier_fits": 375, "comparisons": comparisons,
                                  "limit": "Internal development bridge on original300 exposed specimens; separate from extended330 repeats; no claim of equal total feature-search cost."})
    write_csv(OUT / "oof_predictions.csv.gz", all_predictions)
    write_csv(OUT / "paired_fold_differences.csv", fold_effects)
    write_csv(OUT / "paired_prediction_errors.csv.gz", errors)
    write_csv(OUT / "outer_parameters.csv", [{"context": row["context"], "repeat_seed": row["repeat_seed"],
                                              "outer_fold": row["outer_fold"], **row["chosen"]["config"],
                                              "numeric_gamma": row["numeric_gamma"], "fits": row["actual_new_classifier_fits"],
                                              "cpu_seconds": row["cpu_seconds_this_attempt"]} for row in folds])
    receipt(status, completed_folds=len(folds), missing_folds=missing,
            actual_new_classifier_fits=completed, classifier_fit_attempts=attempted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=("freeze-source", "summarize"))
    parser.add_argument("--fold", type=int, choices=range(15))
    args = parser.parse_args()
    try:
        if args.phase == "freeze-source":
            freeze_source()
        elif args.phase == "summarize":
            summarize()
        elif args.fold is not None:
            run_fold(args.fold)
        else:
            parser.error("choose freeze-source, --fold INDEX, or summarize")
    except (FileNotFoundError, ValueError, OSError) as error:
        receipt("BLOCKED_MISSING_ARTIFACT", error_type=type(error).__name__, error=str(error),
                actual_new_classifier_fits="see attempt CPU ledger; partial fit count may be unavailable")
        raise


if __name__ == "__main__":
    main()
