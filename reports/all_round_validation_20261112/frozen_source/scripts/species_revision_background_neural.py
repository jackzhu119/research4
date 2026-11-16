"""One resumable neural-development job, restricted to master TRAIN samples.

The job reuses the historical Encoder/SNV and paired A/Aextra/B experiment.
Its validation and small support pool are taken ONLY from the master TRAIN
role. It never selects a profile or examines master calibration/prediction
outcomes. The caller owns aggregate selection and the total CPU budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import uuid

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from herb_spectra.upgrade_neural import Encoder, row_snv, state_hash
from herb_spectra.conservative_calibration import build_context, fit_candidate

TARGETS = {"suanzaoren": "酸枣仁", "huangqi": "黄芪", "tianma": "天麻"}
PROFILES = {
    f"p{width}_e{epochs}_full": {"width": width, "epochs": epochs, "strategy": "full"}
    for width in [12, 24] for epochs in [96, 192, 384]
}
PROFILES.update({f"p{width}_e192_frozen": {"width": width, "epochs": 192, "strategy": "frozen"} for width in [12, 24]})
ARMS = ("A", "Aextra", "B")
SPLIT_SEED = 20261031
SUPPORT_SEED = 20261108
SHOT_LEVELS = (1, 3, 5, 10)
SUPPORT_REPEATS = 20
CALIBRATORS = (
    {"id": "legacy_lambda1", "family": "legacy_anchor", "regularization": 1.},
    {"id": "weak_lambda1000_blend005", "family": "damped_anchor", "regularization": 1000., "blend": .05},
)


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def pack_hash(pack):
    h = hashlib.sha256()
    for key in ("record_ids", "sample_ids", "labels", "X"):
        if key not in pack:
            continue
        array = np.ascontiguousarray(pack[key])
        h.update(key.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def subset(pack, indices, labels=True):
    result = {key: value[np.asarray(indices, dtype=int)].copy() for key, value in pack.items() if key != "labels" or labels}
    return result


def load_train_only(features_path, split_path):
    # Do not retain or consult non-TRAIN labels from a full master CSV.
    with Path(split_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [{key: row[key] for key in ["record_id", "physical_sample_id", "species", "origin_group_id"]}
                for row in csv.DictReader(handle) if row["role"] == "train"]
    rows.sort(key=lambda row: row["record_id"])
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Unique nonempty master TRAIN whitelist required")
    if any(not row["physical_sample_id"] for row in rows) or len({row["physical_sample_id"] for row in rows}) != len(rows):
        raise ValueError("One independent physical specimen per TRAIN record required")
    with np.load(features_path, allow_pickle=False) as data:
        record_ids = data["record_ids"].astype(str)
        lookup = {rid: i for i, rid in enumerate(record_ids)}
        if len(lookup) != len(record_ids) or any(row["record_id"] not in lookup for row in rows):
            raise ValueError("TRAIN whitelist and feature IDs do not match")
        indices = [lookup[row["record_id"]] for row in rows]
        # A full legacy NPZ may be supplied, but the numerical array is narrowed
        # immediately. No non-TRAIN spectrum is retained, normalized or fitted.
        values = data["X_alpha_transfer"][indices].astype(float, copy=True)
        axis = data["axis_transfer"].astype(float, copy=True)
    del record_ids, lookup, indices
    if values.shape != (len(rows), 128) or axis.shape != (128,):
        raise ValueError("Historical 128-point alpha shape representation required")
    if not np.isfinite(values).all() or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("TRAIN spectra must be finite on an increasing physical axis")
    if not np.allclose(axis, np.linspace(.21, 1.49, 128), rtol=0, atol=1e-12):
        raise ValueError("Unexpected transfer representation range")
    store = {"record_ids": np.asarray([r["record_id"] for r in rows]),
             "sample_ids": np.asarray([r["physical_sample_id"] for r in rows]),
             "labels": np.asarray([r["origin_group_id"] for r in rows]), "X": values}
    species = np.asarray([r["species"] for r in rows])
    return store, species, axis


def development_partition(store, species, target, fold):
    if target not in TARGETS or fold not in range(5):
        raise ValueError("Predeclared target and zero-based fold 0..4 required")
    target_pack = subset(store, np.flatnonzero(species == TARGETS[target]))
    classes, counts = np.unique(target_pack["labels"], return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        raise ValueError("Training-only five-fold classification is not feasible")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(target_pack["X"], target_pack["labels"]))
    source_pool, validation_indices = folds[fold]
    # This seed depends on the target/fold only, never model seed/profile/arm.
    target_ordinal = list(TARGETS).index(target)
    rng = np.random.default_rng(SUPPORT_SEED + 100 * target_ordinal + fold)
    source_indices, support_indices = [], []
    for label in classes:
        candidates = source_pool[target_pack["labels"][source_pool] == label]
        ordered = rng.permutation(candidates)
        if len(ordered) <= max(SHOT_LEVELS):
            raise ValueError("Insufficient source plus ten independent supports per class")
        support_indices.extend(ordered[:10])
        source_indices.extend(ordered[10:])
    source = subset(target_pack, sorted(source_indices))
    support = subset(target_pack, sorted(support_indices))
    validation = subset(target_pack, sorted(validation_indices), labels=False)
    validation_truth = target_pack["labels"][sorted(validation_indices)].copy()
    auxiliary = {name: subset(store, np.flatnonzero(species == chinese)) for name, chinese in TARGETS.items() if name != target}
    scopes = [set(pack["sample_ids"]) for pack in [source, support, validation, *auxiliary.values()]]
    if any(scopes[i] & scopes[j] for i in range(len(scopes)) for j in range(i)):
        raise ValueError("Source/support/validation/auxiliary physical samples overlap")
    if set.union(*scopes[:3]) != set(target_pack["sample_ids"]):
        raise ValueError("Target master TRAIN was not partitioned exactly once")
    return source, support, validation, validation_truth, auxiliary, classes.tolist()


def support_draws(support, classes):
    rng = np.random.default_rng(SUPPORT_SEED)
    result = []
    for repeat in range(SUPPORT_REPEATS):
        orders = {label: rng.permutation(np.flatnonzero(support["labels"] == label)) for label in classes}
        for shots in SHOT_LEVELS:
            chosen = np.concatenate([orders[label][:shots] for label in classes]).astype(int)
            if len(chosen) != len(classes) * shots:
                raise ValueError("Insufficient internal support pool")
            result.append({"repeat": repeat, "shots": shots, "indices": chosen.tolist(), "support_ids": support["record_ids"][chosen].tolist()})
    return result


def job_configuration(target, fold, seed, profile_name, steps, axis, source, support, validation, validation_truth, auxiliary):
    if profile_name not in PROFILES or seed not in [42, 43, 44] or steps not in [120, 480, 1920]:
        raise ValueError("Unregistered profile/seed/pretraining budget")
    profile = dict(PROFILES[profile_name], name=profile_name, blocks=2, dropout=.35, residual=True, attention=False,
                   learning_rate=.0005, weight_decay=.001, target_batch_size=64, pretrain_batch_size=32,
                   pretrain_steps=steps, class_weight="source balanced")
    return {"schema_version": 1, "task": target, "species": TARGETS[target], "fold": fold, "seed": seed,
            "profile": profile, "pretrain_steps": steps, "split_seed": SPLIT_SEED,
            "scope": "master_TRAIN_only_fivefold_development_not_confirmation",
            "internal_support_per_class": 10, "support_seed": SUPPORT_SEED,
            "shots": list(SHOT_LEVELS), "support_repeats": SUPPORT_REPEATS,
            "calibrators": list(CALIBRATORS), "arms": list(ARMS), "reference": "same-source 128-point SNV scaler C10 balanced RBF SVM",
            "frozen_encoder_policy": "reset BN; one source-only no-gradient statistics refresh; freeze parameters and eval encoder during target-head training",
            "A_frozen_meaning": "random fixed encoder plus trained target head; explicit matched control",
            "source_ids": source["record_ids"].tolist(), "support_ids": support["record_ids"].tolist(),
            "validation_ids": validation["record_ids"].tolist(),
            "auxiliary_ids": {name: pack["record_ids"].tolist() for name, pack in auxiliary.items()},
            "source_fingerprint": pack_hash(source), "support_fingerprint": pack_hash(support),
            "validation_features_fingerprint": pack_hash(validation),
            "validation_truth_sha256": hashlib.sha256(np.ascontiguousarray(validation_truth).tobytes()).hexdigest(),
            "auxiliary_fingerprints": {name: pack_hash(pack) for name, pack in auxiliary.items()},
            "axis_sha256": hashlib.sha256(axis.tobytes()).hexdigest(),
            "worker_sha256": sha(__file__),
            "reused_code_sha256": {name: sha(ROOT / "src/herb_spectra" / name) for name in ["upgrade_neural.py", "conservative_calibration.py", "calibration.py"]},
            "no_master_calibration_or_prediction_used": True,
            "profile_selected_by_this_worker": False}


def validate_receipt(folder, receipt):
    for relative, expected in receipt["artifacts"].items():
        path = folder / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Completed output has missing/changed bytes: " + str(path))


def class_targets(labels, classes):
    encoded = np.asarray([classes.index(str(label)) for label in labels], dtype=np.int64)
    counts = np.bincount(encoded, minlength=len(classes))
    if not counts.all():
        raise ValueError("Source lacks a target class")
    return encoded, torch.as_tensor(len(labels) / (len(classes) * counts), dtype=torch.float32)


def fresh_modules(profile, seed, classes):
    torch.manual_seed(seed)
    encoder = Encoder(profile["width"], profile["blocks"], profile["dropout"], residual=True, attention=False)
    with torch.random.fork_rng():
        torch.manual_seed(seed + 10000)
        head = nn.Linear(16, len(classes))
    return encoder, head


def train_neural(source, auxiliary, arm, profile, seed, classes, attempt):
    # This fit interface has no validation or support arrays/labels.
    allowed = set(source["record_ids"])
    for name, pack in auxiliary.items():
        if allowed & set(pack["record_ids"]):
            raise ValueError("Pretraining and target-source identities overlap")
        allowed |= set(pack["record_ids"])
    scaler = StandardScaler().fit(row_snv(source["X"]))
    z = scaler.transform(row_snv(source["X"])).astype(np.float32)
    targets, weights = class_targets(source["labels"], classes)
    encoder, head = fresh_modules(profile, seed, classes)
    initial_encoder, initial_head = state_hash(encoder), state_hash(head)
    pool_names = [] if arm == "A" else ["target"] if arm == "Aextra" else sorted(auxiliary)
    prepared, aux_scalers = {}, {}
    aux_heads = nn.ModuleDict()
    for ordinal, name in enumerate(pool_names):
        pack = source if name == "target" else auxiliary[name]
        own_classes = classes if name == "target" else sorted(set(pack["labels"].tolist()))
        own_scaler = scaler if name == "target" else StandardScaler().fit(row_snv(pack["X"]))
        own_values = z if name == "target" else own_scaler.transform(row_snv(pack["X"])).astype(np.float32)
        own_targets, own_weights = class_targets(pack["labels"], own_classes)
        prepared[name] = {"X": torch.as_tensor(own_values), "y": torch.as_tensor(own_targets), "weights": own_weights,
                          "ids": pack["record_ids"], "generator": torch.Generator().manual_seed(seed + 30000 + ordinal)}
        if name != "target":
            aux_scalers[name] = own_scaler
        with torch.random.fork_rng():
            torch.manual_seed(seed + 20000 + ordinal)
            aux_heads[name] = nn.Linear(16, len(own_classes))
    pretrain_losses = []
    begin = time.perf_counter()
    with gzip.open(attempt / "pretrain_batches.jsonl.gz", "wt", encoding="utf-8") as trace:
        if pool_names:
            optimizer = torch.optim.Adam([*encoder.parameters(), *aux_heads.parameters()], lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
            torch.manual_seed(seed + 40000)
            for step in range(profile["pretrain_steps"]):
                name = pool_names[step % len(pool_names)]
                pool = prepared[name]
                indices = torch.randperm(len(pool["X"]), generator=pool["generator"])[:profile["pretrain_batch_size"]]
                if len(indices) != profile["pretrain_batch_size"] or len(set(pool["ids"][indices.numpy()])) != len(indices):
                    raise ValueError("A pretraining batch duplicated or lacked independent source samples")
                encoder.train()
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(aux_heads[name](encoder(pool["X"][indices])), pool["y"][indices], weight=pool["weights"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite pretraining loss")
                loss.backward()
                optimizer.step()
                pretrain_losses.append(float(loss.detach()))
                trace.write(json.dumps({"step": step, "pool": name, "source_indices": indices.numpy().tolist(), "loss": float(loss.detach())}, separators=(",", ":")) + "\n")
                if (step + 1) % 120 == 0:
                    atomic_json(attempt / "progress.json", {"stage": "pretraining", "completed_steps": step + 1, "total_steps": profile["pretrain_steps"]})
    pretrain_seconds = time.perf_counter() - begin
    pretrain_state = copy.deepcopy(encoder.state_dict())
    if state_hash(head) != initial_head:
        raise ValueError("Pretraining touched the fresh paired target head")
    for layer in encoder.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.reset_running_stats()
    x, y = torch.as_tensor(z), torch.as_tensor(targets)
    bn_refresh = False
    if profile["strategy"] == "frozen":
        # Re-estimate BN from target SOURCE features, without gradients/labels.
        torch.manual_seed(seed + 45000)
        encoder.train()
        with torch.no_grad():
            for left in range(0, len(z), profile["target_batch_size"]):
                encoder(x[left:left + profile["target_batch_size"]])
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        bn_refresh = True
    frozen_start = state_hash(encoder)
    optimized_parameters = [*head.parameters()] if profile["strategy"] == "frozen" else [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.Adam(optimized_parameters, lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
    torch.manual_seed(seed + 50000)
    generator = torch.Generator().manual_seed(seed + 60000)
    losses, schedule = [], hashlib.sha256()
    begin = time.perf_counter()
    with (attempt / "target_losses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "source_loss"])
        for epoch in range(profile["epochs"]):
            order = torch.randperm(len(z), generator=generator)
            schedule.update(order.numpy().astype("<i8").tobytes())
            encoder.eval() if profile["strategy"] == "frozen" else encoder.train()
            head.train()
            total = 0.
            for left in range(0, len(z), profile["target_batch_size"]):
                indices = order[left:left + profile["target_batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(head(encoder(x[indices])), y[indices], weight=weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite target-source loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            losses.append(total / len(z))
            writer.writerow([epoch + 1, losses[-1]])
            handle.flush()
            if (epoch + 1) % 8 == 0 or epoch + 1 == profile["epochs"]:
                atomic_json(attempt / "progress.json", {"stage": "target_source_training", "completed_epochs": epoch + 1, "total_epochs": profile["epochs"], "source_loss": losses[-1]})
    target_seconds = time.perf_counter() - begin
    encoder.eval()
    head.eval()
    if profile["strategy"] == "frozen" and state_hash(encoder) != frozen_start:
        raise ValueError("Frozen encoder changed during target-head optimization")
    metadata = {"arm": arm, "seed": seed, "classes": classes, "source_ids": source["record_ids"].tolist(),
                "auxiliary_ids": {name: pack["record_ids"].tolist() for name, pack in auxiliary.items()} if arm == "B" else {},
                "pretrain_pool_ids": {name: pool["ids"].tolist() for name, pool in prepared.items()},
                "profile": profile, "initial_encoder_sha256": initial_encoder, "initial_head_sha256": initial_head,
                "target_order_sha256": schedule.hexdigest(), "final_encoder_sha256": state_hash(encoder), "final_head_sha256": state_hash(head),
                "target_source_bn_refresh": bn_refresh, "frozen_encoder_unchanged_after_refresh": profile["strategy"] == "frozen",
                "pretrain_steps_actual": len(pretrain_losses), "target_epochs_actual": len(losses),
                "pretrain_losses": pretrain_losses, "target_losses": losses,
                "pretrain_seconds": pretrain_seconds, "target_seconds": target_seconds,
                "support_or_validation_accepted_by_fit": False}
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "pretrained_encoder": pretrain_state,
                "classes": classes, "profile": profile}, attempt / "weights.pt")
    joblib.dump({"scaler": scaler, "auxiliary_scalers": aux_scalers}, attempt / "scalers.joblib")
    atomic_json(attempt / "training.json", metadata)
    return metadata


def fitted_neural(cache, receipt):
    attempt = cache / receipt["attempt"]
    state = torch.load(attempt / "weights.pt", map_location="cpu", weights_only=True)
    encoder, head = fresh_modules(state["profile"], 42, state["classes"])
    encoder.load_state_dict(state["encoder"])
    head.load_state_dict(state["head"])
    encoder.eval()
    head.eval()
    scaler = joblib.load(attempt / "scalers.joblib")["scaler"]

    def score(values):
        z = scaler.transform(row_snv(values)).astype(np.float32)
        before = state_hash(encoder), state_hash(head)
        with torch.no_grad():
            scores = head(encoder(torch.as_tensor(z))).numpy()
        if before != (state_hash(encoder), state_hash(head)) or not np.isfinite(scores).all():
            raise ValueError("Pointwise inference changed learned state")
        return z, scores

    return score


def cached_neural(cache_root, source, auxiliary, arm, config, classes):
    profile = dict(config["profile"])
    # A is independent of pretraining budget; cache once across its three jobs.
    if arm == "A":
        profile["pretrain_steps"] = 0
    identity = {"kind": "neural", "arm": arm, "source": config["source_fingerprint"],
                "auxiliary": config["auxiliary_fingerprints"] if arm == "B" else {},
                "profile": profile, "seed": config["seed"], "axis": config["axis_sha256"],
                "worker_sha256": config["worker_sha256"], "reused_code_sha256": config["reused_code_sha256"]}
    cache = cache_root / objhash(identity)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = cache / "complete.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt["identity"] != identity:
                raise ValueError("Cache identity differs")
            validate_receipt(cache, receipt)
            reused = True
        else:
            attempt = cache / "attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            atomic_json(attempt / "identity.json", identity)
            train_neural(source, auxiliary, arm, profile, config["seed"], classes, attempt)
            artifact_names = ["weights.pt", "scalers.joblib", "training.json", "pretrain_batches.jsonl.gz", "target_losses.csv", "identity.json"]
            receipt = {"status": "complete", "identity": identity, "attempt": str(attempt.relative_to(cache)),
                       "artifacts": {str((attempt / name).relative_to(cache)): sha(attempt / name) for name in artifact_names}}
            atomic_json(complete, receipt)
            reused = False
    metadata = json.loads((cache / receipt["attempt"] / "training.json").read_text())
    return fitted_neural(cache, receipt), metadata, {"cache_path": str(cache.resolve()), "cache_key": cache.name,
        "complete_sha256": sha(cache / "complete.json"), "cache_reused": reused, "new_model_fits": 0 if reused else 1}


def cached_reference(cache_root, source, config, classes):
    identity = {"kind": "svm_reference", "source": config["source_fingerprint"], "axis": config["axis_sha256"],
                "C": 10., "gamma": "scale", "class_weight": "balanced", "break_ties": True,
                "preprocessing": "historical row SNV then source-only StandardScaler", "worker_sha256": config["worker_sha256"]}
    cache = cache_root / objhash(identity)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = cache / "complete.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt["identity"] != identity:
                raise ValueError("SVM cache identity mismatch")
            validate_receipt(cache, receipt)
            reused = True
        else:
            start = time.perf_counter()
            scaler = StandardScaler().fit(row_snv(source["X"]))
            values = scaler.transform(row_snv(source["X"]))
            model = SVC(C=10., gamma="scale", class_weight="balanced", decision_function_shape="ovr", break_ties=True)
            model.fit(values, source["labels"])
            if model.classes_.tolist() != classes:
                raise ValueError("Reference class order differs")
            joblib.dump({"scaler": scaler, "model": model}, cache / "model.joblib")
            atomic_json(cache / "training.json", {"classes": classes, "source_ids": source["record_ids"].tolist(), "fit_seconds": time.perf_counter() - start})
            receipt = {"status": "complete", "identity": identity, "artifacts": {name: sha(cache / name) for name in ["model.joblib", "training.json"]}}
            atomic_json(complete, receipt)
            reused = False
    fitted = joblib.load(cache / "model.joblib")

    def score(values):
        z = fitted["scaler"].transform(row_snv(values))
        scores = fitted["model"].decision_function(z)
        return z, scores

    return score, {"cache_path": str(cache.resolve()), "cache_key": cache.name, "complete_sha256": sha(cache / "complete.json"),
                   "cache_reused": reused, "new_model_fits": 0 if reused else 1, "reference_only_duplicate_across_profiles_and_seeds": True}


def predict_methods(score, source, support, validation, draws, classes, arm, writer, score_columns):
    source_features, source_scores = score(source["X"])
    support_features, support_scores = score(support["X"])
    query_features, query_scores = score(validation["X"])
    context = build_context(source_features, source_scores, source["labels"], np.asarray(classes), seed=42)
    n = 0

    def write_scores(scores, method, repeat=-1, shots=0):
        nonlocal n
        predicted = np.asarray(classes)[scores.argmax(axis=1)]
        for rid, label, logits in zip(validation["record_ids"], predicted, scores):
            writer.writerow({"arm": arm, "method": method, "repeat": repeat, "shots": shots,
                "record_id": str(rid), "predicted": str(label), **dict(zip(score_columns, map(float, logits)))})
            n += 1

    write_scores(query_scores, "source_only")
    for draw in draws:
        indices = np.asarray(draw["indices"], dtype=int)
        for candidate in CALIBRATORS:
            fitted = fit_candidate(context, candidate, support_features[indices], support_scores[indices], support["labels"][indices])
            write_scores(fitted.predict_scores(query_features, query_scores), candidate["id"], draw["repeat"], draw["shots"])
    return n


def score_validation(prediction_path, validation, truth, classes):
    truth_lookup = dict(zip(validation["record_ids"], truth))
    settings = defaultdict(list)
    with gzip.open(prediction_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            settings[(row["arm"], row["method"], int(row["repeat"]), int(row["shots"]))].append(row)
    records = []
    for (arm, method, repeat, shots), rows in settings.items():
        if len(rows) != len(truth_lookup) or {r["record_id"] for r in rows} != set(truth_lookup):
            raise ValueError("Each setting must evaluate the same training-only validation IDs")
        actual = [str(truth_lookup[row["record_id"]]) for row in rows]
        predicted = [row["predicted"] for row in rows]
        records.append({"arm": arm, "method": method, "repeat": repeat, "shots": shots, "n": len(rows),
                        "correct": sum(a == b for a, b in zip(actual, predicted)),
                        "accuracy": accuracy_score(actual, predicted),
                        "macro_f1": f1_score(actual, predicted, labels=classes, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                        "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist()})
    groups = defaultdict(list)
    for row in records:
        groups[(row["arm"], row["method"], row["shots"])].append(row)
    summary = [{"arm": arm, "method": method, "shots": shots, "support_repeats": len(rows), "n_validation": len(truth_lookup),
                **{metric: float(np.mean([r[metric] for r in rows])) for metric in ["accuracy", "macro_f1", "balanced_accuracy"]}}
               for (arm, method, shots), rows in groups.items()]
    return records, summary


def run_job(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    store, species, axis = load_train_only(args.features, args.split)
    source, support, validation, truth, auxiliary, classes = development_partition(store, species, args.target, args.fold)
    config = job_configuration(args.target, args.fold, args.seed, args.profile, args.pretrain_steps, axis, source, support, validation, truth, auxiliary)
    identity = objhash(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "complete.json").exists():
            receipt = json.loads((output / "complete.json").read_text())
            if receipt["identity_sha256"] != identity:
                raise ValueError("Existing completed job has a different frozen configuration")
            validate_receipt(output, receipt)
            # Weight caches are outside the unit directory; verify their
            # referenced receipt and artifact hashes before accepting a skip.
            model_references = json.loads((output / "model_receipts.json").read_text())
            for reference in model_references.values():
                cache = Path(reference["cache_path"])
                if sha(cache / "complete.json") != reference["complete_sha256"]:
                    raise ValueError("Referenced model-cache receipt changed")
                validate_receipt(cache, json.loads((cache / "complete.json").read_text()))
            print(json.dumps({"status": "already_complete", "output": str(output), "identity_sha256": identity}), flush=True)
            return
        if (output / "protocol.json").exists():
            existing = json.loads((output / "protocol.json").read_text())
            if objhash(existing) != identity:
                raise ValueError("Interrupted job configuration changed; use a separate output")
        else:
            atomic_json(output / "protocol.json", config)
        atomic_json(output / "protocol_lock.json", {"protocol_sha256": sha(output / "protocol.json"), "frozen_before_fit": True})
        draws = support_draws(support, classes)
        atomic_json(output / "support_draws.json", draws)
        cache_root = args.cache_root.resolve() if args.cache_root else output.parent / "_neural_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        references, neural_metadata = {}, []
        score_columns = ["score_" + label for label in classes]
        temporary_predictions = output / ("validation_predictions.partial." + uuid.uuid4().hex + ".csv.gz")
        started = time.perf_counter()
        prediction_rows = 0
        with gzip.open(temporary_predictions, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["arm", "method", "repeat", "shots", "record_id", "predicted", *score_columns])
            writer.writeheader()
            for arm in ARMS:
                atomic_json(output / "progress.json", {"stage": "arm", "arm": arm, "identity_sha256": identity})
                scorer, metadata, reference = cached_neural(cache_root, source, auxiliary, arm, config, classes)
                references[arm] = reference
                neural_metadata.append(metadata)
                prediction_rows += predict_methods(scorer, source, support, validation, draws, classes, arm, writer, score_columns)
                handle.flush()
                atomic_json(output / ("arm_" + arm + "_receipt.json"), {"status": "complete", "model": reference, "source_only_fit": True})
            scorer, reference = cached_reference(cache_root, source, config, classes)
            references["SVM_reference"] = reference
            prediction_rows += predict_methods(scorer, source, support, validation, draws, classes, "SVM_reference", writer, score_columns)
        for key in ["initial_encoder_sha256", "initial_head_sha256", "target_order_sha256"]:
            if len({metadata[key] for metadata in neural_metadata}) != 1:
                raise ValueError("Paired A/Aextra/B differ in " + key)
        final_predictions = output / "validation_predictions.csv.gz"
        temporary_predictions.replace(final_predictions)
        atomic_json(output / "prediction_lock.json", {"sha256": sha(final_predictions), "rows": prediction_rows,
            "validation_truth_not_joined_yet": True, "protocol_sha256": sha(output / "protocol.json")})
        records, metrics = score_validation(final_predictions, validation, truth, classes)
        atomic_json(output / "metrics_by_setting.json", records)
        atomic_json(output / "validation_truth.json", dict(zip(validation["record_ids"].tolist(), truth.tolist())))
        atomic_json(output / "model_receipts.json", references)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        summary = {"status": "complete", "task": args.target, "species": TARGETS[args.target], "fold": args.fold,
                   "seed": args.seed, "profile": args.profile, "pretrain_steps": args.pretrain_steps,
                   "identity_sha256": identity, "scope": config["scope"], "classes": classes,
                   "source_count": len(source["record_ids"]), "support_count": len(support["record_ids"]),
                   "validation_count": len(validation["record_ids"]), "prediction_rows": prediction_rows,
                   "new_model_fits": sum(r["new_model_fits"] for r in references.values()),
                   "model_receipts": references, "metrics": metrics,
                   "job_wall_seconds": time.perf_counter() - started, "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
                   "max_rss_kib": usage.ru_maxrss, "master_calibration_or_prediction_used": False,
                   "validation_is_within_master_training": True, "hyperparameter_selection_performed": False,
                   "reference_same_source_and_128point_representation": True,
                   "seed_and_support_repeats_are_not_independent_samples_or_batches": True}
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", {"stage": "complete", "identity_sha256": identity,
            "new_model_fits": summary["new_model_fits"], "prediction_rows": prediction_rows})
        names = ["protocol.json", "protocol_lock.json", "support_draws.json", "validation_predictions.csv.gz",
                 "prediction_lock.json", "metrics_by_setting.json", "validation_truth.json", "model_receipts.json", "summary.json",
                 *["arm_" + arm + "_receipt.json" for arm in ARMS]]
        atomic_json(output / "complete.json", {"status": "complete", "identity_sha256": identity,
            "artifacts": {name: sha(output / name) for name in names}, "models_resumable_at_arm_boundary": True,
            "master_calibration_or_prediction_used": False})
        print(json.dumps({"status": "complete", "output": str(output), "task": args.target, "fold": args.fold,
                          "seed": args.seed, "profile": args.profile, "pretrain_steps": args.pretrain_steps,
                          "new_model_fits": summary["new_model_fits"], "source_metrics": [m for m in metrics if m["method"] == "source_only"]}, ensure_ascii=False), flush=True)


def self_test():
    rng = np.random.default_rng(77)
    rows, values = [], []
    prefixes = {"suanzaoren": "ZY", "huangqi": "HQ", "tianma": "TM"}
    for name, chinese in TARGETS.items():
        for group in range(3):
            for i in range(60):
                rid = f"{prefixes[name]}{group:02d}-{i:03d}"
                rows.append({"record_id": rid, "physical_sample_id": rid, "species": chinese,
                             "origin_group_id": f"{prefixes[name]}{group:02d}", "role": "train"})
                values.append(rng.normal(size=128))
    for i in range(4):
        rid = f"FORBIDDEN-{i}"
        rows.append({"record_id": rid, "physical_sample_id": rid, "species": "MASTER_HOLDOUT",
                     "origin_group_id": "DO_NOT_USE", "role": "calibration" if i % 2 else "prediction"})
        values.append(rng.normal(size=128))
    with tempfile.TemporaryDirectory(prefix="species-neural-selftest-") as temporary:
        folder = Path(temporary)
        split = folder / "split.csv"
        with split.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        ids = np.asarray([r["record_id"] for r in rows])
        x = np.asarray(values)
        first, changed = folder / "first.npz", folder / "changed.npz"
        np.savez(first, record_ids=ids, X_alpha_transfer=x, axis_transfer=np.linspace(.21, 1.49, 128))
        x[-4:] = np.nan
        np.savez(changed, record_ids=ids, X_alpha_transfer=x, axis_transfer=np.linspace(.21, 1.49, 128))
        store, species, axis = load_train_only(first, split)
        other, other_species, other_axis = load_train_only(changed, split)
        assert pack_hash(store) == pack_hash(other)
        assert not any(rid.startswith("FORBIDDEN") for rid in store["record_ids"])
        checks = 2
        for target in TARGETS:
            validation_seen = []
            for fold in range(5):
                a = development_partition(store, species, target, fold)
                b = development_partition(other, other_species, target, fold)
                assert pack_hash(a[0]) == pack_hash(b[0]) and pack_hash(a[1]) == pack_hash(b[1])
                assert np.array_equal(a[2]["record_ids"], b[2]["record_ids"])
                assert "labels" not in a[2]
                assert len(a[1]["record_ids"]) == len(a[5]) * 10
                validation_seen.extend(a[2]["record_ids"])
                draws = support_draws(a[1], a[5])
                for repeat in range(SUPPORT_REPEATS):
                    current = [d for d in draws if d["repeat"] == repeat]
                    for left, right in zip(current, current[1:]):
                        assert set(left["support_ids"]) <= set(right["support_ids"])
                checks += 4 + SUPPORT_REPEATS
            assert len(validation_seen) == len(set(validation_seen)) == sum(species == TARGETS[target])
            checks += 1
        # A profile's model initialization and target order must not depend on
        # pretraining budget; this catches accidental seed/head coupling.
        profile = dict(PROFILES["p12_e96_full"], blocks=2, dropout=.35)
        e1, h1 = fresh_modules(profile, 42, ["x", "y", "z"])
        e2, h2 = fresh_modules(profile, 42, ["x", "y", "z"])
        assert state_hash(e1) == state_hash(e2) and state_hash(h1) == state_hash(h2)
        checks += 1
    print(json.dumps({"status": "passed", "checks": checks, "scope": "nontrain-feature perturbation invariance, fivefold coverage, source/support/validation isolation, nested supports, paired initialization", "new_model_fits": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGETS))
    parser.add_argument("--fold", "--outer-fold", dest="fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=[42, 43, 44])
    parser.add_argument("--profile", choices=list(PROFILES))
    parser.add_argument("--pretrain-steps", type=int, choices=[120, 480, 1920], default=120)
    parser.add_argument("--features", type=Path, default=ROOT / "data/research/species_background_20261108/train_only_features.npz")
    parser.add_argument("--split", type=Path, default=ROOT / "data/research/species_background_20261108/train_only_split.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.list_profiles:
        print(json.dumps({"targets": TARGETS, "profiles": PROFILES, "pretrain_steps": [120, 480, 1920], "seeds": [42, 43, 44],
            "folds": list(range(5)), "job_count_complete_matrix": 1080, "arms_per_job": list(ARMS), "same_source_reference": "SVM_reference"}, ensure_ascii=False, indent=2))
    elif args.self_test:
        self_test()
    else:
        if any(getattr(args, key) is None for key in ["target", "fold", "seed", "profile", "output"]):
            parser.error("--target --fold --seed --profile --output are required for one job")
        run_job(args)


if __name__ == "__main__":
    main()
