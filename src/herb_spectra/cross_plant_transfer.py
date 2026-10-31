"""Paired, source-only external-plant pretraining for exposed jujube spectra.

The public/teacher domains are separate supervised tasks. Their labels never
become ZY labels. This is exploratory shape engineering, not verified physical
calibration or a new independent confirmation cohort.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import builtins
import io
import os
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .leakage_guard import (FitScope, LeakageError, array_fingerprint,
                            audit_pointwise_inference, source_fingerprint)
from .upgrade_neural import Encoder, row_snv, state_hash, metric_record


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "cross-plant-deep-transfer-20261031-v1"
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
ARMS = ("A", "Aextra", "B", "C")
ROLES = (*ARMS, "source_policy")
SEEDS = (42, 43, 44)
AXIS = np.linspace(.21, 1.49, 128)
AXIS_QUANTITY = "frequency_THz__absorption_coefficient_shape_units_unverified"
SOURCES = {
    "zaoren": ("data/multiplant_database/zaoren/aligned_modalities.npz", "6d1a6917b0f97ab431ec91e78621292b04ddc5cba56a02c0c9c641b2d0035647"),
    "huangqi": ("data/multiplant_database/huangqi/alpha.npz", "70a1afe2404c0a78cd4b262bc1e54857ca6bdf9c9ce02d8aee418b280f479040"),
    "tianma": ("data/multiplant_database/tianma/alpha.npz", "44906eba924bab5bd9b06609a9e97863bafe6cb051af26fa5489c801f6b1a63a"),
    "ginseng": ("data/external/public_thz_20261031_v1/ginseng/observations.npz", "b173223e52dd2170c2765190d0ab87df2889f6ba4ff0151d2ba4dee5dc100c3c"),
}
SPLITS = "reports/staged-nested-optimization/2026-10-31/core/splits.json"
SPLITS_SHA = "fcf14231b4dc1a1e15af8c46310ee6f39e71520b76a7b62fe0e176485f6f5cfe"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def objhash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_versions():
    names = ("src/herb_spectra/cross_plant_transfer.py", "scripts/run_cross_plant_transfer.py",
             "src/herb_spectra/upgrade_neural.py", "src/herb_spectra/leakage_guard.py",
             "src/herb_spectra/classification.py")
    return {name: sha(ROOT / name) for name in names}


def protocol():
    base = {"width": 12, "blocks": 2, "dropout": .35, "residual": True,
            "attention": False, "learning_rate": .0005, "weight_decay": .001,
            "class_weight": "balanced", "target_batch_size": 64,
            "pretrain_steps": 120, "pretrain_batch_size": 32}
    return {"protocol_id": RUN_ID, "seeds": list(SEEDS), "arms": list(ARMS),
            "roles": list(ROLES), "classes": list(CLASSES), "axis": AXIS.tolist(),
            "axis_quantity": AXIS_QUANTITY, "profiles": [dict(base, id="p48", epochs=48), dict(base, id="p96", epochs=96)],
            "sources": {k: {"path": v[0], "sha256": v[1]} for k, v in SOURCES.items()},
            "historical_splits": SPLITS, "historical_splits_sha256": SPLITS_SHA,
            "study_status": "EXPLORATORY_ENGINEERING_SHAPE_TRANSFER_UNVERIFIED_PHYSICAL_COMPATIBILITY",
            "external_class_heads": {"huangqi": 4, "tianma": 4, "ginseng": 2},
            "external_label_meaning": {"huangqi": "existing anonymous filename_groups HQ01..HQ04; source-group supervision, not verified geography",
                                       "tianma": "existing anonymous filename_groups TM01..TM04; source-group supervision, not verified geography",
                                       "ginseng": "public species label0American/1Asian; not ZY origin"},
            "C_task_order": ["huangqi", "tianma", "ginseng"],
            "Aextra_pool": "current target source only; fresh temporary seven-class head",
            "common_profile_selection": "A-only; pool both inner queries684 per seed, then three-seed means; MacroF1/BA/Acc/fixedid",
            "policy_guard": {"macro_f1_margin": .005, "balanced_accuracy_not_lower": True,
                             "accuracy_not_lower": True, "tolerance": 1e-12, "fixed_arm_order": ["Aextra", "B", "C"]},
            "source_pipeline_budget": 240, "outer_pipeline_budget": 60,
            "pretrain_trajectories_if_complete": 225, "cache_reuse": False,
            "unknown_root_ancestry_or_units_are_not_scientific_PASS": True,
            "excluded_data": {"fruit": "unknown physical time/THz calibration", "leaves": "plot curves; independent units/labels missing"}}


def interpolate(values, native_axis, target_axis=AXIS):
    values, native_axis, target_axis = np.asarray(values, float), np.asarray(native_axis, float), np.asarray(target_axis, float)
    if values.ndim != 2 or values.shape[1] != len(native_axis) or not np.isfinite(values).all():
        raise LeakageError("Finite spectra paired to native coordinate required")
    if not np.isfinite(native_axis).all() or not np.all(np.diff(native_axis) > 0) or not np.all(np.diff(target_axis) > 0):
        raise LeakageError("Finite strictly increasing coordinates required")
    if target_axis[0] < native_axis[0] or target_axis[-1] > native_axis[-1]:
        raise LeakageError("Extrapolation is forbidden")
    return np.stack([np.interp(target_axis, native_axis, row) for row in values])


def validate_pack(pack, source=False, expected_axis=AXIS):
    allowed = {"X", "record_ids", "axis"} | ({"source_labels"} if source else set())
    if set(pack) != allowed:
        raise LeakageError("Strict source/query schema differs; truth/metadata may not be features")
    x, ids, axis = np.asarray(pack["X"]), np.asarray(pack["record_ids"]), np.asarray(pack["axis"])
    if ids.ndim != 1 or not len(ids) or len(set(ids.tolist())) != len(ids) or ids.dtype.kind not in "US":
        raise LeakageError("Explicit unique string record IDs required")
    if x.ndim != 2 or x.shape != (len(ids), len(expected_axis)) or x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise LeakageError("Finite samples by fixed spectral bins required; no IDs/paths")
    if axis.shape != np.asarray(expected_axis).shape or not np.array_equal(axis, expected_axis):
        raise LeakageError("Frequency-axis or quantity context mismatch")
    if source:
        y = np.asarray(pack["source_labels"])
        if y.shape != (len(ids),) or y.dtype.kind not in "US" or len(np.unique(y)) < 2 or any(not str(v).strip() for v in y):
            raise LeakageError("Paired source labels required")
    return pack


def read_pack(path, source=False):
    with np.load(path, allow_pickle=False) as data:
        pack = {k: data[k] for k in data.files}
    return validate_pack(pack, source)


def csvwrite(path, rows):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle = opener(path, "xt", encoding="utf-8", newline="")
    else:
        handle = path.open("x", encoding="utf-8", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csvread(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def forbid_named_truth(paths):
    """Reject named truth-file I/O while fitting/predicting; not an OS sandbox."""
    forbidden = {str(Path(path).resolve()) for path in paths}
    attempts = []
    old_builtin, old_io, old_os = builtins.open, io.open, os.open

    def guarded(original, kind):
        def invoke(file, *args, **kwargs):
            if not isinstance(file, int):
                try:
                    value = str(Path(file).resolve())
                except TypeError:
                    value = None
                if value in forbidden:
                    attempts.append({"path": value, "hook": kind, "denied": True})
                    raise LeakageError("Named scoring truth I/O denied during fit/inference")
            return original(file, *args, **kwargs)
        return invoke

    with patch("builtins.open", guarded(old_builtin, "builtins.open")), patch("io.open", guarded(old_io, "io.open")), patch("os.open", guarded(old_os, "os.open")):
        yield attempts


def prepare(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("Preserve attempts; new preparation needs a new directory")
    p = protocol()
    if sha(ROOT / SPLITS) != SPLITS_SHA:
        raise LeakageError("Historical fixed memberships changed")
    domains = {}
    for name, (relative, expected) in SOURCES.items():
        if sha(ROOT / relative) != expected:
            raise LeakageError("Frozen canonical data changed: " + name)
        with np.load(ROOT / relative, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if name == "zaoren":
            x, axis, labels = data["X_alpha"], data["axis_alpha"], data["y"]
        elif name == "ginseng":
            x, axis, labels = data["X_alpha"], data["axis"], data["y"]
        else:
            x, axis, labels = data["X"], data["axis"], data["filename_groups"]
        identifiers = data["record_ids"].astype(str)
        if name != "zaoren":
            identifiers = np.asarray([name + "::" + rid for rid in identifiers])
        domains[name] = validate_pack({"X": interpolate(x, axis), "axis": AXIS.copy(),
                                      "record_ids": identifiers, "source_labels": labels.astype(str)}, True)
    if len(domains["zaoren"]["record_ids"]) != 855 or tuple(np.unique(domains["zaoren"]["source_labels"])) != CLASSES:
        raise LeakageError("The seven-class855 canonical target changed")
    flat = [str(v) for pool in domains.values() for v in pool["record_ids"]]
    if len(flat) != len(set(flat)):
        raise LeakageError("Cross-domain technical IDs collide")
    out.mkdir(parents=True)
    (out / "auxiliary").mkdir()
    auxiliary = {}
    for name in ("huangqi", "tianma", "ginseng"):
        path = out / "auxiliary" / (name + ".npz")
        np.savez_compressed(path, **domains[name])
        auxiliary[name] = {"path": str(path.relative_to(out)), "sha256": sha(path),
                           "record_ids": domains[name]["record_ids"].tolist(),
                           "class_order": np.unique(domains[name]["source_labels"]).tolist(),
                           "unknown_upstream_or_calibration": True}
    target = domains["zaoren"]
    lookup = {rid: i for i, rid in enumerate(target["record_ids"])}
    contexts = []
    for fold in read(ROOT / SPLITS):
        entries = [("inner", j, inner["train_sample_ids"], inner["validation_sample_ids"])
                   for j, inner in enumerate(fold["inner_folds"])]
        entries.append(("outer", None, fold["train_sample_ids"], fold["test_sample_ids"]))
        for stage, inner, source_ids, query_ids in entries:
            if set(source_ids) & set(query_ids) or set(source_ids) | set(query_ids) != set(fold["train_sample_ids"] if stage == "inner" else lookup):
                raise LeakageError("Current-source/query fixed membership mismatch")
            split = f"outer{fold['fold']:02d}." + (f"inner{inner:02d}" if stage == "inner" else "final")
            directory = out / "inputs" / split
            directory.mkdir(parents=True)
            record = {"split_id": split, "stage": stage, "outer_fold": fold["fold"], "inner_fold": inner,
                      "source_ids": source_ids, "query_ids": query_ids}
            for role, identifiers in [("source", source_ids), ("query", query_ids)]:
                indices = [lookup[rid] for rid in identifiers]
                pack = {"X": target["X"][indices], "axis": AXIS.copy(), "record_ids": target["record_ids"][indices]}
                if role == "source":
                    pack["source_labels"] = target["source_labels"][indices]
                path = directory / (role + ".npz")
                np.savez_compressed(path, **validate_pack(pack, role == "source"))
                record[role + "_pack"] = str(path.relative_to(out))
                record[role + "_sha256"] = sha(path)
            truth = out / "scoring_truth" / (split + ".csv")
            truth.parent.mkdir(exist_ok=True)
            csvwrite(truth, [{"record_id": rid, "label": target["source_labels"][lookup[rid]]} for rid in query_ids])
            record.update(truth_path=str(truth.relative_to(out)), truth_sha256=sha(truth))
            contexts.append(record)
    write(out / "protocol.json", p)
    write(out / "preparation.json", {"created_at_utc": utc(), "contexts": contexts, "auxiliary": auxiliary,
                                      "axis_sha256": array_fingerprint(AXIS), "unit_compatibility_verified": False,
                                      "independent_confirmation_established": False})
    write(out / "run_manifest.json", manifest(out))
    return {"target_observations": 855, "external_observations": 850, "contexts": len(contexts), "training_fits": 0}


def manifest(out):
    out = Path(out)
    return {"protocol_sha256": sha(out / "protocol.json"), "preparation_sha256": sha(out / "preparation.json"),
            "source_sha256": source_versions(), "environment": {k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "scikit-learn", "torch", "joblib")}}


def verify_run(out):
    out = Path(out)
    if read(out / "protocol.json") != protocol() or read(out / "run_manifest.json") != manifest(out):
        raise LeakageError("Frozen code, data/protocol metadata or environment changed")
    prep = read(out / "preparation.json")
    for context in prep["contexts"]:
        for role in ("source", "query"):
            if sha(out / context[role + "_pack"]) != context[role + "_sha256"]:
                raise LeakageError("Prepared input byte mismatch")
    for pool in prep["auxiliary"].values():
        if sha(out / pool["path"]) != pool["sha256"]:
            raise LeakageError("Auxiliary bytes changed")
    return prep


class TransferNet(nn.Module):
    def __init__(self, profile, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = Encoder(profile["width"], profile["blocks"], profile["dropout"],
                               residual=profile["residual"], attention=profile["attention"])
        with torch.random.fork_rng():
            torch.manual_seed(seed + 10000)
            self.head = nn.Linear(16, 7)

    def forward(self, inputs):
        return self.head(self.encoder(inputs["alpha"]))


@dataclass
class FittedTransfer:
    model: TransferNet
    scaler: StandardScaler
    auxiliary_scalers: dict
    pretrained_state: dict
    metadata: dict

    def transform(self, query):
        validate_pack(query, False)
        return self.scaler.transform(row_snv(query["X"])).astype(np.float32)

    def predict_scores(self, query):
        values = self.transform(query)
        self.model.eval()
        before = state_hash(self.model)
        with torch.no_grad():
            scores = self.model({"alpha": torch.as_tensor(values)}).cpu().numpy()
        if before != state_hash(self.model) or not np.isfinite(scores).all():
            raise LeakageError("Inference changed learned state or emitted nonfinite logits")
        return scores


def _weights(labels, classes):
    encoded = np.asarray([classes.index(str(v)) for v in labels], dtype=np.int64)
    counts = np.bincount(encoded, minlength=len(classes))
    if not counts.all():
        raise LeakageError("Missing current-source class")
    return encoded, len(labels) / (len(classes) * counts), counts


def fit_pipeline(source, auxiliary, arm, profile, seed, split_id, directory):
    """Train from explicit current-source and auxiliary packs; accepts no query."""
    validate_pack(source, True)
    if arm not in ARMS or tuple(np.unique(source["source_labels"])) != CLASSES:
        raise LeakageError("Unknown arm or incomplete seven-class target source")
    if seed not in SEEDS:
        raise LeakageError("All and only predeclared seeds are allowed")
    pretrain_pools = [] if arm == "A" else ["target"] if arm == "Aextra" else ["ginseng"] if arm == "B" else ["huangqi", "tianma", "ginseng"]
    seen = set(source["record_ids"].tolist())
    for name in pretrain_pools:
        if name == "target":
            continue
        validate_pack(auxiliary[name], True)
        current_ids = set(auxiliary[name]["record_ids"].tolist())
        if seen & current_ids:
            raise LeakageError("External and current-target technical identities overlap")
        if len(np.unique(auxiliary[name]["source_labels"])) != {"ginseng": 2, "huangqi": 4, "tianma": 4}[name]:
            raise LeakageError("External source task class vocabulary changed")
        seen |= current_ids
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    version = objhash(source_versions())
    ids = source["record_ids"]
    scope = FitScope(ids, split_id + "." + arm + ".target", RUN_ID, directory / "fit_scope.jsonl", code_version=version)
    start = time.perf_counter()
    normalized = row_snv(source["X"])
    scaler = scope.fit(StandardScaler(), normalized, sample_ids=ids, component="target_source_scaler")
    z = scaler.transform(normalized).astype(np.float32)
    encoded, weight, counts = _weights(source["source_labels"], list(CLASSES))
    model = TransferNet(profile, seed)
    initial_encoder, initial_head = state_hash(model.encoder), state_hash(model.head)
    prepared, aux_scalers, heads, pool_scopes = {}, {}, nn.ModuleDict(), {}
    for ordinal, name in enumerate(pretrain_pools):
        pool = source if name == "target" else validate_pack(auxiliary[name], True)
        own_classes = list(CLASSES) if name == "target" else np.unique(pool["source_labels"]).tolist()
        current = FitScope(pool["record_ids"], split_id + "." + arm + ".pretrain." + name,
                           RUN_ID + "." + name, directory / "fit_scope.jsonl", code_version=version)
        if len(pool["record_ids"]) < profile["pretrain_batch_size"]:
            raise LeakageError("Cannot satisfy fixed batch without duplicating samples")
        if name == "target":
            pre_scaler, values = scaler, z
        else:
            x = row_snv(pool["X"])
            pre_scaler = current.fit(StandardScaler(), x, sample_ids=pool["record_ids"], component="external_source_scaler")
            values = pre_scaler.transform(x).astype(np.float32)
            aux_scalers[name] = pre_scaler
        target, class_weights, _ = _weights(pool["source_labels"], own_classes)
        prepared[name] = {"pack": pool, "x": values, "y": target, "weights": torch.as_tensor(class_weights, dtype=torch.float32),
                          "classes": own_classes, "generator": torch.Generator().manual_seed(seed + 30000 + ordinal)}
        pool_scopes[name] = current
        with torch.random.fork_rng():
            torch.manual_seed(seed + 20000 + ordinal)
            heads[name] = nn.Linear(16, len(own_classes))
    pre_losses, pre_schedule, target_schedule = [], [], []
    pretrain_begin = time.perf_counter()
    pretrain_checkpoint = {"initial_encoder_sha256": initial_encoder, "initial_zy_head_sha256": initial_head}
    with gzip.open(directory / "batch_trace.jsonl.gz", "xt", encoding="utf-8") as trace:
        def batch_event(stage, name, step, indices, pool, values, targets, loss):
            actual_ids = pool["record_ids"][indices].tolist()
            allowed_ids = pool["record_ids"].tolist()
            if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) - set(allowed_ids):
                raise LeakageError("Actual optimizer batch escaped current source")
            trace.write(json.dumps({"stage": stage, "step": int(step), "head": name,
                                    "fit_sample_ids": actual_ids, "allowed_source_ids_sha256": objhash(allowed_ids),
                                    "X_fingerprint": array_fingerprint(values), "y_fingerprint": array_fingerprint(targets),
                                    "loss": float(loss)}, separators=(",", ":"), allow_nan=False) + "\n")

        if pretrain_pools:
            optimizer = torch.optim.Adam([*model.encoder.parameters(), *heads.parameters()],
                                         lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
            torch.manual_seed(seed + 40000)
            with ExitStack() as learning_scopes:
                for name in pretrain_pools:
                    pool = prepared[name]
                    learning_scopes.enter_context(pool_scopes[name].operation("source_head_pretraining_trajectory",
                        pool["pack"]["record_ids"], X=pool["x"], y=pool["y"]))
                for step in range(profile["pretrain_steps"]):
                    name = pretrain_pools[step % len(pretrain_pools)]
                    pool = prepared[name]
                    selected = torch.randperm(len(pool["x"]), generator=pool["generator"])[:profile["pretrain_batch_size"]].numpy()
                    model.encoder.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = heads[name](model.encoder(torch.as_tensor(pool["x"][selected])))
                    loss = nn.functional.cross_entropy(logits, torch.as_tensor(pool["y"][selected]), weight=pool["weights"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite external-source loss")
                    # Actual optimizer input scope, not merely a declared dataset name.
                    actual_ids = pool["pack"]["record_ids"][selected]
                    if set(actual_ids.tolist()) - set(pool_scopes[name].allowed_ids):
                        raise LeakageError("Forbidden actual pretrain IDs")
                    loss.backward()
                    optimizer.step()
                    batch_event("pretrain", name, step, selected, pool["pack"], pool["x"][selected], pool["y"][selected], loss.detach())
                    pre_losses.append(float(loss.detach()))
                    pre_schedule.append({"pool": name, "indices": selected.tolist()})
            pretrain_checkpoint.update(encoder={k: v.detach().clone() for k, v in model.encoder.state_dict().items()},
                                       heads={k: {n: t.detach().clone() for n, t in head.state_dict().items()} for k, head in heads.items()},
                                       classes={name: prepared[name]["classes"] for name in pretrain_pools})
        pretrain_checkpoint.setdefault("encoder", {k: v.detach().clone() for k, v in model.encoder.state_dict().items()})
        pretrain_checkpoint["fresh_zy_head"] = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
        pretrained_encoder_hash = state_hash(model.encoder)
        before_reset_buffers = {name: value.detach().cpu().numpy().tolist() for name, value in model.encoder.named_buffers()}
        for layer in model.encoder.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.reset_running_stats()
        reset_encoder_hash = state_hash(model.encoder)
        after_reset_buffers = {name: value.detach().cpu().numpy().tolist() for name, value in model.encoder.named_buffers()}
        if state_hash(model.head) != initial_head:
            raise LeakageError("Pretraining modified the paired fresh ZY head")
        pretrain_seconds = time.perf_counter() - pretrain_begin
        optimizer = torch.optim.Adam(model.parameters(), lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
        initial_optimizer_hash = joblib.hash(optimizer.state_dict())
        torch.manual_seed(seed + 50000)
        generator = torch.Generator().manual_seed(seed + 60000)
        target_loss, target_steps = [], 0
        train_begin = time.perf_counter()
        with scope.operation("target_torch_training", ids, X=z, y=encoded) as proof:
            for epoch in range(profile["epochs"]):
                order = torch.randperm(len(z), generator=generator).numpy()
                target_schedule.append(order.tolist())
                total = 0.
                model.train()
                for left in range(0, len(z), profile["target_batch_size"]):
                    selected = order[left:left + profile["target_batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model({"alpha": torch.as_tensor(z[selected])})
                    loss = nn.functional.cross_entropy(logits, torch.as_tensor(encoded[selected]), weight=torch.as_tensor(weight, dtype=torch.float32))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite target-source loss")
                    if set(ids[selected].tolist()) - set(scope.allowed_ids):
                        raise LeakageError("Forbidden actual target IDs")
                    loss.backward()
                    optimizer.step()
                    batch_event("target", "ZY", target_steps, selected, source, z[selected], encoded[selected], loss.detach())
                    target_steps += 1
                    total += float(loss.detach()) * len(selected)
                target_loss.append(total / len(z))
    model.eval()
    metadata = {"created_at_utc": utc(), "arm": arm, "profile": profile, "seed": seed, "split_id": split_id,
                "source_ids": ids.tolist(), "source_class_counts": dict(zip(CLASSES, counts.tolist())),
                "axis_sha256": array_fingerprint(source["axis"]), "axis_quantity": AXIS_QUANTITY,
                "raw_target_source_fingerprint": source_fingerprint(source["X"], source["source_labels"]),
                "initial_encoder_sha256": initial_encoder, "initial_zy_head_sha256": initial_head,
                "pretrained_encoder_sha256": pretrained_encoder_hash, "reset_encoder_sha256": reset_encoder_hash,
                "BN_before_reset": before_reset_buffers, "BN_after_reset": after_reset_buffers,
                "fresh_target_optimizer_initial_sha256": initial_optimizer_hash,
                "target_order_sha256": objhash(target_schedule), "fitted_state_sha256": state_hash(model),
                "pretrain_steps": len(pre_schedule), "target_steps": target_steps,
                "pretrain_pool_steps": dict(Counter(step["pool"] for step in pre_schedule)),
                "pretrain_loss": pre_losses, "target_epoch_loss": target_loss,
                "target_class_weights": weight.tolist(), "pretrain_seconds": pretrain_seconds,
                "target_training_seconds": time.perf_counter() - train_begin, "elapsed_seconds": time.perf_counter() - start,
                "target_model_parameters": sum(v.numel() for v in model.parameters()),
                "source_versions": source_versions(), "query_received_by_training": False,
                "pretrain_cache_reuse": False, "OS_file_isolation": False,
                "auxiliary_identity": {name: {"record_ids": prepared[name]["pack"]["record_ids"].tolist(),
                    "classes": prepared[name]["classes"], "axis_sha256": array_fingerprint(prepared[name]["pack"]["axis"]),
                    "data_fingerprint": source_fingerprint(prepared[name]["pack"]["X"], prepared[name]["pack"]["source_labels"])}
                    for name in pretrain_pools}}
    fitted = FittedTransfer(model, scaler, aux_scalers, pretrain_checkpoint, metadata)
    fitted._leakage_fit_provenance = proof
    joblib.dump(fitted, directory / "pipeline.joblib", compress=3)
    write(directory / "batch_schedule.json", {"pretrain": pre_schedule, "target_epochs": target_schedule})
    metadata["pipeline_sha256"] = sha(directory / "pipeline.joblib")
    metadata["batch_trace_sha256"] = sha(directory / "batch_trace.jsonl.gz")
    metadata["batch_schedule_sha256"] = sha(directory / "batch_schedule.json")
    metadata["fit_scope_sha256"] = sha(directory / "fit_scope.jsonl")
    write(directory / "training.json", metadata)
    return fitted


def predict_rows(fitted, query, common):
    scores = fitted.predict_scores(query)
    predictions = np.asarray(CLASSES)[scores.argmax(1)]
    return [{**common, "record_id": str(rid), "y_predicted": str(pred),
             **{"score_" + label: float(value) for label, value in zip(CLASSES, row)}}
            for rid, pred, row in zip(query["record_ids"], predictions, scores)]


def check_pipeline_evidence(directory):
    directory = Path(directory)
    metadata = read(directory / "training.json")
    for key, filename in [("pipeline_sha256", "pipeline.joblib"), ("batch_trace_sha256", "batch_trace.jsonl.gz"),
                          ("batch_schedule_sha256", "batch_schedule.json"), ("fit_scope_sha256", "fit_scope.jsonl")]:
        if metadata[key] != sha(directory / filename):
            raise LeakageError("Stored learned checkpoint or actual optimizer trace changed")
    return metadata


def _truth(path):
    rows = csvread(path)
    result = {row["record_id"]: row["label"] for row in rows}
    if len(result) != len(rows):
        raise LeakageError("Duplicate scoring truth IDs")
    return result


def metrics(rows, truth):
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(truth):
        raise LeakageError("Pooled prediction denominator mismatch")
    return metric_record([truth[rid] for rid in ids], [row["y_predicted"] for row in rows])


def select_shared_profile(table, profiles=None):
    """Pure source-only shared-epoch selector, also used by small diagnostics."""
    profiles = protocol()["profiles"] if profiles is None else profiles
    rank = lambda row: (-row["mean"]["macro_f1"], -row["mean"]["balanced_accuracy"], -row["mean"]["accuracy"], row["profile"]["id"])
    chosen = min([table["A", p["id"]] for p in profiles], key=rank)
    selected_profile = chosen["profile"]
    baseline = chosen["mean"]
    eligible = []
    for arm in ARMS[1:]:
        candidate = table[arm, selected_profile["id"]]
        m = candidate["mean"]
        if m["macro_f1"] + 1e-12 >= baseline["macro_f1"] + .005 and m["balanced_accuracy"] + 1e-12 >= baseline["balanced_accuracy"] and m["accuracy"] + 1e-12 >= baseline["accuracy"]:
            eligible.append(candidate)
    winner = min(eligible, key=lambda row: (*rank(row)[:3], ARMS.index(row["arm"]))) if eligible else chosen
    return {"shared_profile": selected_profile, "profile_chosen_only_from_A": True, "policy_arm": winner["arm"],
            "source_means_at_shared_profile": {arm: table[arm, selected_profile["id"]]["mean"] for arm in ARMS},
            "outer_used_for_selection": False}


def choices_from_inner(records, rows):
    """Pool both inner query sets per seed, then means; A chooses shared epoch."""
    choices, summaries = [], []
    for outer in range(5):
        table = {}
        for arm in ARMS:
            for profile in protocol()["profiles"]:
                per_seed = []
                for seed in SEEDS:
                    subset = [r for r in rows if int(r["outer_fold"]) == outer and r["arm"] == arm and r["profile"] == profile["id"] and int(r["seed"]) == seed]
                    trials = [r for r in records if r["outer_fold"] == outer and r["arm"] == arm and r["profile"] == profile["id"] and r["seed"] == seed]
                    if len(subset) != 684 or len(trials) != 2:
                        raise LeakageError("Both inner684 source observations and all seeds required")
                    truth = {r["record_id"]: r["y_true"] for r in subset}
                    per_seed.append(metrics(subset, truth))
                summary = {"outer_fold": outer, "arm": arm, "profile": profile,
                           "per_seed": dict(zip(map(str, SEEDS), per_seed)),
                           "mean": {m: float(np.mean([r[m] for r in per_seed])) for m in ("macro_f1", "balanced_accuracy", "accuracy")}}
                summaries.append(summary)
                table[arm, profile["id"]] = summary
        choices.append({"outer_fold": outer, **select_shared_profile(table)})
    return choices, summaries


def _event(out, event, **kwargs):
    with (Path(out) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": event, **kwargs}) + "\n")


def _trial(out, context, arm, profile, seed, auxiliary, outer=False):
    out = Path(out)
    source = read_pack(out / context["source_pack"], True)
    query = read_pack(out / context["query_pack"], False)
    if source["record_ids"].tolist() != context["source_ids"] or query["record_ids"].tolist() != context["query_ids"]:
        raise LeakageError("Pack membership/order changed")
    directory = out / ("outer_models" if outer else "source_models") / context["split_id"] / arm / profile["id"] / f"seed{seed}"
    forbidden = [out / c["truth_path"] for c in read(out / "preparation.json")["contexts"]]
    with forbid_named_truth(forbidden) as denied:
        for path in forbidden:
            try:
                path.open("rb")
            except LeakageError:
                pass
            else:
                raise LeakageError("Named-truth denial probe failed")
        fitted = fit_pipeline(source, auxiliary, arm, profile, seed, context["split_id"], directory)
        transformed = fitted.transform(query)
        audited = transformed if outer else transformed[:7]
        inference = audit_pointwise_inference(fitted.model, {"alpha": audited}, batch_sizes=(1, 3), atol=1e-5, rtol=1e-5)
        common = {"outer_fold": context["outer_fold"], "inner_fold": context["inner_fold"],
                  "arm": arm, "profile": profile["id"], "seed": seed}
        predictions = predict_rows(fitted, query, common)
        csvwrite(directory / "predictions.csv.gz", predictions)
        write(directory / "inference_audit.json", inference)
    write(directory / "truth_access_audit.json", {"named_paths": len(forbidden), "denied_attempts": denied,
                                                "scope": "actual fit, preprocessing and prediction", "OS_sandbox": False})
    record = {**common, "split_id": context["split_id"], "model_directory": str(directory.relative_to(out)),
              "training_sha256": sha(directory / "training.json"), "pipeline_sha256": sha(directory / "pipeline.joblib"),
              "prediction_sha256": sha(directory / "predictions.csv.gz"), "inference_audit_sha256": sha(directory / "inference_audit.json"),
              "truth_access_audit_sha256": sha(directory / "truth_access_audit.json"),
              "elapsed_seconds": fitted.metadata["elapsed_seconds"], "target_steps": fitted.metadata["target_steps"],
              "pretrain_steps": fitted.metadata["pretrain_steps"]}
    return record, predictions


def _check_source_release(out, release):
    verify_run(out)
    m = read(Path(out) / "run_manifest.json")
    if release.get("allow_source_training") is not True or not release.get("created_at_utc") or any(release.get(k) != m[k] for k in ("protocol_sha256", "preparation_sha256", "source_sha256")):
        raise LeakageError("Dated source release must bind full prepared implementation")


def selection(out, release_path):
    out = Path(out)
    release = read(release_path)
    _check_source_release(out, release)
    if (out / "source_release_snapshot.json").exists():
        raise FileExistsError("Preserve actual source attempts")
    write(out / "source_release_snapshot.json", release)
    prep = verify_run(out)
    auxiliary = {name: read_pack(out / pool["path"], True) for name, pool in prep["auxiliary"].items()}
    records, rows = [], []
    _event(out, "source_selection_start")
    for context in [c for c in prep["contexts"] if c["stage"] == "inner"]:
        for arm in ARMS:
            for profile in protocol()["profiles"]:
                for seed in SEEDS:
                    record, predicted = _trial(out, context, arm, profile, seed, auxiliary)
                    if sha(out / context["truth_path"]) != context["truth_sha256"]:
                        raise LeakageError("Inner truth bytes changed before parse")
                    truth = _truth(out / context["truth_path"])
                    record["metrics"] = metrics(predicted, truth)
                    rows.extend({**r, "y_true": truth[r["record_id"]]} for r in predicted)
                    records.append(record)
        _event(out, "inner_context_complete", split=context["split_id"], pipeline_fits=len(records))
        print(json.dumps({"split": context["split_id"], "source_pipelines": len(records), "outer_predictions": False}), flush=True)
    if len(records) != 240:
        raise LeakageError("Incomplete source budget")
    choices, summaries = choices_from_inner(records, rows)
    write(out / "source_records.json", records)
    csvwrite(out / "source_predictions.csv.gz", rows)
    write(out / "candidate_summaries.json", summaries)
    verify_run(out)
    write(out / "selection_lock.json", {"created_at_utc": utc(), "outer_folds": choices,
            "source_pipeline_fits": len(records), "source_release_sha256": sha(out / "source_release_snapshot.json"),
            **{key: sha(out / filename) for key, filename in [("run_manifest_sha256", "run_manifest.json"),
                ("source_records_sha256", "source_records.json"), ("source_predictions_sha256", "source_predictions.csv.gz"),
                ("candidate_summaries_sha256", "candidate_summaries.json")]}, "outer_used_for_selection": False})
    _event(out, "source_selection_locked", selection_lock_sha256=sha(out / "selection_lock.json"))
    return {"source_pipeline_fits": len(records), "selection_lock_sha256": sha(out / "selection_lock.json"), "outer_scored": False}


def verify_selected(out):
    out = Path(out)
    prep = verify_run(out)
    lock = read(out / "selection_lock.json")
    for key, filename in [("source_release_sha256", "source_release_snapshot.json"), ("run_manifest_sha256", "run_manifest.json"),
                          ("source_records_sha256", "source_records.json"), ("source_predictions_sha256", "source_predictions.csv.gz"),
                          ("candidate_summaries_sha256", "candidate_summaries.json")]:
        if lock[key] != sha(out / filename):
            raise LeakageError("Selection dependency bytes changed")
    _check_source_release(out, read(out / "source_release_snapshot.json"))
    for record in read(out / "source_records.json"):
        directory = out / record["model_directory"]
        for key, filename in [("training_sha256", "training.json"), ("pipeline_sha256", "pipeline.joblib"),
                              ("prediction_sha256", "predictions.csv.gz"), ("inference_audit_sha256", "inference_audit.json"),
                              ("truth_access_audit_sha256", "truth_access_audit.json")]:
            if record[key] != sha(directory / filename):
                raise LeakageError("Stored source weight/evidence changed")
        check_pipeline_evidence(directory)
    return prep, lock


def evaluation(out, release_path):
    out = Path(out)
    prep, lock = verify_selected(out)
    release = read(release_path)
    expected = {"selection_lock_sha256": sha(out / "selection_lock.json"), "protocol_sha256": sha(out / "protocol.json"), "run_manifest_sha256": sha(out / "run_manifest.json")}
    if release.get("allow_outer_evaluation") is not True or not release.get("created_at_utc") or any(release.get(k) != v for k, v in expected.items()):
        raise LeakageError("Dated outer release must bind all source choices")
    write(out / "outer_release_snapshot.json", release)
    auxiliary = {name: read_pack(out / pool["path"], True) for name, pool in prep["auxiliary"].items()}
    records, rows, aliases = [], [], []
    _event(out, "outer_prediction_start")
    for context in [c for c in prep["contexts"] if c["stage"] == "outer"]:
        chosen = next(f for f in lock["outer_folds"] if f["outer_fold"] == context["outer_fold"])
        for arm in ARMS:
            for seed in SEEDS:
                record, predicted = _trial(out, context, arm, chosen["shared_profile"], seed, auxiliary, True)
                records.append(record)
                rows.extend({**r, "role": arm} for r in predicted)
                if arm == chosen["policy_arm"]:
                    rows.extend({**r, "role": "source_policy"} for r in predicted)
                    aliases.append({"outer_fold": context["outer_fold"], "seed": seed, "arm": arm,
                                    "model_directory": record["model_directory"], "role": "source_policy"})
    if len(records) != 60 or len(aliases) != 15 or len(rows) != 12825:
        raise LeakageError("Incomplete outer roles/seed/855 denominator")
    write(out / "outer_model_records.json", records)
    write(out / "outer_role_aliases.json", aliases)
    csvwrite(out / "outer_predictions.csv.gz", rows)
    verify_run(out)
    write(out / "prediction_lock.json", {"created_at_utc": utc(), "outer_truth_parsed": False,
        "actual_outer_models": len(records), "alias_cases": len(aliases), "prediction_rows": len(rows),
        **{key: sha(out / filename) for key, filename in [("prediction_sha256", "outer_predictions.csv.gz"),
            ("outer_model_records_sha256", "outer_model_records.json"), ("outer_role_aliases_sha256", "outer_role_aliases.json"),
            ("selection_lock_sha256", "selection_lock.json"), ("outer_release_sha256", "outer_release_snapshot.json"),
            ("run_manifest_sha256", "run_manifest.json"), ("protocol_sha256", "protocol.json")]}})
    _event(out, "outer_predictions_locked", prediction_lock_sha256=sha(out / "prediction_lock.json"))
    return {"actual_outer_models": len(records), "alias_cases": len(aliases), "rows": len(rows), "outer_truth_parsed": False}


def verify_predictions_before_truth(out):
    out = Path(out)
    prep, _ = verify_selected(out)
    lock = read(out / "prediction_lock.json")
    for key, filename in [("prediction_sha256", "outer_predictions.csv.gz"), ("outer_model_records_sha256", "outer_model_records.json"),
                          ("outer_role_aliases_sha256", "outer_role_aliases.json"), ("selection_lock_sha256", "selection_lock.json"),
                          ("outer_release_sha256", "outer_release_snapshot.json"), ("run_manifest_sha256", "run_manifest.json"),
                          ("protocol_sha256", "protocol.json")]:
        if lock[key] != sha(out / filename):
            raise LeakageError("Prediction/scoring dependency changed")
    for record in read(out / "outer_model_records.json"):
        directory = out / record["model_directory"]
        for key, filename in [("training_sha256", "training.json"), ("pipeline_sha256", "pipeline.joblib"),
                              ("prediction_sha256", "predictions.csv.gz"), ("inference_audit_sha256", "inference_audit.json"),
                              ("truth_access_audit_sha256", "truth_access_audit.json")]:
            if record[key] != sha(directory / filename):
                raise LeakageError("Outer fitted evidence changed")
        check_pipeline_evidence(directory)
    for context in prep["contexts"]:
        if context["stage"] == "outer" and sha(out / context["truth_path"]) != context["truth_sha256"]:
            raise LeakageError("All outer truth bytes must be checked before any truth parsing")
    if lock["outer_truth_parsed"] is not False:
        raise LeakageError("Prediction stage parsed outer truth")
    return prep


def score(out):
    out = Path(out)
    prep = verify_predictions_before_truth(out)
    truth = {}
    for context in prep["contexts"]:
        if context["stage"] == "outer":
            current = _truth(out / context["truth_path"])
            if set(truth) & set(current):
                raise LeakageError("One target appears in two outer queries")
            truth.update(current)
    if len(truth) != 855:
        raise LeakageError("Canonical denominator changed")
    rows = csvread(out / "outer_predictions.csv.gz")
    pooled, folds, deltas = [], [], []
    for role in ROLES:
        for seed in SEEDS:
            selected = [r for r in rows if r["role"] == role and int(r["seed"]) == seed]
            metric = metrics(selected, truth)
            pooled.append({"role": role, "seed": seed, "metrics": metric})
            for fold in range(5):
                part = [r for r in selected if int(r["outer_fold"]) == fold]
                folds.append({"role": role, "seed": seed, "outer_fold": fold,
                              "metrics": metrics(part, {r["record_id"]: truth[r["record_id"]] for r in part})})
            if role != "A":
                base = {r["record_id"]: r for r in rows if r["role"] == "A" and int(r["seed"]) == seed}
                corrected = sum(base[r["record_id"]]["y_predicted"] != truth[r["record_id"]] and r["y_predicted"] == truth[r["record_id"]] for r in selected)
                broken = sum(base[r["record_id"]]["y_predicted"] == truth[r["record_id"]] and r["y_predicted"] != truth[r["record_id"]] for r in selected)
                baseline = next(m["metrics"] for m in pooled if m["role"] == "A" and m["seed"] == seed)
                deltas.append({"role": role, "seed": seed, "corrected": corrected, "broken": broken,
                               **{m + "_delta": metric[m] - baseline[m] for m in ("accuracy", "macro_f1", "balanced_accuracy")}})
    summary = [{"role": role, "observations_per_seed": 855, "seeds": list(SEEDS),
                **{m + "_mean": float(np.mean([r["metrics"][m] for r in pooled if r["role"] == role])) for m in ("accuracy", "macro_f1", "balanced_accuracy")},
                **{m + "_sample_sd": float(np.std([r["metrics"][m] for r in pooled if r["role"] == role], ddof=1)) for m in ("accuracy", "macro_f1", "balanced_accuracy")}}
               for role in ROLES]
    write(out / "pooled_metrics_by_seed.json", pooled)
    write(out / "fold_metrics.json", folds)
    write(out / "paired_scratch_deltas.json", deltas)
    write(out / "seed_summary.json", summary)
    csvwrite(out / "scored_outer_predictions.csv.gz", [{**r, "y_true": truth[r["record_id"]]} for r in rows])
    write(out / "scoring_manifest.json", {"created_at_utc": utc(), "new_training_fits": 0, "real_target_observations": 855,
        "three_seeds_are_not_new_specimens": True, "study_status": protocol()["study_status"],
        **{key: sha(out / filename) for key, filename in [("prediction_lock_sha256", "prediction_lock.json"),
            ("prediction_sha256", "outer_predictions.csv.gz"), ("metrics_sha256", "pooled_metrics_by_seed.json"),
            ("summary_sha256", "seed_summary.json"), ("paired_deltas_sha256", "paired_scratch_deltas.json"),
            ("scored_predictions_sha256", "scored_outer_predictions.csv.gz")]}})
    _event(out, "independent_score_complete", new_fits=0)
    return summary
