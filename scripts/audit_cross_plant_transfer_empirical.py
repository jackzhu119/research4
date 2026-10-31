#!/usr/bin/env python3
"""Gated, small cross-plant neural isolation diagnostics.

The diagnostic is deliberately separate from the full-budget transfer study.
Named Python I/O interception is an auditable API boundary, not an OS sandbox.
Source-label permutations preserve the two historical inner-block memberships.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from herb_spectra.classification import classification_metrics
from herb_spectra.leakage_guard import LeakageError

PROTOCOL_ID = "cross-plant-transfer-outer0-empirical-20261031-v1"
ARMS = ("A", "Aextra", "B", "C")
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
PROFILE_IDS = ("diagnostic2", "diagnostic4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def protocol():
    return {
        "protocol_id": PROTOCOL_ID,
        "outer_fold": 0,
        "seed": 42,
        "arms": list(ARMS),
        "target_classes": list(CLASSES),
        "target_profiles": [{"id": PROFILE_IDS[0], "epochs": 2},
                            {"id": PROFILE_IDS[1], "epochs": 4}],
        "pretrain_steps": 4,
        "pretrain_batch_size": 32,
        "source_observations": 684,
        "query_observations": 171,
        "inner_blocks": 2,
        "inner_source_observations": 342,
        "inner_query_observations": 342,
        "common_profile_rule": "A-only pooled684 source OOF Macro-F1, then Balanced Accuracy, Accuracy, fixed profile ID",
        "source_policy_rule": "At common A profile, switch only if Macro-F1 >= A+.005 and BA/Accuracy >= A (tolerance1e-12); otherwise A",
        "null_permutation_seeds": list(range(1000, 1010)),
        "null_unit": "target labels independently shuffled within the two fixed original inner-query blocks; auxiliary labels unchanged",
        "full_C": "all inner fits, common-profile/policy selection and four final fits rerun; external query truth variant changes only protected file bytes",
        "D": "after fitting, alpha[0,0]+=.01 only; model state and other170 predictions/scores unchanged; no claim of full D retraining",
        "pipeline_budget": {"inner_per_repeat": 16, "outer_per_repeat": 4,
                            "per_repeat": 20, "repeats": 12, "total": 240},
        "pretrained_trajectories": 180,
        "pretrain_optimizer_steps": 720,
        "null_pvalue": "(1 + count(null >= observed)) / 11, resolution1/11",
        "all_12_prediction_locks_before_scoring": True,
        "main_candidates_and_budget_unchanged_by_diagnostic": True,
        "fresh_confirmation": False,
        "scientific_status": "EXPLORATORY_ENGINEERING_TRANSFER_UNKNOWN_PHYSICAL_EQUIVALENCE",
        "truth_guard": "named builtins.open/io.open/Path.open/os.open interception; not an OS sandbox",
    }


@contextmanager
def truth_boundary(paths, events, *, allow_scoring_read=False):
    """Deny named query-truth reads in learning and all named writes."""
    protected = {str(Path(p).resolve()) for p in paths}
    old_builtin, old_io, old_os = builtins.open, io.open, os.open

    def check(file, writing, route):
        if isinstance(file, int):
            return
        try:
            name = str(Path(os.fsdecode(file)).resolve())
        except TypeError:
            return
        if name not in protected:
            return
        if writing or not allow_scoring_read:
            events.append({"event": "QUERY_TRUTH_ACCESS_REJECTED", "path": name,
                           "route": route, "write": bool(writing)})
            raise LeakageError("Query truth is isolated from learning and read-only during scoring")
        events.append({"event": "QUERY_TRUTH_SCORING_READ", "path": name, "route": route})

    def patched_builtin(file, mode="r", *args, **kwargs):
        check(file, any(c in str(mode) for c in "wax+"), "builtins.open")
        return old_builtin(file, mode, *args, **kwargs)

    def patched_io(file, mode="r", *args, **kwargs):
        check(file, any(c in str(mode) for c in "wax+"), "io.open/Path.open")
        return old_io(file, mode, *args, **kwargs)

    def patched_os(file, flags, *args, **kwargs):
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        check(file, writing, "os.open")
        return old_os(file, flags, *args, **kwargs)

    builtins.open, io.open, os.open = patched_builtin, patched_io, patched_os
    try:
        yield
    finally:
        builtins.open, io.open, os.open = old_builtin, old_io, old_os


def strict_query(pack):
    if set(pack) != {"record_ids", "X", "axis"}:
        raise LeakageError("Inference pack may contain only X, axis and routing record IDs")
    ids, values, axis = (np.asarray(pack[key]) for key in ("record_ids", "X", "axis"))
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(set(ids.tolist())) != len(ids):
        raise LeakageError("Unique string routing IDs required")
    if axis.ndim != 1 or axis.dtype.kind not in "iuf" or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise LeakageError("Finite increasing physical frequency axis required")
    if values.dtype.kind not in "iuf" or values.shape != (len(ids), len(axis)) or not np.isfinite(values).all():
        raise LeakageError("Finite numerical spectra required, without metadata features")
    return pack


def validate_inner_blocks(source_ids, labels, blocks):
    ids, y = np.asarray(source_ids), np.asarray(labels)
    if ids.ndim != 1 or len(ids) != len(y) or len(set(ids.tolist())) != len(ids):
        raise LeakageError("Unique paired source IDs and labels required")
    if set(y.tolist()) != set(CLASSES) or len(blocks) != 2:
        raise LeakageError("Seven source classes and exactly two historical blocks required")
    flattened = [str(rid) for block in blocks for rid in block]
    if len(flattened) != len(ids) or len(set(flattened)) != len(flattened) or set(flattened) != set(ids.tolist()):
        raise LeakageError("Inner blocks must cover every current source exactly once")
    lookup = dict(zip(ids.tolist(), y.tolist(), strict=True))
    for block in blocks:
        if set(lookup[str(rid)] for rid in block) != set(CLASSES):
            raise LeakageError("Each fixed block must retain all seven source classes")
    return lookup


def permute_source_labels(source_ids, labels, blocks, seed):
    """Permute only current-source labels; preserve block membership/counts."""
    lookup = validate_inner_blocks(source_ids, labels, blocks)
    rng = np.random.default_rng(seed)
    result = dict(lookup)
    for block in blocks:
        old = [lookup[str(rid)] for rid in block]
        changed = rng.permutation(np.asarray(old)).tolist()
        if sorted(old) != sorted(changed):
            raise LeakageError("Label permutation changed source-block counts")
        result.update(zip([str(rid) for rid in block], changed, strict=True))
    return np.asarray([result[str(rid)] for rid in source_ids], dtype=np.asarray(labels).dtype)


def choose_common_profile_and_policy(metrics):
    """Pure source-only selector, shared epoch budget across all four arms."""
    if set(metrics) != set(ARMS) or any(set(metrics[arm]) != set(PROFILE_IDS) for arm in ARMS):
        raise LeakageError("Complete fixed four-arm/two-profile source metrics required")
    for arm in ARMS:
        for profile in PROFILE_IDS:
            if any(not np.isfinite(metrics[arm][profile][key]) for key in METRICS):
                raise LeakageError("Finite actual source metrics required")
    profile = min(PROFILE_IDS, key=lambda p: tuple(-float(metrics["A"][p][k]) for k in METRICS) + (p,))
    baseline = metrics["A"][profile]
    eligible = [arm for arm in ARMS[1:]
                if metrics[arm][profile]["macro_f1"] + 1e-12 >= baseline["macro_f1"] + .005
                and metrics[arm][profile]["balanced_accuracy"] + 1e-12 >= baseline["balanced_accuracy"]
                and metrics[arm][profile]["accuracy"] + 1e-12 >= baseline["accuracy"]]
    selected = min(eligible, key=lambda a: tuple(-float(metrics[a][profile][k]) for k in METRICS) + (ARMS.index(a),)) if eligible else "A"
    return {"common_profile": profile, "policy_arm": selected,
            "eligible_nonbaseline_arms": eligible, "profile_chosen_by": "A_source_OOF_only"}


def pooled_source_metrics(rows, source_ids):
    """Every source record exactly once per arm/profile; no fold averaging."""
    ids = list(map(str, source_ids))
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for profile in PROFILE_IDS:
            cases = [row for row in rows if row["arm"] == arm and row["profile_id"] == profile]
            lookup = {str(row["record_id"]): row for row in cases}
            if len(cases) != len(ids) or len(lookup) != len(ids) or set(lookup) != set(ids):
                raise LeakageError("Source OOF must cover each current source exactly once")
            y = np.asarray([lookup[rid]["truth"] for rid in ids])
            pred = np.asarray([lookup[rid]["prediction"] for rid in ids])
            result[arm][profile] = classification_metrics(y, pred, CLASSES)
    return result


def verify_frozen(out):
    out = Path(out)
    manifest = read(out / "frozen_manifest.json")
    if manifest.get("protocol_id") != PROTOCOL_ID or read(out / "plan.json") != protocol():
        raise LeakageError("Frozen diagnostic protocol changed")
    for name, expected in manifest["files_sha256"].items():
        if sha(out / name) != expected:
            raise LeakageError("Frozen diagnostic inputs changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise LeakageError("Frozen main or diagnostic code changed")
    for name, expected in manifest.get("main_files_sha256", {}).items():
        if sha(name) != expected:
            raise LeakageError("Main preparation/protocol changed after diagnostic freeze")
    return manifest


def check_release(out, gate_path):
    out = Path(out)
    gate = read(gate_path)
    if gate.get("allow_empirical_training") is not True or gate.get("protocol_id") != PROTOCOL_ID or gate.get("frozen_manifest_sha256") != sha(out / "frozen_manifest.json") or not gate.get("created_at_utc"):
        raise LeakageError("Hash-bound root diagnostic training release required")
    verify_frozen(out)
    return gate


def frozen_source_versions():
    from herb_spectra import cross_plant_transfer as core
    names = set(core.source_versions()) | {
        "scripts/audit_cross_plant_transfer_empirical.py",
        "tests/test_transfer_empirical_controls.py",
        "tests/test_cross_plant_transfer.py",
    }
    return {name: sha(ROOT / name) for name in sorted(names)}


def prepare(out, main_output):
    """Copy frozen outer0 sources/queries; only this packaging stage reads truth."""
    from herb_spectra import cross_plant_transfer as core
    out, main_output = Path(out), Path(main_output).resolve()
    if out.exists():
        raise FileExistsError("Preserve preparation attempts; a new directory is required")
    preparation = core.verify_run(main_output)
    contexts = [c for c in preparation["contexts"] if c["outer_fold"] == 0]
    outer = [c for c in contexts if c["stage"] == "outer"]
    inners = sorted([c for c in contexts if c["stage"] == "inner"], key=lambda c: c["inner_fold"])
    if len(outer) != 1 or len(inners) != 2:
        raise LeakageError("Exactly original outer0 and its two inner contexts required")
    outer = outer[0]
    source = core.read_pack(main_output / outer["source_pack"], True)
    query = core.read_pack(main_output / outer["query_pack"], False)
    if len(source["record_ids"]) != 684 or len(query["record_ids"]) != 171:
        raise LeakageError("Historical outer0 source/query denominator changed")
    blocks = [c["query_ids"] for c in inners]
    source_lookup = validate_inner_blocks(source["record_ids"], source["source_labels"], blocks)
    position = {str(rid): i for i, rid in enumerate(source["record_ids"])}
    for context in inners:
        sp = core.read_pack(main_output / context["source_pack"], True)
        qp = core.read_pack(main_output / context["query_pack"], False)
        for ids, pack in [(context["source_ids"], sp), (context["query_ids"], qp)]:
            if pack["record_ids"].tolist() != ids:
                raise LeakageError("Inner order/members changed")
            ix = [position[str(rid)] for rid in ids]
            if not np.array_equal(pack["X"], source["X"][ix]):
                raise LeakageError("Inner source/query spectra do not match current outer source")
        if sp["source_labels"].tolist() != [source_lookup[str(rid)] for rid in sp["record_ids"]]:
            raise LeakageError("Inner labels disagree with current outer-source labels")
    out.mkdir(parents=True)
    copied, local_contexts = {}, []
    for context in [*inners, outer]:
        local = {k: context[k] for k in ("split_id", "stage", "outer_fold", "inner_fold", "source_ids", "query_ids")}
        for role in ("source", "query"):
            source_path = main_output / context[role + "_pack"]
            destination = out / "inputs" / context["split_id"] / (role + ".npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            local[role + "_pack"] = str(destination.relative_to(out))
            copied[str(destination.relative_to(out))] = sha(destination)
        local_contexts.append(local)
    auxiliary = {}
    for name, pool in preparation["auxiliary"].items():
        source_path = main_output / pool["path"]
        destination = out / "auxiliary" / (name + ".npz")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source_path, destination)
        auxiliary[name] = {"path": str(destination.relative_to(out)), "sha256": sha(destination)}
        copied[str(destination.relative_to(out))] = sha(destination)
    truth_dir = out / "scoring_truth"
    truth_dir.mkdir()
    original = truth_dir / "outer0_original.csv"
    if sha(main_output / outer["truth_path"]) != outer["truth_sha256"]:
        raise LeakageError("Main prepared truth bytes changed before packaging")
    shutil.copyfile(main_output / outer["truth_path"], original)
    with original.open(newline="", encoding="utf-8") as stream:
        truth_rows = list(csv.DictReader(stream))
    truth = {row["record_id"]: row["label"] for row in truth_rows}
    if len(truth) != len(truth_rows) or set(truth) != set(query["record_ids"].tolist()):
        raise LeakageError("Frozen query truth membership mismatch during packaging")
    ordered = [truth[str(rid)] for rid in query["record_ids"]]
    changed = np.random.default_rng(4242).permutation(np.asarray(ordered)).tolist()
    if ordered == changed:
        raise LeakageError("Fixed protected query-label trap must change labels")
    trap = truth_dir / "outer0_permuted_trap.csv"
    with trap.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"record_id": str(rid), "label": label} for rid, label in zip(query["record_ids"], changed, strict=True))
    original.chmod(0o444)
    trap.chmod(0o444)
    for path in (original, trap):
        copied[str(path.relative_to(out))] = sha(path)
    write(out / "plan.json", protocol())
    write(out / "preparation.json", {"contexts": local_contexts, "auxiliary": auxiliary,
        "inner_query_blocks": blocks, "source_ids": source["record_ids"].tolist(),
        "query_ids": query["record_ids"].tolist(), "main_output": str(main_output),
        "main_run_manifest_sha256": sha(main_output / "run_manifest.json"),
        "main_protocol_sha256": sha(main_output / "protocol.json"),
        "main_preparation_sha256": sha(main_output / "preparation.json"),
        "query_original_and_trap_sha256_differ": sha(original) != sha(trap),
        "truth_read_by_preparation_only": True, "fit_operations": 0})
    copied.update({name: sha(out / name) for name in ("plan.json", "preparation.json")})
    write(out / "frozen_manifest.json", {"protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": copied, "source_sha256": frozen_source_versions(),
        "main_files_sha256": {str(main_output / name): sha(main_output / name)
            for name in ("run_manifest.json", "protocol.json", "preparation.json")},
        "actual_research_fit_operations": 0, "root_fit_release_required": True})
    return {"status": "PREPARED_NOT_TRAINED", "frozen_manifest_sha256": sha(out / "frozen_manifest.json")}


def source_subset(source, ids, current_labels):
    """Subset only the allowed outer source; null labels propagate to inner source."""
    lookup = {str(rid): i for i, rid in enumerate(source["record_ids"])}
    labels = dict(zip(source["record_ids"].tolist(), np.asarray(current_labels).tolist(), strict=True))
    if set(ids) - set(lookup):
        raise LeakageError("A requested training row lies outside current outer source")
    indices = [lookup[str(rid)] for rid in ids]
    return {"X": source["X"][indices].copy(), "axis": source["axis"].copy(),
            "record_ids": source["record_ids"][indices].copy(),
            "source_labels": np.asarray([labels[str(rid)] for rid in ids], dtype=source["source_labels"].dtype)}


def diagnostic_profiles():
    from herb_spectra import cross_plant_transfer as core
    base = core.protocol()["profiles"][0]
    return [{**base, "id": row["id"], "epochs": row["epochs"], "pretrain_steps": 4}
            for row in protocol()["target_profiles"]]


def stable_tree(value):
    """Value identity for tensor checkpoints, excluding Python storage addresses."""
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): stable_tree(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [stable_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def object_sha(value):
    return hashlib.sha256(json.dumps(stable_tree(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def learned_signature(fitted, directory):
    from herb_spectra.cross_plant_transfer import state_hash
    meta = fitted.metadata
    selected = ("arm", "profile", "seed", "source_ids", "source_class_counts", "axis_sha256",
        "raw_target_source_fingerprint", "initial_encoder_sha256", "initial_zy_head_sha256",
        "pretrained_encoder_sha256", "reset_encoder_sha256", "BN_before_reset", "BN_after_reset",
        "fresh_target_optimizer_initial_sha256", "target_order_sha256", "fitted_state_sha256",
        "pretrain_steps", "target_steps", "pretrain_pool_steps", "pretrain_loss", "target_epoch_loss",
        "target_class_weights", "source_versions", "auxiliary_identity")
    import gzip
    trace_hash = hashlib.sha256()
    with gzip.open(Path(directory) / "batch_trace.jsonl.gz", "rb") as stream:
        for line in stream:
            trace_hash.update(line)
    return {"model_state_sha256": state_hash(fitted.model),
            "scaler": stable_tree({"mean": fitted.scaler.mean_, "var": fitted.scaler.var_, "scale": fitted.scaler.scale_}),
            "auxiliary_scalers": {name: stable_tree({"mean": scaler.mean_, "var": scaler.var_, "scale": scaler.scale_})
                                   for name, scaler in sorted(fitted.auxiliary_scalers.items())},
            "pretrained_checkpoint_sha256": object_sha(fitted.pretrained_state),
            "stable_training_metadata_sha256": object_sha({key: meta[key] for key in selected}),
            "actual_optimizer_trace_content_sha256": trace_hash.hexdigest(),
            "batch_schedule_sha256": sha(Path(directory) / "batch_schedule.json")}


def fitted_evidence(directory):
    return {name: sha(Path(directory) / name) for name in
            ("pipeline.joblib", "training.json", "batch_trace.jsonl.gz", "batch_schedule.json", "fit_scope.jsonl")}


def fit_one_repeat(out, source, query, auxiliary, preparation, run_name, current_labels, probe_truth):
    from herb_spectra import cross_plant_transfer as core
    out = Path(out)
    directory = out / "fits" / run_name
    directory.mkdir(parents=True, exist_ok=False)
    paths = sorted((out / "scoring_truth").glob("*.csv"))
    events, inner_records, source_rows, signatures = [], [], [], {}
    profiles = diagnostic_profiles()
    all_labels = dict(zip(source["record_ids"].tolist(), np.asarray(current_labels).tolist(), strict=True))
    with truth_boundary(paths, events):
        try:
            Path(probe_truth).read_text()
        except LeakageError:
            pass
        else:
            raise LeakageError("A real protected query-truth read must be refused")
        for context in [c for c in preparation["contexts"] if c["stage"] == "inner"]:
            sp = source_subset(source, context["source_ids"], current_labels)
            qp = strict_query(core.read_pack(out / context["query_pack"], False))
            for arm in ARMS:
                for profile in profiles:
                    model_dir = directory / "inner" / context["split_id"] / arm / profile["id"]
                    fitted = core.fit_pipeline(sp, auxiliary, arm, profile, 42, context["split_id"], model_dir)
                    predicted = core.predict_rows(fitted, qp, {})
                    inference = core.audit_pointwise_inference(fitted.model,
                        {"alpha": fitted.transform(qp)[:7]}, batch_sizes=(1, 3), atol=1e-5, rtol=1e-5)
                    key = f"{context['split_id']}/{arm}/{profile['id']}"
                    signature = learned_signature(fitted, model_dir)
                    signatures[key] = signature
                    record = {"key": key, "arm": arm, "profile_id": profile["id"],
                        "source_ids": sp["record_ids"].tolist(), "query_ids": qp["record_ids"].tolist(),
                        "model_directory": str(model_dir.relative_to(out)),
                        "pipeline_sha256": sha(model_dir / "pipeline.joblib"),
                        "training_sha256": sha(model_dir / "training.json"),
                        "evidence_sha256": fitted_evidence(model_dir),
                        "signature": signature, "inference_audit": inference}
                    inner_records.append(record)
                    source_rows.extend({"arm": arm, "profile_id": profile["id"],
                        "record_id": row["record_id"], "truth": all_labels[row["record_id"]],
                        "prediction": row["y_predicted"],
                        **{key: value for key, value in row.items() if key.startswith("score_")}}
                        for row in predicted)
        metrics = pooled_source_metrics(source_rows, source["record_ids"])
        selection = choose_common_profile_and_policy(metrics)
        core_table = {(arm, profile["id"]): {"arm": arm, "profile": profile,
                         "mean": {key: metrics[arm][profile["id"]][key] for key in METRICS}}
                      for arm in ARMS for profile in profiles}
        core_selection = core.select_shared_profile(core_table, profiles)
        if (core_selection["shared_profile"]["id"], core_selection["policy_arm"]) != (selection["common_profile"], selection["policy_arm"]):
            raise LeakageError("Diagnostic independent selector disagrees with production rule")
        write(directory / "source_selection_lock.json", {"selection": selection,
            "pooled_source_metrics": metrics, "inner_records": inner_records,
            "source_rows": source_rows, "outer_truth_read": False})
        chosen = next(profile for profile in profiles if profile["id"] == selection["common_profile"])
        final_source = source_subset(source, source["record_ids"].tolist(), current_labels)
        finals, fitted_by_arm = {}, {}
        for arm in ARMS:
            model_dir = directory / "outer" / arm / chosen["id"]
            fitted = core.fit_pipeline(final_source, auxiliary, arm, chosen, 42, "outer00.final", model_dir)
            predicted = core.predict_rows(fitted, strict_query(query), {})
            inference = core.audit_pointwise_inference(fitted.model, {"alpha": fitted.transform(query)},
                batch_sizes=(1, 3, 64), atol=1e-5, rtol=1e-5)
            signature = learned_signature(fitted, model_dir)
            signatures[f"outer/{arm}"] = signature
            finals[arm] = {"arm": arm, "profile_id": chosen["id"],
                "model_directory": str(model_dir.relative_to(out)),
                "pipeline_sha256": sha(model_dir / "pipeline.joblib"),
                "training_sha256": sha(model_dir / "training.json"),
                "evidence_sha256": fitted_evidence(model_dir),
                "signature": signature, "predictions": predicted, "inference_audit": inference}
            fitted_by_arm[arm] = fitted
        result = {"run": run_name, "selection": selection,
            "source_selection_lock_sha256": sha(directory / "source_selection_lock.json"),
            "signatures": signatures, "finals": finals, "truth_access_events": events,
            "pipeline_fits": 20, "outer_truth_read": False}
        write(directory / "round_prediction_lock.json", result)
    return result, fitted_by_arm


def execute(out, release):
    from herb_spectra import cross_plant_transfer as core
    out = Path(out)
    gate = check_release(out, release)
    write(out / "root_fit_release_snapshot.json", gate)
    preparation = read(out / "preparation.json")
    outer = next(c for c in preparation["contexts"] if c["stage"] == "outer")
    source, query = core.read_pack(out / outer["source_pack"], True), strict_query(core.read_pack(out / outer["query_pack"], False))
    auxiliary = {name: core.read_pack(out / pool["path"], True) for name, pool in preparation["auxiliary"].items()}
    truth_paths = sorted((out / "scoring_truth").glob("*.csv"))
    original, trap = out / "scoring_truth/outer0_original.csv", out / "scoring_truth/outer0_permuted_trap.csv"
    observed, fitted = fit_one_repeat(out, source, query, auxiliary, preparation, "observed", source["source_labels"], original)
    repeated, _ = fit_one_repeat(out, source, query, auxiliary, preparation, "query_truth_permuted_trap", source["source_labels"], trap)
    invariants = {
        "source_choices_equal": observed["selection"] == repeated["selection"],
        "all20_learned_signatures_equal": observed["signatures"] == repeated["signatures"],
        "all4_final_predictions_and_logits_equal": all(observed["finals"][arm]["predictions"] == repeated["finals"][arm]["predictions"] for arm in ARMS),
        "query_truth_files_really_differ": sha(original) != sha(trap),
    }
    if not all(invariants.values()):
        raise LeakageError("Query truth trap changed source choices, learned objects or final predictions")
    perturb = {key: value.copy() for key, value in query.items()}
    perturb["X"][0, 0] += .01
    d_results, d_events = {}, []
    with truth_boundary(truth_paths, d_events):
        for arm, model in fitted.items():
            baseline = observed["finals"][arm]
            model_dir = out / baseline["model_directory"]
            before = learned_signature(model, model_dir)
            predictions = core.predict_rows(model, perturb, {})
            unchanged_state = before == learned_signature(model, model_dir)
            labels_equal = [r["y_predicted"] for r in predictions[1:]] == [r["y_predicted"] for r in baseline["predictions"][1:]]
            old_scores = np.asarray([[row["score_" + label] for label in CLASSES] for row in baseline["predictions"][1:]])
            new_scores = np.asarray([[row["score_" + label] for label in CLASSES] for row in predictions[1:]])
            scores_equal = bool(np.allclose(old_scores, new_scores, atol=1e-5, rtol=1e-5))
            if not (unchanged_state and labels_equal and scores_equal):
                raise LeakageError("One changed query affected learned state or other query rows")
            d_results[arm] = {"model_state_unchanged": unchanged_state,
                "other170_predictions_unchanged": labels_equal, "other170_scores_unchanged": scores_equal,
                "first_row_prediction_before": baseline["predictions"][0], "first_row_prediction_after": predictions[0],
                "full_source_retraining": False}
    nulls = []
    for index, seed in enumerate(protocol()["null_permutation_seeds"]):
        changed = permute_source_labels(source["record_ids"], source["source_labels"], preparation["inner_query_blocks"], seed)
        result, _ = fit_one_repeat(out, source, query, auxiliary, preparation, f"null_{index:02d}_seed{seed}", changed, original)
        nulls.append({"permutation_seed": seed, "run": result["run"],
                      "round_lock_sha256": sha(out / "fits" / result["run"] / "round_prediction_lock.json")})
        print(json.dumps({"diagnostic_nulls_complete": index + 1, "outer_truth_read": False}), flush=True)
    verify_frozen(out)
    write(out / "diagnostic_prediction_lock.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256": sha(out / "frozen_manifest.json"),
        "root_fit_release_sha256": sha(out / "root_fit_release_snapshot.json"),
        "observed_round_lock_sha256": sha(out / "fits/observed/round_prediction_lock.json"),
        "query_truth_trap_round_lock_sha256": sha(out / "fits/query_truth_permuted_trap/round_prediction_lock.json"),
        "query_truth_invariance": invariants, "D_postfit_single_query": d_results,
        "nulls": nulls, "pipeline_fits": 240, "actual_pretrain_steps": 720,
        "outer_truth_parsed": False, "main_selection_or_budget_changed": False})
    return {"status": "DIAGNOSTIC_PREDICTIONS_LOCKED_NOT_SCORED", "pipeline_fits": 240,
            "prediction_lock_sha256": sha(out / "diagnostic_prediction_lock.json")}


def verify_prediction_lock(out):
    out = Path(out)
    verify_frozen(out)
    lock = read(out / "diagnostic_prediction_lock.json")
    if lock["outer_truth_parsed"] is not False or lock["pipeline_fits"] != 240:
        raise LeakageError("Incomplete diagnostic prediction lock")
    if [item["permutation_seed"] for item in lock["nulls"]] != protocol()["null_permutation_seeds"]:
        raise LeakageError("All ten fixed null permutations must be locked before scoring")
    pairs = [("frozen_manifest_sha256", "frozen_manifest.json"),
             ("root_fit_release_sha256", "root_fit_release_snapshot.json"),
             ("observed_round_lock_sha256", "fits/observed/round_prediction_lock.json"),
             ("query_truth_trap_round_lock_sha256", "fits/query_truth_permuted_trap/round_prediction_lock.json")]
    for key, filename in pairs:
        if lock[key] != sha(out / filename):
            raise LeakageError("Diagnostic lock dependency bytes changed")
    for item in lock["nulls"]:
        if item["round_lock_sha256"] != sha(out / "fits" / item["run"] / "round_prediction_lock.json"):
            raise LeakageError("A null prediction lock changed")
    for run in ["observed", "query_truth_permuted_trap", *[item["run"] for item in lock["nulls"]]]:
        directory = out / "fits" / run
        current = read(directory / "round_prediction_lock.json")
        if current["pipeline_fits"] != 20 or current["outer_truth_read"] is not False or set(current["finals"]) != set(ARMS) or len(current["signatures"]) != 20:
            raise LeakageError("Each repeat must contain all four arms and twenty fitted pipelines")
        if current["source_selection_lock_sha256"] != sha(directory / "source_selection_lock.json"):
            raise LeakageError("A source choice lock changed before scoring")
        source_choice = read(directory / "source_selection_lock.json")
        if len(source_choice["inner_records"]) != 16 or len(source_choice["source_rows"]) != 684 * 4 * 2:
            raise LeakageError("Each repeat must contain all source candidates and pooled684 source OOF rows")
        for record in [*source_choice["inner_records"], *current["finals"].values()]:
            model_dir = out / record["model_directory"]
            if record["evidence_sha256"] != fitted_evidence(model_dir):
                raise LeakageError("Optimizer traces, schedules or learning scopes changed before scoring")
            for key, name in [("pipeline_sha256", "pipeline.joblib"), ("training_sha256", "training.json")]:
                if record[key] != sha(model_dir / name):
                    raise LeakageError("Diagnostic weights or training evidence changed before scoring")
    return lock


def score(out):
    """New scoring process; verifies all prediction/weight/input hashes first."""
    out = Path(out)
    lock = verify_prediction_lock(out)
    events = []
    truth_paths = sorted((out / "scoring_truth").glob("*.csv"))
    with truth_boundary(truth_paths, events, allow_scoring_read=True):
        with (out / "scoring_truth/outer0_original.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    truth = {row["record_id"]: row["label"] for row in rows}
    if len(truth) != 171 or len(rows) != 171:
        raise LeakageError("Diagnostic query truth denominator changed")
    observed = read(out / "fits/observed/round_prediction_lock.json")
    runs = [read(out / "fits" / item["run"] / "round_prediction_lock.json") for item in lock["nulls"]]

    def calculate(round_record, role):
        arm = role if role in ARMS else round_record["selection"]["policy_arm"]
        predictions = round_record["finals"][arm]["predictions"]
        ids = [row["record_id"] for row in predictions]
        if len(ids) != 171 or len(set(ids)) != 171 or set(ids) != set(truth):
            raise LeakageError("Actual diagnostic predictions do not cover all171 query records exactly once")
        return classification_metrics(np.asarray([truth[rid] for rid in ids]),
            np.asarray([row["y_predicted"] for row in predictions]), CLASSES)

    roles = (*ARMS, "source_policy")
    metrics = {role: calculate(observed, role) for role in roles}
    null_values = {role: [calculate(run, role) for run in runs] for role in roles}
    from herb_spectra import cross_plant_transfer as core
    preparation = read(out / "preparation.json")
    outer = next(c for c in preparation["contexts"] if c["stage"] == "outer")
    source = core.read_pack(out / outer["source_pack"], True)
    counts = {label: int(np.sum(source["source_labels"] == label)) for label in CLASSES}
    majority = min(CLASSES, key=lambda label: (-counts[label], label))
    majority_metrics = classification_metrics(np.asarray(list(truth.values())), np.full(171, majority), CLASSES)
    result = {"protocol_id": PROTOCOL_ID, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXPLORATORY_LIMITED_OUTER0_LOW_BUDGET_DIAGNOSTIC_ONLY",
        "observed_metrics": metrics, "majority_source_counts": counts,
        "majority_source_chosen_class": majority, "majority_metrics": majority_metrics,
        "null_metrics": null_values,
        "null_summary": {role: {metric: {"observed": metrics[role][metric],
            "null_mean": float(np.mean([value[metric] for value in null_values[role]])),
            "empirical_tail_plus_one": (1 + sum(value[metric] >= metrics[role][metric] for value in null_values[role])) / 11}
            for metric in METRICS} for role in roles},
        "query_truth_invariance": lock["query_truth_invariance"],
        "D_postfit_single_query": lock["D_postfit_single_query"],
        "pipeline_fits": 240, "score_new_fits": 0,
        "main_selection_or_budget_changed": False,
        "truth_scoring_events": events,
        "diagnostic_prediction_lock_sha256": sha(out / "diagnostic_prediction_lock.json")}
    write(out / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-output", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.main_output is None:
            parser.error("prepare requires --main-output")
        result = prepare(args.output, args.main_output)
    elif args.stage == "execute":
        if args.release is None:
            parser.error("execute requires root --release")
        result = execute(args.output, args.release)
    else:
        result = score(args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

