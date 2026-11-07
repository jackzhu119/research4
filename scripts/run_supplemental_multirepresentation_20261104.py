"""Staged supplemental fusion CLI; numerical work is gated until Q is terminal.

plan/register are standard-library only. run supervises one independent process
per prepare/bind/source/fit/predict/score stage. Every scientific output needs a
healthy closed-stage receipt before the next stage can consume it. This source
has not itself executed a numerical experiment merely by being written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import ctypes
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
LEAF = ROOT / "scripts/draft_supplemental_multirepresentation_20261104.py"
DRAFT = ROOT / "data/research/supplemental_multirepresentation_draft_20261104.json"
OLD_SUPERVISOR = ROOT / "scripts/run_conservative_cross_plant_pilot_20261104_v2.py"
ROLE_ROOT = Path("data/splits/supplemental_spectra_20261104_v1")
STAGES = ("prepare", "bind", "source", "fit", "predict", "score")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TASK_ORDER = ("zaoren_legacy7", "huangqi_legacy4", "tianma_legacy4", "zaoren_expanded10")
_LEAF_INSTANCE = None


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "Artifact must be inside this worktree")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}


def checked(record):
    path = (ROOT / record["path"]).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == record["sha256"],
            "Missing/changed bound artifact: " + str(path))
    return path


def save(path, value, *, mutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not mutable:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
        return
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    os.replace(temporary, path)


def csvwrite(path, rows):
    require(rows, "No empty scientific CSV")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def leaf():
    # Its top level and metadata helpers use only the standard library.
    global _LEAF_INSTANCE
    if _LEAF_INSTANCE is None:
        _LEAF_INSTANCE = load_module(LEAF, "supplemental_fusion_leaf")
    return _LEAF_INSTANCE


def root_authority(path, expected_sha):
    """No imports or raw-array reads before these resource/quantum gates."""
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected_sha, "Root release missing or changed")
    auth = read(path)
    require(auth.get("schema") == "research4-post-quantum-fusion-authority-v1"
            and auth.get("root_execution_authorized") is True and auth.get("quantum_terminal_reviewed") is True,
            "Q terminal review and explicit root release required")
    require(auth.get("implementation_review_passed") is True, "Independent implementation review required before execution")
    limits = auth.get("limits", {})
    require(0 < limits.get("cpu_seconds", 0) <= 3600 and 0 < limits.get("wall_seconds", 0) <= 14400
            and limits.get("threads") == 1 and limits.get("device") == "cpu"
            and limits.get("RSS_bytes") == int(1.5 * 1024**3) and limits.get("AS_bytes") == 10 * 1024**3,
            "Root must freeze the bounded CPU-only resource limits")
    q = auth["quantum_ledger"]; qpath = Path(q["path"]).resolve()
    require(qpath.is_file() and sha(qpath) == q["sha256"], "Q ledger differs from terminal review")
    phases = read(qpath)["phases"]
    require(phases and not any(p.get("status") == "RUNNING" for p in phases), "Quantum payload still active/unresolved")
    require(all(p.get("actual_cpu_seconds") is not None for p in phases), "Unresolved quantum CPU accounting")
    qroot = Path(auth["quantum_worktree_root"]).resolve()
    lock = Path(auth["quantum_global_lock"]).resolve()
    require(qpath.is_relative_to(qroot / "reports") and lock.is_file()
            and lock == qroot / ".runtime/pyscf-v4-global.lock", "Bind the existing actual quantum worktree lock")
    return auth


def code_refs():
    paths = [SELF, LEAF, DRAFT, OLD_SUPERVISOR, ROOT / "src/herb_spectra/upgrade_neural.py",
             ROOT / "src/herb_spectra/leakage_guard.py", ROOT / "src/herb_spectra/classification.py"]
    return [ref(p) for p in paths]


def register(out, authority, authority_sha):
    auth = root_authority(authority, authority_sha)
    require(not out.exists() and out.resolve().is_relative_to(ROOT / "reports"), "Fresh independent reports directory required")
    recipe = read(DRAFT); leaf().metadata_plan(recipe)
    out.mkdir(parents=True)
    registered = dict(recipe)
    registered.update(status="REGISTERED_FIXED_RECIPE_BEFORE_NUMERICAL_PREPARATION", registered_at_utc=utc(),
                      executed_model_fits=0, execution_driver=ref(SELF),
                      resource_limits=auth["limits"], previous_resource_proposal_preserved=True,
                      standard_library_environment={"python": sys.version, "executable": sys.executable,
                                                    "platform": list(os.uname())})
    save(out / "protocol.json", registered)
    save(out / "registration.json", {"status": "REGISTERED_NOT_RUN", "protocol": ref(out / "protocol.json"),
         "authority_path": str(Path(authority).resolve()), "authority_sha256": authority_sha,
         "code_refs": code_refs(), "stages": list(STAGES)})
    created = datetime.now(timezone.utc)
    save(out / "resource_ledger.json", {"status": "NOT_RUN", "created_at_utc": created.isoformat(),
         "deadline_at_utc": (created + timedelta(seconds=auth["limits"]["wall_seconds"])).isoformat(),
         "limits": auth["limits"], "closed_child_CPU_seconds": 0., "closed_supervisor_CPU_seconds_observed": 0., "phases": [],
         "actual_CPU_accounting": "wait4 child CPU plus observed supervisor processCPU through phase close; final metadata-write/return overhead not included"})


def registration(out):
    reg = read(out / "registration.json"); checked(reg["protocol"])
    for record in reg["code_refs"]:
        checked(record)
    auth = root_authority(reg["authority_path"], reg["authority_sha256"])
    return reg, auth


def healthy_stage(out, stage):
    ledger = read(out / "resource_ledger.json")
    candidates = [r for r in ledger["phases"] if r["stage"] == stage]
    require(len(candidates) == 1 and candidates[0]["status"] == "EXITED_HEALTHY", "No unique healthy stage producer")
    row = candidates[0]
    require("closed_stage" in row, "Closed-stage hash has not been registered in the resource ledger")
    path = checked(row["closed_stage"])
    require(path == (out / stage / "closed_stage.json").resolve(), "Stage closure path differs")
    receipt = read(path)
    require(receipt["stage"] == stage and receipt["status"] == "DONE_HEALTHY_CLOSED_STAGE"
            and receipt["exit_code"] == 0 and receipt["stop_reason"] is None,
            "Prerequisite stage is not cleanly closed: " + stage)
    require(receipt["actual_child_CPU_seconds"] == row["actual_child_CPU_seconds"], "Closed CPU receipt differs")
    result = read(checked(receipt["worker_result"]))
    require(result["status"] == "DONE" and result["stage"] == stage
            and result["proof_sha256"] == receipt["supervisor_proof"]["sha256"], "Worker/closure proof mismatch")
    checked(receipt["supervisor_proof"])
    outputs = {k: v for k, v in result.items() if isinstance(v, dict) and {"path", "sha256"} <= set(v)}
    require(outputs, "Stage has no anchored scientific/metadata outputs")
    for record in outputs.values():
        checked(record)
    return {**receipt, "verified_outputs": outputs}


def verify_worker_proof(out, stage, path, expected_sha):
    require(sha(path) == expected_sha, "Changed supervisor proof")
    proof = read(path)
    require(proof["stage"] == stage and Path(proof["output"]).resolve() == out.resolve()
            and proof["driver_sha256"] == sha(SELF) and proof["parent_pid"] == os.getppid(),
            "Worker must run under its exact live supervisor")
    stat = Path("/proc", str(os.getppid()), "stat").read_text()
    fields = stat[stat.rfind(")") + 1:].split()
    require(int(fields[19]) == proof["parent_start_ticks"] and os.getsid(0) == os.getpid(), "Supervisor identity/session changed")
    require(resource.getrlimit(resource.RLIMIT_AS) == (proof["AS_bytes"],) * 2,
            "No actual virtual-address cap")
    require(all(os.environ.get(k) == "1" for k in THREAD_KEYS), "Single-thread environment required before imports")
    registration(out)
    return proof


def runtime_budget(proof):
    """External wait4/proc guard remains authoritative during long native calls."""
    def check():
        require(time.process_time() < proof["remaining_cpu_seconds"] - 10, "Finite cumulative CPU budget reached")
        require(datetime.now(timezone.utc) < datetime.fromisoformat(proof["deadline_at_utc"]), "Run deadline reached")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < proof["RSS_bytes"], "Best-effort RSS limit reached")
    return check


def numeric_dependencies():
    import numpy as np
    return np


def read_npz(record, np, fields=None):
    path = checked(record)
    with np.load(path, allow_pickle=False) as archive:
        names = archive.files if fields is None else fields
        require(set(names) <= set(archive.files), "NPZ interface fields missing")
        return {name: archive[name] for name in names}


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def record_map(ids):
    ids = [str(v) for v in ids]
    require(len(set(ids)) == len(ids), "Duplicate modality IDs")
    return {identifier: i for i, identifier in enumerate(ids)}


def incoming_canonical_map(role_rows, metadata):
    metadata = {r["sample_id"]: r for r in metadata}
    selected, aliases = {}, defaultdict(set)
    for row in role_rows:
        key = (row["species"], row["canonical_model_record_id"])
        aliases[key].update((row["sample_id"], row["entity_id"], row["canonical_model_record_id"]))
        meta = metadata[row["sample_id"]]
        aliases[key].update(json.loads(meta["legacy_match_ids"]))
        if str(meta["canonical_for_model"]).lower() == "true":
            require(key not in selected or selected[key] == row["sample_id"], "Ambiguous canonical incoming modality parent")
            selected[key] = row["sample_id"]
    return selected, aliases


def raw_entry(registry, species, mode):
    entries = [r for r in registry["datasets"] if r["species"] == species and r["modality"] == mode]
    require(len(entries) == 1, "Unique registered modality axis family required")
    return {"path": entries[0]["artifact"], "sha256": entries[0]["sha256"]}


def feature_pack(meta, values, axes, *, role, labels=None):
    pack = {"schema": "research4-compact-multirepresentation-pack-v1", "task_id": meta["task_id"], "role": role,
            "record_ids": meta["record_ids"], "entity_ids": meta["entity_ids"], "identity_aliases": meta["identity_aliases"],
            "values": values, "axis_metadata": axes}
    if labels is not None:
        pack["source_labels"] = labels
    return pack


def save_compact(path, pack, np):
    lf = leaf(); lf.validate_feature_pack(pack, source=pack["role"] == "source")
    data = {"X_" + key: np.asarray(value, dtype=np.float64) for key, value in pack["values"].items()}
    metadata = {k: v for k, v in pack.items() if k != "values"}
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **data)
    return ref(path)


def load_compact(record, np, *, source):
    data = read_npz(record, np)
    require(set(data) == {"metadata_json", *("X_" + m for m in leaf().FEATURES)}, "Compact pack contains unknown fields")
    metadata = json.loads(str(data.pop("metadata_json")))
    pack = {**metadata, "values": {key[2:]: value for key, value in data.items()}}
    leaf().validate_feature_pack(pack, source=source)
    return pack


def subset_source(pack, identifiers, *, validation=False):
    positions = record_map(pack["record_ids"]); require(set(identifiers) <= set(positions), "Fold IDs outside source")
    indices = [positions[i] for i in identifiers]
    meta = {key: [pack[key][i] for i in indices] for key in ("record_ids", "entity_ids", "identity_aliases")}
    meta["task_id"] = pack["task_id"]
    values = {mode: pack["values"][mode][indices] for mode in leaf().FEATURES}
    return feature_pack(meta, values, pack["axis_metadata"], role="inner_validation" if validation else "source",
                        labels=None if validation else [pack["source_labels"][i] for i in indices])


def _prepare(out, proof):
    """Only this stage reads raw modalities/full role metadata; it never fits."""
    np = numeric_dependencies(); lf = leaf(); budget = runtime_budget(proof); budget()
    recipe = read(DRAFT); meta_paths = recipe["metadata_references"]
    task_metadata_path = ROOT / meta_paths["source_fold_manifest"]
    tasks = read(task_metadata_path)["tasks"]
    tasks = {t["id"]: t for t in tasks}; require(set(tasks) == set(TASK_ORDER), "Registered task set changed")
    all_roles = rows(ROOT / meta_paths["role_manifest"])
    incoming_roles = rows(ROOT / meta_paths["incoming_role_map"])
    incoming_metadata = rows(ROOT / meta_paths["sample_metadata"])
    incoming_map, extra_aliases = incoming_canonical_map(incoming_roles, incoming_metadata)
    registry = read(ROOT / meta_paths["merged_registry"])
    directory = out / "prepare"; directory.mkdir(exist_ok=True)
    refs = [ref(ROOT / meta_paths[k]) for k in ("source_fold_manifest", "role_manifest", "incoming_role_map",
                                              "sample_metadata", "merged_registry", "source_registry", "active_supplemental")]
    raw_cache, raw_refs = {}, {}
    def raw(species, mode, legacy):
        key = species, mode, legacy
        if key not in raw_cache:
            record = ref(ROOT / "data/multiplant_database" / species / (mode + ".npz")) if legacy else raw_entry(registry, species, mode)
            data = read_npz(record, np, ("X", "axis", "record_ids"))
            require(data["X"].shape == (len(data["record_ids"]), len(data["axis"]))
                    and np.isfinite(data["X"]).all() and np.isfinite(data["axis"]).all(), "Invalid raw modality shape/values")
            require(np.all(np.diff(data["axis"]) > 0), "Raw axis must increase")
            data["positions"] = record_map(data["record_ids"]); raw_cache[key] = data; raw_refs[key] = record
        return raw_cache[key]
    source_tasks, query_tasks, identity_receipts = [], [], []
    for task_id in TASK_ORDER:
        budget(); task = tasks[task_id]; species = task["species"]
        own = [dict(r) for r in all_roles if r["task_id"] == task_id]
        for row in own:
            row["identity_aliases"] = sorted(extra_aliases.get((species, row["record_id"]), set()))
        source_n, support_n, query_n, class_n = lf.TASKS[task_id]
        role_ids = lf.validate_roles(own, {"source": source_n, "support": support_n, "query": query_n})
        records = {r["record_id"]: r for r in own}
        if task_id != "zaoren_expanded10":
            lf.validate_folds(role_ids["source"], task["inner_folds"])
        else:
            require(task["inner_folds"] == [], "Expanded singleton classes cannot create source CV")
        produced = {}; axis_contract = None
        for role in ("source", "query"):
            original = read_npz(task[role + "_pack"], np)
            if role == "query":
                require(not set(original) & {"y", "labels", "truth", "source_labels"}, "Original query exposes truth")
            identifiers = [str(v) for v in original["record_ids"]]
            require(set(identifiers) == set(role_ids[role]) and len(set(identifiers)) == len(identifiers), "Canonical role IDs changed")
            entities = [str(v) for v in original["entity_ids"]]
            require(entities == [records[i]["entity_id"] for i in identifiers], "Canonical entity mapping differs")
            alpha_axis = np.asarray(original["axis"], dtype=np.float64)
            require(alpha_axis.shape == (612,) and original["X"].shape == (len(identifiers), 612), "Canonical alpha612 required")
            grids = {"alpha": np.linspace(.21, 1.49, 128), "n": np.linspace(.21, 1.49, 128),
                     "alpha_long": np.linspace(.21, 1.49, 1024)}
            values = {mode: [] for mode in lf.FEATURES}; native_axes = {"alpha": alpha_axis, "alpha_long": alpha_axis}
            time_rows = []
            for index, identifier in enumerate(identifiers):
                budget(); alpha = original["X"][index]
                for mode in ("alpha", "alpha_long"):
                    values[mode].append(lf.pointwise_row(mode, lf.resample_row(alpha_axis, alpha, grids[mode])))
                for mode in ("n", "time"):
                    old = species == "zaoren" and records[identifier]["stratum"] == "legacy_exposed"
                    incoming_id = identifier if old else incoming_map.get((species, identifier))
                    require(incoming_id, "No unique explicit incoming/canonical modality mapping")
                    data = raw(species, mode, old)
                    require(incoming_id in data["positions"], "Modality missing canonical/explicit incoming ID")
                    axis = np.asarray(data["axis"], dtype=np.float64)
                    if mode in native_axes:
                        require(np.array_equal(native_axes[mode], axis), "Within-task native modality axes differ")
                    else:
                        native_axes[mode] = axis
                    vector = data["X"][data["positions"][incoming_id]]
                    if mode == "n":
                        values[mode].append(lf.pointwise_row(mode, lf.resample_row(axis, vector, grids[mode])))
                    else:
                        time_rows.append(vector)
            prep_release = {k: True for k in ("root_preparation_authorized", "quantum_terminal_reviewed",
                           "raw_inputs_and_source_axes_frozen", "external_actual_CPU_supervisor_active")}
            time_values, time_grid = lf.prepare_time_rows(native_axes["time"], time_rows, preparation_release=prep_release)
            values["time"] = time_values; grids["time"] = time_grid
            axes = {}
            from hashlib import sha256
            for mode in lf.FEATURES:
                native = np.asarray(native_axes[mode], dtype="<f8")
                axes[mode] = {"native_axis_fingerprint": sha256(native.tobytes()).hexdigest(),
                    "native_axis_fingerprint_definition": "sha256_raw_little_endian_float64_axis_bytes",
                    "points": lf.POINTS[mode], "target_axis": grids[mode].tolist(),
                    "axis_kind": "native_time_coordinate_UNIT_UNVERIFIED" if mode == "time" else "frequency_THz",
                    "axis_unit": "NOT_EXPLICITLY_DECLARED" if mode == "time" else "THz_axis_label",
                    "quantity": "refractive_index" if mode == "n" else "time_domain_amplitude" if mode == "time" else "absorption_coefficient",
                    "value_unit": "NOT_EXTERNALLY_VERIFIED", "pointwise_transform": "identity" if mode == "n" else "l2_no_center" if mode == "time" else "snv_population",
                    "pointwise_prepared": True,
                    "resampling": {"method": "polyphase", "up": 512, "down": 3375, "window": ["kaiser", 5.0],
                                   "padtype": "line", "native_points": 6750} if mode == "time" else {"method": "linear_no_extrapolation"},
                    "derived_from": "registered_native_time6750" if mode == "time" else
                                    "canonical_original_alpha612_not_alpha128" if mode == "alpha_long" else "registered_original"}
            if axis_contract is None:
                axis_contract = axes
            require(axes == axis_contract, "Query/native axis contract differs from source-frozen axes")
            aliases = [sorted(lf.identity_aliases(records[i])) for i in identifiers]
            meta = {"task_id": task_id, "record_ids": identifiers, "entity_ids": entities, "identity_aliases": aliases}
            labels = original["y"].astype(str).tolist() if role == "source" else None
            if labels is not None:
                require(labels == [records[i]["label"] for i in identifiers] and len(set(labels)) == class_n, "Source labels differ")
            pack = feature_pack(meta, values, axes, role=role, labels=labels)
            path = directory / task_id / (role + ".npz"); path.parent.mkdir(exist_ok=True)
            produced[role] = save_compact(path, pack, np)
            produced[role + "_ids"] = identifiers
            if role == "source":
                produced["source_alias_union"] = sorted({v for row in aliases for v in row})
            refs.append(task[role + "_pack"])
        source_tasks.append({"task_id": task_id, "source_pack": produced["source"], "source_ids": produced["source_ids"],
                             "source_alias_union": produced["source_alias_union"], "inner_folds": task["inner_folds"]})
        query_tasks.append({"task_id": task_id, "query_pack": produced["query"], "query_ids": produced["query_ids"],
                            "truth": task["truth"], "strata": task["strata"]})
        identity_receipts.append({"task_id": task_id, "source": source_n, "support_metadata_only": support_n,
                                  "query": query_n, "classes": class_n, "same_original_roles": True,
                                  "numeric_preparation_not_biological_independence_proof": True})
        del original, values, time_rows
    save(directory / "source_manifest.json", {"tasks": source_tasks, "query_features_or_truth": False})
    save(directory / "prediction_manifest.json", {"tasks": query_tasks, "query_truth_contents_read": False})
    refs.extend(raw_refs.values())
    save(directory / "preparation.json", {"status": "NUMERICAL_PREPARATION_DONE_EXPLORATORY", "input_refs": refs,
         "source_manifest": ref(directory / "source_manifest.json"), "prediction_manifest": ref(directory / "prediction_manifest.json"),
         "identity_receipts": identity_receipts, "fixed_time_filter": read(DRAFT)["preprocessing"]["time"]["resampling"],
         "sample_count_increased_by_representations": False, "query_truth_contents_read": False})
    return {"status": "DONE", "preparation": ref(directory / "preparation.json"), "model_fits": 0}


def _bind(out, proof):
    preparation_receipt = healthy_stage(out, "prepare")
    preparation_path = checked(preparation_receipt["verified_outputs"]["preparation"])
    prep = read(preparation_path)
    for record in prep["input_refs"]:
        checked(record)
    source = read(checked(prep["source_manifest"])); query = read(checked(prep["prediction_manifest"]))
    for task in source["tasks"]:
        checked(task["source_pack"])
    for task in query["tasks"]:
        checked(task["query_pack"])  # bytes only; no label file contents or fitting
    directory = out / "bind"; directory.mkdir(exist_ok=True)
    gate = _bind_numerical_gate(out, proof, prep, source, query, directory)
    save(directory / "numerical_preflight.json", gate)
    save(directory / "bound.json", {"status": "ALL_INPUTS_BOUND_BEFORE_SOURCE_SELECTION", "protocol": ref(out / "protocol.json"),
         "preparation": ref(preparation_path), "source_manifest": prep["source_manifest"],
         "prediction_manifest": prep["prediction_manifest"], "numerical_preflight": ref(directory / "numerical_preflight.json"),
         "model_fits": 0, "query_truth_joined": False})
    return {"status": "DONE", "binding": ref(directory / "bound.json"), "model_fits": 0}


def bound_sources(out):
    receipt = healthy_stage(out, "bind")
    bound = read(checked(receipt["verified_outputs"]["binding"])); checked(bound["protocol"]); checked(bound["preparation"])
    gate = read(checked(bound["numerical_preflight"]))
    require(gate["status"] == "PASS_REAL_SOURCE_FORWARD_AND_SYNTHETIC_SIGNAL_CONTROLS", "Actual preparation/forward gates incomplete")
    return bound, read(checked(bound["source_manifest"]))["tasks"]


def runtime_release(out, proof, bound, contexts, *, source_locks=False):
    """Derived from actual checked supervisor, binding and closed-stage receipts."""
    return {"root_execution_authorized": True, "quantum_terminal_reviewed": True,
        "prepared_inputs_numerically_audited": True, "source_scope_review_passed": True,
        "all_inputs_and_code_frozen": True, "external_actual_CPU_supervisor_active": True,
        "threads": 1, "device": "cpu", "protocol_sha256": sha(out / "protocol.json"),
        "prepared_inputs_sha256": bound["source_manifest"]["sha256"], "code_sha256": sha(LEAF),
        "fit_contexts": contexts, "all_task_source_epoch_locks_frozen": source_locks}


def fit_context(task_id, phase, arm, seed, epochs, source_ids, inference_ids):
    return {"task_id": task_id, "phase": phase, "arm": arm, "seed": seed, "epochs": epochs,
            "source_ids": list(source_ids), "inference_ids": list(inference_ids)}


def _source(out, proof):
    np = numeric_dependencies(); lf = leaf(); bound, tasks = bound_sources(out); budget = runtime_budget(proof)
    directory = out / "source"; directory.mkdir(exist_ok=True)
    locks, source_models, all_source_rows = {}, [], []
    for task in tasks:
        task_id = task["task_id"]
        if task_id == "zaoren_expanded10":
            continue
        pack = load_compact(task["source_pack"], np, source=True)
        require(pack["record_ids"] == task["source_ids"], "Frozen source order changed")
        labels = dict(zip(pack["record_ids"], pack["source_labels"])); oof = []
        for fold_id, fold in enumerate(task["inner_folds"]):
            source = subset_source(pack, fold["train_ids"])
            validation = subset_source(pack, fold["valid_ids"], validation=True)
            for seed in lf.SEEDS:
                budget(); split_id = f"{task_id}.inner{fold_id}.seed{seed}"
                context = fit_context(task_id, "source_alpha_cv", "alpha_single", seed, 96,
                                      source["record_ids"], validation["record_ids"])
                release = runtime_release(out, proof, bound, {split_id: context})
                dest = directory / task_id / f"fold{fold_id}" / f"seed{seed}"
                lf.fit_source_leaf(source, validation=validation, arm="alpha_single", epochs=96, seed=seed,
                    directory=dest, split_id=split_id, release=release, budget_check=budget,
                    collect_source_epoch_predictions=True)
                prediction = read_npz(ref(dest / "source_epoch_predictions.npz"), np)
                identifiers = prediction["record_ids"].astype(str).tolist()
                require(identifiers == validation["record_ids"], "Source OOF prediction order changed")
                classes = prediction["class_labels"].astype(str).tolist()
                for epoch in lf.EPOCHS:
                    logits = prediction["epoch_" + str(epoch)]
                    require(logits.shape == (len(identifiers), len(classes)) and np.isfinite(logits).all(), "Invalid source logits")
                    predicted = [classes[int(i)] for i in logits.argmax(axis=1)]
                    oof.extend({"epoch": epoch, "seed": seed, "fold": fold_id, "record_id": identifier, "prediction": label}
                               for identifier, label in zip(identifiers, predicted))
                source_models.append({"task_id": task_id, "fold": fold_id, "seed": seed,
                    "training": ref(dest / "training.json"), "model": ref(dest / "model.pt"),
                    "source_epoch_predictions": ref(dest / "source_epoch_predictions.npz")})
        locks[task_id] = lf.select_common_epoch(task_id, labels, task["inner_folds"], oof)
        all_source_rows.extend({"task_id": task_id, **r} for r in oof)
    require(len(source_models) == 27 and set(locks) == set(TASK_ORDER) - {"zaoren_expanded10"}, "Source grid incomplete")
    save(directory / "source_epoch_lock.json", {"status": "ALL_SOURCE_EPOCHS_LOCKED_BEFORE_FINAL_FIT_OR_QUERY",
        "locks": locks, "expanded10_epoch": lf.inherited_epoch("zaoren_expanded10", locks),
        "models": source_models, "source_manifest": bound["source_manifest"], "protocol": ref(out / "protocol.json"),
        "real_query_features_or_labels_used": False})
    csvwrite(directory / "source_oof_predictions.csv", all_source_rows)
    return {"status": "DONE", "source_lock": ref(directory / "source_epoch_lock.json"), "model_fits": 27}


def _fit(out, proof):
    source_receipt = healthy_stage(out, "source")
    np = numeric_dependencies(); lf = leaf(); bound, tasks = bound_sources(out); budget = runtime_budget(proof)
    lock_path = checked(source_receipt["verified_outputs"]["source_lock"]); lock = read(lock_path)
    checked(lock["source_manifest"]); checked(lock["protocol"])
    for model in lock["models"]:
        checked(model["training"]); checked(model["model"]); checked(model["source_epoch_predictions"])
    directory = out / "fit"; directory.mkdir(exist_ok=True)
    models, pair_receipts = [], []
    for task in tasks:
        task_id = task["task_id"]; pack = load_compact(task["source_pack"], np, source=True)
        require(pack["record_ids"] == task["source_ids"], "Final source order differs")
        epochs = lf.inherited_epoch(task_id, lock["locks"])
        for seed in lf.SEEDS:
            pair = {}
            for arm in lf.ARMS:
                budget(); key = f"{task_id}.{arm}.seed{seed}"
                context = fit_context(task_id, "final_source_fit", arm, seed, epochs, pack["record_ids"], [])
                release = runtime_release(out, proof, bound, {key: context}, source_locks=True)
                dest = directory / task_id / arm / f"seed{seed}"
                metadata = lf.fit_source_leaf(pack, arm=arm, epochs=epochs, seed=seed, directory=dest,
                    split_id=key, release=release, budget_check=budget)
                pair[arm] = metadata
                models.append({"model_key": key, "task_id": task_id, "arm": arm, "seed": seed, "epochs": epochs,
                    "model": ref(dest / "model.pt"), "training": ref(dest / "training.json"),
                    "source_identity_aliases": task["source_alias_union"]})
            pair_receipts.append({"task_id": task_id, "seed": seed,
                **lf.audit_capacity_pair(pair["alpha_capacity3"], pair["alpha_n_time"])})
    require(len(models) == 36 and len(pair_receipts) == 12, "All final models and pair controls required")
    save(directory / "model_lock.json", {"status": "ALL_36_FINAL_MODELS_FROZEN_BEFORE_QUERY", "models": models,
         "source_epoch_lock": ref(lock_path), "protocol": ref(out / "protocol.json"),
         "capacity_pairing": pair_receipts, "query_features_or_labels_received": False})
    return {"status": "DONE", "model_lock": ref(directory / "model_lock.json"), "model_fits": 36}


def _predict(out, proof):
    fit_receipt = healthy_stage(out, "fit"); healthy_stage(out, "source")
    np = numeric_dependencies(); lf = leaf(); bound, _ = bound_sources(out); budget = runtime_budget(proof)
    model_lock_path = checked(fit_receipt["verified_outputs"]["model_lock"]); models = read(model_lock_path)
    checked(models["source_epoch_lock"]); checked(models["protocol"])
    prediction_manifest = read(checked(bound["prediction_manifest"]))
    tasks = {t["task_id"]: t for t in prediction_manifest["tasks"]}
    validate_model_grid(models["models"], None, tasks)
    frozen = {}
    # Verify every model before the first real query array is opened.
    for item in models["models"]:
        checked(item["model"]); checked(item["training"])
        frozen[item["model_key"]] = {"training_path": item["training"]["path"], "training_sha256": item["training"]["sha256"],
            "model_path": item["model"]["path"], "model_sha256": item["model"]["sha256"],
            "source_identity_aliases": item["source_identity_aliases"], "query_ids": tasks[item["task_id"]]["query_ids"]}
    release = runtime_release(out, proof, bound, {}, source_locks=True)
    release.update(all_final_models_frozen=True, frozen_models=frozen)
    directory = out / "predict"; directory.mkdir(exist_ok=True); predictions = []
    for task_id in TASK_ORDER:
        query = load_compact(tasks[task_id]["query_pack"], np, source=False)
        require(query["record_ids"] == tasks[task_id]["query_ids"], "Frozen query IDs differ")
        for item in (m for m in models["models"] if m["task_id"] == task_id):
            budget(); dest = directory / task_id / item["arm"] / f"seed{item['seed']}"
            lf.predict_frozen_leaf(query, model_key=item["model_key"], release=release, directory=dest, budget_check=budget)
            predictions.append({"model_key": item["model_key"], "task_id": task_id, "arm": item["arm"], "seed": item["seed"],
                "predictions": ref(dest / "predictions.npz"), "prediction_metadata": ref(dest / "prediction.json")})
    require(len(predictions) == 36, "All arms and seeds must predict before any truth join")
    save(directory / "prediction_lock.json", {"status": "ALL_36_PREDICTIONS_LOCKED_BEFORE_TRUTH_JOIN",
        "models": ref(model_lock_path), "prediction_manifest": bound["prediction_manifest"],
        "predictions": predictions, "query_truth_joined": False})
    return {"status": "DONE", "prediction_lock": ref(directory / "prediction_lock.json"), "model_fits": 0}


def align_predictions(expected_ids, ids, class_labels, logits):
    """Explicit ID join; never compare another arm's original row positions."""
    positions = record_map(ids)
    require(set(positions) == set(expected_ids) and len(positions) == len(expected_ids), "Query IDs missing/extra")
    require(len(set(class_labels)) == len(class_labels) and len(logits) == len(ids), "Invalid prediction vocabulary")
    output = []
    for identifier in expected_ids:
        row = [float(v) for v in logits[positions[identifier]]]
        require(len(row) == len(class_labels) and all(math.isfinite(v) for v in row), "Nonfinite/misordered logits")
        output.append(class_labels[max(range(len(row)), key=lambda i: row[i])])
    return output


def validate_model_grid(models, predictions, tasks):
    expected = {f"{task}.{arm}.seed{seed}" for task in TASK_ORDER for arm in leaf().ARMS for seed in leaf().SEEDS}
    require(set(tasks) == set(TASK_ORDER), "Query task set changed")
    for task_id, task in tasks.items():
        require(len(task["query_ids"]) == leaf().TASKS[task_id][2]
                and len(set(task["query_ids"])) == len(task["query_ids"]), "Fixed query membership count changed")
    require(len(models) == len(expected) and {r["model_key"] for r in models} == expected, "Final model grid missing/duplicate")
    by_key = {}
    for row in models:
        key = f"{row['task_id']}.{row['arm']}.seed{row['seed']}"
        require(row["model_key"] == key and key in expected, "Model key/arm/seed/task mismatch")
        by_key[key] = row
    if predictions is not None:
        require(len(predictions) == len(expected) and {r["model_key"] for r in predictions} == expected,
                "Prediction grid missing/duplicate before truth release")
        for row in predictions:
            model = by_key[row["model_key"]]
            require(all(row[k] == model[k] for k in ("task_id", "arm", "seed")), "Prediction/model identity differs")
    return by_key


def score_labels(truth, predicted, vocabulary):
    require(len(truth) == len(predicted) > 0 and set(truth) | set(predicted) <= set(vocabulary), "Unknown true/predicted labels")
    metric_classes = sorted(set(truth)); per_class = []; confusion = []
    for label in vocabulary:
        tp = sum(a == b == label for a, b in zip(truth, predicted))
        support = sum(a == label for a in truth); positives = sum(b == label for b in predicted)
        precision = tp / positives if positives else 0.; recall = tp / support if support else 0.
        f1 = 2 * tp / (support + positives) if support + positives else 0.
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": f1, "support": support})
        confusion.extend({"true_class": label, "predicted_class": other,
                          "count": sum(a == label and b == other for a, b in zip(truth, predicted))} for other in vocabulary)
    actual = [r for r in per_class if r["class"] in metric_classes]
    return {"accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(truth),
            "macro_f1": statistics.mean(r["f1"] for r in actual),
            "balanced_accuracy": statistics.mean(r["recall"] for r in actual), "metric_classes": metric_classes}, per_class, confusion


def corrected_harmed(truth, baseline, candidate):
    require(len(truth) == len(baseline) == len(candidate) > 0, "Paired score length differs")
    corrected = sum(b != t and c == t for t, b, c in zip(truth, baseline, candidate))
    harmed = sum(b == t and c != t for t, b, c in zip(truth, baseline, candidate))
    return {"corrected": corrected, "harmed": harmed, "net_correct": corrected - harmed,
            "accuracy_delta_pp": 100 * (corrected - harmed) / len(truth)}


def _score(out, proof):
    predict_receipt = healthy_stage(out, "predict"); fit_receipt = healthy_stage(out, "fit")
    source_receipt = healthy_stage(out, "source")
    budget = runtime_budget(proof); np = numeric_dependencies()
    lock_path = checked(predict_receipt["verified_outputs"]["prediction_lock"]); lock = read(lock_path)
    require(lock["models"] == fit_receipt["verified_outputs"]["model_lock"], "Prediction release uses a different model lock")
    model_lock = read(checked(lock["models"])); query_manifest = read(checked(lock["prediction_manifest"]))
    require(model_lock["source_epoch_lock"] == source_receipt["verified_outputs"]["source_lock"],
            "Frozen model/source-selection locks differ before truth release")
    tasks = {t["task_id"]: t for t in query_manifest["tasks"]}
    model_by_key = validate_model_grid(model_lock["models"], lock["predictions"], tasks)
    classes_by_key = {}
    for key, model in model_by_key.items():
        checked(model["model"])
        classes = read(checked(model["training"]))["class_labels"]
        require(len(classes) == leaf().TASKS[model["task_id"]][3] and len(set(classes)) == len(classes), "Frozen class vocabulary changed")
        classes_by_key[key] = classes
    cached = {}
    # Check all bytes, shapes and IDs BEFORE opening the first truth.csv.
    for item in lock["predictions"]:
        checked(item["prediction_metadata"])
        data = read_npz(item["predictions"], np, ("record_ids", "class_labels", "logits"))
        identifiers = data["record_ids"].astype(str).tolist(); classes = data["class_labels"].astype(str).tolist()
        require(classes == classes_by_key[item["model_key"]], "Prediction class columns differ from the frozen model")
        predicted = align_predictions(tasks[item["task_id"]]["query_ids"], identifiers, classes, data["logits"])
        cached[item["model_key"]] = (item, classes, predicted)
    directory = out / "score"; directory.mkdir(exist_ok=True)
    overall, per_class_rows, confusion_rows, pairs, individual = [], [], [], [], []
    for task_id in TASK_ORDER:
        budget(); task = tasks[task_id]; truth_rows = rows(checked(task["truth"]))
        truth_by_id = {r["record_id"]: r for r in truth_rows}
        require(len(truth_by_id) == len(truth_rows) and set(truth_by_id) == set(task["query_ids"]), "Truth IDs differ")
        ids = task["query_ids"]; truth = [truth_by_id[i]["y"] for i in ids]
        subsets = {"overall": list(range(len(ids)))}
        if task_id == "zaoren_expanded10":
            subsets["legacy_exposed171"] = [i for i, identifier in enumerate(ids) if truth_by_id[identifier]["stratum"] == "legacy_exposed"]
            subsets["new3_descriptive_only"] = [i for i, identifier in enumerate(ids) if truth_by_id[identifier]["stratum"] == "new_three_record_group"]
            require(len(subsets["legacy_exposed171"]) == 171 and len(subsets["new3_descriptive_only"]) == 3, "Expanded query strata changed")
        for seed in leaf().SEEDS:
            arms = {}
            for arm in leaf().ARMS:
                key = f"{task_id}.{arm}.seed{seed}"; item, classes, predicted = cached[key]; arms[arm] = predicted
                require(len(classes) == leaf().TASKS[task_id][3], "Class vocabulary changed")
                individual.extend({"task_id": task_id, "arm": arm, "seed": seed, "record_id": identifier,
                    "truth": t, "prediction": p, "correct": t == p, "stratum": truth_by_id[identifier]["stratum"]}
                    for identifier, t, p in zip(ids, truth, predicted))
                for subset, indices in subsets.items():
                    score, class_rows, confusion = score_labels([truth[i] for i in indices], [predicted[i] for i in indices], classes)
                    prefix = {"task_id": task_id, "subset": subset, "arm": arm, "seed": seed}
                    overall.append({**prefix, "N": len(indices), "correct": sum(truth[i] == predicted[i] for i in indices),
                        "accuracy_percent": 100 * score["accuracy"], "macro_f1": score["macro_f1"],
                        "balanced_accuracy_percent": 100 * score["balanced_accuracy"],
                        "metric_classes": json.dumps(score["metric_classes"]), "confirmation_status": "EXPOSED_DEVELOPMENT_ONLY"})
                    per_class_rows.extend({**prefix, **row} for row in class_rows)
                    confusion_rows.extend({**prefix, **row} for row in confusion)
            for baseline, candidate in (("alpha_single", "alpha_capacity3"), ("alpha_single", "alpha_n_time"), ("alpha_capacity3", "alpha_n_time")):
                for subset, indices in subsets.items():
                    result = corrected_harmed([truth[i] for i in indices], [arms[baseline][i] for i in indices],
                                             [arms[candidate][i] for i in indices])
                    pairs.append({"task_id": task_id, "subset": subset, "seed": seed, "baseline": baseline,
                                  "candidate": candidate, "N": len(indices), **result})
    summary = []
    groups = defaultdict(list)
    for row in overall:
        groups[(row["task_id"], row["subset"], row["arm"])].append(row)
    for (task_id, subset, arm), values in groups.items():
        require(sorted(v["seed"] for v in values) == list(leaf().SEEDS), "All seeds required, no best-seed report")
        summary.append({"task_id": task_id, "subset": subset, "arm": arm, "N_per_seed": values[0]["N"],
            **{metric + "_" + suffix: operation(v[metric] for v in values)
               for metric in ("accuracy_percent", "macro_f1", "balanced_accuracy_percent")
               for suffix, operation in (("mean", statistics.mean), ("sampleSD", statistics.stdev))}})
    csvwrite(directory / "per_seed_metrics.csv", overall); csvwrite(directory / "model_comparison.csv", summary)
    csvwrite(directory / "per_class_metrics.csv", per_class_rows); csvwrite(directory / "confusion_matrix_counts.csv", confusion_rows)
    csvwrite(directory / "paired_corrected_harmed.csv", pairs); csvwrite(directory / "all_query_predictions.csv", individual)
    cost_rows = []
    for model in read(checked(lock["models"]))["models"]:
        metadata = read(checked(model["training"]))
        cost_rows.append({"task_id": model["task_id"], "arm": model["arm"], "seed": model["seed"],
            "epochs": model["epochs"], "parameter_count": metadata["parameter_count"],
            "fit_process_CPU_seconds_inside_leaf": metadata["fit_process_CPU_seconds"],
            "model_bytes": checked(model["model"]).stat().st_size})
    csvwrite(directory / "model_costs.csv", cost_rows)
    _plots(directory, summary, read(checked(source_receipt["verified_outputs"]["source_lock"])))
    save(directory / "result.json", {"status": "DONE_DEVELOPMENT_ONLY", "summary": summary,
        "prediction_lock": ref(lock_path), "all_seeds_reported": True, "independent_confirmation": False,
        "new_HQ_TM_independent_specimens": 0, "new_readable_representations": 303,
        "source_fit_count": 27, "final_fit_count": 36, "real_query_labels_used_for_training_or_epoch_selection": False,
        "historical_96_0234_not_reassigned": True, "units_batch_ancestry_limits": "EXPLORATORY_MULTI_REPRESENTATION_ENGINEERING"})
    return {"status": "DONE", "results": ref(directory / "result.json"), "model_fits": 0}


def _plots(directory, summary, source_lock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = list(leaf().ARMS); colors = ("#456990", "#e8a87c", "#50a18e")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, metric in enumerate(("accuracy_percent", "macro_f1")):
        for j, arm in enumerate(arms):
            data = [next(r for r in summary if r["task_id"] == task and r["subset"] == "overall" and r["arm"] == arm) for task in TASK_ORDER]
            axes[k].bar([i + (j - 1) * .24 for i in range(4)], [r[metric + "_mean"] for r in data],
                        yerr=[r[metric + "_sampleSD"] for r in data], width=.23, color=colors[j], label=arm, capsize=2)
        axes[k].set_xticks(range(4), ("ZA7", "HQ4", "TM4", "ZA10")); axes[k].set_ylim(0, 100 if k == 0 else 1)
        axes[k].set_ylabel("Accuracy (%)" if k == 0 else "Macro-F1"); axes[k].grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8); fig.suptitle("Fixed-query development results: all three seeds")
    fig.tight_layout(); fig.savefig(directory / "model_comparison.png", dpi=180); fig.savefig(directory / "model_comparison.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for task, lock in source_lock["locks"].items():
        ax.plot([r["epoch"] for r in lock["candidates"]], [r["macro_f1"] for r in lock["candidates"]], marker="o", label=task)
    ax.set(xlabel="Candidate epoch (source-only)", ylabel="Mean source-OOF Macro-F1", ylim=(0, 1))
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(directory / "source_epoch_selection.png", dpi=180); fig.savefig(directory / "source_epoch_selection.pdf"); plt.close(fig)


def _bind_numerical_gate(out, proof, prep, source, query, directory):
    """Real later source-only forward/alias/filter gates; zero gradient fits."""
    lf = leaf(); release = runtime_release(out, proof, prep, {})
    np, torch, Scaler, Encoder, FitScope, inference_audit = lf.numerical_runtime(release)
    budget = runtime_budget(proof); checks = []
    for task in source["tasks"]:
        pack = load_compact(task["source_pack"], np, source=True); task_id = task["task_id"]
        examples = {m: np.asarray(pack["values"][m][:3], dtype=np.float32) for m in lf.FEATURES}
        capacity = lf.build_model(torch, Encoder, "alpha_capacity3", lf.TASKS[task_id][3], 42)
        multimodal = lf.build_model(torch, Encoder, "alpha_n_time", lf.TASKS[task_id][3], 42)
        require(lf.state_hash(capacity) == lf.state_hash(multimodal), "Actual capacity/real initial weights differ")
        require(sum(p.numel() for p in capacity.parameters()) == sum(p.numel() for p in multimodal.parameters()),
                "Actual capacity/real parameter counts differ")
        for arm, model in (("alpha_capacity3", capacity), ("alpha_n_time", multimodal)):
            model.eval(); selected = {m: examples[m] for m in dict.fromkeys(lf.ARMS[arm])}
            with torch.no_grad():
                logits = model({m: torch.as_tensor(x) for m, x in selected.items()})
            require(tuple(logits.shape) == (3, lf.TASKS[task_id][3]) and bool(torch.isfinite(logits).all()), "Real forward shape/values failed")
            inference_audit(model, selected, batch_sizes=(1, 2), atol=1e-5, rtol=1e-5)
        heldout = next(t for t in query["tasks"] if t["task_id"] == task_id)["query_ids"][0]
        scope = FitScope(pack["record_ids"][:2], "preflight.invalid_fit." + task_id, "supplemental-fusion-preflight",
                         directory / "rejected_fit_scope.jsonl", code_version=sha(SELF))
        estimator = Scaler(); rejected = False
        try:
            scope.fit(estimator, examples["alpha"][:2], sample_ids=[heldout, pack["record_ids"][1]], component="intentional_rejection")
        except ValueError:
            rejected = True
        require(rejected and not hasattr(estimator, "mean_"), "Disallowed-ID fit did not stop before estimator fitting")
        checks.append({"task_id": task_id, "source_only_forward_and_BN": "PASS", "actual_parameter_count": sum(p.numel() for p in capacity.parameters()),
                       "capacity_initial_state_equal": True, "invalid_ID_fit_rejected_before_fit": True, "gradient_training_fits": 0})
        budget()
    from scipy.signal import resample_poly
    native = np.arange(6750, dtype=np.float64)
    alternating = (-1.) ** np.arange(6750)
    high = resample_poly(alternating, 512, 3375, window=("kaiser", 5.0), padtype="line")
    require(float(np.sqrt(np.mean(high[20:-20] ** 2))) < .02, "Anti-alias high-frequency attenuation failed")
    pulse = np.exp(-.5 * ((native - 3333.25) / 3.) ** 2)
    release_preparation = {k: True for k in ("root_preparation_authorized", "quantum_terminal_reviewed",
        "raw_inputs_and_source_axes_frozen", "external_actual_CPU_supervisor_active")}
    filtered, axis = lf.prepare_time_rows(native, pulse[None, :], preparation_release=release_preparation)
    require(filtered.shape == (1, 1024) and np.isclose(np.linalg.norm(filtered), 1.)
            and np.isfinite(filtered).all() and axis[-1] <= native[-1], "Narrow-pulse preparation failed")
    return {"status": "PASS_REAL_SOURCE_FORWARD_AND_SYNTHETIC_SIGNAL_CONTROLS", "tasks": checks,
            "synthetic_high_frequency_interior_RMS": float(np.sqrt(np.mean(high[20:-20] ** 2))),
            "synthetic_controls_are_not_real_plant_samples": True, "gradient_training_fits": 0,
            "query_arrays_or_truth_read": False,
            "runtime_environment": {"python": sys.version, "executable": sys.executable,
                "numpy": np.__version__, "torch": torch.__version__,
                "scipy": sys.modules["scipy"].__version__, "sklearn": sys.modules["sklearn"].__version__,
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "device": "cpu"}}


def worker(out, stage, proof_path, proof_sha):
    proof = verify_worker_proof(out, stage, proof_path, proof_sha)
    function = {"prepare": _prepare, "bind": _bind, "source": _source, "fit": _fit, "predict": _predict, "score": _score}[stage]
    path = out / stage / "worker_result.json"
    try:
        result = function(out, proof)
        save(path, {**result, "stage": stage, "completed_at_utc": utc(), "proof_sha256": proof_sha})
        return 0
    except BaseException as exc:
        save(path, {"status": "FAILED_KEEP_ARTIFACTS", "stage": stage, "exception": type(exc).__name__,
                    "reason": str(exc), "ended_at_utc": utc(), "proof_sha256": proof_sha})
        raise


def stage_allowed(out, stage):
    require(stage in STAGES, "Unknown stage")
    ledger = read(out / "resource_ledger.json")
    require(not any(p["status"] != "EXITED_HEALTHY" for p in ledger["phases"]), "Prior stage failure/unknown blocks automatic continuation")
    completed = [p["stage"] for p in ledger["phases"]]
    require(completed == list(STAGES[:len(completed)]) and len(completed) < len(STAGES)
            and STAGES[len(completed)] == stage, "Stages must run once in fixed order")
    for previous in completed:
        healthy_stage(out, previous)
    return ledger


def closed_resource_verdict(observed, total_cpu, limits, peak_rss, *, deadline_exceeded):
    reason = observed["stop_reason"]
    if total_cpu > limits["cpu_seconds"]:
        reason = reason or "CLOSED_CUMULATIVE_ACTUAL_CPU_LIMIT_EXCEEDED"
    if peak_rss > limits["RSS_bytes"]:
        reason = reason or "CLOSED_WAIT4_PEAK_RSS_LIMIT_EXCEEDED"
    if deadline_exceeded:
        reason = reason or "CLOSED_WALL_DEADLINE_EXCEEDED"
    return observed["exit_code"] == 0 and reason is None, reason


def supervise_stage(out, stage):
    supervisor_started_cpu = time.process_time()
    _, auth = registration(out); ledger = stage_allowed(out, stage)
    require(ledger["limits"] == auth["limits"], "Mutable ledger limits differ from frozen root authority")
    limits = ledger["limits"]; used = ledger["closed_child_CPU_seconds"] + ledger["closed_supervisor_CPU_seconds_observed"]
    remaining = limits["cpu_seconds"] - used
    require(remaining >= 30 and datetime.now(timezone.utc) < datetime.fromisoformat(ledger["deadline_at_utc"]),
            "Finite whole-run CPU/wall resources exhausted; do not reset or score incomplete work")
    stage_dir = out / stage; require(not stage_dir.exists(), "Preserve existing incomplete stage; no automatic scientific retry")
    stage_dir.mkdir(); parent_pid = os.getpid()
    parent_stat = Path("/proc", str(parent_pid), "stat").read_text()
    fields = parent_stat[parent_stat.rfind(")") + 1:].split()
    proof_path = stage_dir / "supervisor_proof.json"
    proof = {"stage": stage, "output": str(out.resolve()), "parent_pid": parent_pid, "parent_start_ticks": int(fields[19]),
             "driver_sha256": sha(SELF), "nonce": str(uuid.uuid4()), "remaining_cpu_seconds": remaining,
             "deadline_at_utc": ledger["deadline_at_utc"], "AS_bytes": limits["AS_bytes"], "RSS_bytes": limits["RSS_bytes"]}
    save(proof_path, proof)
    environment = {**os.environ, **{k: "1" for k in THREAD_KEYS}, "CUDA_VISIBLE_DEVICES": "",
                   "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    def native_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(remaining) - 10), int(remaining)))
        resource.setrlimit(resource.RLIMIT_AS, (limits["AS_bytes"],) * 2)
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
            os._exit(125)
    command = [sys.executable, str(SELF), "_worker", "--output", str(out.resolve()), "--stage", stage,
               "--proof", str(proof_path.resolve()), "--proof-sha256", sha(proof_path)]
    row = {"stage": stage, "status": "RUNNING", "command": command, "started_at_utc": utc(), "prior_closed_child_CPU_seconds": used}
    with (stage_dir / "console.log").open("x") as console:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=native_limits)
        row["pid"] = child.pid
        ledger["phases"].append(row); ledger["status"] = "RUNNING"; save(out / "resource_ledger.json", ledger, mutable=True)
        old = load_module(OLD_SUPERVISOR, "reviewed_single_process_actual_CPU_monitor")
        old.CPU_CAP = limits["cpu_seconds"]; old.WALL_CAP = limits["wall_seconds"]; old.RSS_CAP = limits["RSS_bytes"]
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(ledger["created_at_utc"])).total_seconds()
        observed = old.monitor_owned_child(child, time.monotonic() - elapsed, used)
    usage = observed.pop("usage")
    if usage is None:
        row.update(observed, status="UNKNOWN_CPU_PARTIAL", actual_child_CPU_seconds=None, ended_at_utc=utc())
        ledger["status"] = "UNKNOWN_CPU_PARTIAL"; save(out / "resource_ledger.json", ledger, mutable=True)
        raise RuntimeError("Unknown final wait4 accounting; preserve files and require manual review, never reset budget")
    actual = float(usage.ru_utime + usage.ru_stime); ledger["closed_child_CPU_seconds"] += actual
    parent_observed = time.process_time() - supervisor_started_cpu
    ledger["closed_supervisor_CPU_seconds_observed"] += parent_observed
    healthy, closed_reason = closed_resource_verdict(observed,
        ledger["closed_child_CPU_seconds"] + ledger["closed_supervisor_CPU_seconds_observed"], limits,
        usage.ru_maxrss * 1024, deadline_exceeded=datetime.now(timezone.utc) > datetime.fromisoformat(ledger["deadline_at_utc"]))
    observed["stop_reason"] = closed_reason
    row.update(observed, status="EXITED_HEALTHY" if healthy else "FAILED_CLOSED", ended_at_utc=utc(),
               actual_child_CPU_seconds=actual, wait4_peak_RSS_KiB=usage.ru_maxrss, log_sha256=sha(stage_dir / "console.log"))
    row["supervisor_CPU_seconds_observed_through_phase_close"] = parent_observed
    ledger["status"] = row["status"]; save(out / "resource_ledger.json", ledger, mutable=True)
    if not healthy:
        raise RuntimeError("Stage failed or reached guard; retain true partial state: " + stage)
    result = read(stage_dir / "worker_result.json"); require(result["status"] == "DONE", "No successful worker result")
    save(stage_dir / "closed_stage.json", {"status": "DONE_HEALTHY_CLOSED_STAGE", "stage": stage,
         "exit_code": 0, "stop_reason": None, "actual_child_CPU_seconds": actual,
         "worker_result": ref(stage_dir / "worker_result.json"), "supervisor_proof": ref(proof_path),
         "closed_resource_row": row, "closure_at_utc": utc()})
    row["closed_stage"] = ref(stage_dir / "closed_stage.json")
    save(out / "resource_ledger.json", ledger, mutable=True)
    if stage == "score":
        save(out / "completion.json", {"status": "DONE_DEVELOPMENT_ONLY", "model_fits": 63,
            "source_fits": 27, "final_fits": 36, "actual_child_CPU_seconds": ledger["closed_child_CPU_seconds"],
            "actual_CPU_seconds_measured_through_phase_close": ledger["closed_child_CPU_seconds"] + ledger["closed_supervisor_CPU_seconds_observed"],
            "resource_ledger": ref(out / "resource_ledger.json"), "results": ref(out / "score/result.json"),
            "independent_confirmation": False, "quantum_calculations_by_this_run": 0})
    print(json.dumps({"stage": stage, "status": "DONE_HEALTHY_CLOSED_STAGE", "actual_child_CPU_seconds": actual,
                      "cumulative_child_CPU_seconds": ledger["closed_child_CPU_seconds"]}), flush=True)


def run(out, stage):
    _, auth = registration(out)
    with (out / ".run.lock").open("a") as local_lock, Path(auth["quantum_global_lock"]).open("a") as quantum_lock:
        fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(quantum_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = STAGES if stage == "all" else (stage,)
        for current in stages:
            completed = [p["stage"] for p in read(out / "resource_ledger.json")["phases"] if p["status"] == "EXITED_HEALTHY"]
            if stage == "all" and current in completed:
                healthy_stage(out, current)
                continue
            supervise_stage(out, current)


def plan():
    return {**leaf().metadata_plan(read(DRAFT)), "driver": "FULL_STAGED_CODE_NOT_NUMERICALLY_RUN",
            "stages": list(STAGES), "proposed_CPU_seconds": 3600, "proposed_wall_seconds": 14400,
            "missing_before_execution": ["root terminal-Q and implementation-review authority",
                "actual supervised raw preparation and numerical anti-alias/ID/forward gates",
                "all data/code/axis hashes bound in a fresh run", "actual source/model/prediction/score stages"],
            "root_authority_required_after_quantum": True, "current_accuracy": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--authority", type=Path, required=True)
    register_parser.add_argument("--authority-sha256", required=True)
    run_parser = sub.add_parser("run"); run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    internal = sub.add_parser("_worker"); internal.add_argument("--output", type=Path, required=True)
    internal.add_argument("--stage", choices=STAGES, required=True); internal.add_argument("--proof", type=Path, required=True)
    internal.add_argument("--proof-sha256", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), indent=2)); return
    out = args.output.resolve()
    if args.command == "register": register(out, args.authority, args.authority_sha256)
    elif args.command == "run": run(out, args.stage)
    else: worker(out, args.stage, args.proof, args.proof_sha256)


if __name__ == "__main__":
    main()
