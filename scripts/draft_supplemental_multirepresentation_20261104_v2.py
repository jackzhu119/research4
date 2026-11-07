"""Metadata-safe draft and deferred leaf implementation; no numerical imports here.

This file does not prepare real arrays, train or score when imported or invoked
with --plan. Numerical functions require a separately frozen root execution
release after the quantum task is terminal. The end-to-end role-pack binder and
driver remain NOT_RUN; a draft is not an executable experiment registration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
EPOCHS = (24, 48, 96)
MODALITIES = ("alpha", "n", "time")
FEATURES = MODALITIES + ("alpha_long",)
POINTS = {"alpha": 128, "n": 128, "time": 1024, "alpha_long": 1024}
ARMS = {"alpha_single": ("alpha",), "alpha_capacity3": ("alpha", "alpha", "alpha_long"),
        "alpha_n_time": MODALITIES}
TASKS = {"zaoren_legacy7": (513, 171, 171, 7), "huangqi_legacy4": (241, 80, 82, 4),
         "tianma_legacy4": (242, 80, 83, 4), "zaoren_expanded10": (516, 174, 174, 10)}
_RUNTIME = None
NATIVE_TIME_AXIS_SHA = "fbf09eefe1c1970452b8b33c4dd91400768e3dc1d0065a9ac6e394824fb95887"
UNIFORM_TIME_AXIS_SHA = "fb8570076ac289d8362ffd8e518960fdfc7f5b549bf4ac71e5cd2c222e4b348e"
TIME_RESAMPLING_BASE = {"method": "within_span_linear_uniformization_then_polyphase", "up": 512, "down": 3375,
    "window": ["kaiser", 5.0], "padtype": "line", "native_points": 6750, "uniform_points": 6750,
    "uniformization": "endpoint_linspace_pointwise_linear_no_extrapolation",
    "coordinate_max_deviation_in_steps_limit": .001, "step_max_relative_deviation_limit": .001}


def require(value, message):
    if not value:
        raise ValueError(message)


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def arm_spec(arm):
    require(arm in ARMS, "Unknown/unregistered arm")
    inputs = ARMS[arm]
    return {"branch_inputs": list(inputs), "independent_encoders": len(inputs),
            "branch_points": [POINTS[name] for name in inputs],
            "embedding_dimensions": 16 * len(inputs), "shared_encoder_weights": False}


def identity_aliases(row):
    aliases = [row.get("record_id"), row.get("entity_id")]
    legacy = row.get("legacy_match_ids", [])
    if isinstance(legacy, str):
        legacy = json.loads(legacy)
    require(isinstance(legacy, list), "Explicit legacy alias list required")
    aliases += legacy + list(row.get("identity_aliases", []))
    return {str(value) for value in aliases if value is not None and str(value)}


def validate_roles(rows, expected_counts):
    """Check the complete canonical metadata, never numerical specimen arrays."""
    roles = ("source", "support", "query")
    require(set(expected_counts) == set(roles), "All three roles required")
    require(len({r["record_id"] for r in rows}) == len(rows), "Duplicate canonical records")
    require(Counter(r["role"] for r in rows) == Counter(expected_counts), "Fixed role counts changed")
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in rows:
        require(row.get("entity_id") and row.get("label"), "Entity and class metadata required")
        keys = sorted(identity_aliases(row))
        require(keys, "Missing entity aliases")
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    groups = defaultdict(lambda: {"roles": set(), "labels": set(), "records": set()})
    fingerprints = defaultdict(set)
    for row in rows:
        group = groups[find(sorted(identity_aliases(row))[0])]
        group["roles"].add(row["role"]); group["labels"].add(row["label"])
        group["records"].add(row["record_id"])
        if row.get("alpha_fingerprint"):
            fingerprints[row["alpha_fingerprint"]].add(row["role"])
    require(all(len(g["roles"]) == len(g["labels"]) == len(g["records"]) == 1 for g in groups.values()),
            "Cross-role aliases, class conflicts or multiple canonical records for one entity")
    require(all(len(v) == 1 for v in fingerprints.values()), "Numerically identical alpha crosses roles")
    return {role: [r["record_id"] for r in rows if r["role"] == role] for role in roles}


def validate_folds(source_ids, folds):
    require(len(folds) == 3 and len(set(source_ids)) == len(source_ids), "Exactly three source folds required")
    allowed = set(source_ids); counts = Counter()
    for fold in folds:
        train, valid = fold["train_ids"], fold["valid_ids"]
        require(len(set(train)) == len(train) and len(set(valid)) == len(valid), "Duplicate fold IDs")
        require(not set(train) & set(valid) and set(train) | set(valid) == allowed,
                "Fold overlap or changed source universe")
        counts.update(valid)
    require(counts == Counter({i: 1 for i in source_ids}), "Every source entity must be OOF once")


def classification_metrics(truth, predictions, classes):
    require(len(truth) == len(predictions) > 0 and len(set(classes)) == len(classes), "Invalid score inputs")
    require(set(truth) <= set(classes) and set(predictions) <= set(classes), "Unknown class")
    f1, recalls = [], []
    for label in classes:
        tp = sum(a == b == label for a, b in zip(truth, predictions))
        fn = sum(a == label and b != label for a, b in zip(truth, predictions))
        fp = sum(a != label and b == label for a, b in zip(truth, predictions))
        require(tp + fn > 0, "Source OOF must cover every class")
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        recalls.append(tp / (tp + fn))
    return {"macro_f1": math.fsum(f1) / len(classes),
            "balanced_accuracy": math.fsum(recalls) / len(classes),
            "accuracy": sum(a == b for a, b in zip(truth, predictions)) / len(truth)}


def select_common_epoch(task_id, source_labels, folds, records):
    """Only source OOF rows; input rows contain predictions, not query truth."""
    require(task_id in TASKS and task_id != "zaoren_expanded10", "Expanded singleton classes cannot select epochs")
    source_ids = list(source_labels)
    require(len(source_ids) == TASKS[task_id][0] and len(set(source_labels.values())) == TASKS[task_id][3],
            "Source pool or class coverage differs from fixed roles")
    validate_folds(source_ids, folds)
    allowed_keys = {"epoch", "seed", "fold", "record_id", "prediction"}
    groups = defaultdict(list)
    for row in records:
        require(set(row) == allowed_keys and row["epoch"] in EPOCHS and row["seed"] in SEEDS,
                "Unexpected row fields, epoch or seed")
        require(type(row["fold"]) is int and 0 <= row["fold"] < 3, "Unknown source fold")
        require(row["record_id"] in folds[row["fold"]]["valid_ids"], "Prediction is not this fold's source validation")
        groups[(row["epoch"], row["seed"])].append(row)
    require(set(groups) == {(epoch, seed) for epoch in EPOCHS for seed in SEEDS}, "Incomplete seed/epoch grid")
    classes = sorted(set(source_labels.values())); seed_metrics = []
    for epoch in EPOCHS:
        for seed in SEEDS:
            rows = groups[(epoch, seed)]
            require(Counter(r["record_id"] for r in rows) == Counter({i: 1 for i in source_ids}),
                    "OOF IDs incomplete, duplicated or outside source")
            score = classification_metrics([source_labels[r["record_id"]] for r in rows],
                                           [r["prediction"] for r in rows], classes)
            seed_metrics.append({"epoch": epoch, "seed": seed, **score})
    candidates = [{"epoch": epoch, **{key: math.fsum(r[key] for r in seed_metrics if r["epoch"] == epoch) / 3
                   for key in ("macro_f1", "balanced_accuracy", "accuracy")}} for epoch in EPOCHS]
    chosen = min(candidates, key=lambda r: (-r["macro_f1"], -r["balanced_accuracy"], -r["accuracy"], r["epoch"]))
    return {"task_id": task_id, "epoch": chosen["epoch"], "scope": "SOURCE_OOF_ONLY",
            "source_ids_sha256": objhash(source_ids), "folds_sha256": objhash(folds),
            "seed_metrics": seed_metrics, "candidates": candidates, "query_labels_used": False}


def inherited_epoch(task_id, locks):
    require(task_id in TASKS, "Unknown task")
    owner = "zaoren_legacy7" if task_id == "zaoren_expanded10" else task_id
    require(owner in locks and locks[owner]["scope"] == "SOURCE_OOF_ONLY", "Missing complete source lock")
    require(locks[owner]["epoch"] in EPOCHS, "Unregistered selected epoch")
    return locks[owner]["epoch"]


def resample_row(axis, values, target):
    """Pure pointwise linear interpolation; all axes retain their own meaning."""
    from bisect import bisect_right
    axis, values, target = list(map(float, axis)), list(map(float, values)), list(map(float, target))
    require(len(axis) == len(values) >= 2 and target, "Aligned row and axis required")
    require(all(math.isfinite(v) for v in axis + values + target), "Nonfinite row or axis")
    require(all(b > a for a, b in zip(axis, axis[1:])) and all(b > a for a, b in zip(target, target[1:])),
            "Strictly increasing independent axes required")
    require(target[0] >= axis[0] and target[-1] <= axis[-1], "No extrapolation")
    output = []
    for coordinate in target:
        left = min(max(bisect_right(axis, coordinate) - 1, 0), len(axis) - 2)
        weight = (coordinate - axis[left]) / (axis[left + 1] - axis[left])
        output.append((1 - weight) * values[left] + weight * values[left + 1])
    return output


def pointwise_row(modality, row):
    require(modality in FEATURES, "Unknown physical representation")
    row = list(map(float, row))
    require(row and all(math.isfinite(v) for v in row), "Nonfinite or empty row")
    if modality == "n":
        return row
    if modality in ("alpha", "alpha_long"):
        mean = math.fsum(row) / len(row)
        scale = math.sqrt(math.fsum((v - mean) ** 2 for v in row) / len(row))
        require(scale > 1e-12, "Constant alpha blocks the run")
        return [(v - mean) / scale for v in row]
    scale = math.sqrt(math.fsum(v * v for v in row))
    require(scale > 1e-12, "Zero time waveform blocks the run")
    return [v / scale for v in row]


def axis_float64_sha(axis):
    return hashlib.sha256(struct.pack("<" + "d" * len(axis), *axis)).hexdigest()


def time_axis_profile(axis, *, synthetic=False):
    """Unsupervised coordinates only; preserve native values and diagnose jitter."""
    axis = list(map(float, axis))
    require(len(axis) == 6750 and all(math.isfinite(v) for v in axis)
            and all(b > a for a, b in zip(axis, axis[1:])), "Finite increasing registered6750 axis required")
    native_sha = axis_float64_sha(axis)
    require(synthetic or native_sha == NATIVE_TIME_AXIS_SHA, "New acquisition axis requires a separate review")
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    uniform = [axis[0] + index * step for index in range(len(axis))]; uniform[-1] = axis[-1]
    require(synthetic or axis_float64_sha(uniform) == UNIFORM_TIME_AXIS_SHA, "Frozen endpoint-uniform source axis changed")
    delta = [b - a for a, b in zip(axis, axis[1:])]
    coordinate_error = max(abs(a - b) for a, b in zip(axis, uniform))
    step_error = max(abs(v - step) for v in delta)
    require(coordinate_error / step <= .001 and step_error / step <= .001,
            "Materially irregular native sampling is blocked, not hidden by an allclose tolerance")
    return uniform, {"native_axis_sha256_raw_float64": native_sha,
        "uniform_axis_sha256_raw_float64": axis_float64_sha(uniform),
        "native_start": axis[0], "native_end": axis[-1], "uniform_step": step,
        "delta_min": min(delta), "delta_max": max(delta), "delta_mean": statistics.fmean(delta),
        "delta_median": statistics.median(delta), "delta_population_SD": statistics.pstdev(delta),
        "max_native_uniform_coordinate_deviation": coordinate_error,
        "max_coordinate_deviation_in_steps": coordinate_error / step,
        "max_native_uniform_step_deviation": step_error, "max_relative_step_deviation": step_error / step,
        "coordinates_only_not_selected_from_labels_or_values": True,
        "synthetic_signal_control_not_real_specimens": bool(synthetic), "no_extrapolation": True,
        "time_unit": "NOT_EXPLICITLY_DECLARED"}


def validate_time_resampling(metadata, target_axis):
    require(set(metadata) == set(TIME_RESAMPLING_BASE) | {"axis_audit"}
            and all(metadata[k] == v for k, v in TIME_RESAMPLING_BASE.items()), "Two-step time recipe changed")
    audit = metadata["axis_audit"]
    expected_keys = {"native_axis_sha256_raw_float64", "uniform_axis_sha256_raw_float64", "output_axis_sha256_raw_float64",
        "native_start", "native_end", "uniform_step", "delta_min", "delta_max", "delta_mean", "delta_median", "delta_population_SD",
        "max_native_uniform_coordinate_deviation", "max_coordinate_deviation_in_steps", "max_native_uniform_step_deviation",
        "max_relative_step_deviation", "coordinates_only_not_selected_from_labels_or_values", "synthetic_signal_control_not_real_specimens",
        "no_extrapolation", "time_unit"}
    require(set(audit) == expected_keys and audit["native_axis_sha256_raw_float64"] == NATIVE_TIME_AXIS_SHA
            and audit["synthetic_signal_control_not_real_specimens"] is False
            and audit["no_extrapolation"] is True and audit["coordinates_only_not_selected_from_labels_or_values"] is True
            and audit["time_unit"] == "NOT_EXPLICITLY_DECLARED", "Time coordinate audit/provenance differs")
    for key in ("native_start", "native_end", "uniform_step", "delta_min", "delta_max", "delta_mean", "delta_median", "delta_population_SD",
            "max_native_uniform_coordinate_deviation", "max_coordinate_deviation_in_steps", "max_native_uniform_step_deviation", "max_relative_step_deviation"):
        require(type(audit[key]) in (int, float) and math.isfinite(audit[key]), "Invalid coordinate audit statistic")
    require(audit["uniform_step"] > 0 and audit["delta_min"] > 0 and audit["native_end"] > audit["native_start"]
            and audit["uniform_step"] == (audit["native_end"] - audit["native_start"]) / 6749
            and 0 <= audit["max_coordinate_deviation_in_steps"] <= .001 and 0 <= audit["max_relative_step_deviation"] <= .001,
            "Unreviewed native irregularity")
    uniform = [audit["native_start"] + i * audit["uniform_step"] for i in range(6750)]; uniform[-1] = audit["native_end"]
    expected = [audit["native_start"] + i * audit["uniform_step"] * 3375 / 512 for i in range(1024)]
    require(audit["uniform_axis_sha256_raw_float64"] == axis_float64_sha(uniform) == UNIFORM_TIME_AXIS_SHA
            and audit["output_axis_sha256_raw_float64"] == axis_float64_sha(target_axis)
            and list(target_axis) == expected and expected[-1] <= audit["native_end"], "Actual uniform/filter output coordinates differ")


def prepare_time_rows(axis, rows, *, preparation_release):
    """Deferred anti-alias preparation, NOT_RUN; no Torch is needed here.

    The later preparer must first verify raw artifact hashes, canonical IDs,
    roles and the source-frozen acquisition axis. This leaf does not bind files.
    Filter boundary extension is explicit; target coordinates remain native
    time coordinates with unverified units, never THz.
    """
    require(isinstance(preparation_release, dict) and all(preparation_release.get(k) is True for k in
            ("root_preparation_authorized", "quantum_terminal_reviewed", "raw_inputs_and_source_axes_frozen",
             "external_actual_CPU_supervisor_active")), "No numerical preparation during metadata/Q-only phase")
    import numpy as np
    from scipy.signal import resample_poly
    axis = np.asarray(axis, dtype=np.float64); values = np.asarray(rows, dtype=np.float64)
    require(axis.shape == (6750,) and values.ndim == 2 and values.shape[1] == 6750
            and np.isfinite(axis).all() and np.isfinite(values).all(), "Expected finite registered6750 time coordinates")
    uniform_values, audit = time_axis_profile(axis.tolist(), synthetic=preparation_release.get("synthetic_signal_control") is True)
    uniform_axis = np.asarray(uniform_values, dtype=np.float64)
    require(uniform_axis[0] >= axis[0] and uniform_axis[-1] <= axis[-1], "Uniformization cannot extrapolate")
    # The original rounded coordinates are never treated as already uniform.
    # No row interaction or fitting; all rows use the same frozen acquisition grid.
    interpolated = np.stack([np.interp(uniform_axis, axis, row) for row in values])
    delta = np.diff(uniform_axis)
    require(np.allclose(delta, audit["uniform_step"], rtol=1e-8, atol=1e-12), "Explicit uniformization failed")
    filtered = resample_poly(interpolated, up=512, down=3375, axis=1, window=("kaiser", 5.0), padtype="line")
    require(filtered.shape == (len(values), 1024) and np.isfinite(filtered).all(), "Unexpected anti-alias output")
    norms = np.linalg.norm(filtered, axis=1)
    require((norms > 1e-12).all(), "Zero filtered waveform blocks the task")
    target = uniform_axis[0] + np.arange(1024, dtype=np.float64) * audit["uniform_step"] * 3375 / 512
    require(target[0] >= axis[0] and target[-1] <= axis[-1], "Target time coordinates escaped the acquisition window")
    audit["output_axis_sha256_raw_float64"] = axis_float64_sha(target.tolist())
    return filtered / norms[:, None], target, {**TIME_RESAMPLING_BASE, "axis_audit": audit}


def validate_feature_pack(pack, *, source):
    base = {"schema", "task_id", "role", "record_ids", "entity_ids", "identity_aliases", "values", "axis_metadata"}
    require(set(pack) == base | ({"source_labels"} if source else set()), "Unexpected or truth-bearing feature fields")
    require(pack["schema"] == "research4-compact-multirepresentation-pack-v1", "Wrong prepared pack schema")
    require(pack["task_id"] in TASKS, "Unknown prepared task")
    require(pack["role"] == "source" if source else pack["role"] in ("inner_validation", "query"), "Wrong feature role")
    n = len(pack["record_ids"])
    require(n > 0 and len(set(pack["record_ids"])) == n and len(set(pack["entity_ids"])) == n,
            "Duplicate/empty record or entity list")
    require(all(isinstance(v, str) and v for v in list(pack["record_ids"]) + list(pack["entity_ids"])),
            "Explicit string IDs required outside features")
    require(len(pack["entity_ids"]) == len(pack["identity_aliases"]) == n, "Missing parent identity")
    require(all(isinstance(row, (list, tuple)) and row and all(isinstance(v, str) and v for v in row)
                for row in pack["identity_aliases"]), "Explicit per-record identity alias lists required")
    require(set(pack["values"]) == set(pack["axis_metadata"]) == set(FEATURES), "All paired/derived features required")
    require(all(len(pack["values"][m]) == n for m in FEATURES), "Cross-modality row mismatch")
    axis_keys = {"native_axis_fingerprint", "native_axis_fingerprint_definition", "points", "target_axis",
                 "axis_kind", "axis_unit", "quantity", "value_unit", "pointwise_transform", "pointwise_prepared",
                 "resampling", "derived_from"}
    transforms = {"alpha": "snv_population", "alpha_long": "snv_population", "n": "identity", "time": "l2_no_center"}
    quantities = {"alpha": "absorption_coefficient", "alpha_long": "absorption_coefficient", "n": "refractive_index", "time": "time_domain_amplitude"}
    for mode in FEATURES:
        meta = pack["axis_metadata"][mode]
        require(set(meta) == axis_keys and meta["points"] == POINTS[mode] and meta["pointwise_prepared"] is True,
                "Unknown or unprepared axis metadata fields")
        require(meta["pointwise_transform"] == transforms[mode] and meta["quantity"] == quantities[mode],
                "Quantity or row policy changed")
        axis = meta["target_axis"]
        require(len(axis) == POINTS[mode] and all(isinstance(v, (int, float)) and math.isfinite(v) for v in axis)
                and all(b > a for a, b in zip(axis, axis[1:])), "Invalid independent target axis")
        require(meta["axis_kind"] == ("native_time_coordinate_UNIT_UNVERIFIED" if mode == "time" else "frequency_THz"),
                "Do not treat time coordinates as THz")
        require(all(isinstance(meta[k], str) and meta[k] for k in
                    ("native_axis_fingerprint", "native_axis_fingerprint_definition", "axis_unit", "value_unit")),
                "Missing axis fingerprint or explicit unit status")
        if mode == "time":
            validate_time_resampling(meta["resampling"], axis)
            require(meta["native_axis_fingerprint"] == meta["resampling"]["axis_audit"]["native_axis_sha256_raw_float64"],
                    "Native coordinate fingerprint changed during interpolation")
            require(meta["derived_from"] == "registered_native_time6750", "Time input provenance changed")
        else:
            require(meta["resampling"] == {"method": "linear_no_extrapolation"}, "Frequency interpolation changed")
            require(abs(axis[0] - .21) <= 1e-12 and abs(axis[-1] - 1.49) <= 1e-12, "Frequency window changed")
    short, long = pack["axis_metadata"]["alpha"], pack["axis_metadata"]["alpha_long"]
    require(long["derived_from"] == "canonical_original_alpha612_not_alpha128"
            and short["native_axis_fingerprint"] == long["native_axis_fingerprint"]
            and short["native_axis_fingerprint_definition"] == long["native_axis_fingerprint_definition"],
            "Long alpha control must derive directly from the same canonical original alpha")
    if source:
        require(len(pack["source_labels"]) == n, "Source label count changed")


def assert_pack_separation(source, query):
    validate_feature_pack(source, source=True); validate_feature_pack(query, source=False)
    require(source["task_id"] == query["task_id"], "Cross-task features cannot share a fit")
    aliases = []
    for pack in (source, query):
        own = set(pack["record_ids"]) | set(pack["entity_ids"])
        for row in pack["identity_aliases"]:
            require(isinstance(row, (list, tuple)) and row, "Explicit per-row aliases required")
            own.update(str(v) for v in row)
        aliases.append(own)
    require(not aliases[0] & aliases[1], "Same record/entity/alias crosses fit and inference")


def require_runtime_release(release):
    required = ("root_execution_authorized", "quantum_terminal_reviewed", "prepared_inputs_numerically_audited",
                "source_scope_review_passed", "all_inputs_and_code_frozen", "external_actual_CPU_supervisor_active")
    require(isinstance(release, dict) and all(release.get(k) is True for k in required),
            "DRAFT_NOT_EXECUTABLE: root must bind data/code, review Q and release resources first")
    require(release.get("threads") == 1 and release.get("device") == "cpu", "CPU single-thread only")
    for key in ("protocol_sha256", "prepared_inputs_sha256", "code_sha256"):
        value = release.get(key, "")
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "Missing frozen release hash: " + key)
    own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(release["code_sha256"] == own, "Frozen draft implementation differs")


def numerical_runtime(release):
    """Never called by the metadata CLI or pure tests; deferred until after Q."""
    require_runtime_release(release)
    global _RUNTIME
    if _RUNTIME is None:
        import os
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "Set single-thread environment before numerical imports")
        import numpy as np
        import torch
        from sklearn.preprocessing import StandardScaler
        sys.path.insert(0, str(ROOT / "src"))
        from herb_spectra.upgrade_neural import Encoder
        from herb_spectra.leakage_guard import FitScope, audit_pointwise_inference
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        _RUNTIME = np, torch, StandardScaler, Encoder, FitScope, audit_pointwise_inference
    return _RUNTIME


def validate_fit_context(release, split_id, source, inference, phase, epochs, arm, seed):
    """Root's later bound context fixes actual IDs; a payload's role tag is insufficient."""
    context = release.get("fit_contexts", {}).get(split_id)
    require(isinstance(context, dict) and context.get("task_id") == source["task_id"]
            and context.get("phase") == phase and context.get("epochs") == epochs
            and context.get("arm") == arm and context.get("seed") == seed,
            "Missing or changed frozen fit context")
    require(context.get("source_ids") == list(source["record_ids"]), "Actual source IDs differ from frozen allowance")
    actual = [] if inference is None else list(inference["record_ids"])
    require(context.get("inference_ids") == actual, "Inference IDs differ from frozen allowance")


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def build_model(torch, Encoder, arm, n_classes, seed):
    require(arm in ARMS and seed in SEEDS and n_classes in (4, 7, 10), "Unregistered architecture")
    torch.manual_seed(seed)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = torch.nn.ModuleList([Encoder(12, 2, .35, True, False) for _ in ARMS[arm]])
            self.head = torch.nn.Linear(16 * len(ARMS[arm]), n_classes)
        def forward(self, inputs):
            return self.head(torch.cat([encoder(inputs[name]) for encoder, name in zip(self.encoders, ARMS[arm])], 1))
    return Model()


def fit_source_leaf(source, *, validation=None, arm, epochs, seed, directory, split_id, release,
                    budget_check, collect_source_epoch_predictions=False):
    """Deferred implementation only, NOT_RUN. Caller binds packs and owns staging.

    budget_check is mandatory and must enforce the externally supervised whole
    experiment budget, not create a fresh per-model budget. Real query features
    and truth are never accepted. Inner-validation truth is joined later using
    source IDs only. Final fits take source alone; prediction is another function.
    """
    validate_feature_pack(source, source=True)
    if validation is not None:
        assert_pack_separation(source, validation)
        require(validation["role"] == "inner_validation", "Source fitting never accepts a real query")
    require(seed in SEEDS and arm in ARMS and epochs in EPOCHS, "Unregistered fit")
    require(callable(budget_check), "Whole-run CPU/RSS supervisor callback required")
    require(not collect_source_epoch_predictions or (arm == "alpha_single" and epochs == 96
            and validation is not None), "Epoch predictions only for source alpha CV")
    phase = "source_alpha_cv" if validation is not None else "final_source_fit"
    require(validation is None or collect_source_epoch_predictions, "Only registered source-alpha CV uses validation")
    validate_fit_context(release, split_id, source, validation, phase, epochs, arm, seed)
    if validation is None:
        require(release.get("all_task_source_epoch_locks_frozen") is True, "All source choices must lock first")
    budget_check()
    np, torch, Scaler, Encoder, FitScope, inference_audit = numerical_runtime(release)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    ids = np.asarray(source["record_ids"]); labels = np.asarray(source["source_labels"])
    classes = sorted(set(labels.tolist()))
    require(len(classes) == TASKS[source["task_id"]][3], "Source class coverage changed")
    scope = FitScope(ids, split_id + "." + arm, "supplemental-multirepresentation-draft-v1",
                     directory / "fit_scope.jsonl", code_version=release["code_sha256"])
    used = tuple(dict.fromkeys(ARMS[arm])); x, q, scalers = {}, {}, {}
    for mode in used:
        meta = source["axis_metadata"][mode]
        require(meta.get("points") == POINTS[mode] and meta.get("pointwise_prepared") is True,
                "Prepared independent-axis contracts differ")
        raw = np.asarray(source["values"][mode], dtype=np.float64)
        require(raw.shape == (len(ids), POINTS[mode]) and np.isfinite(raw).all(), "Invalid compact source values")
        scaler = scope.fit(Scaler(), raw, sample_ids=ids, component=mode + "_source_standard_scaler")
        x[mode] = scaler.transform(raw).astype(np.float32)
        if validation is not None:
            require(meta == validation["axis_metadata"][mode], "Validation has a changed independent axis")
            pending = np.asarray(validation["values"][mode], dtype=np.float64)
            require(pending.shape == (len(validation["record_ids"]), POINTS[mode]) and np.isfinite(pending).all(),
                    "Invalid source-validation values")
            q[mode] = scaler.transform(pending).astype(np.float32)
        scalers[mode] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    class_index = {label: i for i, label in enumerate(classes)}
    y = np.asarray([class_index[v] for v in labels], dtype=np.int64)
    weights = np.asarray([len(y) / (len(classes) * sum(y == i)) for i in range(len(classes))], dtype=np.float32)
    model = build_model(torch, Encoder, arm, len(classes), seed)
    initial_state = state_hash(model)
    initial_encoders = [state_hash(encoder) for encoder in model.encoders]
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=.001)
    torch.manual_seed(seed + 50000)
    generator = torch.Generator().manual_seed(seed + 60000)
    losses, checkpoint_logits, schedule = [], {}, []
    started = time.process_time()
    try:
        for epoch in range(1, epochs + 1):
            budget_check(); order = torch.randperm(len(ids), generator=generator).numpy()
            schedule.append(order.tolist()); model.train(); total = 0.
            for left in range(0, len(ids), 64):
                budget_check(); chosen = order[left:left + 64]
                actual = {m: torch.as_tensor(x[m][chosen]) for m in used}
                # Concatenation is only the FitScope fingerprint, never a shared physical CNN axis.
                audit_values = np.concatenate([x[m][chosen] for m in used], axis=1)
                with scope.operation("actual_neural_batch", ids[chosen], X=audit_values, y=y[chosen]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(actual), torch.as_tensor(y[chosen]),
                                                            weight=torch.as_tensor(weights))
                    require(bool(torch.isfinite(loss)), "Nonfinite loss; preserve failed run")
                    loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(chosen)
            losses.append(total / len(ids)); model.eval()
            if collect_source_epoch_predictions and epoch in EPOCHS:
                with torch.no_grad():
                    checkpoint_logits[epoch] = model({m: torch.as_tensor(q[m]) for m in used}).cpu().numpy()
        audit = None
        if validation is not None:
            audit = inference_audit(model, q, batch_sizes=(1, 3), atol=1e-5, rtol=1e-5)
        budget_check()
        torch.save(model.state_dict(), directory / "model.pt")
        if validation is not None:
            np.savez_compressed(directory / "source_epoch_predictions.npz",
                                record_ids=np.asarray(validation["record_ids"]), class_labels=np.asarray(classes),
                                **{"epoch_" + str(k): v for k, v in checkpoint_logits.items()})
        metadata = {"status": "SOURCE_FITTED_NO_QUERY_ACCESS", "task_id": source["task_id"], "arm": arm,
                    "seed": seed, "epochs": epochs, "source_ids": ids.tolist(), "class_labels": classes,
                    "source_axis_metadata": source["axis_metadata"],
                    "validation_ids": [] if validation is None else validation["record_ids"],
                    "source_scalers": scalers, "source_class_weights": weights.tolist(), "source_epoch_loss": losses,
                    "source_batch_order_sha256": objhash(schedule), "branch_spec": arm_spec(arm),
                    "initial_state_sha256": initial_state, "initial_encoder_sha256": initial_encoders,
                    "fitted_state_sha256": state_hash(model),
                    "parameter_count": sum(p.numel() for p in model.parameters()), "inference_audit": audit,
                    "query_features_received": False, "query_labels_received": False, "OS_truth_sandbox": False,
                    "fit_process_CPU_seconds": time.process_time() - started}
        write_json(directory / "training.json", metadata)
        return metadata
    except BaseException as exc:
        write_json(directory / "failure.json", {"status": "FAILED_KEEP_ARTIFACTS", "reason": str(exc),
                   "completed_epochs": len(losses), "query_labels_received": False})
        raise


def predict_frozen_leaf(query, *, model_key, release, directory, budget_check):
    """Deferred label-free prediction after all 36 final models are hash-locked."""
    validate_feature_pack(query, source=False)
    require(query["role"] == "query" and release.get("all_final_models_frozen") is True,
            "Real-query prediction requires the separate complete-model release")
    record = release.get("frozen_models", {}).get(model_key)
    require(isinstance(record, dict) and record.get("query_ids") == list(query["record_ids"]), "Unbound query/model")
    require(callable(budget_check), "Whole-run resource callback required")
    np, torch, _, Encoder, _, inference_audit = numerical_runtime(release)
    def checked(path_key, hash_key):
        path = (ROOT / record[path_key]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Missing/outside frozen model artifact")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record[hash_key], "Frozen model artifact changed")
        return path
    training = json.loads(checked("training_path", "training_sha256").read_text())
    weights = checked("model_path", "model_sha256")
    require(training["task_id"] == query["task_id"] and not training["query_features_received"]
            and not training["validation_ids"], "Final prediction requires a source-only final fit")
    arm, seed, classes = training["arm"], training["seed"], training["class_labels"]
    source_aliases = set(record.get("source_identity_aliases", []))
    require(source_aliases, "Frozen source parent/alias union required")
    incoming = set(query["record_ids"]) | set(query["entity_ids"])
    for row in query["identity_aliases"]:
        incoming.update(row)
    require(not incoming & source_aliases, "Query intersects a source ancestor or alias")
    used = tuple(dict.fromkeys(ARMS[arm])); q = {}
    for mode in used:
        require(training["source_axis_metadata"][mode] == query["axis_metadata"][mode], "Query axis contract changed")
        raw = np.asarray(query["values"][mode], dtype=np.float64)
        require(raw.shape == (len(query["record_ids"]), POINTS[mode]) and np.isfinite(raw).all(), "Invalid query values")
        scaler = training["source_scalers"][mode]
        q[mode] = ((raw - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])).astype(np.float32)
    budget_check(); model = build_model(torch, Encoder, arm, len(classes), seed)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)); model.eval()
    require(state_hash(model) == training["fitted_state_sha256"], "Frozen state replay differs")
    with torch.no_grad():
        logits = model({m: torch.as_tensor(q[m]) for m in used}).cpu().numpy()
    audit = inference_audit(model, q, batch_sizes=(1, 3), atol=1e-5, rtol=1e-5)
    budget_check(); directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "predictions.npz", record_ids=np.asarray(query["record_ids"]),
                        class_labels=np.asarray(classes), logits=logits)
    result = {"status": "LABEL_FREE_PREDICTIONS_AWAIT_INDEPENDENT_SCORING", "model_key": model_key,
              "query_ids": query["record_ids"], "inference_audit": audit, "query_labels_received": False}
    write_json(directory / "prediction.json", result)
    return result


def audit_capacity_pair(capacity, multimodal):
    """Audit real later metadata; pure fixtures can test these guards now."""
    require(capacity["arm"] == "alpha_capacity3" and multimodal["arm"] == "alpha_n_time", "Wrong comparison arms")
    for key in ("task_id", "seed", "epochs", "source_ids", "source_batch_order_sha256",
                "initial_state_sha256", "initial_encoder_sha256", "parameter_count"):
        require(capacity[key] == multimodal[key], "Capacity/real-modality pairing differs: " + key)
    require(not capacity["query_features_received"] and not multimodal["query_features_received"],
            "Final fitting accessed real query features")
    return {"status": "TECHNICAL_METADATA_PAIRING_PASS", "is_accuracy_or_independent_sample_evidence": False}


def metadata_plan(protocol):
    require(protocol["status"] == "DRAFT_METADATA_ONLY_NOT_EXECUTABLE", "This metadata entry is not a release")
    require(protocol["seeds"] == list(SEEDS) and protocol["epoch_selection"]["candidates"] == list(EPOCHS),
            "Draft seed/epoch grid changed")
    require(protocol["arm_order"] == list(ARMS), "All three controls required")
    for arm in ARMS:
        declared = protocol["arms"][arm]
        require(declared["branch_inputs"] == list(ARMS[arm])
                and declared["branch_points"] == [POINTS[name] for name in ARMS[arm]]
                and declared["embedding_dimensions"] == 16 * len(ARMS[arm])
                and declared["encoders_share_weights"] is False,
                "Arm inputs, branch lengths, embeddings or independent weights changed")
    architecture = protocol["architecture"]
    require(all(architecture[k] == v for k, v in {"width": 12, "blocks": 2, "dropout": .35,
            "residual": True, "attention": False, "branch_embedding": 16}.items()), "Frozen architecture recipe changed")
    require(architecture["source_class"] == "src/herb_spectra/upgrade_neural.py:Encoder", "Backbone source changed")
    training = protocol["training"]
    require(all(training[k] == v for k, v in {"learning_rate": .0005, "weight_decay": .001,
            "batch_size": 64, "pretraining": False, "calibration": False, "optimizer": "Adam",
            "device": "cpu", "threads": 1, "early_stopping_on_query": False}.items()), "Optimizer/control recipe changed")
    require(protocol["preprocessing"]["points_by_prepared_feature"] == POINTS
            and all(protocol["preprocessing"][m]["window"] == [.21, 1.49] for m in ("alpha", "n", "alpha_long")),
            "Independent frequency windows changed")
    long = protocol["preprocessing"]["alpha_long"]
    require(long["interpolation"] == "direct_pointwise_linear_from_original_canonical612_alpha_not_from_alpha128"
            and long["derived_feature_not_new_information"] is True, "Long alpha must derive directly from canonical612")
    require(protocol["preprocessing"]["alpha"]["row_transform"] == long["row_transform"] == "SNV_population_std"
            and protocol["preprocessing"]["n"]["row_transform"] == "identity_preserve_between_specimen_mean"
            and protocol["preprocessing"]["time"]["row_transform"] == "divide_by_L2_norm_no_mean_subtraction_after_anti_alias_filter",
            "Fixed per-representation row policies changed")
    require(protocol["preprocessing"]["time"]["resampling"] == {
            "method": "within_span_linear_uniformization_then_scipy.signal.resample_poly", "input_points": 6750,
            "uniform_points": 6750, "output_points": 1024, "up": 512, "down": 3375, "window": ["kaiser", 5.0], "padtype": "line",
            "uniformization": "endpoint_linspace_6750_pointwise_linear_no_extrapolation", "native_coordinate_max_deviation_in_steps": .001,
            "native_step_max_relative_deviation": .001, "uniform_axis_rtol": 1e-8, "uniform_axis_atol": 1e-12,
            "frozen_actual_native_axis_sha256": NATIVE_TIME_AXIS_SHA,
            "frozen_actual_uniform_axis_sha256": UNIFORM_TIME_AXIS_SHA}, "Fixed two-step anti-alias recipe changed")
    expected = {row["id"]: row for row in protocol["tasks"]}
    require(set(expected) == set(TASKS), "All fixed tasks required")
    for task_id, (source, support, query, classes) in TASKS.items():
        row = expected[task_id]
        require(row["classes"] == classes and row["expected_roles"] == {"source": source, "support": support, "query": query},
                "Original role counts changed")
        if task_id != "zaoren_expanded10":
            require(row["source_cv_folds"] == 3 and row["inherit_epoch_from"] is None, "Only source3fold selection allowed")
    require(expected["zaoren_expanded10"]["source_cv_folds"] == 0
            and expected["zaoren_expanded10"]["inherit_epoch_from"] == "zaoren_legacy7", "No singleton CV")
    require(protocol["binding"]["execution_release"] is None, "Never execute using a draft metadata plan")
    return {"status": "NOT_RUN_DRAFT_METADATA_ONLY", "expected_source_alpha_fits": 27,
            "expected_final_fits": 36, "fits_executed": 0, "numerical_inputs_read": 0,
            "runtime_imported": _RUNTIME is not None, "missing_before_execution": [
                "prepare and numerically audit new aligned compact role packs", "bind native time axes and all input/code hashes",
                "independent leakage review and whole-run resource supervisor", "root execution release after quantum terminal review",
                "source-stage driver, immutable locks, separate prediction and scoring release"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/research/supplemental_multirepresentation_draft_20261104_v2.json")
    parser.add_argument("--plan", action="store_true", required=True)
    args = parser.parse_args()
    # This CLI has no execute/train command and cannot import numerical libraries.
    print(json.dumps(metadata_plan(json.loads(args.protocol.read_text())), indent=2))


if __name__ == "__main__":
    main()
