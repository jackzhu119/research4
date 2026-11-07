"""Finite engineering stress tests of five frozen historical R2 OOF models.

No fitting, selection, calibration, new specimens, or independent confirmation.
``prepare`` freezes exact provenance and projects the already-labelled historical
CSV into a truth-free baseline reference (preparation parses the old labelled
rows, then discards truth fields; this is not a claim of never-exposed truth).
``run`` needs a root-reviewed release, replays the exact historical predictions,
writes/locks every perturbed prediction, and only then opens scoring truth.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits


SCHEMA = "frozen-r2-oof-engineering-robustness-20261107-v1"
LOCK_STATUS = "ALL_NATIVE_PREDICTIONS_LOCKED_BEFORE_SCORING_TRUTH_ACCESS"
HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
CORE = "reports/staged-nested-optimization/2026-10-31/core"
GENERATION = "reports/approved-zaoren/2026-10-31/datasets/generation.json"
QUALITY = "reports/approved-zaoren/2026-10-31/data-preparation/quality_report.json"
REGISTRY = "data/active_workbooks.json"
DECLARATION = "data/annotations/zaoren_user_declaration_20261031.json"
TEST_FILE = "tests/test_frozen_oof_robustness_20261107.py"
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
LIMITATIONS = [
    "All 855 records were exposed historical development data; this is not independent confirmation.",
    "Each ZY record is user-declared independent; original collection ancestry was not independently verified.",
    "One source lot per anonymous origin group; origin and source-lot effects cannot be separated.",
    "Synthetic corruptions are engineering spectral-shape tests, not validated instrument noise, batch transfer, or chemical effects.",
    "Repeated corruption seeds and cases reuse the same 855 records; they do not increase specimen count.",
    "ZY02 has five records; per-class counts and denominators must accompany macro metrics.",
    "Frozen SVC.predict vote rules are preserved; OVR scores are not probabilities or an argmax prediction replacement.",
    "No p-values, biological confidence intervals, adaptation, refitting, parameter selection, or model ranking from corruption scores.",
    "Unclipped original alpha values, including high-frequency negatives, are retained.",
    "Preparation parses the already-labelled historical prediction CSV to project a truth-free reference; run-phase scoring access is logically separated, not OS-sandboxed or never-exposed truth.",
]


class Blocked(RuntimeError):
    """A frozen prerequisite or run budget failed; never substitute a model."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def conditions(axis_enabled=True):
    cases = [{"case": "baseline", "kind": "noise", "level": 0., "seed": None}]
    for level in (.001, .005, .01):
        for seed in SEEDS:
            cases.append({"case": f"noise_{level:g}_seed{seed}", "kind": "noise", "level": level, "seed": seed})
    for kind, levels in (("gain", (.9, 1.1)), ("constant_baseline", (-.05, .05)),
                         ("linear_baseline", (-.05, .05))):
        for level in levels:
            cases.append({"case": f"{kind}_{level:g}", "kind": kind, "level": level, "seed": None})
    if axis_enabled:
        for level in (-.002, .002):
            cases.append({"case": f"axis_shift_{level:g}_THz", "kind": "axis_shift", "level": level, "seed": None})
    return cases


def raw_arrays(path):
    """Read no labels: unique groups make the official aggregate exactly a sort."""
    with np.load(path, allow_pickle=False) as z:
        x, axis = z["X_alpha"], z["axis"]
        ids, groups = z["record_ids"].astype(str), z["groups"].astype(str)
    if x.ndim != 2 or x.shape != (855, len(axis)) or not np.isfinite(x).all():
        raise Blocked("Original 855 full alpha matrix is missing, invalid, or changed")
    if axis.ndim != 1 or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
        raise Blocked("Frequency axis must be finite and strictly increasing")
    if len(set(groups)) != 855 or not np.array_equal(ids, groups):
        raise Blocked("Not the frozen one-record/one-group input; do not guess aggregation")
    order = np.argsort(groups)
    return np.asarray(x[order], dtype=float), np.asarray(axis, dtype=float), groups[order]


def source_sd(source):
    """One source-only scalar: RMS of full-frequency population SDs, ddof=0."""
    x = np.asarray(source, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise Blocked("Invalid source partition for engineering amplitude")
    sd = float(np.sqrt(np.mean(np.var(x, axis=0, ddof=0))))
    if not np.isfinite(sd) or sd <= 0:
        raise Blocked("Nonpositive source-fold RMS SD")
    return sd


def corrupt(x, axis, sd, spec, fold):
    """Raw full-spectrum intervention; never fits or inspects query labels."""
    x = np.asarray(x, dtype=float)
    level, kind = spec["level"], spec["kind"]
    if kind == "noise":
        if level == 0:
            return x.copy()
        # Independent fold streams; the same standard-normal stream is reused
        # across magnitudes for a given seed, keeping level curves paired.
        rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], fold]))
        return x + level * sd * rng.standard_normal(x.shape)
    if kind == "gain":
        return x * level
    if kind == "constant_baseline":
        return x + level * sd
    if kind == "linear_baseline":
        ramp = 2 * (axis - axis[0]) / (axis[-1] - axis[0]) - 1
        return x + level * sd * ramp
    if kind == "axis_shift":
        # Positive delta means x_delta(f)=x(f+delta). Full-grid interpolation
        # precedes the historical model mask. Outside original export: clamp.
        return np.asarray([np.interp(axis + level, axis, row) for row in x])
    raise Blocked("Unknown, unfrozen corruption kind")


def validate_splits(splits, ids):
    known, all_query = set(ids), []
    if [s["fold"] for s in splits] != list(range(5)):
        raise Blocked("Exactly the five historical outer folds are required")
    for s in splits:
        train, query = s["train_sample_ids"], s["test_sample_ids"]
        if len(train) != 684 or len(query) != 171 or len(set(train)) != 684 or len(set(query)) != 171:
            raise Blocked("Unexpected fold cardinality or duplicate identity")
        if set(train) & set(query) or set(train) | set(query) != known:
            raise Blocked("Source/query identity overlap or missing sample")
        all_query.extend(query)
    if len(all_query) != 855 or set(all_query) != known:
        raise Blocked("Each original sample must occur in query exactly once")


def model_band(raw, mask):
    # Official preparation masks the full matrix first, then advanced-indexes
    # source/query rows, which produces a C-contiguous float64 matrix. Match
    # that layout too: joblib's historical training fingerprint includes strides.
    return np.ascontiguousarray(np.asarray(raw, dtype=float)[:, mask])


def baseline_reference(path):
    with gzip.open(path, "rt", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "R2"]
    if len(rows) != 855 or len({r["record_id"] for r in rows}) != 855:
        raise Blocked("Historical R2 reference does not contain 855 unique records")
    # Never copy y_true or is_correct into the inference-readable reference.
    return [{k: v for k, v in r.items() if k in ("record_id", "outer_fold", "y_predicted") or k.startswith("score_")}
            for r in rows]


def verify_replay(predictions, reference, atol=1e-12):
    p = {r["record_id"]: r for r in predictions}
    r = {r["record_id"]: r for r in reference}
    if len(p) != len(predictions) or len(r) != len(reference) or set(p) != set(r):
        raise Blocked("Baseline identity mismatch or duplicate predictions")
    maximum = 0.
    for identifier, ref in r.items():
        obs = p[identifier]
        if int(obs["outer_fold"]) != int(ref["outer_fold"]) or obs["y_predicted"] != ref["y_predicted"]:
            raise Blocked("Baseline prediction/fold differs from the frozen historical R2")
        keys = {k for k in ref if k.startswith("score_")}
        if {k for k in obs if k.startswith("score_")} != keys:
            raise Blocked("Baseline class/score columns differ")
        for k in keys:
            delta = abs(float(obs[k]) - float(ref[k]))
            if not np.isfinite(delta) or delta > atol:
                raise Blocked("Baseline decision scores differ from the frozen historical R2")
            maximum = max(maximum, delta)
    return {"identical_predictions": len(p), "scores_max_abs_difference": maximum,
            "scores_absolute_tolerance": atol, "prediction_rule": "historical SVC.predict voting"}


def metric_counts(truth, predicted, classes):
    """Independent confusion-count arithmetic, with all seven classes retained."""
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    if len(truth) != len(predicted) or not len(truth):
        raise Blocked("Scoring row mismatch or empty prediction")
    for t, p in zip(truth, predicted, strict=True):
        if t not in index or p not in index:
            raise Blocked("Unknown scoring class")
        cm[index[t], index[p]] += 1
    support, predicted_count = cm.sum(1), cm.sum(0)
    if np.any(support == 0):
        raise Blocked("Every fixed class must have a query denominator")
    tp = np.diag(cm)
    recalls = tp / support
    denominator = support + predicted_count
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(classes), dtype=float), where=denominator > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recalls.mean()), "correct": int(tp.sum()), "samples": int(cm.sum()),
            "confusion_matrix": cm.tolist(), "per_class": [
                {"class": c, "support": int(support[i]), "correct": int(tp[i]),
                 "predicted": int(predicted_count[i]), "recall": float(recalls[i]),
                 "precision": float(tp[i] / predicted_count[i]) if predicted_count[i] else 0.,
                 "f1": float(f1[i])} for i, c in enumerate(classes)]}


def write_csv(path, rows):
    if not rows:
        raise Blocked("Refuse empty evidence file")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(repo, output):
    """No model loading/prediction/fitting. Freeze source identities and hashes."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists():
        raise Blocked("Preparation output already exists; never overwrite a run")
    core = repo / CORE
    generation, quality, registry = (read_json(repo / p) for p in (GENERATION, QUALITY, REGISTRY))
    manifest, status = read_json(core / "run_manifest.json"), read_json(core / "status.json")
    for name, expected in status["result_sha256"].items():
        if sha(core / name) != expected:
            raise Blocked("Historical core result hash mismatch")
    if status["status"] != "completed":
        raise Blocked("Historical core run incomplete")
    source_files = manifest["source_sha256"]
    for rel, expected in source_files.items():
        if sha(repo / rel) != expected:
            raise Blocked("Official historical inference source changed")
    full, candidate = repo / generation["source_npz"], repo / generation["output_npz"]
    if sha(full) != generation["source_sha256"] or sha(candidate) != generation["output_sha256"]:
        raise Blocked("Full/candidate generation hash chain failed")
    if generation["output_sha256"] != manifest["input_sha256"] or quality["dataset_sha256"] != generation["source_sha256"]:
        raise Blocked("Dataset is not the exact historical R2 input lineage")
    registry_sources = {w["path"]: w["sha256"] for w in registry["workbooks"] if w["dataset"] == "zaoren"}
    qa_sources = {s["path"]: s["sha256"] for s in quality["sources"]}
    if registry_sources != qa_sources or len(qa_sources) != 7:
        raise Blocked("Active seven-workbook registry does not match historical raw source QA")
    for rel, expected in qa_sources.items():
        if sha(repo / rel) != expected:
            raise Blocked("Active workbook source bytes changed")
    axis_enabled = all(str(s["axis_headers"]["alpha"]).strip().lower() == "thz" for s in quality["sources"])
    x, axis, ids = raw_arrays(full)
    xc, axisc, idsc = raw_arrays(candidate)
    protocol = read_json(core / "protocol_snapshot.json")
    mask = (axis >= protocol["frequency"]["lower_inclusive"]) & (axis <= protocol["frequency"]["upper_inclusive"])
    if x.shape != (855, 2578) or xc.shape != (855, 612) or not np.array_equal(ids, idsc):
        raise Blocked("Unexpected historical raw/candidate shapes or identities")
    if not np.array_equal(axis[mask], axisc) or not np.array_equal(x[:, mask], xc):
        raise Blocked("Full raw alpha is not numerically identical to the historical masked candidate")
    splits = read_json(core / "splits.json")
    validate_splits(splits, ids)
    models = []
    for row in read_json(core / "outer_metrics.json"):
        if row["method"] != "R2":
            continue
        model = repo / row["model_path"]
        side = repo / "outputs/staged-nested-20261031/core/deliverable_models" / f"outer_{row['outer_fold']:02d}" / model.with_suffix(".json").name
        if not model.exists():
            model = side.with_suffix(".joblib")
        if not model.exists() or sha(model) != row["model_sha256"] or not side.exists():
            raise Blocked("Exact historical full R2 model/sidecar unavailable; never reconstruct")
        meta = read_json(side)
        split = splits[row["outer_fold"]]
        if meta["model_sha256"] != row["model_sha256"] or meta["source_ids"] != split["train_sample_ids"] or meta["outer_query_ids"] != split["test_sample_ids"]:
            raise Blocked("Model sidecar identity/provenance mismatch")
        if not any(r["method"] == "R2" and r["recipe"] == row["config"] for r in meta["roles"]):
            raise Blocked("Historical R2 role/recipe missing from exact model")
        models.append({"fold": row["outer_fold"], "model": str(model.relative_to(repo)),
                       "model_sha256": row["model_sha256"], "sidecar": str(side.relative_to(repo)),
                       "sidecar_sha256": sha(side), "config": row["config"]})
    models.sort(key=lambda r: r["fold"])
    if [m["fold"] for m in models] != list(range(5)):
        raise Blocked("Missing or duplicate R2 fold model")
    reference = baseline_reference(core / "outer_predictions.csv.gz")
    paths = [GENERATION, QUALITY, REGISTRY, DECLARATION, *qa_sources, *source_files,
             *(str((core / p).relative_to(repo)) for p in
               ("run_manifest.json", "status.json", "protocol_snapshot.json", "splits.json", "outer_metrics.json", "outer_predictions.csv.gz", "summary.json")),
             str(full.relative_to(repo)), str(candidate.relative_to(repo)), str(HERE.relative_to(repo)), TEST_FILE]
    for model in models:
        paths.extend((model["model"], model["sidecar"]))
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "joblib")}
    if versions != manifest["versions"]:
        raise Blocked("Runtime versions differ from the original native-model inference environment")
    output.mkdir(parents=True)
    write_csv(output / "baseline_reference_without_truth.csv.gz", reference)
    plan = {"schema": SCHEMA, "status": "FROZEN_NOT_EXECUTED", "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "expected_historical_baseline_correct": 821, "expected_samples": 855, "expected_features": 612,
            "full_input": str(full.relative_to(repo)), "candidate_input": str(candidate.relative_to(repo)),
            "historical_core": CORE, "historical_versions": versions, "models": models,
            "input_code_hashes": {p: sha(repo / p) for p in sorted(set(paths))},
            "reference_file": "baseline_reference_without_truth.csv.gz",
            "reference_sha256": sha(output / "baseline_reference_without_truth.csv.gz"),
            "axis_shift_status": "VERIFIED_SOURCE_EXPORT_HEADER_THZ" if axis_enabled else "BLOCKED_UNKNOWN_FREQUENCY_UNIT",
            "conditions": conditions(axis_enabled), "stochastic_seeds": list(SEEDS),
            "full_spectrum_points": 2578, "mask_bounds_inclusive": [.1, 1.5],
            "source_sd_definition": "sqrt(mean(var(raw full-frequency source X_alpha, axis=0, ddof=0))) from 684 current outer-source records only",
            "noise_definition": "iid standard normal times level times scalar source SD; seed stream SeedSequence([seed, outer_fold]); same stream across noise levels",
            "linear_baseline_definition": "level * scalar source SD * linspace-by-frequency(-1,+1) on original full axis",
            "axis_shift_definition": "x_delta(f)=x(f+delta); linear interpolation on original full export, constant endpoint clamp, then original fixed mask",
            "prediction_definition": "native frozen StagedSVM.predict/decision_function; no argmax replacement",
            "baseline_gate": "855 exact historical IDs/folds/predictions; every OVR score absolute difference<=1e-12; no perturbation if failed",
            "prep_label_bearing_file_access": True,
            "prep_label_bearing_file_access_purpose": "projection_only; discard historical y_true/is_correct fields; no label-driven choice",
            "scoring_access": "Preparation parses the historically exposed labelled prediction CSV only to project a reference without y_true/is_correct; run does not index NPZ y and parses historical y_true only after all new prediction files have been locked. Logical phase separation, not OS isolation or never-exposed truth.",
            "limits": {"cpu_seconds": 600, "wall_seconds": 1200, "threads": 2, "fit_calls_allowed": 0},
            "limitations": LIMITATIONS}
    write_json(output / "protocol.json", plan)
    return {"status": plan["status"], "protocol": str(output / "protocol.json"),
            "protocol_sha256": sha(output / "protocol.json"), "models": 5, "cases": len(plan["conditions"]),
            "baseline_replay_status": "NOT_RUN_REQUIRES_ROOT_RELEASE", "training": False}


def check_frozen(repo, protocol_path, release_path):
    plan = read_json(protocol_path)
    if plan.get("schema") != SCHEMA or plan.get("conditions") != conditions(plan["axis_shift_status"] == "VERIFIED_SOURCE_EXPORT_HEADER_THZ"):
        raise Blocked("Protocol schema or prespecified conditions changed")
    release = read_json(release_path)
    if release.get("approved") is not True or release.get("protocol_sha256") != sha(protocol_path):
        raise Blocked("Root-reviewed source freeze release is absent or does not match")
    for rel, expected in plan["input_code_hashes"].items():
        if sha(Path(repo) / rel) != expected:
            raise Blocked("Frozen source/input/model changed: " + rel)
    for name, expected in plan["historical_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise Blocked("Frozen runtime package changed: " + name)
    if sha(Path(protocol_path).parent / plan["reference_file"]) != plan["reference_sha256"]:
        raise Blocked("Truth-free baseline reference changed")
    return plan


def forbid_fit(*args, **kwargs):
    raise Blocked("A fit/selection function was called in frozen inference")


def fit_blockers():
    from herb_spectra import staged_nested_search as staged
    stack = ExitStack()
    for obj, name in ((staged, "fit_source_candidate"), (staged, "fit_transform_bundle"),
                      (staged, "_prefix_selector"), (staged.StandardScaler, "fit"),
                      (staged.SVC, "fit"), (staged.MCUVESelector, "fit")):
        stack.enter_context(patch.object(obj, name, forbid_fit))
    return stack


def native_rows(model, x, ids, fold):
    # This callable has no y parameter. Scores are diagnostic, not probabilities.
    predicted = np.asarray(model.predict(x)).astype(str)
    scores = np.asarray(model.predict_scores(x), dtype=float)
    classes = np.asarray(model.classes_).astype(str)
    if classes.tolist() != [f"ZY{i:02d}" for i in range(1, 8)]:
        raise Blocked("Native R2 class order differs from the historical seven groups")
    if predicted.shape != (len(ids),) or scores.shape != (len(ids), 7) or not np.isfinite(scores).all():
        raise Blocked("Unexpected native inference output")
    return [{"record_id": str(i), "outer_fold": int(fold), "y_predicted": str(p),
             **{"score_" + c: float(v) for c, v in zip(classes, s, strict=True)}}
            for i, p, s in zip(ids, predicted, scores, strict=True)]


def validate_native_model(model, config, source_candidate):
    from herb_spectra.staged_nested_search import StagedSVM
    # Historical saving aliases numerically identical recipes across roles;
    # the descriptive candidate ID is intentionally absent from that cache key.
    if not isinstance(model, StagedSVM) or {k: v for k, v in model.config.items() if k != "id"} != {k: v for k, v in config.items() if k != "id"}:
        raise Blocked("Loaded object is not the exact official historical R2 recipe")
    if joblib.hash(np.asarray(source_candidate, dtype=float)) != model.metadata["source_value_fingerprint"]:
        raise Blocked("Actual ordered source alpha fingerprint differs from native training metadata")
    if model.metadata["prediction_rule"] != "SVC standard vote, break_ties=False":
        raise Blocked("Native SVC historical voting policy changed")


def score_locked(repo, plan, output, expected_protocol_sha256):
    lock = read_json(output / "prediction_lock.json")
    expected_cases = {spec["case"] for spec in plan["conditions"]}
    expected_files = {f"predictions/{name}.csv.gz" for name in expected_cases} | {
        "baseline_replay.json", "source_fold_amplitudes.csv", "corruption_amplitude_diagnostics.csv", "axis_boundary_diagnostic.json"}
    if lock.get("status") != LOCK_STATUS or lock.get("protocol_sha256") != expected_protocol_sha256:
        raise Blocked("Missing or mismatched all-predictions/protocol lock")
    if len(expected_cases) != len(plan["conditions"]) or set(lock.get("files_sha256", {})) != expected_files:
        raise Blocked("Prediction lock must cover every exact prespecified case and audit artifact")
    for name, expected in lock["files_sha256"].items():
        if sha(output / name) != expected:
            raise Blocked("Prediction lock changed before scoring")
    replay = read_json(output / "baseline_replay.json")
    if replay.get("status") != "PASS_EXACT_HISTORICAL_PREDICTIONS" or replay.get("identical_predictions") != 855:
        raise Blocked("Exact historical baseline replay gate is missing")
    replay_difference = replay.get("scores_max_abs_difference", float("inf"))
    if not np.isfinite(replay_difference) or replay_difference > 1e-12:
        raise Blocked("Historical baseline decision-score replay gate failed")
    splits = read_json(Path(repo) / plan["historical_core"] / "splits.json")
    expected_ids = [identifier for split in splits for identifier in split["test_sample_ids"]]
    validate_splits(splits, expected_ids)
    expected_folds = {identifier: split["fold"] for split in splits for identifier in split["test_sample_ids"]}
    expected_classes = [f"ZY{i:02d}" for i in range(1, 8)]
    expected_fields = {"record_id", "outer_fold", "y_predicted", *("score_" + c for c in expected_classes)}
    case_rows = {}
    # Validate the complete locked prediction population before opening truth.
    for case in expected_cases:
        with gzip.open(output / "predictions" / (case + ".csv.gz"), "rt", newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames or []) != expected_fields:
                raise Blocked("Locked prediction schema/class order fields changed")
            rows = list(reader)
        if len(rows) != 855 or len({r["record_id"] for r in rows}) != 855 or {r["record_id"] for r in rows} != set(expected_ids):
            raise Blocked("Locked case identity population is not the original 855")
        if any(int(r["outer_fold"]) != expected_folds[r["record_id"]] or r["y_predicted"] not in expected_classes for r in rows):
            raise Blocked("Locked prediction fold or class invalid")
        if not all(np.isfinite(float(r["score_" + c])) for r in rows for c in expected_classes):
            raise Blocked("Locked prediction score nonfinite")
        case_rows[case] = rows
    with gzip.open(Path(repo) / plan["historical_core"] / "outer_predictions.csv.gz", "rt", newline="") as f:
        truth_rows = [r for r in csv.DictReader(f) if r["method"] == "R2"]
    truth = {r["record_id"]: r["y_true"] for r in truth_rows}
    true_folds = {r["record_id"]: int(r["outer_fold"]) for r in truth_rows}
    if len(truth) != 855:
        raise Blocked("Historical scoring truth is not the unique 855")
    classes = sorted(set(truth.values()))
    if {c: sum(v == c for v in truth.values()) for c in classes} != {"ZY01": 350, "ZY02": 5, **{f"ZY{i:02d}": 100 for i in range(3, 8)}}:
        raise Blocked("Scoring class counts changed")
    details, table, recalls = {}, [], []
    for spec in plan["conditions"]:
        rows = case_rows[spec["case"]]
        if len(rows) != 855 or len({r["record_id"] for r in rows}) != 855 or {r["record_id"] for r in rows} != set(truth):
            raise Blocked("Corruption case changes sample identity or count")
        if any(int(r["outer_fold"]) != true_folds[r["record_id"]] for r in rows):
            raise Blocked("Corruption case changes a sample's original OOF role")
        result = metric_counts([truth[r["record_id"]] for r in rows], [r["y_predicted"] for r in rows], classes)
        details[spec["case"]] = result
        table.append({**spec, **{k: result[k] for k in (*METRICS, "samples", "correct")}})
        recalls.extend({"case": spec["case"], **r} for r in result["per_class"])
    baseline = details["baseline"]
    if baseline["correct"] != plan["expected_historical_baseline_correct"]:
        raise Blocked("Exact 821/855 historical correctness replay failed")
    for row in table:
        for metric in METRICS:
            row["paired_baseline_change_" + metric] = row[metric] - baseline[metric]
    curves = []
    for level in (0., .001, .005, .01):
        members = [r for r in table if r["kind"] == "noise" and r["level"] == level]
        for metric in METRICS:
            values = [r[metric] for r in members]
            curves.append({"noise_level": level, "metric": metric, "seed_count": len(values),
                           "mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values),
                           "paired_baseline_change_mean": float(np.mean(values)) - baseline[metric],
                           "interpretation": "fixed-sample corruption-seed range; not a confidence interval"})
    write_csv(output / "metric_summary.csv", table)
    write_csv(output / "per_class_metrics.csv", recalls)
    write_csv(output / "noise_level_curves.csv", curves)
    write_json(output / "confusions_and_metrics.json", {"class_order": classes, "conditions": details})
    return {"historical_baseline_correct": baseline["correct"], "samples": 855, "cases": len(table),
            "prediction_rows_reused_samples": 855 * len(table), "class_counts": {c: sum(v == c for v in truth.values()) for c in classes},
            "all_metrics_include_same_855": True, "actual_new_specimens": 0,
            "baseline_accuracy": baseline["accuracy"], "baseline_macro_f1": baseline["macro_f1"],
            "baseline_balanced_accuracy": baseline["balanced_accuracy"]}


def run(repo, protocol_path, release_path, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = check_frozen(repo, protocol_path, release_path)
    if output.exists():
        raise Blocked("Execution output exists; never overwrite evidence")
    output.mkdir(parents=True)
    started, cpu0 = time.monotonic(), time.process_time()
    cpu_soft = int(resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime) + 600
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    def timed_out(*_):
        raise Blocked("Frozen robustness wall budget exceeded")
    def cpu_exceeded(*_):
        raise Blocked("Frozen robustness CPU budget exceeded")
    old_alarm = signal.signal(signal.SIGALRM, timed_out)
    old_cpu = signal.signal(signal.SIGXCPU, cpu_exceeded)
    signal.alarm(1200)
    try:
        x, axis, ids = raw_arrays(repo / plan["full_input"])
        mask = (axis >= .1) & (axis <= 1.5)
        if x.shape != (855, 2578) or int(mask.sum()) != 612:
            raise Blocked("Frozen raw frequency/mask dimensions changed")
        splits = read_json(repo / plan["historical_core"] / "splits.json")
        validate_splits(splits, ids)
        index = {str(identifier): i for i, identifier in enumerate(ids)}
        by_case = {spec["case"]: [] for spec in plan["conditions"]}
        amplitude, baseline, diagnostics = [], [], []
        with gzip.open(Path(protocol_path).parent / plan["reference_file"], "rt", newline="") as f:
            reference = list(csv.DictReader(f))
        with threadpool_limits(limits=2), fit_blockers():
            # Mandatory gate: replay all five native models before any corruption.
            for spec in plan["models"]:
                fold = spec["fold"]
                query_ids = splits[fold]["test_sample_ids"]
                query = np.asarray([index[i] for i in query_ids])
                source = np.asarray([index[i] for i in splits[fold]["train_sample_ids"]])
                model = joblib.load(repo / spec["model"])
                validate_native_model(model, spec["config"], model_band(x[source], mask))
                baseline.extend(native_rows(model, model_band(x[query], mask), query_ids, fold))
                del model
            replay = verify_replay(baseline, reference)
            by_case["baseline"] = baseline
            write_json(output / "baseline_replay.json", {"status": "PASS_EXACT_HISTORICAL_PREDICTIONS", **replay})
            for spec in plan["models"]:
                fold = spec["fold"]
                source = np.asarray([index[i] for i in splits[fold]["train_sample_ids"]])
                query_ids = splits[fold]["test_sample_ids"]
                query = np.asarray([index[i] for i in query_ids])
                sd = source_sd(x[source])
                amplitude.append({"fold": fold, "source_samples": 684, "query_samples": 171, "source_full_axis_rms_sd": sd})
                model = joblib.load(repo / spec["model"])
                validate_native_model(model, spec["config"], model_band(x[source], mask))
                before = joblib.hash(model)
                for case in plan["conditions"]:
                    if case["case"] == "baseline":
                        continue
                    altered = corrupt(x[query], axis, sd, case, fold)
                    by_case[case["case"]].extend(native_rows(model, model_band(altered, mask), query_ids, fold))
                    difference = altered - x[query]
                    diagnostics.append({"case": case["case"], "outer_fold": fold,
                        "source_full_axis_rms_sd": sd,
                        "actual_full_axis_perturbation_rms": float(np.sqrt(np.mean(difference ** 2))),
                        "actual_model_band_perturbation_rms": float(np.sqrt(np.mean(difference[:, mask] ** 2))),
                        "expected_noise_rms": case["level"] * sd if case["kind"] == "noise" else "not_noise"})
                    if time.process_time() - cpu0 > 600 or time.monotonic() - started > 1200:
                        raise Blocked("Frozen inference resource budget exceeded")
                if joblib.hash(model) != before or sha(repo / spec["model"]) != spec["model_sha256"]:
                    raise Blocked("Native prediction mutated a frozen learned model state")
                del model
        (output / "predictions").mkdir()
        for name, rows in by_case.items():
            write_csv(output / "predictions" / (name + ".csv.gz"), rows)
        write_csv(output / "source_fold_amplitudes.csv", amplitude)
        write_csv(output / "corruption_amplitude_diagnostics.csv", diagnostics)
        write_json(output / "axis_boundary_diagnostic.json", {
            "axis_unit": plan["axis_shift_status"], "endpoint_policy": "constant clamp outside original full export; no rows or columns dropped",
            "shift_cases": [{"shift_THz": delta, "model_features_clamped": int(np.sum((axis[mask] + delta < axis[0]) | (axis[mask] + delta > axis[-1])))}
                            for delta in (-.002, .002)] if "VERIFIED" in plan["axis_shift_status"] else []})
        locked = [*sorted((output / "predictions").glob("*.csv.gz")), output / "baseline_replay.json", output / "source_fold_amplitudes.csv", output / "corruption_amplitude_diagnostics.csv", output / "axis_boundary_diagnostic.json"]
        write_json(output / "prediction_lock.json", {"status": LOCK_STATUS,
                   "protocol_sha256": sha(protocol_path), "locked_utc": datetime.now(timezone.utc).isoformat(),
                   "files_sha256": {str(p.relative_to(output)): sha(p) for p in locked},
                   "fit_calls": 0, "training_labels_loaded": False, "query_labels_loaded_for_inference": False})
        scored = score_locked(repo, plan, output, sha(protocol_path))
        summary = {"schema": SCHEMA, "status": "DONE_EXPLORATORY_SYNTHETIC_CORRUPTION_DIAGNOSTICS", **scored,
                   "protocol_sha256": sha(protocol_path), "release_sha256": sha(release_path),
                   "models_fitted": 0, "unique_native_models": 5, "model_load_calls": 10, "new_specimens": 0,
                   "cpu_seconds": time.process_time() - cpu0, "wall_seconds": time.monotonic() - started,
                   "limits": plan["limits"], "baseline_replay": replay, "limitations": LIMITATIONS}
        write_json(output / "summary.json", summary)
        (output / "README.md").write_text(
            "# Frozen R2 OOF engineering robustness diagnostics\n\n"
            "All prespecified conditions reuse the same 855 historically exposed development records and five exact historical R2 models. "
            "The unchanged baseline replayed every original prediction before corruption. No fitting or selection was allowed.\n\n"
            "See metric_summary.csv for all cases and paired changes from baseline, per_class_metrics.csv for class denominators, "
            "confusions_and_metrics.json for every count matrix, and noise_level_curves.csv for every stochastic seed level. "
            "Seed minimum/maximum values are descriptive and are not confidence intervals.\n\n"
            + "\n".join("- " + s for s in LIMITATIONS) + "\n", encoding="utf-8")
        write_json(output / "artifact_index.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
        return summary
    except Exception as exc:
        write_json(output / "status.json", {"status": "BLOCKED", "reason": str(exc), "models_fitted": 0,
                   "cpu_seconds": time.process_time() - cpu0, "wall_seconds": time.monotonic() - started,
                   "no_partial_result_is_validation": True})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGXCPU, old_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.repo, args.output)
    else:
        if args.protocol is None or args.release is None:
            parser.error("run requires --protocol and --release from root-reviewed frozen preparation")
        result = run(args.repo, args.protocol, args.release, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
