"""H01/H02: fixed-coordinate loss-mass and active-C mechanism controls.

These are finite interventions on predeclared objectives, not a hyperparameter
selection workflow. Existing Source preprocessing and numeric gamma are frozen
for every fitted arm. Query labels cannot enter a fitting interface.
"""
from __future__ import annotations

import copy
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from herb_spectra.spectral_models import FittedSpectralSource
from b_followup_common_20261110 import TaskWriter, endpoint_id, load_parent
from b_followup_models_20261110 import diagnostics, native_predict, nested_support
from b_round3_margin_20261110 import audit_model, compare_models


K_VALUES = (1, 3, 5, 10)
H01_METHODS = ("source_mass_only", "source_support_fixed_C", "source_support_normalized_mass")
C_CONDITIONS = ((0.1, "0p1"), (1.0, "1"), (10.0, "10"))
WEIGHT_SETTINGS = (("balanced", "balanced"), ("none", None))
BASE_C = 10.0


def c_method(C_tag, setting, *, argmax=False):
    return f"C_{C_tag}_{setting}" + ("_argmax" if argmax else "")


def expected_endpoints(job):
    if job["family"] == "H01":
        return [endpoint_id("identity")] + [endpoint_id(method, k)
            for k in K_VALUES for method in H01_METHODS]
    if job["family"] == "H02":
        return [endpoint_id(c_method(tag, setting, argmax=argmax))
            for _, tag in C_CONDITIONS for setting, _ in WEIGHT_SETTINGS
            for argmax in (False, True)]
    raise ValueError("Only the frozen H01/H02 model families are supported")


def fit_count(job):
    if job["family"] == "H01":
        return 8
    if job["family"] == "H02":
        return 2 if job["ratio"] == "B" else 4
    raise ValueError("Unknown model-side family")


def adapter_count(job=None):
    return 0


def fit_source_fixedcoords(X_train, y_train, reference, *, C, class_weight="balanced"):
    """Fit only an SVC, using copies of existing Source coordinate transforms."""
    X_train = np.asarray(X_train, dtype=float)
    labels = np.asarray(y_train, dtype=str)
    C = float(C)
    if (X_train.ndim != 2 or len(X_train) != len(labels) or not len(labels)
            or not np.isfinite(X_train).all() or not np.isfinite(C) or C <= 0):
        raise ValueError("Training matrix/labels/C must be aligned and finite")
    if class_weight not in ("balanced", None):
        raise ValueError("Unknown frozen class-weight condition")
    classes = np.unique(labels)
    if not np.array_equal(classes, reference.classes_):
        raise ValueError("All frozen source classes must occur in the training arm")
    preprocessor = copy.deepcopy(reference.preprocessor_)
    scaler = copy.deepcopy(reference.scaler_)
    values = scaler.transform(preprocessor.transform(X_train))
    gamma = float(reference.estimator_._gamma)
    estimator = SVC(C=C, kernel="rbf", gamma=gamma, class_weight=class_weight,
                    break_ties=True, decision_function_shape="ovr", random_state=42)
    estimator.fit(values, labels)
    if not np.array_equal(estimator.classes_, classes):
        raise ValueError("SVC output class ordering changed")
    if joblib.hash(scaler) != joblib.hash(reference.scaler_) or float(estimator._gamma) != gamma:
        raise ValueError("Fitted arm changed the frozen Source coordinates or gamma")
    classes.setflags(write=False)
    candidate = {"id": "round4_fixed_source_coordinates_rbf_svc", "preprocessing": "snv",
                 "model": "svc", "C": C, "class_weight": class_weight,
                 "scaler_mode": "frozen_source", "gamma_mode": "frozen_source_numeric"}
    metadata = {"fit_scope": "explicit_training_arrays_only", "source_samples": len(labels),
                "class_order": classes.tolist(), "input_features": X_train.shape[1],
                "class_counts": {str(label): int(np.count_nonzero(labels == label)) for label in classes},
                "C": C, "effective_gamma": gamma, "class_weight": class_weight,
                "scaler_mode": "frozen_source", "preprocessing": "source SNV ddof=1"}
    return FittedSpectralSource(candidate, preprocessor, scaler, estimator, classes, metadata)


def loss_mass_C(source_n, support_n, arm):
    """Analytic per-observation C controls; no spectra or outcome enter this map."""
    if (isinstance(source_n, bool) or isinstance(support_n, bool) or source_n <= 0
            or support_n < 0 or int(source_n) != source_n or int(support_n) != support_n):
        raise ValueError("Loss-mass counts must be nonnegative integers with positive Source")
    if arm == "source_mass_only":
        return BASE_C * (source_n + support_n) / source_n
    if arm == "source_support_normalized_mass":
        return BASE_C * source_n / (source_n + support_n)
    if arm in ("identity", "source_support_fixed_C"):
        return BASE_C
    raise ValueError("Unknown loss-mass arm")


def _fixed_coordinates(model, reference):
    detail = diagnostics(model, reference=reference)
    if not detail["scaler_identical_to_source"] or not detail["gamma_identical_to_source"]:
        raise ValueError("All H01/H02 arms must keep Source Scaler and numeric gamma fixed")
    if joblib.hash(model.preprocessor_) != joblib.hash(reference.preprocessor_):
        raise ValueError("Source preprocessing changed")
    return detail


def _load_frozen_refit(job, k, oldroot, round2root):
    from b_round4_common_20261111 import load_frozen_refit
    return load_frozen_refit(job, k, oldroot, round2root)


def _h01(writer, job, oldroot, round2root):
    _, source, _ = load_parent(job, oldroot)
    before_hash = joblib.hash(source)
    source_ids = list(job["source_ids"])
    source_y = writer.labels(source_ids)
    counts = np.unique(source_y, return_counts=True)[1]
    if not np.all(counts == counts[0]) or not np.array_equal(
            source.estimator_.class_weight_, np.ones(len(job["classes"]))):
        raise ValueError("H01 requires the predeclared equal-class Source contexts")
    if float(source.estimator_.C) != BASE_C:
        raise ValueError("H01 expects the frozen C10 source model")
    ns = len(source_ids)
    records = [{"method": "identity", "k": 0, "scores": source.predict_scores(writer.query_X),
                "pred": native_predict(source, writer.query_X), "support_ids": [],
                "detail": {"C": BASE_C, "C_multiplier": 1.0, "labeled_training_n": ns,
                           "support_labels_used": 0, "allocated_support_labels": 0,
                           "effective_loss_mass": BASE_C * ns, "C_times_labeled_n": BASE_C * ns,
                           "source_baseline_loss_mass": BASE_C * ns,
                           "reused_model": True, "new_model_fit": False,
                           "class_weight_all_one": True, **_fixed_coordinates(source, source)}}]
    training_locks, provenance = {}, []
    new_fits = 0
    for k in K_VALUES:
        support_ids = nested_support(job, writer.indexed, k)
        all_ids = source_ids + support_ids
        nk, nt = len(support_ids), len(all_ids)
        training_locks[str(k)] = {"source_statistics_ids": source_ids,
                                  "source_mass_only_labeled_ids": source_ids,
                                  "source_support_labeled_ids": all_ids,
                                  "allocated_support_ids": support_ids}
        for method in H01_METHODS:
            C = loss_mass_C(ns, nk, method)
            train_ids = source_ids if method == "source_mass_only" else all_ids
            started_cpu, started_wall = time.process_time(), time.monotonic()
            reused = method == "source_support_fixed_C"
            if reused:
                model, origin = _load_frozen_refit(job, k, oldroot, round2root)
                provenance.append({"method": method, "k": k, **origin})
            else:
                model = fit_source_fixedcoords(writer.take(train_ids), writer.labels(train_ids),
                                               source, C=C, class_weight="balanced")
                writer.save_model(f"{method}_k{k}.joblib", model)
                new_fits += 1
            if float(model.estimator_.C) != C:
                raise ValueError("Saved/fitted H01 C differs from its analytic loss-mass intervention")
            detail = _fixed_coordinates(model, source)
            if not np.array_equal(model.estimator_.class_weight_, np.ones(len(job["classes"]))):
                raise ValueError("H01 class-weight allocation changed despite balanced training counts")
            # A reused model has its old training cost; loading/evaluation cost is
            # still part of this bundle's CPU rather than new classifier training.
            fit_cpu = time.process_time() - started_cpu if not reused else 0.0
            fit_wall = time.monotonic() - started_wall if not reused else 0.0
            records.append({"method": method, "k": k, "scores": model.predict_scores(writer.query_X),
                            "pred": native_predict(model, writer.query_X), "support_ids": support_ids,
                            "detail": {"C": C, "C_multiplier": C / BASE_C,
                                       "labeled_training_n": len(train_ids), "statistics_training_n": ns,
                                       "support_labels_used": 0 if method == "source_mass_only" else nk,
                                       "allocated_support_labels": nk, "class_weight_all_one": True,
                                       "effective_loss_mass": C * len(train_ids),
                                       "C_times_labeled_n": C * len(train_ids),
                                       "source_baseline_loss_mass": BASE_C * ns,
                                       "total_loss_mass_matches_source": bool(method == "source_support_normalized_mass"),
                                       "source_only_uniform_loss_multiplier": nt / ns if method == "source_mass_only" else None,
                                       "reused_model": reused, "new_model_fit": not reused,
                                       "fit_cpu_seconds": fit_cpu, "fit_wall_seconds": fit_wall,
                                       **detail}})
    # Query truths are joined by TaskWriter only after every arm has been fitted
    # and its predictions generated. No fitting function can receive Query y.
    for record in records:
        writer.record(record["method"], record["k"], record["scores"], pred=record["pred"],
                      support_ids=record["support_ids"], decision_rule="SVC.predict", **record["detail"])
    if joblib.hash(source) != before_hash or new_fits != fit_count(job):
        raise ValueError("H01 source immutability or actual fit accounting failed")
    writer.save_json("training_id_locks.json", training_locks)
    writer.save_json("reused_model_provenance.json", provenance)
    writer.save_json("loss_mass_design.json", {
        "Source_labels": ns, "C0": BASE_C, "source_loss_mass": BASE_C * ns,
        "class_weight_all_one": True, "coordinates_fit_scope": "original Source only",
        "interpretation": "Fixed finite loss-mass controls; added supervised rows remain different covariates/labels, not a uniquely identified label-information cause."})
    return writer.finish(expected_endpoints(job), new_model_fits=new_fits, new_adapter_fits=0,
                         source_model_unchanged=True, reused_model_endpoints=5,
                         selected_C_by_query=False, fixed_source_coordinates=True)


def _load_weight(job, setting, oldroot, round2root):
    from b_round3_common_20261110 import load_weight_model
    return load_weight_model(job, setting, oldroot, round2root)


def _h02(writer, job, oldroot, round2root):
    reference, origin = _load_weight(job, "balanced", oldroot, round2root)
    before_hash = joblib.hash(reference)
    source_ids = list(job["source_ids"])
    X_source = writer.take(source_ids)
    y_source = writer.labels(source_ids)
    if float(reference.estimator_.C) != BASE_C:
        raise ValueError("H02 reference must be the frozen C10 Source model")
    if len(job["classes"]) != 2:
        raise ValueError("H02 is the fixed binary Source-ratio/coverage experiment")
    is_equal_counts = len(set(np.unique(y_source, return_counts=True)[1])) == 1
    if is_equal_counts != (job["ratio"] == "B"):
        raise ValueError("H02 ratio semantics disagree with actual Source counts")
    provenance, cases, new_fits = [], [], 0
    by_condition = {}
    for C, tag in C_CONDITIONS:
        balanced_model = None
        for setting, weight in WEIGHT_SETTINGS:
            method = c_method(tag, setting)
            start_cpu, start_wall = time.process_time(), time.monotonic()
            historical = C == BASE_C
            equivalent = setting == "none" and is_equal_counts
            if equivalent:
                if balanced_model is None or not np.array_equal(
                        balanced_model.estimator_.class_weight_, np.ones(len(job["classes"]))):
                    raise ValueError("Balanced/None reuse requires exactly unit effective class weights")
                model = balanced_model
                provenance.append({"method": method, "C": C, "reused_from_method": c_method(tag, "balanced"),
                                   "reason": "EQUAL_EFFECTIVE_CLASS_WEIGHTS", "effective_class_weights": [1.0, 1.0]})
            elif historical:
                model, old = (reference, origin) if setting == "balanced" else _load_weight(job, setting, oldroot, round2root)
                provenance.append({"method": method, "C": C, "reason": "EXACT_HISTORICAL_C10_ENDPOINT", **old})
            else:
                # We deliberately refit at the new C. Numerical interiority of
                # the C10 solution is not an exact-parameter reuse criterion.
                model = fit_source_fixedcoords(X_source, y_source, reference, C=C, class_weight=weight)
                writer.save_model(f"{method}.joblib", model)
                new_fits += 1
            if setting == "balanced":
                balanced_model = model
            if model.classes_.tolist() != job["classes"] or float(model.estimator_.C) != C:
                raise ValueError("H02 class vocabulary or planned C changed")
            detail = _fixed_coordinates(model, reference)
            scores, predicted = model.predict_scores(writer.query_X), native_predict(model, writer.query_X)
            reused = historical or equivalent
            cases.append({"method": method, "C": C, "C_tag": tag, "setting": setting,
                          "model": model, "scores": scores, "predicted": predicted,
                          "detail": {"C": C, "C_multiplier": C / BASE_C, "class_weight_setting": setting,
                                     "training_n": len(source_ids), "support_labels_used": 0,
                                     "reused_model": reused, "new_model_fit": not reused,
                                     "equal_effective_class_weight_reuse": equivalent,
                                     "fit_cpu_seconds": time.process_time() - start_cpu if not reused else 0.0,
                                     "fit_wall_seconds": time.monotonic() - start_wall if not reused else 0.0,
                                     **detail}})
            by_condition[(C, setting)] = cases[-1]
    # All fitting/prediction precedes Query-label access, including Source-side
    # constraint auditing, whose Query labels are used only for evaluation.
    query_y = writer.labels(job["query_ids"])
    source_rows, mechanisms, audit_by_condition = [], [], {}
    for case in cases:
        audit = audit_model(case["model"], X_source, y_source, source_ids,
                            writer.query_X, query_y, case["setting"])
        audit_by_condition[(case["C"], case["setting"])] = audit
        if not np.array_equal(audit["query_scores"], case["scores"]):
            raise ValueError("H02 saved scores do not reproduce the existing model audit")
        for row in audit["samples"]:
            source_rows.append({"method": case["method"], "C": case["C"], **row})
        summary = {"method": case["method"], "C": case["C"], **audit["summary"]}
        mechanisms.append(summary)
        detail = {**case["detail"], **{key: value for key, value in audit["summary"].items()
                                      if key not in ("analysis", "setting", "interpretation")}}
        writer.record(case["method"], 0, case["scores"], pred=case["predicted"],
                      support_ids=[], decision_rule="SVC.predict", **detail)
        writer.record(case["method"] + "_argmax", 0, case["scores"], support_ids=[],
                      decision_rule="score_argmax", **{**detail, "new_model_fit": False})
    for C, tag in C_CONDITIONS:
        first, second = by_condition[(C, "balanced")], by_condition[(C, "none")]
        pair = compare_models(first["model"], second["model"], audit_by_condition[(C, "balanced")],
                              audit_by_condition[(C, "none")], query_y)
        mechanisms.append({**pair, "analysis": "weight_pair_at_C", "C": C,
                           "method": f"C_{tag}_none_minus_balanced"})
        if C != BASE_C:
            for setting, _ in WEIGHT_SETTINGS:
                baseline, changed = by_condition[(BASE_C, setting)], by_condition[(C, setting)]
                shift = compare_models(baseline["model"], changed["model"], audit_by_condition[(BASE_C, setting)],
                                       audit_by_condition[(C, setting)], query_y)
                mechanisms.append({**shift, "analysis": "C_dose_within_weight", "C": C,
                                   "setting": setting, "method": f"C_{tag}_minus_C10_{setting}"})
    if joblib.hash(reference) != before_hash or new_fits != fit_count(job):
        raise ValueError("H02 reference immutability or exact fitting count failed")
    pd.DataFrame(source_rows).to_csv(writer.temp / "source_C_bounds.csv", index=False)
    pd.DataFrame(mechanisms).to_csv(writer.temp / "C_constraint_mechanisms.csv", index=False)
    writer.save_json("C_constraint_mechanisms.json", mechanisms)
    writer.save_json("reused_model_provenance.json", provenance)
    writer.save_json("training_id_locks.json", {"labeled_ids": source_ids, "statistics_ids": source_ids,
                                               "numeric_gamma_from_original_source": float(reference.estimator_._gamma)})
    return writer.finish(expected_endpoints(job), new_model_fits=new_fits, new_adapter_fits=0,
                         fixed_source_coordinates=True, reference_source_unchanged=True,
                         C_grid_used_for_selection=False,
                         interpretation="All predeclared C interventions are reported; active-bound tolerances are numerical descriptions, never a stopping or model-selection rule.")


def execute(job, X, metadata, axis, output, input_hash, oldroot, round2root, round3root):
    """Execute one frozen model-side bundle; caller handles sealed resumes."""
    writer = TaskWriter(job, X, metadata, axis, output, input_hash, oldroot)
    if job["family"] == "H01":
        return _h01(writer, job, oldroot, round2root)
    if job["family"] == "H02":
        return _h02(writer, job, oldroot, round2root)
    raise ValueError("Unknown H01/H02 family")
