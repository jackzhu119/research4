#!/usr/bin/env python3
"""E3: a locked-mask by locked-numeric-SVM-parameter 2x2 diagnostic.

All four cells may be refitted (at most 60 fits over15 contexts). Diagonal
replay is checked against archived per-specimen predictions; discrepancies
are retained and explicitly change the interpretation to a new unified
condition diagnostic. No tuning or cell selection is performed.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from e2_utils import (REPORT, alpha_mask, assert_context_split, contexts, core_data,
                      csv_write, dump, file_hash, ids_hash, indices_for, load_job,
                      numeric_parameters, read_json, receipt, snv)

OUT = REPORT / "e3"
METHOD_A = "selection/alpha_snv_c2_k62"
METHOD_B = "rank_context/true_n"


def measures(y, prediction, classes):
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0))}


def original_predictions(result, test_ids):
    rows = result.get("predictions")
    if not isinstance(rows, list):
        raise FileNotFoundError("Archived diagonal specimen predictions missing")
    mapping = {str(row["record_id"]): str(row.get("prediction", row.get("y_pred"))) for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(map(str, test_ids)):
        raise ValueError("Archived diagonal OOF IDs differ")
    return np.array([mapping[str(value)] for value in test_ids])


def verify_source_transform(result, directory, scaler, mask):
    base = result.get("job", {}).get("family", result.get("family"))
    if base not in ("selection", "rank_context"):
        raise ValueError("Unexpected source family")
    if result.get("chosen_parameters", {}).get("preprocessing") != "fixed":
        raise ValueError("Source preprocessing differs from source fixed alpha-SNV transform")
    path = directory / "fitted_artifacts.npz"
    note = {"saved_scaler_artifact_available": path.exists(),
            "source_transform_definition": "pinned study_worker: per-spectrum ddof1 alpha SNV, train-fit per-column StandardScaler, then locked alpha mask",
            "selected_alpha_columns": mask.tolist()}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            mean, scale = saved["scaler_mean"], saved["scaler_scale"]
            if len(mean) == 1224:
                mean, scale = mean[612:], scale[612:]
            if len(mean) != 612 or not np.allclose(mean, scaler.mean_, atol=1e-12, rtol=1e-12) or not np.allclose(scale, scaler.scale_, atol=1e-12, rtol=1e-12):
                raise ValueError("Archived full alpha-column scaler differs")
            if "selected_global_indices" in saved and not np.array_equal(np.sort(saved["selected_global_indices"] - 612), mask):
                raise ValueError("Saved selected coordinates differ from result")
        note["saved_scaler_sha256"] = file_hash(path)
        note["saved_scaler_matches"] = True
    return note


def run_context(number):
    if number not in range(15):
        raise ValueError("Context outside frozen first3 by fivefold queue")
    context = contexts()[number]
    a, split_a, dir_a = load_job(context["methods"][METHOD_A])
    b, split_b, dir_b = load_job(context["methods"][METHOD_B])
    assert_context_split(context, split_a)
    assert_context_split(context, split_b)
    data = core_data()
    train = indices_for(data["ids"], context["outer_train_ids"])
    test = indices_for(data["ids"], context["outer_test_ids"])
    alpha = snv(data["alpha"])
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(alpha[train])
    transformed_test = scaler.transform(alpha[test])
    masks = {"A": alpha_mask(a), "B": alpha_mask(b)}
    if any(len(mask) != 62 for mask in masks.values()):
        raise ValueError("Expected both locked final alpha-only K62 masks")
    parameters = {"A": numeric_parameters(a), "B": numeric_parameters(b)}
    transform_a = verify_source_transform(a, dir_a, scaler, masks["A"])
    transform_b = verify_source_transform(b, dir_b, scaler, masks["B"])
    # Ranking in1224 and fitting on62 alpha columns retain the same independent
    # alpha column centering/scaling; n is not a classifier input.
    expected = {"AA": original_predictions(a, context["outer_test_ids"]),
                "BB": original_predictions(b, context["outer_test_ids"])}
    classes = np.asarray(data["classes"]).astype(str)
    true = classes[data["y"][test]]
    y_train = data["y"][train]
    predictions, scores, fits, equivalences, decision_scores = {}, {}, 0, [], {}
    model_cache = {}
    cell_costs = {}
    target = OUT / ("context" + str(number))
    target.mkdir(parents=True, exist_ok=True)
    lock = {"context_index": number, "repeat": context["repeat"], "seed": context["seed"], "fold": context["fold"],
            "method_A": METHOD_A, "method_B": METHOD_B, "job_A": context["methods"][METHOD_A], "job_B": context["methods"][METHOD_B],
            "outer_train_ids": context["outer_train_ids"], "outer_test_ids": context["outer_test_ids"],
            "input_sha256": data["input_sha256"], "masks_local_alpha_indices": {key: value.tolist() for key, value in masks.items()},
            "parameters_actual_numeric_gamma": parameters, "new_selection_or_tuning": False,
            "class_weight": "balanced", "kernel": "rbf", "random_state": 42,
            "source_transform_A": transform_a, "source_transform_B": transform_b}
    dump(target / "locked_design.json", lock)
    with threadpool_limits(limits=1):
        for mask_name, parameter_name in (("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")):
            cell = mask_name + parameter_name
            mask, p = masks[mask_name], parameters[parameter_name]
            identity = (tuple(mask.tolist()), p["C"], p["gamma"])
            cpu0, wall0 = time.process_time(), time.monotonic()
            if identity in model_cache:
                source_cell = model_cache[identity]
                predictions[cell] = predictions[source_cell].copy()
                decision_scores[cell] = decision_scores[source_cell].copy()
                equivalences.append({"cell": cell, "identical_to": source_cell, "refit_skipped": True})
                actual_fits = 0
            else:
                model = SVC(C=p["C"], gamma=p["gamma"], class_weight="balanced", kernel="rbf", random_state=42)
                model.fit(transformed_train[:, mask], y_train)
                predictions[cell] = classes[model.predict(transformed_test[:, mask])]
                decision_scores[cell] = model.decision_function(transformed_test[:, mask])
                fits += 1
                actual_fits = 1
                model_cache[identity] = cell
                np.savez_compressed(target / ("model_" + cell + ".npz"),
                                    support_vectors=model.support_vectors_, dual_coef=model.dual_coef_,
                                    intercept=model.intercept_, support=model.support_, n_support=model.n_support_,
                                    classes=model.classes_, decision_scores=decision_scores[cell], scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                                    selected_alpha_indices=mask)
            scores[cell] = measures(true, predictions[cell], classes)
            cell_costs[cell] = {"new_classifier_fits": actual_fits, "cpu_seconds": time.process_time() - cpu0,
                                "wall_seconds": time.monotonic() - wall0,
                                "predictor_count": len(mask), "input_dependency": "all612 alpha points for per-spectrum SNV"}
            # An immutable completed cell checkpoint survives a later cell failure.
            dump(target / ("cell_" + cell + ".json"), {"cell": cell, "metrics": scores[cell],
                                                        "predictions": predictions[cell].tolist(), "cost": cell_costs[cell]})
    replay = {cell: {"exact_prediction_match": bool(np.array_equal(predictions[cell], expected[cell])),
                     "mismatched_specimens": int(np.sum(predictions[cell] != expected[cell]))}
              for cell in ("AA", "BB")}
    for cell, directory in (("AA", dir_a), ("BB", dir_b)):
        saved_path = directory / "fitted_artifacts.npz"
        if saved_path.exists():
            with np.load(saved_path, allow_pickle=False) as saved:
                if "decision_scores" in saved:
                    original = saved["decision_scores"]
                    difference = float(np.max(np.abs(original - decision_scores[cell]))) if original.shape == decision_scores[cell].shape else None
                    agree = original.shape == decision_scores[cell].shape and bool(np.allclose(original, decision_scores[cell], rtol=1e-10, atol=1e-10))
                    replay[cell].update(decision_scores_match_within_numerical_tolerance=agree,
                                        decision_scores_max_abs_difference=difference)
    exact = all(row["exact_prediction_match"] and row.get("decision_scores_match_within_numerical_tolerance", True) for row in replay.values())
    effects = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        aa, ab, ba, bb = [scores[cell][metric] for cell in ("AA", "AB", "BA", "BB")]
        effects[metric] = {"mask_B_minus_A_at_theta_A": ba - aa, "mask_B_minus_A_at_theta_B": bb - ab,
                           "theta_B_minus_A_at_mask_A": ab - aa, "theta_B_minus_A_at_mask_B": bb - ba,
                           "interaction_difference_of_differences": bb - ba - ab + aa,
                           "units": "fraction; multiply100 for percentage points"}
    rows = []
    for i, record_id in enumerate(context["outer_test_ids"]):
        rows.append({"record_id": record_id, "physical_specimen_id": record_id, "true_label": true[i],
                     **{cell: predictions[cell][i] for cell in predictions},
                     "archived_AA": expected["AA"][i], "archived_BB": expected["BB"][i],
                     "mask_change_theta_A": int(predictions["BA"][i] != predictions["AA"][i]),
                     "mask_change_theta_B": int(predictions["BB"][i] != predictions["AB"][i]),
                     "parameter_change_mask_A": int(predictions["AB"][i] != predictions["AA"][i]),
                     "parameter_change_mask_B": int(predictions["BB"][i] != predictions["BA"][i])})
    csv_write(target / "specimen_predictions.csv", rows)
    result = {"status": "COMPLETED", "context_index": number, "repeat": context["repeat"], "seed": context["seed"], "fold": context["fold"],
              "interpretation_mode": "archived_diagonals_exactly_replayed" if exact else "new_unified_condition_diagnostic",
              "archived_diagonal_replay": replay, "metrics": scores, "effects": effects,
              "confusion_matrices": {cell: confusion_matrix(true, predictions[cell], labels=classes).tolist() for cell in predictions},
              "class_order": classes.tolist(), "cell_costs": cell_costs, "new_classifier_fits": fits,
              "new_pls_coefficient_fits": 0, "equivalent_cells": equivalences,
              "outer_train_ids_sha256": ids_hash(context["outer_train_ids"]), "outer_test_ids_sha256": ids_hash(context["outer_test_ids"]),
              "code_sha256": file_hash(__file__),
              "limitations": ["exploratory ranking-context mechanism within background protocol, not the old V4 joint pipeline",
                              "four cells are not candidates for selecting an unbiased best final model",
                              "does not establish chemical causation of n",
                              "all unfavorable cells and replay discrepancies retained"]}
    dump(target / "result.json", result)
    receipt("COMPLETED", "Locked 2x2 mask/parameter diagnostic saved; both original diagonals checked", new_classifier_fits=fits, new_pls_fits=0,
            exact_archived_diagonal_replay=exact, artifacts=[str(target.relative_to(REPORT))])


def summarize():
    completed, missing, effects, counts = [], [], [], 0
    for number in range(15):
        path = OUT / ("context" + str(number)) / "result.json"
        if not path.exists():
            missing.append(number)
            continue
        value = read_json(path)
        completed.append(value)
        counts += int(value["new_classifier_fits"])
        for metric, quantities in value["effects"].items():
            effects.append({"context_index": number, "repeat": value["repeat"], "seed": value["seed"], "fold": value["fold"],
                            "metric": metric, "interpretation_mode": value["interpretation_mode"], **quantities})
    csv_write(OUT / "paired_factorial_effects.csv", effects)
    aggregate = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        keys = ("mask_B_minus_A_at_theta_A", "mask_B_minus_A_at_theta_B", "theta_B_minus_A_at_mask_A",
                "theta_B_minus_A_at_mask_B", "interaction_difference_of_differences")
        for key in keys:
            values = np.array([row["effects"][metric][key] for row in completed])
            if len(values):
                aggregate.append({"metric": metric, "effect": key, "context_count": len(values),
                                  "mean": float(values.mean()), "median": float(np.median(values)),
                                  "min": float(values.min()), "max": float(values.max()),
                                  "positive": int(np.sum(values > 0)), "zero": int(np.sum(values == 0)), "negative": int(np.sum(values < 0)),
                                  "inference": "descriptive fixed15 dependent contexts; no independent-repeat t test"})
    csv_write(OUT / "effect_descriptions.csv", aggregate)
    status = "COMPLETED" if not missing else "BLOCKED_MISSING_ARTIFACT"
    value = {"status": status, "completed_contexts": len(completed), "expected_contexts": 15, "missing_contexts": missing,
             "new_classifier_fits": counts, "new_pls_coefficient_fits": 0,
             "strictly_replayed_contexts": sum(row["interpretation_mode"] == "archived_diagonals_exactly_replayed" for row in completed),
             "unified_condition_contexts": sum(row["interpretation_mode"] == "new_unified_condition_diagnostic" for row in completed),
             "effect_descriptions": aggregate,
             "limitations": "No favorable-cell selection, chemical causal inference or equivalence to old V4 joint workflow"}
    dump(OUT / "summary.json", value)
    receipt(status, "All locked factorial contexts summarized, including replay discrepancies", new_classifier_fits=0, new_pls_fits=0,
            cumulative_E3_classifier_fits=counts, artifacts=["e3/paired_factorial_effects.csv", "e3/effect_descriptions.csv", "e3/summary.json"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if (args.context is None) == (not args.summarize):
        parser.error("Choose exactly one --context or --summarize")
    try:
        if args.summarize:
            summarize()
        else:
            run_context(args.context)
    except FileNotFoundError as error:
        receipt("BLOCKED_MISSING_ARTIFACT", str(error), new_classifier_fits=0, new_pls_fits=0)
        print(str(error), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
