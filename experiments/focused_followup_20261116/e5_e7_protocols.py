#!/usr/bin/env python3
"""Read-only workbook provenance, disabled MLP audit and prospective batch gate.

This entry point never fits a model and never updates original workbook/paper
bytes.  New-batch validation only validates metadata; it cannot enable training.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path
import posixpath
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(os.environ.get("FOCUSED_RESULT_ROOT", ROOT / "reports/focused_followup_20261116"))
CACHE = Path(os.environ.get("FOCUSED_SOURCE_ROOT", ROOT / ".cache/focused_followup_20261116"))
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def receipt(task, status, details):
    ident = os.environ.get("FOCUSED_TASK_ID", task)
    dump(REPORT / "task_results" / (ident + ".json"), {"task_id": ident, "module": task,
         "status": status, "new_classifier_fits": 0, "new_pls_fits": 0, **details})


def inspect_workbook(path):
    """Read only named active workbook XML; no raw/archive discovery."""
    ans = {}
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
                strings.append("".join(si.itertext()))
        relations = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS)
        for sheet in sheets:
            name = sheet.attrib["name"]
            if name not in ("时域", "频域", "折射率", "吸收系数"):
                continue
            target = relations[sheet.attrib["{" + REL + "}id"]]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            headers, axis, ref_count = [], [], 0
            with z.open(member) as f:
                for _, elem in ET.iterparse(f, events=["end"]):
                    if elem.tag != "{" + NS["m"] + "}row":
                        continue
                    vals = []
                    for cell in elem:
                        v = cell.find("m:v", NS)
                        v = v.text if v is not None else "".join(cell.itertext())
                        if cell.attrib.get("t") == "s":
                            v = strings[int(v)]
                        vals.append((cell.attrib.get("r", ""), v))
                    if not headers:
                        headers = [v for _, v in vals]
                    else:
                        a = next((v for addr, v in vals if re.fullmatch(r"A[0-9]+", addr)), None)
                        if a is not None:
                            axis.append(a)
                        if name in ("时域", "频域"):
                            b = next((v for addr, v in vals if re.fullmatch(r"B[0-9]+", addr)), None)
                            try:
                                float(b)
                                ref_count += 1
                            except (TypeError, ValueError):
                                pass
                    elem.clear()
            # Header is evidence of an export label, not proof of the transform.
            axis_hash = hashlib.sha256("\n".join(axis).encode()).hexdigest()
            ans[name] = {"headers": headers, "axis_count": len(axis), "axis_export_strings_sha256": axis_hash,
                         "reference_numeric_rows": ref_count}
    return ans


def units_report():
    out = REPORT / "e5"
    registry = read_json(ROOT / "data/active_workbooks.json")
    declaration_path = ROOT / "data/annotations/acquisition_conditions_user_declaration_20261112.json"
    declaration = read_json(declaration_path)
    declaration_hash = sha(declaration_path)
    workbooks, grids, mismatch = [], [], []
    for item in registry["workbooks"]:
        p = ROOT / item["path"]
        observed = sha(p)
        if observed != item["sha256"]:
            raise ValueError("Active workbook bytes differ from registry: " + item["path"])
        sheets = inspect_workbook(p)
        time_headers = sheets.get("时域", {}).get("headers", [])
        freq_headers = sheets.get("频域", {}).get("headers", [])
        thickness = Counter()
        for h in time_headers[2:]:
            match = re.search(r"_([0-9.]+)mm$", h)
            thickness[match.group(1) + " mm" if match else "unknown"] += 1
        alpha_grid = sheets.get("吸收系数", {}).get("axis_export_strings_sha256")
        n_grid = sheets.get("折射率", {}).get("axis_export_strings_sha256")
        pair_grid_equal = bool(alpha_grid and alpha_grid == n_grid)
        if not pair_grid_equal:
            mismatch.append({"workbook": item["path"], "problem": "alpha/n exported axis strings differ or missing"})
        have_raw = sheets.get("时域", {}).get("reference_numeric_rows", 0) > 0 and len(time_headers) > 2
        have_ref_header = len(time_headers) > 1 and time_headers[1] == "参考值"
        statuses = ["BLOCKED_MISSING_EXPORT_DEFINITION"]
        if not have_raw:
            statuses.insert(0, "BLOCKED_MISSING_RAW_SIGNAL")
        row = {
            "origin_group_id": item["origin_group_id"], "species": item["species"], "workbook": item["path"],
            "sha256": observed, "registry_hash_verified": True, "export_software": "unknown", "export_software_version": "unknown",
            "frequency_unit": "THz (export header Thz)" if freq_headers and freq_headers[0] == "Thz" else "unknown",
            "frequency_unit_evidence": "frequency/alpha/n first-column header only; actual transform settings unknown",
            "time_header": time_headers[0] if time_headers else "unknown", "time_unit": "unknown",
            "nominal_thickness_counts": dict(thickness), "thickness_unit": "mm (sample-header suffix)",
            "independently_measured_thickness": "unknown", "thickness_basis": "original time-sheet sample-column labels; not independent metrology",
            "alpha_definition": "user-declared corrected equivalent-transfer slab equations; exporter implementation unknown",
            "alpha_export_unit": "unknown", "raw_sample_time_trace_available": have_raw,
            "workbook_reference_trace_available": have_raw and have_ref_header,
            "sample_reference_pairing": "shared workbook reference column; specimen/reference acquisition identity unknown",
            "reference_type": "PE blank pellet (user declaration)",
            "PE_response_correction": "applied (user-confirmed); software/settings/PE constants/reference thickness unknown",
            "alpha_n_export_grid_equal": pair_grid_equal, "reconstruction_status": statuses,
            "evidence_source": [item["path"] + " first-row XML and axis", str(declaration_path.relative_to(ROOT))],
            "declaration_sha256": declaration_hash,
            "confirmation_scope": "header observations + shared user declaration; no inference of unknown units or processing factors",
        }
        workbooks.append(row)
        for name, view in sheets.items():
            grids.append({"origin_group_id": item["origin_group_id"], "sheet": name,
                          "axis_header": view["headers"][0] if view["headers"] else "unknown",
                          "axis_count": view["axis_count"], "axis_export_strings_sha256": view["axis_export_strings_sha256"]})
    table(out / "workbook_units_reference_audit.csv", workbooks)
    table(out / "export_axis_audit.csv", grids)
    # Twelve diagnostics selected by material and documented header exceptions,
    # never by classification score or closeness of reconstructed curves.
    choices = ["ZY02", "ZY01", "ZR01", "ZY03", "ZY04", "ZY07", "HQ01", "HQ03", "HQ04", "TM01", "TM03", "TM04"]
    byid = {w["origin_group_id"]: w for w in workbooks}
    plan = []
    for i, gid in enumerate(choices):
        w = byid[gid]
        plan.append({"diagnostic_pair": i + 1, "origin_group_id": gid, "species": w["species"], "workbook": w["workbook"],
                     "sample_column_rule": "first sample column in original order; ZY02 first column preserves nominal1mm exception",
                     "reference_column_rule": "same workbook time-sheet column B; requires acquisition-identity confirmation",
                     "export_workflow": "unknown", "status": w["reconstruction_status"],
                     "required_before_reconstruction": ["time unit", "actual sample/reference pairing", "documented FFT/window/phase settings",
                                                            "PE correction transfer definition", "thickness metrology and alpha export unit"]})
    dump(out / "bounded_reconstruction_protocol.json", {
        "planned_pairs": plan, "new_reconstructions_performed": 0,
        "formula_scope": declaration["reference"]["effective_transfer_definition"],
        "formulas": declaration["equations"],
        "prohibited": ["applying an air-reference formula directly to uncorrected PE ratio", "adjusting unknown factors to match old curves", "rescaling n from its range"],
        "comparison_metrics": ["maximum absolute difference on frozen common physical-frequency axis", "RMSE and relative L2 error with denominator definition frozen before reconstruction"],
        "acceptance_tolerances": "unknown: must derive from documented exporter precision and matched numerical processing before observing reconstruction errors",
        "acceptance_status": "BLOCKED_MISSING_EXPORT_DEFINITION", "scope": "12 workflow diagnostics, not an external validation sample-size argument"})
    status = "BLOCKED_MISSING_ARTIFACT"
    dump(out / "summary.json", {"status": status, "completed_scope": "18 workbook hashes, headers, nominal thicknesses, reference availability and exported grids audited",
        "workbooks": len(workbooks), "grid_mismatches": mismatch,
        "affected_comparison_pause": bool(mismatch), "pause_scope": "only affected workbook/grid merges" if mismatch else "none: no alpha/n axis mismatch detected",
        "unit_limit": "Unknown alpha/time units and correction implementation prevent optical reconstruction; they do not by themselves invalidate every fixed-prediction analysis.",
        "raw_signal_status": "workbook time and shared reference traces present; individual acquisition pairing not independently confirmed",
        "reconstruction_status": "BLOCKED_MISSING_EXPORT_DEFINITION", "new_fits": 0})
    (out / "REPORT.md").write_text("# E5: Workbook units and PE reference provenance\n\n"
        "All 18 active workbook hashes are checked against the authorized registry. The original first-column frequency label is `Thz`; time units and absorption export units remain unknown. Nominal thicknesses come from sample headers, including two 1 mm ZY02 columns; these are not independently measured thicknesses.\n\n"
        "Time traces and a shared workbook reference column are present. The user confirmed a PE blank reference and PE-response correction, but the acquisition pairing, exporter version and correction settings are unavailable. The 12-pair diagnostic therefore remains BLOCKED_MISSING_EXPORT_DEFINITION; no optical constants were reconstructed or rescaled. The table records missing raw traces separately if encountered. Unknown single units do not alone certify or invalidate all saved prediction comparisons.\n", encoding="utf-8")
    return {"status": status, "summary": "e5/summary.json"}


def mlp_audit():
    out = REPORT / "e6"
    rows = []
    table_path = ROOT / "paper/current/tables/all_final_model_settings.csv"
    with table_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if "mlp" in " ".join(str(v) for v in row.values()).lower():
                rows.append(row)
    table(out / "existing_mlp_summary_rows.csv", rows, list(rows[0]) if rows else ["status"])
    v4 = CACHE / "v4/runs"
    archives = []
    diagnostic_rows, replay_rows, curves = [], [], []
    feature_path = CACHE / "v4/train_only_features.npz"
    feature_lookup, alpha = {}, None
    if feature_path.is_file():
        import numpy as np
        import joblib
        if sha(feature_path) != "33540088d94293a87501b9a995d3002689214c5a8ca6694398818b5af0fb9f13":
            raise ValueError("MLP replay input is not the exact sealed original V4 TRAIN feature pack")
        with np.load(feature_path, allow_pickle=False) as pack:
            feature_lookup = {str(v): i for i, v in enumerate(pack["record_ids"])}
            alpha = pack["X_alpha"].copy()
    for p in sorted(v4.glob("matched_r*_f*/summary.json")):
        d = read_json(p)
        # Preserve the actual serialized diagnostic subtree rather than guessing
        # the early stopping IDs from a newly generated random split.
        relevant = {}
        def visit(obj, path=""):
            if isinstance(obj, dict):
                for k, val in obj.items():
                    key = path + "/" + str(k)
                    if "mlp" in str(k).lower():
                        relevant[key] = val
                    else:
                        visit(val, key)
            elif isinstance(obj, list):
                for i, val in enumerate(obj):
                    visit(val, path + "/" + str(i))
        visit(d)
        for row in d.get("results", []):
            if row.get("family") == "fullalpha_MLP":
                relevant["results/fullalpha_MLP"] = row
        if relevant:
            archives.append({"path": str(p.relative_to(CACHE)), "sha256": sha(p), "diagnostics": relevant})
        folder = p.parent
        checkpoint = folder / "models/fullalpha_MLP.joblib"
        complete = read_json(folder / "complete.json")
        expected = complete.get("artifacts", {}).get("models/fullalpha_MLP.joblib")
        if not checkpoint.is_file() or not expected or sha(checkpoint) != expected:
            raise ValueError("Existing MLP checkpoint missing or hash differs from pinned complete.json: " + str(folder))
        if alpha is None:
            raise ValueError("MLP checkpoint replay requires R1 original TRAIN feature recovery before this audit")
        # Only deserialize bytes from the user's fixed trusted repository after
        # verifying the immutable source seal; never load arbitrary uploads.
        bundle = joblib.load(checkpoint)
        model, scaler = bundle["model"], bundle["scaler"]
        manifest = read_json(folder / "job_manifest.json")
        ids = manifest["outer_validation_record_ids"]
        source_ids = manifest["source_record_ids"]
        if set(ids) & set(source_ids) or len(ids) != 60 or len(source_ids) != 240:
            raise ValueError("MLP archived outer roles are invalid")
        x = alpha[[feature_lookup[rid] for rid in ids]]
        if bundle["preprocessing"] == "snv":
            x = (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, ddof=1, keepdims=True), np.finfo(float).eps)
        elif bundle["preprocessing"] != "raw":
            raise ValueError("Unsupported original MLP representation; do not silently substitute")
        predicted = model.predict(scaler.transform(x))
        labels = bundle.get("class_labels")
        if labels is not None:
            predicted = np.asarray(labels)[np.asarray(predicted, dtype=int)]
        with gzip.open(folder / "validation_predictions.csv.gz", "rt", encoding="utf-8-sig", newline="") as f:
            saved = {r["record_id"]: r for r in csv.DictReader(f) if r["family"] == "fullalpha_MLP" and str(r["is_primary_k"]) == "1"}
        if set(saved) != set(ids):
            raise ValueError("MLP archived predictions do not cover exact outer-test IDs")
        mismatch = []
        for rid, pred in zip(ids, predicted):
            match = str(pred) == saved[rid]["prediction"]
            if not match:
                mismatch.append(rid)
            replay_rows.append({"context": folder.name, "record_id": rid, "saved_prediction": saved[rid]["prediction"],
                                "checkpoint_prediction": str(pred), "identical": match, "y_true": saved[rid]["y_true"]})
        if mismatch:
            raise ValueError("MLP checkpoint predictions do not strictly replay in " + folder.name + ": " + str(mismatch))
        losses = [float(v) for v in getattr(model, "loss_curve_", [])]
        validation_scores = [float(v) for v in (getattr(model, "validation_scores_", None) or [])]
        optimizer = getattr(model, "_optimizer", None)
        epochs = int(getattr(model, "n_iter_", bundle.get("n_iter", 0)))
        if bundle.get("n_iter") is not None and epochs != int(bundle["n_iter"]):
            raise ValueError("MLP checkpoint epoch diagnostics disagree")
        diagnostic_rows.append({"context": folder.name, "checkpoint_sha256": expected, "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_hash_verified": True, "prediction_replay_n": len(ids), "prediction_mismatches": len(mismatch),
            "preprocessing": bundle["preprocessing"], "hidden_layer_sizes": list(model.hidden_layer_sizes), "alpha": float(model.alpha),
            "learning_rate_init": float(model.learning_rate_init), "actual_epochs": epochs,
            "best_validation_epoch_earliest_max": int(np.argmax(validation_scores)) + 1 if validation_scores else "unknown",
            "best_validation_accuracy": float(model.best_validation_score_) if getattr(model, "best_validation_score_", None) is not None else "unknown",
            "internal_monitor": "accuracy", "external_selection": "inner macro-F1 primary",
            "loss_curve_length": len(losses), "validation_curve_length": len(validation_scores),
            "optimizer_update_steps": int(optimizer.t) if hasattr(optimizer, "t") else "unknown",
            "training_samples_seen_counter": int(model.t_) if hasattr(model, "t_") else "unknown",
            "convergence_warnings": bundle.get("convergence_warnings", "unknown"),
            "scaler_fit_ids_sha256": hashlib.sha256(json.dumps(source_ids, separators=(",", ":")).encode()).hexdigest(),
            "scaler_fit_ids_count": len(source_ids), "actual_gradient_train_ids": "unknown", "actual_early_stopping_ids": "unknown"})
        curves.append({"context": folder.name, "checkpoint_sha256": expected, "loss_curve": losses,
                       "internal_validation_accuracy_curve": validation_scores, "actual_scaler_fit_ids": source_ids,
                       "outer_test_ids": ids, "actual_early_stopping_ids": "not archived; never reconstructed from a new split"})
    table(out / "checkpoint_diagnostics.csv", diagnostic_rows, list(diagnostic_rows[0]) if diagnostic_rows else ["status"])
    table(out / "checkpoint_prediction_replay.csv", replay_rows, list(replay_rows[0]) if replay_rows else ["status"])
    dump(out / "checkpoint_training_curves.json", curves)
    dump(out / "existing_checkpoint_diagnostic_audit.json", {
        "status": "REUSED_VERIFIED" if archives else "BLOCKED_MISSING_ARTIFACT",
        "summary_table_source": str(table_path.relative_to(ROOT)), "summary_table_sha256": sha(table_path),
        "summary_rows": len(rows), "restored_diagnostics": archives,
        "prediction_evidence": "V4 validation_predictions.csv family fullalpha_MLP; source hashes recovered by E1",
        "model_checkpoint_evidence": {"hash_verified_and_inspected": len(diagnostic_rows), "prediction_replay_rows": len(replay_rows), "source": "pinned V4 complete.json model seals"},
        "actual_early_stopping_ids": "unknown unless explicitly logged in original checkpoint evidence",
        "loss_curve_limit": "saved checkpoints or diagnostics cannot retroactively establish unlogged actual early-stop entity IDs",
        "no_new_training": True})
    dump(out / "pending_mlp_config.json", {
        "RUN_MLP_EXTENSION": False, "RUN_MLP_FIXED_EPOCH": False, "status": "SKIPPED_BY_CONFIG",
        "protocol": "V4-compatible 3 repeats x5 outer x2 inner; frozen actual IDs from E1, separate from330 repeats",
        "representation": "full612 alpha SNV(ddof1), training-fold StandardScaler",
        "hidden_layer_sizes": [[32], [64]], "alpha": [0.0001, 0.001, 0.01], "learning_rate_init": [0.0003, 0.001],
        "candidates": 12, "solver": "adam", "activation": "relu", "batch_size": 32,
        "max_iter": 300, "validation_fraction": 0.2, "n_iter_no_change": 15,
        "tuning_seed": 42, "final_optimization_seeds": [42, 20261116, 20261117],
        "inner_selection": "macro-F1 first, preserve original V4 candidate-order tie break",
        "early_stopping_monitor": "sklearn internal accuracy; distinct from external macro-F1 selection",
        "planned_fits_if_explicitly_enabled": 405, "optional_fixed_epoch_extra_fits": 45,
        "required_actual_logs": ["gradient_train_ids", "early_stopping_ids", "inner_validation_ids", "outer_test_ids", "scaler_fit_ids", "actual_epochs", "best_epoch", "loss_curve", "validation_accuracy_curve", "update_steps", "model_sha256", "predictions"],
        "fixed_epoch_rule": "ceil(median of winning configuration inner-best epochs), no outer labels; explicitly verify actual completed epochs",
        "limits": ["outer-fit accuracy includes early-stop samples and is not pure gradient-train accuracy", "fixed-epoch contrast changes both sample use and stopping", "do not expand network/grid based on scores"]})
    dump(out / "summary.json", {"status": "SKIPPED_BY_CONFIG", "new_nn_fits": 0, "pending_fit_count": 405,
                               "existing_summary_rows": len(rows), "restored_diagnostic_files": len(archives),
                               "existing_checkpoint_hashes_verified": len(diagnostic_rows), "strictly_replayed_outer_predictions": len(replay_rows),
                               "independent_specimens": 300 if len(replay_rows) == 900 else "unknown"})
    return {"status": "SKIPPED_BY_CONFIG", "summary": "e6/summary.json"}


META_FIELDS = ["task", "class_label", "origin", "species", "procurement_lot_id", "procurement_evidence_path", "preparation_batch_id",
               "preparation_evidence_path", "physical_specimen_id", "pellet_id", "technical_scan_id", "measurement_batch_id", "measurement_date",
               "record_id", "workbook_path", "file_sha256", "reference_record_id", "role", "role_freeze_id", "label_access_stage", "is_new_real_batch",
               "pre_screen_labels_count", "label_cost", "cost_currency", "notes"]


def batch_protocol():
    out = REPORT / "e7"
    b = {n: read_json(ROOT / ("paper/current/protocols/" + n + ".json")) for n in ["B1", "B2", "B3", "B4", "B5"]}
    source_paths = [ROOT / ("paper/current/protocols/" + n + ".json") for n in b]
    # Exact definitions copied as machine-readable fragments with source hashes.
    inherited = {"B1_source_model": b["B1"]["source_model"], "B2_F02": b["B2"]["F02"], "B3_G02": b["B3"]["G02"],
                 "B4_H01": b["B4"]["H01"], "B5_I01": b["B5"]["I01"], "B5_I02": b["B5"]["I02"]}
    methods = [
        {"key": "no_update", "features": "source-fitted612 alpha SNV scaler", "labels": "Source only", "C": "C_source", "gamma": "original actual numeric gamma", "refit": False},
        {"key": "true_labels_full_refit", "features": "SNV rowwise; StandardScaler fitted Source+Support", "labels": "Source+true Support", "C": "C_source", "gamma": "scale recomputed on fitted representation; B2F02 full-pipeline endpoint", "refit": True},
        {"key": "frozen_scaler_gamma_refit", "features": "Source-fitted frozen scaler", "labels": "Source+true Support", "C": "C_source", "gamma": "original actual numeric gamma", "refit": True},
        {"key": "constant_C_times_N_refit", "features": "Source-fitted frozen scaler", "labels": "Source+true Support", "C": "C_source*N_source/(N_source+N_support)", "gamma": "original actual numeric gamma", "refit": True, "basis": "B4H01 augmented mass-normalized arm"},
        {"key": "source_loss_mass_control", "features": "Source-fitted frozen scaler", "labels": "Source only", "C": "C_source*(N_source+N_support)/N_source", "gamma": "original actual numeric gamma", "refit": True, "basis": "B4H01 source-only mass intervention"},
        {"key": "unlabeled_support_statistics", "features": "StandardScaler Source+Support spectra", "labels": "Source labels only; Support true label values excluded", "C": "C_source", "gamma": "original actual numeric gamma", "refit": True, "basis": "B3G02; class-stratified Support sampling remains label-informed"},
        {"key": "cyclic_wrong_support_labels", "features": "StandardScaler Source+Support spectra", "labels": "Source true; Support mapped to next class modulo frozen class order", "C": "C_source", "gamma": "original actual numeric gamma", "refit": True, "basis": "B3G02 artificial negative control, not a finding of mislabeled originals"},
    ]
    tasks = [
        {"task": "binary_material", "classes": ["枣仁", "酸枣仁"], "target": "material identity"},
        {"task": "four_material", "classes": ["枣仁", "酸枣仁", "黄芪", "天麻"], "target": "material identity"},
        {"task": "five_source_closed_set", "classes": ["ZY03", "ZY04", "ZY05", "ZY06", "ZY07"], "target": "酸枣仁 origin, all five labels present in training and test"},
    ]
    protocol = {
        "protocol_id": "focused-new-real-batch-support-query-20261116-v1", "status": "BLOCKED_WAITING_NEW_BATCH_DATA",
        "training_enabled": False, "tasks": tasks,
        "identity_hierarchy": ["procurement_lot", "preparation_batch", "physical_specimen", "pellet", "technical_scan"],
        "independence": "Require actual procurement/preparation/measurement evidence. Renaming, scans or repeated pellets do not create new independent batches.",
        "example_sampling_structure": {"real_procurement_lots_per_class": 3, "batches_per_lot": 1, "physical_specimens_per_batch": 30,
            "pellets_per_specimen": 1, "technical_scans_per_pellet": 2, "roles": "first two independently documented lots development; third untouched final-test lot; every class in every role",
            "support_pool_per_class_in_target_batch": 10, "fixed_query_per_class_in_target_batch": 20,
            "scope": "Illustrative operational design, not a powered or sufficient batch count; final batch Support use defines adaptation evaluation, not an untouched zero-shot endpoint"},
        "source_freeze": {"before_target_access": ["actual source IDs", "source model bytes/hash", "source scaler mean/scale", "C and actual numeric gamma", "class order", "feature/sampling/update rules", "cost convention"],
            "default_model": b["B1"]["source_model"], "selection": "Source-only inner validation; retain all prespecified arms; never select on Query"},
        "target_role_freeze": {"unit": "physical specimen", "roles": ["support_pool", "query"], "max_support_per_class": 10,
            "support_budgets_per_class": [1, 3, 5, 10], "draw_seeds": [2026111600, 2026111601, 2026111602],
            "nested_samples": True, "query_identical_across_budgets_methods_draws": True,
            "label_stratification": "uses true labels to satisfy per-class quota; log every screened label and costs including discarded candidates",
            "query_access": "isolated evaluation only, after prediction artifact sealed; no cross-sample statistics, selection, thresholds, early stop, fallback or method choice",
            "transductive": "Any Query-unlabeled distribution use requires separately named protocol and is forbidden in this one"},
        "methods": methods, "class_weight": "balanced inherited B1 model; document realized class effective C; no silent switch to B5I02 None",
        "exact_inherited_definition_fragments": inherited,
        "definition_sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "fixed_paper_commit": "47f3f69a8cf6d0cce063da702e818f5e093537de"} for p in source_paths],
        "metrics": ["accuracy", "balanced_accuracy", "macro_f1", "per_class_recall", "confusion_matrix", "paired_per_specimen_prediction_changes"],
        "uncertainty": "Keep physical specimens nested in actual batches; report per-batch results. Repeated Support draws do not increase Query specimen/batch N. At currently one known batch/source, between-batch variance is not identifiable.",
        "acceptance_before_training": ["all metadata required fields populated from evidence", "physical entities disjoint Source/Support/Query", "actual lots present in Source and target for every class", "frozen model+split+input hashes", "paired reference/grid/unit audits passed for intended merge", "Query inaccessible to fitting and selection", "new training explicitly enabled in a new authorization"],
        "entry_point": "python experiments/focused_followup_20261116/e5_e7_protocols.py --validate-new-batch metadata.csv",
        "entry_point_effect": "metadata validation only; never fits or enables a model",
    }
    dump(out / "new_batch_protocol.json", protocol)
    table(out / "metadata_template.csv", [], META_FIELDS)
    table(out / "label_screening_cost_template.csv", [], ["task", "candidate_id", "class_label", "label_requested_at", "label_cost", "currency", "accepted_to_support", "quota_reason", "query_accessed"])
    table(out / "batch_cost_template.csv", [], ["task", "class_label", "procurement_lot_id", "preparation_batch_id", "procurement_cost", "preparation_cost_per_specimen", "scan_cost", "label_cost", "physical_specimens", "pellets_per_specimen", "scans_per_pellet", "currency", "evidence_source"])
    sensitivity = []
    for task in tasks:
        for specimens in [20, 30, 50]:
            for rho in [0.0, 0.1, 0.3, 0.5]:
                nc = len(task["classes"])
                n = nc * 3 * specimens
                deff = 1 + (specimens - 1) * rho
                sensitivity.append({"task": task["task"], "classes": nc, "real_lots_per_class_assumed": 3,
                    "physical_specimens_per_lot_assumed": specimens, "technical_scans_per_specimen_assumed": 2,
                    "total_physical_specimens": n, "total_scans": n * 2, "within_lot_correlation_assumed": rho,
                    "design_effect_heuristic": deff, "effective_n_heuristic": n / deff,
                    "target_accuracy_difference_assumed_pp": 5,
                    "cost_formula": "classes*3*(procurement_cost_per_lot + specimens*(preparation_cost+2*scan_cost+label_cost))",
                    "power_claim": "none: correlation and5pp are assumptions; historical single-batch data cannot identify between-lot variance"})
    table(out / "sampling_budget_sensitivity.csv", sensitivity)
    (out / "REPORT.md").write_text("# E7: Real-batch sampling and Support/Query protocol\n\n"
        "Status: BLOCKED_WAITING_NEW_BATCH_DATA. The binary, four-material and five-source closed-set tasks have separate class definitions. The metadata template records procurement lot, preparation batch, physical specimen, pellet and scan. Each class needs actual distinct lots on Source and target sides. Holding out a source label is not the same five-class closed-set task.\n\n"
        "Three real lots per class and 30 physical specimens per lot are an example only. Costs remain explicit unknown inputs. The sensitivity table labels every within-lot correlation and5-percentage-point target difference as hypothetical; it makes no power claim. Technical scans and repeated Support draws do not create independent specimens.\n\n"
        "Freeze Source and all seven update rules before target access. The maximum Support pool and a physical-entity-disjoint Query are fixed first. Support1/3/5/10 use identical Query and nested sampled IDs. Quota sampling uses labels, so the screening ledger includes all labels spent to meet quotas. Query cannot enter cross-sample statistics, tuning or fallback. The supplied executable entry point validates metadata and cannot train; new real data and separate explicit enablement remain required.\n", encoding="utf-8")
    return {"status": "BLOCKED_WAITING_NEW_BATCH_DATA", "protocol": "e7/new_batch_protocol.json"}


def validate_metadata(path):
    errors = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(META_FIELDS) - set(reader.fieldnames or []))
        rows = list(reader)
    if missing:
        errors.append("missing columns: " + ",".join(missing))
    if not rows:
        errors.append("no new-batch records supplied")
    required = ["task", "class_label", "procurement_lot_id", "procurement_evidence_path", "preparation_batch_id", "preparation_evidence_path", "physical_specimen_id", "pellet_id", "technical_scan_id", "measurement_batch_id", "record_id", "file_sha256", "role", "role_freeze_id"]
    entities, counts, lots = defaultdict(set), defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    for i, row in enumerate(rows, 2):
        for field in required:
            if not row.get(field) or row.get(field, "").lower() == "unknown":
                errors.append(f"row{i}: missing evidence field {field}")
        role, task, cls = row.get("role"), row.get("task"), row.get("class_label")
        if role in ("support_pool", "query") and row.get("is_new_real_batch", "").lower() != "true":
            errors.append(f"row{i}: new real target batch is not confirmed")
        if role not in ("source", "support_pool", "query"):
            errors.append(f"row{i}: invalid role {role}")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("file_sha256", "")):
            errors.append(f"row{i}: file_sha256 must be the actual64-digit lowercase SHA256")
        if role == "query" and row.get("label_access_stage") not in ("not_accessed", "sealed_evaluation_only"):
            errors.append(f"row{i}: Query label-access stage violates sealed evaluation")
        entities[(task, row.get("physical_specimen_id"))].add(role)
        counts[task][(cls, role)] += 1
        lots[task][(cls, role)].add(row.get("procurement_lot_id"))
    for (task, specimen), roles in entities.items():
        if len(roles) > 1:
            errors.append(f"{task}/{specimen}: physical specimen overlaps roles {sorted(roles)}")
    expected = {"binary_material": ["枣仁", "酸枣仁"], "four_material": ["枣仁", "酸枣仁", "黄芪", "天麻"], "five_source_closed_set": ["ZY03", "ZY04", "ZY05", "ZY06", "ZY07"]}
    for task in counts:
        if task not in expected:
            errors.append("unknown task: " + str(task))
            continue
        for cls in expected[task]:
            for role in ("source", "support_pool", "query"):
                if not counts[task][(cls, role)]:
                    errors.append(f"{task}/{cls}: missing role {role}")
            if lots[task][(cls, "source")] & (lots[task][(cls, "support_pool")] | lots[task][(cls, "query")]):
                errors.append(f"{task}/{cls}: Source lot overlaps target lots")
    return {"metadata_path": str(path), "sha256": sha(path), "records": len(rows), "errors": errors,
            "metadata_valid": not errors, "training_enabled": False,
            "status": "BLOCKED_MISSING_ARTIFACT" if errors else "COMPLETED",
            "scope": "metadata checks only; acquisition-evidence authenticity and model enablement require separate review"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["all", "units", "mlp", "batch"], default="all")
    p.add_argument("--validate-new-batch", type=Path)
    args = p.parse_args(argv)
    if args.validate_new_batch:
        result = validate_metadata(args.validate_new_batch)
        dump(REPORT / "e7/new_batch_validation.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["metadata_valid"] else 2
    results = {}
    if args.phase in ("all", "units"):
        results["E5"] = units_report()
    if args.phase in ("all", "mlp"):
        results["E6"] = mlp_audit()
    if args.phase in ("all", "batch"):
        results["E7"] = batch_protocol()
    artifacts = []
    for module in results:
        folder = REPORT / module.lower()
        try:
            artifacts.append(str(folder.relative_to(ROOT)))
        except ValueError:
            artifacts.append(str(folder))
    receipt("E5_E6_E7", "COMPLETED", {"module_states": results, "artifacts": artifacts,
            "scope": "requested audit/protocol deliverables completed; optical reconstruction and new-batch training remain blocked"})
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
