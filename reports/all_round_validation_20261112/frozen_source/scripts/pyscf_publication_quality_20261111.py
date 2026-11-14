#!/usr/bin/env python3
"""Independent publication qualification from original molecular derivatives.

This is a new audit, not a revision of historical analysis flags. It authenticates
the complete displacement ledger and tests derivative identities before allowing
an isolated-molecule harmonic/IR claim. It does not identify experimental peaks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import pyscf_repair_modes_20261111 as modes

SCHEMA = "research4-pyscf-publication-quality-v1"
THRESHOLDS = {
    "orbital_gradient_norm": 1e-8,
    "energy_recheck_hartree": 1e-10,
    "electron_count_error": 1e-7,
    "nuclear_gradient_max_hartree_per_bohr": 1e-5,
    "nuclear_gradient_rms_hartree_per_bohr": 3e-6,
    "raw_relative_antisymmetry": 1e-3,
    "projected_relative_step_difference": 5e-3,
    "force_translation_relative_residual": 1e-3,
    "force_rotation_covariance_relative_residual": 1e-3,
    "dipole_translation_relative_residual": 1e-3,
    "dipole_rotation_covariance_relative_residual": 1e-3,
    "frequency_absolute_tolerance_cm1": 0.5,
    "frequency_relative_tolerance": 0.005,
    "degenerate_frequency_gap_cm1": 1.0,
    "degenerate_frequency_gap_relative": 0.001,
    "matched_subspace_minimum_overlap": 0.95,
    "ir_cluster_absolute_tolerance_km_mol": 0.1,
    "ir_cluster_relative_tolerance": 0.03,
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(value):
    if isinstance(value, (bool, str)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def electronic_quality(point):
    """Require individual native, fresh-Fock, energy and electron-count evidence."""
    audit = point.get("electronic_audit", point)
    residual = _finite(audit.get("fresh_orbital_gradient_norm"))
    energy = _finite(audit.get("fresh_energy_difference_hartree", audit.get("energy_recheck_difference_hartree")))
    electrons = _finite(audit.get("electron_count_error"))
    return bool(
        point.get("status") == "DONE" and point.get("qualified") is True and point.get("native_converged") is True
        and residual is not None and 0 <= residual <= THRESHOLDS["orbital_gradient_norm"]
        and energy is not None and abs(energy) <= THRESHOLDS["energy_recheck_hartree"]
        and electrons is not None and abs(electrons) <= THRESHOLDS["electron_count_error"]
        and _finite(point.get("energy_hartree")) is not None
    )


def _finite_array(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Incomplete/nonfinite original molecular array")
    return array


def _check_point_arrays(point, folder, manifest):
    for name, filename in (("coordinates_bohr", "coordinates_bohr.npy"),
                           ("gradient", "gradient.npy"), ("dipole_au", "dipole_au.npy")):
        target = folder / filename
        reference = point.get("array_refs", {}).get(name)
        if not isinstance(reference, dict) or not reference.get("sha256"):
            raise ValueError("Missing original point array seal: " + str(target))
        digest = sha256(target)
        if digest != reference["sha256"]:
            raise ValueError("Original point array seal mismatch: " + str(target))
        if not np.array_equal(np.load(target, allow_pickle=False), np.asarray(point[name])):
            raise ValueError("Original point JSON/array mismatch: " + str(target))
        manifest[str(target.resolve())] = digest


def derivative_sum_rules(hessian, dipole_derivative, coords, gradient, dipole, charge, masses):
    """Use exact translation and rotation covariance, including nonzero force.

    H (a x R) = a x gradient, D (a x R) = a x dipole, and
    D translations = charge * I. The rotation origin is the center of mass.
    The gradient correction prevents rejecting valid, slightly nonstationary data.
    """
    n = len(masses)
    centered = coords - np.average(coords, axis=0, weights=masses)
    center_of_mass = np.average(coords, axis=0, weights=masses)
    com_dipole = dipole - charge * center_of_mass
    translations = np.tile(np.eye(3), (n, 1)) / math.sqrt(n)
    hnorm = max(float(np.linalg.norm(hessian)), np.finfo(float).tiny)
    dnorm = max(float(np.linalg.norm(dipole_derivative)), np.finfo(float).tiny)
    rotation_h, rotation_d = [], []
    for axis in np.eye(3):
        direction = np.cross(axis, centered).ravel()
        length = float(np.linalg.norm(direction))
        if length > 1e-12:
            rotation_h.append((hessian @ direction - np.cross(axis, gradient).ravel()) / length)
            rotation_d.append((dipole_derivative @ direction - np.cross(axis, com_dipole)) / length)
    values = {
        "force_translation_relative_residual": float(np.linalg.norm(hessian @ translations) / hnorm),
        "force_rotation_covariance_relative_residual": float(np.linalg.norm(rotation_h) / hnorm),
        "dipole_translation_relative_residual": float(np.linalg.norm(dipole_derivative @ translations - charge * np.eye(3) / math.sqrt(n)) / dnorm),
        "dipole_rotation_covariance_relative_residual": float(np.linalg.norm(rotation_d) / dnorm),
    }
    values["force_sum_rules_passed"] = all(values[k] <= THRESHOLDS[k] for k in
        ("force_translation_relative_residual", "force_rotation_covariance_relative_residual"))
    values["dipole_sum_rules_passed"] = all(values[k] <= THRESHOLDS[k] for k in
        ("dipole_translation_relative_residual", "dipole_rotation_covariance_relative_residual"))
    return values


def _clusters(frequencies):
    groups = []
    for index, frequency in enumerate(frequencies):
        if not groups:
            groups.append([index])
            continue
        previous = frequencies[groups[-1][-1]]
        gap = max(THRESHOLDS["degenerate_frequency_gap_cm1"],
                  THRESHOLDS["degenerate_frequency_gap_relative"] * max(abs(frequency), abs(previous)))
        if abs(frequency - previous) <= gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def compare_steps(coarse, fine):
    """Compare every frequency and degenerate-subspace total IR/tensor strength."""
    delta = fine["projected_hessian"] - coarse["projected_hessian"]
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(fine["projected_hessian"]), np.finfo(float).tiny))
    spectral = float(np.linalg.norm(delta, ord=2)) if delta.size else 0.0
    overlap = np.abs(coarse["mass_weighted_modes"].T @ fine["mass_weighted_modes"])
    rows, columns = linear_sum_assignment(-overlap)
    inverse = {int(column): int(row) for row, column in zip(rows, columns)}
    frequencies = fine["signed_frequencies_cm1"]
    frequency_checks = []
    for j in range(len(frequencies)):
        i = inverse[j]
        difference = float(abs(coarse["signed_frequencies_cm1"][i] - frequencies[j]))
        tolerance = max(THRESHOLDS["frequency_absolute_tolerance_cm1"],
                        THRESHOLDS["frequency_relative_tolerance"] * abs(float(frequencies[j])))
        frequency_checks.append({"coarse_index": i, "fine_index": j,
            "absolute_difference_cm1": difference, "tolerance_cm1": tolerance,
            "passed": difference <= tolerance})
    groups = []
    factor = modes.physical_conversion_constants()["atomic_dipole_derivative_squared_to_km_mol"]
    for final_indices in _clusters(frequencies):
        first_indices = [inverse[j] for j in final_indices]
        subspace = coarse["mass_weighted_modes"][:, first_indices].T @ fine["mass_weighted_modes"][:, final_indices]
        singular = np.linalg.svd(subspace, compute_uv=False)
        minimum_overlap = float(np.min(singular)) if len(singular) else 0.0
        # Tensor D_Q D_Q^T and its trace are invariant under rotation within a
        # degenerate normal-mode subspace, unlike individual mode intensities.
        first_d = coarse["normal_coordinate_dipole_derivatives_au_per_sqrt_amu"][:, first_indices]
        final_d = fine["normal_coordinate_dipole_derivatives_au_per_sqrt_amu"][:, final_indices]
        first_tensor, final_tensor = factor * first_d @ first_d.T, factor * final_d @ final_d.T
        difference = float(np.linalg.norm(final_tensor - first_tensor))
        scale = max(float(np.linalg.norm(first_tensor)), float(np.linalg.norm(final_tensor)))
        tolerance = THRESHOLDS["ir_cluster_absolute_tolerance_km_mol"] + THRESHOLDS["ir_cluster_relative_tolerance"] * scale
        groups.append({"coarse_indices": first_indices, "fine_indices": final_indices,
            "minimum_subspace_overlap": minimum_overlap,
            "coarse_total_ir_km_mol": float(np.trace(first_tensor)),
            "fine_total_ir_km_mol": float(np.trace(final_tensor)),
            "ir_tensor_difference_km_mol": difference, "ir_tensor_tolerance_km_mol": tolerance,
            "subspace_passed": minimum_overlap >= THRESHOLDS["matched_subspace_minimum_overlap"],
            "ir_passed": difference <= tolerance})
    return {
        "projected_relative_step_difference": relative,
        "projected_spectral_step_difference_hartree_per_amu_bohr2": spectral,
        "matrix_agreement_passed": relative <= THRESHOLDS["projected_relative_step_difference"],
        "positive_spectral_margin_passed": bool(len(fine["eigenvalues"]) and fine["eigenvalues"][0] > spectral),
        "frequency_checks": frequency_checks, "all_frequencies_agree": bool(frequency_checks) and all(x["passed"] for x in frequency_checks),
        "ir_subspace_checks": groups,
        "all_subspaces_agree": bool(groups) and all(x["subspace_passed"] for x in groups),
        "ir_converged": bool(groups) and all(x["ir_passed"] for x in groups),
        "interpretation": "Empirical step sensitivity, not a rigorous error bound or a grid/basis convergence proof.",
    }


def audit_publication(modes_dir, finite_differences_dir, center_point, *, electronic_stability_passed, output=None):
    """Authenticate original data and independently issue conservative claim gates.

    Missing/corrupt evidence yields a failed gate and a persisted receipt, never
    a pass inherited from historical flags. Local restart densities are unused.
    """
    folder, fd = Path(modes_dir), Path(finite_differences_dir)
    destination = Path(output) if output is not None else folder / "publication_quality.json"
    manifest, issues, steps_data, comparisons = {}, [], [], []
    gates = {key: False for key in (
        "source_artifacts_authenticated", "distinct_steps", "center_electronic_quality",
        "center_stationary", "complete_unique_displacement_ledger", "consistent_electronic_model_settings", "all_points_electronically_qualified",
        "raw_derivatives_reassembled", "all_vibrational_eigenvalues_positive", "raw_antisymmetry_passed",
        "all_step_comparisons_passed", "force_sum_rules_passed", "dipole_sum_rules_passed",
        "ir_step_convergence_passed", "orbital_stability_passed")}
    gates["orbital_stability_passed"] = electronic_stability_passed is True
    original_flags = None
    point_count = 0
    try:
        result_path = folder / "result.json"
        saved = json.loads(result_path.read_text())
        original_flags = saved.get("qualification")
        manifest[str(result_path.resolve())] = sha256(result_path)
        modes.verify_analysis(folder, write_receipt=False)
        provenance = saved["input_provenance"]
        expected_fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if expected_fingerprint != saved["input_fingerprint"]:
            raise ValueError("Mode-analysis provenance fingerprint mismatch")
        for artifact in saved["artifacts"].values():
            path = folder / artifact["path"]
            manifest[str(path.resolve())] = sha256(path)
        symbols = saved["symbols"]
        ncart, natom = 3 * len(symbols), len(symbols)
        steps = provenance["steps_bohr"]
        count = saved["matrix_count"]
        if count != len(steps) or count < 2:
            raise ValueError("At least two complete, explicitly labeled finite-difference steps required")
        gates["distinct_steps"] = all(_finite(h) is not None and h > 0 for h in steps) and len(set(steps)) == len(steps)
        if not gates["distinct_steps"]:
            issues.append("Distinct positive finite-difference steps were not demonstrated")
        with np.load(folder / "analysis_arrays.npz", allow_pickle=False) as arrays:
            coords = _finite_array(arrays["coordinates_bohr"], (natom, 3))
            gradient = _finite_array(center_point.get("gradient"), (natom, 3))
            dipole = _finite_array(center_point.get("dipole_au"), (3,))
            if center_point.get("symbols") != symbols or not np.array_equal(np.asarray(center_point.get("coordinates_bohr")), coords):
                raise ValueError("Center point identity/geometry differs from the normal-mode center")
            center_settings = center_point.get("settings")
            if not isinstance(center_settings, dict) or not center_settings:
                raise ValueError("Center point electronic model settings are missing")
            gates["center_electronic_quality"] = electronic_quality(center_point)
            gmax, grms = float(np.max(np.abs(gradient))), float(np.sqrt(np.mean(gradient ** 2)))
            gates["center_stationary"] = gmax <= THRESHOLDS["nuclear_gradient_max_hartree_per_bohr"] and grms <= THRESHOLDS["nuclear_gradient_rms_hartree_per_bohr"]
            all_qualified, fingerprints, ledger_complete = True, set(), True
            expected_folders = {f"h{i}" for i in range(count)}
            actual_folders = {p.name for p in fd.glob("h*") if p.is_dir()}
            if actual_folders != expected_folders:
                raise ValueError("Unexpected/missing finite-difference step directory")
            for hi, step in enumerate(steps):
                expected_names = {f"c{j:03d}-{sign}" for j in range(ncart) for sign in ("plus", "minus")}
                actual_names = {p.name for p in (fd / f"h{hi}").glob("c*") if p.is_dir()}
                if actual_names != expected_names:
                    raise ValueError("Missing, duplicate or unexpected Cartesian displacement identity")
                points = {1: [], -1: []}
                for j in range(ncart):
                    for sign, label in ((1, "plus"), (-1, "minus")):
                        point_dir = fd / f"h{hi}" / f"c{j:03d}-{label}"
                        point_path = point_dir / "result.json"
                        point = json.loads(point_path.read_text())
                        manifest[str(point_path.resolve())] = sha256(point_path)
                        displaced = coords.copy()
                        displaced.flat[j] += sign * step
                        if point.get("symbols") != symbols or not np.array_equal(np.asarray(point.get("coordinates_bohr")), displaced):
                            raise ValueError("Original displacement geometry/atom identity mismatch: " + str(point_dir))
                        if point.get("settings") != center_settings:
                            raise ValueError("Electronic model settings differ between center and displacement: " + str(point_dir))
                        fingerprint = point.get("input_fingerprint")
                        if not isinstance(fingerprint, str) or not fingerprint or fingerprint in fingerprints:
                            ledger_complete = False
                        fingerprints.add(fingerprint)
                        _check_point_arrays(point, point_dir, manifest)
                        all_qualified = all_qualified and electronic_quality(point)
                        _finite_array(point["gradient"], (natom, 3))
                        _finite_array(point["dipole_au"], (3,))
                        points[sign].append(point)
                        point_count += 1
                # Independent assembly uses columns d(g_i)/d(R_j).
                hessian = np.stack([(np.asarray(p["gradient"]) - np.asarray(m["gradient"])).ravel() / (2 * step)
                                    for p, m in zip(points[1], points[-1])], axis=1)
                derivative = np.stack([(np.asarray(p["dipole_au"]) - np.asarray(m["dipole_au"])) / (2 * step)
                                       for p, m in zip(points[1], points[-1])], axis=1)
                if not np.array_equal(hessian, arrays[f"step_{hi}_raw_hessian"]) or not np.array_equal(derivative, arrays[f"step_{hi}_dipole_derivative_au"]):
                    raise ValueError("Original displacement reassembly differs from saved mode inputs")
                analysis = modes.normal_modes(symbols, coords, hessian, derivative)
                rules = derivative_sum_rules(hessian, derivative, coords, gradient, dipole,
                    float(center_point.get("settings", {}).get("charge", 0)), analysis["masses_amu"])
                steps_data.append({"step_bohr": step, "analysis": analysis, "sum_rules": rules})
            gates["complete_unique_displacement_ledger"] = ledger_complete and point_count == 2 * ncart * count
            gates["consistent_electronic_model_settings"] = True
            gates["all_points_electronically_qualified"] = all_qualified
            gates["raw_derivatives_reassembled"] = True
            gates["source_artifacts_authenticated"] = True
        analyses = [x["analysis"] for x in steps_data]
        fine_index = int(np.argmin(steps))
        for i, analysis in enumerate(analyses):
            if i != fine_index:
                comparisons.append({"coarse_index": i, "fine_index": fine_index,
                    **compare_steps(analysis, analyses[fine_index])})
        gates["all_vibrational_eigenvalues_positive"] = all(len(a["eigenvalues"]) > 0 and np.all(a["eigenvalues"] > 0) for a in analyses)
        gates["raw_antisymmetry_passed"] = all(a["diagnostics"]["raw_relative_antisymmetry"] <= THRESHOLDS["raw_relative_antisymmetry"] for a in analyses)
        gates["all_step_comparisons_passed"] = bool(comparisons) and all(
            c["matrix_agreement_passed"] and c["positive_spectral_margin_passed"] and c["all_frequencies_agree"] and c["all_subspaces_agree"] for c in comparisons)
        gates["force_sum_rules_passed"] = all(x["sum_rules"]["force_sum_rules_passed"] for x in steps_data)
        gates["dipole_sum_rules_passed"] = all(x["sum_rules"]["dipole_sum_rules_passed"] for x in steps_data)
        gates["ir_step_convergence_passed"] = bool(comparisons) and all(c["ir_converged"] for c in comparisons)
    except Exception as error:
        issues.append(type(error).__name__ + ": " + str(error))
    minimum_keys = [key for key in gates if key not in ("dipole_sum_rules_passed", "ir_step_convergence_passed")]
    minimum = all(gates[key] for key in minimum_keys)
    ready = all(gates.values())
    receipt = {
        "schema": SCHEMA, "status": "AUDIT_COMPLETE", "at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": THRESHOLDS, "thresholds_relaxed_automatically": False,
        "gates": gates, "failed_gates": [key for key, value in gates.items() if not value], "issues": issues,
        "center_point_sha256": hashlib.sha256(json.dumps(center_point, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "authenticated_artifact_sha256": manifest, "original_displacement_point_count": point_count,
        "step_diagnostics": [{"step_bohr": x["step_bohr"], **x["analysis"]["diagnostics"], "sum_rules": x["sum_rules"]} for x in steps_data],
        "step_comparisons": comparisons,
        "historical_qualification_flags": original_flags,
        "historical_flags_used_as_acceptance": False,
        "minimum_verified": minimum, "paper_ready_molecular_harmonic_data": ready,
        "chemical_peak_assignment_verified": False, "global_conformer_minimum_verified": False,
        "scope": "Original derivative ledger authenticated and numerical molecular harmonic/IR qualification independently checked; no new SCF, grid/basis convergence or experimental solid-state peak assignment established.",
    }
    atomic_json(destination, receipt)
    return receipt
