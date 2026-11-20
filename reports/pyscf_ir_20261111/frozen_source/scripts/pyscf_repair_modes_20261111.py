#!/usr/bin/env python3
"""Signed molecular modes and IR intensities from complete numerical derivatives.

The Hessian convention is ``H[i,j] = d(gradient[i])/d(coordinate[j])`` in
Hartree/Bohr**2. Dipole derivatives have shape (3, 3*N), in atomic dipole/Bohr,
and must differentiate the *total* nuclear-plus-electronic molecular dipole.
No SCF, geometry optimization, frequency sign editing, or mode deletion occurs
here. Raw Hessian antisymmetry is reported before symmetric diagonalization.

Average atomic masses are PySCF's standard elemental masses, not electron masses.
Mass-weighted eigenvectors u obey u.T@u=I. The corresponding Cartesian normal
coordinates are L=M**(-1/2)u, with units amu**(-1/2); Euclidean-unit Cartesian
display directions are saved separately and are never used for IR intensities.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import constants
from scipy.optimize import linear_sum_assignment


SCHEMA = "research4-pyscf-frequency-repair-modes-v1"
QUALIFICATION_THRESHOLDS = {
    "nuclear_gradient_max_hartree_per_bohr": 1e-5,
    "nuclear_gradient_rms_hartree_per_bohr": 3e-6,
    "fresh_orbital_gradient_norm": 1e-7,
    "absolute_energy_recheck_difference_hartree": 1e-10,
    "absolute_electron_count_error_if_provided": 1e-7,
    "raw_relative_antisymmetry": 1e-3,
    "projected_relative_step_difference": 5e-3,
}

# Same standard average masses as pyscf.data.elements.MASSES in PySCF 2.13.0.
# This small fallback lets purely algebraic tests run without loading PySCF.
# Elements outside this explicit table require PySCF's complete mass registry.
_STANDARD_MASS_FALLBACK = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403163, "Ne": 20.1797, "Na": 22.98976928,
    "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(value.shape).encode() + b"\0" + value.tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def physical_conversion_constants() -> dict[str, float | str]:
    """Expose the complete unit convention used by frequencies and intensities.

    For eigenvalues in Eh/(amu*a0**2), cyclic frequency is
    sqrt(Eh/(amu*a0**2))*sqrt(eigenvalue)/(2*pi). Divide by c*100 for cm^-1.

    Standard integrated IR intensity is N_A/(12*epsilon_0*c**2) times the squared
    dipole derivative in C**2/kg. An atomic dipole/Bohr is e, so the factor is
    N_A*e**2/(12*epsilon_0*c**2*m_u*1000) km/mol. The standard integrated km/mol
    convention does not include ln(10), used separately for decadic broadening.
    This agrees with pyscf-properties' prop.infrared.rhf.kernel_ir conversion.
    """
    hartree = constants.physical_constants["atomic unit of energy"][0]
    bohr = constants.physical_constants["Bohr radius"][0]
    amu = constants.physical_constants["atomic mass constant"][0]
    angular_per_au = math.sqrt(hartree / (amu * bohr**2))
    return {
        "hartree_joule": hartree,
        "bohr_metre": bohr,
        "atomic_mass_unit_kg": amu,
        "speed_of_light_metre_per_second": constants.c,
        "elementary_charge_coulomb": constants.e,
        "avogadro_per_mole": constants.N_A,
        "vacuum_permittivity_farad_per_metre": constants.epsilon_0,
        "eigenvalue_sqrt_to_cm1": angular_per_au / (2 * math.pi * constants.c * 100),
        "cm1_to_THz": constants.c * 100 / 1e12,
        "atomic_dipole_derivative_squared_to_km_mol": (
            constants.N_A * constants.e**2
            / (12 * constants.epsilon_0 * constants.c**2 * amu * 1000)
        ),
        "IR_convention": "standard integrated km/mol, no ln(10), unbroadened",
        "IR_reference_implementation": (
            "https://github.com/pyscf/properties/blob/master/pyscf/prop/infrared/rhf.py"
        ),
    }


def atomic_masses(symbols: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return standard isotope-average atomic masses in amu, preserving atom order."""
    try:
        from pyscf.data import elements
    except ImportError:
        try:
            masses = np.asarray([_STANDARD_MASS_FALLBACK[x] for x in symbols], dtype=float)
        except KeyError as error:
            raise ValueError("PySCF is required for this element's standard mass") from error
    else:
        try:
            masses = np.asarray([elements.MASSES[elements.charge(x)] for x in symbols], dtype=float)
        except (IndexError, KeyError, RuntimeError) as error:
            raise ValueError("Unrecognized elemental symbol or mass") from error
    if masses.shape != (len(symbols),) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Finite positive standard masses are required; ghost atoms are unsupported")
    return masses


def external_basis(
    coordinates_bohr: np.ndarray, masses_amu: np.ndarray, rank_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return orthonormal external and vibrational bases in mass-weighted space.

    Six candidate vectors are mass-weighted translations and COM rotations.
    Normalizing each nonzero candidate before SVD avoids a length-unit-dependent
    translation/rotation rank decision. A linear molecule has only five external
    directions. No assumption of six directions is imposed on atoms or diatomics.
    """
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    if coordinates.shape != (len(masses), 3) or len(masses) == 0:
        raise ValueError("Coordinates must have shape (N,3) for N positive masses")
    if not np.isfinite(coordinates).all() or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Finite coordinates and positive finite masses are required")
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    root_mass = np.sqrt(masses)[:, None]
    axes = np.eye(3)
    translations = [(root_mass * axis).reshape(-1) for axis in axes]
    rotations = [(root_mass * np.cross(axis, centered)).reshape(-1) for axis in axes]
    candidates = np.stack(translations + rotations, axis=1)
    lengths = np.linalg.norm(candidates, axis=0)
    # Exactly vanishing rotation axes carry no information and remain zero.
    normalized = candidates / np.where(lengths > 1e-14, lengths, 1.0)
    normalized[:, lengths <= 1e-14] = 0.0
    left, singular, _ = np.linalg.svd(normalized, full_matrices=True)
    threshold = rank_tolerance * singular[0] if len(singular) else rank_tolerance
    rank = int(np.count_nonzero(singular > threshold))
    external = left[:, :rank]
    vibrational = left[:, rank:]
    return external, vibrational, {
        "external_rank": rank,
        "vibrational_mode_count": int(3 * len(masses) - rank),
        "external_candidate_singular_values": singular.tolist(),
        "external_rank_relative_tolerance": rank_tolerance,
        "external_basis_orthogonality_error": float(np.linalg.norm(external.T @ external - np.eye(rank))),
        "vibrational_basis_external_overlap": float(np.linalg.norm(external.T @ vibrational)),
        "rotor_type": "ATOM" if len(masses) == 1 else ("LINEAR" if rank == 5 else "NONLINEAR"),
    }


def signed_frequencies(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(eigenvalues, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Finite signed eigenvalues are required")
    conversion = physical_conversion_constants()
    cm1 = np.sign(values) * np.sqrt(np.abs(values)) * conversion["eigenvalue_sqrt_to_cm1"]
    return cm1, cm1 * conversion["cm1_to_THz"]


def ir_intensities(
    dipole_derivative_au: np.ndarray, cartesian_mass_normalized_modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract total-dipole derivatives with mass-normalized Cartesian modes.

    Dipole derivative rows are x/y/z dipole components; columns are atom-major
    Cartesian coordinates. Normal modes have shape (3*N, n_modes), in amu^-1/2.
    """
    derivatives = np.asarray(dipole_derivative_au, dtype=float)
    modes = np.asarray(cartesian_mass_normalized_modes, dtype=float)
    if derivatives.ndim != 2 or derivatives.shape[0] != 3 or modes.ndim != 2 or derivatives.shape[1] != modes.shape[0]:
        raise ValueError("Dipole derivative must be (3,3N) and modes must be (3N,n_modes)")
    if not np.isfinite(derivatives).all() or not np.isfinite(modes).all():
        raise ValueError("Finite total-dipole derivatives and normal modes are required")
    normal_derivatives = derivatives @ modes
    factor = physical_conversion_constants()["atomic_dipole_derivative_squared_to_km_mol"]
    intensities = factor * np.einsum("xi,xi->i", normal_derivatives, normal_derivatives)
    return intensities, normal_derivatives


def normal_modes(
    symbols: list[str] | tuple[str, ...], coordinates_bohr: np.ndarray,
    hessian: np.ndarray, dipole_derivative: np.ndarray | None = None,
) -> dict[str, Any]:
    """Analyze one full, raw Cartesian Hessian, retaining every signed mode."""
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    masses = atomic_masses(symbols)
    external, basis, geometry = external_basis(coordinates, masses)
    raw = np.asarray(hessian, dtype=float)
    ncart = len(symbols) * 3
    if raw.shape != (ncart, ncart) or not np.isfinite(raw).all():
        raise ValueError("A finite complete (3N,3N) Hessian is required; incomplete arrays cannot be zero-filled")
    antisymmetric = raw - raw.T
    raw_norm = float(np.linalg.norm(raw))
    denominator = max(raw_norm, np.finfo(float).tiny)
    symmetric = (raw + raw.T) / 2
    inverse_root_mass = 1 / np.sqrt(np.repeat(masses, 3))
    weighted = inverse_root_mass[:, None] * symmetric * inverse_root_mass[None, :]
    projected = basis.T @ weighted @ basis
    eigenvalues, reduced_modes = np.linalg.eigh(projected)
    modes = basis @ reduced_modes
    # Eigenvectors have an arbitrary sign. Canonical signs aid deterministic replay.
    if modes.shape[1]:
        indices = np.argmax(np.abs(modes), axis=0)
        signs = np.sign(modes[indices, np.arange(modes.shape[1])])
        modes *= np.where(signs == 0, 1, signs)[None, :]
    cartesian = inverse_root_mass[:, None] * modes
    lengths = np.linalg.norm(cartesian, axis=0)
    display = cartesian / np.where(lengths > 0, lengths, 1)[None, :]
    cm1, thz = signed_frequencies(eigenvalues)
    intensities = normal_dipole = None
    if dipole_derivative is not None:
        intensities, normal_dipole = ir_intensities(dipole_derivative, cartesian)
    diagnostics = {
        **geometry,
        "raw_relative_antisymmetry": float(np.linalg.norm(antisymmetric) / denominator),
        "raw_max_absolute_antisymmetry_hartree_per_bohr2": float(np.max(np.abs(antisymmetric))),
        "mass_weighted_external_residual_frobenius": float(np.linalg.norm(weighted @ external)),
        "mass_weighted_external_residual_relative": float(np.linalg.norm(weighted @ external) / max(np.linalg.norm(weighted), np.finfo(float).tiny)),
        "mass_weighted_mode_orthogonality_error": float(np.linalg.norm(modes.T @ modes - np.eye(len(eigenvalues)))),
        "positive_count": int(np.count_nonzero(eigenvalues > 0)),
        "imaginary_count": int(np.count_nonzero(eigenvalues < 0)),
        "exact_zero_count": int(np.count_nonzero(eigenvalues == 0)),
        "lowest_signed_frequency_cm1": float(cm1[0]) if len(cm1) else None,
        "lowest_signed_eigenvalue_hartree_per_amu_bohr2": float(eigenvalues[0]) if len(eigenvalues) else None,
        "positive_modes_in_0p1_to_1p5_THz": int(np.count_nonzero((thz >= 0.1) & (thz <= 1.5) & (eigenvalues > 0))),
    }
    return {
        "masses_amu": masses, "external_basis": external, "vibrational_basis": basis,
        "raw_hessian": raw, "symmetric_hessian": symmetric,
        "mass_weighted_hessian": weighted, "projected_hessian": projected,
        "eigenvalues": eigenvalues, "mass_weighted_modes": modes,
        "cartesian_mass_normalized_modes": cartesian, "cartesian_unit_display_modes": display,
        "reduced_masses_amu": 1 / np.where(lengths > 0, lengths**2, np.inf),
        "signed_frequencies_cm1": cm1, "signed_frequencies_THz": thz,
        "IR_intensities_km_mol": intensities,
        "normal_coordinate_dipole_derivatives_au_per_sqrt_amu": normal_dipole,
        "diagnostics": diagnostics,
    }


def _lookup(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    for container in ("diagnostics", "quality", "electronic_diagnostics", "fresh_diagnostics"):
        nested = data.get(container)
        if isinstance(nested, dict):
            for name in names:
                value = nested.get(name)
                if value is not None:
                    return value
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (bool, str)) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _field_present(data: dict[str, Any], name: str) -> bool:
    if name in data:
        return True
    return any(
        isinstance(data.get(container), dict) and name in data[container]
        for container in ("diagnostics", "quality", "electronic_diagnostics", "fresh_diagnostics")
    )


def electronic_quality(data: dict[str, Any] | None) -> dict[str, Any]:
    """Apply independent bounds without overriding a stricter upstream failure.

    The core calculation uses its stricter orbital-residual target of 1e-8.
    Its explicit ``qualified=False`` (or invalid declared qualification) vetoes
    this analysis even when the separate 1e-7 analysis bound would pass. A true
    upstream flag never replaces the independent checks performed here.
    """
    data = data or {}
    native = _lookup(data, "native_converged", "native_pyscf_converged") is True
    residual = _finite_number(_lookup(data, "fresh_orbital_gradient_norm", "orbital_gradient_norm"))
    energy = _finite_number(_lookup(data, "energy_recheck_difference_hartree", "fresh_energy_difference_hartree"))
    upstream_declared = _field_present(data, "qualified")
    upstream_qualified = _lookup(data, "qualified")
    upstream_ok = not upstream_declared or upstream_qualified is True
    electron_declared = _field_present(data, "electron_count_error")
    electron_error = _finite_number(_lookup(data, "electron_count_error"))
    electron_ok = not electron_declared or (
        electron_error is not None
        and abs(electron_error) <= QUALIFICATION_THRESHOLDS["absolute_electron_count_error_if_provided"]
    )
    orbital_ok = residual is not None and 0 <= residual <= QUALIFICATION_THRESHOLDS["fresh_orbital_gradient_norm"]
    energy_ok = energy is not None and abs(energy) <= QUALIFICATION_THRESHOLDS["absolute_energy_recheck_difference_hartree"]
    return {
        "native_converged": native,
        "fresh_orbital_gradient_norm": residual,
        "energy_recheck_difference_hartree": energy,
        "orbital_residual_passed": orbital_ok,
        "energy_recheck_passed": energy_ok,
        "upstream_qualification_declared": upstream_declared,
        "upstream_qualified": upstream_qualified,
        "upstream_qualification_passed": upstream_ok,
        "electron_count_error_declared": electron_declared,
        "electron_count_error": electron_error,
        "electron_count_check_passed": bool(electron_ok),
        "qualified": bool(native and orbital_ok and energy_ok and upstream_ok and electron_ok),
    }


def _point_quality_summary(point_quality: Any) -> dict[str, Any]:
    if point_quality is None:
        return {"metadata_present": False, "all_points_qualified": False, "all_native_converged": False}
    if isinstance(point_quality, dict):
        if "points" in point_quality:
            return _point_quality_summary(point_quality["points"])
        if "all_points_qualified" in point_quality:
            # The orchestrator can supply its independently sealed point summary.
            # Record that this is an aggregate claim, not individual replay here.
            return {**point_quality, "metadata_present": True, "qualification_evidence": "ORCHESTRATOR_AGGREGATE"}
        point_quality = [point_quality]
    if not isinstance(point_quality, (list, tuple)):
        raise ValueError("point_quality must be point dictionaries or a sealed aggregate")
    flattened = []
    for item in point_quality:
        if isinstance(item, (list, tuple)):
            flattened.extend(item)
        else:
            flattened.append(item)
    if any(not isinstance(item, dict) for item in flattened):
        raise ValueError("Each point's quality metadata must be a dictionary")
    checks = [electronic_quality(item) for item in flattened]
    finite_residuals = [item["fresh_orbital_gradient_norm"] for item in checks if item["fresh_orbital_gradient_norm"] is not None]
    return {
        "metadata_present": True, "qualification_evidence": "INDIVIDUAL_ELECTRONIC_RECHECKS",
        "point_count": len(checks),
        "qualified_point_count": sum(item["qualified"] for item in checks),
        "native_unconverged_point_count": sum(not item["native_converged"] for item in checks),
        "all_points_qualified": bool(checks) and all(item["qualified"] for item in checks),
        "all_native_converged": bool(checks) and all(item["native_converged"] for item in checks),
        "max_fresh_orbital_gradient_norm": max(finite_residuals) if finite_residuals else None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _matrix_sequence(values: Any, dimension: tuple[int, int], name: str) -> list[np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.shape == dimension:
        array = array[None, :, :]
    if array.ndim != 3 or tuple(array.shape[1:]) != dimension or not len(array):
        raise ValueError(f"{name} must be complete matrices of shape {dimension}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite or missing numerical entries")
    return [np.array(item, copy=True) for item in array]


def analyze_modes(
    symbols: list[str] | tuple[str, ...], coordinates_bohr: np.ndarray,
    hessians: Any, dipole_derivatives: Any, output: str | Path,
    center_quality: dict[str, Any] | None = None, point_quality: Any = None,
    *, steps_bohr: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Write complete signed-mode/IR data and conservative qualification metadata.

    The usual matrix order is h=.001 then h=.0005 Bohr. Explicit ``steps_bohr``
    labels are optional; qualification requires at least two complete matrices.
    Step disagreement is an empirical numerical error diagnostic, *not* a
    rigorous mathematical error bound or independent electronic stability test.
    """
    symbols = list(symbols)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    if coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Finite atom-order-preserving coordinates are required")
    ncart = 3 * len(symbols)
    matrices = _matrix_sequence(hessians, (ncart, ncart), "hessians")
    derivatives = None if dipole_derivatives is None else _matrix_sequence(dipole_derivatives, (3, ncart), "dipole_derivatives")
    if derivatives is not None and len(derivatives) != len(matrices):
        raise ValueError("Each Hessian must have a matching complete dipole derivative")
    if steps_bohr is not None:
        steps = [float(item) for item in steps_bohr]
        if len(steps) != len(matrices) or any(not math.isfinite(item) or item <= 0 for item in steps):
            raise ValueError("Positive finite step labels must match the Hessian count")
    else:
        steps = [0.001, 0.0005] if len(matrices) == 2 else [None] * len(matrices)
    metadata = _jsonable({"center_quality": center_quality, "point_quality": point_quality})
    fingerprint_payload = {
        "schema": SCHEMA, "source_sha256": sha256(__file__), "symbols": symbols,
        "coordinates_hash": _array_hash(coordinates),
        "hessian_hashes": [_array_hash(item) for item in matrices],
        "dipole_derivative_hashes": None if derivatives is None else [_array_hash(item) for item in derivatives],
        "steps_bohr": steps, "quality_metadata": metadata,
    }
    digest = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    previous = out / "result.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("input_fingerprint") != digest:
            raise ValueError("Analysis inputs changed; use a new output directory instead of replacing saved science")
        verify_analysis(out)
        return saved
    analyses = [normal_modes(symbols, coordinates, item, None if derivatives is None else derivatives[i]) for i, item in enumerate(matrices)]
    first, final = analyses[0], analyses[-1]
    center = electronic_quality(center_quality)
    nuclear_max = _finite_number(_lookup(center_quality or {}, "nuclear_gradient_max", "nuclear_gradient_max_hartree_per_bohr"))
    nuclear_rms = _finite_number(_lookup(center_quality or {}, "nuclear_gradient_rms", "nuclear_gradient_rms_hartree_per_bohr"))
    nuclear_ok = (
        nuclear_max is not None and nuclear_rms is not None
        and nuclear_max >= 0 and nuclear_rms >= 0
        and nuclear_max <= QUALIFICATION_THRESHOLDS["nuclear_gradient_max_hartree_per_bohr"]
        and nuclear_rms <= QUALIFICATION_THRESHOLDS["nuclear_gradient_rms_hartree_per_bohr"]
    )
    stability = _lookup(center_quality or {}, "orbital_stability_passed") is True
    center.update(nuclear_gradient_max=nuclear_max, nuclear_gradient_rms=nuclear_rms,
                  nuclear_stationarity_passed=bool(nuclear_ok), orbital_stability_passed=stability)
    points = _point_quality_summary(point_quality)
    expected_point_count = 2 * ncart * len(matrices)
    points["expected_central_difference_point_count"] = expected_point_count
    points["complete_point_quality_ledger"] = bool(
        points.get("point_count") == expected_point_count
    )
    step_comparison = None
    if len(analyses) >= 2:
        delta = final["projected_hessian"] - first["projected_hessian"]
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(final["projected_hessian"]), np.finfo(float).tiny))
        spectral = float(np.linalg.norm(delta, ord=2)) if delta.size else 0.0
        low_eigen = float(final["eigenvalues"][0]) if len(final["eigenvalues"]) else None
        overlap = np.abs(first["mass_weighted_modes"].T @ final["mass_weighted_modes"])
        rows, columns = linear_sum_assignment(-overlap)
        matched_frequency_difference = np.abs(first["signed_frequencies_cm1"][rows] - final["signed_frequencies_cm1"][columns])
        step_comparison = {
            "first_matrix_index": 0, "final_matrix_index": len(analyses) - 1,
            "projected_relative_step_difference": relative,
            "projected_spectral_step_difference_hartree_per_amu_bohr2": spectral,
            "lowest_final_eigenvalue_exceeds_empirical_spectral_difference": bool(low_eigen is not None and low_eigen > spectral),
            "sorted_frequency_max_absolute_difference_cm1": float(np.max(np.abs(first["signed_frequencies_cm1"] - final["signed_frequencies_cm1"]))) if len(rows) else None,
            "maximum_overlap_matched_frequency_max_absolute_difference_cm1": float(np.max(matched_frequency_difference)) if len(rows) else None,
            "maximum_overlap_assignment_final_indices_zero_based": columns.tolist(),
            "matched_mode_absolute_overlap_minimum": float(np.min(overlap[rows, columns])) if len(rows) else None,
            "interpretation": "Empirical finite-step sensitivity, not a rigorous error bound; individual vectors/IR intensities in degenerate subspaces are basis-dependent.",
        }
    dual_step_ok = bool(step_comparison and step_comparison["projected_relative_step_difference"] <= QUALIFICATION_THRESHOLDS["projected_relative_step_difference"])
    empirical_positive_margin = bool(step_comparison and step_comparison["lowest_final_eigenvalue_exceeds_empirical_spectral_difference"])
    asymmetry_ok = all(item["diagnostics"]["raw_relative_antisymmetry"] <= QUALIFICATION_THRESHOLDS["raw_relative_antisymmetry"] for item in analyses)
    all_positive = all(item["diagnostics"]["imaginary_count"] == 0 and item["diagnostics"]["exact_zero_count"] == 0 and len(item["eigenvalues"]) > 0 for item in analyses)
    stationary = bool(center["qualified"] and nuclear_ok)
    nuclear_minimum = bool(
        stationary and points.get("all_points_qualified") is True
        and points.get("all_native_converged") is True
        and points["complete_point_quality_ledger"]
        and asymmetry_ok and dual_step_ok and empirical_positive_margin and all_positive
    )
    minimum_verified = bool(nuclear_minimum and stability)
    qualification = {
        "center_electronic_quality": center, "point_electronic_quality": points,
        "dual_step_comparison_available": step_comparison is not None,
        "raw_antisymmetry_passed": asymmetry_ok, "projected_step_agreement_passed": dual_step_ok,
        "empirical_positive_spectral_margin_passed": empirical_positive_margin,
        "all_vibrational_eigenvalues_strictly_positive_at_all_steps": all_positive,
        "stationary_geometry_verified": stationary,
        "nuclear_minimum_numerically_supported": nuclear_minimum,
        "orbital_stability_passed": stability,
        "minimum_verified": minimum_verified,
        "paper_ready_molecular_harmonic_data": bool(minimum_verified and derivatives is not None),
        "thresholds_are_qualification_not_execution_stops": True,
        "scope": "Local isolated-molecule harmonic calculation at the stated method; no global-conformer minimum or solid-state chemical peak assignment is established.",
    }
    arrays = {"coordinates_bohr": coordinates, "masses_amu": final["masses_amu"]}
    for index, item in enumerate(analyses):
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                arrays[f"step_{index}_{key}"] = value
        if derivatives is not None:
            arrays[f"step_{index}_dipole_derivative_au"] = derivatives[index]
    temporary = out / "analysis_arrays.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out / "analysis_arrays.npz")
    fieldnames = ["matrix_index", "step_bohr", "mode_index_one_based", "signed_eigenvalue_hartree_per_amu_bohr2", "signed_frequency_cm1", "signed_frequency_THz", "imaginary", "reduced_mass_amu", "IR_intensity_km_mol", "positive_in_0p1_to_1p5_THz"]
    csv_tmp = out / "all_modes.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(analyses):
            for mode in range(len(item["eigenvalues"])):
                thz = float(item["signed_frequencies_THz"][mode])
                writer.writerow({
                    "matrix_index": index, "step_bohr": steps[index], "mode_index_one_based": mode + 1,
                    "signed_eigenvalue_hartree_per_amu_bohr2": float(item["eigenvalues"][mode]),
                    "signed_frequency_cm1": float(item["signed_frequencies_cm1"][mode]),
                    "signed_frequency_THz": thz, "imaginary": bool(item["eigenvalues"][mode] < 0),
                    "reduced_mass_amu": float(item["reduced_masses_amu"][mode]),
                    "IR_intensity_km_mol": None if item["IR_intensities_km_mol"] is None else float(item["IR_intensities_km_mol"][mode]),
                    "positive_in_0p1_to_1p5_THz": bool(0.1 <= thz <= 1.5 and item["eigenvalues"][mode] > 0),
                })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_tmp, out / "all_modes.csv")
    summaries = []
    for index, item in enumerate(analyses):
        summaries.append({"matrix_index": index, "step_bohr": steps[index], **item["diagnostics"],
                          "signed_frequencies_cm1": item["signed_frequencies_cm1"].tolist(),
                          "signed_frequencies_THz": item["signed_frequencies_THz"].tolist(),
                          "IR_intensities_km_mol": None if item["IR_intensities_km_mol"] is None else item["IR_intensities_km_mol"].tolist()})
    result = {
        "schema": SCHEMA, "status": "ANALYSIS_COMPUTED", "at_utc": utc_now(),
        "input_fingerprint": digest, "input_provenance": fingerprint_payload,
        "symbols": symbols, "atom_count": len(symbols), "matrix_count": len(matrices),
        "mass_source": "pyscf.data.elements.MASSES standard isotope-average masses; explicit matching fallback for common elements",
        "constants": physical_conversion_constants(), "qualification_thresholds": QUALIFICATION_THRESHOLDS,
        "steps": summaries, "step_comparison": step_comparison,
        "qualification": qualification, "minimum_verified": minimum_verified,
        "IR_status": "TOTAL_DIPOLE_FINITE_DIFFERENCES_COMPUTED" if derivatives is not None else "NOT_COMPUTED",
        "chemical_peak_assignment_verified": False,
        "notes": [
            "All signed vibrational modes are retained; negative eigenvalues are imaginary modes, never replaced by absolute frequencies.",
            "Dipole-derivative projections on imaginary directions are diagnostic values, not absorption transitions of a stable minimum.",
            "Raw antisymmetry is measured before averaging H with H.T for a real symmetric vibrational problem.",
            "Cartesian display vectors have Euclidean unit norm; mass-normalized Cartesian vectors, not display vectors, enter IR contraction.",
            "Finite-step disagreement is empirical evidence of numerical sensitivity, not a rigorous bound on all numerical or model errors.",
            "Positive projected modes do not replace an independently checked fresh nuclear gradient or electronic orbital-stability test.",
            "Gas-phase molecular IR intensities and frequencies do not establish drug-solid THz peak identities.",
        ],
        "artifacts": {name: {"path": name, "sha256": sha256(out / name)} for name in ("analysis_arrays.npz", "all_modes.csv")},
    }
    _atomic_json(out / "result.json", result)
    return result


def verify_analysis(output: str | Path, *, write_receipt: bool = True) -> dict[str, Any]:
    """Replay saved numerical matrices and independently check all CSV frequencies.

    This verifies the derivative-to-mode and IR mathematics; it is explicitly not
    an independent electronic-structure or displacement-gradient recalculation.
    """
    out = Path(output)
    result = json.loads((out / "result.json").read_text())
    if result.get("schema") != SCHEMA:
        raise ValueError("Unrecognized mode-analysis schema")
    for artifact in result.get("artifacts", {}).values():
        path = out / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError("Saved analysis artifact hash mismatch: " + str(path))
    max_frequency_error = max_ir_error = 0.0
    with np.load(out / "analysis_arrays.npz", allow_pickle=False) as arrays:
        coordinates = arrays["coordinates_bohr"]
        if _array_hash(coordinates) != result["input_provenance"]["coordinates_hash"]:
            raise ValueError("Saved coordinates do not match the analysis inputs")
        expected = []
        for index in range(result["matrix_count"]):
            raw = arrays[f"step_{index}_raw_hessian"]
            if _array_hash(raw) != result["input_provenance"]["hessian_hashes"][index]:
                raise ValueError("Saved Hessian does not match the declared analysis inputs")
            dipole = arrays.get(f"step_{index}_dipole_derivative_au")
            if dipole is not None and _array_hash(dipole) != result["input_provenance"]["dipole_derivative_hashes"][index]:
                raise ValueError("Saved dipole derivative does not match the declared analysis inputs")
            replay = normal_modes(result["symbols"], coordinates, raw, dipole)
            saved_frequencies = np.asarray(result["steps"][index]["signed_frequencies_cm1"])
            if replay["signed_frequencies_cm1"].shape != saved_frequencies.shape:
                raise ValueError("Vibrational mode count changed during replay")
            difference = float(np.max(np.abs(replay["signed_frequencies_cm1"] - saved_frequencies))) if len(saved_frequencies) else 0.0
            max_frequency_error = max(max_frequency_error, difference)
            if not np.allclose(replay["signed_frequencies_cm1"], saved_frequencies, rtol=1e-11, atol=1e-8):
                raise ValueError("Signed frequencies differ during independent mode replay")
            if dipole is not None:
                saved_ir = np.asarray(result["steps"][index]["IR_intensities_km_mol"])
                error = float(np.max(np.abs(replay["IR_intensities_km_mol"] - saved_ir))) if len(saved_ir) else 0.0
                max_ir_error = max(max_ir_error, error)
                if not np.allclose(replay["IR_intensities_km_mol"], saved_ir, rtol=1e-10, atol=1e-8):
                    raise ValueError("IR intensity differs during numerical replay")
            expected.append(replay)
    with (out / "all_modes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sum(len(item["eigenvalues"]) for item in expected):
        raise ValueError("Saved mode CSV row count is incomplete or duplicated")
    seen = set()
    for row in rows:
        index = int(row["matrix_index"])
        mode = int(row["mode_index_one_based"]) - 1
        key = (index, mode)
        if key in seen or index < 0 or index >= len(expected) or mode < 0 or mode >= len(expected[index]["eigenvalues"]):
            raise ValueError("Mode CSV identities are duplicated or out of range")
        seen.add(key)
        replay = expected[index]
        for field, values in (("signed_frequency_cm1", replay["signed_frequencies_cm1"]), ("signed_frequency_THz", replay["signed_frequencies_THz"]), ("signed_eigenvalue_hartree_per_amu_bohr2", replay["eigenvalues"]), ("reduced_mass_amu", replay["reduced_masses_amu"])):
            if not math.isclose(float(row[field]), float(values[mode]), rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError("CSV mode value differs from saved numerical derivatives: " + field)
        if (row["imaginary"].lower() == "true") != bool(replay["eigenvalues"][mode] < 0):
            raise ValueError("CSV imaginary-mode flag contradicts the signed eigenvalue")
        if replay["IR_intensities_km_mol"] is not None and not math.isclose(float(row["IR_intensity_km_mol"]), float(replay["IR_intensities_km_mol"][mode]), rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("CSV IR intensity differs from saved dipole derivatives")
    receipt = {"schema": SCHEMA, "status": "VERIFIED", "at_utc": utc_now(),
               "matrix_count": result["matrix_count"], "mode_rows_checked": len(rows),
               "maximum_frequency_replay_error_cm1": max_frequency_error,
               "maximum_IR_intensity_replay_error_km_mol": max_ir_error,
               "result_sha256": sha256(out / "result.json"),
               "scope": "Saved derivatives to signed modes/IR replay, not new SCF or numerical derivative evaluation"}
    if write_receipt:
        _atomic_json(out / "independent_mode_verification.json", receipt)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, required=True, help="Replay an existing analysis directory")
    args = parser.parse_args()
    print(json.dumps(verify_analysis(args.verify), ensure_ascii=False))
