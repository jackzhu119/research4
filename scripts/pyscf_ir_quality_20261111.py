"""Independent, JSON-serializable numerical qualification for harmonic IR.

Inputs use a fixed dipole origin, Cartesian coordinates in Bohr, total dipoles
in e*Bohr, Cartesian dipole derivatives in e, and masses in unified atomic mass
units. Cartesian normal modes are rows and satisfy L M L.T = identity. Quality
failures are evidence for refinement, never instructions to stop a calculation.
No Richardson estimate below is a rigorous error bound.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# CODATA 2022 values; exact SI definitions where applicable. These constants do
# not depend on a pre-existing IR result or the production conversion routine.
AVOGADRO = 6.02214076e23
ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906892e-27
VACUUM_PERMITTIVITY = 8.8541878188e-12
LIGHT_SPEED = 299792458.0
FINE_STRUCTURE = 7.2973525643e-3
ELECTRON_MASS_KG = 9.1093837139e-31
BOHR_METERS = 5.29177210544e-11


def prefactor_si() -> float:
    """km/mol per (e/sqrt(amu))**2, from SI absorption constants."""
    return (AVOGADRO * ELEMENTARY_CHARGE**2 /
            (12 * VACUUM_PERMITTIVITY * LIGHT_SPEED**2 * ATOMIC_MASS_KG * 1000))


def prefactor_atomic() -> float:
    """Independent atomic-constant form used by official PySCF properties."""
    return (FINE_STRUCTURE**2 * (1e-3 / ATOMIC_MASS_KG) * ELECTRON_MASS_KG *
            AVOGADRO * math.pi * BOHR_METERS / 3)


def _failed(message: str) -> dict[str, Any]:
    return {"qualified": False, "valid": False, "failures": [message]}


def _finite(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be nonempty and finite")
    return a


def _modes(value: Any, masses: Any) -> tuple[np.ndarray, np.ndarray]:
    mass = _finite(masses, "masses").reshape(-1)
    if np.any(mass <= 0):
        raise ValueError("masses must be positive")
    modes = _finite(value, "modes")
    if modes.ndim == 3 and modes.shape[1:] == (mass.size, 3):
        modes = modes.reshape(modes.shape[0], -1)
    if modes.ndim != 2 or modes.shape[1] != 3 * mass.size:
        raise ValueError("modes must have shape (mode, atom, 3) or (mode, 3*atom)")
    return modes, mass


def _series(values: Any, steps: Any, tail: tuple[int, ...], name: str):
    hs = _finite(steps, "steps").reshape(-1)
    if hs.size < 2 or np.any(hs <= 0) or np.unique(hs).size != hs.size:
        raise ValueError("at least two distinct positive steps are required")
    a = _finite(values, name)
    if a.shape != (hs.size,) + tail:
        raise ValueError(f"{name} must have shape {(hs.size,) + tail}, got {a.shape}")
    order = np.argsort(hs)[::-1]
    return a[order], hs[order]


def validate_mode_normalization(modes: Any, masses: Any, tolerance: float = 1e-7):
    """Reject Euclidean-renormalized, duplicate, missing, or invalid modes."""
    try:
        ls, mass = _modes(modes, masses)
        gram = (ls * np.repeat(mass, 3)) @ ls.T
        error = float(np.max(np.abs(gram - np.eye(ls.shape[0]))))
        ok = error <= tolerance
        return {"qualified": ok, "valid": True, "maximum_mass_gram_error": error,
                "number_of_modes": int(ls.shape[0]), "tolerance": tolerance,
                "failures": [] if ok else ["modes_are_not_mass_orthonormal"]}
    except (TypeError, ValueError) as exc:
        return _failed(str(exc))


def _vector_check(a: np.ndarray, b: np.ndarray, absolute: float, relative: float):
    errors = np.linalg.norm(a - b, axis=-1)
    scales = np.maximum(np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1))
    tolerances = absolute + relative * scales
    return errors, scales, tolerances, errors <= tolerances


def _intensity(projected: np.ndarray) -> np.ndarray:
    return prefactor_si() * np.sum(projected**2, axis=-1)


def _richardson(values: np.ndarray, hs: np.ndarray) -> dict[str, Any]:
    """Central-difference O(h^2) estimate, with explicit noise diagnostics."""
    ratio2 = (hs[-2] / hs[-1])**2
    fine, previous = values[-1], values[-2]
    extrapolated = fine + (fine - previous) / (ratio2 - 1)
    result = {
        "assumed_leading_power": 2,
        "is_rigorous_error_bound": False,
        "extrapolated_projected_derivatives": extrapolated.tolist(),
        "estimated_finest_correction_norm": np.linalg.norm(extrapolated - fine, axis=-1).tolist(),
        "three_step_diagnostic_available": bool(hs.size >= 3),
        "interpretation": "asymptotic estimate only; SCF/grid noise or higher orders can invalidate it",
    }
    if hs.size >= 3:
        earlier = values[-3] - values[-2]
        later = values[-2] - values[-1]
        en = np.linalg.norm(earlier, axis=-1)
        ln = np.linalg.norm(later, axis=-1)
        expected = (hs[-3]**2 - hs[-2]**2) / (hs[-2]**2 - hs[-1]**2)
        resolvable = (en > 1e-12) & (ln > 1e-12)
        cosine = np.divide(np.sum(earlier * later, axis=-1), en * ln,
                           out=np.zeros_like(en), where=resolvable)
        observed = np.divide(en, ln, out=np.zeros_like(en), where=resolvable)
        # This detects failed asymptotic behavior; it does not assign a rigorous
        # stochastic noise model from only three deterministic samples.
        suspicious = resolvable & ((cosine < 0.5) | (observed < expected / 2) |
                                    (observed > expected * 2))
        result.update({
            "expected_difference_norm_ratio": float(expected),
            "observed_difference_norm_ratios": [float(x) if r else None for x, r in zip(observed, resolvable)],
            "difference_vector_cosines": [float(x) if r else None for x, r in zip(cosine, resolvable)],
            "non_asymptotic_or_noise_suspected": suspicious.tolist(),
            "differences_below_diagnostic_resolution": (~resolvable).tolist(),
        })
    return result


def common_frame_derivative_comparison(
    derivative_series: Any, steps: Any, modes: Any, masses: Any, *,
    vector_absolute_tolerance: float = 1e-4,
    vector_relative_tolerance: float = 0.01,
    intensity_absolute_tolerance: float = 0.1,
    intensity_relative_tolerance: float = 0.02,
    weak_peak_km_mol: float = 1.0,
) -> dict[str, Any]:
    """Compare h-series Cartesian derivatives in one fixed mode frame.

    derivative_series is (step, atom, cartesian, dipole) or (step, 3*atom,
    dipole). modes must come from ONE Hessian, not one changing set per h.
    All tolerance decisions concern scientific qualification only. The three
    step diagnostic also rejects visibly non-asymptotic vectors at otherwise
    small pairwise errors; exact/roundoff-identical derivatives remain valid.
    """
    try:
        ls, mass = _modes(modes, masses)
        a = _finite(derivative_series, "derivative_series")
        if a.ndim == 4 and a.shape[1:] == (mass.size, 3, 3):
            a = a.reshape(a.shape[0], 3 * mass.size, 3)
        a, hs = _series(a, steps, (3 * mass.size, 3), "derivative_series")
        norm = validate_mode_normalization(ls, mass)
        projected = np.einsum("mk,skd->smd", ls, a)
        errors, scales, limits, vector_ok = _vector_check(
            projected[-1], projected[-2], vector_absolute_tolerance,
            vector_relative_tolerance)
        intensities = _intensity(projected)
        di = np.abs(intensities[-1] - intensities[-2])
        si = np.maximum(intensities[-1], intensities[-2])
        ilimits = intensity_absolute_tolerance + intensity_relative_tolerance * si
        intensity_ok = di <= ilimits
        rich = _richardson(projected, hs)
        suspicious = np.asarray(rich.get("non_asymptotic_or_noise_suspected", [False] * len(ls)))
        significant_suspicious = suspicious & (errors > vector_absolute_tolerance)
        failures = list(norm["failures"])
        if not np.all(vector_ok):
            failures.append("projected_derivative_vectors_not_converged")
        if not np.all(intensity_ok):
            failures.append("common_frame_ir_intensities_not_converged")
        if np.any(significant_suspicious):
            failures.append("three_step_asymptotic_behavior_not_established")
        return {
            "qualified": not failures, "valid": True, "failures": failures,
            "steps_bohr_descending": hs.tolist(), "mass_normalization": norm,
            "prefactor_km_mol": prefactor_si(),
            "prefactor_atomic_formula_relative_difference": abs(prefactor_si() / prefactor_atomic() - 1),
            "projected_derivative_series": projected.tolist(),
            "ir_intensity_series_km_mol": intensities.tolist(),
            "mode_derivative_absolute_differences": errors.tolist(),
            "mode_derivative_tolerances": limits.tolist(),
            "mode_derivative_relative_differences": np.divide(errors, scales, out=np.zeros_like(errors), where=scales > 0).tolist(),
            "mode_intensity_absolute_differences_km_mol": di.tolist(),
            "mode_intensity_tolerances_km_mol": ilimits.tolist(),
            "weak_peaks": (si < weak_peak_km_mol).tolist(),
            "cartesian_derivative_max_absolute_difference": float(np.max(np.abs(a[-1] - a[-2]))),
            "cartesian_derivative_frobenius_difference": float(np.linalg.norm(a[-1] - a[-2])),
            "richardson": rich,
        }
    except (TypeError, ValueError) as exc:
        return _failed(str(exc))


def directional_comparison(
    projected: Any, direct_series: Any, steps: Any, *,
    vector_absolute_tolerance: float = 2e-4,
    vector_relative_tolerance: float = 0.02,
    intensity_absolute_tolerance: float = 0.2,
    intensity_relative_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Cross-check D.T L against directly displaced normal-coordinate dipoles.

    direct_series has shape (step, mode, 3), each entry being
    [mu(R+delta*L)-mu(R-delta*L)]/(2*delta); steps are delta in
    Bohr*sqrt(amu). projected is the (mode, 3) Cartesian contraction.
    The mode rows and signs MUST be the same in both paths.
    """
    try:
        p = _finite(projected, "projected")
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("projected must have shape (mode, 3)")
        direct, hs = _series(direct_series, steps, p.shape, "direct_series")
        er, sc, lim, ok = _vector_check(p, direct[-1], vector_absolute_tolerance,
                                       vector_relative_tolerance)
        ce, cs, cl, co = _vector_check(direct[-1], direct[-2],
                                      vector_absolute_tolerance, vector_relative_tolerance)
        pi, di = _intensity(p), _intensity(direct[-1])
        ie = np.abs(pi - di)
        il = intensity_absolute_tolerance + intensity_relative_tolerance * np.maximum(pi, di)
        failures = []
        if not np.all(ok):
            failures.append("direct_directional_and_cartesian_vectors_disagree")
        if not np.all(co):
            failures.append("direct_directional_derivatives_not_converged")
        if not np.all(ie <= il):
            failures.append("direct_directional_and_cartesian_intensities_disagree")
        return {
            "qualified": not failures, "valid": True, "failures": failures,
            "steps_normal_coordinate_descending": hs.tolist(),
            "vector_absolute_differences": er.tolist(), "vector_tolerances": lim.tolist(),
            "direct_step_vector_differences": ce.tolist(),
            "direct_step_vector_tolerances": cl.tolist(),
            "cartesian_intensities_km_mol": pi.tolist(),
            "direct_intensities_km_mol": di.tolist(),
            "intensity_absolute_differences_km_mol": ie.tolist(),
            "richardson": _richardson(direct, hs),
        }
    except (TypeError, ValueError) as exc:
        return _failed(str(exc))


def degenerate_subspace_comparison(
    modes_a: Any, modes_b: Any, derivatives_a: Any, derivatives_b: Any,
    masses: Any, *, minimum_subspace_cosine: float = 0.999,
    vector_absolute_tolerance: float = 2e-4, vector_relative_tolerance: float = 0.02,
    intensity_absolute_tolerance: float = 0.2, intensity_relative_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Compare one matched mode group without depending on signs/basis choices.

    derivatives_a/b are the corresponding projected modal dipole vectors.
    Call separately for each physically matched near-degenerate group.
    """
    try:
        la, mass = _modes(modes_a, masses)
        lb, _ = _modes(modes_b, mass)
        if la.shape != lb.shape:
            raise ValueError("matched subspaces must have equal dimensions")
        da = _finite(derivatives_a, "derivatives_a")
        db = _finite(derivatives_b, "derivatives_b")
        if da.shape != (la.shape[0], 3) or db.shape != da.shape:
            raise ValueError("projected derivatives must have shape (mode, 3)")
        na, nb = validate_mode_normalization(la, mass), validate_mode_normalization(lb, mass)
        overlap = (lb * np.repeat(mass, 3)) @ la.T
        u, sigma, vt = np.linalg.svd(overlap)
        rotation = u @ vt
        aligned = rotation @ da
        difference = float(np.linalg.norm(db - aligned))
        limit = vector_absolute_tolerance + vector_relative_tolerance * max(float(np.linalg.norm(db)), float(np.linalg.norm(da)))
        ia, ib = float(np.sum(_intensity(da))), float(np.sum(_intensity(db)))
        ilimit = intensity_absolute_tolerance + intensity_relative_tolerance * max(ia, ib)
        failures = na["failures"] + nb["failures"]
        if float(np.min(sigma)) < minimum_subspace_cosine:
            failures.append("mode_subspaces_do_not_match")
        if difference > limit:
            failures.append("aligned_subspace_derivatives_disagree")
        if abs(ia - ib) > ilimit:
            failures.append("subspace_total_intensities_disagree")
        return {"qualified": not failures, "valid": True, "failures": failures,
                "principal_angle_cosines": sigma.tolist(),
                "aligned_derivative_frobenius_difference": difference,
                "aligned_derivative_tolerance": limit,
                "summed_intensities_km_mol": [ia, ib],
                "summed_intensity_tolerance_km_mol": ilimit}
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed(str(exc))


def rigid_body_residuals(
    derivatives: Any, coords_bohr: Any, dipole_au: Any, charge: float,
    *, origin_bohr: Any = (0.0, 0.0, 0.0), absolute_tolerance: float = 5e-4,
) -> dict[str, Any]:
    """Check fixed-origin total-dipole translation and rotation identities.

    The charge can be nonzero. Electronic-only derivatives fail this check
    unless the omitted nuclear derivative happens to vanish.
    """
    try:
        coords = _finite(coords_bohr, "coords_bohr")
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords_bohr must have shape (atom, 3)")
        d = _finite(derivatives, "derivatives")
        if d.shape == (coords.size, 3):
            d = d.reshape(coords.shape[0], 3, 3)
        if d.shape != (coords.shape[0], 3, 3):
            raise ValueError("derivatives must have shape (atom, cartesian, dipole)")
        mu = _finite(dipole_au, "dipole_au")
        origin = _finite(origin_bohr, "origin_bohr")
        if mu.shape != (3,) or origin.shape != (3,) or not np.isfinite(charge):
            raise ValueError("dipole, origin and charge have invalid shape/value")
        trans = d.sum(axis=0) - float(charge) * np.eye(3)
        rotations = []
        for axis in np.eye(3):
            velocity = np.cross(axis, coords - origin)
            predicted = np.einsum("aid,ai->d", d, velocity)
            expected = np.cross(axis, mu - float(charge) * origin)
            rotations.append(predicted - expected)
        rotations = np.asarray(rotations)
        tm, rm = float(np.max(np.abs(trans))), float(np.max(np.abs(rotations)))
        failures = []
        if tm > absolute_tolerance:
            failures.append("fixed_origin_translation_identity_failed")
        if rm > absolute_tolerance:
            failures.append("fixed_origin_rotation_identity_failed")
        return {"qualified": not failures, "valid": True, "failures": failures,
                "charge": float(charge), "origin_bohr": origin.tolist(),
                "translation_residual_unit": "e",
                "rotation_residual_unit": "e*Bohr",
                "translation_max_absolute_residual": tm,
                "rotation_max_absolute_residual": rm,
                "translation_residual_tensor": trans.tolist(),
                "rotation_residual_tensor": rotations.tolist(),
                "absolute_tolerance": absolute_tolerance}
    except (TypeError, ValueError) as exc:
        return _failed(str(exc))
