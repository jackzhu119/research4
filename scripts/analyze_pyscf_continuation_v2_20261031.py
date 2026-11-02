"""Failure-aware V2 final audit; never fabricate result.json or evaluate DFT.

Extends the frozen first analyzer by deriving explicit PARTIAL evidence for an
ended native worker that never wrote its final result. Original files remain
unchanged, and unfinalized snapshots never become accepted frequencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("legacy_pyscf_analysis_helpers", ROOT / "scripts/analyze_pyscf_spinosin_pilot_20261031.py")
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)
sha, read, write, csvwrite = LEGACY.sha, LEGACY.read, LEGACY.write, LEGACY.csvwrite
CM1_TO_THZ = .0299792458
BOHR_TO_ANGSTROM = .52917721092  # PySCF2.14.0 lib.param.BOHR; independently checked at replay.
PEAK_PATH = ROOT / "reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv"
PEAK_SHA256 = "b3a5f59157f42859f68edf0a8760d9f811f83ea7947c978cc0a9cbc98a4ba8b4"
SOURCE_BASE = "aef64c65295bee03764da1dd72c21d32ed3e6c24"
THRESHOLDS = {"convergence_energy": 1e-6, "convergence_grms": 3e-4,
              "convergence_gmax": 4.5e-4, "convergence_drms": .0012, "convergence_dmax": .0018}
METHOD = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1,
          "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "density_fitting": True,
          "auxbasis": "def2-universal-jfit", "hessian_auxbasis_response": 2,
          "density_fit_backend": "INCORE_DFJ_RELEASE_BEFORE_GRAD_V3",
          "charge": 0, "spin_2S": 0, "multiplicity": 1}
COMPOSITIONS = {"water": {"H": 2, "O": 1}, "spinosin": {"C": 28, "H": 32, "O": 15}}


def method_matches(settings):
    common = {key: value for key, value in METHOD.items()
              if key not in ("density_fitting", "auxbasis", "hessian_auxbasis_response", "density_fit_backend")}
    backend = settings.get("density_fit_backend")
    supported = ((backend == "INCORE_DFJ_RELEASE_BEFORE_GRAD_V3"
                  and settings.get("density_fitting") is True
                  and settings.get("auxbasis") == "def2-universal-jfit"
                  and settings.get("hessian_auxbasis_response") == 2)
                 or (backend == "SCREENED_DIRECT_RKS_V4" and settings.get("density_fitting") is False
                     and settings.get("auxbasis") is None and settings.get("hessian_auxbasis_response") is None
                     and settings.get("direct_scf") is True and settings.get("direct_scf_tol") == 1e-13
                     and settings.get("global_four_center_ERI_cache_allowed") is False))
    return (all(settings.get(key) == value for key, value in common.items()) and supported
            and settings.get("optimizer_thresholds") == THRESHOLDS
            and settings.get("scf_conv_tol_hartree") == 1e-8
            and settings.get("threads") in (1, 2, 3, 4))


def valid_modes(result, identity):
    """An actual computed array may still fail minimum/frequency acceptance."""
    if identity not in COMPOSITIONS or not method_matches(result.get("settings", {})):
        return False
    count, natoms = (3, 3) if identity == "water" else (219, 75)
    required_true = ("geometry_optimized", "optimizer_converged", "scf_converged",
                     "final_gradient_thresholds_passed", "frequency_validation_passed",
                     "stationary_minimum_verified", "all_frequencies_real_positive",
                     "normal_modes_complete", "harmonic_analysis_excludes_translations_and_rotations",
                     "normal_mode_hessian_symmetrized")
    if (result.get("identity") != identity or result.get("atoms") != natoms
            or result.get("status") != "DONE" or result.get("frequencies_status") != "DONE"
            or not all(result.get(key) is True for key in required_true)
            or result.get("normal_mode_count") != count or result.get("expected_normal_mode_count") != count
            or result.get("imaginary_frequency_count") != 0
            or result.get("IR_intensities_status") != "NOT_COMPUTED"):
        return False
    try:
        cm = np.asarray(result.get("frequencies_cm1", []), dtype=float)
        thz = np.asarray(result.get("frequencies_thz", []), dtype=float)
        grms, gmax = float(result["atom_rms_hartree_per_bohr"]), float(result["atom_max_hartree_per_bohr"])
    except (TypeError, ValueError, KeyError):
        return False
    return bool(cm.shape == thz.shape == (count,) and np.isfinite(cm).all()
                and np.isfinite(thz).all() and np.all(cm > 0)
                and np.allclose(thz, cm * CM1_TO_THZ, atol=1e-12, rtol=1e-12)
                and 0 <= grms <= THRESHOLDS["convergence_grms"]
                and 0 <= gmax <= THRESHOLDS["convergence_gmax"])


def authorize(root, release):
    ledger_path = root / "cpu_budget.json"
    ledger = read(ledger_path)
    if (release.get("allow_analysis_export") is not True or not release.get("created_at_utc")
            or release.get("cpu_budget_sha256") != sha(ledger_path)):
        raise ValueError("Dated root release must bind final continuation ledger")
    if ledger.get("schema") != "research4-pyscf-continuation-budget-v2":
        raise ValueError("Not the separately authorized continuation budget schema")
    if any(row.get("status") in ("RUNNING", "BUDGET_VIOLATION_REQUIRES_REVIEW") for row in ledger["phases"]):
        raise ValueError("Unfinished or unreviewed budget-violation phase cannot enter final analysis")
    auth_path = Path(ledger["authorization_path"])
    auth_path = auth_path if auth_path.is_absolute() else ROOT / auth_path
    if sha(auth_path) != ledger["authorization_sha256"]:
        raise ValueError("Recorded authorization bytes changed")
    authorization = read(auth_path)
    checkpoint = Path(ledger["source_checkpoint_path"])
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    if (ledger.get("source_base_commit") != SOURCE_BASE
            or authorization.get("source_base_commit") != SOURCE_BASE
            or authorization.get("source_checkpoint_sha256") != ledger["source_checkpoint_sha256"]
            or sha(checkpoint) != ledger["source_checkpoint_sha256"]):
        raise ValueError("Immutable source base/checkpoint differs from continuation authorization")
    cap = float(ledger["quantum_cpu_cap_seconds"])
    used = np.asarray([row["actual_cpu_seconds"] for row in ledger["phases"]], dtype=float)
    if (not np.isfinite(used).all() or np.any(used < 0)
            or not np.isclose(used.sum(), ledger["consumed_cpu_seconds"], atol=1e-7, rtol=1e-12)
            or not 0 < cap <= min(43200, authorization["max_quantum_cpu_seconds"])
            or used.sum() > cap + 1e-7):
        raise ValueError("Independent phase CPU sum/cap disagrees with continuation authorization")
    for row in ledger["phases"]:
        if row.get("status") not in ("EXITED", "PARTIAL_OR_FAILED", "FAILED_TO_START"):
            raise ValueError("Unknown final phase status")
        if row.get("threads") not in (1, 2, 3, 4):
            raise ValueError("Thread count outside authorized bounds")
    return ledger, auth_path


def gradient_statistics(values):
    g = np.asarray(values, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3 or not np.isfinite(g).all():
        raise ValueError("Finite atom-by3 gradient required")
    lengths = np.linalg.norm(g, axis=1)
    return {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(g * g))),
            "component_max_hartree_per_bohr": float(np.max(np.abs(g))),
            "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(lengths * lengths))),
            "atom_max_hartree_per_bohr": float(np.max(lengths)),
            "gradient_norm_hartree_per_bohr": float(np.linalg.norm(g))}


def check_geometry(symbols, coordinates, identity):
    coordinates = np.asarray(coordinates, dtype=float)
    if (Counter(symbols) != Counter(COMPOSITIONS[identity])
            or coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all()):
        raise ValueError("Geometry composition, atom count or finite-coordinate mismatch")
    return coordinates


def audit_step(metadata_path, identity, phase, input_hashes):
    meta = read(metadata_path)
    prefix = metadata_path.with_suffix("")
    gradient_path = prefix.with_suffix(".gradient.npy")
    xyz_path = prefix.with_suffix(".xyz")
    coordinates_path = prefix.with_suffix(".coordinates_angstrom.npy")
    bohr_path = prefix.with_suffix(".coordinates_bohr.npy")
    paths = [metadata_path, gradient_path, xyz_path, coordinates_path, bohr_path]
    if not all(path.is_file() for path in paths):
        raise ValueError("A completed geometry JSON lacks its paired gradient/coordinate artifacts")
    if meta.get("scf_converged") is not True or not np.isfinite(float(meta["energy_hartree"])):
        raise ValueError("Completed gradient point has unconverged or nonfinite SCF evidence")
    log_path = metadata_path.parent.parent / "pyscf.log"
    converged_energies = [float(value) for value in re.findall(r"converged SCF energy\s*=\s*([-+0-9.eE]+)", log_path.read_text())]
    cycle = int(meta["cycle"])
    if (cycle < 1 or len(converged_energies) < cycle
            or not np.isclose(converged_energies[cycle - 1], float(meta["energy_hartree"]), atol=2e-8, rtol=0)):
        raise ValueError("Paired geometry energy differs from its ordered converged SCF log")
    input_hashes[str(log_path)] = sha(log_path)
    gradient = np.load(gradient_path, allow_pickle=False)
    symbols, rounded = LEGACY.xyz(xyz_path)
    precise = check_geometry(symbols, np.load(coordinates_path, allow_pickle=False), identity)
    bohr = np.load(bohr_path, allow_pickle=False)
    if (gradient.shape != precise.shape or bohr.shape != precise.shape
            or not np.allclose(rounded, precise, atol=1e-11, rtol=0)
            or not np.allclose(bohr * BOHR_TO_ANGSTROM, precise, atol=1e-10, rtol=0)):
        raise ValueError("Paired gradient/coordinates or Bohr/Angstrom units disagree")
    stats = gradient_statistics(gradient)
    for key, value in stats.items():
        if not np.isclose(meta[key], value, atol=1e-13, rtol=1e-11):
            raise ValueError("Saved gradient statistic disagrees with array: " + key)
    for key, path in [("gradient_npy_sha256", gradient_path),
                      ("coordinates_angstrom_npy_sha256", coordinates_path),
                      ("coordinates_bohr_npy_sha256", bohr_path)]:
        if meta.get(key) != sha(path):
            raise ValueError("Paired array SHA differs: " + key)
    for path in paths:
        input_hashes[str(path)] = sha(path)
    passed = stats["atom_rms_hartree_per_bohr"] <= .0003 and stats["atom_max_hartree_per_bohr"] <= .00045
    if meta.get("final_gradient_thresholds_passed") is not passed:
        raise ValueError("Stored gradient threshold flag disagrees with actual array")
    return {"phase": phase, "identity": identity, "cycle": int(meta["cycle"]),
            "energy_hartree": float(meta["energy_hartree"]), "scf_converged": True,
            **stats, "gradient_thresholds_passed": passed, "gradient_npy_sha256": sha(gradient_path)}


def phase_location(root, row):
    command = row["command"]
    if "--output" not in command:
        if row["status"] == "FAILED_TO_START":
            return None
        raise ValueError("Recorded command lacks explicit worker output")
    location = Path(command[command.index("--output") + 1])
    location = location if location.is_absolute() else ROOT / location
    if not location.resolve().is_relative_to(root.resolve()):
        raise ValueError("Worker output outside authorized continuation run")
    return location.resolve()


def completed_phase_result(row, location, input_hashes):
    """Read closed-phase evidence; an absent final result is always PARTIAL."""
    if row.get("status") == "RUNNING" or not row.get("ended_at_utc"):
        raise ValueError("Only actually ended phases may derive missing-result evidence")
    if location is not None and (location / "result.json").is_file():
        result = read(location / "result.json")
        return {**result, "worker_result_missing": False, "completion_evidence_kind": "FINAL_WORKER_RESULT_JSON"}
    snapshots = {}
    for filename in ("input.json", "progress.json"):
        if location is not None and (location / filename).is_file():
            snapshots[filename] = read(location / filename)
            input_hashes[str(location / filename)] = sha(location / filename)
    initial, progress = snapshots.get("input.json", {}), snapshots.get("progress.json", {})
    for key in ("identity", "script_sha256", "settings"):
        if key in initial and key in progress and initial[key] != progress[key]:
            raise ValueError("Unfinalized worker snapshots disagree about immutable " + key)
    result = {**initial, **progress}
    original_frequency_claim = result.get("frequencies_status", "NOT_RUN")
    artifacts = {}
    names = ("calculation.chk", "hessian_hartree_per_bohr2.npy", "hessian_used_for_analysis.npy",
             "hessian_cache_manifest.json", "normal_modes.npy", "frequencies.csv", "gradient_hartree_per_bohr.npy")
    for filename in names:
        path = location / filename if location is not None else None
        if path is not None and path.is_file():
            artifacts[filename] = {"sha256": sha(path), "bytes": path.stat().st_size}
            input_hashes[str(path)] = sha(path)
    # No alternate result is written into the worker directory. Snapshot fields
    # describe last observed work and do not certify native completion.
    result.update(status="PARTIAL", worker_result_missing=True,
                  completion_evidence_kind="PARENT_EXIT_AND_UNFINALIZED_INPUT_PROGRESS_ONLY",
                  parent_exit_code=row.get("exit_code"), parent_ended_at_utc=row["ended_at_utc"],
                  snapshot_event=progress.get("event"),
                  unfinalized_frequency_status_claim=original_frequency_claim,
                  frequencies_status=("NOT_CONFIRMED_MISSING_WORKER_RESULT" if original_frequency_claim == "DONE" else original_frequency_claim),
                  frequency_validation_passed=False, stationary_minimum_verified=False,
                  IR_intensities_status="NOT_COMPUTED", observed_closed_artifacts=artifacts,
                  stop_reason="Worker exited without result.json; actual parent exit_code=" + str(row.get("exit_code")))
    return result


def frequency_parent_guard(root, row, result, input_hashes):
    """A different numerical backend requires its own actual optimized parent."""
    if result.get("identity") == "water":
        return
    command = row["command"]
    if result.get("phase") != "spinosin-frequency" or "--optimization-result" not in command:
        raise ValueError("Spinosin frequencies require the explicit completed optimization parent")
    parent_path = Path(command[command.index("--optimization-result") + 1])
    parent_path = parent_path if parent_path.is_absolute() else ROOT / parent_path
    if not parent_path.resolve().is_relative_to(root):
        raise ValueError("Frequency optimization parent outside current continuation run")
    if sha(parent_path) != result.get("optimization_result_sha256"):
        raise ValueError("Frequency parent optimization result hash differs")
    parent = read(parent_path)
    if (parent.get("status") != "DONE" or parent.get("identity") != "spinosin"
            or not all(parent.get(key) is True for key in ("geometry_optimized", "optimizer_converged",
                                                           "scf_converged", "final_gradient_thresholds_passed"))
            or not method_matches(parent.get("settings", {}))):
        raise ValueError("Frequency parent optimization lacks actual convergence")
    keys = list(METHOD) + ["direct_scf", "direct_scf_tol", "global_four_center_ERI_cache_allowed"]
    if any(parent["settings"].get(key) != result["settings"].get(key) for key in keys):
        raise ValueError("Frequency backend/method differs from its optimized parent")
    if (parent.get("optimized_xyz_sha256") != result.get("input_xyz_sha256")
            or not parent.get("optimized_coordinates_angstrom_npy_sha256")
            or parent["optimized_coordinates_angstrom_npy_sha256"] != result.get("input_coordinates_angstrom_npy_sha256")):
        raise ValueError("Frequency input geometry differs from its completed optimization")
    for option, expected in [("--input-xyz", parent["optimized_xyz_sha256"]),
                              ("--input-coordinates-npy", parent["optimized_coordinates_angstrom_npy_sha256"])]:
        if option not in command:
            raise ValueError("Frequency input lacks complete-precision parent geometry option")
        path = Path(command[command.index(option) + 1]); path = path if path.is_absolute() else ROOT / path
        if sha(path) != expected:
            raise ValueError("Frequency parent geometry file hash differs")
        input_hashes[str(path)] = sha(path)
    input_hashes[str(parent_path)] = sha(parent_path)


def audit_final_gradient(location, result, input_hashes):
    path = location / "gradient_hartree_per_bohr.npy"
    if not path.exists():
        if result.get("geometry_optimized") is True:
            raise ValueError("Optimized claim lacks its fresh final gradient")
        return {}
    gradient = np.load(path, allow_pickle=False)
    if gradient.shape != (sum(COMPOSITIONS[result["identity"]].values()), 3):
        raise ValueError("Fresh final gradient atom count differs")
    stats = gradient_statistics(gradient)
    for key, value in stats.items():
        if not np.isclose(value, result[key], atol=1e-13, rtol=1e-11):
            raise ValueError("Final gradient statistic differs from actual saved array")
    passed = stats["atom_rms_hartree_per_bohr"] <= .0003 and stats["atom_max_hartree_per_bohr"] <= .00045
    if result.get("final_gradient_thresholds_passed") is not passed:
        raise ValueError("Final gradient threshold flag disagrees with array")
    if result.get("geometry_optimized") is True and not (passed and result.get("optimizer_converged") is True and result.get("scf_converged") is True):
        raise ValueError("Optimized claim lacks independent convergence conditions")
    input_hashes[str(path)] = sha(path)
    return stats


def replay_harmonic(symbols, bohr, hessian):
    """Re-diagonalize saved small matrices only. No SCF/derivative evaluation."""
    for site in (".runtime/quantum-upgrade-site", ".runtime/quantum-pilot-site"):
        sys.path.insert(0, str(ROOT / site))
    from pyscf import gto, lib
    from pyscf.hessian import thermo
    lib.num_threads(1)
    if abs(lib.param.BOHR - BOHR_TO_ANGSTROM) > 1e-14:
        raise ValueError("PySCF Bohr conversion differs from recorded convention")
    molecule = gto.M(atom=list(zip(symbols, bohr)), unit="Bohr", basis="def2-svp",
                     charge=0, spin=0, symmetry=False, verbose=0)
    result = thermo.harmonic_analysis(molecule, hessian, exclude_trans=True,
                                     exclude_rot=True, imaginary_freq=False)
    return np.asarray(result["freq_wavenumber"]), np.asarray(result["norm_mode"])


def load_modes(location, result, input_hashes, replay=True):
    """Keep computed-but-unaccepted frequencies, while verifying their real files."""
    identity = result["identity"]
    if identity not in COMPOSITIONS:
        raise ValueError("Unknown molecule identity")
    natoms = sum(COMPOSITIONS[identity].values())
    cm = np.asarray(result["frequencies_cm1"], dtype=float)
    thz = np.asarray(result["frequencies_thz"], dtype=float)
    modes = np.load(location / "normal_modes.npy", allow_pickle=False)
    raw = np.load(location / "hessian_hartree_per_bohr2.npy", allow_pickle=False)
    used = np.load(location / "hessian_used_for_analysis.npy", allow_pickle=False)
    coordinates = np.load(location / "normal_mode_geometry_angstrom.npy", allow_pickle=False)
    bohr = np.load(location / "normal_mode_geometry_bohr.npy", allow_pickle=False)
    symbols, rounded = LEGACY.xyz(location / "normal_mode_geometry.xyz")
    check_geometry(symbols, coordinates, identity)
    if (cm.ndim != 1 or not len(cm) or thz.shape != cm.shape or not np.isfinite(cm).all()
            or not np.isfinite(thz).all() or modes.shape != (len(cm), natoms, 3)
            or not np.isfinite(modes).all() or np.any(np.linalg.norm(modes.reshape(len(cm), -1), axis=1) == 0)
            or raw.shape != (natoms, natoms, 3, 3) or used.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(used).all()
            or bohr.shape != coordinates.shape or not np.isfinite(bohr).all()
            or not np.allclose(thz, cm * CM1_TO_THZ, atol=1e-12, rtol=1e-12)
            or not np.allclose(rounded, coordinates, atol=1e-11, rtol=0)
            or not np.allclose(bohr * BOHR_TO_ANGSTROM, coordinates, atol=1e-10, rtol=0)):
        raise ValueError("Computed frequency, Hessian, mode or geometry arrays are inconsistent")
    asym = float(np.max(np.abs(raw - raw.transpose(1, 0, 3, 2))))
    if (asym > 1e-5 or not np.allclose(used, .5 * (raw + raw.transpose(1, 0, 3, 2)), atol=1e-14, rtol=1e-12)
            or not np.isclose(asym, result["hessian_max_symmetry_difference"], atol=1e-14, rtol=1e-10)):
        raise ValueError("Raw Hessian asymmetry or saved analysis symmetrization differs")
    binding = {"raw_hessian_sha256": "hessian_hartree_per_bohr2.npy",
               "analysis_hessian_sha256": "hessian_used_for_analysis.npy",
               "normal_mode_geometry_angstrom_npy_sha256": "normal_mode_geometry_angstrom.npy"}
    for key, filename in binding.items():
        if result.get(key) != sha(location / filename):
            raise ValueError("Frequency evidence hash differs: " + key)
    with (location / "frequencies.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(cm):
        raise ValueError("Frequency CSV and result cardinalities disagree")
    for i, row in enumerate(csv_rows):
        if (int(row["mode"]) != i + 1 or row["IR_intensity_status"] != "NOT_COMPUTED"
                or not np.isclose(float(row["frequency_cm1"]), cm[i], atol=1e-10, rtol=0)
                or not np.isclose(float(row["frequency_thz"]), thz[i], atol=1e-10, rtol=0)):
            raise ValueError("Frequency CSV differs from actual result")
    gradient_path = location / "gradient_hartree_per_bohr.npy"
    statistics = gradient_statistics(np.load(gradient_path, allow_pickle=False))
    for key, value in statistics.items():
        if not np.isclose(value, result[key], atol=1e-13, rtol=1e-11):
            raise ValueError("Fresh final gradient statistics differ")
    replay_info = {"status": "NOT_REQUESTED", "quantum_jobs": 0}
    if replay:
        replay_cm, replay_modes = replay_harmonic(symbols, bohr, used)
        if replay_cm.shape != cm.shape or not np.allclose(replay_cm, cm, atol=1e-6, rtol=1e-9):
            raise ValueError("Saved symmetric Hessian harmonic frequency replay differs")
        replay_info = {"status": "SAVED_MATRIX_DIAGONALIZATION_PASS", "quantum_jobs": 0,
                       "max_absolute_frequency_difference_cm1": float(np.max(np.abs(replay_cm - cm))),
                       "recomputed_mode_shape": list(replay_modes.shape)}
    files = ["normal_modes.npy", "hessian_hartree_per_bohr2.npy", "hessian_used_for_analysis.npy",
             "normal_mode_geometry_angstrom.npy", "normal_mode_geometry_bohr.npy",
             "normal_mode_geometry.xyz", "frequencies.csv", "gradient_hartree_per_bohr.npy"]
    for filename in files:
        input_hashes[str(location / filename)] = sha(location / filename)
    accepted = valid_modes(result, identity)
    rows = [{"phase": location.name, "identity": identity, "mode": i + 1,
             "frequency_cm1": float(c), "frequency_thz": float(t), "imaginary": bool(c < 0),
             "accepted_for_comparison": accepted, "IR_intensity_status": "NOT_COMPUTED"}
            for i, (c, t) in enumerate(zip(cm, thz))]
    return rows, modes, (symbols, coordinates), replay_info


def plot_reports(output, steps, water, accepted, experimental, mode_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for phase in dict.fromkeys(row["phase"] for row in steps if row["identity"] == "spinosin"):
        records = [row for row in steps if row["phase"] == phase]
        cycles = [row["cycle"] for row in records]
        axes[0].plot(cycles, [row["energy_hartree"] for row in records], ".-", label=phase)
        for key, style, label in [("atom_rms_hartree_per_bohr", ".-", "RMS"), ("atom_max_hartree_per_bohr", "--", "max")]:
            axes[1].semilogy(cycles, [row[key] for row in records], style, label=phase + " " + label)
    axes[0].set(xlabel="Completed geometry evaluation", ylabel="Paired SCF energy / Hartree")
    axes[1].set(xlabel="Completed geometry evaluation", ylabel="Atom-vector gradient / Hartree Bohr$^{-1}$")
    axes[1].axhline(.0003, ls=":", c="gray", label="RMS threshold")
    axes[1].axhline(.00045, ls=":", c="black", label="max threshold")
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=6)
    fig.suptitle("Actual Spinosin trajectory: a completed evaluation is not convergence")
    LEGACY.save_figure(fig, output, "spinosin_optimization_progress"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for identity, rows, ax in [("water", water, axes[0]), ("spinosin", accepted, axes[1])]:
        if rows:
            for phase in dict.fromkeys(row["phase"] for row in rows):
                values = [row["frequency_thz"] for row in rows if row["phase"] == phase]
                y = list(dict.fromkeys(row["phase"] for row in rows)).index(phase)
                ax.plot(values, [y] * len(values), "|", ms=12, label=phase)
            ax.legend(fontsize=6)
        else:
            ax.text(.5, .5, "No accepted frequencies\nNOT_RUN or validation failed", transform=ax.transAxes, ha="center")
        ax.set(xlabel="Mode frequency / THz", ylabel="Calculation phase", title=identity + ": positions only, no IR intensities")
    LEGACY.save_figure(fig, output, "actual_mode_positions"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for row in experimental:
        axes[0].plot(row["experimental_frequency_thz"], row["outer_fold"], "|", ms=12, c="#ac5423")
    axes[0].set(title="Historical source-only experimental candidates", ylabel="Source fold")
    low = [row for row in accepted if .1 <= row["frequency_thz"] <= 1.5]
    if low:
        for row in low:
            axes[1].plot(row["frequency_thz"], row["mode"], "|", ms=12, c="#286090")
        axes[1].set(title="Validated isolated Spinosin positions", ylabel="Actual mode index")
    else:
        axes[1].text(.5, .5, "Comparison BLOCKED\nNo accepted Spinosin modes in range\nThis is not a measured zero absorption result", transform=axes[1].transAxes, ha="center")
        axes[1].set(title="No theoretical absorption curve", yticks=[])
    for ax in axes:
        ax.set(xlim=(.1, 1.5), xlabel="Frequency / THz")
    fig.suptitle("Frequency proximity cannot confirm chemical attribution")
    LEGACY.save_figure(fig, output, "mechanism_comparison_status"); plt.close(fig)
    for phase, rows, modes, geometry in mode_sets:
        symbols, coords = geometry
        for row in rows:
            if not .1 <= row["frequency_thz"] <= 1.5:
                continue
            mode = modes[row["mode"] - 1]; scale = .35 / max(np.linalg.norm(mode, axis=1).max(), 1e-30)
            fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection="3d")
            LEGACY.atom_plot(ax, symbols, coords, mode * scale)
            ax.set_title(f"Spinosin {phase}, mode {row['mode']}: {row['frequency_thz']:.6f} THz\nArbitrary display amplitude; no IR intensity", fontsize=9)
            fig.subplots_adjust(left=.15, right=.85, bottom=.15, top=.88)
            stem = f"{phase}_mode_{row['mode']:03d}"
            LEGACY.save_figure(fig, output, stem); plt.close(fig)
            frames = []
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                displaced = coords + np.sin(angle) * mode * scale
                frames.extend([str(len(symbols)), "Arbitrary display displacement; not thermal amplitude or IR intensity"] +
                              [f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for s, p in zip(symbols, displaced)])
            (output / (stem + ".xyz")).write_text("\n".join(frames) + "\n")


def run(root, output, release_path):
    root, output, release_path = root.resolve(), output.resolve(), release_path.resolve()
    if output.exists():
        raise FileExistsError("Preserve existing report; use a fresh output directory")
    ledger, authorization_path = authorize(root, read(release_path))
    if sha(PEAK_PATH) != PEAK_SHA256:
        raise ValueError("Historical experimental candidate bytes changed")
    experimental = LEGACY.experimental_rows(PEAK_PATH)
    if len(experimental) != 20:
        raise ValueError("Expected all20 historical experimental candidates")
    input_hashes = {str(root / "cpu_budget.json"): sha(root / "cpu_budget.json"), str(authorization_path): sha(authorization_path),
                    str(PEAK_PATH): sha(PEAK_PATH), str(release_path): sha(release_path), str(Path(__file__)): sha(__file__)}
    input_hashes[str(Path(LEGACY.__file__))] = sha(LEGACY.__file__)
    parent_checkpoint = Path(ledger["source_checkpoint_path"])
    parent_checkpoint = parent_checkpoint if parent_checkpoint.is_absolute() else ROOT / parent_checkpoint
    input_hashes[str(parent_checkpoint)] = sha(parent_checkpoint)
    phases, steps, computed, water, accepted, replays, mode_sets, missing_results = [], [], [], [], [], [], [], []
    for row in ledger["phases"]:
        driver_log = Path(row["log"])
        driver_log = driver_log if driver_log.is_absolute() else ROOT / driver_log
        if not driver_log.resolve().is_relative_to(root) or sha(driver_log) != row["log_sha256"]:
            raise ValueError("Phase driver log path/hash differs from final ledger")
        input_hashes[str(driver_log)] = sha(driver_log)
        location = phase_location(root, row)
        result = completed_phase_result(row, location, input_hashes)
        if location is not None:
            for name in ("result.json", "input.json", "progress.json"):
                if (location / name).exists(): input_hashes[str(location / name)] = sha(location / name)
        identity = result.get("identity", "UNKNOWN")
        if identity in COMPOSITIONS and not method_matches(result.get("settings", {})):
            raise ValueError("Unexpected electronic-structure method/settings")
        final_gradient = audit_final_gradient(location, result, input_hashes) if identity in COMPOSITIONS else {}
        if identity in COMPOSITIONS and not result.get("script_sha256"):
            raise ValueError("Identified worker lacks its executed source SHA")
        if result.get("script_sha256"):
            script_paths = [Path(value) for value in row["command"] if str(value).endswith(".py")]
            if len(script_paths) != 1:
                raise ValueError("Worker source path is not uniquely recorded")
            script_path = script_paths[0] if script_paths[0].is_absolute() else ROOT / script_paths[0]
            if sha(script_path) != result.get("script_sha256"):
                raise ValueError("Current worker source differs from recorded execution SHA")
            input_hashes[str(script_path)] = sha(script_path)
        phases.append({"phase": row["phase"], "identity": identity, "budget_status": row["status"],
                       "scientific_status": result.get("status", "PARTIAL_MISSING_WORKER_RESULT"),
                       "worker_result_missing": result["worker_result_missing"],
                       "completion_evidence_kind": result["completion_evidence_kind"],
                       "snapshot_event": result.get("snapshot_event"),
                       "actual_cpu_seconds": row["actual_cpu_seconds"], "wall_seconds": row["elapsed_wall_seconds"],
                       "threads": row["threads"], "exit_code": row["exit_code"],
                       "geometry_optimized": result.get("geometry_optimized", False),
                       "backend": result.get("settings", {}).get("density_fit_backend"),
                       "fresh_gradient_rms_hartree_per_bohr": final_gradient.get("atom_rms_hartree_per_bohr"),
                       "fresh_gradient_max_hartree_per_bohr": final_gradient.get("atom_max_hartree_per_bohr"),
                       "frequencies_status": result.get("frequencies_status", "NOT_RUN"),
                       "IR_intensity_status": "NOT_COMPUTED", "stop_reason": result.get("stop_reason", row.get("failure"))})
        if result["worker_result_missing"]:
            missing_results.append({"phase": row["phase"], "worker_output_directory": str(location),
                "worker_result_missing": True, "derived_status": "PARTIAL", "parent_exit_code": row.get("exit_code"),
                "actual_cpu_seconds": row["actual_cpu_seconds"], "ended_at_utc": row["ended_at_utc"],
                "identity_from_unfinalized_snapshots": identity, "snapshot_event": result.get("snapshot_event"),
                "unfinalized_frequency_status_claim": result.get("unfinalized_frequency_status_claim"),
                "frequency_validation_permitted": False, "no_result_json_fabricated": True,
                "observed_closed_artifacts": result.get("observed_closed_artifacts", {})})
        if identity in COMPOSITIONS:
            for path in sorted((location / "geometry_steps").glob("step_*.json")):
                steps.append(audit_step(path, identity, row["phase"], input_hashes))
        if result.get("frequencies_status") == "DONE":
            frequency_parent_guard(root, row, result, input_hashes)
            mode_rows, modes, geometry, replay = load_modes(location, result, input_hashes)
            computed.extend(mode_rows); replays.append({"phase": row["phase"], **replay})
            if valid_modes(result, identity):
                if identity == "water": water.extend(mode_rows)
                else:
                    accepted.extend(mode_rows); mode_sets.append((row["phase"], mode_rows, modes, geometry))
    low = [row for row in accepted if .1 <= row["frequency_thz"] <= 1.5]
    pairs = LEGACY.comparison_rows(experimental, low)
    output.mkdir(parents=True)
    frequency_fields = ["phase", "identity", "mode", "frequency_cm1", "frequency_thz", "imaginary", "accepted_for_comparison", "IR_intensity_status"]
    csvwrite(output / "phase_status_and_resources.csv", phases, list(phases[0]) if phases else ["phase", "scientific_status"])
    csvwrite(output / "actual_paired_geometry_steps.csv", steps, list(steps[0]) if steps else ["phase", "cycle", "energy_hartree"])
    for filename, rows in [("computed_frequencies_including_unvalidated.csv", computed), ("water_validated_frequencies.csv", water),
                           ("spinosin_validated_all_frequencies.csv", accepted), ("spinosin_validated_0p1_1p5_thz_modes.csv", low)]:
        csvwrite(output / filename, rows, frequency_fields)
    csvwrite(output / "experimental_candidate_positions.csv", experimental, list(experimental[0]))
    csvwrite(output / "experiment_theory_pairs.csv", pairs, list(pairs[0]))
    write(output / "saved_matrix_frequency_replays.json", replays)
    write(output / "missing_worker_result_audit.json", missing_results)
    plot_reports(output, steps, water, accepted, experimental, mode_sets)
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_only": True,
               "new_SCF_gradient_Hessian_evaluations": 0, "actual_cpu_seconds": ledger["consumed_cpu_seconds"],
               "quantum_cpu_cap_seconds": ledger["quantum_cpu_cap_seconds"], "paired_geometry_evaluations": len(steps),
               "ended_phases_without_worker_result": len(missing_results), "no_worker_result_json_fabricated": True,
               "spinosin_paired_geometry_evaluations": sum(row["identity"] == "spinosin" for row in steps),
               "computed_frequency_rows_including_unvalidated": len(computed), "validated_water_frequency_rows": len(water),
               "validated_spinosin_frequency_rows": len(accepted), "validated_spinosin_modes_0p1_1p5_thz": len(low),
               "spinosin_frequency_interpretation": "VALIDATED_ISOLATED_MOLECULE_MODES_ONLY" if accepted else "BLOCKED_NO_ACCEPTED_MODES",
               "NOT_RUN_is_not_a_computed_zero_spectrum": True, "chemical_attribution_confirmed": False,
               "IR_intensities_status": "NOT_COMPUTED", "classification_modified": False,
               "experimental_candidate_rows": len(experimental), "comparison_pair_rows": len(pairs), "input_sha256": input_hashes}
    for path, expected in input_hashes.items():
        if sha(path) != expected:
            raise ValueError("Input changed during read-only final analysis: " + path)
    write(output / "summary.json", summary)
    (output / "README.md").write_text("# PySCF续算的实际结果审计\n\n本目录只读重算保存梯度统计与Hessian正常模式，不启动SCF、梯度或Hessian导数计算。\n\n"
        + f"实际配对几何评估{len(steps)}条，水通过频率{len(water)}条，斯皮诺素通过频率{len(accepted)}条。多后端水记录不是不同分子。\n\n"
        + "computed_frequencies_including_unvalidated.csv保留未通过极小点／正频率检查的真实计算频率，不能据此作峰归属；未计算则只保留表头。NOT_RUN不等于零吸收或不存在低频模式。\n\n"
        + "已结束但没有result.json的原生进程以父进程真实退出记录和原样input／progress快照派生PARTIAL，明确worker_result_missing；不向计算目录补造result.json。快照中的频率完成声明没有最终结果时一律不获得验证资格，已有Hessian／缓存文件仅作为实际部分执行证据，不能代替完成频率计算。详见missing_worker_result_audit.json。\n\n"
        + "所有模式图只用实际频率分析的完整精度几何；箭头／动画幅度任意缩放，不是热运动幅度或IR强度。IR均NOT_COMPUTED，频率位置不代表吸收强度。全部实验候选与全部合格范围内模式作笛卡尔对照，不按接近程度挑构象或计算方法。\n\n"
        + "孤立分子模式不包括固态晶格、集体振动与复杂基质。频率接近不能确认药材的化学归属，需要纯品／含量测定等证据。本次不修改96.0234%分类模型。\n")
    write(output / "files_sha256.json", {str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
                                         for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.run_root, args.output, args.release)
    print(json.dumps({key: value for key, value in summary.items() if key != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
