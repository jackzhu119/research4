"""Closed V5 modes audit and descriptive mechanism export; no quantum engine.

Default: Spinosin only (75 atoms, 219 modes). Explicit control-validation is
restricted to water/ethanol and never generates Spinosin mechanism comparisons.
Equal-height sticks and rescaled XYZ frames are display aids, not IR or MD.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys

# One analysis process; do not borrow the quantum worker's four-thread policy.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
CONSUMER_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
REPLAY = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5/real-restart-validation-independent-audit/replay_restarts_v5.py"
REPLAY_SHA = "49a12e894e0a1afa48ebffbb80391dc4f414a8168a4368f8b807b9769ec46ad5"
PEAKS = ROOT / "reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv"
PEAK_SHA = "b3a5f59157f42859f68edf0a8760d9f811f83ea7947c978cc0a9cbc98a4ba8b4"
CM1_TO_THZ = .0299792458
BOHR_TO_ANGSTROM = .52917721092
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}
COMPOSITIONS = {"Spinosin": {"C": 28, "H": 32, "O": 15},
                "water_software_control": {"O": 1, "H": 2},
                "ethanol_software_control": {"C": 2, "H": 6, "O": 1}}
FREQ_FIELDS = ("case", "mode", "frequency_cm1", "frequency_THz", "imaginary",
               "qualified_for_mechanism", "IR_intensity_status")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(path, expected, name):
    require(sha(path) == expected, "Frozen analysis dependency changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def checked(path, hashes, expected=None):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Missing/symlink input: " + str(path))
    require(path.stat().st_size <= 64 * 1024**2, "Analysis input exceeds small-artifact limit")
    path = path.resolve()
    digest = sha(path)
    require(expected is None or digest == expected, "Input SHA changed: " + str(path))
    hashes[str(path)] = {"sha256": digest, "bytes": path.stat().st_size,
                         "producer_hash_present": expected is not None}
    return path


def read_json(path, hashes, expected=None):
    return json.loads(checked(path, hashes, expected).read_text())


def array(path, shape, hashes, expected=None):
    value = np.load(checked(path, hashes, expected), allow_pickle=False)
    require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(),
            "Invalid complete float64 array: " + str(path))
    return value


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csvwrite(path, rows, fields):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def case_for_symbols(symbols, control):
    composition = dict(Counter(symbols))
    if not control:
        require(composition == COMPOSITIONS["Spinosin"] and len(symbols) == 75,
                "Default input must be Spinosin C28H32O15, 75 atoms; controls require explicit flag")
        return "Spinosin"
    for label in ("water_software_control", "ethanol_software_control"):
        if composition == COMPOSITIONS[label]:
            return label
    raise ValueError("Control-validation allows only actual closed water or ethanol")


def validate_frequency_rows(rows, cm, thz):
    require(len(rows) == len(cm) and np.isfinite(cm).all() and np.isfinite(thz).all(),
            "Missing/nonfinite frequency rows")
    require(np.allclose(thz, cm * CM1_TO_THZ, atol=1e-12, rtol=0), "JSON THz conversion differs")
    for index, row in enumerate(rows):
        require(int(row["mode"]) == index + 1 and float(row["frequency_cm1"]) == cm[index]
                and float(row["frequency_thz"]) == thz[index]
                and row["imaginary"] == str(bool(cm[index] < 0))
                and row["IR_intensity_status"] == "NOT_COMPUTED", "CSV/JSON signed frequency mismatch")


def numpy_mode_audit(raw, used, modes, cm, coords_bohr, symbols, replay=None):
    """Independent NumPy TR projection; no PySCF import or Hessian evaluation."""
    replay = replay or load_frozen(REPLAY, REPLAY_SHA, "frozen_numpy_TR_replay")
    n, count = len(symbols), 3 * len(symbols) - 6
    require(raw.shape == used.shape == (n, n, 3, 3) and modes.shape == (count, n, 3)
            and cm.shape == (count,) and coords_bohr.shape == (n, 3), "Incomplete mode/Hessian dimensions")
    require(all(np.isfinite(x).all() and not np.iscomplexobj(x) for x in (raw, used, modes, cm, coords_bohr)),
            "Nonfinite/complex mode evidence")
    require(np.array_equal(used, (raw + raw.transpose(1, 0, 3, 2)) * .5), "Unexpected Hessian symmetrization")
    asymmetry = float(np.max(np.abs(raw - raw.transpose(1, 0, 3, 2))))
    require(asymmetry <= 1e-5, "Raw Hessian asymmetry exceeds the frozen threshold")
    mass = np.array([MASS[symbol] for symbol in symbols])
    projection, basis = replay.projector(coords_bohr, mass)
    matrix = np.einsum("pqxy,p,q->pqxy", used, mass**-.5, mass**-.5).transpose(0, 2, 1, 3).reshape(3*n, 3*n)
    eigenvalues, _ = np.linalg.eigh(basis.T @ matrix @ basis)
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * replay.FACTOR
    require(len(eigenvalues) == count and np.allclose(frequencies, cm, atol=1e-6, rtol=1e-9),
            "Independent signed normal-mode frequency replay differs")
    z = (modes * np.sqrt(mass)[None, :, None]).reshape(count, 3*n).T
    gram = float(np.max(np.abs(z.T @ z - np.eye(count))))
    external = float(np.max(np.abs(projection @ z - z)))
    saved_eigenvalues = np.sign(cm) * (np.abs(cm) / replay.FACTOR)**2
    residual = float(np.max(np.abs(projection @ matrix @ projection @ z - z * saved_eigenvalues[None, :])))
    require(max(gram, external, residual) <= 1e-8, "Mode orthogonality/TR/projected eigen-equation failed")
    return {"status": "PASS_INDEPENDENT_NUMPY_TR_REPLAY", "mode_count": count,
            "raw_Hessian_max_asymmetry": asymmetry,
            "max_frequency_difference_cm1": float(np.max(np.abs(frequencies-cm))),
            "mass_weighted_orthogonality_error": gram, "external_projection_error": external,
            "projected_eigen_equation_residual": residual,
            "projected_eigen_residual_units": "Eh/(amu Bohr^2)",
            "eigen_residual_is_not_a_CPHF_residual": True, "signed_imaginary_modes_retained": True,
            "masses_amu": MASS, "frequency_factor_cm1": float(replay.FACTOR),
            "arbitrary_signs_and_degenerate_rotations_allowed": True}


def global_kernel_check(value, result, n):
    expected = {"status": "DONE_INTERMEDIATE_CAPTURE_ONLY", "native_global_kernel_returned_normally": True,
                "H1AO_complete": True, "global_Hessian_assembly_unchanged": True, "partial_kernel_tiling_used": False,
                "response_block_atoms": 8, "native_conv_tol_cpscf": 1e-8,
                "native_max_cycle": 50, "native_level_shift": 0,
                "independent_CPHF_residual_norm": "NOT_COMPUTED"}
    require(all(value.get(key) == item for key, item in expected.items()), "Global Hessian/response completion missing")
    require(value["dimensions"]["atoms"] == n and value["response_complete_global_atom_ids"] == list(range(n))
            and [x["atom_id"] for x in value["H1AO_atoms"]] == list(range(n)), "Full atom response coverage missing")
    require(value["partial_hessian"]["native_function_returned_normally"] is True, "RKS partial Hessian not complete")
    binding = value["binding"]
    require(hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() == value["binding_sha256"],
            "Global Hessian binding hash differs")
    require(binding["settings"] == result["settings"] and binding["input_evidence"]["source_snapshot_sha256"] == result["snapshot_manifest_sha256"]
            and binding["input_evidence"]["actual_entry_source_sha256"] == CONSUMER_SHA
            and binding["input_evidence"]["consumer_resource_proof_sha256"] == result["resource_supervision"]["proof_sha256"],
            "Full Hessian source/proof binding differs")
    chunks = [list(range(i, min(i+8, n))) for i in range(0, n, 8)]
    require([b["global_atom_ids"] for b in value["response_blocks"]] == chunks, "Response atom blocks changed")
    for block, atoms in zip(value["response_blocks"], chunks):
        require(block["native_solver_returned_normally"] is True and [x["atom_id"] for x in block["arrays"]] == atoms
                and block["residual_norm"] == "NOT_COMPUTED", "Incomplete CPHF response block")
    return {"status": "COMPLETE_GLOBAL_RKS_RESPONSE_METADATA_PASS", "large_H1_MO1_arrays_read": False,
            "CPHF_residual_independently_measured": False}


def experimental_candidates(hashes, path=PEAKS, expected_sha=PEAK_SHA):
    with checked(path, hashes, expected_sha).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 20, "Retain all 20 overlapping-fold candidate records")
    result = []
    for row in rows:
        value = {key: int(row[key]) for key in ("outer_fold", "candidate_number", "frequency_index")}
        frequency, cm = float(row["frequency_thz"]), float(row["wavenumber_cm1"])
        require(.1 <= frequency <= 1.5 and np.isclose(frequency, cm * CM1_TO_THZ, atol=1e-10, rtol=0)
                and row["classification_prior_used"] == "False" and not row["chemical_assignment"],
                "Unexpected experimental candidate content or unit")
        result.append({**value, "experimental_frequency_THz": frequency, "experimental_wavenumber_cm1": cm,
                       "independent_chemical_peak": False, "absolute_alpha_units": "UNKNOWN",
                       "chemical_assignment": "NOT_CONFIRMED"})
    require({(r["outer_fold"], r["candidate_number"]) for r in result} == {(i, j) for i in range(5) for j in range(1, 5)},
            "Missing or duplicate fold/candidate record")
    return result


def cartesian_pairs(experimental, modes):
    require(len(experimental) == 20, "No deduplication of overlapping-fold records")
    return [{**peak, "theoretical_mode": mode["mode"], "theoretical_frequency_THz": mode["frequency_THz"],
             "theoretical_wavenumber_cm1": mode["frequency_cm1"],
             "signed_delta_THz": mode["frequency_THz"]-peak["experimental_frequency_THz"],
             "absolute_delta_THz": abs(mode["frequency_THz"]-peak["experimental_frequency_THz"]),
             "signed_delta_cm1": mode["frequency_cm1"]-peak["experimental_wavenumber_cm1"],
             "absolute_delta_cm1": abs(mode["frequency_cm1"]-peak["experimental_wavenumber_cm1"]),
             "IR_intensity_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
             "comparison": "ALL_PAIRS_DESCRIPTIVE_ONLY_NO_NEAREST_PEAK_ASSIGNMENT"}
            for peak in experimental for mode in modes]


def mechanism_qualified(cm, *, control, fresh_gradient_verified):
    values = np.asarray(cm, float)
    return bool(not control and fresh_gradient_verified and values.shape == (219,)
                and np.isfinite(values).all() and np.all(values > 0))


def animate(path, symbols, xyz, vector, mode, cm, label, amplitude=.15, frames=31):
    require(amplitude > 0 and frames >= 3 and np.isfinite(vector).all(), "Invalid display animation")
    scale = np.max(np.linalg.norm(vector, axis=1))
    require(scale > 0, "Zero normal mode")
    with Path(path).open("x") as handle:
        for phase in np.linspace(0, 2*np.pi, frames):
            positions = xyz + amplitude * np.sin(phase) * vector / scale
            handle.write(f"{len(symbols)}\n{label}; actual mode {mode}; {cm:.12g} cm^-1; DISPLAY_ONLY_NOT_MD; arbitrary max amplitude {amplitude} Angstrom; source atom order; no IR intensity\n")
            for symbol, position in zip(symbols, positions):
                handle.write(symbol + " " + " ".join(f"{x:.12f}" for x in position) + "\n")


def plots(output, rows, low, case, experimental=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.plot([r["mode"] for r in rows], [r["frequency_THz"] for r in rows], ".-", ms=2, lw=.6)
    ax.axhline(0, color="gray", lw=.7)
    ax.set(xlabel="Actual normal-mode index", ylabel="Signed frequency / THz",
           title=case+": computed modes (not IR intensities)")
    for extension in ("png", "pdf"):
        fig.savefig(output / ("signed_mode_frequencies."+extension), dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2 if experimental is not None else 1, 1, figsize=(8, 4.5), squeeze=False, constrained_layout=True)
    axis = axes[0, 0]
    axis.vlines([r["frequency_THz"] for r in low], 0, 1, color="#286090", lw=1)
    axis.set(xlim=(.1, 1.5), ylim=(0, 1.2), xlabel="Frequency / THz", ylabel="Equal display height",
             title=case+": isolated-molecule mode positions, not IR absorption")
    if not low:
        axis.text(.5, .5, "No computed modes in this window", transform=axis.transAxes, ha="center")
    if experimental is not None:
        axis = axes[1, 0]
        axis.scatter([r["experimental_frequency_THz"] for r in experimental], [r["outer_fold"] for r in experimental], s=12, color="#555555")
        axis.set(xlim=(.1, 1.5), yticks=range(5), xlabel="Candidate position / THz", ylabel="Development fold",
                 title="20 overlapping-fold records; not 20 independent chemical peaks")
    for extension in ("png", "pdf"):
        fig.savefig(output / ("low_frequency_positions_not_IR."+extension), dpi=200)
    plt.close(fig)


def analyze(result_path, expected_sha, output, *, control=False):
    output, result_path = Path(output).resolve(), Path(result_path).resolve()
    require(not output.exists(), "Refuse overwrite of existing analysis")
    hashes = {}
    require(expected_sha and result_path.is_relative_to(ROOT / "reports"), "Explicit closed report path and result SHA required")
    result = read_json(result_path, hashes, expected_sha)
    require(result.get("status") == "FREQUENCIES_COMPUTED" and result.get("frequencies_status") == "DONE"
            and result.get("complete_global_Hessian") is True and result.get("complete_RHF_helper") is True
            and result.get("normal_modes_complete") is True,
            "Target frequencies NOT_RUN or incomplete; do not read molecular arrays")
    require(result.get("atoms") in ((3, 9) if control else (75,)), "Control/default atom-count mismatch")
    consumer = load_frozen(CONSUMER, CONSUMER_SHA, "frozen_V5_postprocess_consumer")
    evidence = {"schema": consumer.EXECUTION_SCHEMA, "source_worker": result["source_worker"],
                "resource_supervision": result["resource_supervision"], "output_directory": str(result_path.parent),
                "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "phase": result["phase"]}
    closure = consumer.verify_closed_execution(evidence, expected_snapshot_sha=result["snapshot_manifest_sha256"], expected_output=result_path.parent)
    require(closure["producer_result_sha256"] == expected_sha and result.get("no_new_SCF_performed") is True
            and result.get("stable_MF_state_unchanged") is True and result.get("IR_intensities_status") == "NOT_COMPUTED",
            "Incomplete source/closed-result evidence")
    require(closure["actual_cpu_seconds"] <= result["resource_supervision"]["proof"]["cpu_hard_limit_seconds"],
            "Closed producer exceeded its recorded phase CPU grant")
    gate = result["gradient_gate"]["result"]
    qualified_gradient = consumer.qualify_gradient_receipt(gate["path"], gate["sha256"], result["snapshot_manifest_sha256"], result["boot_id"])
    require(qualified_gradient == result["gradient_gate"], "Original same-state fresh-gradient receipt differs")
    gradient_result = read_json(gate["path"], hashes, gate["sha256"])
    # Only a closed, healthy exact source permits the first molecular-array read.
    w = consumer.state_module()
    manifest, molecule, source_arrays, binding = w.load_snapshot_payload(result["snapshot_directory"], expected_manifest_sha256=result["snapshot_manifest_sha256"])
    require(binding == result["source_state_binding"], "Exact source MF binding differs")
    symbols = [row[0] for row in molecule["_atom"]]
    case = case_for_symbols(symbols, control)
    if not control:
        require("spinosin" in str(result["identity"]).lower()
                and "spinosin" in str(binding["producer"]["identity"]).lower(), "Source identity is not declared Spinosin")
    n, count, location = len(symbols), 3*len(symbols)-6, result_path.parent
    require(result["atoms"] == n and result["normal_mode_count"] == result["expected_normal_mode_count"] == count,
            "Complete nonlinear mode count differs")
    gradient = array(Path(gate["path"]).parent / "gradient_hartree_per_bohr.npy", (n, 3), hashes, gradient_result["gradient_npy_sha256"])
    delta_g = float(np.max(np.abs(gradient-source_arrays["gradient"])))
    lengths = np.linalg.norm(source_arrays["gradient"], axis=1)
    require(delta_g <= 1e-8 and np.sqrt(np.mean(lengths**2)) <= 3e-4 and lengths.max() <= 4.5e-4,
            "Source/fresh restored gradient not stationary or not identical")
    require(np.isclose(delta_g, gradient_result["restored_gradient_max_abs_difference_hartree_per_bohr"], atol=1e-13, rtol=0),
            "Gradient difference metadata inconsistent")
    xyz = array(location / "normal_mode_geometry_angstrom.npy", (n, 3), hashes)
    bohr = array(location / "normal_mode_geometry_bohr.npy", (n, 3), hashes)
    require(np.array_equal(xyz, source_arrays["coordinates_angstrom"])
            and w.frozen_base().array_fingerprint(bohr) == manifest["molecule_fingerprint"]["coordinates_bohr"]
            and np.allclose(bohr * BOHR_TO_ANGSTROM, xyz, atol=1e-10, rtol=0), "Source atom order or geometry/unit changed")
    raw = array(location / "hessian_hartree_per_bohr2.npy", (n, n, 3, 3), hashes, result["raw_hessian_sha256"])
    used = array(location / "hessian_used_for_analysis.npy", raw.shape, hashes, result["analysis_hessian_sha256"])
    modes = array(location / "normal_modes.npy", (count, n, 3), hashes)
    reduced_mass = array(location / "reduced_mass_amu.npy", (count,), hashes)
    require(np.allclose(reduced_mass, 1/np.sum(modes**2, axis=(1, 2)), atol=1e-10, rtol=1e-10), "Reduced masses disagree with normal modes")
    cm, thz = np.asarray(result["frequencies_cm1"], float), np.asarray(result["frequencies_thz"], float)
    with checked(location / "frequencies.csv", hashes).open(newline="") as handle:
        validate_frequency_rows(list(csv.DictReader(handle)), cm, thz)
    numerical = numpy_mode_audit(raw, used, modes, cm, bohr, symbols)
    kernel = global_kernel_check(read_json(location / "hessian_cache_manifest.json", hashes, result["hessian_cache_manifest_sha256"]), result, n)
    positive = bool(np.all(cm > 0))
    require(result["imaginary_frequency_count"] == int(np.sum(cm < 0))
            and result["all_frequencies_real_positive"] is positive
            and result["stationary_minimum_verified"] is positive
            and result["frequency_validation_passed"] is positive, "Minimum/imaginary flags contradict actual frequencies")
    qualified = mechanism_qualified(cm, control=control, fresh_gradient_verified=True)
    rows = [{"case": case, "mode": i+1, "frequency_cm1": float(value), "frequency_THz": float(thz[i]),
             "imaginary": bool(value < 0), "qualified_for_mechanism": qualified,
             "IR_intensity_status": "NOT_COMPUTED"} for i, value in enumerate(cm)]
    low = [r for r in rows if .1 <= r["frequency_THz"] <= 1.5]
    experimental = experimental_candidates(hashes) if qualified else None
    pairs = cartesian_pairs(experimental, low) if qualified else None
    output.mkdir(parents=True)
    csvwrite(output / "all_frequencies_cm1_THz.csv", rows, FREQ_FIELDS)
    csvwrite(output / "normal_modes_0p1_to_1p5_THz.csv", low, FREQ_FIELDS)
    if pairs is not None:
        fields = [*experimental[0], "theoretical_mode", "theoretical_frequency_THz", "theoretical_wavenumber_cm1",
                  "signed_delta_THz", "absolute_delta_THz", "signed_delta_cm1", "absolute_delta_cm1",
                  "IR_intensity_status", "chemical_attribution_verified", "comparison"]
        csvwrite(output / "all_experimental_theory_pairs.csv", pairs, fields)
    plots(output, rows, low, case, experimental)
    animated = []
    selections = ([rows[0]] if control and positive else low[:6]) if positive else []
    if selections:
        (output / "animations").mkdir()
    for row in selections:
        target = output / "animations" / f"{case}_mode_{row['mode']:03d}.xyz"
        animate(target, symbols, xyz, modes[row["mode"]-1], row["mode"], row["frequency_cm1"], case)
        animated.append(target.name)
    for path, record in hashes.items():
        require(sha(path) == record["sha256"], "Closed input changed during postprocessing: " + path)
    audit = {"schema": "research4-V5-independent-modes-analysis-v1", "status": "DONE_CONTROL_VALIDATION" if control else "DONE_COMPUTED_MODES_AUDITED",
             "case": case, "created_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_source_sha256": sha(__file__),
             "consumer_sha256": CONSUMER_SHA, "numpy_replay_source_sha256": REPLAY_SHA,
             "producer_closure": closure, "source_result_sha256": expected_sha,
             "snapshot_manifest_sha256": result["snapshot_manifest_sha256"],
             "source_SCF_closed_producer": binding["closed_producer"], "restored_gradient_gate": qualified_gradient,
             "energy_hartree": manifest["energy_hartree"],
             "electronic_method": {key: binding["method_settings"][key] for key in
                 ("xc", "basis", "grid_level", "dispersion", "density_fitting", "direct_scf_tol")},
             "atoms": n, "normal_modes": count, "all_frequencies_positive": positive,
             "imaginary_frequency_count": int(np.sum(cm < 0)), "zero_frequency_count": int(np.sum(cm == 0)),
             "qualified_for_mechanism": qualified, "mechanism_pairing_status": "ALL_PAIRS_DESCRIPTIVE_ONLY" if qualified else "BLOCKED_CONTROL_OR_NONPOSITIVE_MINIMUM",
             "mechanism_qualification_scope": "Positive harmonic isolated-molecule calculation only; not chemical attribution, IR activity, or a global conformer minimum",
             "experimental_fold_records": 20 if qualified else "NOT_READ", "independent_chemical_peak_count": "NOT_ESTABLISHED",
             "low_frequency_mode_count": len(low), "cartesian_comparison_count": len(pairs) if pairs is not None else "NOT_GENERATED",
             "numpy_replay": numerical, "global_kernel": kernel, "fresh_gradient_max_difference_Eh_per_Bohr": delta_g,
             "source_gradient_atom_RMS_Eh_per_Bohr": float(np.sqrt(np.mean(lengths**2))),
             "source_gradient_atom_max_Eh_per_Bohr": float(lengths.max()),
             "source_MF_unchanged_verified": True, "normal_modes_and_CSV_without_producer_file_hash": "CURRENT_HASH_RECORDED_AND_CHECKED_AGAINST_ANCHORED_HESSIAN_AND_JSON; NO_OLD_HASH_INVENTED",
             "animations": animated, "animation_policy": "DISPLAY_ONLY_NOT_MD; arbitrary 0.15Angstrom max displacement; source atom ordering; first six low modes or one software-control mode",
             "IR_intensities_status": "NOT_COMPUTED", "experimental_absolute_alpha_units": "UNKNOWN_NOT_IR_EQUIVALENT",
             "chemical_attribution_verified": False, "theory_or_source_peaks_enter_classifier": False,
             "Spinosin_target_status": "NOT_RUN_CONTROL_ONLY" if control else "ACTUAL_CLOSED_FREQUENCIES_AUDITED",
             "new_quantum_evaluations": 0, "PySCF_imported": any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules),
             "postprocessing_environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "matplotlib": sys.modules["matplotlib"].__version__,
                 "max_virtual_address_space_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                 "CPU_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "UNAVAILABLE",
                 "BLAS_threads": os.environ["OPENBLAS_NUM_THREADS"]},
             "input_hashes": hashes, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    require(not audit["PySCF_imported"], "Postprocessor must not import PySCF")
    write_json(output / "checks.json", audit)
    (output / "README.md").write_text(f"{case}: {count} actual signed modes audited. Mechanism qualification: {qualified}. IR NOT_COMPUTED; IR activity is unknown and a computed mode may be IR inactive. Equal-height sticks are mode positions, not intensities. XYZ frames have arbitrary display amplitude and are not molecular dynamics. Isolated-molecule frequencies are not solid medicinal-material THz absorption; no chemical assignment is confirmed, and one conformer is not a global-minimum search. Twenty overlapping-fold experimental records, when eligible, are retained in a full Cartesian comparison rather than nearest-peak matching. Experimental absolute alpha units are unknown. Theory and candidate peaks never enter the classifier. Control runs do not represent Spinosin calculations.\n")
    seal_output(output)
    return audit


def seal_output(output):
    records = [{"path": str(p.relative_to(output)), "sha256": sha(p), "bytes": p.stat().st_size}
               for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "artifact_index.json", {"analysis_source_sha256": sha(__file__), "files": records})


def preflight(output):
    output = Path(output).resolve()
    require(not output.exists(), "Preserve existing preflight")
    require(sha(CONSUMER) == CONSUMER_SHA and sha(REPLAY) == REPLAY_SHA and sha(PEAKS) == PEAK_SHA,
            "Frozen prerequisites differ")
    output.mkdir(parents=True)
    value = {"status": "NOT_RUN", "target": "Spinosin C28H32O15,75atoms,219modes",
             "target_frequencies_read": False, "new_quantum_evaluations": 0,
             "qualified_for_mechanism": False, "IR_intensities_status": "NOT_COMPUTED",
             "analysis_source_sha256": sha(__file__), "consumer_sha256": CONSUMER_SHA,
             "numpy_helper_sha256": REPLAY_SHA, "experimental_candidates_sha256": PEAK_SHA}
    write_json(output / "preflight.json", value)
    (output / "README.md").write_text("NOT_RUN: no Spinosin molecular arrays or frequencies have been read. After actual complete healthy V5 closure, pass --result and its explicit SHA to this script. --control-validation is limited to actual closed water/ethanol software controls. No empty frequency table is presented as a target result.\n")
    seal_output(output)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--result-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-validation", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    # Restrict this process only. No new dependencies or quantum environment edits.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (min(cap, soft) if soft > 0 else cap, min(cap, hard) if hard > 0 else cap))
    if args.preflight:
        require(args.result is None and not args.control_validation, "Preflight does not consume any result")
        result = preflight(args.output)
    else:
        require(args.result is not None and args.result_sha256, "Explicit result and result SHA required")
        result = analyze(args.result, args.result_sha256, args.output, control=args.control_validation)
    print(json.dumps({k: result.get(k) for k in ("status", "case", "normal_modes", "qualified_for_mechanism", "Spinosin_target_status")}))


if __name__ == "__main__":
    main()
