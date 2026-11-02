"""Screened direct RKS V4: explicit no-density-fitting control and vibrations.

The parent process supplies CPU/wall limits. Each worker writes restart evidence
while computing. Merely returning a geometry is never optimization success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CM1_TO_THZ = 0.0299792458
FORMULAS = {"water": {"H": 2, "O": 1}, "spinosin": {"C": 28, "H": 32, "O": 15}}
THRESHOLDS = {"convergence_energy": 1e-6, "convergence_grms": 3e-4,
              "convergence_gmax": 4.5e-4, "convergence_drms": 1.2e-3,
              "convergence_dmax": 1.8e-3}


class BudgetStop(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def xyz_text(symbols, coordinates, title):
    coordinates = np.asarray(coordinates, float)
    if coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid XYZ coordinates")
    return str(len(symbols)) + "\n" + title.replace("\n", " ") + "\n" + "\n".join(
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coordinates)) + "\n"


def read_xyz(path, expected):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("Complete XYZ required")
    declared = int(lines[0])
    rows = [line.split() for line in lines[2:] if line.strip()]
    if len(rows) != declared or any(len(row) != 4 for row in rows):
        raise ValueError("XYZ count/column mismatch")
    symbols = [row[0] for row in rows]
    coordinates = np.asarray([[float(v) for v in row[1:]] for row in rows])
    if Counter(symbols) != Counter(expected) or not np.isfinite(coordinates).all():
        raise ValueError("XYZ composition or finite-coordinate check failed")
    if np.any(np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)[np.triu_indices(declared, 1)] < .25):
        raise ValueError("Coincident or implausibly overlapping atoms")
    return symbols, coordinates


def gradient_summary(gradient):
    values = np.asarray(gradient, float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Finite atom-by-three nuclear gradient required")
    atom_norms = np.linalg.norm(values, axis=1)
    return {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(values ** 2))),
            "component_max_hartree_per_bohr": float(np.max(np.abs(values))),
            "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(atom_norms ** 2))),
            "atom_max_hartree_per_bohr": float(np.max(atom_norms)),
            "gradient_norm_hartree_per_bohr": float(np.linalg.norm(values))}


def stationary_checks(gradient):
    info = gradient_summary(gradient)
    # geomeTRIC uses atom displacement-vector gradient norms, not 3N scalar RMS.
    passed = (info["atom_rms_hartree_per_bohr"] <= THRESHOLDS["convergence_grms"]
              and info["atom_max_hartree_per_bohr"] <= THRESHOLDS["convergence_gmax"])
    return {**info, "final_gradient_thresholds_passed": bool(passed)}


def require_optimized_result(result, geometry_path, settings, coordinates_path=None):
    if result.get("status") != "DONE" or not result.get("geometry_optimized"):
        raise ValueError("Frequency stage blocked: optimization not DONE")
    if result.get("optimizer_converged") is not True or result.get("scf_converged") is not True:
        raise ValueError("Frequency stage blocked: unconverged optimization/SCF")
    if result.get("final_gradient_thresholds_passed") is not True:
        raise ValueError("Frequency stage blocked: gradient not stationary")
    if result.get("optimized_xyz_sha256") != sha(geometry_path):
        raise ValueError("Optimized geometry bytes changed")
    expected_coordinates_sha = result.get("optimized_coordinates_angstrom_npy_sha256")
    if not expected_coordinates_sha or coordinates_path is None or expected_coordinates_sha != sha(coordinates_path):
        raise ValueError("Frequency stage requires the same full-precision optimized coordinates")
    for key in ("xc", "basis", "grid_level", "dispersion", "density_fitting", "auxbasis", "density_fit_backend",
                "direct_scf", "direct_scf_tol", "scf_conv_tol_hartree", "optimizer_thresholds",
                "global_four_center_ERI_cache_allowed", "charge", "spin_2S", "multiplicity"):
        if result.get("settings", {}).get(key) != settings[key]:
            raise ValueError("Frequency method differs from optimized method: " + key)


def frequencies_payload(wavenumbers, modes, atom_count):
    if np.iscomplexobj(wavenumbers) or np.iscomplexobj(modes):
        raise ValueError("Signed real frequencies/modes required; do not discard imaginary components")
    freq, modes = np.asarray(wavenumbers, float), np.asarray(modes, float)
    expected = 3 * atom_count - 6
    if freq.shape != (expected,) or modes.shape != (expected, atom_count, 3):
        raise ValueError("Incomplete/nonlinear molecular vibrational mode count")
    if not np.isfinite(freq).all() or not np.isfinite(modes).all():
        raise ValueError("Nonfinite frequency or normal mode")
    if np.any(np.linalg.norm(modes.reshape(expected, -1), axis=1) == 0):
        raise ValueError("Zero normal-mode vectors are not valid vibration evidence")
    return {"normal_mode_count": expected, "expected_normal_mode_count": expected,
            "imaginary_frequency_count": int(np.sum(freq < 0)),
            "frequencies_cm1": freq.tolist(), "frequencies_thz": (freq * CM1_TO_THZ).tolist(),
            "IR_intensities_status": "NOT_COMPUTED",
            "normal_modes_complete": True, "all_frequencies_real_positive": bool(np.all(freq > 0))}


def prepare_hessian_for_analysis(hessian, atom_count):
    raw = np.asarray(hessian, float)
    if raw.shape != (atom_count, atom_count, 3, 3) or not np.isfinite(raw).all():
        raise ValueError("Incomplete or nonfinite raw Hessian")
    transposed = raw.transpose(1, 0, 3, 2)
    asymmetry = float(np.max(np.abs(raw - transposed)))
    if asymmetry > 1e-5:
        raise ValueError("Raw Hessian symmetry check failed; do not hide asymmetry")
    return (raw + transposed) * .5, asymmetry


def frequency_minimum_checks(info):
    passed = bool(info["normal_modes_complete"] and info["all_frequencies_real_positive"]
                  and info["normal_mode_count"] == info["expected_normal_mode_count"]
                  and info["imaginary_frequency_count"] == 0)
    return {"stationary_minimum_verified": passed, "frequency_validation_passed": passed}



def direct_integral_policy(mf):
    """Require the reviewed no-DF, no-global-four-center-ERI execution policy."""
    if getattr(mf, "with_df", None) is not None:
        raise RuntimeError("Direct RKS control must not carry a density-fitting object")
    if getattr(mf, "_eri", None) is not None:
        raise RuntimeError("Global four-center ERI cache is forbidden in this control")
    if getattr(mf.mol, "incore_anyway", None) is not False or mf.direct_scf is not True:
        raise RuntimeError("Screened direct SCF and incore_anyway=False are required")
    if mf.direct_scf_tol != 1e-13:
        raise RuntimeError("Direct SCF screening tolerance changed from reviewed setting")
    return {"storage": "SCREENED_DIRECT_NO_GLOBAL_ERI", "density_fitting": False,
            "global_four_center_ERI_cached": False, "array_saved_to_output": False}


def screened_direct_rks_type(parent_type, direct_get_jk):
    """Bypass RHF's small-molecule incore heuristic, including for water tests.

    Delegate precisely to the normal SCF screened-direct contraction. This is not
    a new electronic-structure approximation and does not loosen screening.
    """
    class ScreenedDirectRKS(parent_type):
        def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
            direct_integral_policy(self)
            return direct_get_jk(self, mol=mol, dm=dm, hermi=hermi,
                                 with_j=with_j, with_k=with_k, omega=omega)
    return ScreenedDirectRKS


def direct_gradient_type(parent_type, emit, memory_mb=None):
    """State/timing wrapper calling the unchanged standard non-DF RKS kernel."""
    class ScreenedDirectRKSGradients(parent_type):
        def reset(self, mol=None):
            emit("gradient_scanner_geometry_reset", {})
            return super().reset(mol)

        def kernel(self, *args, **kwargs):
            if not self.base.converged:
                raise RuntimeError("Unconverged SCF cannot enter nuclear gradient")
            policy = direct_integral_policy(self.base)
            emit("direct_gradient_started", {"scf_converged": True,
                "SCF_energy_hartree": float(self.base.e_tot), "direct_integral_policy": policy,
                "current_rss_mb_before_gradient": float(memory_mb()) if memory_mb is not None else None})
            cpu, wall = time.process_time(), time.perf_counter()
            gradient = super().kernel(*args, **kwargs)
            values = np.asarray(gradient)
            if values.shape != (self.mol.natm, 3) or not np.isfinite(values).all():
                raise RuntimeError("Incomplete or nonfinite direct RKS nuclear gradient")
            emit("direct_gradient_kernel_complete", {
                "gradient_cpu_seconds": time.process_time() - cpu,
                "gradient_wall_seconds": time.perf_counter() - wall,
                "direct_integral_policy": direct_integral_policy(self.base)})
            return gradient
    return ScreenedDirectRKSGradients


def worker_settings(args):
    return {"xc": "PBE", "basis": "def2-svp", "grid_level": args.grid_level,
            "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "density_fitting": False,
            "auxbasis": None, "hessian_auxbasis_response": None,
            "scf_conv_tol_hartree": 1e-8, "scf_max_cycle": 80, "threads": args.threads,
            "max_memory_mb": 8000, "density_fit_backend": "SCREENED_DIRECT_RKS_V4",
            "direct_scf": True, "direct_scf_tol": 1e-13,
            "global_four_center_ERI_cache_allowed": False,
            "charge": 0, "spin_2S": 0, "multiplicity": 1,
            "optimizer_maxsteps": args.maxsteps, "optimizer_thresholds": THRESHOLDS}


def precise_input_coordinates(xyz_coordinates, npy_path=None, expected_sha=None):
    if npy_path is None:
        if expected_sha:
            raise ValueError("Coordinate SHA supplied without a coordinate NPY")
        return np.asarray(xyz_coordinates, float)
    if not expected_sha or sha(npy_path) != expected_sha:
        raise ValueError("Full precision coordinate NPY requires its expected SHA")
    values = np.load(npy_path, allow_pickle=False)
    if values.shape != np.asarray(xyz_coordinates).shape or not np.isfinite(values).all():
        raise ValueError("Full precision coordinates have invalid shape or values")
    if not np.allclose(values, xyz_coordinates, atol=5e-12, rtol=0):
        raise ValueError("Full precision coordinates differ from provenance-bound XYZ")
    return values


def atomic_npy(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def install_sites():
    for relative in (".runtime/quantum-upgrade-site", ".runtime/quantum-pilot-site"):
        sys.path.insert(0, str(ROOT / relative))


def execute(args):
    install_sites()
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.grad.rks import Gradients as RKSGradients
    from pyscf.scf.hf import SCF as DirectSCF
    from pyscf.geomopt import geometric_solver
    from pyscf.hessian import thermo

    started = utc()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new phase directory; preserve prior calculations")
    out.mkdir(parents=True)
    identity = "water" if args.phase == "water-optfreq" else "spinosin"
    symbols, coordinates = read_xyz(args.input_xyz, FORMULAS[identity])
    coordinates = precise_input_coordinates(coordinates, args.input_coordinates_npy,
                                            args.expected_coordinates_sha256)
    provenance = None
    if identity == "spinosin":
        if args.provenance_json is None:
            raise ValueError("Reviewed Spinosin provenance JSON required")
        provenance = json.loads(args.provenance_json.read_text())
        if (provenance.get("identity_verified") is not True
                or provenance.get("stereochemistry_verified") is not True
                or provenance.get("geometry_origin") not in ("documented_conformer_generation", "database_3d")
                or not provenance.get("source_url")
                or provenance.get("input_xyz_sha256") != sha(args.input_xyz)):
            raise ValueError("Unverified identity/stereochemistry or mismatched source geometry")
    if args.expected_input_sha256 and sha(args.input_xyz) != args.expected_input_sha256:
        raise ValueError("Input geometry SHA mismatch")
    settings = worker_settings(args)
    lib.num_threads(args.threads)
    log_handle = (out / "pyscf.log").open("w", buffering=1)
    state = {"status": "PARTIAL", "phase": args.phase, "identity": identity,
             "started_at_utc": started, "settings": settings, "input_xyz_sha256": sha(args.input_xyz),
             "input_xyz_source": str(args.input_xyz), "input_provenance": provenance,
             "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pyscf": pyscf.__version__,
                          "geometric": importlib.metadata.version("geometric")},
             "script_sha256": sha(__file__), "geometry_optimized": False,
             "actual_pyscf_threads": int(lib.num_threads()),
             "frequencies_status": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED",
             "optimizer_converged": False, "scf_converged": False}
    if args.input_coordinates_npy:
        state.update(input_coordinates_angstrom_npy_source=str(args.input_coordinates_npy),
                     input_coordinates_angstrom_npy_sha256=sha(args.input_coordinates_npy))
    write(out / "input.json", state)
    shutil.copy2(args.input_xyz, out / "input.xyz")
    atomic_npy(out / "input_coordinates_angstrom.npy", coordinates)
    checkpoint = out / "calculation.chk"
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise ValueError("Requested checkpoint is missing")
        shutil.copy2(args.checkpoint, checkpoint)
        state["initial_checkpoint_sha256"] = sha(args.checkpoint)

    def progress(event, **kwargs):
        record = {"utc": utc(), "event": event, "process_cpu_seconds": time.process_time() - cpu_start,
                  "wall_seconds": time.perf_counter() - wall_start, **kwargs}
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        write(out / "progress.json", {**state, **record})

    def molecule(coords):
        mol = gto.Mole()
        mol.stdout = log_handle
        mol.build(atom=list(zip(symbols, np.asarray(coords))), unit="Angstrom", basis=settings["basis"],
                         charge=0, spin=0, symmetry=False, max_memory=8000, verbose=4)
        mol.incore_anyway = False
        return mol

    def scf_method(mol):
        state["scf_converged"] = False
        Type = screened_direct_rks_type(dft.rks.RKS, DirectSCF.get_jk)
        mf = Type(mol)
        mf.xc = settings["xc"]
        mf.max_memory = settings["max_memory_mb"]
        mf.direct_scf = settings["direct_scf"]
        mf.direct_scf_tol = settings["direct_scf_tol"]
        mf._eri = None
        mf.mol.incore_anyway = False
        direct_integral_policy(mf)
        mf.grids.level = settings["grid_level"]
        mf.conv_tol = settings["scf_conv_tol_hartree"]
        mf.max_cycle = settings["scf_max_cycle"]
        mf.chkfile = str(checkpoint)
        if checkpoint.exists():
            mf.init_guess = "chkfile"
        def scf_callback(env):
            state["scf_converged"] = False
            progress("scf_cycle", cycle=int(env.get("cycle", -1)),
                     latest_scf_iteration_energy_hartree=float(env["e_tot"]), scf_converged=False)
        mf.callback = scf_callback
        return mf

    def gradient_event(event, details):
        if event == "gradient_scanner_geometry_reset":
            state["scf_converged"] = False
        elif event == "direct_gradient_started":
            state.update(scf_converged=details["scf_converged"],
                         latest_completed_SCF_energy_hartree=details["SCF_energy_hartree"],
                         direct_integral_policy=details["direct_integral_policy"])
        elif event == "direct_gradient_kernel_complete":
            state["last_gradient_timing"] = details
        progress(event, **details)

    GradientClass = direct_gradient_type(RKSGradients, gradient_event,
                                         memory_mb=lambda: lib.current_memory()[0])

    def gradient_method(mf):
        gradient = GradientClass(mf)
        gradient.grid_response = False
        return gradient

    def final_scf_gradient(mol):
        mf = scf_method(mol)
        energy = float(mf.kernel())
        state.update(scf_converged=bool(mf.converged), energy_hartree=energy,
                     atoms=mol.natm, electrons=mol.nelectron, basis_functions=mol.nao_nr())
        if not mf.converged or not np.isfinite(energy):
            raise RuntimeError("SCF not converged; no derivative success claimed")
        gradient = np.asarray(gradient_method(mf).kernel())
        state["direct_integral_policy"] = direct_integral_policy(mf)
        state.update(stationary_checks(gradient))
        np.save(out / "gradient_hartree_per_bohr.npy", gradient)
        progress("scf_gradient_complete", **{k: state[k] for k in ("energy_hartree", "scf_converged",
                  "atom_rms_hartree_per_bohr", "atom_max_hartree_per_bohr")})
        return mf, gradient

    def normal_modes(mf, mol):
        if not state.get("geometry_optimized") or not state.get("final_gradient_thresholds_passed"):
            raise ValueError("Nonstationary/unoptimized structure cannot enter Hessian stage")
        atomic_npy(out / "normal_mode_geometry_angstrom.npy", mol.atom_coords(unit="Angstrom"))
        atomic_npy(out / "normal_mode_geometry_bohr.npy", mol.atom_coords(unit="Bohr"))
        (out / "normal_mode_geometry.xyz").write_text(xyz_text(symbols, mol.atom_coords(unit="Angstrom"),
                                                             "Actual full-precision analysis geometry is saved in NPY"))
        state["normal_mode_geometry_angstrom_npy_sha256"] = sha(out / "normal_mode_geometry_angstrom.npy")
        progress("hessian_started")
        direct_integral_policy(mf)
        hobj = mf.Hessian()
        if type(hobj).__module__ != "pyscf.hessian.rks":
            raise RuntimeError("Reviewed standard non-DF RKS Hessian is required")
        state.update(hessian_backend=type(hobj).__module__, hessian_density_fitting=False)
        hessian = np.asarray(hobj.kernel())
        direct_integral_policy(mf)
        np.save(out / "hessian_hartree_per_bohr2.npy", hessian)
        used_hessian, asymmetry = prepare_hessian_for_analysis(hessian, mol.natm)
        np.save(out / "hessian_used_for_analysis.npy", used_hessian)
        analysis = thermo.harmonic_analysis(mol, used_hessian, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
        info = frequencies_payload(analysis["freq_wavenumber"], analysis["norm_mode"], mol.natm)
        np.save(out / "normal_modes.npy", analysis["norm_mode"])
        np.save(out / "reduced_mass_amu.npy", analysis["reduced_mass"])
        with (out / "frequencies.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["mode", "frequency_cm1", "frequency_thz", "imaginary", "IR_intensity_status"])
            for i, value in enumerate(info["frequencies_cm1"], 1):
                writer.writerow([i, value, value * CM1_TO_THZ, value < 0, "NOT_COMPUTED"])
        state.update(info, frequencies_status="DONE", hessian_max_symmetry_difference=asymmetry,
                     raw_hessian_sha256=sha(out / "hessian_hartree_per_bohr2.npy"),
                     analysis_hessian_sha256=sha(out / "hessian_used_for_analysis.npy"),
                     harmonic_analysis_excludes_translations_and_rotations=True,
                     mass_convention="PySCF default isotope-average atomic masses",
                     normal_mode_hessian_symmetrized=True,
                     normal_mode_hessian_symmetrization="(raw + raw.transpose(1,0,3,2)) / 2 after raw asymmetry<=1e-5",
                     **frequency_minimum_checks(info))
        progress("frequencies_complete", mode_count=info["normal_mode_count"],
                 imaginary_modes=info["imaginary_frequency_count"])

    def stop_handler(signum, _frame):
        raise BudgetStop("Worker interrupted by resource supervisor signal " + str(signum))

    old_handlers = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGTERM, signal.SIGXCPU)}
    try:
        mol = molecule(coordinates)
        state.update(atoms=mol.natm, electrons=mol.nelectron, basis_functions=mol.nao_nr())
        progress("worker_started", basis_functions=mol.nao_nr())
        if args.phase == "spinosin-frequency":
            if args.optimization_result is None:
                raise ValueError("Frequency phase requires actual optimization result JSON")
            optimized = json.loads(args.optimization_result.read_text())
            require_optimized_result(optimized, args.input_xyz, settings, args.input_coordinates_npy)
            state.update(optimizer_converged=True, geometry_optimized=True,
                         optimization_result_sha256=sha(args.optimization_result))
            mf, _ = final_scf_gradient(mol)
            if not state["final_gradient_thresholds_passed"]:
                raise RuntimeError("Fresh same-method gradient no longer satisfies stationarity")
            normal_modes(mf, mol)
        elif args.phase == "spinosin-scf-gradient":
            final_scf_gradient(mol)
        else:
            mf = scf_method(mol)
            (out / "geometry_steps").mkdir()

            def geometry_callback(env):
                current = env["mol"]
                cycle = int(env["self"].cycle)
                gradient = np.asarray(env["gradients"])
                text = xyz_text(symbols, current.atom_coords(unit="Angstrom"), f"PySCF geometry cycle {cycle}; not convergence certificate")
                step = out / "geometry_steps" / f"step_{cycle:04d}"
                step.with_suffix(".xyz").write_text(text)
                atomic_npy(step.with_suffix(".coordinates_angstrom.npy"), current.atom_coords(unit="Angstrom"))
                atomic_npy(step.with_suffix(".coordinates_bohr.npy"), current.atom_coords(unit="Bohr"))
                np.save(step.with_suffix(".gradient.npy"), gradient)
                info = {"cycle": cycle, "energy_hartree": float(env["energy"]),
                        "scf_converged": bool(env["g_scanner"].converged),
                        "coordinates_angstrom_npy_sha256": sha(step.with_suffix(".coordinates_angstrom.npy")),
                        "coordinates_bohr_npy_sha256": sha(step.with_suffix(".coordinates_bohr.npy")),
                        "gradient_npy_sha256": sha(step.with_suffix(".gradient.npy")),
                        **stationary_checks(gradient)}
                write(step.with_suffix(".json"), info)
                temporary = out / "last_geometry.xyz.writing"
                temporary.write_text(text)
                os.replace(temporary, out / "last_geometry.xyz")
                state.update(last_geometry_cycle=cycle, last_geometry_energy_hartree=info["energy_hartree"],
                             energy_hartree=info["energy_hartree"], scf_converged=info["scf_converged"],
                             last_geometry_gradient_checks=stationary_checks(gradient),
                             direct_integral_policy=direct_integral_policy(env["g_scanner"].base))
                progress("geometry_cycle", **info)

            progress("geometry_optimization_started")
            converged, optimized_mol = geometric_solver.kernel(gradient_method(mf), maxsteps=args.maxsteps,
                assert_convergence=True, callback=geometry_callback, **THRESHOLDS)
            state["optimizer_converged"] = bool(converged)
            optimized_path = out / "optimized.xyz"
            optimized_path.write_text(xyz_text(symbols, optimized_mol.atom_coords(unit="Angstrom"),
                                               f"Optimizer returned converged={bool(converged)}; inspect result.json"))
            state["optimized_xyz_sha256"] = sha(optimized_path)
            atomic_npy(out / "optimized_coordinates_angstrom.npy", optimized_mol.atom_coords(unit="Angstrom"))
            state["optimized_coordinates_angstrom_npy_sha256"] = sha(out / "optimized_coordinates_angstrom.npy")
            # The direct backend must never own a full four-center ERI cache.
            # Keep its MO checkpoint, then construct a fresh same-method SCF.
            direct_integral_policy(mf)
            del mf
            gc.collect()
            mf, _ = final_scf_gradient(optimized_mol)
            state["geometry_optimized"] = bool(converged and state["final_gradient_thresholds_passed"])
            if not state["geometry_optimized"]:
                state.update(status="PARTIAL", stop_reason="Optimizer/fresh gradient convergence conditions not all met")
            elif args.phase == "water-optfreq":
                normal_modes(mf, optimized_mol)
        if args.phase == "spinosin-scf-gradient" or state.get("geometry_optimized"):
            state["status"] = "DONE"
        if args.phase in ("water-optfreq", "spinosin-frequency") and state.get("frequencies_status") != "DONE":
            state["status"] = "PARTIAL"
        elif args.phase in ("water-optfreq", "spinosin-frequency") and not state.get("frequency_validation_passed"):
            state.update(status="PARTIAL", stop_reason="Frequencies computed but nonpositive modes prevent a verified minimum")
    except (BudgetStop, KeyboardInterrupt) as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), interrupted=True)
    except Exception as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), exception_type=type(exc).__name__)
        import traceback
        (out / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        state.update(completed_at_utc=utc(), wall_seconds=time.perf_counter() - wall_start,
                     process_cpu_seconds=time.process_time() - cpu_start,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     checkpoint_present=checkpoint.exists(),
                     checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None)
        write(out / "result.json", state)
        log_handle.flush()
        log_handle.close()
    print(json.dumps({k: state.get(k) for k in ("status", "phase", "energy_hartree", "geometry_optimized", "frequencies_status", "stop_reason")}))
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("water-optfreq", "spinosin-scf-gradient", "spinosin-optimize", "spinosin-frequency"))
    p.add_argument("--input-xyz", required=True, type=Path)
    p.add_argument("--input-coordinates-npy", type=Path, help="Optional SHA-bound Angstrom coordinates matching XYZ")
    p.add_argument("--expected-coordinates-sha256")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--provenance-json", type=Path)
    p.add_argument("--expected-input-sha256")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--optimization-result", type=Path)
    p.add_argument("--grid-level", type=int, default=1, choices=(1, 2, 3))
    p.add_argument("--maxsteps", type=int, default=35)
    p.add_argument("--threads", type=int, default=1, choices=(1, 2, 3, 4))
    a = p.parse_args()
    if a.maxsteps < 1 or a.maxsteps > 100:
        p.error("Bounded optimizer maxsteps must be1..100")
    result = execute(a)
    sys.exit(0 if result["status"] == "DONE" else 2)


if __name__ == "__main__":
    main()
