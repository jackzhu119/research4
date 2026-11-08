"""Explicitly sourced ethanol-only Opt/Hessian engineering validation.

Root alone executes this program through the cumulative budget runner. Importing
it performs no quantum work and creates no default molecular geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/pyscf_spinosin_bounded_hessian_worker_20261031.py"
IDENTITY = "validation_ethanol"
COMPOSITION = {"C": 2, "H": 6, "O": 1}
INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
TOLERANCES = {"energy_hartree": 1e-8, "gradient_hartree_per_bohr": 1e-8,
              "raw_hessian_hartree_per_bohr2": 1e-6, "frequencies_cm1": 1e-2}
CASES = (("standard_full", None), ("response_block4", 4), ("response_block8", 8))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_source(xyz_path, provenance_path, expected_sha, coordinates_path=None):
    """Require an actually supplied, verified9-atom ethanol source; never guess."""
    xyz_path, provenance_path = Path(xyz_path), Path(provenance_path)
    if not expected_sha or sha(xyz_path) != expected_sha:
        raise ValueError("Explicit expected XYZ SHA must match supplied structure")
    provenance = json.loads(provenance_path.read_text())
    expected = {"identity": IDENTITY, "validation_only": True, "identity_verified": True,
                "stereochemistry_verified": True, "molecule": "ethanol", "formula": "C2H6O",
                "atom_count": 9, "electron_count": 26, "charge": 0, "multiplicity": 1,
                "composition": COMPOSITION, "smiles": "CCO", "inchikey": INCHIKEY}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Root-supplied provenance is not the verified ethanol validation model")
    if (provenance.get("input_xyz_sha256") != expected_sha or not provenance.get("source_url")
            or provenance.get("coordinate_units") != "angstrom"):
        raise ValueError("Trusted source URL, coordinate units and provenance hash required")
    lines = xyz_path.read_text().splitlines(); rows = [line.split() for line in lines[2:] if line.strip()]
    if not lines or int(lines[0]) != 9 or len(rows) != 9 or any(len(row) != 4 for row in rows):
        raise ValueError("Exactly9 explicitly supplied XYZ atom rows required")
    symbols = [row[0] for row in rows]; coords = np.asarray([[float(v) for v in row[1:]] for row in rows])
    if Counter(symbols) != Counter(COMPOSITION) or not np.isfinite(coords).all():
        raise ValueError("Ethanol composition or finite coordinates mismatch")
    if coordinates_path is not None:
        coordinates_path = Path(coordinates_path)
        if sha(coordinates_path) != provenance.get("coordinates_sha256"):
            raise ValueError("Supplied full-precision source coordinates differ from provenance")
        precise = np.load(coordinates_path, allow_pickle=False)
        if precise.shape != (9, 3) or not np.isfinite(precise).all() or not np.allclose(precise, coords, atol=1e-11, rtol=0):
            raise ValueError("Full-precision source geometry differs from supplied XYZ")
        coords = precise
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    if np.min(distances[np.triu_indices(9, 1)]) < .25 or np.linalg.matrix_rank(coords - coords.mean(0), tol=1e-10) < 2:
        raise ValueError("Overlapping or linear coordinates cannot use the nonlinear21-mode convention")
    return symbols, coords, provenance


def compare_case(reference, candidate):
    """Compare every atom-pair Hessian entry, not only within response blocks."""
    arrays = {}
    for key, shape in [("gradient", (9, 3)), ("raw_hessian", (9, 9, 3, 3)),
                       ("analysis_hessian", (9, 9, 3, 3)), ("frequencies_cm1", (21,))]:
        first, second = np.asarray(reference[key]), np.asarray(candidate[key])
        if first.shape != shape or second.shape != shape or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("Incomplete/nonfinite full-system comparison array: " + key)
        arrays[key] = float(np.max(np.abs(second - first)))
    energy_difference = abs(float(candidate["energy_hartree"]) - float(reference["energy_hartree"]))
    if not np.isfinite(energy_difference):
        raise ValueError("Finite comparison energy required")
    checks = {"energy": energy_difference <= TOLERANCES["energy_hartree"],
              "gradient": arrays["gradient"] <= TOLERANCES["gradient_hartree_per_bohr"],
              "full_raw_hessian": arrays["raw_hessian"] <= TOLERANCES["raw_hessian_hartree_per_bohr2"],
              "all21_frequencies": arrays["frequencies_cm1"] <= TOLERANCES["frequencies_cm1"]}
    return {"max_abs_energy_difference_hartree": energy_difference,
            "max_abs_gradient_difference_hartree_per_bohr": arrays["gradient"],
            "max_abs_raw_hessian_difference_hartree_per_bohr2": arrays["raw_hessian"],
            "max_abs_analysis_hessian_difference_hartree_per_bohr2": arrays["analysis_hessian"],
            "max_abs_frequency_difference_cm1": arrays["frequencies_cm1"],
            "all_atom_pair_cross_couplings_compared": True, "checks": checks,
            "numerical_equivalence_passed": bool(all(checks.values())),
            "candidate_imaginary_frequency_count": int(np.sum(np.asarray(candidate["frequencies_cm1"]) < 0)),
            "candidate_all21_frequencies_positive": bool(np.all(np.asarray(candidate["frequencies_cm1"]) > 0))}


def compare_same_state(mf, evaluate, fingerprint, persist):
    """Injectable orchestration: same MF object, geometry and orbital gauge."""
    original = fingerprint(mf); reference = None; comparisons = []
    for label, block in CASES:
        if fingerprint(mf) != original:
            raise RuntimeError("MF geometry/orbitals changed before engineering case")
        actual = evaluate(mf, label, block)
        if fingerprint(mf) != original:
            raise RuntimeError("MF geometry/orbitals changed during engineering case")
        if reference is None:
            reference = actual
        difference = compare_case(reference, actual)
        record = {"case": label, "response_block_atoms": block, **difference}
        persist(label, actual, record)
        comparisons.append(record)
    return comparisons


def import_engineering_worker():
    spec = importlib.util.spec_from_file_location("bounded_hessian_engineering_v7", WORKER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def execute(args):
    symbols, coordinates, provenance = verify_source(args.input_xyz, args.provenance_json,
        args.expected_input_sha256, args.input_coordinates_npy)
    if not 1 <= args.threads <= 4 or not 1 <= args.maxsteps <= 100:
        raise ValueError("Bounded1..4threads and1..100optimizer steps required")
    out = args.output.resolve()
    if out.exists(): raise FileExistsError("Preserve earlier validation outputs")
    out.mkdir(parents=True)
    engine = import_engineering_worker(); engine.install_sites()
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.scf import _vhf
    from pyscf.scf.hf import SCF as DirectSCF
    from pyscf.grad.rks import Gradients as RKSGradients
    from pyscf.geomopt import geometric_solver
    from pyscf.hessian import thermo, rhf as rhf_hessian
    from pyscf.lib import linalg_helper
    lib.num_threads(args.threads)
    settings = engine.worker_settings(SimpleNamespace(grid_level=1, threads=args.threads,
        maxsteps=args.maxsteps, hessian_block_atoms=4, native_direct_driver_threads=1))
    started = utc(); wall, cpu = time.perf_counter(), time.process_time()
    state = {"identity": IDENTITY, "validation_only": True, "status": "PARTIAL", "started_at_utc": started,
        "atoms": 9, "electrons": 26, "settings": settings, "script_sha256": sha(__file__),
        "engineering_helper_source_sha256": sha(WORKER), "input_xyz_sha256": sha(args.input_xyz),
        "provenance_sha256": sha(args.provenance_json), "source_provenance": provenance,
        "versions": {"python": sys.version.split()[0], "pyscf": pyscf.__version__, "numpy": np.__version__,
                     "geometric": importlib.metadata.version("geometric")},
        "tolerances_fixed_before_results": TOLERANCES, "geometry_optimized": False,
        "optimizer_converged": False, "scf_converged": False,
        "frequencies_status": "VALIDATION_ONLY_NOT_HERBAL_THEORY", "IR_intensities_status": "NOT_COMPUTED",
        "numerical_equivalence_passed": False, "stationary_minimum_verified": False,
        "not_a_plant_or_acid_jujube_result": True, "new_Spinosin_frequencies": 0}
    write(out / "input.json", state)
    log_handle = (out / "pyscf.log").open("w", buffering=1)
    checkpoint = out / "calculation.chk"
    def progress(event, **details):
        record = {"utc": utc(), "event": event, "process_cpu_seconds": time.process_time()-cpu,
                  "wall_seconds": time.perf_counter()-wall, **details}
        with (out / "events.jsonl").open("a") as handle: handle.write(json.dumps(record, allow_nan=False) + "\n")
        write(out / "progress.json", {**state, **record})
    Type = engine.screened_direct_rks_type(dft.rks.RKS, DirectSCF.get_jk)
    Gradient = engine.direct_gradient_type(RKSGradients, lambda event, details: progress(event, **details),
                                           memory_mb=lambda: lib.current_memory()[0])
    def scf_method(mol):
        method = Type(mol); method.xc = "PBE"; method.max_memory = 8000
        method.direct_scf = True; method.direct_scf_tol = 1e-13
        method.conv_tol = 1e-8; method.max_cycle = 80; method.conv_tol_cpscf = 1e-8
        method.grids.level = 1; method._eri = None; mol.incore_anyway = False
        method.chkfile = str(checkpoint)
        if checkpoint.exists(): method.init_guess = "chkfile"
        engine.direct_integral_policy(method)
        return method
    def gradient_method(method):
        obj = Gradient(method); obj.grid_response = False; return obj
    def fingerprint(method):
        return {"geometry": engine.array_fingerprint(method.mol.atom_coords()),
                "MO_coeff": engine.array_fingerprint(method.mo_coeff),
                "MO_energy": engine.array_fingerprint(method.mo_energy),
                "MO_occupancy": engine.array_fingerprint(method.mo_occ), "energy_hartree": float(method.e_tot)}
    def stop(signum, _frame): raise engine.BudgetStop("Validation resource signal " + str(signum))
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGXCPU)}
    driver_guard = engine.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
        native_threads=1, emit=lambda details: progress("native_direct_driver_entered", native_control=details))
    try:
        with driver_guard:
            mol = gto.Mole(); mol.stdout = log_handle
            mol.build(atom=list(zip(symbols, coordinates)), unit="Angstrom", basis="def2-svp",
                      charge=0, spin=0, symmetry=False, verbose=4, max_memory=8000)
            mol.incore_anyway = False; method = scf_method(mol)
            (out / "geometry_steps").mkdir()
            def callback(env):
                cycle = int(env["self"].cycle); prefix = out / "geometry_steps" / f"step_{cycle:04d}"
                gradient = np.asarray(env["gradients"]); current = env["mol"]
                np.save(prefix.with_suffix(".gradient.npy"), gradient)
                np.save(prefix.with_suffix(".coordinates_angstrom.npy"), current.atom_coords(unit="Angstrom"))
                write(prefix.with_suffix(".json"), {"cycle": cycle, "energy_hartree": float(env["energy"]),
                    "scf_converged": bool(env["g_scanner"].converged), **engine.stationary_checks(gradient)})
                progress("validation_geometry_cycle", cycle=cycle, energy_hartree=float(env["energy"]))
            converged, optimized = geometric_solver.kernel(gradient_method(method), maxsteps=args.maxsteps,
                assert_convergence=True, callback=callback, **engine.THRESHOLDS)
            state["optimizer_converged"] = bool(converged)
            (out / "optimized.xyz").write_text(engine.xyz_text(symbols, optimized.atom_coords(unit="Angstrom"),
                                                              "Ethanol validation model, not a plant spectrum"))
            np.save(out / "optimized_coordinates_angstrom.npy", optimized.atom_coords(unit="Angstrom"))
            method = scf_method(optimized); energy = float(method.kernel())
            state.update(scf_converged=bool(method.converged), energy_hartree=energy)
            if not method.converged or not np.isfinite(energy): raise RuntimeError("Fresh validation SCF not converged")
            gradient = np.asarray(gradient_method(method).kernel()); np.save(out / "gradient_hartree_per_bohr.npy", gradient)
            state.update(engine.stationary_checks(gradient))
            state["geometry_optimized"] = bool(converged and state["final_gradient_thresholds_passed"])
            if not state["geometry_optimized"]: raise RuntimeError("Validation Opt/fresh-gradient gate failed; no Hessian")
            progress("validation_opt_fresh_gradient_passed")
            np.save(out / "same_MF_MO_coeff.npy", method.mo_coeff)
            np.save(out / "same_MF_MO_energy.npy", method.mo_energy)
            np.save(out / "same_MF_MO_occupancy.npy", method.mo_occ)
            same_fingerprint = fingerprint(method); write(out / "same_MF_fingerprint.json", same_fingerprint)
            def evaluate(same_method, label, block):
                case_out = out / label; case_out.mkdir(); start_cpu, start_wall = time.process_time(), time.perf_counter()
                case_settings = engine.worker_settings(SimpleNamespace(grid_level=1, threads=args.threads,
                    maxsteps=args.maxsteps, hessian_block_atoms=block or 4, native_direct_driver_threads=1))
                hobj = same_method.Hessian(); hobj.max_cycle = 50; hobj.level_shift = 0
                if type(hobj).__module__ != "pyscf.hessian.rks": raise RuntimeError("Standard non-DF RKS Hessian required")
                if block is None:
                    with engine.hessian_memory_context(hobj, case_settings, emit=lambda event, meta: progress(event, case=label, metadata=meta)):
                        raw = np.asarray(hobj.kernel())
                else:
                    cache = args.hessian_cache_dir.resolve() / label
                    raw = np.asarray(engine.run_checkpointed_hessian(hobj, cache, case_out / "hessian_cache_manifest.json",
                        case_settings, block_atoms=block, versions=state["versions"],
                        native_lindep=float(inspect.signature(linalg_helper.krylov).parameters["lindep"].default),
                        response_operator_factory=rhf_hessian.gen_vind,
                        input_evidence={"validation_only": True, "identity": IDENTITY,
                            "geometry_optimized": True, "final_gradient_thresholds_passed": True,
                            "same_MF_fingerprint": same_fingerprint, "source_provenance_sha256": sha(args.provenance_json)}))
                np.save(case_out / "hessian_hartree_per_bohr2.npy", raw)
                used, asym = engine.prepare_hessian_for_analysis(raw, 9); np.save(case_out / "hessian_used_for_analysis.npy", used)
                normal = thermo.harmonic_analysis(same_method.mol, used, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
                info = engine.frequencies_payload(normal["freq_wavenumber"], normal["norm_mode"], 9)
                np.save(case_out / "normal_modes.npy", normal["norm_mode"])
                np.save(case_out / "reduced_mass_amu.npy", normal["reduced_mass"])
                actual_gradient = np.asarray(gradient_method(same_method).kernel())
                np.save(case_out / "post_Hessian_gradient_hartree_per_bohr.npy", actual_gradient)
                initial_energy_difference = abs(float(same_method.e_tot) - energy)
                initial_gradient_difference = float(np.max(np.abs(actual_gradient - gradient)))
                with (case_out / "frequencies.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle); writer.writerow(["mode", "frequency_cm1", "frequency_thz", "IR_intensity_status", "validation_only"])
                    writer.writerows((i+1, cm, cm*engine.CM1_TO_THZ, "NOT_COMPUTED", True) for i, cm in enumerate(info["frequencies_cm1"]))
                write(case_out / "result.json", {"identity": IDENTITY, "validation_only": True, "case": label,
                    "settings": case_settings, "energy_hartree": float(same_method.e_tot),
                    "same_MF_fingerprint_before": same_fingerprint,
                    "same_MF_fingerprint_after": fingerprint(same_method),
                    "normal_mode_geometry_source": str(out / "optimized_coordinates_angstrom.npy"),
                    "normal_mode_geometry_source_sha256": sha(out / "optimized_coordinates_angstrom.npy"),
                    "raw_hessian_sha256": sha(case_out / "hessian_hartree_per_bohr2.npy"),
                    "analysis_hessian_sha256": sha(case_out / "hessian_used_for_analysis.npy"),
                    "normal_modes_file_sha256": sha(case_out / "normal_modes.npy"),
                    "raw_hessian_max_asymmetry": asym, "CPU_seconds": time.process_time()-start_cpu,
                    "wall_seconds": time.perf_counter()-start_wall, **info, **engine.frequency_minimum_checks(info),
                    "max_abs_energy_difference_from_initial_fresh_SCF_hartree": initial_energy_difference,
                    "max_abs_gradient_difference_from_initial_fresh_gradient_hartree_per_bohr": initial_gradient_difference,
                    "CPHF_residual_norm": "NOT_COMPUTED", "native_solver_normal_return_is_not_measured_residual": True})
                if (initial_energy_difference > TOLERANCES["energy_hartree"]
                        or initial_gradient_difference > TOLERANCES["gradient_hartree_per_bohr"]):
                    raise RuntimeError("Hessian case changed the initial fresh SCF/gradient beyond fixed tolerance")
                return {"energy_hartree": float(same_method.e_tot), "gradient": actual_gradient,
                        "raw_hessian": raw, "analysis_hessian": used,
                        "frequencies_cm1": np.asarray(info["frequencies_cm1"])}
            def persist(label, actual, comparison):
                write(out / label / "comparison_to_standard.json", comparison)
                progress("validation_Hessian_case_complete", case=label, comparison=comparison)
            comparisons = compare_same_state(method, evaluate, fingerprint, persist)
            write(out / "all_case_comparisons.json", comparisons)
            state.update(numerical_equivalence_passed=all(row["numerical_equivalence_passed"] for row in comparisons),
                         all_cases_positive21_frequency_minimum=all(row["candidate_all21_frequencies_positive"] for row in comparisons),
                         comparisons=comparisons, same_MF_geometry_and_orbitals_unchanged=True)
            state["stationary_minimum_verified"] = bool(state["all_cases_positive21_frequency_minimum"])
            state["status"] = "DONE" if state["numerical_equivalence_passed"] else "PARTIAL"
            if not state["numerical_equivalence_passed"]: state["stop_reason"] = "Predeclared engineering numerical equivalence threshold failed"
    except (engine.BudgetStop, KeyboardInterrupt) as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), interrupted=True)
    except Exception as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), exception_type=type(exc).__name__)
        import traceback
        (out / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        for signum, previous in handlers.items(): signal.signal(signum, previous)
        state.update(completed_at_utc=utc(), wall_seconds=time.perf_counter()-wall,
                     process_cpu_seconds=time.process_time()-cpu,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     native_direct_driver_control=driver_guard.summary())
        if state["native_direct_driver_control"].get("restoration_failures"):
            state.update(status="PARTIAL", stop_reason="Native thread restoration failure")
        write(out / "result.json", state); log_handle.close()
    print(json.dumps({key: state.get(key) for key in ["identity", "status", "geometry_optimized", "numerical_equivalence_passed", "stationary_minimum_verified", "stop_reason"]}))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xyz", required=True, type=Path)
    parser.add_argument("--provenance-json", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--input-coordinates-npy", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hessian-cache-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maxsteps", type=int, default=50)
    args = parser.parse_args(); result = execute(args)
    sys.exit(0 if result["status"] == "DONE" else 2)


if __name__ == "__main__":
    main()
