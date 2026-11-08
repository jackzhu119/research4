#!/usr/bin/env python3
"""Bounded prospective moving-grid PBE numerical-Hessian experiment.

This module never edits or imports historical producers/results. A successful
numerical calculation and a certified minimum are separate statements. Units
inside the worker are Bohr, Hartree, and Hartree/Bohr.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback
import time

SCHEMA = "research4-full-moving-grid-numerical-hessian-v2"
SETTINGS = {
    "xc": "PBE", "basis": "def2-SVP", "grid_level": 3,
    "dispersion": "NONE_EXPLICIT", "charge": 0, "spin": 0,
    "density_fitting": False, "full_native_veff_every_cycle": True,
    "grid_response": True, "direct_scf_tol": 1e-15,
    "energy_tolerance_hartree": 1e-12, "orbital_gradient_tolerance": 1e-9,
    "main_orbital_gradient_trigger": 1e-10, "maximum_SCF_cycles": 100,
    "maximum_SCF_calls_per_point": 3, "max_memory_mb": 6000,
    "solver_pipeline": "NEWTON_SEED_THEN_UP_TO_TWO_STRICT_NATIVE_CERTIFICATIONS",
    "newton_seed_max_macrocycles": 30, "newton_seed_max_inner_cycles": 12,
    "newton_seed_energy_tolerance": 1e-13, "newton_seed_gradient_tolerance": 1e-11,
    "maximum_strict_certification_calls": 2,
    "newton_seed_is_certification": False,
    "optimization_max_steps": 40, "nuclear_rms_tolerance": 1e-5,
    "nuclear_max_tolerance": 3e-5,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_array(path, array):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def read_ref(ref):
    import numpy as np
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("Checkpoint hash differs: " + str(path))
    array = np.load(path, allow_pickle=False)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite checkpoint: " + str(path))
    return array


def load_optimization(path):
    path = Path(path)
    result = json.loads(path.read_text())
    if result.get("schema") != SCHEMA or result.get("source_sha256") != sha(__file__):
        raise ValueError("Optimization schema/source differs")
    if result.get("status") != "OPTIMIZED_GEOMETRY_CERTIFIED":
        raise ValueError("Optimization prerequisite is not certified")
    if result.get("settings") != SETTINGS:
        raise ValueError("Optimization method/settings differ")
    require_formula(result.get("symbols", []))
    if not result.get("optimizer_converged") or result.get("nuclear_gradient_rms", math.inf) > 1e-5 or result.get("nuclear_gradient_max", math.inf) > 3e-5:
        raise ValueError("Optimizer or fresh nuclear-gradient gate is not passed")
    zero_ref = result["fresh_zero_result"]
    if sha(zero_ref["path"]) != zero_ref["sha256"]:
        raise ValueError("Fresh-zero result hash differs")
    zero = json.loads(Path(zero_ref["path"]).read_text())
    if zero.get("status") != "ELECTRONIC_POINT_CERTIFIED" or not zero.get("electronic_certification_passed") or not zero.get("grid_response"):
        raise ValueError("Fresh moving-grid electronic zero is not certified")
    if any(zero.get(key) != result.get(key) for key in ("coordinates_bohr", "density", "symbols")):
        raise ValueError("Fresh-zero and optimization state differ")
    coords = read_ref(result["coordinates_bohr"])
    density = read_ref(result["density"])
    if coords.shape != (17, 3):
        raise ValueError("Expected all 17 protocatechuic-acid atoms")
    return result, coords, density


def require_formula(symbols):
    if Counter(symbols) != Counter({"C": 7, "H": 6, "O": 4}):
        raise ValueError("Expected neutral singlet C7H6O4, with all 17 atoms")


def new_molecule(symbols, coordinates, unit="Bohr", stdout=None):
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [(symbol, tuple(float(x) for x in row)) for symbol, row in zip(symbols, coordinates)]
    mol.unit = unit
    mol.charge, mol.spin = 0, 0
    mol.basis = SETTINGS["basis"]
    mol.verbose = 4
    mol.max_memory = SETTINGS["max_memory_mb"]
    if stdout is not None:
        mol.stdout = stdout
    mol.build()
    return mol


def molecule_from_xyz(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[0].strip())
    rows = [line.split() for line in lines[2:] if line.strip()]
    if len(rows) != count or any(len(row) != 4 for row in rows):
        raise ValueError("Invalid explicit XYZ molecule")
    symbols = [row[0] for row in rows]
    require_formula(symbols)
    return new_molecule(symbols, [[float(x) for x in row[1:]] for row in rows], "Angstrom")


def make_scf(mol, checkpoint):
    """Use native direct AO integration, removing incremental-J history only."""
    import numpy as np
    from pyscf import dft
    from pyscf.dft import rks

    class FullNativeRKS(dft.rks.RKS):
        def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
            return rks.get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=hermi)

    mf = FullNativeRKS(mol)
    mf.xc = SETTINGS["xc"]
    mf.grids.level = SETTINGS["grid_level"]
    mf.direct_scf = True
    mf.direct_scf_tol = SETTINGS["direct_scf_tol"]
    mf.max_memory = SETTINGS["max_memory_mb"]
    mf.max_cycle = SETTINGS["maximum_SCF_cycles"]
    mf.conv_tol = SETTINGS["energy_tolerance_hartree"]
    mf.conv_tol_grad = SETTINGS["main_orbital_gradient_trigger"]
    mf.chkfile = str(checkpoint)
    mf._strict_checks = []

    def strict_check(env):
        # This callback is called by actual main AND actual extra iterations.
        # No historical false flag is relabelled and no final OR test is used.
        gradient = np.linalg.norm(mf.get_grad(env["mo_coeff"], env["mo_occ"], env["fock"]))
        delta = float(env["e_tot"] - env["last_hf_e"])
        extra = env["conv_tol"] > SETTINGS["energy_tolerance_hartree"]
        bound = SETTINGS["orbital_gradient_tolerance"] if extra else SETTINGS["main_orbital_gradient_trigger"]
        passed = bool(abs(delta) <= SETTINGS["energy_tolerance_hartree"] and gradient <= bound)
        row = {"cycle": int(env["cycle"]) + 1, "actual_extra_cycle": bool(extra),
               "energy_hartree": float(env["e_tot"]), "previous_energy_hartree": float(env["last_hf_e"]),
               "energy_change_hartree": delta, "full_fock_orbital_gradient_norm": float(gradient),
               "orbital_bound": bound, "passed": passed}
        mf._strict_checks.append(row)
        return passed

    mf.check_convergence = strict_check
    return mf


def point_fingerprint(mol, dm0):
    import numpy as np
    payload = {"settings": SETTINGS, "source_sha256": sha(__file__),
               "pyscf_version": importlib.metadata.version("pyscf"), "symbols": [mol.atom_symbol(i) for i in range(mol.natm)],
               "coordinates_sha256": hashlib.sha256(np.asarray(mol.atom_coords(), dtype="<f8").tobytes()).hexdigest(),
               "initial_density_sha256": None if dm0 is None else hashlib.sha256(np.asarray(dm0, dtype="<f8").tobytes()).hexdigest()}
    return signature(payload), payload


def run_newton_seed(mol, out, dm0):
    """Finite Newton preconditioner; even a converged seed is not certification."""
    import numpy as np
    start = time.process_time()
    base = make_scf(mol, out / "newton-seed.chk")
    # Newton has a different convergence environment. Never inherit the
    # ordinary-SCF callback; a separate fresh ordinary driver certifies it.
    base.check_convergence = None
    seed = base.newton()
    seed.check_convergence = None
    seed._scf.check_convergence = None
    seed.max_cycle = SETTINGS["newton_seed_max_macrocycles"]
    seed.max_cycle_inner = SETTINGS["newton_seed_max_inner_cycles"]
    seed.conv_tol = SETTINGS["newton_seed_energy_tolerance"]
    seed.conv_tol_grad = SETTINGS["newton_seed_gradient_tolerance"]
    energy = float(seed.kernel(dm0=None if dm0 is None else np.array(dm0, copy=True)))
    density = np.asarray(seed.make_rdm1())
    if not math.isfinite(energy) or not np.isfinite(density).all():
        raise ValueError("Newton preconditioner returned nonfinite state")
    record = {"solver": "FULL_AO_MO_NATIVE_NEWTON_PRECONDITIONER",
              "native_seed_converged": bool(seed.converged),
              "status": "UNCERTIFIED_INITIAL_GUESS",
              "energy_hartree_NOT_CERTIFIED": energy,
              "max_macrocycles": SETTINGS["newton_seed_max_macrocycles"],
              "max_inner_cycles": SETTINGS["newton_seed_max_inner_cycles"],
              "AH_max_cycle": int(seed.ah_max_cycle),
              "density": save_array(out / "local-only/newton_seed_density.npy", density),
              "mo_coeff": save_array(out / "local-only/newton_seed_mo_coeff.npy", seed.mo_coeff),
              "mo_occ": save_array(out / "local-only/newton_seed_mo_occ.npy", seed.mo_occ),
              "mo_energy": save_array(out / "local-only/newton_seed_mo_energy.npy", seed.mo_energy),
              "cpu_seconds_process_time": time.process_time() - start,
              "custom_ordinary_convergence_callback_disabled": True}
    save_json(out / "newton_seed_result.json", record)
    return density, record


def strict_certification_call(mol, out, density, call_index):
    import numpy as np
    start = time.process_time()
    mf = make_scf(mol, out / ("strict-call-%d.chk" % call_index))
    energy = float(mf.kernel(dm0=np.array(density, copy=True)))
    if not math.isfinite(energy):
        raise ValueError("Nonfinite strict ordinary SCF energy")
    actual_density = np.array(mf.make_rdm1(), copy=True)
    hcore = mf.get_hcore()
    fresh_veff = mf.get_veff(mol, actual_density)
    fresh_fock = mf.get_fock(hcore, mf.get_ovlp(), fresh_veff, actual_density,
                             cycle=-1, diis=None, level_shift_factor=0)
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fresh_fock)))
    fresh_energy = float(mf.energy_tot(actual_density, hcore, fresh_veff))
    if not math.isfinite(residual) or not math.isfinite(fresh_energy):
        raise ValueError("Nonfinite independent full-Fock audit")
    last = mf._strict_checks[-1] if mf._strict_checks else {}
    certified = bool(mf.converged and last.get("actual_extra_cycle") and last.get("passed")
                     and residual <= SETTINGS["orbital_gradient_tolerance"]
                     and abs(fresh_energy-energy) <= SETTINGS["energy_tolerance_hartree"])
    record = {"call_index": call_index, "native_pyscf_converged": bool(mf.converged),
              "energy_hartree": energy, "strict_iteration_checks": mf._strict_checks,
              "fresh_full_fock_orbital_gradient_norm": residual,
              "fresh_energy_hartree": fresh_energy,
              "final_energy_recheck_difference_hartree": fresh_energy-energy,
              "electronic_certification_passed": certified,
              "initial_density_sha256": hashlib.sha256(np.asarray(density, dtype="<f8").tobytes()).hexdigest(),
              "actual_endpoint_density": save_array(out / "local-only" / ("strict_call_%d_density.npy" % call_index), actual_density),
              "cpu_seconds_process_time": time.process_time() - start}
    if Path(mf.chkfile).exists():
        record["rolling_checkpoint"] = {"path": str(Path(mf.chkfile).resolve()), "sha256": sha(mf.chkfile)}
    save_json(out / ("strict_call_%d_result.json" % call_index), record)
    return mf, actual_density, record


def evaluate_point(mol, output, dm0=None, nuclear_gradient=True):
    """At most one Newton seed plus two strict ordinary certification calls."""
    import numpy as np
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint, inputs = point_fingerprint(mol, dm0)
    result_path = out / "result.json"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get("input_fingerprint") != fingerprint:
            raise ValueError("Point exists with different coordinates/settings/density")
        if old.get("status") != "ELECTRONIC_POINT_CERTIFIED":
            raise ValueError("Previously failed/partial point cannot automatically retry")
        density = read_ref(old["density"])
        gradient = read_ref(old["gradient"]) if nuclear_gradient else None
        if nuclear_gradient and not old.get("grid_response"):
            raise ValueError("Saved point lacks moving-grid gradient")
        return old, density, gradient
    if (out / "started.json").exists():
        raise ValueError("Unclosed prior point requires independent audit")
    save_json(out / "started.json", {"input_fingerprint": fingerprint, "inputs": inputs, "at_utc": utc()})
    state = {"schema": SCHEMA, "status": "RUNNING", "input_fingerprint": fingerprint,
             "settings": SETTINGS, "inputs": inputs, "started_at_utc": utc(),
             "IR": "NOT_COMPUTED", "native_pyscf_converged": False,
             "actual_SCF_calls_started": 0, "certification_calls": [],
             "initial_guess_scope": "UNCONVERGED_OR_CERTIFIED_INPUT_USED_ONLY_AS_GUESS"}
    save_json(out / "progress.json", state)
    log = (out / "pyscf.log").open("a", buffering=1)
    mol = mol.copy(); mol.stdout = log
    mf = None
    try:
        state["actual_SCF_calls_started"] += 1
        save_json(out / "progress.json", state)
        density, seed_record = run_newton_seed(mol, out, dm0)
        state["newton_seed"] = seed_record
        save_json(out / "progress.json", state)
        for call_index in range(1, SETTINGS["maximum_strict_certification_calls"] + 1):
            state["actual_SCF_calls_started"] += 1
            save_json(out / "progress.json", state)
            mf, density, record = strict_certification_call(mol, out, density, call_index)
            state["certification_calls"].append(record)
            for key in ("native_pyscf_converged", "energy_hartree", "strict_iteration_checks",
                        "fresh_full_fock_orbital_gradient_norm", "fresh_energy_hartree",
                        "final_energy_recheck_difference_hartree", "electronic_certification_passed"):
                state[key] = record[key]
            state.update(density=save_array(out / "density.npy", density),
                         coordinates_bohr=save_array(out / "coordinates_bohr.npy", mol.atom_coords()),
                         symbols=[mol.atom_symbol(i) for i in range(mol.natm)])
            save_json(out / "progress.json", state)
            if record["electronic_certification_passed"]:
                break
            # Only the immediately preceding actual endpoint may seed call 2.
            # There is no third certification call or point-level retry.
        if not state.get("electronic_certification_passed"):
            raise ValueError("Both bounded ordinary SCF certification endpoints failed strict final-pair/full-Fock gates")
        gradient = None
        if nuclear_gradient:
            grad_driver = mf.nuc_grad_method(); grad_driver.grid_response = True
            gradient = np.asarray(grad_driver.kernel(), dtype=float)
            if gradient.shape != (mol.natm, 3) or not np.isfinite(gradient).all():
                raise ValueError("Incomplete/nonfinite nuclear gradient")
            state.update(grid_response=True, gradient=save_array(out / "gradient.npy", gradient),
                         nuclear_gradient_rms=float(np.sqrt(np.mean(gradient**2))),
                         nuclear_gradient_max=float(np.max(np.abs(gradient))))
        else:
            state["grid_response"] = False
        state.update(status="ELECTRONIC_POINT_CERTIFIED", completed_at_utc=utc())
        if Path(mf.chkfile).exists():
            state["rolling_checkpoint"] = {"path": str(Path(mf.chkfile).resolve()), "sha256": sha(mf.chkfile)}
        save_json(result_path, state); save_json(out / "progress.json", state)
        return state, density, gradient
    except BaseException as error:
        state.update(status="POINT_FAILED", error_type=type(error).__name__,
                     error=str(error), completed_at_utc=utc())
        save_json(result_path, state); save_json(out / "progress.json", state)
        raise
    finally:
        log.close()


def stage_controls(args):
    import numpy as np
    out = Path(args.output)
    mol = new_molecule(["O", "H", "H"], [[0., 0., 0.], [.75695, 0., .58588], [-.75695, 0., .58588]], "Angstrom")
    zero, density, gradient = evaluate_point(mol, out / "zero")
    direction = np.zeros((3, 3)); direction[1, 0] = 1.
    derivatives = []
    points = []
    for step in (.001, .0005):
        pair = []
        for sign in (1, -1):
            displaced = mol.copy()
            displaced.set_geom_(mol.atom_coords() + sign * step * direction, unit="Bohr")
            result, _, _ = evaluate_point(displaced, out / ("h%.4f-%s" % (step, "plus" if sign > 0 else "minus")), density, False)
            pair.append(result["energy_hartree"])
            points.append(result)
        derivatives.append(math.fsum((pair[0], -pair[1])) / (2 * step))
    analytic = float(np.sum(gradient * direction))
    errors = [abs(value - analytic) for value in derivatives]
    passed = max(errors) <= 1e-5 and abs(derivatives[0]-derivatives[1]) <= 1e-5
    result = base_result("CONTROLS_PASSED" if passed else "CONTROLS_FAILED")
    result.update(analytic_directional_gradient=analytic, finite_energy_derivatives=derivatives,
                  absolute_errors=errors, step_halving_difference=abs(derivatives[0]-derivatives[1]),
                  absolute_tolerance=1e-5, actual_SCF_point_count=5,
                  electronic_point_results=[{"path": str(path.resolve()), "sha256": sha(path)} for path in
                      [out / "zero" / "result.json"] + [out / ("h%.4f-%s" % (h, sign)) / "result.json"
                          for h in (.001, .0005) for sign in ("plus", "minus")]])
    return result


def stage_optimize(args):
    import numpy as np
    from pyscf.geomopt import addons, geometric_solver
    if not args.xyz:
        raise ValueError("--xyz is required")
    mol = molecule_from_xyz(args.xyz)
    output = Path(args.output)
    initial_density = None
    initial_ref = None
    if args.initial_density is not None:
        if not args.initial_density_sha256 or sha(args.initial_density) != args.initial_density_sha256:
            raise ValueError("Explicit initial-density SHA-256 is required and must match")
        if not args.initial_density_source_result:
            raise ValueError("Historical guess needs --initial-density-source-result provenance")
        source = json.loads(args.initial_density_source_result.read_text())
        source_inputs = source.get("inputs", {})
        if source.get("schema") != "research4-full-moving-grid-numerical-hessian-v1" or source_inputs.get("source_sha256") != "aecb5de96cca3f2cc39892384418499da39b70e1d44a8b9af8c51ad9105f79e3":
            raise ValueError("Initial guess must originate from the authenticated preserved v1 producer")
        if source.get("density", {}).get("sha256") != args.initial_density_sha256:
            raise ValueError("Historical endpoint density hash does not match the supplied guess")
        method_keys = ("xc", "basis", "grid_level", "charge", "spin", "density_fitting", "grid_response")
        if any(source.get("settings", {}).get(key) != SETTINGS[key] for key in method_keys):
            raise ValueError("Historical guess belongs to a different molecular model")
        if source.get("symbols") != [mol.atom_symbol(i) for i in range(mol.natm)]:
            raise ValueError("Historical guess atom order differs")
        if not np.allclose(read_ref(source["coordinates_bohr"]), mol.atom_coords(), rtol=0, atol=1e-8):
            raise ValueError("Historical guess nuclear coordinates differ")
        initial_density = np.load(args.initial_density, allow_pickle=False)
        if initial_density.shape != (mol.nao_nr(), mol.nao_nr()) or not np.isfinite(initial_density).all():
            raise ValueError("Initial unconverged density has wrong shape/nonfinite values")
        if np.max(np.abs(initial_density - initial_density.T)) > 1e-10:
            raise ValueError("Initial unconverged density is not symmetric")
        if abs(np.einsum("ij,ji->", initial_density, mol.intor_symmetric("int1e_ovlp")) - 80.) > 1e-6:
            raise ValueError("Initial unconverged density does not describe 80 electrons at input geometry")
        initial_ref = {"path": str(args.initial_density.resolve()), "sha256": args.initial_density_sha256,
                       "scope": "UNCONVERGED_HISTORICAL_GUESS_ONLY_NOT_CERTIFIED",
                       "source_result": {"path": str(args.initial_density_source_result.resolve()), "sha256": sha(args.initial_density_source_result)},
                       "source_native_pyscf_converged": source.get("native_pyscf_converged")}
    latest = {"density": initial_density, "count": 0}

    def scan(actual_mol):
        index = latest["count"]
        result, dm, gradient = evaluate_point(actual_mol, output / "optimization_steps" / ("step-%03d" % index), latest["density"])
        latest.update(density=dm, count=index+1)
        return result["energy_hartree"], gradient

    method = addons.as_pyscf_method(mol, scan)
    converged, optimized = geometric_solver.kernel(method, maxsteps=40,
        convergence_energy=1e-8, convergence_grms=1e-5, convergence_gmax=3e-5,
        convergence_drms=4e-4, convergence_dmax=1.2e-3)
    zero, density, gradient = evaluate_point(optimized, output / "fresh-final-zero", latest["density"])
    rms = float(np.sqrt(np.mean(gradient**2))); maximum = float(np.max(np.abs(gradient)))
    passed = bool(converged and rms <= 1e-5 and maximum <= 3e-5)
    result = base_result("OPTIMIZED_GEOMETRY_CERTIFIED" if passed else "OPTIMIZATION_NOT_CERTIFIED")
    result.update(optimizer_converged=bool(converged), maximum_optimizer_steps=40,
                  initial_density_guess=initial_ref,
                  actual_gradient_evaluations=latest["count"] + 1,
                  original_xyz={"path": str(Path(args.xyz).resolve()), "sha256": sha(args.xyz)},
                  symbols=zero["symbols"], coordinates_bohr=zero["coordinates_bohr"], density=zero["density"],
                  nuclear_gradient_rms=rms, nuclear_gradient_max=maximum,
                  fresh_zero_result={"path": str((output / "fresh-final-zero/result.json").resolve()), "sha256": sha(output / "fresh-final-zero/result.json")})
    return result


def assemble_hessian(gradients_plus, gradients_minus, step):
    import numpy as np
    plus = np.asarray(gradients_plus, dtype=float)
    minus = np.asarray(gradients_minus, dtype=float)
    if plus.shape != minus.shape or plus.ndim != 3 or plus.shape[0] != plus.shape[1] * 3 or plus.shape[2] != 3:
        raise ValueError("Need complete plus/minus gradient for every Cartesian coordinate")
    if not np.isfinite(plus).all() or not np.isfinite(minus).all() or not step > 0:
        raise ValueError("Invalid gradient or displacement")
    return ((plus - minus) / (2. * step)).reshape(plus.shape[0], -1).T


def stage_hessian(args):
    import numpy as np
    if not args.optimization or args.step not in (.001, .0005):
        raise ValueError("--optimization and predeclared --step .001/.0005 are required")
    optimum, coordinates, density = load_optimization(args.optimization)
    require_formula(optimum["symbols"])
    output = Path(args.output)
    gradients = {1: {}, -1: {}}
    progress = base_result("RUNNING")
    progress.update(step_bohr=args.step, optimization_sha256=sha(args.optimization),
                    expected_points=102, actual_qualified_points=0, failed_points=[],
                    maximum_attempts_per_point=1, maximum_SCF_calls_per_point=3)
    save_json(output / "progress.json", progress)
    try:
        for coordinate in range(51):
            for sign in (1, -1):
                name = "c%03d-%s" % (coordinate, "plus" if sign > 0 else "minus")
                changed = np.array(coordinates, copy=True)
                changed.flat[coordinate] += sign * args.step
                mol = new_molecule(optimum["symbols"], changed)
                try:
                    point, _, gradient = evaluate_point(mol, output / "points" / name, density)
                    gradients[sign][coordinate] = gradient
                    progress["actual_qualified_points"] += 1
                except Exception as error:
                    # Failed or interrupted prior point never earns a retry.
                    # Independent unattempted coordinates still add useful evidence.
                    progress["failed_points"].append({"point": name, "error_type": type(error).__name__, "error": str(error)})
                save_json(output / "progress.json", progress)
        if progress["actual_qualified_points"] != 102:
            progress.update(status="NUMERICAL_HESSIAN_PARTIAL", completed_at_utc=utc())
            return progress
        raw = assemble_hessian([gradients[1][i] for i in range(51)],
                               [gradients[-1][i] for i in range(51)], args.step)
        result = base_result("NUMERICAL_HESSIAN_COMPLETE")
        result.update(step_bohr=args.step, optimization_sha256=sha(args.optimization),
                      expected_points=102, actual_qualified_points=102,
                      raw_hessian=save_array(output / "raw_hessian.npy", raw),
                      coordinates_bohr=optimum["coordinates_bohr"], symbols=optimum["symbols"],
                      raw_asymmetry_max=float(np.max(np.abs(raw-raw.T))),
                      raw_asymmetry_frobenius=float(np.linalg.norm(raw-raw.T)),
                      raw_relative_asymmetry=float(np.linalg.norm(raw-raw.T) / max(np.linalg.norm(raw), 1e-300)),
                      all_points_same_zero_density_sha256=optimum["density"]["sha256"],
                      full_cartesian_dimension=51, vibrational_mode_count_expected=45)
        return result
    except BaseException as error:
        progress.update(status="NUMERICAL_HESSIAN_PARTIAL", error_type=type(error).__name__, error=str(error),
                        completed_at_utc=utc())
        save_json(output / "result.json", progress)
        raise


def vibrational_modes(hessian, coordinates, masses):
    import numpy as np
    from scipy import constants
    hessian = np.asarray(hessian, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(masses)
    if hessian.shape != (3*n, 3*n) or coordinates.shape != (n, 3) or np.any(masses <= 0):
        raise ValueError("Inconsistent Hessian, coordinates, or masses")
    center = np.sum(coordinates * masses[:, None], axis=0) / masses.sum()
    centered = coordinates - center
    rootmass = np.sqrt(masses)
    external = []
    for axis in np.eye(3):
        external.append((rootmass[:, None] * np.broadcast_to(axis, (n, 3))).ravel())
    for axis in np.eye(3):
        external.append((rootmass[:, None] * np.cross(axis, centered)).ravel())
    u, singular, _ = np.linalg.svd(np.array(external).T, full_matrices=True)
    rank = int(np.sum(singular > singular.max() * 1e-10))
    if rank != 6:
        raise ValueError("Protocol requires a nonlinear molecule with six external modes")
    projector = u[:, rank:]
    scale = np.repeat(rootmass, 3)
    weighted = .5 * (hessian + hessian.T) / scale[:, None] / scale[None, :]
    values, vectors = np.linalg.eigh(projector.T @ weighted @ projector)
    conversion = math.sqrt(constants.physical_constants["Hartree energy"][0] /
                           (constants.physical_constants["Bohr radius"][0]**2 * constants.physical_constants["atomic mass constant"][0])) / (2 * math.pi * constants.c * 100)
    frequencies = np.sign(values) * np.sqrt(np.abs(values)) * conversion
    return frequencies, values, projector @ vectors, rank


def stage_analyze(args):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from pyscf.data import elements
    if not args.hessians or len(args.hessians) != 2:
        raise ValueError("Exactly two --hessians result.json inputs are required")
    records = [json.loads(Path(path).read_text()) for path in args.hessians]
    if any(row.get("status") != "NUMERICAL_HESSIAN_COMPLETE" or row.get("settings") != SETTINGS or row.get("schema") != SCHEMA or row.get("source_sha256") != sha(__file__) for row in records):
        raise ValueError("Both full, method-matched numerical Hessians must complete")
    if {row["step_bohr"] for row in records} != {.001, .0005}:
        raise ValueError("The two predeclared distinct steps are required")
    records.sort(key=lambda row: -row["step_bohr"])
    if records[0]["optimization_sha256"] != records[1]["optimization_sha256"]:
        raise ValueError("Hessians were not evaluated at the same certified geometry")
    if any(records[0][key] != records[1][key] for key in ("coordinates_bohr", "symbols", "all_points_same_zero_density_sha256")):
        raise ValueError("Hessian geometry/atom ordering differs")
    coordinates = read_ref(records[0]["coordinates_bohr"])
    symbols = records[0]["symbols"]
    require_formula(symbols)
    masses = np.array([elements.MASSES[elements.charge(symbol)] for symbol in symbols])
    hessians = [read_ref(row["raw_hessian"]) for row in records]
    modes = [vibrational_modes(hessian, coordinates, masses) for hessian in hessians]
    overlap = np.abs(modes[0][2].T @ modes[1][2])
    coarse, fine = linear_sum_assignment(-overlap)
    matched_difference = modes[1][0][fine] - modes[0][0][coarse]
    sorted_difference = modes[1][0] - modes[0][0]
    imaginary = [int(np.sum(mode[0] < 0)) for mode in modes]
    scale = np.repeat(np.sqrt(masses), 3)
    weighted = [.5 * (matrix + matrix.T) / scale[:, None] / scale[None, :] for matrix in hessians]
    # Both mode bases span the same six-external-mode orthogonal complement.
    common_basis = modes[0][2]
    projected = [common_basis.T @ matrix @ common_basis for matrix in weighted]
    difference = projected[1] - projected[0]
    relative_step_difference = float(np.linalg.norm(difference) / max(np.linalg.norm(projected[1]), 1e-300))
    uncertainty_margin = float(np.linalg.norm(difference, ord=2))
    asymmetries = [float(np.linalg.norm(matrix-matrix.T) / max(np.linalg.norm(matrix), 1e-300)) for matrix in hessians]
    numeric_consistent = bool(max(asymmetries) <= 1e-3 and relative_step_difference <= 1e-3)
    positive_with_margin = bool(all(np.min(mode[1]) > uncertainty_margin for mode in modes))
    frequency_bound = max(5., .01 * max(float(np.max(np.abs(mode[0]))) for mode in modes))
    result = base_result("NUMERICAL_HESSIAN_ANALYZED")
    result.update(input_hessians=[{"path": str(Path(p).resolve()), "sha256": sha(p)} for p in args.hessians],
                  signed_frequencies_cm1=[mode[0].tolist() for mode in modes],
                  signed_eigenvalues_atomic_units=[mode[1].tolist() for mode in modes],
                  imaginary_counts=imaginary, projected_external_rank=6, vibrational_modes=45,
                  matching={"coarse_mode_zero_based": coarse.tolist(), "fine_mode_zero_based": fine.tolist(),
                            "absolute_mass_weighted_overlaps": overlap[coarse, fine].tolist(),
                            "signed_frequency_differences_cm1": matched_difference.tolist()},
                  sorted_frequency_difference_cm1=sorted_difference.tolist(),
                  sorted_frequency_report_bound_cm1=frequency_bound,
                  sorted_frequency_report_within_bound=bool(np.max(np.abs(sorted_difference)) <= frequency_bound),
                  raw_relative_antisymmetry=asymmetries,
                  projected_mass_weighted_hessian_relative_step_difference=relative_step_difference,
                  projected_mass_weighted_hessian_operator_difference=uncertainty_margin,
                  minimum_eigenvalue_exceeds_step_uncertainty_margin=positive_with_margin,
                  numerical_step_consistency_passed=numeric_consistent,
                  numerical_step_consistency_bounds={"relative_antisymmetry_each": 1e-3, "projected_hessian_relative_frobenius_difference": 1e-3},
                  minimum_verified=bool(numeric_consistent and positive_with_margin and imaginary == [0, 0]),
                  minimum_scope="This isolated PBE/def2-SVP grid3 model and certified geometry only; no chemical assignment",
                  original_analytic_Hessian_or_imaginary_modes_modified=False)
    for index, mode in enumerate(modes):
        result["mass_weighted_modes_%d" % index] = save_array(Path(args.output) / ("mass_weighted_modes_%d.npy" % index), mode[2])
        result["projected_mass_weighted_hessian_%d" % index] = save_array(Path(args.output) / ("projected_mass_weighted_hessian_%d.npy" % index), projected[index])
    result["common_vibrational_orthonormal_basis"] = save_array(Path(args.output) / "common_vibrational_basis.npy", common_basis)
    result["atomic_masses_amu"] = save_array(Path(args.output) / "atomic_masses_amu.npy", masses)
    return result


def base_result(status):
    return {"schema": SCHEMA, "status": status, "settings": SETTINGS, "at_utc": utc(),
            "IR": "NOT_COMPUTED", "chemical_assignment": "NOT_PERFORMED",
            "classifier_results_modified": False, "source_sha256": sha(__file__)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["controls", "optimize", "hessian", "analyze"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--optimization", type=Path)
    parser.add_argument("--initial-density", type=Path)
    parser.add_argument("--initial-density-sha256")
    parser.add_argument("--initial-density-source-result", type=Path)
    parser.add_argument("--step", type=float)
    parser.add_argument("--hessians", nargs=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse only authenticated qualified points; failed points never retry")
    args = parser.parse_args(argv)
    if args.threads not in (1, 2):
        parser.error("Only one or two threads are allowed")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from pyscf import lib
    lib.num_threads(args.threads)
    try:
        result = {"controls": stage_controls, "optimize": stage_optimize,
                  "hessian": stage_hessian, "analyze": stage_analyze}[args.stage](args)
        result.update(threads=args.threads, pyscf_version=importlib.metadata.version("pyscf"))
        save_json(args.output / "result.json", result)
        return 0 if result["status"] in {"CONTROLS_PASSED", "OPTIMIZED_GEOMETRY_CERTIFIED", "NUMERICAL_HESSIAN_COMPLETE", "NUMERICAL_HESSIAN_ANALYZED"} else 2
    except BaseException as error:
        path = args.output / "result.json"
        prior = json.loads(path.read_text()) if path.exists() else base_result("FAILED")
        prior.update(status=prior["status"] if prior.get("status") == "NUMERICAL_HESSIAN_PARTIAL" else "FAILED",
                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                     completed_at_utc=utc())
        save_json(path, prior)
        return 2


if __name__ == "__main__":
    sys.exit(main())
