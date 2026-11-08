"""Bounded, explicit rebuilt-grid finite-displacement SCF/gradient diagnostics.

Default invocation writes NOT_RUN preparation only. Actual displaced points
require the unchanged V8 cumulative guardian and exclusive closed-main queue.
Every point is a fresh producer: no historical molecule, array, result or ledger
is edited. This worker does NOT compute a new Hessian or a verified minimum.

Grids are rebuilt at each displaced geometry while the stock nuclear gradient
uses grid_response=False, as in the source calculation. Thus this is a
rebuilt-grid diagnostic, not an exact finite difference of a common fixed
quadrature functional. Disagreement alone does not establish a Hessian bug.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v8.py"
ANALYZER_SHA = "da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6"
AUDITOR = ROOT / "scripts/audit_pyscf_saved_cphf_residual_20261105.py"
AUDITOR_SHA = "6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c"
POINT_SCHEMA = "research4-cloud-followup-scf-gradient-v1"
POINT_SUCCESS = "DISPLACED_SCF_GRADIENT_COMPUTED"
ANALYSIS_SCHEMA = "research4-cloud-followup-directional-finite-difference-v1"
ANALYSIS_SUCCESS = "DIRECTIONAL_FINITE_DIFFERENCE_ANALYZED"
SOURCE_SNAPSHOT = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRID_LIMITATION = (
    "REBUILT_ATOM_CENTERED_GRID_WITH_GRID_RESPONSE_FALSE_NOT_EXACT_COMMON_"
    "FIXED_QUADRATURE_FINITE_DIFFERENCE; difference alone cannot diagnose a bug"
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def checked(record, *, max_bytes=32 * 1024**2):
    original = Path(record["path"])
    require(not original.is_symlink(), "Symlink source rejected")
    p = original.resolve()
    require(p.is_file() and p.is_relative_to(ROOT), "Existing repository source required")
    require(p.stat().st_size <= max_bytes and sha(p) == record["sha256"], "Source hash or size differs")
    return p


def read(path):
    p = Path(path)
    require(p.is_file() and p.stat().st_size <= 32 * 1024**2, "Bounded JSON required")
    return json.loads(p.read_text())


def save(path, value, *, exclusive=False):
    p = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with p.open("x") as handle:
            handle.write(content)
    else:
        tmp = p.with_name(p.name + ".tmp-" + str(os.getpid()))
        with tmp.open("x") as handle:
            handle.write(content)
        os.replace(tmp, p)


def load(path, expected, name):
    require(sha(path) == expected, "Frozen dependency source differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_displacement(mode_index, coordinate_index, step_bohr, sign, atoms=None):
    require((mode_index is not None) != (coordinate_index is not None), "Choose one mode OR one Cartesian coordinate")
    if mode_index is not None:
        require(type(mode_index) is int and mode_index >= 1, "Mode numbers are positive and one based")
        if atoms is not None:
            require(mode_index <= 3 * atoms - 6, "Mode number exceeds complete nonlinear mode space")
    else:
        require(type(coordinate_index) is int and coordinate_index >= 0, "Cartesian coordinate numbers are zero based")
        if atoms is not None:
            require(coordinate_index < 3 * atoms, "Coordinate number exceeds complete molecule")
    require(type(sign) is int and sign in (-1, 1), "A point has one explicitly signed displacement")
    require(math.isfinite(step_bohr) and 1e-4 <= step_bohr <= 0.02, "Bounded 0.0001--0.02 Bohr Euclidean displacement required")


def fresh_output(path):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT / "reports") and not p.exists(), "Fresh separate reports directory required")
    p.mkdir(parents=True)
    return p


def sole_phase(guard, supervision):
    proof = supervision["proof"]
    ledger = guard.read(guard.root_path(proof["ledger_path"]))
    guard.validate_ledger(ledger, allow_running=True)
    running = [p for p in ledger["phases"] if p.get("status") == "RUNNING"]
    require(len(running) == 1 and running[0]["phase"] == proof["phase"], "Exactly this one supervised Q phase may run")
    scripts = [guard.root_path(x) for x in running[0]["command"] if isinstance(x, str) and x.endswith(".py")]
    require(Path(__file__).resolve() in scripts, "Actual guardian command does not bind this worker")


@contextmanager
def closed_main(args, analyzer):
    record = {"path": str(args.main_pipeline_state.resolve()), "sha256": args.main_pipeline_state_sha256}
    path = checked(record)
    auditor = load(AUDITOR, AUDITOR_SHA, "cloud_followup_main_closure_validator")
    lock = ROOT / ".runtime/research4-materialized-quantum-queue.lock"
    require(lock.is_file() and not lock.is_symlink(), "Existing main quantum queue lock required")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auditor.validate_main_closed_state(read(path))
        yield record
        checked(record)


def source_gate(args, analyzer):
    path = args.source_result.resolve()
    gate = analyzer.qualify(path, args.source_result_sha256, control=args.control_validation)
    result = gate["result"]
    if args.control_validation:
        require(result["case"] in ("water_software_control", "ethanol_software_control"), "Authenticated software control required")
    else:
        require(result["case"] == "Spinosin" and result["atoms"] == 75 and result["basis_functions"] == 762
                and result["snapshot_manifest_sha256"] == SOURCE_SNAPSHOT, "Only original complete Spinosin C001 source is allowed")
    validate_displacement(args.mode_index, args.coordinate_index, args.step_bohr, args.sign, result["atoms"])
    return gate


def direction_and_hessian(args, gate, np):
    result = gate["result"]
    n = result["atoms"]
    directory = args.source_result.resolve().parent
    raw_file = checked({"path": str(directory / "hessian_hartree_per_bohr2.npy"), "sha256": result["raw_hessian_sha256"]})
    raw = np.load(raw_file, allow_pickle=False)
    require(raw.dtype == np.float64 and raw.shape == (n, n, 3, 3) and np.isfinite(raw).all(), "Complete original raw Hessian required")
    references = {"raw_hessian": ref(raw_file)}
    if args.mode_index is None:
        q = np.zeros((n, 3), dtype=float)
        q.reshape(-1)[args.coordinate_index] = 1.
    else:
        require(isinstance(args.mode_array_sha256, str) and len(args.mode_array_sha256) == 64,
                "Root protocol must explicitly bind the previously audited mode array SHA")
        mode_file = checked({"path": str(directory / "normal_modes.npy"), "sha256": args.mode_array_sha256},
                            max_bytes=8 * 1024**2)
        # The global result records all actual modes. The independent analyzer
        # additionally verifies the matrix/mode replay, but no mode is re-fit here.
        mode_array = np.load(mode_file, allow_pickle=False)
        require(mode_array.dtype == np.float64
                and mode_array.shape == (3*n-6, n, 3) and np.isfinite(mode_array).all(), "Complete bounded saved mode vectors required")
        references["normal_modes"] = ref(mode_file)
        vector = mode_array[args.mode_index - 1]
        norm = float(np.linalg.norm(vector))
        require(norm > 0 and math.isfinite(norm), "Finite nonzero mode vector required")
        q = vector / norm
    require(np.isclose(np.sum(q*q), 1., atol=1e-12, rtol=0), "Euclidean Cartesian unit direction required")
    hq = np.einsum("abxy,by->ax", raw, q)
    curvature = float(np.sum(q * hq))
    return q, hq, curvature, references


def execute(args):
    guard = load(GUARD, GUARD_SHA, "cloud_followup_v8_guard")
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    require(args.threads == 2, "Reviewed two-thread V8 resource policy required")
    sole_phase(guard, supervision)
    analyzer = load(ANALYZER, ANALYZER_SHA, "cloud_followup_authenticated_source_analyzer")
    with closed_main(args, analyzer) as main_ref:
        gate = source_gate(args, analyzer)
        return execute_closed(args, analyzer, gate, supervision, main_ref)


def execute_closed(args, analyzer, gate, supervision, main_ref):
    import numpy as np
    from pyscf import lib
    from pyscf.scf import _vhf, addons
    from pyscf.grad.rks import Gradients
    lib.num_threads(2)
    w = load(analyzer.STATE, analyzer.STATE_SHA, "cloud_followup_exact_state_restorer")
    base = w.frozen_base()
    q, hq, curvature, references = direction_and_hessian(args, gate, np)
    out = fresh_output(args.output)
    log = (out / "pyscf.log").open("x", buffering=1)
    source = gate["result"]
    state = {"schema": POINT_SCHEMA, "status": "PARTIAL", "stage": "rebuilt-grid-displaced-scf-gradient",
        "started_at_utc": utc(), "source_worker": ref(__file__), "resource_supervision": supervision,
        "output_directory": str(out), "stable_MF_state_unchanged": False,
        "stable_MF_flag_scope": "ORIGINAL_SOURCE_MF_IMMUTABILITY_NOT_DISPLACED_STATE_EQUALITY_OR_ORBITAL_STABILITY",
        "source_result": ref(args.source_result), "snapshot_manifest_sha256": source["snapshot_manifest_sha256"],
        "case": source["case"], "source_arrays": references, "closed_main_pipeline_state": main_ref,
        "displacement": {"mode_index_1based": args.mode_index, "coordinate_index_0based": args.coordinate_index,
            "step_bohr": args.step_bohr, "sign": args.sign, "direction_normalization": "EUCLIDEAN_CARTESIAN_UNIT_NORM",
            "maximum_atom_displacement_bohr": args.step_bohr * float(np.linalg.norm(q, axis=1).max())},
        "analytic_directional_curvature_Eh_per_Bohr2": curvature,
        "grid_diagnostic_limitation": GRID_LIMITATION, "IR_intensities_status": "NOT_COMPUTED",
        "new_full_Hessian": "NOT_COMPUTED", "stationary_minimum_verified": False,
        "chemical_assignment_verified": False, "old_scientific_results_modified": False,
        "new_classifier_training": False, "scf_converged": False, "gradient_completed": False}
    save(out / "input.json", {**state, "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}, exclusive=True)
    state["input_json_sha256"] = sha(out / "input.json")
    cpu, wall = time.process_time(), time.perf_counter()
    def emit(event, detail=None):
        row = {"utc": utc(), "event": event, "CPU_seconds": time.process_time()-cpu,
               "wall_seconds": time.perf_counter()-wall, **(detail or {})}
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        save(out / "progress.json", {**state, **row})
    def stop(signum, frame):
        raise base.BudgetStop("Bounded diagnostic received resource signal " + str(signum))
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
        native_threads=2, emit=lambda d: emit("native_direct_driver_signature", d))
    try:
        native.install()
        original, binding = w.restore_snapshot(Path(source["snapshot_directory"]),
            expected_manifest_sha256=source["snapshot_manifest_sha256"], expected_source_sha256=analyzer.STATE_SHA, stdout=log)
        require(binding == source["source_state_binding"], "Restored exact original state differs")
        original_fingerprint = w.molecule_physical_fingerprint(original.mol)
        def source_mf_fingerprints():
            return {key:base.array_fingerprint(value) for key,value in {
                "MO_coeff":original.mo_coeff, "MO_energy":original.mo_energy, "MO_occ":original.mo_occ,
                "grid_coordinates":original.grids.coords, "grid_weights":original.grids.weights}.items()}
        original_mf_payload = source_mf_fingerprints()
        n, nao = original.mol.natm, original.mol.nao_nr()
        symbols = [original.mol.atom_symbol(i) for i in range(n)]
        require(n == source["atoms"] and nao == source["basis_functions"]
                and original.mo_coeff.shape == (nao, nao) and original.mol.charge == 0
                and original.mol.spin == 0 and original.mol.nelectron == int(original.mo_occ.sum()),
                "Complete neutral closed-shell original AO/MO and electron count required")
        source_xyz = original.mol.atom_coords(unit="Bohr")
        new_xyz = source_xyz + args.sign * args.step_bohr * q
        # A separate Mole owns a deep-copy geometry and a newly constructed grid.
        mol = original.mol.copy()
        mol.set_geom_(new_xyz, unit="Bohr", inplace=True)
        mol.stdout = log
        require(w.molecule_physical_fingerprint(original.mol) == original_fingerprint,
                "Creating diagnostic geometry changed original molecule")
        require([mol.atom_symbol(i) for i in range(n)] == symbols and mol.nao_nr() == nao
                and mol.charge == original.mol.charge and mol.spin == original.mol.spin
                and np.array_equal(mol.atom_coords(unit="Bohr"), new_xyz), "Atom order, full AO space or exact displaced geometry changed")
        settings = dict(binding["method_settings"])
        require(settings["xc"] == "PBE" and settings["basis"].lower() == "def2-svp"
                and settings["grid_level"] == 1 and settings["dispersion"] == "NONE_EXPLICIT_PBE_PILOT_NOT_D3"
                and settings["gradient_grid_response"] is False, "Frozen PBE/def2-SVP/grid1/no-D3 gradient model required")
        settings.update(threads=2, native_direct_driver_threads=2)
        mf = w.make_scf(mol, settings, out / "scf.chk")
        require(mf.grids.coords is None and mf.grids.weights is None and mf.disp is None,
                "Each point must rebuild its own atom-centered grid without adding D3")
        state["settings"] = settings
        state["atoms"], state["basis_functions"], state["occupied_orbitals"] = n, nao, int(np.count_nonzero(original.mo_occ > 0))
        state["symbols"] = symbols
        state["reference_energy_hartree"] = float(original.e_tot)
        state["reference_state_binding"] = binding
        state["source_geometry_fingerprint"] = original_fingerprint
        base.atomic_npy(out / "direction_cartesian_unit.npy", q)
        base.atomic_npy(out / "analytic_H_times_direction.npy", hq)
        base.atomic_npy(out / "source_coordinates_bohr.npy", source_xyz)
        base.atomic_npy(out / "displaced_coordinates_bohr.npy", new_xyz)
        (out / "displaced.xyz").write_text(base.xyz_text(symbols, mol.atom_coords(unit="Angstrom"), "Finite displacement diagnostic; NOT an optimized structure"))
        dm0 = addons.project_dm_nr2nr(original.mol, original.make_rdm1(), mol)
        state["initial_guess"] = "PROJECTED_EXACT_SOURCE_DENSITY_TO_DISPLACED_AO_BASIS"
        def callback(env):
            emit("SCF_cycle", {"cycle_0based": int(env.get("cycle", -1)),
                "energy_hartree": float(env["e_tot"]) if "e_tot" in env else None})
        mf.callback = callback
        emit("SCF_started", {"actual_new_geometry": True, "grid_rebuilt": True})
        mf.kernel(dm0=dm0)
        state["scf_converged"] = bool(mf.converged)
        state["energy_hartree"] = float(mf.e_tot)
        require(mf.converged and math.isfinite(mf.e_tot), "Displaced SCF did not converge")
        require(mf.mo_coeff.shape == (nao, nao) and np.count_nonzero(mf.mo_occ > 0) == state["occupied_orbitals"],
                "Displaced full AO/MO or occupied space differs")
        state["displaced_grid_points"] = int(len(mf.grids.weights))
        emit("SCF_complete", {"energy_hartree": state["energy_hartree"]})
        cls = base.direct_gradient_type(Gradients, emit, memory_mb=lambda: lib.current_memory()[0])
        gradient_obj = cls(mf)
        gradient_obj.grid_response = False
        gradient = np.asarray(gradient_obj.kernel())
        require(gradient.shape == (n, 3) and np.isfinite(gradient).all(), "Complete finite displaced nuclear gradient required")
        base.atomic_npy(out / "gradient_hartree_per_bohr.npy", gradient)
        state["gradient_completed"] = True
        state["gradient_hartree_per_bohr"] = gradient.tolist()
        state["gradient_projected_Eh_per_Bohr"] = float(np.sum(q * gradient))
        state["status"] = POINT_SUCCESS
        state["computed_array_files"] = {name: ref(out / name) for name in
            ("direction_cartesian_unit.npy", "analytic_H_times_direction.npy", "source_coordinates_bohr.npy",
             "displaced_coordinates_bohr.npy", "gradient_hartree_per_bohr.npy")}
        state["checkpoint"] = {**ref(out / "scf.chk"), "storage_policy": "LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"}
        require(w.molecule_physical_fingerprint(original.mol) == original_fingerprint, "Original source geometry changed during calculation")
        require(source_mf_fingerprints() == original_mf_payload, "Original source MF payload changed during calculation")
        for record in references.values():
            checked(record)
        checked(state["source_result"])
        state["stable_MF_state_unchanged"] = True
        state["original_MF_payload_fingerprints"] = original_mf_payload
        emit("point_completed", {"gradient_completed": True})
    except BaseException as exc:
        state["status"] = "PARTIAL_RESOURCE_STOP" if isinstance(exc, base.BudgetStop) else "FAILED_DIAGNOSTIC"
        state["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        emit("point_failed", {"error": state["error"]})
    finally:
        native.restore()
        state["native_direct_driver_control"] = native.summary()
        if state["native_direct_driver_control"].get("restoration_failures") != 0 or not state["native_direct_driver_control"].get("original_driver_restored"):
            state["status"] = "FAILED_NATIVE_DRIVER_RESTORATION"
        state["ended_at_utc"], state["worker_CPU_seconds"], state["worker_wall_seconds"] = utc(), time.process_time()-cpu, time.perf_counter()-wall
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        log.close()
        save(out / "result.json", state, exclusive=True)
    return state


def analyze(args):
    """Only small sealed points are combined; no quantum engine is imported."""
    plus_path = args.plus_result.resolve(); minus_path = args.minus_result.resolve()
    plus, minus = read(plus_path), read(minus_path)
    require(plus["schema"] == minus["schema"] == POINT_SCHEMA and plus["status"] == minus["status"] == POINT_SUCCESS,
            "Two actually completed displaced points required")
    d1, d2 = plus["displacement"], minus["displacement"]
    require(d1["sign"] == 1 and d2["sign"] == -1, "Explicit + and - records required")
    for key in ("source_result", "source_worker", "source_arrays", "snapshot_manifest_sha256", "settings", "symbols", "source_geometry_fingerprint"):
        require(plus[key] == minus[key], "Source or method differs across +/- points: " + key)
    for key in ("mode_index_1based", "coordinate_index_0based", "step_bohr", "direction_normalization"):
        require(d1[key] == d2[key], "Displacement pair differs: " + key)
    require(plus["reference_energy_hartree"] == minus["reference_energy_hartree"], "Reference energy differs")
    checked(plus["source_result"])
    guard = load(GUARD, GUARD_SHA, "cloud_followup_closed_point_guard")
    for path, result in ((plus_path, plus), (minus_path, minus)):
        # The genuine guardian, result SHA and actual worker command must close.
        evidence = {"source_worker": result["source_worker"], "resource_supervision": result["resource_supervision"],
                    "output_directory": str(path.parent)}
        closure = guard.verify_closed_phase(evidence, expected_worker_path=Path(__file__).resolve(),
            expected_worker_sha256=plus["source_worker"]["sha256"], expected_result_schema=POINT_SCHEMA,
            allowed_result_statuses=(POINT_SUCCESS,), expected_output=path.parent)
        require(closure["producer_result_sha256"] == sha(path), "Closed point result differs")
    import numpy as np
    def array(result, name):
        record = result["computed_array_files"][name]
        value = np.load(checked(record), allow_pickle=False)
        require(value.dtype == np.float64 and value.shape == (plus["atoms"], 3) and np.isfinite(value).all(), "Point array shape/dtype differs")
        return value
    q = array(plus, "direction_cartesian_unit.npy")
    hq = array(plus, "analytic_H_times_direction.npy")
    require(np.array_equal(q, array(minus, "direction_cartesian_unit.npy"))
            and np.array_equal(hq, array(minus, "analytic_H_times_direction.npy")), "Original unit direction or analytic Hq differs")
    step = d1["step_bohr"]
    source_xyz = array(plus, "source_coordinates_bohr.npy")
    require(np.array_equal(source_xyz, array(minus, "source_coordinates_bohr.npy")), "Source coordinates differ across the pair")
    require(np.array_equal(array(plus, "displaced_coordinates_bohr.npy"), source_xyz + step*q)
            and np.array_equal(array(minus, "displaced_coordinates_bohr.npy"), source_xyz - step*q),
            "Actual displaced geometry does not equal source +/- declared step times direction")
    fd = (array(plus, "gradient_hartree_per_bohr.npy") - array(minus, "gradient_hartree_per_bohr.npy")) / (2*step)
    difference = fd-hq
    fd_curv, analytic_curv = float(np.sum(q*fd)), float(np.sum(q*hq))
    energy_curv = (plus["energy_hartree"]+minus["energy_hartree"]-2*plus["reference_energy_hartree"])/(step*step)
    out = fresh_output(args.output)
    base = load(ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py",
        "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d", "cloud_followup_analysis_array_writer")
    base.atomic_npy(out / "finite_difference_H_times_direction.npy", fd)
    base.atomic_npy(out / "finite_difference_minus_analytic_Hq.npy", difference)
    state = {"schema": ANALYSIS_SCHEMA, "status": ANALYSIS_SUCCESS, "computed_at_utc": utc(),
        "source_worker": ref(__file__), "source_result": plus["source_result"], "source_points": [ref(plus_path),ref(minus_path)],
        "new_quantum_evaluations": 0, "case": plus["case"], "displacement_pair": d1,
        "analytic_directional_curvature_Eh_per_Bohr2": analytic_curv,
        "finite_gradient_directional_curvature_Eh_per_Bohr2": fd_curv,
        "finite_energy_directional_curvature_Eh_per_Bohr2": energy_curv,
        "Hq_max_abs_difference_Eh_per_Bohr2": float(np.max(np.abs(difference))),
        "Hq_RMS_difference_Eh_per_Bohr2": float(np.sqrt(np.mean(difference*difference))),
        "Hq_Frobenius_difference_Eh_per_Bohr2": float(np.linalg.norm(difference)),
        "directional_curvature_difference_Eh_per_Bohr2": fd_curv-analytic_curv,
        "relative_directional_difference": abs(fd_curv-analytic_curv)/abs(analytic_curv) if analytic_curv != 0 else None,
        "source_signed_frequency_cm1": None if d1["mode_index_1based"] is None else
            read(checked(plus["source_result"]))["frequencies_cm1"][d1["mode_index_1based"]-1],
        "grid_diagnostic_limitation": GRID_LIMITATION, "new_numeric_PASS_FAIL_threshold": "NOT_SET",
        "energy_only_is_not_sufficient_derivative_verification": True,
        "stationary_minimum_verified": False, "cause_of_negative_modes": "NOT_ESTABLISHED_BY_THIS_DIAGNOSTIC_ALONE",
        "IR_intensities_status": "NOT_COMPUTED"}
    save(out / "result.json", state, exclusive=True)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    p.add_argument("--source-result", type=Path); p.add_argument("--source-result-sha256")
    p.add_argument("--main-pipeline-state", type=Path); p.add_argument("--main-pipeline-state-sha256")
    p.add_argument("--mode-index", type=int); p.add_argument("--coordinate-index", type=int)
    p.add_argument("--mode-array-sha256")
    p.add_argument("--step-bohr", type=float); p.add_argument("--sign", type=int)
    p.add_argument("--control-validation", action="store_true"); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--plus-result", type=Path); p.add_argument("--minus-result", type=Path)
    args = p.parse_args()
    if args.execute:
        require(args.source_result is not None and args.source_result_sha256 and args.main_pipeline_state is not None
                and args.main_pipeline_state_sha256 and args.step_bohr is not None, "Explicit authenticated source/main/step inputs required")
        require(args.plus_result is None and args.minus_result is None, "Point execution cannot combine old points")
        validate_displacement(args.mode_index, args.coordinate_index, args.step_bohr, args.sign)
        value = execute(args)
    elif args.analyze_only:
        require(args.plus_result is not None and args.minus_result is not None and args.source_result is None,
                "Analysis uses two actual closed point results")
        value = analyze(args)
    else:
        require(all(x is None for x in (args.source_result,args.source_result_sha256,args.main_pipeline_state,args.main_pipeline_state_sha256,
            args.mode_index,args.coordinate_index,args.mode_array_sha256,args.step_bohr,args.sign,args.plus_result,args.minus_result)) and not args.control_validation,
            "Metadata preparation does not read target inputs")
        out = fresh_output(args.output)
        value = {"schema": POINT_SCHEMA, "status": "NOT_RUN", "prepared_at_utc": utc(), "source_worker": ref(__file__),
            "new_quantum_evaluations": 0, "numerical_packages_imported": False, "target_arrays_read": False,
            "grid_diagnostic_limitation": GRID_LIMITATION, "IR_intensities_status": "NOT_COMPUTED"}
        save(out / "preflight.json", value, exclusive=True)
    print(json.dumps({key:value.get(key) for key in ("schema","status","case","gradient_completed")},ensure_ascii=False))
    return 0 if value["status"] in ("NOT_RUN",POINT_SUCCESS,ANALYSIS_SUCCESS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
