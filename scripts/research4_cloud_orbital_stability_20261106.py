"""Bounded native PySCF RKS internal-orbital-stability diagnostic.

Default execution is metadata-only NOT_RUN. Actual execution requires the
frozen cumulative V8 guardian, a healthy closed original response producer,
and the closed main-queue lock. No SCF, optimization, nuclear Hessian, new
classification, external/spin-breaking stability, or IR intensity is computed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts/audit_pyscf_saved_cphf_residual_20261105.py"
AUDIT_SHA = "6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c"
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
CONSUMER = ROOT / "scripts/pyscf_staged_consumer_20261105_v8.py"
CONSUMER_SHA = "3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585"
NATIVE_SOURCES = {
    "scf/stability.py": "58ceea68eb56702ec41fa0a844b60c225ef53800ed4fb5b1fea307cd5bc50674",
    "soscf/newton_ah.py": "7276458382a9530771b85b0c0d1aa39fd42d9b98e69c65d9b4a5410222a70c98",
    "lib/linalg_helper.py": "9dcd5d4582c3695818e05fa72217c7764f87d5491ed29acb7b64d2725ee130e0",
}
SCHEMA = "research4-native-internal-orbital-stability-diagnostic-v1"
SUCCESS = "DONE_NATIVE_INTERNAL_ORBITAL_STABILITY_ANALYSIS"


def require(value, message):
    if not value:
        raise ValueError(message)


def load_dependencies():
    import hashlib
    import importlib.util
    require(hashlib.sha256(AUDIT.read_bytes()).hexdigest() == AUDIT_SHA,
            "Frozen diagnostic helper source differs")
    spec = importlib.util.spec_from_file_location("research4_frozen_saved_response_audit", AUDIT)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    return audit


def validate_solver_policy(threads, max_cycle, nroots, tol):
    require(type(threads) is int and threads == 2, "Exactly two supervised threads required")
    require(type(max_cycle) is int and 1 <= max_cycle <= 30, "Bounded Davidson max_cycle must be 1..30")
    require(type(nroots) is int and nroots == 3, "Exactly three native internal-stability roots required")
    require(type(tol) in (float, int) and tol == 1e-4, "Reviewed native Davidson tolerance is 1e-4")


def sole_diagnostic_phase(guard, supervision):
    proof = supervision["proof"]
    ledger = guard.read(guard.root_path(proof["ledger_path"]))
    guard.validate_ledger(ledger, allow_running=True)
    active = [row for row in ledger["phases"] if row.get("status") == "RUNNING"]
    require(len(active) == 1 and active[0]["phase"] == proof["phase"],
            "A different or concurrent quantum phase is active")
    scripts = [guard.root_path(x) for x in active[0]["command"] if isinstance(x, str) and x.endswith(".py")]
    require(Path(__file__).resolve() in scripts, "Live guardian must bind this actual diagnostic script")
    return ledger


def floats(values):
    return [float(value) for value in values]


@contextmanager
def capture_native_davidson(linalg_helper, *, max_cycle, nroots, tol, emit, record):
    """Observe the actual native davidson1 return without new operator actions.

    Native rhf_stability calls lib.davidson, whose frozen function delegates to
    linalg_helper.davidson1. We preserve that algorithm and call signature,
    bound its cycle/memory policy, and capture its returned convergence flags.
    Callback residuals are explicitly intermediate, not final measurements.
    """
    original = linalg_helper.davidson1
    signature = inspect.signature(original)
    require(record.get("native_davidson1_calls", 0) == 0, "Fresh observer record required")

    def observed(*args, **kwargs):
        require(record.get("native_davidson1_calls", 0) == 0, "Only one internal-stability Davidson invocation allowed")
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        require(bound.arguments["nroots"] == nroots and bound.arguments["tol"] == tol,
                "Native internal-stability solver arguments differ")
        require(bound.arguments["callback"] is None, "Unexpected existing Davidson callback")
        bound.arguments["max_cycle"] = max_cycle
        bound.arguments["max_memory"] = 2048
        record["native_davidson1_calls"] = 1
        record["effective_native_solver_policy"] = {
            "max_cycle": max_cycle, "nroots": nroots, "tol": tol,
            "max_space": bound.arguments["max_space"], "max_memory_mb": 2048,
            "native_tol_residual_argument": bound.arguments["tol_residual"],
            "effective_native_residual_tolerance": tol**0.5 if bound.arguments["tol_residual"] is None else bound.arguments["tol_residual"],
            "final_residual_norms": "NOT_RETURNED_BY_NATIVE_API_NOT_COMPUTED",
            "algorithm": "FROZEN_PYSCF_LIB_DAVIDSON1_UNCHANGED",
        }

        def callback(env):
            row = {"iteration_zero_based": int(env["icyc"]), "roots_Eh": floats(env["e"]),
                   "root_residual_norms": floats(env["dx_norm"]),
                   "root_energy_changes_Eh": floats(env["de"]),
                   "native_convergence_flags": [bool(x) for x in env["conv"]],
                   "scope": "INTERMEDIATE_CALLBACK_NOT_FINAL_ROOT_RESIDUALS"}
            record.setdefault("intermediate_callbacks", []).append(row)
            emit("native_davidson_intermediate", row)

        bound.arguments["callback"] = callback
        emit("native_davidson_begin", record["effective_native_solver_policy"])
        result = original(*bound.args, **bound.kwargs)
        require(isinstance(result, tuple) and len(result) == 3, "Unexpected native davidson1 return shape")
        conv, eigenvalues, vectors = result
        record["actual_returned_convergence_flags"] = [bool(x) for x in conv]
        record["actual_returned_roots_Eh"] = floats(eigenvalues)
        record["actual_returned_root_count"] = len(vectors)
        emit("native_davidson_returned", {
            "native_convergence_flags": record["actual_returned_convergence_flags"],
            "roots_Eh": record["actual_returned_roots_Eh"], "vectors": len(vectors)})
        return result

    linalg_helper.davidson1 = observed
    try:
        yield
    finally:
        require(linalg_helper.davidson1 is observed, "Native observer binding unexpectedly changed")
        linalg_helper.davidson1 = original
        record["native_davidson1_original_binding_restored"] = linalg_helper.davidson1 is original


def interpret_native_result(record, native_stable, nroots=3):
    roots = record.get("actual_returned_roots_Eh", [])
    flags = record.get("actual_returned_convergence_flags", [])
    count = record.get("actual_returned_root_count", 0)
    import math
    complete = (len(roots) == len(flags) == count == nroots
                and all(flags) and all(math.isfinite(x) for x in roots))
    if not complete:
        return {"status": "PARTIAL", "internal_orbital_stability": "UNDETERMINED_NATIVE_SOLVER_NOT_FULLY_CONVERGED",
                "native_raw_stable_flag_not_a_convergence_claim": bool(native_stable),
                "native_all_requested_roots_converged": False}
    require(bool(native_stable) == (roots[0] >= -1e-5), "Native stability flag and lowest root disagree")
    return {"status": SUCCESS, "internal_orbital_stability": "STABLE" if native_stable else "UNSTABLE",
            "native_all_requested_roots_converged": True,
            "native_internal_stability_eigenvalue_threshold_Eh": -1e-5}


def execute(args):
    validate_solver_policy(args.threads, args.max_cycle, args.nroots, args.tol)
    audit = load_dependencies()
    guard = audit.load(GUARD, GUARD_SHA, "orbital_stability_actual_v8_guard")
    consumer = audit.load(CONSUMER, CONSUMER_SHA, "orbital_stability_frozen_consumer")
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    sole_diagnostic_phase(guard, supervision)
    with audit.closed_main_queue(args, consumer) as main_state:
        gate = audit.metadata_gate(args, consumer, guard)
        return execute_closed(args, audit, consumer, supervision, main_state, gate)


def execute_closed(args, audit, consumer, supervision, main_state, gate):
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports") and not output.exists(), "Fresh independent report directory required")
    require(not output.is_relative_to(args.response_result.resolve().parent), "Never write inside a historical source")
    import numpy as np
    from pyscf import lib
    from pyscf.scf import stability, _vhf, cphf, hf, _response_functions as response
    from pyscf.soscf import newton_ah
    from pyscf.hessian import rhf, rks
    from pyscf.dft import numint, gen_grid, rks as dft_rks
    from pyscf.grad import rks as grad_rks, rhf as grad_rhf
    from pyscf.lib import linalg_helper
    lib.num_threads(2)
    modules = {"scf/stability.py": stability, "soscf/newton_ah.py": newton_ah,
               "lib/linalg_helper.py": linalg_helper}
    native_sources = {}
    pyscf_root = ROOT / ".runtime/quantum-upgrade-site/pyscf"
    for relative, module in modules.items():
        path = Path(module.__file__).resolve()
        require(path == pyscf_root / relative and audit.sha(path) == NATIVE_SOURCES[relative],
                "Reviewed native stability/Newton/Davidson source differs")
        native_sources[relative] = {"path": str(path), "sha256": NATIVE_SOURCES[relative]}
    require(stability.rhf_stability.__globals__ is stability.__dict__
            and stability.rhf_internal.__globals__ is stability.__dict__
            and stability.lib is lib and lib.davidson.__globals__ is linalg_helper.__dict__
            and newton_ah.gen_g_hop_rhf.__globals__ is newton_ah.__dict__,
            "Actual stability-to-Davidson native call chain differs")
    w = consumer.load_source(consumer.STATE, consumer.STATE_SHA, "orbital_stability_original_state")
    v5 = consumer.load_source(consumer.V5, consumer.V5_SHA, "orbital_stability_original_gradient")
    base = consumer.load_source(consumer.BASE, consumer.BASE_SHA, "orbital_stability_native_driver")
    core = consumer.load_source(consumer.CORE, consumer.CORE_SHA, "orbital_stability_runtime_binding")
    xc_core = consumer.load_source(consumer.XC_CORE, consumer.XC_CORE_SHA, "orbital_stability_native_xc_binding")
    output.mkdir(parents=True); log = (output / "pyscf.log").open("x", buffering=1)
    value = gate["result"]
    state = {"schema": SCHEMA, "stage": "internal-orbital-stability", "status": "PARTIAL",
             "started_at_utc": audit.utc(), "source_worker": consumer.artifact(__file__),
             "output_directory": str(output), "stable_MF_state_unchanged": False,
             "stable_MF_flag_scope": "ORIGINAL_SOURCE_IMMUTABILITY_NOT_ELECTRONIC_STABILITY_VERDICT",
             "source_response": gate["record"], "source_response_closure": gate["response"][3],
             "snapshot_manifest_sha256": value["snapshot_manifest_sha256"],
             "actual_closed_main_pipeline_state": main_state, "resource_supervision": supervision,
             "actual_stability_sources": native_sources,
             "scope": "RKS_INTERNAL_REAL_RESTRICTED_ORBITAL_STABILITY_ONLY",
             "external_complex_spin_breaking_stability": "NOT_RUN",
             "nuclear_stationary_minimum_verified": False, "chemical_assignment_verified": False,
             "IR_intensities_status": "NOT_COMPUTED", "no_new_SCF_optimization_nuclear_Hessian_or_classifier": True,
             "source_MF_immutability_is_not_electronic_stability": True,
             "negative_nuclear_modes_cause": "NOT_ESTABLISHED_BY_THIS_SINGLE_CHECK",
             "original_results_changed": False, "davidson": {"native_davidson1_calls": 0}}
    consumer.save(output / "input.json", {**state, "arguments": {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, exclusive=True)
    state["input_json_sha256"] = audit.sha(output / "input.json")
    started_cpu, started_wall = time.process_time(), time.perf_counter()

    def emit(event, details=None):
        row = {"utc": audit.utc(), "event": event, "CPU_seconds": time.process_time()-started_cpu,
               "wall_seconds": time.perf_counter()-started_wall, **(details or {})}
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        consumer.save(output / "progress.json", {**state, **row})

    def stop(signum, frame):
        raise base.BudgetStop("Native internal-stability resource signal " + str(signum))

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
        native_threads=2, emit=lambda d: emit("native_driver_signature", d))
    try:
        native.install()
        state["gradient_gate"] = v5.qualify_gradient_receipt(value["gradient_gate"]["result"]["path"],
            value["gradient_gate"]["result"]["sha256"], value["snapshot_manifest_sha256"], v5.boot_id())
        mf, binding = w.restore_snapshot(Path(value["snapshot_directory"]),
            expected_manifest_sha256=value["snapshot_manifest_sha256"],
            expected_source_sha256=consumer.STATE_SHA, stdout=log)
        require(binding == value["source_state_binding"], "Exact source state binding differs")
        state["actual_imported_native_sources"] = consumer.bind_actual_native_modules(mf._numint,
            xc_core=xc_core, rhf_core=core, rhf=rhf, rks=rks, numint=numint,
            grad_rks=grad_rks, gen_grid=gen_grid, linalg_helper=linalg_helper, mf=mf,
            dft_rks=dft_rks, hf=hf, grad_rhf=grad_rhf, response=response, cphf=cphf,
            lib=lib, vhf=_vhf)
        mf.max_memory = 3000
        before = v5.physical_state(mf)
        require(mf.mo_coeff.shape == (gate["nao"], gate["nao"])
                and np.count_nonzero(mf.mo_occ > 0) == value["occupied_orbitals"],
                "Complete source AO/MO and occupied spaces required")
        state.update(case=value["case"], atoms=gate["n"], basis_functions=gate["nao"],
                     occupied_orbitals=value["occupied_orbitals"], energy_hartree=float(mf.e_tot),
                     settings={**binding["method_settings"], "threads": 2, "max_memory_mb": 3000,
                               "native_davidson_max_cycle": args.max_cycle, "nroots": args.nroots,
                               "tol": args.tol, "external": False, "internal": True})
        with capture_native_davidson(linalg_helper, max_cycle=args.max_cycle, nroots=args.nroots,
                                     tol=args.tol, emit=emit, record=state["davidson"]):
            mo_i, mo_e, stable_i, stable_e = stability.rhf_stability(
                mf, internal=True, external=False, verbose=4, return_status=True,
                nroots=args.nroots, tol=args.tol)
        require(mo_e is None and stable_e is None, "Unexpected external stability computation")
        require(state["davidson"].get("native_davidson1_calls") == 1,
                "Exactly one actual native internal-stability solver required")
        require(mo_i is not None and mo_i.shape == mf.mo_coeff.shape,
                "Native returned orbital matrix is incomplete")
        state.update(interpret_native_result(state["davidson"], stable_i, args.nroots))
        state["native_returned_candidate_orbitals_saved"] = False
        state["candidate_orbitals_not_used_to_replace_or_reoptimize_original_MF"] = True
        require(v5.physical_state(mf) == before, "Original exact MF changed during native diagnostic")
        repeat = audit.metadata_gate(args, consumer, audit.load(GUARD, GUARD_SHA, "orbital_stability_final_guard"))
        require(repeat["result"] == gate["result"] and repeat["physical"] == gate["physical"],
                "Original closed scientific source changed")
        state["source_MF_and_closed_producer_unchanged"] = True
        state["stable_MF_state_unchanged"] = True
        emit("internal_stability_diagnostic_returned", {"status": state["status"],
             "internal_orbital_stability": state["internal_orbital_stability"]})
    except BaseException as error:
        state.update(status="PARTIAL", exception_type=type(error).__name__, stop_reason=str(error))
        (output / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        native.restore(); state["native_direct_driver_control"] = native.summary()
        if state["native_direct_driver_control"].get("restoration_failures") != 0 or not state["native_direct_driver_control"].get("original_driver_restored"):
            state.update(status="PARTIAL", stop_reason="Native driver restoration failed")
        for signum, handler in handlers.items(): signal.signal(signum, handler)
        state.update(completed_at_utc=audit.utc(), process_cpu_seconds=time.process_time()-started_cpu,
                     wall_seconds=time.perf_counter()-started_wall)
        log.close(); consumer.save(output / "result.json", state, exclusive=True)
    return state


def preflight(output):
    audit = load_dependencies()
    require(not output.exists(), "Fresh metadata-only report directory required")
    references = []
    for p, expected in ((AUDIT, AUDIT_SHA), (GUARD, GUARD_SHA), (CONSUMER, CONSUMER_SHA)):
        require(audit.sha(p) == expected, "Frozen dependency source changed")
        references.append({"path": str(p), "sha256": expected})
    output.mkdir(parents=True)
    value = {"schema": SCHEMA, "status": "NOT_RUN", "source_worker": consumer_record(audit),
             "source_dependencies": references, "quantum_operator_evaluations": 0,
             "numerical_packages_imported": False, "target_arrays_read": False,
             "internal_stability": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED",
             "policy": {"max_cycle": 30, "nroots": 3, "tol": 1e-4, "threads": 2,
                        "external": False, "full_AO_MO": True,
                        "density_fitting": False, "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3"}}
    with (output / "preflight.json").open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.write("\n")
    return value


def consumer_record(audit):
    return {"path": str(Path(__file__).resolve()), "sha256": audit.sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--response-result", type=Path)
    parser.add_argument("--response-result-sha256")
    parser.add_argument("--main-pipeline-state", type=Path)
    parser.add_argument("--main-pipeline-state-sha256")
    parser.add_argument("--control-validation", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-cycle", type=int, default=30)
    parser.add_argument("--nroots", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.set_defaults(atom_index=0, whole_block=False)
    args = parser.parse_args()
    validate_solver_policy(args.threads, args.max_cycle, args.nroots, args.tol)
    if args.execute:
        require(args.response_result is not None and args.response_result_sha256,
                "Actual closed original response source and SHA required")
        value = execute(args)
    else:
        require(args.response_result is None and args.main_pipeline_state is None and not args.control_validation,
                "Metadata preparation does not consume quantum producer inputs")
        value = preflight(args.output)
    print(json.dumps({k: value.get(k) for k in ("status", "internal_orbital_stability", "native_all_requested_roots_converged")}))
    return 0 if value["status"] in (SUCCESS, "NOT_RUN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
