"""Optional saved-CPHF equation audit. Default is metadata NOT_RUN.

Only root may execute this worker under the frozen V8 quantum guardian after
the main quantum queue is closed. It evaluates the full native response
operator once at the saved solution; it never runs a response solver, SCF,
gradient, Hessian or classifier. Old producer results remain immutable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
CONSUMER = ROOT / "scripts/pyscf_staged_consumer_20261105_v8.py"
CONSUMER_SHA = "3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585"
SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
CONTROL_SHAS = {"water_software_control": "123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4",
    "ethanol_software_control": "9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba"}
SCHEMA = "research4-saved-cphf-equation-residual-audit-v1"
SUCCESS = "SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(file, expected, name):
    require(sha(file) == expected, "Frozen source differs")
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def select_atoms(available, atom_index, whole_block):
    require(isinstance(available, list) and available and len(set(available)) == len(available), "Exact source atom list required")
    require((atom_index is not None) != whole_block, "Choose one atom or the whole completed B8 block")
    if whole_block:
        return list(available)
    require(type(atom_index) is int and atom_index in available, "Selected atom is not in the completed response")
    return [atom_index]


def sole_audit_phase(guard, supervision):
    proof = supervision["proof"]
    ledger = guard.read(guard.root_path(proof["ledger_path"]))
    guard.validate_ledger(ledger, allow_running=True)
    active = [p for p in ledger["phases"] if p.get("status") == "RUNNING"]
    require(len(active) == 1 and active[0]["phase"] == proof["phase"], "No concurrent main quantum phase is permitted")
    command = active[0]["command"]
    scripts = [guard.root_path(p) for p in command if isinstance(p, str) and p.endswith(".py")]
    require(Path(__file__).resolve() in scripts, "Live phase must bind this actual audit entrypoint")
    return ledger


def validate_main_closed_state(value):
    invocations = value.get("invocations", [])
    require(invocations and invocations[-1].get("status") in
        ("PIPELINE_HEALTHY_CLOSED", "BOUNDARY_CLOSED", "BLOCKED_REQUIRES_ROOT_REVIEW_NO_AUTOMATIC_RETRY"),
        "The main pipeline must be actually closed, not merely between stages")
    require(value.get("stages") and all(s.get("status") in ("HEALTHY_CLOSED", "BLOCKED_REQUIRES_ROOT_REVIEW")
        and not s.get("guardian_unresolved") for s in value["stages"].values()),
        "A main quantum stage is still active or unresolved")
    analysis = value.get("analysis", {})
    require(analysis.get("status") in (None, "HEALTHY_CLOSED", "BLOCKED_REQUIRES_ROOT_REVIEW")
        and not analysis.get("child_unresolved"), "Main independent analysis is still active or unresolved")


@contextmanager
def closed_main_queue(args, consumer):
    require(args.main_pipeline_state is not None and args.main_pipeline_state_sha256,
        "Explicit actual closed main state and SHA are required")
    record = {"path": str(args.main_pipeline_state.resolve()), "sha256": args.main_pipeline_state_sha256}
    lock = ROOT / ".runtime/research4-materialized-quantum-queue.lock"
    require(lock.is_file() and not lock.is_symlink(), "Use the existing real main-queue lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_main_closed_state(consumer.read(consumer.checked(record)))
        yield record
        consumer.checked(record)


def metadata_gate(args, consumer, guard):
    record = {"path": str(args.response_result.resolve()), "sha256": args.response_result_sha256}
    value = consumer.read(consumer.checked(record))
    require(value.get("schema") == consumer.SCHEMA and value.get("stage") == "response-block"
            and value.get("status") == "STAGE_COMPLETE", "A real completed V8 response is required")
    if args.control_validation:
        require(CONTROL_SHAS.get(value.get("case")) == value.get("snapshot_manifest_sha256"), "Exact trusted water/ethanol control source required")
    else:
        require(value.get("case") == "Spinosin" and value.get("atoms") == 75
                and value.get("snapshot_manifest_sha256") == SNAPSHOT_SHA
                and value.get("gradient_gate", {}).get("result", {}).get("sha256") == GRADIENT_SHA,
                "Only the original actual Spinosin C001 source/G is eligible")
        input_doc = consumer.read(consumer.checked({"path": str(args.response_result.resolve().parent / "input.json"),
            "sha256": value["input_json_sha256"]}))
        original_args = input_doc["arguments"]
        consumer.require_control_release(original_args["control_release"], original_args["control_release_sha256"], guard)
    cache_dir = args.response_result.resolve().parent / "stage-cache"
    cache = consumer.read(consumer.checked({"path": str(cache_dir / "manifest.json"), "sha256": value["cache_manifest_sha256"]}))
    physical = cache["physical_binding"]; n, nao = value["atoms"], value["basis_functions"]
    a_ref, b_ref = value["partial_result_dependency"], value["h1_result_dependency"]
    a = consumer.qualify_partial(a_ref, physical, n, nao, guard)
    b = consumer.qualify_stage(b_ref, "h1ao", physical, n, nao, guard)
    c = consumer.qualify_response(record, physical, n, nao, guard, a_ref, b_ref)
    require(c[0] == value and b[0]["partial_result_dependency"] == a_ref, "Qualified source chain differs")
    selected = select_atoms(value["global_atom_ids"], args.atom_index, args.whole_block)
    return {"result": value, "response": c, "partial": a, "h1": b, "physical": physical,
        "record": record, "selected": selected, "n": n, "nao": nao}


def matrix_metrics(np, value):
    return {"max_abs": float(np.max(np.abs(value))),
        "frobenius": float(np.linalg.norm(value)), "RMS": float(np.sqrt(np.mean(value * value)))}


def execute(args):
    guard = load(GUARD, GUARD_SHA, "actual_v8_residual_audit_guard")
    consumer = load(CONSUMER, CONSUMER_SHA, "actual_v8_saved_response_qualifier")
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    require(args.threads == 2, "Use the reviewed two-thread resource policy")
    sole_audit_phase(guard, supervision)
    with closed_main_queue(args, consumer) as main_state_ref:
        return execute_closed(args, guard, consumer, supervision, main_state_ref)


def execute_closed(args, guard, consumer, supervision, main_state_ref):
    gate = metadata_gate(args, consumer, guard)  # No numerical import until actual source closure.
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports") and not output.exists(), "Fresh separate report directory required")
    source_dir = args.response_result.resolve().parent
    require(not output.is_relative_to(source_dir), "Never write into a historical producer directory")
    import numpy as np
    from pyscf import lib
    from pyscf.scf import _vhf, cphf, hf, _response_functions as response
    from pyscf.hessian import rhf, rks
    from pyscf.dft import numint, gen_grid, rks as dft_rks
    from pyscf.grad import rks as grad_rks, rhf as grad_rhf
    from pyscf.lib import linalg_helper
    lib.num_threads(2)
    helper = consumer.load_source(consumer.HELPER, consumer.HELPER_SHA, "residual_audit_frozen_helper")
    w = consumer.load_source(consumer.STATE, consumer.STATE_SHA, "residual_audit_original_state")
    v5 = consumer.load_source(consumer.V5, consumer.V5_SHA, "residual_audit_original_gradient")
    base = consumer.load_source(consumer.BASE, consumer.BASE_SHA, "residual_audit_native_guards")
    core = consumer.load_source(consumer.CORE, consumer.CORE_SHA, "residual_audit_runtime_binding")
    xc_core = consumer.load_source(consumer.XC_CORE, consumer.XC_CORE_SHA, "residual_audit_native_XC_binding")
    output.mkdir(parents=True); log = (output / "pyscf.log").open("x", buffering=1)
    value = gate["result"]
    state = {"schema": SCHEMA, "stage": "saved-response-residual-audit", "status": "PARTIAL", "started_at_utc": utc(),
        "source_worker": consumer.artifact(__file__), "resource_supervision": supervision,
        "snapshot_manifest_sha256": value["snapshot_manifest_sha256"], "source_response": gate["record"],
        "actual_closed_main_pipeline_state": main_state_ref,
        "source_response_closure": gate["response"][3], "selected_atom_ids": gate["selected"],
        "scope": "SELECTED_NUCLEAR_RHS_WITH_FULL_MOLECULAR_AO_MO_OPERATOR", "IR_intensities_status": "NOT_COMPUTED",
        "claim": "SAME_OPERATOR_EQUATION_RESIDUAL_NOT_INDEPENDENT_FULL_DFT_OR_NEW_SOLUTION",
        "source_response_residual_norm_record_unchanged": value["residual_norm"],
        "no_SCF_gradient_Hessian_or_response_solver_run": True, "stable_MF_state_unchanged": False,
        "convergence_threshold_status": "NOT_SET_NOT_A_PASS_FAIL_GATE", "old_producer_modified": False,
        "fresh_native_vind_calls_completed": 0}
    consumer.save(output / "input.json", {**state, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, exclusive=True)
    state["input_json_sha256"] = sha(output / "input.json")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    def emit(event, details=None):
        row = {"utc": utc(), "event": event, "CPU_seconds": time.process_time()-start_cpu,
               "wall_seconds": time.perf_counter()-start_wall, **(details or {})}
        with (output / "events.jsonl").open("a") as f: f.write(json.dumps(row, allow_nan=False)+"\n")
        consumer.save(output / "progress.json", {**state, **row})
    def stop(signum, frame):
        raise base.BudgetStop("Saved-response residual audit resource signal " + str(signum))
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads, native_threads=2,
        emit=lambda d: emit("native_driver_signature", d))
    used_inputs = [gate["record"]]
    try:
        native.install()
        state["gradient_gate"] = v5.qualify_gradient_receipt(value["gradient_gate"]["result"]["path"],
            value["gradient_gate"]["result"]["sha256"], value["snapshot_manifest_sha256"], v5.boot_id())
        mf, binding = w.restore_snapshot(Path(value["snapshot_directory"]),
            expected_manifest_sha256=value["snapshot_manifest_sha256"], expected_source_sha256=consumer.STATE_SHA, stdout=log)
        require(binding == value["source_state_binding"], "Exact source state differs")
        require(helper.case_for([mf.mol.atom_symbol(i) for i in range(mf.mol.natm)], args.control_validation) == value["case"],
            "Restored full molecular identity differs")
        state["actual_imported_native_sources"] = consumer.bind_actual_native_modules(mf._numint, xc_core=xc_core,
            rhf_core=core, rhf=rhf, rks=rks, numint=numint, grad_rks=grad_rks, gen_grid=gen_grid,
            linalg_helper=linalg_helper, mf=mf, dft_rks=dft_rks, hf=hf, grad_rhf=grad_rhf,
            response=response, cphf=cphf, lib=lib, vhf=_vhf)
        hobj = mf.Hessian(); hobj.max_cycle, hobj.level_shift = 50, 0
        settings = {**binding["method_settings"], "threads": 2, "hessian_max_memory_mb": 3000}
        before = v5.physical_state(mf)
        require(v5.stock_operator_gate(mf, hobj) == value["stock_hamiltonian_gate"], "Stock Hamiltonian differs")
        with base.hessian_memory_context(hobj, settings, emit=emit):
            physical = {"schema": consumer.PHYSICAL_SCHEMA, "snapshot_manifest_sha256": value["snapshot_manifest_sha256"],
                "stable_snapshot_binding": binding, "actual_runtime": core.runtime_state_fingerprint(hobj),
                "numerical_policy": consumer.NUMERICAL_POLICY, "frozen_derivative_sources": core._source_identity(rhf),
                "native_threads": 2, "native_krylov_lindep": float(inspect.signature(linalg_helper.krylov).parameters["lindep"].default)}
            require(physical == gate["physical"], "Original complete response operator state differs")
            C, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
            occupied, virtual = occ > 0, occ == 0
            nao, nmo = C.shape; nocc = int(np.count_nonzero(occupied))
            require(C.shape == (gate["nao"], gate["nao"]) and nocc == value["occupied_orbitals"], "Complete AO/MO/occupied spaces required")
            require(all(x.dtype == np.float64 and np.isfinite(x).all() for x in (C, eps, occ))
                and np.all((occ == 0) | (occ == 2)), "Original real closed-shell orbital payload required")
            S = mf.mol.intor_symmetric("int1e_ovlp")
            dS = -mf.mol.intor("int1e_ipovlp", comp=3)
            slices = mf.mol.aoslice_by_atom(); Cocc = C[:, occupied]
            def payload(source, key):
                rows = [row for row in source[1]["arrays"] if row["key"] == key]
                require(len(rows) == 1, "Unique actual cached response/H1 array required")
                row = rows[0]; ref = {"path": str(source[2] / row["file"]), "sha256": row["file_sha256"]}
                p = consumer.checked(ref); v = np.load(p, allow_pickle=False)
                require(v.dtype == np.float64 and list(v.shape) == row["shape"] and np.isfinite(v).all()
                    and base.array_fingerprint(v) == row["payload"], "Actual used cache payload differs")
                used_inputs.append(ref); return v
            W, H, Q, saved_E1 = [], [], [], []
            for atom in gate["selected"]:
                W.append(payload(gate["response"], f"MO1_atom_{atom:03d}"))
                saved_E1.append(payload(gate["response"], f"MO_E1_atom_{atom:03d}"))
                h1 = payload(gate["h1"], f"H1AO_atom_{atom:03d}")
                _, _, p0, p1 = slices[atom]; s1 = np.zeros((3, nao, nao))
                s1[:, p0:p1] += dS[:, p0:p1]
                s1[:, :, p0:p1] += dS[:, p0:p1].transpose(0, 2, 1)
                H.append(np.asarray([C.T @ x @ Cocc for x in h1]))
                Q.append(np.asarray([C.T @ x @ Cocc for x in s1]))
            W, H, Q, saved_E1 = map(np.vstack, (W, H, Q, saved_E1))
            U = np.einsum("pa,xai->xpi", C.T @ S, W)
            delta = eps[virtual, None] - eps[occupied]
            require(np.isfinite(delta).all() and np.all(delta != 0), "Nonzero finite native energy denominators required")
            emit("fresh_full_native_operator_build_begin")
            fx = rhf.gen_vind(mf, C, occ)
            emit("fresh_full_native_operator_build_end")
            emit("fresh_native_vind_at_saved_U_begin", {"RHS": len(U)})
            V = fx(U).reshape(U.shape)
            state["fresh_native_vind_calls_completed"] = 1
            emit("fresh_native_vind_at_saved_U_end", {"RHS": len(U)})
            hs = H - Q * eps[occupied]
            r_virtual = delta * U[:, virtual] + hs[:, virtual] + V[:, virtual]
            r_preconditioned = r_virtual / delta
            r_occupied = U[:, occupied] + .5 * Q[:, occupied]
            e1_fresh = (hs + V)[:, occupied] + U[:, occupied] * (eps[occupied, None] - eps[occupied])
            e1_difference = e1_fresh - saved_E1
            reconstructed = np.einsum("ap,xpi->xai", C, U)
            normalization = U[:, occupied] + U[:, occupied].transpose(0, 2, 1) + Q[:, occupied]
            rows = []
            for i in range(len(U)):
                parts = (delta * U[i, virtual], hs[i, virtual], V[i, virtual])
                denominator = sum(float(np.linalg.norm(p)) for p in parts)
                row = {"atom_id": gate["selected"][i//3], "axis": "xyz"[i%3],
                    "virtual_residual_unit": "Eh/Bohr", "preconditioned_and_occupied_unit": "1/Bohr",
                    "virtual": matrix_metrics(np, r_virtual[i]), "preconditioned": matrix_metrics(np, r_preconditioned[i]),
                    "occupied_gauge": matrix_metrics(np, r_occupied[i]),
                    "normalization": matrix_metrics(np, normalization[i]), "E1_difference_unit": "Eh/Bohr",
                    "E1_fresh_minus_saved": matrix_metrics(np, e1_difference[i]),
                    "AO_coefficient_reconstruction_unit": "AO_coefficient/Bohr",
                    "AO_coefficient_reconstruction": matrix_metrics(np, reconstructed[i]-W[i]),
                    "relative_term_scaled_residual": float(np.linalg.norm(r_virtual[i]))/denominator if denominator else None,
                    "relative_term_scaled_residual_denominator_zero": denominator == 0,
                    "no_new_convergence_threshold": True}
                require(all(np.isfinite(p).all() for p in (r_virtual[i], r_preconditioned[i], r_occupied[i], e1_difference[i])), "Nonfinite residual")
                rows.append(row)
            state.update(status=SUCCESS, metrics_by_nuclear_RHS=rows, checked_nuclear_RHS=len(U),
                total_molecular_nuclear_RHS=3*gate["n"], all_molecular_RHS_checked=len(U) == 3*gate["n"],
                source_MO_metric_max_abs=float(np.max(np.abs(C.T @ S @ C - np.eye(nmo)))),
                source_MO_metric_unit="dimensionless",
                minimum_absolute_energy_gap_Eh=float(np.min(np.abs(delta))),
                native_requested_trial_tolerance=value["requested_effective_CPHF_tolerance"],
                actual_native_krylov_lindep=physical["native_krylov_lindep"],
                native_trial_criterion_is_not_measured_residual=True,
                saved_E1_uses_pre_final_fixed_point_potential=True,
                actual_formula="R_ai=(eps_a-eps_i)U_ai+H_ai-eps_i*Q_ai+freshV_ai(U)",
                fresh_operator_source="FROZEN_NATIVE_RHF_GEN_VIND_FULL_RKS_RESPONSE", no_new_solver_or_optimization=True)
            require(core.runtime_state_fingerprint(hobj) == physical["actual_runtime"], "Response operator changed")
        require(v5.physical_state(mf) == before, "Exact source MF changed")
        for ref in used_inputs: consumer.checked(ref)
        repeat = metadata_gate(args, consumer, guard)
        require(repeat["result"] == gate["result"] and repeat["physical"] == gate["physical"]
            and repeat["selected"] == gate["selected"], "Closed source chain changed during residual audit")
        state["complete_closed_source_chain_rechecked_after_measurement"] = True
        state["stable_MF_state_unchanged"] = True
    except BaseException as error:
        state.update(status="PARTIAL", exception_type=type(error).__name__, stop_reason=str(error))
        (output / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        native.restore(); state["native_direct_driver_control"] = native.summary()
        if state["native_direct_driver_control"].get("restoration_failures") != 0 or not state["native_direct_driver_control"].get("original_driver_restored"):
            state.update(status="PARTIAL", stop_reason="Native operator restoration failed")
        for signum, handler in handlers.items(): signal.signal(signum, handler)
        state.update(completed_at_utc=utc(), process_cpu_seconds=time.process_time()-start_cpu,
            wall_seconds=time.perf_counter()-start_wall)
        log.close(); consumer.save(output / "result.json", state, exclusive=True)
    return state


def preflight(output):
    require(not output.exists(), "Fresh metadata-only preparation directory required")
    refs = []
    for file, expected in ((GUARD, GUARD_SHA), (CONSUMER, CONSUMER_SHA)):
        require(sha(file) == expected, "Frozen dependency differs")
        refs.append({"path": str(file), "sha256": expected})
    output.mkdir(parents=True)
    value = {"schema": SCHEMA, "status": "NOT_RUN", "source_worker": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "source_dependencies": refs, "quantum_operator_evaluations": 0, "actual_residuals": "NOT_RUN",
        "target_arrays_read": False, "numerical_packages_imported": False,
        "IR_intensities": "NOT_COMPUTED", "old_results_changed": False,
        "requirement": "Root-only real V8 guardian; all main Q closed; real healthy response refs; no arbitrary convergence threshold"}
    with (output / "preflight.json").open("x") as f: json.dump(value, f, indent=2); f.write("\n")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--execute", action="store_true")
    p.add_argument("--response-result", type=Path); p.add_argument("--response-result-sha256")
    p.add_argument("--main-pipeline-state", type=Path); p.add_argument("--main-pipeline-state-sha256")
    p.add_argument("--atom-index", type=int); p.add_argument("--whole-block", action="store_true")
    p.add_argument("--control-validation", action="store_true"); p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.execute:
        require(args.response_result is not None and args.response_result_sha256, "Explicit actual result and SHA required")
        value = execute(args)
    else:
        require(args.response_result is None and args.main_pipeline_state is None and not args.control_validation
            and args.atom_index is None and not args.whole_block,
            "Metadata preparation does not consume response inputs")
        value = preflight(args.output)
    print(json.dumps({k: value.get(k) for k in ("status", "checked_nuclear_RHS", "fresh_native_vind_calls_completed")}))
    return 0 if value["status"] in (SUCCESS, "NOT_RUN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
