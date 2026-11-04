"""Supervised full RKS Hessian from a verified exact MF; restartable RHF units.

Stock reference, same-state gradient verification and incremental segments are
separate phases. No new SCF is performed. A checkpoint stop is not a frequency
result. Completed frequencies and a positive-frequency minimum are distinct.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
INCREMENTAL_PATH = ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py"
INCREMENTAL_SHA = "ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55"
BASE_PATH = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
BASE_SHA = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"
SUPERVISOR_PATH = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"
SUPERVISOR_SHA = "33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee"
SCHEMA = "research4-resumable-full-rks-hessian-result-v1"
EXECUTION_SCHEMA = "research4-rhf-unit-producer-v1"
GRADIENT_TOLERANCE = 1e-8
SUCCESS = {"DONE", "CHECKPOINT_SEGMENT_COMPLETE", "FREQUENCIES_COMPUTED"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_source(path, expected, name):
    if sha(path) != expected:
        raise ValueError("Frozen numerical dependency changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_module():
    return load_source(STATE_PATH, STATE_SHA, "research4_exact_state_hessian_dependency")


def boot_id():
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if len(value) != 36:
        raise ValueError("A Linux boot identity is required for restore-gradient gating")
    return value


def healthy_closed_row(row):
    return (row.get("status") == "EXITED" and row.get("exit_code") == 0
            and isinstance(row.get("ended_at_utc"), str)
            and all(row.get(key) is False for key in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed"))
            and not row.get("residual_processes_require_review_after_main_payload_exit")
            and type(row.get("actual_cpu_seconds")) in (float, int)
            and math.isfinite(row["actual_cpu_seconds"]) and row["actual_cpu_seconds"] >= 0)


def verify_closed_execution(evidence, *, expected_snapshot_sha, expected_output=None):
    """Inspect actual immutable proof, closed V5 ledger, source, log and result."""
    w = state_module()
    if evidence.get("schema") != EXECUTION_SCHEMA or evidence.get("snapshot_manifest_sha256") != expected_snapshot_sha:
        raise ValueError("Unit producer does not bind the exact source MF")
    source = evidence.get("source_worker", {})
    if w.checked_artifact(source) != Path(__file__).resolve() or source.get("sha256") != sha(__file__):
        raise ValueError("Unexpected unit producer entry source")
    supervision = evidence["resource_supervision"]
    proof_path = Path(supervision["proof_path"]).resolve()
    if sha(proof_path) != supervision["proof_sha256"]:
        raise ValueError("Unit producer proof bytes changed")
    proof = json.loads(proof_path.read_text())
    if (proof != supervision["proof"] or proof.get("schema") != "research4-pyscf-actual-cpu-supervisor-proof-v5"
            or proof.get("supervisor_script_sha256") != SUPERVISOR_SHA
            or supervision.get("live_parent_process_and_source_verified") is not True):
        raise ValueError("Unit lacks an actual verified V5 producer proof")
    ledger_path = w.resolve_path(proof["ledger_path"])
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("schema") != "research4-pyscf-actual-cpu-budget-v5" or ledger.get("runner_sha256") != SUPERVISOR_SHA:
        raise ValueError("Unit producer ledger differs")
    rows = [r for r in ledger.get("phases", []) if r.get("phase") == proof.get("phase")]
    if len(rows) != 1 or not healthy_closed_row(rows[0]):
        raise ValueError("Unit producer has not closed with healthy EXITED/0")
    row = rows[0]
    if row.get("supervisor_proof_sha256") != sha(proof_path) or w.resolve_path(row["supervisor_proof_path"]) != proof_path:
        raise ValueError("Closed producer row/proof mismatch")
    log = w.resolve_path(row["log"])
    if sha(log) != row.get("log_sha256"):
        raise ValueError("Closed unit producer log changed")
    output = w.resolve_path(evidence["output_directory"])
    if expected_output and output != Path(expected_output).resolve():
        raise ValueError("Unit producer output changed")
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    if (result.get("schema") != SCHEMA or result.get("status") not in SUCCESS
            or result.get("source_worker") != source or result.get("resource_supervision") != supervision
            or result.get("snapshot_manifest_sha256") != expected_snapshot_sha
            or result.get("phase") != evidence.get("phase")):
        raise ValueError("Unit has no matching successful producer result")
    guard = result.get("native_direct_driver_control", {})
    if guard.get("restoration_failures") != 0 or guard.get("original_driver_restored") is not True:
        raise ValueError("Producer native driver did not restore cleanly")
    command = row.get("command", [])
    if str(Path(__file__).resolve()) not in [str(w.resolve_path(t)) for t in command if isinstance(t, str) and t.endswith(".py")]:
        raise ValueError("Actual producer command does not invoke this reviewed entry")
    if "--output" not in command or w.resolve_path(command[command.index("--output")+1]) != output:
        raise ValueError("Actual producer command does not bind its result directory")
    if not result.get("stable_MF_state_unchanged"):
        raise ValueError("Producer did not verify unchanged source MF")
    return {"proof_sha256": sha(proof_path), "producer_result_sha256": sha(result_path),
            "phase": row["phase"], "actual_cpu_seconds": row["actual_cpu_seconds"],
            "closed_row_sha256": hashlib.sha256(canonical(row).encode()).hexdigest()}


def load_forensic_release(path, expected_sha, snapshot_sha):
    if path is None:
        if expected_sha:
            raise ValueError("Forensic SHA supplied without release")
        return None
    path = Path(path).resolve()
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError("Explicit root forensic release SHA required")
    value = json.loads(path.read_text())
    if (value.get("schema") != "research4-explicit-root-rhf-unit-forensic-release-v1"
            or value.get("explicit_root_forensic_approval") is not True
            or value.get("snapshot_manifest_sha256") != snapshot_sha
            or value.get("hessian_worker_sha256") != sha(__file__)
            or not value.get("reason") or not isinstance(value.get("allowed_records"), list)):
        raise ValueError("Unsupported root forensic release")
    value["release_path"] = str(path)
    value["release_sha256"] = expected_sha
    return value


def verify_cached_unit_producers(cache_dir, snapshot_sha, *, forensic_release=None):
    """A numeric completion marker alone is not a successful producer phase."""
    cache_dir = Path(cache_dir).resolve()
    units = cache_dir / "units"
    if not units.exists():
        return []
    if units.is_symlink():
        raise ValueError("Symlink checkpoint directory rejected")
    accepted = []
    complete_paths = sorted(units.glob("*/attempt-*/complete.json"))
    if len(complete_paths) > 1000:
        raise ValueError("Unexpected number of completed RHF units")
    for path in complete_paths:
        if path.is_symlink() or path.stat().st_size > 131072:
            raise ValueError("Unsafe RHF completion record")
        record = json.loads(path.read_text())
        evidence = record.get("execution_evidence", {})
        if evidence.get("phase") != "hessian-segment":
            raise ValueError("RHF unit producer phase is not a supervised segment")
        try:
            producer = verify_closed_execution(evidence, expected_snapshot_sha=snapshot_sha)
            accepted.append({"record": str(path), "sha256": sha(path), "qualification": "CLOSED_HEALTHY_V5_PRODUCER", "producer": producer})
        except (ValueError, KeyError, OSError, IndexError) as exc:
            allowed = [] if forensic_release is None else forensic_release["allowed_records"]
            match = [r for r in allowed if Path(r.get("path", "")).resolve() == path
                     and r.get("sha256") == sha(path)
                     and r.get("producer_proof_sha256") == evidence.get("resource_supervision", {}).get("proof_sha256")]
            if len(match) != 1:
                raise ValueError("RHF unit producer not independently qualified: " + str(path)) from exc
            # Explicit forensic override never rewrites the failed/unknown exit.
            accepted.append({"record": str(path), "sha256": sha(path), "qualification": "EXPLICIT_ROOT_FORENSIC_RELEASE",
                             "original_closed_parent_check_failure": str(exc),
                             "release_sha256": forensic_release["release_sha256"]})
    return accepted


def qualify_gradient_receipt(path, expected_sha, snapshot_sha, current_boot):
    path = Path(path).resolve()
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError("A frozen same-state gradient-check result SHA is required")
    result = json.loads(path.read_text())
    if (result.get("schema") != SCHEMA or result.get("phase") != "restore-gradient-check"
            or result.get("status") != "DONE" or result.get("boot_id") != current_boot
            or result.get("snapshot_manifest_sha256") != snapshot_sha
            or result.get("restore_gradient_comparison_passed") is not True
            or result.get("final_gradient_thresholds_passed") is not True):
        raise ValueError("Same-boot exact-state gradient gate not satisfied")
    evidence = {"schema": EXECUTION_SCHEMA, "source_worker": result["source_worker"],
                "resource_supervision": result["resource_supervision"], "output_directory": str(path.parent),
                "snapshot_manifest_sha256": snapshot_sha, "phase": result["phase"]}
    return {"result": state_module().artifact_record(path),
            "producer": verify_closed_execution(evidence, expected_snapshot_sha=snapshot_sha),
            "scope": "SAME_BOOT_RESTORED_STATE_NO_NEW_SCF"}


def physical_state(mf):
    w = state_module()
    arrays = w.snapshot_arrays(mf, np.zeros((mf.mol.natm, 3)))
    return {"molecule": w.molecule_physical_fingerprint(mf.mol),
            "arrays": {k: w.frozen_base().array_fingerprint(v) for k, v in arrays.items() if k != "gradient"},
            "energy_hartree": float(mf.e_tot), "grid_settings": w.grid_settings(mf.grids),
            "xc": mf.xc, "nlc": mf.nlc, "disp": getattr(mf, "disp", None),
            "numint_cutoff": float(mf._numint.cutoff), "numint_omega": mf._numint.omega}


def stock_operator_gate(mf, hobj):
    """Disallow decorators/custom hcore paths before sharing derivative units."""
    import pyscf
    from pyscf.scf import hf
    from pyscf.hessian import rhf, rks
    from pyscf.dft import rks as dft_rks
    from pyscf.grad import rhf as grad_rhf
    for owner in (mf, mf.mol, hobj):
        for name in ("with_x2c", "with_solvent", "mm_mol", "with_mm", "with_qmmm", "with_dftd3"):
            if getattr(owner, name, None) is not None:
                raise ValueError("Decorated core Hamiltonian is outside the reviewed scope: " + name)
    if getattr(mf.mol, "_pseudo", None) or getattr(mf.mol, "pseudo", None) or mf.mol.has_ecp():
        raise ValueError("Pseudo potentials/ECPs are outside the reviewed scope")
    if type(hobj) is not rks.Hessian or type(hobj).__module__ != "pyscf.hessian.rks":
        raise ValueError("Only the original stock non-DF RKS Hessian is allowed")
    identities = ((mf.get_hcore, hf.SCF.get_hcore),
                  (mf.nuc_grad_method, dft_rks.RKS.nuc_grad_method),
                  (hobj.get_hcore, rhf.HessianBase.get_hcore),
                  (hobj.hcore_generator, rhf.HessianBase.hcore_generator))
    if any(getattr(actual, "__func__", None) is not reference for actual, reference in identities):
        raise ValueError("Custom core Hamiltonian or gradient factory hook rejected")
    gradient = mf.nuc_grad_method()  # construction only; never call its kernel/core
    if (getattr(gradient.hcore_generator, "__func__", None) is not grad_rhf.hcore_generator
            or getattr(gradient.get_hcore, "__func__", None) is not grad_rhf.GradientsBase.get_hcore):
        raise ValueError("Custom gradient core Hamiltonian hook rejected")
    expected = {"MF.get_hcore": (mf.get_hcore, hf.__file__),
                "Hessian.get_hcore": (hobj.get_hcore, rhf.__file__),
                "Hessian.hcore_generator": (hobj.hcore_generator, rhf.__file__)}
    origins = {}
    for name, (function, source) in expected.items():
        actual = Path(inspect.getsourcefile(function)).resolve()
        if actual != Path(source).resolve():
            raise ValueError("Custom core Hamiltonian hook rejected: " + name)
        origins[name] = {"file": str(actual), "sha256": sha(actual), "qualified_name": function.__qualname__}
    original = rhf._partial_hess_ejk
    if original.__name__ != "_partial_hess_ejk" or Path(inspect.getsourcefile(original)).resolve() != Path(rhf.__file__).resolve():
        raise ValueError("Stock reference helper is already patched")
    if getattr(mf, "with_df", None) is not None or getattr(mf, "disp", None) is not None:
        raise ValueError("Only plain non-DF PBE without dispersion is allowed")
    return origins


def publish_modes(out, mf, raw, state, base):
    from pyscf.hessian import thermo
    raw = np.asarray(raw)
    base.atomic_npy(out / "hessian_hartree_per_bohr2.npy", raw)
    used, asymmetry = base.prepare_hessian_for_analysis(raw, mf.mol.natm)
    base.atomic_npy(out / "hessian_used_for_analysis.npy", used)
    analysis = thermo.harmonic_analysis(mf.mol, used, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
    info = base.frequencies_payload(analysis["freq_wavenumber"], analysis["norm_mode"], mf.mol.natm)
    base.atomic_npy(out / "normal_modes.npy", analysis["norm_mode"])
    base.atomic_npy(out / "reduced_mass_amu.npy", analysis["reduced_mass"])
    with (out / "frequencies.csv").open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["mode", "frequency_cm1", "frequency_thz", "imaginary", "IR_intensity_status"])
        for index, frequency in enumerate(info["frequencies_cm1"], 1):
            writer.writerow([index, frequency, frequency*base.CM1_TO_THZ, frequency < 0, "NOT_COMPUTED"])
    state.update(info, **base.frequency_minimum_checks(info), frequencies_status="DONE",
                 status="FREQUENCIES_COMPUTED", hessian_max_symmetry_difference=asymmetry,
                 complete_RHF_helper=True, complete_global_Hessian=True,
                 raw_hessian_sha256=sha(out / "hessian_hartree_per_bohr2.npy"),
                 analysis_hessian_sha256=sha(out / "hessian_used_for_analysis.npy"),
                 full_system_CPHF_returned_normally=True,
                 measured_CPHF_residual_norm="NOT_COMPUTED",
                 normal_return_is_not_a_measured_residual=True,
                 harmonic_analysis_excludes_translations_and_rotations=True,
                 mass_convention="PySCF default isotope-average atomic masses",
                 raw_hessian_symmetrization="(raw + raw.transpose(1,0,3,2))/2 only after raw asymmetry<=1e-5")


def execute(args):
    w = state_module()
    base = load_source(BASE_PATH, BASE_SHA, "research4_v8_resumable_hessian_dependency")
    supervisor = load_source(SUPERVISOR_PATH, SUPERVISOR_SHA, "research4_v5_resumable_hessian_supervisor")
    incremental = load_source(INCREMENTAL_PATH, INCREMENTAL_SHA, "research4_incremental_hessian_dependency")
    supervision = supervisor.verify_resource_supervision(expected_threads=args.threads)
    w.install_sites()
    from pyscf import lib
    from pyscf.scf import _vhf
    from pyscf.grad.rks import Gradients
    from pyscf.hessian import rhf
    from pyscf.lib import linalg_helper
    lib.num_threads(args.threads)
    if not 1 <= args.threads <= 4 or not 1 <= args.max_new_units <= 150:
        raise ValueError("Bounded threads and finite new-unit count required")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Preserve prior Hessian reports; select a fresh output directory")
    out.mkdir(parents=True)
    log = (out / "pyscf.log").open("x", buffering=1)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    state = {"schema": SCHEMA, "phase": args.phase, "status": "PARTIAL", "started_at_utc": utc(),
             "source_worker": w.artifact_record(__file__), "resource_supervision": supervision,
             "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "snapshot_directory": str(args.snapshot.resolve()),
             "state_worker_sha256": STATE_SHA, "incremental_module_sha256": INCREMENTAL_SHA,
             "frozen_V8_sha256": BASE_SHA, "boot_id": boot_id(),
             "frequencies_status": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED",
             "complete_RHF_helper": False, "complete_global_Hessian": False,
             "no_new_SCF_performed": True, "RKS_XC_H1_CPHF_cache_restore": "NOT_IMPLEMENTED_SAVE_ONLY",
             "RHF_cache_restore": "NOT_ENTERED", "stable_MF_state_unchanged": False}
    w.save_json(out / "input.json", {**state, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, exclusive=True)
    def emit(event, details=None, **values):
        record = {"utc": utc(), "event": event, "process_cpu_seconds": time.process_time()-started_cpu,
                  "wall_seconds": time.perf_counter()-started_wall, **(details or {}), **values}
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
        w.save_json(out / "progress.json", {**state, **record})
    def stop(signum, _frame):
        raise base.BudgetStop("Supervisor resource signal " + str(signum))
    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    guard = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads, native_threads=args.threads,
             emit=lambda details: emit("native_driver_signature", details))
    mf = before = context = None
    try:
        guard.install()
        mf, binding = w.restore_snapshot(args.snapshot, expected_manifest_sha256=args.snapshot_manifest_sha256,
                                         expected_source_sha256=STATE_SHA, stdout=log)
        state["state_restoration_status"] = mf._research4_restoration_status
        state["source_state_binding"] = binding
        state["identity"] = binding["producer"]["identity"]
        state["versions"] = binding["environment"]
        state["settings"] = {**binding["method_settings"], "threads": args.threads,
                             "native_direct_driver_threads": args.threads,
                             "hessian_max_memory_mb": 3000, "hessian_max_cycle": 50,
                             "hessian_level_shift": 0, "hessian_response_block_atoms": 8,
                             "engineering_worker_version": "RESUMABLE_RHF_FULL_STOCK_RKS_V1"}
        state["geometry_optimized"] = True
        state["scf_converged"] = True
        state["energy_hartree"] = float(mf.e_tot)
        state["atoms"] = int(mf.mol.natm)
        state["basis_functions"] = int(mf.mol.nao_nr())
        before = physical_state(mf)
        base.atomic_npy(out / "normal_mode_geometry_angstrom.npy", mf.mol.atom_coords(unit="Angstrom"))
        base.atomic_npy(out / "normal_mode_geometry_bohr.npy", mf.mol.atom_coords(unit="Bohr"))
        symbols = [mf.mol.atom_symbol(i) for i in range(mf.mol.natm)]
        (out / "normal_mode_geometry.xyz").write_text(base.xyz_text(symbols, mf.mol.atom_coords(unit="Angstrom"), "Restored exact MF geometry; no new optimization"))
        hobj = mf.Hessian()
        hobj.max_cycle, hobj.level_shift = 50, 0
        state["stock_hamiltonian_gate"] = stock_operator_gate(mf, hobj)
        if args.phase == "restore-gradient-check":
            gradient_cls = base.direct_gradient_type(Gradients, emit, memory_mb=lambda: lib.current_memory()[0])
            gradient_method = gradient_cls(mf)
            gradient_method.grid_response = False
            gradient = np.asarray(gradient_method.kernel())
            base.atomic_npy(out / "gradient_hartree_per_bohr.npy", gradient)
            saved = np.load(args.snapshot / "gradient.npy", allow_pickle=False)
            if gradient.shape != saved.shape or not np.isfinite(gradient).all():
                raise ValueError("Restored gradient dimensions differ from source")
            difference = float(np.max(np.abs(gradient-saved)))
            state.update(base.stationary_checks(gradient),
                         restored_gradient_max_abs_difference_hartree_per_bohr=difference,
                         fixed_gradient_comparison_tolerance=GRADIENT_TOLERANCE,
                         restore_gradient_comparison_passed=difference <= GRADIENT_TOLERANCE,
                         gradient_npy_sha256=sha(out / "gradient_hartree_per_bohr.npy"))
            if not state["restore_gradient_comparison_passed"] or not state["final_gradient_thresholds_passed"]:
                raise RuntimeError("Same-state restored gradient check failed")
            state["status"] = "DONE"
        else:
            if args.gradient_check_result is None:
                raise ValueError("A closed same-boot restored-gradient gate is required")
            state["gradient_gate"] = qualify_gradient_receipt(args.gradient_check_result, args.gradient_check_result_sha256,
                                                              args.snapshot_manifest_sha256, state["boot_id"])
            state["final_gradient_thresholds_passed"] = True
            cache_root = args.rhf_cache.resolve()
            if not cache_root.is_relative_to(ROOT / "outputs") or cache_root == ROOT / "outputs":
                raise ValueError("A dedicated ignored outputs/RHF cache directory is required")
            forensic = load_forensic_release(args.forensic_unit_release, args.forensic_unit_release_sha256,
                                               args.snapshot_manifest_sha256)
            if args.phase == "hessian-segment":
                state["prior_RHF_unit_producer_checks"] = verify_cached_unit_producers(cache_root, args.snapshot_manifest_sha256,
                                                                                       forensic_release=forensic)
            else:
                state["stock_partial_helper_unpatched_reference"] = True
            execution = {"schema": EXECUTION_SCHEMA, "source_worker": state["source_worker"],
                         "resource_supervision": supervision, "output_directory": str(out),
                         "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "phase": args.phase}
            saveonly = ROOT / "outputs/pyscf-spinosin-resumable-20261104-v5/hessian-saveonly" / out.name
            if saveonly.exists():
                raise FileExistsError("Save-only cache directory already exists")
            with base.hessian_memory_context(hobj, state["settings"], emit=emit):
                context = incremental.IncrementalPureJContext(hobj, cache_root, binding,
                    max_new_units=args.max_new_units, execution_evidence=execution) if args.phase == "hessian-segment" else nullcontext()
                with context:
                    if args.phase == "hessian-segment":
                        state["RHF_cache_restore"] = "SAME_STATE_COMPLETE_PREFIX_RESTORED_WITH_VERIFIED_PRODUCERS"
                        state["incremental_before"] = context.progress()
                    raw = base._run_checkpointed_hessian_in_memory_context(hobj, saveonly,
                        out / "hessian_cache_manifest.json", state["settings"], block_atoms=8,
                        versions=state["versions"], response_operator_factory=rhf.gen_vind,
                        native_lindep=float(inspect.signature(linalg_helper.krylov).parameters["lindep"].default),
                        input_evidence={"source_snapshot_sha256": args.snapshot_manifest_sha256,
                                        "actual_entry_source_sha256": sha(__file__),
                                        "consumer_resource_proof_sha256": supervision["proof_sha256"],
                                        "geometry_optimized": True, "final_gradient_thresholds_passed": True,
                                        "RHF_partial_only_cross_process_restore": args.phase == "hessian-segment"})
                    if args.phase == "hessian-segment":
                        state["incremental_after"] = context.progress()
            publish_modes(out, mf, raw, state, base)
        state["stable_MF_state_unchanged"] = physical_state(mf) == before
        if not state["stable_MF_state_unchanged"]:
            raise RuntimeError("Original geometry/MOs/actual grid changed during the consumer phase")
    except incremental.CheckpointSegmentComplete as exc:
        state.update(status="CHECKPOINT_SEGMENT_COMPLETE", incremental_after=exc.progress,
                     scientific_stage_status="PARTIAL_RHF_CHECKPOINT_ONLY", frequencies_status="NOT_RUN",
                     complete_RHF_helper=False, complete_global_Hessian=False)
        state["stable_MF_state_unchanged"] = mf is not None and physical_state(mf) == before
        if not state["stable_MF_state_unchanged"]:
            state.update(status="PARTIAL", stop_reason="Source MF changed at checkpoint stop")
    except BaseException as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), exception_type=type(exc).__name__)
        (out / "exception_traceback.txt").write_text(traceback.format_exc())
        if mf is not None and before is not None:
            try:
                state["stable_MF_state_unchanged"] = physical_state(mf) == before
            except Exception:
                state["stable_MF_state_unchanged"] = False
    finally:
        guard.restore()
        state["native_direct_driver_control"] = guard.summary()
        if state["native_direct_driver_control"]["restoration_failures"]:
            state.update(status="PARTIAL", stop_reason="Native driver restoration failed")
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if (out / "hessian_cache_manifest.json").exists():
            state["hessian_cache_manifest_sha256"] = sha(out / "hessian_cache_manifest.json")
        state.update(completed_at_utc=utc(), wall_seconds=time.perf_counter()-started_wall,
                     process_cpu_seconds=time.process_time()-started_cpu,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        log.flush()
        log.close()
        w.save_json(out / "result.json", state, exclusive=True)
    print(json.dumps({key: state.get(key) for key in ("phase", "status", "identity", "frequencies_status", "normal_mode_count", "stop_reason")}))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("restore-gradient-check", "stock-hessian-reference", "hessian-segment"))
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-units", type=int, default=10)
    parser.add_argument("--rhf-cache", type=Path, default=ROOT / "outputs/pyscf-spinosin-resumable-20261104-v5/rhf-cache")
    parser.add_argument("--gradient-check-result", type=Path)
    parser.add_argument("--gradient-check-result-sha256")
    parser.add_argument("--forensic-unit-release", type=Path)
    parser.add_argument("--forensic-unit-release-sha256")
    args = parser.parse_args()
    result = execute(args)
    raise SystemExit(0 if result["status"] in SUCCESS else 2)


if __name__ == "__main__":
    main()
