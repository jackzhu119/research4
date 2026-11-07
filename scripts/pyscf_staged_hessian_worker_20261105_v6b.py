"""V6B contingency: complete full RKS partial, full H1AO, then stock global kernel.

Preparation/import is stdlib-only. Numerical work requires --execute and an
actual reviewed V6B parent. V5 snapshots/gradient proofs remain genuine V5.
Caches from failed/RUNNING producers are rejected; no forensic compatibility.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
STATE = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
V5 = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
V5_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
BASE = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
BASE_SHA = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"
CORE = ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py"
CORE_SHA = "ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55"
SPINOSIN_SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
SPINOSIN_GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
SCHEMA = "research4-staged-full-rks-hessian-result-v6b"
CACHE_SCHEMA = "research4-complete-full-rks-stage-cache-v6b"
PHYSICAL_SCHEMA = "research4-full-rks-physical-binding-v6b"
RHF_EVIDENCE_SCHEMA = "research4-stage-rhf-unit-producer-v6b"
STAGES = ("rks-partial", "h1ao", "global")
SUCCESS = {"rks-partial": ("STAGE_COMPLETE",), "h1ao": ("STAGE_COMPLETE",), "global": ("FREQUENCIES_COMPUTED",)}
NUMERICAL_POLICY = {"xc": "PBE", "basis": "def2-svp", "density_fitting": False,
                    "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "full_AO_MO_space": True,
                    "grid_changed": False, "MF_Hessian_memory_mb": 3000,
                    "stock_hess_elec_max_memory_argument_mb": 4000,
                    "conv_tol_cpscf": 1e-8, "hessian_max_cycle": 50, "level_shift": 0,
                    "response_block_atoms": 8}


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked(record):
    path = Path(record["path"])
    require(not path.is_symlink() and path.is_file(), "Missing/symlink stage artifact")
    require(sha(path) == record["sha256"], "Closed artifact SHA changed: " + str(path))
    return path.resolve()


def read(path):
    return json.loads(Path(path).read_text())


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
            handle.flush(); os.fsync(handle.fileno())
        fsync_directory(path.parent)
    else:
        temporary = path.with_name(path.name+".writing")
        with temporary.open("w") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)


def load_source(path, expected, name):
    require(len(expected) == 64 and sha(path) == expected, "Unfrozen/changed dependency: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_atoms(atmlst, n):
    values = list(range(n)) if atmlst is None else list(atmlst)
    require(values == list(range(n)), "Only complete original-order atoms are supported")
    return values


def validate_cache_call(kind, values, atoms, mf, same_array):
    """Normalize stock None defaults; validate explicit arguments, not only MF."""
    require(kind in ("partial", "h1", "nuclear"), "Unsupported cached contribution")
    complete_atoms(values.get("atmlst"), atoms)
    if kind == "nuclear":
        require(values.get("mol") is None or values["mol"] is mf.mol, "Nuclear call uses another molecule")
        return
    for key in (("mo_energy", "mo_coeff", "mo_occ") if kind == "partial" else ("mo_coeff", "mo_occ")):
        require(key in values and (values[key] is None or same_array(values[key], getattr(mf, key))),
                "Stock cache call changed explicit full MO arguments")
    if kind == "partial":
        require(values.get("max_memory") == 4000, "Stock hess_elec memory argument changed")
    else:
        require(values.get("chkfile") is None, "H1 checkpoint-output calls unsupported")


def validate_manifest(manifest, *, stage, physical_binding, atoms, nao):
    """Metadata only; invoked before any stage NPY is read."""
    require(stage in ("rks-partial", "h1ao") and manifest.get("schema") == CACHE_SCHEMA
            and manifest.get("stage") == stage and manifest.get("complete") is True,
            "Unsupported/incomplete stage cache; failed native arrays are quarantined")
    require(manifest.get("physical_binding") == physical_binding
            and manifest.get("physical_binding_sha256") == digest(physical_binding),
            "Full-system physical state or numerical policy differs")
    expected = ([{"key": "full_RKS_partial", "shape": [atoms, atoms, 3, 3]}] if stage == "rks-partial" else
                [{"key": f"H1AO_atom_{i:03d}", "shape": [3, nao, nao]} for i in range(atoms)])
    arrays = manifest.get("arrays", [])
    require(len(arrays) == len(expected), "Complete stage array count differs")
    for row, wanted in zip(arrays, expected):
        require(row.get("key") == wanted["key"] and row.get("shape") == wanted["shape"]
                and row.get("dtype") == "float64" and isinstance(row.get("file"), str)
                and Path(row["file"]).name == row["file"] and row["file"].endswith(".npy")
                and len(row.get("file_sha256", "")) == 64 and isinstance(row.get("payload"), dict),
                "Stage array identity/shape/dtype/hash differs")
    require(len({r["file"] for r in arrays}) == len(arrays), "Duplicate stage payload path")
    require(manifest.get("includes_nuclear_hessian") is False
            and manifest.get("includes_CPHF_response") is False, "Do not double-add nuclear or response terms")


def qualify_stage(result_record, stage, physical_binding, atoms, nao, guard):
    """Healthy source/proof/log/result and complete manifest before NPY access."""
    path = checked(result_record)
    require(path.name == "result.json" and path.is_relative_to(ROOT / "reports"), "Closed report result required")
    value = read(path)
    require(value.get("schema") == SCHEMA and value.get("stage") == stage
            and value.get("phase") == stage and value.get("status") in SUCCESS[stage]
            and value.get("no_new_SCF_performed") is True and value.get("complete_RKS_partial") is True
            and value.get("atoms") == atoms and value.get("basis_functions") == nao
            and value.get("physical_binding_sha256") == digest(physical_binding)
            and (stage != "h1ao" or value.get("complete_H1AO") is True), "Unsupported/incomplete V6B stage producer")
    evidence = {"source_worker": value["source_worker"], "resource_supervision": value["resource_supervision"],
                "output_directory": str(path.parent), "snapshot_manifest_sha256": value["snapshot_manifest_sha256"], "stage": stage}
    closure = guard.verify_closed_phase(evidence, expected_worker_path=Path(__file__).resolve(),
        expected_worker_sha256=sha(__file__), expected_result_schema=SCHEMA,
        allowed_result_statuses=SUCCESS[stage], expected_output=path.parent,
        expected_snapshot_sha256=physical_binding["snapshot_manifest_sha256"])
    require(closure["producer_result_sha256"] == result_record["sha256"]
            and value.get("stable_MF_state_unchanged") is True
            and value.get("native_direct_driver_control", {}).get("original_driver_restored") is True
            and value["native_direct_driver_control"].get("restoration_failures") == 0,
            "Stage source MF/operator was not restored cleanly")
    manifest_path = path.parent / "stage-cache" / "manifest.json"
    require(not manifest_path.is_symlink() and not manifest_path.parent.is_symlink()
            and sha(manifest_path) == value["cache_manifest_sha256"], "Stage manifest hash changed")
    manifest = read(manifest_path)
    validate_manifest(manifest, stage=stage, physical_binding=physical_binding, atoms=atoms, nao=nao)
    require(manifest.get("producer") == {"result_path": str(path), "source_worker_sha256": sha(__file__),
                "supervisor_proof_sha256": value["resource_supervision"]["proof_sha256"], "stage": stage},
            "Cache producer binding differs")
    for row in manifest["arrays"]:
        payload = manifest_path.parent / row["file"]
        require(not payload.is_symlink() and payload.is_file() and sha(payload) == row["file_sha256"],
                "Closed complete stage payload missing/changed before new scientific work")
    return value, manifest, manifest_path.parent, closure


def verify_rhf_producers(cache, snapshot_sha, guard, v5):
    """Dispatch genuine V5 and healthy V6B producers; never accept forensic units."""
    cache = Path(cache).resolve()
    require(cache.is_relative_to(ROOT / "outputs") and cache != ROOT / "outputs", "Dedicated outputs/RHF cache required")
    require(not (cache / "units").is_symlink(), "Symlink RHF units rejected")
    records = sorted((cache / "units").glob("*/attempt-*/complete.json"))
    require(len(records) <= 150, "Duplicate/excess RHF units")
    result = []
    for path in records:
        record = read(path); evidence = record.get("execution_evidence", {})
        if evidence.get("schema") == v5.EXECUTION_SCHEMA and evidence.get("source_worker", {}).get("sha256") == V5_SHA:
            require(evidence.get("phase") == "hessian-segment", "Unsupported old RHF producer phase")
            closure = v5.verify_closed_execution(evidence, expected_snapshot_sha=snapshot_sha)
            qualification = "CLOSED_HEALTHY_V5_PRODUCER"
        elif evidence.get("schema") == RHF_EVIDENCE_SCHEMA and evidence.get("source_worker", {}).get("sha256") == sha(__file__):
            require(evidence.get("phase") == evidence.get("stage") == "rks-partial", "Unsupported V6B RHF stage")
            closure = guard.verify_closed_phase(evidence, expected_worker_path=Path(__file__).resolve(),
                expected_worker_sha256=sha(__file__), expected_result_schema=SCHEMA,
                allowed_result_statuses=SUCCESS["rks-partial"], expected_snapshot_sha256=snapshot_sha)
            qualification = "CLOSED_HEALTHY_V6B_FULL_RKS_PARTIAL_PRODUCER"
        else:
            raise ValueError("Unsupported RHF cache producer schema/source; quarantine failed/RUNNING native arrays")
        result.append({"record": artifact(path), "qualification": qualification, "producer": closure})
    require(sorted(read(p)["index"] for p in records) == list(range(len(records))), "RHF prefix must be continuous")
    return result


class CachedStageProviders:
    """Owned full-system cache copies; exactly one stock partial/H1/nuclear call."""
    def __init__(self, hobj, partial_loader, h1_loader, validator):
        self.hobj, self.partial_loader, self.h1_loader, self.validator = hobj, partial_loader, h1_loader, validator
        self.counts = {"partial": 0, "h1": 0, "nuclear": 0}

    def __enter__(self):
        obj = self.hobj
        self.missing = object()
        self.saved = {name: obj.__dict__.get(name, self.missing) for name in ("partial_hess_elec", "make_h1", "hess_nuc")}
        self.originals = {name: getattr(obj, name) for name in self.saved}
        require(not any(getattr(f, "_research4_stage_provider", False) for f in self.originals.values()), "Do not nest cache providers")
        def wrapped(kind, original, loader):
            signature = inspect.signature(original)
            def call(_obj, *args, **kwargs):
                bound = signature.bind(*args, **kwargs); bound.apply_defaults()
                self.validator(kind, bound.arguments)
                self.counts[kind] += 1
                require(self.counts[kind] == 1, "Global stock kernel repeated a cached/nuclear contribution")
                return original(*args, **kwargs) if loader is None else loader()
            call._research4_stage_provider = True
            return types.MethodType(call, obj)
        try:
            obj.partial_hess_elec = wrapped("partial", self.originals["partial_hess_elec"], self.partial_loader)
            obj.make_h1 = wrapped("h1", self.originals["make_h1"], self.h1_loader)
            obj.hess_nuc = wrapped("nuclear", self.originals["hess_nuc"], None)
        except BaseException:
            self.restore_methods()
            raise
        return self

    def restore_methods(self):
        for name, value in self.saved.items():
            if value is self.missing:
                self.hobj.__dict__.pop(name, None)
            else:
                self.hobj.__dict__[name] = value

    def __exit__(self, *exc):
        self.restore_methods()

    def require_once(self):
        require(self.counts == {"partial": 1, "h1": 1, "nuclear": 1}, "Incomplete/repeated full global assembly")


def case_for(symbols, control):
    from collections import Counter
    composition = dict(Counter(symbols))
    if not control:
        require(len(symbols) == 75 and composition == {"C": 28, "H": 32, "O": 15}, "Default target must be Spinosin75")
        return "Spinosin"
    for name, expected in (("water_software_control", {"H": 2, "O": 1}), ("ethanol_software_control", {"C": 2, "H": 6, "O": 1})):
        if composition == expected:
            return name
    raise ValueError("Explicit controls allow only water/ethanol; never relabel controls as Spinosin")


def execute(args):
    # Parent/proof qualification occurs before importing any numerical package.
    guard_module = load_source(GUARD, GUARD_SHA, "research4_v6b_staged_parent")
    supervision = guard_module.verify_resource_supervision(expected_threads=args.threads)
    require(1 <= args.threads <= 4 and args.stage in STAGES, "Unsupported bounded stage")
    if not args.control_validation:
        require(args.snapshot_manifest_sha256 == SPINOSIN_SNAPSHOT_SHA
                and args.gradient_check_result_sha256 == SPINOSIN_GRADIENT_SHA, "Use exact frozen Spinosin source/G")
    for key in ("snapshot", "gradient_check_result", "output"):
        require(getattr(args, key) is not None, "Explicit input/output required: " + key)
    if args.stage == "rks-partial":
        require(args.rhf_cache is not None, "Full RKS partial requires a healthy RHF cache")
    else:
        require(args.partial_result is not None and args.partial_result_sha256, "Healthy complete partial producer required")
    if args.stage == "global":
        require(args.h1_result is not None and args.h1_result_sha256, "Healthy complete75 H1AO producer required")
    import numpy as np
    from pyscf import lib
    from pyscf.scf import _vhf
    from pyscf.hessian import rhf, rks
    from pyscf.lib import linalg_helper
    lib.num_threads(args.threads)
    w = load_source(STATE, STATE_SHA, "research4_stage_original_V5_state")
    v5 = load_source(V5, V5_SHA, "research4_stage_original_V5_consumer")
    base = load_source(BASE, BASE_SHA, "research4_stage_frozen_V8_kernel")
    core = load_source(CORE, CORE_SHA, "research4_stage_frozen_incremental_core")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports") and not output.exists(), "Fresh report directory required")
    output.mkdir(parents=True)
    log = (output / "pyscf.log").open("x", buffering=1)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    state = {"schema": SCHEMA, "stage": args.stage, "phase": args.stage, "status": "PARTIAL", "started_at_utc": utc(),
             "source_worker": artifact(__file__), "resource_supervision": supervision,
             "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "snapshot_directory": str(args.snapshot.resolve()),
             "boot_id": v5.boot_id(), "no_new_SCF_performed": True, "stable_MF_state_unchanged": False,
             "source_MF_stability_status": "NOT_CHECKED",
             "frequencies_status": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED", "complete_global_Hessian": False,
             "complete_RHF_helper": False, "complete_RKS_partial": False, "complete_H1AO": False,
             "physical_binding_separate_from_stage_producer": True,
             "V5_analyzer_compatibility": "BLOCKED_V6B_PROOF_SCHEMA_REQUIRES_REVIEWED_V6B_POSTPROCESSOR"}
    save_json(output / "input.json", {**state, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, exclusive=True)
    def emit(event, details=None, **more):
        row = {"utc": utc(), "event": event, "CPU_seconds": time.process_time()-start_cpu,
               "wall_seconds": time.perf_counter()-start_wall, **(details or {}), **more}
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
        save_json(output / "progress.json", {**state, **row})
    def stop(signum, _frame):
        raise base.BudgetStop("V6B parent resource signal " + str(signum))
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
        native_threads=args.threads, emit=lambda details: emit("native_driver_signature", details))
    mf = before = hobj = pending = None
    input_records = []
    try:
        native.install()
        state["gradient_gate"] = v5.qualify_gradient_receipt(args.gradient_check_result, args.gradient_check_result_sha256,
                                                              args.snapshot_manifest_sha256, state["boot_id"])
        state["final_gradient_thresholds_passed"] = True
        # Original V5 source closure remains V5; no new SCF or proof translation.
        mf, binding = w.restore_snapshot(args.snapshot, expected_manifest_sha256=args.snapshot_manifest_sha256,
                                         expected_source_sha256=STATE_SHA, stdout=log)
        symbols = [mf.mol.atom_symbol(i) for i in range(mf.mol.natm)]
        state.update(case=case_for(symbols, args.control_validation), atoms=int(mf.mol.natm), basis_functions=int(mf.mol.nao_nr()),
                     source_state_binding=binding, versions=binding["environment"], energy_hartree=float(mf.e_tot))
        state["settings"] = {**binding["method_settings"], "threads": args.threads,
            "native_direct_driver_threads": args.threads, "hessian_max_memory_mb": 3000,
            "hessian_max_cycle": 50, "hessian_level_shift": 0, "hessian_response_block_atoms": 8,
            "engineering_worker_version": "STAGED_COMPLETE_FULL_RKS_V6B"}
        hobj = mf.Hessian(); hobj.max_cycle, hobj.level_shift = 50, 0
        state["stock_hamiltonian_gate"] = v5.stock_operator_gate(mf, hobj)
        identities = ((hobj.partial_hess_elec, rks.partial_hess_elec), (hobj.make_h1, rks.make_h1),
                      (hobj.hess_elec, rhf.hess_elec), (hobj.solve_mo1, rhf.HessianBase.solve_mo1),
                      (hobj.kernel, rhf.HessianBase.kernel), (hobj.hess_nuc, rhf.HessianBase.hess_nuc))
        require(all(getattr(f, "__func__", None) is expected for f, expected in identities), "Nonstock Hessian stage method rejected")
        before = v5.physical_state(mf)
        with base.hessian_memory_context(hobj, state["settings"], emit=emit):
            physical = {"schema": PHYSICAL_SCHEMA, "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
                        "stable_snapshot_binding": binding, "actual_runtime": core.runtime_state_fingerprint(hobj),
                        "numerical_policy": NUMERICAL_POLICY, "frozen_derivative_sources": core._source_identity(rhf),
                        "native_threads": args.threads,
                        "native_krylov_lindep": float(inspect.signature(linalg_helper.krylov).parameters["lindep"].default)}
            state["physical_binding_sha256"] = digest(physical)
            n, nao = state["atoms"], state["basis_functions"]
            def write_array(directory, key, value, shape):
                value = np.asarray(value)
                require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), "Invalid complete stage array")
                target = directory / (key+".npy")
                with target.open("xb") as handle:
                    np.save(handle, value, allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
                fsync_directory(directory)
                return {"key": key, "file": target.name, "file_sha256": sha(target), "shape": list(shape),
                        "dtype": str(value.dtype), "payload": base.array_fingerprint(value)}
            def load_array(directory, row):
                path = directory / row["file"]
                require(not path.is_symlink() and sha(path) == row["file_sha256"], "Closed stage payload SHA differs")
                value = np.load(path, allow_pickle=False)
                require(value.dtype == np.float64 and list(value.shape) == row["shape"]
                        and np.isfinite(value).all() and base.array_fingerprint(value) == row["payload"], "Closed stage payload differs")
                return value
            partial_source = h1_source = None
            if args.stage != "rks-partial":
                record = {"path": str(args.partial_result.resolve()), "sha256": args.partial_result_sha256}
                partial_source = qualify_stage(record, "rks-partial", physical, n, nao, guard_module)
                input_records.append(record)
                state["partial_stage_producer"] = partial_source[3]
                state["complete_RHF_helper"] = state["complete_RKS_partial"] = True
            if args.stage == "global":
                record = {"path": str(args.h1_result.resolve()), "sha256": args.h1_result_sha256}
                h1_source = qualify_stage(record, "h1ao", physical, n, nao, guard_module)
                require(h1_source[0].get("partial_result_dependency") == input_records[0], "H1AO and global use different partial producer")
                input_records.append(record); state["H1_stage_producer"] = h1_source[3]
                state["complete_H1AO"] = True
            if args.stage in ("rks-partial", "h1ao"):
                directory = output / "stage-cache"; directory.mkdir()
                if args.stage == "rks-partial":
                    prior = verify_rhf_producers(args.rhf_cache, args.snapshot_manifest_sha256, guard_module, v5)
                    if not args.control_validation:
                        require(len(prior) in (149, 150), "Target contingency accepts only healthy149/150 RHF seed")
                    state["prior_RHF_unit_producer_checks"] = prior
                    evidence = {"schema": RHF_EVIDENCE_SCHEMA, "source_worker": state["source_worker"],
                        "resource_supervision": supervision, "output_directory": str(output),
                        "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "phase": "rks-partial", "stage": "rks-partial"}
                    with core.IncrementalPureJContext(hobj, args.rhf_cache, binding, max_new_units=150, execution_evidence=evidence) as context:
                        state["incremental_before"] = context.progress()
                        # Exactly the stock hess_elec argument, NOT hobj.max_memory.
                        value = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ, list(range(n)), 4000, None)
                        state["incremental_after"] = context.progress()
                    require(state["incremental_after"]["total_units_completed"] == 2*n, "RHF helper incomplete")
                    arrays = [write_array(directory, "full_RKS_partial", value, (n, n, 3, 3))]
                    state["complete_RHF_helper"] = state["complete_RKS_partial"] = True
                else:
                    value = hobj.make_h1(mf.mo_coeff, mf.mo_occ, None, list(range(n)), None)
                    require(len(value) == n, "H1AO must retain every global atom")
                    arrays = [write_array(directory, f"H1AO_atom_{i:03d}", value[i], (3, nao, nao)) for i in range(n)]
                    state["complete_H1AO"] = True
                    state["partial_result_dependency"] = input_records[0]
                pending = {"schema": CACHE_SCHEMA, "stage": args.stage, "complete": True,
                    "physical_binding": physical, "physical_binding_sha256": digest(physical), "arrays": arrays,
                    "includes_nuclear_hessian": False, "includes_CPHF_response": False,
                    "producer": {"result_path": str(output / "result.json"), "source_worker_sha256": sha(__file__),
                                 "supervisor_proof_sha256": supervision["proof_sha256"], "stage": args.stage}}
                state["status"] = "STAGE_COMPLETE"
            else:
                def validate_call(kind, values):
                    validate_cache_call(kind, values, n, mf,
                        lambda actual, reference: np.asarray(actual).dtype == np.float64 and np.array_equal(actual, reference))
                def partial_loader():
                    return load_array(partial_source[2], partial_source[1]["arrays"][0]).copy()
                def h1_loader():
                    # Fresh owned loads; do not retain a second full1GB cache in RAM.
                    return [load_array(h1_source[2], row).copy() for row in h1_source[1]["arrays"]]
                with CachedStageProviders(hobj, partial_loader, h1_loader, validate_call) as providers:
                    saveonly = ROOT / "outputs/pyscf-staged-hessian-v6b" / output.name
                    raw = base._run_checkpointed_hessian_in_memory_context(hobj, saveonly,
                        output / "hessian_cache_manifest.json", state["settings"], block_atoms=8,
                        versions=state["versions"], response_operator_factory=rhf.gen_vind,
                        native_lindep=physical["native_krylov_lindep"],
                        input_evidence={"source_snapshot_sha256": args.snapshot_manifest_sha256,
                            "actual_entry_source_sha256": sha(__file__), "consumer_resource_proof_sha256": supervision["proof_sha256"],
                            "geometry_optimized": True, "final_gradient_thresholds_passed": True,
                            "partial_stage_result_sha256": input_records[0]["sha256"], "H1_stage_result_sha256": input_records[1]["sha256"]})
                    providers.require_once()
                    state["full_global_contribution_calls"] = dict(providers.counts)
                capture_path = output / "hessian_cache_manifest.json"
                capture = read(capture_path)
                # V8's original hook now calls a verified cache provider. Do not
                # mislabel that as a fresh native partial calculation in V6B.
                capture["partial_hessian"].update(native_function_returned_normally=False,
                    cache_provider_returned_normally=True, native_partial_executed_in_this_phase=False,
                    healthy_native_partial_source=input_records[0])
                capture["V6B_stage_execution_origin"] = {"partial": "HEALTHY_COMPLETE_V6B_CACHE_COPY",
                    "H1AO": "HEALTHY_COMPLETE_V6B_CACHE_FRESH_LOADS", "CPHF": "FRESH_FULL_SYSTEM_STOCK_B8",
                    "nuclear": "FRESH_STOCK_ONCE", "native_make_h1_executed_in_this_phase": False,
                    "H1AO_capture_duplicates_closed_source_bytes": 3*n*nao*nao*8,
                    "duplicate_H1_capture_IO_counted_not_elided": True}
                save_json(capture_path, capture)
                state["RKS_partial_and_H1_cache_restore"] = capture["V6B_stage_execution_origin"]
                state["stock_hamiltonian_gate_after_provider_restore"] = v5.stock_operator_gate(mf, hobj)
                for name, coordinates in (("normal_mode_geometry_angstrom.npy", mf.mol.atom_coords(unit="Angstrom")),
                                          ("normal_mode_geometry_bohr.npy", mf.mol.atom_coords(unit="Bohr"))):
                    base.atomic_npy(output / name, coordinates)
                (output / "normal_mode_geometry.xyz").write_text(base.xyz_text(symbols, mf.mol.atom_coords(unit="Angstrom"), "V6B exact restored full-system geometry"))
                v5.publish_modes(output, mf, raw, state, base)
            require(core.runtime_state_fingerprint(hobj) == physical["actual_runtime"], "Complete source state changed inside stage")
        require(v5.physical_state(mf) == before, "Source MF changed across complete stage")
        state["stable_MF_state_unchanged"] = True
        state["source_MF_stability_status"] = "VERIFIED_UNCHANGED"
        for record in input_records:
            checked(record)
        # The source files remain immutable even if stock mutates its owned copies.
        for source in (partial_source, h1_source):
            if source:
                for row in source[1]["arrays"]:
                    require(sha(source[2] / row["file"]) == row["file_sha256"], "Cached source was contaminated in place")
    except BaseException as exc:
        state.update(status="PARTIAL", exception_type=type(exc).__name__, stop_reason=str(exc))
        if mf is not None and before is not None:
            try:
                state["stable_MF_state_unchanged"] = v5.physical_state(mf) == before
                state["source_MF_stability_status"] = "VERIFIED_UNCHANGED" if state["stable_MF_state_unchanged"] else "CHANGED"
            except Exception:
                state["source_MF_stability_status"] = "CHECK_FAILED"
        (output / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        native.restore(); state["native_direct_driver_control"] = native.summary()
        if (state["native_direct_driver_control"].get("restoration_failures") != 0
                or state["native_direct_driver_control"].get("original_driver_restored") is not True):
            state.update(status="PARTIAL", stop_reason="Native operator restoration failed")
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if state["status"] == "STAGE_COMPLETE" and state["stable_MF_state_unchanged"] and pending:
            manifest_path = output / "stage-cache" / "manifest.json"
            save_json(manifest_path, pending, exclusive=True)
            state["cache_manifest_sha256"] = sha(manifest_path)
        if (output / "hessian_cache_manifest.json").exists():
            state["hessian_cache_manifest_sha256"] = sha(output / "hessian_cache_manifest.json")
        state.update(completed_at_utc=utc(), process_cpu_seconds=time.process_time()-start_cpu,
                     wall_seconds=time.perf_counter()-start_wall, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        log.flush(); log.close()
        save_json(output / "result.json", state, exclusive=True)
    return state


def preflight(output):
    output = Path(output).resolve()
    require(not output.exists(), "Preserve earlier V6B preparations")
    dependencies = []
    for path, expected in ((GUARD, GUARD_SHA), (STATE, STATE_SHA), (V5, V5_SHA), (BASE, BASE_SHA), (CORE, CORE_SHA)):
        require(len(expected) == 64 and sha(path) == expected, "Unfrozen/changed preparation source")
        dependencies.append(artifact(path))
    output.mkdir(parents=True)
    value = {"schema": SCHEMA, "status": "NOT_RUN", "target": "Spinosin75/full219 signed modes",
             "source_worker": artifact(__file__), "stages": list(STAGES), "numerical_policy": NUMERICAL_POLICY,
             "guard_source_sha256": GUARD_SHA, "guard_frozen": True, "frozen_dependencies": dependencies,
             "small_molecule_controls": "NOT_RUN_WATER_ETHANOL_REQUIRED_AFTER_ACTIVE_V5_PHASE",
             "stage_cache_compatibility": "ONLY_NEW_V6B_HEALTHY_COMPLETE_STAGE_MANIFESTS; OLD_FAILED_NATIVE_ARRAYS_QUARANTINED",
             "old_RHF_seed_compatibility": "HEALTHY_V5_149_SAME_EXACT_STATE_WITH_FROZEN_CORE; LAST1_RECOMPUTED_UNDER_V6B",
             "V5_analyzer_compatibility": "BLOCKED_NEW_V6B_PROOF_SCHEMA_NEEDS_REVIEWED_POSTPROCESSOR",
             "target_arrays_or_numerical_packages_read": False, "QM_launched": False,
             "actual_allocation_or_ledger_created": False, "IR_intensities_status": "NOT_COMPUTED"}
    save_json(output / "preflight.json", value, exclusive=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-manifest-sha256")
    parser.add_argument("--gradient-check-result", type=Path)
    parser.add_argument("--gradient-check-result-sha256")
    parser.add_argument("--rhf-cache", type=Path)
    parser.add_argument("--partial-result", type=Path)
    parser.add_argument("--partial-result-sha256")
    parser.add_argument("--h1-result", type=Path)
    parser.add_argument("--h1-result-sha256")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--control-validation", action="store_true")
    args = parser.parse_args()
    result = execute(args) if args.execute else preflight(args.output)
    print(json.dumps({k: result.get(k) for k in ("status", "stage", "frequencies_status", "complete_global_Hessian")}))
    if args.execute and result["status"] not in ("STAGE_COMPLETE", "FREQUENCIES_COMPUTED"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
