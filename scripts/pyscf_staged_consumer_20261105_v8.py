"""V8 continuation: authenticated legacy A/H1 -> native B8 response -> full Hessian.

The full-AO/MO numerical operations are copied from the frozen V7 consumer.
New V8 supervision and result/cache schemas are explicit, while legacy V7 A/H1
remain qualified only by their original frozen worker and V6b guardian.
Import/preflight are stdlib only. Actual controls must release target execution.
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
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
LEGACY_GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
LEGACY_GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
LEGACY_CONSUMER = ROOT / "scripts/pyscf_staged_consumer_20261105_v7.py"
LEGACY_CONSUMER_SHA = "6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe"
LEGACY_SCHEMA = "research4-staged-full-rks-hessian-result-v7"
CONTROL_RELEASE_SCHEMA = "research4-v8-actual-small-control-release-v1"
HELPER = ROOT / "scripts/pyscf_staged_hessian_worker_20261105_v6b.py"
HELPER_SHA = "f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d"
PARTIAL_WORKER = ROOT / "scripts/pyscf_xc_chunk_worker_20261105_v7.py"
PARTIAL_WORKER_SHA = "00e710eb0d637dbd073939bea5d4496993cb68f29efad1439322df3972415f02"
XC_CORE = ROOT / "scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py"
XC_CORE_SHA = "da1e2b0267c48e748a2bcb7d3f2ee3110d8e2288aa340d23dab7a12c994ddf8e"
LINALG_HELPER_SHA = "9dcd5d4582c3695818e05fa72217c7764f87d5491ed29acb7b64d2725ee130e0"
CPHF_SOURCE_SHA = "e39e3a06fc33e23307a8558d2410bc1137e08ef9c27d48b4ce95fd20a22a63e3"
RESPONSE_MODULE_SHAS = {
    "dft/rks.py": "fcc8d3dbd03df368f3b70f3ce5999be6a19c608a6ff0bdc1890074616f0befa8",
    "scf/hf.py": "34ea1a9057da3a2d5aa37fe51d4e7305580bcb8e0118a5f17ba95ba5bb36c22f",
    "grad/rhf.py": "d30fd1ab2a00997183802da28da775f6a29a6c9705c55b19c04d66a1a575267f",
    "scf/_response_functions.py": "806b3bbf00f1127bfbb15c74f5c7b022bb8f58ca01945f502d6082dd0b35f411"}
STATE = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
V5 = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
V5_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
BASE = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
BASE_SHA = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"
CORE = ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py"
CORE_SHA = "ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55"
SCHEMA = "research4-staged-full-rks-hessian-result-v8"
CACHE_SCHEMA = "research4-complete-full-rks-stage-cache-v8"
RESPONSE_SCHEMA = "research4-complete-full-rks-response-block-cache-v8"
RESPONSE_INDEX_SCHEMA = "research4-v8-closed-response-results-index-v1"
PHYSICAL_SCHEMA = "research4-full-rks-physical-binding-v6b"
COMPATIBILITY = "BLOCKED_V8_PRODUCER_REQUIRES_REVIEWED_V8_POSTPROCESSOR"
SPINOSIN_SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
SPINOSIN_GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
STAGES = ("h1ao", "response-block", "global-assemble")
NUMERICAL_POLICY = {"xc": "PBE", "basis": "def2-svp", "density_fitting": False,
    "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "full_AO_MO_space": True,
    "grid_changed": False, "MF_Hessian_memory_mb": 3000,
    "stock_hess_elec_max_memory_argument_mb": 4000, "conv_tol_cpscf": 1e-8,
    "hessian_max_cycle": 50, "level_shift": 0, "response_block_atoms": 8}


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def root_path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 32 * 1024**2,
            "Missing, symlink or oversized metadata")
    return json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def checked(record):
    supplied = Path(record["path"])
    p = supplied if supplied.is_absolute() else ROOT / supplied
    require(not p.is_symlink() and not any(q.is_symlink() for q in p.parents), "Symlink artifact/parent rejected")
    require(p.is_file() and sha(p) == record["sha256"], "Bound artifact changed")
    return p.resolve()


def save(path, value, *, exclusive=False):
    p = Path(path)
    temporary = p if exclusive else p.with_name(p.name + ".writing")
    with temporary.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    if not exclusive:
        os.replace(temporary, p)
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_source(path, expected, name):
    require(len(expected) == 64 and sha(path) == expected, "Unfrozen/changed source: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def partial_worker():
    return load_source(PARTIAL_WORKER, PARTIAL_WORKER_SHA, "research4_v7_real_XC_partial_source")


def response_blocks(atoms):
    require(type(atoms) is int and atoms in (3, 9, 75), "Only genuine control/Spinosin complete atom counts")
    return [list(range(i, min(i + 8, atoms))) for i in range(0, atoms, 8)]


def bind_actual_native_modules(ni, *, xc_core, rhf_core, rhf, rks, numint, grad_rks, gen_grid,
                              linalg_helper, mf, dft_rks, hf, grad_rhf, response, cphf, lib, vhf):
    """Check the actual executed module/function objects, not another file tree."""
    require(type(ni) is numint.NumInt, "Actual plain NumInt instance required")
    records = xc_core.validate_runtime_sources(ni, numint=numint, rks=rks, grad_rks=grad_rks, gen_grid=gen_grid)
    expected = xc_core.PYSCF_ROOT / "hessian/rhf.py"
    require(Path(rhf.__file__).resolve() == expected and sha(expected) == rhf_core.PINNED["hessian/rhf.py"]
            and rks.rhf_hess is rhf, "Actual RHF module differs from reviewed full-system implementation")
    functions = (rhf.hess_elec, rhf.solve_mo1, rhf.hess_nuc, rhf.gen_vind, rhf.HessianBase.solve_mo1)
    require(all(f.__globals__ is rhf.__dict__ and Path(inspect.getsourcefile(f)).resolve() == expected for f in functions),
            "Actual RHF callable namespace/source differs")
    for name, module, expected_sha in (("lib/linalg_helper.py", linalg_helper, LINALG_HELPER_SHA),
                                       ("scf/cphf.py", rhf.cphf, CPHF_SOURCE_SHA)):
        path = xc_core.PYSCF_ROOT / name
        require(Path(module.__file__).resolve() == path and sha(path) == expected_sha,
                "Actual response solver module differs: " + name)
        records[name] = {"path": str(path), "sha256": expected_sha}
    for name, module in (("dft/rks.py", dft_rks), ("scf/hf.py", hf), ("grad/rhf.py", grad_rhf),
                         ("scf/_response_functions.py", response), ("scf/_vhf.py", vhf)):
        path = xc_core.PYSCF_ROOT / name
        expected_sha = rhf_core.PINNED[name] if name == "scf/_vhf.py" else RESPONSE_MODULE_SHAS[name]
        require(Path(module.__file__).resolve() == path and sha(path) == expected_sha,
                "Actual full-system response module differs: " + name)
        records[name] = {"path": str(path), "sha256": expected_sha}
    require(rhf.cphf.solve.__globals__ is rhf.cphf.__dict__
            and Path(inspect.getsourcefile(rhf.cphf.solve)).resolve() == xc_core.PYSCF_ROOT / "scf/cphf.py"
            and rhf.cphf is cphf and rhf.rhf is grad_rhf and rhf._vhf is vhf
            and lib.krylov is linalg_helper.krylov
            and rhf.cphf.lib.krylov is linalg_helper.krylov
            and linalg_helper.krylov.__globals__ is linalg_helper.__dict__
            and getattr(mf.gen_response, "__func__", None) is response._gen_rhf_response
            and response._gen_rhf_response.__globals__ is response.__dict__, "Actual CPHF/Krylov/full-response callable binding differs")
    records["hessian/rhf.py"] = {"path": str(expected), "sha256": rhf_core.PINNED["hessian/rhf.py"]}
    return records


def option(command, name):
    require(command.count(name) == 1 and command.index(name) + 1 < len(command), "Unique actual argv required: " + name)
    return command[command.index(name) + 1]


def actual_command(value, directory, guard):
    proof = value["resource_supervision"]["proof"]
    ledger = read(proof["ledger_path"])
    guard.validate_ledger(ledger, allow_running=True)
    rows = [r for r in ledger["phases"] if r.get("phase") == proof["phase"]]
    require(len(rows) == 1 and guard.healthy_closed_row(rows[0]), "A real healthy closed producer is required")
    command = rows[0]["command"]
    require(sha(directory / "input.json") == value["input_json_sha256"], "Actual saved input document changed")
    args = read(directory / "input.json")["arguments"]
    require(command.count("--execute") == 1 and args["execute"] is True
            and option(command, "--stage") == args["stage"] == value["stage"]
            and root_path(option(command, "--output")) == directory
            and int(option(command, "--threads")) == args["threads"] == value["settings"]["threads"] == 2
            and command.count("--control-validation") == int(args["control_validation"]), "Actual stage/source/thread argv differs")
    keys = [("snapshot", True), ("snapshot_manifest_sha256", False),
            ("gradient_check_result", True), ("gradient_check_result_sha256", False),
            ("partial_result", True), ("partial_result_sha256", False)]
    if value["stage"] != "h1ao":
        keys += [("h1_result", True), ("h1_result_sha256", False)]
    if value["stage"] == "response-block":
        require(int(option(command, "--block-index")) == args["block_index"] == value["block_index"], "Response block argv differs")
    if value["stage"] == "global-assemble":
        keys += [("response_index", True), ("response_index_sha256", False)]
    if not args["control_validation"]:
        keys += [("control_release", True), ("control_release_sha256", False)]
    for key, is_path in keys:
        observed = option(command, "--" + key.replace("_", "-"))
        require(root_path(observed) == root_path(args[key]) if is_path else observed == args[key],
                "Bound input differs from actual argv: " + key)
    require(value["snapshot_manifest_sha256"] == args["snapshot_manifest_sha256"], "Snapshot argv differs")
    return command, args, rows[0]


def closed_stage(record, stage, physical, atoms, nao, guard, statuses=("STAGE_COMPLETE",)):
    path = checked(record); value = read(path)
    require(path.name == "result.json" and path.is_relative_to(ROOT / "reports"), "A report result is required")
    require(value.get("schema") == SCHEMA and value.get("stage") == value.get("phase") == stage
            and value.get("status") in statuses and value.get("atoms") == atoms
            and value.get("basis_functions") == nao and value.get("no_new_SCF_performed") is True
            and value.get("stable_MF_state_unchanged") is True
            and value.get("source_MF_stability_status") == "VERIFIED_UNCHANGED"
            and all(value.get(k) is True for k in ("complete_RHF_helper", "complete_RKS_partial", "complete_H1AO", "final_gradient_thresholds_passed"))
            and value.get("physical_binding_sha256") == digest(physical)
            and value.get("IR_intensities_status") == "NOT_COMPUTED"
            and value.get("V5_analyzer_compatibility") == COMPATIBILITY
            and value.get("native_direct_driver_control", {}).get("original_driver_restored") is True
            and value["native_direct_driver_control"].get("restoration_failures") == 0,
            "Incomplete/foreign V7 stage")
    evidence = {"source_worker": value["source_worker"], "resource_supervision": value["resource_supervision"],
        "output_directory": str(path.parent), "snapshot_manifest_sha256": value["snapshot_manifest_sha256"], "stage": stage}
    closure = guard.verify_closed_phase(evidence, expected_worker_path=Path(__file__).resolve(),
        expected_worker_sha256=sha(__file__), expected_result_schema=SCHEMA,
        allowed_result_statuses=statuses, expected_output=path.parent,
        expected_snapshot_sha256=physical["snapshot_manifest_sha256"])
    require(closure["producer_result_sha256"] == record["sha256"], "Result differs from healthy producer")
    actual_command(value, path.parent, guard)
    return value, path.parent, closure


def legacy_modules():
    """The old producer is verified by its real, unchanged old guardian."""
    old_guard = load_source(LEGACY_GUARD, LEGACY_GUARD_SHA, "research4_v8_legacy_V6b_guard")
    old_consumer = load_source(LEGACY_CONSUMER, LEGACY_CONSUMER_SHA, "research4_v8_legacy_V7_consumer")
    return old_consumer, old_guard


def qualify_partial(record, physical, atoms, nao, guard):
    old_consumer, old_guard = legacy_modules()
    require(old_consumer.SCHEMA == LEGACY_SCHEMA
            and old_consumer.PHYSICAL_SCHEMA == PHYSICAL_SCHEMA
            and old_consumer.NUMERICAL_POLICY == NUMERICAL_POLICY,
            "Explicit legacy A physical/numerical contract differs")
    return old_consumer.qualify_partial(record, physical, atoms, nao, old_guard)


def verify_arrays(directory, rows, expected):
    require(len(rows) == len(expected), "Complete array coverage differs")
    require(len({r.get("file") for r in rows}) == len(rows), "Duplicate array filenames")
    for row, want in zip(rows, expected):
        require(all(row.get(k) == v for k, v in want.items()) and row.get("dtype") == "float64"
                and isinstance(row.get("file"), str) and Path(row["file"]).name == row["file"]
                and row["file"].endswith(".npy") and isinstance(row.get("payload"), dict),
                "Array identity/shape/type differs")
        checked({"path": str(directory / row["file"]), "sha256": row["file_sha256"]})


def cache_manifest(value, directory, stage, physical, schema):
    path = directory / "stage-cache/manifest.json"
    require(not path.parent.is_symlink() and sha(path) == value["cache_manifest_sha256"], "Stage manifest differs")
    m = read(path)
    require(m.get("schema") == schema and m.get("stage") == stage and m.get("complete") is True
            and m.get("physical_binding") == physical and m.get("physical_binding_sha256") == digest(physical)
            and m.get("producer") == {"result_path": str(directory / "result.json"),
                "source_worker_sha256": sha(__file__), "supervisor_proof_sha256": value["resource_supervision"]["proof_sha256"],
                "stage": stage}, "Wrong complete V7 producer/physical binding")
    return m, path.parent


def qualify_stage(record, stage, physical, atoms, nao, guard):
    require(stage == "h1ao", "A retains its separate genuine XC qualifier; this entry is H1 only")
    supplied = read(checked(record))
    if supplied.get("schema") == LEGACY_SCHEMA:
        old_consumer, old_guard = legacy_modules()
        require(old_consumer.PHYSICAL_SCHEMA == PHYSICAL_SCHEMA
                and old_consumer.NUMERICAL_POLICY == NUMERICAL_POLICY,
                "Explicit legacy H1 physical/numerical contract differs")
        return old_consumer.qualify_stage(record, stage, physical, atoms, nao, old_guard)
    require(supplied.get("schema") == SCHEMA, "Unknown H1 producer schema is never relabelled")
    value, directory, closure = closed_stage(record, stage, physical, atoms, nao, guard)
    require(value.get("complete_H1AO") is True, "Complete global H1 is required")
    manifest, cache = cache_manifest(value, directory, stage, physical, CACHE_SCHEMA)
    require(manifest.get("includes_nuclear_hessian") is False and manifest.get("includes_CPHF_response") is False,
            "H1 is not a nuclear or CPHF contribution")
    verify_arrays(cache, manifest["arrays"], [{"key": f"H1AO_atom_{i:03d}", "shape": [3, nao, nao]} for i in range(atoms)])
    return value, manifest, cache, closure


def qualify_response(record, physical, atoms, nao, guard, partial_ref, h1_ref, nocc=None):
    value, directory, closure = closed_stage(record, "response-block", physical, atoms, nao, guard)
    index = value.get("block_index"); blocks = response_blocks(atoms)
    require(type(index) is int and 0 <= index < len(blocks), "Invalid B8 response block index")
    selected = blocks[index]
    require(value.get("global_atom_ids") == selected and value.get("partial_result_dependency") == partial_ref
            and value.get("h1_result_dependency") == h1_ref and value.get("native_solver_returned_normally") is True
            and value.get("full_AO_MO_response_operator") is True and value.get("residual_norm") == "NOT_COMPUTED"
            and value.get("requested_effective_CPHF_tolerance") == 1e-8 * len(selected), "Wrong full-system response/source")
    nocc = value["occupied_orbitals"] if nocc is None else nocc
    require(type(nocc) is int and nocc > 0 and value["occupied_orbitals"] == nocc, "Occupied space differs")
    manifest, cache = cache_manifest(value, directory, "response-block", physical, RESPONSE_SCHEMA)
    require(manifest.get("block_index") == index and manifest.get("global_atom_ids") == selected
            and manifest.get("includes_nuclear_hessian") is False and manifest.get("full_global_response_complete") is False,
            "Response cache is not one exact RHS block")
    wanted = []
    for atom in selected:
        wanted += [{"key": f"MO1_atom_{atom:03d}", "atom_id": atom, "kind": "MO1", "shape": [3, nao, nocc]},
                   {"key": f"MO_E1_atom_{atom:03d}", "atom_id": atom, "kind": "MO_E1", "shape": [3, nocc, nocc]}]
    verify_arrays(cache, manifest["arrays"], wanted)
    return value, manifest, cache, closure


def qualify_response_index(record, physical, atoms, nao, guard, partial_ref, h1_ref, nocc=None):
    path = checked(record); index = read(path)
    require(index.get("schema") == RESPONSE_INDEX_SCHEMA and isinstance(index.get("results"), list)
            and len(index["results"]) == len(response_blocks(atoms)), "Full B8 response index required")
    require(len({root_path(r["path"]) for r in index["results"]}) == len(index["results"]), "Duplicate response producer")
    sources = [qualify_response(ref, physical, atoms, nao, guard, partial_ref, h1_ref, nocc) for ref in index["results"]]
    require([s[0]["block_index"] for s in sources] == list(range(len(sources)))
            and [a for s in sources for a in s[0]["global_atom_ids"]] == list(range(atoms)), "Missing/reordered response coverage")
    return sources, index, record


@contextmanager
def AssemblyProviders(hobj, partial, validator):
    """Only the electronic partial is cached; unexpected native recomputation fails."""
    missing = object(); names = ("partial_hess_elec", "make_h1", "solve_mo1", "hess_nuc")
    saved = {n: hobj.__dict__.get(n, missing) for n in names}
    originals = {n: getattr(hobj, n) for n in names}
    counts = {"partial": 0, "h1": 0, "response_solver": 0, "nuclear": 0}
    def cached(_obj, *args, **kwargs):
        values = inspect.signature(originals["partial_hess_elec"]).bind(*args, **kwargs); values.apply_defaults()
        validator("partial", dict(values.arguments)); counts["partial"] += 1
        require(counts["partial"] == 1, "Repeated partial contribution")
        return partial.copy()
    def forbidden(name):
        def call(_obj, *args, **kwargs):
            counts[name] += 1
            raise RuntimeError("Explicit complete H1/MO response forbids native recomputation: " + name)
        return call
    def nuclear(_obj, *args, **kwargs):
        values = inspect.signature(originals["hess_nuc"]).bind(*args, **kwargs); values.apply_defaults()
        validator("nuclear", dict(values.arguments)); counts["nuclear"] += 1
        require(counts["nuclear"] == 1, "Repeated nuclear contribution")
        return originals["hess_nuc"](*args, **kwargs)
    try:
        for n, f in (("partial_hess_elec", cached), ("make_h1", forbidden("h1")),
                     ("solve_mo1", forbidden("response_solver")), ("hess_nuc", nuclear)):
            setattr(hobj, n, types.MethodType(f, hobj))
        yield counts
        require(counts == {"partial": 1, "h1": 0, "response_solver": 0, "nuclear": 1}, "Full global contribution counts differ")
    finally:
        for n, value in saved.items():
            if value is missing:
                hobj.__dict__.pop(n, None)
            else:
                hobj.__dict__[n] = value


class NativeVindObserver:
    """Log native operator calls; never alter inputs, outputs or solver policy.

    A call is not a Krylov iteration and no residual is measured here.
    """
    def __init__(self, native, emit):
        self.native, self.emit = native, emit
        self.started = self.completed = self.failed = 0

    def __call__(self, value):
        self.started += 1
        index = self.started
        self.emit("native_vind_call_begin", {"call_index": index,
            "input_shape": list(value.shape), "not_a_Krylov_iteration_or_residual": True})
        try:
            returned = self.native(value)
        except BaseException as exc:
            self.failed += 1
            self.emit("native_vind_call_failed", {"call_index": index,
                "exception_type": type(exc).__name__, "native_exception_preserved": True})
            raise
        self.completed += 1
        self.emit("native_vind_call_end", {"call_index": index,
            "output_shape": list(returned.shape), "native_return_object_forwarded_unchanged": True})
        return returned

    def summary(self):
        return {"calls_started": self.started, "calls_completed": self.completed,
            "calls_failed": self.failed, "changes_to_input_output_or_solver": False,
            "calls_are_not_Krylov_iteration_counts": True, "residual_norm": "NOT_COMPUTED"}


def require_control_release(path, expected_sha, guard):
    """Accept actual V8 controls, with legacy controls separately authenticated."""
    release = read(checked({"path": str(path), "sha256": expected_sha}))
    require(release.get("schema") == CONTROL_RELEASE_SCHEMA
            and all(release.get(k) is True for k in ("root_execution_authorized",
                "v8_actual_water_ethanol_controls_reviewed", "no_new_budget_beyond_registered_v8")),
            "Explicit root release of actual V8 controls is required")
    refs = {str(checked(row)): row for row in release["code_refs"]}
    require(len(refs) == len(release["code_refs"]), "Duplicate released code path")
    for source, expected in ((Path(__file__).resolve(), sha(__file__)), (GUARD, GUARD_SHA)):
        require(str(source) in refs and refs[str(source)]["sha256"] == expected,
                "Released V8 consumer/guardian differs")
    legacy_ref = release["legacy_v7_control_release"]
    partial_worker().require_control_release(checked(legacy_ref), legacy_ref["sha256"],
        consumer_source={"path": str(LEGACY_CONSUMER), "sha256": LEGACY_CONSUMER_SHA})
    seen = set()
    for control in release["controls"]:
        case = control["case"]
        require(case in ("water_software_control", "ethanol_software_control") and case not in seen,
                "Exactly both distinct controls are required")
        seen.add(case); n = 3 if case == "water_software_control" else 9
        index_path = checked(control["global_analysis_index"]); index = read(index_path)
        require(index["V8_worker_sha256"] == sha(__file__)
                and any(row["sha256"] == index["analysis_source_sha256"] for row in refs.values()),
                "Actual V8 analyzer source was not released")
        entries = [row for row in index["files"] if row["path"] == "checks.json"]
        require(len(entries) == 1, "Unique actual independent checks required")
        entry = entries[0]
        checks = read(checked({"path": str(index_path.parent / "checks.json"),
            "sha256": entry["sha256"]}))
        require(checks["schema"] == "research4-V8-independent-staged-modes-analysis-v1"
                and checks["status"] == "DONE_CONTROL_NUMERICAL_REPLAY" and checks["case"] == case
                and checks["normal_modes"] == 3 * n - 6 and checks["all_frequencies_positive"] is True
                and checks["normal_mode_count_complete"] is True
                and checks["IR_intensities_status"] == "NOT_COMPUTED"
                and checks["qualified_for_mechanism"] is False and checks["new_quantum_evaluations"] == 0
                and checks["actual_parent_sha256"] == GUARD_SHA
                and checks["actual_V8_worker_sha256"] == sha(__file__)
                and checks["analysis_source_sha256"] == index["analysis_source_sha256"]
                and checks["actual_old_stock_control_comparison"]["status"]
                    == "PASS_ACTUAL_V8_VS_AUTHENTICATED_V5_STOCK_FULL_MATRIX_AND_ALL_MODES",
                "Actual full native Hessian/all-mode V8 control comparison did not pass")
        finals = [root_path(filename) for filename, row in checks["input_hashes"].items()
                  if Path(filename).name == "result.json" and row["sha256"] == checks["source_result_sha256"]]
        require(len(finals) == 1, "Unique actual complete V8 global producer required")
        global_path = checked({"path": str(finals[0]), "sha256": checks["source_result_sha256"]})
        result = read(global_path)
        evidence = {"source_worker": result["source_worker"], "resource_supervision": result["resource_supervision"],
            "output_directory": str(global_path.parent), "snapshot_manifest_sha256": result["snapshot_manifest_sha256"],
            "stage": "global-assemble"}
        closure = guard.verify_closed_phase(evidence, expected_worker_path=Path(__file__).resolve(),
            expected_worker_sha256=sha(__file__), expected_result_schema=SCHEMA,
            allowed_result_statuses=("FREQUENCIES_COMPUTED",), expected_output=global_path.parent,
            expected_snapshot_sha256=result["snapshot_manifest_sha256"])
        require(closure == checks["producer_closure"] and result["case"] == case
                and result["normal_mode_count"] == 3 * n - 6,
                "New control's actual healthy global producer differs")
    require(seen == {"water_software_control", "ethanol_software_control"}, "Both actual V8 controls required")
    return release


def execute(args):
    guard = load_source(GUARD, GUARD_SHA, "research4_v8_true_continuation_parent")
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    require(args.threads == 2 and args.stage in STAGES, "Only reviewed two-thread V7 stages")
    partial_module = partial_worker()  # no numerical import in this module or its metadata qualifier
    require(args.partial_result is not None and args.partial_result_sha256, "Real V7 A required")
    if not args.control_validation:
        require(args.snapshot_manifest_sha256 == SPINOSIN_SNAPSHOT_SHA
                and args.gradient_check_result_sha256 == SPINOSIN_GRADIENT_SHA, "Only original Spinosin C001 source/G")
        require(args.control_release is not None and args.control_release_sha256, "Actual reviewed V7 native-control release required")
        require_control_release(args.control_release, args.control_release_sha256, guard)
    if args.stage != "h1ao":
        require(args.h1_result is not None and args.h1_result_sha256, "Complete healthy B required")
    if args.stage == "global-assemble":
        require(args.response_index is not None and args.response_index_sha256, "Complete healthy B8 response index required")
    import numpy as np
    from pyscf import lib
    from pyscf.scf import _vhf
    from pyscf.scf import cphf, hf, _response_functions as response
    from pyscf.hessian import rhf, rks
    from pyscf.dft import numint, gen_grid
    from pyscf.dft import rks as dft_rks
    from pyscf.grad import rks as grad_rks
    from pyscf.grad import rhf as grad_rhf
    from pyscf.lib import linalg_helper
    lib.num_threads(2)
    helper = load_source(HELPER, HELPER_SHA, "research4_v7_frozen_V6b_numerical_helpers_only")
    w = load_source(STATE, STATE_SHA, "research4_v7_original_state")
    v5 = load_source(V5, V5_SHA, "research4_v7_original_gradient_and_modes")
    base = load_source(BASE, BASE_SHA, "research4_v7_frozen_native_thread_and_memory_helpers")
    core = load_source(CORE, CORE_SHA, "research4_v7_full_physical_state_fingerprint")
    xc_core = load_source(XC_CORE, XC_CORE_SHA, "research4_v7_frozen_actual_native_XC_binding")
    output = args.output.resolve(); require(output.is_relative_to(ROOT / "reports") and not output.exists(), "Fresh report output required")
    output.mkdir(parents=True); log = (output / "pyscf.log").open("x", buffering=1)
    state = {"schema": SCHEMA, "stage": args.stage, "phase": args.stage, "status": "PARTIAL", "started_at_utc": utc(),
        "source_worker": artifact(__file__), "resource_supervision": supervision, "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
        "snapshot_directory": str(args.snapshot.resolve()), "boot_id": v5.boot_id(), "no_new_SCF_performed": True,
        "stable_MF_state_unchanged": False, "source_MF_stability_status": "NOT_CHECKED", "frequencies_status": "NOT_RUN",
        "IR_intensities_status": "NOT_COMPUTED", "complete_global_Hessian": False, "complete_RHF_helper": False,
        "complete_RKS_partial": False, "complete_H1AO": False, "V5_analyzer_compatibility": COMPATIBILITY}
    save(output / "input.json", {**state, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, exclusive=True)
    state["input_json_sha256"] = sha(output / "input.json")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    def emit(event, details=None, **more):
        row = {"utc": utc(), "event": event, "CPU_seconds": time.process_time()-start_cpu,
               "wall_seconds": time.perf_counter()-start_wall, **(details or {}), **more}
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
        save(output / "progress.json", {**state, **row})
    def stop(signum, frame):
        raise base.BudgetStop("V8 actual parent resource signal " + str(signum))
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads, native_threads=2,
                                               emit=lambda d: emit("native_driver_signature", d))
    mf = before = pending = None; sources = []; input_records = []
    try:
        native.install()
        state["gradient_gate"] = v5.qualify_gradient_receipt(args.gradient_check_result, args.gradient_check_result_sha256,
                                                            args.snapshot_manifest_sha256, state["boot_id"])
        state["final_gradient_thresholds_passed"] = True
        mf, binding = w.restore_snapshot(args.snapshot, expected_manifest_sha256=args.snapshot_manifest_sha256,
                                         expected_source_sha256=STATE_SHA, stdout=log)
        state["actual_imported_native_sources"] = bind_actual_native_modules(mf._numint, xc_core=xc_core,
            rhf_core=core, rhf=rhf, rks=rks, numint=numint, grad_rks=grad_rks, gen_grid=gen_grid,
            linalg_helper=linalg_helper, mf=mf, dft_rks=dft_rks, hf=hf, grad_rhf=grad_rhf,
            response=response, cphf=cphf, lib=lib, vhf=_vhf)
        symbols = [mf.mol.atom_symbol(i) for i in range(mf.mol.natm)]
        state.update(case=helper.case_for(symbols, args.control_validation), atoms=int(mf.mol.natm), basis_functions=int(mf.mol.nao_nr()),
                     source_state_binding=binding, versions=binding["environment"], energy_hartree=float(mf.e_tot))
        state["settings"] = {**binding["method_settings"], "threads": 2, "native_direct_driver_threads": 2,
            "hessian_max_memory_mb": 3000, "hessian_max_cycle": 50, "hessian_level_shift": 0,
            "hessian_response_block_atoms": 8, "engineering_worker_version": "STAGED_COMPLETE_FULL_RKS_V8"}
        hobj = mf.Hessian(); hobj.max_cycle, hobj.level_shift = 50, 0
        state["stock_hamiltonian_gate"] = v5.stock_operator_gate(mf, hobj)
        identities = ((hobj.partial_hess_elec, rks.partial_hess_elec), (hobj.make_h1, rks.make_h1),
            (hobj.hess_elec, rhf.hess_elec), (hobj.solve_mo1, rhf.HessianBase.solve_mo1), (hobj.hess_nuc, rhf.HessianBase.hess_nuc))
        require(all(getattr(f, "__func__", None) is original for f, original in identities), "Nonstock full Hessian operators")
        before = v5.physical_state(mf)
        with base.hessian_memory_context(hobj, state["settings"], emit=emit):
            physical = {"schema": PHYSICAL_SCHEMA, "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
                "stable_snapshot_binding": binding, "actual_runtime": core.runtime_state_fingerprint(hobj),
                "numerical_policy": NUMERICAL_POLICY, "frozen_derivative_sources": core._source_identity(rhf),
                "native_threads": 2, "native_krylov_lindep": float(inspect.signature(linalg_helper.krylov).parameters["lindep"].default)}
            state["physical_binding_sha256"] = digest(physical)
            n, nao = state["atoms"], state["basis_functions"]; nocc = int(np.count_nonzero(mf.mo_occ > 0))
            require(mf.mo_coeff.shape == (nao, nao), "Full square AO/MO space required")
            state["occupied_orbitals"] = nocc
            partial_ref = {"path": str(args.partial_result.resolve()), "sha256": args.partial_result_sha256}
            a = qualify_partial(partial_ref, physical, n, nao, guard); input_records.append(partial_ref); sources.append(a)
            state.update(partial_result_dependency=partial_ref, partial_stage_producer=a[3], complete_RHF_helper=True,
                         complete_RKS_partial=True, complete_XC_partial_lineage=a[4])
            b = None
            if args.stage != "h1ao":
                h1_ref = {"path": str(args.h1_result.resolve()), "sha256": args.h1_result_sha256}
                b = qualify_stage(h1_ref, "h1ao", physical, n, nao, guard)
                require(b[0]["partial_result_dependency"] == partial_ref, "H1 uses another A")
                input_records.append(h1_ref); sources.append(b)
                state.update(h1_result_dependency=h1_ref, H1_stage_producer=b[3], complete_H1AO=True)
            def write_array(directory, key, value, shape, **extra):
                value = np.asarray(value)
                require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), "Invalid complete stage array")
                target = directory / (key + ".npy")
                with target.open("xb") as handle:
                    np.save(handle, value, allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
                return {"key": key, "file": target.name, "file_sha256": sha(target), "shape": list(shape),
                        "dtype": str(value.dtype), "payload": base.array_fingerprint(value), **extra}
            def load_array(directory, row):
                p = checked({"path": str(directory / row["file"]), "sha256": row["file_sha256"]})
                v = np.load(p, allow_pickle=False)
                require(v.dtype == np.float64 and list(v.shape) == row["shape"] and np.isfinite(v).all()
                        and base.array_fingerprint(v) == row["payload"], "Payload differs from actual closed source")
                return v
            if args.stage in ("h1ao", "response-block"):
                directory = output / "stage-cache"; directory.mkdir()
                if args.stage == "h1ao":
                    h1 = hobj.make_h1(mf.mo_coeff, mf.mo_occ, None, list(range(n)), None)
                    require(len(h1) == n, "H1 must retain every atom")
                    arrays = [write_array(directory, f"H1AO_atom_{i:03d}", h1[i], (3, nao, nao)) for i in range(n)]
                    state["complete_H1AO"] = True
                    pending = {"schema": CACHE_SCHEMA, "includes_CPHF_response": False}
                else:
                    blocks = response_blocks(n); require(type(args.block_index) is int and 0 <= args.block_index < len(blocks), "Exact original B8 block required")
                    selected = blocks[args.block_index]
                    h1 = [load_array(b[2], row) for row in b[1]["arrays"]]
                    emit("native_vind_operator_build_begin")
                    fx = NativeVindObserver(rhf.gen_vind(mf, mf.mo_coeff, mf.mo_occ), emit)
                    emit("native_vind_operator_build_end")
                    mo1, moe1 = hobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1, fx=fx,
                                              atmlst=selected, max_memory=4000, verbose=None)
                    require(len(mo1) == len(moe1) == n and all(mo1[i] is None and moe1[i] is None for i in set(range(n))-set(selected)),
                            "Native response must preserve complete atom indexing and exact RHS selection")
                    arrays = []
                    for atom in selected:
                        arrays += [write_array(directory, f"MO1_atom_{atom:03d}", mo1[atom], (3, nao, nocc), atom_id=atom, kind="MO1"),
                                   write_array(directory, f"MO_E1_atom_{atom:03d}", moe1[atom], (3, nocc, nocc), atom_id=atom, kind="MO_E1")]
                    state["native_vind_call_observer"] = fx.summary()
                    state.update(block_index=args.block_index, global_atom_ids=selected, native_solver_returned_normally=True,
                        full_AO_MO_response_operator=True, response_operator_fresh_full_system_per_phase=True,
                        requested_effective_CPHF_tolerance=float(mf.conv_tol_cpscf)*len(selected), residual_norm="NOT_COMPUTED",
                        normal_return_does_not_prove_measured_residual_below_requested_tolerance=True)
                    pending = {"schema": RESPONSE_SCHEMA, "block_index": args.block_index, "global_atom_ids": selected,
                               "full_global_response_complete": False}
                pending.update(stage=args.stage, complete=True, physical_binding=physical, physical_binding_sha256=digest(physical),
                    arrays=arrays, includes_nuclear_hessian=False,
                    producer={"result_path": str(output / "result.json"), "source_worker_sha256": sha(__file__),
                              "supervisor_proof_sha256": supervision["proof_sha256"], "stage": args.stage})
                state["status"] = "STAGE_COMPLETE"
            else:
                index_ref = {"path": str(args.response_index.resolve()), "sha256": args.response_index_sha256}
                responses, index_doc, _ = qualify_response_index(index_ref, physical, n, nao, guard, partial_ref, h1_ref, nocc)
                input_records.extend([index_ref, *index_doc["results"]]); sources.extend(responses)
                h1 = [load_array(b[2], row) for row in b[1]["arrays"]]
                mo1, moe1 = [None] * n, [None] * n
                for response in responses:
                    for row in response[1]["arrays"]:
                        target = mo1 if row["kind"] == "MO1" else moe1
                        require(target[row["atom_id"]] is None, "Duplicate response atom/kind")
                        target[row["atom_id"]] = load_array(response[2], row)
                require(all(x is not None for x in mo1 + moe1), "Complete full-system response required")
                partial = load_array(a[2], a[1]["arrays"][0])
                def validate(kind, values):
                    helper.validate_cache_call(kind, values, n, mf,
                        lambda actual, reference: np.asarray(actual).dtype == np.float64 and np.array_equal(actual, reference))
                with AssemblyProviders(hobj, partial, validate) as counts:
                    raw = hobj.hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ, mo1=mo1, mo_e1=moe1,
                        h1ao=h1, atmlst=list(range(n)), max_memory=4000, verbose=None)
                    raw = raw + hobj.hess_nuc(mf.mol, atmlst=list(range(n)))
                require(np.asarray(raw).shape == (n, n, 3, 3) and np.isfinite(raw).all(), "Full global Hessian incomplete")
                capture = {"schema": "research4-v8-full-global-assembly-capture-v1", "status": "DONE_COMPLETE_GLOBAL_ASSEMBLY",
                    "global_atom_ids": list(range(n)), "partial_result_dependency": partial_ref, "h1_result_dependency": h1_ref,
                    "response_index_dependency": index_ref, "response_result_dependencies": index_doc["results"],
                    "response_blocks": [{"block_index": s[0]["block_index"], "global_atom_ids": s[0]["global_atom_ids"],
                        "result": ref, "closure": s[3]} for ref, s in zip(index_doc["results"], responses)],
                    "contribution_calls": counts, "explicit_full_H1AO_and_MO1_MOE1_supplied": True,
                    "stock_hess_elec_returned_normally": True, "stock_nuclear_returned_normally": True,
                    "global_cross_atom_contractions_preserved": True, "nuclear_added_once": True,
                    "H1AO_source_referenced_without_duplicate_capture": True,
                    "CPHF_executed_in_this_phase": False, "all_response_blocks_healthily_closed": True,
                    "measured_CPHF_residual_norm": "NOT_COMPUTED", "IR_intensities_status": "NOT_COMPUTED",
                    "source_worker": artifact(__file__), "physical_binding_sha256": digest(physical)}
                save(output / "global_assembly_capture.json", capture, exclusive=True)
                state.update(global_assembly_capture_sha256=sha(output / "global_assembly_capture.json"),
                    full_global_contribution_calls=dict(counts), response_index_dependency=index_ref,
                    response_result_dependencies=index_doc["results"], full_system_CPHF_executed_in_this_phase=False,
                    full_system_CPHF_completion_basis="ALL_HEALTHY_FULL_AO_MO_B8_RESPONSE_PRODUCERS",
                    stock_hamiltonian_gate_after_provider_restore=v5.stock_operator_gate(mf, hobj))
                for name, coordinates in (("normal_mode_geometry_angstrom.npy", mf.mol.atom_coords(unit="Angstrom")),
                                          ("normal_mode_geometry_bohr.npy", mf.mol.atom_coords(unit="Bohr"))):
                    base.atomic_npy(output / name, coordinates)
                (output / "normal_mode_geometry.xyz").write_text(base.xyz_text(symbols, mf.mol.atom_coords(unit="Angstrom"), "V8 full original atom-order geometry"))
                v5.publish_modes(output, mf, raw, state, base)
            require(core.runtime_state_fingerprint(hobj) == physical["actual_runtime"], "Full source state changed inside V7 stage")
        require(v5.physical_state(mf) == before, "Exact source MF changed")
        state.update(stable_MF_state_unchanged=True, source_MF_stability_status="VERIFIED_UNCHANGED")
        for record in input_records:
            checked(record)
        for source in sources:
            for row in source[1]["arrays"]:
                checked({"path": str(source[2] / row["file"]), "sha256": row["file_sha256"]})
    except BaseException as exc:
        state.update(status="PARTIAL", exception_type=type(exc).__name__, stop_reason=str(exc))
        (output / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        native.restore(); state["native_direct_driver_control"] = native.summary()
        if state["native_direct_driver_control"].get("restoration_failures") != 0 or state["native_direct_driver_control"].get("original_driver_restored") is not True:
            state.update(status="PARTIAL", stop_reason="Native operator restoration failed")
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if state["status"] == "STAGE_COMPLETE" and state["stable_MF_state_unchanged"] and pending:
            save(output / "stage-cache/manifest.json", pending, exclusive=True)
            state["cache_manifest_sha256"] = sha(output / "stage-cache/manifest.json")
        state.update(completed_at_utc=utc(), process_cpu_seconds=time.process_time()-start_cpu,
            wall_seconds=time.perf_counter()-start_wall, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        log.flush(); log.close(); save(output / "result.json", state, exclusive=True)
    return state


def preflight(output):
    output = Path(output).resolve(); require(not output.exists(), "Fresh preparation output required")
    refs = []
    for p, expected in ((GUARD, GUARD_SHA), (LEGACY_GUARD, LEGACY_GUARD_SHA), (LEGACY_CONSUMER, LEGACY_CONSUMER_SHA),
                        (HELPER, HELPER_SHA), (PARTIAL_WORKER, PARTIAL_WORKER_SHA),
                        (XC_CORE, XC_CORE_SHA), (STATE, STATE_SHA), (V5, V5_SHA), (BASE, BASE_SHA), (CORE, CORE_SHA)):
        require(sha(p) == expected, "Frozen dependency changed")
        refs.append(artifact(p))
    output.mkdir(parents=True)
    result = {"status": "NOT_RUN", "schema": SCHEMA, "source_worker": artifact(__file__), "stages": list(STAGES),
        "physical_schema_reused_for_identical_mathematical_state_only": PHYSICAL_SCHEMA, "frozen_dependencies": refs,
        "partial_worker_status": "NOT_FROZEN" if len(PARTIAL_WORKER_SHA) != 64 else "PINNED_NOT_EXECUTED",
        "partial_worker_expected_sha256": PARTIAL_WORKER_SHA, "new_quantum_calculations": 0,
        "numerical_packages_imported": False, "arrays_read": False, "IR_intensities_status": "NOT_COMPUTED",
        "native_control_release": "NOT_RUN_REQUIRED_BEFORE_SPINOSIN", "budgets_modified": False,
        "old_V6b_sources_or_schemas_modified": False}
    save(output / "preflight.json", result, exclusive=True); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true"); p.add_argument("--stage", choices=STAGES)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--snapshot", type=Path)
    p.add_argument("--snapshot-manifest-sha256"); p.add_argument("--gradient-check-result", type=Path)
    p.add_argument("--gradient-check-result-sha256"); p.add_argument("--partial-result", type=Path)
    p.add_argument("--partial-result-sha256"); p.add_argument("--h1-result", type=Path); p.add_argument("--h1-result-sha256")
    p.add_argument("--block-index", type=int); p.add_argument("--response-index", type=Path); p.add_argument("--response-index-sha256")
    p.add_argument("--control-release", type=Path); p.add_argument("--control-release-sha256")
    p.add_argument("--threads", type=int, default=2); p.add_argument("--control-validation", action="store_true")
    args = p.parse_args(); result = execute(args) if args.execute else preflight(args.output)
    print(json.dumps({k: result.get(k) for k in ("status", "stage", "frequencies_status", "complete_global_Hessian")}))
    if args.execute and result["status"] not in ("STAGE_COMPLETE", "FREQUENCIES_COMPUTED"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
