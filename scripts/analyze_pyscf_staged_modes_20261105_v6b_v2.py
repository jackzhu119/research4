"""V6b-only closed staged-mode analysis; numerical imports occur after gates.

This is not a compatibility switch for the frozen V5 a2 analyzer. Genuine V5
source/G receipts retain their original schema; genuine V6b A/B/global producers
are verified with the frozen V6b parent. No SCF, gradients or Hessians are run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/pyscf_staged_hessian_worker_20261105_v6b.py"
WORKER_SHA = "f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d"
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
STATE = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
V5 = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
V5_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
A2 = ROOT / "scripts/analyze_spinosin_v5_modes_20261104.py"
A2_SHA = "a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154"
SCHEMA = "research4-staged-full-rks-hessian-result-v6b"
SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
CM1_TO_THZ = .0299792458
BOHR_TO_ANGSTROM = .52917721092
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}
CONTROL_SOURCES = {
    "water_software_control": {
        "snapshot_sha256": "123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4",
        "input_sha256": "865877bc001ebca8a5a9fde48dbb16cd9a21504a14b435c23a276bfb6f5e9ded",
        "identity": "validation_water_verified_old_optimized_geometry", "formula": "H2O",
        "stock_result": "reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-stock-hessian-001/result.json",
        "stock_result_sha256": "986be9aa60ff6810693169055e078d202dc7135f8a6ddf61c47f0a921018a930"},
    "ethanol_software_control": {
        "snapshot_sha256": "9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba",
        "input_sha256": "7c3ddf300ab6cb806a475a8e08c3a11944c3cc1e744332dab65ffbd52b638e07",
        "identity": "validation_ethanol_verified_old_optimized_geometry", "formula": "C2H6O",
        "stock_result": "reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-stock-hessian-001/result.json",
        "stock_result_sha256": "6c35b214b22e927185ca2298674fb8e20e1a6fede06bbecc33be22e7910d1e61"}}
FREQ_FIELDS = ("case", "mode", "frequency_cm1", "frequency_THz", "imaginary",
               "qualified_for_mechanism", "IR_intensity_status")
NUMERICAL = None


def require(value, message):
    if not value:
        raise ValueError(message)


def root_path(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value, *, compact=True):
    options = {"sort_keys": True, "allow_nan": False}
    if compact:
        options["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def checked(path, hashes, expected=None, *, maximum_bytes=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing/symlink artifact")
    require(maximum_bytes is None or path.stat().st_size <= maximum_bytes, "Artifact exceeds bounded metadata/array size")
    path = path.resolve(); value = sha(path)
    require(expected is None or value == expected, "Artifact hash differs: " + str(path))
    hashes[str(path)] = {"sha256": value, "bytes": path.stat().st_size, "producer_hash_present": expected is not None}
    return path


def read_json(path, hashes, expected=None):
    return json.loads(checked(path, hashes, expected, maximum_bytes=32 * 1024**2).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def frozen_module(path, expected, name):
    require(sha(path) == expected, "Frozen source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def frozen_metadata_subset(path, expected, functions, literal_constants, extra=None):
    """Use unmodified pure functions from frozen V5 without its top-level NumPy.

    Only named function definitions and literal constants are compiled. Imports,
    decorators and every other top-level statement are excluded. This does not
    translate old schemas or permit V6b producers through V5 validators.
    """
    require(sha(path) == expected, "Frozen V5 metadata source differs")
    tree = ast.parse(Path(path).read_text()); selected = []; constants = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            require(not node.decorator_list, "Metadata subset must not execute decorators")
            selected.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in literal_constants:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    require({n.name for n in selected} == set(functions) and set(constants) == set(literal_constants),
            "Frozen pure metadata interface changed")
    namespace = {"__file__": str(path), "ROOT": ROOT, "Path": Path, "json": json,
                 "hashlib": hashlib, "math": math, **constants, **(extra or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def v5_metadata_validators():
    state = frozen_metadata_subset(STATE, STATE_SHA,
        ("sha", "canonical", "document_hash", "resolve_path", "artifact_record", "checked_artifact", "validate_closed_producer"),
        ("BASE_SHA", "SUPERVISOR_SHA", "RESULT_SCHEMA"))
    consumer = frozen_metadata_subset(V5, V5_SHA,
        ("sha", "canonical", "healthy_closed_row", "verify_closed_execution", "qualify_gradient_receipt"),
        ("SCHEMA", "SUCCESS", "EXECUTION_SCHEMA", "SUPERVISOR_SHA"), {"state_module": lambda: state})
    return state, consumer


def case_for(symbols, control):
    composition = dict(Counter(symbols))
    if not control:
        require(composition == {"C": 28, "H": 32, "O": 15} and len(symbols) == 75, "Default is original Spinosin75 only")
        return "Spinosin"
    for label, wanted in (("water_software_control", {"H": 2, "O": 1}),
                          ("ethanol_software_control", {"C": 2, "H": 6, "O": 1})):
        if composition == wanted:
            return label
    raise ValueError("Explicit control mode permits only real water/ethanol")


def control_identity_gate(case, snapshot_sha, producer, input_document):
    """Formula alone never authenticates ethanol versus its C2H6O isomers."""
    expected = CONTROL_SOURCES[case]
    require(snapshot_sha == expected["snapshot_sha256"]
            and producer["identity"] == input_document["molecule_id"] == expected["identity"]
            and producer["input_manifest"]["sha256"] == expected["input_sha256"]
            and input_document["formula"] == expected["formula"]
            and input_document["connectivity_verified"] is True,
            "Control requires exact trusted old source/input, never formula-only chemical identity")


def global_header(result, *, control):
    require(result.get("schema") == SCHEMA and result.get("stage") == result.get("phase") == "global"
            and result.get("status") == "FREQUENCIES_COMPUTED" and result.get("frequencies_status") == "DONE",
            "Only an actual complete V6b global result is eligible")
    for key in ("no_new_SCF_performed", "stable_MF_state_unchanged", "complete_RHF_helper", "complete_RKS_partial",
                "complete_H1AO", "complete_global_Hessian", "normal_modes_complete", "final_gradient_thresholds_passed",
                "full_system_CPHF_returned_normally", "harmonic_analysis_excludes_translations_and_rotations"):
        require(result.get(key) is True, "Incomplete global evidence: " + key)
    n = result.get("atoms")
    require(type(n) is int and n in ((3, 9) if control else (75,)), "Control/default atom count differs")
    require(result.get("normal_mode_count") == result.get("expected_normal_mode_count") == 3 * n - 6,
            "Incomplete nonlinear normal-mode count")
    require(result.get("source_MF_stability_status") == "VERIFIED_UNCHANGED"
            and result.get("IR_intensities_status") == "NOT_COMPUTED"
            and result.get("measured_CPHF_residual_norm") == "NOT_COMPUTED",
            "Do not infer measured CPHF residuals or IR intensities")
    require(result.get("full_global_contribution_calls") == {"partial": 1, "h1": 1, "nuclear": 1},
            "Global contributions missing or double-added")
    require(result.get("V5_analyzer_compatibility") == "BLOCKED_V6B_PROOF_SCHEMA_REQUIRES_REVIEWED_V6B_POSTPROCESSOR",
            "Never relabel V6b as V5 compatibility")
    return n


def command_option(command, name):
    require(command.count(name) == 1 and command.index(name) + 1 < len(command), "Actual closed command lacks unique " + name)
    return command[command.index(name) + 1]


def stage_command_gate(command, args, result, output, stage, control, source_args):
    """Anchor input.json declarations to each actual closed parent argv."""
    require(command.count("--control-validation") == int(control) and args["control_validation"] is control,
            "Control/default flag differs from actual producer command")
    require(command.count("--execute") == 1 and args["execute"] is True
            and args["stage"] == command_option(command, "--stage") == stage
            and root_path(command_option(command, "--output")) == root_path(output)
            and int(command_option(command, "--threads")) == args["threads"] == result["settings"]["threads"],
            "Actual stage/output/thread arguments differ")
    keys = [("--snapshot", "snapshot"), ("--snapshot-manifest-sha256", "snapshot_manifest_sha256"),
            ("--gradient-check-result", "gradient_check_result"),
            ("--gradient-check-result-sha256", "gradient_check_result_sha256")]
    if stage in ("h1ao", "global"):
        keys += [("--partial-result", "partial_result"), ("--partial-result-sha256", "partial_result_sha256")]
    if stage == "global":
        keys += [("--h1-result", "h1_result"), ("--h1-result-sha256", "h1_result_sha256")]
    if stage == "rks-partial":
        keys += [("--rhf-cache", "rhf_cache")]
    for flag, key in keys:
        observed = command_option(command, flag)
        require((root_path(observed) == root_path(args[key])) if not key.endswith("sha256") else observed == args[key],
                "Input.json differs from actual closed command: " + key)
    for key in ("snapshot", "snapshot_manifest_sha256", "gradient_check_result", "gradient_check_result_sha256", "threads"):
        require((root_path(args[key]) == root_path(source_args[key])) if key in ("snapshot", "gradient_check_result")
                else args[key] == source_args[key], "Actual stages changed original source/G/threads: " + key)
    if stage == "h1ao":
        require(root_path(args["partial_result"]) == root_path(source_args["partial_result"])
                and args["partial_result_sha256"] == source_args["partial_result_sha256"], "B actual command does not consume global A")
    require(args["snapshot_manifest_sha256"] == result["snapshot_manifest_sha256"], "Stage snapshot argument differs")


def closed_command(result, hashes, guard):
    proof = result["resource_supervision"]["proof"]
    ledger_path = Path(proof["ledger_path"])
    ledger = read_json(ledger_path, hashes)
    guard.validate_ledger(ledger, allow_running=False)
    rows = [r for r in ledger["phases"] if r.get("phase") == proof["phase"]]
    require(len(rows) == 1 and guard.healthy_closed_row(rows[0]), "Actual producing phase is not healthily closed")
    row = rows[0]
    require(row["actual_cpu_seconds"] <= proof["cpu_hard_limit_seconds"], "Producer exceeded its CPU grant")
    checked(row["log"], hashes, row["log_sha256"])
    checked(result["resource_supervision"]["proof_path"], hashes, result["resource_supervision"]["proof_sha256"])
    return row["command"]


def retain_v5_closure(result_path, supervision, closure, hashes):
    """Keep independently validated V5 receipt bytes in the final race check."""
    proof = read_json(root_path(supervision["proof_path"]), hashes, closure["proof_sha256"])
    require(proof == supervision["proof"], "Original V5 proof document differs")
    ledger = read_json(root_path(proof["ledger_path"]), hashes)
    rows = [r for r in ledger["phases"] if r.get("phase") == closure["phase"]]
    require(len(rows) == 1 and digest(rows[0]) == closure["closed_row_sha256"],
            "Original V5 closed row changed during metadata qualification")
    # The genuine state-producer closure contains log_sha256; the genuine
    # fresh-G verify_closed_execution closure intentionally does not. Both
    # authenticate this same closed row, whose hash-bound log is authoritative.
    log_sha = rows[0]["log_sha256"]
    require(isinstance(log_sha, str) and len(log_sha) == 64
            and all(c in "0123456789abcdef" for c in log_sha), "Closed V5 log SHA is missing or malformed")
    require("log_sha256" not in closure or closure["log_sha256"] == log_sha, "Optional closure log SHA contradicts its bound row")
    checked(root_path(rows[0]["log"]), hashes, log_sha)
    checked(root_path(result_path), hashes, closure.get("closed_result_sha256", closure.get("producer_result_sha256")))


def source_gate(result, input_args, control, hashes):
    """Actual V5 snapshot and freshG provenance, before any numerical import."""
    state, v5 = v5_metadata_validators()
    snapshot_dir = Path(result["snapshot_directory"]).resolve()
    require(snapshot_dir == root_path(input_args["snapshot"]), "Global source snapshot path differs")
    snapshot_sha = result["snapshot_manifest_sha256"]
    manifest_path = snapshot_dir / "manifest.json"
    manifest = read_json(manifest_path, hashes, snapshot_sha)
    require(manifest.get("schema") == "research4-exact-pbe-mf-snapshot-v1"
            and manifest.get("status") == "EXACT_SCF_GRADIENT_STATE_SAVED"
            and manifest.get("source_worker", {}).get("sha256") == STATE_SHA,
            "Original V5 snapshot metadata differs")
    require(state.checked_artifact(manifest["source_worker"]) == STATE
            and manifest["frozen_V8_sha256"] == state.BASE_SHA and manifest["supervisor_sha256"] == state.SUPERVISOR_SHA,
            "Original snapshot sources differ")
    original_closure = state.validate_closed_producer(manifest, manifest_path)
    retain_v5_closure(manifest["producer"]["result_path"], manifest["producer"]["resource_supervision"],
                      original_closure, hashes)
    molecule = read_json(snapshot_dir / "molecule.json", hashes, manifest["molecule"]["sha256"])
    symbols = [r[0] for r in molecule["_atom"]]; case = case_for(symbols, control)
    physical = manifest["molecule_fingerprint"]
    require(physical["natm"] == result["atoms"] == len(symbols) and physical["charge"] == physical["spin"] == 0
            and physical["cart"] is False and physical["nao"] == result["basis_functions"], "Source molecular identity differs")
    require(result["case"] == case, "Reported case does not match the actual source molecule")
    g = result["gradient_gate"]["result"]
    require(root_path(g["path"]) == root_path(input_args["gradient_check_result"])
            and g["sha256"] == input_args["gradient_check_result_sha256"], "Actual fresh-G input differs")
    gradient = read_json(g["path"], hashes, g["sha256"])
    qualified_g = v5.qualify_gradient_receipt(g["path"], g["sha256"], snapshot_sha, result["boot_id"])
    require(qualified_g == result["gradient_gate"], "Genuine V5 same-state fresh-G gate differs")
    retain_v5_closure(g["path"], gradient["resource_supervision"], qualified_g["producer"], hashes)
    if not control:
        require(snapshot_sha == SNAPSHOT_SHA and g["sha256"] == GRADIENT_SHA, "Original Spinosin C001 source/G required")
        require("spinosin" in str(manifest["producer"]["identity"]).lower(), "Source not declared Spinosin")
    else:
        control_input = read_json(root_path(manifest["producer"]["input_manifest"]["path"]), hashes,
                                  manifest["producer"]["input_manifest"]["sha256"])
        control_identity_gate(case, snapshot_sha, manifest["producer"], control_input)
    binding = {"schema": manifest["schema"], "snapshot_manifest_sha256": snapshot_sha,
               "snapshot_directory": str(snapshot_dir), "producer": manifest["producer"],
               "molecule_fingerprint": physical, "arrays": manifest["arrays"], "environment": manifest["environment"],
               "method_settings": manifest["settings"], "source_worker": manifest["source_worker"], "closed_producer": original_closure}
    require(binding == result["source_state_binding"] and result["energy_hartree"] == manifest["energy_hartree"],
            "Exact original source binding/energy differs")
    return {"snapshot_manifest": manifest, "molecule": molecule, "symbols": symbols, "case": case,
            "binding": binding, "source_closure": original_closure, "gradient_result": gradient,
            "gradient_gate": qualified_g}


def stage_capture_gate(capture, result, partial_ref, h1_ref, n):
    expected = {"status": "DONE_INTERMEDIATE_CAPTURE_ONLY", "native_global_kernel_returned_normally": True,
        "H1AO_complete": True, "global_Hessian_assembly_unchanged": True, "partial_kernel_tiling_used": False,
        "response_block_atoms": 8, "native_conv_tol_cpscf": 1e-8, "native_max_cycle": 50, "native_level_shift": 0,
        "independent_CPHF_residual_norm": "NOT_COMPUTED"}
    require(all(capture.get(k) == v for k, v in expected.items()), "Complete stock global response metadata missing")
    require(capture["dimensions"]["atoms"] == n and capture["response_complete_global_atom_ids"] == list(range(n))
            and [r["atom_id"] for r in capture["H1AO_atoms"]] == list(range(n)), "Global atom coverage incomplete")
    partial = capture["partial_hessian"]
    require(partial.get("native_function_returned_normally") is False
            and partial.get("cache_provider_returned_normally") is True
            and partial.get("native_partial_executed_in_this_phase") is False
            and partial.get("healthy_native_partial_source") == partial_ref,
            "V6b reused partial must not be labelled a new native partial call")
    origin = {"partial": "HEALTHY_COMPLETE_V6B_CACHE_COPY", "H1AO": "HEALTHY_COMPLETE_V6B_CACHE_FRESH_LOADS",
        "CPHF": "FRESH_FULL_SYSTEM_STOCK_B8", "nuclear": "FRESH_STOCK_ONCE", "native_make_h1_executed_in_this_phase": False,
        "H1AO_capture_duplicates_closed_source_bytes": 3 * n * result["basis_functions"]**2 * 8,
        "duplicate_H1_capture_IO_counted_not_elided": True}
    require(capture["V6B_stage_execution_origin"] == result["RKS_partial_and_H1_cache_restore"] == origin,
            "V6b contribution origin differs")
    binding = capture["binding"]; evidence = binding["input_evidence"]
    require(digest(binding, compact=False) == capture["binding_sha256"] and binding["settings"] == result["settings"],
            "Global capture binding differs")
    require(evidence["source_snapshot_sha256"] == result["snapshot_manifest_sha256"]
            and evidence["actual_entry_source_sha256"] == WORKER_SHA
            and evidence["consumer_resource_proof_sha256"] == result["resource_supervision"]["proof_sha256"]
            and evidence["partial_stage_result_sha256"] == partial_ref["sha256"]
            and evidence["H1_stage_result_sha256"] == h1_ref["sha256"]
            and evidence["geometry_optimized"] is True and evidence["final_gradient_thresholds_passed"] is True,
            "Global source/parent/A/B input evidence differs")
    chunks = [list(range(i, min(i + 8, n))) for i in range(0, n, 8)]
    require([b["global_atom_ids"] for b in capture["response_blocks"]] == chunks, "Full B8 response coverage differs")
    for block, atoms in zip(capture["response_blocks"], chunks):
        require(block["native_solver_returned_normally"] is True and block["residual_norm"] == "NOT_COMPUTED"
                and [r["atom_id"] for r in block["arrays"]] == atoms, "Response block incomplete")
    return {"status": "FULL_V6B_GLOBAL_CACHE_ORIGIN_AND_RESPONSE_METADATA_PASS", "CPHF_residual_measured": False}


def control_reference_gate(source, hashes):
    """Authenticate a genuine old V5 STOCK result, never relabel it as V6b."""
    expected = CONTROL_SOURCES[source["case"]]
    path = root_path(expected["stock_result"])
    value = read_json(path, hashes, expected["stock_result_sha256"])
    _, v5 = v5_metadata_validators()
    require(value["schema"] == v5.SCHEMA and value["status"] == "FREQUENCIES_COMPUTED"
            and value["phase"] == "stock-hessian-reference" and value["frequencies_status"] == "DONE"
            and value["complete_global_Hessian"] is True and value["normal_modes_complete"] is True
            and value["source_state_binding"] == source["binding"]
            and value["snapshot_manifest_sha256"] == expected["snapshot_sha256"],
            "Old control reference is not a complete exact-source V5 STOCK calculation")
    evidence = {"schema": v5.EXECUTION_SCHEMA, "source_worker": value["source_worker"],
        "resource_supervision": value["resource_supervision"], "output_directory": str(path.parent),
        "snapshot_manifest_sha256": value["snapshot_manifest_sha256"], "phase": value["phase"]}
    closure = v5.verify_closed_execution(evidence, expected_snapshot_sha=value["snapshot_manifest_sha256"], expected_output=path.parent)
    require(closure["producer_result_sha256"] == expected["stock_result_sha256"], "Old stock closed result differs")
    retain_v5_closure(path, value["resource_supervision"], closure, hashes)
    return {"result": value, "path": str(path), "closure": closure}


def qualify(result_path, expected_sha, *, control=False, hashes=None):
    """All source/parent/A/B/capture gates before the first NumPy/array read."""
    hashes = {} if hashes is None else hashes
    path = Path(result_path).resolve()
    require(path.is_relative_to(ROOT / "reports") and path.name == "result.json", "Explicit report result required")
    result = read_json(path, hashes, expected_sha); n = global_header(result, control=control)
    worker = frozen_module(WORKER, WORKER_SHA, "frozen_v6b_staged_metadata")
    guard = frozen_module(GUARD, GUARD_SHA, "frozen_v6b_closed_parent")
    evidence = {"source_worker": result["source_worker"], "resource_supervision": result["resource_supervision"],
        "output_directory": str(path.parent), "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "stage": "global"}
    closure = guard.verify_closed_phase(evidence, expected_worker_path=WORKER, expected_worker_sha256=WORKER_SHA,
        expected_result_schema=SCHEMA, allowed_result_statuses=("FREQUENCIES_COMPUTED",),
        expected_output=path.parent, expected_snapshot_sha256=result["snapshot_manifest_sha256"])
    require(closure["producer_result_sha256"] == expected_sha, "Global closed result SHA differs")
    command = closed_command(result, hashes, guard)
    args = read_json(path.parent / "input.json", hashes)["arguments"]
    stage_command_gate(command, args, result, path.parent, "global", control, args)
    source = source_gate(result, args, control, hashes)
    partial_ref = {"path": str(root_path(args["partial_result"])), "sha256": args["partial_result_sha256"]}
    h1_ref = {"path": str(root_path(args["h1_result"])), "sha256": args["h1_result_sha256"]}
    a = read_json(partial_ref["path"], hashes, partial_ref["sha256"])
    b = read_json(h1_ref["path"], hashes, h1_ref["sha256"])
    stage_args = {}
    # Independently close BOTH actual producers and inspect their bound argv
    # before the frozen cache qualifier hashes any large stage payload bytes.
    for record, value, stage in ((partial_ref, a, "rks-partial"), (h1_ref, b, "h1ao")):
        directory = root_path(record["path"]).parent
        evidence = {"source_worker": value["source_worker"], "resource_supervision": value["resource_supervision"],
            "output_directory": str(directory), "snapshot_manifest_sha256": value["snapshot_manifest_sha256"], "stage": stage}
        closed = guard.verify_closed_phase(evidence, expected_worker_path=WORKER, expected_worker_sha256=WORKER_SHA,
            expected_result_schema=SCHEMA, allowed_result_statuses=("STAGE_COMPLETE",), expected_output=directory,
            expected_snapshot_sha256=result["snapshot_manifest_sha256"])
        require(closed["producer_result_sha256"] == record["sha256"], "Closed A/B result SHA differs")
        stage_args[stage] = read_json(directory / "input.json", hashes)["arguments"]
        stage_command_gate(closed_command(value, hashes, guard), stage_args[stage], value, directory, stage, control, args)
        require(value["source_state_binding"] == result["source_state_binding"]
                and value["gradient_gate"] == result["gradient_gate"] and value["boot_id"] == result["boot_id"],
                "Stages changed source state, fresh gradient or recorded boot")
    _, v5_pure = v5_metadata_validators()
    rhf_producers = worker.verify_rhf_producers(root_path(stage_args["rks-partial"]["rhf_cache"]),
        result["snapshot_manifest_sha256"], guard, v5_pure)
    before, after = a["incremental_before"]["total_units_completed"], a["incremental_after"]["total_units_completed"]
    require(len(rhf_producers) == after == 2 * n and 0 <= before <= after
            and rhf_producers[:before] == a["prior_RHF_unit_producer_checks"],
            "Complete healthy RHF source chain differs from actual A prefix")
    for row in rhf_producers:
        checked(row["record"]["path"], hashes, row["record"]["sha256"])
    manifest = read_json(Path(partial_ref["path"]).parent / "stage-cache/manifest.json", hashes, a["cache_manifest_sha256"])
    physical = manifest["physical_binding"]
    require(physical["stable_snapshot_binding"] == source["binding"]
            and physical["snapshot_manifest_sha256"] == result["snapshot_manifest_sha256"]
            and physical["numerical_policy"] == worker.NUMERICAL_POLICY
            and physical["schema"] == worker.PHYSICAL_SCHEMA
            and digest(physical) == result["physical_binding_sha256"], "Global/A original physical state differs")
    # Source/G metadata is now closed. These calls verify complete A/B manifests
    # and hash payload BYTES, never load their potentially-gigantic H1 arrays.
    a_value, a_manifest, a_dir, a_closure = worker.qualify_stage(partial_ref, "rks-partial", physical, n, result["basis_functions"], guard)
    b_value, b_manifest, b_dir, b_closure = worker.qualify_stage(h1_ref, "h1ao", physical, n, result["basis_functions"], guard)
    require(b_value["partial_result_dependency"] == partial_ref
            and result["partial_stage_producer"] == a_closure and result["H1_stage_producer"] == b_closure,
            "Global and H1 do not share the actual A/B producers")
    read_json(h1_ref["path"], hashes, h1_ref["sha256"])
    read_json(b_dir / "manifest.json", hashes, b_value["cache_manifest_sha256"])
    for directory, cache in ((a_dir, a_manifest), (b_dir, b_manifest)):
        for row in cache["arrays"]:
            # Frozen qualify_stage just checked these exact bytes; record its
            # evidence without a second first-pass read of the large H1 files.
            path_key = str((directory / row["file"]).resolve())
            hashes[path_key] = {"sha256": row["file_sha256"], "bytes": Path(path_key).stat().st_size,
                                "producer_hash_present": True, "verified_by": "FROZEN_V6B_qualify_stage"}
    capture = read_json(path.parent / "hessian_cache_manifest.json", hashes, result["hessian_cache_manifest_sha256"])
    kernel = stage_capture_gate(capture, result, partial_ref, h1_ref, n)
    control_reference = control_reference_gate(source, hashes) if control else None
    return {"result": result, "source": source, "closure": closure, "partial_ref": partial_ref, "h1_ref": h1_ref,
            "partial_closure": a_closure, "H1_closure": b_closure, "RHF_producers": rhf_producers,
            "physical_binding": physical, "kernel": kernel,
            "control_reference": control_reference,
            "result_path": str(path), "hashes": hashes, "metadata_gate_passed_before_numerical_import": True}


def eigenvalue_clusters(values, *, atol=1e-8, rtol=1e-6):
    """Fixed near-degenerate clusters in Eh/(amu Bohr^2); no fit/peak choice."""
    values = [float(v) for v in values]
    require(values and all(math.isfinite(v) for v in values)
            and all(b >= a for a, b in zip(values, values[1:])), "Sorted finite eigenvalues required")
    groups = [[0]]
    for index in range(1, len(values)):
        gap = values[index] - values[index - 1]
        if gap <= atol + rtol * max(abs(values[index]), abs(values[index - 1])):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def numpy_runtime(gate):
    global NUMERICAL
    require(gate.get("metadata_gate_passed_before_numerical_import") is True, "No numerical imports before real provenance gates")
    if NUMERICAL is None:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        import numpy as np
        a2 = frozen_module(A2, A2_SHA, "frozen_a2_numerical_helpers_only_not_V6b_gate")
        require(a2.MASS == MASS and a2.CM1_TO_THZ == CM1_TO_THZ, "Validated mass/unit conventions changed")
        NUMERICAL = np, a2
    return NUMERICAL


def array(path, shape, hashes, np, expected=None):
    path = checked(path, hashes, expected, maximum_bytes=64 * 1024**2)
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(), "Invalid complete numerical array")
    return value


def independent_subspace_audit(raw, used, modes, cm, coords_bohr, symbols, np, a2):
    base_audit = a2.numpy_mode_audit(raw, used, modes, cm, coords_bohr, symbols)
    replay = a2.load_frozen(a2.REPLAY, a2.REPLAY_SHA, "V6b_independent_numpy_TR_projector")
    n = len(symbols); mass = np.asarray([MASS[s] for s in symbols], dtype=np.float64)
    projection, basis = replay.projector(coords_bohr, mass)
    matrix = np.einsum("pqxy,p,q->pqxy", used, mass**-.5, mass**-.5).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    eigenvalues, eigenvectors = np.linalg.eigh(basis.T @ matrix @ basis)
    independent = basis @ eigenvectors
    saved = (modes * np.sqrt(mass)[None, :, None]).reshape(3 * n - 6, 3 * n).T
    errors = []
    for indices in eigenvalue_clusters(eigenvalues.tolist()):
        left = independent[:, indices]; right = saved[:, indices]
        error = float(np.max(np.abs(left @ left.T - right @ right.T)))
        singular = np.linalg.svd(left.T @ right, compute_uv=False)
        angle_error = float(np.max(np.abs(singular - 1.)))
        require(error <= 1e-6 and angle_error <= 1e-6, "Independent eigenvector/near-degenerate subspace differs")
        errors.append({"modes_one_based": [i + 1 for i in indices], "projector_max_error": error,
                       "principal_overlap_singular_value_max_error": angle_error})
    return {**base_audit, "independent_eigenvectors_and_subspaces": errors,
            "subspace_projector_and_overlap_tolerance": 1e-6,
            "eigenvalue_cluster_absolute_gap_Eh_per_amu_Bohr2": 1e-8,
            "eigenvalue_cluster_relative_gap": 1e-6,
            "signs_and_within_cluster_orthogonal_rotations_are_arbitrary": True,
            "independent_mass_convention": MASS}


def validate_minimum_flags(result, frequencies):
    values = [float(v) for v in frequencies]
    require(values and all(math.isfinite(v) for v in values), "Actual finite signed frequencies required")
    positive = all(v > 0 for v in values)
    require(result["imaginary_frequency_count"] == sum(v < 0 for v in values)
            and result["all_frequencies_real_positive"] is positive
            and result["stationary_minimum_verified"] is positive
            and result["frequency_validation_passed"] is positive,
            "Minimum/imaginary flags contradict actual complete signed frequencies")
    return positive


def compare_actual_control(reference, raw, used, modes, cm, coords_bohr, symbols, hashes, np, a2):
    """Future real all-matrix/all-mode comparison; numerical execution NOT_RUN now."""
    value, location = reference["result"], Path(reference["path"]).parent
    n, count = len(symbols), 3 * len(symbols) - 6
    old_raw = array(location / "hessian_hartree_per_bohr2.npy", (n, n, 3, 3), hashes, np, value["raw_hessian_sha256"])
    old_used = array(location / "hessian_used_for_analysis.npy", old_raw.shape, hashes, np, value["analysis_hessian_sha256"])
    old_modes = array(location / "normal_modes.npy", (count, n, 3), hashes, np)
    old_coords = array(location / "normal_mode_geometry_bohr.npy", (n, 3), hashes, np)
    old_cm, old_thz = np.asarray(value["frequencies_cm1"], dtype=float), np.asarray(value["frequencies_thz"], dtype=float)
    require(np.array_equal(old_coords, coords_bohr), "Old stock and V6b use different source coordinates")
    with checked(location / "frequencies.csv", hashes).open(newline="") as handle:
        a2.validate_frequency_rows(list(csv.DictReader(handle)), old_cm, old_thz)
    old_audit = independent_subspace_audit(old_raw, old_used, old_modes, old_cm, old_coords, symbols, np, a2)
    raw_error, used_error = float(np.max(np.abs(raw - old_raw))), float(np.max(np.abs(used - old_used)))
    require(raw_error <= 1e-8 and used_error <= 1e-8 and np.allclose(cm, old_cm, atol=1e-5, rtol=1e-9),
            "Real V6b control differs from exact-source old stock full matrix/all signed modes")
    replay = a2.load_frozen(a2.REPLAY, a2.REPLAY_SHA, "V6b_control_saved_mode_comparison")
    eigenvalues = np.sign(old_cm) * (np.abs(old_cm) / replay.FACTOR)**2
    mass = np.asarray([MASS[s] for s in symbols], dtype=float)
    left = (modes * np.sqrt(mass)[None, :, None]).reshape(count, 3 * n).T
    right = (old_modes * np.sqrt(mass)[None, :, None]).reshape(count, 3 * n).T
    subspace_errors = []
    for indices in eigenvalue_clusters(eigenvalues.tolist()):
        a, b = left[:, indices], right[:, indices]
        error = float(np.max(np.abs(a @ a.T - b @ b.T)))
        require(error <= 1e-5, "Actual V6b/old-stock control mode subspace differs")
        subspace_errors.append({"modes_one_based": [i + 1 for i in indices], "projector_max_error": error})
    return {"status": "PASS_ACTUAL_V6B_VS_AUTHENTICATED_V5_STOCK_FULL_MATRIX_AND_ALL_MODES",
        "old_stock_result": reference["path"], "old_stock_closed_producer": reference["closure"],
        "raw_Hessian_max_abs_difference_Eh_per_Bohr2": raw_error,
        "analysis_Hessian_max_abs_difference_Eh_per_Bohr2": used_error,
        "all_signed_frequency_max_abs_difference_cm1": float(np.max(np.abs(cm - old_cm))),
        "mode_count": count, "old_stock_independent_replay": old_audit, "mode_subspaces": subspace_errors,
        "matrix_atol_Eh_per_Bohr2": 1e-8, "frequency_atol_cm1": 1e-5, "frequency_rtol": 1e-9,
        "mode_subspace_atol": 1e-5, "control_is_not_medicinal_plant_evidence": True}


def mechanism_qualified(values, *, control, fresh_gradient_verified):
    return bool(not control and fresh_gradient_verified and len(values) == 219
                and all(math.isfinite(float(v)) and float(v) > 0 for v in values))


def seal(output):
    files = [{"path": str(p.relative_to(output)), "sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "artifact_index.json", {"analysis_source_sha256": sha(__file__), "a2_numerical_helper_sha256": A2_SHA,
               "V6b_worker_sha256": WORKER_SHA, "files": files})


def _analyze_closed(result_path, expected_sha, output, *, control=False):
    output = Path(output).resolve(); require(not output.exists(), "Preserve earlier analysis; use fresh output")
    gate = qualify(result_path, expected_sha, control=control)
    np, a2 = numpy_runtime(gate)
    result, source, hashes = gate["result"], gate["source"], gate["hashes"]
    # The original, unmodified V5 payload verifier is used only AFTER the new
    # pure V6b+A+B+source/G gate. No MF is restored and no PySCF engine imported.
    v5 = frozen_module(V5, V5_SHA, "genuine_V5_source_payload_verifier_after_gate")
    state = v5.state_module()
    manifest, molecule, source_arrays, binding = state.load_snapshot_payload(result["snapshot_directory"],
        expected_manifest_sha256=result["snapshot_manifest_sha256"])
    require(binding == source["binding"], "Numerically verified exact source binding differs")
    for record in manifest["arrays"].values():
        source_path = Path(result["snapshot_directory"]) / record["file"]
        hashes[str(source_path.resolve())] = {"sha256": record["sha256"], "bytes": source_path.stat().st_size,
            "producer_hash_present": True, "verified_by": "FROZEN_V5_load_snapshot_payload"}
    qualified_g = v5.qualify_gradient_receipt(source["gradient_gate"]["result"]["path"],
        source["gradient_gate"]["result"]["sha256"], result["snapshot_manifest_sha256"], result["boot_id"])
    require(qualified_g == source["gradient_gate"], "Original V5 numerical-side G validation differs")
    symbols, case = source["symbols"], source["case"]
    n, count, location = len(symbols), 3 * len(symbols) - 6, Path(result_path).resolve().parent
    g_record = source["gradient_gate"]["result"]
    gradient = array(Path(g_record["path"]).parent / "gradient_hartree_per_bohr.npy", (n, 3), hashes, np,
                     source["gradient_result"]["gradient_npy_sha256"])
    delta_g = float(np.max(np.abs(gradient - source_arrays["gradient"])))
    lengths = np.linalg.norm(source_arrays["gradient"], axis=1)
    require(delta_g <= 1e-8 and np.sqrt(np.mean(lengths**2)) <= 3e-4 and lengths.max() <= 4.5e-4,
            "Exact source/fresh restored gradient is inconsistent or not stationary")
    require(np.isclose(delta_g, source["gradient_result"]["restored_gradient_max_abs_difference_hartree_per_bohr"],
                       atol=1e-13, rtol=0), "Fresh-gradient numerical metadata differs")
    xyz = array(location / "normal_mode_geometry_angstrom.npy", (n, 3), hashes, np)
    bohr = array(location / "normal_mode_geometry_bohr.npy", (n, 3), hashes, np)
    require(np.array_equal(xyz, source_arrays["coordinates_angstrom"])
            and state.frozen_base().array_fingerprint(bohr) == manifest["molecule_fingerprint"]["coordinates_bohr"]
            and np.allclose(bohr * BOHR_TO_ANGSTROM, xyz, atol=1e-10, rtol=0), "Source atom order/coordinates differ")
    raw = array(location / "hessian_hartree_per_bohr2.npy", (n, n, 3, 3), hashes, np, result["raw_hessian_sha256"])
    used = array(location / "hessian_used_for_analysis.npy", raw.shape, hashes, np, result["analysis_hessian_sha256"])
    modes = array(location / "normal_modes.npy", (count, n, 3), hashes, np)
    reduced = array(location / "reduced_mass_amu.npy", (count,), hashes, np)
    require(np.allclose(reduced, 1 / np.sum(modes**2, axis=(1, 2)), atol=1e-10, rtol=1e-10), "Reduced masses disagree with modes")
    cm, thz = np.asarray(result["frequencies_cm1"], dtype=float), np.asarray(result["frequencies_thz"], dtype=float)
    require(cm.shape == thz.shape == (count,), "Actual complete frequency arrays required")
    with checked(location / "frequencies.csv", hashes).open(newline="") as handle:
        a2.validate_frequency_rows(list(csv.DictReader(handle)), cm, thz)
    numerical = independent_subspace_audit(raw, used, modes, cm, bohr, symbols, np, a2)
    control_comparison = compare_actual_control(gate["control_reference"], raw, used, modes, cm, bohr,
        symbols, hashes, np, a2) if control else "NOT_A_SOFTWARE_CONTROL"
    positive = validate_minimum_flags(result, cm.tolist())
    qualified = mechanism_qualified(cm.tolist(), control=control, fresh_gradient_verified=True)
    frequency_rows = [{"case": case, "mode": i + 1, "frequency_cm1": float(v), "frequency_THz": float(thz[i]),
        "imaginary": bool(v < 0), "qualified_for_mechanism": qualified, "IR_intensity_status": "NOT_COMPUTED"}
        for i, v in enumerate(cm)]
    low = [r for r in frequency_rows if .1 <= r["frequency_THz"] <= 1.5]
    experimental = a2.experimental_candidates(hashes) if qualified else None
    pairs = a2.cartesian_pairs(experimental, low) if qualified else None
    output.mkdir(parents=True)
    a2.csvwrite(output / "all_frequencies_cm1_THz.csv", frequency_rows, FREQ_FIELDS)
    a2.csvwrite(output / "normal_modes_0p1_to_1p5_THz.csv", low, FREQ_FIELDS)
    if pairs is not None:
        fields = [*experimental[0], "theoretical_mode", "theoretical_frequency_THz", "theoretical_wavenumber_cm1",
                  "signed_delta_THz", "absolute_delta_THz", "signed_delta_cm1", "absolute_delta_cm1",
                  "IR_intensity_status", "chemical_attribution_verified", "comparison"]
        a2.csvwrite(output / "all_experimental_theory_pairs.csv", pairs, fields)
    a2.plots(output, frequency_rows, low, case + (" [not an all-positive minimum]" if not positive else ""), experimental)
    chosen = ([frequency_rows[0]] if control else [r for r in frequency_rows if r["imaginary"]][:3] + low[:6])
    animations = []
    if chosen:
        (output / "animations").mkdir()
    for row in chosen:
        target = output / "animations" / f"{case}_mode_{row['mode']:03d}.xyz"
        label = case + (" IMAGINARY_DIAGNOSTIC_NOT_MINIMUM" if row["imaginary"] else "")
        a2.animate(target, symbols, xyz, modes[row["mode"] - 1], row["mode"], row["frequency_cm1"], label)
        animations.append(target.name)
    for path, record in hashes.items():
        require(sha(path) == record["sha256"], "Closed input changed during postprocessing: " + path)
    require(not any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules), "Analysis must not import a quantum engine")
    checks = {"schema": "research4-V6b-independent-staged-modes-analysis-v1",
        "status": "DONE_CONTROL_NUMERICAL_REPLAY" if control else "DONE_COMPUTED_MODES_AUDITED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "case": case,
        "analysis_source_sha256": sha(__file__), "actual_V6b_worker_sha256": WORKER_SHA, "actual_V6b_parent_sha256": GUARD_SHA,
        "a2_reused_for_numerical_helpers_only": A2_SHA, "old_a2_V6b_compatibility_claimed": False,
        "source_result_sha256": expected_sha, "producer_closure": gate["closure"],
        "partial_producer": gate["partial_closure"], "H1_producer": gate["H1_closure"],
        "healthy_RHF_unit_producers": gate["RHF_producers"],
        "source_SCF_closed_producer": source["source_closure"], "restored_gradient_gate": source["gradient_gate"],
        "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "physical_binding_sha256": result["physical_binding_sha256"],
        "metadata_provenance_gates_before_numpy_or_array_access": True,
        "global_completion_status": "ACTUAL_FULL_HESSIAN_AND_ALL_SIGNED_MODES_AUDITED",
        "atoms": n, "normal_modes": count, "normal_mode_count_complete": True, "all_frequencies_positive": positive,
        "imaginary_frequency_count": int(np.sum(cm < 0)), "zero_frequency_count": int(np.sum(cm == 0)),
        "qualified_for_mechanism": qualified, "minimum_qualification_separate_from_completion": True,
        "mechanism_pairing_status": "ALL_PAIRS_DESCRIPTIVE_ONLY" if qualified else "BLOCKED_CONTROL_OR_NONPOSITIVE_MINIMUM",
        "experimental_fold_records": 20 if qualified else "NOT_READ", "independent_chemical_peak_count": "NOT_ESTABLISHED",
        "low_frequency_mode_count": len(low), "cartesian_comparison_count": len(pairs) if pairs is not None else "NOT_GENERATED",
        "independent_numpy_replay": numerical, "global_capture": gate["kernel"],
        "actual_old_stock_control_comparison": control_comparison,
        "fresh_gradient_max_difference_Eh_per_Bohr": delta_g,
        "source_gradient_atom_RMS_Eh_per_Bohr": float(np.sqrt(np.mean(lengths**2))),
        "source_gradient_atom_max_Eh_per_Bohr": float(lengths.max()), "energy_hartree": manifest["energy_hartree"],
        "electronic_method": {key: binding["method_settings"][key] for key in
            ("xc", "basis", "grid_level", "dispersion", "density_fitting", "direct_scf_tol")},
        "IR_intensities_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
        "control_chemical_identity": "EXACT_TRUSTED_OLD_SNAPSHOT_AND_INPUT_RECIPE" if control else "NOT_A_SOFTWARE_CONTROL",
        "theory_or_peaks_enter_classifier": False, "isolated_molecule_is_not_solid_herbal_spectrum": True,
        "one_conformer_is_not_a_global_minimum_search": True, "new_quantum_evaluations": 0,
        "normal_modes_and_CSV_without_old_producer_hash": "CURRENT_HASH_PLUS_ANCHORED_HESSIAN_JSON_REPLAY_NO_OLD_HASH_INVENTED",
        "animations": animations, "animation_policy": "DISPLAY_ONLY_NOT_MD; arbitrary0.15Angstrom; negative diagnostic modes retained",
        "group_participation_adapter": "NOT_RUN_NOT_IMPLEMENTED_FOR_V6b; original V5 adapter unchanged",
        "input_hashes": hashes, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": sys.modules["matplotlib"].__version__}}
    write_json(output / "checks.json", checks)
    (output / "README.md").write_text(
        f"{case}: {count} actual signed modes independently audited from a genuine closed V6b A/B/global chain. "
        f"All-positive minimum: {positive}; mechanism qualification: {qualified}. "
        "IR NOT_COMPUTED. Ticks have equal display height, not absorption intensity; mode IR activity is unknown. "
        "XYZ frames have arbitrary display amplitude and are not MD; negative modes are instability diagnostics. "
        "Only a positive Spinosin minimum is compared descriptively to all20 overlapping-fold candidate records; "
        "these are not20 independent chemical peaks and frequency proximity is not chemical identification. "
        "Isolated-molecule harmonic modes omit solid-state lattice and intermolecular effects. "
        "No theory modifies the classifier or its historical96.0234% result. Control results are not Spinosin.\n")
    seal(output)
    return checks


def preflight(output):
    output = Path(output).resolve(); require(not output.exists(), "Preserve earlier preparation")
    refs = []
    for path, expected in ((WORKER, WORKER_SHA), (GUARD, GUARD_SHA), (STATE, STATE_SHA), (V5, V5_SHA), (A2, A2_SHA)):
        require(sha(path) == expected, "Frozen small source dependency differs")
        refs.append({"path": str(path), "sha256": expected, "bytes": path.stat().st_size})
    v5_metadata_validators()  # compile pure metadata definitions only, never NumPy
    output.mkdir(parents=True)
    result = {"status": "NOT_RUN", "analysis_schema": "research4-V6b-independent-staged-modes-analysis-v1",
              "source_dependencies": refs, "target_arrays_read": False, "numerical_packages_imported": False,
              "controls_numerical_validation": "NOT_RUN_AFTER_Q_REQUIRED", "Spinosin_frequencies": "NOT_RUN",
              "IR_intensities_status": "NOT_COMPUTED", "old_a2_V6b_gate_unchanged": True,
              "actual_budget_registered": False, "quantum_calculations": 0}
    write_json(output / "preflight.json", result); seal(output)
    return result


@contextmanager
def exclusive_closed_quantum():
    lock = ROOT / ".runtime/pyscf-v4-global.lock"
    require(lock.is_file(), "Use existing real Q global lock, never create another")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def analyze(result_path, expected_sha, output, *, control=False):
    """Public API: never numerically inspect arrays while the real Q lock is held."""
    with exclusive_closed_quantum():
        return _analyze_closed(result_path, expected_sha, output, control=control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path); parser.add_argument("--result-sha256")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--control-validation", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cpu-seconds", type=int, default=300); parser.add_argument("--wall-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.preflight:
        require(args.result is None and not args.control_validation, "Preflight never consumes actual results")
        result = preflight(args.output)
    else:
        require(args.result is not None and args.result_sha256 and 1 <= args.cpu_seconds <= 600
                and 1 <= args.wall_seconds <= 1800, "Explicit result/SHA and bounded postprocessing limits required")
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds + 5))
        def timeout(signum, frame):
            raise TimeoutError("Finite postprocessing resource limit")
        signal.signal(signal.SIGALRM, timeout); signal.alarm(args.wall_seconds)
        signal.signal(signal.SIGXCPU, timeout)
        result = analyze(args.result, args.result_sha256, args.output, control=args.control_validation)
        signal.alarm(0)
    print(json.dumps({key: result.get(key) for key in ("status", "case", "normal_modes", "qualified_for_mechanism")}))


if __name__ == "__main__":
    main()
