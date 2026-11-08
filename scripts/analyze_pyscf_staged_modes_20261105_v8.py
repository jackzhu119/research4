"""V8 independent modes: genuine legacy V5/V7 inputs and new V8 producers.

The V7 analyzer is immutable. Numerical formulas are copied unchanged, allowing
version labels only. Legacy A/H1 retain their actual original sources, schemas
and guardian proofs; new response/global producers require the new frozen V8
guard. No SCF, gradients or Hessians are run by this analyzer. Unfrozen source
pins mean NOT_RUN, never successful validation.
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
import time
import types

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/pyscf_staged_consumer_20261105_v8.py"
WORKER_SHA = "3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585"
LEGACY_WORKER = ROOT / "scripts/pyscf_staged_consumer_20261105_v7.py"
LEGACY_WORKER_SHA = "6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe"
V7_ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v7_v2.py"
V7_ANALYZER_SHA = "76e2eddebf681f9adf2ab7c5f9fa0bdc126b0c93f081cc93a606009f31c94705"
ORIGINAL_NUMERICAL_SOURCE = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v6b_v2.py"
ORIGINAL_NUMERICAL_SOURCE_SHA = "38dde5e185c4fc48d58732982e0164b3a2f5430a26bdfc1076b21103d1d75c6a"
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
LEGACY_GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
LEGACY_GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
STATE = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
V5 = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
V5_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
A2 = ROOT / "scripts/analyze_spinosin_v5_modes_20261104.py"
A2_SHA = "a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154"
SCHEMA = "research4-staged-full-rks-hessian-result-v8"
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
    translate old schemas or permit V7 producers through V5 validators.
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
    require(result.get("schema") == SCHEMA and result.get("stage") == result.get("phase") == "global-assemble"
            and result.get("status") == "FREQUENCIES_COMPUTED" and result.get("frequencies_status") == "DONE",
            "Only an actual complete V8 global result is eligible")
    for key in ("no_new_SCF_performed", "stable_MF_state_unchanged", "complete_RHF_helper", "complete_RKS_partial",
                "complete_H1AO", "complete_global_Hessian", "normal_modes_complete", "final_gradient_thresholds_passed",
                "full_system_CPHF_returned_normally", "harmonic_analysis_excludes_translations_and_rotations"):
        require(result.get(key) is True, "Incomplete global evidence: " + key)
    n = result.get("atoms")
    require(type(n) is int and n in ((3, 9) if control else (75,)), "Control/default atom count differs")
    require(result.get("normal_mode_count") == result.get("expected_normal_mode_count") == 3 * n - 6,
            "Incomplete nonlinear normal-mode count")
    require(result.get("full_system_CPHF_executed_in_this_phase") is False
            and result.get("full_system_CPHF_completion_basis") == "ALL_HEALTHY_FULL_AO_MO_B8_RESPONSE_PRODUCERS",
            "V8 final assembly must not claim a new CPHF solve")
    require(result.get("source_MF_stability_status") == "VERIFIED_UNCHANGED"
            and result.get("IR_intensities_status") == "NOT_COMPUTED"
            and result.get("measured_CPHF_residual_norm") == "NOT_COMPUTED",
            "Do not infer measured CPHF residuals or IR intensities")
    require(result.get("full_global_contribution_calls") == {"partial": 1, "h1": 0, "response_solver": 0, "nuclear": 1},
            "Global contributions missing or double-added")
    require(result.get("V5_analyzer_compatibility") == "BLOCKED_V8_PRODUCER_REQUIRES_REVIEWED_V8_POSTPROCESSOR",
            "Never relabel V8 as legacy compatibility")
    return n


def command_option(command, name):
    require(command.count(name) == 1 and command.index(name) + 1 < len(command), "Actual closed command lacks unique " + name)
    return command[command.index(name) + 1]


def stage_command_gate(command, args, result, output, stage, control, source_args):
    """Bind new B/response/assembly declarations to actual closed parent argv."""
    require(stage in ("h1ao", "response-block", "global-assemble"), "V7 A has its separate decomposition argv gate")
    require(command.count("--control-validation") == int(control) and args["control_validation"] is control,
            "Control/default flag differs from actual producer command")
    require(command.count("--execute") == 1 and args["execute"] is True
            and args["stage"] == command_option(command, "--stage") == stage
            and root_path(command_option(command, "--output")) == root_path(output)
            and int(command_option(command, "--threads")) == args["threads"] == result["settings"]["threads"] == 2,
            "Actual V7 stage/output/selected-two-thread arguments differ")
    keys = [("--snapshot", "snapshot"), ("--snapshot-manifest-sha256", "snapshot_manifest_sha256"),
            ("--gradient-check-result", "gradient_check_result"),
            ("--gradient-check-result-sha256", "gradient_check_result_sha256"),
            ("--partial-result", "partial_result"), ("--partial-result-sha256", "partial_result_sha256")]
    if stage in ("response-block", "global-assemble"):
        keys += [("--h1-result", "h1_result"), ("--h1-result-sha256", "h1_result_sha256")]
    if stage == "response-block":
        require(int(command_option(command, "--block-index")) == args["block_index"] == result["block_index"],
                "Actual response block index differs")
    if stage == "global-assemble":
        keys += [("--response-index", "response_index"), ("--response-index-sha256", "response_index_sha256")]
    if not control:
        keys += [("--control-release", "control_release"), ("--control-release-sha256", "control_release_sha256")]
    for flag, key in keys:
        observed = command_option(command, flag)
        require((root_path(observed) == root_path(args[key])) if not key.endswith("sha256") else observed == args[key],
                "Input.json differs from actual V7 closed command: " + key)
    shared = ("snapshot", "snapshot_manifest_sha256", "gradient_check_result", "gradient_check_result_sha256",
              "threads", "partial_result", "partial_result_sha256")
    if stage in ("response-block", "global-assemble"):
        shared += ("h1_result", "h1_result_sha256")
    if not control:
        shared += ("control_release", "control_release_sha256")
    path_keys = {"snapshot", "gradient_check_result", "partial_result", "h1_result", "control_release"}
    for key in shared:
        require((root_path(args[key]) == root_path(source_args[key])) if key in path_keys else args[key] == source_args[key],
                "Actual V7 stages changed shared source/A/B/release: " + key)
    require(args["snapshot_manifest_sha256"] == result["snapshot_manifest_sha256"], "Stage snapshot argument differs")


def h1_command_gate(command, args, result, output, control, global_args):
    """Keep old H1 release identity while requiring identical physical inputs."""
    legacy = result.get("schema") == "research4-staged-full-rks-hessian-result-v7"
    require(legacy or result.get("schema") == SCHEMA, "Unknown H1 producer schema")
    stage_command_gate(command, args, result, output, "h1ao", control, args if legacy else global_args)
    if legacy:
        shared = ("snapshot", "snapshot_manifest_sha256", "gradient_check_result",
                  "gradient_check_result_sha256", "threads", "partial_result", "partial_result_sha256")
        path_keys = {"snapshot", "gradient_check_result", "partial_result"}
        require(all((root_path(args[k]) == root_path(global_args[k])) if k in path_keys else args[k] == global_args[k]
                    for k in shared), "Legacy H1 and new global changed original physical inputs")


def guardian_for_closed_producer(result, guard):
    """Select a genuine era, never submit a V6b ledger to V8 validation."""
    proof = result["resource_supervision"]["proof"]
    identity = (proof.get("schema"), root_path(proof.get("supervisor_script_path", "")),
                proof.get("supervisor_script_sha256"))
    if identity == ("research4-pyscf-continuation-supervisor-proof-v6b", LEGACY_GUARD, LEGACY_GUARD_SHA):
        require(result.get("schema") == "research4-staged-full-rks-hessian-result-v7"
                and result.get("stage") == "h1ao", "Only authentic legacy H1 may use the old guardian here")
        return frozen_module(LEGACY_GUARD, LEGACY_GUARD_SHA, "genuine_legacy_v6b_closed_parent")
    require(identity == ("research4-pyscf-continuation-supervisor-proof-v8", GUARD, GUARD_SHA),
            "Unknown or mixed closed guardian era/source")
    require(result.get("schema") == SCHEMA and result.get("stage") in ("h1ao", "response-block", "global-assemble"),
            "New response/global declarations require the genuine V8 guardian")
    require(getattr(guard, "PROOF_SCHEMA", None) == identity[0], "New guardian schema differs")
    return guard


def closed_command(result, hashes, guard):
    proof = result["resource_supervision"]["proof"]
    ledger_path = Path(proof["ledger_path"])
    ledger = read_json(ledger_path, hashes)
    actual_guard = guardian_for_closed_producer(result, guard)
    actual_guard.validate_ledger(ledger, allow_running=False)
    rows = [r for r in ledger["phases"] if r.get("phase") == proof["phase"]]
    require(len(rows) == 1 and actual_guard.healthy_closed_row(rows[0]), "Actual producing phase is not healthily closed")
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


def stage_capture_gate(capture, result, partial_ref, h1_ref, response_index_ref, response_refs, response_closures, n):
    """V8 assembly uses explicit legacy H1/new MO1/MOE1 and no new solve."""
    expected = {"schema": "research4-v8-full-global-assembly-capture-v1",
        "status": "DONE_COMPLETE_GLOBAL_ASSEMBLY",
        "contribution_calls": {"partial": 1, "h1": 0, "response_solver": 0, "nuclear": 1},
        "global_atom_ids": list(range(n)), "partial_result_dependency": partial_ref,
        "h1_result_dependency": h1_ref, "response_index_dependency": response_index_ref,
        "explicit_full_H1AO_and_MO1_MOE1_supplied": True,
        "stock_hess_elec_returned_normally": True, "stock_nuclear_returned_normally": True,
        "global_cross_atom_contractions_preserved": True, "nuclear_added_once": True,
        "IR_intensities_status": "NOT_COMPUTED", "measured_CPHF_residual_norm": "NOT_COMPUTED",
        "H1AO_source_referenced_without_duplicate_capture": True, "CPHF_executed_in_this_phase": False,
        "all_response_blocks_healthily_closed": True, "response_result_dependencies": response_refs}
    require(all(capture.get(k) == v for k, v in expected.items()), "Actual V7 global assembly metadata differs")
    require(result["full_global_contribution_calls"] == expected["contribution_calls"], "V7 global call counts differ")
    require(capture.get("physical_binding_sha256") == result["physical_binding_sha256"]
            and capture.get("source_worker", {}).get("sha256") == WORKER_SHA
            and root_path(capture["source_worker"]["path"]) == WORKER,
            "Actual V7 assembly source/physical binding differs")
    chunks = [list(range(i, min(i + 8, n))) for i in range(0, n, 8)]
    blocks = capture.get("response_blocks")
    require(isinstance(blocks, list) and len(blocks) == len(chunks) == len(response_refs) == len(response_closures),
            "V7 global response block count differs")
    for i, (block, ids, ref, closure) in enumerate(zip(blocks, chunks, response_refs, response_closures)):
        require(block.get("block_index") == i and block.get("global_atom_ids") == ids
                and block.get("result") == ref and block.get("closure") == closure,
                "Full global B8 source coverage/order/closure differs")
    return {"status": "FULL_V8_EXPLICIT_GLOBAL_ASSEMBLY_AND_HEALTHY_B8_METADATA_PASS",
        "CPHF_residual_measured": False, "response_blocks": len(blocks),
        "full_global_atom_ids": list(range(n)), "cached_H1_duplicated_by_V8_save_only": False,
        "response_solver_executed_in_final_assembly": False}


def control_reference_gate(source, hashes):
    """Authenticate a genuine old V5 STOCK result, never relabel it as V7."""
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


def numerical_formula_source_gate(candidate_source=None):
    """Anchor the V7 numerical formulas; allow only three helper label edits."""
    require(sha(ORIGINAL_NUMERICAL_SOURCE) == ORIGINAL_NUMERICAL_SOURCE_SHA, "Original numerical source differs")
    require(sha(V7_ANALYZER) == V7_ANALYZER_SHA, "Frozen V7 independent analyzer differs")
    target = Path(__file__) if candidate_source is None else Path(candidate_source)
    old_text, new_text = V7_ANALYZER.read_text(), target.read_text()
    old_functions = {n.name: n for n in ast.parse(old_text).body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in ast.parse(new_text).body if isinstance(n, ast.FunctionDef)}
    names = ("eigenvalue_clusters", "numpy_runtime", "array", "independent_subspace_audit",
             "validate_minimum_flags", "compare_actual_control", "mechanism_qualified",
             "v5_metadata_validators", "source_gate", "control_reference_gate")
    for name in names:
        require(name in old_functions and name in new_functions, "Missing independent numerical/physical gate")
        text = ast.get_source_segment(old_text, old_functions[name])
        if name in ("numpy_runtime", "independent_subspace_audit", "compare_actual_control"):
            text = text.replace("V7", "V8").replace("v7", "v8")
        expected = ast.parse(text).body[0]
        require(ast.dump(expected, include_attributes=False) == ast.dump(new_functions[name], include_attributes=False),
                "Unreviewed numerical formula/source gate changed: " + name)
    # All arithmetic/control-flow between the gate and export metadata remains
    # byte-for-byte equivalent in AST, including signed modes and plots.
    def numerical_prefix(function):
        nodes = []
        for node in function.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "checks" for t in node.targets):
                break
            nodes.append(node)
        return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    require(numerical_prefix(old_functions["_analyze_closed"]) == numerical_prefix(new_functions["_analyze_closed"]),
            "Unreviewed numerical assembly/export prefix changed")
    return {"status": "UNCHANGED_FROZEN_V7_NUMERICAL_FORMULAS_HELPER_LABELS_ONLY",
        "original_source_sha256": ORIGINAL_NUMERICAL_SOURCE_SHA,
        "frozen_V7_analyzer_sha256": V7_ANALYZER_SHA, "functions": list(names),
        "full_numerical_analysis_prefix_AST_unchanged": True,
        "new_candidate_source_sha256": sha(target), "numerical_imports_or_execution": False}


def expected_native_sources(worker):
    """Read only hash-pinned literal source dictionaries; never import PySCF."""
    xc = frozen_metadata_subset(worker.XC_CORE, worker.XC_CORE_SHA, (), ("PINNED",)).PINNED
    rhf = frozen_metadata_subset(worker.CORE, worker.CORE_SHA, (), ("PINNED",)).PINNED
    expected = {**xc, "hessian/rhf.py": rhf["hessian/rhf.py"], "scf/_vhf.py": rhf["scf/_vhf.py"],
        "lib/linalg_helper.py": worker.LINALG_HELPER_SHA, "scf/cphf.py": worker.CPHF_SOURCE_SHA,
        **worker.RESPONSE_MODULE_SHAS}
    require(len(expected) == 12, "Complete reviewed native source set required")
    return expected


def validate_actual_native_receipt(value, worker, hashes):
    expected = expected_native_sources(worker)
    records = value.get("actual_imported_native_sources")
    require(isinstance(records, dict) and set(records) == set(expected), "Incomplete actual imported native module receipt")
    native_root = ROOT / ".runtime/quantum-upgrade-site/pyscf"
    for name, expected_sha in expected.items():
        record = records[name]
        require(isinstance(record, dict) and root_path(record["path"]) == native_root / name
                and record.get("sha256") == expected_sha, "Actual native module path/source differs: " + name)
        checked(native_root / name, hashes, expected_sha, maximum_bytes=2*1024**2)
    return {"status": "ACTUAL_EXECUTED_MODULE_RECEIPT_AND_PINNED_NATIVE_SOURCES_VERIFIED",
        "module_count": len(expected), "sources": records, "quantum_engine_imported_by_this_check": False}


def closed_shell_occupied_space(molecule_fingerprint):
    electrons = molecule_fingerprint.get("nelectron")
    require(type(electrons) is int and electrons > 0 and electrons % 2 == 0
            and molecule_fingerprint.get("charge") == molecule_fingerprint.get("spin") == 0,
            "Exact neutral singlet closed-shell electron count required")
    return electrons // 2


def validate_extra_chain(chain, hashes):
    """Require the actually verified V7 decomposition, not a compatibility flag."""
    require(chain.get("schema") == "research4-v7-complete-xc-partial-lineage-v1",
            "Actual V7 partial source-chain contract required")
    for key in ("all_chunk_producers_healthy", "all_required_grid_blocks_covered_exactly_once",
                "RHF_added_once", "XCdiag_once", "XCderiv2_once",
                "actual_partial_command_verified", "actual_component_commands_verified"):
        require(chain.get(key) is True, "Incomplete V7 source chain: " + key)
    require(chain.get("nuclear_or_CPHF_included") is False, "V7 A must not include nuclear/response terms")
    records = chain.get("verified_artifacts")
    require(isinstance(records, list) and records, "Complete V7 chain artifact inventory required")
    seen = set()
    for record in records:
        require(isinstance(record, dict) and set(("path", "sha256", "bytes")) <= set(record)
                and type(record["bytes"]) is int and record["bytes"] >= 0,
                "Malformed verified V7 chain artifact")
        path = root_path(record["path"])
        require(path not in seen and path.is_relative_to(ROOT), "Duplicate/outside V7 chain artifact")
        seen.add(path)
        require(path.stat().st_size == record["bytes"], "V7 chain artifact byte size changed")
        checked(path, hashes, record["sha256"])
    require(isinstance(chain.get("RHF_producers"), list) and chain["RHF_producers"],
            "Healthy full-system RHF producer chain required")
    row_refs = chain.get("closed_ledger_rows")
    require(isinstance(row_refs, list) and row_refs, "Actual closed component ledger rows required")
    _, v5_pure = v5_metadata_validators()
    for record in row_refs:
        require(set(("path", "phase", "closed_row_sha256")) <= set(record), "Malformed closed component row reference")
        ledger = read_json(root_path(record["path"]), hashes)
        rows = [row for row in ledger.get("phases", []) if row.get("phase") == record["phase"]]
        require(len(rows) == 1 and v5_pure.healthy_closed_row(rows[0]), "V7 component source row is not healthy")
        observed = {"json_sort_keys_default": digest(rows[0], compact=False),
                    "json_sort_keys_compact": digest(rows[0], compact=True)}
        matching = [style for style, value in observed.items() if value == record["closed_row_sha256"]]
        require(matching and ("hash_style" not in record or record["hash_style"] in matching),
                "Actual closed V7 component row hash differs")
    return chain


def qualify(result_path, expected_sha, *, control=False, hashes=None):
    """Mixed-era source/parent/A/H1/response gates precede NumPy and NPY."""
    hashes = {} if hashes is None else hashes
    require(len(WORKER_SHA) == 64 and all(c in "0123456789abcdef" for c in WORKER_SHA),
            "V8 consumer source SHA is not frozen; numerical execution NOT_RUN")
    require(len(GUARD_SHA) == 64 and all(c in "0123456789abcdef" for c in GUARD_SHA),
            "V8 guardian source SHA is not frozen; numerical execution NOT_RUN")
    formula_gate = numerical_formula_source_gate()
    checked(ORIGINAL_NUMERICAL_SOURCE, hashes, ORIGINAL_NUMERICAL_SOURCE_SHA, maximum_bytes=128*1024)
    path = Path(result_path).resolve()
    require(path.is_relative_to(ROOT / "reports") and path.name == "result.json", "Explicit report result required")
    result = read_json(path, hashes, expected_sha); n = global_header(result, control=control)
    worker = frozen_module(WORKER, WORKER_SHA, "frozen_v8_staged_metadata")
    guard = frozen_module(GUARD, GUARD_SHA, "frozen_v8_actual_parent")
    require(worker.SCHEMA == SCHEMA and worker.COMPATIBILITY == result["V5_analyzer_compatibility"],
            "V8 consumer schema/compatibility contract differs")
    evidence = {"source_worker": result["source_worker"], "resource_supervision": result["resource_supervision"],
        "output_directory": str(path.parent), "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "stage": "global-assemble"}
    closure = guard.verify_closed_phase(evidence, expected_worker_path=WORKER, expected_worker_sha256=WORKER_SHA,
        expected_result_schema=SCHEMA, allowed_result_statuses=("FREQUENCIES_COMPUTED",),
        expected_output=path.parent, expected_snapshot_sha256=result["snapshot_manifest_sha256"])
    require(closure["producer_result_sha256"] == expected_sha, "Global closed result SHA differs")
    native_receipts = [validate_actual_native_receipt(result, worker, hashes)]
    command = closed_command(result, hashes, guard)
    args = read_json(path.parent / "input.json", hashes, result["input_json_sha256"])["arguments"]
    stage_command_gate(command, args, result, path.parent, "global-assemble", control, args)
    source = source_gate(result, args, control, hashes)
    nocc = closed_shell_occupied_space(source["snapshot_manifest"]["molecule_fingerprint"])
    require(result.get("occupied_orbitals") == nocc, "Global occupied space differs from original molecule electron count")
    partial_ref = {"path": str(root_path(args["partial_result"])), "sha256": args["partial_result_sha256"]}
    h1_ref = {"path": str(root_path(args["h1_result"])), "sha256": args["h1_result_sha256"]}
    a = read_json(partial_ref["path"], hashes, partial_ref["sha256"])
    b = read_json(h1_ref["path"], hashes, h1_ref["sha256"])
    # A has a distinct V7 decomposition/assembly argv and a distinct real
    # source. Only the new V7 qualifier verifies that entire actual chain.
    # It is never submitted to old V6b rks-partial command or producer gates.
    manifest = read_json(Path(partial_ref["path"]).parent / "stage-cache/manifest.json", hashes, a["cache_manifest_sha256"])
    physical = manifest["physical_binding"]
    require(physical["stable_snapshot_binding"] == source["binding"]
            and physical["snapshot_manifest_sha256"] == result["snapshot_manifest_sha256"]
            and physical["numerical_policy"] == worker.NUMERICAL_POLICY
            and physical["schema"] == worker.PHYSICAL_SCHEMA
            and digest(physical) == result["physical_binding_sha256"], "Global/A original physical state differs")
    a_value, a_manifest, a_dir, a_closure, extra_chain = worker.qualify_partial(
        partial_ref, physical, n, result["basis_functions"], guard)
    validate_extra_chain(extra_chain, hashes)
    b_value, b_manifest, b_dir, b_closure = worker.qualify_stage(
        h1_ref, "h1ao", physical, n, result["basis_functions"], guard)
    require(a_value == a and b_value == b, "V7 A/B changed during independent metadata gates")
    native_receipts.append(validate_actual_native_receipt(b, worker, hashes))
    for value in (a, b):
        require(value["source_state_binding"] == result["source_state_binding"]
                and value["gradient_gate"] == result["gradient_gate"] and value["boot_id"] == result["boot_id"],
                "Stages changed original source/G/boot")
    b_args = read_json(b_dir.parent / "input.json", hashes, b["input_json_sha256"])["arguments"]
    # The legacy H1 genuinely used the old control release, not the newly
    # executed V8 release. Authenticate its own argv and separately require
    # unchanged molecular source/G/A/thread inputs across eras.
    h1_command_gate(closed_command(b, hashes, guard), b_args, b, b_dir.parent, control, args)
    require(b_value["partial_result_dependency"] == partial_ref
            and result["partial_stage_producer"] == a_closure and result["H1_stage_producer"] == b_closure,
            "Global and H1 do not share actual healthy A/B producers")
    read_json(b_dir / "manifest.json", hashes, b_value["cache_manifest_sha256"])
    for directory, cache in ((a_dir, a_manifest), (b_dir, b_manifest)):
        for row in cache["arrays"]:
            path_key = str((directory / row["file"]).resolve())
            hashes[path_key] = {"sha256": row["file_sha256"], "bytes": Path(path_key).stat().st_size,
                "producer_hash_present": True, "verified_by": "FROZEN_V7_ACTUAL_PRODUCER_QUALIFICATION"}
    response_index_ref = {"path": str(root_path(args["response_index"])), "sha256": args["response_index_sha256"]}
    responses, response_index, checked_index = worker.qualify_response_index(response_index_ref, physical, n,
        result["basis_functions"], guard, partial_ref, h1_ref, nocc=nocc)
    require(checked_index == response_index_ref and response_index.get("schema") == "research4-v8-closed-response-results-index-v1",
            "Actual V8 response index differs")
    read_json(response_index_ref["path"], hashes, response_index_ref["sha256"])
    response_refs = response_index["results"]
    require(result.get("response_result_dependencies") == response_refs
            and len(response_refs) == len(responses) == (n + 7)//8,
            "V7 assembly response dependency count/order differs")
    response_closures = []
    for index, (ref, (value, cache, directory, response_closure)) in enumerate(zip(response_refs, responses)):
        require(value.get("block_index") == index and value.get("global_atom_ids") == list(range(index*8, min((index+1)*8, n)))
                and value["source_state_binding"] == result["source_state_binding"]
                and value["gradient_gate"] == result["gradient_gate"] and value["boot_id"] == result["boot_id"]
                and value["occupied_orbitals"] == nocc,
                "Actual V7 response block changed atom coverage/source/G/boot/occupied space")
        native_receipts.append(validate_actual_native_receipt(value, worker, hashes))
        ref_path = root_path(ref["path"])
        read_json(ref_path, hashes, ref["sha256"])
        response_args = read_json(ref_path.parent / "input.json", hashes, value["input_json_sha256"])["arguments"]
        stage_command_gate(closed_command(value, hashes, guard), response_args, value, ref_path.parent,
            "response-block", control, args)
        read_json(directory / "manifest.json", hashes, value["cache_manifest_sha256"])
        for row in cache["arrays"]:
            payload = directory / row["file"]
            hashes[str(payload.resolve())] = {"sha256": row["file_sha256"], "bytes": payload.stat().st_size,
                "producer_hash_present": True, "verified_by": "FROZEN_V8_HEALTHY_RESPONSE_BLOCK_QUALIFICATION"}
        response_closures.append(response_closure)
    capture = read_json(path.parent / "global_assembly_capture.json", hashes, result["global_assembly_capture_sha256"])
    kernel = stage_capture_gate(capture, result, partial_ref, h1_ref, response_index_ref,
        response_refs, response_closures, n)
    control_reference = control_reference_gate(source, hashes) if control else None
    return {"result": result, "source": source, "closure": closure, "partial_ref": partial_ref, "h1_ref": h1_ref,
        "partial_closure": a_closure, "H1_closure": b_closure, "RHF_producers": extra_chain["RHF_producers"],
        "response_closures": response_closures, "response_index": response_index_ref,
        "actual_imported_native_source_receipts": native_receipts,
        "V7_complete_partial_source_chain": extra_chain, "numerical_formula_source_gate": formula_gate,
        "physical_binding": physical, "kernel": kernel,
        "control_reference": control_reference, "result_path": str(path), "hashes": hashes,
        "metadata_gate_passed_before_numerical_import": True}


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
        a2 = frozen_module(A2, A2_SHA, "frozen_a2_numerical_helpers_only_not_V8_gate")
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
    replay = a2.load_frozen(a2.REPLAY, a2.REPLAY_SHA, "V8_independent_numpy_TR_projector")
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
    require(np.array_equal(old_coords, coords_bohr), "Old stock and V8 use different source coordinates")
    with checked(location / "frequencies.csv", hashes).open(newline="") as handle:
        a2.validate_frequency_rows(list(csv.DictReader(handle)), old_cm, old_thz)
    old_audit = independent_subspace_audit(old_raw, old_used, old_modes, old_cm, old_coords, symbols, np, a2)
    raw_error, used_error = float(np.max(np.abs(raw - old_raw))), float(np.max(np.abs(used - old_used)))
    require(raw_error <= 1e-8 and used_error <= 1e-8 and np.allclose(cm, old_cm, atol=1e-5, rtol=1e-9),
            "Real V8 control differs from exact-source old stock full matrix/all signed modes")
    replay = a2.load_frozen(a2.REPLAY, a2.REPLAY_SHA, "V8_control_saved_mode_comparison")
    eigenvalues = np.sign(old_cm) * (np.abs(old_cm) / replay.FACTOR)**2
    mass = np.asarray([MASS[s] for s in symbols], dtype=float)
    left = (modes * np.sqrt(mass)[None, :, None]).reshape(count, 3 * n).T
    right = (old_modes * np.sqrt(mass)[None, :, None]).reshape(count, 3 * n).T
    subspace_errors = []
    for indices in eigenvalue_clusters(eigenvalues.tolist()):
        a, b = left[:, indices], right[:, indices]
        error = float(np.max(np.abs(a @ a.T - b @ b.T)))
        require(error <= 1e-5, "Actual V8/old-stock control mode subspace differs")
        subspace_errors.append({"modes_one_based": [i + 1 for i in indices], "projector_max_error": error})
    return {"status": "PASS_ACTUAL_V8_VS_AUTHENTICATED_V5_STOCK_FULL_MATRIX_AND_ALL_MODES",
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
               "V8_worker_sha256": WORKER_SHA, "legacy_V7_worker_sha256": LEGACY_WORKER_SHA, "original_numerical_source_sha256": ORIGINAL_NUMERICAL_SOURCE_SHA, "files": files})


def _analyze_closed(result_path, expected_sha, output, *, control=False):
    started_at = datetime.now(timezone.utc).isoformat()
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    output = Path(output).resolve(); require(not output.exists(), "Preserve earlier analysis; use fresh output")
    gate = qualify(result_path, expected_sha, control=control)
    np, a2 = numpy_runtime(gate)
    result, source, hashes = gate["result"], gate["source"], gate["hashes"]
    # The original, unmodified V5 payload verifier is used only AFTER the new
    # pure V7+A+B+source/G/chunk gate. No MF is restored and no PySCF engine imported.
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
    nocc = closed_shell_occupied_space(manifest["molecule_fingerprint"])
    occupations = source_arrays["MO_occ"]
    require(occupations.dtype == np.float64 and occupations.shape == (result["basis_functions"],)
            and np.count_nonzero(occupations > 0) == nocc == result["occupied_orbitals"]
            and np.array_equal(occupations, np.where(occupations > 0, 2., 0.))
            and float(occupations.sum()) == manifest["molecule_fingerprint"]["nelectron"]
            and source_arrays["MO_coeff"].shape == (result["basis_functions"], result["basis_functions"]),
            "Exact full-square MO/neutral closed-shell occupation payload differs")
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
    checks = {"schema": "research4-V8-independent-staged-modes-analysis-v1",
        "status": "DONE_CONTROL_NUMERICAL_REPLAY" if control else "DONE_COMPUTED_MODES_AUDITED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "started_at_utc": started_at,
        "process_cpu_seconds": time.process_time()-started_cpu, "wall_seconds": time.perf_counter()-started_wall,
        "case": case,
        "analysis_source_sha256": sha(__file__), "actual_V8_worker_sha256": WORKER_SHA, "legacy_V7_worker_sha256": LEGACY_WORKER_SHA, "actual_parent_sha256": GUARD_SHA,
        "actual_parent_schema": "research4-pyscf-continuation-supervisor-proof-v8",
        "legacy_producers_are_not_relabelled_as_v8": True,
        "a2_reused_for_numerical_helpers_only": A2_SHA, "old_a2_V8_compatibility_claimed": False,
        "original_numerical_source_sha256": ORIGINAL_NUMERICAL_SOURCE_SHA,
        "copied_numerical_formulas_match_frozen_V7_with_label_changes_only": True,
        "numerical_formula_source_gate": gate["numerical_formula_source_gate"],
        "old_V7_producer_gate_used_for_V8": False,
        "source_result_sha256": expected_sha, "producer_closure": gate["closure"],
        "partial_producer": gate["partial_closure"], "H1_producer": gate["H1_closure"],
        "healthy_full_B8_response_producers": gate["response_closures"], "response_index": gate["response_index"],
        "healthy_RHF_unit_producers": gate["RHF_producers"],
        "V7_complete_partial_source_chain": gate["V7_complete_partial_source_chain"],
        "source_SCF_closed_producer": source["source_closure"], "restored_gradient_gate": source["gradient_gate"],
        "snapshot_manifest_sha256": result["snapshot_manifest_sha256"], "physical_binding_sha256": result["physical_binding_sha256"],
        "metadata_provenance_gates_before_numpy_or_array_access": True,
        "actual_imported_native_source_receipts": gate["actual_imported_native_source_receipts"],
        "global_completion_status": "ACTUAL_FULL_HESSIAN_AND_ALL_SIGNED_MODES_AUDITED",
        "atoms": n, "occupied_orbitals_independently_verified": nocc,
        "full_square_MO_space_independently_verified": True,
        "normal_modes": count, "normal_mode_count_complete": True, "all_frequencies_positive": positive,
        "imaginary_frequency_count": int(np.sum(cm < 0)), "zero_frequency_count": int(np.sum(cm == 0)),
        "qualified_for_mechanism": qualified, "minimum_qualification_separate_from_completion": True,
        "MF_stability_flag_scope": "IMMUTABILITY_OF_RESTORED_MF_NOT_ORBITAL_STABILITY_TEST",
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
        "group_participation_adapter": "NOT_RUN_NOT_IMPLEMENTED_FOR_V8; original V5 adapter unchanged",
        "input_hashes": hashes, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": sys.modules["matplotlib"].__version__}}
    write_json(output / "checks.json", checks)
    (output / "README.md").write_text(
        f"{case}: {count} actual signed modes independently audited from a genuine mixed legacy-V7 A/H1 and new-V8 full-B8/assembly chain. "
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
    output = Path(output).resolve(); require(not output.exists(), "Preserve earlier V8 preparation")
    refs = []
    for path, expected in ((LEGACY_GUARD, LEGACY_GUARD_SHA), (LEGACY_WORKER, LEGACY_WORKER_SHA),
                           (V7_ANALYZER, V7_ANALYZER_SHA), (STATE, STATE_SHA), (V5, V5_SHA), (A2, A2_SHA),
                           (ORIGINAL_NUMERICAL_SOURCE, ORIGINAL_NUMERICAL_SOURCE_SHA)):
        require(sha(path) == expected, "Frozen source dependency differs")
        refs.append({"path": str(path), "sha256": expected, "bytes": path.stat().st_size})
    new_frozen = {}
    for label, path, expected in (("worker", WORKER, WORKER_SHA), ("guardian", GUARD, GUARD_SHA)):
        good = len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)
        new_frozen[label] = good
        if good:
            require(path.is_file() and sha(path) == expected, "Frozen V8 dependency changed")
            refs.append({"path": str(path), "sha256": expected, "bytes": path.stat().st_size})
        else:
            refs.append({"path": str(path), "sha256": "NOT_FROZEN", "status": "NOT_RUN_UNFROZEN"})
    frozen = all(new_frozen.values())
    v5_metadata_validators()
    formula_gate = numerical_formula_source_gate()
    output.mkdir(parents=True)
    result = {"status": "NOT_RUN" if frozen else "NOT_RUN_BLOCKED_UNFROZEN_V8_DEPENDENCY",
        "analysis_schema": "research4-V8-independent-staged-modes-analysis-v1",
        "source_dependencies": refs, "V8_consumer_source_frozen": new_frozen["worker"],
        "V8_guardian_source_frozen": new_frozen["guardian"],
        "numerical_formula_source_gate": formula_gate,
        "target_arrays_read": False, "numerical_packages_imported": False,
        "controls_numerical_validation": "NOT_RUN_REAL_MULTIBLOCK_WATER_ETHANOL_REQUIRED",
        "Spinosin_frequencies": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED",
        "old_V7_analyzer_consumer_guard_unchanged": True,
        "old_38dde_76e2_6e9_b985_unchanged": True,
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
