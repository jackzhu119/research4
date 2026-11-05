"""V7 independent RHF/XC chunk and A-assembly worker. Preparation is NOT_RUN.

Every real stage requires the actual frozen V6b budget parent. Its proof stays
V6b; this worker/results are explicitly V7 and never impersonate f137/V6b.
Only healthy closed native contributions are accepted by the metadata consumer.
"""
from __future__ import annotations
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pyscf-spinosin-grid-checkpoint-20261105-v7"
CACHE_ROOT = ROOT / "outputs/pyscf-spinosin-grid-checkpoint-20261105-v7"
SELF = Path(__file__).resolve()
XC_CORE = ROOT / "scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py"
XC_CORE_SHA = "da1e2b0267c48e748a2bcb7d3f2ee3110d8e2288aa340d23dab7a12c994ddf8e"
V6B = ROOT / "scripts/pyscf_staged_hessian_worker_20261105_v6b.py"
V6B_SHA = "f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d"
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v6b.py"
GUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
STATE = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
V5 = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
V5_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
BASE = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
BASE_SHA = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"
RHF_CORE = ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py"
RHF_CORE_SHA = "ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55"
ANALYZER_V2 = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v6b_v2.py"
ANALYZER_V2_SHA = "38dde5e185c4fc48d58732982e0164b3a2f5430a26bdfc1076b21103d1d75c6a"
SCHEMA = "research4-staged-full-rks-hessian-result-v7"
CACHE_SCHEMA = "research4-complete-full-rks-stage-cache-v7"
PHYSICAL_SCHEMA = "research4-full-rks-physical-binding-v6b"  # Same real physical state, not a producer schema.
NUMERICAL_POLICY = {"xc": "PBE", "basis": "def2-svp", "density_fitting": False,
    "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "full_AO_MO_space": True,
    "grid_changed": False, "MF_Hessian_memory_mb": 3000,
    "stock_hess_elec_max_memory_argument_mb": 4000, "conv_tol_cpscf": 1e-8,
    "hessian_max_cycle": 50, "level_shift": 0, "response_block_atoms": 8}
SUCCESS = {"xc-plan": "XC_PLAN_COMPUTED", "rhf-partial": "RHF_PARTIAL_COMPUTED",
    "xc-diag-chunk": "XC_CHUNK_COMPUTED", "xc-deriv2-chunk": "XC_CHUNK_COMPUTED",
    "rks-partial": "STAGE_COMPLETE", "control-reference": "CONTROL_REFERENCE_COMPUTED",
    "control-compare": "CONTROL_PARTIAL_DIFFERENTIAL_PASS"}
SPINOSIN_SNAPSHOT = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
SPINOSIN_G = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
CONTROL_SNAPSHOTS = {"123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4": "water_software_control",
    "9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba": "ethanol_software_control"}


def require(value, message):
    if not value: raise ValueError(message)


def utc(): return datetime.now(timezone.utc).isoformat()
def root_path(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def artifact(path):
    path = Path(path).resolve(); return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked(record):
    path = root_path(record["path"])
    require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
        and path.stat().st_size <= 64 * 1024**2 and sha(path) == record["sha256"], "Changed/missing/oversize artifact")
    require("bytes" not in record or record["bytes"] == path.stat().st_size, "Artifact size differs")
    return path


def read(path): return json.loads(Path(path).read_text())
def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def save(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n"); handle.flush()
        os.fsync(handle.fileno())
    sync_directory(path.parent)


def load_source(path, expected, name):
    require(len(expected) == 64 and sha(path) == expected, "Frozen source not ready/changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def metadata_modules():
    core = load_source(XC_CORE, XC_CORE_SHA, "V7_XC_chunk_metadata")
    guard = load_source(GUARD, GUARD_SHA, "original_real_V6b_budget_parent")
    native = load_source(V6B, V6B_SHA, "original_V6b_physical_binding_metadata")
    require(native.PHYSICAL_SCHEMA == PHYSICAL_SCHEMA and native.NUMERICAL_POLICY == NUMERICAL_POLICY, "Physical model/policy changed")
    return core, guard, native


def option(command, name):
    require(command.count(name) == 1 and command.index(name) + 1 < len(command), "Unique actual option required: " + name)
    return command[command.index(name) + 1]


def component_command(value, path, closure, guard):
    proof = value["resource_supervision"]["proof"]
    ledger = read(root_path(proof["ledger_path"]))
    rows = [r for r in ledger["phases"] if r.get("phase") == proof["phase"]]
    require(len(rows) == 1 and guard.healthy_closed_row(rows[0]), "Actual component phase not healthy")
    row = rows[0]; command = row["command"]; args = read(path.parent / "input.json")["arguments"]
    require(command.count("--execute") == 1 and option(command, "--stage") == args["stage"] == value["stage"]
        and root_path(option(command, "--output")) == path.parent
        and int(option(command, "--threads")) == args["threads"] == value["settings"]["threads"]
        and command.count("--control-validation") == int(args["control_validation"]), "Actual component stage/identity arguments differ")
    for flag, key in (("--snapshot", "snapshot"), ("--snapshot-manifest-sha256", "snapshot_manifest_sha256"),
            ("--gradient-check-result", "gradient_check_result"), ("--gradient-check-result-sha256", "gradient_check_result_sha256")):
        observed = option(command, flag)
        require(observed == args[key] if key.endswith("sha256") else root_path(observed) == root_path(args[key]), "Actual source/G argv differs")
    require(args["snapshot_manifest_sha256"] == value["snapshot_manifest_sha256"]
        and args["gradient_check_result_sha256"] == value["gradient_gate"]["result"]["sha256"]
        and root_path(args["gradient_check_result"]) == root_path(value["gradient_gate"]["result"]["path"]), "Declared source/G differs")
    keys = []
    if value["stage"] == "xc-plan": keys += [("--xc-max-memory-mb", "xc_max_memory_mb")]
    if value["stage"] in ("xc-diag-chunk", "xc-deriv2-chunk"):
        keys += [("--plan-result", "plan_result"), ("--plan-result-sha256", "plan_result_sha256"),
            ("--start-block", "start_block"), ("--block-count", "block_count")]
    if value["stage"] == "rks-partial": keys += [("--component-index", "component_index"), ("--component-index-sha256", "component_index_sha256")]
    if value["stage"] == "control-compare":
        keys += [("--partial-result", "partial_result"), ("--partial-result-sha256", "partial_result_sha256"),
            ("--reference-result", "reference_result"), ("--reference-result-sha256", "reference_result_sha256")]
    if value["stage"] == "rhf-partial" and args.get("rhf_cache") is not None: keys += [("--rhf-cache", "rhf_cache")]
    if not args["control_validation"]:
        keys += [("--control-release", "control_release"), ("--control-release-sha256", "control_release_sha256")]
    for flag, key in keys:
        observed = option(command, flag)
        if key.endswith("sha256"): require(observed == args[key], "Actual component hash argv differs")
        elif key in ("start_block", "block_count"): require(int(observed) == args[key], "Actual chunk range argv differs")
        elif key == "xc_max_memory_mb": require(float(observed) == args[key], "Actual frozen memory argv differs")
        else: require(root_path(observed) == root_path(args[key]), "Actual component artifact argv differs")
    records = [artifact(path), artifact(path.parent / "input.json"), artifact(root_path(value["resource_supervision"]["proof_path"])), artifact(root_path(row["log"]))]
    require(records[0]["sha256"] == closure["producer_result_sha256"]
        and records[2]["sha256"] == closure["proof_sha256"] and records[3]["sha256"] == closure["log_sha256"], "Closed component receipt bytes differ")
    if not args["control_validation"]:
        actual_release = artifact(checked(value["actual_control_release"]))
        require(root_path(actual_release["path"]) == root_path(args["control_release"])
            and actual_release["sha256"] == args["control_release_sha256"], "Actual root control release changed")
        records.append(actual_release)
    return args, records, {"path": str(root_path(proof["ledger_path"])), "phase": row["phase"],
        "closed_row_sha256": closure["closed_row_sha256"], "hash_style": "json_sort_keys_default"}


def qualify_component(record, stage, physical, atoms, nao, guard=None):
    core, actual_guard, _ = metadata_modules(); guard = guard or actual_guard
    path = checked(record); value = read(path)
    require(path.name == "result.json" and path.is_relative_to(REPORT_ROOT)
        and value.get("schema") == SCHEMA and value.get("stage") == stage
        and value.get("status") == SUCCESS[stage] and value.get("atoms") == atoms and value.get("basis_functions") == nao
        and value.get("physical_binding_sha256") == digest(physical) and value.get("physical_binding") == physical
        and value.get("source_state_binding") == physical["stable_snapshot_binding"]
        and value.get("stable_MF_state_unchanged") is True and value.get("no_new_SCF_performed") is True,
        "Unsupported/incomplete actual V7 component")
    evidence = {"source_worker": value["source_worker"], "resource_supervision": value["resource_supervision"],
        "output_directory": str(path.parent), "snapshot_manifest_sha256": value["snapshot_manifest_sha256"], "stage": stage}
    closure = guard.verify_closed_phase(evidence, expected_worker_path=SELF, expected_worker_sha256=sha(SELF),
        expected_result_schema=SCHEMA, allowed_result_statuses=(SUCCESS[stage],), expected_output=path.parent,
        expected_snapshot_sha256=physical["snapshot_manifest_sha256"])
    require(closure["producer_result_sha256"] == record["sha256"] and value["XC_core_source_sha256"] == XC_CORE_SHA,
        "V7 component source/result differs")
    require(value["actual_imported_native_sources"] == {name: {"path": str(core.PYSCF_ROOT / name), "sha256": expected}
        for name, expected in core.PINNED.items()}, "Actually executed PySCF source identity differs")
    args, records, row = component_command(value, path, closure, guard)
    return value, closure, args, records, row


def qualify_plans(record, physical, atoms, nao, guard=None):
    value, closure, args, records, row = qualify_component(record, "xc-plan", physical, atoms, nao, guard)
    path = checked(value["plans_artifact"]); plans = read(path)
    core, _, _ = metadata_modules()
    require(set(plans) == set(core.KINDS), "Both independently frozen XC plans required")
    for plan in plans.values():
        core.validate_plan(plan)
        require(plan["physical_binding_sha256"] == digest(physical) and plan["natm"] == atoms and plan["nao"] == nao
            and plan["max_memory_mb"] == args["xc_max_memory_mb"]
            and plan["full_global_grid_and_masks"] == value["actual_full_grid_binding"]
            and plan["blocksize_mode"] == ("EXPLICIT_CONTROL224" if args["control_validation"] else "FROZEN_ORIGINAL_AUTOMATIC_FORMULA"), "XC plan source/memory policy changed")
    require(float(option_from_args_metadata(args, "xc_max_memory_mb")) == 2000., "Initial V7 only supports explicit2000MB frozen XC plans")
    return plans, {"record": record, "producer": closure, "gradient_gate": value["gradient_gate"],
        "boot_id": value["boot_id"]}, records + [artifact(path)], [row]


def option_from_args_metadata(args, key): return args[key]


def payload(value, key):
    records = value["arrays"]; own = [r for r in records if r["key"] == key]
    require(len(own) == 1, "Unique component payload required: " + key)
    row = own[0]; directory = root_path(value["output_directory"]) / "stage-cache"
    require(Path(row["file"]).name == row["file"] and row["file"].endswith(".npy") and row["dtype"] == "float64", "Payload identity/dtype changed")
    path = checked({"path": str(directory / row["file"]), "sha256": row["file_sha256"]})
    return row, path


def _assembly_sources(index_record, physical, atoms, nao, guard):
    core, _, _ = metadata_modules(); path = checked(index_record); index = read(path)
    require(index.get("schema") == "research4-v7-XC-component-index-v1" and set(index) == {"schema", "RHF_partial", "XC_plan", "XC_chunks"}, "Explicit complete component index required")
    plans, plan_producer, records, ledger_rows = qualify_plans(index["XC_plan"], physical, atoms, nao, guard)
    records.append(artifact(path))
    rhf, rhf_closure, _, own, row = qualify_component(index["RHF_partial"], "rhf-partial", physical, atoms, nao, guard)
    require(rhf["gradient_gate"] == plan_producer["gradient_gate"] and rhf["boot_id"] == plan_producer["boot_id"],
        "Plan and native RHF use different same-state freshG/boot")
    require(all(rhf["actual_full_grid_binding"] == plan["full_global_grid_and_masks"] for plan in plans.values()),
        "Native RHF and XC plans use different full-grid identity")
    require(rhf["complete_RHF_helper"] is True and rhf["includes_XC_or_nuclear_or_CPHF"] is False, "RHF component must include only nativee1+ej")
    rr, rp = payload(rhf, "full_RHF_partial"); require(rr["shape"] == [atoms, atoms, 3, 3], "Complete native RHF shape required")
    records += own + [artifact(rp)]; ledger_rows.append(row)
    chunks, ordered = [], []
    for record in index["XC_chunks"]:
        path = checked(record); stage = read(path)["stage"]
        require(stage in ("xc-diag-chunk", "xc-deriv2-chunk"), "Non-XC or unsupported component in XC list")
        value, closure, args, own, row = qualify_component(record, stage, physical, atoms, nao, guard)
        kind = value["XC_chunk"]["kind"]
        require(value["plan_result_dependency"] == index["XC_plan"] and kind == stage[:-6]
            and value["gradient_gate"] == rhf["gradient_gate"] and value["boot_id"] == rhf["boot_id"], "Chunk source/G/plan changed")
        require(value["actual_full_grid_binding"] == plans[kind]["full_global_grid_and_masks"], "Actual chunk grid differs from frozen source plan")
        require(root_path(args["plan_result"]) == root_path(index["XC_plan"]["path"])
            and args["plan_result_sha256"] == index["XC_plan"]["sha256"]
            and args["start_block"] == value["XC_chunk"]["start_block"] and args["block_count"] == value["XC_chunk"]["block_count"], "Actual chunk arguments differ")
        require(value["XC_chunk"]["native_helper_returned_normally"] is True
            and value["XC_chunk"]["original_block_loop_restored"] is True
            and value["XC_chunk"]["all_selected_original_blocks_completed"] == [b["index"] for b in value["XC_chunk"]["selected_blocks"]]
            and value["XC_chunk"]["no_extra_ipip_nuclear_RHF_or_CPHF_added"] is True, "Native chunk was incomplete or double-added")
        ar, ap = payload(value, "contracted_XC_partial"); require(ar["shape"] == [atoms, atoms, 3, 3], "Complete XC contraction shape required")
        chunks.append(value["XC_chunk"]); ordered.append((kind, value["XC_chunk"]["start_block"], value, ap, record, closure))
        records += own + [artifact(ap)]; ledger_rows.append(row)
    cover = core.coverage(plans, chunks)
    ordered.sort(key=lambda r: (r[0], r[1]))
    return {"index": index, "plans": plans, "plan_producer": plan_producer,
        "RHF": rhf, "RHF_closure": rhf_closure, "RHF_path": rp,
        "chunks": ordered, "coverage": cover, "records": records, "closed_ledger_rows": ledger_rows}


def qualify_partial(record, physical_binding, atoms, nao, guard):
    """Stdlib V7 A contract: full healthy RHF+XC chain before any payload load."""
    value, closure, args, records, row = qualify_component(record, "rks-partial", physical_binding, atoms, nao, guard)
    require(value["complete_RHF_helper"] is True and value["complete_RKS_partial"] is True
        and value["V6b_producer_compatibility"] == "BLOCKED_NEW_V7_XC_CHUNK_PRODUCER_REQUIRES_NEW_CONSUMER", "V7 completeA cannot impersonate frozenf137")
    require(root_path(args["component_index"]) == root_path(value["component_index"]["path"])
        and args["component_index_sha256"] == value["component_index"]["sha256"], "Actual assembly component index differs")
    sources = _assembly_sources(value["component_index"], physical_binding, atoms, nao, guard)
    require(value["gradient_gate"] == sources["RHF"]["gradient_gate"]
        and value["boot_id"] == sources["RHF"]["boot_id"]
        and value["actual_full_grid_binding"] == sources["RHF"]["actual_full_grid_binding"]
        and value["actual_XC_coverage"] == sources["coverage"], "Assembly freshG/coverage changed")
    directory = checked(record).parent / "stage-cache"; manifest_path = directory / "manifest.json"
    require(sha(manifest_path) == value["cache_manifest_sha256"], "Final A cache manifest changed")
    manifest = read(manifest_path)
    require(manifest["schema"] == CACHE_SCHEMA and manifest["stage"] == "rks-partial" and manifest["complete"] is True
        and manifest["physical_binding"] == physical_binding and manifest["physical_binding_sha256"] == digest(physical_binding)
        and manifest["includes_nuclear_hessian"] is False and manifest["includes_CPHF_response"] is False,
        "Full V7 A manifest not complete or physical state changed")
    require(manifest["producer"] == {"result_path": str(checked(record)), "source_worker_sha256": sha(SELF),
        "supervisor_proof_sha256": value["resource_supervision"]["proof_sha256"], "stage": "rks-partial"}, "V7 A producer binding changed")
    require(len(manifest["arrays"]) == 1, "Unique complete A required")
    ar = manifest["arrays"][0]
    require(ar["key"] == "full_RKS_partial" and ar["file"] == "full_RKS_partial.npy"
        and ar["shape"] == [atoms, atoms, 3, 3] and ar["dtype"] == "float64", "Complete A payload identity differs")
    ap = checked({"path": str(directory / ar["file"]), "sha256": ar["file_sha256"]})
    unique = {r["path"]: r for r in records + sources["records"] + [artifact(manifest_path), artifact(ap), artifact(XC_CORE), artifact(SELF)]}
    extra = {"schema": "research4-v7-complete-xc-partial-lineage-v1", "RHF_partial": sources["index"]["RHF_partial"],
        "XC_plan": sources["plan_producer"], "XC_chunks": [{"kind": r[0], "start_block": r[1], "record": r[4], "producer": r[5]} for r in sources["chunks"]],
        "RHF_producers": [{"record": sources["index"]["RHF_partial"], "producer": sources["RHF_closure"]}],
        "coverage": sources["coverage"], "all_chunk_producers_healthy": True,
        "all_required_grid_blocks_covered_exactly_once": True, "RHF_added_once": True, "XCdiag_once": True,
        "XCderiv2_once": True, "nuclear_or_CPHF_included": False, "actual_partial_command_verified": True,
        "actual_component_commands_verified": True, "verified_artifacts": list(unique.values()),
        "closed_ledger_rows": [row, *sources["closed_ledger_rows"]], "old_failed_costs_not_erased": True}
    return value, manifest, directory, closure, extra


def require_control_release(path, expected_sha, consumer_source=None):
    require(path is not None and expected_sha, "No targetSpinosin quantum work before actual multi-block water/ethanol controls")
    release = read(checked({"path": str(path), "sha256": expected_sha}))
    require(release.get("schema") == "research4-v7-actual-small-control-release-v1"
        and all(release.get(k) is True for k in ("root_execution_authorized", "v7_actual_water_ethanol_controls_reviewed", "no_budget_increase")), "Explicit root actual-control release required")
    refs = {str(checked(r)): r for r in release["code_refs"]}
    require(len(refs) == len(release["code_refs"]), "Duplicate released code path")
    require(str(SELF) in refs and refs[str(SELF)]["sha256"] == sha(SELF)
        and str(XC_CORE) in refs and refs[str(XC_CORE)]["sha256"] == XC_CORE_SHA, "Control release uses different V7 A/core")
    consumer = root_path(consumer_source["path"]) if consumer_source else ROOT / "scripts/pyscf_staged_consumer_20261105_v7.py"
    require(str(consumer) in refs and (consumer_source is None or refs[str(consumer)]["sha256"] == consumer_source["sha256"]), "Control release uses different downstream consumer")
    core, guard, _ = metadata_modules(); seen = set()
    for control in release["controls"]:
        case = control["case"]; require(case in ("water_software_control", "ethanol_software_control") and case not in seen, "Both actual distinct controls required")
        seen.add(case); n = 3 if case == "water_software_control" else 9
        partial = read(checked(control["partial_differential_result"]))
        qualify_component(control["partial_differential_result"], "control-compare", partial["physical_binding"], n, partial["basis_functions"], guard)
        require(partial["case"] == case and partial["control_all_AO_and_partial_differences_passed"] is True, "Actual native AO/partial controls did not pass")
        index_path = checked(control["global_analysis_index"]); index = read(index_path)
        require(index["V7_worker_sha256"] == refs[str(consumer)]["sha256"]
            and any(r["sha256"] == index["analysis_source_sha256"] for r in release["code_refs"]), "Unreviewed actual V7 analyzer export")
        rows = [r for r in index["files"] if r["path"] == "checks.json"]; require(len(rows) == 1, "Unique actual analyzer checks required")
        result = read(checked({"path": str(index_path.parent / "checks.json"), **{k: rows[0][k] for k in ("sha256", "bytes")}}))
        require(result["schema"] == "research4-V7-independent-staged-modes-analysis-v1"
            and result["status"] == "DONE_CONTROL_NUMERICAL_REPLAY" and result["case"] == case
            and result["normal_modes"] == 3 * n - 6 and result["all_frequencies_positive"] is True
            and result["normal_mode_count_complete"] is True
            and result["IR_intensities_status"] == "NOT_COMPUTED" and result["qualified_for_mechanism"] is False
            and result["actual_parent_sha256"] == GUARD_SHA and result["new_quantum_evaluations"] == 0
            and result["actual_old_stock_control_comparison"]["status"] == "PASS_ACTUAL_V7_VS_AUTHENTICATED_V5_STOCK_FULL_MATRIX_AND_ALL_MODES",
            "Actual complete full-Hessian/all-signed-frequency controls did not pass")
        require(result["analysis_source_sha256"] == index["analysis_source_sha256"]
            and result["actual_V7_worker_sha256"] == refs[str(consumer)]["sha256"], "Actual control analyzer/consumer source differs")
        final_paths = [root_path(p) for p, r in result["input_hashes"].items()
            if Path(p).name == "result.json" and r["sha256"] == result["source_result_sha256"]]
        require(len(final_paths) == 1, "Unique actual complete control global result required")
        global_path = checked({"path": str(final_paths[0]), "sha256": result["source_result_sha256"]})
        global_value = read(global_path)
        evidence = {"source_worker": global_value["source_worker"], "resource_supervision": global_value["resource_supervision"],
            "output_directory": str(global_path.parent), "snapshot_manifest_sha256": global_value["snapshot_manifest_sha256"], "stage": "global-assemble"}
        closed = guard.verify_closed_phase(evidence, expected_worker_path=consumer,
            expected_worker_sha256=refs[str(consumer)]["sha256"], expected_result_schema=SCHEMA,
            allowed_result_statuses=("FREQUENCIES_COMPUTED",), expected_output=global_path.parent,
            expected_snapshot_sha256=global_value["snapshot_manifest_sha256"])
        require(closed == result["producer_closure"] and global_value["case"] == case
            and global_value["normal_mode_count"] == 3 * n - 6, "Control complete global healthy producer differs")
    require(seen == {"water_software_control", "ethanol_software_control"}, "Both actual controls required before target")
    return release


def execute(args):
    # Actual live parent, old budget lineage and target control release FIRST.
    core, guard, old = metadata_modules()
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    require(args.stage in SUCCESS and args.threads == 2 and args.xc_max_memory_mb == 2000,
            "Frozen V7 stage/thread/memory settings required")
    if args.control_validation:
        require(args.snapshot_manifest_sha256 in CONTROL_SNAPSHOTS, "Only authenticated original water/ethanol control snapshots")
    else:
        require(args.snapshot_manifest_sha256 == SPINOSIN_SNAPSHOT and args.gradient_check_result_sha256 == SPINOSIN_G,
            "Original Spinosin C001 source/G only")
        require_control_release(args.control_release, args.control_release_sha256)
    output = args.output.resolve()
    require(output.is_relative_to(REPORT_ROOT) and output != REPORT_ROOT and not output.exists(), "Fresh new V7 report namespace required")
    output.mkdir(parents=True)
    state = {"schema": SCHEMA, "stage": args.stage, "phase": args.stage, "status": "PARTIAL", "started_at_utc": utc(),
        "source_worker": artifact(SELF), "XC_core_source_sha256": XC_CORE_SHA, "resource_supervision": supervision,
        "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "snapshot_directory": str(args.snapshot.resolve()),
        "no_new_SCF_performed": True, "stable_MF_state_unchanged": False, "source_MF_stability_status": "NOT_CHECKED",
        "frequencies_status": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED", "complete_RHF_helper": False,
        "complete_RKS_partial": False, "complete_H1AO": False, "complete_global_Hessian": False,
        "V6b_producer_compatibility": "BLOCKED_NEW_V7_XC_CHUNK_PRODUCER_REQUIRES_NEW_CONSUMER",
        "V7_analyzer_compatibility": "REQUIRES_REVIEWED_V7_POSTPROCESSOR_NOT_38dde_V6b",
        "actual_control_release": None if args.control_validation else artifact(args.control_release),
        "output_directory": str(output), "arrays": [], "XC_checkpoint_policy": core.POLICY}
    save(output / "input.json", {**state, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    import numpy as np
    from pyscf import lib
    from pyscf.scf import _vhf
    from pyscf.hessian import rhf, rks
    from pyscf.dft import numint, gen_grid
    from pyscf.grad import rks as grad_rks
    from pyscf.lib import linalg_helper
    lib.num_threads(args.threads)
    w = load_source(STATE, STATE_SHA, "V7_original_exact_source")
    v5 = load_source(V5, V5_SHA, "V7_original_V5_source_gradient")
    base = load_source(BASE, BASE_SHA, "V7_original_V8_native_operator")
    rhf_core = load_source(RHF_CORE, RHF_CORE_SHA, "V7_original_incremental_RHF")
    log = (output / "pyscf.log").open("x", buffering=1)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    def emit(event, details=None, **more):
        record = {"utc": utc(), "event": event, "CPU_seconds": time.process_time() - cpu0, **(details or {}), **more}
        with (output / "events.jsonl").open("a") as handle: handle.write(json.dumps(record, allow_nan=False) + "\n")
    native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
        native_threads=args.threads, emit=lambda details: emit("native_driver_signature", **details))
    mf = before = pending = None
    def stop(signum, frame): raise base.BudgetStop("V7 resource signal " + str(signum))
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    def save_array(key, value):
        value = np.asarray(value)
        require(value.dtype == np.float64 and np.isfinite(value).all(), "Finite float64 component required")
        directory = output / "stage-cache"; directory.mkdir(exist_ok=True)
        path = directory / (key + ".npy")
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
        sync_directory(directory)
        row = {"key": key, "file": path.name, "file_sha256": sha(path), "shape": list(value.shape),
            "dtype": "float64", "payload": base.array_fingerprint(value)}
        state["arrays"].append(row); return row
    def load_array(value, key):
        row, path = payload(value, key); array = np.load(path, allow_pickle=False)
        require(array.dtype == np.float64 and list(array.shape) == row["shape"] and np.isfinite(array).all()
            and base.array_fingerprint(array) == row["payload"], "Closed native payload changed")
        return array
    try:
        core.check_sources(); native.install()
        state["boot_id"] = v5.boot_id()
        state["gradient_gate"] = v5.qualify_gradient_receipt(args.gradient_check_result, args.gradient_check_result_sha256,
            args.snapshot_manifest_sha256, state["boot_id"])
        state["final_gradient_thresholds_passed"] = True
        mf, binding = w.restore_snapshot(args.snapshot, expected_manifest_sha256=args.snapshot_manifest_sha256,
            expected_source_sha256=STATE_SHA, stdout=log)
        state["actual_imported_native_sources"] = core.validate_runtime_sources(mf._numint,
            numint=numint, rks=rks, grad_rks=grad_rks, gen_grid=gen_grid)
        symbols = [mf.mol.atom_symbol(i) for i in range(mf.mol.natm)]
        case = old.case_for(symbols, args.control_validation)
        require(not args.control_validation or case == CONTROL_SNAPSHOTS[args.snapshot_manifest_sha256], "Authenticated control identity differs")
        state.update(case=case, atoms=int(mf.mol.natm), basis_functions=int(mf.mol.nao_nr()), source_state_binding=binding,
            versions=binding["environment"], energy_hartree=float(mf.e_tot))
        n, nao = state["atoms"], state["basis_functions"]
        settings = {**binding["method_settings"], "threads": args.threads, "native_direct_driver_threads": args.threads,
            "hessian_max_memory_mb": 3000, "hessian_max_cycle": 50, "hessian_level_shift": 0,
            "hessian_response_block_atoms": 8, "engineering_worker_version": "V7_HEALTHY_CLOSED_GRID_CHUNKS"}
        state["settings"] = settings
        hobj = mf.Hessian(); hobj.max_cycle, hobj.level_shift = 50, 0
        state["stock_hamiltonian_gate"] = v5.stock_operator_gate(mf, hobj)
        before = v5.physical_state(mf)
        with base.hessian_memory_context(hobj, settings, emit=emit):
            physical = {"schema": PHYSICAL_SCHEMA, "snapshot_manifest_sha256": args.snapshot_manifest_sha256,
                "stable_snapshot_binding": binding, "actual_runtime": rhf_core.runtime_state_fingerprint(hobj),
                "numerical_policy": NUMERICAL_POLICY, "frozen_derivative_sources": rhf_core._source_identity(rhf),
                "native_threads": args.threads,
                "native_krylov_lindep": float(inspect.signature(linalg_helper.krylov).parameters["lindep"].default)}
            state.update(physical_binding=physical, physical_binding_sha256=digest(physical))
            grid = core.grid_binding(hobj, np, base.array_fingerprint)
            state["actual_full_grid_binding"] = grid
            if args.stage == "xc-plan":
                plans = {kind: core.make_plan(kind=kind, nao=nao, natm=n, ngrids=grid["full_grid_points"],
                    max_memory_mb=args.xc_max_memory_mb, grid_binding=grid, physical_binding_sha256=digest(physical),
                    control=args.control_validation) for kind in core.KINDS}
                save(output / "XC_plans.json", plans); state["plans_artifact"] = artifact(output / "XC_plans.json")
            elif args.stage == "rhf-partial":
                require(getattr(hobj.partial_hess_elec, "__func__", None) is rks.partial_hess_elec, "Native unmodified RKS method required")
                if not args.control_validation:
                    require(args.rhf_cache is not None and args.rhf_cache.resolve().is_relative_to(CACHE_ROOT), "New V7 healthy149 cache namespace required")
                    prior = old.verify_rhf_producers(args.rhf_cache, args.snapshot_manifest_sha256, guard, v5)
                    require(len(prior) == 2 * n - 1 == 149, "Only old healthy149 seed may enter target RHF-only phase")
                    state["RHF_seed_producer_checks"] = prior
                    evidence = {"schema": "research4-V7-RHF-only-unit-producer-v1", "source_worker": state["source_worker"],
                        "resource_supervision": supervision, "output_directory": str(output),
                        "snapshot_manifest_sha256": args.snapshot_manifest_sha256, "phase": "rhf-partial", "stage": "rhf-partial"}
                    with rhf_core.IncrementalPureJContext(hobj, args.rhf_cache, binding, max_new_units=1, execution_evidence=evidence) as context:
                        state["incremental_before"] = context.progress()
                        e1, ej, ek = rhf._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ, list(range(n)), 4000, None, with_k=False)
                        state["incremental_after"] = context.progress()
                    require(state["incremental_after"]["total_units_completed"] == 2 * n, "Full native RHF did not return")
                else:
                    e1, ej, ek = rhf._partial_hess_ejk(hobj, mf.mo_energy, mf.mo_coeff, mf.mo_occ, list(range(n)), 4000, None, with_k=False)
                    state["RHF_seed_producer_checks"] = []
                save_array("full_RHF_partial", e1 + ej)
                state.update(complete_RHF_helper=True, includes_XC_or_nuclear_or_CPHF=False,
                    RHF_math="STOCK_rhf._partial_hess_ejk_e1_plus_ej_PBE_zero_hybrid")
            elif args.stage in ("xc-diag-chunk", "xc-deriv2-chunk"):
                require(args.plan_result is not None and args.plan_result_sha256, "Healthy frozen plan required")
                plan_record = {"path": str(args.plan_result.resolve()), "sha256": args.plan_result_sha256}
                plans, plan_producer, _, _ = qualify_plans(plan_record, physical, n, nao, guard)
                require(plan_producer["gradient_gate"] == state["gradient_gate"] and plan_producer["boot_id"] == state["boot_id"],
                    "Current restored source/G differs from actual plan")
                kind = args.stage[:-6]; plan = plans[kind]
                require(plan["full_global_grid_and_masks"] == grid and plan["nao"] == nao and plan["natm"] == n,
                    "Actual full grid/masks/molecule differs from frozen plan")
                value, details, raw = core.native_xc_chunk(hobj, plan, args.start_block, args.block_count,
                    emit=lambda event, **info: emit(event, **info), return_control_AO=args.control_validation)
                save_array("contracted_XC_partial", value)
                if raw is not None: save_array("control_native_XC_AO", raw)
                state.update(XC_chunk=details, plan_result_dependency=plan_record,
                    control_AO_saved_only_for_authenticated_small_molecule=raw is not None)
            elif args.stage == "rks-partial":
                require(args.component_index is not None and args.component_index_sha256, "Explicit immutable component index required")
                record = {"path": str(args.component_index.resolve()), "sha256": args.component_index_sha256}
                sources = _assembly_sources(record, physical, n, nao, guard)
                total = load_array(sources["RHF"], "full_RHF_partial").copy()
                for _, _, chunk, _, _, _ in sources["chunks"]: total += load_array(chunk, "contracted_XC_partial")
                require(total.shape == (n, n, 3, 3), "Full A shape mismatch")
                row = save_array("full_RKS_partial", total)
                pending = {"schema": CACHE_SCHEMA, "stage": "rks-partial", "complete": True,
                    "physical_binding": physical, "physical_binding_sha256": digest(physical), "arrays": [row],
                    "includes_nuclear_hessian": False, "includes_CPHF_response": False,
                    "producer": {"result_path": str(output / "result.json"), "source_worker_sha256": sha(SELF),
                        "supervisor_proof_sha256": supervision["proof_sha256"], "stage": "rks-partial"}}
                state.update(component_index=record, actual_XC_coverage=sources["coverage"],
                    complete_RHF_helper=True, complete_RKS_partial=True,
                    actual_assembly="ONE_RHF_plus_ordered_complete_XCdiag_and_XCderiv2_chunks_no_nuclear_CPHF")
            elif args.stage == "control-reference":
                require(args.control_validation, "Reference stage only authenticated small controls")
                dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ); slices = mf.mol.aoslice_by_atom()
                diagonal = rks._get_vxc_diag(hobj, mf.mo_coeff, mf.mo_occ, 2000)
                derivative2 = rks._get_vxc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, 2000)
                save_array("native_full_XC_diag_AO", diagonal); save_array("native_full_XC_deriv2_AO", derivative2)
                save_array("native_full_RKS_partial", hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ, list(range(n)), 4000, None))
                save_array("native_full_XC_diag_contracted", core.contract_xc("xc-diag", diagonal, dm0, slices, n, np))
                save_array("native_full_XC_deriv2_contracted", core.contract_xc("xc-deriv2", derivative2, dm0, slices, n, np))
                state["reference_scope"] = "UNMODIFIED_NATIVE_FULL_GRID_XC_AO_AND_RKS_PARTIAL_NOT_FREQUENCIES"
            elif args.stage == "control-compare":
                require(args.control_validation and args.partial_result is not None and args.reference_result is not None,
                    "Actual healthy small-control A and native reference required")
                part_record = {"path": str(args.partial_result.resolve()), "sha256": args.partial_result_sha256}
                part, _, _, _, _ = qualify_partial(part_record, physical, n, nao, guard)
                ref_record = {"path": str(args.reference_result.resolve()), "sha256": args.reference_result_sha256}
                reference, _, _, _, _ = qualify_component(ref_record, "control-reference", physical, n, nao, guard)
                sources = _assembly_sources(part["component_index"], physical, n, nao, guard)
                differences = {}
                for kind, key in (("xc-diag", "native_full_XC_diag_AO"), ("xc-deriv2", "native_full_XC_deriv2_AO")):
                    target = load_array(reference, key); summed = np.zeros_like(target)
                    for own_kind, _, chunk, _, _, _ in sources["chunks"]:
                        if own_kind == kind: summed += load_array(chunk, "control_native_XC_AO")
                    difference = float(np.max(np.abs(target - summed))); differences[kind + "_full_AO_max_abs_error"] = difference
                    require(difference <= 1e-8, "Chunk/native complete AO XC difference exceeded fixed1e-8")
                partial_difference = float(np.max(np.abs(load_array(part, "full_RKS_partial") - load_array(reference, "native_full_RKS_partial"))))
                require(partial_difference <= 1e-8, "Complete assembled/native RKS partial differs")
                rhf_difference = load_array(reference, "native_full_RKS_partial") - load_array(reference, "native_full_XC_diag_contracted") - load_array(reference, "native_full_XC_deriv2_contracted")
                rhf_difference = float(np.max(np.abs(rhf_difference - load_array(sources["RHF"], "full_RHF_partial"))))
                require(rhf_difference <= 1e-8, "Independent native full RHF contribution differs")
                save(output / "partial_differential.json", {"status": "PASS_ACTUAL_NATIVE_AO_AND_FULL_PARTIAL_DIFFERENTIAL",
                    **differences, "full_RKS_partial_max_abs_error": partial_difference, "RHF_partial_max_abs_error": rhf_difference,
                    "fixed_absolute_tolerance": 1e-8, "complete_native_Hessian_and_frequencies": "NOT_TESTED_BY_THIS_STAGE"})
                state.update(control_all_AO_and_partial_differences_passed=True, partial_differential=artifact(output / "partial_differential.json"),
                    partial_result_dependency=part_record, reference_result_dependency=ref_record,
                    complete_Hessian_frequency_control_required_separately=True)
            require(core.grid_binding(hobj, np, base.array_fingerprint) == grid, "Native phase changed full frozen grid/masks")
        state["stable_MF_state_unchanged"] = v5.physical_state(mf) == before
        require(state["stable_MF_state_unchanged"], "Source MF changed")
        state["source_MF_stability_status"] = "VERIFIED_UNCHANGED"
        state["status"] = SUCCESS[args.stage]
    except BaseException as exc:
        state.update(status="PARTIAL", exception_type=type(exc).__name__, stop_reason=str(exc))
        (output / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        native.restore(); state["native_direct_driver_control"] = native.summary()
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        if state["native_direct_driver_control"].get("original_driver_restored") is not True or state["native_direct_driver_control"].get("restoration_failures") != 0:
            state.update(status="PARTIAL", stop_reason="Original native operator was not restored")
        if state["status"] == "STAGE_COMPLETE" and pending is not None:
            save(output / "stage-cache/manifest.json", pending); state["cache_manifest_sha256"] = sha(output / "stage-cache/manifest.json")
        state.update(completed_at_utc=utc(), process_cpu_seconds=time.process_time() - cpu0,
            wall_seconds=time.perf_counter() - wall0, peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        log.flush(); log.close(); save(output / "result.json", state)
    return state


def preflight(output):
    require(not output.exists(), "Fresh V7 metadata preflight required")
    core, _, _ = metadata_modules(); _, evidence = core.filtered_loop_ast()
    output.mkdir(parents=True)
    value = {"status": "NOT_RUN", "schema": SCHEMA, "source_worker": artifact(SELF), "XC_core": artifact(XC_CORE),
        "real_parent": artifact(GUARD), "native_sources": core.check_sources(), "AST_selection_evidence": evidence,
        "old_f137_source_unchanged": sha(V6B) == V6B_SHA, "actual_numerical_control_tests": "NOT_RUN",
        "actual_Spinosin_quantum_work_authorized": False, "target_arrays_or_Q_ledger_read": False,
        "native_XC_values_or_partial_matrices_computed": False,
        "frozen_control_blocksize": 224, "frozen_target_memory_mb": 2000, "IR_intensities_status": "NOT_COMPUTED",
        "orphan_CPU_cost_not_forgiven": True, "no_automatic_budget_increase": True}
    save(output / "preflight.json", value); return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--stage", choices=tuple(SUCCESS))
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--snapshot", type=Path); parser.add_argument("--snapshot-manifest-sha256")
    parser.add_argument("--gradient-check-result", type=Path); parser.add_argument("--gradient-check-result-sha256")
    parser.add_argument("--control-validation", action="store_true"); parser.add_argument("--xc-max-memory-mb", type=float, default=2000.)
    parser.add_argument("--plan-result", type=Path); parser.add_argument("--plan-result-sha256")
    parser.add_argument("--start-block", type=int, default=0); parser.add_argument("--block-count", type=int, default=1)
    parser.add_argument("--rhf-cache", type=Path); parser.add_argument("--component-index", type=Path); parser.add_argument("--component-index-sha256")
    parser.add_argument("--partial-result", type=Path); parser.add_argument("--partial-result-sha256")
    parser.add_argument("--reference-result", type=Path); parser.add_argument("--reference-result-sha256")
    parser.add_argument("--control-release", type=Path); parser.add_argument("--control-release-sha256")
    args = parser.parse_args()
    if args.execute:
        require(args.stage and args.snapshot is not None and args.snapshot_manifest_sha256
            and args.gradient_check_result is not None and args.gradient_check_result_sha256, "Explicit native source/G/stage required")
        result = execute(args)
    else:
        require(args.stage is None and args.snapshot is None, "Metadata preflight never consumes target state")
        result = preflight(args.output)
    print(json.dumps({k: result.get(k) for k in ("status", "stage", "case", "atoms")}))
    return 0 if result["status"] in (*SUCCESS.values(), "NOT_RUN") else 2


if __name__ == "__main__": raise SystemExit(main())

