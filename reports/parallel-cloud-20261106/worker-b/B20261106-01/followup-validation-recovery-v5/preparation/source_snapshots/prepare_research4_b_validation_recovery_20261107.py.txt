"""Prepare the finite v5 checkpoint recovery; never execute science or Git.

--build-reviewed-spec creates an additive 33-job proposal. A later invocation
with --spec/--spec-sha256 prepares its pinned manifest for independent review.
Every output uses exclusive creation; old allocations and artifacts stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/parallel-cloud-20261106/worker-b/B20261106-01"
PREVIOUS = BASE / "followup-validation-v4"
WORK = BASE / "followup-validation-recovery-v5"
SUPERVISOR = ROOT / "scripts/research4_b_validation_supervisor_recovery_20261107.py"
WORKER = ROOT / "scripts/research4_b_validation_recovery_worker_20261107.py"
HELPER = ROOT / "scripts/research4_parallel_b_git_return_20261106.py"
PRIOR_MANIFEST = PREVIOUS / "manifest_v2_ROOT_AUTHORIZED.json"
CLOSURE_REVIEW = WORK / "preparation/v4_closure_remote_review.json"
SPEC_SCHEMA = "research4-b-validation-recovery-spec-v5"
PROBE_ACID = "acid-g3-checkpoint-probe"
PROBE_CALY = "caly-g3-checkpoint-probe"
PRIORITY_CENTERS = ("acid-g3-fixed-center", "caly-g3-regrid-center")


def load_supervisor():
    spec = importlib.util.spec_from_file_location("r4_b_validation_recovery_prepare", SUPERVISOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def unique_refs(refs):
    found = {}
    for binding in refs:
        key = binding["path"]
        if key in found and found[key] != binding:
            raise ValueError("Conflicting references to " + key)
        found[key] = binding
    return list(found.values())


def predecessor_bindings(s, state):
    return {"predecessor_manifest": s.ref(PRIOR_MANIFEST), "predecessor_ledger": s.ref(PREVIOUS / "cpu_budget.json"),
        "predecessor_state": s.ref(PREVIOUS / "state.json"),
        "predecessor_supervisor": s.ref(ROOT / "scripts/research4_b_validation_supervisor_20261107.py"),
        "predecessor_allocation": s.ref(ROOT / ".runtime/worker-b-validation-allocation.json"),
        "predecessor_closure_review": s.ref(CLOSURE_REVIEW), "inherited_results": state["results"],
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def prepare(spec_path, spec_sha256):
    s = load_supervisor(); spec_path = s.path(spec_path); spec = s.read(spec_path, spec_sha256)
    s.require(spec.get("schema") == SPEC_SCHEMA and spec.get("jobs"), "Reviewed finite v5 spec required")
    s.require(not any((WORK / name).exists() for name in ("manifest_ROOT_AUTHORIZED.json", "return_manifest.json", "cpu_budget.json", "state.json"))
        and not (ROOT / ".runtime/worker-b-validation-recovery-allocation.json").exists(),
        "V5 manifest/allocation already exists; never reset a prepared or spent allocation")
    prior = s.read(PRIOR_MANIFEST); state = s.read(PREVIOUS / "state.json")
    bindings = predecessor_bindings(s, state)
    s.require(spec["predecessor_bindings"] == bindings, "Predecessor inputs or recovery boot changed since spec preparation")
    manifest = {key: prior[key] for key in ("run_id", "branch", "python_executable", "cpu_cap_seconds", "deadline_utc",
        "threads", "memory_gib", "rss_stop_gib")}
    manifest.update(schema=s.MANIFEST_SCHEMA, work_directory=str(WORK.relative_to(ROOT)),
        ledger=str((WORK / "cpu_budget.json").relative_to(ROOT)), reserve_cpu_seconds=spec["reserve_cpu_seconds"],
        jobs=spec["jobs"], same_allocation_no_budget_reset=True, **bindings)
    s.require(manifest["reserve_cpu_seconds"] >= 300, "Retain explicit300 CPU-second reserve")
    carry = s.validate_carry(manifest)
    s.require(datetime.now(timezone.utc) < datetime.fromisoformat(manifest["deadline_utc"]), "Original deadline expired")
    refs = unique_refs([s.ref(SUPERVISOR), s.ref(__file__), *spec["source_refs"]])
    inputs = unique_refs([s.ref(spec_path), *spec["input_refs"]])
    manifest["authorization_ref"] = spec["authorization_ref"]
    for binding in refs + inputs + [manifest["authorization_ref"]]: s.checked(binding)
    manifest["source_refs"] = refs; manifest["input_refs"] = inputs
    s.validate(manifest, "0" * 64)
    return_path = WORK / "return_manifest.json"
    upload = spec.get("upload", {"enabled": True})
    if upload.get("enabled"):
        allowlist = upload["allowlist"]; prefix = WORK.name + "/"
        s.require(isinstance(allowlist, list) and allowlist and all(isinstance(p, str) and p.startswith(prefix)
            and ".." not in Path(p).parts for p in allowlist), "Return allowlist must remain within v5")
        returning = {"schema": "research4-b-validation-recovery-return-manifest-v5", "role": "B", "run_id": s.RUN_ID,
            "work_directory": str(BASE.relative_to(ROOT)), "ledger": manifest["ledger"],
            "upload": {"branch": s.BRANCH, "allowlist": allowlist}}
        s.save(return_path, returning, exclusive=True)
        manifest["source_refs"] = unique_refs([*refs, s.ref(HELPER)])
        manifest["input_refs"] = unique_refs([*inputs, s.ref(return_path)])
        manifest["upload"] = {"enabled": True, "helper": s.ref(HELPER), "manifest": s.ref(return_path)}
    else:
        manifest["upload"] = {"enabled": False}
    target = WORK / "manifest_ROOT_AUTHORIZED.json"; s.save(target, manifest, exclusive=True)
    s.validate(manifest, s.sha(target))
    return {"status": "PREPARED_NOT_RUN", "manifest": s.ref(target), "jobs": len(manifest["jobs"]),
        "inherited_successes": len(manifest["inherited_results"]), "recovery_boot_id": manifest["recovery_boot_id"],
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": s.CAP - carry["budget_charged_cpu_seconds"],
        "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in manifest["jobs"]), "new_quantum_evaluations": 0, "git_mutations": 0}


def recovery_job_order(prior_jobs, state):
    """Return the root-reviewed center-first topology without changing scope."""
    pending = [j for j in prior_jobs if state["jobs"][j["id"]]["status"] != "DONE_SEALED"]
    centers = [j for j in pending if j["id"].endswith("-center")]
    signed = [j for j in pending if j["id"].endswith(("-minus", "-plus"))]
    aggregate = [j for j in pending if j not in centers and j not in signed]
    if len(pending) != 31 or len(centers) != 5 or len(signed) != 20 or len(aggregate) != 6:
        raise ValueError("Expected the exact closed partial v4 remainder:5 centers,20 signed points,6 aggregates")
    by_id = {j["id"]: j for j in centers}
    ordered_centers = [by_id[name] for name in PRIORITY_CENTERS]
    ordered_centers += [j for j in centers if j["id"] not in PRIORITY_CENTERS]
    return [*ordered_centers, *signed, *aggregate]


def old_request(s, job):
    argv = job["argv"]
    p = s.path(argv[argv.index("--request") + 1]); expected = argv[argv.index("--request-sha256") + 1]
    return p, s.read(p, expected)


def failed_checkpoint_inputs(s, job_id, ledger):
    rows = [r for r in ledger["phases"] if r["job_id"] == job_id]
    s.require(len(rows) == 1 and rows[0]["status"] == "FAILED_CLOSED" and rows[0]["stop_reason"] == "ACTUAL_CPU_GUARD",
        "Expected one trusted old CPU-guard closure for " + job_id)
    row = rows[0]; result = s.path(row["output_directory"]) / "result.json"
    s.require(s.sha(result) == row["result_sha256"], "Closed failed result changed")
    old_result = s.read(result)
    progress = [binding for binding in old_result.get("output_artifacts", [])
        if Path(binding["path"]).name == "progress.json"]
    s.require(len(progress) == 1, "One historical progress snapshot required for checkpoint provenance")
    s.checked(progress[0])
    checkpoint = result.parent / "fresh_point_scf.chk"
    s.require(checkpoint.is_file() and not checkpoint.is_symlink(), "Preserved failed SCF checkpoint is absent")
    return {"warmstart_result": s.ref(result), "warmstart_checkpoint": s.ref(checkpoint),
        "closed_phase": row["phase"], "wait4_receipt": row["wait4_receipt"], "last_progress": progress[0],
        "checkpoint_hash_scope": "First explicitly pinned during v5 preparation; old partial result did not publish a checkpoint checksum."}


def build_reviewed_spec():
    s = load_supervisor(); prior = s.read(PRIOR_MANIFEST); state = s.read(PREVIOUS / "state.json")
    ledger = s.read(PREVIOUS / "cpu_budget.json"); bindings = predecessor_bindings(s, state)
    carry = s.validate_carry(bindings)
    target = WORK / "reviewed_spec_ROOT_AUTHORIZED.json"
    s.require(not target.exists(), "Preserve existing v5 reviewed spec")
    s.require(len(state["results"]) == 43, "Exactly43 prior sealed successes must be inherited")
    ordered = recovery_job_order(prior["jobs"], state); original_by_id = {j["id"]: j for j in prior["jobs"]}
    old_counts = Counter(r["job_id"] for r in ledger["phases"])
    center_ids = [j["id"] for j in ordered if j["id"].endswith("-center")]
    checkpoint_inputs = {name: failed_checkpoint_inputs(s, name, ledger) for name in center_ids}
    root_protocol = WORK / "preparation/root_recovery_specification.json"
    science_review = s.ref(WORK / "preparation/independent_newton_science_review.json")
    s.require(science_review["sha256"] == "43ef85d3db32c6a35c768167965efe40fa8f0f403097250bd7161e956b02d7d7",
        "Pinned independent recovery science review changed")
    inputs = [*state["results"].values(), s.ref(CLOSURE_REVIEW), s.ref(root_protocol), science_review, *[bindings[key] for key in
        ("predecessor_manifest", "predecessor_ledger", "predecessor_state", "predecessor_supervisor", "predecessor_allocation")]]
    for values in checkpoint_inputs.values():
        inputs.extend([values["warmstart_result"], values["warmstart_checkpoint"], values["wait4_receipt"], values["last_progress"]])
    checkpoint_manifest = WORK / "preparation/checkpoint_input_bindings.json"
    s.save(checkpoint_manifest, {"schema": "research4-b-v5-checkpoint-input-bindings-v1", "inputs": checkpoint_inputs}, exclusive=True)
    inputs.append(s.ref(checkpoint_manifest))
    grant_path = WORK / "preparation/user_authorization_validation_recovery.json"
    grant = {"schema": "research4-b-validation-recovery-user-authorization-v5", "recorded_at_utc": s.utc(),
        "user_instruction": "授权你更多时间，恢复一直继续跑",
        "interpretation": "Authorizes checkpoint diagnosis and repaired finite continuation. More time does not create or refund CPU; the original cap and still-sufficient deadline remain unchanged.",
        "cpu_cap_seconds": s.CAP, "deadline_utc": prior["deadline_utc"], "recovery_boot_id": bindings["recovery_boot_id"],
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": s.CAP - carry["budget_charged_cpu_seconds"], "reserve_cpu_seconds": 300,
        "old_successes_reused_without_recomputation": 43, "old_guard_failures_charged_and_preserved": 5,
        "predeclared_new_jobs": {"checkpoint_probes": 2, "repaired_centers": 5, "signed_points": 20, "aggregates": 6},
        "order": "Two independent probes, acid-g3-fixed and caly-g3-regrid representative centers, remaining centers, signed pairs, aggregates.",
        "settings": {"direct_scf_tol": 1e-15, "scf_conv_tol": 1e-12, "scf_conv_tol_grad": 1e-9,
            "fresh_fock_gradient_qualification_max": 1e-8, "small_rho_cutoff": 0, "scf_solver": "warmstart_cdiis_then_newton",
            "warmstart_scf_max_cycle": 8, "newton_max_cycle": 12, "newton_ah_max_cycle": 20, "newton_ah_lindep": 1e-24},
        "probe_controls": {"direct_scf_tolerances": [1e-13, 1e-15], "SCF_kernel_or_nuclear_gradient_runs": 0},
        "probe_decision_rule": "Probe results are recorded; they cannot silently change any predeclared numerical threshold or solver setting.",
        "failure_policy": "Each old failed center has one remaining attempt; scientific qualification failures are not success and are not retried. No unknown closure restart; true operational retries require wait4 evidence and remaining pair budget.",
        "predecessor_closure_review": s.ref(CLOSURE_REVIEW), "root_recovery_protocol": s.ref(root_protocol), "hourly_automation_created": False,
        "preservation": "Every old source, state, ledger, allocation, failure artifact and result remains unchanged."}
    s.save(grant_path, grant, exclusive=True); inputs.append(s.ref(grant_path))
    sources = [s.ref(WORKER), s.ref(SUPERVISOR), s.ref(__file__)]; snapshots = []
    for source in (WORKER, SUPERVISOR, Path(__file__).resolve(),
        ROOT / "scripts/test_research4_b_validation_supervisor_recovery_20261107.py",
        ROOT / "scripts/test_research4_b_validation_recovery_worker_20261107.py"):
        s.require(source.is_file(), "Reviewed v5 source/test absent: " + str(source))
        dest = WORK / "preparation/source_snapshots" / (source.name + ".txt"); dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle: handle.write(source.read_bytes())
        sources.append(s.ref(source)); inputs.append(s.ref(dest))
        snapshots.append({"source": s.ref(source), "exact_byte_snapshot": s.ref(dest)})
    snapshot_manifest = WORK / "preparation/source_snapshots/source_refs.json"
    s.save(snapshot_manifest, {"files": snapshots}, exclusive=True); inputs.append(s.ref(snapshot_manifest))
    jobs = []; changes = []
    python = str(s.python_path(prior["python_executable"]))
    def write_request(name, request):
        p = WORK / "requests" / (name + ".json"); s.save(p, request, exclusive=True)
        binding = s.ref(p); inputs.append(binding)
        return p, binding
    def argv_for(p, binding):
        return [python, str(WORKER), "--request", str(p), "--request-sha256", binding["sha256"], "--output", "{output}", "--execute"]
    for name, center_name, cpu, wall in ((PROBE_ACID, "acid-g3-fixed-center", 120, 600),
        (PROBE_CALY, "caly-g3-regrid-center", 240, 900)):
        old_p, request = old_request(s, original_by_id[center_name]); inputs.append(s.ref(old_p))
        request = dict(request); request.pop("gradient_response_controls", None)
        request.update(operation="checkpoint_probe", direct_scf_tol=1e-15,
            probe_direct_scf_tolerances=[1e-13, 1e-15],
            warmstart_result=checkpoint_inputs[center_name]["warmstart_result"],
            warmstart_checkpoint=checkpoint_inputs[center_name]["warmstart_checkpoint"])
        p, binding = write_request(name, request)
        jobs.append({"id": name, "argv": argv_for(p, binding), "cpu_seconds": cpu, "wall_seconds": wall,
            "depends_on": [], "max_attempts": 1, "result_schema": "research4-b-diagnostic-result-v1",
            "success_statuses": ["CHECKPOINT_PROBE_COMPLETE"]})
    for old_job in ordered:
        name = old_job["id"]; old_p, request = old_request(s, old_job); inputs.append(s.ref(old_p))
        request = dict(request); job = json.loads(json.dumps(old_job)); operation = request["operation"]
        if operation in ("point", "fixed_grid_center", "fixed_grid_point"):
            s.require(request["scf_conv_tol"] == 1e-12 and request["scf_conv_tol_grad"] == 1e-9
                and request["scf_qualification_gradient_norm"] == 1e-8 and request["small_rho_cutoff"] == 0,
                "Predecessor numerical qualification scope differs")
            request.update(direct_scf_tol=1e-15, scf_solver="warmstart_cdiis_then_newton",
                warmstart_scf_max_cycle=8, newton_max_cycle=12, newton_ah_max_cycle=20)
        if name in checkpoint_inputs:
            request.update(warmstart_result=checkpoint_inputs[name]["warmstart_result"],
                warmstart_checkpoint=checkpoint_inputs[name]["warmstart_checkpoint"])
            job["depends_on"] = [PROBE_ACID if name.startswith("acid-") else PROBE_CALY]
        p, binding = write_request(name, request)
        # Preserve every old dynamic point/center/aggregate CLI binding. Only
        # the interpreter/worker/request selector and warmstart flag are new.
        old_argv = old_job["argv"]
        job["argv"] = argv_for(p, binding) + old_argv[old_argv.index("--execute") + 1:]
        if name.endswith(("-minus", "-plus")):
            s.require(len(old_job["depends_on"]) == 1, "Signed point must have one matching center")
            center = old_job["depends_on"][0]
            job["argv"].extend(["--warmstart-result", "{result:" + center + "}",
                "--warmstart-result-sha256", "{sha256:" + center + "}"])
        remaining_attempts = min(2, old_job.get("max_attempts", 1) - old_counts[name])
        s.require(remaining_attempts >= 1, "Historical attempts exhausted: " + name)
        job["max_attempts"] = remaining_attempts
        if name in checkpoint_inputs: s.require(remaining_attempts == 1, "Old failed center must retain exactly one remaining attempt")
        jobs.append(job)
        changes.append({"job_id": name, "old_request": s.ref(old_p), "new_request": binding,
            "historical_phase_count": old_counts[name], "remaining_max_attempts": remaining_attempts,
            "cpu_seconds": job["cpu_seconds"], "wall_seconds": job["wall_seconds"], "depends_on": job["depends_on"]})
    s.require(len(jobs) == 33 and {j["id"] for j in jobs[2:]} == {j["id"] for j in ordered}, "Recovery job scope expanded")
    change_path = WORK / "preparation/recovery_plan_bindings.json"
    s.save(change_path, {"schema": "research4-b-validation-recovery-plan-bindings-v5", "changes": changes,
        "old_success_ids_reused": sorted(state["results"]), "new_job_ids_in_order": [j["id"] for j in jobs]}, exclusive=True)
    inputs.append(s.ref(change_path))
    prefix = WORK.name + "/"
    core_arrays = ("gradient.npy", "gradient_grid_response_true.npy", "direction.npy", "geometry_bohr.npy", "predicted_Hq.npy")
    allowlist = [prefix + "*.json", prefix + "*.jsonl", prefix + "detached-*.console.log",
        *[prefix + "preparation/**/*" + extension for extension in (".json", ".csv", ".txt", ".md")],
        prefix + "requests/*.json", *[prefix + "science/**/*" + extension for extension in (".json", ".jsonl", ".csv", ".log", ".txt")],
        *[prefix + "science/**/" + name for name in core_arrays], prefix + "execution/*.json", prefix + "execution/*.log"]
    spec = {"schema": SPEC_SCHEMA, "reserve_cpu_seconds": 300, "source_refs": unique_refs(sources),
        "input_refs": unique_refs(inputs), "authorization_ref": s.ref(grant_path), "predecessor_bindings": bindings,
        "jobs": jobs, "upload": {"enabled": True, "allowlist": allowlist},
        "local_only_artifacts": ["*.chk", "density.npy", "grids_coords.npy", "grids_weights.npy", "grid_non0tab.npy"],
        "local_only_policy": "Keep all original artifacts intact with source/result/seal hashes; only the finite small return whitelist is uploaded.",
        "new_quantum_evaluations_at_preparation": 0, "hourly_automation_created": False}
    s.save(target, spec, exclusive=True)
    return {"status": "SPEC_WRITTEN_NOT_RUN_NO_MANIFEST", "spec": s.ref(target), "jobs": len(jobs),
        "inherited_successes": len(bindings["inherited_results"]), "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in jobs),
        "remaining_cpu_seconds": s.CAP - carry["budget_charged_cpu_seconds"], "new_quantum_evaluations": 0, "git_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec"); parser.add_argument("--spec-sha256"); parser.add_argument("--build-reviewed-spec", action="store_true")
    args = parser.parse_args()
    if args.build_reviewed_spec:
        if args.spec or args.spec_sha256: parser.error("Use build mode or pinned spec preparation")
        result = build_reviewed_spec()
    else:
        if not args.spec or not args.spec_sha256: parser.error("Pinned --spec and --spec-sha256 are required")
        result = prepare(args.spec, args.spec_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
