"""Prepare v6 after an explicit cumulative90,000 CPU-second authorization.

This script never runs science or Git. --build-reviewed-spec creates the
finite20-point/6-aggregate proposal; pinned --spec preparation creates only
reviewable metadata. Original history, failed work and allocations stay intact.
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
PREVIOUS = BASE / "followup-validation-recovery-v5"
WORK = BASE / "followup-validation-recovery-v6"
SUPERVISOR = ROOT / "scripts/research4_b_validation_supervisor_recovery_v6_20261107.py"
WORKER = ROOT / "scripts/research4_b_validation_recovery_worker_v6_20261107.py"
HELPER = ROOT / "scripts/research4_parallel_b_git_return_20261106.py"
PRIOR_MANIFEST = PREVIOUS / "manifest_ROOT_AUTHORIZED.json"
CLOSURE_REVIEW = WORK / "preparation/v5_closure_remote_review.json"
EXTENSION_AUTHORIZATION = WORK / "preparation/extension_authorization.json"
EXTENSION_SHA256 = "9a12364fa757992654c35e83d77445153bb90350e345d76ba5e4be7737e6b4b9"
AH_SCIENCE_REVIEW_SHA256 = "5072df1c263d9c33747c50243bfbe5beb25f148cadd166b6754a60de481cf89f"
AH_TOY_PROVENANCE_SHA256 = "2750298293079a6bcec089e05eb9b181768e599d5723863a13591592c8bc15fd"
ALLOCATION = ROOT / ".runtime/worker-b-validation-recovery-v6-allocation.json"
EXTENSION_ALLOCATION = ROOT / ".runtime/worker-b-cpu-extension-20261107.json"
SPEC_SCHEMA = "research4-b-validation-recovery-spec-v6"
SMOKE = "acid-g3-fixed-mode01-h0010-minus"
EFFECTIVE_CAP = 90000
ORIGINAL_CAP = 58800
ADDITIONAL_CAP = EFFECTIVE_CAP - ORIGINAL_CAP
POINT_CAPS = {("acid", 3): (600, 1200), ("acid", 5): (900, 1800), ("caly", 3): (2400, 3600)}


def load_supervisor():
    spec = importlib.util.spec_from_file_location("r4_b_validation_recovery_v6_prepare", SUPERVISOR)
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
        "predecessor_supervisor": s.ref(ROOT / "scripts/research4_b_validation_supervisor_recovery_20261107.py"),
        "predecessor_allocation": s.ref(ROOT / ".runtime/worker-b-validation-recovery-allocation.json"),
        "predecessor_closure_review": s.ref(CLOSURE_REVIEW), "inherited_results": state["results"],
        "extension_authorization": s.ref(EXTENSION_AUTHORIZATION),
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def checked_extension(s, prior):
    grant = s.read(EXTENSION_AUTHORIZATION, EXTENSION_SHA256)
    s.require(grant.get("schema") == "research4-b-cpu-extension-authorization-v1"
        and grant.get("run_id") == s.RUN_ID and grant.get("approved") is True and grant.get("single_use") is True
        and grant.get("original_cpu_cap_seconds") == ORIGINAL_CAP and grant.get("additional_cpu_seconds") == ADDITIONAL_CAP
        and grant.get("effective_cpu_cap_seconds") == EFFECTIVE_CAP and grant.get("user_specified_numeric_total") == EFFECTIVE_CAP
        and grant.get("deadline_utc") == prior["deadline_utc"], "Exact single-use cumulative90,000 CPU-second grant required")
    s.require(prior["cpu_cap_seconds"] == ORIGINAL_CAP and s.CAP == EFFECTIVE_CAP,
        "Frozen predecessor cap stays58,800; only new v6 cap is90,000")
    return grant


def prepare(spec_path, spec_sha256):
    s = load_supervisor(); spec_path = s.path(spec_path); spec = s.read(spec_path, spec_sha256)
    s.require(spec.get("schema") == SPEC_SCHEMA and spec.get("jobs"), "Reviewed finite v6 spec required")
    s.require(not any((WORK / name).exists() for name in ("manifest_ROOT_AUTHORIZED.json", "return_manifest.json", "cpu_budget.json", "state.json"))
        and not ALLOCATION.exists() and not EXTENSION_ALLOCATION.exists(),
        "V6 preparation or the single-use extension claim already exists; never reset it")
    prior = s.read(PRIOR_MANIFEST); state = s.read(PREVIOUS / "state.json"); checked_extension(s, prior)
    bindings = predecessor_bindings(s, state)
    s.require(spec["predecessor_bindings"] == bindings, "Predecessor, extension grant or boot changed after spec preparation")
    manifest = {key: prior[key] for key in ("run_id", "branch", "python_executable", "deadline_utc", "threads", "memory_gib", "rss_stop_gib")}
    manifest.update(schema=s.MANIFEST_SCHEMA, cpu_cap_seconds=EFFECTIVE_CAP, work_directory=str(WORK.relative_to(ROOT)),
        ledger=str((WORK / "cpu_budget.json").relative_to(ROOT)), reserve_cpu_seconds=spec["reserve_cpu_seconds"],
        jobs=spec["jobs"], historical_costs_preserved_without_refund=True, **bindings)
    manifest["smoke_job_id"] = spec["smoke_job_id"]
    manifest["fail_fast_quantum_job_ids"] = spec["fail_fast_quantum_job_ids"]
    s.require(manifest["reserve_cpu_seconds"] >= 300, "Retain explicit300 CPU-second reserve")
    carry = s.validate_carry(manifest)
    s.require(datetime.now(timezone.utc) < datetime.fromisoformat(manifest["deadline_utc"]), "Original deadline expired")
    refs = unique_refs([s.ref(SUPERVISOR), s.ref(__file__), *spec["source_refs"]])
    inputs = unique_refs([s.ref(spec_path), *spec["input_refs"]]); manifest["authorization_ref"] = spec["authorization_ref"]
    for binding in refs + inputs + [manifest["authorization_ref"]]: s.checked(binding)
    manifest["source_refs"] = refs; manifest["input_refs"] = inputs
    s.validate(manifest, "0" * 64)
    return_path = WORK / "return_manifest.json"; upload = spec.get("upload", {"enabled": True})
    if upload.get("enabled"):
        allowlist = upload["allowlist"]; prefix = WORK.name + "/"
        s.require(isinstance(allowlist, list) and allowlist and all(isinstance(p, str) and p.startswith(prefix)
            and ".." not in Path(p).parts for p in allowlist), "Return allowlist must remain within v6")
        returning = {"schema": "research4-b-validation-recovery-return-manifest-v6", "role": "B", "run_id": s.RUN_ID,
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
        "effective_cpu_cap_seconds": EFFECTIVE_CAP, "one_time_additional_cpu_seconds": ADDITIONAL_CAP,
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": EFFECTIVE_CAP - carry["budget_charged_cpu_seconds"],
        "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in manifest["jobs"]), "new_quantum_evaluations": 0, "git_mutations": 0}


def recovery_job_order(prior_jobs, state):
    """Use the real first point as smoke, then its adjacent partner and pairs."""
    pending = [j for j in prior_jobs if state["jobs"][j["id"]]["status"] != "DONE_SEALED"]
    signed = [j for j in pending if j["id"].endswith(("-minus", "-plus"))]
    aggregate = [j for j in pending if j not in signed]
    if len(pending) != 26 or len(signed) != 20 or len(aggregate) != 6 or signed[0]["id"] != SMOKE:
        raise ValueError("Expected exact v5 remainder:20 signed points beginning with the real smoke, plus6 aggregates")
    groups = [j.get("admission_group") for j in signed]
    if any(groups[i] is None or groups[i] != groups[i + 1] for i in range(0, 20, 2)) or len(set(groups)) != 10:
        raise ValueError("All ten signed pairs must remain adjacent")
    return [*signed, *aggregate]


def fail_fast_dependencies(ordered_jobs):
    """Keep original center/point dependencies and append explicit Q gates."""
    planned = {}; previous_quantum = None
    for job in ordered_jobs:
        name = job["id"]; dependencies = list(job["depends_on"])
        if name.endswith(("-minus", "-plus")):
            if name != SMOKE: dependencies.append(SMOKE)
            if previous_quantum: dependencies.append(previous_quantum)
            previous_quantum = name
        else:
            dependencies.append(SMOKE)
        planned[name] = list(dict.fromkeys(dependencies))
    return planned


def old_request(s, job):
    argv = job["argv"]; p = s.path(argv[argv.index("--request") + 1]); expected = argv[argv.index("--request-sha256") + 1]
    return p, s.read(p, expected)


def failed_checkpoint_inputs(s, job_id, ledger):
    rows = [r for r in ledger["phases"] if r["job_id"] == job_id]
    s.require(len(rows) == 1 and rows[0]["status"] == "FAILED_CLOSED", "Expected one trusted old failed signed-point closure: " + job_id)
    row = rows[0]; result = s.path(row["output_directory"]) / "result.json"
    s.require(s.sha(result) == row["result_sha256"], "Closed failed result changed")
    old_result = s.read(result)
    progress = [binding for binding in old_result.get("output_artifacts", []) if Path(binding["path"]).name == "progress.json"]
    s.require(len(progress) == 1, "One historical progress snapshot required for target checkpoint provenance")
    s.checked(progress[0]); checkpoint = result.parent / "fresh_point_scf.chk"
    s.require(checkpoint.is_file() and not checkpoint.is_symlink(), "Failed target checkpoint missing")
    return {"warmstart_result": s.ref(result), "warmstart_checkpoint": s.ref(checkpoint), "last_progress": progress[0],
        "closed_phase": row["phase"], "wait4_receipt": row["wait4_receipt"],
        "checkpoint_hash_scope": "First explicitly pinned for v6 from the preserved closed v5 target output; the old partial result did not seal a checkpoint hash."}


def build_reviewed_spec():
    s = load_supervisor(); prior = s.read(PRIOR_MANIFEST); state = s.read(PREVIOUS / "state.json")
    ledger = s.read(PREVIOUS / "cpu_budget.json"); checked_extension(s, prior); bindings = predecessor_bindings(s, state)
    carry = s.validate_carry(bindings)
    target = WORK / "reviewed_spec_ROOT_AUTHORIZED.json"
    s.require(not target.exists(), "Preserve an existing v6 reviewed spec")
    s.require(len(state["results"]) == 50, "Exactly50 old sealed results must be inherited without recomputation")
    ordered = recovery_job_order(prior["jobs"], state); old_counts = Counter(r["job_id"] for r in ledger["phases"])
    acid_ids = [j["id"] for j in ordered if j["id"].startswith("acid-") and j["id"].endswith(("-minus", "-plus"))]
    s.require(len(acid_ids) == 16, "Exactly16 failed acid target checkpoints required")
    checkpoints = {name: failed_checkpoint_inputs(s, name, ledger) for name in acid_ids}
    inputs = [*state["results"].values(), *[bindings[key] for key in ("predecessor_manifest", "predecessor_ledger", "predecessor_state",
        "predecessor_supervisor", "predecessor_allocation", "predecessor_closure_review", "extension_authorization")]]
    science_review = WORK / "preparation/independent_AH_science_review.json"
    s.read(science_review, AH_SCIENCE_REVIEW_SHA256); inputs.append(s.ref(science_review))
    toy_provenance = WORK / "preparation/ah_solver_diagnostics/source_refs.json"
    toy_bindings = s.read(toy_provenance, AH_TOY_PROVENANCE_SHA256); inputs.append(s.ref(toy_provenance))
    s.require(len(toy_bindings.get("copies", [])) == 2, "Exactly two preserved AH toy diagnostic artifacts required")
    for copied in toy_bindings["copies"]:
        original = copied["original"]; exact_copy = copied["exact_byte_copy"]
        original_path = s.checked(original); copied_path = s.checked(exact_copy)
        s.require(original["sha256"] == exact_copy["sha256"] and original["bytes"] == exact_copy["bytes"],
            "AH diagnostic copy must preserve exact original bytes")
        s.require(original_path.stat().st_size == original["bytes"] and copied_path.stat().st_size == exact_copy["bytes"],
            "AH diagnostic provenance file sizes differ")
        inputs.extend([original, exact_copy])
    # Add additive root/independent preparation documents that exist before
    # production generation; their hashes become part of the frozen protocol.
    for p in sorted((WORK / "preparation").glob("*.json")): inputs.append(s.ref(p))
    for values in checkpoints.values():
        inputs.extend([values["warmstart_result"], values["warmstart_checkpoint"], values["last_progress"], values["wait4_receipt"]])
    checkpoint_manifest = WORK / "preparation/checkpoint_input_bindings.json"
    s.save(checkpoint_manifest, {"schema": "research4-b-v6-target-checkpoint-input-bindings-v1", "inputs": checkpoints}, exclusive=True)
    inputs.append(s.ref(checkpoint_manifest))
    grant_path = WORK / "preparation/user_authorization_validation_recovery_v6.json"
    grant = {"schema": "research4-b-validation-recovery-user-authorization-v6", "recorded_at_utc": s.utc(),
        "user_instructions_verbatim": ["授权你更多时间，恢复一直继续跑", "没运行恢复自动运行", "cpu秒给90000秒，不要中断一直运行后续任务"],
        "interpretation": "The numeric instruction sets one new cumulative90,000 CPU-second cap, not an additional90,000 seconds. Preserve all actual costs and the1,000-second permanent nonmeasured debit; the old deadline remains unchanged.",
        "original_frozen_cpu_cap_seconds": ORIGINAL_CAP, "single_use_additional_cpu_seconds": ADDITIONAL_CAP,
        "effective_cpu_cap_seconds": EFFECTIVE_CAP, "extension_authorization": s.ref(EXTENSION_AUTHORIZATION),
        "deadline_utc": prior["deadline_utc"], "recovery_boot_id": bindings["recovery_boot_id"],
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": EFFECTIVE_CAP - carry["budget_charged_cpu_seconds"], "reserve_cpu_seconds": 300,
        "old_successes_reused_without_recomputation": 50, "old_failed_points_retained_with_all_costs": 16,
        "predeclared_jobs": {"signed_points_including_one_real_smoke": 20, "aggregates": 6},
        "smoke_job_id": SMOKE, "smoke_is_not_a_duplicate": True,
        "gate": "Every subsequent quantum point depends on smoke success and on the preceding quantum point. Keep both signs adjacent and reserve both unfulfilled caps plus300 before admission/retry.",
        "solver_controls": {"newton_AH_inner_conv_tol": 1e-22, "old_v5_newton_AH_inner_conv_tol": 1e-12,
            "newton_AH_lindep": 1e-24, "SCF_energy_tolerance": 1e-12, "SCF_orbital_gradient_tolerance": 1e-9,
            "fresh_Fock_orbital_gradient_norm_qualification": 1e-8},
        "scientific_method_settings": "Unchanged from qualified v5 centers and v5 point requests; only the solver's AH inner tolerance implementation changes. Fixed points use the sealed v5 center grid and energy.",
        "warmstart": "Sixteen acid points use their same-target preserved v5 failed result/checkpoint/progress; four untouched Calycosin points use the sealed v5 center checkpoint. MO data is only an initial guess.",
        "attempt_policy": "Acid: one remaining cross-version attempt. Calycosin: at most two; no unknown closure restart or scientific-negative-as-success.",
        "predecessor_closure_review": s.ref(CLOSURE_REVIEW), "hourly_automation_created": False,
        "preservation": "All earlier sources, results, state, ledgers, allocations, failures and checkpoints remain unchanged."}
    s.save(grant_path, grant, exclusive=True); inputs.append(s.ref(grant_path))
    sources = [s.ref(WORKER), s.ref(SUPERVISOR), s.ref(__file__)]; snapshots = []
    for source in (WORKER, SUPERVISOR, Path(__file__).resolve(),
        ROOT / "scripts/test_research4_b_validation_supervisor_recovery_v6_20261107.py",
        ROOT / "scripts/test_research4_b_validation_recovery_worker_v6_20261107.py"):
        s.require(source.is_file(), "Reviewed v6 source/test absent: " + str(source))
        dest = WORK / "preparation/source_snapshots" / (source.name + ".txt"); dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle: handle.write(source.read_bytes())
        sources.append(s.ref(source)); inputs.append(s.ref(dest)); snapshots.append({"source": s.ref(source), "exact_byte_snapshot": s.ref(dest)})
    snapshot_manifest = WORK / "preparation/source_snapshots/source_refs.json"
    s.save(snapshot_manifest, {"files": snapshots}, exclusive=True); inputs.append(s.ref(snapshot_manifest))
    jobs = []; changes = []; dependency_plan = fail_fast_dependencies(ordered)
    python = str(s.python_path(prior["python_executable"]))
    for old_job in ordered:
        name = old_job["id"]; old_p, request = old_request(s, old_job); inputs.append(s.ref(old_p))
        request = dict(request); job = json.loads(json.dumps(old_job)); is_point = name.endswith(("-minus", "-plus"))
        s.require(request["operation"] in (("point", "fixed_grid_point") if is_point else ("aggregate",)), "Unexpected recovery operation")
        if is_point:
            s.require(request["scf_conv_tol"] == 1e-12 and request["scf_conv_tol_grad"] == 1e-9
                and request["scf_qualification_gradient_norm"] == 1e-8 and request["direct_scf_tol"] == 1e-15
                and request["small_rho_cutoff"] == 0, "Qualified v5 scientific settings changed")
            center_ids = [key for key in old_job["depends_on"] if key.endswith("-center")]
            s.require(len(center_ids) == 1 and center_ids[0] in state["results"], "One inherited qualified v5 center required")
            center = center_ids[0]; center_ref = state["results"][center]; s.checked(center_ref)
            if name in checkpoints:
                request.update(warmstart_result=checkpoints[name]["warmstart_result"], warmstart_checkpoint=checkpoints[name]["warmstart_checkpoint"])
            else:
                request.pop("warmstart_result", None); request.pop("warmstart_checkpoint", None)
            job["depends_on"] = dependency_plan[name]
            family = "acid" if name.startswith("acid-") else "caly"
            job["cpu_seconds"], job["wall_seconds"] = POINT_CAPS[(family, request["grid_level"])]
        else:
            job["depends_on"] = dependency_plan[name]
        p = WORK / "requests" / (name + ".json"); s.save(p, request, exclusive=True); binding = s.ref(p); inputs.append(binding)
        job["argv"] = [python, str(WORKER), "--request", str(p), "--request-sha256", binding["sha256"], "--output", "{output}", "--execute"]
        if is_point:
            if request["operation"] == "fixed_grid_point":
                job["argv"].extend(["--center-result", str(s.path(center_ref["path"])), "--center-result-sha256", center_ref["sha256"]])
            if name not in checkpoints:
                job["argv"].extend(["--warmstart-result", str(s.path(center_ref["path"])), "--warmstart-result-sha256", center_ref["sha256"]])
        else:
            old_argv = old_job["argv"]; job["argv"].extend(old_argv[old_argv.index("--execute") + 1:])
        remaining_attempts = min(2, old_job.get("max_attempts", 1) - old_counts[name])
        s.require(remaining_attempts >= 1 and (name not in checkpoints or remaining_attempts == 1), "Historical attempt reset: " + name)
        job["max_attempts"] = remaining_attempts
        if name == SMOKE: job["smoke_success_gate"] = True
        jobs.append(job)
        changes.append({"job_id": name, "old_request": s.ref(old_p), "new_request": binding,
            "historical_phase_count": old_counts[name], "remaining_max_attempts": remaining_attempts,
            "cpu_seconds": job["cpu_seconds"], "wall_seconds": job["wall_seconds"], "depends_on": job["depends_on"],
            "warmstart_source": "SAME_TARGET_FAILED_V5_CHECKPOINT" if name in checkpoints else "QUALIFIED_V5_CENTER" if is_point else "INHERITED_AND_NEW_POINT_REFS"})
    s.require(len(jobs) == 26 and jobs[0]["id"] == SMOKE and len({j["id"] for j in jobs}) == 26, "Exact finite v6 scope required")
    change_path = WORK / "preparation/recovery_plan_bindings.json"
    s.save(change_path, {"schema": "research4-b-validation-recovery-plan-bindings-v6", "changes": changes,
        "old_success_ids_reused": sorted(state["results"]), "job_ids_in_order": [j["id"] for j in jobs]}, exclusive=True)
    inputs.append(s.ref(change_path)); prefix = WORK.name + "/"
    core_arrays = ("gradient.npy", "gradient_grid_response_true.npy", "direction.npy", "geometry_bohr.npy", "predicted_Hq.npy")
    allowlist = [prefix + "*.json", prefix + "*.jsonl", prefix + "detached-*.console.log",
        *[prefix + "preparation/**/*" + extension for extension in (".json", ".csv", ".txt", ".md")], prefix + "requests/*.json",
        *[prefix + "science/**/*" + extension for extension in (".json", ".jsonl", ".csv", ".log", ".txt")],
        *[prefix + "science/**/" + name for name in core_arrays], prefix + "execution/*.json", prefix + "execution/*.log"]
    spec = {"schema": SPEC_SCHEMA, "reserve_cpu_seconds": 300, "source_refs": unique_refs(sources), "input_refs": unique_refs(inputs),
        "authorization_ref": s.ref(grant_path), "predecessor_bindings": bindings, "jobs": jobs,
        "smoke_job_id": SMOKE, "fail_fast_quantum_job_ids": [j["id"] for j in jobs if j["id"].endswith(("-minus", "-plus"))],
        "upload": {"enabled": True, "allowlist": allowlist}, "effective_cpu_cap_seconds": EFFECTIVE_CAP,
        "single_use_additional_cpu_seconds": ADDITIONAL_CAP,
        "local_only_artifacts": ["*.chk", "density.npy", "grids_coords.npy", "grids_weights.npy", "grid_non0tab.npy"],
        "local_only_policy": "All science remains intact locally with recorded hashes; only finite core gradients/directions/Hq and small metadata/logs are returned.",
        "new_quantum_evaluations_at_preparation": 0, "hourly_automation_created": False}
    s.save(target, spec, exclusive=True)
    return {"status": "SPEC_WRITTEN_NOT_RUN_NO_MANIFEST", "spec": s.ref(target), "jobs": len(jobs),
        "smoke_job_id": SMOKE, "inherited_successes": len(bindings["inherited_results"]),
        "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in jobs),
        "effective_cpu_cap_seconds": EFFECTIVE_CAP, "remaining_cpu_seconds": EFFECTIVE_CAP - carry["budget_charged_cpu_seconds"],
        "new_quantum_evaluations": 0, "git_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec"); parser.add_argument("--spec-sha256"); parser.add_argument("--build-reviewed-spec", action="store_true")
    args = parser.parse_args()
    if args.build_reviewed_spec:
        if args.spec or args.spec_sha256: parser.error("Use build mode or pinned spec preparation")
        result = build_reviewed_spec()
    else:
        if not args.spec or not args.spec_sha256: parser.error("Pinned --spec and --spec-sha256 required")
        result = prepare(args.spec, args.spec_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
