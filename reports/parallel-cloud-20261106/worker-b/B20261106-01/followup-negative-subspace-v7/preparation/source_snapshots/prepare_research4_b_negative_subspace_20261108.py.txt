"""Prepare a finite v7 negative-subspace investigation without running it.

--plan-only prints the31-job inventory without creating files. Reviewed
jobs/requests are supplied through a pinned spec to prepare a manifest. The
existing90,000 CPU-second authorization is historical; this round adds no CPU,
does not write an extension claim, and preserves all earlier artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/parallel-cloud-20261106/worker-b/B20261106-01"
PREVIOUS = BASE / "followup-validation-recovery-v6"
WORK = BASE / "followup-negative-subspace-v7"
SUPERVISOR = ROOT / "scripts/research4_b_negative_subspace_supervisor_20261108.py"
WORKER = ROOT / "scripts/research4_b_negative_subspace_worker_20261108.py"
HELPER = ROOT / "scripts/research4_parallel_b_git_return_20261106.py"
PRIOR_MANIFEST = PREVIOUS / "manifest_ROOT_AUTHORIZED.json"
CLOSURE_REVIEW = WORK / "preparation/v6_closure_completion_review.json"
CLOSURE_REVIEW_SHA256 = "d8e39c3f7a3920595be142ffbd0fb2a3b66e4119eeaea292fc47530b004b18ba"
CLOSURE_PROVENANCE = WORK / "preparation/v6_closure_evidence_source_refs.json"
CLOSURE_PROVENANCE_SHA256 = "4b8e8f02bb6935d5d5a6ee39b74eb5bec7c47178589b54e3d18e12136f2fba82"
ROOT_PROTOCOL = WORK / "preparation/research_questions_ROOT_AUTHORIZED.json"
ROOT_PROTOCOL_SHA256 = "98adefbfcc3b84f9baa56e2e34e4fa754e0d44d7743511b588a5d408cc9d2bf0"
MODE1_COUPLING_EVIDENCE = WORK / "preparation/existing_mode1_coupling_evidence.json"
MODE1_COUPLING_SHA256 = "94a6aa173c2d452a9f90d31d1e9daae2021a30838bf257880da5a1ff9b382348"
EXTENSION_AUTHORIZATION = PREVIOUS / "preparation/extension_authorization.json"
EXTENSION_SHA256 = "9a12364fa757992654c35e83d77445153bb90350e345d76ba5e4be7737e6b4b9"
HISTORICAL_EXTENSION_CLAIM = ROOT / ".runtime/worker-b-cpu-extension-20261107.json"
ALLOCATION = ROOT / ".runtime/worker-b-negative-subspace-allocation.json"
SPEC_SCHEMA = "research4-b-negative-subspace-spec-v7"
CAP = 90000
RESERVE = 300
BASE_CASES = (("acid", "protocatechuic_acid", (2, 3, 4)), ("caly", "calycosin", (2, 3, 4, 5, 6, 7)))
SMOKE = "acid-g3-negative-mode02-h0010-minus"
POINT_CAPS = {"protocatechuic_acid": (600, 1200), "calycosin": (2400, 3600)}


def load_supervisor():
    spec = importlib.util.spec_from_file_location("r4_b_negative_subspace_prepare", SUPERVISOR)
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


def base_point_plan():
    """Return the fixed scientific base inventory, with adjacent signed pairs."""
    planned = []
    for label, case, modes in BASE_CASES:
        center_id = label + "-g3-regrid-center"
        for mode in modes:
            pair = label + "-g3-negative-mode%02d-h0010" % mode
            for sign in (-1, 1):
                planned.append({"id": pair + ("-minus" if sign < 0 else "-plus"),
                    "case": case, "operation": "point", "direction": {"type": "mode", "index": mode},
                    "grid_level": 3, "step_bohr": .001, "sign": sign, "gradient_response_controls": [False, True],
                    "inherited_center_id": center_id, "admission_group": pair, "max_attempts": 2})
    return planned


def negative_subspace_plan():
    """Return the root-reviewed31-job order and its exact data dependencies."""
    base = base_point_plan(); planned = []; all_new_points = []; all_old_points = []
    for label, case, new_modes in BASE_CASES:
        center = label + "-g3-regrid-center"
        old_mode1 = [label + "-g3-regrid-mode01-h0010-" + sign for sign in ("minus", "plus")]
        case_base = [dict(point) for point in base if point["case"] == case]
        for point in case_base:
            point["kind"] = "basis_point"; point["depends_on"] = [center]
            all_new_points.append(point["id"]); planned.append(point)
        subspace = label + "-g3-negative-subspace"
        basis_refs = [*old_mode1, *[point["id"] for point in case_base]]
        planned.append({"id": subspace, "case": case, "kind": "subspace_aggregate", "operation": "subspace_aggregate",
            "mode_indices": [1, *new_modes], "step_bohr": .001, "inherited_center_id": center,
            "point_ids": basis_refs, "depends_on": [center, *basis_refs], "max_attempts": 2})
        combo_points = []
        for h in (.001, .0005):
            pair = label + "-g3-negative-combo-minimum-h%04d" % round(h * 10000)
            for sign in (-1, 1):
                name = pair + ("-minus" if sign < 0 else "-plus")
                dependencies = [center, subspace]
                point = {"id": name, "case": case, "kind": "combination_point", "operation": "point",
                    "direction": {"type": "subspace_minimum", "index": 1}, "grid_level": 3,
                    "step_bohr": h, "sign": sign, "gradient_response_controls": [False, True],
                    "inherited_center_id": center, "subspace_job_id": subspace, "admission_group": pair,
                    "depends_on": list(dict.fromkeys(dependencies)), "max_attempts": 2}
                combo_points.append(name); all_new_points.append(name); planned.append(point)
        planned.append({"id": label + "-g3-negative-combo-comparison", "case": case, "kind": "combo_aggregate",
            "operation": "aggregate", "point_ids": [center, *combo_points],
            "depends_on": [center, subspace, *combo_points], "max_attempts": 2})
        all_old_points.extend([center, *[label + "-g3-regrid-mode01-" + step + "-" + sign
            for step in ("h0010", "h0005") for sign in ("minus", "plus")]])
    final_points = [*all_old_points, *all_new_points]
    planned.append({"id": "all-negative-subspace-comparison", "kind": "final_aggregate", "operation": "aggregate",
        "point_ids": final_points, "depends_on": final_points, "max_attempts": 2})
    def select(case, kind, h=None):
        return [item for item in planned if item.get("case") == case and item["kind"] == kind
            and (h is None or item.get("step_bohr") == h)]
    acid, caly = "protocatechuic_acid", "calycosin"
    ordered = [*select(acid, "basis_point"), *select(acid, "subspace_aggregate"),
        *select(acid, "combination_point", .001), *select(caly, "basis_point"), *select(caly, "subspace_aggregate"),
        *select(acid, "combination_point", .0005), *select(acid, "combo_aggregate"),
        *select(caly, "combination_point"), *select(caly, "combo_aggregate"), planned[-1]]
    # Rebuild the independent group chains after the reviewed interleaving. Physical
    # dependencies remain the matching center and, for candidate points, the
    # corresponding sealed subspace selection.
    previous_by_group = {}
    for item in ordered:
        if item["operation"] != "point": continue
        label = "acid" if item["case"] == acid else "caly"
        group = label + ("-baseline" if item["kind"] == "basis_point" else "-combo")
        dependencies = [item["inherited_center_id"]]
        if item.get("subspace_job_id"): dependencies.append(item["subspace_job_id"])
        if group in previous_by_group: dependencies.append(previous_by_group[group])
        item["depends_on"] = list(dict.fromkeys(dependencies)); item["fail_fast_group"] = group
        previous_by_group[group] = item["id"]
    return ordered


def quantum_groups(planned):
    groups = {}
    for item in planned:
        if item["operation"] == "point": groups.setdefault(item["fail_fast_group"], []).append(item["id"])
    return groups


def checked_historical_extension(s, prior):
    """Read the already-used authorization and claim; never create either."""
    grant = s.read(EXTENSION_AUTHORIZATION, EXTENSION_SHA256)
    claim = s.read(HISTORICAL_EXTENSION_CLAIM)
    s.require(prior["cpu_cap_seconds"] == CAP == s.CAP
        and grant.get("schema") == "research4-b-cpu-extension-authorization-v1"
        and grant.get("approved") is True and grant.get("single_use") is True
        and grant.get("effective_cpu_cap_seconds") == CAP and grant.get("additional_cpu_seconds") == 31200
        and grant.get("original_cpu_cap_seconds") == 58800 and grant.get("deadline_utc") == prior["deadline_utc"],
        "Retain the existing historical90,000 CPU-second ceiling and deadline")
    s.require(claim.get("schema") == "research4-b-single-use-cpu-extension-claim-v1"
        and claim.get("run_id") == s.RUN_ID and claim.get("grant_id") == grant.get("grant_id")
        and claim.get("extension_authorization", {}).get("sha256") == EXTENSION_SHA256
        and s.checked(claim["extension_authorization"]) == s.path(EXTENSION_AUTHORIZATION)
        and claim.get("manifest_sha256") == s.sha(PRIOR_MANIFEST)
        and claim.get("effective_cpu_cap_seconds") == CAP
        and claim.get("additional_cpu_authorized_seconds") == 31200
        and s.path(claim["ledger"]) == PREVIOUS / "cpu_budget.json",
        "The prior one-time extension must already be claimed by the frozen v6 ledger")
    return grant, claim


def predecessor_bindings(s, state):
    return {"predecessor_manifest": s.ref(PRIOR_MANIFEST), "predecessor_ledger": s.ref(PREVIOUS / "cpu_budget.json"),
        "predecessor_state": s.ref(PREVIOUS / "state.json"),
        "predecessor_supervisor": s.ref(ROOT / "scripts/research4_b_validation_supervisor_recovery_v6_20261107.py"),
        "predecessor_allocation": s.ref(ROOT / ".runtime/worker-b-validation-recovery-v6-allocation.json"),
        "predecessor_closure_review": s.ref(CLOSURE_REVIEW), "historical_extension_claim": s.ref(HISTORICAL_EXTENSION_CLAIM),
        "extension_authorization": s.ref(EXTENSION_AUTHORIZATION), "inherited_results": state["results"],
        "recovery_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def prepare(spec_path, spec_sha256):
    s = load_supervisor(); spec_path = s.path(spec_path); spec = s.read(spec_path, spec_sha256)
    s.require(spec.get("schema") == SPEC_SCHEMA and isinstance(spec.get("jobs"), list) and spec["jobs"],
        "Pinned reviewed finite v7 spec required")
    s.require(not any((WORK / name).exists() for name in ("manifest_ROOT_AUTHORIZED.json", "return_manifest.json", "cpu_budget.json", "state.json"))
        and not ALLOCATION.exists(), "V7 preparation/allocation already exists; preserve it instead of resetting")
    prior = s.read(PRIOR_MANIFEST); state = s.read(PREVIOUS / "state.json"); checked_historical_extension(s, prior)
    s.require(len(state["results"]) == 76, "Exactly76 historical successes must be inherited")
    bindings = predecessor_bindings(s, state)
    s.require(spec["predecessor_bindings"] == bindings, "Historical inputs or current boot changed after spec review")
    manifest = {key: prior[key] for key in ("run_id", "branch", "python_executable", "deadline_utc", "threads", "memory_gib", "rss_stop_gib")}
    manifest.update(schema=s.MANIFEST_SCHEMA, cpu_cap_seconds=CAP, work_directory=str(WORK.relative_to(ROOT)),
        ledger=str((WORK / "cpu_budget.json").relative_to(ROOT)), reserve_cpu_seconds=spec.get("reserve_cpu_seconds", RESERVE),
        jobs=spec["jobs"], additional_cpu_seconds_this_round=0, historical_costs_preserved_without_refund=True, **bindings)
    for field in ("smoke_job_id", "fail_fast_quantum_groups"):
        if field in spec: manifest[field] = spec[field]
    s.require(manifest["reserve_cpu_seconds"] >= RESERVE, "Retain the explicit300 CPU-second reserve")
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
            and ".." not in Path(p).parts for p in allowlist), "Return allowlist must remain within v7")
        returning = {"schema": "research4-b-negative-subspace-return-manifest-v7", "role": "B", "run_id": s.RUN_ID,
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
        "cpu_cap_seconds": CAP, "additional_cpu_seconds_this_round": 0,
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": CAP - carry["budget_charged_cpu_seconds"],
        "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in manifest["jobs"]),
        "new_quantum_evaluations": 0, "git_mutations": 0, "historical_extension_claim_written": False}


def build_reviewed_spec():
    """Write root's finite31-job proposal after every component is frozen."""
    s = load_supervisor(); prior = s.read(PRIOR_MANIFEST); state = s.read(PREVIOUS / "state.json")
    checked_historical_extension(s, prior)
    s.read(ROOT_PROTOCOL, ROOT_PROTOCOL_SHA256)
    s.read(MODE1_COUPLING_EVIDENCE, MODE1_COUPLING_SHA256)
    s.read(CLOSURE_REVIEW, CLOSURE_REVIEW_SHA256)
    provenance = s.read(CLOSURE_PROVENANCE, CLOSURE_PROVENANCE_SHA256)
    original_review = s.checked(provenance["original"]); copied_review = s.checked(provenance["exact_byte_copy"])
    s.require(provenance["original"]["sha256"] == provenance["exact_byte_copy"]["sha256"] == CLOSURE_REVIEW_SHA256
        and copied_review == CLOSURE_REVIEW and original_review.stat().st_size == provenance["original"]["bytes"]
        and copied_review.stat().st_size == provenance["exact_byte_copy"]["bytes"], "Closed history copy provenance differs")
    bindings = predecessor_bindings(s, state); carry = s.validate_carry(bindings)
    target = WORK / "reviewed_spec_ROOT_AUTHORIZED.json"
    s.require(not target.exists() and not ALLOCATION.exists() and not (WORK / "cpu_budget.json").exists(),
        "Preserve an existing v7 proposal or allocation")
    s.require(len(state["results"]) == 76, "Exactly76 sealed inherited successes required")
    planned = negative_subspace_plan(); quantum = [p for p in planned if p["operation"] == "point"]
    s.require(len(planned) == 31 and len(quantum) == 26 and quantum[0]["id"] == SMOKE, "Exact root-reviewed v7 scope required")
    prior_by_id = {job["id"]: job for job in prior["jobs"]}
    inputs = [*state["results"].values(), s.ref(ROOT_PROTOCOL), s.ref(MODE1_COUPLING_EVIDENCE), s.ref(CLOSURE_PROVENANCE), provenance["original"], *[bindings[key] for key in ("predecessor_manifest", "predecessor_ledger", "predecessor_state",
        "predecessor_supervisor", "predecessor_allocation", "predecessor_closure_review", "historical_extension_claim", "extension_authorization")]]
    for p in sorted((WORK / "preparation").glob("*.json")): inputs.append(s.ref(p))
    base_requests = {}; center_inputs = {}
    method_keys = ("schema", "threads", "source_worker", "case", "frequency_result", "grid_level", "gradient_grid_response",
        "scf_conv_tol", "scf_conv_tol_grad", "scf_qualification_gradient_norm", "small_rho_cutoff", "grid_prune",
        "gradient_response_controls", "direct_scf_tol", "scf_solver", "warmstart_scf_max_cycle", "newton_max_cycle", "newton_ah_max_cycle")
    for label, case, _ in BASE_CASES:
        old_job = prior_by_id[label + "-g3-regrid-mode01-h0010-minus"]; argv = old_job["argv"]
        old_request_path = s.path(argv[argv.index("--request") + 1])
        request = s.read(old_request_path, argv[argv.index("--request-sha256") + 1]); inputs.append(s.ref(old_request_path))
        base = {key: request[key] for key in method_keys if key in request}
        s.require(base["case"] == case and base["grid_level"] == 3 and base["scf_conv_tol"] == 1e-12
            and base["scf_conv_tol_grad"] == 1e-9 and base["scf_qualification_gradient_norm"] == 1e-8
            and base["direct_scf_tol"] == 1e-15 and base["small_rho_cutoff"] == 0
            and base["gradient_response_controls"] == [False, True], "Frozen qualified v6 physical settings differ")
        base_requests[case] = base; inputs.extend([base["frequency_result"], base["source_worker"]])
        center_id = label + "-g3-regrid-center"; center_ref = state["results"][center_id]
        center_result = s.read(s.checked(center_ref))
        checkpoints = [binding for binding in center_result.get("checkpoint_files", []) if Path(binding["path"]).suffix == ".chk"]
        s.require(len(checkpoints) == 1 and center_result.get("scf_qualified") is True and center_result.get("scf_converged") is True,
            "One trusted qualified inherited center checkpoint required")
        s.checked(checkpoints[0]); inputs.extend([center_ref, checkpoints[0]])
        center_inputs[case] = {"id": center_id, "result": center_ref, "checkpoint": checkpoints[0]}
    center_manifest = WORK / "preparation/inherited_center_inputs.json"
    s.save(center_manifest, {"schema": "research4-b-v7-inherited-center-inputs-v1", "cases": center_inputs}, exclusive=True)
    inputs.append(s.ref(center_manifest))
    grant_path = WORK / "preparation/user_authorization_negative_subspace.json"
    grant = {"schema": "research4-b-negative-subspace-user-authorization-v7", "recorded_at_utc": s.utc(),
        "user_instruction": "进一步处理已有问题", "additional_cpu_seconds_this_round": 0,
        "interpretation": "Continue the finite investigation within the already-authorized cumulative90,000 CPU-second cap; no new extension claim, refund, or deadline extension.",
        "cpu_cap_seconds": CAP, "deadline_utc": prior["deadline_utc"], "recovery_boot_id": bindings["recovery_boot_id"],
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": CAP - carry["budget_charged_cpu_seconds"], "reserve_cpu_seconds": RESERVE,
        "old_successes_inherited_without_recomputation": 76,
        "new_jobs": {"basis_points": 18, "subspace_aggregates": 2, "combination_points": 8, "combination_comparisons": 2, "final_aggregate": 1},
        "scientific_scope": "Use original negative-mode directions:acid2..4/Calycosin2..7 at h0.001Bohr plus inherited mode1/centers. Build both False/True K and Cartesian Gram G, select the True Ksym/G minimum diagnostic candidate deterministically regardless of eigenvalue sign, and independently test it at h0.001/0.0005Bohr.",
        "limitations": "An unproven common True-grid point set or a large antisymmetric component remains a recorded limitation; a selected vector is a hypothesis for independent energy/gradient finite differences, not a certified energy-Hessian eigenmode or full-dimensional minimum.",
        "settings": "Reuse qualified v6 grid3 rebuilt-grid physical settings, both gradient-response controls and unchanged SCF/fresh-Fock qualification. New directions use the sealed matching center only as a warmstart guess.",
        "order": "Acid basis/subspace and h0.001 candidate pair; Calycosin basis/subspace; acid h0.0005 pair/comparison; Calycosin candidate pairs/comparison; final comparison.",
        "smoke_job_id": SMOKE, "independent_quantum_success_groups": quantum_groups(planned),
        "failure_policy": "At most two preserved attempts per new job. Scientific qualification failure remains negative/incomplete. Unknown closure never restarts; an insufficient complete-pair budget leaves that pair missing, never zero-filled.",
        "historical_extension_claim": bindings["historical_extension_claim"], "extension_authorization": bindings["extension_authorization"],
        "predecessor_closure_review": bindings["predecessor_closure_review"], "root_scientific_protocol": s.ref(ROOT_PROTOCOL), "hourly_automation_created": False,
        "preservation": "All previous sources, ledgers, markers, failures, science and checkpoint bytes remain unchanged."}
    s.save(grant_path, grant, exclusive=True); inputs.append(s.ref(grant_path))
    sources = [s.ref(WORKER), s.ref(SUPERVISOR), s.ref(__file__)]; snapshots = []
    for source in (WORKER, SUPERVISOR, Path(__file__).resolve(),
        ROOT / "scripts/test_research4_b_negative_subspace_supervisor_20261108.py",
        ROOT / "scripts/test_research4_b_negative_subspace_worker_20261108.py"):
        s.require(source.is_file(), "Reviewed v7 source/test absent: " + str(source))
        dest = WORK / "preparation/source_snapshots" / (source.name + ".txt"); dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle: handle.write(source.read_bytes())
        sources.append(s.ref(source)); inputs.append(s.ref(dest)); snapshots.append({"source": s.ref(source), "exact_byte_snapshot": s.ref(dest)})
    snapshot_manifest = WORK / "preparation/source_snapshots/source_refs.json"
    s.save(snapshot_manifest, {"files": snapshots}, exclusive=True); inputs.append(s.ref(snapshot_manifest))
    jobs = []; python = str(s.python_path(prior["python_executable"]))
    new_ids = {p["id"] for p in planned}
    def append_result(argv, flag, job_id):
        if job_id in new_ids:
            argv.extend(["--" + flag + "-result", "{result:" + job_id + "}", "--" + flag + "-result-sha256", "{sha256:" + job_id + "}"])
        else:
            binding = state["results"][job_id]; s.checked(binding); inputs.append(binding)
            argv.extend(["--" + flag + "-result", str(s.path(binding["path"])), "--" + flag + "-result-sha256", binding["sha256"]])
    for item in planned:
        case = item.get("case"); operation = item["operation"]
        if operation == "point":
            request = {**base_requests[case], "operation": "point", "direction": item["direction"], "step_bohr": item["step_bohr"], "sign": item["sign"]}
            cpu, wall = POINT_CAPS[case]; success = "POINT_COMPUTED"
        elif operation == "subspace_aggregate":
            request = {**base_requests[case], "operation": operation, "mode_indices": item["mode_indices"], "step_bohr": .001}
            request.pop("gradient_response_controls", None); cpu, wall, success = 60, 300, "SUBSPACE_AGGREGATE_COMPLETE"
        else:
            request = {"schema": "research4-b-diagnostic-request-v1", "threads": 2, "operation": "aggregate", "expected_points": item["point_ids"]}
            cpu, wall, success = 60, 300, "AGGREGATE_COMPLETE"
        request_path = WORK / "requests" / (item["id"] + ".json"); s.save(request_path, request, exclusive=True)
        binding = s.ref(request_path); inputs.append(binding)
        argv = [python, str(WORKER), "--request", str(request_path), "--request-sha256", binding["sha256"], "--output", "{output}", "--execute"]
        if operation == "point":
            append_result(argv, "warmstart", item["inherited_center_id"])
            if item.get("subspace_job_id"): append_result(argv, "subspace", item["subspace_job_id"])
        elif operation == "subspace_aggregate":
            append_result(argv, "center", item["inherited_center_id"])
            for point_id in item["point_ids"]: append_result(argv, "point", point_id)
        else:
            for point_id in item["point_ids"]: append_result(argv, "point", point_id)
        job = {"id": item["id"], "argv": argv, "cpu_seconds": cpu, "wall_seconds": wall, "depends_on": item["depends_on"],
            "max_attempts": 2, "result_schema": "research4-b-diagnostic-result-v1", "success_statuses": [success]}
        if item.get("admission_group"): job["admission_group"] = item["admission_group"]
        if item["id"] == SMOKE: job["smoke_success_gate"] = True
        jobs.append(job)
    plan_path = WORK / "preparation/negative_subspace_plan.json"
    s.save(plan_path, {"schema": "research4-b-negative-subspace-plan-v7", "jobs": planned,
        "old_successes_reused": sorted(state["results"]), "additional_cpu_seconds_this_round": 0}, exclusive=True)
    inputs.append(s.ref(plan_path)); prefix = WORK.name + "/"
    core_arrays = ("gradient.npy", "gradient_grid_response_true.npy", "direction.npy", "geometry_bohr.npy", "predicted_Hq.npy",
        "predicted_subspace_Hq.npy", "predicted_subspace_false_Hq.npy", "predicted_subspace_true_Hq.npy",
        "selected_true_direction_false_Hq.npy", "selected_true_direction_true_Hq.npy")
    allowlist = [prefix + "*.json", prefix + "*.jsonl", prefix + "detached-*.console.log",
        *[prefix + "preparation/**/*" + extension for extension in (".json", ".csv", ".txt", ".md")], prefix + "requests/*.json",
        *[prefix + "science/**/*" + extension for extension in (".json", ".jsonl", ".csv", ".log", ".txt")],
        *[prefix + "science/**/" + name for name in core_arrays], prefix + "science/**/subspace_*.npy",
        prefix + "science/**/selected_direction_true.npy", prefix + "science/**/original_Hessian_projected_K.npy",
        prefix + "execution/*.json", prefix + "execution/*.log"]
    spec = {"schema": SPEC_SCHEMA, "reserve_cpu_seconds": RESERVE, "source_refs": unique_refs(sources), "input_refs": unique_refs(inputs),
        "authorization_ref": s.ref(grant_path), "predecessor_bindings": bindings, "jobs": jobs,
        "smoke_job_id": SMOKE, "fail_fast_quantum_groups": quantum_groups(planned),
        "upload": {"enabled": True, "allowlist": allowlist}, "cpu_cap_seconds": CAP, "additional_cpu_seconds_this_round": 0,
        "local_only_artifacts": ["*.chk", "density.npy", "grids_coords.npy", "grids_weights.npy", "grid_non0tab.npy"],
        "local_only_policy": "Preserve all artifacts locally with result/seal hashes; return only core gradients/directions/Hq and small metadata/logs.",
        "new_quantum_evaluations_at_preparation": 0, "hourly_automation_created": False}
    s.save(target, spec, exclusive=True)
    return {"status": "SPEC_WRITTEN_NOT_RUN_NO_MANIFEST", "spec": s.ref(target), "jobs": len(jobs), "quantum_points": len(quantum),
        "smoke_job_id": SMOKE, "inherited_successes": len(state["results"]), "additional_cpu_seconds_this_round": 0,
        "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in jobs),
        "remaining_cpu_seconds": CAP - carry["budget_charged_cpu_seconds"], "new_quantum_evaluations": 0, "git_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec"); parser.add_argument("--spec-sha256"); parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--build-reviewed-spec", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        if args.spec or args.spec_sha256 or args.build_reviewed_spec: parser.error("Use one preparation mode")
        plan = negative_subspace_plan(); points = [p for p in plan if p["operation"] == "point"]
        result = {"status": "PURE_PLAN_NOT_RUN", "jobs": plan, "job_count": len(plan), "point_count": len(points),
            "pair_count": len({p["admission_group"] for p in points}), "new_quantum_evaluations": 0,
            "additional_cpu_seconds_this_round": 0, "creates_files": False,
            "smoke_job_id": SMOKE}
    elif args.build_reviewed_spec:
        if args.spec or args.spec_sha256: parser.error("Use build mode or pinned-spec preparation")
        result = build_reviewed_spec()
    else:
        if not args.spec or not args.spec_sha256: parser.error("Pinned --spec and --spec-sha256 required")
        result = prepare(args.spec, args.spec_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
