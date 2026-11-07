"""Prepare a pinned finite v4 queue from a reviewed spec; never run science/Git.

Spec schema research4-b-validation-spec-v4: jobs (the supervisor job objects),
source_refs (worker/source refs), input_refs (request/evidence refs), optional
authorization_ref, and optional upload {enabled, allowlist}. Every output is
created exclusively. The original allocation and all predecessor files stay
read-only; only the completed v3 remainder may carry forward.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/parallel-cloud-20261106/worker-b/B20261106-01"
PREVIOUS = BASE / "followup-rootcause-v3"
WORK = BASE / "followup-validation-v4"
SUPERVISOR = ROOT / "scripts/research4_b_validation_supervisor_20261107.py"
HELPER = ROOT / "scripts/research4_parallel_b_git_return_20261106.py"
SPEC_SCHEMA = "research4-b-validation-spec-v4"


def load_supervisor():
    spec = importlib.util.spec_from_file_location("r4_b_validation_prepare_supervisor", SUPERVISOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def unique_refs(refs):
    found = {}
    for value in refs:
        key = value["path"]
        if key in found and found[key] != value:
            raise ValueError("Conflicting references to " + key)
        found[key] = value
    return list(found.values())


def prepare(spec_path, spec_sha256):
    s = load_supervisor()
    spec_path = s.path(spec_path)
    spec = s.read(spec_path, spec_sha256)
    s.require(spec.get("schema") == SPEC_SCHEMA, "Exact reviewed v4 spec required")
    s.require(isinstance(spec.get("jobs"), list) and spec["jobs"], "Explicit finite jobs required")
    s.require(not any((WORK / name).exists() for name in ("manifest_ROOT_AUTHORIZED.json", "return_manifest.json", "cpu_budget.json", "state.json"))
              and not s.path(".runtime/worker-b-validation-allocation.json").exists(),
              "V4 preparation/allocation already exists; preserve it instead of resetting")
    prior_manifest_path = PREVIOUS / "manifest_ROOT_AUTHORIZED.json"
    prior = s.read(prior_manifest_path)
    manifest = {key: prior[key] for key in ("run_id", "branch", "python_executable", "cpu_cap_seconds",
                "deadline_utc", "threads", "memory_gib", "rss_stop_gib", "reserve_cpu_seconds")}
    manifest["reserve_cpu_seconds"] = spec.get("reserve_cpu_seconds", 300)
    s.require(type(manifest["reserve_cpu_seconds"]) in (int, float) and manifest["reserve_cpu_seconds"] >= 300,
              "V4 reserve must explicitly retain at least300 CPU seconds")
    manifest.update(schema=s.MANIFEST_SCHEMA, work_directory=str(WORK.relative_to(ROOT)),
        ledger=str((WORK / "cpu_budget.json").relative_to(ROOT)),
        predecessor_manifest=s.ref(prior_manifest_path), predecessor_ledger=s.ref(PREVIOUS / "cpu_budget.json"),
        predecessor_state=s.ref(PREVIOUS / "state.json"),
        predecessor_supervisor=s.ref(ROOT / "scripts/research4_b_rootcause_supervisor_20261107.py"),
        predecessor_allocation=s.ref(ROOT / ".runtime/worker-b-rootcause-allocation.json"),
        jobs=spec["jobs"], same_allocation_no_budget_reset=True)
    # Validate every prior seal, closed wait4 row and allocation before writing.
    carry = s.validate_carry(manifest)
    s.require(datetime.now(timezone.utc) < datetime.fromisoformat(manifest["deadline_utc"]),
              "Original allocation deadline has expired")
    refs = unique_refs([s.ref(SUPERVISOR), s.ref(__file__), *spec.get("source_refs", [])])
    inputs = unique_refs([s.ref(spec_path), *spec.get("input_refs", [])])
    if spec.get("authorization_ref"):
        s.checked(spec["authorization_ref"])
        manifest["authorization_ref"] = spec["authorization_ref"]
        inputs = unique_refs([*inputs, spec["authorization_ref"]])
    for binding in refs + inputs:
        s.checked(binding)
    upload = spec.get("upload", {"enabled": True})
    return_path = WORK / "return_manifest.json"
    return_manifest = None
    if upload.get("enabled"):
        prefix = WORK.name + "/"
        allowlist = upload.get("allowlist", [prefix + "*.json", prefix + "*.jsonl",
            prefix + "detached-*.console.log", prefix + "preparation/**/*.json", prefix + "requests/*.json",
            prefix + "science/**/*.json", prefix + "science/**/*.csv", prefix + "science/**/*.log",
            prefix + "science/**/*.txt", prefix + "execution/*.json", prefix + "execution/*.log"])
        s.require(isinstance(allowlist, list) and allowlist and all(isinstance(p, str) and p.startswith(prefix)
                  and ".." not in Path(p).parts for p in allowlist), "Upload allowlist must stay within v4")
        return_manifest = {"schema": "research4-b-validation-return-manifest-v4", "role": "B",
            "run_id": s.RUN_ID, "work_directory": str(BASE.relative_to(ROOT)), "ledger": manifest["ledger"],
            "upload": {"branch": s.BRANCH, "allowlist": allowlist}}
        refs = unique_refs([*refs, s.ref(HELPER)])
    manifest["source_refs"] = refs
    manifest["input_refs"] = inputs
    # Preflight all queue bindings before leaving any prepared output. Upload
    # refs only become available after the return manifest is written below.
    s.validate(manifest, "0" * 64)
    if return_manifest is not None:
        s.save(return_path, return_manifest, exclusive=True)
        binding = s.ref(return_path)
        manifest["upload"] = {"enabled": True, "helper": s.ref(HELPER), "manifest": binding}
        manifest["input_refs"] = unique_refs([*inputs, binding])
    else:
        manifest["upload"] = {"enabled": False}
    target = WORK / "manifest_ROOT_AUTHORIZED.json"
    s.save(target, manifest, exclusive=True)
    final_sha = s.sha(target)
    s.validate(manifest, final_sha)
    return {"status": "PREPARED_NOT_RUN", "manifest": s.ref(target), "jobs": len(manifest["jobs"]),
            "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
            "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
            "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
            "remaining_cpu_seconds": s.CAP - carry["budget_charged_cpu_seconds"],
            "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in manifest["jobs"]),
            "new_quantum_evaluations": 0, "git_mutations": 0}


def build_reviewed_spec():
    """Write the finite root-reviewed plan for a separate manifest review step."""
    s = load_supervisor()
    previous = s.read(PREVIOUS / "manifest_ROOT_AUTHORIZED.json")
    state = s.read(PREVIOUS / "state.json")
    carry = s.validate_carry({"predecessor_manifest": s.ref(PREVIOUS / "manifest_ROOT_AUTHORIZED.json"),
        "predecessor_ledger": s.ref(PREVIOUS / "cpu_budget.json"), "predecessor_state": s.ref(PREVIOUS / "state.json"),
        "predecessor_supervisor": s.ref(ROOT / "scripts/research4_b_rootcause_supervisor_20261107.py"),
        "predecessor_allocation": s.ref(ROOT / ".runtime/worker-b-rootcause-allocation.json")})
    target = WORK / "reviewed_spec_ROOT_AUTHORIZED.json"
    s.require(not target.exists(), "Preserve existing reviewed v4 spec")
    original_state = s.read(BASE / "state.json")
    frequencies = {case: original_state["results"][case + "-frequency"]
                   for case in ("water", "protocatechuic_acid", "calycosin")}
    for binding in frequencies.values(): s.checked(binding)
    worker = ROOT / "scripts/research4_b_validation_worker_20261107.py"
    paper = ROOT / "scripts/research4_b_paper_validation_20261107.py"
    original_worker = s.ref(ROOT / "scripts/research4_parallel_b_quantum_20261106.py")
    paper_request = WORK / "preparation/paper_validation_request.json"
    paper_data = s.read(paper_request)
    inputs = [*frequencies.values(), original_worker, s.ref(paper_request), *paper_data["inputs"].values(),
              s.ref(WORK / "preparation/research_questions_ROOT_AUTHORIZED.json"),
              s.ref(WORK / "preparation/v3_closure_review.json")]
    for binding in inputs: s.checked(binding)
    grant_path = WORK / "preparation/user_authorization_validation.json"
    grant = {"schema": "research4-b-validation-user-authorization-v4", "recorded_at_utc": s.utc(),
        "user_instruction": "你多设计一下验证去解决这些问题并自主探索一下我的这个论文还需要哪方面的实验和验证你都设置好在后台自动运行计算，我授权你72小时内拥有全部权限所有多设计一些后台自动的实验",
        "interpretation": "Authorizes finite new experiments and routine repairs within the inherited allocation; no new CPU budget, refund, deadline extension or hourly reporting task.",
        "run_id": s.RUN_ID, "cpu_cap_seconds": s.CAP, "deadline_utc": previous["deadline_utc"],
        "actual_closed_cpu_seconds": carry["actual_closed_cpu_seconds"],
        "permanent_conservative_debit_cpu_seconds": carry["conservative_debit_cpu_seconds"],
        "budget_charged_cpu_seconds": carry["budget_charged_cpu_seconds"],
        "remaining_cpu_seconds": s.CAP - carry["budget_charged_cpu_seconds"], "reserve_cpu_seconds": 300,
        "maximum_attempts_per_job": 2, "automatic_retry": "Only evidenced operational failures after real wait4 closure; scientific SCF qualification failures and valid negative results are not retried.",
        "paired_admission": "Reserve both unsatisfied point caps plus300 CPU seconds before either sign or any retry.",
        "protocol": s.ref(WORK / "preparation/research_questions_ROOT_AUTHORIZED.json"),
        "predecessor_closure_review": s.ref(WORK / "preparation/v3_closure_review.json")}
    s.save(grant_path, grant, exclusive=True); inputs.append(s.ref(grant_path))
    # Preserve the two original small planning artifacts byte for byte.
    copied = []
    for name in ("independent_v3_final_review.json", "proposed_finite_direction_tests.csv"):
        old = PREVIOUS / "analysis-next-round" / name
        new = WORK / "preparation/v3_analysis" / name; new.parent.mkdir(parents=True, exist_ok=True)
        with new.open("xb") as handle: handle.write(old.read_bytes())
        copied.append({"original": s.ref(old), "copied_exact_bytes": s.ref(new)})
        inputs.append(s.ref(new))
    copied_path = WORK / "preparation/v3_analysis/source_refs.json"; s.save(copied_path, {"files": copied}, exclusive=True)
    inputs.append(s.ref(copied_path))
    # The frozen return helper does not accept .gz. Publish a reversible JSON
    # envelope of each original <1MiB compressed Git blob, without altering it.
    envelopes = []
    for binding in paper_data["inputs"].values():
        p = s.checked(binding)
        if p.suffix == ".gz":
            s.require(p.stat().st_size <= 1024 ** 2, "Paper compressed blob exceeds reviewed1MiB limit")
            envelopes.append({"original": binding, "encoding": "base64_of_original_gzip_bytes",
                              "base64": base64.b64encode(p.read_bytes()).decode("ascii")})
    blob_path = WORK / "preparation/paper_inputs/returnable_gzip_blobs.json"
    s.save(blob_path, {"schema": "research4-paper-input-blob-envelope-v1", "blobs": envelopes}, exclusive=True)
    inputs.append(s.ref(blob_path))
    sources = [s.ref(worker), s.ref(paper)]
    snapshots = []
    for p in (worker, paper, SUPERVISOR, Path(__file__).resolve(),
              ROOT / "scripts/test_research4_b_validation_supervisor_20261107.py",
              ROOT / "scripts/test_research4_b_validation_worker_20261107.py",
              ROOT / "scripts/test_research4_b_paper_validation_20261107.py"):
        s.require(p.is_file(), "Final reviewed source/test is absent: " + str(p))
        dest = WORK / "preparation/source_snapshots" / (p.name + ".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle: handle.write(p.read_bytes())
        sources.append(s.ref(p))
        snapshots.append({"source": s.ref(p), "exact_byte_snapshot": s.ref(dest)})
        inputs.append(s.ref(dest))
    snapshot_path = WORK / "preparation/source_snapshots/source_refs.json"
    s.save(snapshot_path, {"files": snapshots}, exclusive=True); inputs.append(s.ref(snapshot_path))
    jobs, points = [], []
    def add(name, request, cpu, wall, dependencies=(), center=None, minus=None, plus=None,
            point_ids=(), admission_group=None, external=None):
        request_path = WORK / "requests" / (name + ".json")
        s.save(request_path, {"schema": "research4-b-diagnostic-request-v1", "threads": 2,
                              "source_worker": original_worker, **request}, exclusive=True)
        binding = s.ref(request_path); inputs.append(binding)
        argv = [str(s.python_path(previous["python_executable"])), str(worker), "--request", str(request_path),
                "--request-sha256", binding["sha256"], "--output", "{output}", "--execute"]
        for flag, job_id in (("center", center), ("minus", minus), ("plus", plus)):
            if job_id: argv.extend(["--" + flag + "-result", "{result:" + job_id + "}", "--" + flag + "-result-sha256", "{sha256:" + job_id + "}"])
        for flag, value in (external or {}).items():
            s.checked(value); inputs.append(value)
            argv.extend(["--" + flag + "-result", str(s.path(value["path"])), "--" + flag + "-result-sha256", value["sha256"]])
        for job_id in point_ids:
            argv.extend(["--point-result", "{result:" + job_id + "}", "--point-result-sha256", "{sha256:" + job_id + "}"])
        success = {"fixed_grid_center": "FIXED_GRID_CENTER_COMPUTED", "fixed_grid_point": "POINT_COMPUTED",
                   "point": "POINT_COMPUTED", "response_audit": "RESPONSE_AUDIT_COMPLETE", "aggregate": "AGGREGATE_COMPLETE"}[request["operation"]]
        job = {"id": name, "argv": argv, "cpu_seconds": cpu, "wall_seconds": wall,
               "depends_on": list(dependencies), "max_attempts": 2,
               "result_schema": "research4-b-diagnostic-result-v1", "success_statuses": [success]}
        if admission_group: job["admission_group"] = admission_group
        jobs.append(job)
    paper_binding = s.ref(paper_request)
    jobs.append({"id": "paper-existing-prediction-validation", "argv": [str(s.python_path(previous["python_executable"])),
        str(paper), "--request", str(paper_request), "--request-sha256", paper_binding["sha256"], "--output", "{output}", "--execute"],
        "cpu_seconds": 120, "wall_seconds": 300, "depends_on": [], "max_attempts": 2,
        "result_schema": "research4-b-paper-validation-result-v1", "success_statuses": ["DONE_CONDITIONAL_PREDICTION_DIAGNOSTICS"]})
    # These response jobs deliberately keep every old grid/SCF setting; only
    # explicit CPHF convergence and Krylov linear-dependence settings change.
    for tag, direction in (("translate0", {"type": "translation", "axis": 0}), ("mode01", {"type": "mode", "index": 1})):
        add("water-cphf-tight-" + tag, {"operation": "response_audit", "case": "water", "frequency_result": frequencies["water"],
            "grid_level": 1, "gradient_grid_response": False, "scf_conv_tol": 1e-12, "direction": direction,
            "finite_difference_step_bohr": .0005, "cphf_tol": 1e-13, "cphf_max_cycle": 200,
            "cphf_krylov_lindep": 1e-24, "include_fock_fd": False, "include_stock_direct_control": False}, 120, 600,
            external={"center": state["results"]["water-fixed-center"],
                "minus": state["results"]["water-fixed-" + tag + "-h0005-minus"],
                "plus": state["results"]["water-fixed-" + tag + "-h0005-plus"]})
    def batch(case, grid, fixed, cpu, wall):
        label = {"water": "water", "protocatechuic_acid": "acid", "calycosin": "caly"}[case] + "-g%d-" % grid + ("fixed" if fixed else "regrid")
        directions = [{"type": "translation", "axis": 0}, {"type": "mode", "index": 1}] if case == "water" else [{"type": "mode", "index": 1}]
        base = {"case": case, "frequency_result": frequencies[case], "grid_level": grid,
                "gradient_grid_response": False, "scf_conv_tol": 1e-12, "scf_conv_tol_grad": 1e-9,
                "scf_qualification_gradient_norm": 1e-8, "small_rho_cutoff": 0, "grid_prune": "default"}
        if not fixed: base["gradient_response_controls"] = [False, True]
        center = label + "-center"
        add(center, {**base, "operation": "fixed_grid_center" if fixed else "point",
            "direction": directions[0], "step_bohr": 0, "sign": 0}, cpu, wall)
        group_points = [center]
        for direction in directions:
            tag = "translate0" if direction["type"] == "translation" else "mode01"
            for h in (.001, .0005):
                pair = label + "-" + tag + "-h%04d" % round(h * 10000)
                for sign in (-1, 1):
                    name = pair + ("-minus" if sign < 0 else "-plus")
                    add(name, {**base, "operation": "fixed_grid_point" if fixed else "point",
                        "direction": direction, "step_bohr": h, "sign": sign}, cpu, wall,
                        dependencies=[center] if fixed else (), center=center if fixed else None, admission_group=pair)
                    group_points.append(name)
        add(label + "-comparison", {"operation": "aggregate", "expected_points": group_points}, 30, 180,
            dependencies=group_points, point_ids=group_points)
        points.extend(group_points)
    for grid, cpu, wall in ((3, 60, 180), (5, 90, 240)):
        for fixed in (True, False): batch("water", grid, fixed, cpu, wall)
    for grid, cpu, wall in ((3, 350, 600), (5, 500, 900)):
        for fixed in (True, False): batch("protocatechuic_acid", grid, fixed, cpu, wall)
    batch("calycosin", 3, False, 1500, 1800)
    add("all-validation-points-comparison", {"operation": "aggregate", "expected_points": points}, 60, 240,
        dependencies=points, point_ids=points)
    prefix = WORK.name + "/"
    core_arrays = ("gradient.npy", "gradient_grid_response_true.npy", "direction.npy", "geometry_bohr.npy", "predicted_Hq.npy",
        "H1q_AO.npy", "S1q_AO.npy", "H1q_MO.npy", "S1q_MO.npy", "U_MO.npy", "induced_response_MO.npy",
        "occupied_energy_response.npy", "density_derivative_analytic.npy", "density_derivative_FD.npy")
    allowlist = [prefix + "*.json", prefix + "*.jsonl", prefix + "detached-*.console.log",
        *[prefix + "preparation/**/*" + ext for ext in (".json", ".csv", ".txt", ".md")], prefix + "requests/*.json",
        *[prefix + "science/**/*" + ext for ext in (".json", ".jsonl", ".csv", ".log", ".txt")],
        *[prefix + "science/**/" + name for name in core_arrays], prefix + "execution/*.json", prefix + "execution/*.log"]
    spec = {"schema": SPEC_SCHEMA, "reserve_cpu_seconds": 300, "source_refs": sources,
        "input_refs": unique_refs(inputs), "authorization_ref": s.ref(grant_path), "jobs": jobs,
        "upload": {"enabled": True, "allowlist": allowlist}, "local_only_arrays": ["density.npy", "grids_coords.npy", "grids_weights.npy", "grid_non0tab.npy", "*.chk"],
        "local_only_policy": "All scientific arrays remain intact locally and retain worker/seal hashes; only the finite return whitelist is published.",
        "plan_counts": {"new_center_and_displaced_points": len(points), "response_audits_reusing_old_points": 2, "paper_jobs": 1}}
    s.save(target, spec, exclusive=True)
    return {"status": "SPEC_WRITTEN_NOT_RUN_NO_MANIFEST", "spec": s.ref(target), "jobs": len(jobs),
            "points": len(points), "first_attempt_cpu_caps_sum": sum(j["cpu_seconds"] for j in jobs),
            "remaining_cpu_seconds": s.CAP - carry["budget_charged_cpu_seconds"], "new_quantum_evaluations": 0, "git_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--build-reviewed-spec", action="store_true")
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
