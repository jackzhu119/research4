"""Closed V4 pure-J evidence audit. No SCF, gradient or Hessian evaluation.

Only an explicit root release for all frozen stages permits molecular reads.
Signed computed modes are retained separately from qualified isolated-molecule
modes. RHF-before-XC caches are not a full PBE Hessian or a restart facility.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "pyscf-spinosin-pure-j-20261101-v4"
SUPERVISOR = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261101.py"
ENTRY = ROOT / "scripts/pyscf_spinosin_pure_j_worker_20261101.py"
V8 = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
CORE_PATH = ROOT / "scripts/analyze_pyscf_continuation_v2_20261031.py"
V3_ANALYZER = ROOT / "scripts/analyze_pyscf_monitored_results_20261031.py"
EXECUTION_LOCK_SHA = "3af7acecce03964892c91fcc5d32d9ccd9e5c7abfd29e84b1fc1595bd96a9dab"
PEAK_PATH = ROOT / "reports/pyscf-spinosin-continuation-20261031-v3/final-analysis/experimental_candidate_positions.csv"
PEAK_SHA = "56ba3f3b860de3e649fc260d08b81983e681a4d42dd5723d4a82492ce141d3f8"
PINNED = {SUPERVISOR: "354ff51b5c2af2810033656d9972948262cfebb4dabb3978fca2d021f97e2b8f",
          ENTRY: "e3215e563b4429212084cfccbcec98e6a9f482bd8c3352d53d3064402649b450",
          V8: "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d",
          CORE_PATH: "31ebde796b057e82444bede277ccb1c1af926cefd742101620eff983b74f6bd8",
          V3_ANALYZER: "92ef86dedcbef4a8879fd53a15fa2a209ea19c761c203b1a91177d60e2908a08",
          ROOT/"scripts/analyze_pyscf_spinosin_pilot_20261031.py": "1bd0a8ac445a72f7a2dbbc3ffcd38ee0c9e868f2b90043738b83386d019b288b"}
STOP_FLAGS = ("cpu_terminated", "memory_terminated", "wall_terminated", "monitor_failed", "hard_killed",
              "actual_cpu_guard_TERM_sent", "owned_child_pidfd_containment_requested",
              "owned_unreaped_child_direct_PID_containment_requested")
CM1_TO_THZ = .0299792458
FREQUENCY_FIELDS = ["phase", "identity", "mode", "frequency_cm1", "frequency_thz", "imaginary",
                    "accepted_for_comparison", "IR_intensity_status"]


def module(name, path):
    # Verify the entire frozen helper chain before any helper module executes.
    for source,expected in PINNED.items():
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError("Frozen transitive source changed: "+str(source))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


CORE = module("frozen_v2_pure_j_analysis_helpers", CORE_PATH)
V3 = module("frozen_v3_pure_j_analysis_helpers", V3_ANALYZER)
sha, read, write, csvwrite = CORE.sha, CORE.read, CORE.write, CORE.csvwrite


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def pin(path, hashes, expected=None):
    path = resolved(path)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError("Evidence SHA-256 differs: " + str(path))
    hashes[str(path)] = actual
    return path


def option(command, name):
    if command.count(name) != 1 or command.index(name)+1 >= len(command):
        raise ValueError("Exactly one recorded argument required: " + name)
    return command[command.index(name)+1]


def healthy_parent(row):
    return bool(row.get("status") == "EXITED" and row.get("exit_code") == 0
                and row.get("ended_at_utc") and not any(row.get(key) for key in STOP_FLAGS)
                and not row.get("actual_cpu_stop_observation") and not row.get("resource_stop_reason")
                and not row.get("residual_processes_require_review_after_main_payload_exit"))


def cost_totals(ledger, v3, v2):
    """Separate actual pools, including failed/overrun cost; never invent savings."""
    result = {}
    for label, value in (("V4", ledger), ("V3", v3), ("V2", v2)):
        values = [float(row["actual_cpu_seconds"]) for row in value["phases"]]
        total = math.fsum(values)
        if (not all(math.isfinite(x) and x >= 0 for x in values)
                or not math.isclose(total, float(value["consumed_cpu_seconds"]), abs_tol=1e-7, rel_tol=1e-12)):
            raise ValueError("Closed actual CPU sum differs: " + label)
        result[label + "_actual_CPU_seconds"] = total
        result[label + "_configured_pool_CPU_seconds"] = float(value["quantum_cpu_cap_seconds"])
        result[label + "_pool_overrun_CPU_seconds"] = max(0., total-float(value["quantum_cpu_cap_seconds"]))
    result["V2_V3_V4_actual_CPU_seconds_scope_excludes_original_pilot_and_nonquantum_tests"] = sum(
        result[label + "_actual_CPU_seconds"] for label in ("V2", "V3", "V4"))
    result["V4_is_separate_36_CPU_hour_pool_not_continuation_of_original_48_hour_claim"] = True
    return result


def authorize(root, release, validator=None):
    """No worker result or molecular array is read until this entire gate passes."""
    root = root.resolve(); path = root/"cpu_budget.json"; ledger = read(path)
    if any(row.get("status") == "RUNNING" or not row.get("ended_at_utc") for row in ledger["phases"]):
        raise ValueError("All frozen V4 stages must actually end before analysis")
    if (release.get("allow_analysis_export") is not True or release.get("cpu_budget_sha256") != sha(path)
            or not release.get("created_at_utc") or resolved(release.get("run_root", root)) != root):
        raise ValueError("Root release must bind this final V4 ledger and run root")
    if datetime.fromisoformat(release["created_at_utc"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Root release requires timezone-aware creation time")
    if ledger.get("schema") != "research4-pyscf-actual-cpu-budget-v4":
        raise ValueError("The independent actual-CPU V4 ledger is required")
    if any(row.get("status") not in ("EXITED", "PARTIAL_OR_FAILED", "FAILED_TO_START", "BUDGET_VIOLATION_REQUIRES_REVIEW") for row in ledger["phases"]):
        raise ValueError("Unknown phase status")
    if any(row.get("status") == "BUDGET_VIOLATION_REQUIRES_REVIEW" for row in ledger["phases"]) and release.get("allow_disclosed_budget_violation_analysis") is not True:
        raise ValueError("Budget violation requires explicit disclosed-failure root review")
    hashes = {}; pin(path, hashes, release["cpu_budget_sha256"])
    lock_path = root/"execution_lock.json"
    if release.get("execution_lock_sha256") != EXECUTION_LOCK_SHA:
        raise ValueError("Root release must bind the fixed pre-quantum execution lock")
    lock = read(pin(lock_path, hashes, EXECUTION_LOCK_SHA))
    if lock.get("status") != "SOURCE_AND_COMMANDS_FROZEN_BEFORE_QUANTUM":
        raise ValueError("Pre-quantum execution lock is not fixed")
    for target, key in ((Path(__file__), "analyzer_source_sha256"), (ENTRY, "entry_source_sha256"),
                        (SUPERVISOR, "supervisor_source_sha256")):
        if release.get(key) != sha(target): raise ValueError("Released source hash differs: " + key)
        pin(target, hashes, release[key])
    for source, expected in PINNED.items(): pin(source, hashes, expected)
    for section in ("source_sha256", "input_sha256"):
        for filename, expected in lock[section].items(): pin(filename, hashes, expected)
    stages = lock["phases"]
    if [row["phase"] for row in ledger["phases"]]!=list(stages):
        raise ValueError("Final phases must preserve the frozen control-before-target order")
    if len(ledger["phases"]) != len(stages) or {row["phase"] for row in ledger["phases"]} != set(stages):
        raise ValueError("Final ledger must contain each frozen stage exactly once")
    closed_cpu=0.
    for row in ledger["phases"]:
        if not math.isclose(float(row.get("actual_cpu_closed_batch_cost_seconds",float("nan"))),closed_cpu,abs_tol=1e-7,rel_tol=1e-12):
            raise ValueError("CPU guard did not bind actual preceding completed phase costs")
        closed_cpu+=float(row["actual_cpu_seconds"])
        if row["command"] != stages[row["phase"]]["payload_argv"]:
            raise ValueError("Actual payload differs from its fixed execution command")
    official = module("official_actual_cpu_v4_closed_audit", SUPERVISOR)
    authorization = (validator or official.validate_ledger)(ledger)
    if ledger.get("runner_sha256") != sha(SUPERVISOR): raise ValueError("Actual supervisor source differs")
    pin(ledger["authorization_path"], hashes, ledger["authorization_sha256"])
    v3path = pin(authorization["prior_closed_ledger_path"], hashes, authorization["prior_closed_ledger_sha256"])
    prior_v3 = read(v3path)
    bound = prior_v3["prior_closed_ledger"]
    v2path = pin(bound["path"], hashes, bound["sha256"]); prior_v2 = read(v2path)
    if any(row.get("status") == "RUNNING" for value in (prior_v3, prior_v2) for row in value["phases"]):
        raise ValueError("Historical ledger unexpectedly active")
    totals = cost_totals(ledger, prior_v3, prior_v2)
    if (not math.isclose(totals["V3_actual_CPU_seconds"],111023.488913,abs_tol=1e-7,rel_tol=0)
            or not math.isclose(totals["V2_actual_CPU_seconds"],14846.81297,abs_tol=1e-7,rel_tol=0)):
        raise ValueError("Bound historical actual costs differ from fixed closure")
    pin(CORE_PATH, hashes); pin(V3_ANALYZER, hashes); pin(CORE.LEGACY.__file__, hashes)
    return ledger, authorization, lock, hashes, totals, prior_v3, prior_v2


def audit_closed_proof(root, row, hashes):
    log = pin(row["log"], hashes, row["log_sha256"])
    if not log.is_relative_to(root): raise ValueError("Driver log escapes V4 run")
    proof_path = resolved(row["supervisor_proof_path"])
    if not proof_path.is_relative_to(root): raise ValueError("Supervisor proof escapes V4 run")
    if row["status"] == "FAILED_TO_START" and not row.get("supervisor_proof_sha256"):
        if proof_path.is_file(): pin(proof_path, hashes)
        return {"status":"FAILED_TO_START_WITHOUT_BOUND_PROOF", "healthy_parent":False}
    proof = read(pin(proof_path, hashes, row["supervisor_proof_sha256"]))
    expected = {"schema":"research4-pyscf-actual-cpu-supervisor-proof-v4", "phase":row["phase"],
        "threads":4,"address_space_soft_limit_bytes":24*1024**3,"address_space_hard_limit_bytes":24*1024**3,
        "rss_stop_bytes":10*1024**3,"cgroup_kill_bytes":12*1024**3,"expected_cgroup_memory_max_bytes":16*1024**3,
        "actual_cpu_guard_enabled":True,"actual_cpu_guard_source":"PROC_STAT_CUMULATIVE_FROM_PROCESS_BIRTH",
        "actual_cpu_cleanup_reserve_seconds":120,"actual_cpu_batch_cap_seconds":129600}
    if any(proof.get(k) != v for k,v in expected.items()): raise ValueError("Archived V4 guard contract differs")
    if (resolved(proof["ledger_path"]) != root/"cpu_budget.json"
            or resolved(proof["supervisor_script_path"]) != SUPERVISOR
            or proof.get("supervisor_script_sha256") != PINNED[SUPERVISOR]
            or proof.get("child_pid") != row.get("pid")
            or proof.get("target_pgid") != proof.get("child_pid")
            or proof.get("target_session_id") != proof.get("child_pid")):
        raise ValueError("Archived supervisor source/ledger/PID/session differs")
    for key in ("child_pid","child_start_ticks","supervisor_pid","supervisor_start_ticks","actual_cpu_clock_ticks_per_second"):
        if type(proof.get(key)) is not int or proof[key] <= 0: raise ValueError("Invalid historical process identity")
    for key in ("cpu_soft_limit_seconds","cpu_hard_limit_seconds","wall_limit_seconds","actual_cpu_stop_threshold_seconds",
                "actual_cpu_closed_batch_cost_seconds","actual_cpu_cleanup_reserve_seconds",
                "actual_cpu_guard_source","actual_cpu_clock_ticks_per_second","actual_cpu_batch_cap_seconds","cpu_guard_kill_grace_seconds"):
        if proof.get(key) != row.get(key): raise ValueError("Archived proof/phase resource mismatch: "+key)
    if proof["actual_cpu_stop_threshold_seconds"] != row["granted_cpu_seconds"]-120:
        raise ValueError("Wrong cumulative-CPU stop boundary")
    for key,maximum in (("monitor_interval_seconds",.1),("memory_kill_grace_seconds",1.),("cpu_guard_kill_grace_seconds",1.)):
        if not isinstance(proof.get(key),(int,float)) or isinstance(proof.get(key),bool) or not 0 < proof[key] <= maximum:
            raise ValueError("Unreviewed monitor timing")
    return {"status":"ARCHIVED_V4_STARTUP_PROOF_AND_CLOSED_ROW_VERIFIED", "healthy_parent":healthy_parent(row) and float(row["actual_cpu_seconds"])<=float(row["granted_cpu_seconds"]),
        "proof_sha256":sha(proof_path),"startup_proof_is_not_live_PID_or_hard_RSS_guarantee":True}


def frequency_rows(location, identity, phase, hashes):
    """Retain actual CSV signs even when later evidence fails qualification."""
    path=location/"frequencies.csv"
    if not path.is_file(): return []
    pin(path,hashes); rows=[]
    with path.open(encoding="utf-8-sig",newline="") as handle:
        for i,value in enumerate(csv.DictReader(handle),1):
            cm=float(value["frequency_cm1"]); thz=float(value.get("frequency_thz",value.get("frequency_THz")))
            if (int(value["mode"]) != i or not np.isfinite([cm,thz]).all()
                    or not np.isclose(thz,cm*CM1_TO_THZ,atol=1e-10,rtol=0)
                    or value.get("IR_intensity_status") != "NOT_COMPUTED"):
                raise ValueError("Actual signed frequency CSV is inconsistent")
            rows.append({"phase":phase,"identity":identity,"mode":i,"frequency_cm1":cm,"frequency_thz":thz,
                "imaginary":bool(cm<0),"accepted_for_comparison":False,"IR_intensity_status":"NOT_COMPUTED"})
    return rows


def compare_mode_subspaces(cm, saved, replayed):
    """Compare real normalized mode spaces; allow signs/degenerate rotations."""
    cm=np.asarray(cm,float); left=np.asarray(saved); right=np.asarray(replayed)
    if (np.iscomplexobj(left) or np.iscomplexobj(right) or left.shape != right.shape or left.shape[0] != len(cm)
            or not np.isfinite(cm).all() or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Invalid mode arrays")
    left=left.reshape(len(cm),-1); right=right.reshape(len(cm),-1)
    if np.any(np.linalg.norm(left,axis=1)==0) or np.any(np.linalg.norm(right,axis=1)==0):
        raise ValueError("Zero mode is not vibration evidence")
    worst=0.; start=0
    while start<len(cm):
        end=start+1
        while end<len(cm) and abs(cm[end]-cm[start]) <= 1e-5: end+=1
        a=left[start:end].T; b=right[start:end].T
        if np.linalg.matrix_rank(a) != end-start or np.linalg.matrix_rank(b) != end-start:
            raise ValueError("Rank-deficient mode cluster")
        qa=np.linalg.qr(a,mode="reduced")[0]; qb=np.linalg.qr(b,mode="reduced")[0]
        distance=float(np.linalg.norm(qa-qb@(qb.T@qa),ord=2))
        worst=max(worst,distance); start=end
    if worst>1e-6: raise ValueError("Saved mode directions differ from Hessian replay")
    return {"status":"SAVED_REAL_MODE_SUBSPACES_MATCH_HESSIAN_REPLAY","maximum_projector_distance":worst,
        "degeneracy_cluster_width_cm1":1e-5,"arbitrary_mode_signs_allowed":True}


def validate_global_manifest(value,result,gradient_sha,geometry_sha,natom=75):
    """Small JSON completion evidence; never load ignored large H1/MO caches."""
    expected={"status":"DONE_INTERMEDIATE_CAPTURE_ONLY","native_global_kernel_returned_normally":True,
        "H1AO_complete":True,"global_Hessian_assembly_unchanged":True,"partial_kernel_tiling_used":False,
        "integral_backend":"SCREENED_DIRECT_RKS_V4","response_block_atoms":8,
        "native_conv_tol_cpscf":1e-8,"native_max_cycle":50,"native_level_shift":0,
        "independent_CPHF_residual_norm":"NOT_COMPUTED","requested_tolerance_is_not_a_measured_residual":True}
    if any(value.get(k)!=v for k,v in expected.items()): raise ValueError("No complete unchanged global PBE Hessian kernel evidence")
    if value.get("dimensions",{}).get("atoms")!=natom or value.get("response_complete_global_atom_ids")!=list(range(natom)):
        raise ValueError("Incomplete full-system response atom coverage")
    if ([a.get("atom_id") for a in value.get("H1AO_atoms",[])]!=list(range(natom))
            or (value.get("partial_hessian") or {}).get("native_function_returned_normally") is not True):
        raise ValueError("Complete RKS electronic partial and global H1AO metadata required; RHF early cache is insufficient")
    binding=value.get("binding",{})
    if hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()!=value.get("binding_sha256"):
        raise ValueError("Global Hessian binding canonical SHA differs")
    if binding.get("worker_script_sha256")!=PINNED[V8] or binding.get("settings")!=result.get("settings"):
        raise ValueError("Global capture scaffold or actual pure-J settings differ")
    evidence=binding.get("input_evidence",{})
    required={"geometry_optimized":True,"final_gradient_thresholds_passed":True,
        "gradient_npy_sha256":gradient_sha,"normal_mode_geometry_npy_sha256":geometry_sha,
        "optimization_result_sha256":result.get("optimization_result_sha256"),
        "resource_supervision_proof_sha256":result.get("resource_supervision",{}).get("proof_sha256")}
    if any(evidence.get(k)!=v for k,v in required.items()): raise ValueError("Global capture fresh-gradient/geometry/parent/proof binding differs")
    blocks=value.get("response_blocks",[])
    expected_chunks=[list(range(i,min(i+8,natom))) for i in range(0,natom,8)]
    if [b.get("global_atom_ids") for b in blocks]!=expected_chunks: raise ValueError("Response blocks do not preserve the frozen all-atom B8 grouping")
    for block,atoms in zip(blocks,expected_chunks):
        if (block.get("native_solver_returned_normally") is not True
                or [a.get("atom_id") for a in block.get("arrays",[])]!=atoms
                or block.get("residual_norm")!="NOT_COMPUTED"
                or block.get("normal_return_does_not_prove_measured_residual_below_requested_tolerance") is not True
                or not np.isclose(float(block.get("requested_effective_CPHF_tolerance",float("nan"))),len(atoms)*1e-8,atol=1e-16,rtol=0)):
            raise ValueError("Response block completion/tolerance metadata differs")
    return {"status":"FULL_GLOBAL_KERNEL_AND_COMPLETE_RESPONSE_METADATA_VERIFIED","atoms":natom,
        "H1AO_atom_records":natom,"response_blocks":len(blocks),"large_intermediate_arrays_loaded":False,
        "RHF_early_cache_alone_is_not_sufficient":True,"CPHF_residual_independently_measured":False}


def audit_global_manifest(location,result,hashes):
    value=read(pin(location/"hessian_cache_manifest.json",hashes,result.get("hessian_cache_manifest_sha256","MISSING")))
    return validate_global_manifest(value,result,sha(location/"gradient_hartree_per_bohr.npy"),
        sha(location/"normal_mode_geometry_angstrom.npy"))


def qualified_modes(result, rows, parent_ok, scientific_checks_ok):
    if (not parent_ok or not scientific_checks_ok or result.get("worker_result_missing")
            or result.get("identity") != "spinosin" or len(rows) != 219
            or any(r["identity"] != "spinosin" or r["mode"] != i+1 or not r["frequency_cm1"]>0
                   for i,r in enumerate(rows)) or not CORE.valid_modes(result,"spinosin")):
        return False
    settings=result.get("settings",{})
    return bool(settings.get("engineering_worker_version")=="PURE_PBE_J_PARTIAL_V9_SAVE_ONLY"
        and settings.get("partial_hessian_engine")=="PURE_PBE_J_PARTIAL_V9_SAVE_ONLY"
        and settings.get("density_fit_backend")=="SCREENED_DIRECT_RKS_V4"
        and settings.get("completed_pure_j_equivalence_evidence") and settings.get("threads")==4)


def all_candidate_pairs(experimental, low_modes):
    if len(experimental)!=20: raise ValueError("Retain all20 fixed experimental candidates")
    return [{**observed,"theory_phase":mode["phase"],"theoretical_mode":mode["mode"],
        "theoretical_frequency_thz":mode["frequency_thz"],"theoretical_wavenumber_cm1":mode["frequency_cm1"],
        "absolute_difference_thz":abs(observed["experimental_frequency_thz"]-mode["frequency_thz"]),
        "absolute_difference_cm1":abs(observed["experimental_wavenumber_cm1"]-mode["frequency_cm1"]),
        "comparison_status":"ALL_PAIRS_DESCRIPTIVE_ONLY_NO_FREQUENCY_SCALING_OR_ATTRIBUTION",
        "IR_intensity_status":"NOT_COMPUTED","chemical_attribution_verified":False}
        for observed in experimental for mode in low_modes]


def experimental_rows(hashes):
    pin(PEAK_PATH,hashes,PEAK_SHA)
    with PEAK_PATH.open(encoding="utf-8-sig",newline="") as handle: values=list(csv.DictReader(handle))
    rows=[]
    for value in values:
        row={key:int(value[key]) for key in ("outer_fold","candidate_number","frequency_index")}
        row.update({key:float(value[key]) for key in ("experimental_frequency_thz","experimental_wavenumber_cm1")})
        if (not .1<=row["experimental_frequency_thz"]<=1.5
                or not np.isclose(row["experimental_frequency_thz"],row["experimental_wavenumber_cm1"]*CM1_TO_THZ,atol=1e-10,rtol=0)
                or value["chemical_assignment"]!="NOT_CONFIRMED"):
            raise ValueError("Fixed experimental candidate data differ")
        rows.append({**row,"chemical_assignment":"NOT_CONFIRMED"})
    if len(rows)!=20: raise ValueError("Exactly20 fixed experimental candidate rows required")
    return rows


def assert_entry(result, hashes):
    value=result.get("actual_entrypoint",{})
    if resolved(value.get("path",""))!=ENTRY or value.get("sha256")!=PINNED[ENTRY]:
        raise ValueError("Actual pure-J entrypoint differs")
    pin(ENTRY,hashes,value["sha256"])
    if result.get("settings",{}).get("actual_entrypoint",{}).get("sha256")!=PINNED[ENTRY]:
        raise ValueError("Settings actual entrypoint differs")
    # The old field names the genuine imported scaffold, not the new entry.
    if result.get("phase")=="spinosin-frequency" and (result.get("script_sha256")!=PINNED[V8]
            or result.get("original_V8_source_sha256")!=PINNED[V8]
            or result.get("original_script_sha256_field_scope")!="FROZEN_V8_EXECUTION_SCAFFOLD_NOT_THIS_ENTRYPOINT"):
        raise ValueError("Dual producer/scaffold provenance differs")


def audit_fresh_scf_log(location,result,hashes):
    path=pin(location/"pyscf.log",hashes)
    energies=[float(x) for x in re.findall(r"converged SCF energy\s*=\s*([-+0-9.eE]+)",path.read_text())]
    if (result.get("scf_converged") is not True or not energies or not np.isfinite(energies).all()
            or not np.isclose(energies[-1],float(result["energy_hartree"]),atol=2e-8,rtol=0)):
        raise ValueError("Fresh converged SCF energy log differs from result")
    return {"status":"LAST_ACTUAL_CONVERGED_SCF_LOG_ENERGY_MATCHES_RESULT","energy_hartree":energies[-1]}


def parent_optimization_audit(root, row, result, prior_v3, hashes):
    prior_root=resolved(prior_v3["authorization_path"]).parent
    args=row["command"]; start=args.index("--validation-results")+1
    paths=[resolved(p) for p in args[start:start+2]]
    entry=module("closed_pure_j_launch_gate_only",ENTRY)
    evidence=entry.verify_completed_equivalence_records(paths)
    if result.get("settings",{}).get("completed_pure_j_equivalence_evidence")!=evidence:
        raise ValueError("Current Spinosin settings do not bind both actual closed pure-J gates")
    for p in paths: pin(p,hashes)
    CORE.frequency_parent_guard(prior_root,row,result,hashes)
    parent_path=resolved(option(row["command"],"--optimization-result")); parent=read(parent_path)
    owners=[r for r in prior_v3["phases"] if "--output" in r.get("command",[]) and
        resolved(option(r["command"],"--output"))==parent_path.parent]
    if len(owners)!=1 or not healthy_parent(owners[0]): raise ValueError("Old actual optimized parent is not healthy and unique")
    proof=V3.audit_closed_proof(prior_root,prior_root/"cpu_budget.json",owners[0],hashes)
    if not proof["frequency_acceptance_allowed"] or parent.get("script_sha256")!=PINNED[V8]:
        raise ValueError("Old optimized parent source/proof differs")
    CORE.audit_final_gradient(parent_path.parent,parent,hashes)
    steps=[CORE.audit_step(p,"spinosin",owners[0]["phase"],hashes)
        for p in sorted((parent_path.parent/"geometry_steps").glob("step_*.json")) if read(p).get("scf_converged") is True]
    return {"status":"OLD_V3_OPTIMIZATION_AND_CURRENT_FRESH_GRADIENT_REQUIRED","old_result_sha256":sha(parent_path),
        "new_geometry_optimization_performed":False,"old_parent_actual_CPU_seconds":owners[0]["actual_cpu_seconds"]},steps


def control_case_timings(location,identity,hashes):
    rows=[]
    for label in ("stock_partial_B8_native4","pure_j_partial_B8_native4"):
        path=location/label/"result.json"
        if not path.is_file(): continue
        value=read(pin(path,hashes))
        cpu=float(value["CPU_seconds"]); wall=float(value["wall_seconds"])
        if not np.isfinite([cpu,wall]).all() or min(cpu,wall)<0: raise ValueError("Invalid actual case timing")
        rows.append({"identity":identity,"case":label,"actual_worker_process_CPU_seconds":cpu,
            "actual_wall_seconds":wall,"normal_modes":value.get("normal_mode_count"),
            "scope":"SOFTWARE_CONTROL_ONLY_SINGLE_RUN_NOT_SPINOSIN_SPEEDUP"})
    return rows


def audit_control(location,result,row,hashes):
    """Read only the closed validation molecule; compare every full-matrix entry."""
    identity=result["identity"]; n=3 if identity=="water" else 9
    if not (result.get("validation_only") is True and result.get("new_geometry_optimization_performed") is False
            and result.get("status")=="DONE" and result.get("validation_passed") is True
            and all(result.get(k) is True for k in ("scf_converged","final_gradient_thresholds_passed",
                "numerical_equivalence_passed","same_MF_geometry_MO_grid_retained","stationary_minimum_verified"))):
        raise ValueError("Small-molecule gate not completed")
    geometry=np.load(pin(location/"actual_geometry_angstrom.npy",hashes),allow_pickle=False)
    symbols,_=CORE.LEGACY.xyz(pin(option(row["command"],"--input-xyz"),hashes))
    precise=np.load(pin(option(row["command"],"--input-coordinates-npy"),hashes),allow_pickle=False)
    if not np.allclose(geometry,precise,atol=1e-11,rtol=0): raise ValueError("Validation geometry drifted from exact closed input")
    audit_fresh_scf_log(location,result,hashes)
    if geometry.shape!=(n,3) or not np.isfinite(geometry).all(): raise ValueError("Bad validation geometry")
    fingerprint=read(pin(location/"same_MF_fingerprint.json",hashes))
    fresh=np.load(pin(location/"fresh_gradient_hartree_per_bohr.npy",hashes),allow_pickle=False)
    stats=CORE.gradient_statistics(fresh)
    if stats["atom_rms_hartree_per_bohr"]>3e-4 or stats["atom_max_hartree_per_bohr"]>4.5e-4:
        raise ValueError("Validation fresh gradient is not stationary")
    for key,value in stats.items():
        if not np.isclose(value,result[key],atol=1e-13,rtol=1e-11): raise ValueError("Validation fresh G statistics differ")
    cases=[]; timings=[]
    for label in ("stock_partial_B8_native4","pure_j_partial_B8_native4"):
        at=location/label; meta=read(pin(at/"result.json",hashes)); arrays={}
        for key,name in (("raw","hessian_hartree_per_bohr2.npy"),("used","hessian_used_for_analysis.npy"),
                         ("modes","normal_modes.npy"),("cm","frequencies_cm1.npy"),("gradient","post_Hessian_gradient_hartree_per_bohr.npy")):
            arrays[key]=np.load(pin(at/name,hashes),allow_pickle=False)
        raw,used=arrays["raw"],arrays["used"]; cm=arrays["cm"]
        if (raw.shape!=(n,n,3,3) or used.shape!=raw.shape or arrays["modes"].shape!=(3*n-6,n,3)
                or arrays["gradient"].shape!=(n,3) or fresh.shape!=(n,3)
                or not all(np.isfinite(a).all() for a in arrays.values())
                or float(np.max(np.abs(raw-raw.transpose(1,0,3,2))))>1e-5
                or not np.isclose(float(np.max(np.abs(raw-raw.transpose(1,0,3,2)))),meta["raw_hessian_max_asymmetry"],atol=1e-14,rtol=1e-10)
                or meta.get("IR_intensities_status")!="NOT_COMPUTED"
                or meta.get("pure_j_helper_enabled")!=(label=="pure_j_partial_B8_native4")
                or meta.get("both_cases_original_V8_B8_native4_global_assembly") is not True
                or meta.get("stock_partial_function_object_and_source_restored") is not True
                or meta.get("actual_entrypoint",{}).get("sha256")!=PINNED[ENTRY]
                or meta.get("energy_difference_from_initial_fresh_SCF",float("inf"))>1e-8
                or meta.get("gradient_difference_from_initial_fresh_G",float("inf"))>1e-8
                or not np.allclose(used,.5*(raw+raw.transpose(1,0,3,2)),atol=1e-14,rtol=1e-12)
                or max(abs(cm-np.asarray(meta["frequencies_cm1"])))>1e-10
                or not np.all(cm>0) or len(cm)!=3*n-6 or meta.get("same_MF_fingerprint_after")!=fingerprint
                or not np.allclose(arrays["gradient"],fresh,atol=1e-8,rtol=0)):
            raise ValueError("Validation full matrix/modes/unchanged MF checks failed")
        for key,name in (("raw_hessian_sha256","hessian_hartree_per_bohr2.npy"),
                         ("analysis_hessian_sha256","hessian_used_for_analysis.npy"),("normal_modes_sha256","normal_modes.npy")):
            pin(at/name,hashes,meta[key])
        replay_cm,replay_modes=CORE.replay_harmonic(symbols,geometry/CORE.BOHR_TO_ANGSTROM,used)
        if not np.allclose(cm,replay_cm,atol=1e-6,rtol=1e-9): raise ValueError("Validation saved Hessian frequency replay differs")
        mode_check=compare_mode_subspaces(cm,arrays["modes"],replay_modes)
        cases.append(arrays)
        timings.append({"identity":identity,"case":label,"actual_worker_process_CPU_seconds":meta["CPU_seconds"],
            "actual_wall_seconds":meta["wall_seconds"],"normal_modes":len(cm),"scope":"SOFTWARE_CONTROL_ONLY_SINGLE_RUN_NOT_SPINOSIN_SPEEDUP"})
        write_data={"case":label,"saved_matrix_replay":"PASS",**mode_check}
        arrays["audit"]=write_data
    rawdiff=float(np.max(np.abs(cases[0]["raw"]-cases[1]["raw"]))); freqdiff=float(np.max(np.abs(cases[0]["cm"]-cases[1]["cm"])))
    if rawdiff>1e-6 or freqdiff>.01: raise ValueError("Closed pure-J full-matrix equivalence gate failed")
    return {"identity":identity,"validation_only":True,"raw_hessian_max_difference":rawdiff,
        "frequency_max_difference_cm1":freqdiff,"all_atom_pairs_compared":True,"actual_modes_each_case":3*n-6,
        "all_modes_positive":True,"healthy_supervised_parent":healthy_parent(row),"replay":[a["audit"] for a in cases]},timings


def render_report(summary,phases,timings,output):
    lines=["# PySCF斯皮诺素纯PBE J-only批次实际结果","",
        "本报告只读取root明确释放的已关闭批次，未运行新的SCF、梯度或Hessian。", "",
        f"斯皮诺素实际计算频率记录：{summary['computed_Spinosin_frequency_rows_including_unvalidated']}；合格频率：{summary['validated_spinosin_frequency_rows']}；0.1–1.5 THz合格模式：{summary['validated_spinosin_low_band_modes']}。",
        f"状态：{summary['Spinosin_frequency_status']}。IR强度为NOT_COMPUTED。未算／失败的空表不代表零吸收。", "",
        "| 阶段 | 父进程状态 | worker状态 | 实际CPU秒 | 申请CPU秒 | CPU守卫停止 |", "|---|---|---|---:|---:|---|"]
    lines += [f"| {p['phase']} | {p['budget_status']} | {p['scientific_status']} | {p['actual_cpu_seconds']:.6f} | {p['configured_phase_CPU_seconds']} | {p['cpu_terminated']} |" for p in phases]
    lines += ["",f"新V4实际CPU {summary['V4_actual_CPU_seconds']:.6f}秒；历史V3 {summary['V3_actual_CPU_seconds']:.6f}秒；历史V2 {summary['V2_actual_CPU_seconds']:.6f}秒。",
        f"三批次明确范围累计 {summary['V2_V3_V4_actual_CPU_seconds_scope_excludes_original_pilot_and_nonquantum_tests']:.6f}秒；不含最初pilot、非量子测试或等待墙钟。新V4是独立36 CPU小时预算，不声称仍处于旧48小时池内。历史或新超额按实际值保留。", "",
        "| 软件验证分子 | 实际方案 | 实际CPU秒 | 实际墙钟秒 |", "|---|---|---:|---:|"]
    lines += [f"| {x['identity']} | {x['case']} | {x['actual_worker_process_CPU_seconds']:.6f} | {x['actual_wall_seconds']:.6f} |" for x in timings]
    lines += ["", "水和乙醇均只是软件控制，同一MF上的原完整partial与pure-J完整Hessian逐项比较。所有时序均保留，包括乙醇CPU略升；单次小分子时序不能推断75原子斯皮诺素加速，旧失败不作加速比的分母。", "",
        "computed_frequencies_including_unvalidated.csv保留实际正负频率。只有同方法旧优化父结果、健康实际监督父进程、当前fresh梯度、完整raw／sym Hessian、219模式、模式空间重放及全正频均合格后，才导出斯皮诺素解释表和振型动画。RHF e1／ej／zero-ek早期缓存不等于完整PBE Hessian；缓存恢复仍NOT_IMPLEMENTED_SAVE_ONLY。", "",
        f"实验位置表固定20条；完整候选×全部合格频段模式的笛卡尔对照共{summary['all_candidate_mode_pair_rows']}条。没有频率缩放拟合、最近峰筛选或测试集选峰。频率接近不构成化学归属确认。", "",
        "模型是一个孤立中性斯皮诺素构象，PBE/def2-SVP、grid1、无DF、无D3，尚未研究系统构象／基组／网格收敛。药材固态的晶格、分子间氢键、集体振动与复杂基质没有由本计算代表；需要纯品和化学测定验证。模式动画幅度任意可视缩放，不代表热振幅或IR强度。", "",
        "本工作不修改酸枣仁分类器或96.0234%准确率，不声称DFT提高分类准确率。CPU／RSS为尽力轮询防护，历史启动proof并不证明硬RSS保证或旧PID仍活跃。", "",
        "输出包括signed原始频率、模式索引、父优化实际轨迹、失败与资源记录、原矩阵重放和全部输入SHA。所有源文件／旧实验保持不变。", "",
        "可重复的后处理命令（必须使用同一真实root release与新输出目录）：", "", "```sh",
        f"python scripts/analyze_pyscf_pure_j_results_20261101.py --run-root reports/{RUN_NAME} --output NEW_OUTPUT --release ROOT_FINAL_RELEASE.json --report docs/PYSCF_SPINOSIN_PURE_J_RESULTS_20261101.md", "```", ""]
    return "\n".join(lines)


def run(root,output,release_path,report_path=None,budget_validator=None):
    root,output,release_path=root.resolve(),output.resolve(),release_path.resolve()
    if output.exists() or not output.is_relative_to(root): raise FileExistsError("Use a fresh analysis output inside V4")
    if report_path is not None:
        report_path=report_path.resolve()
        if report_path!=(ROOT/"docs/PYSCF_SPINOSIN_PURE_J_RESULTS_20261101.md").resolve():
            raise ValueError("Only the new requested pure-J results document may be generated")
        if report_path.exists(): raise FileExistsError("Preserve existing report")
    ledger,authorization,lock,hashes,totals,prior_v3,prior_v2=authorize(root,read(release_path),budget_validator)
    pin(release_path,hashes); experimental=experimental_rows(hashes)
    computed=[]; qualified=[]; steps=[]; mode_sets=[]; phases=[]; failures=[]; proofs=[]; checks=[]; controls=[]; timings=[]; water_rows=[]
    control_success=True; artifact_index=[]
    for row in ledger["phases"]:
        proof=audit_closed_proof(root,row,hashes); proofs.append({"phase":row["phase"],**proof})
        location=CORE.phase_location(root,row); result=CORE.completed_phase_result(row,location,hashes)
        for name in ("result.json","input.json","progress.json","hessian_cache_manifest.json","rhf_helper_cache_manifest.json"):
            if location is not None and (location/name).is_file(): pin(location/name,hashes)
        identity=result.get("identity","UNKNOWN"); errors=[]; rows=[]; gradient={}
        if not result.get("worker_result_missing"): assert_entry(result,hashes)
        supervision=result.get("resource_supervision",{})
        if supervision and supervision.get("proof_sha256")!=proof.get("proof_sha256"):
            raise ValueError("Worker startup proof differs from actual closed parent")
        if not result.get("worker_result_missing") and not supervision: raise ValueError("Worker lacks archived startup proof")
        if identity in ("water","validation_ethanol"):
            try:
                timings.extend(control_case_timings(location,identity,hashes))
                control,timing=audit_control(location,result,row,hashes)
                if not proof["healthy_parent"]: raise ValueError("Software control parent not healthy")
                controls.append(control)
                if identity=="water":
                    water_rows.extend(frequency_rows(location/"pure_j_partial_B8_native4","water",row["phase"]+"_PURE_J_SOFTWARE_CONTROL",hashes))
            except (ValueError,KeyError,OSError) as exc: errors.append(str(exc));control_success=False
        elif identity=="spinosin" or option(row["command"],"--phase")=="spinosin-frequency":
            identity="spinosin"
            artifacts={}
            for name in ("frequencies.csv","normal_modes.npy","hessian_hartree_per_bohr2.npy","hessian_used_for_analysis.npy",
                         "normal_mode_geometry_angstrom.npy","normal_mode_geometry_bohr.npy","normal_mode_geometry.xyz"):
                if (location/name).is_file():
                    artifact=pin(location/name,hashes);artifacts[name]={"path":str(artifact),"sha256":hashes[str(artifact)],"bytes":artifact.stat().st_size}
            artifact_index.append({"phase":row["phase"],"identity":"spinosin","actual_closed_files":artifacts,
                "presence_is_not_qualification":True,"no_missing_file_created":True})
            try: rows=frequency_rows(location,identity,row["phase"],hashes)
            except (ValueError,KeyError,OSError) as exc: errors.append("frequency CSV: "+str(exc))
            scientific_ok=False; parent_ok=False
            try:
                parent,trajectory=parent_optimization_audit(root,row,result,prior_v3,hashes); steps.extend(trajectory)
                gradient=CORE.audit_final_gradient(location,result,hashes)
                audit_fresh_scf_log(location,result,hashes)
                checks.append({"phase":row["phase"],"parent_optimization":parent,"fresh_gradient":gradient})
                parent_ok=True
            except (ValueError,KeyError,OSError) as exc: errors.append("parent/freshG: "+str(exc))
            if rows:
                try:
                    global_audit=audit_global_manifest(location,result,hashes)
                    audited,modes,geometry,replay=CORE.load_modes(location,result,hashes)
                    replay_cm,replay_modes=CORE.replay_harmonic(geometry[0],np.load(location/"normal_mode_geometry_bohr.npy",allow_pickle=False),
                        np.load(location/"hessian_used_for_analysis.npy",allow_pickle=False))
                    mode_check=compare_mode_subspaces([r["frequency_cm1"] for r in rows],modes,replay_modes)
                    if len(audited)!=len(rows) or any(abs(a['frequency_cm1']-b['frequency_cm1'])>1e-10 for a,b in zip(audited,rows)):
                        raise ValueError("Actual CSV rows differ from full result arrays")
                    checks.append({"phase":row["phase"],"saved_matrix_replay":replay,"mode_subspace_check":mode_check,"global_kernel_metadata":global_audit})
                    scientific_ok=True
                except (ValueError,KeyError,OSError) as exc: errors.append("Hessian/modes: "+str(exc))
                accepted=qualified_modes(result,rows,parent_ok and proof["healthy_parent"],scientific_ok and control_success and {c["identity"] for c in controls}=={"water","validation_ethanol"})
                rows=[{**r,"accepted_for_comparison":accepted} for r in rows]
                computed.extend(rows)
                if accepted: qualified.extend(rows);mode_sets.append((row["phase"],rows,modes,geometry))
        elif identity!="UNKNOWN": errors.append("Unrecognized molecular identity")
        phases.append({"phase":row["phase"],"identity":identity,"budget_status":row["status"],
            "scientific_status":result.get("status","PARTIAL"),"frequencies_status":result.get("frequencies_status","NOT_RUN"),"worker_result_missing":result.get("worker_result_missing",False),
            "parent_healthy":proof["healthy_parent"],"actual_cpu_seconds":row["actual_cpu_seconds"],
            "configured_phase_CPU_seconds":row["granted_cpu_seconds"],"profiling_CPU_soft_seconds":row["cpu_soft_limit_seconds"],
            "profiling_CPU_hard_seconds":row["cpu_hard_limit_seconds"],"actual_sampled_guard_threshold_seconds":row["actual_cpu_stop_threshold_seconds"],
            "sampled_phase_CPU_seconds":row.get("peak_actual_cpu_seconds_sampled"),"actual_CPU_over_grant_seconds":max(0.,row["actual_cpu_seconds"]-row["granted_cpu_seconds"]),
            "wall_seconds":row["elapsed_wall_seconds"],"exit_code":row["exit_code"],"cpu_terminated":row.get("cpu_terminated",False),
            "memory_terminated":row.get("memory_terminated",False),"wall_terminated":row.get("wall_terminated",False),
            "monitor_failed":row.get("monitor_failed",False),"resource_stop_reason":row.get("resource_stop_reason"),
            "peak_group_RSS_bytes_sampled":row.get("peak_group_rss_bytes_sampled"),
            "stop_reason":result.get("stop_reason",row.get("failure")),"audit_errors":"; ".join(errors),"IR_intensity_status":"NOT_COMPUTED"})
        if errors or not proof["healthy_parent"] or result.get("status")!="DONE":
            failures.append({"phase":row["phase"],"identity":identity,"actual_parent_status":row["status"],
                "actual_parent_exit_code":row["exit_code"],"actual_worker_status":result.get("status","PARTIAL"),
                "worker_result_missing":result.get("worker_result_missing",False),"audit_errors":errors,
                "stop_reason":result.get("stop_reason",row.get("failure")),"no_result_json_fabricated":True})
    low=[r for r in qualified if .1<=r["frequency_thz"]<=1.5]; pairs=all_candidate_pairs(experimental,low)
    summary={"completed_at_utc":datetime.now(timezone.utc).isoformat(),"analysis_only":True,"new_SCF_gradient_Hessian_evaluations":0,
        **totals,"computed_Spinosin_frequency_rows_including_unvalidated":len(computed),"validated_spinosin_frequency_rows":len(qualified),
        "validated_spinosin_low_band_modes":len(low),"Spinosin_frequency_status":"QUALIFIED_ISOLATED_SINGLE_CONFORMER_MODES_ONLY" if qualified else ("COMPUTED_UNQUALIFIED_SIGNED_MODES_PRESERVED" if computed else "NO_COMPLETED_SPINOSIN_FREQUENCIES"),
        "experimental_candidate_rows":20,"all_candidate_mode_pair_rows":len(pairs),"failed_or_partial_phase_count":len(failures),
        "ended_phases_without_worker_result":sum(p["worker_result_missing"] for p in phases),"water_and_ethanol_are_software_controls_only":True,
        "Spinosin_new_geometry_optimizations":0,"actual_old_V3_Spinosin_paired_geometry_steps":len(steps),
        "IR_intensities_status":"NOT_COMPUTED","chemical_attribution_confirmed":False,"frequency_scaling_fitted":False,
        "classification_modified":False,"classification_accuracy_96p0234_percent_unchanged":True,
        "NOT_RUN_is_not_zero_absorption":True,"cache_restoration":"NOT_IMPLEMENTED_SAVE_ONLY",
        "RHF_e1_ej_zeroek_is_not_complete_PBE_Hessian":True,"no_75_atom_speedup_claim_from_controls":True,
        "actual_CPU_and_RSS_monitoring_are_best_effort":True,"input_sha256":hashes}
    for path,expected in hashes.items():
        if sha(path)!=expected: raise ValueError("Input evidence changed during analysis: "+path)
    output.mkdir(parents=True)
    for name,values,fields in (("phase_status_and_resources.csv",phases,list(phases[0]) if phases else ["phase"]),
        ("computed_frequencies_including_unvalidated.csv",computed,FREQUENCY_FIELDS),
        ("water_software_control_frequencies.csv",water_rows,FREQUENCY_FIELDS),
        ("spinosin_qualified_all_frequencies.csv",qualified,FREQUENCY_FIELDS),("spinosin_qualified_0p1_1p5_thz_modes.csv",low,FREQUENCY_FIELDS),
        ("experimental_candidate_positions.csv",experimental,list(experimental[0])),
        ("experiment_theory_all_pairs.csv",pairs,list(pairs[0]) if pairs else ["theoretical_mode","comparison_status"]),
        ("actual_old_V3_optimization_trajectory.csv",steps,list(steps[0]) if steps else ["phase","cycle","energy_hartree"]),
        ("software_control_case_timings.csv",timings,list(timings[0]) if timings else ["identity","case"])):
        csvwrite(output/name,values,fields)
    historical=[]
    for label,doc in (("V2",prior_v2),("V3",prior_v3),("V4",ledger)):
        for row in doc["phases"]:
            grant=row.get("granted_cpu_seconds",row.get("requested_cpu_seconds"))
            historical.append({"batch":label,"phase":row["phase"],"status":row["status"],"actual_CPU_seconds":row["actual_cpu_seconds"],
                "configured_grant_CPU_seconds":grant,"actual_over_grant_CPU_seconds":None if grant is None else max(0.,row["actual_cpu_seconds"]-grant),
                "actual_sampled_CPU_guard_configured":row.get("actual_cpu_guard_enabled",False),
                "actual_CPU_guard_sample_recorded":row.get("peak_actual_cpu_seconds_sampled") is not None,
                "sampled_CPU_peak":row.get("peak_actual_cpu_seconds_sampled"),
                "scope":"CLOSED_V2_V3_V4_ONLY_NOT_ORIGINAL_PILOT_OR_NONQUANTUM_TESTS"})
    csvwrite(output/"historical_and_new_actual_CPU_costs.csv",historical,list(historical[0]))
    write(output/"failure_evidence.json",failures);write(output/"archived_supervision_audit.json",proofs)
    write(output/"computed_mode_artifact_index.json",artifact_index)
    write(output/"full_matrix_and_parent_audits.json",checks);write(output/"software_controls_audit.json",controls)
    CORE.plot_reports(output,steps,water_rows,qualified,experimental,mode_sets)
    V3.computed_frequency_plot(output,computed)
    V3.cost_plot(output,phases)
    report=render_report(summary,phases,timings,output);(output/"README.md").write_text(report)
    if report_path is not None:
        report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(report)
    write(output/"summary.json",summary)
    write(output/"files_sha256.json",{str(p.relative_to(output)):{"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run-root","output","release"):p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--report",type=Path)
    args=p.parse_args(); result=run(args.run_root,args.output,args.release,args.report)
    print(json.dumps({k:v for k,v in result.items() if k!="input_sha256"},ensure_ascii=False))


if __name__=="__main__":main()
