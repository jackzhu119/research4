#!/usr/bin/env python3
"""Read-only-first, stdlib-only summary of the finite Research4 A003 queue.

No quantum code, model, installation, Git command or numeric array loader is
imported or executed.  A report is written only after the original queue,
guardians, workers, wrapper and Git have actually closed.  Binary checkpoints
are not read: their producer-declared hashes and sizes remain explicitly
distinct from the small artifacts rehashed by this postprocessor.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reports/cloud-followup-20261106-v1/queue-configs/host-a-004/config_ROOT_APPROVED.json"
GRANT = CONFIG.with_name("root_grant_ROOT_APPROVED.json")
MAX_JSON = 8 * 1024**2
MAX_BINARY_HASH = 1024**2
MAX_TEXT_HASH = 8 * 1024**2
NEVER_READ = {".chk", ".h5", ".hdf", ".hdf5", ".npz", ".joblib", ".pkl", ".pt", ".pth"}
ARRAY_SUFFIXES = {".npy", ".bin", ".dat"}
TERMINAL_INVOCATIONS = {"HEALTHY_BOUNDARY_CLOSED", "COMPLETED", "COMPLETED_WITH_BLOCKED_JOBS",
                        "BLOCKED_REQUIRES_ROOT_REVIEW"}
TERMINAL_JOBS = {"DONE_SEALED", "FAILED_CLOSED", "PARTIAL_SCIENTIFIC_CLOSED",
                 "BLOCKED_DEPENDENCY_NOT_SUCCESS", "BLOCKED_RESOURCE_FEASIBILITY", "NOT_RUN"}
CLOSED_PHASES = {"EXITED", "PARTIAL_OR_FAILED", "FAILED_TO_START"}
GRID_CAVEAT = ("Rebuilt atom-centred grids with gradient_grid_response=False do not form an exact "
               "common fixed-quadrature finite difference. A discrepancy alone does not identify its cause. "
               "Exact sign disagreements near zero are not a physical significance test. No new PASS/FAIL threshold is set.")
OLD_RESULT_SHA = "253c10a92c083f54aaeb3f1e407e095790649737f5f876bd3cecfd35bfb4f9b7"
LEGACY_RESIDUAL_WORKER_SHA = "6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c"


class EvidenceError(RuntimeError):
    pass


class PendingWork(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise EvidenceError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def location(value):
    supplied = Path(value)
    require(not supplied.is_symlink(), "Symlink metadata/artifact rejected")
    path = supplied.resolve() if supplied.is_absolute() else (ROOT / supplied).resolve()
    require(path.is_relative_to(ROOT), "Evidence must remain in the Research4 checkout")
    return path


def digest_small(path):
    path = location(path)
    require(path.suffix.lower() not in NEVER_READ, "Checkpoint content is forbidden to this summary")
    limit = MAX_BINARY_HASH if path.suffix.lower() in ARRAY_SUFFIXES else MAX_TEXT_HASH
    require(path.stat().st_size <= limit, "Large artifact is metadata-only, never read by summary")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def metadata(path, expected=None):
    path = location(path)
    require(path.is_file() and path.suffix == ".json" and path.stat().st_size <= MAX_JSON,
            "Small explicit JSON metadata required")
    raw = path.read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected, "Metadata SHA mismatch: " + str(path))
    value = json.loads(raw)
    require(isinstance(value, (dict, list)), "Structured JSON metadata required")
    return value


def ref(path):
    path = location(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest_small(path), "bytes": path.stat().st_size}


def verify_reference(record, *, base=None):
    require(isinstance(record, dict) and isinstance(record.get("path"), str)
            and re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")), "Explicit artifact path/SHA required")
    raw = Path(record["path"])
    path = location((Path(base) / raw) if base is not None and not raw.is_absolute() else raw)
    require(path.is_file(), "Missing sealed artifact: " + str(path))
    declared_size = record.get("bytes", record.get("size_bytes"))
    size = path.stat().st_size
    require(declared_size is None or declared_size == size, "Sealed artifact size changed: " + str(path))
    forbidden = path.suffix.lower() in NEVER_READ
    oversized = size > (MAX_BINARY_HASH if path.suffix.lower() in ARRAY_SUFFIXES else MAX_TEXT_HASH)
    if forbidden or oversized:
        return {"path": str(path.relative_to(ROOT)), "declared_sha256": record["sha256"], "bytes": size,
                "verification": "PRODUCER_DECLARED_HASH_AND_CURRENT_SIZE_ONLY_CONTENT_NOT_READ",
                "reason": "FORBIDDEN_CHECKPOINT" if forbidden else "OVERSIZED_ARTIFACT"}
    actual = digest_small(path)
    require(actual == record["sha256"], "Sealed small artifact SHA changed: " + str(path))
    return {"path": str(path.relative_to(ROOT)), "sha256": actual, "bytes": size,
            "verification": "ACTUAL_SMALL_FILE_SHA256_RECOMPUTED_NO_ARRAY_LOADING"}


def load_context(config_path, grant_path):
    config_path, grant_path = location(config_path), location(grant_path)
    config, grant = metadata(config_path), metadata(grant_path)
    require(config.get("schema") == "research4-cloud-followup-queue-config-v1"
            and grant.get("schema") == "research4-cloud-followup-root-grant-v1", "Unsupported A queue metadata")
    config_sha, grant_sha = digest_small(config_path), digest_small(grant_path)
    require(grant.get("config_sha256") == config_sha and grant.get("root_execution_approved") is True,
            "Actual root config grant differs")
    state_exists = location(config["state"]).exists()
    state = metadata(config["state"]) if state_exists else {
        "config_sha256": config_sha, "grant_sha256": grant_sha,
        "invocations": [], "jobs": {}, "actual_queue_state_exists": False}
    require(state.get("config_sha256") == config_sha and state.get("grant_sha256") == grant_sha,
            "Actual state must bind this config and grant")
    ledger = metadata(config["ledger"])
    require(ledger.get("quantum_cpu_cap_seconds") == config["quantum_cpu_cap_seconds"], "Selected cumulative pool changed")
    require(len({j["id"] for j in config["jobs"]}) == len(config["jobs"]), "Duplicate configured job")
    return config, grant, state, ledger


def reuse_header(config):
    """Cheap read-only inventory; full closed-producer checks occur in execute."""
    if not config.get("reused_evidence"):
        return {"records": 0, "job_ids": [], "full_provenance_rechecked_in_inventory": False}
    record = config["reused_evidence"]
    document = metadata(record["path"], record["sha256"])
    require(document.get("schema") == "research4-root-audited-reused-evidence-v1"
            and document.get("root_approved") is True, "Root audited reuse inventory required")
    ids = [item["job"]["id"] for item in document["records"]]
    require(len(ids) == len(set(ids)) and not set(ids) & {j["id"] for j in config["jobs"]},
            "Current and reused job inventories overlap")
    return {"records": len(ids), "job_ids": ids, "registry_SHA_verified": True,
            "full_provenance_rechecked_in_inventory": False}


def readiness_reasons(config, state, ledger):
    reasons = []
    invocations = state.get("invocations", [])
    if not invocations:
        reasons.append("NO_ACTUAL_QUEUE_INVOCATION")
    for invocation in invocations:
        if invocation.get("status") not in TERMINAL_INVOCATIONS:
            reasons.append("QUEUE_INVOCATION_NOT_TERMINAL:" + str(invocation.get("id")))
    for job in config["jobs"]:
        status = state.get("jobs", {}).get(job["id"], {}).get("status")
        if status not in TERMINAL_JOBS:
            reasons.append("SCIENCE_JOB_PENDING_OR_UNKNOWN:" + job["id"] + ":" + str(status))
    for phase in ledger.get("phases", []):
        if phase.get("status") not in CLOSED_PHASES:
            reasons.append("LEDGER_PHASE_OPEN_OR_UNKNOWN:" + str(phase.get("phase")))
    return reasons


def forensic_hints(config, state, ledger):
    """Describe stale metadata without releasing it as completed science."""
    hints = []
    for job in config["jobs"]:
        saved = state.get("jobs", {}).get(job["id"], {})
        rows = [p for p in ledger.get("phases", []) if p.get("phase") == saved.get("phase")]
        if saved.get("status") == "RUNNING" and len(rows) == 1 and rows[0].get("status") in CLOSED_PHASES:
            hints.append({"job_id": job["id"], "original_queue_status": "RUNNING",
                          "actual_ledger_status": rows[0]["status"],
                          "classification": "PARTIAL_FORENSIC_REQUIRED_NOT_SCIENTIFIC_RELEASE",
                          "PID_absence_or_closed_row_does_not_upgrade_job": True})
    return hints


def proc_identity(pid):
    proc = Path("/proc") / str(pid)
    first = (proc / "stat").read_text(); fields = first[first.rfind(")")+2:].split()
    uid = next(line for line in (proc / "status").read_text().splitlines() if line.startswith("Uid:")).split()[1:]
    second = (proc / "stat").read_text(); repeat = second[second.rfind(")")+2:].split()
    require(fields[19] == repeat[19] and len(uid) == 4 and len(set(uid)) == 1, "Process identity changed during scan")
    return {"pid": int(pid), "start_ticks": int(fields[19]), "uid": int(uid[0]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "state": fields[0], "comm": first[first.find("(")+1:first.rfind(")")],
            "pgrp": int(fields[2]), "session": int(fields[3]),
            "CPU_seconds": sum(int(fields[i]) for i in (11,12,13,14))/os.sysconf("SC_CLK_TCK")}


def process_conflict(args):
    if not args:
        return None
    executable = Path(args[0]).name
    if executable == "git" or executable.startswith("git-"):
        return "GIT_ACTIVE"
    if executable.startswith(("python", "pypy")):
        scripts = [Path(a).name for a in args[1:] if a.endswith(".py")]
        for name in scripts:
            if name == Path(__file__).name:
                continue
            if name.startswith(("run_research4_host_a_", "research4_cloud_followup_queue_")):
                return "ORIGINAL_WRAPPER_OR_QUEUE_ACTIVE"
            if any(s in name for s in ("pyscf", "cloud_followup_worker", "cloud_orbital_stability",
                                       "cloud_small_molecule", "parallel_b_quantum", "parallel_b_supervisor")):
                return "QUANTUM_OR_GUARD_PROCESS_ACTIVE"
    return None


def no_active_science_or_git(grant):
    conflicts, observations = [], []
    exceptions = grant.get("opaque_process_identity_exceptions", [])
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            identity = proc_identity(int(proc.name))
            if identity["state"] in ("Z", "X", "x"):
                continue
            try:
                args = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            except PermissionError:
                repeated = proc_identity(int(proc.name))
                keys = ("pid", "start_ticks", "uid", "comm", "boot_id")
                matched = (all(identity[k] == repeated[k] for k in keys) and identity["uid"] == 0
                           and identity["comm"] in ("dockerd", "containerd")
                           and any(all(identity[k] == item.get(k) for k in keys) for item in exceptions))
                require(matched, "Opaque process identity is not the exact root-granted daemon")
                observations.append({**identity, "category": "EXACT_ROOT_GRANTED_OPAQUE_DAEMON"})
                continue
            category = process_conflict(args)
            if category:
                conflicts.append({**identity, "category": category,
                                  "script_basenames": [Path(a).name for a in args[1:] if a.endswith(".py")]})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise EvidenceError("Process visibility cannot establish actual closure") from error
    if conflicts:
        raise PendingWork("Actual PID/birth/UID/boot scan still sees active science, wrapper or Git: "
                          + json.dumps(conflicts, ensure_ascii=False))
    return {"verified_at_utc": utc(), "live_conflicts": [], "opaque_daemon_observations": observations,
            "full_command_lines_or_environment_printed": False}


def dotted(value, key):
    for part in key.split("."):
        require(isinstance(value, dict) and part in value, "Missing declared result field: " + key)
        value = value[part]
    return value


def status_label(raw):
    if raw == "DONE_SEALED":
        return "DONE"
    if raw in ("PARTIAL_SCIENTIFIC_CLOSED", "FAILED_CLOSED"):
        return "PARTIAL"
    if str(raw).startswith("BLOCKED"):
        return "BLOCKED"
    return "NOT_RUN"


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def verify_job(job, saved, ledger, evidence):
    seal = saved.get("seal") or saved.get("transport_seal")
    if not seal:
        return None
    require(isinstance(seal.get("result"), dict), "Saved job seal lacks result")
    evidence.append(verify_reference(seal["result"]))
    result = metadata(seal["result"]["path"], seal["result"]["sha256"])
    require(result.get("schema") == job["result_schema"], "Actual result schema differs")
    producer = result.get("source_worker", {})
    require(location(producer.get("path", "")) == location(job["worker"]["path"])
            and producer.get("sha256") == job["worker"]["sha256"], "Actual result source binding differs")
    evidence.append(verify_reference(job["worker"]))
    output = location(saved["output"])
    require(location(seal["result"]["path"]) == output / "result.json", "Seal result is outside its actual output")
    for artifact in seal.get("artifacts", []):
        relative = Path(artifact["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe sealed artifact path")
        evidence.append(verify_reference({"path": str(output / relative),
                                         "sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"]}))
    if seal.get("log"):
        evidence.append(verify_reference(seal["log"]))
    if job["kind"] == "quantum":
        rows = [row for row in ledger["phases"] if row["phase"] == seal.get("phase") == saved.get("phase")]
        require(len(rows) == 1 and rows[0]["status"] == "EXITED" and rows[0]["exit_code"] == 0,
                "Scientific seal lacks a real successful closed quantum phase")
        row = rows[0]
        require(row_hash(row) == seal["closed_row_sha256"], "Closed CPU ledger row changed")
        require(math.isclose(float(row["actual_cpu_seconds"]), float(seal["actual_cpu_seconds"]), abs_tol=1e-9),
                "Seal and actual ledger CPU disagree")
        evidence.append(verify_reference(seal["proof"]))
        proof = metadata(seal["proof"]["path"], seal["proof"]["sha256"])
        supervision = result.get("resource_supervision", {})
        require(supervision.get("proof") == proof and supervision.get("proof_sha256") == seal["proof"]["sha256"]
                and supervision.get("live_parent_process_and_source_verified") is True,
                "Actual worker did not verify its authentic live supervisor proof")
        require(proof.get("phase") == row["phase"]
                and location(proof["ledger_path"]) == location(ledger["continuation_ledger_path"])
                and proof.get("supervisor_script_sha256") == ledger["runner_sha256"], "Wrong actual producer proof lineage")
        require(row.get("supervisor_proof_sha256") == seal["proof"]["sha256"]
                and row.get("log_sha256") == seal["log"]["sha256"], "Proof/log differs from actual closed ledger")
        command = row.get("command", [])
        require(command.count("--output") == 1
                and location(command[command.index("--output")+1]) == output,
                "Actual closed producer command does not bind the output")
        if result.get("output_directory") is not None:
            require(location(result["output_directory"]) == output, "Actual scientific output declaration differs")
        else:
            # Fixed 6ff is a known, frozen legacy producer. Its input JSON
            # binds the actual output even though result.json lacks the field.
            # No other producer receives this explicitly scoped compatibility.
            require(job["worker"]["sha256"] == LEGACY_RESIDUAL_WORKER_SHA
                    and result.get("schema") == "research4-saved-cphf-equation-residual-audit-v1",
                    "Missing scientific output field from an unqualified producer")
            source_input = output/"input.json"
            evidence.append(verify_reference({"path": str(source_input), "sha256": result.get("input_json_sha256", "")}))
            original_input = metadata(source_input, result["input_json_sha256"])
            require(location(original_input["arguments"]["output"]) == output,
                    "Frozen legacy producer input output differs from actual command")
    else:
        require(result.get("new_quantum_evaluations") == 0 and seal.get("new_quantum_evaluations") == 0,
                "Analysis-only source must declare no new quantum evaluations")
    if saved["status"] == "DONE_SEALED":
        require(result.get("status") in job["success_statuses"], "DONE seal does not meet its declared scientific status")
        for key, expected in job.get("required_result_fields", {}).items():
            require(dotted(result, key) == expected, "DONE result violates declared field: " + key)
    return result


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), "Finite measured scalar required: " + label)
    return float(value)


def sign(value):
    return -1 if value < 0 else 1 if value > 0 else 0


def fd_row(job_id, result, origin="A003"):
    pair = result["displacement_pair"]
    analytic = finite(result["analytic_directional_curvature_Eh_per_Bohr2"], "analytic curvature")
    gradient = finite(result["finite_gradient_directional_curvature_Eh_per_Bohr2"], "FD gradient curvature")
    energy = finite(result["finite_energy_directional_curvature_Eh_per_Bohr2"], "FD energy curvature")
    return {"origin": origin, "job_id": job_id, "case": result["case"],
            "mode_index_1based": pair.get("mode_index_1based"),
            "coordinate_index_0based": pair.get("coordinate_index_0based"),
            "step_bohr": finite(pair["step_bohr"], "step"),
            "analytic_curvature_Eh_per_Bohr2": analytic, "FD_gradient_curvature_Eh_per_Bohr2": gradient,
            "FD_energy_curvature_Eh_per_Bohr2": energy, "analytic_sign": sign(analytic),
            "FD_gradient_sign": sign(gradient), "exact_sign_disagreement": sign(analytic) != sign(gradient),
            "absolute_curvature_difference_Eh_per_Bohr2": abs(gradient-analytic),
            "source_relative_directional_difference": result.get("relative_directional_difference"),
            "Hq_max_abs_difference_Eh_per_Bohr2": result.get("Hq_max_abs_difference_Eh_per_Bohr2"),
            "source_signed_frequency_cm1": result.get("source_signed_frequency_cm1"),
            "new_numeric_PASS_FAIL_threshold": "NOT_SET",
            "grid_caveat": result.get("grid_diagnostic_limitation", GRID_CAVEAT)}


def step_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"], row["mode_index_1based"], row["coordinate_index_0based"]].append(row)
    output = []
    for (case, mode, coordinate), values in grouped.items():
        for a, b in itertools.combinations(sorted(values, key=lambda x: x["step_bohr"], reverse=True), 2):
            if a["step_bohr"] == b["step_bohr"]:
                continue
            ga, gb = a["FD_gradient_curvature_Eh_per_Bohr2"], b["FD_gradient_curvature_Eh_per_Bohr2"]
            denominator = max(abs(ga), abs(gb))
            output.append({"case": case, "mode_index_1based": mode, "coordinate_index_0based": coordinate,
                           "larger_step_bohr": a["step_bohr"], "smaller_step_bohr": b["step_bohr"],
                           "FD_gradient_absolute_step_difference_Eh_per_Bohr2": abs(ga-gb),
                           "FD_gradient_relative_step_difference": abs(ga-gb)/denominator if denominator else None,
                           "FD_energy_absolute_step_difference_Eh_per_Bohr2": abs(a["FD_energy_curvature_Eh_per_Bohr2"]-b["FD_energy_curvature_Eh_per_Bohr2"]),
                           "step_consistency_PASS_FAIL": "NOT_SET_NO_EXTRAPOLATION_OR_CAUSE_ASSIGNMENT"})
    return output


def residual_rows(job_id, result):
    rows = result.get("metrics_by_nuclear_RHS", [])
    require(isinstance(rows, list) and result.get("checked_nuclear_RHS") == len(rows), "Measured residual RHS count differs")
    selected = result.get("selected_atom_ids", [])
    pairs = [(row["atom_id"], row["axis"]) for row in rows]
    require(len(set(pairs)) == len(pairs) and all(atom in selected and axis in ("x", "y", "z") for atom, axis in pairs),
            "Duplicate/invalid measured nuclear RHS")
    output = []
    for row in rows:
        output.append({"job_id": job_id, "atom_index_zero_based": row["atom_id"], "axis": row["axis"],
                       "virtual_max_abs_Eh_per_Bohr": finite(row["virtual"]["max_abs"], "residual"),
                       "virtual_RMS_Eh_per_Bohr": finite(row["virtual"]["RMS"], "residual RMS"),
                       "preconditioned_max_abs_per_Bohr": finite(row["preconditioned"]["max_abs"], "preconditioned residual"),
                       "occupied_gauge_max_abs_per_Bohr": finite(row["occupied_gauge"]["max_abs"], "occupied gauge"),
                       "relative_term_scaled_residual": row.get("relative_term_scaled_residual"),
                       "new_convergence_threshold": "NOT_SET_NOT_A_PASS_FAIL_GATE"})
    return output


def residual_coverage(rows):
    unique = {(row["atom_index_zero_based"], row["axis"]) for row in rows}
    require(all(0 <= atom < 75 and axis in ("x", "y", "z") for atom, axis in unique),
            "Residual direction outside the original 75-atom molecule")
    return {"unique_measured_nuclear_RHS": len(unique), "total_molecular_nuclear_RHS": 225,
            "duplicate_RHS_rows_not_counted_as_extra_coverage": len(rows)-len(unique),
            "all225_measured": len(unique) == 225, "numerical_convergence_PASS_FAIL": "NOT_SET"}


def orbital_rows(job_id, result):
    davidson = result.get("davidson", {})
    roots = davidson.get("actual_returned_roots_Eh", [])
    flags = davidson.get("actual_returned_convergence_flags", [])
    require(len(roots) == len(flags), "Orbital returned root/convergence count differs")
    requested = davidson.get("effective_native_solver_policy", {}).get("nroots", 3)
    complete = (result.get("native_all_requested_roots_converged") is True
                and len(flags) == requested and bool(flags) and all(flag is True for flag in flags))
    verdict = result.get("internal_orbital_stability") if complete else "UNDETERMINED_NATIVE_SOLVER_NOT_FULLY_CONVERGED"
    return [{"job_id": job_id, "case": result.get("case"), "root_index_zero_based": i,
             "returned_root_Eh": finite(value, "orbital root"), "native_root_converged": flags[i],
             "all_requested_roots_converged": complete, "qualified_internal_orbital_verdict": verdict,
             "native_raw_stable_flag_is_not_a_convergence_claim": True,
             "final_root_residual_norm": "NOT_RETURNED_BY_NATIVE_API_NOT_COMPUTED",
             "nuclear_minimum_verdict": "NOT_ESTABLISHED_BY_ORBITAL_CHECK"} for i, value in enumerate(roots)]


def contextual_controls(config, ledger, evidence):
    fd, orbital, context = [], [], []
    candidates = [r for r in config.get("immutable_inputs", [])
                  if Path(r["path"]).name == "host-a003-reused-controls-and-failure-forensics-001.json"]
    for record in candidates:
        evidence.append(verify_reference(record)); document = metadata(record["path"], record["sha256"])
        queue_ref = document["original_queue"]; evidence.append(verify_reference(queue_ref))
        old_state = metadata(queue_ref["path"], queue_ref["sha256"])
        for item in document["reused_actual_successes"]:
            evidence.append(verify_reference(item)); result = metadata(item["path"], item["sha256"])
            old = old_state["jobs"][item["job"]]; seal = old["seal"]
            require(old["status"] == "DONE_SEALED" and seal["result"]["sha256"] == item["sha256"], "Reused control seal differs")
            for artifact in seal.get("artifacts", []):
                evidence.append(verify_reference({"path": str(location(old["output"])/artifact["relative_path"]),
                                                 "sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"]}))
            if seal.get("proof"):
                evidence.append(verify_reference(seal["proof"])); rows = [r for r in ledger["phases"] if r["phase"] == seal["phase"]]
                require(len(rows) == 1 and row_hash(rows[0]) == seal["closed_row_sha256"], "Reused control real closure differs")
            if seal.get("log"):
                evidence.append(verify_reference(seal["log"]))
            context.append({"job_id": item["job"], "case": result.get("case"), "status": "DONE", "actual_result_status": result["status"]})
            if result.get("schema") == "research4-cloud-followup-directional-finite-difference-v1":
                fd.append(fd_row(item["job"], result, "REUSED_A002_WATER_CONTROL"))
            if result.get("schema") == "research4-native-internal-orbital-stability-diagnostic-v1":
                orbital.extend(orbital_rows(item["job"], result))
        ethanol = document["ethanol_source_correction"]
        for key in ("result", "proof", "log"):
            evidence.append(verify_reference(ethanol[key]))
        result = metadata(ethanol["result"]["path"], ethanol["result"]["sha256"])
        rows = [r for r in ledger["phases"] if r["phase"] == ethanol["actual_closed_phase"]]
        require(len(rows) == 1 and math.isclose(rows[0]["actual_cpu_seconds"], ethanol["actual_cpu_seconds"], abs_tol=1e-9),
                "Actual optional ethanol cost differs")
        orbital.extend(orbital_rows("reused-ethanol-orbital-stability", result))
        context.append({"job_id": "reused-ethanol-orbital-stability", "case": "ethanol", "status": "PARTIAL",
                        "actual_result_status": result["status"], "verdict": "UNDETERMINED_NATIVE_SOLVER_NOT_FULLY_CONVERGED"})
    return fd, orbital, context


def reused_records(config, ledger, evidence):
    """Accept old science only through an explicit, hash-bound ROOT audit.

    In particular, an old RUNNING job is never directly upgraded from PID
    absence. Its unchanged original state and the new forensic release remain
    separate facts in the output status matrix.
    """
    if not config.get("reused_evidence"):
        return [], [], []
    registry_ref = config["reused_evidence"]
    evidence.append(verify_reference(registry_ref))
    registry = metadata(registry_ref["path"], registry_ref["sha256"])
    require(registry.get("schema") == "research4-root-audited-reused-evidence-v1"
            and registry.get("root_approved") is True and isinstance(registry.get("forensic_reason"), str)
            and registry["forensic_reason"], "Explicit ROOT forensic reuse qualification required")
    for key in ("source_queue_config", "source_queue_grant", "source_queue_state", "metadata_adapter_source"):
        evidence.append(verify_reference(registry[key]))
    old_config = metadata(registry["source_queue_config"]["path"], registry["source_queue_config"]["sha256"])
    old_grant = metadata(registry["source_queue_grant"]["path"], registry["source_queue_grant"]["sha256"])
    old_state = metadata(registry["source_queue_state"]["path"], registry["source_queue_state"]["sha256"])
    require(old_state["config_sha256"] == registry["source_queue_config"]["sha256"]
            and old_grant["config_sha256"] == old_state["config_sha256"]
            and old_state["grant_sha256"] == registry["source_queue_grant"]["sha256"],
            "Original queue/source/state bindings differ")
    records = registry.get("records")
    require(isinstance(records, list) and 0 < len(records) <= 32, "Finite explicit reuse records required")
    old_jobs = {job["id"]: job for job in old_config["jobs"]}
    occupied = {job["id"] for job in config["jobs"]}; rows = []; scopes = []
    for record in records:
        job = record["job"]
        require(job["id"] not in occupied and old_jobs.get(job["id"]) == job,
                "Duplicate or altered forensic-reused job descriptor")
        occupied.add(job["id"])
        require(record.get("status") == "DONE_VERIFIED_FROM_CLOSED_PROVENANCE"
                and record.get("scientific_release") is True
                and type(record.get("legacy_output_binding")) is bool,
                "Reuse record has no actual scientific release")
        require(record.get("source_queue_state") == registry["source_queue_state"],
                "Reused record points to another original state")
        output = location(record["output_directory"]); seal = record["seal"]
        require(location(seal["result"]["path"]) == output/"result.json", "Forensic result/output differs")
        original = old_state.get("jobs", {}).get(job["id"])
        require(isinstance(original, dict) and location(original["output"]) == output,
                "Original job/output provenance missing")
        if record["legacy_output_binding"]:
            require(job["worker"]["sha256"] == LEGACY_RESIDUAL_WORKER_SHA
                    and seal.get("phase") == original.get("phase"), "Legacy compatibility has wrong producer/phase")
        else:
            require(original.get("status") == "DONE_SEALED" and original.get("seal") == seal,
                    "Non-legacy reused seal changed from actual completed source")
        saved = {"status": "DONE_SEALED", "output": str(output), "phase": seal.get("phase"), "seal": seal}
        result = verify_job(job, saved, ledger, evidence)
        rows.append({"job": job, "saved": saved, "result": result,
                     "original_queue_status": original.get("status"),
                     "forensic_reason": registry["forensic_reason"],
                     "source_queue_state": registry["source_queue_state"],
                     "legacy_output_binding": record["legacy_output_binding"]})
        scopes.append((old_config["phase_prefix"], job["id"]))
    return rows, scopes, old_config.get("immutable_inputs", [])


def collect(config, state, ledger, evidence):
    jobs, costs, fd, rhs, orbital = [], [], [], [], []
    reused, old_scopes, old_inputs = reused_records(config, ledger, evidence)
    scopes = [(config["phase_prefix"], job["id"]) for job in config["jobs"]] + old_scopes
    actual_phases = []
    for row in ledger["phases"]:
        matches = [job_id for prefix, job_id in scopes
                   if row["phase"].startswith(prefix+"-"+job_id+"-attempt")]
        require(len(matches) <= 1, "Actual phase ambiguously belongs to current/reused jobs")
        if matches:
            actual_phases.append((row, matches[0]))
    for row, job_id in actual_phases:
        cpu = finite(row.get("actual_cpu_seconds"), "actual closed phase CPU")
        require(cpu >= 0, "Negative closed CPU")
        costs.append({"phase": row["phase"], "job_id": job_id, "status": row["status"],
                      "actual_CPU_seconds": cpu, "exit_code": row.get("exit_code"),
                      "resource_stop_reason": row.get("resource_stop_reason"),
                      "CPU_terminated": row.get("cpu_terminated"), "wall_terminated": row.get("wall_terminated")})
    all_records = [{"job": job, "saved": state["jobs"][job["id"]], "origin": "CURRENT_CONFIG"}
                   for job in config["jobs"]]
    all_records += [{**record, "origin": "ROOT_AUDITED_REUSED_EVIDENCE"} for record in reused]
    protected_outputs = []
    for record in all_records:
        job, saved = record["job"], record["saved"]
        if saved.get("output"):
            protected_outputs.append(str(location(saved["output"])))
        result = record.get("result") if "result" in record else verify_job(job, saved, ledger, evidence)
        own_costs = [row for row in costs if row["job_id"] == job["id"]]
        jobs.append({"job_id": job["id"], "kind": job["kind"], "queue_status": saved["status"],
                     "origin": record["origin"],
                     "original_queue_status_preserved": record.get("original_queue_status", saved["status"]),
                     "forensic_reason": record.get("forensic_reason"),
                     "status": status_label(saved["status"]), "phase_count": len(own_costs),
                     "actual_closed_CPU_seconds": sum(row["actual_CPU_seconds"] for row in own_costs) if own_costs else None,
                     "CPU_scope": "SELECTED_CUMULATIVE_QUANTUM_LEDGER" if own_costs else "NOT_IN_QUANTUM_LEDGER_NOT_ZERO_MEASURED_CPU",
                     "actual_result_status": result.get("status") if result else None,
                     "result_sha256": (saved.get("seal") or saved.get("transport_seal") or {}).get("result", {}).get("sha256"),
                     "reason": saved.get("reason", saved.get("resource_stop_reason", saved.get("queue_status")))})
        if not result:
            continue
        schema = result.get("schema")
        if schema == "research4-cloud-followup-directional-finite-difference-v1":
            fd.append(fd_row(job["id"], result, record["origin"]))
        elif schema == "research4-saved-cphf-equation-residual-audit-v1":
            rhs.extend(residual_rows(job["id"], result))
        elif schema == "research4-native-internal-orbital-stability-diagnostic-v1":
            orbital.extend(orbital_rows(job["id"], result))
    combined_inputs = {}
    for record in config.get("immutable_inputs", []) + old_inputs:
        previous = combined_inputs.get(record["path"])
        require(previous is None or previous["sha256"] == record["sha256"], "Current/reused immutable evidence versions conflict")
        combined_inputs[record["path"]] = record
    context_config = {**config, "immutable_inputs": list(combined_inputs.values())}
    control_fd, control_orbital, controls = contextual_controls(context_config, ledger, evidence)
    fd.extend(control_fd); orbital.extend(control_orbital)
    historical_ref = next((r for r in combined_inputs.values() if r.get("sha256") == OLD_RESULT_SHA), None)
    require(historical_ref is not None, "Immutable original 219-mode result reference required")
    evidence.append(verify_reference(historical_ref)); historical = metadata(historical_ref["path"], OLD_RESULT_SHA)
    frequencies = historical["frequencies_cm1"]
    require(len(frequencies) == 219 and sum(x < 0 for x in frequencies) == 25
            and sum(x > 0 for x in frequencies) == 194 and historical["stationary_minimum_verified"] is False,
            "Original signed modes or qualification unexpectedly changed")
    return {"jobs": jobs, "costs": costs, "fd": fd, "steps": step_rows(fd), "rhs": rhs, "orbital": orbital,
            "protected_scientific_output_directories": protected_outputs,
            "controls": controls, "historical": {"source": historical_ref, "signed_modes": 219, "positive": 194,
                "imaginary": 25, "stationary_minimum_verified": False, "qualified_for_mechanism": False},
            "residual_coverage": residual_coverage(rhs)}


def csvwrite(path, rows, fallback):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fallback, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def jsonwrite(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")


def write_report(output, data, evidence, config_path, grant_path, config, ledger, process_observation):
    require(not output.exists(), "Fresh independent report directory required; never overwrite")
    require(not any(output.is_relative_to(location(p)) for p in data.get("protected_scientific_output_directories", [])),
            "Do not append a postprocessor inside an existing sealed scientific result directory")
    output.mkdir(parents=True)
    tables = [("job_status.csv", data["jobs"], ["job_id", "status"]),
              ("phase_costs.csv", data["costs"], ["phase", "actual_CPU_seconds"]),
              ("directional_curvatures.csv", data["fd"], ["job_id", "case", "step_bohr"]),
              ("step_dependence.csv", data["steps"], ["case", "mode_index_1based", "larger_step_bohr", "smaller_step_bohr"]),
              ("residual_RHS.csv", data["rhs"], ["job_id", "atom_index_zero_based", "axis"]),
              ("orbital_roots.csv", data["orbital"], ["job_id", "returned_root_Eh", "native_root_converged"])]
    for name, rows, fallback in tables:
        csvwrite(output/name, rows, fallback)
    counts = Counter(row["status"] for row in data["jobs"])
    disagreements = [row for row in data["fd"] if row["exact_sign_disagreement"]]
    binary_not_read = [row for row in evidence if row["verification"].startswith("PRODUCER_DECLARED")]
    result = {"schema": "research4-cloud-followup-postrun-analysis-v1", "status": "POSTRUN_ANALYSIS_COMPLETED",
              "created_at_utc": utc(), "source_worker": ref(__file__), "new_quantum_evaluations": 0, "models_fitted": 0,
              "source_config": ref(config_path), "source_grant": ref(grant_path), "source_state": ref(config["state"]),
              "selected_actual_ledger": ref(config["ledger"]), "all_selected_queue_jobs_settled": True,
              "actual_no_active_process_verification": process_observation,
              "status_matrix": data["jobs"], "status_counts": {key: counts.get(key, 0) for key in ("DONE", "PARTIAL", "NOT_RUN", "BLOCKED")},
              "resources": {"current_and_reused_phase_actual_CPU_seconds_including_failed_attempts": sum(row["actual_CPU_seconds"] for row in data["costs"]),
                  "cumulative_actual_CPU_seconds": ledger["consumed_cpu_seconds"], "fixed_cumulative_CPU_cap_seconds": ledger["quantum_cpu_cap_seconds"],
                  "remaining_CPU_seconds": ledger["remaining_quantum_cpu_seconds"], "analysis_job_CPU_not_measured_by_quantum_ledger": True,
                  "failed_costs_refunded_or_original_ledgers_modified": False},
              "original_Spinosin": data["historical"], "finite_difference_rows": len(data["fd"]),
              "exact_sign_disagreements": disagreements, "step_dependence": data["steps"],
              "finite_difference_numeric_PASS_FAIL_threshold": "NOT_SET", "grid_caveat": GRID_CAVEAT,
              "cause_of_original_negative_modes": "NOT_ESTABLISHED_BY_THESE_DIAGNOSTICS_ALONE",
              "residual_coverage": data["residual_coverage"], "orbital_results": data["orbital"],
              "reused_control_statuses": data["controls"], "small_artifact_verification": evidence,
              "checkpoints_or_large_arrays_not_read": binary_not_read,
              "IR_intensities_status": "NOT_COMPUTED", "chemical_assignment_verified": False,
              "theory_enters_classifier": False, "historical_classifier_accuracy": "96.0234% prior internal CV only, unchanged",
              "standard_THZ_HPLC_LCMS_new_batches": "NOT_RUN_NO_NEW_LAB_DATA",
              "no_quantum_Git_install_ledger_model_label_or_main_changes_by_postprocessor": True}
    jsonwrite(output/"result.json", result)
    lines = ["# 实际闭合队列与ROOT审核复用结果的异常汇总", "",
             "本报告只汇总已保存结果；未启动量子计算、训练、安装或Git，未修改原账本或历史结果。", "",
             "原Spinosin结果仍为219模式＝194正频＋25虚频，稳定极小点及峰归属资格未通过。", "",
             "| 项目 | 状态 | 实际结果状态 | 实际量子CPU秒（含该项目失败尝试） |", "|---|---|---|---:|"]
    for row in data["jobs"]:
        cpu = "未计入量子账本，不当作0" if row["actual_closed_CPU_seconds"] is None else str(row["actual_closed_CPU_seconds"])
        lines.append(f"| {row['job_id']} | {row['status']} | {row['actual_result_status'] or row['queue_status']} | {cpu} |")
    lines += ["", "## 实际方向曲率", "",
              "单位均为Eh/Bohr²；位移按Cartesian欧氏单位向量、步长为Bohr。下表保留符号，没有事后通过阈值。", "",
              "| 来源/项目 | 步长 | 解析曲率 | 梯度差分曲率 | 能量差分曲率 | 精确符号不同 |",
              "|---|---:|---:|---:|---:|---|"]
    for row in data["fd"]:
        lines.append(f"| {row['origin']}/{row['job_id']} | {row['step_bohr']} | {row['analytic_curvature_Eh_per_Bohr2']:.12g} | {row['FD_gradient_curvature_Eh_per_Bohr2']:.12g} | {row['FD_energy_curvature_Eh_per_Bohr2']:.12g} | {row['exact_sign_disagreement']} |")
    lines += ["", GRID_CAVEAT, "", "水控制中的相对差及近零分量符号也如实保留。DONE表示计算完成；原stock完整Hessian对照通过，不等于本次FD数值通过。",
              "", "## 响应与轨道检查", "",
              f"实际唯一核位移残差覆盖为 {data['residual_coverage']['unique_measured_nuclear_RHS']}/225；重复方向不重复计数。残差仅测量，未设置新的收敛PASS/FAIL阈值。",
              "轨道根逐一报告native收敛标志。仅原始stable标志或正根不足以判定已收敛，内部电子稳定性也不证明核坐标极小点。",
              "", "## 证据及边界", "",
              "小文件逐个重算SHA。CHK/HDF等检查点和大数组只记录原生产者封存SHA及当前大小，未读其内容，不能称已独立重哈希。",
              "所有失败成本保留。IR强度NOT_COMPUTED，无峰归属、无新分类准确率；96.0234%保持既有内部交叉验证语义。标准品、HPLC/LC-MS及独立新批次仍NOT_RUN。",
              ""]
    (output/"README.md").write_text("\n".join(lines))
    jsonwrite(output/"artifact_index.json", {"files": [ref(p) for p in sorted(output.iterdir()) if p.is_file()], "self_hash_included": False})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--grant", type=Path, default=GRANT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attempt", default=datetime.now(timezone.utc).strftime("postrun-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        require(not (args.check_only and args.execute), "Choose check-only or execute")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.attempt) is not None, "Safe report attempt name required")
        config, grant, state, ledger = load_context(args.config, args.grant)
        reasons = readiness_reasons(config, state, ledger)
        if args.check_only or not args.execute:
            reused = reuse_header(config)
            current_counts = Counter(state.get("jobs", {}).get(job["id"], {}).get("status", "NOT_RUN")
                                     for job in config["jobs"])
            print(json.dumps({"status": "PENDING_READ_ONLY" if reasons else "QUEUE_METADATA_CLOSED_REQUIRES_FINAL_PROCESS_SCAN",
                              "ready_by_metadata": not reasons, "pending_reasons": reasons,
                              "forensic_metadata_hints_not_scientific_release": forensic_hints(config, state, ledger),
                              "planned_jobs_total": len(config["jobs"])+reused["records"],
                              "current_config_jobs": len(config["jobs"]), "ROOT_audited_reused_inventory": reused,
                              "actual_current_job_status_counts": dict(current_counts),
                              "report_writes": 0, "new_quantum_evaluations": 0}, ensure_ascii=False))
            return 0
        if reasons:
            raise PendingWork("; ".join(reasons))
        observation = no_active_science_or_git(grant)
        evidence = []
        for record in config.get("source_refs", []) + config.get("immutable_inputs", []):
            evidence.append(verify_reference(record))
        data = collect(config, state, ledger, evidence)
        # Re-read actual state/ledger and process identities immediately before
        # creating the fresh report. No missing PID is treated as completion.
        config2, grant2, state2, ledger2 = load_context(args.config, args.grant)
        require(config2 == config and grant2 == grant and state2 == state and ledger2 == ledger,
                "Evidence changed during report preparation; retry read-only after closure")
        no_active_science_or_git(grant)
        output = location(args.output or ROOT/"reports/cloud-followup-analysis-20261106-v1"/args.attempt)
        require(output.is_relative_to(ROOT/"reports"), "Fresh report under repository reports required")
        result = write_report(output, data, evidence, args.config, args.grant, config, ledger, observation)
        print(json.dumps({"status": result["status"], "output": str(output), "status_counts": result["status_counts"],
                          "new_quantum_evaluations": 0}, ensure_ascii=False))
        return 0
    except PendingWork as error:
        print(json.dumps({"status": "BLOCKED_PENDING_SCIENCE_NO_REPORT_WRITES", "reason": str(error), "new_quantum_evaluations": 0}))
        return 3
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED_EVIDENCE_NO_SUCCESS_CLAIM", "reason": str(error), "error_type": type(error).__name__,
                          "new_quantum_evaluations": 0}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
