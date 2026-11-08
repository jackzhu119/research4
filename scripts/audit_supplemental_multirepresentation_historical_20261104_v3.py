"""Historical no-fit NN verification, with archived Q review and a fresh grant.

This utility does not authorize or expose old training commands. It preserves
the original NN authority, elapsed training deadline, actual v2 audit and live
quantum ledger. Numerical helpers are explicitly reused without monkeypatching
their authority, budgets, paths, or globals. Current Q is read only under its
real global lock by an actual, separately authorized audit invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "scripts/audit_supplemental_multirepresentation_20261104_v2.py"
OLD_SHA = "7ef9829efd2bdfefc75e4a7e59813f6713ba8779342176aedc3adca899bc6306"
LEAF = ROOT / "scripts/historical_supplemental_nn_numerical_20261104_v3.py"
QGUARD_NAME = "run_pyscf_continuation_budget_20261105_v6b.py"
QGUARD_SHA = "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5"
STAGES = ("prepare", "bind", "source", "fit", "predict", "score")
Q_MUTABLE = {"consumed_cpu_seconds", "remaining_quantum_cpu_seconds", "phases"}
NUMERIC = {"numpy", "torch", "pyscf", "scipy", "sklearn"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 16 * 1024**2, "Bounded metadata JSON required")
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")


def checked(record):
    path = Path(record["path"])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size <= 16 * 1024**2,
            "Bounded in-worktree NN metadata artifact required")
    require(sha(path) == record["sha256"], "Changed NN metadata artifact")
    if "bytes" in record:
        require(path.stat().st_size == record["bytes"], "Metadata byte count differs")
    return path


def load_frozen(path, expected_sha, name):
    require(Path(path).is_file() and sha(path) == expected_sha, "Frozen source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(not NUMERIC & set(sys.modules), "Metadata imports must remain numerical-package-free")
    return module


def aware(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset() is not None, "Recorded timezone-aware UTC timestamp required")
    return result.astimezone(timezone.utc)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def phase_costs(phases):
    require(isinstance(phases, list) and phases, "A nonempty complete Q phase list is required")
    names, values = set(), []
    for row in phases:
        name, cost = row.get("phase"), row.get("actual_cpu_seconds")
        require(name and name not in names, "Duplicate/missing Q phase identity")
        require(row.get("status") in {"EXITED", "PARTIAL_OR_FAILED"} and row.get("ended_at_utc")
                and not row.get("residual_processes_require_review_after_main_payload_exit"),
                "Active, unknown or unresolved Q phase; historical arrays cannot be read")
        require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0, "Unknown Q CPU is never zero")
        names.add(name); values.append(cost)
    return math.fsum(values)


def compare_append_only(archived, current):
    """Strict complete old prefix and unchanged caps/lineage; failed costs stay."""
    require(set(current) == set(archived) and Q_MUTABLE <= set(archived), "Q top-level schema changed")
    require(canonical({k: v for k, v in archived.items() if k not in Q_MUTABLE})
            == canonical({k: v for k, v in current.items() if k not in Q_MUTABLE}),
            "Q immutable source/caps/registry/lineage changed")
    original = archived["phases"]; phases = current["phases"]
    require(len(phases) >= len(original) and canonical(phases[:len(original)]) == canonical(original),
            "Archived Q phase prefix was altered, removed or relabelled")
    old_used, new_used = phase_costs(original), phase_costs(phases)
    for value in (archived, current):
        used = phase_costs(value["phases"])
        require(type(value["consumed_cpu_seconds"]) in (int, float)
                and math.isclose(used, value["consumed_cpu_seconds"], rel_tol=1e-12, abs_tol=1e-7), "Q closed CPU total differs")
        require(math.isclose(max(0., value["quantum_cpu_cap_seconds"] - used), value["remaining_quantum_cpu_seconds"],
                             rel_tol=1e-12, abs_tol=1e-7), "Q remaining budget differs")
    require(new_used >= old_used, "Q actual costs were rolled back")
    return {"status": "PASS_APPEND_ONLY_PREFIX_AND_ACCOUNTING", "archived_phase_count": len(original),
            "current_phase_count": len(phases), "new_closed_phase_count": len(phases) - len(original),
            "archived_consumed_CPU": old_used, "current_consumed_CPU": new_used,
            "failed_costs_retained": True, "Q_success_not_required_or_inferred": True}


def historical_time_checks(protocol, ledger, records):
    """Verify past production times against its past deadline, never today's clock."""
    deadline = aware(ledger["deadline_at_utc"])
    created, registered, continued = (aware(ledger["created_at_utc"]), aware(protocol["registered_at_utc"]),
                                      aware(ledger["continuation_registered_at_utc"]))
    require(created <= registered <= continued <= deadline, "Historical registration/deadline chain differs")
    require([r["stage"] for r in records] == list(STAGES), "Historical stage chronology incomplete")
    previous = continued; result = []
    for record in records:
        start, completed, ended, closed = (aware(record[k]) for k in ("started_at_utc", "completed_at_utc", "ended_at_utc", "closure_at_utc"))
        require(previous <= start <= completed <= ended <= closed <= deadline, "Historical stage exceeded/reordered its original deadline")
        require(record["proof_deadline_at_utc"] == ledger["deadline_at_utc"], "Original proof deadline was changed")
        previous = closed
        result.append({"stage": record["stage"], "original_deadline_pass": True, "closed_at_utc": record["closure_at_utc"]})
    return {"status": "PASS_HISTORICAL_PRODUCTION_WITHIN_ORIGINAL_DEADLINE", "original_deadline_at_utc": ledger["deadline_at_utc"],
            "stages": result, "current_clock_used_for_old_training_validity": False,
            "per_fit_UTC_not_recorded": True, "live_original_parent_PID_not_rechecked": True}


def verify_nn_history(run, old, archive_ref, preservation_ref):
    registration = read(run / "registration.json")
    authority_path = Path(registration["authority_path"]).resolve()
    require(authority_path.is_relative_to(ROOT) and sha(authority_path) == registration["authority_sha256"], "Original NN authority changed")
    authority = read(authority_path)
    require(authority.get("schema") == "research4-post-quantum-fusion-authority-v1" and authority.get("root_execution_authorized") is True
            and authority.get("quantum_terminal_reviewed") is True and authority.get("implementation_review_passed") is True,
            "Original NN authority was not an explicit terminal-Q release")
    require(Path(authority["authorized_output_path"]).resolve() == run, "Original NN authority bound another run")
    archive_path, receipt_path = checked(archive_ref), checked(preservation_ref)
    qref = authority["quantum_ledger"]
    require(archive_path != Path(qref["path"]).resolve() and sha(archive_path) == qref["sha256"]
            and archive_path.stat().st_size == qref["bytes"], "Archive is not the exact original terminal-Q bytes")
    preservation = read(receipt_path); archive = read(archive_path)
    require(preservation["schema"] == "research4-immutable-terminal-Q-review-preservation-v1"
            and preservation["status"] == "EXACT_SNAPSHOT_PRESERVED_AFTER_SUCCESSFUL_NN_AND_AUDIT_BEFORE_Q_CONTINUATION"
            and preservation["original_terminal_sha256"] == qref["sha256"]
            and Path(preservation["original_live_ledger_path"]).resolve() == Path(qref["path"]).resolve()
            and checked(preservation["snapshot"]) == archive_path,
            "Archived Q preservation receipt differs from the original authority")
    require(archive["runner_sha256"] == QGUARD_SHA
            and Path(archive["continuation_ledger_path"]).resolve() == Path(qref["path"]).resolve(), "Archive Q source/registered path differs")
    compare_append_only(archive, archive)
    for record in registration["code_refs"]:
        old.reference(record)
    protocol = read(old.reference(registration["protocol"]))
    stages, ledger = old.closed_chain(run)  # Frozen pure historical receipts/math, no old clock gate.
    completion = read(run / "completion.json")
    require(old.reference(completion["resource_ledger"]) == run / "resource_ledger.json"
            and old.reference(completion["results"]) == old.reference(stages["score"]["results"]),
            "Historical completion no longer binds the exact original ledger/results")
    scope = old.cumulative_scope(authority, ledger)
    records = []
    driver = protocol["execution_driver"]
    driver_path = old.reference(driver)
    for row in ledger["phases"]:
        closed = read(old.reference(row["closed_stage"]))
        proof_ref = closed["supervisor_proof"]; proof_path = old.reference(proof_ref); proof = read(proof_path)
        worker = read(old.reference(closed["worker_result"]))
        require(proof["stage"] == row["stage"] and Path(proof["output"]).resolve() == run
                and proof["driver_sha256"] == driver["sha256"], "Historical supervisor source/stage/output changed")
        command = [protocol["standard_library_environment"]["executable"], str(driver_path), "_worker", "--output", str(run),
                   "--stage", row["stage"], "--proof", str(proof_path), "--proof-sha256", proof_ref["sha256"]]
        require(row["command"] == command, "Historical supervised command changed")
        records.append({"stage": row["stage"], "started_at_utc": row["started_at_utc"], "ended_at_utc": row["ended_at_utc"],
            "completed_at_utc": worker["completed_at_utc"], "closure_at_utc": closed["closure_at_utc"],
            "proof_deadline_at_utc": proof["deadline_at_utc"]})
    timing = historical_time_checks(protocol, ledger, records)
    require(aware(preservation["created_at_utc"]) >= aware(records[-1]["closure_at_utc"]), "Preservation predates the completed NN job")
    source_lock = read(old.reference(stages["source"]["source_lock"]))
    models = read(old.reference(stages["fit"]["model_lock"]))
    predicted = read(old.reference(stages["predict"]["prediction_lock"]))
    require(predicted["models"] == stages["fit"]["model_lock"]
            and models["source_epoch_lock"] == stages["source"]["source_lock"], "Original NN locks were changed")
    context = {"stages": stages, "source_lock": source_lock, "models_lock": models, "prediction_lock": predicted,
               "source_manifest": read(old.reference(source_lock["source_manifest"])),
               "prediction_manifest": read(old.reference(predicted["prediction_manifest"]))}
    report = {"status": "PASS_ARCHIVE_AND_ORIGINAL_NN_PROVENANCE", "original_NN_authority_SHA": sha(authority_path),
        "archived_Q_SHA": sha(archive_path), "preservation_receipt_SHA": sha(receipt_path),
        "original_Q_prefix_phases": len(archive["phases"]), "historical_timing": timing,
        "original_NN_CPU_and_reserve_scope": scope, "old_training_permission_extended": False,
        "original_Q_live_SHA_not_required_for_archived_history": True}
    return context, report, authority, archive


def validate_current_grant(path, expected_sha, run, output, replay_models):
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size <= 256 * 1024, "Bounded fresh root audit grant required")
    require(sha(path) == expected_sha, "Fresh root audit grant missing/changed")
    grant = read(path)
    require(grant.get("schema") == "research4-historical-NN-audit-grant-v3" and grant.get("root_execution_authorized") is True
            and grant.get("historical_verification_only") is True and grant.get("training_authorized") is False,
            "A fresh no-training audit grant is required")
    require(Path(grant["historical_run"]).resolve() == run and Path(grant["authorized_output_path"]).resolve() == output,
            "Fresh grant binds another run/output")
    require(grant["audit_source_sha256"] == sha(__file__) and grant["numerical_leaf"]["sha256"] == sha(LEAF)
            and checked(grant["numerical_leaf"]) == LEAF and grant["frozen_helper_sha256"] == OLD_SHA
            and grant["frozen_Q_guard_sha256"] == QGUARD_SHA, "Fresh grant code pins differ")
    require(not replay_models or grant.get("allow_weight_replay") is True, "Weight replay not enabled by fresh root grant")
    now = datetime.now(timezone.utc); issued = aware(grant["issued_at_utc"]); expires = aware(grant["expires_at_utc"])
    authorization = grant["current_user_authorization"]
    evidence = read(checked(authorization))
    user_end = aware(evidence[authorization["deadline_field"]])
    require(issued <= now < expires <= user_end, "Fresh grant or current user authorization expired")
    limits = grant["limits"]
    require(type(limits["cpu_seconds"]) is int and 10 <= limits["cpu_seconds"] <= 180
            and type(limits["wall_seconds"]) is int and 10 <= limits["wall_seconds"] <= 600
            and limits["threads"] == 1 and limits["device"] == "cpu"
            and limits["RSS_bytes"] == int(1.5 * 1024**3) and limits["AS_bytes"] == 10 * 1024**3,
            "Fresh historical audit must be independently bounded180CPU/600wall")
    require(not output.exists() and output.is_relative_to(ROOT / "reports") and output != run
            and not output.is_relative_to(run) and not run.is_relative_to(output), "A fresh separate audit output is required")
    return grant


def verify_grant_anchors(grant, run):
    """All historical graph/hash work is charged inside the fresh budget."""
    anchors = grant["historical_NN_anchors"]
    require(set(anchors) == {"registration", "completion", "resource_ledger"}, "Explicit original NN metadata hashes required")
    for name, record in anchors.items():
        require(checked(record) == run / (name + ".json"), "Original NN anchor path/hash changed")
    prior_audit = grant["original_successful_audit"]
    old_summary = read(checked(prior_audit["summary"]))
    old_closure = read(checked(prior_audit["closure"]))
    require(old_summary["status"] == "PASS_INDEPENDENT_POST_Q_NUMERICAL_AUDIT"
            and old_summary["source_code_sha256"] == OLD_SHA
            and old_summary["native_model_replay"]["status"] == "DONE_ACTUAL_NO_FIT_REPLAY"
            and old_summary["native_model_replay"]["models_replayed"] == 36
            and old_closure["status"] == "EXITED_HEALTHY", "Original actual successful v2 audit is not preserved")


class FreshAuditBudget:
    """Only the new explicit audit grant; never consumes/resets the old NN/Q ledgers."""
    def __init__(self, output, grant, *, started_at=None):
        self.output, self.grant = output, grant
        self.started = time.monotonic() if started_at is None else started_at
        self.cpu = grant["limits"]["cpu_seconds"]; self.rss = grant["limits"]["RSS_bytes"]
        elapsed = time.monotonic() - self.started
        expiry = (aware(grant["expires_at_utc"]) - datetime.now(timezone.utc)).total_seconds()
        self.wall = min(grant["limits"]["wall_seconds"], elapsed + expiry)
        require(self.wall - elapsed >= 5, "Fresh audit grant has insufficient remaining time")

    def __enter__(self):
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGXCPU)}
        def stop(signum, frame):
            raise RuntimeError("FRESH_HISTORICAL_AUDIT_RESOURCE_LIMIT")
        for sig in self.handlers:
            signal.signal(sig, stop)
        old_cpu = resource.getrlimit(resource.RLIMIT_CPU)[1]
        hard_cpu = min(self.cpu, old_cpu) if old_cpu >= 0 else self.cpu
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, hard_cpu - 2), hard_cpu))
        old_as = resource.getrlimit(resource.RLIMIT_AS)[1]
        hard_as = min(old_as, self.grant["limits"]["AS_bytes"]) if old_as >= 0 else self.grant["limits"]["AS_bytes"]
        resource.setrlimit(resource.RLIMIT_AS, (hard_as, hard_as))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))}); os.environ["CUDA_VISIBLE_DEVICES"] = ""
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.wall - (time.monotonic() - self.started)))
        self.check(); return self

    def record(self):
        return {"actual_process_CPU_seconds_including_startup": time.process_time(), "wall_seconds": time.monotonic() - self.started,
            "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "fresh_grant_limits": self.grant["limits"], "fresh_grant_expires_at_utc": self.grant["expires_at_utc"],
            "old_NN_or_Q_budget_modified": False, "old_training_deadline_not_extended": True}

    def check(self):
        require(time.process_time() < self.cpu and time.monotonic() - self.started < self.wall
                and datetime.now(timezone.utc) < aware(self.grant["expires_at_utc"]), "Fresh audit CPU/time grant reached")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= self.rss, "Fresh audit best-effort RSS guard reached")

    def __exit__(self, kind, error, traceback):
        try:
            if self.output.exists():
                if kind is None:
                    self.check(); write_new(self.output / "historical_audit_closed.json", {"status": "EXITED_HEALTHY",
                        "resource": self.record(), "closure_at_utc": datetime.now(timezone.utc).isoformat(),
                        "snapshot_precedes_only_final_receipt_write_and_return": True})
                else:
                    write_new(self.output / "historical_audit_failure.json", {"status": "PARTIAL_OR_FAILED",
                        "error_type": kind.__name__, "error": str(error), "resource": self.record()})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
        return False


def execute(run, output, grant_path, grant_sha, replay_models):
    started = time.monotonic()
    require(run.is_relative_to(ROOT / "reports"), "Historical NN run must stay in its original worktree")
    grant = validate_current_grant(grant_path, grant_sha, run, output, replay_models)
    with FreshAuditBudget(output, grant, started_at=started) as budget:
        verify_grant_anchors(grant, run)
        old = load_frozen(OLD, OLD_SHA, "frozen_historical_nn_audit_7ef")
        context, history, original_authority, archive = verify_nn_history(run, old, grant["archived_Q_snapshot"], grant["archive_preservation_receipt"])
        budget.check()
        qroot = Path(original_authority["quantum_worktree_root"]).resolve()
        lock = Path(original_authority["quantum_global_lock"]).resolve()
        current_path = Path(original_authority["quantum_ledger"]["path"]).resolve()
        require(lock == qroot / ".runtime/pyscf-v4-global.lock" and lock.is_file() and current_path.is_relative_to(qroot / "reports"),
                "Must use the actual original Q global lock/ledger")
        guard_path = qroot / "scripts" / QGUARD_NAME
        require(sha(guard_path) == QGUARD_SHA, "Frozen Q metadata guard differs")
        with lock.open("r") as qlock:
            fcntl.flock(qlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Only here may this utility read the live mutable Q ledger.
            require(current_path.stat().st_size <= 16 * 1024**2, "Bounded current Q metadata required")
            current_bytes = current_path.read_bytes()
            current = json.loads(current_bytes)
            prefix = compare_append_only(archive, current)
            guard = load_frozen(guard_path, QGUARD_SHA, "frozen_Q_ledger_guard_b985")
            guard.validate_ledger(current, allow_running=False)
            require(current_path.read_bytes() == current_bytes, "Current Q ledger changed while its actual lock was held")
            prefix.update(current_live_Q_sha256=hashlib.sha256(current_bytes).hexdigest(), frozen_Q_guard_validation=True,
                          guard_scope="metadata/lineage/cost only; no Q scientific result arrays read")
            require(sha(LEAF) == grant["numerical_leaf"]["sha256"], "Historical numerical leaf changed")
            numerical = load_frozen(LEAF, grant["numerical_leaf"]["sha256"], "historical_nn_numerical_v3")
            output.mkdir(parents=True)
            write_new(output / "current_grant_claim.json", {"status": "CLAIMED_ONCE", "grant_sha256": grant_sha,
                "grant_path": str(Path(grant_path).resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "no_old_NN_training_permission": True})
            metadata = {"status": "PASS_HISTORICAL_METADATA_NEW_NUMERICAL_AUDIT_NOT_YET_COMPLETE", "history": history,
                        "current_Q_prefix": prefix, "fresh_grant_sha256": grant_sha, "resource": budget.record()}
            write_new(output / "historical_metadata_proof.json", metadata)
            context.update(historical_training_valid=True, current_Q_prefix_verified=True, current_grant_verified=True,
                current_grant_sha256=grant_sha, leaf_code_sha256=grant["numerical_leaf"]["sha256"])
            result = numerical.run_checks(run, output, old, context, budget, replay_models)
            budget.check(); require(current_path.read_bytes() == current_bytes, "Q metadata changed before historical audit closed")
            summary = {"status": "PASS_NEW_HISTORICAL_VERIFICATION_NO_TRAINING", "historical_training_deadline_remains_original": True,
                "current_replay_is_not_original_training": True, "original_authority_or_ledgers_or_prior_audits_modified": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(), "new_grant_sha256": grant_sha,
                "historical_metadata": metadata, "numerical_verification": result, "resource": budget.record(),
                "audit_source_sha256": sha(__file__), "frozen_numerical_helper_sha256": OLD_SHA,
                "numerical_leaf_sha256": grant["numerical_leaf"]["sha256"]}
            write_new(output / "historical_summary.json", summary)
            return summary


def plan(output=None):
    require(not NUMERIC & set(sys.modules), "Historical preparation must stay stdlib-only")
    value = {"status": "NOT_RUN_HISTORICAL_UTILITY_PREPARED_ONLY", "actual_arrays_or_current_Q_read": 0,
        "actual_model_fits": 0, "actual_weight_replays": 0, "old_training_command_available": False,
        "new_current_grant_required": True, "old_training_validity_uses_recorded_times_not_now": True,
        "live_Q_requires_unchanged_immutable_header_exact_full_archived_prefix_and_no_active_phase": True,
        "numerical_imports_deferred_until_real_Q_lock_and_historical_grant": True,
        "new_max_CPU_seconds": 180, "new_max_wall_seconds": 600, "old_NN_Q_budget_resets": False,
        "source_refs": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
                        for p in (Path(__file__), OLD, LEAF) if p.exists()]}
    if output is not None:
        write_new(output, value)
    return value


def grant_template(run, output):
    """Cold, non-authorizing template; root must issue current times/permission."""
    def record(path):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Template uses existing archived NN metadata only")
        return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}
    registration = read(run / "registration.json")
    original = Path(registration["authority_path"]).resolve()
    directory = original.parent
    old_audit = ROOT / "reports/supplemental-multirepresentation-independent-audit-20261104-v2"
    authorization = ROOT / "data/research/supplement_authorization_20261104.json"
    return {"schema": "research4-historical-NN-audit-grant-v3", "root_execution_authorized": False,
        "historical_verification_only": True, "training_authorized": False, "allow_weight_replay": True,
        "historical_run": str(run), "authorized_output_path": str(output),
        "issued_at_utc": "ROOT_MUST_ISSUE_CURRENT_UTC", "expires_at_utc": "ROOT_MUST_BOUND_BY_CURRENT_USER_AUTHORIZATION",
        "audit_source_sha256": sha(__file__), "numerical_leaf": record(LEAF), "frozen_helper_sha256": OLD_SHA,
        "frozen_Q_guard_sha256": QGUARD_SHA,
        "historical_NN_anchors": {name: record(run / (name + ".json")) for name in ("registration", "completion", "resource_ledger")},
        "original_successful_audit": {"summary": record(old_audit / "summary.json"), "closure": record(old_audit / "audit_stage_closed.json")},
        "archived_Q_snapshot": record(directory / "terminal-quantum-ledger-at-NN-release.json"),
        "archive_preservation_receipt": record(directory / "terminal-quantum-ledger-preservation.json"),
        "current_user_authorization": {**record(authorization), "deadline_field": "window_end_utc"},
        "limits": {"cpu_seconds": 180, "wall_seconds": 600, "threads": 1, "device": "cpu",
                   "RSS_bytes": 1610612736, "AS_bytes": 10737418240},
        "note": "TEMPLATE_NOT_EXECUTABLE; never reauthorizes or modifies old NN training/Q budgets"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cold = sub.add_parser("plan"); cold.add_argument("--output", type=Path)
    template = sub.add_parser("grant-template"); template.add_argument("--run", type=Path, required=True)
    template.add_argument("--audit-output", type=Path, required=True); template.add_argument("--output", type=Path, required=True)
    actual = sub.add_parser("audit"); actual.add_argument("--run", type=Path, required=True)
    actual.add_argument("--output", type=Path, required=True); actual.add_argument("--grant", type=Path, required=True)
    actual.add_argument("--grant-sha256", required=True); actual.add_argument("--replay-models", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.output)
    elif args.command == "grant-template":
        result = grant_template(args.run.resolve(), args.audit_output.resolve()); write_new(args.output, result)
    else:
        result = execute(args.run.resolve(), args.output.resolve(), args.grant, args.grant_sha256, args.replay_models)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
