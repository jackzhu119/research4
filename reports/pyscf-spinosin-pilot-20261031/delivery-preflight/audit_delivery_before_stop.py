"""Read-only repository/artifact/CPU snapshot while quantum work is active.

No checkpoint contents, SCF/integral/gradient/Hessian calls or Git mutations.
Creates only new files in this delivery-preflight directory.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
PILOT = HERE.parent
BASE = "3fc996e99cce8ff5799b5a2af9b565f982dfc899"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def verify_sealed_index(path):
    data = json.loads(path.read_text())
    if "files" in data and isinstance(data["files"], dict):
        entries = data["files"]
    elif "files_sha256" in data and isinstance(data["files_sha256"], dict):
        entries = data["files_sha256"]
    else:
        entries = data
    records = []
    for name, entry in entries.items():
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            continue
        item = path.parent / name
        actual = sha(item) if item.is_file() else None
        records.append({"path": str(item.relative_to(ROOT)), "expected_sha256": expected,
                        "actual_sha256": actual, "matches": actual == expected})
    return {"index": str(path.relative_to(ROOT)), "index_sha256": sha(path),
            "checked_files": len(records), "all_match": bool(records) and all(r["matches"] for r in records),
            "records": records}


def sample_worker_cpu(phase):
    pid = phase["pid"]
    try:
        # First parenthesis-delimited field can contain spaces; fields after
        # the final ')' begin with stat field3, so utime/stime are indices11/12.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        ticks = os.sysconf("SC_CLK_TCK")
        actual_cpu = (int(fields[11]) + int(fields[12])) / ticks
        child_cpu = (int(fields[13]) + int(fields[14])) / ticks
        expected_script = phase["command"][1]
        command_matches = expected_script in command
        return {"pid": pid, "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": command, "matches_ledger_worker_script": command_matches,
                "process_state": fields[0], "self_user_plus_system_cpu_seconds": actual_cpu,
                "completed_children_cpu_seconds": child_cpu,
                "CPU_seconds_including_reaped_children": actual_cpu + child_cpu,
                "clock_ticks_per_second": ticks,
                "process_start_boot_clock_ticks": int(fields[19]),
                "scope": "Live Linux/proc sample only; supervisor wait4 remains final accounting."}
    except FileNotFoundError:
        return {"pid": pid, "status": "NO_LONGER_PRESENT_WAIT_FOR_FINAL_LEDGER"}


def main():
    names = ["audit_before_stop.json", "cpu_budget_before_stop.snapshot.json",
             "cached_progress_before_stop.snapshot.json", "files_sha256.before-stop.json"]
    assert not any((HERE / name).exists() for name in names)
    started = datetime.now(timezone.utc).isoformat()
    changed = git("diff", "--no-ext-diff", "--name-only", "-z", BASE).decode().split("\0")
    changed = [path for path in changed if path]
    numstat = git("diff", "--no-ext-diff", "--numstat", BASE).decode().splitlines()
    additions = []
    for line in numstat:
        added, removed, path = line.split("\t", 2)
        additions.append({"path": path, "added_lines": added, "removed_lines": removed})
    allowed = {"AGENTS.md", "README.md"}
    historical_preserved = all(path in allowed for path in changed)
    allowed_changes_are_additions = all(record["removed_lines"] == "0" for record in additions)
    base_files = [path for path in git("ls-tree", "-r", "--name-only", "-z", BASE).decode().split("\0") if path]
    protected_counts = {prefix: sum(path.startswith(prefix + "/") for path in base_files)
                        for prefix in ("data", "models", "reports", "docs", "scripts", "tests")}
    index_checks = [verify_sealed_index(path) for path in sorted(PILOT.rglob("files_sha256*.json"))
                    if HERE not in path.parents]

    artifact_inventory = []
    for path in sorted(PILOT.rglob("*")):
        if not path.is_file() or HERE in path.parents:
            continue
        info = path.stat()
        artifact_inventory.append({"path": str(path.relative_to(ROOT)), "bytes": info.st_size,
                                   "mtime_ns": info.st_mtime_ns,
                                   "is_symlink": path.is_symlink()})
    huge = [record for record in artifact_inventory if record["bytes"] >= 10 * 1024**2]
    checkpoints = [record for record in artifact_inventory if record["path"].endswith(".chk")]
    generated_cache_names = [record for record in artifact_inventory
        if re.search(r"(?:^|/)(?:cderi|_cderi|.*\.eri|.*\.h5|.*\.hdf5|.*\.tmp|.*\.swap)$", record["path"], re.I)]
    checkpoint_policy = {"content_read": False, "all_checkpoint_checks_are_existence_and_size_only": True,
                         "reason": "Cached Spinosin worker still active; no claim of final checkpoint validity."}

    budget_bytes = (PILOT / "cpu_budget.json").read_bytes()
    progress_bytes = (PILOT / "spinosin-cached-optimization/progress.json").read_bytes()
    (HERE / names[1]).write_bytes(budget_bytes)
    (HERE / names[2]).write_bytes(progress_bytes)
    budget, progress = json.loads(budget_bytes), json.loads(progress_bytes)
    ended = [phase for phase in budget["phases"] if "actual_cpu_seconds" in phase]
    active = [phase for phase in budget["phases"] if phase.get("status") == "RUNNING"]
    ended_sum = sum(phase["actual_cpu_seconds"] for phase in ended)
    active_samples = [sample_worker_cpu(phase) for phase in active]
    valid_active_samples = all(sample.get("matches_ledger_worker_script") for sample in active_samples)
    active_sum = sum(sample.get("CPU_seconds_including_reaped_children", 0.) for sample in active_samples)
    live_total = ended_sum + active_sum

    failed_phase_evidence = {}
    for name in ("spinosin-scf-gradient/result.json", "spinosin-scf-gradient/pyscf.log",
                 "spinosin-scf-gradient/events.jsonl", "spinosin-scf-gradient/calculation.chk",
                 "spinosin_scf_gradient_driver.log", "direct_J_stop_request.json"):
        path = PILOT / name
        evidence = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
        if path.exists() and path.suffix != ".chk":
            evidence["sha256"] = sha(path)
        failed_phase_evidence[name] = evidence

    passed = (historical_preserved and allowed_changes_are_additions and
              all(record["all_match"] for record in index_checks) and not huge and not generated_cache_names and
              abs(ended_sum - budget["consumed_cpu_seconds"]) < 1e-9 and
              valid_active_samples and live_total < budget["quantum_cpu_cap_seconds"] and
              all(record["exists"] for record in failed_phase_evidence.values()))
    result = {
        "status": "PASS_POINT_IN_TIME_PREFLIGHT_NOT_FINAL_COMPLETION" if passed else "REVIEW_REQUIRED",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)), "quantum_and_integral_jobs_started_by_this_audit": 0,
        "repository": {"base_commit": BASE, "HEAD_at_snapshot": git("rev-parse", "HEAD").decode().strip(),
            "branch_at_snapshot": git("branch", "--show-current").decode().strip(),
            "tracked_changed_paths_from_base": changed, "numstat": additions,
            "old_tracked_files_unchanged_except_allowed_additive_notes": historical_preserved,
            "allowed_notes_have_zero_deleted_lines": allowed_changes_are_additions,
            "base_tracked_file_count": len(base_files), "base_file_count_by_protected_directory": protected_counts,
            "coverage_limit": "Git tracked comparison against base; this does not re-hash every ignored local raw/dependency file or attest to new untracked files not yet committed.",
            "status_porcelain_snapshot": git("status", "--porcelain=v1", "--untracked-files=normal").decode(),
            "git_mutations_by_this_audit": False},
        "sealed_index_checks": index_checks,
        "artifacts": {"inventory": artifact_inventory, "file_count": len(artifact_inventory),
            "total_bytes": sum(record["bytes"] for record in artifact_inventory),
            "largest_files": sorted(artifact_inventory, key=lambda record: record["bytes"], reverse=True)[:15],
            "engineering_large_file_flag_threshold_bytes": 10 * 1024**2,
            "files_above_large_flag": huge, "unnecessary_integral_or_swap_cache_candidates": generated_cache_names,
            "checkpoints": checkpoints, "checkpoint_policy": checkpoint_policy,
            "failed_or_interrupted_direct_J_phase_evidence": failed_phase_evidence,
            "free_disk_bytes_at_snapshot": shutil.disk_usage(ROOT).free,
            "dependencies_scope": "Runtime dependencies remain in ignored .runtime/ outside the new report directory; no wheel or integral cache is needed in Git."},
        "CPU": {"ledger_snapshot_path": str((HERE / names[1]).relative_to(ROOT)),
            "ledger_snapshot_sha256": sha(HERE / names[1]),
            "finished_phase_actual_CPU_sum_seconds": ended_sum,
            "ledger_consumed_matches_finished_phase_sum": abs(ended_sum - budget["consumed_cpu_seconds"]) < 1e-9,
            "live_active_samples": active_samples,
            "live_total_quantum_CPU_sample_seconds": live_total,
            "live_remaining_quantum_CPU_sample_seconds": budget["quantum_cpu_cap_seconds"] - live_total,
            "quantum_cpu_cap_seconds": budget["quantum_cpu_cap_seconds"],
            "non_quantum_reserved_seconds_not_measured_by_quantum_ledger": budget["non_quantum_reserve_seconds"],
            "user_total_cpu_cap_seconds": budget["user_total_cpu_cap_seconds"],
            "pending_final_recomputation": True,
            "limitation": "Running-phase time is absent from ledger consumed/remaining fields until wait4. Live/proc estimate is not a final total; preparation/test CPU reserve is not a measurement of all preparation CPU."},
        "active_phase_snapshot": {"path": str((HERE / names[2]).relative_to(ROOT)),
            "sha256": sha(HERE / names[2]), "record_utc": progress.get("utc"),
            "status_at_snapshot": progress.get("status"),
            "last_completed_geometry_cycle": progress.get("last_geometry_cycle"),
            "geometry_optimized_at_snapshot": progress.get("geometry_optimized"),
            "frequencies_status_at_snapshot": progress.get("frequencies_status"),
            "not_a_final_result": True},
        "scientific_method_risks": [
            "PBE/def2-SVP/grid1/no dispersion is an explicit isolated-molecule feasibility pilot; no D3(BJ), basis/grid convergence or solid-state phonon claim.",
            "A converged SCF at one geometry is not a converged optimized structure. Fresh final same-method gradient and optimizer convergence remain necessary before Hessian.",
            "Low-frequency torsions are conformer-, grid- and environment-sensitive; real Spinosin frequencies cannot be inferred from water validation or unfinished optimization.",
            "No Spinosin Hessian/frequency success is attested while the optimization worker is active. IR intensities are NOT_COMPUTED.",
            "C001 is a documented ETKDG/MMFF-generated verified-stereo conformer, not an experimental coordinate record or a proven global minimum.",
            "Cached DF-J uses a real5.718GB ndarray; safe water equivalence does not prove Spinosin numerical accuracy or ensure memory at future geometries.",
            "Requested-formula A/B cached DF tensors exceed this environment; A source-entry identity ambiguity must remain explicit.",
            "Theoretical/experimental peak proximity alone cannot establish compound identity. Existing spectra, classification models and96.0234% baseline remain separate.",
        ],
        "follow_up_after_parent_stops_phase": [
            "Recompute final wait4 phase CPU sum and compare with ledger; no RUNNING records should remain.",
            "Independently recompute every completed saved Spinosin gradient statistic and identify the last complete geometry without conflating later SCF checkpoint geometry.",
            "Verify final result/checkpoint SHA only after the worker has exited; preserve interrupted direct-J evidence and all optimizer steps.",
            "Repeat Git diff/status and large-file scan after final reports/docs are generated, before commit/push.",
        ],
    }
    write_new(HERE / names[0], result)
    files = [Path(__file__), HERE / names[0], HERE / names[1], HERE / names[2]]
    write_new(HERE / names[3], {"scope": "Delivery before-stop snapshot, immutable; later final review uses a separate index.",
                              "files_sha256": {path.name: sha(path) for path in files}})
    print(json.dumps({"status": result["status"], "tracked_changes": changed,
        "base_file_count": len(base_files), "sealed_indices_checked": len(index_checks),
        "sealed_files_checked": sum(record["checked_files"] for record in index_checks),
        "sealed_indices_all_match": all(record["all_match"] for record in index_checks),
        "quantum_artifact_bytes_at_snapshot": result["artifacts"]["total_bytes"],
        "large_files": huge, "live_total_quantum_CPU_sample_seconds": live_total,
        "before_stop_index_sha256": sha(HERE / names[3])}, indent=2))


if __name__ == "__main__":
    main()
