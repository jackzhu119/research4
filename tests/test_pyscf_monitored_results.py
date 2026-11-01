"""Synthetic final-ledger/array orchestration only; no QM or active checkpoints."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("monitored_results_analysis", ROOT / "scripts/analyze_pyscf_monitored_results_20261031.py")
analysis = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(analysis)
FIX = importlib.util.spec_from_file_location("frozen_analysis_shape_fixtures", ROOT / "tests/test_pyscf_continuation_analysis.py")
fixtures = importlib.util.module_from_spec(FIX); FIX.loader.exec_module(fixtures)


def budget_fixture(root):
    root.mkdir(); prior_dir = root.parent / "prior"; prior_dir.mkdir()
    prior = prior_dir / "cpu_budget.json"
    analysis.write(prior, {"schema": "research4-pyscf-continuation-budget-v2", "consumed_cpu_seconds": 10.,
                          "quantum_cpu_cap_seconds": 43200, "phases": [{"status": "EXITED", "actual_cpu_seconds": 10.}]})
    checkpoint = prior_dir / "source.chk"; checkpoint.write_bytes(b"Synthetic hash fixture, never loaded as molecular checkpoint")
    auth = root / "authorization.json"
    authorization = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "authorization_text": "Unit test only: no real calculation authorized",
        "new_budget_is_separately_authorized": True, "prior_budget_preserved": True,
        "source_base_commit": "aef64c65295bee03764da1dd72c21d32ed3e6c24", "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": analysis.sha(checkpoint), "prior_closed_ledger_path": str(prior),
        "prior_closed_ledger_sha256": analysis.sha(prior), "prior_consumed_cpu_seconds": 10.,
        "max_quantum_cpu_seconds": 172800, "max_run_wall_seconds": 86400, "max_phase_cpu_seconds": 86400,
        "max_phase_wall_seconds": 43200, "max_threads": 4, "max_memory_gib": 24, "guard_semantics": analysis.GUARD,
        "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3, "monitor_interval_seconds": .1,
        "memory_kill_grace_seconds": 1., "expected_cgroup_memory_max_bytes": 16*1024**3}
    analysis.write(auth, authorization); created = datetime.now(timezone.utc)
    ledger = {"schema": "research4-pyscf-monitored-budget-v3", "runner_sha256": analysis.sha(analysis.SUPERVISOR),
        "authorization_path": str(auth), "authorization_sha256": analysis.sha(auth),
        "source_checkpoint_path": str(checkpoint), "source_checkpoint_sha256": analysis.sha(checkpoint),
        "source_base_commit": authorization["source_base_commit"], "quantum_cpu_cap_seconds": 172800,
        "run_wall_cap_seconds": 86400, "created_at_utc": created.isoformat(),
        "run_deadline_at_utc": (created + timedelta(seconds=86400)).isoformat(), "consumed_cpu_seconds": 0., "phases": [],
        "prior_closed_ledger": {"path": str(prior), "sha256": analysis.sha(prior), "actual_cpu_seconds": 10.,
                                "phase_count": 1, "no_RUNNING_phases": True}}
    analysis.write(root / "cpu_budget.json", ledger)
    return ledger


def closed_phase(root, ledger, identity="spinosin", missing=False, negative=False, label=None):
    location = root / (label or ("synthetic-" + identity)); location.mkdir()
    row = {"phase": location.name, "status": "PARTIAL_OR_FAILED" if negative or missing else "EXITED",
        "exit_code": 1 if missing else (2 if negative else 0), "actual_cpu_seconds": 2., "elapsed_wall_seconds": 1.,
        "threads": 4, "pid": 12345, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_soft_limit_seconds": 900, "cpu_hard_limit_seconds": 1000, "wall_limit_seconds": 1000,
        "command": ["synthetic", str(analysis.ETHANOL_VALIDATOR if identity == "validation_ethanol" else analysis.WORKER), "--output", str(location)],
        "memory_terminated": False, "monitor_failed": False, "wall_terminated": False, "hard_killed": False,
        "peak_group_rss_bytes_sampled": 1024**3, "peak_cgroup_memory_bytes_sampled": 2*1024**3}
    log = root / (location.name + ".log"); log.write_text("Synthetic driver log, not actual PySCF output\n")
    row.update(log=str(log), log_sha256=analysis.sha(log))
    proof_path = root / (location.name + ".proof.json")
    proof = {"schema": "research4-pyscf-resource-supervisor-proof-v3", "guard_semantics": analysis.GUARD,
        "address_space_soft_limit_bytes": 24*1024**3, "address_space_hard_limit_bytes": 24*1024**3,
        "rss_stop_bytes": 10*1024**3, "cgroup_kill_bytes": 12*1024**3,
        "expected_cgroup_memory_max_bytes": 16*1024**3, "phase": row["phase"], "threads": 4,
        "ledger_path": str(root / "cpu_budget.json"), "supervisor_script_path": str(analysis.SUPERVISOR),
        "supervisor_script_sha256": analysis.sha(analysis.SUPERVISOR), "child_pid": row["pid"],
        "supervisor_pid": 23456, "supervisor_start_ticks": 120, "child_start_ticks": 125,
        "cpu_soft_limit_seconds": 900, "cpu_hard_limit_seconds": 1000, "wall_limit_seconds": 1000,
        "monitor_interval_seconds": .1, "memory_kill_grace_seconds": 1.}
    analysis.write(proof_path, proof); row.update(supervisor_proof_path=str(proof_path), supervisor_proof_sha256=analysis.sha(proof_path))
    result = fixtures.result_fixture("water" if identity == "water" else "spinosin")
    result["settings"] = {**fixtures.direct_settings(), "engineering_worker_version": "SUPERVISED_PARALLEL_DIRECT_V8_SAVE_ONLY",
                          "resource_supervision_required": True}
    result.update(identity=identity, script_sha256=analysis.sha(analysis.ETHANOL_VALIDATOR if identity == "validation_ethanol" else analysis.WORKER),
                  resource_supervision={"proof_sha256": row["supervisor_proof_sha256"]})
    if identity == "validation_ethanol":
        result.update(atoms=9, validation_only=True, engineering_helper_source_sha256=analysis.sha(analysis.WORKER))
    if negative:
        result.update(status="PARTIAL", imaginary_frequency_count=1, all_frequencies_real_positive=False,
                      frequency_validation_passed=False, stationary_minimum_verified=False)
        result["frequencies_cm1"][0] = -5.; result["frequencies_thz"][0] = -5.*analysis.CORE.CM1_TO_THZ
    if missing:
        result.update(status="PARTIAL", geometry_optimized=False, frequencies_status="NOT_RUN", event="hessian_started")
        analysis.write(location / "input.json", result); analysis.write(location / "progress.json", result)
    else:
        analysis.write(location / "result.json", result)
    ledger["phases"].append(row); ledger["consumed_cpu_seconds"] += row["actual_cpu_seconds"]
    analysis.write(root / "cpu_budget.json", ledger)
    return location, row, result


def release(root):
    path = root.parent / "release.json"
    state = root / "controller-state.json"
    analysis.write(state, {"status": "DONE", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                           "ledger_sha256_at_end": analysis.sha(root / "cpu_budget.json")})
    analysis.write(path, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "allow_analysis_export": True,
                          "cpu_budget_sha256": analysis.sha(root / "cpu_budget.json"),
                          "pipeline_state_path": str(state), "pipeline_state_sha256": analysis.sha(state)})
    return path


def fake_modes(location, result, hashes):
    # Shape-only fixture for orchestration tests; never a real molecular result.
    identity = result["identity"]; count, atoms = (3, 3) if identity == "water" else (219, 75)
    rows = [{"phase": location.name, "identity": identity, "mode": i+1, "frequency_cm1": float(cm),
        "frequency_thz": float(cm*analysis.CORE.CM1_TO_THZ), "imaginary": bool(cm < 0),
        "accepted_for_comparison": False, "IR_intensity_status": "NOT_COMPUTED"}
        for i, cm in enumerate(result["frequencies_cm1"])]
    return rows, np.ones((count, atoms, 3)), (["H"]*atoms, np.zeros((atoms, 3))), {"status": "SYNTHETIC_SHAPE_FIXTURE_ONLY", "quantum_jobs": 0}


class MonitoredResultsTests(unittest.TestCase):
    def test_official_v3_authorization_and_prior_sum_pass_without_qm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; budget_fixture(root); gate = release(root)
            _, _, hashes, sums = analysis.authorize(root, analysis.read(gate))
            self.assertEqual(sums, {"V3": 0., "prior_V2": 10.})
            self.assertIn(str(root.parent/"prior/cpu_budget.json"), hashes)

    def test_running_or_prior_tampering_blocks_final_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; ledger = budget_fixture(root)
            ledger["phases"] = [{"status": "RUNNING"}]; analysis.write(root/"cpu_budget.json", ledger)
            gate = release(root)
            with self.assertRaises(ValueError): analysis.authorize(root, analysis.read(gate))
            ledger["phases"] = []; analysis.write(root/"cpu_budget.json", ledger)
            (root.parent/"prior/cpu_budget.json").write_text("{}")
            gate = release(root)
            with self.assertRaises(ValueError): analysis.authorize(root, analysis.read(gate))

    def test_final_release_hash_and_independent_cpu_total_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; ledger = budget_fixture(root); gate = release(root)
            ledger["consumed_cpu_seconds"] = 3.; analysis.write(root/"cpu_budget.json", ledger)
            with self.assertRaises(ValueError): analysis.authorize(root, analysis.read(gate))
            gate = release(root)
            with self.assertRaises(ValueError): analysis.authorize(root, analysis.read(gate))

    def test_between_stage_closed_ledger_does_not_make_running_pipeline_final(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; budget_fixture(root); gate = release(root)
            data = analysis.read(gate); state = Path(data["pipeline_state_path"])
            analysis.write(state, {"status":"RUNNING", "ledger_sha256_at_end":analysis.sha(root/"cpu_budget.json")})
            data["pipeline_state_sha256"] = analysis.sha(state); analysis.write(gate, data)
            with self.assertRaises(ValueError): analysis.authorize(root, data)

    def test_frequency_parent_must_belong_to_true_closed_successful_ledger_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; ledger = budget_fixture(root)
            parent = root/"fake-parent"; parent.mkdir(); source = parent/"result.json"
            analysis.write(source, {"phase":"spinosin-optimize", "script_sha256":analysis.sha(analysis.WORKER)})
            row = {"command":["synthetic","--optimization-result",str(source)]}
            with patch.object(analysis.CORE,"frequency_parent_guard",return_value=None):
                with self.assertRaises(ValueError):
                    analysis.frequency_parent_audit(root,ledger,row,{"identity":"spinosin"},{})

    def test_archived_proof_and_log_contract_verified_not_claimed_live(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; ledger = budget_fixture(root); _, row, _ = closed_phase(root, ledger)
            result = analysis.audit_closed_proof(root, root/"cpu_budget.json", row, {})
            self.assertTrue(result["frequency_acceptance_allowed"])
            self.assertTrue(result["RSS_monitor_best_effort_not_hard_limit"])
            proof_path = Path(row["supervisor_proof_path"]); data = analysis.read(proof_path)
            data["rss_stop_bytes"] += 1; analysis.write(proof_path, data)
            row["supervisor_proof_sha256"] = analysis.sha(proof_path)
            with self.assertRaises(ValueError): analysis.audit_closed_proof(root, root/"cpu_budget.json", row, {})

    def test_memory_terminated_phase_is_not_frequency_eligible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"; ledger = budget_fixture(root); _, row, _ = closed_phase(root, ledger)
            row.update(status="PARTIAL_OR_FAILED", memory_terminated=True)
            self.assertFalse(analysis.audit_closed_proof(root, root/"cpu_budget.json", row, {})["frequency_acceptance_allowed"])

    def run_synthetic(self, identity="spinosin", missing=False, negative=False):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)/"run"; ledger = budget_fixture(root)
        parent_path = None
        if identity == "spinosin" and not missing:
            parent, _, parent_result = closed_phase(root,ledger,"spinosin",label="synthetic-true-opt-parent")
            parent_result.update(phase="spinosin-optimize",frequencies_status="NOT_RUN")
            np.save(parent/"gradient_hartree_per_bohr.npy",np.zeros((75,3)))
            analysis.write(parent/"result.json",parent_result); parent_path = parent/"result.json"
        location, owner, result = closed_phase(root, ledger, identity, missing, negative)
        if parent_path is not None:
            owner["command"] += ["--optimization-result",str(parent_path)]
            analysis.write(root/"cpu_budget.json",ledger)
        gate = release(root); peaks = root.parent/"peaks.csv"
        import csv
        with peaks.open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["outer_fold", "candidate_number", "frequency_index", "frequency_thz", "wavenumber_cm1"])
            writer.writerows((fold, number, number, 1.08+.01*number, (1.08+.01*number)/analysis.CORE.CM1_TO_THZ)
                             for fold in range(5) for number in range(1,5))
        with patch.object(analysis, "PEAK_PATH", peaks), patch.object(analysis, "PEAK_SHA256", analysis.sha(peaks)), \
             patch.object(analysis.CORE, "audit_final_gradient", return_value={}), \
             patch.object(analysis.CORE, "frequency_parent_guard", return_value=None), \
             patch.object(analysis.CORE, "load_modes", side_effect=fake_modes), \
             patch.object(analysis.CORE, "plot_reports", return_value=None), patch.object(analysis, "cost_plot", return_value=None), \
             patch.object(analysis, "computed_frequency_plot", return_value=None):
            summary = analysis.run(root, root.parent/"export", gate)
        return root, location, result, summary

    def test_native_missing_result_stays_partial_and_no_file_is_fabricated(self):
        root, location, _, summary = self.run_synthetic(missing=True)
        self.assertEqual(summary["ended_phases_without_worker_result"], 1)
        self.assertEqual(summary["validated_spinosin_frequency_rows"], 0)
        self.assertFalse((location/"result.json").exists())
        failures = analysis.read(root.parent/"export/actual_failure_evidence.json")
        self.assertTrue(failures[0]["worker_result_missing"])

    def test_real_signed_computed_rows_are_not_promoted_to_qualified_modes(self):
        root, location, result, summary = self.run_synthetic(negative=True)
        self.assertEqual(summary["computed_frequency_rows_including_unvalidated"], 219)
        self.assertEqual(summary["validated_spinosin_frequency_rows"], 0)
        self.assertEqual(analysis.read(location/"result.json")["frequencies_cm1"][0], -5.)
        import csv
        with (root.parent/"export/experiment_theory_pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertTrue(all(row["theoretical_frequency_thz"] == "" for row in csv.DictReader(handle)))

    def test_water_and_ethanol_never_become_spinosin_modes(self):
        for identity in ("water", "validation_ethanol"):
            _, _, _, summary = self.run_synthetic(identity=identity)
            self.assertEqual(summary["validated_spinosin_frequency_rows"], 0)
            self.assertTrue(summary["water_and_ethanol_are_validation_only"])
            if identity == "validation_ethanol": self.assertEqual(summary["computed_frequency_rows_including_unvalidated"], 0)

    def test_complete_qualified_shape_can_enter_spinosin_scope(self):
        _, _, _, summary = self.run_synthetic()
        self.assertEqual(summary["validated_spinosin_frequency_rows"], 219)
        self.assertFalse(summary["chemical_attribution_confirmed"])

    def test_missing_rss_is_not_plotted_as_measured_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            analysis.cost_plot(output, [{"phase":"missing", "actual_cpu_seconds":0., "peak_group_RSS_bytes_sampled":None},
                                        {"phase":"measured", "actual_cpu_seconds":2., "peak_group_RSS_bytes_sampled":1024**3}])
            self.assertTrue((output/"actual_phase_costs.png").exists())
            self.assertTrue((output/"actual_phase_costs.pdf").exists())


if __name__ == "__main__": unittest.main()
