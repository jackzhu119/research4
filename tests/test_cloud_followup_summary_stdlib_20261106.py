"""No-Q tests of closure, provenance and interpretation in the postprocessor."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("postrun_summary_tested", REPO/"scripts/research4_cloud_followup_summary_20261106.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def reference(path):
    return {"path": str(path), "sha256": sha(path)}


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="research4-summary-test-"); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = mock.patch.object(M, "ROOT", self.root); self.patch.start(); self.addCleanup(self.patch.stop)

    def quantum_fixture(self, *, legacy=True, wrong_input_output=False):
        output = self.root/"reports/fake-job"; output.mkdir(parents=True)
        worker = self.root/"scripts/fake_worker.py"; worker.parent.mkdir()
        worker.write_bytes((REPO/"scripts/audit_pyscf_saved_cphf_residual_20261105.py").read_bytes()
                           if legacy else b"# Synthetic producer fixture, never executed\n")
        inp = output/"input.json"
        write_json(inp, {"arguments": {"output": str(self.root/"reports/other-output" if wrong_input_output else output)}})
        log = self.root/"reports/job.log"; log.write_text("Unexecuted stdlib evidence fixture\n")
        proof = {"phase": "old-fake-job-attempt001", "ledger_path": str(self.root/"ledger.json"),
                 "supervisor_script_sha256": "b"*64}
        proof_file = self.root/"reports/job.proof.json"; write_json(proof_file, proof)
        result = {"schema": "research4-saved-cphf-equation-residual-audit-v1", "status": "SAVED_RESPONSE_RESIDUALS_MEASURED_NOT_CONVERGENCE_GATED",
                  "source_worker": reference(worker), "input_json_sha256": sha(inp),
                  "resource_supervision": {"proof": proof, "proof_sha256": sha(proof_file),
                                            "live_parent_process_and_source_verified": True}}
        write_json(output/"result.json", result)
        row = {"phase": proof["phase"], "status": "EXITED", "exit_code": 0,
               "actual_cpu_seconds": 1.25, "command": ["python", str(worker), "--output", str(output)],
               "supervisor_proof_sha256": sha(proof_file), "log_sha256": sha(log)}
        ledger = {"phases": [row], "continuation_ledger_path": str(self.root/"ledger.json"), "runner_sha256": "b"*64}
        seal = {"phase": row["phase"], "result": reference(output/"result.json"),
                "closed_row_sha256": M.row_hash(row), "actual_cpu_seconds": 1.25,
                "proof": reference(proof_file), "log": reference(log), "artifacts": []}
        job = {"id": "fake-job", "kind": "quantum", "worker": reference(worker),
               "result_schema": result["schema"], "success_statuses": [result["status"]]}
        saved = {"status": "DONE_SEALED", "output": str(output), "phase": row["phase"], "seal": seal}
        return job, saved, ledger

    def test_missing_PID_does_not_close_running_job(self):
        config = {"jobs": [{"id": "x"}]}
        state = {"invocations": [{"id": "q", "status": "RUNNING"}], "jobs": {"x": {"status": "RUNNING", "phase": "p"}}}
        ledger = {"phases": [{"phase": "p", "status": "EXITED"}]}
        self.assertTrue(M.readiness_reasons(config, state, ledger))
        hints = M.forensic_hints(config, state, ledger)
        self.assertEqual(hints[0]["classification"], "PARTIAL_FORENSIC_REQUIRED_NOT_SCIENTIFIC_RELEASE")
        self.assertEqual(state["jobs"]["x"]["status"], "RUNNING")

    def test_open_or_unknown_global_ledger_blocks_even_terminal_queue(self):
        config = {"jobs": [{"id": "x"}]}
        state = {"invocations": [{"id": "q", "status": "COMPLETED"}], "jobs": {"x": {"status": "DONE_SEALED"}}}
        for status in ("RUNNING", "UNKNOWN", "CPU_UNACCOUNTED"):
            self.assertTrue(M.readiness_reasons(config, state, {"phases": [{"phase": "other", "status": status}]}))
        self.assertEqual(M.readiness_reasons(config, state, {"phases": [{"phase": "p", "status": "EXITED"}]}), [])

    def test_checkpoint_is_metadata_only_and_not_opened(self):
        path = self.root/"state.chk"; path.write_bytes(b"synthetic checkpoint")
        with mock.patch.object(Path, "open", side_effect=AssertionError("Checkpoint content must not be read")):
            observed = M.verify_reference({"path": str(path), "sha256": "a"*64, "bytes": path.stat().st_size})
        self.assertEqual(observed["reason"], "FORBIDDEN_CHECKPOINT")
        self.assertIn("CONTENT_NOT_READ", observed["verification"])

    def test_large_array_is_not_read_even_for_checksum(self):
        path = self.root/"big.npy"
        with path.open("wb") as stream: stream.truncate(M.MAX_BINARY_HASH+1)
        with mock.patch.object(Path, "open", side_effect=AssertionError("Large array must not be read")):
            observed = M.verify_reference({"path": str(path), "sha256": "a"*64, "bytes": path.stat().st_size})
        self.assertEqual(observed["reason"], "OVERSIZED_ARTIFACT")

    def test_changed_small_metadata_is_rejected(self):
        path = self.root/"small.json"; path.write_text("{}")
        original = reference(path); path.write_text('{"changed":true}')
        with self.assertRaises(M.EvidenceError): M.verify_reference(original)

    def test_legacy_output_requires_actual_command_and_hashed_input(self):
        job, saved, ledger = self.quantum_fixture()
        observed = M.verify_job(job, saved, ledger, [])
        self.assertEqual(observed["source_worker"]["sha256"], M.LEGACY_RESIDUAL_WORKER_SHA)
        ledger["phases"][0]["command"][-1] = str(self.root/"reports/other")
        saved["seal"]["closed_row_sha256"] = M.row_hash(ledger["phases"][0])
        with self.assertRaises(M.EvidenceError): M.verify_job(job, saved, ledger, [])

    def test_legacy_input_output_mismatch_is_rejected(self):
        job, saved, ledger = self.quantum_fixture(wrong_input_output=True)
        with self.assertRaises(M.EvidenceError): M.verify_job(job, saved, ledger, [])

    def test_unknown_producer_does_not_receive_legacy_alias(self):
        job, saved, ledger = self.quantum_fixture(legacy=False)
        with self.assertRaises(M.EvidenceError): M.verify_job(job, saved, ledger, [])

    def test_audited_reuse_preserves_original_stale_status_and_requires_ROOT(self):
        job, saved, ledger = self.quantum_fixture()
        old_config = self.root/"old_config.json"; old_grant = self.root/"old_grant.json"; old_state = self.root/"old_state.json"
        write_json(old_config, {"phase_prefix": "old", "jobs": [job], "immutable_inputs": []})
        write_json(old_grant, {"config_sha256": sha(old_config)})
        write_json(old_state, {"config_sha256": sha(old_config), "grant_sha256": sha(old_grant),
                              "jobs": {job["id"]: {**saved, "status": "RUNNING", "seal": None}}})
        before = old_state.read_bytes()
        registry = self.root/"registry.json"
        body = {"schema": "research4-root-audited-reused-evidence-v1", "root_approved": True,
                "forensic_reason": "Synthetic test of actual closed legacy producer with stale queue metadata",
                "source_queue_config": reference(old_config), "source_queue_grant": reference(old_grant),
                "source_queue_state": reference(old_state), "metadata_adapter_source": job["worker"],
                "records": [{"job": job, "seal": saved["seal"], "source_queue_state": reference(old_state),
                             "output_directory": saved["output"], "status": "DONE_VERIFIED_FROM_CLOSED_PROVENANCE",
                             "scientific_release": True, "legacy_output_binding": True}]}
        write_json(registry, body)
        rows, _, _ = M.reused_records({"jobs": [], "reused_evidence": reference(registry)}, ledger, [])
        self.assertEqual(rows[0]["original_queue_status"], "RUNNING")
        self.assertEqual(old_state.read_bytes(), before)
        body["root_approved"] = False; write_json(registry, body)
        with self.assertRaises(M.EvidenceError): M.reused_records({"jobs": [], "reused_evidence": reference(registry)}, ledger, [])

    def fd_result(self, *, analytic=-.03, gradient=.005, step=.01):
        return {"case": "Spinosin", "displacement_pair": {"mode_index_1based": 1, "step_bohr": step},
                "analytic_directional_curvature_Eh_per_Bohr2": analytic,
                "finite_gradient_directional_curvature_Eh_per_Bohr2": gradient,
                "finite_energy_directional_curvature_Eh_per_Bohr2": gradient+.001,
                "grid_diagnostic_limitation": M.GRID_CAVEAT}

    def test_sign_disagreement_and_near_zero_are_descriptive_not_PASS(self):
        row = M.fd_row("m1", self.fd_result())
        self.assertTrue(row["exact_sign_disagreement"])
        self.assertEqual(row["new_numeric_PASS_FAIL_threshold"], "NOT_SET")
        near = M.fd_row("water-near-zero", self.fd_result(analytic=-1e-12, gradient=1e-12))
        self.assertTrue(near["exact_sign_disagreement"])
        self.assertIn("not a physical significance test", near["grid_caveat"])

    def test_step_dependence_is_measured_not_extrapolated_or_selected(self):
        a = M.fd_row("large", self.fd_result(gradient=.005, step=.01))
        z = M.fd_row("small", self.fd_result(gradient=.005001, step=.005))
        rows = M.step_rows([a,z]); self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["FD_gradient_absolute_step_difference_Eh_per_Bohr2"], .000001)
        self.assertIn("NO_EXTRAPOLATION", rows[0]["step_consistency_PASS_FAIL"])

    def test_duplicate_RHS_does_not_increase_physical_coverage(self):
        rows = [{"atom_index_zero_based": 0, "axis": a} for a in ("x","y","z")]
        coverage = M.residual_coverage(rows+rows)
        self.assertEqual(coverage["unique_measured_nuclear_RHS"], 3)
        self.assertEqual(coverage["duplicate_RHS_rows_not_counted_as_extra_coverage"], 3)
        self.assertFalse(coverage["all225_measured"])

    def test_native_positive_roots_without_convergence_are_undetermined(self):
        result = {"case": "ethanol", "internal_orbital_stability": "STABLE",
                  "native_all_requested_roots_converged": False,
                  "davidson": {"actual_returned_roots_Eh": [3.1,3.2,3.3],
                               "actual_returned_convergence_flags": [False,False,False]}}
        rows = M.orbital_rows("ethanol", result)
        self.assertTrue(all(x["qualified_internal_orbital_verdict"].startswith("UNDETERMINED") for x in rows))
        self.assertTrue(all(x["final_root_residual_norm"].startswith("NOT_RETURNED") for x in rows))

    def test_wrapper_guard_and_Git_are_conflicts_but_postrun_watcher_is_not(self):
        self.assertIsNotNone(M.process_conflict(["python", "run_research4_host_a_20261106_v4.py"]))
        self.assertIsNotNone(M.process_conflict(["python", "run_pyscf_continuation_budget_20261105_v8.py"]))
        self.assertIsNotNone(M.process_conflict(["git", "push"]))
        self.assertIsNone(M.process_conflict(["python", "research4_postrun_analysis_watch_20261106.py"]))

    def test_check_only_and_pending_execute_never_create_report(self):
        config = {"jobs": [{"id": "x"}]}; state = {"invocations": [{"id": "q", "status": "RUNNING"}],
                                                    "jobs": {"x": {"status": "RUNNING"}}}; ledger = {"phases": []}
        output = self.root/"reports/should-not-exist"
        with mock.patch.object(M, "load_context", return_value=(config,{},state,ledger)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(M.main(["--check-only", "--output", str(output)]), 0)
                self.assertEqual(M.main(["--execute", "--output", str(output)]), 3)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
