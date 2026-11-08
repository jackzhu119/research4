"""Tiny metadata/NumPy fixtures; never run SCF, gradients or integrals."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pyscf_resumable_hessian_worker_20261103.py"


class ResumableHessianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("resumable_hessian_pure_fixture", SCRIPT)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.w = self.worker

    def row(self):
        value = {"status": "EXITED", "exit_code": 0, "ended_at_utc": "2026-11-04T00:00:00+00:00", "actual_cpu_seconds": 0.}
        value.update({key: False for key in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed")})
        return value

    def fixture(self, phase="hessian-segment", status="CHECKPOINT_SEGMENT_COMPLETE"):
        w = self.w
        out = self.path / "producer"
        out.mkdir()
        log = self.path / "parent.log"
        log.write_text("Synthetic software fixture only\n")
        ledger_file = self.path / "ledger.json"
        proof_file = self.path / "proof.json"
        proof = {"schema": "research4-pyscf-actual-cpu-supervisor-proof-v4", "supervisor_script_sha256": w.SUPERVISOR_SHA,
                 "phase": "fixture_phase", "ledger_path": str(ledger_file)}
        proof_file.write_text(json.dumps(proof))
        supervision = {"proof_path": str(proof_file), "proof_sha256": w.sha(proof_file), "proof": proof,
                       "live_parent_process_and_source_verified": True}
        source = w.state_module().artifact_record(SCRIPT)
        evidence = {"schema": w.EXECUTION_SCHEMA, "source_worker": source, "resource_supervision": supervision,
                    "output_directory": str(out), "snapshot_manifest_sha256": "a"*64, "phase": phase}
        result = {"schema": w.SCHEMA, "status": status, "phase": phase, "source_worker": source,
                  "resource_supervision": supervision, "snapshot_manifest_sha256": "a"*64,
                  "native_direct_driver_control": {"restoration_failures": 0, "original_driver_restored": True},
                  "stable_MF_state_unchanged": True, "boot_id": "synthetic-boot",
                  "restore_gradient_comparison_passed": True, "final_gradient_thresholds_passed": True}
        (out / "result.json").write_text(json.dumps(result))
        row = {**self.row(), "phase": "fixture_phase", "supervisor_proof_sha256": w.sha(proof_file),
               "supervisor_proof_path": str(proof_file), "log": str(log), "log_sha256": w.sha(log),
               "command": ["python", str(SCRIPT), "--output", str(out)]}
        ledger_file.write_text(json.dumps({"schema": "research4-pyscf-actual-cpu-budget-v4", "runner_sha256": w.SUPERVISOR_SHA, "phases": [row]}))
        return evidence, result, ledger_file, row

    def test_healthy_closed_row(self):
        self.assertTrue(self.w.healthy_closed_row(self.row()))

    def test_each_guard_failure_rejected(self):
        for key in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed"):
            row = self.row()
            row[key] = True
            with self.subTest(key=key):
                self.assertFalse(self.w.healthy_closed_row(row))

    def test_running_or_failed_or_unknown_exit_rejected(self):
        for mutation in ({"status": "RUNNING"}, {"exit_code": 2}, {"ended_at_utc": None}, {"actual_cpu_seconds": -1.}, {"actual_cpu_seconds": float("nan")}):
            self.assertFalse(self.w.healthy_closed_row({**self.row(), **mutation}))

    def test_closed_producer_real_files_verified(self):
        evidence, _, _, _ = self.fixture()
        actual = self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)
        self.assertEqual(actual["phase"], "fixture_phase")

    def test_wrong_snapshot_rejected(self):
        evidence, _, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "exact source MF"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="b"*64)

    def test_self_reported_execution_not_sufficient(self):
        evidence, _, ledger, _ = self.fixture()
        value = json.loads(ledger.read_text())
        value["phases"][0]["status"] = "RUNNING"
        ledger.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "healthy EXITED"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_changed_parent_log_rejected(self):
        evidence, _, _, _ = self.fixture()
        (self.path / "parent.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_changed_proof_rejected(self):
        evidence, _, _, _ = self.fixture()
        (self.path / "proof.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "proof bytes"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_wrong_command_output_rejected(self):
        evidence, _, ledger, _ = self.fixture()
        value = json.loads(ledger.read_text())
        value["phases"][0]["command"][-1] = str(self.path / "different")
        ledger.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "result directory"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_stable_state_false_rejected(self):
        evidence, result, _, _ = self.fixture()
        result["stable_MF_state_unchanged"] = False
        (Path(evidence["output_directory"]) / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "unchanged source MF"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_normal_checkpoint_not_full_frequency_status(self):
        evidence, result, _, _ = self.fixture()
        self.assertEqual(result["status"], "CHECKPOINT_SEGMENT_COMPLETE")
        self.assertNotEqual(result["status"], "FREQUENCIES_COMPUTED")
        self.assertIn(result["status"], self.w.SUCCESS)
        self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_unclean_result_not_accepted(self):
        evidence, _, _, _ = self.fixture(status="PARTIAL")
        with self.assertRaisesRegex(ValueError, "successful producer"):
            self.w.verify_closed_execution(evidence, expected_snapshot_sha="a"*64)

    def test_gradient_gate_same_boot(self):
        evidence, _, _, _ = self.fixture(phase="restore-gradient-check", status="DONE")
        path = Path(evidence["output_directory"]) / "result.json"
        actual = self.w.qualify_gradient_receipt(path, self.w.sha(path), "a"*64, "synthetic-boot")
        self.assertEqual(actual["scope"], "SAME_BOOT_RESTORED_STATE_NO_NEW_SCF")

    def test_gradient_gate_different_boot_rejected(self):
        evidence, _, _, _ = self.fixture(phase="restore-gradient-check", status="DONE")
        path = Path(evidence["output_directory"]) / "result.json"
        with self.assertRaisesRegex(ValueError, "Same-boot"):
            self.w.qualify_gradient_receipt(path, self.w.sha(path), "a"*64, "new-boot")

    def test_gradient_gate_comparison_failure_rejected(self):
        evidence, result, _, _ = self.fixture(phase="restore-gradient-check", status="DONE")
        result["restore_gradient_comparison_passed"] = False
        path = Path(evidence["output_directory"]) / "result.json"
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "Same-boot"):
            self.w.qualify_gradient_receipt(path, self.w.sha(path), "a"*64, "synthetic-boot")

    def test_each_cached_unit_gets_closed_producer_gate(self):
        evidence, _, _, _ = self.fixture()
        cache = self.path / "cache"
        record = cache / "units/000000-diag-0000/attempt-000001/complete.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"execution_evidence": evidence}))
        actual = self.w.verify_cached_unit_producers(cache, "a"*64)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["qualification"], "CLOSED_HEALTHY_V4_PRODUCER")

    def test_unclean_cached_unit_requires_explicit_forensic(self):
        evidence, _, ledger, _ = self.fixture()
        data = json.loads(ledger.read_text())
        data["phases"][0]["status"] = "RUNNING"
        ledger.write_text(json.dumps(data))
        cache = self.path / "cache"
        record = cache / "units/000000-diag-0000/attempt-000001/complete.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"execution_evidence": evidence}))
        with self.assertRaisesRegex(ValueError, "not independently qualified"):
            self.w.verify_cached_unit_producers(cache, "a"*64)
        release = {"allowed_records": [{"path": str(record), "sha256": self.w.sha(record),
                                        "producer_proof_sha256": evidence["resource_supervision"]["proof_sha256"]}],
                   "release_sha256": "f"*64}
        actual = self.w.verify_cached_unit_producers(cache, "a"*64, forensic_release=release)
        self.assertEqual(actual[0]["qualification"], "EXPLICIT_ROOT_FORENSIC_RELEASE")
        self.assertEqual(json.loads(ledger.read_text())["phases"][0]["status"], "RUNNING")

    def test_forensic_release_needs_explicit_root_and_hash(self):
        path = self.path / "forensic.json"
        release = {"schema": "research4-explicit-root-rhf-unit-forensic-release-v1", "explicit_root_forensic_approval": True,
                   "snapshot_manifest_sha256": "a"*64, "hessian_worker_sha256": self.w.sha(SCRIPT),
                   "reason": "Synthetic fixture only", "allowed_records": []}
        path.write_text(json.dumps(release))
        with self.assertRaisesRegex(ValueError, "release SHA"):
            self.w.load_forensic_release(path, None, "a"*64)
        actual = self.w.load_forensic_release(path, self.w.sha(path), "a"*64)
        self.assertEqual(actual["allowed_records"], [])

    def test_negative_frequency_preserved_with_complete_modes(self):
        base = self.w.state_module().frozen_base()
        modes = np.ones((3, 3, 3))
        info = base.frequencies_payload(np.array([-2., 10., 20.]), modes, 3)
        self.assertTrue(info["normal_modes_complete"])
        self.assertEqual(info["imaginary_frequency_count"], 1)
        self.assertEqual(info["frequencies_cm1"][0], -2.)
        self.assertEqual(info["IR_intensities_status"], "NOT_COMPUTED")
        self.assertFalse(base.frequency_minimum_checks(info)["stationary_minimum_verified"])

    def test_incomplete_modes_rejected(self):
        base = self.w.state_module().frozen_base()
        with self.assertRaisesRegex(ValueError, "mode count"):
            base.frequencies_payload(np.ones(2), np.ones((2, 3, 3)), 3)


if __name__ == "__main__":
    unittest.main()
