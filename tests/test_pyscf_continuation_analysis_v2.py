"""Synthetic native-exit auditing only; no quantum jobs or real final export."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continuation_analysis_v2", ROOT / "scripts/analyze_pyscf_continuation_v2_20261031.py")
analysis = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(analysis)
FIXTURES = importlib.util.spec_from_file_location("previous_analysis_test_fixtures", ROOT / "tests/test_pyscf_continuation_analysis.py")
fixtures = importlib.util.module_from_spec(FIXTURES); FIXTURES.loader.exec_module(fixtures)


def ended_row():
    return {"phase": "synthetic_native_failure", "status": "PARTIAL_OR_FAILED", "exit_code": 1,
            "ended_at_utc": "synthetic-ended", "actual_cpu_seconds": 1., "elapsed_wall_seconds": 1., "threads": 2}


class MissingWorkerResultTests(unittest.TestCase):
    def test_existing_final_result_is_preserved_without_derived_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp); actual = {"identity": "water", "status": "DONE"}
            analysis.write(location / "result.json", actual)
            result = analysis.completed_phase_result(ended_row(), location, {})
            self.assertEqual(result["status"], "DONE"); self.assertFalse(result["worker_result_missing"])
            self.assertEqual(analysis.read(location / "result.json"), actual)

    def test_native_exit_derives_partial_from_input_progress_and_real_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp); initial = fixtures.result_fixture("water")
            initial.update(status="PARTIAL", geometry_optimized=False, frequencies_status="NOT_RUN")
            analysis.write(location / "input.json", initial)
            analysis.write(location / "progress.json", {**initial, "event": "hessian_started"})
            np.save(location / "hessian_hartree_per_bohr2.npy", np.zeros((3, 3, 3, 3)))
            hashes = {}; result = analysis.completed_phase_result(ended_row(), location, hashes)
            self.assertEqual(result["status"], "PARTIAL"); self.assertTrue(result["worker_result_missing"])
            self.assertEqual(result["identity"], "water"); self.assertEqual(result["parent_exit_code"], 1)
            self.assertEqual(result["snapshot_event"], "hessian_started")
            self.assertIn("hessian_hartree_per_bohr2.npy", result["observed_closed_artifacts"])
            self.assertFalse((location / "result.json").exists())

    def test_unfinalized_frequency_done_claim_never_passes_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp); complete_claim = fixtures.result_fixture()
            analysis.write(location / "input.json", complete_claim)
            analysis.write(location / "progress.json", {**complete_claim, "event": "frequencies_complete"})
            result = analysis.completed_phase_result(ended_row(), location, {})
            self.assertEqual(result["unfinalized_frequency_status_claim"], "DONE")
            self.assertEqual(result["frequencies_status"], "NOT_CONFIRMED_MISSING_WORKER_RESULT")
            self.assertFalse(result["frequency_validation_passed"])
            self.assertFalse(analysis.valid_modes(result, "spinosin"))
            self.assertFalse((location / "result.json").exists())

    def test_snapshot_identity_source_and_method_disagreements_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp); initial = {"identity": "water", "script_sha256": "one", "settings": {"basis": "def2-svp"}}
            analysis.write(location / "input.json", initial)
            for key, bad in [("identity", "spinosin"), ("script_sha256", "two"), ("settings", {"basis": "sto-3g"})]:
                analysis.write(location / "progress.json", {**initial, key: bad})
                with self.assertRaises(ValueError): analysis.completed_phase_result(ended_row(), location, {})

    def test_running_or_unended_phase_cannot_use_failure_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            for row in [{**ended_row(), "status": "RUNNING"}, {k: v for k, v in ended_row().items() if k != "ended_at_utc"}]:
                with self.assertRaises(ValueError): analysis.completed_phase_result(row, Path(temp), {})

    def test_failed_to_start_with_no_worker_files_has_no_fabricated_identity(self):
        row = {**ended_row(), "status": "FAILED_TO_START", "exit_code": None, "actual_cpu_seconds": 0.}
        result = analysis.completed_phase_result(row, None, {})
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(result["worker_result_missing"])
        self.assertNotIn("identity", result)
        self.assertEqual(result["observed_closed_artifacts"], {})

    def test_full_synthetic_native_exit_keeps_paired_steps_and_blank_theory(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp); root = project / "run"; root.mkdir(); phase = root / "failed-water"
            fixtures.step_fixture(phase)
            worker = root / "synthetic_worker.py"; worker.write_text("# Never executed; source-binding fixture only\n")
            result = fixtures.result_fixture("water")
            result.update(status="PARTIAL", script_sha256=analysis.sha(worker), geometry_optimized=False,
                          frequencies_status="NOT_RUN", event="hessian_started")
            analysis.write(phase / "input.json", result); analysis.write(phase / "progress.json", result)
            driver = root / "driver.log"; driver.write_text("Synthetic native malloc failure, no actual quantum execution\n")
            ledger, _ = fixtures.ledger_fixture(root)
            ledger["phases"][0] = {**ended_row(), "command": ["fixture", str(worker), "--output", str(phase)],
                                   "log": str(driver), "log_sha256": analysis.sha(driver)}
            analysis.write(root / "cpu_budget.json", ledger)
            release = project / "release.json"
            analysis.write(release, {"allow_analysis_export": True, "created_at_utc": "synthetic",
                                     "cpu_budget_sha256": analysis.sha(root / "cpu_budget.json")})
            peaks = project / "peaks.csv"
            with peaks.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["outer_fold", "candidate_number", "frequency_index", "frequency_thz", "wavenumber_cm1"])
                writer.writerows((fold, number, number, 1.08 + .01 * number, (1.08 + .01 * number) / analysis.CM1_TO_THZ)
                                 for fold in range(5) for number in range(1, 5))
            before = {p.name: analysis.sha(p) for p in phase.iterdir() if p.is_file()}
            with patch.object(analysis, "PEAK_PATH", peaks), patch.object(analysis, "PEAK_SHA256", analysis.sha(peaks)):
                summary = analysis.run(root, project / "export", release)
            self.assertEqual(summary["ended_phases_without_worker_result"], 1)
            self.assertEqual(summary["paired_geometry_evaluations"], 1)
            self.assertEqual(summary["computed_frequency_rows_including_unvalidated"], 0)
            self.assertTrue(summary["no_worker_result_json_fabricated"])
            self.assertFalse((phase / "result.json").exists())
            self.assertEqual({p.name: analysis.sha(p) for p in phase.iterdir() if p.is_file()}, before)
            audit = analysis.read(project / "export/missing_worker_result_audit.json")
            self.assertFalse(audit[0]["frequency_validation_permitted"])
            self.assertEqual(audit[0]["parent_exit_code"], 1)
            with (project / "export/phase_status_and_resources.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["worker_result_missing"], "True")
            self.assertEqual(rows[0]["scientific_status"], "PARTIAL")
            with (project / "export/experiment_theory_pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertTrue(all(pair["theoretical_frequency_thz"] == "" for pair in pairs))


if __name__ == "__main__":
    unittest.main()
