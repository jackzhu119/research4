"""Synthetic audit guards only: never launch SCF, gradients or Hessian jobs."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_pyscf_continuation_20261031.py"
SPEC = importlib.util.spec_from_file_location("continuation_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def result_fixture(identity="spinosin"):
    count, atoms = (219, 75) if identity == "spinosin" else (3, 3)
    cm = np.linspace(5, 500, count)
    return {"identity": identity, "atoms": atoms, "status": "DONE",
            "settings": {**analysis.METHOD, "optimizer_thresholds": dict(analysis.THRESHOLDS),
                         "scf_conv_tol_hartree": 1e-8, "threads": 2},
            "geometry_optimized": True, "optimizer_converged": True, "scf_converged": True,
            "final_gradient_thresholds_passed": True, "frequency_validation_passed": True,
            "stationary_minimum_verified": True, "all_frequencies_real_positive": True,
            "normal_modes_complete": True, "harmonic_analysis_excludes_translations_and_rotations": True,
            "normal_mode_hessian_symmetrized": True, "frequencies_status": "DONE",
            "normal_mode_count": count, "expected_normal_mode_count": count,
            "imaginary_frequency_count": 0, "IR_intensities_status": "NOT_COMPUTED",
            "atom_rms_hartree_per_bohr": 0., "atom_max_hartree_per_bohr": 0.,
            "frequencies_cm1": cm.tolist(), "frequencies_thz": (cm * analysis.CM1_TO_THZ).tolist()}


def direct_settings():
    return {**result_fixture()["settings"], "density_fit_backend": "SCREENED_DIRECT_RKS_V4",
            "density_fitting": False, "auxbasis": None, "hessian_auxbasis_response": None,
            "direct_scf": True, "direct_scf_tol": 1e-13, "global_four_center_ERI_cache_allowed": False}


def ledger_fixture(root, status="EXITED"):
    auth = root / "authorization.json"
    checkpoint = root / "source.chk"; checkpoint.write_bytes(b"Synthetic source hash fixture, never used for SCF")
    analysis.write(auth, {"max_quantum_cpu_seconds": 43200, "source_base_commit": analysis.SOURCE_BASE,
                          "source_checkpoint_sha256": analysis.sha(checkpoint)})
    value = {"schema": "research4-pyscf-continuation-budget-v2",
             "authorization_path": str(auth), "authorization_sha256": analysis.sha(auth),
             "source_base_commit": analysis.SOURCE_BASE, "source_checkpoint_path": str(checkpoint),
             "source_checkpoint_sha256": analysis.sha(checkpoint),
             "quantum_cpu_cap_seconds": 43200, "consumed_cpu_seconds": 1.,
             "phases": [{"phase": "synthetic", "status": status, "actual_cpu_seconds": 1.,
                         "elapsed_wall_seconds": 1., "threads": 2, "exit_code": 0}]}
    analysis.write(root / "cpu_budget.json", value)
    release = {"allow_analysis_export": True, "created_at_utc": "synthetic",
               "cpu_budget_sha256": analysis.sha(root / "cpu_budget.json")}
    return value, release


def step_fixture(location):
    steps = location / "geometry_steps"; steps.mkdir(parents=True)
    prefix = steps / "step_0001"
    coordinates = np.asarray([[0., 0., 0.], [.9, .1, 0.], [-.3, .9, 0.]])
    gradient = np.asarray([[.001, .002, .003], [0., .001, 0.], [0., 0., .001]])
    np.save(prefix.with_suffix(".gradient.npy"), gradient)
    np.save(prefix.with_suffix(".coordinates_angstrom.npy"), coordinates)
    np.save(prefix.with_suffix(".coordinates_bohr.npy"), coordinates / analysis.BOHR_TO_ANGSTROM)
    prefix.with_suffix(".xyz").write_text("3\nSynthetic fixture, not a computed molecule\n" +
        "\n".join(f"{symbol} {point[0]:.12f} {point[1]:.12f} {point[2]:.12f}" for symbol, point in zip(["O", "H", "H"], coordinates)) + "\n")
    metadata = {"cycle": 1, "energy_hartree": -1., "scf_converged": True,
                **analysis.gradient_statistics(gradient), "final_gradient_thresholds_passed": False,
                "gradient_npy_sha256": analysis.sha(prefix.with_suffix(".gradient.npy")),
                "coordinates_angstrom_npy_sha256": analysis.sha(prefix.with_suffix(".coordinates_angstrom.npy")),
                "coordinates_bohr_npy_sha256": analysis.sha(prefix.with_suffix(".coordinates_bohr.npy"))}
    analysis.write(prefix.with_suffix(".json"), metadata)
    (location / "pyscf.log").write_text("Synthetic log fixture only\nconverged SCF energy = -1.0\n")
    return prefix.with_suffix(".json")


def mode_fixture(location):
    location.mkdir()
    result = result_fixture("water")
    cm = np.asarray(result["frequencies_cm1"])
    coords = np.asarray([[0., 0., 0.], [.9, .1, 0.], [-.3, .9, 0.]])
    raw = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
    raw[0, 1, 0, 1] += 1e-7
    used = .5 * (raw + raw.transpose(1, 0, 3, 2))
    np.save(location / "normal_modes.npy", np.eye(9)[:3].reshape(3, 3, 3))
    np.save(location / "hessian_hartree_per_bohr2.npy", raw)
    np.save(location / "hessian_used_for_analysis.npy", used)
    np.save(location / "normal_mode_geometry_angstrom.npy", coords)
    np.save(location / "normal_mode_geometry_bohr.npy", coords / analysis.BOHR_TO_ANGSTROM)
    np.save(location / "gradient_hartree_per_bohr.npy", np.zeros((3, 3)))
    (location / "normal_mode_geometry.xyz").write_text("3\nSynthetic modes, not DFT results\nO 0 0 0\nH .9 .1 0\nH -.3 .9 0\n")
    with (location / "frequencies.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["mode", "frequency_cm1", "frequency_thz", "IR_intensity_status"])
        writer.writerows((i + 1, value, value * analysis.CM1_TO_THZ, "NOT_COMPUTED") for i, value in enumerate(cm))
    result.update(analysis.gradient_statistics(np.zeros((3, 3))),
                  hessian_max_symmetry_difference=float(np.max(np.abs(raw - raw.transpose(1, 0, 3, 2)))),
                  raw_hessian_sha256=analysis.sha(location / "hessian_hartree_per_bohr2.npy"),
                  analysis_hessian_sha256=analysis.sha(location / "hessian_used_for_analysis.npy"),
                  normal_mode_geometry_angstrom_npy_sha256=analysis.sha(location / "normal_mode_geometry_angstrom.npy"))
    return result


class ContinuationAnalysisTests(unittest.TestCase):
    def test_complete_shape_fixture_can_pass_both_identities(self):
        for identity in ["water", "spinosin"]:
            self.assertTrue(analysis.valid_modes(result_fixture(identity), identity))
        self.assertFalse(analysis.valid_modes(result_fixture("water"), "spinosin"))

    def test_zero_negative_nonfinite_and_wrong_units_cannot_pass(self):
        for value in [0., -1., np.nan]:
            result = result_fixture(); result["frequencies_cm1"][0] = value
            self.assertFalse(analysis.valid_modes(result, "spinosin"))
        result = result_fixture(); result["frequencies_thz"][0] += .1
        self.assertFalse(analysis.valid_modes(result, "spinosin"))

    def test_mode_count_optimizer_and_false_minimum_flags_rejected(self):
        for key, value in [("normal_mode_count", 218), ("geometry_optimized", False),
                           ("optimizer_converged", False), ("stationary_minimum_verified", False),
                           ("normal_mode_hessian_symmetrized", False), ("IR_intensities_status", "MADE_UP")]:
            result = result_fixture(); result[key] = value
            self.assertFalse(analysis.valid_modes(result, "spinosin"), key)

    def test_method_and_numerical_gradient_checks_are_not_trusted_flags(self):
        result = result_fixture(); result["settings"]["grid_level"] = 2
        self.assertFalse(analysis.valid_modes(result, "spinosin"))

    def test_direct_backend_is_separate_and_requires_its_safe_settings(self):
        result = result_fixture(); result["settings"] = direct_settings()
        self.assertTrue(analysis.valid_modes(result, "spinosin"))
        for key, value in [("density_fitting", True), ("direct_scf", False),
                           ("global_four_center_ERI_cache_allowed", True), ("auxbasis", "def2-universal-jfit")]:
            settings = direct_settings(); settings[key] = value
            self.assertFalse(analysis.method_matches(settings), key)

    def test_frequency_cannot_borrow_an_optimization_from_another_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); xyz = root / "optimized.xyz"; xyz.write_text("Synthetic geometry binding only\n")
            precise = root / "optimized.npy"; np.save(precise, np.zeros((75, 3)))
            parent = result_fixture(); parent.update(optimized_xyz_sha256=analysis.sha(xyz),
                optimized_coordinates_angstrom_npy_sha256=analysis.sha(precise))
            parent_path = root / "parent.json"; analysis.write(parent_path, parent)
            result = result_fixture(); result.update(phase="spinosin-frequency", settings=direct_settings(),
                input_xyz_sha256=analysis.sha(xyz), input_coordinates_angstrom_npy_sha256=analysis.sha(precise),
                optimization_result_sha256=analysis.sha(parent_path))
            row = {"command": ["fixture", "--optimization-result", str(parent_path), "--input-xyz", str(xyz), "--input-coordinates-npy", str(precise)]}
            with self.assertRaises(ValueError): analysis.frequency_parent_guard(root, row, result, {})
            parent["settings"] = direct_settings(); analysis.write(parent_path, parent)
            result["optimization_result_sha256"] = analysis.sha(parent_path)
            hashes = {}; analysis.frequency_parent_guard(root, row, result, hashes)
            self.assertEqual(len(hashes), 3)
            precise.write_bytes(b"tampered")
            with self.assertRaises(ValueError): analysis.frequency_parent_guard(root, row, result, {})

    def test_missing_final_gradient_cannot_support_optimized_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                analysis.audit_final_gradient(Path(temp), result_fixture(), {})
        result = result_fixture(); result["atom_max_hartree_per_bohr"] = .001
        self.assertFalse(analysis.valid_modes(result, "spinosin"))

    def test_running_or_violated_budget_is_not_a_final_report(self):
        with tempfile.TemporaryDirectory() as temp:
            for status in ["RUNNING", "BUDGET_VIOLATION_REQUIRES_REVIEW"]:
                root = Path(temp); _, release = ledger_fixture(root, status)
                with self.assertRaises(ValueError): analysis.authorize(root, release)

    def test_cpu_total_and_authorization_hash_are_independent_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ledger, release = ledger_fixture(root)
            self.assertEqual(analysis.authorize(root, release)[0]["consumed_cpu_seconds"], 1)
            ledger["consumed_cpu_seconds"] = 3.; analysis.write(root / "cpu_budget.json", ledger)
            release["cpu_budget_sha256"] = analysis.sha(root / "cpu_budget.json")
            with self.assertRaises(ValueError): analysis.authorize(root, release)
            ledger, release = ledger_fixture(root)
            (root / "authorization.json").write_text("{}")
            with self.assertRaises(ValueError): analysis.authorize(root, release)

    def test_output_escape_and_reuse_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir()
            with self.assertRaises(ValueError):
                analysis.phase_location(root, {"command": ["worker", "--output", str(root.parent)], "status": "EXITED"})
            with self.assertRaises(FileExistsError): analysis.run(root, root, root / "irrelevant")

    def test_immutable_checkpoint_and_source_base_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ledger, release = ledger_fixture(root)
            (root / "source.chk").write_bytes(b"tampered fixture")
            with self.assertRaises(ValueError): analysis.authorize(root, release)
            ledger, release = ledger_fixture(root); ledger["source_base_commit"] = "wrong"
            analysis.write(root / "cpu_budget.json", ledger)
            release["cpu_budget_sha256"] = analysis.sha(root / "cpu_budget.json")
            with self.assertRaises(ValueError): analysis.authorize(root, release)

    def test_completed_step_recomputes_five_gradient_statistics_and_energy_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata = step_fixture(Path(temp)); bindings = {}
            row = analysis.audit_step(metadata, "water", "synthetic", bindings)
            self.assertEqual(row["cycle"], 1); self.assertFalse(row["gradient_thresholds_passed"])
            self.assertEqual(len(bindings), 6)
            changed = analysis.read(metadata); changed["atom_max_hartree_per_bohr"] *= 2; analysis.write(metadata, changed)
            with self.assertRaises(ValueError): analysis.audit_step(metadata, "water", "synthetic", {})

    def test_missing_paired_array_or_wrong_scf_energy_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata = step_fixture(Path(temp))
            (Path(temp) / "pyscf.log").write_text("converged SCF energy = -2.0\n")
            with self.assertRaises(ValueError): analysis.audit_step(metadata, "water", "synthetic", {})
            metadata.with_suffix(".gradient.npy").unlink()
            with self.assertRaises(ValueError): analysis.audit_step(metadata, "water", "synthetic", {})

    def test_saved_raw_and_sym_hessians_and_mode_geometry_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "modes"; result = mode_fixture(location)
            rows, modes, geometry, replay = analysis.load_modes(location, result, {}, replay=False)
            self.assertEqual(len(rows), 3); self.assertEqual(modes.shape, (3, 3, 3))
            self.assertEqual(replay["status"], "NOT_REQUESTED"); self.assertEqual(len(geometry[0]), 3)
            np.save(location / "hessian_used_for_analysis.npy", np.zeros((3, 3, 3, 3)))
            with self.assertRaises(ValueError): analysis.load_modes(location, result, {}, replay=False)

    def test_negative_computed_frequencies_kept_but_not_accepted(self):
        result = result_fixture(); result["frequencies_cm1"][0] = -5.
        result["frequencies_thz"][0] = -5. * analysis.CM1_TO_THZ
        self.assertFalse(analysis.valid_modes(result, "spinosin"))
        observed = [{"experimental_frequency_thz": 1.08, "experimental_wavenumber_cm1": 1.08 / analysis.CM1_TO_THZ}]
        blocked = analysis.LEGACY.comparison_rows(observed, [])
        self.assertIsNone(blocked[0]["theoretical_frequency_thz"])
        self.assertIsNone(blocked[0]["absolute_difference_thz"])

    def test_gated_synthetic_partial_export_does_not_invent_frequencies(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp); root = project / "run"; root.mkdir()
            phase = root / "synthetic-water"; step_fixture(phase)
            result = result_fixture("water"); result.update(status="PARTIAL", geometry_optimized=False, frequencies_status="NOT_RUN")
            worker = root / "synthetic_worker.py"; worker.write_text("# Synthetic source fixture, never executed\n")
            result["script_sha256"] = analysis.sha(worker); analysis.write(phase / "result.json", result)
            driver_log = root / "driver.log"; driver_log.write_text("Synthetic driver log, no quantum execution\n")
            ledger, _ = ledger_fixture(root)
            ledger["phases"][0].update(command=["synthetic", str(worker), "--output", str(phase)], exit_code=2,
                                      status="PARTIAL_OR_FAILED", log=str(driver_log), log_sha256=analysis.sha(driver_log))
            analysis.write(root / "cpu_budget.json", ledger)
            release = project / "release.json"
            analysis.write(release, {"allow_analysis_export": True, "created_at_utc": "synthetic",
                                     "cpu_budget_sha256": analysis.sha(root / "cpu_budget.json")})
            peaks = project / "peaks.csv"
            with peaks.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["outer_fold", "candidate_number", "frequency_index", "frequency_thz", "wavenumber_cm1"])
                writer.writerows((fold, number, number, 1.08 + .01 * number, (1.08 + .01 * number) / analysis.CM1_TO_THZ)
                                 for fold in range(5) for number in range(1, 5))
            with patch.object(analysis, "PEAK_PATH", peaks), patch.object(analysis, "PEAK_SHA256", analysis.sha(peaks)):
                summary = analysis.run(root, project / "export", release)
            self.assertEqual(summary["new_SCF_gradient_Hessian_evaluations"], 0)
            self.assertEqual(summary["validated_spinosin_frequency_rows"], 0)
            self.assertEqual(summary["computed_frequency_rows_including_unvalidated"], 0)
            self.assertEqual(summary["experimental_candidate_rows"], 20)
            self.assertTrue((project / "export/mechanism_comparison_status.pdf").exists())
            with (project / "export/experiment_theory_pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(row["theoretical_frequency_thz"] == "" for row in rows))
            driver_log.write_text("changed")
            with self.assertRaises(ValueError): analysis.run(root, project / "changed-driver-export", release)


if __name__ == "__main__":
    unittest.main()
