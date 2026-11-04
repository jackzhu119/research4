"""Pure/synthetic postprocessing tests; no quantum engine or new DFT."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/analyze_spinosin_v5_modes_20261104.py"
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("independent_v5_mode_tests", SOURCE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
new_quantum = {key for key in set(sys.modules)-before if key == "pyscf" or key.startswith("pyscf.")}


class ModeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unit-closed-modes-", dir=ROOT / "reports")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def synthetic(self, eigenvalues=(1e-4, 4e-4, 9e-4)):
        helper = analysis.load_frozen(analysis.REPLAY, analysis.REPLAY_SHA, "closed_numpy_fixture")
        symbols = ["O", "H", "H"]
        coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        mass = np.array([analysis.MASS[s] for s in symbols])
        _, basis = helper.projector(coordinates, mass)
        matrix = basis @ np.diag(eigenvalues) @ basis.T
        hessian = matrix.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3) * np.sqrt(mass)[:, None, None, None] * np.sqrt(mass)[None, :, None, None]
        hessian = (hessian+hessian.transpose(1, 0, 3, 2))*.5
        modes = basis.T.reshape(3, 3, 3)/np.sqrt(mass)[None, :, None]
        cm = np.sign(eigenvalues)*np.sqrt(np.abs(eigenvalues))*helper.FACTOR
        return hessian, modes, cm, coordinates, symbols

    def test_import_does_not_import_quantum_engine(self):
        self.assertEqual(new_quantum, set())

    def test_explicit_control_label_never_spinosin(self):
        self.assertEqual(analysis.case_for_symbols(["O", "H", "H"], True), "water_software_control")
        with self.assertRaisesRegex(ValueError, "Default input"):
            analysis.case_for_symbols(["O", "H", "H"], False)
        with self.assertRaisesRegex(ValueError, "only actual closed water or ethanol"):
            analysis.case_for_symbols(["C"]*28+["H"]*32+["O"]*15, True)

    def test_default_spinosin_exact_composition(self):
        symbols = ["C"]*28+["H"]*32+["O"]*15
        self.assertEqual(analysis.case_for_symbols(symbols, False), "Spinosin")
        with self.assertRaises(ValueError):
            analysis.case_for_symbols(symbols[:-1]+["H"], False)

    def test_positive_complete_numpy_replay(self):
        hessian, modes, cm, xyz, symbols = self.synthetic()
        result = analysis.numpy_mode_audit(hessian, hessian.copy(), modes, cm, xyz, symbols)
        self.assertEqual(result["mode_count"], 3)
        self.assertLess(result["max_frequency_difference_cm1"], 1e-7)

    def test_imaginary_mode_is_preserved_and_not_abs_transformed(self):
        hessian, modes, cm, xyz, symbols = self.synthetic((-1e-4, 4e-4, 9e-4))
        self.assertLess(cm[0], 0)
        analysis.numpy_mode_audit(hessian, hessian.copy(), modes, cm, xyz, symbols)
        with self.assertRaisesRegex(ValueError, "signed normal-mode"):
            analysis.numpy_mode_audit(hessian, hessian.copy(), modes, abs(cm), xyz, symbols)

    def test_mode_sign_and_degenerate_basis_rotation_allowed(self):
        hessian, modes, cm, xyz, symbols = self.synthetic((1e-4, 1e-4, 9e-4))
        rotation = np.array([[.6, .8], [-.8, .6]])
        modes[:2] = np.einsum("ij,jab->iab", rotation, modes[:2])
        modes[2] *= -1
        analysis.numpy_mode_audit(hessian, hessian.copy(), modes, cm, xyz, symbols)

    def test_wrong_mode_vector_or_order_rejected(self):
        hessian, modes, cm, xyz, symbols = self.synthetic()
        with self.assertRaisesRegex(ValueError, "eigen-equation"):
            analysis.numpy_mode_audit(hessian, hessian.copy(), modes[[1, 0, 2]], cm, xyz, symbols)

    def test_missing_spinosin_mode_rejected_before_eigensolver(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            analysis.numpy_mode_audit(np.zeros((75, 75, 3, 3)), np.zeros((75, 75, 3, 3)),
                np.zeros((218, 75, 3)), np.zeros(218), np.zeros((75, 3)), ["C"]*28+["H"]*32+["O"]*15)

    def test_unprescribed_symmetrization_rejected(self):
        hessian, modes, cm, xyz, symbols = self.synthetic()
        wrong = hessian.copy(); wrong[0, 0, 0, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, "symmetrization"):
            analysis.numpy_mode_audit(hessian, wrong, modes, cm, xyz, symbols)

    def test_frequency_csv_requires_signs_units_and_no_ir(self):
        cm = np.array([-2., 10., 20.]); thz = cm*analysis.CM1_TO_THZ
        rows = [{"mode": str(i+1), "frequency_cm1": str(x), "frequency_thz": str(thz[i]),
                 "imaginary": str(bool(x < 0)), "IR_intensity_status": "NOT_COMPUTED"} for i, x in enumerate(cm)]
        analysis.validate_frequency_rows(rows, cm, thz)
        rows[0]["imaginary"] = "False"
        with self.assertRaisesRegex(ValueError, "signed frequency"):
            analysis.validate_frequency_rows(rows, cm, thz)

    def test_partial_target_never_reads_arrays(self):
        path = self.root / "result.json"
        path.write_text(json.dumps({"status": "CHECKPOINT_SEGMENT_COMPLETE"}))
        with patch.object(analysis.np, "load", side_effect=AssertionError("Active array read")) as loader:
            with self.assertRaisesRegex(ValueError, "NOT_RUN or incomplete"):
                analysis.analyze(path, analysis.sha(path), self.root / "out")
        loader.assert_not_called()

    def test_healthy_closure_precedes_any_array_read(self):
        path = self.root / "result.json"
        path.write_text(json.dumps({"status": "FREQUENCIES_COMPUTED", "frequencies_status": "DONE", "atoms": 75,
            "complete_global_Hessian": True, "complete_RHF_helper": True, "normal_modes_complete": True,
            "source_worker": {}, "resource_supervision": {}, "snapshot_manifest_sha256": "frozen", "phase": "hessian-segment"}))
        def failure(*a, **k): raise ValueError("Not healthy closed")
        consumer = SimpleNamespace(EXECUTION_SCHEMA="fixture", verify_closed_execution=failure)
        with patch.object(analysis, "load_frozen", return_value=consumer), \
                patch.object(analysis.np, "load", side_effect=AssertionError("Early array read")) as loader:
            with self.assertRaisesRegex(ValueError, "Not healthy closed"):
                analysis.analyze(path, analysis.sha(path), self.root / "out")
        loader.assert_not_called()

    def test_restored_gradient_parent_gate_precedes_snapshot_array_loader(self):
        path = self.root / "result.json"
        path.write_text(json.dumps({"status": "FREQUENCIES_COMPUTED", "frequencies_status": "DONE", "atoms": 75,
            "complete_global_Hessian": True, "complete_RHF_helper": True, "normal_modes_complete": True,
            "source_worker": {}, "resource_supervision": {"proof": {"cpu_hard_limit_seconds": 1200}},
            "snapshot_manifest_sha256": "frozen", "phase": "hessian-segment", "boot_id": "fixture",
            "no_new_SCF_performed": True, "stable_MF_state_unchanged": True, "IR_intensities_status": "NOT_COMPUTED",
            "gradient_gate": {"result": {"path": "fixture-not-read", "sha256": "fixture"}}}))
        def failure(*args, **kwargs): raise ValueError("Gradient producer not healthy closed")
        source_loader = Mock(side_effect=AssertionError("Snapshot loader must not run before gradient parent gate"))
        consumer = SimpleNamespace(EXECUTION_SCHEMA="fixture", state_module=source_loader,
            verify_closed_execution=lambda *a, **k: {"producer_result_sha256": analysis.sha(path), "actual_cpu_seconds": 1.},
            qualify_gradient_receipt=failure)
        with patch.object(analysis, "load_frozen", return_value=consumer):
            with self.assertRaisesRegex(ValueError, "Gradient producer"):
                analysis.analyze(path, analysis.sha(path), self.root / "out")
        source_loader.assert_not_called()

    def test_actual_20_fold_records_keep_overlap_and_unknown_alpha(self):
        rows = analysis.experimental_candidates({})
        self.assertEqual(len(rows), 20)
        self.assertLess(len({r["experimental_frequency_THz"] for r in rows}), 20)
        self.assertTrue(all(r["absolute_alpha_units"] == "UNKNOWN" and r["independent_chemical_peak"] is False for r in rows))

    def test_pairs_are_full_cartesian_not_nearest_matching(self):
        records = analysis.experimental_candidates({})
        modes = [{"mode": 1, "frequency_THz": 1., "frequency_cm1": 1/analysis.CM1_TO_THZ},
                 {"mode": 2, "frequency_THz": 1.4, "frequency_cm1": 1.4/analysis.CM1_TO_THZ}]
        pairs = analysis.cartesian_pairs(records, modes)
        self.assertEqual(len(pairs), 40)
        self.assertEqual({(p["outer_fold"], p["candidate_number"], p["theoretical_mode"]) for p in pairs},
                         {(i, j, k) for i in range(5) for j in range(1, 5) for k in (1, 2)})
        self.assertTrue(all(p["chemical_attribution_verified"] is False and p["IR_intensity_status"] == "NOT_COMPUTED" for p in pairs))

    def test_mechanism_gate_never_filters_bad_modes_or_promotes_controls(self):
        positive = np.ones(219)
        self.assertTrue(analysis.mechanism_qualified(positive, control=False, fresh_gradient_verified=True))
        for frequency in (-1., 0., np.nan):
            values = positive.copy(); values[0] = frequency
            self.assertFalse(analysis.mechanism_qualified(values, control=False, fresh_gradient_verified=True))
        self.assertFalse(analysis.mechanism_qualified(positive[:-1], control=False, fresh_gradient_verified=True))
        self.assertFalse(analysis.mechanism_qualified(positive, control=True, fresh_gradient_verified=True))
        self.assertFalse(analysis.mechanism_qualified(positive, control=False, fresh_gradient_verified=False))

    def test_candidate_tamper_rejected(self):
        target = self.root / "peaks.csv"
        target.write_bytes(analysis.PEAKS.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "SHA changed"):
            analysis.experimental_candidates({}, target)

    def test_animation_preserves_source_order_and_display_amplitude(self):
        target = self.root / "display.xyz"
        xyz = np.zeros((3, 3)); vector = np.eye(3)
        analysis.animate(target, ["O", "H", "H"], xyz, vector, 1, 100., "control", frames=5)
        lines = target.read_text().splitlines()
        self.assertEqual(len(lines), 25)
        for frame in range(5):
            self.assertIn("DISPLAY_ONLY_NOT_MD", lines[frame*5+1])
            self.assertEqual([x.split()[0] for x in lines[frame*5+2:frame*5+5]], ["O", "H", "H"])
        displaced = np.array([[float(x) for x in row.split()[1:]] for row in lines[7:10]])
        self.assertAlmostEqual(float(np.linalg.norm(displaced, axis=1).max()), .15)

    def test_preflight_has_no_fake_frequency_table_and_refuses_overwrite(self):
        output = self.root / "preflight"
        result = analysis.preflight(output)
        self.assertEqual(result["status"], "NOT_RUN")
        self.assertFalse(result["target_frequencies_read"])
        self.assertFalse(list(output.glob("*.csv")))
        with self.assertRaisesRegex(ValueError, "Preserve existing"):
            analysis.preflight(output)


if __name__ == "__main__":
    unittest.main()
