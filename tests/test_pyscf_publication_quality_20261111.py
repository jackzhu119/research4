"""Regression tests for evidence/claim failures the original mode gate missed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "scripts", HERE.parent / "pyscf_audit"):
    sys.path.insert(0, str(candidate))
import pyscf_repair_modes_20261111 as modes
import pyscf_publication_quality_20261111 as quality


def save_array(folder, name, value):
    path = folder / (name + ".npy")
    np.save(path, value, allow_pickle=False)
    return {"path": str(path), "sha256": quality.sha256(path)}


def point_metadata(symbols, coords, gradient, dipole, energy, folder=None):
    value = {
        "status": "DONE", "qualified": True, "native_converged": True,
        "symbols": symbols, "coordinates_bohr": coords.tolist(),
        "gradient": gradient.tolist(), "dipole_au": dipole.tolist(),
        "energy_hartree": float(energy), "settings": {"charge": 0},
        "input_fingerprint": hashlib.sha256(coords.tobytes()).hexdigest(),
        "nuclear_gradient_max": float(np.max(np.abs(gradient))),
        "nuclear_gradient_rms": float(np.sqrt(np.mean(gradient ** 2))),
        "electronic_audit": {"fresh_orbital_gradient_norm": 1e-9,
            "fresh_energy_difference_hartree": 0.0, "electron_count_error": 0.0},
    }
    if folder is not None:
        folder.mkdir(parents=True)
        value["array_refs"] = {
            "coordinates_bohr": save_array(folder, "coordinates_bohr", coords),
            "gradient": save_array(folder, "gradient", gradient),
            "dipole_au": save_array(folder, "dipole_au", dipole),
        }
        quality.atomic_json(folder / "result.json", value)
    return value


def make_case(root, *, steps=(0.001, 0.0005), rigid_contamination=0.0,
              ir_scales=(1.0, 1.0), negative=False):
    symbols = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 1.11], [-1.43, 0.0, 1.11]])
    masses = modes.atomic_masses(symbols)
    external, basis, _ = modes.external_basis(coords, masses)
    massroot = np.sqrt(np.repeat(masses, 3))
    eigenvalues = np.array([-0.001 if negative else 0.001, 0.003, 0.005])
    hessian = massroot[:, None] * (basis @ np.diag(eigenvalues) @ basis.T) * massroot[None, :]
    if rigid_contamination:
        hessian += rigid_contamination * np.outer(massroot * external[:, 0], massroot * external[:, 0])
    # Neutral, zero-dipole harmonic model: total dipole has derivatives only
    # in the internal vibration subspace, so both rigid-body sum rules hold.
    derivative = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.2], [0.3, 0.2, 0.1]]) @ basis.T * massroot[None, :]
    center = point_metadata(symbols, coords, np.zeros_like(coords), np.zeros(3), -75.0)
    all_qualities, hessians, derivatives = [], [], []
    fd = root / "finite-differences"
    for hi, step in enumerate(steps):
        positive, negative_points = [], []
        dip = derivative * ir_scales[hi]
        for column in range(coords.size):
            for sign, label, target in ((1, "plus", positive), (-1, "minus", negative_points)):
                displaced = coords.copy()
                displaced.flat[column] += sign * step
                delta = (displaced - coords).ravel()
                point = point_metadata(symbols, displaced, (hessian @ delta).reshape(coords.shape),
                    dip @ delta, -75.0 + 0.5 * delta @ hessian @ delta,
                    fd / f"h{hi}" / f"c{column:03d}-{label}")
                target.append(point)
                all_qualities.append({"qualified": True, "native_converged": True,
                    "fresh_orbital_gradient_norm": 1e-9, "energy_recheck_difference_hartree": 0.0,
                    "electron_count_error": 0.0})
        hessians.append(np.stack([(np.array(p["gradient"]) - np.array(m["gradient"])).ravel() / (2 * step)
                                 for p, m in zip(positive, negative_points)], axis=1))
        derivatives.append(np.stack([(np.array(p["dipole_au"]) - np.array(m["dipole_au"])) / (2 * step)
                                    for p, m in zip(positive, negative_points)], axis=1))
    cq = {**all_qualities[0], "nuclear_gradient_max": 0.0, "nuclear_gradient_rms": 0.0,
          "orbital_stability_passed": True}
    original = modes.analyze_modes(symbols, coords, hessians, derivatives, root / "modes",
                                  center_quality=cq, point_quality=all_qualities, steps_bohr=steps)
    return center, original


class PublicationQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def audit(self, center, **kwargs):
        return quality.audit_publication(self.root / "modes", self.root / "finite-differences", center,
            electronic_stability_passed=kwargs.get("electronic_stability_passed", True))

    def test_complete_physical_derivatives_pass_and_receipt_authenticates_original_points(self):
        center, _ = make_case(self.root)
        receipt = self.audit(center)
        self.assertTrue(receipt["paper_ready_molecular_harmonic_data"], receipt)
        self.assertEqual(receipt["original_displacement_point_count"], 36)
        self.assertGreaterEqual(len(receipt["authenticated_artifact_sha256"]), 147)
        self.assertFalse(receipt["chemical_peak_assignment_verified"])
        self.assertFalse(receipt["global_conformer_minimum_verified"])
        self.assertTrue((self.root / "modes/publication_quality.json").is_file())

    def test_repeating_one_step_is_not_step_convergence(self):
        center, original = make_case(self.root, steps=(0.001, 0.001))
        self.assertTrue(original["qualification"]["paper_ready_molecular_harmonic_data"])
        receipt = self.audit(center)
        self.assertFalse(receipt["gates"]["distinct_steps"])
        self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])

    def test_projection_cannot_hide_rigid_body_curvature(self):
        center, original = make_case(self.root, rigid_contamination=0.2)
        self.assertTrue(original["qualification"]["paper_ready_molecular_harmonic_data"])
        receipt = self.audit(center)
        self.assertFalse(receipt["gates"]["force_sum_rules_passed"])
        self.assertFalse(receipt["minimum_verified"])

    def test_ir_divergence_vetoes_ir_claim_while_preserving_verified_minimum(self):
        center, original = make_case(self.root, ir_scales=(1.0, 100.0))
        self.assertTrue(original["qualification"]["paper_ready_molecular_harmonic_data"])
        receipt = self.audit(center)
        self.assertTrue(receipt["minimum_verified"], receipt)
        self.assertFalse(receipt["gates"]["ir_step_convergence_passed"])
        self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])

    def test_missing_point_is_not_a_complete_ledger(self):
        center, _ = make_case(self.root)
        (self.root / "finite-differences/h0/c000-plus/result.json").unlink()
        receipt = self.audit(center)
        self.assertFalse(receipt["gates"]["complete_unique_displacement_ledger"])
        self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])

    def test_wrong_displacement_identity_is_rejected_before_reassembly(self):
        center, _ = make_case(self.root)
        point = self.root / "finite-differences/h0/c000-plus/result.json"
        value = json.loads(point.read_text())
        value["coordinates_bohr"][0][0] *= -1
        quality.atomic_json(point, value)
        receipt = self.audit(center)
        self.assertTrue(any("identity mismatch" in x for x in receipt["issues"]))
        self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])

    def test_changed_raw_gradient_bytes_are_rejected(self):
        center, _ = make_case(self.root)
        point = self.root / "finite-differences/h0/c000-plus/gradient.npy"
        value = np.load(point)
        value[0, 0] += 0.1
        np.save(point, value)
        receipt = self.audit(center)
        self.assertTrue(any("seal mismatch" in x for x in receipt["issues"]))
        self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])

    def test_mixed_grid_or_electronic_model_settings_are_rejected(self):
        center, _ = make_case(self.root)
        point = self.root / "finite-differences/h0/c000-plus/result.json"
        original = json.loads(point.read_text())
        for changed in ({"grid_level": 7}, {"xc": "B3LYP"}, {"basis": "def2-TZVP"},
                        {"density_fitting": True}):
            with self.subTest(changed=changed):
                value = dict(original, settings={**original["settings"], **changed})
                quality.atomic_json(point, value)
                receipt = self.audit(center)
                self.assertFalse(receipt["gates"]["consistent_electronic_model_settings"])
                self.assertTrue(any("Electronic model settings differ" in x for x in receipt["issues"]))
                self.assertFalse(receipt["minimum_verified"])
                self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])

    def test_electronic_flag_cannot_replace_fresh_strict_residual(self):
        center, _ = make_case(self.root)
        point = self.root / "finite-differences/h0/c000-plus/result.json"
        value = json.loads(point.read_text())
        value["electronic_audit"]["fresh_orbital_gradient_norm"] = 2e-8
        quality.atomic_json(point, value)
        receipt = self.audit(center)
        self.assertFalse(receipt["gates"]["all_points_electronically_qualified"])
        self.assertFalse(receipt["minimum_verified"])

    def test_missing_stability_evidence_is_not_pass(self):
        center, _ = make_case(self.root)
        self.assertFalse(self.audit(center, electronic_stability_passed=None)["minimum_verified"])

    def test_different_center_geometry_is_not_accepted(self):
        center, _ = make_case(self.root)
        center["coordinates_bohr"][0][0] += 0.01
        receipt = self.audit(center)
        self.assertFalse(receipt["paper_ready_molecular_harmonic_data"])
        self.assertTrue(any("Center point identity" in x for x in receipt["issues"]))

    def test_historical_true_flag_cannot_override_actual_negative_curvature(self):
        center, original = make_case(self.root, negative=True)
        self.assertFalse(original["minimum_verified"])
        path = self.root / "modes/result.json"
        changed = json.loads(path.read_text())
        changed["minimum_verified"] = True
        changed["qualification"]["paper_ready_molecular_harmonic_data"] = True
        quality.atomic_json(path, changed)
        receipt = self.audit(center)
        self.assertFalse(receipt["gates"]["all_vibrational_eigenvalues_positive"])
        self.assertFalse(receipt["minimum_verified"])

    def test_degenerate_mode_rotation_preserves_subspace_total_ir(self):
        symbols = ["O", "H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 1.11], [-1.43, 0.0, 1.11]])
        masses = modes.atomic_masses(symbols)
        _, basis, _ = modes.external_basis(coords, masses)
        massroot = np.sqrt(np.repeat(masses, 3))
        hessian = massroot[:, None] * (basis @ np.diag([0.001, 0.001, 0.005]) @ basis.T) * massroot[None, :]
        derivative = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.2], [0.3, 0.2, 0.1]]) @ basis.T * massroot[None, :]
        first = modes.normal_modes(symbols, coords, hessian, derivative)
        final = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in first.items()}
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        final["mass_weighted_modes"][:, :2] = final["mass_weighted_modes"][:, :2] @ rotation
        final["normal_coordinate_dipole_derivatives_au_per_sqrt_amu"][:, :2] = final["normal_coordinate_dipole_derivatives_au_per_sqrt_amu"][:, :2] @ rotation
        comparison = quality.compare_steps(first, final)
        self.assertTrue(comparison["ir_converged"], comparison)
        self.assertTrue(comparison["all_subspaces_agree"], comparison)


if __name__ == "__main__":
    unittest.main()
