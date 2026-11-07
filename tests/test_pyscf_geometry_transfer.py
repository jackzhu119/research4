"""Synthetic transfer-guard fixtures, never quantum or reported research results."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("geometry_transfer", ROOT / "scripts/prepare_pyscf_geometry_transfer_20261031.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.source = self.root / "phase"; self.source.mkdir()
        self.output, self.ledger, self.release = self.root / "prepared", self.root / "budget.json", self.root / "release.json"
        original = (ROOT / "reports/pyscf-spinosin-pilot-20261031/conformers/C001_mmff94s.xyz").read_text().splitlines()
        self.symbols = [line.split()[0] for line in original[2:]]
        self.coords = np.asarray([line.split()[1:] for line in original[2:]], float)
        self.xyz = "75\nSYNTHETIC GUARD FIXTURE, NOT A QUANTUM RESULT\n" + "\n".join(original[2:]) + "\n"
        self.settings = {"density_fit_backend": "SCREENED_DIRECT_RKS_V4", "density_fitting": False,
            "auxbasis": None, "xc": "PBE", "basis": "def2-svp", "grid_level": 1,
            "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "charge": 0, "spin_2S": 0,
            "multiplicity": 1, "direct_scf": True, "direct_scf_tol": 1e-13,
            "scf_conv_tol_hartree": 1e-8, "global_four_center_ERI_cache_allowed": False,
            "optimizer_thresholds": dict(worker.THRESHOLDS)}
        (self.source / "optimized.xyz").write_text(self.xyz)
        np.save(self.source / "optimized_coordinates_angstrom.npy", self.coords)
        np.save(self.source / "gradient_hartree_per_bohr.npy", np.ones((75, 3)) * 1e-5)
        self.result = {"identity": "spinosin", "phase": "spinosin-optimize", "atoms": 75, "electrons": 320,
            "status": "DONE", "geometry_optimized": True, "optimizer_converged": True,
            "scf_converged": True, "final_gradient_thresholds_passed": True, "settings": self.settings,
            "optimized_xyz_sha256": worker.sha(self.source / "optimized.xyz"),
            "optimized_coordinates_angstrom_npy_sha256": worker.sha(self.source / "optimized_coordinates_angstrom.npy"),
            "energy_hartree": -100.0, **self.stats(1e-5)}
        self.budget = {"phases": [{"phase": "synthetic-source", "status": "EXITED", "exit_code": 0,
            "actual_cpu_seconds": 1.0, "ended_at_utc": "2026-11-01T00:00:01+00:00",
            "command": ["fixture-worker", "--output", str(self.source)]}]}
        self.bind()

    @staticmethod
    def stats(value):
        return {"component_rms_hartree_per_bohr": value, "component_max_hartree_per_bohr": value,
                "atom_rms_hartree_per_bohr": math.sqrt(3) * value,
                "atom_max_hartree_per_bohr": math.sqrt(3) * value, "gradient_norm_hartree_per_bohr": 15 * value}

    def bind(self):
        worker.write(self.source / "result.json", self.result)
        worker.write(self.ledger, self.budget)
        worker.write(self.release, {"created_at_utc": "2026-11-01T00:00:02+00:00", "allow_geometry_transfer": True,
            "cpu_budget_sha256": worker.sha(self.ledger), "source_result_sha256": worker.sha(self.source / "result.json")})

    def prepare(self, purpose="frequency"):
        return worker.prepare(self.source / "result.json", self.source, self.ledger, self.output, purpose, self.release)

    def partial(self):
        self.result.update(status="PARTIAL", geometry_optimized=False, optimizer_converged=False,
                           energy_hartree=-9999., optimized_xyz_sha256="must-not-use-returned-trial")
        directory = self.source / "geometry_steps"; directory.mkdir()
        for cycle in (1, 2):
            step = directory / f"step_{cycle:04d}"
            step.with_suffix(".xyz").write_text(self.xyz)
            np.save(step.with_suffix(".coordinates_angstrom.npy"), self.coords)
            np.save(step.with_suffix(".gradient.npy"), np.ones((75, 3)) * .001)
            worker.write(step.with_suffix(".json"), {"cycle": cycle, "scf_converged": True,
                "energy_hartree": -10. * cycle, **self.stats(.001),
                "coordinates_angstrom_npy_sha256": worker.sha(step.with_suffix(".coordinates_angstrom.npy")),
                "gradient_npy_sha256": worker.sha(step.with_suffix(".gradient.npy"))})
        (self.source / "calculation.chk").write_text("SYNTHETIC TRIAL CHECKPOINT MUST NEVER BE READ")
        self.bind()


class GeometryTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)

    def test_frequency_preserves_exact_xyz_and_fullprecision_hashes(self):
        f = self.fixture; result = f.prepare()
        self.assertEqual(result["new_quantum_jobs"], 0)
        self.assertFalse(result["worker_launched"])
        self.assertEqual(worker.sha(f.output / "transfer_geometry.xyz"), f.result["optimized_xyz_sha256"])
        self.assertEqual(worker.sha(f.output / "transfer_coordinates_angstrom.npy"), f.result["optimized_coordinates_angstrom_npy_sha256"])
        provenance = worker.read(f.output / "transfer_geometry_provenance.json")
        self.assertTrue(provenance["stereochemistry_verified"])
        self.assertEqual(provenance["source_inchikey"], "VGGSULWDCMWZPO-ODEMIOGVSA-N")

    def test_partial_only_last_completed_step_and_never_checkpoint(self):
        f = self.fixture; f.partial()
        original_read = Path.read_bytes
        def no_checkpoint(path):
            if path.suffix == ".chk": raise AssertionError("Checkpoint must not be read")
            return original_read(path)
        with mock.patch.object(Path, "read_bytes", no_checkpoint):
            result = f.prepare("optimization")
        self.assertEqual(result["selection"]["source_cycle"], 2)
        self.assertEqual(result["selection"]["paired_source_energy_hartree"], -20.)
        self.assertFalse(result["selection"]["source_geometry_optimized"])
        self.assertFalse(result["checkpoint_read"])
        self.assertFalse(result["optimizer_history_restored"])

    def test_running_ledger_rejected_even_if_source_has_final_json(self):
        f = self.fixture; f.budget["phases"][0]["status"] = "RUNNING"; f.bind()
        with self.assertRaises(ValueError): f.prepare()
        self.assertFalse(f.output.exists())

    def test_ledger_and_result_hash_binding_rejects_mutation(self):
        f = self.fixture; f.ledger.write_text(f.ledger.read_text() + " ")
        with self.assertRaises(ValueError): f.prepare()
        f.bind(); (f.source / "result.json").write_text((f.source / "result.json").read_text() + " ")
        with self.assertRaises(ValueError): f.prepare()

    def test_source_must_be_ended_recorded_owner(self):
        f = self.fixture; f.budget["phases"][0]["command"][-1] = str(f.root / "other-source"); f.bind()
        with self.assertRaises(ValueError): f.prepare()

    def test_release_must_be_dated_after_phase_end(self):
        f = self.fixture; release = worker.read(f.release); release["created_at_utc"] = "2026-11-01T00:00:00+00:00"; worker.write(f.release, release)
        with self.assertRaises(ValueError): f.prepare()

    def test_df_method_and_partial_state_cannot_be_frequency_source(self):
        f = self.fixture; f.settings.update(density_fitting=True, density_fit_backend="INCORE_DFJ_RELEASE_BEFORE_GRAD_V3"); f.bind()
        with self.assertRaises(ValueError): f.prepare()
        f.settings.update(density_fitting=False, density_fit_backend="SCREENED_DIRECT_RKS_V4"); f.partial()
        with self.assertRaises(ValueError): f.prepare()

    def test_changed_coordinate_hash_rejected(self):
        f = self.fixture; np.save(f.source / "optimized_coordinates_angstrom.npy", f.coords + .1)
        with self.assertRaises(ValueError): f.prepare()

    def test_false_stationarity_flag_cannot_override_actual_gradient(self):
        f = self.fixture; np.save(f.source / "gradient_hartree_per_bohr.npy", np.ones((75, 3)) * .001)
        f.result.update(f.stats(.001)); f.bind()
        with self.assertRaises(ValueError): f.prepare()

    def test_corrupt_completed_gradient_never_falls_back_to_trial(self):
        f = self.fixture; f.partial(); p = f.source / "geometry_steps/step_0002.gradient.npy"
        np.save(p, np.zeros((74, 3)))
        with self.assertRaises(ValueError): f.prepare("optimization")

    def test_identity_verification_failure_blocks_output(self):
        f = self.fixture
        with mock.patch.object(worker, "verify_identity", side_effect=ValueError("stereochemical identity mismatch")) as verify:
            with self.assertRaises(ValueError): f.prepare()
        verify.assert_called_once(); self.assertFalse(f.output.exists())

    def test_existing_output_and_source_result_alias_refused(self):
        f = self.fixture; f.prepare(); before = worker.sha(f.output / "transfer_manifest.json")
        with self.assertRaises(ValueError): f.prepare()
        self.assertEqual(worker.sha(f.output / "transfer_manifest.json"), before)
        with self.assertRaises(ValueError): worker.prepare(f.source / "other.json", f.source, f.ledger, f.root / "fresh", "frequency", f.release)


if __name__ == "__main__":
    unittest.main()
