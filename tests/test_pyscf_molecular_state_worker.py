"""Pure JSON/tiny NumPy fixtures; no SCF, gradient, optimizer or integrals."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pyscf_molecular_state_worker_20261103.py"


class MolecularStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("molecular_state_pure_test", SCRIPT)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.w = self.worker

    def record(self, path):
        return self.w.artifact_record(path)

    def input(self):
        xyz = self.path / "input.xyz"
        xyz.write_text("3\nSynthetic water-coordinate fixture only\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
        coordinates = self.path / "coordinates.npy"
        np.save(coordinates, np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), allow_pickle=False)
        manifest = {"schema": self.w.INPUT_SCHEMA, "molecule_id": "synthetic_fixture_not_a_calculation",
                    "atom_symbols": ["O", "H", "H"], "composition": {"O": 1, "H": 2}, "formula": "H2O",
                    "charge": 0, "spin_2S": 0, "multiplicity": 1, "radical_electrons": 0,
                    "electron_count": 10, "connectivity_verified": True, "stereochemistry_checked": True,
                    "input_xyz": self.record(xyz), "coordinates_angstrom_npy": self.record(coordinates),
                    "source_artifacts": []}
        return manifest

    def write_input(self, value):
        path = self.path / "input.json"
        path.write_text(json.dumps(value))
        return path

    def arrays(self):
        return {"MO_coeff": np.eye(2), "MO_energy": np.array([-1., 1.]), "MO_occ": np.array([2., 0.]),
                "grid_coordinates": np.array([[0., 0., 0.], [1., 1., 1.]]), "grid_weights": np.ones(2),
                "coordinates_angstrom": np.array([[0., 0., 0.], [1., 0., 0.]]), "gradient": np.zeros((2, 3)),
                "grid_non0tab": np.ones((1, 1), dtype=np.uint8)}

    def snapshot(self, closed=True):
        w = self.w
        directory = self.path / "snapshot"
        directory.mkdir()
        arrays = self.arrays()
        files = {}
        for name, array in arrays.items():
            target = directory / (name + ".npy")
            np.save(target, array, allow_pickle=False)
            files[name] = {"file": target.name, "sha256": w.sha(target), **w.frozen_base().array_fingerprint(array)}
        molecule = directory / "molecule.json"
        molecule.write_text('{"synthetic":"not a real molecule or calculation"}')
        log = self.path / "payload.log"
        log.write_text("Pure fixture, not scientific evidence\n")
        proof_file = self.path / "proof.json"
        ledger_file = self.path / "ledger.json"
        proof = {"schema": "research4-pyscf-actual-cpu-supervisor-proof-v4", "phase": "fixture",
                 "ledger_path": str(ledger_file), "supervisor_script_sha256": w.SUPERVISOR_SHA}
        proof_file.write_text(json.dumps(proof))
        supervision = {"proof_path": str(proof_file), "proof_sha256": w.sha(proof_file), "proof": proof}
        source = self.record(SCRIPT)
        manifest = {"schema": w.SNAPSHOT_SCHEMA, "status": "EXACT_SCF_GRADIENT_STATE_SAVED",
                    "source_worker": source, "frozen_V8_sha256": w.BASE_SHA, "supervisor_sha256": w.SUPERVISOR_SHA,
                    "molecule": {"file": molecule.name, "sha256": w.sha(molecule)}, "arrays": files,
                    "molecule_fingerprint": {"natm": 2, "nao": 2, "nelectron": 2},
                    "settings": {"xc": "PBE"}, "environment": {"synthetic": True}, "energy_hartree": -1.,
                    "producer": {"result_path": str(self.path / "result.json"),
                                 "resource_supervision": supervision, "input_manifest": {"synthetic": True}}}
        target = directory / "manifest.json"
        target.write_text(json.dumps(manifest))
        result = {"schema": w.RESULT_SCHEMA, "status": "DONE", "snapshot_manifest": self.record(target),
                  "resource_supervision": supervision, "source_worker": source, "settings": manifest["settings"],
                  "versions": manifest["environment"], "input_manifest": manifest["producer"]["input_manifest"],
                  "energy_hartree": -1., "geometry_optimized": True, "scf_converged": True,
                  "final_gradient_thresholds_passed": True,
                  "native_direct_driver_control": {"original_driver_restored": True, "restoration_failures": 0}}
        (self.path / "result.json").write_text(json.dumps(result))
        row = {"phase": "fixture", "status": "EXITED" if closed else "RUNNING", "exit_code": 0,
               "ended_at_utc": "2026-11-04T00:00:00+00:00", "actual_cpu_seconds": 0.,
               "supervisor_proof_path": str(proof_file), "supervisor_proof_sha256": w.sha(proof_file),
               "log": str(log), "log_sha256": w.sha(log), "command": ["python", str(SCRIPT)]}
        row.update({name: False for name in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed")})
        ledger = {"schema": "research4-pyscf-actual-cpu-budget-v4", "runner_sha256": w.SUPERVISOR_SHA, "phases": [row]}
        ledger_file.write_text(json.dumps(ledger))
        return directory, manifest, row, ledger_file

    def test_input_full_atom_order_valid(self):
        path = self.write_input(self.input())
        manifest, symbols, coords = self.w.validate_input(path, self.w.sha(path))
        self.assertEqual(symbols, ["O", "H", "H"])
        self.assertEqual(coords.shape, (3, 3))

    def test_input_sha_mandatory(self):
        path = self.write_input(self.input())
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            self.w.validate_input(path, None)

    def test_same_composition_wrong_atom_order_rejected(self):
        value = self.input()
        value["atom_symbols"] = ["H", "H", "O"]
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "atom order"):
            self.w.validate_input(path, self.w.sha(path))

    def test_electron_count_rejected(self):
        value = self.input()
        value["electron_count"] = 12
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "Electron"):
            self.w.validate_input(path, self.w.sha(path))

    def test_formula_rejected(self):
        value = self.input()
        value["formula"] = "H2O2"
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "formula"):
            self.w.validate_input(path, self.w.sha(path))

    def test_screen_alias_disagreement_rejected(self):
        arrays = self.arrays()
        arrays["grid_screen_index"] = np.zeros((1, 1), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            self.w.validate_arrays(arrays, 2, 2, 2)

    def test_screen_full_basis_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "full basis"):
            self.w.validate_arrays(self.arrays(), 2, 2, 2, nbas=2)

    def test_radical_rejected(self):
        value = self.input()
        value["radical_electrons"] = 1
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "singlets"):
            self.w.validate_input(path, self.w.sha(path))

    def test_missing_graph_review_rejected(self):
        value = self.input()
        value["connectivity_verified"] = False
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "Connectivity"):
            self.w.validate_input(path, self.w.sha(path))

    def test_coordinate_tamper_rejected(self):
        value = self.input()
        coordinates = Path(value["coordinates_angstrom_npy"]["path"])
        np.save(coordinates, np.full((3, 3), 10.))
        value["coordinates_angstrom_npy"] = self.record(coordinates)
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "coordinates disagree"):
            self.w.validate_input(path, self.w.sha(path))

    def test_checkpoint_without_order_binding_rejected(self):
        value = self.input()
        checkpoint = self.path / "fake.chk"
        checkpoint.write_bytes(b"Not HDF, must never be parsed")
        value["initial_checkpoint"] = self.record(checkpoint)
        path = self.write_input(value)
        with self.assertRaisesRegex(ValueError, "atom-order"):
            self.w.validate_input(path, self.w.sha(path))

    def test_closed_shell_complete_MO_shape_required(self):
        arrays = self.arrays()
        arrays["MO_coeff"] = np.ones((2, 1))
        with self.assertRaisesRegex(ValueError, "square AO/MO"):
            self.w.validate_arrays(arrays, 2, 2, 2)

    def test_fractional_occupations_rejected(self):
        arrays = self.arrays()
        arrays["MO_occ"] = np.array([1., 1.])
        with self.assertRaisesRegex(ValueError, "occupations"):
            self.w.validate_arrays(arrays, 2, 2, 2)

    def test_invalid_screening_table_rejected(self):
        arrays = self.arrays()
        arrays["grid_non0tab"] = np.ones((1, 1))
        with self.assertRaisesRegex(ValueError, "screen table"):
            self.w.validate_arrays(arrays, 2, 2, 2)

    def test_grid_mismatch_rejected(self):
        arrays = self.arrays()
        arrays["grid_weights"] = np.ones(3)
        with self.assertRaisesRegex(ValueError, "quadrature"):
            self.w.validate_arrays(arrays, 2, 2, 2)

    def test_nan_rejected(self):
        arrays = self.arrays()
        arrays["MO_energy"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.w.validate_arrays(arrays, 2, 2, 2)

    def test_snapshot_closed_payload_gate(self):
        directory, manifest, _, _ = self.snapshot()
        actual, _, arrays, binding = self.w.load_snapshot_payload(directory)
        self.assertEqual(actual, manifest)
        self.assertEqual(binding["closed_producer"]["phase"], "fixture")
        np.testing.assert_array_equal(arrays["MO_coeff"], np.eye(2))

    def test_running_producer_rejected(self):
        directory, _, _, _ = self.snapshot(closed=False)
        with self.assertRaisesRegex(ValueError, "not closed cleanly"):
            self.w.load_snapshot_payload(directory)

    def test_each_parent_failure_rejected(self):
        directory, _, row, ledger = self.snapshot()
        original = json.loads(ledger.read_text())
        for field in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed"):
            document = deepcopy(original)
            document["phases"][0][field] = True
            ledger.write_text(json.dumps(document))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "not closed cleanly"):
                self.w.load_snapshot_payload(directory)

    def test_changed_mo_array_rejected(self):
        directory, _, _, _ = self.snapshot()
        np.save(directory / "MO_coeff.npy", np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "byte SHA"):
            self.w.load_snapshot_payload(directory)

    def test_changed_molecule_document_rejected(self):
        directory, _, _, _ = self.snapshot()
        (directory / "molecule.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Molecule document SHA"):
            self.w.load_snapshot_payload(directory)

    def test_changed_parent_log_rejected(self):
        directory, _, _, _ = self.snapshot()
        (self.path / "payload.log").write_text("Changed")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.w.load_snapshot_payload(directory)

    def test_changed_result_energy_rejected(self):
        directory, _, _, _ = self.snapshot()
        result = json.loads((self.path / "result.json").read_text())
        result["energy_hartree"] = 123.
        (self.path / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "matching DONE"):
            self.w.load_snapshot_payload(directory)

    def test_expected_snapshot_hash_rejected(self):
        directory, _, _, _ = self.snapshot()
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            self.w.load_snapshot_payload(directory, expected_manifest_sha256="0"*64)

    def test_restore_can_never_disable_parent_gate(self):
        with self.assertRaisesRegex(ValueError, "always requires"):
            self.w.restore_snapshot(self.path, require_closed_producer=False)

    def test_immutable_json_publication(self):
        path = self.path / "immutable.json"
        self.w.save_json(path, {"first": True}, exclusive=True)
        with self.assertRaises(FileExistsError):
            self.w.save_json(path, {"second": True}, exclusive=True)
        self.assertEqual(json.loads(path.read_text()), {"first": True})

    def test_immutable_array_publication(self):
        path = self.path / "immutable.npy"
        self.w.save_immutable_npy(path, np.array([1., 2.]))
        with self.assertRaises(FileExistsError):
            self.w.save_immutable_npy(path, np.array([7., 8.]))
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), [1., 2.])

    def test_grid_callable_cannot_resolve_arbitrary_module(self):
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            self.w.resolve_callable({"module": "os", "qualname": "system"})

    def test_bounds(self):
        for threads in (0, 5, True):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                self.w.method_settings(threads)
        for steps in (0, 101, False):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.w.method_settings(4, steps)


if __name__ == "__main__":
    unittest.main()
