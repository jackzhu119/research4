"""Parser and orchestration mocks only; no real molecule or quantum execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_pyscf_large_block_engineering_20261031.py"
SPEC = importlib.util.spec_from_file_location("large_block_validation", SCRIPT)
validation = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(validation)


def source_fixture(root):
    # Arbitrary nonlinear points solely for parser tests, never sent to PySCF.
    coords = np.asarray([[i * .7, (i % 2) * .6, (i % 3) * .4] for i in range(9)])
    symbols = ["C", "C", "O"] + ["H"] * 6
    xyz = root / "synthetic.xyz"
    xyz.write_text("9\nSynthetic parser fixture, not a chemical structure\n" +
        "\n".join(f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for s, p in zip(symbols, coords)) + "\n")
    precise = root / "synthetic.npy"; np.save(precise, coords)
    record = {"identity": "validation_ethanol", "validation_only": True, "identity_verified": True,
              "stereochemistry_verified": True, "molecule": "ethanol", "formula": "C2H6O",
              "atom_count": 9, "electron_count": 26, "charge": 0, "multiplicity": 1,
              "composition": {"C": 2, "H": 6, "O": 1}, "smiles": "CCO",
              "inchikey": validation.INCHIKEY, "source_url": "https://example.invalid/unit-test-only",
              "coordinate_units": "angstrom", "input_xyz_sha256": validation.sha(xyz),
              "coordinates_sha256": validation.sha(precise)}
    provenance = root / "synthetic_provenance.json"; provenance.write_text(json.dumps(record))
    return xyz, precise, provenance, record


def comparison_fixture():
    # Pure matrices chosen to test full9x9 block coverage; not a real Hessian.
    raw = np.zeros((9, 9, 3, 3)); raw[0, 8, 0, 0] = raw[8, 0, 0, 0] = .01
    return {"energy_hartree": -1., "gradient": np.zeros((9, 3)), "raw_hessian": raw,
            "analysis_hessian": raw.copy(), "frequencies_cm1": np.linspace(5., 100., 21)}


class LargeBlockEngineeringTests(unittest.TestCase):
    def test_no_default_source_and_explicit_sha_required(self):
        with tempfile.TemporaryDirectory() as temp:
            xyz, precise, provenance, _ = source_fixture(Path(temp))
            symbols, coords, record = validation.verify_source(xyz, provenance, validation.sha(xyz), precise)
            self.assertEqual(len(symbols), 9); self.assertEqual(coords.shape, (9, 3))
            self.assertTrue(record["validation_only"])
            for expected in [None, "wrong"]:
                with self.assertRaises(ValueError): validation.verify_source(xyz, provenance, expected)

    def test_wrong_identity_formula_charge_key_and_validation_scope_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            xyz, _, provenance, record = source_fixture(Path(temp))
            for key, value in [("identity", "spinosin"), ("formula", "C28H32O15"),
                               ("charge", 1), ("electron_count", 25), ("validation_only", False),
                               ("inchikey", "wrong"), ("identity_verified", False)]:
                provenance.write_text(json.dumps({**record, key: value}))
                with self.assertRaises(ValueError): validation.verify_source(xyz, provenance, validation.sha(xyz))

    def test_wrong_precise_geometry_hash_and_coordinate_unit_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            xyz, precise, provenance, record = source_fixture(Path(temp))
            np.save(precise, np.zeros((9, 3)))
            with self.assertRaises(ValueError): validation.verify_source(xyz, provenance, validation.sha(xyz), precise)
            provenance.write_text(json.dumps({**record, "coordinate_units": "bohr"}))
            with self.assertRaises(ValueError): validation.verify_source(xyz, provenance, validation.sha(xyz))

    def test_wrong_xyz_atom_count_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            xyz, _, provenance, record = source_fixture(Path(temp)); xyz.write_text(xyz.read_text().replace("9\n", "8\n", 1))
            record["input_xyz_sha256"] = validation.sha(xyz); provenance.write_text(json.dumps(record))
            with self.assertRaises(ValueError): validation.verify_source(xyz, provenance, validation.sha(xyz))

    def test_actual_energy_and_gradient_differences_trigger_the_fixed_gate(self):
        first, second = comparison_fixture(), comparison_fixture(); second["energy_hartree"] += 2e-8
        self.assertFalse(validation.compare_case(first, second)["checks"]["energy"])
        second = comparison_fixture(); second["gradient"][0, 0] = 2e-8
        self.assertFalse(validation.compare_case(first, second)["checks"]["gradient"])

    def test_all_pairs_and_all21_modes_compared(self):
        first, second = comparison_fixture(), comparison_fixture()
        record = validation.compare_case(first, second)
        self.assertTrue(record["numerical_equivalence_passed"])
        self.assertTrue(record["all_atom_pair_cross_couplings_compared"])
        second["raw_hessian"][0, 8, 0, 0] += 2e-6
        record = validation.compare_case(first, second)
        self.assertFalse(record["numerical_equivalence_passed"])
        self.assertFalse(record["checks"]["full_raw_hessian"])

    def test_negative_modes_do_not_fake_a_positive_minimum_or_prevent_numeric_equivalence(self):
        first, second = comparison_fixture(), comparison_fixture()
        first["frequencies_cm1"][0] = second["frequencies_cm1"][0] = -5.
        record = validation.compare_case(first, second)
        self.assertTrue(record["numerical_equivalence_passed"])
        self.assertFalse(record["candidate_all21_frequencies_positive"])
        self.assertEqual(record["candidate_imaginary_frequency_count"], 1)

    def test_incomplete_nonfinite_and_excess_frequency_difference_fail(self):
        first, second = comparison_fixture(), comparison_fixture(); second["frequencies_cm1"][0] += .02
        self.assertFalse(validation.compare_case(first, second)["numerical_equivalence_passed"])
        for key, value in [("raw_hessian", np.zeros((8, 8, 3, 3))),
                           ("frequencies_cm1", np.zeros(20)), ("gradient", np.full((9, 3), np.nan))]:
            second = comparison_fixture(); second[key] = value
            with self.assertRaises(ValueError): validation.compare_case(first, second)

    def test_same_mf_instance_and_fixed_case_order_without_model_rebuilding(self):
        mf = {"fingerprint": "unchanged"}; seen = []; persisted = []
        def evaluate(actual, label, block):
            seen.append((id(actual), label, block)); return comparison_fixture()
        results = validation.compare_same_state(mf, evaluate, lambda actual: actual["fingerprint"],
            lambda label, actual, record: persisted.append((label, record["numerical_equivalence_passed"])))
        self.assertEqual([label for _, label, _ in seen], ["standard_full", "response_block4", "response_block8"])
        self.assertEqual([block for _, _, block in seen], [None, 4, 8])
        self.assertEqual(len({identifier for identifier, _, _ in seen}), 1)
        self.assertEqual(len(results), 3); self.assertEqual(len(persisted), 3)

    def test_geometry_or_orbital_mutation_refused(self):
        mf = {"fingerprint": "original"}
        def mutate(actual, label, block):
            actual["fingerprint"] = "changed"; return comparison_fixture()
        with self.assertRaises(RuntimeError):
            validation.compare_same_state(mf, mutate, lambda actual: actual["fingerprint"], lambda *args: None)


if __name__ == "__main__":
    unittest.main()
