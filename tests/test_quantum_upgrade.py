"""Input/lineage safeguards; parser fixtures are synthetic, not calculations."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("quantum_upgrade", Path(__file__).parents[1]/"scripts/prepare_quantum_upgrade.py")
quantum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantum)


class QuantumUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.xyz = "3\nsynthetic water geometry validator fixture\nO 0 0 0\nH .75 .58 0\nH -.75 .58 0\n"
        self.provenance = {"structure_verified":True,"source_url":"https://example.test/synthetic-fixture",
                           "stereochemistry_status":"verified","geometry_origin":"database_3d",
                           "xyz_sha256":hashlib.sha256(self.xyz.encode()).hexdigest()}

    def test_formula_atom_counts_and_electron_parity(self):
        self.assertEqual(sum(quantum.formula_counts("C28H32O15").values()),75)
        self.assertEqual(sum(quantum.formula_counts("C58H94O26").values()),178)
        self.assertEqual(sum(quantum.formula_counts("C52H84O21").values()),157)
        with self.assertRaises(ValueError): quantum.validated_xyz(self.xyz,"H2O",self.provenance,0,2)

    def test_unverified_or_changed_structure_rejected(self):
        with self.assertRaises(ValueError): quantum.validated_xyz(self.xyz,"H2O",{},0,1)
        with self.assertRaises(ValueError): quantum.validated_xyz(self.xyz+"\n","H2O",self.provenance,0,1)
        with self.assertRaises(ValueError): quantum.validated_xyz(self.xyz,"C28H32O15",self.provenance,0,1)

    def test_generated_input_only_for_verified_exact_composition(self):
        text = quantum.gaussian_input(self.xyz,"H2O",self.provenance,0,1,"synthetic format fixture")
        self.assertIn("Opt=(Tight,CalcFC) Freq",text)
        self.assertTrue(text.endswith("\n\n"))
        with self.assertRaises(ValueError): quantum.gaussian_input(self.xyz,"H2O",self.provenance,0,1,"bad\nroute")

    def test_gaussian_incomplete_and_imaginary_diagnostics(self):
        text = "Optimization completed.\n Frequencies -- 10.0 -20.0 30.0\n IR Inten    -- 1.0 2.0 3.0\n Normal termination of Gaussian 16"
        value = quantum.parse_gaussian_frequency_log(text)
        self.assertEqual(value["imaginary_frequency_count"],1)
        self.assertFalse(value["candidate_ready_for_interpretation"])
        self.assertAlmostEqual(value["frequencies_thz"][0],.299792458)
        self.assertFalse(quantum.parse_gaussian_frequency_log(text.replace("Normal termination of Gaussian 16","Error termination"))["normal_termination"])

    def test_theory_prior_rejects_test_selected_or_unrun_data(self):
        prior = {"locked_protocol_sha256":"p","scope":"training_only","fit_sample_ids":["train"],
                 "calculation_status":"COMPLETED","source_hash":"s"}
        self.assertFalse(quantum.validate_theory_prior(prior,["train"],["test"],"p")["absolute_zero_leakage_proven"])
        for extra in ({"fit_sample_ids":["test"]},{"test_score_selected":True},{"label_conditioned":True},
                      {"calculation_status":"NOT_RUN"},{"locked_protocol_sha256":"other"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                quantum.validate_theory_prior(prior|extra,["train"],["test"],"p")

    def test_frequency_parser_does_not_certify_physics_or_accept_bad_intensity(self):
        text = "Optimization completed.\n Frequencies -- 10 20 30\n IR Inten    -- 1 2 3\n Normal termination of Gaussian 16"
        self.assertTrue(quantum.parse_gaussian_frequency_log(text)["basic_frequency_block_checks_passed"])
        self.assertFalse(quantum.parse_gaussian_frequency_log(text)["candidate_ready_for_interpretation"])
        self.assertFalse(quantum.parse_gaussian_frequency_log(text.replace("1 2 3", "1 NaN 3"))["basic_frequency_block_checks_passed"])
        self.assertFalse(quantum.parse_gaussian_frequency_log(text.replace("1 2 3", "1 2"))["basic_frequency_block_checks_passed"])

    def test_external_prior_must_be_locked_before_evaluation(self):
        prior = {"locked_protocol_sha256":"p","scope":"preregistered_external","fit_sample_ids":[],
                 "calculation_status":"COMPLETED","source_hash":"s","locked_before_outer_evaluation":True}
        self.assertEqual(quantum.validate_theory_prior(prior,[],["test"],"p")["scope"],"preregistered_external")
        with self.assertRaises(ValueError):quantum.validate_theory_prior(prior|{"fit_sample_ids":["train"]},["train"],["test"],"p")


if __name__ == "__main__":
    unittest.main()
