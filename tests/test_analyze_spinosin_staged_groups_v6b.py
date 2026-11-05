"""Stdlib-only contracts and static C001 metadata; no target modes or imports."""
import ast
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_spinosin_staged_groups_20261105_v6b.py"
SPEC = importlib.util.spec_from_file_location("v6b_group_adapter_pure_tests", SCRIPT)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def checks():
    return {"schema": "research4-V6b-independent-staged-modes-analysis-v1", "status": "DONE_COMPUTED_MODES_AUDITED",
        "case": "Spinosin", "analysis_source_sha256": M.ANALYZER_SHA,
        "actual_V6b_worker_sha256": "f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d",
        "actual_V6b_parent_sha256": "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5",
        "old_a2_V6b_compatibility_claimed": False, "snapshot_manifest_sha256": M.SNAPSHOT_SHA,
        "atoms": 75, "normal_modes": 219, "normal_mode_count_complete": True,
        "global_completion_status": "ACTUAL_FULL_HESSIAN_AND_ALL_SIGNED_MODES_AUDITED",
        "all_frequencies_positive": True, "imaginary_frequency_count": 0, "zero_frequency_count": 0,
        "qualified_for_mechanism": True, "minimum_qualification_separate_from_completion": True,
        "metadata_provenance_gates_before_numpy_or_array_access": True,
        "IR_intensities_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
        "theory_or_peaks_enter_classifier": False, "new_quantum_evaluations": 0,
        "restored_gradient_gate": {"result": {"sha256": M.GRADIENT_SHA}},
        "independent_numpy_replay": {"status": "PASS_INDEPENDENT_NUMPY_TR_REPLAY"}}


class PureContracts(unittest.TestCase):
    def test_cold_import_and_frozen_static_package_are_stdlib_only(self):
        hashes = {}; static = M.static_package(hashes)
        self.assertEqual(len(static["groups"]), 22)
        groups = static["groups"]
        partitions = [g for g in groups if g["family"] == M.PARTITION]
        overlaps = [g for g in groups if g["family"] == M.OVERLAP]
        self.assertEqual((len(partitions), len(overlaps)), (4, 18))
        self.assertEqual(sorted(i for g in partitions for i in g["indices_zero_based"]), list(range(75)))
        self.assertEqual([len(g["indices_zero_based"]) for g in partitions], [32, 20, 1, 22])
        self.assertEqual(static["audit"]["C001_to_current_zero_based_index_map"], list(range(75)))
        self.assertFalse(any(k.split(".")[0] in {"numpy", "scipy", "torch", "pyscf", "rdkit"} for k in sys.modules))
        self.assertFalse(any(Path(p).suffix.lower() in (".npy", ".npz", ".chk") for p in hashes))

    def test_source_v6b_checks_qualify_only_actual_full_positive_original_target(self):
        M.validate_analysis_checks(checks())
        for change in ({"schema": "research4-V5-independent-modes-analysis-v1"}, {"analysis_source_sha256": "0" * 64},
                {"case": "water_software_control"}, {"normal_modes": 218}, {"atoms": 3}, {"qualified_for_mechanism": False},
                {"all_frequencies_positive": False}, {"imaginary_frequency_count": 1}, {"zero_frequency_count": 1},
                {"normal_mode_count_complete": False}, {"IR_intensities_status": "COMPUTED"},
                {"chemical_attribution_verified": True}, {"theory_or_peaks_enter_classifier": True},
                {"snapshot_manifest_sha256": "0" * 64}, {"qualified_for_mechanism": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError): M.validate_analysis_checks({**checks(), **change})
        bad = checks(); bad["restored_gradient_gate"]["result"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): M.validate_analysis_checks(bad)

    def test_negative_zero_nonfinite_and_wrong_unit_modes_block_before_numbers(self):
        value = {"frequencies_cm1": [1.] * 219, "frequencies_thz": [M.CM1_TO_THZ] * 219,
            "imaginary_frequency_count": 0, "all_frequencies_real_positive": True,
            "stationary_minimum_verified": True, "frequency_validation_passed": True}
        self.assertEqual(len(M.validate_positive_frequencies(value)[0]), 219)
        for number in (-1., 0., float("nan"), float("inf"), True):
            bad = deepcopy(value); bad["frequencies_cm1"][0] = number
            with self.subTest(number=number), self.assertRaises(ValueError): M.validate_positive_frequencies(bad)
        bad = deepcopy(value); bad["frequencies_thz"][0] = 1.
        with self.assertRaises(ValueError): M.validate_positive_frequencies(bad)
        bad = deepcopy(value); bad["frequency_validation_passed"] = False
        with self.assertRaises(ValueError): M.validate_positive_frequencies(bad)

    def test_runtime_rejected_without_authentic_metadata_gate(self):
        with self.assertRaises(ValueError): M.numerical_runtime({})
        self.assertNotIn("numpy", sys.modules)

    def test_public_api_always_takes_existing_global_q_lock(self):
        events = []
        class Lock:
            def __enter__(self): events.append("lock")
            def __exit__(self, *exc): events.append("unlock")
        class Analyzer:
            def exclusive_closed_quantum(self): return Lock()
        with patch.object(M, "modes_analyzer", return_value=Analyzer()), \
                patch.object(M, "_analyze_closed", side_effect=lambda *a, **k: events.append("analyze") or {"fixture": True}):
            self.assertEqual(M.analyze("unused", "unused", "unused", "unused", "unused"), {"fixture": True})
        self.assertEqual(events, ["lock", "analyze", "unlock"])

    def test_rejected_qualify_never_calls_numeric_helper_or_creates_result(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "must_not_exist"
            with patch.object(M, "static_package", return_value={}), \
                    patch.object(M, "qualify_analysis", side_effect=ValueError("synthetic unqualified minimum")), \
                    patch.object(M, "numerical_runtime", side_effect=AssertionError("must remain cold")) as numeric:
                with self.assertRaisesRegex(ValueError, "unqualified"):
                    M._analyze_closed("unused", "unused", "unused", "unused", out)
                numeric.assert_not_called(); self.assertFalse(out.exists())

    def test_scalar_m_d_squared_is_mass_weighted_not_unweighted(self):
        mode = [[1., 0., 0.], [1., 0., 0.]]
        groups = [{"indices_zero_based": [0]}, {"indices_zero_based": [1]}, {"indices_zero_based": [0, 1]}]
        result = M.scalar_participation(mode, [1., 3.], groups)
        self.assertEqual(result, [.25, .75, 1.])
        self.assertEqual(sum(result[:2]), 1.)
        self.assertGreater(sum(result), 1.)  # The overlap group is never a disjoint partition.

    def test_scalar_sign_and_global_scale_invariant_but_not_degenerate_basis_invariant(self):
        groups = [{"indices_zero_based": [0]}, {"indices_zero_based": [1]}]
        base = [[1., 0., 0.], [2., 0., 0.]]
        actual = M.scalar_participation(base, [1., 2.], groups)
        for scale in (-4., 1e100, 1e-100):
            scaled = [[v * scale for v in row] for row in base]
            for a, b in zip(actual, M.scalar_participation(scaled, [1., 2.], groups)): self.assertAlmostEqual(a, b)
        basis1 = M.scalar_participation([[1., 0., 0.], [0., 0., 0.]], [1., 1.], groups)
        rotated = M.scalar_participation([[2**-.5, 0., 0.], [2**-.5, 0., 0.]], [1., 1.], groups)
        self.assertNotEqual(basis1, rotated)

    def test_scalar_invalid_masses_indices_and_zero_vectors_rejected(self):
        good = [{"indices_zero_based": [0]}]
        for mode, masses, groups in (([[0., 0., 0.]], [1.], good), ([[1., 0., 0.]], [0.], good),
                ([[float("nan"), 0., 0.]], [1.], good), ([[1., 0., 0.]], [1.], [{"indices_zero_based": [1]}]),
                ([[1., 0., 0.]], [1.], [{"indices_zero_based": [0, 0]}])):
            with self.assertRaises(ValueError): M.scalar_participation(mode, masses, groups)

    def test_no_top_level_numerical_import_or_old_v5_gate_use(self):
        tree = ast.parse(SCRIPT.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import): self.assertFalse(any(v.name.split(".")[0] in {"numpy", "scipy", "torch", "pyscf", "rdkit"} for v in node.names))
            if isinstance(node, ast.ImportFrom): self.assertNotIn((node.module or "").split(".")[0], {"numpy", "scipy", "torch", "pyscf", "rdkit"})
        functions = {n.name: ast.get_source_segment(SCRIPT.read_text(), n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertLess(functions["qualify_analysis"].index("validate_analysis_checks(value)"), functions["qualify_analysis"].index("gate = analyzer.qualify("))
        self.assertLess(functions["_analyze_closed"].index("gate = qualify_analysis("), functions["_analyze_closed"].index("np, a2, helper = numerical_runtime(gate)"))
        self.assertNotIn("qualify_analysis(", functions["preflight"])
        self.assertNotIn("terminal_quantum(", functions["preflight"])
        self.assertIn("helper.participation(modes, masses, groups)", functions["_analyze_closed"])
        self.assertNotIn("old.qualify_analysis(", SCRIPT.read_text())


if __name__ == "__main__": unittest.main()
