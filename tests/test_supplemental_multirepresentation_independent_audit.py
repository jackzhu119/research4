"""Pure-stdlib audit preparation tests; no NN run, arrays or numerical imports."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_supplemental_multirepresentation_20261104.py"
SPEC = importlib.util.spec_from_file_location("independent_fusion_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)


class IndependentAuditTests(unittest.TestCase):
    def test_plan_is_not_run_and_keeps_numeric_imports_cold(self):
        value = AUDIT.plan()
        self.assertEqual(value["status"], "NOT_RUN_AUDIT_PREPARED_ONLY")
        self.assertEqual(value["actual_predictions_or_arrays_read"], 0)
        self.assertEqual(value["actual_metrics_recomputed"], 0)
        self.assertEqual(value["expected_after_healthy_run"]["query_prediction_artifacts"], 36)
        self.assertFalse(set(AUDIT.FORBIDDEN_NUMERICAL_MODULES) & set(sys.modules))

    def test_numerical_import_and_optional_replay_are_deferred(self):
        parsed = ast.parse(PATH.read_text())
        for node in parsed.body:
            if isinstance(node, ast.Import):
                self.assertFalse({n.name.split(".")[0] for n in node.names} & set(AUDIT.FORBIDDEN_NUMERICAL_MODULES))
        torch_owners = []
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef):
                imports = {n.name.split(".")[0] for child in ast.walk(node) if isinstance(child, ast.Import) for n in child.names}
                if "torch" in imports:
                    torch_owners.append(node.name)
                self.assertNotIn("pyscf", imports)
        self.assertEqual(torch_owners, ["replay_saved_models"])
        replay = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "replay_saved_models")
        calls = [n.func.attr for n in ast.walk(replay) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(set(calls) & {"fit", "backward", "step"})
        self.assertIn("no_grad", calls)

    def test_complete_three_seed_grid_required_before_truth(self):
        values = [{"task_id": task, "arm": arm, "seed": seed, "model_key": f"{task}.{arm}.seed{seed}"}
                  for task in AUDIT.TASKS for arm in AUDIT.ARMS for seed in AUDIT.SEEDS]
        AUDIT.validate_grid(values, values)
        with self.assertRaisesRegex(ValueError, "Incomplete/duplicate"):
            AUDIT.validate_grid(values, values[:-1])
        duplicate = values[:-1] + [values[0]]
        with self.assertRaisesRegex(ValueError, "Incomplete/duplicate"):
            AUDIT.validate_grid(values, duplicate)

    def test_explicit_id_alignment_rejects_repeated_rows(self):
        self.assertEqual(AUDIT.align(["b", "a"], ["a", "b"], ["B", "A"]), ["A", "B"])
        with self.assertRaisesRegex(ValueError, "Duplicate/missing"):
            AUDIT.align(["a", "b", "a"], ["a", "b"], ["A", "B", "A"])

    def test_subset_macro_preserves_outside_class_errors_in_confusion(self):
        value = AUDIT.metrics(["A", "B"], ["A", "C"], ["A", "B", "C"])
        self.assertEqual(value["accuracy_percent"], 50.)
        self.assertEqual(value["macro_f1"], .5)
        self.assertEqual(value["balanced_accuracy_percent"], 50.)
        self.assertEqual(sum(r["count"] for r in value["confusion"]), 2)
        self.assertEqual(next(r["count"] for r in value["confusion"] if r["true_class"] == "B" and r["predicted_class"] == "C"), 1)

    def test_paired_corrected_and_harmed_are_both_reported(self):
        value = AUDIT.pair(["A", "B", "C"], ["B", "B", "A"], ["A", "C", "C"])
        self.assertEqual((value["corrected"], value["harmed"], value["net_correct"]), (2, 1, 1))
        self.assertAlmostEqual(value["accuracy_delta_pp"], 100 / 3)

    def test_old171_and_new3_require_exact_id_membership_not_just_counts(self):
        old = [f"old-{i}" for i in range(171)]; new = ["new-1", "new-2", "new-3"]
        AUDIT.validate_expanded_strata(old + new, old, old, new)
        swapped_old = old[:-1] + [new[0]]; swapped_new = [old[-1]] + new[1:]
        with self.assertRaisesRegex(ValueError, "identities differ"):
            AUDIT.validate_expanded_strata(old + new, old, swapped_old, swapped_new)

    def test_no_audit_budget_or_weight_execution_in_plan(self):
        with mock.patch.object(AUDIT, "AuditBudget", side_effect=AssertionError("actual audit only")), \
             mock.patch.object(AUDIT, "replay_saved_models", side_effect=AssertionError("no replay in plan")):
            value = AUDIT.plan()
        self.assertEqual(value["optional_native_replay"]["status"], "NOT_RUN")
        self.assertEqual(value["optional_native_replay"]["actual_models_replayed"], 0)


if __name__ == "__main__":
    unittest.main()
