"""Pure-stdlib audit preparation tests; no NN run, arrays or numerical imports."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_supplemental_multirepresentation_20261104_v2.py"
SPEC = importlib.util.spec_from_file_location("independent_fusion_audit_v2", PATH)
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

    def carry_fixture(self, root):
        prior = Path(root) / "reports/prior"; prior.mkdir(parents=True)
        original = Path(root) / "reports/original_authority.json"
        original.write_text(json.dumps({"limits": {"cpu_seconds": 3600}}))
        protocol = prior / "protocol.json"; protocol.write_text(json.dumps({"executed_model_fits": 0}))
        def ref(path):
            return {"path": str(path.relative_to(root)), "sha256": AUDIT.sha(path), "bytes": path.stat().st_size}
        registration = prior / "registration.json"
        registration.write_text(json.dumps({"protocol": ref(protocol), "authority_path": str(original.resolve()), "authority_sha256": AUDIT.sha(original)}))
        ledger = prior / "resource_ledger.json"
        ledger.write_text(json.dumps({"status": "FAILED_CLOSED", "created_at_utc": "2026-11-04T19:38:58.123369+00:00",
            "deadline_at_utc": "2026-11-04T23:38:58.123369+00:00", "limits": {"cpu_seconds": 3600},
            "phases": [{"stage": "prepare", "status": "FAILED_CLOSED", "actual_child_CPU_seconds": 2.,
                        "supervisor_CPU_seconds_observed_through_phase_close": .25}],
            "closed_child_CPU_seconds": 2., "closed_supervisor_CPU_seconds_observed": .25}))
        diagnostic = Path(root) / "reports/diagnostic.json"
        diagnostic.write_text(json.dumps({"status": "DONE_AXIS_AUDIT_NO_FITS", "exit_code": 0, "wall_terminated": False,
            "model_fits": 0, "numerical_packages_imported_by_supervisor": False,
            "actual_wait4_child_CPU_seconds": .09, "observed_supervisor_process_CPU_seconds": .001,
            "CPU_charge_seconds": .091}))
        charged = 2. + .25 + .09 + .001 + 10
        carry = {"schema": "research4-fusion-bounded-continuation-v2", "prior_run": ref(protocol), "ledger_ref": ref(ledger),
            "registration_ref": ref(registration), "original_authority_ref": ref(original), "diagnostic_ref": ref(diagnostic),
            "prior_actual_CPU": 2.25, "original_cpu_cap": 3600,
            "original_deadline_at_utc": "2026-11-04T23:38:58.123369+00:00", "diagnostic_exact_wait4_CPU": .09,
            "diagnostic_supervisor_CPU_observed": .001, "first_unmeasured_diagnostic_conservative_charge_CPU": 10,
            "charged_prior": charged, "remaining_authorized_CPU": int(3600 - charged)}
        return {"limits": {"cpu_seconds": 3400}, "continuation_accounting": carry}

    def test_prior_costs_and_unmeasured_reserve_are_distinct_and_cannot_be_dropped(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(AUDIT, "ROOT", Path(directory)):
            authority = self.carry_fixture(Path(directory))
            result = AUDIT.prior_continuation(authority)
            self.assertEqual(result["prior_actual_CPU"], 2.25)
            self.assertAlmostEqual(result["diagnostic_actual_CPU"], .091)
            self.assertEqual(result["conservative_unmeasured_charge_not_actual_CPU"], 10)
            omitted = deepcopy(authority); omitted["continuation_accounting"]["first_unmeasured_diagnostic_conservative_charge_CPU"] = 0
            with self.assertRaisesRegex(ValueError, "reserve was dropped"):
                AUDIT.prior_continuation(omitted)

    def test_cumulative_scope_rejects_deadline_reset(self):
        carried = {"charged_prior_CPU": 12.349431818, "original_created_at_utc": "start",
                   "original_deadline_at_utc": "original_deadline", "carry": {"reserve": 10}}
        authority = {"limits": {"cpu_seconds": 3400}}
        ledger = {"limits": authority["limits"], "created_at_utc": "start", "deadline_at_utc": "original_deadline",
            "continuation_accounting": {"charged_prior_CPU_seconds": 12.349431818, "carry": {"reserve": 10}},
            "closed_child_CPU_seconds": 100., "closed_supervisor_CPU_seconds_observed": 1.,
            "phases": [{"status": "EXITED_HEALTHY", "actual_child_CPU_seconds": 100., "supervisor_CPU_seconds_observed_through_phase_close": 1.}]}
        with mock.patch.object(AUDIT, "prior_continuation", return_value=carried):
            actual = AUDIT.cumulative_scope(authority, ledger)
            self.assertAlmostEqual(actual["charged_before_audit_CPU"], 113.349431818)
            ledger["deadline_at_utc"] = "fresh_deadline"
            with self.assertRaisesRegex(ValueError, "reset original wall"):
                AUDIT.cumulative_scope(authority, ledger)

    def test_new_actual_CPU_grand_totals_cannot_hide_phase_costs(self):
        ledger = {"closed_child_CPU_seconds": 10., "closed_supervisor_CPU_seconds_observed": 1.,
                  "phases": [{"status": "EXITED_HEALTHY", "actual_child_CPU_seconds": 10., "supervisor_CPU_seconds_observed_through_phase_close": 1.}]}
        self.assertEqual(AUDIT.verified_new_actual_cpu(ledger), 11.)
        ledger["closed_child_CPU_seconds"] = 0.
        with self.assertRaisesRegex(ValueError, "phase sum"):
            AUDIT.verified_new_actual_cpu(ledger)
        ledger["closed_child_CPU_seconds"] = 10.
        ledger["phases"][0]["actual_child_CPU_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            AUDIT.verified_new_actual_cpu(ledger)

    def test_audit_cpu_fits_original_scope_and_cannot_extend_deadline(self):
        future = (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat()
        scope = {"original_deadline_at_utc": future, "charged_before_audit_CPU": 3412.349431818}
        budget = AUDIT.AuditBudget(Path("NOT_WRITTEN"), 180, 600, scope=scope)
        self.assertLessEqual(budget.wall_seconds, 300)
        scope["charged_before_audit_CPU"] = 3500
        with self.assertRaisesRegex(ValueError, "original3600 CPU"):
            AUDIT.AuditBudget(Path("NOT_WRITTEN"), 180, 600, scope=scope)
        scope["charged_before_audit_CPU"] = 100
        scope["original_deadline_at_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "original3600 CPU"):
            AUDIT.AuditBudget(Path("NOT_WRITTEN"), 180, 600, scope=scope)

    def test_time_repair_rejects_wrong_axis_pin_and_endpoint_step(self):
        audit = {"native_axis_sha256_raw_float64": "bad", "uniform_axis_sha256_raw_float64": AUDIT.UNIFORM_TIME_AXIS_SHA}
        metadata = {"resampling": {**AUDIT.TIME_RECIPE, "axis_audit": audit}, "native_axis_fingerprint": "bad", "target_axis": []}
        with self.assertRaisesRegex(ValueError, "registered acquisition"):
            AUDIT.validate_time_repair(metadata)
        audit.update(native_axis_sha256_raw_float64=AUDIT.NATIVE_TIME_AXIS_SHA, no_extrapolation=True,
            coordinates_only_not_selected_from_labels_or_values=True, synthetic_signal_control_not_real_specimens=False,
            time_unit="NOT_EXPLICITLY_DECLARED", native_start=0., native_end=6749., uniform_step=2.,
            delta_min=1., delta_max=1., delta_mean=1., delta_median=1., delta_population_SD=0.,
            max_native_uniform_coordinate_deviation=0., max_coordinate_deviation_in_steps=0.,
            max_native_uniform_step_deviation=0., max_relative_step_deviation=0.)
        metadata["native_axis_fingerprint"] = AUDIT.NATIVE_TIME_AXIS_SHA
        with self.assertRaisesRegex(ValueError, "endpoints/step"):
            AUDIT.validate_time_repair(metadata)


if __name__ == "__main__":
    unittest.main()
