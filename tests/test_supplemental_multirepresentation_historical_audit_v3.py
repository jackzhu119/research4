"""Cold metadata-only historical tests; no live Q, arrays or numerical imports."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_supplemental_multirepresentation_historical_20261104_v3.py"
SPEC = importlib.util.spec_from_file_location("historical_nn_metadata_v3", PATH)
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


class HistoricalMetadataTests(unittest.TestCase):
    def archive(self):
        phases = [{"phase": f"archived-{i}", "status": "PARTIAL_OR_FAILED" if i % 11 == 0 else "EXITED",
                   "actual_cpu_seconds": float(i + 1), "ended_at_utc": "2000-01-01T00:00:00+00:00"} for i in range(55)]
        consumed = sum(p["actual_cpu_seconds"] for p in phases)
        return {"schema": "fixture", "quantum_cpu_cap_seconds": 216000,
                "source_checkpoint_sha256": "immutable-source", "continuation_source": {"original_phase_count": 49},
                "consumed_cpu_seconds": consumed, "remaining_quantum_cpu_seconds": 216000 - consumed,
                "phases": phases}

    def append(self, archive, **overrides):
        current = deepcopy(archive)
        new = {"phase": "legitimate-new-failure", "status": "PARTIAL_OR_FAILED", "actual_cpu_seconds": 3.5,
               "ended_at_utc": "2000-01-01T01:00:00+00:00", **overrides}
        current["phases"].append(new)
        if type(new["actual_cpu_seconds"]) in (int, float):
            current["consumed_cpu_seconds"] += new["actual_cpu_seconds"]
            current["remaining_quantum_cpu_seconds"] -= new["actual_cpu_seconds"]
        return current

    def test_cold_plan_does_not_read_live_quantum_or_numerical_data(self):
        value = MOD.plan()
        self.assertEqual(value["status"], "NOT_RUN_HISTORICAL_UTILITY_PREPARED_ONLY")
        self.assertEqual(value["actual_arrays_or_current_Q_read"], 0)
        self.assertFalse(value["old_training_command_available"])
        self.assertFalse(MOD.NUMERIC & set(sys.modules))

    def test_legitimate_append_preserves_all55_rows_and_failed_costs(self):
        archive = self.archive()
        result = MOD.compare_append_only(archive, self.append(archive))
        self.assertEqual(result["archived_phase_count"], 55)
        self.assertEqual(result["current_phase_count"], 56)
        self.assertTrue(result["failed_costs_retained"])
        self.assertFalse(result["Q_success_not_required_or_inferred"] is False)

    def test_rollback_relabel_cap_or_lineage_change_is_rejected(self):
        archive = self.archive()
        bad = []
        current = self.append(archive); current["phases"][0]["status"] = "EXITED"; bad.append(current)
        current = self.append(archive); current["quantum_cpu_cap_seconds"] += 1; bad.append(current)
        current = self.append(archive); current["continuation_source"]["original_phase_count"] = 1; bad.append(current)
        current = deepcopy(archive); current["phases"].pop(); bad.append(current)
        current = self.append(archive); current["hidden_rollback_allowed"] = True; bad.append(current)
        for candidate in bad:
            with self.subTest(candidate=repr(candidate)[:80]), self.assertRaises(ValueError):
                MOD.compare_append_only(archive, candidate)

    def test_running_unknown_cpu_and_unclosed_process_are_rejected(self):
        archive = self.archive()
        for candidate in (self.append(archive, status="RUNNING", actual_cpu_seconds=None),
                          self.append(archive, actual_cpu_seconds=None),
                          self.append(archive, residual_processes_require_review_after_main_payload_exit=True)):
            with self.assertRaises(ValueError):
                MOD.compare_append_only(archive, candidate)

    def test_recorded_historical_deadline_can_be_past_today(self):
        protocol = {"registered_at_utc": "2000-01-01T00:00:01+00:00"}
        ledger = {"created_at_utc": "2000-01-01T00:00:00+00:00", "continuation_registered_at_utc": "2000-01-01T00:00:02+00:00",
                  "deadline_at_utc": "2000-01-01T01:00:00+00:00"}
        records = []
        for i, stage in enumerate(MOD.STAGES):
            minute = i + 1
            records.append({"stage": stage, "started_at_utc": f"2000-01-01T00:{minute:02}:00+00:00",
                "completed_at_utc": f"2000-01-01T00:{minute:02}:01+00:00", "ended_at_utc": f"2000-01-01T00:{minute:02}:02+00:00",
                "closure_at_utc": f"2000-01-01T00:{minute:02}:03+00:00", "proof_deadline_at_utc": ledger["deadline_at_utc"]})
        result = MOD.historical_time_checks(protocol, ledger, records)
        self.assertFalse(result["current_clock_used_for_old_training_validity"])
        records[-1]["closure_at_utc"] = "2000-01-01T01:00:01+00:00"
        with self.assertRaisesRegex(ValueError, "original deadline"):
            MOD.historical_time_checks(protocol, ledger, records)

    def test_past_proof_deadline_cannot_be_rewritten_or_timezone_omitted(self):
        with self.assertRaises(ValueError):
            MOD.aware("2000-01-01T00:00:00")
        a = self.archive(); current = self.append(a)
        current["phases"][0]["actual_cpu_seconds"] = 1  # Canonical original float1.0 must stay byte-equivalent.
        with self.assertRaisesRegex(ValueError, "prefix"):
            MOD.compare_append_only(a, current)

    def test_non_authorizing_template_cannot_start_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            grant = Path(directory) / "grant.json"
            grant.write_text(json.dumps({"schema": "research4-historical-NN-audit-grant-v3",
                "root_execution_authorized": False, "historical_verification_only": True, "training_authorized": False}))
            with self.assertRaisesRegex(ValueError, "fresh no-training audit grant"):
                MOD.validate_current_grant(grant, MOD.sha(grant), ROOT / "reports/not-read",
                                           ROOT / "reports/not-created", True)

    def test_all_historical_graph_work_is_inside_entry_anchored_fresh_budget(self):
        tree = ast.parse(PATH.read_text())
        execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        self.assertEqual(execute.body[0].targets[0].id, "started")
        budget = next(n for n in execute.body if isinstance(n, ast.With))
        call = budget.items[0].context_expr
        self.assertEqual(call.func.id, "FreshAuditBudget")
        self.assertEqual(next(k.value.id for k in call.keywords if k.arg == "started_at"), "started")
        charged_names = {n.func.id for n in ast.walk(budget) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertTrue({"verify_grant_anchors", "load_frozen", "verify_nn_history"} <= charged_names)
        preceding = execute.body[:execute.body.index(budget)]
        uncharged_names = {n.func.id for item in preceding for n in ast.walk(item)
                           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertFalse({"verify_grant_anchors", "load_frozen", "verify_nn_history"} & uncharged_names)

    def test_old_supervisor_deadline_is_an_exact_immutable_record(self):
        protocol = {"registered_at_utc": "2000-01-01T00:00:01+00:00"}
        ledger = {"created_at_utc": "2000-01-01T00:00:00+00:00", "continuation_registered_at_utc": "2000-01-01T00:00:02+00:00",
                  "deadline_at_utc": "2000-01-01T01:00:00+00:00"}
        records = [{"stage": stage, "started_at_utc": f"2000-01-01T00:{i+1:02}:00+00:00",
                    "completed_at_utc": f"2000-01-01T00:{i+1:02}:01+00:00", "ended_at_utc": f"2000-01-01T00:{i+1:02}:02+00:00",
                    "closure_at_utc": f"2000-01-01T00:{i+1:02}:03+00:00", "proof_deadline_at_utc": ledger["deadline_at_utc"]}
                   for i, stage in enumerate(MOD.STAGES)]
        records[2]["proof_deadline_at_utc"] = "2000-01-01T02:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "proof deadline"):
            MOD.historical_time_checks(protocol, ledger, records)

    def test_no_numerical_import_or_training_entry_is_exposed(self):
        tree = ast.parse(PATH.read_text())
        imports = set()
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(n.name.split(".")[0] for n in node.names)
            if isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        self.assertFalse(imports & MOD.NUMERIC)
        self.assertFalse(set(calls) & {"fit", "fit_source_leaf", "backward", "Popen", "audit", "terminal_authority"})
        self.assertIn("validate_ledger", calls)


if __name__ == "__main__":
    unittest.main()
