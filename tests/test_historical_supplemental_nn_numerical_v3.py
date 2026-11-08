"""Standard-library-only tests; never execute numerical historical checks."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/historical_supplemental_nn_numerical_20261104_v3.py"
SPEC = importlib.util.spec_from_file_location("historical_numerical_leaf_v3", PATH)
LEAF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEAF)


class Budget:
    def __init__(self):
        self.calls = 0
    def check(self):
        self.calls += 1


def metadata_fixture():
    """Software-only metadata, never scientific predictions or array packs."""
    sources, tasks, records, stored = [], [], [], {}
    locks = {t: {"epoch": 24} for t in LEAF.TASKS if t != "zaoren_expanded10"}
    for task_id, (n, q, _) in LEAF.TASKS.items():
        ids = [f"FIXTURE-{task_id}-{i}" for i in range(n)]
        sources.append({"task_id": task_id, "source_ids": ids})
        tasks.append({"task_id": task_id, "query_ids": [f"FIXTURE-QUERY-{task_id}-{i}" for i in range(q)]})
        for arm in LEAF.ARMS:
            for seed in LEAF.SEEDS:
                key = f"{task_id}.{arm}.seed{seed}"
                record = {"model_key": key, "task_id": task_id, "arm": arm, "seed": seed,
                    "epochs": 24, "model": {"path": key + ".pt"}, "training": {"path": key + ".json"}}
                records.append(record)
                stored[key + ".json"] = {"task_id": task_id, "arm": arm, "seed": seed, "epochs": 24,
                    "source_ids": ids.copy(), "validation_ids": [], "query_features_received": False,
                    "query_labels_received": False, "source_batch_order_sha256": "FIXTURE_ORDER",
                    "initial_state_sha256": "FIXTURE_INIT", "initial_encoder_sha256": ["FIXTURE_ENCODER"] * 3,
                    "parameter_count": 123, "source_class_weights": [1.],
                    "branch_spec": {"branch_points": [128] if arm == "alpha_single" else [128, 128, 1024]}}
    model_ref = {"path": "FIXTURE_MODEL_LOCK", "sha256": "a" * 64}
    source_ref = {"path": "FIXTURE_SOURCE_LOCK", "sha256": "b" * 64}
    context = {"stages": {"fit": {"model_lock": model_ref}, "source": {"source_lock": source_ref}},
        "source_lock": {"locks": locks}, "models_lock": {"models": records, "source_epoch_lock": source_ref},
        "prediction_lock": {"models": model_ref, "predictions": deepcopy(records)},
        "source_manifest": {"tasks": sources}, "prediction_manifest": {"tasks": tasks}}
    def grid(models, predictions):
        expected = {f"{t}.{a}.seed{s}" for t in LEAF.TASKS for a in LEAF.ARMS for s in LEAF.SEEDS}
        if any(len(v) != 36 or {r["model_key"] for r in v} != expected for v in [models, predictions]):
            raise ValueError("Incomplete fixture grid")
    old = SimpleNamespace(validate_grid=grid, reference=lambda r: r["path"], read=lambda p: stored[p])
    return context, old, stored


class HistoricalLeafTests(unittest.TestCase):
    def test_import_keeps_numerical_modules_cold(self):
        self.assertFalse(set(sys.modules) & {"numpy", "torch", "pyscf", "scipy", "sklearn"})
        parsed = ast.parse(PATH.read_text())
        owners = {}
        for node in parsed.body:
            if isinstance(node, ast.Import):
                self.assertFalse({n.name.split(".")[0] for n in node.names}
                                 & {"numpy", "torch", "pyscf", "scipy", "sklearn"})
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Import):
                        for name in child.names:
                            if name.name in ("numpy", "torch", "pyscf"):
                                owners.setdefault(name.name, []).append(node.name)
        self.assertEqual(owners, {"numpy": ["run_checks"]})

    def test_no_old_top_level_authority_budget_claim_or_training_calls(self):
        parsed = ast.parse(PATH.read_text())
        calls = [n.func.attr for n in ast.walk(parsed)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(set(calls) & {"audit", "terminal_authority", "AuditBudget", "claim_scope", "fit", "backward", "step"})
        self.assertIn("source_epoch_audit", calls)
        self.assertIn("replay_saved_models", calls)

    def test_missing_parent_proof_blocks_before_any_array_or_input_read(self):
        with mock.patch.object(LEAF, "metadata_checks", side_effect=AssertionError("metadata must remain cold")):
            with self.assertRaisesRegex(ValueError, "Historical training"):
                LEAF.run_checks(Path("NEVER_READ_OLD_RUN"), Path("NEVER_CREATED_OUTPUT"),
                                SimpleNamespace(), {}, Budget(), False)
        self.assertNotIn("numpy", sys.modules)

    def test_grant_must_bind_exact_leaf_before_old_helper_read(self):
        context = {"historical_training_valid": True, "current_Q_prefix_verified": True,
                   "current_grant_verified": True, "current_grant_sha256": "a" * 64,
                   "leaf_code_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "bind this exact numerical leaf"):
            LEAF.validate_entry(Path("NEVER_READ"), Path("NEVER_CREATED"), SimpleNamespace(), context, Budget())

    def test_metadata_grid_and_capacity_pairing(self):
        context, old, _ = metadata_fixture(); budget = Budget()
        sources, tasks, training, pairing = LEAF.metadata_checks(old, context, budget)
        self.assertEqual((len(sources), len(tasks), len(training), len(pairing), budget.calls), (4, 4, 36, 12, 36))

    def test_missing_seed_or_changed_source_ID_blocks(self):
        context, old, stored = metadata_fixture()
        context["prediction_lock"]["predictions"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete fixture grid"):
            LEAF.metadata_checks(old, context, Budget())
        context, old, stored = metadata_fixture()
        stored["zaoren_legacy7.alpha_single.seed42.json"]["source_ids"][0] = "WRONG_SOURCE"
        with self.assertRaisesRegex(ValueError, "source IDs changed"):
            LEAF.metadata_checks(old, context, Budget())

    def test_changed_C_M_initialization_blocks(self):
        context, old, stored = metadata_fixture()
        stored["huangqi_legacy4.alpha_n_time.seed44.json"]["initial_state_sha256"] = "DIFFERENT"
        with self.assertRaisesRegex(ValueError, "C/M pairing differs: initial_state"):
            LEAF.metadata_checks(old, context, Budget())

    def test_query_access_or_unlocked_epoch_blocks(self):
        for field, value in (("query_features_received", True), ("epochs", 48)):
            context, old, stored = metadata_fixture()
            stored["tianma_legacy4.alpha_single.seed42.json"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                LEAF.metadata_checks(old, context, Budget())

    def test_complete_prediction_gate_precedes_truth_and_replay_is_optional(self):
        source = PATH.read_text()
        self.assertLess(source.index('predictions[item["model_key"]] ='), source.index('if replay_models:'))
        self.assertLess(source.index('if replay_models:'), source.index('old.csv_rows(old.reference(task["truth"]))'))
        self.assertIn('"status": "NOT_RUN_HASH_VERIFICATION_ONLY"', source)
        self.assertIn('directory = output / "historical_numerical"', source)
        self.assertIn('replay_dir = directory / "historical_native_replay"', source)


if __name__ == "__main__":
    unittest.main()
