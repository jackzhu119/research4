"""Pure stdlib tests for the staged driver; no spectra or numerical imports."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_supplemental_multirepresentation_20261104.py"
SPEC = importlib.util.spec_from_file_location("supplemental_multirepresentation_driver", PATH)
DRIVER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(DRIVER)


class DriverTests(unittest.TestCase):
    def test_plan_stays_cold_and_has_full_stage_order(self):
        result = DRIVER.plan()
        self.assertEqual(result["stages"], ["prepare", "bind", "source", "fit", "predict", "score"])
        self.assertEqual((result["fits_executed"], result["numerical_inputs_read"]), (0, 0))
        self.assertEqual((result["expected_source_alpha_fits"], result["expected_final_fits"]), (27, 36))
        self.assertIsNone(result["current_accuracy"])
        for name in ("numpy", "scipy", "torch", "sklearn", "matplotlib"):
            self.assertNotIn(name, sys.modules)

    def test_top_level_imports_are_only_standard_library(self):
        for node in ast.parse(PATH.read_text()).body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.split(".")[0] in {"numpy", "scipy", "torch", "sklearn", "matplotlib"} for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], {"numpy", "scipy", "torch", "sklearn", "herb_spectra"})

    def authority_fixture(self, directory, running=False):
        root = Path(directory); qroot = root / "quantum"; (qroot / "reports/pilot").mkdir(parents=True)
        (qroot / ".runtime").mkdir(); lock = qroot / ".runtime/pyscf-v4-global.lock"; lock.touch()
        ledger = qroot / "reports/pilot/cpu_budget.json"
        ledger.write_text(json.dumps({"phases": [{"status": "RUNNING" if running else "EXITED", "actual_cpu_seconds": 1.}]}))
        auth = {"schema": "research4-post-quantum-fusion-authority-v1", "root_execution_authorized": True,
                "quantum_terminal_reviewed": True, "implementation_review_passed": True,
                "quantum_worktree_root": str(qroot), "quantum_global_lock": str(lock),
                "quantum_ledger": {"path": str(ledger), "sha256": DRIVER.sha(ledger)},
                "limits": {"cpu_seconds": 3600, "wall_seconds": 14400, "threads": 1, "device": "cpu",
                           "RSS_bytes": int(1.5 * 1024**3), "AS_bytes": 10 * 1024**3}}
        path = root / "authority.json"; path.write_text(json.dumps(auth))
        return path, auth

    def test_terminal_quantum_and_explicit_review_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.authority_fixture(directory, running=True)
            with self.assertRaises(ValueError): DRIVER.root_authority(path, DRIVER.sha(path))
        with tempfile.TemporaryDirectory() as directory:
            path, auth = self.authority_fixture(directory)
            DRIVER.root_authority(path, DRIVER.sha(path))
            auth["implementation_review_passed"] = False; path.write_text(json.dumps(auth))
            with self.assertRaises(ValueError): DRIVER.root_authority(path, DRIVER.sha(path))

    def test_authority_cannot_enlarge_budget_or_use_a_different_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path, auth = self.authority_fixture(directory)
            auth["limits"]["cpu_seconds"] = 3601; path.write_text(json.dumps(auth))
            with self.assertRaises(ValueError): DRIVER.root_authority(path, DRIVER.sha(path))
            auth["limits"]["cpu_seconds"] = 3600
            fake = Path(directory) / "pyscf-v4-global.lock"; fake.touch()
            auth["quantum_global_lock"] = str(fake); path.write_text(json.dumps(auth))
            with self.assertRaises(ValueError): DRIVER.root_authority(path, DRIVER.sha(path))

    def test_unknown_quantum_cpu_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path, auth = self.authority_fixture(directory)
            ledger = Path(auth["quantum_ledger"]["path"])
            ledger.write_text(json.dumps({"phases": [{"status": "PARTIAL", "actual_cpu_seconds": None}]}))
            auth["quantum_ledger"]["sha256"] = DRIVER.sha(ledger); path.write_text(json.dumps(auth))
            with self.assertRaises(ValueError): DRIVER.root_authority(path, DRIVER.sha(path))

    def test_fixed_stage_order_and_partial_stops(self):
        with mock.patch.object(DRIVER, "read", return_value={"phases": []}):
            DRIVER.stage_allowed(Path("NOT_READ"), "prepare")
            with self.assertRaises(ValueError): DRIVER.stage_allowed(Path("NOT_READ"), "source")
        ledger = {"phases": [{"stage": "prepare", "status": "EXITED_HEALTHY"}, {"stage": "bind", "status": "EXITED_HEALTHY"}]}
        with mock.patch.object(DRIVER, "read", return_value=ledger), mock.patch.object(DRIVER, "healthy_stage"):
            DRIVER.stage_allowed(Path("NOT_READ"), "source")
        ledger["phases"][-1]["status"] = "UNKNOWN_CPU_PARTIAL"
        with mock.patch.object(DRIVER, "read", return_value=ledger):
            with self.assertRaises(ValueError): DRIVER.stage_allowed(Path("NOT_READ"), "source")

    def test_failed_closure_cannot_release_next_stage(self):
        receipt = {"stage": "source", "status": "DONE_HEALTHY_CLOSED_STAGE", "exit_code": 0,
                   "stop_reason": "RSS_GUARD", "worker_result": {}}
        ledger = {"phases": [{"stage": "source", "status": "EXITED_HEALTHY", "closed_stage": {}}]}
        out = Path("NOT_READ").resolve()
        with mock.patch.object(DRIVER, "read", side_effect=[ledger, receipt]), \
             mock.patch.object(DRIVER, "checked", return_value=out / "source/closed_stage.json"):
            with self.assertRaises(ValueError): DRIVER.healthy_stage(out, "source")

    def test_changed_source_lock_is_caught_through_closed_receipt(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(DRIVER, "ROOT", Path(directory)):
            out = Path(directory) / "reports/run"; stage = out / "source"; stage.mkdir(parents=True)
            proof = stage / "proof.json"; proof.write_text('{}')
            source_lock = stage / "source_lock.json"; source_lock.write_text('{"epoch":24}')
            result = stage / "worker_result.json"
            result.write_text(json.dumps({"status": "DONE", "stage": "source", "proof_sha256": DRIVER.sha(proof),
                                          "source_lock": DRIVER.ref(source_lock)}))
            row = {"stage": "source", "status": "EXITED_HEALTHY", "actual_child_CPU_seconds": 1.}
            receipt = {"stage": "source", "status": "DONE_HEALTHY_CLOSED_STAGE", "exit_code": 0,
                       "stop_reason": None, "actual_child_CPU_seconds": 1.,
                       "worker_result": DRIVER.ref(result), "supervisor_proof": DRIVER.ref(proof)}
            closed = stage / "closed_stage.json"; closed.write_text(json.dumps(receipt))
            row["closed_stage"] = DRIVER.ref(closed)
            (out / "resource_ledger.json").write_text(json.dumps({"phases": [row]}))
            DRIVER.healthy_stage(out, "source")
            source_lock.write_text('{"epoch":48}')
            with self.assertRaises(ValueError): DRIVER.healthy_stage(out, "source")

    def test_incoming_join_uses_explicit_canonical_parent(self):
        metadata = [{"sample_id": "incoming-a", "canonical_for_model": "True", "legacy_match_ids": '["old-canonical"]'},
                    {"sample_id": "incoming-alias", "canonical_for_model": "False", "legacy_match_ids": '["old-canonical"]'}]
        role_rows = [{"sample_id": r["sample_id"], "species": "huangqi", "canonical_model_record_id": "old-canonical",
                      "entity_id": "incoming-measured-entity"} for r in metadata]
        chosen, aliases = DRIVER.incoming_canonical_map(role_rows, metadata)
        self.assertEqual(chosen[("huangqi", "old-canonical")], "incoming-a")
        self.assertIn("incoming-alias", aliases[("huangqi", "old-canonical")])
        metadata[1]["canonical_for_model"] = "True"
        with self.assertRaises(ValueError): DRIVER.incoming_canonical_map(role_rows, metadata)

    def test_actual_registry_path_can_still_be_v1(self):
        registry = {"datasets": [{"species": "tianma", "modality": "time",
                                 "artifact": "data/supplemental/teacher_20261104_v1/merged/tianma/time.npz", "sha256": "FIXTURE"}]}
        record = DRIVER.raw_entry(registry, "tianma", "time")
        self.assertIn("_v1/", record["path"])
        registry["datasets"].append(deepcopy(registry["datasets"][0]))
        with self.assertRaises(ValueError): DRIVER.raw_entry(registry, "tianma", "time")

    def test_fold_context_has_actual_ids_not_only_role_flags(self):
        result = DRIVER.fit_context("zaoren_legacy7", "source_alpha_cv", "alpha_single", 42, 96, ["train"], ["valid"])
        self.assertEqual(result["source_ids"], ["train"])
        self.assertEqual(result["inference_ids"], ["valid"])
        self.assertNotIn("query_truth", result)

    def test_predictions_align_by_ID_not_position(self):
        predicted = DRIVER.align_predictions(["a", "b"], ["b", "a"], ["C0", "C1"], [[0., 1.], [2., 0.]])
        self.assertEqual(predicted, ["C0", "C1"])
        with self.assertRaises(ValueError): DRIVER.align_predictions(["a", "b"], ["a", "a"], ["C0"], [[1.], [1.]])

    def test_nonfinite_or_missing_predictions_block_score(self):
        with self.assertRaises(ValueError): DRIVER.align_predictions(["a"], ["a"], ["C0"], [[float("nan")]])
        with self.assertRaises(ValueError): DRIVER.align_predictions(["a", "b"], ["a"], ["C0"], [[1.]])

    def test_new3_metrics_allow_wrong_prediction_to_an_old_class(self):
        result, classes, confusion = DRIVER.score_labels(["n1", "n2", "n3"], ["old", "n2", "n3"], ["old", "n1", "n2", "n3"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["metric_classes"], ["n1", "n2", "n3"])
        self.assertEqual(sum(r["count"] for r in confusion), 3)

    def test_corrected_harmed_net_matches_accuracy_change(self):
        truth = ["a", "a", "b", "b"]; baseline = ["b", "a", "b", "a"]; candidate = ["a", "b", "b", "b"]
        result = DRIVER.corrected_harmed(truth, baseline, candidate)
        self.assertEqual((result["corrected"], result["harmed"], result["net_correct"]), (2, 1, 1))
        self.assertEqual(result["accuracy_delta_pp"], 25.)

    def test_incomplete_grid_never_opens_truth(self):
        lock = {"models": {}, "prediction_manifest": {}, "predictions": []}
        receipt = {"verified_outputs": {"prediction_lock": {}, "model_lock": {}, "source_lock": {}}}
        with mock.patch.object(DRIVER, "healthy_stage", return_value=receipt), mock.patch.object(DRIVER, "numeric_dependencies", return_value=None), \
             mock.patch.object(DRIVER, "checked", return_value=Path("NOT_READ")), \
             mock.patch.object(DRIVER, "read", side_effect=[lock, {"models": [], "source_epoch_lock": {}}, {"tasks": []}]), \
             mock.patch.object(DRIVER, "rows", side_effect=AssertionError("truth must remain unopened")):
            with self.assertRaises(ValueError): DRIVER._score(Path("NOT_READ"), {})

    def grid_fixture(self):
        tasks = {task: {"task_id": task, "query_ids": [task + ".q" + str(i) for i in range(DRIVER.leaf().TASKS[task][2])]}
                 for task in DRIVER.TASK_ORDER}
        models = [{"model_key": f"{task}.{arm}.seed{seed}", "task_id": task, "arm": arm, "seed": seed,
                   "model": {"path": "weight:" + f"{task}.{arm}.seed{seed}"},
                   "training": {"path": "training:" + f"{task}.{arm}.seed{seed}"}}
                  for task in DRIVER.TASK_ORDER for arm in DRIVER.leaf().ARMS for seed in DRIVER.leaf().SEEDS]
        predictions = [{k: v for k, v in r.items() if k not in ("model", "training")} for r in models]
        for row in predictions:
            row.update(predictions={"path": "prediction:" + row["model_key"]}, prediction_metadata={"path": "meta"})
        return tasks, models, predictions

    def test_exact_unique_model_and_prediction_grid_is_required(self):
        tasks, models, predictions = self.grid_fixture()
        DRIVER.validate_model_grid(models, predictions, tasks)
        predictions[-1] = deepcopy(predictions[0])
        with self.assertRaises(ValueError): DRIVER.validate_model_grid(models, predictions, tasks)
        with self.assertRaises(ValueError): DRIVER.validate_model_grid(models[:-1], None, tasks)

    def test_changed_class_column_order_blocks_before_truth(self):
        tasks, models, predictions = self.grid_fixture()
        lock = {"models": {"path": "MODEL_LOCK"}, "prediction_manifest": {"path": "QUERY_MANIFEST"}, "predictions": predictions}
        def healthy(out, stage):
            return {"verified_outputs": {"prediction_lock": {"path": "PRED_LOCK"}, "model_lock": {"path": "MODEL_LOCK"},
                                         "source_lock": {"path": "SOURCE_LOCK"}}}
        def read(path):
            name = str(path)
            if name == "PRED_LOCK": return lock
            if name == "MODEL_LOCK": return {"models": models, "source_epoch_lock": {"path": "SOURCE_LOCK"}}
            if name == "QUERY_MANIFEST": return {"tasks": list(tasks.values())}
            if name.startswith("training:"):
                task = name.split(":", 1)[1].split(".")[0]
                return {"class_labels": ["C" + str(i) for i in range(DRIVER.leaf().TASKS[task][3])]}
            raise AssertionError("Unexpected metadata read: " + name)
        class Strings(list):
            def astype(self, value): return self
            def tolist(self): return list(self)
        def load_npz(record, np, fields):
            key = record["path"].split(":", 1)[1]; task = key.split(".")[0]
            classes = ["C" + str(i) for i in range(DRIVER.leaf().TASKS[task][3])]
            return {"record_ids": Strings(tasks[task]["query_ids"]), "class_labels": Strings(list(reversed(classes))),
                    "logits": [[1.] * len(classes) for _ in tasks[task]["query_ids"]]}
        with mock.patch.object(DRIVER, "healthy_stage", side_effect=healthy), \
             mock.patch.object(DRIVER, "numeric_dependencies", return_value=None), \
             mock.patch.object(DRIVER, "checked", side_effect=lambda record: Path(record["path"])), \
             mock.patch.object(DRIVER, "read", side_effect=read), mock.patch.object(DRIVER, "read_npz", side_effect=load_npz), \
             mock.patch.object(DRIVER, "rows", side_effect=AssertionError("truth must remain unopened")):
            with self.assertRaisesRegex(ValueError, "class columns"):
                DRIVER._score(Path("NOT_READ"), {})

    def test_closed_actual_CPU_overrun_cannot_be_DONE(self):
        limits = {"cpu_seconds": 3600, "RSS_bytes": 1000}
        healthy, reason = DRIVER.closed_resource_verdict({"exit_code": 0, "stop_reason": None}, 3600.001,
                                                        limits, 500, deadline_exceeded=False)
        self.assertFalse(healthy); self.assertIn("ACTUAL_CPU", reason)
        healthy, reason = DRIVER.closed_resource_verdict({"exit_code": 0, "stop_reason": None}, 10,
                                                        limits, 1001, deadline_exceeded=False)
        self.assertFalse(healthy); self.assertIn("RSS", reason)


if __name__ == "__main__":
    unittest.main()
