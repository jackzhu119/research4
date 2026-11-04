"""End-to-end isolation tests with tiny synthetic packs, never real spectra."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

from herb_spectra.leakage_guard import LeakageError, array_fingerprint
from herb_spectra.supplemental_benchmark import file_sha256, write_new_json
from herb_spectra.supplemental_execution import Budget, BudgetExceeded, fit_logged, preflight, run_source, run_predict, run_score, source_calibration


class SupplementalExecutionTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / "reports").mkdir()
        data = root / "data"
        data.mkdir()
        rng = np.random.default_rng(20261104)
        axis = np.linspace(.1, 1.5, 18)
        refs, packs = {}, {}
        for role, count in (("source", 18), ("support", 3), ("query", 3)):
            pack = {"X": rng.normal(size=(count, len(axis))), "axis": axis,
                    "record_ids": np.array([f"{role}-{i}" for i in range(count)]),
                    "entity_ids": np.array([f"entity-{role}-{i}" for i in range(count)]),
                    "legacy_match_ids": np.array(["[]"] * count),
                    "batch_ids": np.array(["SYNTHETIC_BATCH"] * count)}
            labels = np.array((["A", "B", "C"] * count)[:count])
            pack["X"][labels == "B", 3] += 3
            pack["X"][labels == "C", 8] -= 3
            if role != "query":
                pack["y"] = labels
            path = data / (role + ".npz")
            np.savez_compressed(path, **pack)
            refs[role] = {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
            packs[role] = pack
        truth_path = data / "query_truth.csv"
        with truth_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record_id", "y", "entity_id"])
            writer.writeheader()
            writer.writerows(dict(record_id=i, y=c, entity_id=e) for i, c, e in
                             zip(packs["query"]["record_ids"], ["A", "B", "C"], packs["query"]["entity_ids"]))
        anchor = data / "code_anchor.py"
        anchor.write_text("# synthetic test inventory anchor\n", encoding="utf-8")
        ready = data / "intake_ready.json"
        ready.write_text('{"status":"SYNTHETIC_READY"}\n', encoding="utf-8")
        folds = []
        for train, valid in StratifiedKFold(3, shuffle=True, random_state=7).split(packs["source"]["X"], packs["source"]["y"]):
            folds.append({"train_ids": packs["source"]["record_ids"][train].tolist(),
                          "valid_ids": packs["source"]["record_ids"][valid].tolist()})
        task = {"id": "SYNTHETIC_THREE_CLASS", "classes": ["A", "B", "C"], "quantity": "synthetic_alpha",
                **{role + "_pack": ref for role, ref in refs.items()}, "inner_folds": folds,
                "axis_sha256": array_fingerprint(axis),
                "truth": {"path": str(truth_path.relative_to(root)), "sha256": file_sha256(truth_path)},
                "source_calibration_episodes": [], "calibration": {"conservative_policy": "global_no_op", "reason": "SYNTHETIC_FIXED_CONTROL"},
                "calibration_draws": [{"draw_id": 0, "shots": {"1": packs["support"]["record_ids"].tolist()}}],
                "strata": {"first_class": [str(packs["query"]["record_ids"][0])]}}
        protocol = {"schema": "SYNTHETIC_PROTOCOL", "dataset_version": "SYNTHETIC_ONLY", "seeds": [42, 43, 44],
            "resources": {"cpu_seconds": 6000, "wall_seconds": 14400, "rss_bytes": 1610612736},
            "calibration": {}, "code_files": {str(anchor.relative_to(root)): file_sha256(anchor)},
            "intake_manifest": {"path": str(ready.relative_to(root)), "sha256": file_sha256(ready)},
            "candidates": {
                "mcuve_svm": [{"id": "mc", "family": "mcuve_svm", "preprocessing": "snv", "mc": {"runs": 2, "components": 1, "ratio": .2}, "C": 10., "gamma": "scale", "class_weight": "balanced"}],
                "svm": [{"id": "sv", "family": "svm", "preprocessing": "snv", "mc": None, "C": 1., "gamma": "scale"}],
                "shrinkage_lda": [{"id": "ld", "family": "shrinkage_lda", "preprocessing": "snv", "mc": None, "shrinkage": "auto"}],
                "mlp": [{"id": "ml", "family": "mlp", "preprocessing": "snv", "mc": None, "hidden": [4], "alpha": .001, "max_iter": 2}]},
            "tasks": [task]}
        protocol_path = data / "protocol.json"
        write_new_json(protocol_path, protocol)
        release = data / "release_source.json"
        write_new_json(release, {"protocol_sha256": file_sha256(protocol_path), "allowed_stages": ["source"], "data_ready": True})
        return protocol_path.relative_to(root), release.relative_to(root), truth_path

    def test_missing_release_blocks_before_any_real_fit_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol, _, _ = self.fixture(folder)
            with self.assertRaisesRegex(LeakageError, "release file"):
                run_source(folder, protocol, "reports/blocked", None)
            self.assertFalse((Path(folder) / "reports/blocked").exists())

    def test_source_lock_then_query_release_then_separate_truth_join(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, release, truth_path = self.fixture(root)
            output = "reports/SYNTHETIC_RUN"
            original_open = Path.open

            def forbid_query_truth(path, *args, **kwargs):
                if path.resolve() == truth_path.resolve():
                    raise AssertionError("Source/prediction stage attempted to open query truth")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", forbid_query_truth):
                audit = preflight(root, protocol)
                self.assertEqual(audit["status"], "PREFLIGHT_ONLY_NO_MODEL_FIT")
                source_result = run_source(root, protocol, output, release)
                self.assertEqual(source_result["status"], "DONE_SOURCE_LOCKED_NO_QUERY_PREDICTIONS")
                with self.assertRaisesRegex(LeakageError, "does not unlock"):
                    run_predict(root, protocol, output, release)
                query_release = root / "data/release_query.json"
                write_new_json(query_release, {"protocol_sha256": file_sha256(root / protocol),
                    "allowed_stages": ["predict"], "data_ready": True,
                    "source_lock_sha256": file_sha256(root / output / "source_lock.json")})
                predicted = run_predict(root, protocol, output, query_release.relative_to(root))
                self.assertEqual(predicted["status"], "DONE_PREDICTIONS_LOCKED_NOT_SCORED")
            scored = run_score(root, protocol, output)
            self.assertEqual(scored["query_truth_first_opened_in"], "score")
            aggregate = json.loads((root / output / "scoring/aggregate_metrics.json").read_text())
            main = [r for r in aggregate if r["method"] == "none" and r["stratum"] == "all"]
            self.assertEqual(len(main), 4)
            self.assertTrue(all(r["seeds"] == [42, 43, 44] for r in main))
            manifest = json.loads((root / output / "prediction_lock.json").read_text())
            conservative = [p for p in manifest["predictions"] if p["method"] == "conservative"]
            self.assertTrue(all(p["adapter"]["status"] == "NO_OP" for p in conservative))
            with self.assertRaises(FileExistsError):
                run_score(root, protocol, output)
            missing_seed = dict(manifest)
            missing_seed["predictions"] = [p for p in manifest["predictions"]
                if not (p["family"] == "svm" and p["seed"] == 44 and p["method"] == "none")]
            (root / output / "prediction_lock.json").write_text(json.dumps(missing_seed), encoding="utf-8")
            with patch.object(Path, "open", forbid_query_truth):
                with self.assertRaisesRegex(LeakageError, "complete registered"):
                    run_score(root, protocol, output)

    def test_source_calibration_episodes_are_a_separate_source_only_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol_path, _, _ = self.fixture(root)
            protocol = json.loads((root / protocol_path).read_text())
            task = protocol["tasks"][0]
            task["calibration"] = {}
            with np.load(root / task["source_pack"]["path"], allow_pickle=False) as archive:
                ids, labels = archive["record_ids"], archive["y"]
            episodes = []
            for i in range(6):
                base, support, query = [], [], []
                for label in task["classes"]:
                    ordered = np.roll(ids[labels == label], i)
                    base.extend(ordered[:3])
                    support.extend(ordered[3:4])
                    query.extend(ordered[4:])
                episodes.append({"base_ids": base, "support_ids": support, "query_ids": query,
                    "shots": {"1": support}, "stage": "selection" if i < 3 else "gate"})
            task["source_calibration_episodes"] = episodes
            output = root / "reports/calibration_synthetic"
            output.mkdir()
            budget = Budget(protocol["resources"])
            policy = source_calibration(root, task, protocol["candidates"]["mcuve_svm"][0],
                                        protocol, output, "SYNTHETIC_TEST_CODE", budget)
            self.assertEqual(policy["evaluations"], 6 * 12)
            self.assertEqual(budget.completed_fits, 6)
            rows_path = output / "calibration_selection" / task["id"] / "source_episode_results.json.gz"
            with gzip.open(rows_path, "rt", encoding="utf-8") as handle:
                rows = json.load(handle)
            self.assertEqual({r["stage"] for r in rows}, {"selection", "gate"})
            for row in rows:
                for event in row["fit_events"]:
                    if event["event"] == "fit_completed":
                        allowed = episodes[row["episode"]]["support_ids"]
                        self.assertTrue(set(event["fit_sample_ids"]) <= set(allowed))

    def test_budget_stop_locks_only_wholly_completed_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol_path, release_path, _ = self.fixture(root)
            protocol = json.loads((root / protocol_path).read_text())
            parent = protocol["tasks"][0]
            second = {**parent, "id": "SYNTHETIC_SECOND_TASK", "inherit_selection_from": parent["id"], "inner_folds": []}
            protocol["tasks"].append(second)
            (root / protocol_path).write_text(json.dumps(protocol), encoding="utf-8")
            (root / release_path).write_text(json.dumps({"protocol_sha256": file_sha256(root / protocol_path),
                "allowed_stages": ["source"], "data_ready": True}), encoding="utf-8")

            def bounded_fit(out, task, *args, **kwargs):
                if task["id"] == second["id"]:
                    raise BudgetExceeded("SYNTHETIC_FORCED_STOP_BEFORE_SECOND_TASK_FIT")
                return fit_logged(out, task, *args, **kwargs)

            with patch("herb_spectra.supplemental_execution.fit_logged", bounded_fit):
                result = run_source(root, protocol_path, "reports/SYNTHETIC_BUDGET", release_path)
            self.assertEqual(result["status"], "PARTIAL_RESOURCE_LIMIT_COMPLETE_TASKS_ONLY")
            lock = json.loads((root / "reports/SYNTHETIC_BUDGET/source_lock.json").read_text())
            self.assertEqual(lock["completed_task_ids"], [parent["id"]])
            self.assertEqual(lock["task_status"][second["id"]], "NOT_RUN_RESOURCE_LIMIT")
            self.assertEqual(len(lock["models"]), 4 * 3)
            self.assertTrue(all(model["task"] == parent["id"] for model in lock["models"]))


if __name__ == "__main__":
    unittest.main()
