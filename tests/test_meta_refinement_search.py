"""Small numerical/role tests for anchored first-order MAML refinement."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call

from herb_spectra.meta_refinement_search import (anchored_adaptation, candidate_grid, choose_configuration, context_meta_seed,
    class_weights, evaluate_development, evaluate_final, meta_grid, model_scores, train_meta_pair)
from herb_spectra.neural_base_search import DenseClassifier
from herb_spectra.paper_fewshot_classification import sample_meta_episodes, state_fingerprint


class MetaRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.model = DenseClassifier(7, (12,), 7)
        self.y = np.repeat(np.arange(7), [3, 1, 3, 3, 3, 3, 3])
        self.x = (np.eye(7)[self.y] + np.random.default_rng(7).normal(0, .01, (len(self.y), 7))).astype(np.float32)
        self.ids = np.asarray([f"sample-{i}" for i in range(len(self.y))])
        self.config = {"id": "test", "updates": 2, "inner_learning_rate": .01,
                       "outer_learning_rate": .001, "source_anchor_weight": 1.}
        self.selection = {"minimum_accuracy_gain": .001, "max_macro_f1_loss": .005,
                          "max_balanced_accuracy_loss": .005, "reference_meta_id": "m000"}

    def test_predeclared_grid_has_four_sources_and_sixteen_paths(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 16); self.assertEqual(len({c["id"] for c in grid}), 16)
        self.assertEqual({m["updates"] for m in meta_grid()}, {100, 500})
        self.assertTrue(all(sum(c["meta_id"] == m["id"] for c in grid) == 4 for m in meta_grid()))

    def test_final_context_seed_accepts_none_and_is_distinct_from_development(self):
        final = context_meta_seed({"seed": 42, "episode": None})
        self.assertEqual(final, 1642)
        self.assertNotIn(final, [context_meta_seed({"seed": 42, "episode": e}) for e in range(6)])

    def test_source_freeze_eligibility_and_shared_task_records(self):
        before = state_fingerprint(self.model)
        models, record = train_meta_pair(self.model, self.x, self.y, self.ids, self.config, seed=13, class_weight="balanced")
        self.assertEqual(before, state_fingerprint(self.model))
        self.assertEqual(record["policy"]["excluded_insufficient_classes"], [1])
        self.assertTrue(record["same_task_ids_and_update_counts"])
        self.assertFalse(record["equal_FLOPs"])
        for task in record["source_tasks"]:
            self.assertFalse(set(task["support_ids"]) & set(task["query_ids"]))
            self.assertEqual(len(task["support_ids"]), 3)
        self.assertEqual(set(models), {"supervised", "fomaml"})
        self.assertTrue(all(state_fingerprint(m) != before for m in models.values()))

    def test_one_update_matches_independent_first_order_gradient(self):
        config = dict(self.config, updates=1)
        fitted, _ = train_meta_pair(self.model, self.x, self.y, self.ids, config, seed=29, class_weight="balanced")
        expected = deepcopy(self.model); opt = torch.optim.Adam(expected.parameters(), lr=config["outer_learning_rate"])
        tasks, _ = sample_meta_episodes(self.y, self.ids, episodes=1, seed=29, way=3, support=1, query=1)
        task = tasks[0]; x = torch.tensor(self.x); y = torch.tensor(self.y)
        parameters = dict(expected.named_parameters())
        support = F.cross_entropy(expected(x[task["support"]]), y[task["support"]])
        inner = torch.autograd.grad(support, tuple(parameters.values()))
        fast = {k: (v-config["inner_learning_rate"]*g).detach().requires_grad_(True)
                for (k, v), g in zip(parameters.items(), inner)}
        query = F.cross_entropy(functional_call(expected, fast, (x[task["query"]],)), y[task["query"]])
        first_order = torch.autograd.grad(query, tuple(fast.values()))
        counts = torch.bincount(y, minlength=7).float(); weights = len(y)/(7*counts)
        anchor = F.cross_entropy(expected(x), y, weight=weights)
        replay = torch.autograd.grad(anchor, tuple(parameters.values()))
        for p, g, a in zip(parameters.values(), first_order, replay):
            p.grad = (g+a).detach().clone()
        opt.step()
        for name, tensor in expected.state_dict().items():
            torch.testing.assert_close(tensor, fitted["fomaml"].state_dict()[name], rtol=0, atol=0)

    def test_supervised_control_independent_of_inner_learning_rate(self):
        first, _ = train_meta_pair(self.model, self.x, self.y, self.ids, self.config, seed=11)
        second, _ = train_meta_pair(self.model, self.x, self.y, self.ids, dict(self.config, inner_learning_rate=.2), seed=11)
        self.assertEqual(state_fingerprint(first["supervised"]), state_fingerprint(second["supervised"]))

    def test_no_target_no_replay_is_identity_and_initial_stays_frozen(self):
        before = state_fingerprint(self.model)
        config = dict(candidate_grid()[0], source_replay_weight=0., steps=5)
        result = anchored_adaptation(self.model, self.x, self.y, None, None, self.x, [config], target_loss=False)
        np.testing.assert_array_equal(result[config["id"]], model_scores(self.model, self.x).argmax(1))
        self.assertEqual(before, state_fingerprint(self.model))

    def test_query_predictions_are_pointwise(self):
        config = dict(candidate_grid()[0], steps=2)
        supports = np.eye(7, dtype=np.float32); labels = np.arange(7)
        whole = anchored_adaptation(self.model, self.x, self.y, supports, labels, self.x, [config])[config["id"]]
        pieces = [anchored_adaptation(self.model, self.x, self.y, supports, labels, row[None], [config])[config["id"]][0]
                  for row in self.x[:4]]
        np.testing.assert_array_equal(whole[:4], pieces)

    def test_support_labels_required_and_known(self):
        with self.assertRaises(ValueError):
            anchored_adaptation(self.model, self.x, self.y, None, None, self.x, candidate_grid()[:1])
        with self.assertRaises(ValueError):
            anchored_adaptation(self.model, self.x, self.y, np.eye(7, dtype=np.float32), np.zeros(7, dtype=int), self.x, candidate_grid()[:1])
        with self.assertRaises(ValueError):
            train_meta_pair(self.model, self.x, self.y, np.repeat("same", len(self.y)), self.config)

    def _selection_rows(self, *, target=.8, source=.7, matched=.81):
        rows = []
        for candidate in candidate_grid():
            for strategy in ("fomaml", "supervised"):
                for stage in ("selection", "gate"):
                    row = {"candidate_id": candidate["id"], "strategy": strategy, "stage": stage}
                    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                        row[metric] = target; row["source_"+metric] = source; row["matched_"+metric] = matched
                    rows.append(row)
        return rows

    def test_selection_rejects_gain_explained_by_replay_control(self):
        lock = choose_configuration(self._selection_rows(), candidate_grid(), self.selection)
        self.assertIsNone(lock["selected"])
        self.assertTrue(lock["fallback_to_reference_source"])

    def test_selection_requires_gate_and_breaks_ties_by_budget(self):
        rows = self._selection_rows(target=.82, matched=.7)
        lock = choose_configuration(rows, candidate_grid(), self.selection)
        self.assertEqual(lock["selected"]["id"], candidate_grid()[0]["id"])
        self.assertFalse(lock["selected_using_final_query"])
        for row in rows:
            if row["stage"] == "gate":
                row["accuracy"] = .6
        self.assertIsNone(choose_configuration(rows, candidate_grid(), self.selection)["selected"])

    def test_balanced_source_weights_keep_every_class(self):
        labels = torch.tensor(self.y)
        weights = class_weights(labels, 7, "balanced")
        self.assertAlmostEqual(float(weights[labels].sum()), len(labels), places=5)
        self.assertGreater(float(weights[1]), float(weights[0]))
        with self.assertRaises(ValueError):
            class_weights(torch.zeros(4, dtype=torch.long), 7, "balanced")

    def _context(self, final=False):
        classes = np.asarray([f"ZY{i+1:02d}" for i in range(7)])
        context = {"episode": None if final else 0, "seed": 42,
            "candidate": {"class_weight": "none"}, "classes": classes,
            "model": self.model, "fingerprint": state_fingerprint(self.model),
            "preprocessor_sha256": "fixture-preprocessor", "selection_lock_sha256": "fixture-lock",
            "source": {"X": self.x, "y": self.y, "record_ids": self.ids},
            "support_pool": {"X": np.eye(7, dtype=np.float32), "y": np.arange(7),
                             "record_ids": np.asarray([f"support-{i}" for i in range(7)])},
            "prediction": {"X": np.eye(7, dtype=np.float32),
                           "record_ids": np.asarray([f"query-{i}" for i in range(7)])}}
        if not final:
            context["prediction"]["y"] = np.arange(7)
        truth = {f"query-{i}": classes[i] for i in range(7)}
        return context, truth

    def test_complete_final_fixture_and_resume_without_refitting(self):
        context, truth = self._context(final=True)
        protocol = {"final": {"support_draws": 2, "support_seed": 31}}
        config = dict(self.config, id="m000")
        lock = {"selected": None, "reference_meta_id": "m000", "diagnostic": candidate_grid()[0],
                "supervised_diagnostic": candidate_grid()[0]}
        with TemporaryDirectory() as directory, patch("herb_spectra.meta_refinement_search.meta_grid", return_value=[config]):
            rows = evaluate_final(context, truth, lock, protocol, Path(directory))
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(r["total"] == 7 for r in rows))
            with patch("herb_spectra.meta_refinement_search.train_meta_pair", side_effect=AssertionError("unexpected refit")):
                self.assertEqual(rows, evaluate_final(context, truth, lock, protocol, Path(directory), resume=True))
            self.assertNotIn("y", context["prediction"])

    def test_complete_development_fixture_records_all_candidates_and_controls(self):
        context, _ = self._context()
        protocol = {"development": {"support_draws": 2, "support_seed": 31, "selection_episodes": 4}}
        config = dict(self.config, id="m000"); candidates = candidate_grid()[:4]
        with TemporaryDirectory() as directory, patch("herb_spectra.meta_refinement_search.meta_grid", return_value=[config]), \
                patch("herb_spectra.meta_refinement_search.candidate_grid", return_value=candidates):
            rows = evaluate_development(context, protocol, Path(directory))
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r["stage"] == "selection" for r in rows))
            self.assertTrue(all("source_confusion_matrix" in r and "matched_confusion_matrix" in r for r in rows))
            self.assertEqual(rows, evaluate_development(context, protocol, Path(directory), resume=True))


if __name__ == "__main__":
    unittest.main()
