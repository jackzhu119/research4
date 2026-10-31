import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from herb_spectra.image_fusion_search import (
    _verify_release, align_scores, candidate_grid, complementarity, evaluate_context,
    fallback_candidate, fused_probabilities, object_digest, probability_metrics,
    select_fusion, softmax_probabilities, verify_context,
)


def context_fixture(final=False):
    rng = np.random.default_rng(55)
    counts = [210, 3, 60, 60, 60, 60, 60] if final else [51, 1, 51, 51, 51, 51, 51]
    source_y = np.repeat(np.arange(7), counts)
    source_ids = np.array([f"ZY{c+1:02d}-source-{i:03d}" for c in range(7) for i in range(counts[c])])
    query_y = np.repeat(np.arange(7), [70, 1, 20, 20, 20, 20, 20] if final else [17, 1, 17, 17, 17, 17, 17])
    query_ids = np.array([f"query-{i:03d}" for i in range(len(query_y))])
    included = np.bincount(source_y)[source_y] >= 3
    providers = ("svm", "mlp", "mlp_sup100")
    return {"episode": None if final else 0, "seed": 42,
        "classes": np.array([f"ZY{i:02d}" for i in range(1, 8)]),
        "source_ids": source_ids, "source_y": source_y,
        "prediction": {"record_ids": query_ids,
            "scores_by_provider": {p: rng.normal(size=(len(query_y), 7)) for p in providers},
            **({} if final else {"y": query_y})},
        "oof": {"record_ids": source_ids[included], "y": source_y[included],
            "scores_by_provider": {p: rng.normal(size=(included.sum(), 7)) for p in providers}, "folds": []},
        "temperatures": {"svm": .7, "mlp": 2., "mlp_sup100": 3.},
        "source_fingerprints": {p: "frozen-"+p for p in providers}, "lineage": {"no_query_fit": True}}


def aggregate_fixture():
    return {stage: {c["id"]: {"macro_f1": .5, "accuracy": .7, "balanced_accuracy": .6,
                              "evaluations": 12 if stage == "selection" else 6}
                    for c in candidate_grid()} for stage in ("selection", "gate")}


class ImageFusionTests(unittest.TestCase):
    def test_class_alignment_permutation_and_unknown_rejection(self):
        scores = np.array([[2., 7., 1.], [4., 3., 8.]])
        np.testing.assert_array_equal(align_scores(scores, ["B", "C", "A"], ["A", "B", "C"]),
                                      scores[:, [2, 0, 1]])
        with self.assertRaises(ValueError):
            align_scores(scores, ["A", "A", "C"], ["A", "B", "C"])

    def test_temperature_preserves_argmax_and_probability_sum(self):
        scores = np.array([[100., -90., 0.], [1., 2., 4.], [-2., -2., -2.]])
        for t in (.05, 1., 20.):
            p = softmax_probabilities(scores, t)
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-15)
            np.testing.assert_array_equal(p.argmax(1), scores.argmax(1))
        with self.assertRaises(ValueError):
            softmax_probabilities(scores, 0.)

    def test_fusion_endpoints_and_pointwise_invariance(self):
        context = context_fixture()
        scores, temps, classes = context["prediction"]["scores_by_provider"], context["temperatures"], context["classes"]
        candidate = fallback_candidate()
        np.testing.assert_allclose(fused_probabilities(scores, temps, candidate, classes),
                                   softmax_probabilities(scores["mlp_sup100"], temps["mlp_sup100"]))
        candidate = {**candidate, "svm_weight": 1.}
        np.testing.assert_allclose(fused_probabilities(scores, temps, candidate, classes),
                                   softmax_probabilities(scores["svm"], temps["svm"]))
        candidate = {**candidate, "svm_weight": .5}
        all_p = fused_probabilities(scores, temps, candidate, classes)
        one_p = fused_probabilities({p: v[[8]] for p, v in scores.items()}, temps, candidate, classes)
        np.testing.assert_array_equal(all_p[[8]], one_p)

    def test_probability_metrics_known_brier_and_boundary_ece(self):
        p = np.array([[1., 0.], [.25, .75]])
        value = probability_metrics(p, np.array([0, 1]))
        self.assertAlmostEqual(value["brier"], .0625)
        self.assertAlmostEqual(value["nll"], -np.log(.75)/2)
        self.assertAlmostEqual(value["ece"], .125)
        with self.assertRaises(ValueError):
            probability_metrics(p*.5, np.array([0, 1]))

    def test_complementarity_oracle_only_upper_bound(self):
        value = complementarity([0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0], 2)
        self.assertEqual([value[k] for k in ("both_correct", "svm_only_correct", "mlp_only_correct", "both_wrong")], [1, 1, 1, 1])
        self.assertEqual(value["oracle_accuracy_upper_bound"], .75)
        self.assertTrue(value["oracle_is_truth_based_unavailable_predictor"])

    def test_selection_requires_beating_both_endpoints_and_gate(self):
        aggregate = aggregate_fixture()
        candidate = candidate_grid()[2]
        for stage in aggregate:
            aggregate[stage][candidate["id"]]["macro_f1"] = .55
        self.assertEqual(select_fusion(aggregate)["selected"], candidate)
        svm_endpoint = candidate_grid()[4]["id"]
        aggregate["selection"][svm_endpoint]["macro_f1"] = .56
        self.assertEqual(select_fusion(aggregate)["selected"], fallback_candidate())
        aggregate["selection"][svm_endpoint]["macro_f1"] = .5
        aggregate["gate"][candidate["id"]]["accuracy"] = .68
        self.assertEqual(select_fusion(aggregate)["selected"], fallback_candidate())

    def test_macro_priority_tie_uses_accuracy_then_middle_weight(self):
        aggregate = aggregate_fixture()
        first, middle = candidate_grid()[1], candidate_grid()[2]
        for stage in aggregate:
            aggregate[stage][first["id"]]["macro_f1"] = .552
            aggregate[stage][middle["id"]]["macro_f1"] = .551
        self.assertEqual(select_fusion(aggregate)["selected"], middle)

    def test_development_records_preserve_all_context_arrays(self):
        context = context_fixture()
        before = object_digest(context)
        with tempfile.TemporaryDirectory() as d:
            rows, record = evaluate_context(context, Path(d), methods={"chosen": fallback_candidate()})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_support_labels_used"], 0)
            self.assertEqual(record["context_fingerprint_before"], record["context_fingerprint_after"])
            self.assertTrue((Path(d)/"predictions.csv.gz").is_file())
        self.assertEqual(object_digest(context), before)

    def test_final_payload_rejects_truth_or_oofoverlap(self):
        context = context_fixture(final=True)
        verify_context(context, final=True)
        context["prediction"]["y"] = np.zeros(171, dtype=int)
        with self.assertRaises(ValueError):
            verify_context(context, final=True)
        context = context_fixture()
        context["oof"]["record_ids"][0] = "not-a-source-specimen"
        with self.assertRaises(ValueError):
            verify_context(context, final=False)

    def test_global_release_must_cover_both_fixed_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            records = []
            for name in ("own", "score"):
                run = root/name; run.mkdir()
                lock, protocol, manifest = run/"selection_lock.json", run/"protocol_snapshot.json", run/"run_manifest.json"
                lock.write_text(name+" lock"); protocol.write_text(name+" protocol")
                manifest.write_text(json.dumps({"protocol_sha256": sha(protocol)}))
                records.append({"name": name, "selection_lock": str(lock), "selection_lock_sha256": sha(lock),
                    "protocol": str(protocol), "protocol_sha256": sha(protocol), "run_manifest_sha256": sha(manifest)})
            own, score, release = root/"own/selection_lock.json", root/"score/selection_lock.json", root/"release.json"
            release.write_text(json.dumps({"rounds": records}))
            _verify_release(release, own, score)
            score.write_text("changed source lock")
            with self.assertRaises(ValueError):
                _verify_release(release, own, score)


if __name__ == "__main__":
    unittest.main()
