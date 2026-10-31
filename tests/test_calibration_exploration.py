"""Protocol tests on synthetic specimens; no medicinal-plant performance claims."""
from copy import deepcopy
from pathlib import Path
import gzip
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from herb_spectra.calibration_exploration import (ProjectionCache, choose_with_gate,
    complexity, development_episodes, encoded, fast_metrics, feature_payload,
    final_predict_draw, fit_fixed_source, nested_draws, object_digest,
    validate_protocol, run_exploration, digest, verify_existing_training_plan)
from herb_spectra.conservative_calibration import build_context, candidate_grid


def specimens(counts=(30, 3, 15), seed=4):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["A", "B", "C"], counts)
    X = rng.normal(size=(len(labels), 10))
    X += np.asarray([{"A": 0, "B": 1, "C": 2}[v] for v in labels])[:, None] * np.linspace(-1.,1.,10)
    return {"X_alpha": X, "y": labels,
            "record_ids": np.asarray([f"p{i:03d}" for i in range(len(labels))])}


class CalibrationExplorationTests(unittest.TestCase):
    def test_manual_metrics_equal_independent_library_metrics(self):
        truth = np.asarray([0, 0, 0, 1, 2, 2, 2, 2])
        prediction = np.asarray([0, 1, 0, 1, 0, 2, 1, 2])
        actual = fast_metrics(truth, prediction, 3, detailed=True)
        self.assertAlmostEqual(actual["accuracy"], accuracy_score(truth, prediction))
        self.assertAlmostEqual(actual["balanced_accuracy"], balanced_accuracy_score(truth, prediction))
        self.assertAlmostEqual(actual["macro_f1"], f1_score(truth, prediction, average="macro"))
        self.assertEqual(sum(map(sum,actual["confusion_matrix"])), len(truth))

    def test_development_roles_are_complete_disjoint_and_cycle_rare_class(self):
        data = specimens()
        episodes = development_episodes(data, 12, 7)
        rare = set(np.flatnonzero(data["y"] == "B"))
        arrangements = set()
        for episode in episodes:
            roles = [set(episode[k]) for k in ["source", "support_pool", "query"]]
            self.assertEqual(set.union(*roles), set(range(len(data["y"]))))
            self.assertTrue(all(not roles[i]&roles[j] for i in range(3) for j in range(i+1,3)))
            arrangements.add(tuple(next(iter(role&rare)) for role in roles))
        self.assertEqual(len(arrangements), 6)
        self.assertEqual([x.tolist() for x in episodes[0].values()],
                         [x.tolist() for x in development_episodes(data, 12, 7)[0].values()])

    def test_supports_are_nested_shared_and_insufficient_k_is_rejected(self):
        labels = np.repeat(["A", "B"], 12)
        draws = nested_draws(labels, ["A", "B"], [1,3,10], 3, 9)
        for repeat in range(3):
            cases = {d["shots_per_class"]:set(d["indices"]) for d in draws if d["repeat"] == repeat}
            self.assertTrue(cases[1] < cases[3] < cases[10])
        with self.assertRaisesRegex(ValueError, "not feasible"):
            nested_draws(np.array(["A","B"]), ["A","B"], [3])

    def test_matched_control_guard_and_late_gate_fallback(self):
        candidates = [{"id":"i","family":"identity"},
                      {"id":"p","family":"prototype_shrinkage","blend":.25,"pseudocount":5,"gate_margin":None}]
        base = {"accuracy":.8,"macro_f1":.8,"balanced_accuracy":.8,"failures":0,
                "matched_accuracy":.8,"matched_macro_f1":.8,"matched_balanced_accuracy":.8}
        prototype = dict(base, accuracy=.9, matched_accuracy=.91)
        stages = {stage:{"i":deepcopy(base),"p":deepcopy(prototype)} for stage in ["selection","gate"]}
        self.assertEqual(choose_with_gate(stages,candidates)["selected"]["id"],"i")
        for stage in stages: stages[stage]["p"]["matched_accuracy"] = .82
        self.assertEqual(choose_with_gate(stages,candidates)["selected"]["id"],"p")
        stages["gate"]["p"]["accuracy"] = .79
        self.assertEqual(choose_with_gate(stages,candidates)["selected"]["id"],"i")

    def test_query_truth_and_batch_features_never_enter_fitting(self):
        data = specimens(counts=(15,15,15))
        pipe = fit_fixed_source(data["X_alpha"], data["y"], "full")
        before = joblib.hash(pipe)
        z, scores = feature_payload(pipe, data["X_alpha"])
        context = build_context(z,scores,data["y"],pipe.classes_)
        support = np.asarray([0,15,30]); query = np.asarray([1,2,16,17,31,32])
        candidate = next(c for c in candidate_grid() if c["family"] == "mistake_ridge")
        first,_ = final_predict_draw(context,candidate,z,scores,data["y"],z[query],scores[query],support)
        changed_truth = data["y"][query][::-1]
        first_metrics = fast_metrics(encoded(data["y"][query],pipe.classes_),first,3)
        changed_metrics = fast_metrics(encoded(changed_truth,pipe.classes_),first,3)
        self.assertNotEqual(first_metrics, changed_metrics)
        single,_ = final_predict_draw(context,candidate,z,scores,data["y"],z[query[:1]],scores[query[:1]],support)
        assert_array_equal(single,first[:1])
        self.assertEqual(joblib.hash(pipe),before)

    def test_projection_cache_is_pure_transform_and_holds_exact_references(self):
        data = specimens(counts=(8,8,8))
        classes=np.unique(data["y"]); scores=np.eye(3)[encoded(data["y"],classes)]
        original=build_context(data["X_alpha"],scores,data["y"],classes)
        before=joblib.hash(original); cache=ProjectionCache(original)
        values=data["X_alpha"][:4].copy()
        a=cache.project(values); b=cache.project(values)
        self.assertIs(a,b)
        np.testing.assert_allclose(a, original.project(values))
        self.assertEqual(joblib.hash(original),before)

    def test_protocol_grid_hash_rejects_changed_search_and_complexity_is_explicit(self):
        grid=candidate_grid()
        protocol={"candidates":{"count":len(grid),"grid_sha256":object_digest(grid)},
                  "selection":{"family_rank":{"identity":0,"mistake_bias":1,"mistake_kernel":2,
                    "mistake_ridge":3,"prototype_shrinkage":4,"damped_anchor":5,"legacy_anchor":6}},
                  "source_model":{"preprocessing":"snv","C":10},"model_seed":42,
                  "development":{"selection_episodes":2,"gate_episodes":1,"episodes":3}}
        validate_protocol(protocol,grid)
        bad=deepcopy(grid);bad[0]["new_parameter"]=1
        with self.assertRaisesRegex(ValueError,"grid"):
            validate_protocol(protocol,bad)
        low={"id":"x","family":"mistake_bias","strength":.1}
        high={"id":"z","family":"damped_anchor","blend":.01,"regularization":1000}
        self.assertLess(complexity(low),complexity(high))

    def test_synthetic_end_to_end_checkpoint_resume_and_provenance_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rng=np.random.default_rng(12)
            ids=[]; labels=[]; records=[]; blocks=[]
            counts={"training":12,"calibration":5,"prediction":5}
            for role,count in counts.items():
                for ci,label in enumerate(["A","B","C"]):
                    for j in range(count):
                        identifier=f"{role}-{label}-{j}"
                        ids.append(identifier);labels.append(label)
                        blocks.append(rng.normal(size=10)+ci*np.linspace(-1,1,10))
                        records.append({"record_id":identifier,"physical_sample_id":identifier,
                                        "origin_group_id":label,"role":role})
            X=np.asarray(blocks); y=np.asarray(labels); identifiers=np.asarray(ids)
            input_path=root/'input.npz';plan_path=root/'plan.json';protocol_path=root/'protocol.json'
            np.savez(input_path,X_alpha=X,X_n=X+.1,axis=np.linspace(.1,1.5,10),y=y,
                     record_ids=identifiers,groups=identifiers,batch_ids=np.repeat('',len(y)),
                     labels_verified=np.ones(len(y),bool),dataset_version=np.asarray('synthetic'))
            plan={"dataset_version":"synthetic","records":records,"split_seed":1,
                  "counts_by_origin":{label:counts.copy() for label in ['A','B','C']}}
            plan_path.write_text(json.dumps(plan))
            source=np.asarray([row['role']=='training' for row in records])
            pipe=fit_fixed_source(X[source],y[source],'full')
            model_path=root/'source.joblib';joblib.dump(pipe,model_path)
            model_manifest=root/'model_manifest.json'
            model_manifest.write_text(json.dumps({"models":[{"selector":"full","path":str(model_path),"sha256":digest(model_path)}]}))
            grid=[candidate_grid()[0],candidate_grid()[1],next(c for c in candidate_grid() if c['family']=='mistake_bias')]
            protocol={"protocol_id":"synthetic_engineering_fixture","scope":"synthetic only",
                "candidates":{"count":len(grid),"grid_sha256":object_digest(grid)},
                "selection":{"family_rank":{"identity":0,"mistake_bias":1,"mistake_kernel":2,"mistake_ridge":3,"prototype_shrinkage":4,"damped_anchor":5,"legacy_anchor":6},
                    "minimum_accuracy_gain":.001,"max_macro_f1_loss":.005,"max_balanced_accuracy_loss":.005,"tie_tolerance":.001},
                "source_model":{"preprocessing":"snv","C":10,"existing_seven_class_models":str(model_manifest),"training_plan_reference":str(plan_path)},
                "model_seed":42,"selectors":["full"],"dataset_version":"synthetic",
                "frequency":{"lower_inclusive":.1,"upper_inclusive":1.5,"expected_points":10},
                "development":{"selection_episodes":2,"gate_episodes":1,"episodes":3,"split_seed":1,"support_seed":2,"support_draws":2},
                "final":{"support_draws":2,"support_seed":3,"random_query_seed":4,"random_query_examples":3},
                "scopes":[{"id":"synthetic_three_class","exclude_groups":[],"shots":[1],"role_counts":{"training":36,"calibration":15,"prediction":15}}]}
            protocol_path.write_text(json.dumps(protocol));output=root/'run'
            with patch('herb_spectra.calibration_exploration.candidate_grid',return_value=grid):
                summary=run_exploration(input_path,plan_path,protocol_path,output)
                self.assertTrue(summary)
                self.assertTrue((output/'selection_lock.json').is_file())
                status=json.loads((output/'status.json').read_text())
                self.assertFalse(status['final_query_used_for_selection'])
                with gzip.open(output/'final/synthetic_three_class/full/predictions_k1.csv.gz','rt') as handle:
                    rows=list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertTrue(all(int(r['is_correct'])==int(r['y_true']==r['y_predicted']) for r in rows))
                with patch('herb_spectra.calibration_exploration.fit_fixed_source',side_effect=AssertionError('must not refit')):
                    self.assertEqual(run_exploration(input_path,plan_path,protocol_path,output,resume=True),summary)
                protocol['selection']['minimum_accuracy_gain']=.002
                protocol_path.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(ValueError,'Resume refused'):
                    run_exploration(input_path,plan_path,protocol_path,output,resume=True)

    def test_existing_model_cannot_be_reused_with_changed_training_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);original=root/'original.json';changed=root/'changed.json'
            original.write_text(json.dumps({'records':[{'record_id':'A','role':'training'},{'record_id':'B','role':'prediction'}]}))
            changed.write_text(json.dumps({'records':[{'record_id':'B','role':'training'},{'record_id':'A','role':'prediction'}]}))
            with self.assertRaisesRegex(ValueError,'training cohort'):
                verify_existing_training_plan(changed,original)
            self.assertEqual(verify_existing_training_plan(original,original),digest(original))


if __name__ == "__main__":
    unittest.main()
