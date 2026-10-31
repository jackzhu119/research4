"""Train-only latent-search protocol tests, using synthetic spectra only."""
from copy import deepcopy
import gzip
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from herb_spectra.calibration_exploration import development_episodes, fast_metrics
from herb_spectra.latent_adaptation import build_latent_context, latent_candidate_grid
from herb_spectra.latent_calibration_search import (LatentTransformCache, aggregate_latent,
    choose_latent_with_gate, evaluate_latent_episode, object_digest, prediction_only,
    run_latent_search, validate_source_run)
from herb_spectra.spectral_models import fit_source_candidate, source_candidate_grid
from herb_spectra.spectral_source_search import run_source_search, source_reference


def fixture(root):
    rng=np.random.default_rng(29)
    records,labels,role_names=[],[],[]
    for label,counts in zip(["A","B","C"],[(12,4,4),(3,1,1),(12,4,4)]):
        for role,count in zip(["training","calibration","prediction"],counts):
            for i in range(count):
                records.append(f"{label}-{role}-{i:02d}");labels.append(label);role_names.append(role)
    X=rng.normal(size=(len(labels),21))+np.asarray([{"A":0,"B":1,"C":2}[v] for v in labels])[:,None]*np.linspace(-1,1,21)
    arrays={"X_alpha":X,"X_n":X/4+2,"axis":np.linspace(.1,1.5,21),
        "record_ids":np.asarray(records),"groups":np.asarray(records),"y":np.asarray(labels),
        "labels_verified":np.ones(len(labels),dtype=bool),"batch_ids":np.repeat("",len(labels)),
        "dataset_version":np.asarray("synthetic-v2")}
    plan={"dataset_version":"synthetic-v2","split_seed":9,"records":[
        {"record_id":v,"physical_sample_id":v,"origin_group_id":y,"role":role}
        for v,y,role in zip(records,labels,role_names)]}
    source_grid=source_candidate_grid()
    sources=[source_reference(source_grid),next(c for c in source_grid if c["model"]=="plsda" and c["preprocessing"]=="snv"),
             next(c for c in source_grid if c["model"]=="ridge" and c["preprocessing"]=="snv")]
    latent_grid=latent_candidate_grid()
    latent=[next(c for c in latent_grid if c["family"]==family) for family in
            ["identity","anchored_ridge","local_memory","mahalanobis_centroid"]]
    settings={"episodes":6,"selection_episodes":4,"model_seed":42,
              "minimum_accuracy_gain":.001,"max_macro_f1_loss":.005,
              "max_balanced_accuracy_loss":.005,"tie_tolerance":.001}
    protocol={"protocol_id":"synthetic-stageB-v1","dataset_version":"synthetic-v2",
        "scope":"synthetic engineering test, not herb evidence",
        "frequency":{"lower_inclusive":.1,"upper_inclusive":1.5,"expected_points":21},
        "fixed_role_counts":{"training":27,"calibration":9,"prediction":9},
        "candidates":{"source_count":len(sources),"source_grid_sha256":object_digest(sources),
                      "latent_count":len(latent),"latent_grid_sha256":object_digest(latent)},
        "source_search":{**settings,"split_seed":31},
        "calibration_search":{**settings,"support_draws":2,"support_seed":32,"shots_per_class":1},
        "final":{"support_draws":2,"support_seed":33,"random_query_seed":34,"random_query_examples":3,
                 "model_seed":42,"shots_per_class":1},
        "full_export_sensitivity":{"enabled":False}}
    ip,pp,pr=(root/name for name in ["data.npz","plan.json","protocol.json"])
    np.savez_compressed(ip,**arrays);pp.write_text(json.dumps(plan));pr.write_text(json.dumps(protocol))
    return arrays,plan,protocol,sources,latent,ip,pp,pr


class LatentCalibrationSearchTests(unittest.TestCase):
    def test_matched_control_rejects_auxiliary_classifier_gain_and_gate_has_no_second_chase(self):
        grid=latent_candidate_grid();candidates=[grid[0],grid[1],grid[-1]]
        base={"accuracy":.8,"macro_f1":.8,"balanced_accuracy":.8,
              "matched_accuracy":.8,"matched_macro_f1":.8,"matched_balanced_accuracy":.8,
              "evaluations":4,"failures":0}
        stats={stage:{c["id"]:deepcopy(base) for c in candidates} for stage in ["selection","gate"]}
        for stage in stats:stats[stage][candidates[1]["id"]].update(accuracy=.91,matched_accuracy=.92)
        self.assertEqual(choose_latent_with_gate(stats,candidates)["selected"]["family"],"identity")
        for stage in stats:stats[stage][candidates[1]["id"]]["matched_accuracy"]=.8
        self.assertEqual(choose_latent_with_gate(stats,candidates)["selected"]["id"],candidates[1]["id"])
        stats["gate"][candidates[1]["id"]]["accuracy"]=.7
        for stage in stats:stats[stage][candidates[-1]["id"]].update(accuracy=.85,macro_f1=.85,balanced_accuracy=.85)
        result=choose_latent_with_gate(stats,candidates)
        self.assertEqual(result["proposed"]["id"],candidates[1]["id"])
        self.assertEqual(result["selected"]["family"],"identity")

    def test_projection_and_distance_cache_are_pure_pointwise_source_transforms(self):
        rng=np.random.default_rng(7);X=rng.normal(size=(24,8));y=np.repeat(["A","B","C"],8)
        scores=np.eye(3)[np.repeat([0,1,2],8)]
        context=build_latent_context(X,scores,y,np.unique(y));before=joblib.hash(context)
        cache=LatentTransformCache(context);q=rng.normal(size=(5,8))
        a=cache.project(q);self.assertIs(a,cache.project(q))
        b=cache.source_distances(q);self.assertIs(b,cache.source_distances(q))
        assert_allclose(b,context.source_distances(q));assert_allclose(cache.design(q)[:,:-1],a)
        self.assertEqual(before,joblib.hash(context))

    def test_prediction_boundary_ignores_query_truth_and_query_batch_has_no_fit(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(30,8));y=np.repeat(["A","B","C"],10)
        scores=np.eye(3)[np.repeat([0,1,2],10)];context=build_latent_context(X,scores,y,np.unique(y))
        before=joblib.hash(context);indices=np.array([0,10,20]);query=np.array([1,2,11,12,21,22])
        for candidate in [latent_candidate_grid()[1],latent_candidate_grid()[28],latent_candidate_grid()[-1]]:
            pred,_=prediction_only(context,candidate,X,scores,y,X[query],scores[query],indices)
            single,_=prediction_only(context,candidate,X,scores,y,X[query[:1]],scores[query[:1]],indices)
            assert_array_equal(single,pred[:1])
            first=fast_metrics(np.array([0,0,1,1,2,2]),pred,3)
            changed=fast_metrics(np.array([2,2,0,0,1,1]),pred,3)
            self.assertNotEqual(first,changed)
        self.assertEqual(before,joblib.hash(context))

    def test_source_cache_cohort_and_missing_stage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,sources,_,ip,pp,pr=fixture(root)
            with self.assertRaisesRegex(ValueError,"complete"):
                validate_source_run(root/"missing",ip,pp,pr)
            with patch("herb_spectra.spectral_source_search.source_candidate_grid",return_value=sources):
                run_source_search(ip,pp,pr,root/"source")
            lock,_,_=validate_source_run(root/"source",ip,pp,pr)
            self.assertFalse(lock["final_query_scored"])
            changed=root/"other-plan.json";changed.write_text(pp.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"cohort"):
                validate_source_run(root/"source",ip,changed,pr)

    def test_episode_uses_source_model_context_only_and_dedicated_support(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(18,21));y=np.repeat(["A","B","C"],6)
        training={"X_alpha":X,"y":y,"axis":np.linspace(.1,1.5,21),
                  "record_ids":np.asarray([f"id{i}" for i in range(18)])}
        partition={"source":np.array([0,1,2,6,7,8,12,13,14]),
                   "support_pool":np.array([3,9,15]),"query":np.array([4,5,10,11,16,17])}
        candidate=source_reference(source_candidate_grid())
        model=fit_source_candidate(candidate,X[partition["source"]],y[partition["source"]],training["axis"])
        context_fit=build_latent_context
        config={"model_seed":42,"support_draws":2,"support_seed":7,"selection_episodes":4}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_path=root/"model.joblib";joblib.dump(model,source_path)
            with patch("herb_spectra.latent_calibration_search.load_source_model",return_value=model),\
                 patch("herb_spectra.latent_calibration_search.source_model_path",return_value=source_path),\
                 patch("herb_spectra.latent_calibration_search.build_latent_context",wraps=context_fit) as fit:
                result=evaluate_latent_episode(training,partition,candidate,root,[latent_candidate_grid()[0]],
                                               config,0,root/"scores.csv.gz")
            assert_allclose(fit.call_args.args[0],model.transform(X[partition["source"]]))
            self.assertEqual(fit.call_args.args[2].tolist(),y[partition["source"]].tolist())
            self.assertEqual(result["new_source_fit_calls"],0)
            self.assertEqual(result["source_fingerprint_before"],result["source_fingerprint_after"])
            self.assertEqual(result["context_fingerprint_before"],result["context_fingerprint_after"])

    def test_complete_synthetic_run_resume_without_fit_and_frozen_query_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arrays,plan,_,sources,latent,ip,pp,pr=fixture(root)
            with patch("herb_spectra.spectral_source_search.source_candidate_grid",return_value=sources):
                run_source_search(ip,pp,pr,root/"source")
            with patch("herb_spectra.latent_calibration_search.latent_candidate_grid",return_value=latent):
                first=run_latent_search(ip,pp,pr,root/"source",root/"latent")
                with patch("herb_spectra.latent_calibration_search.fit_source_candidate",side_effect=AssertionError("refit")):
                    second=run_latent_search(ip,pp,pr,root/"source",root/"latent",resume=True)
                self.assertEqual(first,second)
            status=json.loads((root/"latent/status.json").read_text())
            self.assertEqual(status["new_development_source_fits"],0)
            self.assertEqual(status["sensitivity_source_fits"],0)
            query_ids={r["record_id"] for r in plan["records"] if r["role"]=="prediction"}
            for file in (root/"latent/final").glob("*/predictions.csv.gz"):
                with gzip.open(file,"rt",encoding="utf-8") as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual({r["record_id"] for r in rows},query_ids)
                self.assertTrue(all(int(r["is_correct"])==int(r["y_true"]==r["y_predicted"]) for r in rows))
            changed=json.loads(pr.read_text());changed["calibration_search"]["support_draws"]=3
            pr.write_text(json.dumps(changed))
            with patch("herb_spectra.latent_calibration_search.latent_candidate_grid",return_value=latent):
                with self.assertRaisesRegex(ValueError,"protocol"):
                    run_latent_search(ip,pp,pr,root/"source",root/"latent",resume=True)


if __name__=="__main__":
    unittest.main()
