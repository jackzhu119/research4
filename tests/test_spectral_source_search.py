"""Source-search protocol tests on synthetic spectra, not herb results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from numpy.testing import assert_allclose

from herb_spectra.classification import build_pipeline, class_scores
from herb_spectra.spectral_models import fit_source_candidate, source_candidate_grid
from herb_spectra.spectral_source_search import (aggregate_source_results, evaluate_source_episode,
    load_source_model, object_digest, run_source_search, select_sources, source_metadata_path,
    source_model_path, source_reference)


def synthetic_input(directory):
    rng = np.random.default_rng(11)
    records, labels, roles = [], [], []
    for c, counts in zip(["A", "B", "C"], [(12,4,4),(3,1,1),(12,4,4)]):
        for role, count in zip(["training", "calibration", "prediction"], counts):
            for i in range(count):
                identifier = f"{c}-{role}-{i:02d}"
                records.append(identifier);labels.append(c);roles.append(role)
    X = rng.normal(size=(len(records),21))
    X += np.asarray([{"A":0,"B":1,"C":2}[v] for v in labels])[:,None]*np.linspace(-1,1,21)
    arrays = {"X_alpha":X,"X_n":X/3+2,"axis":np.linspace(.1,1.5,21),
              "record_ids":np.asarray(records),"groups":np.asarray(records),
              "y":np.asarray(labels),"batch_ids":np.repeat("",len(records)),
              "labels_verified":np.ones(len(records),dtype=bool),"dataset_version":np.asarray("synthetic-v1")}
    plan = {"dataset_version":"synthetic-v1", "split_seed":17,
            "records":[{"record_id":identifier,"physical_sample_id":identifier,
                        "origin_group_id":label,"role":role}
                       for identifier,label,role in zip(records,labels,roles)]}
    grid = source_candidate_grid()
    reference = source_reference(grid)
    candidates = [reference, next(c for c in grid if c["preprocessing"]=="snv" and c["model"]=="plsda"),
                  next(c for c in grid if c["preprocessing"]=="snv" and c["model"]=="ridge")]
    protocol = {"protocol_id":"synthetic-source-v1","dataset_version":"synthetic-v1",
                "frequency":{"lower_inclusive":.1,"upper_inclusive":1.5,"expected_points":21},
                "fixed_role_counts":{"training":27,"calibration":9,"prediction":9},
                "candidates":{"source_count":len(candidates),"source_grid_sha256":object_digest(candidates)},
                "source_search":{"episodes":6,"selection_episodes":4,"split_seed":18,"model_seed":42,
                    "minimum_accuracy_gain":.001,"max_macro_f1_loss":.005,
                    "max_balanced_accuracy_loss":.005,"tie_tolerance":.001}}
    input_path = directory/"data.npz";np.savez_compressed(input_path,**arrays)
    plan_path=directory/"plan.json";plan_path.write_text(json.dumps(plan))
    protocol_path=directory/"protocol.json";protocol_path.write_text(json.dumps(protocol))
    return arrays,plan,protocol,candidates,input_path,plan_path,protocol_path


class SpectralSourceSearchTests(unittest.TestCase):
    def test_reference_source_matches_existing_pipeline_class_scores(self):
        rng=np.random.default_rng(5);X=rng.normal(size=(30,21));y=np.repeat(["A","B","C"],10)
        c=source_reference(source_candidate_grid())
        new=fit_source_candidate(c,X,y,np.linspace(.1,1.5,21))
        old=build_pipeline(block_sizes=(21,),random_state=42)
        old.set_params(preprocess__mode="snv",model__C=10.,model__break_ties=True)
        old.fit(X,y)
        query=rng.normal(size=(8,21))
        assert_allclose(new.predict_scores(query),class_scores(old,query),atol=1e-12,rtol=1e-12)

    def test_source_fit_only_receives_source_rows_and_prediction_is_frozen(self):
        rng=np.random.default_rng(8);X=rng.normal(size=(15,21));y=np.repeat(["A","B","C"],5)
        training={"X_alpha":X,"y":y,"axis":np.linspace(.1,1.5,21),
                  "record_ids":np.asarray([f"r{i}" for i in range(15)])}
        partition={"source":np.array([0,1,5,6,10,11]),"support_pool":np.array([2,7,12]),
                   "query":np.array([3,4,8,9,13,14])}
        c=source_reference(source_candidate_grid())
        with tempfile.TemporaryDirectory() as tmp:
            with patch("herb_spectra.spectral_source_search.fit_source_candidate",wraps=fit_source_candidate) as fit:
                meta=evaluate_source_episode(training,partition,c,Path(tmp),0,4,42)
                assert_allclose(fit.call_args.args[1],X[partition["source"]])
                self.assertEqual(fit.call_args.args[2].tolist(),y[partition["source"]].tolist())
            self.assertEqual(meta["status"],"pass")
            self.assertFalse(meta["final_query_scored"])
            self.assertEqual(meta["source_fingerprint_before"],meta["source_fingerprint_after"])
            cached=load_source_model(tmp,c["id"],0)
            self.assertEqual(joblib.hash(cached),meta["source_fingerprint_before"])
            with source_model_path(tmp,c["id"],0).open("ab") as f:f.write(b"changed")
            with self.assertRaisesRegex(ValueError,"changed"):
                load_source_model(tmp,c["id"],0)

    def test_source_selection_gate_does_not_try_second_best_after_failure(self):
        grid=source_candidate_grid();ref=source_reference(grid)
        other=[c for c in grid if c["preprocessing"]=="snv" and c["model"] in {"plsda","ridge"}][:2]
        candidates=[ref,*other]
        base={"accuracy":.80,"macro_f1":.80,"balanced_accuracy":.80,"failures":0,"evaluations":4}
        aggregates={c["id"]:{"selection":deepcopy(base),"gate":deepcopy(base)} for c in candidates}
        aggregates[other[0]["id"]]["selection"].update(accuracy=.90,macro_f1=.9,balanced_accuracy=.9)
        aggregates[other[0]["id"]]["gate"].update(accuracy=.79)
        aggregates[other[1]["id"]]["selection"].update(accuracy=.85,macro_f1=.85,balanced_accuracy=.85)
        aggregates[other[1]["id"]]["gate"].update(accuracy=.87,macro_f1=.87,balanced_accuracy=.87)
        choice=select_sources(aggregates,candidates)
        self.assertEqual(choice["proposed"]["id"],other[0]["id"])
        self.assertEqual(choice["selected"]["id"],ref["id"])
        self.assertTrue(choice["fallback_to_reference"])

    def test_macro_safeguard_rejects_accuracy_only_source_gain(self):
        grid=source_candidate_grid();ref=source_reference(grid)
        pls=next(c for c in grid if c["model"]=="plsda")
        base={"accuracy":.8,"macro_f1":.8,"balanced_accuracy":.8,"failures":0,"evaluations":4}
        aggregate={c["id"]:{s:deepcopy(base) for s in ["selection","gate"]} for c in [ref,pls]}
        for stage in aggregate[pls["id"]]:aggregate[pls["id"]][stage].update(accuracy=.9,macro_f1=.7)
        self.assertEqual(select_sources(aggregate,[ref,pls])["selected"]["id"],ref["id"])

    def test_end_to_end_checkpoint_resume_and_query_feature_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arrays,plan,protocol,candidates,ip,pp,pr=synthetic_input(root)
            with patch("herb_spectra.spectral_source_search.source_candidate_grid",return_value=candidates):
                first=run_source_search(ip,pp,pr,root/"first")
                with patch("herb_spectra.spectral_source_search.fit_source_candidate",side_effect=AssertionError("refit")):
                    replay=run_source_search(ip,pp,pr,root/"first",resume=True)
                self.assertEqual(first,replay)
                changed={k:v.copy() for k,v in arrays.items()}
                q=[i for i,r in enumerate(plan["records"]) if r["role"]=="prediction"]
                changed["X_alpha"][q]+=999
                np.savez_compressed(root/"changed.npz",**changed)
                second=run_source_search(root/"changed.npz",pp,pr,root/"second")
            for key in ["reference","proposed","selected","family_best","candidates_for_adaptation"]:
                self.assertEqual(first[key],second[key])
            self.assertEqual(first["source_fits_recorded"],18)
            self.assertFalse(first["final_query_scored"])
            with patch("herb_spectra.spectral_source_search.source_candidate_grid",return_value=candidates):
                pp.write_text(pp.read_text()+" ")
                with self.assertRaisesRegex(ValueError,"Resume rejected"):
                    run_source_search(ip,pp,pr,root/"first",resume=True)

    def test_frozen_grid_protocol_rejects_changed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,candidates,ip,pp,pr=synthetic_input(root)
            changed=deepcopy(candidates);changed[0]["C"]=999
            with patch("herb_spectra.spectral_source_search.source_candidate_grid",return_value=changed):
                with self.assertRaisesRegex(ValueError,"grid"):
                    run_source_search(ip,pp,pr,root/"run")

    def test_outer_prediction_truth_changes_neither_development_scores_nor_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arrays,plan,_,candidates,ip,pp,pr=synthetic_input(root)
            with patch("herb_spectra.spectral_source_search.source_candidate_grid",return_value=candidates):
                first=run_source_search(ip,pp,pr,root/"first")
                changed={k:v.copy() for k,v in arrays.items()}; changed_plan=deepcopy(plan)
                query=[i for i,r in enumerate(plan["records"]) if r["role"]=="prediction"]
                changed["y"][query]=changed["y"][query][::-1]
                for i in query: changed_plan["records"][i]["origin_group_id"]=str(changed["y"][i])
                altered_input=root/"altered.npz";np.savez_compressed(altered_input,**changed)
                altered_plan=root/"altered_plan.json";altered_plan.write_text(json.dumps(changed_plan))
                second=run_source_search(altered_input,altered_plan,pr,root/"second")
            for name in ["reference","proposed","selected","family_best","candidates_for_adaptation"]:
                self.assertEqual(first[name],second[name])
            a=json.loads((root/"first/source_development_aggregate.json").read_text())
            b=json.loads((root/"second/source_development_aggregate.json").read_text())
            self.assertEqual(a,b)


if __name__ == "__main__":
    unittest.main()
