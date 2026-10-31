from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

from herb_spectra.paper_fewshot_classification import state_fingerprint
from herb_spectra.staged_nested_search import fit_source_candidate
from herb_spectra.staged_nested_combinations import (FAMILIES,SmallNestedCNN,
    _check_inner_roles,_check_release,choose_family,cnn_encode,cnn_logits,digest,
    fit_cached_caller,fit_cnn_source,fit_mcuve,fit_stable_selectors,inherited_config,
    numeric_fingerprint,pairwise_jaccard,ranked_topk,run_search,source_resamples)


class StagedNestedCombinationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(120)
        cls.X=rng.normal(size=(21,612));cls.y=np.repeat(np.asarray([f"C{i}" for i in range(7)]),3)
        cls.ids=np.asarray([f"fixture{i:03d}" for i in range(21)]);cls.axis=np.linspace(.1,1.5,612)
        cls.config={"preprocessing":"snv","mc":{"runs":2,"components":1,"ratio":.05},
            "svm":{"C":1.,"gamma":"scale","class_weight":"balanced"}}
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name)/"caller"
        cls.models,cls.common=fit_cached_caller(cls.directory,cls.X,cls.y,cls.axis,cls.ids,cls.config,cnn_epochs=2)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_each_outer_inherits_its_own_sequential_choices(self):
        a=deepcopy(self.config);b=deepcopy(self.config);b["mc"]["ratio"]=.2;b["preprocessing"]="sg1_snv"
        def context(c):return {"choices":{"R1":{"selected":{"mc":c["mc"]}},"R2":{"selected":{"svm":c["svm"]}},"R3":{"selected":c}}}
        self.assertEqual(inherited_config(context(a)),a);self.assertEqual(inherited_config(context(b)),b)
        c=context(a);c["choices"]["R3"]["selected"]["mc"]={"runs":99}
        with self.assertRaisesRegex(ValueError,"same outer"):inherited_config(c)

    def test_source_resamples_are_shared_reproducible_and_min_two(self):
        parts=source_resamples(self.y,self.ids);again=source_resamples(self.y,self.ids)
        self.assertEqual(len(parts),10)
        for a,b in zip(parts,again):
            np.testing.assert_array_equal(a["indices"],b["indices"])
            self.assertEqual(len(a["indices"]),len(set(a["indices"].tolist())))
            self.assertGreaterEqual(min(a["class_counts"].values()),2)
            self.assertEqual(a["source_ids"],self.ids[a["indices"]].tolist())
        with self.assertRaisesRegex(ValueError,"at least2"):
            source_resamples(np.asarray(["a","b","b"]),np.asarray(["s1","s2","s3"]))

    def test_frequency_topk_ties_and_zero_supplement(self):
        selected,meta=ranked_topk([.5,0,0,.5],[3,0,2,1],3)
        self.assertEqual(selected.tolist(),[0,2,3]);self.assertEqual(meta["zero_frequency_supplement_count"],1)

    def test_pairwise_jaccard_exact_values(self):
        result=pairwise_jaccard([[0,1],[1,2],[0,1]])
        self.assertEqual(len(result["pairs"]),3)
        self.assertAlmostEqual(result["mean"],(1/3+1+1/3)/3)

    def test_stable_frequencies_and_fit_budget_are_recorded(self):
        stable=self.common["stability"]
        self.assertEqual(len(stable["source_resamples"]),10)
        for name in ("stable_mcuve","stable_cars","stable_joint"):
            self.assertEqual(len(stable["selected_indices"][name]),31)
            self.assertEqual(len(stable["selection_frequencies"][name]),612)
        a=np.asarray(stable["selection_frequencies"]["stable_mcuve"]);b=np.asarray(stable["selection_frequencies"]["stable_cars"])
        np.testing.assert_allclose(stable["selection_frequencies"]["stable_joint"],.5*a+.5*b)
        budget=stable["fit_budget"]
        self.assertEqual(budget["budget_mcuve_coefficient_rounds"],20)
        self.assertEqual(budget["stable_mcuve_coefficient_rounds"],20)
        self.assertEqual(budget["tie_breaker_extra_pls_fits"],0)
        self.assertEqual(budget["stable_cars_multivariate_internal_cv_pls_fits"],240)

    def test_ordinary_model_exactly_matches_core_R3_rule(self):
        core=fit_source_candidate({"id":"fixture",**self.config},self.X,self.y,self.axis,42,{})
        model=self.models["ordinary_mcuve"]
        np.testing.assert_array_equal(core.predict(self.X),model.predict(self.X))
        np.testing.assert_allclose(core.predict_scores(self.X),model.predict_scores(self.X),rtol=0,atol=1e-12)
        self.assertEqual(model.metadata["prediction_rule"],"SVC.predict standard vote, break_ties=False")

    def test_CNN_architecture_source_budget_and_shared_encoder(self):
        model=self.models["cnn_softmax"];control=self.models["cnn_embedding_svm"]
        self.assertEqual(sum(p.numel() for p in model.network.parameters()),9223)
        self.assertEqual(state_fingerprint(model.network),state_fingerprint(control.network))
        self.assertEqual(model.metadata["selected_features"],612)
        self.assertEqual(control.metadata["classifier_features"],32)
        self.assertFalse(model.metadata["feature_selection_applied"])
        self.assertEqual(model.metadata["neural_training"]["epochs"],2)
        self.assertEqual(len(model.metadata["neural_training"]["losses"]),2)
        self.assertFalse(any(isinstance(layer,torch.nn.BatchNorm1d) for layer in model.network.modules()))

    def test_CNN_predictions_pointwise_and_source_state_frozen(self):
        model=self.models["cnn_softmax"];before=numeric_fingerprint(model)
        individual=model.predict_scores(self.X[:1]);group=model.predict_scores(self.X[:7])[:1]
        np.testing.assert_allclose(individual,group,rtol=1e-5,atol=1e-6)
        self.assertEqual(before,numeric_fingerprint(model))
        other=self.models["cnn_embedding_svm"];before=numeric_fingerprint(other)
        other.predict(self.X[:3]);self.assertEqual(before,numeric_fingerprint(other))

    def test_checkpoint_all_families_resume_without_refitting(self):
        before={str(p):digest(p) for p in self.directory.rglob('*') if p.is_file()}
        with patch('herb_spectra.staged_nested_combinations.fit_caller_models',side_effect=AssertionError('no fit')):
            restored,common=fit_cached_caller(self.directory,self.X,self.y,self.axis,self.ids,self.config,cnn_epochs=2,resume=True)
        for family in FAMILIES:
            self.assertEqual(numeric_fingerprint(restored[family]),numeric_fingerprint(self.models[family]))
            np.testing.assert_array_equal(restored[family].predict(self.X),self.models[family].predict(self.X))
        self.assertEqual(before,{str(p):digest(p) for p in self.directory.rglob('*') if p.is_file()})

    def test_inner_role_partition_rejects_any_outer_test_use(self):
        context={'train_indices':np.arange(8),'test_indices':np.arange(8,10)}
        _check_inner_roles(context,np.arange(4),np.arange(4,8))
        with self.assertRaisesRegex(ValueError,'isolation'):_check_inner_roles(context,np.arange(4),np.asarray([4,5,6,8]))

    def test_family_selection_is_macro_first_then_simple(self):
        rows=[{'family':f,'metrics':{'macro_f1':.7,'balanced_accuracy':.8,'accuracy':.9}} for f in FAMILIES for _ in range(2)]
        self.assertEqual(choose_family(rows)['selected']['family'],'ordinary_mcuve')
        for row in rows:
            if row['family']=='cnn_embedding_svm':row['metrics']['macro_f1']=.8
        self.assertEqual(choose_family(rows)['selected']['family'],'cnn_embedding_svm')

    def test_global_outer_release_requires_exact_three_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'r4';out.mkdir();protocol=Path(td)/'p.json';protocol.write_text('{}')
            for name in ['selection_lock.json','run_manifest.json']:(out/name).write_text('{}')
            release=Path(td)/'release.json'
            record={'selection_lock':str(out/'selection_lock.json'),'selection_lock_sha256':digest(out/'selection_lock.json'),
                'run_manifest_sha256':digest(out/'run_manifest.json'),'protocol_sha256':digest(protocol)}
            release.write_text(json.dumps({'rounds':[record]}))
            self.assertEqual(_check_release(out,protocol,release),release)
            protocol.write_text('{"different":true}')
            with self.assertRaisesRegex(ValueError,'hashes changed'):_check_release(out,protocol,release)

    def test_full_selection_status_handles_real_metrics_without_total(self):
        # Synthetic orchestration fixture: no real data/model fit or outer scoring.
        rows=[{'outer_fold':o,'inner_fold':i,'stage':'R4','family':f,'candidate_id':f,'validation_samples':342,
               'metrics':{'macro_f1':.5,'balanced_accuracy':.5,'accuracy':.5}}
              for o in range(5) for i in range(2) for f in FAMILIES]
        contexts=[{'outer_fold':o,'choices':{'R1':{'selected':{'mc':self.config['mc']}},
            'R2':{'selected':{'svm':self.config['svm']}},'R3':{'selected':self.config}},
            'data':{},'inner_folds':[{},{}]} for o in range(5)]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';protocol=Path(td)/'protocol.json'
            protocol.write_text(json.dumps({'families':list(FAMILIES),'cnn':{'epochs':200},'model_seed':42}))
            manifest={'core_selection_lock_sha256':'fixture'}
            def inner(data,context,inherited,inner_no,fold,output,seed,resume=False):
                return [r for r in rows if r['outer_fold']==context['outer_fold'] and r['inner_fold']==inner_no]
            with patch('herb_spectra.staged_nested_combinations._manifest',return_value=manifest),\
                 patch('herb_spectra.staged_nested_combinations.load_selection_contexts',return_value=contexts),\
                 patch('herb_spectra.staged_nested_combinations._inner_evaluate',side_effect=inner):
                result=run_search(protocol,protocol,protocol,out,out)
            self.assertEqual(result['inner_evaluations'],70)
            self.assertEqual(result['inner_prediction_rows'],70*342)
            self.assertFalse(result['outer_test_scored'])


if __name__=='__main__':unittest.main()
