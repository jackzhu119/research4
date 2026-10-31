"""Boundary tests use synthetic arrays only; no scientific score is reported."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from herb_spectra.leakage_guard import LeakageError, audit_pointwise_inference
from herb_spectra.selection import MCUVESelector
from herb_spectra.upgrade_neural import (CLASSES, FAMILIES, PROFILES, MODALITIES,
    UpgradeNet, fit_source, fixed_blocks, row_snv, save_fitted, load_fitted,
    read_pack, score_predictions, state_hash, prepare)


class UpgradeNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng=np.random.default_rng(2101)
        self.X=self.rng.normal(size=(14,24))
        self.y=np.repeat(np.asarray(CLASSES),2)
        self.ids=np.asarray([f"synthetic-{i:02d}" for i in range(14)])
        self.profile={**PROFILES[0],"width":4,"epochs":2,"mc_runs":3,"mc_components":1,"mc_ratio":.25}

    def test_all_prespecified_families_are_label_free_bn_inductive(self):
        inputs={m:self.X[:3].astype(np.float32) for m in ("alpha","n","time","frequency")}
        inputs["mcuve"]=self.X[:3,:6].astype(np.float32)
        for family in (*FAMILIES,"mcuve_only","dual_alpha_mcuve_se"):
            torch.manual_seed(42)
            model=UpgradeNet(family,self.profile,{k:v.shape[1] for k,v in inputs.items()})
            report=audit_pointwise_inference(model,inputs,batch_sizes=(1,2),atol=1e-5,rtol=1e-5)
            self.assertEqual(report["samples_checked"],3)
            self.assertTrue(report["parameters_and_buffers_unchanged"])

    def test_fixed_blocks_do_not_use_other_samples(self):
        axis=np.arange(24,dtype=float)
        whole,out_axis,blocks=fixed_blocks(self.X,axis,5)
        alone,_,_=fixed_blocks(self.X[:1],axis,5)
        np.testing.assert_array_equal(whole[:1],alone)
        self.assertEqual(whole.shape,(14,5));self.assertEqual(len(blocks),5)
        self.assertTrue(np.all(np.diff(out_axis)>0))

    def test_inference_pack_rejects_truth_even_if_unused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"bad.npz";np.savez(p,X_alpha=self.X,record_ids=self.ids,query_labels=self.y)
            with self.assertRaisesRegex(LeakageError,"truth"):read_pack(p)

    def test_unknown_metadata_or_filename_feature_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"bad.npz"
            for extra in ({"disguised_truth":self.y},{"X_filename":self.ids[:,None]}):
                np.savez(p,X_alpha=self.X,record_ids=self.ids,**extra)
                with self.assertRaisesRegex(LeakageError,"unknown"):read_pack(p)
            np.savez(p,X_alpha=self.ids[:,None],record_ids=self.ids)
            with self.assertRaisesRegex(LeakageError,"numeric"):read_pack(p)

    def test_scoring_rejects_repeated_oof_denominator(self):
        rows=[{"record_id":"a","y_predicted":"ZY01"},{"record_id":"a","y_predicted":"ZY02"}]
        with self.assertRaisesRegex(LeakageError,"Duplicate"):score_predictions(rows,{"a":"ZY01"})

    def test_source_fit_internal_pls_matches_original_and_logs_each_fit(self):
        with tempfile.TemporaryDirectory() as t:
            log=Path(t)/"fits.jsonl"
            fitted=fit_source({"alpha":self.X},self.y,self.ids,"dual_alpha_mcuve",self.profile,42,"synthetic.inner",log,"test-code")
            z=fitted.scalers["alpha"].transform(row_snv(self.X))
            ordinary=MCUVESelector(n_runs=3,n_components=1,n_features_to_select=6,
                sample_fraction=.8,task="classification",random_state=42).fit(z,self.y,groups=self.ids)
            np.testing.assert_array_equal(fitted.selector.selected_indices,ordinary.selected_indices_)
            np.testing.assert_array_equal(fitted.selector.coefficients,ordinary.coefficients_)
            entries=[json.loads(s) for s in log.read_text().splitlines()]
            self.assertTrue(all(set(r.get("fit_sample_ids",[]))<=set(self.ids) for r in entries))
            internal=[r for r in entries if r["event"]=="fit_completed" and "pls" in r.get("component","").lower()]
            self.assertEqual(len(internal),21)
            # Nonadjacent MC variables are consumed by Linear layers, not a CNN.
            self.assertFalse(any(isinstance(m,torch.nn.Conv1d) for m in fitted.model.mc.modules()))

    def test_changed_query_or_truth_cannot_mutate_source_artifact(self):
        with tempfile.TemporaryDirectory() as t:
            source={"alpha":self.X}
            fitted=fit_source(source,self.y,self.ids,"cnn_alpha",self.profile,42,"synthetic.inner",Path(t)/"fits.jsonl","test-code")
            before=state_hash(fitted.model);scaler=fitted.scalers["alpha"].mean_.copy()
            scores=fitted.predict_scores({"alpha":self.X[:7]})
            changed=self.X[:7].copy();changed[0]=changed[0][::-1]*3
            fitted.predict_scores({"alpha":changed})
            rows=[{"record_id":str(i),"y_predicted":CLASSES[int(c)]} for i,c in enumerate(scores.argmax(1))]
            score_predictions(rows,{str(i):CLASSES[i] for i in range(7)})
            score_predictions(rows,{str(i):CLASSES[(i+1)%7] for i in range(7)})
            self.assertEqual(before,state_hash(fitted.model));np.testing.assert_array_equal(scaler,fitted.scalers["alpha"].mean_)
            # Refit from the same source and seed reproduces the fitted state.
            again=fit_source(source,self.y,self.ids,"cnn_alpha",self.profile,42,"synthetic.inner",Path(t)/"fits2.jsonl","test-code")
            self.assertEqual(before,state_hash(again.model))

    def test_saved_weight_cannot_be_loaded_for_another_fold(self):
        with tempfile.TemporaryDirectory() as t:
            fitted=fit_source({"alpha":self.X},self.y,self.ids,"cnn_alpha",self.profile,42,"synthetic.inner",Path(t)/"fits.jsonl","test-code")
            directory=Path(t)/"model";save_fitted(fitted,directory)
            restored=load_fitted(directory,expected_split_id="synthetic.inner",expected_source_ids=self.ids)
            np.testing.assert_array_equal(fitted.predict_scores({"alpha":self.X}),restored.predict_scores({"alpha":self.X}))
            with self.assertRaisesRegex(LeakageError,"Cross-fold"):load_fitted(directory,expected_split_id="different.inner")
            with self.assertRaisesRegex(LeakageError,"source"):load_fitted(directory,expected_source_ids=self.ids[::-1])

    def test_database_pairing_value_changes_fail_before_preparation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"protocol.json"
            data=Path(t)/"data.npz";np.savez(data,X_alpha=self.X)
            p.write_text(json.dumps({"input":str(data),"input_sha256":"wrong"}))
            with self.assertRaisesRegex(LeakageError,"pairing"):prepare(p,Path(t)/"run")


if __name__=="__main__":unittest.main()
