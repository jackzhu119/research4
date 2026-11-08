"""Synthetic fit-free semantic checks; no real models, inference, or SVC fits."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_source_shift_augmentation_20261107.py'
spec = importlib.util.spec_from_file_location('source_shift_augmentation', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class SourceAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.axis=np.array([.1,.2,.3,.4]);self.mask=np.array([True,True,True,False])
        self.raw=np.tile(np.array([1.,2.,3.,4.]),(684,1))
        self.source=[f's{i:04d}' for i in range(684)];self.query=[f'q{i:04d}' for i in range(171)]
        self.y=np.array(['ZY01']*684)

    def fake_model(self):
        classes=np.array([f'ZY{i:02d}' for i in range(1,8)])
        weights={c:float(i+1)/4 for i,c in enumerate(classes)}
        return SimpleNamespace(classes_=classes,metadata={'training_class_weights':weights,'effective_gamma':.012345},
            estimator=SimpleNamespace(C=1000.,_gamma=.012345,class_weight_=np.array(list(weights.values()))))

    def test_exact_two_recipes_six_unranked_cases(self):
        self.assertEqual(m.RECIPES,(('half_median_bin',.5),('one_median_bin',1.)))
        cases=m.evaluation_specs();self.assertEqual(len(cases),6)
        self.assertEqual([s['delta_THz'] for s in cases],[0.,-.002,.002]*2)
        self.assertEqual([s['baseline_case'] for s in cases],['baseline','axis_shift_-0.002_THz','axis_shift_0.002_THz']*2)

    def test_raw_shift_before_mask_and_each_physical_weight_one(self):
        x,y,w,prov=m.augmented_source(self.raw,self.axis,self.mask,.05,self.source,self.y)
        self.assertEqual(x.shape,(2052,3));self.assertEqual(y.shape,(2052,))
        np.testing.assert_allclose(x[0],[1.,2.,3.]);np.testing.assert_allclose(x[684],[1.5,2.5,3.5]);np.testing.assert_allclose(x[1368],[1.,1.5,2.5])
        self.assertTrue(x.flags.c_contiguous)
        self.assertEqual(sum(p['physical_id']==self.source[0] for p in prov),3)
        m.assert_fit_scope(self.source,self.query,prov,y,self.y,w)
        np.testing.assert_array_equal(w,np.full(2052,1/3))

    def test_source_query_identity_overlap_fails(self):
        _,y,w,prov=m.augmented_source(self.raw,self.axis,self.mask,.01,self.source,self.y)
        with self.assertRaises(m.Blocked):m.assert_fit_scope(self.source,[self.source[0]]+self.query[1:],prov,y,self.y,w)

    def test_wrong_variant_weight_or_label_fails(self):
        _,y,w,prov=m.augmented_source(self.raw,self.axis,self.mask,.01,self.source,self.y)
        for mode in ['weight','label','variant','identity']:
            pp=[dict(p) for p in prov]; yy=y.copy();ww=w.copy()
            if mode=='weight':ww[0]=1.
            if mode=='label':yy[0]='ZY02'
            if mode=='variant':pp[0]['variant']=1
            if mode=='identity':pp[0]['physical_id']=self.query[0]
            with self.assertRaises(m.Blocked):m.assert_fit_scope(self.source,self.query,pp,yy,self.y,ww)

    def test_fixed_numeric_gamma_C_and_original_explicit_weights(self):
        model=self.fake_model();p=m.frozen_parameters(model)
        self.assertEqual(p['gamma'],.012345);self.assertEqual(p['C'],1000.)
        self.assertEqual(p['class_weight'],model.metadata['training_class_weights'])
        self.assertIsInstance(p['class_weight'],dict)
        self.assertFalse(p['break_ties']);self.assertEqual(p['decision_function_shape'],'ovr')
        model.estimator._gamma=.2
        with self.assertRaises(m.Blocked):m.frozen_parameters(model)
        model=self.fake_model();model.estimator.class_weight_[0]+=1
        with self.assertRaises(m.Blocked):m.frozen_parameters(model)

    def test_fit_spy_receives_only_fixed_parameters_and_one_third_weights(self):
        class SpySVC:
            def __init__(self,**params):self.params=params
            def fit(self,x,y,sample_weight):
                self.seen=(x.copy(),y.copy(),sample_weight.copy());return self
        processed=np.ones((2052,3));y=np.tile(self.y,3);w=np.full(2052,1/3)
        from herb_spectra.leakage_guard import FitScope
        variant_ids=[f'dependent-row-{i}' for i in range(2052)]
        scope=FitScope(variant_ids,'synthetic-source-only','synthetic-dependent-variants',code_version='test')
        est=m.fit_new_svc(self.fake_model(),processed,y,w,scope,variant_ids,svc_type=SpySVC)
        self.assertEqual(est.params['gamma'],.012345);self.assertEqual(est.params['C'],1000.)
        np.testing.assert_array_equal(est.seen[2],w)
        np.testing.assert_array_equal(est.seen[1],y)
        self.assertEqual([e['event'] for e in scope.entries],['fit_start','fit_completed'])
        self.assertEqual(len(est._leakage_fit_provenance['fit_sample_ids']),2052)

    def test_per_fold_label_projection_excludes_each_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'candidate.npz';ids=np.array([f'i{i:04d}' for i in range(855)]);labels=np.array(['ZY01']*855)
            np.savez(path,groups=ids,y=labels)
            splits=[{'fold':fold,'test_sample_ids':ids[fold*171:(fold+1)*171].tolist(),
                'train_sample_ids':np.concatenate([ids[:fold*171],ids[(fold+1)*171:]]).tolist()} for fold in range(5)]
            packs=m.source_projection(path,splits)
            self.assertEqual(len(packs),5)
            for pack,split in zip(packs,splits):
                self.assertEqual(len(pack['source_y']),684);self.assertEqual(pack['source_ids'],split['train_sample_ids'])
                self.assertFalse(set(pack['source_ids']) & set(split['test_sample_ids']))

    def test_preprocessor_and_selector_fit_entries_rejected_without_fits(self):
        from herb_spectra import staged_nested_search as staged
        with m.prevent_preprocessor_fits():
            for obj in [staged.StandardScaler(),staged.ScatterPreprocessor('msc')]:
                with self.assertRaises(m.Blocked):obj.fit(np.ones((2,3)))
            with self.assertRaises(m.Blocked):staged.fit_transform_bundle(None,None,None,None,None,None)

    def test_scoring_incomplete_prediction_or_evidence_lock_blocks_before_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);plan={'evaluations':m.evaluation_specs()};protocol='a'*64
            expected={f"predictions/{s['recipe']}__{s['case']}.csv.gz":'b'*64 for s in plan['evaluations']}
            for pred in [{},expected]:
                m.r.write_json(output/'prediction_lock.json',{'status':'ALL_SIX_CASES_LOCKED_BEFORE_NEW_SCORING_TRUTH_ACCESS',
                    'protocol_sha256':protocol,'predictions_sha256':pred,'evidence_sha256':{},'fit_count':10})
                with patch.object(m.gzip,'open',side_effect=AssertionError('No truth/prediction open permitted')):
                    with self.assertRaises(m.Blocked):m.score(output,plan,protocol)

    def test_weight_ceiling_checked_in_memory_before_disk_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib'
            with self.assertRaises(m.Blocked):m.save_capped_weights({'synthetic':'fixture'},path,1)
            self.assertFalse(path.exists())
            m.save_capped_weights({'synthetic':'fixture'},path,10000)
            self.assertEqual(m.joblib.load(path),{'synthetic':'fixture'})


if __name__=='__main__':unittest.main()
