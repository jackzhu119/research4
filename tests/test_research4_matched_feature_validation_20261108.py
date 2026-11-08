"""Source isolation, matched K and resumable evidence tests using tiny data."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_feature_worker',ROOT/'scripts/research4_matched_feature_validation_20261108.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TransformTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(531)
        self.axis=np.linspace(.1,1.5,8);self.n=rng.normal(size=(16,8))+np.arange(8)
        self.a=rng.normal(size=(16,8))+np.arange(8)**2
        self.y=np.asarray(['a']*8+['b']*8);self.ids=np.asarray(['ID%02d'%i for i in range(16)])

    @staticmethod
    def ranking(z,y,runs,seed):
        return np.arange(z.shape[1]),np.arange(z.shape[1],0,-1,dtype=float)

    def fitted(self,family):
        with patch.object(m,'one_ranking',side_effect=self.ranking):
            return m.fit_transform(self.n,self.a,self.y,self.ids,self.axis,m.FAMILIES[family])

    def test_matched_absolute_k_and_modality_annotations(self):
        for family in m.FAMILIES:
            t=self.fitted(family)
            for k in (2,5,8):
                z,indices,weights=m.selected_matrix(t,self.n,self.a,k,1.)
                self.assertEqual(z.shape,(16,k));self.assertEqual(len(set(indices)),k)
                self.assertTrue(np.all(weights==1))

    def test_all_declared_k_are_absolute_not_ratio(self):
        for width in (612,1224):
            t={'global_indices':np.arange(width),'ranks':[np.arange(width)],'spec':(None,'raw','mc20'),'axis':np.arange(612)}
            for k in m.K_VALUES:self.assertEqual(len(m.chosen_indices(t,k)),k)

    def test_odd_block_budget_is_exact_and_pays_two_streams(self):
        t=self.fitted('block_half_nraw_asnv20');z,ix,_=m.selected_matrix(t,self.n,self.a,5,1.)
        self.assertEqual(sum(ix<8),2);self.assertEqual(sum(ix>=8),3)
        self.assertEqual(t['metadata']['coefficient_draws'],40)
        self.assertEqual(t['metadata']['ovr_pls_fits'],80)

    def test_weight_is_applied_after_scaling(self):
        t=self.fitted('weighted_concat_nraw_asnv20')
        base,indices,_=m.selected_matrix(t,self.n,self.a,5,1.)
        half,half_indices,w=m.selected_matrix(t,self.n,self.a,5,.5)
        np.testing.assert_array_equal(indices,half_indices)
        np.testing.assert_allclose(half,base*.5)
        np.testing.assert_allclose(half.var(axis=0),np.full(5,.25))
        np.testing.assert_allclose(w,np.full(5,.5))

    def test_alpha_columns_never_receive_n_weight(self):
        t=self.fitted('weighted_concat_nraw_asnv20')
        base,indices,_=m.selected_matrix(t,self.n,self.a,12,1.)
        half,_,w=m.selected_matrix(t,self.n,self.a,12,.25)
        np.testing.assert_allclose(half[:,indices>=8],base[:,indices>=8])
        self.assertTrue(np.all(w[indices>=8]==1))

    def test_query_values_do_not_refit_scaler_or_ranking(self):
        t=self.fitted('concat_nraw_asnv20');mean=t['mean'].copy();rank=t['ranks'][0].copy()
        z,_,_=m.selected_matrix(t,self.n[:3],self.a[:3],5,1.)
        m.selected_matrix(t,self.n[:3]+1e9,self.a[:3]*-30,5,.25)
        again,_,_=m.selected_matrix(t,self.n[:3],self.a[:3],5,1.)
        np.testing.assert_array_equal(t['mean'],mean);np.testing.assert_array_equal(t['ranks'][0],rank)
        np.testing.assert_array_equal(z,again)

    def test_selector_has_no_query_or_test_fit_argument(self):
        with self.assertRaises(TypeError):
            m.fit_transform(self.n,self.a,self.y,self.ids,self.axis,m.FAMILIES['alpha_raw20'],query=self.a)

    def test_fit_id_duplicates_rejected(self):
        bad=self.ids.copy();bad[1]=bad[0]
        with self.assertRaises(ValueError):m.fit_transform(self.n,self.a,self.y,bad,self.axis,m.FAMILIES['alpha_raw20'])

    def test_rank_vote_and_mc100_have_equal_draw_budget(self):
        calls=[]
        def ranking(z,y,runs,seed):calls.append((runs,seed));return self.ranking(z,y,runs,seed)
        with patch.object(m,'one_ranking',side_effect=ranking):
            stable=m.fit_transform(self.n,self.a,self.y,self.ids,self.axis,m.FAMILIES['alpha_snv_stability5x20'])
        self.assertEqual(calls,[(20,42+1009*i)for i in range(5)])
        single=self.fitted('alpha_snv100')
        self.assertEqual(stable['metadata']['coefficient_draws'],single['metadata']['coefficient_draws'])
        self.assertEqual(stable['metadata']['ovr_pls_fits'],single['metadata']['ovr_pls_fits'])

    def test_tiny_actual_pls_ranking_invariant_to_class_name_permutation(self):
        rng=np.random.default_rng(72);x=rng.normal(size=(24,12));y=np.asarray(['a']*8+['b']*8+['c']*8)
        r1,i1=m.one_ranking(x,y,20,42)
        changed=np.asarray([{'a':'z','b':'x','c':'y'}[v]for v in y]);r2,i2=m.one_ranking(x,changed,20,42)
        np.testing.assert_allclose(i1,i2,rtol=1e-9,atol=1e-10);np.testing.assert_array_equal(r1,r2)

    def test_cache_is_source_identity_specific_and_sealed(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(m,'one_ranking',side_effect=self.ranking)as fitted:
            one=m.transform_cache(tmp,'job',self.n,self.a,self.y,self.ids,self.axis,m.FAMILIES['alpha_raw20'])
            again=m.transform_cache(tmp,'job',self.n,self.a,self.y,self.ids,self.axis,m.FAMILIES['alpha_raw20'])
            self.assertEqual(fitted.call_count,1);self.assertEqual(one['cache_path'],again['cache_path'])
            other_ids=np.asarray(['OTHER'+v for v in self.ids])
            other=m.transform_cache(tmp,'job',self.n,self.a,self.y,other_ids,self.axis,m.FAMILIES['alpha_raw20'])
            self.assertNotEqual(one['cache_path'],other['cache_path']);self.assertEqual(fitted.call_count,2)
            p=Path(one['cache_path'])/'metadata.json';p.write_text('{}')
            with self.assertRaises(ValueError):m.transform_cache(tmp,'job',self.n,self.a,self.y,self.ids,self.axis,m.FAMILIES['alpha_raw20'])


class SelectionAndEvidenceTests(unittest.TestCase):
    def test_K_selected_on_inner_scores_only(self):
        rows=[]
        for k,score,outer in [(31,.82,.3),(62,.79,.99)]:
            for fold in (0,1):
                rows.append({'config':{'k':k,'C':1.,'gamma':'scale','weight':1.},'inner_fold':fold,
                             'metrics':{'macro_f1':score,'balanced_accuracy':score,'accuracy':score},'outer_accuracy':outer})
        self.assertEqual(m.select_inner(rows)['config']['k'],31)
        for row in rows:row['outer_accuracy']=1-row['outer_accuracy']
        self.assertEqual(m.select_inner(rows)['config']['k'],31)

    def test_missing_inner_fold_rejected(self):
        with self.assertRaises(ValueError):m.select_inner([{'config':{'k':31},'inner_fold':0,'metrics':{}}])

    def test_primary_tie_prefers_smaller_k_and_declared_unweighted_control(self):
        rows=[]
        for k in (62,31):
            for weight in (.25,1.):
                for fold in(0,1):rows.append({'config':{'k':k,'C':1.,'gamma':'scale','weight':weight},'inner_fold':fold,
                    'metrics':{'macro_f1':.7,'balanced_accuracy':.7,'accuracy':.7}})
        self.assertEqual(m.select_inner(rows)['config'],{'k':31,'C':1.,'gamma':'scale','weight':1.})

    def test_failed_cost_preserved_checkpoint_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'stage.json'
            with self.assertRaises(RuntimeError):m.checkpoint(path,'id',lambda:(_ for _ in()).throw(RuntimeError('toy failure')))
            failed=list(Path(tmp).glob('*.attempt-*'));self.assertEqual(len(failed),1)
            self.assertEqual(m.read(failed[0])['status'],'FAILED_OR_INTERRUPTED')
            self.assertGreaterEqual(m.read(failed[0])['actual_cpu_seconds'],0.)
            first=m.checkpoint(path,'id',lambda:{'answer':42})
            second=m.checkpoint(path,'id',lambda:(_ for _ in()).throw(AssertionError('must not rerun')))
            self.assertEqual(first,second);self.assertTrue(failed[0].exists())
            path.write_text('{}')
            with self.assertRaises(ValueError):m.checkpoint(path,'id',lambda:{})

    def test_interrupted_attempt_is_not_reported_done(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'stage.json'
            with self.assertRaises(KeyboardInterrupt):m.checkpoint(path,'id',lambda:(_ for _ in()).throw(KeyboardInterrupt('toy stop')))
            attempt=next(Path(tmp).glob('*.attempt-*'))
            self.assertEqual(m.read(attempt)['status'],'FAILED_OR_INTERRUPTED');self.assertFalse(path.exists())

    def test_complete_metrics_include_each_class_precision_recall_f1(self):
        result=m.metric(np.asarray(['a','a','b','b']),np.asarray(['a','a','a','a']),np.asarray(['a','b']))
        self.assertEqual(result['correct'],2);self.assertEqual(result['denominator'],4)
        self.assertEqual(result['per_class'][1]['recall'],0.)
        self.assertEqual(result['confusion_matrix'],[[2,0],[2,0]])

    def test_mlp_early_stop_is_explicit_and_fit_uses_only_source(self):
        received={}
        class ToyModel:
            n_iter_=3
            def fit(self,x,y):received['fit_x']=x.copy();received['fit_y']=y.copy();return self
            def predict(self,q):return np.repeat(received['fit_y'][0],len(q))
        def factory(**kwargs):received['kwargs']=kwargs;return ToyModel()
        rng=np.random.default_rng(12);train=rng.normal(size=(20,8));query=rng.normal(size=(7,8))+100
        with patch.object(m,'MLPClassifier',side_effect=factory):
            pred,bundle=m.fit_auxiliary({'family':'fullalpha_MLP','preprocessing':'raw','hidden':32},train,
                                      np.asarray(['a']*10+['b']*10),query,np.arange(8))
        self.assertEqual(received['fit_x'].shape,(20,8));self.assertEqual(len(pred),7)
        self.assertTrue(received['kwargs']['early_stopping']);self.assertEqual(received['kwargs']['validation_fraction'],.2)
        np.testing.assert_allclose(bundle['scaler'].mean_,train.mean(axis=0))


class IsolatedInputTests(unittest.TestCase):
    @staticmethod
    def metadata():
        rows=[]
        for group in ['ZY03','ZY04','ZY05','ZY06','ZY07']:
            for j in range(60):
                rid='%s-%03d'%(group,j);rows.append({'record_id':rid,'physical_sample_id':rid,'species':'酸枣仁','origin_group_id':group,'role':'train'})
        for kind,count in [('枣仁',216),('黄芪',241),('天麻',242)]:
            for j in range(count):
                rid='%s-%03d'%(kind,j);rows.append({'record_id':rid,'physical_sample_id':rid,'species':kind,'origin_group_id':'unneeded_aux_group','role':'train'})
        return rows

    def files(self,root,rows,with_data=False):
        split=root/'train.csv'
        with split.open('w',newline='',encoding='utf-8-sig')as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        features=root/'train.npz'
        if with_data:
            rng=np.random.default_rng(42);n=rng.normal(size=(999,250));a=rng.normal(size=(999,250))
            np.savez_compressed(features,record_ids=np.asarray([r['record_id']for r in rows]),
                                X_n=n,X_alpha=a,axis=np.linspace(.1,1.5,250))
            m.atomic(root/'input_lock.json',{'training_records':999,'training_feature_sha256':m.sha(features),
                     'training_split_sha256':m.sha(split),'training_allowlist':[r['record_id']for r in rows]})
        return features,split

    def test_nontrain_row_rejected_before_opening_features(self):
        with tempfile.TemporaryDirectory()as tmp:
            rows=self.metadata();rows[-1]['role']='prediction';features,split=self.files(Path(tmp),rows)
            with self.assertRaisesRegex(ValueError,'Only isolated999TRAIN'):m.load_inputs(features,split)

    def test_duplicate_physical_sample_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            rows=self.metadata();rows[1]['physical_sample_id']=rows[0]['physical_sample_id'];features,split=self.files(Path(tmp),rows)
            with self.assertRaisesRegex(ValueError,'Unique independently'):m.load_inputs(features,split)

    def test_only_pure_acid_rows_reach_model_and_lock_is_checked(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);rows=self.metadata();features,split=self.files(root,rows,with_data=True)
            n,a,y,ids,physical,axis=m.load_inputs(features,split)
            self.assertEqual(n.shape,(300,250));self.assertEqual(a.shape,n.shape)
            self.assertEqual(set(y),{'ZY03','ZY04','ZY05','ZY06','ZY07'})
            self.assertTrue(all(i.startswith('ZY')for i in ids));self.assertEqual(len(set(physical)),300)
            lock=m.read(root/'input_lock.json');lock['training_feature_sha256']='0'*64;m.atomic(root/'input_lock.json',lock)
            with self.assertRaisesRegex(ValueError,'TRAIN input lock'):m.load_inputs(features,split)


if __name__=='__main__':unittest.main()
