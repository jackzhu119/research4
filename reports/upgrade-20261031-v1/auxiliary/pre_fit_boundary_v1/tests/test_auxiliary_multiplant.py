from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from herb_spectra.auxiliary_multiplant import (FUSIONS,PROFILES,fit_candidate,grids,
    read_pack,save_fitted,load_fitted,metrics)
from herb_spectra.leakage_guard import LeakageError,audit_pointwise_inference
from herb_spectra.upgrade_neural import state_hash


class AuxiliaryMultiplantTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.X=np.random.default_rng(7).normal(size=(32,40))
        self.ids=np.asarray([f'unit-source-{i:02d}' for i in range(32)])
        self.labels=np.asarray([f'HQ{i%4+1:02d}' for i in range(32)])
    def tearDown(self):self.tmp.cleanup()
    def fit(self,candidate,**kwargs):
        return fit_candidate({'alpha':self.X},self.labels,self.ids,'huangqi',candidate,42,'unit-split',self.folder/'scope.jsonl','unit-code',expected_source_ids=kwargs.get('expected',self.ids))

    def test_actual_source_ids_must_match_independent_context(self):
        with self.assertRaisesRegex(LeakageError,'fixed allowed source'):
            self.fit(grids()['full_svm'][0],expected=[*self.ids[:-1],'unit-forbidden-query'])

    def test_two_plants_cannot_share_one_class_space(self):
        labels=self.labels.copy();labels[0]='TM01'
        with self.assertRaisesRegex(LeakageError,'separate plant label'):
            fit_candidate({'alpha':self.X},labels,self.ids,'huangqi',grids()['full_svm'][0],42,'unit-split',None,'unit',expected_source_ids=self.ids)

    def test_inference_pack_rejects_labels_and_metadata_features(self):
        p=self.folder/'pack.npz';np.savez(p,record_ids=self.ids,X_alpha=self.X,source_labels=self.labels)
        with self.assertRaisesRegex(LeakageError,'pack schema'):read_pack(p)
        np.savez(p,record_ids=self.ids,X_alpha=np.asarray([['filename.xlsx']]*32))
        with self.assertRaisesRegex(LeakageError,'paired spectra'):read_pack(p)

    def test_original_mcuve_pls_calls_are_source_scoped(self):
        fitted=self.fit(grids()['mcuve_svm'][0])
        lines=[json.loads(v) for v in (self.folder/'scope.jsonl').read_text().splitlines()]
        pls=[r for r in lines if r['event']=='fit_start' and r['component']=='MC-UVE.PLSRegression.fit']
        # Legacy classification MCUVE fits one PLS response per source class.
        self.assertEqual(len(pls),10*4)
        self.assertTrue(all(set(r['fit_sample_ids'])<=set(self.ids) for r in pls))
        self.assertEqual(len(fitted.selector.selected_indices_),4)

    def test_cached_weight_refuses_another_fold_or_source(self):
        fitted=self.fit(grids()['full_svm'][0]);p=self.folder/'pipeline.joblib';save_fitted(fitted,p)
        self.assertEqual(load_fitted(p,expected_split='unit-split',expected_ids=self.ids).classes,tuple(f'HQ{i:02d}' for i in range(1,5)))
        with self.assertRaisesRegex(LeakageError,'Cross-fold'):load_fitted(p,expected_split='other-fold')
        with self.assertRaisesRegex(LeakageError,'Cross-source'):load_fitted(p,expected_ids=self.ids[::-1])

    def test_four_class_nn_batchnorm_inference_is_pointwise_and_frozen(self):
        candidate={**PROFILES[0],'id':'unit-only','kind':'se','width':4,'blocks':1,'epochs':1}
        fitted=self.fit(candidate);before=state_hash(fitted.estimator)
        values=fitted.transform({'alpha':self.X[:7]})
        result=audit_pointwise_inference(fitted.estimator,values,batch_sizes=(1,3),atol=1e-5,rtol=1e-5)
        self.assertTrue(result['parameters_and_buffers_unchanged']);self.assertEqual(before,state_hash(fitted.estimator))
        self.assertEqual(fitted.predict_scores({'alpha':self.X[:7]}).shape,(7,4))

    def test_missing_modality_routes_are_explicit_not_silent_class_deletion(self):
        self.assertEqual(FUSIONS['huangqi'],('time','frequency'))
        self.assertEqual(FUSIONS['tianma'],('alpha','n'))
        self.assertEqual(len(grids()['full_svm']),4);self.assertEqual(len(grids()['mcuve_svm']),8)

    def test_scoring_rejects_repeated_sample_denominator(self):
        truth={'a':'HQ01','b':'HQ02'};rows=[{'record_id':'a','y_predicted':'HQ01'},{'record_id':'a','y_predicted':'HQ02'}]
        with self.assertRaisesRegex(LeakageError,'denominator'):metrics(rows,truth,('HQ01','HQ02','HQ03','HQ04'))


if __name__=='__main__':unittest.main()
