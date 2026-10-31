"""Synthetic invariance/selection checks, not estimates of herbal accuracy."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch

from herb_spectra.neural_base_search import (candidate_grid,reference_candidate,fit_preprocessor,
    transform_features,DenseClassifier,fit_model,model_scores,choose_source,_train_cached)
from herb_spectra.paper_fewshot_classification import state_fingerprint


class NeuralBaseSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        rng=np.random.default_rng(4);y=np.repeat(np.arange(7),8)
        axis=np.linspace(0,4,64)
        x=np.asarray([np.sin((c+1)*axis)+rng.normal(0,.08,len(axis)) for c in y])
        return x,y

    def test_grid_and_reference_are_unique_and_predeclared(self):
        grid=candidate_grid();self.assertEqual(len(grid),24);self.assertEqual(len({c['id'] for c in grid}),24)
        ref=reference_candidate();self.assertEqual(ref['pca_components'],20)
        self.assertEqual(ref['hidden_sizes'],[64,32]);self.assertEqual(ref['epochs'],200)
        self.assertEqual(ref['class_weight'],'balanced')

    def test_preprocessing_is_source_fitted_and_prediction_batch_independent(self):
        x,y=self.fixture();c=reference_candidate();p=fit_preprocessor(x,c)
        before=(p['scaler'].mean_.copy(),p['pca'].components_.copy())
        z=transform_features(p,x[:8]);single=transform_features(p,x[:1])
        assert_allclose(single,z[:1],atol=2e-6);assert_array_equal(before[0],p['scaler'].mean_)
        assert_array_equal(before[1],p['pca'].components_)
        self.assertEqual(z.shape,(8,20));self.assertEqual(z.dtype,np.float32)

    def test_constant_wrong_frequency_and_excessive_pca_are_rejected(self):
        x,y=self.fixture();c=reference_candidate();p=fit_preprocessor(x,c)
        with self.assertRaisesRegex(ValueError,'constant'):transform_features(p,np.ones((1,64)))
        with self.assertRaisesRegex(ValueError,'frequency'):transform_features(p,x[:2,:63])
        bad=deepcopy(c);bad['pca_components']=64
        with self.assertRaisesRegex(ValueError,'PCA'):fit_preprocessor(x,bad)

    def test_dense_scores_are_pointwise_and_do_not_change_state(self):
        torch.manual_seed(3);m=DenseClassifier(20,[64,32]);x=np.random.default_rng(1).normal(size=(6,20)).astype('float32')
        before=state_fingerprint(m);assert_allclose(model_scores(m,x)[:1],model_scores(m,x[:1]),atol=1e-7)
        self.assertEqual(before,state_fingerprint(m));self.assertEqual(m.encode(torch.from_numpy(x)).shape,(6,32))

    def test_training_is_deterministic_and_records_source_metrics(self):
        x,y=self.fixture();c=reference_candidate();c['epochs']=2;p=fit_preprocessor(x,c);z=transform_features(p,x)
        a,losses=fit_model(z,y,c,42);b,_=fit_model(z,y,c,42)
        self.assertEqual(state_fingerprint(a),state_fingerprint(b));self.assertIn('source_accuracy',losses[-1])
        self.assertEqual(a._source_initial_fingerprint,b._source_initial_fingerprint)
        self.assertNotEqual(a._source_initial_fingerprint,state_fingerprint(a))

    def test_training_refuses_missing_class_or_fractional_labels(self):
        x,y=self.fixture();c=reference_candidate();c['epochs']=1;p=fit_preprocessor(x,c);z=transform_features(p,x)
        with self.assertRaisesRegex(ValueError,'seven'):fit_model(z[y!=1],y[y!=1],c)
        with self.assertRaisesRegex(ValueError,'integer'):fit_model(z,y.astype(float),c)

    def test_source_selection_gate_can_reject_proposed_improvement(self):
        grid=candidate_grid();ref=reference_candidate();other=grid[0]
        rows=[]
        for c in grid:
            for stage in ('selection','gate'):
                accuracy=.8
                if c['id']==other['id']:accuracy=.85 if stage=='selection' else .79
                rows.append({'candidate_id':c['id'],'stage':stage,'accuracy':accuracy,
                             'balanced_accuracy':.8,'macro_f1':.8})
        selection={'reference_candidate_id':ref['id'],'accuracy_tolerance':.001,'minimum_accuracy_gain':.001,
                   'max_macro_f1_loss':.005,'max_balanced_accuracy_loss':.005}
        chosen=choose_source(rows,grid,selection)
        self.assertTrue(chosen['fallback_to_reference']);self.assertEqual(chosen['selected'],ref)
        self.assertEqual(chosen['proposed'],other);self.assertFalse(chosen['selected_using_final_query'])

    def test_cached_source_weights_reload_and_tampering_is_detected(self):
        x,y=self.fixture();classes=np.asarray([str(i) for i in range(7)])
        source={'record_ids':np.asarray([f'id{i}' for i in range(len(y))]),'y':classes[y]}
        c=reference_candidate();c['epochs']=1;p=fit_preprocessor(x,c);z=transform_features(p,x)
        with tempfile.TemporaryDirectory() as d:
            a,meta=_train_cached(d,source,classes,c,42,p,z)
            b,_=_train_cached(d,source,classes,c,42,p,z,resume=True)
            self.assertEqual(state_fingerprint(a),state_fingerprint(b))
            with (Path(d)/'model.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'artifacts'):_train_cached(d,source,classes,c,42,p,z,resume=True)


if __name__ == '__main__':
    unittest.main()
