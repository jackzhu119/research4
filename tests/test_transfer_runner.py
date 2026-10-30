import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


spec=importlib.util.spec_from_file_location('transfer_runner',Path(__file__).resolve().parents[1]/'scripts/run_transfer_research.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class TransferProtocolTests(unittest.TestCase):
    def test_support_is_nested_query_is_fixed_and_rng_reproduces(self):
        first=runner.support_query_splits(80,[1,3,5,10,20],repeats=3,seed=7)
        repeated=runner.support_query_splits(80,[1,3,5,10,20],repeats=3,seed=7)
        for a,b in zip(first,repeated):
            assert_array_equal(a['query'],b['query']);assert_array_equal(a['support_pool'],b['support_pool'])
            self.assertFalse(set(a['query']) & set(a['support_pool']))
            self.assertEqual(len(a['query']),60)
            for k in [1,3,5,10,20]:
                self.assertTrue(set(a['support_pool'][:k]) <= set(a['support_pool']))
                self.assertFalse(set(a['query']) & set(a['support_pool'][:k]))

    def test_record_overlap_rejected_even_when_spectra_differ(self):
        rng=np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.npz'
            np.savez(path,X_source=rng.normal(size=(9,4)),y_source=np.arange(9.),
                     source_ids=np.asarray([f's-{i}' for i in range(9)]),
                     X_target=rng.normal(size=(6,4)),y_target=np.arange(6.),
                     target_ids=np.asarray(['s-0']+[f't-{i}' for i in range(5)]))
            with self.assertRaisesRegex(ValueError,'overlap target pool'):runner.validate_input(path)

    def test_target_only_one_shot_mean_but_joint_keeps_source_training(self):
        X=np.arange(24.,dtype=float).reshape(12,2);y=X[:,0]*2+4
        model=Pipeline([('model',PLSRegression(n_components=1,scale=False))]).fit(X,y)
        query=np.asarray([[3.,4.],[7.,8.]])
        support=np.asarray([[11.,12.]]);target=np.asarray([26.])
        prediction,meta=runner.fit_frozen_head(model,X,y,support,target,query,joint=False)
        assert_array_equal(prediction,[26.,26.]);self.assertEqual(meta['n_components'],0)
        joint,joint_meta=runner.fit_frozen_head(model,X,y,support,target,query,joint=True)
        self.assertEqual(joint_meta['training_rows'],13)
        self.assertEqual(joint_meta['n_components'],1)
        self.assertFalse(np.array_equal(joint,prediction))


if __name__=='__main__':unittest.main()
