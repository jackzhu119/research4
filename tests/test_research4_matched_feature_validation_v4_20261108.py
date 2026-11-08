"""Regression for sklearn1.8 MLP early stopping with string source labels.

All real fits here use tiny synthetic arrays; no canonical research data,
background queue, original annotation or frozen V3 file is modified.
"""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V3=ROOT/'scripts/research4_matched_feature_validation_20261108.py'
V4=ROOT/'scripts/research4_matched_feature_validation_20261108_v4.py'
spec=importlib.util.spec_from_file_location('matched_feature_v4_worker',V4)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class StringMLPEarlyStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(911)
        cls.y=np.repeat(np.asarray(['ZY03','ZY04','ZY05','ZY06','ZY07']),20)
        encoded=np.repeat(np.arange(5),20)
        cls.train=rng.normal(scale=.05,size=(100,12))
        cls.train[:,:5]+=np.eye(5)[encoded]*3
        cls.query=cls.train[[0,21,42,63,84]]+rng.normal(scale=.02,size=(5,12))
        cls.axis=np.linspace(.1,1.5,12)
        cls.original_y=cls.y.copy()
        cls.config={'family':'fullalpha_MLP','preprocessing':'raw','hidden':32}
        with m.threadpool_limits(limits=1):
            cls.prediction,cls.bundle=m.fit_auxiliary(cls.config,cls.train,cls.y,cls.query,cls.axis)

    def test_real_tiny_five_string_class_early_stopping_completes(self):
        model=self.bundle['model']
        self.assertTrue(model.early_stopping)
        self.assertGreaterEqual(model.n_iter_,1)
        self.assertTrue(np.isfinite(model.validation_scores_).all())
        np.testing.assert_array_equal(model.classes_,np.arange(5))
        self.assertEqual(self.prediction.shape,(5,))
        self.assertTrue(set(self.prediction)<=set(self.y))

    def test_original_labels_unchanged_and_only_model_targets_encoded(self):
        np.testing.assert_array_equal(self.y,self.original_y)
        self.assertEqual(self.y.dtype.kind,'U')
        self.assertEqual(self.bundle['class_labels'],np.unique(self.y).tolist())
        self.assertEqual(self.bundle['target_encoding'],'source_unique_labels_to_contiguous_int')

    def test_serialized_bundle_reproduces_string_predictions(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'tiny_model.joblib';joblib.dump(self.bundle,path)
            restored=joblib.load(path)
            values=m.SpectralPreprocessor(restored['preprocessing'],self.axis).transform(self.query)
            integer=restored['model'].predict(restored['scaler'].transform(values))
            actual=np.asarray(restored['class_labels'])[integer.astype(int)]
            np.testing.assert_array_equal(actual,self.prediction)

    def test_lda_remains_unencoded_and_matches_frozen_v3(self):
        oldspec=importlib.util.spec_from_file_location('matched_feature_frozen_v3_test',V3)
        old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
        config={'family':'fullalpha_shrinkLDA','preprocessing':'snv'}
        with m.threadpool_limits(limits=1):
            before,_=old.fit_auxiliary(config,self.train,self.y,self.query,self.axis)
            after,bundle=m.fit_auxiliary(config,self.train,self.y,self.query,self.axis)
        np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(bundle['model'].classes_,np.unique(self.y))
        self.assertIsNone(bundle['class_labels']);self.assertEqual(bundle['target_encoding'],'NONE')

    def test_source_encoding_does_not_use_query_values_or_labels(self):
        observed=[]
        class Fake:
            def fit(self,x,y):observed.append((x.copy(),y.copy()));return self
            def predict(self,x):return np.zeros(len(x),dtype=int)
        with patch.object(m,'MLPClassifier',return_value=Fake()):
            p1,b1=m.fit_auxiliary(self.config,self.train,self.y,self.query,self.axis)
            p2,b2=m.fit_auxiliary(self.config,self.train,self.y,self.query+1e6,self.axis)
        self.assertTrue(all(len(y)==100 and np.issubdtype(y.dtype,np.integer)for x,y in observed))
        np.testing.assert_array_equal(observed[0][1],observed[1][1])
        np.testing.assert_array_equal(b1['scaler'].mean_,b2['scaler'].mean_)
        np.testing.assert_array_equal(p1,p2)

    def test_invalid_model_integer_mapping_is_rejected(self):
        class Bad:
            def fit(self,x,y):return self
            def predict(self,x):return np.full(len(x),5,dtype=int)
        with patch.object(m,'MLPClassifier',return_value=Bad()),self.assertRaisesRegex(ValueError,'outside its source-only integer encoding'):
            m.fit_auxiliary(self.config,self.train,self.y,self.query,self.axis)

    def test_frozen_v3_unchanged_and_only_auxiliary_function_differs(self):
        self.assertEqual(m.sha(V3),'96fc728da02d2917443413ca5ddc8c005983be0695df19aff95a82072ab309d6')
        before=ast.parse(V3.read_text());after=ast.parse(V4.read_text())
        def scientific_ast(module):
            return [ast.dump(n,include_attributes=False)for n in module.body
                    if not(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name=='fit_auxiliary')]
        self.assertEqual(scientific_ast(before),scientific_ast(after))
        self.assertEqual(m.FAMILIES['alpha_snv100'],(None,'snv','mc100'))
        self.assertEqual(m.FAMILIES['alpha_snv_stability5x20'],(None,'snv','rankvote5x20'))


def load_tests(loader, tests, pattern):
    # Reuse every existing matched-K/source isolation test against V4, without
    # editing either frozen source or its test file.
    path=ROOT/'tests/test_research4_matched_feature_validation_20261108.py'
    oldspec=importlib.util.spec_from_file_location('matched_feature_reused_v3_tests_for_v4',path)
    reused=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(reused)
    reused.m=m
    tests.addTests(loader.loadTestsFromModule(reused))
    return tests


if __name__=='__main__':unittest.main()
