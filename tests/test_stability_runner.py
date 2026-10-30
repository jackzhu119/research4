import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.base import clone

from herb_spectra.stability import StableVariableSelector


MODULE_SPEC=importlib.util.spec_from_file_location('stability_research_runner',
                  Path(__file__).resolve().parents[1]/'scripts/run_stability_research.py')
runner=importlib.util.module_from_spec(MODULE_SPEC);MODULE_SPEC.loader.exec_module(runner)


class StabilityRunnerTests(unittest.TestCase):
    def test_ten_distinct_paths_match_dimensions_and_record_real_budget_difference(self):
        self.assertEqual(len(runner.MODEL_NAMES),10)
        for method in runner.MODEL_NAMES:
            pipe,grid,budget=runner.make_research_pipeline(method,76,40,['raw','snv'])
            clone(pipe)
            self.assertEqual(grid['preprocess__mode'],['raw','snv'])
            if method.startswith(('mcuve20_','mcuve100_','stable_mcuve_')):
                self.assertEqual(pipe.named_steps['selector'].n_features_to_select,8)
            if method.startswith('stable_'):
                self.assertIsInstance(pipe.named_steps['selector'],StableVariableSelector)
                self.assertEqual(budget['base_coefficient_resamples'],100)
                self.assertEqual(budget['tie_break_coefficient_fits'],10)
            if method=='cars_plsda':self.assertEqual(grid['model__n_components'],[1,2])

    def test_small_real_execution_keeps_heldout_ids_and_resampling_inside_training(self):
        rng=np.random.default_rng(826)
        X=rng.normal(size=(54,12));y=np.repeat([0,1,2],18)
        X[:,0]+=y*3
        train=np.r_[np.arange(14),np.arange(18,32),np.arange(36,50)]
        test=np.setdiff1d(np.arange(54),train)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'data.npz'
            np.savez_compressed(path,X_train=X[train],y_train=y[train],X_test=X[test]+100,
                         y_test=y[test],train_ids=np.asarray([f'sample-{i}' for i in train]),
                         test_ids=np.asarray([f'sample-{i}' for i in test]))
            spec={'id':'synthetic','task':'classification','source_url':'synthetic engineering fixture',
                  'data_path':str(path),'preprocess_modes':['raw'],'identity_scope':'synthetic row IDs'}
            rows=runner.run_dataset(spec,root/'run',models=['stable_mcuve_plsda'])
            self.assertEqual(rows[0]['status'],'pass')
            diag=json.loads((root/'run/stable_mcuve_plsda/stability_diagnostics.json').read_text())
            self.assertEqual(diag['training_unit_count'],len(train))
            self.assertTrue(all(max(indices)<len(train) for indices in diag['resample_unit_indices']))
            folds=json.loads((root/'run/inner_folds.json').read_text())
            query_ids={f'sample-{i}' for i in test}
            for fold in folds:
                self.assertFalse(query_ids & (set(fold['train_ids'])|set(fold['validation_ids'])))
            self.assertEqual(rows[0]['selected_feature_count'],5)
            self.assertEqual(rows[0]['candidate_count'],4)
            self.assertTrue((root/'run/stable_mcuve_plsda/cv_candidates.json').exists())

    def test_changed_data_hash_rejected_before_any_classifier_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data.npz';data.write_bytes(b'not a matching source')
            with self.assertRaisesRegex(ValueError,'NPZ hash differs'):
                runner.run_dataset({'task':'classification','data_path':str(data),
                                    'data_sha256':'wrong'},root/'run')


if __name__=='__main__':unittest.main()
