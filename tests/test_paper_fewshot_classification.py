"""Synthetic neural/protocol checks; these are not herb prediction results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch

from herb_spectra.calibration_exploration import fast_metrics
from herb_spectra.paper_fewshot_classification import (SmallResNet1D,
    all_strategy_predictions, candidate_grid, choose_strategies, finetune_predictions,
    meta_fit, model_scores, object_digest, prototype_context, cached_prototype_predictions,
    prototype_scores, run_pilot, sample_meta_episodes, snv_tensor, state_fingerprint,
    supervised_fit, validate_protocol)


def specimens(counts=(8,1,8,8,8,8,8),seed=9,points=64):
    rng=np.random.default_rng(seed)
    y=np.repeat(np.arange(7),counts)
    axis=np.linspace(0,2*np.pi,points)
    x=np.asarray([np.sin((c+1)*axis)+rng.normal(0,.03,points) for c in y])
    ids=np.asarray([f"p{i:03d}" for i in range(len(y))])
    return snv_tensor(x),y,ids


class PaperFewshotClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_snv_is_per_row_and_constant_spectra_rejected(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(3,64))
        a=snv_tensor(x).numpy()
        assert_allclose(a[:,0].mean(1),0,atol=1e-7)
        assert_allclose(a[:,0].std(1,ddof=1),1,atol=1e-6)
        assert_array_equal(snv_tensor(x[:1]).numpy(),a[:1])
        with self.assertRaisesRegex(ValueError,"constant"):snv_tensor(np.ones((2,64)))

    def test_groupnorm_prediction_is_batch_independent(self):
        x,y,ids=specimens();torch.manual_seed(2);model=SmallResNet1D()
        before=state_fingerprint(model)
        a=model_scores(model,x[:6]);b=model_scores(model,x[:1])
        assert_allclose(a[:1],b,atol=2e-7)
        self.assertEqual(before,state_fingerprint(model))
        self.assertFalse(any(isinstance(m,torch.nn.BatchNorm1d) for m in model.modules()))

    def test_meta_sampling_is_source_disjoint_and_rare_eligibility_is_dynamic(self):
        x,y,ids=specimens()
        tasks,policy=sample_meta_episodes(y,ids,episodes=12,query=1)
        self.assertEqual(policy['excluded_insufficient_classes'],[1])
        for task in tasks:
            self.assertFalse(set(task['support'])&set(task['query']))
            self.assertEqual(len(task['support']),3)
            self.assertNotIn(1,task['classes'])
        xx,yy,ii=specimens(counts=(8,3,8,8,8,8,8))
        _,final_policy=sample_meta_episodes(yy,ii,episodes=1,query=1)
        self.assertEqual(final_policy['eligible_classes'],list(range(7)))

    def test_fomaml_has_real_source_query_gradients_and_preserves_initializer(self):
        x,y,ids=specimens();torch.manual_seed(3);initial=SmallResNet1D()
        before=state_fingerprint(initial)
        trained,record=meta_fit(initial,x,y,ids,strategy='fomaml',episodes=2,query=1)
        self.assertNotEqual(before,state_fingerprint(trained))
        self.assertEqual(before,state_fingerprint(initial))
        self.assertTrue(record['first_order'])
        self.assertTrue(all(np.isfinite(r['source_meta_query_cross_entropy']) for r in record['losses']))

    def test_supervised_continuation_and_prototype_share_source_task_records(self):
        x,y,ids=specimens();torch.manual_seed(5);initial=SmallResNet1D()
        before=state_fingerprint(initial)
        continued,a=meta_fit(initial,x,y,ids,strategy='supervised',episodes=2,query=1)
        proto,b=meta_fit(initial,x,y,ids,strategy='prototype',episodes=2,query=1)
        self.assertEqual(a['episodes'],b['episodes'])
        self.assertNotEqual(before,state_fingerprint(continued));self.assertNotEqual(before,state_fingerprint(proto))
        self.assertEqual(before,state_fingerprint(initial))

    def test_zero_prototype_and_cached_pointwise_scores_are_exact_matched_controls(self):
        x,y,ids=specimens();torch.manual_seed(6);model=SmallResNet1D()
        support=np.asarray([np.flatnonzero(y==c)[0] for c in range(7)])
        plain=prototype_scores(model,x,y,x[:10],blend=0)
        again=prototype_scores(model,x,y,x[:10],support_x=x[support],support_y=y[support],blend=0)
        assert_array_equal(plain,again)
        cache=prototype_context(model,x,y,x[:10])
        assert_array_equal(np.argmax(plain,1),cached_prototype_predictions(model,cache,x[support],y[support],0))
        direct=prototype_scores(model,x,y,x[:10],support_x=x[support],support_y=y[support],blend=.25)
        assert_array_equal(np.argmax(direct,1),cached_prototype_predictions(model,cache,x[support],y[support],.25))

    def test_support_finetuning_copies_source_and_truth_only_affects_scoring(self):
        x,y,ids=specimens();torch.manual_seed(7);model=SmallResNet1D()
        support=np.asarray([np.flatnonzero(y==c)[0] for c in range(7)])
        before=state_fingerprint(model)
        config=[c for c in candidate_grid() if c['strategy']=='full_ft'][:2]
        pred=finetune_predictions(model,x[support],y[support],x[:12],config)
        self.assertEqual(before,state_fingerprint(model))
        self.assertEqual(set(pred),{c['id'] for c in config})
        p=next(iter(pred.values()));a=fast_metrics(y[:12],p,7)
        alternate=(y[:12]+1)%7;b=fast_metrics(alternate,p,7)
        self.assertNotEqual(a,b)

    def test_development_only_guard_has_identity_fallback(self):
        candidates=[{'id':'x','strategy':s,'learning_rate':.001,'steps':1,'blend':.1} for s in ['head_ft','full_ft','prototype','fomaml_ft']]
        rows=[]
        for stage in ['selection','gate']:
            for c in candidates:
                rows.append({'strategy':c['strategy'],'candidate_id':'x','stage':stage,
                    'accuracy':.85 if stage=='selection' else .79,'macro_f1':.8,'balanced_accuracy':.8,
                    'matched_accuracy':.8,'matched_macro_f1':.8,'matched_balanced_accuracy':.8})
        selection={'minimum_accuracy_gain':.001,'max_macro_f1_loss':.005,'max_balanced_accuracy_loss':.005}
        choices=choose_strategies(rows,candidates,selection)
        self.assertTrue(all(value['selected'] is None for value in choices.values()))

    def test_synthetic_end_to_end_and_resume_do_not_refit(self):
        protocol=json.loads(Path('data/research/paper_neural_classification_protocol_20261031.json').read_text())
        protocol['candidate_grid_sha256']=object_digest(candidate_grid())
        protocol['seeds']=[42];protocol['frequency']['expected_points']=64
        protocol['development'].update(episodes=2,selection_episodes=1,gate_episodes=1,support_draws=1)
        protocol['training'].update(epochs=1,meta_episodes=1)
        protocol['final'].update(support_draws=1,training_samples=53,calibration_pool_samples=14,query_samples=14,random_query_examples=3)
        validate_protocol(protocol)
        counts=(10,3,8,8,8,8,8);labels=[];records=[];X=[];rows=[]
        rng=np.random.default_rng(13);axis=np.linspace(.1,1.5,64)
        for c,n in enumerate(counts):
            for role,count in [('training',n),('calibration',2),('prediction',2)]:
                for j in range(count):
                    identifier=f'{c}-{role}-{j}'
                    labels.append(str(c));records.append(identifier)
                    X.append(np.sin((c+1)*axis)+rng.normal(0,.03,len(axis)))
                    rows.append({'record_id':identifier,'physical_sample_id':identifier,'origin_group_id':str(c),'role':role})
        X=np.asarray(X);records=np.asarray(records);labels=np.asarray(labels)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);np.savez(p/'input.npz',X_alpha=X,X_n=X/2+1,axis=axis,record_ids=records,groups=records,y=labels,
                                 batch_ids=np.repeat('',len(X)),labels_verified=np.ones(len(X),dtype=bool),dataset_version=protocol['dataset_version'])
            plan={'dataset_version':protocol['dataset_version'],'split_seed':1,'records':rows}
            (p/'plan.json').write_text(json.dumps(plan));(p/'protocol.json').write_text(json.dumps(protocol))
            summary=run_pilot(p/'input.npz',p/'plan.json',p/'protocol.json',p/'out')
            self.assertEqual(len(summary),12)
            before={str(q):q.stat().st_mtime_ns for q in (p/'out').rglob('*.pt')}
            self.assertEqual(summary,run_pilot(p/'input.npz',p/'plan.json',p/'protocol.json',p/'out',resume=True))
            self.assertEqual(before,{str(q):q.stat().st_mtime_ns for q in (p/'out').rglob('*.pt')})
            status=json.loads((p/'out/status.json').read_text());self.assertFalse(status['paper_exact_reproduction'])
            self.assertEqual(status['prediction_rows'],12*14)


if __name__ == '__main__':
    unittest.main()
