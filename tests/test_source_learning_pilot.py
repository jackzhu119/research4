import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from herb_spectra.leakage_guard import FitScope,LeakageError
from herb_spectra.upgrade_neural import UpgradeNet,PROFILES,CLASSES,fit_source,state_hash

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_learning_pilot',ROOT/'scripts/run_source_learning_pilot.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PilotBoundaries(unittest.TestCase):
    def test_fixed_protocol_budget_and_classes(self):
        self.assertEqual(p.CHECKPOINTS,(12,24,48,96));self.assertEqual(p.FAMILIES,('cnn_alpha','fusion_all_four'));self.assertEqual(p.profile()['epochs'],96)
        self.assertEqual(p.CLASSES,tuple(f'ZY{i:02}' for i in range(1,8)))

    def test_role_overlap_and_unknown_feature_rejected(self):
        source={'record_ids':np.asarray([f'ID{i}' for i in range(342)]),'source_labels':np.asarray([CLASSES[i%7] for i in range(342)])}
        for name,n in [('alpha',612),('n',612),('time',512),('frequency',256)]:source['X_'+name]=np.zeros((342,n))
        val={k:v.copy() for k,v in source.items() if k!='source_labels'}
        with self.assertRaisesRegex(LeakageError,'disjoint'):p.validate_pair(source,val)
        val['record_ids']=np.asarray([f'V{i}' for i in range(342)]);p.validate_pair(source,val)
        val['true_labels_disguised']=source['source_labels']
        with self.assertRaisesRegex(LeakageError,'schema'):p.validate_pair(source,val)

    def test_outside_fit_rejected_before_scaler_fit(self):
        with tempfile.TemporaryDirectory() as d:
            scope=FitScope(['source1','source2'],'fixture','fixture',Path(d)/'scope.jsonl',code_version='fixture');scaler=StandardScaler()
            with self.assertRaises(LeakageError):scope.fit(scaler,np.ones((2,3)),sample_ids=['source1','validation1'],component='negative_scaler')
            self.assertFalse(hasattr(scaler,'mean_'))

    def test_outer_and_other_neural_paths_denied(self):
        for path in [ROOT/'data/multiplant_database/zaoren/aligned_modalities.npz',p.OLD/'scoring_truth/outer00.final.csv',p.OLD/'inputs/outer00.final/inference.npz',p.BASE/'pooled_metrics_by_seed.json']:
            with self.assertRaises(PermissionError):p.check_read_access(path,set())
        p.check_read_access(ROOT/'reports/new-pilot/inputs/source.npz',set())

    def test_actual_python_audit_hook_rejects_before_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            code="import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('pilot',"+repr(str(ROOT/'scripts/run_source_learning_pilot.py'))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.install_read_guard(Path("+repr(d)+"))"
            r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            report=json.loads((Path(d)/'outer_read_rejection.json').read_text());self.assertTrue(report['read_rejected_before_bytes']);self.assertFalse(report['OS_sandbox'])

    def test_changed_inner_truth_rejected_before_semantic_parse(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);paths={k:base/k for k in ['source','validation','truth']}
            for key,path in paths.items():path.write_bytes(('fixture-'+key).encode())
            context={'source_pack_sha256':p.sha(paths['source']),'inference_pack_sha256':p.sha(paths['validation'])};frozen={'scoring_truth/outer00.inner00.csv':p.sha(paths['truth'])}
            p.check_inner_input_hashes(paths,context,frozen)
            paths['truth'].write_bytes(b'changed labels but same hypothetical IDs')
            with self.assertRaisesRegex(LeakageError,'Frozen inner truth bytes changed'):p.check_inner_input_hashes(paths,context,frozen)

    def test_missing_release_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);p.write_json(out/'protocol.json',{});p.write_json(out/'input_manifest.json',{})
            with self.assertRaisesRegex(LeakageError,'Root source release'):p.verify_release(out,{})

    def test_evaluation_preserves_optimizer_bn_rng_gradients(self):
        torch.manual_seed(42);model=UpgradeNet('cnn_alpha',{'width':4,'blocks':1,'dropout':.2},{'alpha':16});optimizer=torch.optim.Adam(model.parameters())
        x={'alpha':torch.randn(14,16)};labels=np.asarray([c for c in CLASSES for _ in range(2)]);before=state_hash(model)
        metrics,scores=p.evaluate(model,optimizer,x,labels,torch.ones(7))
        self.assertEqual(before,state_hash(model));self.assertTrue(model.training);self.assertEqual(scores.shape,(14,7));self.assertEqual(metrics['total'],14)
        class Mutator(nn.Module):
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.zeros(7));self.register_buffer('counter',torch.tensor(0))
            def forward(self,inputs):self.counter.add_(1);return self.weight.expand(len(inputs['alpha']),7)
        bad=Mutator();opt=torch.optim.Adam(bad.parameters())
        with self.assertRaisesRegex(LeakageError,'Evaluation changed'):p.evaluate(bad,opt,x,labels,torch.ones(7))

    def test_eval_insertions_do_not_change_frozen_training_math(self):
        # Two synthetic14-record fits: one frozen implementation, one pilot loop.
        # These fixture trajectories are not real medicinal-plant experiments.
        rng=np.random.default_rng(7);raw=rng.normal(size=(14,16));ids=np.asarray([f'fixture{i}' for i in range(14)]);labels=np.asarray([c for c in CLASSES for _ in range(2)])
        config={**PROFILES[1],'epochs':2,'width':4,'blocks':1}
        with tempfile.TemporaryDirectory() as d:
            old=fit_source({'alpha':raw},labels,ids,'cnn_alpha',config,42,'fixture',Path(d)/'old.jsonl','fixture')
            scope=FitScope(ids,'fixture','fixture',Path(d)/'new.jsonl',code_version='fixture');source={'record_ids':ids,'source_labels':labels,'X_alpha':raw};x,_=p.fit_transforms(source,'cnn_alpha',scope)
            torch.manual_seed(42);model=UpgradeNet('cnn_alpha',config,{'alpha':16});optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=.001);generator=torch.Generator().manual_seed(42);tensors={'alpha':torch.as_tensor(x['alpha'])};target=torch.as_tensor(p.encode(labels));weights=torch.ones(7);criterion=nn.CrossEntropyLoss(weight=weights)
            losses=[]
            for _ in range(2):losses.append(p.train_epoch(model,optimizer,criterion,tensors,target,generator));p.evaluate(model,optimizer,tensors,labels,weights)
            self.assertEqual(state_hash(model),old.metadata['fitted_state_sha256']);np.testing.assert_array_equal(losses,old.metadata['loss_curve'])


if __name__=='__main__':unittest.main()
