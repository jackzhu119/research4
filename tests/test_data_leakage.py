"""Positive and deliberately leaking fixtures; no scientific accuracy claims."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from herb_spectra.leakage_guard import (
    FitScope, LeakageError, LineageRecord, assert_modalities_aligned,
    audit_pointwise_inference, guarded_cache_key, load_guarded_cache,
    instrument_mcuve_pls,
    save_guarded_cache, source_fingerprint, validate_fixed_partitions,
    validate_lineage_partitions,
)


class DataLeakageTests(unittest.TestCase):
    def test_unknown_is_blocked_and_user_declaration_is_exploratory(self):
        unknown = [LineageRecord("a"), LineageRecord("b")]
        roles = {"train":["a"], "test":["b"]}
        self.assertEqual(validate_lineage_partitions(unknown,roles)["status"], "BLOCKED")
        known = [LineageRecord(v, "physical_"+v, independence_evidence="user_declared_independent_20261031") for v in ("a","b")]
        result = validate_lineage_partitions(known,roles)
        self.assertEqual(result["status"], "EXPLORATORY")
        self.assertFalse(result["independent_confirmation_established"])

    def test_common_ancestor_augmentation_and_repreparation_leak_rejected(self):
        records = [LineageRecord("source", "P0"),
                   LineageRecord("repressed", "P1", "P0"),
                   LineageRecord("augmented", "P2", "repressed")]
        with self.assertRaises(LeakageError):
            validate_lineage_partitions(records,{"train":["source"],"test":["augmented"]})
        with self.assertRaises(LeakageError):
            validate_lineage_partitions(records,{"train":["source"],"validation":["repressed"]})

    def test_same_sample_different_modalities_cannot_cross_roles(self):
        rows = [LineageRecord("p_alpha","P", modality="alpha"),
                LineageRecord("p_n","P", modality="n")]
        with self.assertRaises(LeakageError):
            validate_lineage_partitions(rows,{"train":["p_alpha"],"test":["p_n"]})
        # One base record can carry multiple aligned modalities, inside one role.
        same = [LineageRecord("p","P",modality=m,independence_evidence="user_declared") for m in ("alpha","n")]
        self.assertEqual(validate_lineage_partitions(same,{"train":["p"]})["status"],"EXPLORATORY")

    def test_conflicting_label_and_cyclic_lineage_rejected(self):
        with self.assertRaises(LeakageError):
            validate_lineage_partitions([LineageRecord("a","P",label="x"),LineageRecord("b","P",label="y")],{"train":["a","b"]})
        with self.assertRaises(LeakageError):
            validate_lineage_partitions([LineageRecord("a","P1","b"),LineageRecord("b","P2","a")],{"train":["a","b"]})

    def test_saved_partition_digest_and_coverage(self):
        roles = {"train":["a","b"], "test":["c"]}
        digest = validate_fixed_partitions(roles,expected_ids=["a","b","c"])
        self.assertEqual(validate_fixed_partitions(roles,expected_digest=digest),digest)
        with self.assertRaises(LeakageError):
            validate_fixed_partitions({"train":["b","a"],"test":["c"]},expected_digest=digest)
        with self.assertRaises(LeakageError):
            validate_fixed_partitions({"train":["a"],"test":["a"]})

    def test_modality_pairing_must_match_identity_and_order(self):
        self.assertTrue(assert_modalities_aligned(["a","b"],{"alpha":["a","b"],"n":["a","b"]}))
        for identifiers in (["b","a"],["a"],["a","a"]):
            with self.assertRaises(LeakageError):
                assert_modalities_aligned(["a","b"],{"n":identifiers})

    def test_fit_scope_blocks_forbidden_data_before_estimator_fit(self):
        scope = FitScope(["a","b"],"outer0.inner0","v1")
        model = StandardScaler()
        with self.assertRaises(LeakageError):
            scope.fit(model,np.ones((2,3)),sample_ids=["a","query"],component="scaler")
        self.assertFalse(hasattr(model,"mean_"))
        self.assertEqual(scope.entries[-1]["event"],"fit_rejected")

    def test_fit_calls_record_actual_source_rows_and_custom_operation(self):
        X=np.arange(12,dtype=float).reshape(4,3); y=np.array([0,0,1,1]); ids=["a","b","c","d"]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fit.jsonl"
            scope=FitScope(ids,"outer0.inner0","v1",path,code_version="code1")
            scaler=scope.fit(StandardScaler(),X,sample_ids=ids,component="scaler")
            with scope.operation("torch_train",ids,X=X,y=y) as proof:
                self.assertEqual(proof["data_fingerprint"],source_fingerprint(X,y))
            self.assertTrue(np.array_equal(scaler.mean_,X.mean(0)))
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[-1]["fit_sample_ids"],ids)
            self.assertEqual(rows[-1]["split_id"],"outer0.inner0")

    def test_fit_scope_checks_shapes_and_rejects_identifier_features(self):
        scope=FitScope(["a","b"],"outer0","v1")
        with self.assertRaises(LeakageError):
            scope.fit(StandardScaler(),np.ones((3,2)),sample_ids=["a","b"],component="scaler")
        with self.assertRaises(LeakageError):
            scope.fit(StandardScaler(),np.array([["ZY01","path"],["ZY02","file"]]),sample_ids=["a","b"],component="scaler")
        with self.assertRaises(LeakageError):
            with scope.operation("train",["a","b"],X={"alpha":np.ones((2,3)),"n":np.ones((1,3))}):
                self.fail("unpaired modalities must not reach training")

    def test_cache_binds_actual_fit_and_every_identity_field(self):
        X=np.arange(12,dtype=float).reshape(4,3);ids=["a","b","c","d"]
        args=dict(dataset_version="v1",train_ids=ids,split_id="outer0.inner0",preprocessing={"mode":"snv"},
                  selector={"MC":10,"K":2},code_version="code1",seed=42,data_fingerprint=source_fingerprint(X))
        key=guarded_cache_key(**args)
        scope=FitScope(ids,"outer0.inner0","v1",code_version="code1")
        model=scope.fit(StandardScaler(),X,sample_ids=ids,component="scaler")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"scaler.joblib"
            save_guarded_cache(path,model,key)
            restored=load_guarded_cache(path,key)
            self.assertTrue(np.array_equal(restored.mean_,model.mean_))
            mutations={"dataset_version":"v2","train_ids":["a","b","c","e"],"split_id":"outer1.inner0",
                       "preprocessing":{"mode":"msc"},"selector":{"MC":20,"K":3},"code_version":"code2", "seed":43,
                       "data_fingerprint":source_fingerprint(X+1)}
            for field,value in mutations.items():
                with self.subTest(field=field),self.assertRaises(LeakageError):
                    load_guarded_cache(path,guarded_cache_key(**(args|{field:value})))
            with self.assertRaises(LeakageError):
                save_guarded_cache(Path(directory)/"wrong_fold.joblib",model,guarded_cache_key(**(args|{"split_id":"outer1.inner0"})))
            with self.assertRaises(LeakageError):
                load_guarded_cache(path,str(key))

    def test_mc_internal_pls_calls_are_bound_and_mathematics_unchanged(self):
        from sklearn.cross_decomposition import PLSRegression
        from herb_spectra.selection import MCUVESelector
        rng=np.random.default_rng(42);X=rng.normal(size=(18,8));y=np.repeat([0,1,2],6)
        ids=[f"p{i}" for i in range(len(X))]
        baseline=MCUVESelector(n_runs=3,n_components=1,n_features_to_select=3,task="classification",random_state=42).fit(X,y,groups=ids)
        selector=MCUVESelector(n_runs=3,n_components=1,n_features_to_select=3,task="classification",random_state=42)
        scope=FitScope(ids,"outer0.inner0","v1")
        original=PLSRegression.fit
        with instrument_mcuve_pls(scope,X,ids):
            scope.fit(selector,X,y,sample_ids=ids,component="MC-UVE",groups=ids)
        self.assertIs(PLSRegression.fit,original)
        self.assertTrue(np.array_equal(selector.selected_indices_,baseline.selected_indices_))
        self.assertTrue(np.array_equal(selector.signed_stability_,baseline.signed_stability_))
        self.assertTrue(np.array_equal(selector.coefficients_,baseline.coefficients_))
        inner=[r for r in scope.entries if r["event"]=="fit_completed" and r["component"]=="MC-UVE.PLSRegression.fit"]
        self.assertEqual(len(inner),9)
        self.assertTrue(all(set(r["fit_sample_ids"])<=set(ids) and len(r["fit_sample_ids"])<len(ids) for r in inner))

    def test_mc_internal_foreign_or_ambiguous_rows_rejected_and_hooks_restored(self):
        from sklearn.cross_decomposition import PLSRegression
        from herb_spectra import selection
        rng=np.random.default_rng(42);X=rng.normal(size=(8,4));ids=[f"p{i}" for i in range(8)]
        scope=FitScope(ids,"outer0","v1"); original=PLSRegression.fit
        with self.assertRaises(LeakageError):
            with instrument_mcuve_pls(scope,X,ids):
                selection._coefficients(X+100,np.eye(2)[np.arange(8)%2],1)
        self.assertIs(PLSRegression.fit,original)
        with self.assertRaises(LeakageError):
            with instrument_mcuve_pls(scope,np.ones((8,4)),ids):
                self.fail("Ambiguous numerical row binding must fail")

    def test_inference_eval_no_grad_and_batchnorm_buffers_unchanged(self):
        import torch
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.fc=torch.nn.Linear(3,2);self.observed=[]
            def forward(self,x):
                self.observed.append((self.training,torch.is_grad_enabled()))
                return self.fc(self.bn(x))
        torch.manual_seed(42);net=Net();X=torch.randn(7,3)
        result=audit_pointwise_inference(net,X)
        self.assertEqual(result["status"],"TECHNICAL_CHECK_PASS")
        self.assertTrue(all(state==(False,False) for state in net.observed))
        self.assertEqual(int(net.bn.num_batches_tracked),0)

    def test_inference_rejects_query_batch_normalization_and_mutation(self):
        import torch
        class BatchDependent(torch.nn.Module):
            def forward(self,x):
                return x-x.mean(dim=0,keepdim=True)
        with self.assertRaises(LeakageError):
            audit_pointwise_inference(BatchDependent(),torch.arange(18,dtype=torch.float32).reshape(6,3))
        class Mutating(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("counter",torch.tensor(0))
            def forward(self,x):
                self.counter+=1;return x
        with self.assertRaises(LeakageError):
            audit_pointwise_inference(Mutating(),torch.ones(5,3))
        with self.assertRaises(LeakageError):
            audit_pointwise_inference(torch.nn.BatchNorm1d(3,track_running_stats=False),torch.randn(5,3))

    def test_paired_dictionary_inference_is_pointwise(self):
        import torch
        class Multi(torch.nn.Module):
            def forward(self,x):
                return torch.cat([x["alpha"],x["n"]],dim=1)
        result=audit_pointwise_inference(Multi(),{"alpha":np.ones((5,2)),"n":np.arange(10).reshape(5,2)})
        self.assertEqual(result["samples_checked"],5)
        self.assertTrue(result["parameters_and_buffers_unchanged"])


if __name__ == "__main__":
    unittest.main()
