"""Synthetic schema/guard tests only; never read running benchmark results."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import export_public_thz_results_20261031 as export


class PublicTHzExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_partial_or_unreleased_score_never_creates_output(self):
        source=self.root/"benchmark";source.mkdir();out=self.root/"reports"
        release=self.root/"release.json";self.write(release,{"allow_aggregate_export":True,"created_at_utc":"fixture"})
        with self.assertRaisesRegex(export.ExportError,"score is incomplete"):
            export.export({"ginseng":source},out,release)
        self.assertFalse(out.exists())
        self.write(source/"scoring_manifest.json",{"fixture":True})
        with self.assertRaisesRegex(export.ExportError,"not bound"):
            export.export({"ginseng":source},out,release)
        self.assertFalse(out.exists())

    def test_post_score_root_release_exact_hash_required(self):
        source=self.root/"benchmark";source.mkdir();self.write(source/"scoring_manifest.json",{"fixture":True})
        release={"allow_aggregate_export":True,"created_at_utc":"fixture","ginseng_scoring_manifest_sha256":export.sha(source/"scoring_manifest.json")}
        export.verify_release(release,{"ginseng":source})
        (source/"scoring_manifest.json").write_text('{"fixture":"tampered"}')
        with self.assertRaises(export.ExportError):export.verify_release(release,{"ginseng":source})
        with self.assertRaises(export.ExportError):export.verify_release({}, {"ginseng":source})

    def classification_fixture(self):
        y=np.array(["0"]*22+["1"]*20);metrics={};predictions=[]
        for role in export.STAGES:
            metrics[role]={}
            for seed in export.SEEDS:
                pred=y.copy();pred[:seed-41]=np.where(pred[:seed-41]=="0","1","0")
                metrics[role][str(seed)]={"accuracy":accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),
                    "balanced_accuracy":balanced_accuracy_score(y,pred),"correct":int((y==pred).sum()),"total":42,
                    "confusion_matrix":confusion_matrix(y,pred,labels=["0","1"]).tolist(),"class_order":["0","1"]}
                predictions.extend({"role":role,"seed":seed,"record_id":f"synthetic{i}","y_true":a,"y_predicted":b}
                                   for i,(a,b) in enumerate(zip(y,pred)))
        self.write(self.root/"pooled_metrics_by_seed.json",metrics)
        with gzip.open(self.root/"scored_outer_predictions.csv.gz","wt",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        return {"outer_roles":list(export.STAGES),"protocol_id":"synthetic-test-only","scientific_status":"FIXTURE"},metrics

    def test_every_seed_metric_independently_recomputed_and_confusions_aggregate(self):
        protocol,metrics=self.classification_fixture()
        rows,confusions=export.classification_rows(self.root,protocol)
        self.assertEqual(len(rows),15);self.assertEqual(len(confusions),60)
        self.assertEqual({r["seed"] for r in rows},set(export.SEEDS))
        classes=export.per_class_rows(confusions)
        self.assertEqual(len(classes),30)
        self.assertEqual(sum(r["support"] for r in classes if r["role"]=="R0" and r["seed"]==42),42)
        metrics["R0"]["42"]["macro_f1"]+=.01
        self.write(self.root/"pooled_metrics_by_seed.json",metrics)
        with self.assertRaisesRegex(export.ExportError,"recalculation disagrees"):
            export.classification_rows(self.root,protocol)

    def test_same_protocol_gains_only_and_sample_sd_all_seeds(self):
        protocol,_=self.classification_fixture();rows,_=export.classification_rows(self.root,protocol)
        gains=export.within_protocol_rounds(rows)
        self.assertEqual(len(gains),15)
        self.assertTrue(all(r["accuracy_gain_percentage_points"]==0 for r in gains))
        summary=export.summarize(rows,("accuracy","macro_f1","balanced_accuracy"))
        expected=np.std([41/42,40/42,39/42],ddof=1)
        self.assertAlmostEqual(summary[0]["accuracy_sample_sd"],expected)
        self.assertIn("42:41/42",summary[0]["correct_each_seed"])
        wrong=[dict(r) for r in rows];wrong[3]["dataset"]="different-task"
        with self.assertRaisesRegex(export.ExportError,"Cross-dataset"):
            export.within_protocol_rounds(wrong)

    def test_fruit_only_aggregate_metrics_read_negative_r2_and_all_seeds_kept(self):
        metrics=[{"task":"MA_DM","role":"dummy_mean","seed":seed,"rmse":1.,"mae":.8,"r2":-.5,
                  "observations":40,"fruit_groups":10,"y_true":"MUST_NOT_COPY","y_predicted":"MUST_NOT_COPY"}
                 for seed in export.SEEDS]
        self.write(self.root/"pooled_metrics_by_seed.json",metrics)
        protocol={"tasks":{"MA_DM":None},"outer_roles":["dummy_mean"],"protocol_id":"synthetic-fruit","scientific_status":"FIXTURE"}
        rows=export.regression_rows(self.root,protocol)
        self.assertTrue(all(r["r2"]==-.5 for r in rows))
        self.assertNotIn("y_true",rows[0]);self.assertNotIn("y_predicted",rows[0])
        self.assertNotIn("accuracy",rows[0])
        self.assertFalse((self.root/"scored_outer_predictions.csv.gz").exists())
        self.write(self.root/"pooled_metrics_by_seed.json",metrics[:2])
        with self.assertRaisesRegex(export.ExportError,"All fruit"):
            export.regression_rows(self.root,protocol)

    def test_selected_config_whitelist_blocks_target_and_nested_array_exports(self):
        safe=export.safe_config({"C":10,"source_targets":[99],"mc":{"runs":20,"components":2,"ratio":.1,"targets":[9]},"support_vectors":[8]})
        self.assertEqual(safe,{"C":10,"mc":{"runs":20,"components":2,"ratio":.1}})
        with self.assertRaises(export.ExportError):export.safe_member(self.root,"../../outside")
        history=export.HISTORICAL_CONTEXT
        self.assertAlmostEqual(history["accuracy"]*100,96.0233918128655)
        self.assertEqual(history["correct"],821);self.assertEqual(history["total"],855)

    def test_fruit_model_summary_excludes_private_metadata_and_counts_unique_alias_cost(self):
        records=[];aliases=[];folds=[]
        for outer in (0,1):
            config={"id":"R1_fixed","family":"full_SVR","mc":None,"C":10}
            folds.append({"task":"MA_DM","outer":outer,"choices":{role:{"config":config} for role in ("full_SVR","source_policy")}})
            for seed in export.SEEDS:
                model=self.root/f"models/fold{outer}/seed{seed}.joblib"
                model.parent.mkdir(parents=True,exist_ok=True);model.write_bytes(b"synthetic model bytes; never deserialized")
                meta={"selected_indices":[outer,outer+1],"source_targets":[987654.321],"query_ids":["PRIVATE-FRUIT-ID"]}
                self.write(model.with_suffix('.json'),meta)
                rec={"task":"MA_DM","outer":outer,"seed":seed,"config":config,
                     "estimator_path":str(model.relative_to(self.root)),"metadata_sha256":export.sha(model.with_suffix('.json')),
                     "estimator_sha256":export.sha(model),"elapsed_seconds":2.}
                records.append(rec);aliases.append({"task":"MA_DM","outer":outer,"role":"source_policy","seed":seed,"estimator_path":rec["estimator_path"]})
        for name,items in [("outer_model_records.json",records),("inner_records.json",records),("outer_role_aliases.json",aliases)]:
            self.write(self.root/name,items)
        protocol={"outer_roles":["full_SVR","source_policy"],"scientific_status":"FIXTURE"}
        models,stability,costs=export.selected_models_and_costs(self.root,"fruit",protocol,{"outer_folds":folds})
        self.assertEqual(len(models),12)
        self.assertEqual(sum(r['shared_existing_estimator'] for r in models),6)
        serialized=json.dumps([models,stability,costs])
        self.assertNotIn("PRIVATE-FRUIT-ID",serialized);self.assertNotIn("987654",serialized)
        self.assertNotIn("selected_input_indices",serialized);self.assertNotIn("source_targets",serialized)
        self.assertTrue(all(abs(r['mean_pairwise_selected_column_jaccard']-1/3)<1e-12 for r in stability))
        total=export.total_cost_rows(costs)
        outer=next(r for r in total if r['phase']=='outer_unique_models')
        self.assertEqual(outer['actual_unique_estimator_records'],6)
        self.assertEqual(outer['sum_recorded_trial_seconds'],12.)


if __name__=="__main__":unittest.main()
