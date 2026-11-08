"""Tiny source-proof/cache/OOF checks; no real scientific fit or Git."""
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('compact_readout.py')
spec=importlib.util.spec_from_file_location('compact_runtime_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'runtime_repo';self.root.mkdir()
        self.old=Path(self.temp.name)/'old_run';self.old.mkdir()
        (self.root/'evidence').mkdir();(self.root/'reports').mkdir()
        (self.old/'control/receipts').mkdir(parents=True)
        self.patch=patch.object(m,'REPO',self.root);self.patch.start()
        self.rows=[]
        specs=[('酸枣仁',300,'ZY'),('黄芪',241,'HQ'),('天麻',242,'TM'),('枣仁',216,'ZR')]
        for species,count,prefix in specs:
            for i in range(count):
                groups=5 if species=='酸枣仁' else 4
                self.rows.append({'record_id':f'{prefix}-{i:04d}','physical_sample_id':f'{prefix}-{i:04d}',
                    'species':species,'origin_group_id':f'{prefix}{i%groups:02d}','role':'train'})
        self.split=self.old/'train_only_split.csv'
        with self.split.open('w',encoding='utf-8-sig',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        self.input=self.old/'source.py';self.input.write_text('immutable scientific source')
        self.prior={'jobs':{},'input_hashes':{}};self.jobs=[]
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def make_job(self,seed=42,fold=0,missing_arm=False):
        name=f'neural_seed{seed}_fold{fold}';out=self.old/'runs'/name;out.mkdir(parents=True)
        truth={r['record_id']:r['origin_group_id']for i,r in enumerate(self.rows[:300])if i%5==fold}
        with gzip.open(out/'validation_predictions.csv.gz','wt',encoding='utf-8',newline='')as f:
            header=['arm','method','repeat','shots','record_id','predicted','score_ZY00']
            w=csv.DictWriter(f,fieldnames=header);w.writeheader()
            for arm in m.ARMS[:-1]if missing_arm else m.ARMS:
                for rid,label in truth.items():
                    w.writerow({'arm':arm,'method':'source_only','repeat':-1,'shots':0,
                        'record_id':rid,'predicted':label,'score_ZY00':1})
                    # Out-of-pool calibration rows must not enter base metrics.
                    w.writerow({'arm':arm,'method':'legacy_lambda1','repeat':0,'shots':1,
                        'record_id':'NOT_A_QUERY','predicted':'not_a_source_class','score_ZY00':0})
        m.write_new(out/'summary.json',{'status':'complete'})
        m.write_new(out/'validation_truth.json',truth)
        m.write_new(out/'complete.json',{'status':'complete','artifacts':{
            n:m.sha(out/n)for n in ['summary.json','validation_predictions.csv.gz','validation_truth.json']}})
        job={'id':name,'kind':'neural','command':['python','unchanged_worker.py','--target','suanzaoren',
             '--profile','p24_e192_full','--pretrain-steps','480','--seed',str(seed),'--outer-fold',str(fold),
             '--split',str(self.split),'--output',str(out)],'cwd':str(self.old),
             'inputs':[str(self.input)],'outputs':[str(out/n)for n in ['complete.json','summary.json','validation_predictions.csv.gz']],
             'max_retries':1}
        hashes={str(self.input):m.sha(self.input)}
        receipt={'job_sha256':m.objhash(job),'input_hashes':hashes,'output_hashes':{n:m.sha(n)for n in job['outputs']}}
        rp=self.old/'control/receipts'/(name+'.json');m.write_new(rp,receipt)
        self.prior['jobs'][name]={'status':'completed','receipt_sha256':m.sha(rp)};self.prior['input_hashes'][name]=hashes
        self.jobs.append(job);return job
    def finish(self):
        m.write_new(self.root/'original_manifest.json',{'jobs':self.jobs,'protocol_path':str(self.old/'protocol.json')})
        m.write_new(self.root/'evidence/prior_inner_closed.json',self.prior)
    def test_full_three_seed_fivefold_base_pooled_only(self):
        for seed in (42,43,44):
            for fold in range(5):self.make_job(seed,fold)
        self.finish();out=self.root/'reports/first';value=m.summarize(self.root,out)
        report=m.read(out/'summary.json')
        self.assertEqual(value['new_stream_scans'],15);self.assertEqual(len(report['results']),4)
        for result in report['results']:
            self.assertEqual(result['status'],'complete');self.assertEqual(result['independent_specimens'],300)
            self.assertEqual(result['mean_metrics']['accuracy'],1.)
            self.assertEqual(result['per_seed'][0]['metrics']['denominator'],300)
        self.assertLess((out/'summary.json').stat().st_size,m.SUMMARY_CAP)
    def test_second_report_reuses_sealed_base_without_opening_gzip(self):
        self.make_job();self.finish();m.summarize(self.root,self.root/'reports/first')
        with patch.object(m,'HashingReader',side_effect=AssertionError('no repeated decompression')):
            result=m.summarize(self.root,self.root/'reports/second')
        self.assertEqual(result['cache_hits'],1);self.assertEqual(result['new_stream_scans'],0)
    def test_missing_arm_is_not_a_closed_base_evaluation(self):
        self.make_job(missing_arm=True);self.finish()
        with self.assertRaisesRegex(ValueError,'arm lacks exact'):m.summarize(self.root,self.root/'reports/bad')
    def test_model_source_change_invalidates_old_closure(self):
        self.make_job();self.finish();self.input.write_text('different source')
        with self.assertRaisesRegex(ValueError,'input changed'):m.summarize(self.root,self.root/'reports/bad')
    def test_changed_prediction_stat_triggers_rehash_and_failure(self):
        job=self.make_job();self.finish();m.summarize(self.root,self.root/'reports/first')
        path=Path(m.argument(job,'--output'))/'validation_predictions.csv.gz'
        path.write_bytes(path.read_bytes()+b'\x00')
        with self.assertRaisesRegex(ValueError,'hash/size differs'):m.summarize(self.root,self.root/'reports/bad')
    def test_cache_content_tamper_rejected(self):
        job=self.make_job();self.finish();m.summarize(self.root,self.root/'reports/first')
        path=self.root/'readout_cache'/(job['id']+'.json');value=m.read(path);value['rows'][0][3]='WRONG'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'cache was altered'):m.summarize(self.root,self.root/'reports/bad')
    def test_remaining_scientific_descriptor_cannot_change(self):
        job=self.make_job();self.finish();changed={**job,'max_retries':0}
        m.write_new(self.root/'remaining_jobs.json',{'jobs':[changed]})
        with self.assertRaisesRegex(ValueError,'descriptor differs'):m.load_context(self.root)
    def test_same_job_cannot_be_completed_twice_across_ledgers(self):
        job=self.make_job();self.finish();(self.root/'work/control').mkdir(parents=True)
        m.write_new(self.root/'work/control/state.json',{'jobs':{job['id']:{'status':'completed'}},'input_hashes':{}})
        with self.assertRaisesRegex(ValueError,'completed twice'):m.summarize(self.root,self.root/'reports/bad')
    def test_initial_snapshot_without_current_state_is_supported(self):
        self.make_job();self.finish();context=m.load_context(self.root)
        self.assertEqual(context[4]['jobs'],{})


class MetricAndSafetyTests(unittest.TestCase):
    def test_imbalanced_balanced_accuracy_is_not_total_accuracy(self):
        result=m.metrics(Counter({('a','a'):8,('a','b'):1,('b','a'):1}),['a','b'])
        self.assertEqual(result['accuracy'],.8);self.assertAlmostEqual(result['balanced_accuracy'],4/9)
    def test_duplicate_across_folds_is_partial(self):
        groups={('suanzaoren','p',480,'A'):{42:{0:{'one':('a','a')},1:{'one':('a','a')},2:{},3:{},4:{}}}}
        result=m.aggregate(groups,{'suanzaoren':{'one':'a'}})[0]
        self.assertEqual(result['status'],'partial');self.assertIsNone(result['per_seed'][0]['metrics'])
    def test_remote_allowlist(self):
        self.assertTrue(m.remote_allowed('https://github.com/jackzhu119/research4.git'))
        self.assertFalse(m.remote_allowed('https://github.com/another/research4.git'))
        self.assertFalse(m.remote_allowed('https://evil.test/jackzhu119/research4'))
    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):m.inside('/outside',Path('/approved'))


if __name__=='__main__':unittest.main()
