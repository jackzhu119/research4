"""Meaningful tiny source-seal, OOF-coverage and publication-policy checks."""
import copy
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'scripts/research4_compact_evidence_20261108.py'
spec=importlib.util.spec_from_file_location('compact_evidence_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MetricsTests(unittest.TestCase):
    def test_unbalanced_metrics_not_accuracy_alias(self):
        x=m.metrics(Counter({('A','A'):8,('A','B'):1,('B','A'):1}))
        self.assertAlmostEqual(x['accuracy'],.8)
        self.assertAlmostEqual(x['balanced_accuracy'],4/9)
        self.assertNotEqual(x['macro_f1'],x['accuracy'])
    def groups(self, *, neural=False, calibrated=False, missing=False):
        descriptor={'kind':'neural'if neural else'matched_classical','method':'M','selection_scope':'primary',
            'k':'inner_selected','representation':'alpha','source_count':190 if neural else 240,
            'auxiliary':'hq_tm'if neural else'','calibration':'weak'if calibrated else'none',
            'target_epochs':192 if neural else 0,
            'shots_per_class':1 if calibrated else 0}
        key=json.dumps(descriptor,sort_keys=True,separators=(',',':'))
        g={key:{}}
        for seed in ([42,43,44]if neural else[20261031,20261108,42]):
            for draw in(range(20)if calibrated else[-1]):
                if missing and seed==44 and draw==19:continue
                g[key][(seed,draw)]={0:{'a':('A','A')},1:{'b':('B','B')},2:{},3:{},4:{}}
        return g
    def test_three_repeats_are_two_unique_specimens(self):
        r=m.aggregate(self.groups(),{'a','b'})[0]
        self.assertEqual(r['status'],'complete');self.assertEqual(r['unique_specimens'],2)
        self.assertEqual(len(r['repeats']),3);self.assertEqual(r['mean_metrics']['accuracy'],1.)
    def test_duplicate_across_folds_not_complete(self):
        g=self.groups();next(iter(g.values()))[(42,-1)][2]['a']=('A','A')
        self.assertEqual(m.aggregate(g,{'a','b'})[0]['status'],'partial')
    def test_missing_outer_fold_not_complete(self):
        g=self.groups();del next(iter(g.values()))[(42,-1)][4]
        self.assertFalse(m.aggregate(g,{'a','b'})[0]['eligible_for_development_selection'])
    def test_all20_draws_included_without_independent_n_inflation(self):
        r=m.aggregate(self.groups(neural=True,calibrated=True),{'a','b'})[0]
        self.assertEqual(r['status'],'complete');self.assertEqual(r['unique_specimens'],2)
        self.assertEqual(r['repeats'][0]['complete_support_draws'],20)
        self.assertEqual(r['repeats'][0]['metrics']['denominator_per_support_draw'],2)
    def test_nineteen_support_draws_not_complete(self):
        r=m.aggregate(self.groups(neural=True,calibrated=True,missing=True),{'a','b'})[0]
        self.assertEqual(r['status'],'partial')
    def test_scope_metadata_is_not_selection_scope(self):
        row={'model':'neural','setting':'source_only','shots':'0'}
        r=m.grouping(row,{'seed':42,'scope':'source-only-not-blind','representation':'alpha512','source_count':190},'neural')
        self.assertEqual(r['selection_scope'],'primary');self.assertEqual(r['calibration'],'source_only')
    def test_block_representation_reported(self):
        self.assertEqual(m.grouping({'family':'concat_nraw_asnv20'},{},'matched_classical')['representation'],'n_alpha')
    def test_target_epoch_budget_has_distinct_group_identity(self):
        row={'model':'neural','setting':'source_only','shots':0}
        a=m.grouping(row,{'target_epochs_actual':144},'neural')
        b=m.grouping(row,{'target_epochs_actual':192},'neural')
        self.assertNotEqual(a,b)
    def test_paired_bootstrap_clusters_three_repeats_by_physical_id(self):
        original=self.groups();oldkey=next(iter(original));descriptor=json.loads(oldkey)
        base=copy.deepcopy(original[oldkey]);descriptor['method']='alpha_snv20'
        basekey=json.dumps(descriptor,sort_keys=True,separators=(',',':'))
        alternative=copy.deepcopy(base)
        for folds in alternative.values():folds[0]['a']=('A','B')
        descriptor['method']='n_snv20';altkey=json.dumps(descriptor,sort_keys=True,separators=(',',':'))
        groups={basekey:base,altkey:alternative};results=m.aggregate(groups,{'a','b'})
        effect=m.paired_classical(groups,results,{'a','b'})[0]
        self.assertEqual(effect['independent_physical_specimens'],2)
        self.assertEqual(effect['repeated_prediction_damaged'],3)
        self.assertEqual(effect['mean_accuracy_delta'],-.5)


class ReadoutFixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.repo=Path(self.tmp.name)
        self.base=self.repo/'data/research/evidence_validation_20261108_v3';self.base.mkdir(parents=True)
        self.originals=m.REPO,m.BASE,m.__file__;m.REPO=self.repo;m.BASE=self.base
        fake=self.repo/'scripts/research4_compact_evidence_20261108.py';fake.parent.mkdir();fake.write_text('fake source hash anchor')
        m.__file__=str(fake)
        (self.base/'control/receipts').mkdir(parents=True)
        (self.base/'control/logs').mkdir()
        self.rows=[]
        for i in range(999):
            self.rows.append({'record_id':f'R{i:04d}','physical_sample_id':f'R{i:04d}','role':'train',
                'species':'酸枣仁'if i<300 else'黄芪','origin_group_id':f'ZY{3+i//60:02d}'if i<300 else'HQ01'})
        with(self.base/'train_only_split.csv').open('w',encoding='utf-8-sig',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        (self.base/'train_only_features.npz').write_bytes(b'not-read-as-scientific-array-only-hashed')
        lock={'training_feature_sha256':m.sha(self.base/'train_only_features.npz'),
            'training_split_sha256':m.sha(self.base/'train_only_split.csv')}
        m.write(self.base/'input_lock.json',lock)
        protocol={'input_lock_sha256':m.sha(self.base/'input_lock.json'),
            'worker_code_sha256':{'scripts/research4_compact_evidence_20261108.py':m.sha(fake)}}
        m.write(self.base/'protocol.json',protocol)
        self.jobs=[];self.state={'jobs':{},'input_hashes':{},'status':'running','budget_cpu_seconds':88000.,'spent_cpu_seconds':1.}
        self.input=self.base/'protocol.json'
    def tearDown(self):
        m.REPO,m.BASE,m.__file__=self.originals;self.tmp.cleanup()
    def prepare(self):
        for seed in [20261031,20261108,42]:
            for fold in range(5):
                name=f'j_{seed}_{fold}';out=self.base/'runs'/name;out.mkdir(parents=True)
                summary={'repeat_seed':seed,'outer_fold':fold,'source_count':240}
                m.write(out/'summary.json',summary)
                with gzip.open(out/'validation_predictions.csv.gz','wt',encoding='utf-8',newline='')as f:
                    cols=['family','k','is_primary_k','record_id','y_true','prediction','repeat_seed','outer_fold']
                    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
                    for k in [31,62]:
                        for i,row in enumerate(self.rows[:300]):
                            if i%5==fold:w.writerow({'family':'alpha_snv20','k':k,'is_primary_k':int(k==(31 if fold%2 else 62)),
                                'record_id':row['record_id'],'y_true':row['origin_group_id'],'prediction':row['origin_group_id'],
                                'repeat_seed':seed,'outer_fold':fold})
                m.write(out/'complete.json',{'status':'complete','artifacts':{p.name:m.sha(p)for p in out.iterdir()}})
                log=self.base/'control/logs'/(name+'.log');log.write_text('closed fixture')
                job={'id':name,'kind':'matched_classical','command':['fake'], 'cwd':str(self.repo),
                    'inputs':[str(self.input)],'outputs':[str(out/p)for p in ['complete.json','summary.json','validation_predictions.csv.gz']]}
                hashes={str(self.input):m.sha(self.input)}
                receipt={'job_sha256':m.object_hash(job),'input_hashes':hashes,
                    'output_hashes':{p:m.sha(p)for p in job['outputs']},'log_path':str(log),'log_sha256':m.sha(log)}
                rp=self.base/'control/receipts'/(name+'.json');m.write(rp,receipt)
                self.jobs.append(job);self.state['jobs'][name]={'status':'completed','receipt_sha256':m.sha(rp)}
                self.state['input_hashes'][name]=hashes
        manifest={'run_id':m.RUN,'cpu_budget_seconds':88000,'protocol_sha256':m.sha(self.base/'protocol.json'),'jobs':self.jobs}
        m.write(self.base/'jobs.json',manifest)
        self.state['manifest_sha256']=m.object_hash({'manifest':manifest,'normalized_jobs':self.jobs,'budget_cpu_seconds':88000.})
        m.write(self.base/'control/state.json',self.state)
    def test_real_sealed_tiny_readout_merges_primary_k_across_folds(self):
        self.prepare();out=self.base/'snapshots/test';m.summarize(self.base,out)
        report=m.read(out/'summary.json');primary=[r for r in report['results']if r['selection_scope']=='primary']
        self.assertEqual(len(primary),1);self.assertEqual(primary[0]['k'],'inner_selected')
        self.assertEqual(primary[0]['status'],'complete');self.assertEqual(primary[0]['unique_specimens'],300)
        self.assertEqual(primary[0]['mean_metrics']['accuracy'],1.)
        self.assertEqual(len(report['results']),3);self.assertLess((out/'summary.json').stat().st_size,2*1024**2)
    def test_output_tamper_rejected(self):
        self.prepare();p=Path(self.jobs[0]['outputs'][1]);p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact|output'):m.summarize(self.base,self.base/'snapshots/bad')
    def test_master_label_mismatch_rejected(self):
        self.prepare();self.rows[0]['origin_group_id']='ZY99'
        # Freeze a genuinely changed TRAIN annotation and corresponding lock;
        # old sealed predictions must still fail truth agreement.
        with(self.base/'train_only_split.csv').open('w',encoding='utf-8-sig',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        lock=m.read(self.base/'input_lock.json');lock['training_split_sha256']=m.sha(self.base/'train_only_split.csv')
        m.write(self.base/'input_lock.json',lock)
        with self.assertRaises(ValueError):m.summarize(self.base,self.base/'snapshots/mismatch')
    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):m.contained(self.repo/'outside.json',self.base)

    def mock_backup(self, change_at=None):
        """Exercise Git transaction checks without any real Git invocation."""
        output=self.base/'backups/test'
        job={'id':'backup_test','kind':'backup','outputs':[str(output/'receipt.json')],'max_retries':0}
        m.write(self.base/'jobs.json',{'jobs':[job]})
        state={'jobs':{},'status':'running','budget_cpu_seconds':88000.,'spent_cpu_seconds':5.,'active':{'job_id':'backup_test'}}
        calls=[];stage=[];head='a'*40;branch_calls=0
        def fake(*args):
            nonlocal head,branch_calls
            calls.append(args)
            if args[:2]==('branch','--show-current'):
                branch_calls+=1
                return ('main'if branch_calls==change_at else m.BRANCH).encode()+b'\n'
            if args[:2]==('remote','get-url'):return b'https://github.com/jackzhu119/research4.git\n'
            if args[:2]==('diff','--cached'):return ('\0'.join(stage)+('\0'if stage else'')).encode()
            if args[:2]==('rev-parse','--show-object-format'):return b'sha1\n'
            if args[:2]==('rev-parse','HEAD'):return head.encode()+b'\n'
            if args[0]=='ls-tree':return b''
            if args[0]=='add':stage.extend(args[2:]);return b''
            if args[0]=='commit':head='b'*40;return b''
            if args[0]=='rev-parse'and':'in args[1]:
                content=(self.repo/args[1].split(':',1)[1]).read_bytes()
                return hashlib.sha1(('blob %d\0'%len(content)).encode()+content).hexdigest().encode()+b'\n'
            if args[0]=='push':return b''
            if args[0]=='ls-remote':return (head+'\trefs/heads/'+m.BRANCH+'\n').encode()
            raise AssertionError(args)
        with patch.object(m,'context',return_value=({'jobs':[job]}, {}, {},state,[job])),patch.object(m,'git',side_effect=fake):
            m.backup(self.base,output,execute=True)
        return m.read(output/'receipt.json'),calls
    def test_publication_uses_verified_commit_not_HEAD_refspec(self):
        report,calls=self.mock_backup()
        self.assertEqual(report['status'],'DONE')
        self.assertIn(('push','origin','b'*40+':refs/heads/'+m.BRANCH),calls)
    def test_switching_to_main_before_commit_blocks_mutation(self):
        report,calls=self.mock_backup(change_at=3)
        self.assertEqual(report['status'],'NOT_BACKED_UP')
        self.assertFalse(any(x[0]in('commit','push')for x in calls))
    def test_switching_to_main_before_push_blocks_publication(self):
        report,calls=self.mock_backup(change_at=4)
        self.assertEqual(report['status'],'NOT_BACKED_UP')
        self.assertFalse(any(x[0]=='push'for x in calls))


class PublicationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.patch=patch.object(m,'REPO',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def test_latest_report_precedes_big_prediction_under_delta_cap(self):
        (self.root/'late.md').write_bytes(b'x'*40);(self.root/'early.csv.gz').write_bytes(b'y'*90)
        candidates={'early.csv.gz':{'priority':4,'bytes':90,'sha256':m.sha(self.root/'early.csv.gz')},
                    'late.md':{'priority':0,'bytes':40,'sha256':m.sha(self.root/'late.md')}}
        with patch.object(m,'DELTA_CAP',100),patch.object(m,'FILE_CAP',100):chosen,_,omitted,total=m.select_delta(candidates,{},'sha1')
        self.assertEqual(set(chosen),{'late.md'});self.assertIn('early.csv.gz',omitted);self.assertEqual(total,40)
    def test_immutable_committed_artifact_cannot_change(self):
        (self.root/'a.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Committed immutable'):m.select_delta({'a.json':{'priority':0,'bytes':2,'sha256':m.sha(self.root/'a.json')}},{'a.json':'wrong'},'sha1')
    def test_repository_allowlist(self):
        self.assertTrue(m.allowed_remote('https://github.com/jackzhu119/research4.git'))
        self.assertFalse(m.allowed_remote('https://github.com/elsewhere/research4.git'))
        self.assertFalse(m.allowed_remote('https://evil.test/jackzhu119/research4'))


if __name__=='__main__':unittest.main()
