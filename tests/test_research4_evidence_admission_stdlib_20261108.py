"""Small synthetic admission/closure checks; no scientific child or Git use."""
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'scripts/research4_evidence_admission_20261108.py'
spec=importlib.util.spec_from_file_location('admission_under_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='evidence-admission-tests-')
        self.root=Path(self.tmp.name);self.base=self.root/'data/research/evidence_validation_20261108_v3'
        self.base.mkdir(parents=True);self.old=self.root/'old-state.json'
        self.patches=[patch.object(m,'ROOT',self.root),patch.object(m,'BASE',self.base),patch.object(m,'OLD_STATE',self.old)]
        for p in self.patches:p.start()
        now=datetime.now(timezone.utc)
        self.config={'schema':m.SCHEMA,'run_id':'evidence-validation-20261108-v3',
            'authorization_observed_utc':(now-timedelta(seconds=1)).isoformat(),
            'deadline_utc':(now+timedelta(hours=47)).isoformat()}
        self.cfg=self.base/'admission_config.json';self.cfg.write_text(json.dumps(self.config))
        self.script=self.root/'scripts/science.py';self.script.parent.mkdir();self.script.write_text('# fake script, never executed\n')
        self.output=self.base/'runs/science001';self.output.mkdir(parents=True)
        self.admission=self.base/'admission/science001'
        self.argv=[sys.executable,'-B',str(self.script),'--output',str(self.output)]
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def old(self,status='running',job='neural_task'):
        obj={'status':status,'owner':{'pid':123,'start_ticks':45},
             'active':{'pid':456,'start_ticks':78,'job_id':job} if job else None}
        self.old.write_text(json.dumps(obj));return obj
    def helper(self):
        b=m.helper()
        b.same_process=lambda identity:True
        return b
    def test_finite_limits(self):
        cfg,end=m.limits(self.config)
        self.assertEqual(cfg['max_attempts'],4);self.assertEqual(cfg['start_cgroup_bytes'],9*m.GIB)
    def test_old_grant_not_accepted(self):
        self.config['run_id']='species-relaunch-20261108-v2'
        with self.assertRaises(RuntimeError):m.limits(self.config)
    def test_memory_limits_cannot_expand(self):
        for key,value in [('start_cgroup_bytes',10*m.GIB),('stop_cgroup_bytes',12*m.GIB),('own_rss_stop_bytes',2*m.GIB)]:
            cfg=copy.deepcopy(self.config);cfg[key]=value
            with self.assertRaises(RuntimeError):m.limits(cfg)
    def test_attempts_cannot_expand(self):
        self.config['max_attempts']=5
        with self.assertRaises(RuntimeError):m.limits(self.config)
    def test_deadline_cannot_expand(self):
        self.config['deadline_utc']=(datetime.fromisoformat(self.config['authorization_observed_utc'])+timedelta(hours=49)).isoformat()
        with self.assertRaises(RuntimeError):m.limits(self.config)
    def test_disk_floor_cannot_reduce(self):
        self.config['minimum_disk_bytes']=m.GIB
        with self.assertRaises(RuntimeError):m.limits(self.config)
    def test_missing_old_state_blocks_not_completes(self):
        self.assertEqual(m.old_queue_reason(self.old,self.helper()),'old_queue_state_unavailable')
    def test_heavy_old_stage_waits(self):
        for name in ['new_grant_snapshot_00625','new_grant_backup_00625','new_summary']:
            self.old.write_text(json.dumps({'status':'running','owner':{},'active':{'job_id':name}}))
            # A real owner identity is mandatory before stage consideration.
            b=self.helper();obj=json.loads(self.old.read_text());obj['owner']={'pid':123};self.old.write_text(json.dumps(obj))
            self.assertEqual(m.old_queue_reason(self.old,b),'old_queue_heavy_summary_or_backup')
    def test_old_missing_pid_does_not_imply_finished(self):
        self.old.write_text(json.dumps({'status':'running','owner':{'pid':999},'active':{'job_id':'neural'}}))
        b=self.helper();b.same_process=lambda identity:False
        self.assertEqual(m.old_queue_reason(self.old,b),'old_queue_identity_unverified')
    def test_real_old_owner_between_jobs_allows_resource_gated_admission(self):
        self.old.write_text(json.dumps({'status':'running','owner':{'pid':123},'active':None}))
        b=self.helper()
        self.assertIsNone(m.old_queue_reason(self.old,b))
        b.same_process=lambda identity:False
        self.assertEqual(m.old_queue_reason(self.old,b),'old_queue_identity_unverified')
    def test_old_unclean_active_blocks(self):
        self.old.write_text(json.dumps({'status':'paused','active':{'pid':999}}))
        self.assertEqual(m.old_queue_reason(self.old,self.helper()),'old_queue_unclean_active_state')
    def test_closed_old_state_allows_resource_check(self):
        self.old.write_text(json.dumps({'status':'completed','active':None}))
        self.assertIsNone(m.old_queue_reason(self.old,self.helper()))
    def test_memory_admission_checks_before_old_state(self):
        cfg,_=m.limits(self.config)
        self.assertEqual(m.admission_reason(cfg,self.helper(),memory=9*m.GIB,free_disk=3*m.GIB),'global_memory_admission')
    def test_actual_scientific_path_is_new_root(self):
        script,out=m.science_output(self.argv)
        self.assertEqual(script,self.script);self.assertEqual(out,self.output)
    def test_old_science_output_rejected(self):
        argv=self.argv[:-1]+['/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2/runs/x']
        with self.assertRaises(RuntimeError):m.science_output(argv)
    def test_shell_command_rejected(self):
        with self.assertRaises(RuntimeError):m.science_output(['/bin/bash','-c','echo fake'])
    def seal(self):
        (self.output/'small.json').write_text('{}\n')
        (self.output/'complete.json').write_text(json.dumps({'status':'complete','artifacts':{'small.json':m.sha(self.output/'small.json')}}))
    def test_completed_artifacts_verified(self):
        self.seal();self.assertEqual(m.validate_complete(self.output)['artifact_count'],1)
    def test_completed_artifact_mutation_rejected(self):
        self.seal();(self.output/'small.json').write_text('changed')
        with self.assertRaises(RuntimeError):m.validate_complete(self.output)
    def test_artifact_escape_rejected(self):
        (self.output/'complete.json').write_text(json.dumps({'status':'complete','artifacts':{'../outside':'bad'}}))
        with self.assertRaises(RuntimeError):m.validate_complete(self.output)
    def test_helper_nonce_distinct_without_source_mutation(self):
        before=m.sha(m.SUPERVISOR);b=self.helper()
        self.assertEqual(b.TOKEN_KEY,'RESEARCH4_EVIDENCE_CHILD_TOKEN');self.assertEqual(m.sha(m.SUPERVISOR),before)
    def fake_run(self, *, interrupted=False, mutate_during_admission=False, mutate_running=False):
        b=self.helper();actual_stat=b.proc_stat
        class FakeProcess:
            pid=424242
            def poll(self):return None if interrupted else 0
            def wait(self,timeout=None):return -15 if interrupted else 0
        def fake_stat(pid):
            if pid==os.getpid():return actual_stat(pid)
            return {'pid':pid,'start_ticks':765,'session':pid,'pgrp':pid,'state':'R','cpu_seconds':0.}
        def no_signal(*args):raise AssertionError('test must never send actual signals')
        calls=[]
        def admit(*args,**kwargs):
            if mutate_during_admission:self.script.write_text('# changed while waiting')
            return None
        def memory():
            calls.append(1)
            if mutate_running and len(calls)>1:self.script.write_text('# changed during fake science')
            return 12*m.GIB if interrupted else m.GIB
        with patch.object(m,'admission_reason',side_effect=admit),patch.object(m,'memory_current',side_effect=memory),\
             patch.object(m,'free_disk',return_value=4*m.GIB),patch.object(m.subprocess,'Popen',return_value=FakeProcess()) as spawn,\
             patch.object(b,'proc_stat',side_effect=fake_stat),patch.object(m,'owned_rows',return_value=[]),\
             patch.object(m,'close_owned') as close,patch.object(b,'signal_owned_groups',side_effect=no_signal):
            code=m.run(self.base,self.admission,self.argv,b)
        return code,spawn.call_count,close.call_count
    def test_fake_success_requires_real_small_file_seals(self):
        self.seal();code,spawns,closes=self.fake_run()
        self.assertEqual((code,spawns,closes),(0,1,0))
        result=json.loads((self.admission/'receipt.json').read_text())
        self.assertEqual(result['status'],'DONE');self.assertTrue(result['closure_confirmed'])
        self.assertEqual(result['attempts'][0]['identity']['pid'],424242)
    def test_four_resource_attempts_then_fail_closed(self):
        code,spawns,closes=self.fake_run(interrupted=True)
        self.assertEqual((code,spawns,closes),(1,4,4))
        result=json.loads((self.admission/'receipt.json').read_text())
        self.assertEqual(result['status'],'BLOCKED');self.assertFalse(result['resume_eligible'])
        self.assertEqual(len(result['attempts']),4)
    def test_completed_invocation_cannot_run_again(self):
        self.seal();self.fake_run()
        with self.assertRaises(RuntimeError):self.fake_run()
    def test_source_drift_between_admission_and_popen_starts_no_child(self):
        code,spawns,closes=self.fake_run(mutate_during_admission=True)
        self.assertEqual((code,spawns,closes),(1,0,0))
        result=json.loads((self.admission/'receipt.json').read_text())
        self.assertIn('changed before spawn',result['error'])
    def test_running_source_drift_closes_once_without_resource_retry(self):
        code,spawns,closes=self.fake_run(interrupted=True,mutate_running=True)
        self.assertEqual((code,spawns,closes),(1,1,1))
        result=json.loads((self.admission/'receipt.json').read_text())
        self.assertEqual(result['attempts'][0]['reason'],'scientific_source_drift')
    def test_nan_prior_cpu_cannot_be_refunded_by_resume(self):
        self.seal();self.fake_run()
        p=self.admission/'state.json';old=json.loads(p.read_text())
        old.update(status='BLOCKED',resume_eligible=True,own_cpu_seconds=float('nan'))
        p.write_text(json.dumps(old))
        with self.assertRaises(RuntimeError):self.fake_run()
    def test_closed_cache_advice_is_globally_rate_limited_for_new_root(self):
        calls=[]
        fake=SimpleNamespace(advise_closed=lambda:calls.append('mock') or {'status':'MOCK_ONLY'})
        ref={'path':'mock','sha256':'a'*64}
        with patch.object(m,'load_closed_cache_advice',return_value=fake),\
             patch.object(m.time,'monotonic',side_effect=[1000.,1001.,1060.]):
            first=m.maybe_advise_closed(self.base,ref,self.helper())
            second=m.maybe_advise_closed(self.base,ref,self.helper())
            third=m.maybe_advise_closed(self.base,ref,self.helper())
        self.assertEqual(calls,['mock','mock']);self.assertIsNone(second)
        self.assertEqual(first['status'],'DONE');self.assertEqual(third['started_monotonic'],1060.)
        clock=json.loads((self.base/'cache-advice/clock.json').read_text())
        self.assertFalse(clock['signals_sent']);self.assertFalse(clock['old_queue_bytes_or_ledger_modified'])
    def test_closed_cache_helper_hash_is_required_before_import(self):
        p=self.root/'scripts/research4_closed_cache_advice_20261108.py'
        p.write_text('raise AssertionError("must not import changed helper")\n')
        with self.assertRaisesRegex(RuntimeError,'path/hash changed'):
            m.load_closed_cache_advice({'path':str(p),'sha256':'0'*64})


if __name__=='__main__':unittest.main()
