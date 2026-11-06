"""Finite postrun observer safety/identity tests; no Q, Git or subprocess work."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research4_postrun_analysis_watch_20261106.py'
SPEC=importlib.util.spec_from_file_location('postrun_watch_stdlib',SCRIPT)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class PostrunWatchSafety(unittest.TestCase):
    def fixture(self,root,success=True):
        work=root/'work';work.mkdir();runtime=root/'.runtime';runtime.mkdir()
        wrapper={'pid':10,'start_ticks':20,'uid':1000,'boot_id':'fixture-boot-only'}
        queueid={'pid':11,'start_ticks':21,'uid':1000,'boot_id':'fixture-boot-only'}
        qi={'id':'fixture-q1','status':'COMPLETED_WITH_BLOCKED_JOBS','ended_at_utc':'2026-11-06T00:00:00Z','identity':queueid}
        q={'config_sha256':'cfg','grant_sha256':'grant','invocations':[qi],'jobs':{'a':{'status':'DONE_SEALED'}}}
        qp=work/'queue-state.json';qp.write_text(json.dumps(q))
        w={'identity':wrapper,'source_sha256':'wrapper-sha','queue_invocations':[{'attempt':'fixture-q1','identity':queueid}],
           'status':qi['status'],'quantum_phases_closed':True,'ended_at_utc':'2026-11-06T00:00:01Z'}
        wp=work/'wrapper-state.json';wp.write_text(json.dumps(w))
        latest={**w,'actual_upload_succeeded':success};lp=runtime/'latest.json';lp.write_text(json.dumps(latest))
        manifest=work/'manifest.json';manifest.write_text('{}')
        grant=work/'grant.json';grant.write_text(json.dumps({'opaque_process_identity_exceptions':[]}))
        cfg={'deadline_utc':M.DEADLINE,'wrapper_state':str(wp),'wrapper_latest':str(lp),'wrapper_source':{'sha256':'wrapper-sha'},
             'queue_config':{'sha256':'cfg'},'queue_grant':{'path':str(grant),'sha256':'grant'},
             'original_return_manifest':{'sha256':'original-manifest-sha'},'uploader_source':{'sha256':'uploader-sha'},
             'return_manifest':{'path':str(manifest),'sha256':'manifest'}}
        launch={'identity':wrapper};qconfig={'state':str(qp)}
        if success:
            receipt=work/'git-return/fixture-q1-upload1/return_receipt.json';receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(self.receipt()))
        return work,cfg,qconfig,launch,wp,qp,lp

    def receipt(self):
        return {'status':'DONE_ACTUAL_PUSH_AND_REMOTE_REF_VERIFIED','science_commit':'a'*40,
                'manifest_sha256':'original-manifest-sha','source_helper_sha256':'uploader-sha','actual_result_upload':'DONE',
                'all_committed_scientific_blobs_match':True,'main_modified':False,'force_push':False,
                'branch':'research/pyscf-spinosin-mechanism-20261031'}

    def observe_fixture(self,root,values,process='GONE'):
        work,cfg,qconfig,launch,*_=values
        uploader=SimpleNamespace(require_no_active_quantum=Mock(return_value={'no_active_quantum_observed':True}))
        with patch.object(M,'WORK',work),patch.object(M,'ROOT',root),patch.object(M,'checked',side_effect=lambda r:Path(r['path'])),\
             patch.object(M,'identity_state',return_value=process),patch.object(M,'load_uploader',return_value=uploader),\
             patch.object(M,'require_no_live_git'),patch.object(M.subprocess,'run',side_effect=AssertionError('No subprocess during observation')),\
             patch.object(M.subprocess,'Popen',side_effect=AssertionError('No launch during observation')):
            return M.observe(cfg,qconfig,launch),uploader

    def test_import_is_stdlib_and_does_not_import_scientific_engines(self):
        self.assertFalse(any(x.split('.')[0] in {'numpy','pyscf','scipy','torch'} for x in sys.modules))
        self.assertEqual(M.aware(M.DEADLINE),M.aware('2026-11-09T08:25:17.730193Z'))

    def test_birth_uid_boot_and_zombie_are_distinguished_from_pid_absence(self):
        expected={'pid':1,'start_ticks':2,'uid':1000,'boot_id':'original'}
        for state,want in [('R','LIVE'),('S','LIVE'),('Z','CLOSED_ZOMBIE'),('X','CLOSED_ZOMBIE')]:
            self.assertEqual(M.identity_state(expected,lambda p:{**expected,'state':state}),want)
        def missing(pid):raise FileNotFoundError()
        self.assertEqual(M.identity_state(expected,missing),'GONE')
        for key,value in [('pid',2),('start_ticks',3),('uid',0),('boot_id','other')]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                M.identity_state(expected,lambda p:{**expected,key:value,'state':'R'})

    def test_live_original_queue_only_waits_and_does_not_run_closure_or_git(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=self.fixture(root)
            result,uploader=self.observe_fixture(root,values,'LIVE')
            self.assertEqual(result['status'],'WAIT_ORIGINAL_QUEUE_ALIVE')
            uploader.require_no_active_quantum.assert_not_called()

    def test_real_terminal_wrapper_and_verified_prior_upload_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=self.fixture(root)
            result,uploader=self.observe_fixture(root,values)
            self.assertEqual(result['status'],'READY_ANALYSIS_AND_RETURN')
            self.assertTrue(result['original_upload_succeeded'])
            self.assertEqual(len(result['original_upload_receipts']),1)
            uploader.require_no_active_quantum.assert_called_once()
            for path in values[0].glob('git-return/*/return_receipt.json'):path.unlink()
            with self.assertRaises(RuntimeError):self.observe_fixture(root,values)

    def test_explicit_original_upload_failure_allows_readonly_analysis_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=self.fixture(root,False)
            result,_=self.observe_fixture(root,values)
            self.assertEqual(result['status'],'READY_ANALYSIS_ORIGINAL_UPLOAD_FAILED')
            self.assertFalse(result['original_upload_succeeded'])
            self.assertEqual(result['original_upload_receipts'],[])

    def test_missing_or_stale_original_upload_state_is_not_assumed_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=self.fixture(root)
            latest=values[-1];j=json.loads(latest.read_text());j.pop('actual_upload_succeeded');latest.write_text(json.dumps(j))
            with self.assertRaises(RuntimeError):self.observe_fixture(root,values)
            j['actual_upload_succeeded']=True;j['identity']['start_ticks']=999;latest.write_text(json.dumps(j))
            with self.assertRaises(RuntimeError):self.observe_fixture(root,values)

    def test_real_git_receipts_require_exact_SHA_helper_manifest_and_scope(self):
        value=self.receipt();M.verify_git_receipt(value,'original-manifest-sha','uploader-sha')
        for key,bad in [('science_commit','planned-sha'),('science_commit','a'*39),('source_helper_sha256','wrong'),
                        ('manifest_sha256','wrong'),('actual_result_upload','NOT_RUN'),('main_modified',True),
                        ('force_push',True),('all_committed_scientific_blobs_match',False),('branch','main')]:
            with self.subTest(key=key,bad=bad),self.assertRaises(RuntimeError):
                M.verify_git_receipt({**value,key:bad},'original-manifest-sha','uploader-sha')

    def test_pid_absence_never_substitutes_for_actual_queue_terminal_status(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=self.fixture(root)
            qp=values[-2];j=json.loads(qp.read_text());j['invocations'][-1]['status']='RUNNING';qp.write_text(json.dumps(j))
            with self.assertRaises(RuntimeError):self.observe_fixture(root,values)

    def test_duplicate_observers_cannot_hold_same_lock(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'watch'
            with M.sole_watch_lock(folder):
                with self.assertRaises(BlockingIOError):
                    with M.sole_watch_lock(folder):self.fail('duplicate observer entered')
            with M.sole_watch_lock(folder):pass

    def test_default_check_does_not_execute_or_create_attempt(self):
        cfg,qc,grant,launch={'fixture':True},{},{},{}
        argv=['watch','--config','/fixture/root.json','--config-sha256','0'*64]
        with patch.object(sys,'argv',argv),patch.object(M,'validate_config',return_value=(cfg,qc,grant,launch)),\
             patch.object(M,'observe',return_value={'status':'WAIT_ORIGINAL_WRAPPER_ALIVE'}),\
             patch.object(M,'execute',side_effect=AssertionError('No actual watcher in default check')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(M.main(),0)
        self.assertIn('CHECK_ONLY_NO_WRITES_NO_SUBPROCESSES',out.getvalue())

    def test_no_quantum_scheduler_install_or_signalling_api_is_present(self):
        tree=ast.parse(SCRIPT.read_text())
        forbidden={'kill','killpg','pidfd_send_signal','system','execv','execve'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,forbidden)
        text=SCRIPT.read_text()
        self.assertNotIn('automations_create',text)
        self.assertNotIn('pip install',text)
        self.assertIn('range(1,4)',text)
        self.assertIn('RLIMIT_CPU,(120,150)',text)
        self.assertIn('signal.alarm(',text)


if __name__=='__main__':unittest.main()
