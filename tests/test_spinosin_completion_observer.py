"""Pure metadata fixtures. Never touch the live target, Git or quantum code."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import subprocess
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/observe_spinosin_v5_completion_20261104.py"


class CompletionObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s=importlib.util.spec_from_file_location("completion_observer_pure_fixtures",SCRIPT)
        cls.w=importlib.util.module_from_spec(s);s.loader.exec_module(cls.w)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.report=self.path/"reports/v5";self.report.mkdir(parents=True)
        self.addCleanup(patch.stopall)
        patch.object(self.w,"ROOT",self.path).start();patch.object(self.w,"REPORT",self.report).start()
        self.t=datetime(2026,11,4,10,tzinfo=timezone.utc)

    def test_exact_marker_only(self):
        self.assertEqual(self.w.terminal_publication('noise\n'+json.dumps({"status":"DONE_PUBLISHED","commit":"a"*40})),"a"*40)
        self.assertIsNone(self.w.terminal_publication('DONE_PUBLISHED abc\n'+json.dumps({"wrapper":{"status":"DONE_PUBLISHED","commit":"a"*40}})))

    def test_malformed_or_duplicate_marker_rejected(self):
        for text in (json.dumps({"status":"DONE_PUBLISHED","commit":"bad"}),
                     json.dumps({"status":"DONE_PUBLISHED","commit":"a"*40,"extra":True}),
                     '\n'.join([json.dumps({"status":"DONE_PUBLISHED","commit":"a"*40})]*2)):
            with self.subTest(text=text),self.assertRaises(ValueError):self.w.terminal_publication(text)

    def test_unpublished_done_keeps_waiting(self):
        self.assertEqual(self.w.poll_decision({"status":"DONE"},"",self.t+timedelta(seconds=10),self.t),("WAIT",None))

    def test_partial_job_stops(self):
        self.assertEqual(self.w.poll_decision({"status":"PARTIAL"},"",self.t+timedelta(seconds=10),self.t)[0],"BLOCKED_PARTIAL_JOB")

    def test_timeout_stops(self):
        self.assertEqual(self.w.poll_decision({"status":"RUNNING"},"",self.t,self.t)[0],"BLOCKED_OBSERVATION_DEADLINE")

    def test_terminal_marker_requires_done_json(self):
        marker=json.dumps({"status":"DONE_PUBLISHED","commit":"a"*40})
        with self.assertRaisesRegex(ValueError,"disagree"):
            self.w.poll_decision({"status":"RUNNING"},marker,self.t+timedelta(seconds=10),self.t)

    def test_deadline_does_not_expand_ledger(self):
        old=self.path/'old.json';old.write_text(json.dumps({"run_deadline_at_utc":self.t.isoformat()}))
        ledger={"migration":{"source_v4_path":str(old),"source_v4_sha256":self.w.sha(old)},"run_deadline_at_utc":self.t.isoformat()}
        self.assertEqual(self.w.observation_deadline(ledger,{"authorized_end_at_utc":(self.t+timedelta(days=3)).isoformat()}),self.t+timedelta(seconds=600))
        self.assertEqual(self.w.observation_deadline(ledger,{"authorized_end_at_utc":(self.t+timedelta(seconds=100)).isoformat()}),self.t+timedelta(seconds=100))
        ledger['run_deadline_at_utc']=(self.t+timedelta(hours=1)).isoformat()
        with self.assertRaisesRegex(ValueError,"deadline changed"):self.w.observation_deadline(ledger,{"authorized_end_at_utc":self.t.isoformat()})

    def test_git_commit_must_match_local_and_remote(self):
        git=Mock(side_effect=[self.w.BRANCH,'a'*40,'a'*40+' refs/heads/'+self.w.BRANCH,''])
        self.w.verify_git('a'*40,SimpleNamespace(git=git));self.assertEqual(git.call_count,4)

    def test_git_wrong_local_head_stops_before_remote(self):
        git=Mock(side_effect=[self.w.BRANCH,'b'*40])
        with self.assertRaisesRegex(ValueError,'current research HEAD'):self.w.verify_git('a'*40,SimpleNamespace(git=git))
        self.assertEqual(git.call_count,2)

    def fixture_job(self):
        w=self.w
        snap=self.report/'source/manifest.json';snap.parent.mkdir();snap.write_text('{}')
        grad=self.report/'gradient.json';grad.write_text('{"synthetic":"no molecular arrays"}')
        gate=self.report/'gate.json';gate.write_text('{"synthetic_gate":true}')
        patch.object(w,'SNAPSHOT_SHA',w.sha(snap)).start();patch.object(w,'GRADIENT_SHA',w.sha(grad)).start();patch.object(w,'GATE_SHA',w.sha(gate)).start()
        manifest={name:w.record(path) for name,path in [('snapshot_manifest',snap),('gradient_check',grad),('scientific_gate',gate)]}
        result_dir=self.report/'auto-segments/round-001/segment-final';result_dir.mkdir(parents=True)
        result={"status":"FREQUENCIES_COMPUTED","frequencies_status":"DONE","normal_mode_count":219,"atoms":75,
                "frequencies_cm1":[-1.]+[float(i+1) for i in range(218)],"complete_RHF_helper":True,"complete_global_Hessian":True,
                "stable_MF_state_unchanged":True,"snapshot_manifest_sha256":w.SNAPSHOT_SHA,"source_worker":{"sha256":w.CONSUMER_SHA},
                "IR_intensities_status":"NOT_COMPUTED","resource_supervision":{},"phase":"hessian-segment","boot_id":"fixture"}
        result_path=result_dir/'result.json';result_path.write_text(json.dumps(result))
        controller={"status":"DONE","completed_units":150,"total_units":150,"source_sha256":w.CONTROLLER_SHA,
                    "worker_sha256":w.CONSUMER_SHA,"snapshot_manifest_sha256":w.SNAPSHOT_SHA,"final_result":w.record(result_path),
                    "cache_directory":str(self.path/'outputs/pyscf-spinosin-resumable-20261104-v5/spinosin-C001-restart')}
        controller_path=result_dir.parent/'result.json';controller_path.write_text(json.dumps(controller))
        (self.report/'cpu_budget.json').write_text(json.dumps({"phases":[{"status":"EXITED","actual_cpu_seconds":1.}],"consumed_cpu_seconds":1.}))
        status={"status":"DONE","manifest_sha256":w.JOB_MANIFEST_SHA,"round":1,"final_result":controller['final_result']}
        worker=SimpleNamespace(EXECUTION_SCHEMA='fixture',verify_closed_execution=Mock(return_value={"closed":True}),
                               verify_cached_unit_producers=Mock(return_value=[{"qualification":"CLOSED_HEALTHY_V5_PRODUCER"}]*150),
                               qualify_gradient_receipt=Mock(return_value={}),state_module=lambda:SimpleNamespace(validate_closed_producer=Mock(return_value={"closed":True})))
        patch.object(w,'load_module',return_value=worker).start()
        return status,manifest,result_path,result,worker

    def test_signed_complete_job_qualifies_without_array_reads(self):
        status,manifest,_,_,worker=self.fixture_job()
        actual=self.w.qualify_closed_job(status,manifest)
        self.assertEqual(actual['healthy_RHF_unit_producer_count'],150)
        worker.verify_closed_execution.assert_called_once()
        self.assertEqual(list(self.path.rglob('*.npy')),[])

    def test_active_quantum_blocks(self):
        status,manifest,_,_,_=self.fixture_job()
        (self.report/'cpu_budget.json').write_text(json.dumps({"phases":[{"status":"RUNNING","actual_cpu_seconds":None}]}))
        with self.assertRaisesRegex(ValueError,'still active'):self.w.qualify_closed_job(status,manifest)

    def test_incomplete_result_blocks_before_loader(self):
        status,manifest,path,result,worker=self.fixture_job();result['normal_mode_count']=218;path.write_text(json.dumps(result))
        status['final_result']=self.w.record(path);controller_path=path.parent.parent/'result.json';controller=json.loads(controller_path.read_text());controller['final_result']=status['final_result'];controller_path.write_text(json.dumps(controller))
        with self.assertRaisesRegex(ValueError,'219 modes'):self.w.qualify_closed_job(status,manifest)
        worker.verify_closed_execution.assert_not_called()

    def test_unqualified_forensic_unit_blocks(self):
        status,manifest,_,_,worker=self.fixture_job();worker.verify_cached_unit_producers.return_value[-1]={"qualification":"EXPLICIT_ROOT_FORENSIC_RELEASE"}
        with self.assertRaisesRegex(ValueError,'no automatic forensic'):self.w.qualify_closed_job(status,manifest)

    def analysis(self,qualified=False):
        out=self.path/'analysis';out.mkdir()
        checks={"schema":"research4-V5-independent-modes-analysis-v1","status":"DONE_COMPUTED_MODES_AUDITED","case":"Spinosin",
                "source_result_sha256":"a"*64,"normal_modes":219,"snapshot_manifest_sha256":self.w.SNAPSHOT_SHA,
                "analysis_source_sha256":self.w.ANALYZER_SHA,"IR_intensities_status":"NOT_COMPUTED","chemical_attribution_verified":False,
                "PySCF_imported":False,"Spinosin_target_status":"ACTUAL_CLOSED_FREQUENCIES_AUDITED","new_quantum_evaluations":0,
                "theory_or_source_peaks_enter_classifier":False,"qualified_for_mechanism":qualified,
                "postprocessing_environment":{"CPU_affinity":[0],"BLAS_threads":"1","max_virtual_address_space_bytes":1024**3},
                "peak_rss_kib":1024}
        (out/'checks.json').write_text(json.dumps(checks));(out/'signed.csv').write_text('synthetic_fixture_only\n')
        index={"analysis_source_sha256":self.w.ANALYZER_SHA,"files":[{"path":"signed.csv","sha256":self.w.sha(out/'signed.csv')}]}
        (out/'artifact_index.json').write_text(json.dumps(index));return out,checks

    def test_computed_negative_modes_do_not_require_mechanism_claim(self):
        out,_=self.analysis(False);actual=self.w.verify_analysis(out,'a'*64)
        self.assertFalse(actual['qualified_for_mechanism'])

    def test_unqualified_minimum_cannot_get_peak_pairing(self):
        out,_=self.analysis(False);(out/'all_experimental_theory_pairs.csv').write_text('invalid')
        with self.assertRaisesRegex(ValueError,'must not receive'):self.w.verify_analysis(out,'a'*64)

    def test_output_tamper_rejected(self):
        out,_=self.analysis();(out/'signed.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.w.verify_analysis(out,'a'*64)

    def test_control_analysis_cannot_be_target_success(self):
        out,checks=self.analysis();checks['case']='water_software_control';(out/'checks.json').write_text(json.dumps(checks))
        with self.assertRaisesRegex(ValueError,'descriptive audit'):self.w.verify_analysis(out,'a'*64)

    def test_expired_subprocess_never_launches(self):
        with patch.object(self.w,'now',return_value=self.t),patch.object(self.w.subprocess,'Popen') as popen:
            with self.assertRaises(TimeoutError):self.w.subprocess_until(['never'],self.path/'never.log',self.t)
            popen.assert_not_called()

    def test_reaped_subprocess_interrupt_never_signals_reusable_pid(self):
        child=Mock();child.returncode=0;child.wait.side_effect=KeyboardInterrupt
        with patch.object(self.w,'now',return_value=self.t),patch.object(self.w.subprocess,'Popen',return_value=child),patch.object(self.w.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):
                self.w.subprocess_until(['synthetic'],self.path/'synthetic.log',self.t+timedelta(seconds=10))
            kill.assert_not_called()

    def test_unreaped_subprocess_timeout_targets_only_private_group(self):
        child=Mock();child.pid=99999;child.returncode=None;child.wait.side_effect=[subprocess.TimeoutExpired('synthetic',5),0]
        with patch.object(self.w,'now',return_value=self.t),patch.object(self.w.subprocess,'Popen',return_value=child),patch.object(self.w.os,'killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.w.subprocess_until(['synthetic'],self.path/'synthetic.log',self.t+timedelta(seconds=10))
            self.assertEqual(kill.call_args.args[0],99999)
            self.assertEqual(child.wait.call_count,2)

    def test_readonly_git_uses_remaining_deadline_with_cleanup_reserve(self):
        with patch.object(self.w,'now',return_value=self.t),patch.object(self.w.subprocess,'check_output',return_value='synthetic\n') as query:
            self.assertEqual(self.w.readonly_git(self.t+timedelta(seconds=10),'rev-parse','HEAD'),'synthetic')
            self.assertEqual(query.call_args.kwargs['timeout'],5)

    def test_narrow_observer_scope_rejects_quantum_and_repo_roots(self):
        for path in (self.path,self.report,self.report/'auto-segments/round-001',self.report/'v5-source-state'):
            with self.subTest(path=path),self.assertRaises(ValueError):self.w.narrow_observer_directory(path)
        self.assertEqual(self.w.narrow_observer_directory(self.report/'completion-observer-001'),self.report/'completion-observer-001')

    def test_report_keeps_limits_and_classifier(self):
        checks={'imaginary_frequency_count':2,'zero_frequency_count':0,'low_frequency_mode_count':5,'qualified_for_mechanism':False}
        text=self.w.report_text({'status':'MODES_AUDITED_READY_FOR_PUBLICATION'},checks)
        self.assertIn('虚频2个',text);self.assertIn('无D3',text);self.assertIn('96.0234%',text);self.assertIn('NOT_COMPUTED',text)


if __name__=='__main__':unittest.main()
