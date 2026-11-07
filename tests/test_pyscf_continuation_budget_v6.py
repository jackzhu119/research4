"""Synthetic V6 registration/closure tests; no real ledger, payload or QM imports."""
from argparse import Namespace
import ast
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6.py'


class V6ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('v6_continuation_synthetic', SCRIPT)
        cls.r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.r)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.old_path = self.path/'closed_v5.json'
        self.auth_path = self.path/'explicit_root_decision.json'
        self.new_path = self.path/'v6.json'
        self.now = datetime(2026,11,4,23,tzinfo=timezone.utc)
        self.old_auth = {
            'guard_semantics':self.r.GUARD_SEMANTICS,
            'rss_stop_bytes':10*1024**3,'cgroup_kill_bytes':12*1024**3,
            'monitor_interval_seconds':.1,'memory_kill_grace_seconds':1.,
            'expected_cgroup_memory_max_bytes':16*1024**3,
            'actual_cpu_cleanup_reserve_seconds':120,'cpu_guard_kill_grace_seconds':1.,
            'actual_cpu_guard_source':self.r.ACTUAL_CPU_GUARD_SOURCE,
            'prior_closed_ledger_path':str(self.path/'closed_v3.json'),
            'prior_closed_ledger_sha256':'3'*64,'prior_consumed_cpu_seconds':70.,
            'prior_budget_is_historical_only':True}
        self.source = {
            'schema':'research4-pyscf-actual-cpu-budget-v5','runner_sha256':self.r.V5_SHA,
            'created_at_utc':'2026-11-04T07:20:50+00:00',
            'run_deadline_at_utc':'2026-11-05T01:20:50+00:00',
            'quantum_cpu_cap_seconds':129600,'run_wall_cap_seconds':64800,
            'consumed_cpu_seconds':37562.5,'remaining_quantum_cpu_seconds':92037.5,
            'source_checkpoint_path':str(self.path/'closed_source.chk'),
            'source_checkpoint_sha256':'c'*64,'source_base_commit':'a'*40,
            'authorization_path':str(self.path/'v5_auth.json'),'authorization_sha256':'e'*64,
            'prior_closed_ledger':{'path':str(self.path/'closed_v3.json'),'sha256':'3'*64},
            'prior_consumed_cpu_seconds':70.,'prior_budget_is_historical_only':True,
            'migration':{'source_v4_path':str(self.path/'closed_v4.json'),'source_v4_sha256':'4'*64},
            'phases':[
                {'phase':'v5_good','status':'EXITED','exit_code':0,
                 'ended_at_utc':'2026-11-04T10:00:00+00:00','actual_cpu_seconds':26.5},
                {'phase':'v5_failed_expensive','status':'PARTIAL_OR_FAILED','exit_code':-15,
                 'ended_at_utc':'2026-11-04T22:00:00+00:00','actual_cpu_seconds':37536.,
                 'cpu_terminated':True,'monitor_failed':False}]}
        self.write(self.old_path,self.source)
        self.auth = {
            **self.old_auth,'schema':self.r.AUTH_SCHEMA,'root_approved_for_execution':True,
            'no_budget_reset':True,'no_automatic_budget_growth':True,
            'v6_supervisor_source_sha256':self.r.sha(SCRIPT),
            'authorization_text':'SYNTHETIC TEST ONLY: explicitly bounded root allocation.',
            'research_authorization_evidence':'SYNTHETIC recorded evidence; never an actual user grant.',
            'research_authorization_deadline':'2026-11-06T00:00:00+00:00',
            'created_at_utc':'2026-11-04T22:01:00+00:00',
            'source_v5_ledger_path':str(self.old_path),'source_v5_ledger_sha256':self.r.sha(self.old_path),
            'source_v5_runner_sha256':self.r.V5_SHA,
            'source_v5_consumed_cpu_seconds':self.source['consumed_cpu_seconds'],
            'original_v5_created_at_utc':self.source['created_at_utc'],
            'original_v5_run_deadline_at_utc':self.source['run_deadline_at_utc'],
            'original_v5_quantum_cpu_cap_seconds':129600,'original_v5_run_wall_cap_seconds':64800,
            'original_v5_authorization_path':self.source['authorization_path'],
            'original_v5_authorization_sha256':self.source['authorization_sha256'],
            'source_base_commit':'b'*40,'additional_cpu_seconds':43200,'deadline_extension_seconds':43200,
            'continuation_ledger_path':str(self.new_path),
            'max_quantum_cpu_seconds':172800,'max_run_wall_seconds':108000,
            'max_phase_cpu_seconds':122400,'max_phase_wall_seconds':43200,
            'max_threads':4,'max_memory_gib':24}
        self.write(self.auth_path,self.auth)
        for target, value in (
            ('frozen_v5_module',lambda:SimpleNamespace(validate_ledger=lambda source:deepcopy(self.old_auth))),
            ('validate_prior_closed_ledger',lambda auth:{'synthetic_only':True}),
            ('CONTINUATION_REGISTRY',self.path/'isolated_synthetic_registry'),
            ('utc',lambda:self.now)):
            p=patch.object(self.r,target,value);p.start();self.addCleanup(p.stop)

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value,sort_keys=True)+'\n')

    def args(self):
        return Namespace(ledger=self.new_path,authorization=self.auth_path,
            authorization_sha256=self.r.sha(self.auth_path),source_v5_ledger=self.old_path,
            source_v5_sha256=self.r.sha(self.old_path),source_base_commit='b'*40)

    def authorize(self,**updates):
        self.auth.update(updates);self.write(self.auth_path,self.auth)
        return self.r.validate_authorization(self.auth_path,self.r.sha(self.auth_path))

    def register(self):
        with patch.object(self.r,'lineage_locks',lambda auth:nullcontext()):
            return self.r.register_continuation(self.args())

    def test_register_preserves_all_failed_costs_and_prefix(self):
        before=self.old_path.read_bytes();ledger=self.register()
        self.assertEqual(ledger['phases'],self.source['phases'])
        self.assertEqual(ledger['consumed_cpu_seconds'],37562.5)
        self.assertEqual(ledger['remaining_quantum_cpu_seconds'],135237.5)
        self.assertEqual(self.old_path.read_bytes(),before)
        self.assertEqual(ledger['created_at_utc'],self.source['created_at_utc'])
        self.assertEqual(ledger['original_v5_run_deadline_at_utc'],self.source['run_deadline_at_utc'])
        self.assertEqual(ledger['run_deadline_at_utc'],'2026-11-05T13:20:50+00:00')
        self.assertFalse(ledger['continuation_allocation']['new_72h_clock_started'])

    def test_registration_does_not_launch_any_payload(self):
        with patch.object(self.r.subprocess,'Popen',side_effect=AssertionError('No subprocess allowed')):
            self.register()

    def test_register_never_overwrites(self):
        self.register();before=self.new_path.read_bytes()
        with self.assertRaises(FileExistsError):self.register()
        self.assertEqual(self.new_path.read_bytes(),before)

    def duplicate_attempt(self, source, suffix):
        self.old_path=self.path/('copied_v5_'+suffix+'.json');self.write(self.old_path,source)
        self.new_path=self.path/('duplicate_v6_'+suffix+'.json')
        self.auth_path=self.path/('duplicate_auth_'+suffix+'.json')
        self.auth.update(source_v5_ledger_path=str(self.old_path),
            source_v5_ledger_sha256=self.r.sha(self.old_path),
            source_v5_consumed_cpu_seconds=source['consumed_cpu_seconds'],
            continuation_ledger_path=str(self.new_path))
        self.write(self.auth_path,self.auth)
        with self.assertRaises(FileExistsError):self.register()
        self.assertFalse(self.new_path.exists())

    def test_same_source_cannot_allocate_second_sequential_pool(self):
        self.register()
        self.duplicate_attempt(deepcopy(self.source),'same')

    def test_annotation_or_later_phase_cannot_bypass_single_budget_lineage(self):
        self.register()
        annotated=deepcopy(self.source);annotated['unrelated_note']='Not a new budget.'
        self.duplicate_attempt(annotated,'note')
        later=deepcopy(self.source);later['phases'].append({'phase':'later_closed_v5',
            'status':'EXITED','ended_at_utc':'2026-11-04T22:30:00+00:00','actual_cpu_seconds':12.})
        later['consumed_cpu_seconds']+=12.;later['remaining_quantum_cpu_seconds']-=12.
        self.duplicate_attempt(later,'later')

    def test_authorization_must_bind_exact_ledger_path(self):
        self.authorize(continuation_ledger_path=str(self.path/'other.json'))
        with self.assertRaisesRegex(ValueError,'sole ledger path'):self.register()

    def test_relative_authorized_ledger_rejected_before_claim(self):
        with self.assertRaisesRegex(ValueError,'canonical and absolute'):
            self.authorize(continuation_ledger_path='reports/relative_v6.json')
        self.assertFalse(self.r.CONTINUATION_REGISTRY.exists())

    def test_run_rejects_ledger_alias_before_any_payload(self):
        ledger=self.register();alias=self.path/'copied_v6.json';self.write(alias,ledger)
        with patch.object(self.r.subprocess,'Popen',side_effect=AssertionError('No payload')):
            with self.assertRaisesRegex(ValueError,'copied alias'):
                self.r.run_phase(Namespace(ledger=alias),ledger)

    def test_missing_single_allocation_claim_fails_closed(self):
        ledger=self.register();Path(ledger['continuation_registry_claim']['path']).unlink()
        with self.assertRaisesRegex(ValueError,'lineage claim'):self.r.validate_ledger(ledger)

    def test_partial_registration_does_not_allow_a_fresh_pool(self):
        original=self.r.save_new_ledger
        def fail_new_ledger(path,value):
            if Path(path)==self.new_path:raise OSError('Synthetic crash after immutable claim')
            return original(path,value)
        with patch.object(self.r,'save_new_ledger',fail_new_ledger):
            with self.assertRaisesRegex(OSError,'Synthetic crash'):self.register()
        self.assertFalse(self.new_path.exists())
        with self.assertRaises(FileExistsError):self.register()

    def test_zero_increment_preserves_cap_and_deadline(self):
        self.authorize(additional_cpu_seconds=0,deadline_extension_seconds=0,
                       max_quantum_cpu_seconds=129600,max_run_wall_seconds=64800)
        ledger=self.register()
        self.assertEqual(ledger['quantum_cpu_cap_seconds'],129600)
        self.assertEqual(ledger['run_deadline_at_utc'],self.source['run_deadline_at_utc'])

    def test_no_enabled_default_decision(self):
        for key,value in [('root_approved_for_execution',False),('no_budget_reset',False),
                          ('no_automatic_budget_growth',False),('research_authorization_evidence','')]:
            original=deepcopy(self.auth)
            with self.subTest(key=key),self.assertRaises(ValueError):self.authorize(**{key:value})
            self.auth=original

    def test_increment_bounds_and_booleans_rejected(self):
        for key in ('additional_cpu_seconds','deadline_extension_seconds'):
            for value in (-1,43201,True,1.5,None):
                original=deepcopy(self.auth)
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.authorize(**{key:value})
                self.auth=original

    def test_explicit_cumulative_cap_must_match(self):
        with self.assertRaisesRegex(ValueError,'Cumulative caps'):self.authorize(max_quantum_cpu_seconds=172801)

    def test_research_end_not_reset_from_registration(self):
        with self.assertRaisesRegex(ValueError,'research authorization deadline'):
            self.authorize(research_authorization_deadline='2026-11-05T12:00:00+00:00')

    def test_naive_and_future_root_timestamp_rejected(self):
        for value in ('2026-11-04T22:00:00','2026-11-05T02:00:00+00:00'):
            original=deepcopy(self.auth)
            with self.subTest(value=value),self.assertRaises(ValueError):self.authorize(created_at_utc=value)
            self.auth=original

    def test_original_history_fields_must_match(self):
        for key in ('source_v5_consumed_cpu_seconds','original_v5_created_at_utc',
                    'original_v5_run_deadline_at_utc','original_v5_quantum_cpu_cap_seconds',
                    'original_v5_run_wall_cap_seconds','original_v5_authorization_sha256'):
            original=deepcopy(self.auth)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'preserved exactly'):
                self.authorize(**{key:'reset'})
            self.auth=original

    def test_source_sha_is_rechecked_not_remigrated(self):
        ledger=self.register();self.source['phases'][0]['actual_cpu_seconds']=0
        self.write(self.old_path,self.source)
        with self.assertRaisesRegex(ValueError,'SHA'):self.r.validate_ledger(ledger)

    def test_wrong_source_schema_or_runner_rejected(self):
        for key,value in [('schema','research4-pyscf-actual-cpu-budget-v4'),('runner_sha256','0'*64)]:
            changed=deepcopy(self.source);changed[key]=value;self.write(self.old_path,changed)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'reviewed V5'):
                self.r.validate_source_v5(self.old_path,self.r.sha(self.old_path))

    def test_source_symlink_rejected(self):
        link=self.path/'linked.json';link.symlink_to(self.old_path)
        with self.assertRaisesRegex(ValueError,'Symlink'):self.r.validate_source_v5(link,self.r.sha(self.old_path))

    def test_running_unknown_nonfinite_and_residual_source_fail_closed(self):
        for updates in ({'status':'RUNNING'},{'actual_cpu_seconds':None},{'actual_cpu_seconds':float('nan')},
                        {'actual_cpu_seconds':-1},{'actual_cpu_seconds':True},
                        {'residual_processes_require_review_after_main_payload_exit':True}):
            changed=deepcopy(self.source);changed['phases'][1].update(updates);self.write(self.old_path,changed)
            with self.subTest(updates=updates),self.assertRaises(ValueError):
                self.r.validate_source_v5(self.old_path,self.r.sha(self.old_path))

    def test_failed_source_cost_sum_mismatch_rejected(self):
        changed=deepcopy(self.source);changed['consumed_cpu_seconds']=26.5;self.write(self.old_path,changed)
        with self.assertRaisesRegex(ValueError,'ALL closed V5'):self.r.validate_source_v5(self.old_path,self.r.sha(self.old_path))

    def test_lineage_prefix_omission_and_cost_reset_rejected(self):
        original=self.register()
        mutations=[lambda x:x['phases'].pop(),lambda x:x.update(consumed_cpu_seconds=0),
                   lambda x:x['phases'][1].update(actual_cpu_seconds=0),
                   lambda x:x.update(created_at_utc='2026-11-04T23:00:00+00:00'),
                   lambda x:x['migration'].update(source_v4_sha256='0'*64),
                   lambda x:x['continuation_source'].update(failed_costs_carried=False),
                   lambda x:x['continuation_allocation'].update(new_72h_clock_started=True),
                   lambda x:x['continuation_allocation'].update(authorization_sha256='0'*64)]
        for mutate in mutations:
            ledger=deepcopy(original);mutate(ledger)
            with self.subTest(mutation=mutate),self.assertRaises((ValueError,RuntimeError)):
                self.r.validate_ledger(ledger)

    def test_new_failure_cost_also_carried(self):
        ledger=self.register();ledger['phases'].append({'phase':'v6_failed','status':'PARTIAL_OR_FAILED',
            'ended_at_utc':'2026-11-04T23:01:00+00:00','actual_cpu_seconds':123.25})
        ledger['consumed_cpu_seconds']+=123.25;ledger['remaining_quantum_cpu_seconds']-=123.25
        self.r.validate_ledger(ledger)
        ledger['consumed_cpu_seconds']-=123.25
        with self.assertRaisesRegex(ValueError,'Every actual'):self.r.validate_ledger(ledger)

    def test_live_mode_only_one_new_running_row(self):
        ledger=self.register();ledger['phases'].append({'phase':'v6_live','status':'RUNNING'})
        with self.assertRaises(RuntimeError):self.r.validate_ledger(ledger)
        self.r.validate_ledger(ledger,allow_running=True)
        ledger['phases'].append({'phase':'v6_live2','status':'RUNNING'})
        with self.assertRaises(RuntimeError):self.r.validate_ledger(ledger,allow_running=True)

    def test_guard_semantics_and_prior_bank_cannot_change(self):
        for key,value in [('rss_stop_bytes',11*1024**3),('cgroup_kill_bytes',13*1024**3),
                          ('monitor_interval_seconds',.2),('prior_consumed_cpu_seconds',0.)]:
            original=deepcopy(self.auth)
            with self.subTest(key=key),self.assertRaises(ValueError):self.authorize(**{key:value})
            self.auth=original

    def test_expired_register_fails_without_creating_ledger(self):
        self.now=datetime(2026,11,5,14,tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError,'expired'):self.register()
        self.assertFalse(self.new_path.exists())

    def test_phase_limits_use_remaining_cost_and_original_plus_explicit_deadline(self):
        ledger=self.register();auth=self.r.validate_ledger(ledger)
        args=Namespace(threads=4,memory_gib=24,cpu_seconds=43200,wall_seconds=43200)
        limits=self.r.phase_limits(ledger,auth,args,now=datetime(2026,11,5,13,19,tzinfo=timezone.utc))
        self.assertEqual(limits['actual_cpu_closed_batch_cost_seconds'],37562.5)
        self.assertEqual(limits['actual_cpu_batch_cap_seconds'],172800)
        self.assertEqual(limits['wall_limit_seconds'],108)
        self.assertEqual(limits['address_space_limit_gib'],24)

    def test_parser_has_register_run_and_no_initialize(self):
        choices=self.r.parser()._subparsers._group_actions[0].choices
        self.assertEqual(set(choices),{'register','run'})

    def test_immutable_new_ledger_writer_no_replace(self):
        self.r.save_new_ledger(self.new_path,{'synthetic':1})
        with self.assertRaises(FileExistsError):self.r.save_new_ledger(self.new_path,{'synthetic':2})
        self.assertEqual(json.loads(self.new_path.read_text()),{'synthetic':1})

    def producer_fixture(self):
        ledger=self.register();worker=self.path/'frozen_stage.py';worker.write_text('# synthetic no-Q stage\n')
        output=self.path/'stage_output';output.mkdir();log=self.path/'stage.log';log.write_text('synthetic closed log\n')
        proof={'schema':self.r.PROOF_SCHEMA,'phase':'v6_rks','ledger_path':str(self.new_path),
            'supervisor_script_sha256':self.r.sha(SCRIPT),'supervisor_script_path':str(SCRIPT)}
        proof_path=self.path/'stage.proof.json';self.write(proof_path,proof)
        supervision={'proof_path':str(proof_path),'proof_sha256':self.r.sha(proof_path),'proof':proof,
            'live_parent_process_and_source_verified':True}
        source={'path':str(worker),'sha256':self.r.sha(worker)}
        evidence={'source_worker':source,'resource_supervision':supervision,'output_directory':str(output),
            'snapshot_manifest_sha256':'s'*64,'stage':'rks-partial'}
        result={'schema':'synthetic-staged-result','status':'STAGE_COMPLETE','source_worker':source,
            'resource_supervision':supervision,'snapshot_manifest_sha256':'s'*64,
            'stage':'rks-partial','stable_MF_state_unchanged':True,
            'native_direct_driver_control':{'restoration_failures':0,'original_driver_restored':True}}
        self.write(output/'result.json',result)
        row={'phase':'v6_rks','status':'EXITED','exit_code':0,'ended_at_utc':'2026-11-04T23:01:00+00:00',
             'actual_cpu_seconds':100.,'wall_terminated':False,'memory_terminated':False,'cpu_terminated':False,
             'monitor_failed':False,'hard_killed':False,'command':['python',str(worker),'--output',str(output)],
             'supervisor_proof_path':str(proof_path),'supervisor_proof_sha256':self.r.sha(proof_path),
             'log':str(log),'log_sha256':self.r.sha(log)}
        ledger['phases'].append(row);ledger['consumed_cpu_seconds']+=100;ledger['remaining_quantum_cpu_seconds']-=100
        self.write(self.new_path,ledger)
        kwargs=dict(expected_worker_path=worker,expected_worker_sha256=self.r.sha(worker),
            expected_result_schema='synthetic-staged-result',allowed_result_statuses=('STAGE_COMPLETE',),
            expected_output=output,expected_snapshot_sha256='s'*64)
        return evidence,kwargs,ledger,result

    def test_closed_stage_gate_accepts_only_explicit_real_closure_fixture(self):
        evidence,kwargs,ledger,_=self.producer_fixture()
        receipt=self.r.verify_closed_phase(evidence,**kwargs)
        self.assertEqual(receipt['actual_cpu_seconds'],100.)
        self.assertEqual(receipt['phase'],'v6_rks')
        # A separate current consumer may run, but source producer remains closed.
        ledger['phases'].append({'phase':'v6_consumer','status':'RUNNING'});self.write(self.new_path,ledger)
        self.r.verify_closed_phase(evidence,**kwargs)

    def test_unhealthy_parent_never_released(self):
        evidence,kwargs,ledger,_=self.producer_fixture();original=deepcopy(ledger)
        for changes in ({'status':'PARTIAL_OR_FAILED'},{'exit_code':1},{'cpu_terminated':True},
                        {'monitor_failed':True},{'memory_terminated':True},{'hard_killed':True}):
            changed=deepcopy(original);changed['phases'][-1].update(changes);self.write(self.new_path,changed)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.r.verify_closed_phase(evidence,**kwargs)

    def test_result_proof_log_and_source_mismatch_rejected(self):
        evidence,kwargs,ledger,result=self.producer_fixture()
        for changes in ({'status':'RUNNING'},{'stable_MF_state_unchanged':False},{'stage':'h1ao'},
                        {'snapshot_manifest_sha256':'wrong'},
                        {'native_direct_driver_control':{'restoration_failures':1,'original_driver_restored':False}}):
            changed=deepcopy(result);changed.update(changes);self.write(Path(evidence['output_directory'])/'result.json',changed)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.r.verify_closed_phase(evidence,**kwargs)
        self.write(Path(evidence['output_directory'])/'result.json',result)
        Path(ledger['phases'][-1]['log']).write_text('modified')
        with self.assertRaisesRegex(ValueError,'log changed'):self.r.verify_closed_phase(evidence,**kwargs)

    def test_closure_caller_must_choose_explicit_success_status(self):
        evidence,kwargs,_,_=self.producer_fixture();kwargs['allowed_result_statuses']=()
        with self.assertRaisesRegex(ValueError,'explicitly list'):self.r.verify_closed_phase(evidence,**kwargs)

    def test_reviewed_monitor_functions_ast_unchanged_from_v5(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        old=functions(self.r.V5_SOURCE);new=functions(SCRIPT)
        for name in ('parse_proc_stat','read_process_identity','identity_matches','owned_identity_matches',
                     'terminal_reap_observation','bounded_owned_reap','signal_bound_group',
                     'actual_cpu_observation','process_group_snapshot','read_cgroup_memory',
                     'resource_runtime_snapshot','phase_limits'):
            with self.subTest(name=name):self.assertEqual(old[name],new[name])

    def test_only_stdlib_imports_in_guard(self):
        banned={'numpy','scipy','torch','pyscf','geometric','rdkit'}
        for node in ast.walk(ast.parse(SCRIPT.read_text())):
            if isinstance(node,ast.Import):names=[n.name.split('.')[0] for n in node.names]
            elif isinstance(node,ast.ImportFrom):names=[(node.module or '').split('.')[0]]
            else:continue
            self.assertFalse(banned.intersection(names))


if __name__=='__main__':
    unittest.main()
