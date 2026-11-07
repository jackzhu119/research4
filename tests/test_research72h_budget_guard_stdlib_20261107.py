"""Pure stdlib proof/accounting fixtures; no processes, signals, Q, Git or real ledger writes."""
import ast
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch,Mock

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'scripts/run_research4_research72h_budget_20261107.py'
LEGACY=REPO/'scripts/run_pyscf_continuation_budget_20261105_v8.py'
spec=importlib.util.spec_from_file_location('research72_guard_test',SOURCE)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return g.sha(path)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'reports/research72h-20261107-v1/cpu_budget.json'
        self.path.parent.mkdir(parents=True)
        self.source=self.root/'reports/old/cpu_budget.json'
        self.old={'schema':g.OLD_SCHEMA,'phases':[
            {'phase':'old-ok','status':'EXITED','ended_at_utc':'2026-11-07T00:00:00+00:00','actual_cpu_seconds':2.75},
            {'phase':'old-fail','status':'PARTIAL_OR_FAILED','ended_at_utc':'2026-11-07T00:01:00+00:00','actual_cpu_seconds':7.125}],
            'consumed_cpu_seconds':9.875,'remaining_quantum_cpu_seconds':980.125}
        self.oldsha=write(self.source,self.old)
        self.authpath=self.path.parent/'authorization.json'
        self.auth={'schema':g.AUTH_SCHEMA,'authorization_source':'ACTUAL_USER_MESSAGE_CURRENT_CHAT','user_text':'Authorized prospective72h',
            'observed_at_utc':'2026-11-07T02:50:37.289380+00:00','run_deadline_at_utc':'2026-11-10T02:50:37.289380+00:00',
            'authorized_additional_cpu_seconds':900000.0,'maximum_worker_threads':2,'phase_memory_RSS_softstop_GiB':10,
            'host_cgroup_memory_GiB':16,'minimum_workspace_free_bytes':64*1024**2,'source_ledger_must_remain_byte_identical':True,
            'source_ledger':str(self.source),'source_ledger_sha256':self.oldsha,'source_consumed_cpu_seconds':9.875,
            'source_unused_cpu_seconds_retired_not_recredited':980.125,'new_ledger_path':str(self.path),
            'new_cumulative_cap_cpu_seconds':900009.875}
        self.authsha=write(self.authpath,self.auth)
        self.args=SimpleNamespace(ledger=self.path,authorization=self.authpath,authorization_sha256=self.authsha)
        for name,value in [('ROOT',self.root),('EXPECTED_AUTH_SHA',self.authsha),('CONTINUATION_REGISTRY',self.root/'.runtime/claims'),
                           ('utc',lambda:datetime(2026,11,7,3,tzinfo=timezone.utc)),
                           ('old_guard',lambda:SimpleNamespace(validate_ledger=lambda *args,**kwargs:{},lineage_locks=lambda *args:nullcontext())),
                           ('lineage_locks',lambda *args:nullcontext())]:
            p=patch.object(g,name,value);p.start();self.addCleanup(p.stop)
    def init(self):
        return g.initialize(self.args)
    def authorize_changed(self,**changes):
        self.auth.update(changes);self.authsha=write(self.authpath,self.auth);self.args.authorization_sha256=self.authsha
        with patch.object(g,'EXPECTED_AUTH_SHA',self.authsha):
            return g.validate_authorization(self.authpath,self.authsha)
    def append_row(self,ledger,row):
        ledger['phases'].append(row)
        ledger['consumed_cpu_seconds']+=row.get('actual_cpu_seconds') or 0
        ledger['remaining_quantum_cpu_seconds']-=row.get('actual_cpu_seconds') or 0
        g.save(self.path,ledger)
    def test_exact_failed_cost_prefix_carry_without_old_unused_credit(self):
        result=self.init()
        self.assertEqual(result['phases'],self.old['phases'])
        self.assertEqual(result['consumed_cpu_seconds'],9.875)
        self.assertEqual(result['quantum_cpu_cap_seconds'],900009.875)
        self.assertEqual(result['remaining_quantum_cpu_seconds'],900000.)
        self.assertEqual(result['old_unused_CPU_seconds_retired_not_recredited'],980.125)
        self.assertEqual(g.sha(self.source),self.oldsha)
        g.validate_ledger(result)
    def test_wrong_auth_sha_cannot_register(self):
        with self.assertRaises(ValueError):g.validate_authorization(self.authpath,'a'*64)
    def test_changed_old_bytes_cannot_register(self):
        self.source.write_text(self.source.read_text()+' ')
        with self.assertRaises(ValueError):self.init()
    def test_no_recredited_unused_or_larger_allocation(self):
        for changes in ({'new_cumulative_cap_cpu_seconds':900990.},{'authorized_additional_cpu_seconds':900001.}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.authorize_changed(**changes)
            self.auth.update({'new_cumulative_cap_cpu_seconds':900009.875,'authorized_additional_cpu_seconds':900000.})
    def test_new72h_clock_cannot_be_reset_or_extended(self):
        with self.assertRaises(ValueError):self.authorize_changed(run_deadline_at_utc='2026-11-10T03:50:37.289380+00:00')
    def test_single_authorization_claim_prevents_reset_after_deleted_fixture_ledger(self):
        self.init();self.path.unlink()
        with self.assertRaises(FileExistsError):self.init()
    def test_existing_ledger_never_overwritten(self):
        self.init()
        with self.assertRaises(FileExistsError):self.init()
    def test_old_failed_prefix_drop_rejected(self):
        value=self.init();value['phases'].pop(1)
        with self.assertRaises(ValueError):g.validate_ledger(value)
    def test_all_new_failed_costs_must_be_counted(self):
        value=self.init();row={'phase':'new-failed','status':'PARTIAL_OR_FAILED','ended_at_utc':'2026-11-07T03:05:00+00:00','actual_cpu_seconds':3.25}
        value['phases'].append(row)
        with self.assertRaises(ValueError):g.validate_ledger(value)
        value['consumed_cpu_seconds']+=3.25;value['remaining_quantum_cpu_seconds']-=3.25
        g.validate_ledger(value)
    def test_unknown_cpu_or_postexit_process_audit_blocks_recovery(self):
        for extra in ({'actual_cpu_seconds':None,'actual_CPU_is_unknown_do_not_resume_budget':True},
                      {'actual_cpu_seconds':1.,'post_exit_process_audit_failure':'unreadable group'}):
            with self.subTest(extra=extra):
                value=self.init() if not self.path.exists() else g.read(self.path)
                value=deepcopy(value);value['phases'].append({'phase':'unknown','status':'PARTIAL_OR_FAILED','ended_at_utc':'x',**extra})
                with self.assertRaises((ValueError,RuntimeError)):g.validate_ledger(value)
    def test_running_or_reboot_never_infers_closure(self):
        value=self.init();value['phases'].append({'phase':'unknown','status':'RUNNING','pid':999999,'boot_id':'oldboot'})
        with self.assertRaises(RuntimeError):g.validate_ledger(value)
        g.validate_ledger(value,allow_running=True)
    def test_no_legacy_live_launch(self):
        with self.assertRaises(ValueError):g.run_phase(SimpleNamespace(),self.old)
    def test_limits_finite_cap_deadline_threads_cleanup(self):
        value=self.init();auth=g.validate_ledger(value)
        args=SimpleNamespace(threads=2,memory_gib=24,cpu_seconds=12000,wall_seconds=28800)
        limits=g.phase_limits(value,auth,args)
        self.assertEqual(limits['actual_cpu_stop_threshold_seconds'],11880)
        self.assertEqual(limits['wall_limit_seconds'],28800)
        for attr,bad in [('threads',3),('cpu_seconds',60001),('wall_seconds',28801),('memory_gib',25)]:
            old=getattr(args,attr);setattr(args,attr,bad)
            with self.subTest(attr=attr),self.assertRaises(ValueError):g.phase_limits(value,auth,args)
            setattr(args,attr,old)
        with self.assertRaises(ValueError):g.phase_limits(value,auth,args,now=datetime(2026,11,10,3,tzinfo=timezone.utc))
    def test_identity_reuse_never_matches_original_child(self):
        expected={'pid':42,'start_ticks':100,'uid':1000,'pgrp':42,'session':42}
        actual={**expected,'state':'S','uids':[1000]*4}
        self.assertTrue(g.identity_matches(expected,actual))
        for key in ('start_ticks','uid','pgrp','session'):
            changed={**actual,key:actual[key]+1}
            self.assertFalse(g.identity_matches(expected,changed))
    def test_live_oldboot_or_oldschema_proof_rejected(self):
        for schema,boot in ((g.PROOF_SCHEMA,'oldboot'),(g.OLD_PROOF_SCHEMA,'oldboot')):
            p=self.root/'fakeproof.json';write(p,{'schema':schema,'guard_semantics':g.GUARD_SEMANTICS,
                'terminal_exit_reap_grace_seconds':.5,'boot_id':boot})
            with self.assertRaises(ValueError):g.verify_resource_supervision(p)
    def test_startup_handshake_waits_for_exact_persisted_child_then_stops(self):
        proof={'phase':'job','child_pid':42,'child_start_ticks':100,'target_uid':1000,'target_pgid':42,'target_session_id':42}
        row={'phase':'job','status':'RUNNING','child_identity':{'pid':42,'start_ticks':100,'uid':1000,'pgrp':42,'session':42},'supervisor_proof_sha256':'s'}
        pending={'schema':g.SCHEMA,'continuation_ledger_path':str(self.path),'phases':[{'phase':'job','status':'RUNNING'}]}
        ready={**pending,'phases':[row]}
        reader=Mock(side_effect=[pending,ready]);sleeper=Mock()
        with patch.object(g,'validate_ledger',return_value={}):
            result=g.await_live_ledger_binding(self.path,proof,'s',reader=reader,clock=Mock(side_effect=[0.,.1]),sleeper=sleeper)
            self.assertEqual(result['phases'][0],row);self.assertEqual(reader.call_count,2);sleeper.assert_called_once()
            with self.assertRaises(ValueError):
                g.await_live_ledger_binding(self.path,proof,'s',reader=lambda path:pending,
                    clock=Mock(side_effect=[0.,2.]),sleeper=Mock())
            for bad in ({**ready,'phases':[]},{**ready,'phases':[{**row,'child_identity':{'pid':99}}]},
                        {**ready,'phases':[{**row,'supervisor_proof_sha256':'other'}]},
                        {**ready,'phases':[{**row,'actual_cpu_closed_batch_cost_seconds':0}]}):
                with self.assertRaises(ValueError):
                    g.await_live_ledger_binding(self.path,proof,'s',reader=lambda path:bad,clock=lambda:0.,sleeper=Mock())
        with patch.object(g,'validate_ledger',side_effect=ValueError('counter changed')):
            with self.assertRaisesRegex(ValueError,'counter changed'):
                g.await_live_ledger_binding(self.path,proof,'s',reader=lambda path:pending,clock=lambda:0.,sleeper=Mock())
    def closed_fixture(self,*,failed=False):
        ledger=self.init();worker=self.root/'scripts/worker.py';worker.parent.mkdir();worker.write_text('# fake no-Q fixture\n')
        output=self.path.parent/'job';output.mkdir();log=self.path.parent/'job.log';log.write_text('fake closed log\n')
        proof={'schema':g.PROOF_SCHEMA,'supervisor_script_path':str(SOURCE),'supervisor_script_sha256':g.sha(SOURCE),
            'ledger_path':str(self.path),'phase':'job','boot_id':'testboot','child_pid':42,'child_start_ticks':100,
            'target_uid':1000,'target_pgid':42,'target_session_id':42}
        pp=self.path.parent/'proof.json';ps=write(pp,proof)
        row={'phase':'job','status':'PARTIAL_OR_FAILED' if failed else 'EXITED','exit_code':2 if failed else 0,
            'ended_at_utc':'2026-11-07T03:01:00+00:00','actual_cpu_seconds':1.5,'boot_id':'testboot',
            'owned_process_closure_confirmed':True,'child_identity':{'pid':42,'start_ticks':100,'uid':1000,'pgrp':42,'session':42},
            'supervisor_proof_path':str(pp),'supervisor_proof_sha256':ps,'log':str(log),'log_sha256':g.sha(log),
            'command':['python',str(worker),'--output',str(output)],'wall_terminated':False,'memory_terminated':False,
            'cpu_terminated':False,'monitor_failed':False,'hard_killed':False}
        self.append_row(ledger,row)
        return ledger,row,worker,output
    def test_generic_ML_closure_requires_no_payload_proof_or_result(self):
        ledger,row,worker,out=self.closed_fixture()
        result=g.verify_closed_phase({'ledger_path':str(self.path),'phase':'job'},expected_worker_path=worker,
            expected_worker_sha256=g.sha(worker),expected_output=out)
        self.assertTrue(result['scientific_success']);self.assertTrue(result['all_owned_processes_closed'])
        self.assertFalse((out/'result.json').exists())
    def test_known_closed_failure_can_move_to_next_independent_job(self):
        ledger,row,worker,out=self.closed_fixture(failed=True)
        with self.assertRaises(ValueError):g.verify_closed_row(self.path,'job',expected_worker_path=worker,expected_worker_sha256=g.sha(worker))
        result=g.verify_closed_row(self.path,'job',expected_worker_path=worker,expected_worker_sha256=g.sha(worker),allow_failure=True)
        self.assertFalse(result['scientific_success']);self.assertEqual(result['actual_cpu_seconds'],1.5)
    def test_generic_closed_PID_birth_or_proof_log_tamper_rejected(self):
        ledger,row,worker,out=self.closed_fixture();row['child_identity']['start_ticks']+=1;g.save(self.path,ledger)
        with self.assertRaises(ValueError):g.verify_closed_row(self.path,'job',expected_worker_path=worker,expected_worker_sha256=g.sha(worker))
    def test_unresolved_owned_process_closure_not_generic_failure(self):
        ledger,row,worker,out=self.closed_fixture(failed=True);row['owned_process_closure_confirmed']=False;g.save(self.path,ledger)
        with self.assertRaises(ValueError):g.verify_closed_row(self.path,'job',expected_worker_path=worker,expected_worker_sha256=g.sha(worker),allow_failure=True)
    def test_native_monitor_core_AST_preserved_and_heartbeat_disk_boot_present(self):
        old={n.name:n for n in ast.parse(LEGACY.read_text()).body if isinstance(n,ast.FunctionDef)}
        new={n.name:n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)}
        for name in ('parse_proc_stat','read_process_identity','identity_matches','owned_identity_matches','terminal_reap_observation',
                     'bounded_owned_reap','signal_bound_group','actual_cpu_observation','process_group_snapshot','read_cgroup_memory',
                     'resource_runtime_snapshot','phase_limits','healthy_closed_row'):
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        source=SOURCE.read_text()
        for literal in ('last_monitor_CPU_observation','last_monitor_workspace_free_bytes','row["child_identity"] = dict(expected_identity)',
                        'DISK_FREE_GUARD_FAILED_CLOSED','"boot_id":Path','UNRESOLVED_START_FAILURE','owned_process_closure_confirmed'):
            self.assertIn(literal,source)
    def test_production_auth_and_legacy_source_have_exact_frozen_bytes(self):
        self.assertEqual(g.sha(LEGACY),'7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65')
        observed=REPO/'reports/research72h-20261107-v1/resource-protocol/user_authorization_observed-001.json'
        self.assertEqual(g.sha(observed),'058c24d076567b14054f78467c001a7858ef2601a070f3c5b97c8b074bc9981d')


if __name__=='__main__':unittest.main()
