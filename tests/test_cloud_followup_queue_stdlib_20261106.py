"""Pure standard-library policy tests; no quantum process is ever started."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE=Path(__file__).resolve().parents[1]/'scripts/research4_cloud_followup_queue_20261106.py'
spec=importlib.util.spec_from_file_location('followup_queue_under_test',SOURCE)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


class QueuePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for folder in ('scripts','reports/queue','data','.runtime'):(self.root/folder).mkdir(parents=True)
        self.guard=self.root/'scripts/guard.py';self.guard.write_text('# guard fixture, never executed\n')
        self.helper=self.root/'scripts/helper.py';self.helper.write_text('# metadata fixture, never executed\n')
        self.worker=self.root/'scripts/worker.py';self.worker.write_text('# worker fixture, never executed\n')
        self.input=self.root/'data/input.json';self.input.write_text('{}\n')
        for name,value in {'ROOT':self.root,'GUARD':self.guard,'GUARD_SHA':q.sha(self.guard),'HELPERS':self.helper,'HELPERS_SHA':q.sha(self.helper),'LEDGER':self.root/'reports/ledger.json'}.items():
            context=patch.object(q,name,value);context.start();self.addCleanup(context.stop)
        self.job={'id':'point-plus','kind':'quantum','worker':{'path':str(self.worker),'sha256':q.sha(self.worker)},'payload_argv':[str(self.root/'.runtime/bin/python'),str(self.worker),'--output','{output}'],'result_schema':'fixture-only','success_statuses':['DONE'],'cpu_seconds':1000,'wall_seconds':900}
        self.config={'schema':q.SCHEMA,'ledger':str(q.LEDGER),'quantum_cpu_cap_seconds':q.CAP,'deadline_utc':'2026-11-09T08:25:17.730193+00:00','work_directory':str(self.root/'reports/queue'),'state':str(self.root/'reports/queue/state.json'),'monitor':str(self.root/'reports/queue/monitor.jsonl'),'minimum_remaining_cpu_seconds':125,'CPU_reserve_seconds':600,'phase_prefix':'finite-fixture','source_refs':[{'path':str(p),'sha256':q.sha(p)} for p in (self.guard,self.helper,self.worker)],'immutable_inputs':[{'path':str(self.input),'sha256':q.sha(self.input)}],'python_executable':'.runtime/bin/python','pythonpath':['.runtime/quantum-pilot-site','.runtime/quantum-upgrade-site'],'jobs':[self.job]}
        self.grant={'schema':q.GRANT_SCHEMA,'config_sha256':'CONFIG_FIXTURE','driver_sha256':q.sha(SOURCE),'root_execution_approved':True,'root_serial_scheduling_confirmed':True,'no_new_quantum_budget':True,'no_deadline_extension':True,'expires_at_utc':self.config['deadline_utc'],'max_invocations':2,'launcher_cpu_seconds_per_invocation':60,'wall_seconds_per_invocation':3600,'launcher_rss_stop_bytes':64*1024**2}

    def validate(self):q.validate(self.config,self.grant,'CONFIG_FIXTURE')

    def test_earlier_deadline_and_valid_contract(self):self.validate()

    def test_later_deadline_rejected(self):
        self.config['deadline_utc']='2026-11-10T00:00:00+00:00';self.grant['expires_at_utc']=self.config['deadline_utc']
        with self.assertRaises(ValueError):self.validate()

    def test_no_cap_or_ledger_alias(self):
        self.config['quantum_cpu_cap_seconds']+=1
        with self.assertRaises(ValueError):self.validate()

    def test_future_result_placeholder_rejected(self):
        self.job['payload_argv']+=['--source','{result:not-done}']
        with self.assertRaises(ValueError):self.validate()

    def test_output_not_user_supplied_path(self):
        self.job['payload_argv'][-1]=str(self.root/'data')
        with self.assertRaises(ValueError):self.validate()

    def test_source_mutation_rejected(self):
        self.worker.write_text('# changed source\n')
        with self.assertRaises(ValueError):self.validate()

    def test_duplicate_job_rejected(self):
        self.config['jobs'].append(copy.deepcopy(self.job))
        with self.assertRaises(ValueError):self.validate()

    def test_unbounded_retry_is_not_supported(self):
        self.grant['max_invocations']=100
        with self.assertRaises(ValueError):self.validate()

    def test_followup_parent_lock_does_not_deadlock_worker_main_lock(self):
        with q.followup_lock():
            original=self.root/'.runtime/research4-materialized-quantum-queue.lock'
            with original.open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with q.followup_lock():pass

    def test_success_result_fields_are_real_gates(self):
        output=self.root/'reports/queue/out';output.mkdir()
        (output/'result.json').write_text(json.dumps({'schema':'fixture-only','status':'DONE','numerical':{'passed':False}}))
        self.job['required_result_fields']={'numerical.passed':True}
        with self.assertRaises(ValueError):q.qualify_result(self.job,output)

    def test_output_directory_seal_detects_array_mutation(self):
        output=self.root/'reports/queue/out';output.mkdir();p=output/'gradient.dat';p.write_bytes(b'first')
        first=q.output_artifacts(output);p.write_bytes(b'other')
        self.assertNotEqual(first,q.output_artifacts(output))

    def test_optional_partial_is_preserved_without_release_to_dependencies(self):
        self.job['optional']=True
        state={'results':{}};current={'status':'RUNNING'};events=[]
        result=q.close_optional_scientific_partial(state,self.job,current,{'result':{'path':'fixture','sha256':'fixture'}},{'status':'PARTIAL','schema':'fixture-only'},'not scientifically qualified',lambda event,data:events.append((event,data)))
        self.assertIsNone(result)
        self.assertEqual(current['status'],'PARTIAL_SCIENTIFIC_CLOSED')
        self.assertFalse(current['scientific_release'])
        self.assertNotIn(self.job['id'],state['results'])
        self.assertFalse(events[0][1]['scientific_release'])

    def test_mandatory_scientific_partial_cannot_be_silently_skipped(self):
        with self.assertRaises(ValueError):
            q.close_optional_scientific_partial({'results':{}},self.job,{}, {},{'status':'PARTIAL'},'bad',lambda *args:None)

    def test_analysis_cannot_use_quantum_execute_entry_point(self):
        self.job.update(kind='analysis',cpu_seconds=30,wall_seconds=60)
        self.job['payload_argv'].append('--execute')
        with self.assertRaises(ValueError):self.validate()

    def test_analysis_zero_quantum_claim_must_come_from_result(self):
        self.job.update(kind='analysis')
        output=self.root/'reports/queue/out';output.mkdir()
        (output/'result.json').write_text(json.dumps({'schema':'fixture-only','status':'DONE','new_quantum_evaluations':1}))
        with self.assertRaises(q.ScientificQualificationError):q.qualify_result(self.job,output)

    def test_dynamic_result_hash_binds_previous_file(self):
        p=self.root/'reports/previous.json';p.write_text('{}')
        refs={'previous':{'path':str(p),'sha256':q.sha(p)}}
        self.assertEqual(q.materialize(['{result:previous}','{sha256:previous}','{output}'],self.root/'reports/out',refs),[str(p),q.sha(p),str(self.root/'reports/out')])
        p.write_text('{"changed":true}')
        with self.assertRaises(ValueError):q.materialize(['{result:previous}'],self.root/'reports/out',refs)

    def test_closed_cpu_guard_failure_can_retry_but_unknown_sigkill_cannot(self):
        class Guard:
            @staticmethod
            def validate_ledger(value):return None
        row={'phase':'p','status':'PARTIAL_OR_FAILED','actual_cpu_seconds':42.,'resource_stop_reason':'ACTUAL_CPU_GUARD','cpu_terminated':True}
        self.assertTrue(q.retryable_closed_resource_failure(Guard,{'phases':[row]},'p'))
        row['resource_stop_reason']='UNKNOWN_SIGKILL'
        self.assertFalse(q.retryable_closed_resource_failure(Guard,{'phases':[row]},'p'))
        row.update(resource_stop_reason='ACTUAL_CPU_GUARD',monitor_failed=True)
        self.assertFalse(q.retryable_closed_resource_failure(Guard,{'phases':[row]},'p'))


if __name__=='__main__':unittest.main()
