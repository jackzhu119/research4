"""Real-process tests for quality-independent continuation and one CPU grant."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_pyscf_continuous_20261108_v3.py'
spec=importlib.util.spec_from_file_location('continuous_budget', SCRIPT)
budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)

class ContinuousBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=budget.ROOT/'reports',prefix='continuous-budget-test-')
        self.directory=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def config(self,jobs,cap=80000,**extra):
        value={'run_dir':str(self.directory/'run'),'total_cpu_seconds':cap,'jobs':jobs,**extra}
        p=self.directory/'config.json';p.write_text(json.dumps(value));return p
    def job(self,name,code,**extra):
        return {'id':name,'argv':[sys.executable,'-c',code],'wall_seconds':10,'success_statuses':['PASSED'],**extra}
    def invoke(self,path):
        result=subprocess.run([sys.executable,str(SCRIPT),'--config',str(path),'--run'],capture_output=True,timeout=25)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        return json.loads((self.directory/'run/ledger.json').read_text())
    def test_warning_does_not_block_dependent(self):
        warning="import os,json;json.dump({'status':'GEOMETRY_AVAILABLE','optimizer_converged':False},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))"
        okay="import os,json;json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))"
        ledger=self.invoke(self.config([self.job('warning',warning),self.job('dependent',okay,depends_on=['warning'])]))
        self.assertEqual(ledger['jobs']['warning']['status'],'COMPLETED_WITH_WARNINGS')
        self.assertEqual(ledger['jobs']['dependent']['status'],'COMPLETED')
        self.assertEqual(ledger['status'],'ALL_WORK_COMPLETE')
    def test_actual_failed_attempt_retried_costs_accumulate(self):
        code="import os,json,sys;sum(i*i for i in range(500000));json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'));sys.exit(2 if os.environ['R4_ATTEMPT']=='1' else 0)"
        ledger=self.invoke(self.config([self.job('recover',code)]))
        row=ledger['jobs']['recover'];self.assertEqual(row['status'],'COMPLETED');self.assertEqual(len(row['attempts']),2)
        self.assertGreater(row['attempts'][0]['actual_cpu_seconds'],.01)
        self.assertAlmostEqual(row['actual_cpu_seconds'],sum(x['actual_cpu_seconds'] for x in row['attempts']))
        self.assertAlmostEqual(ledger['payload_cpu_seconds'],row['actual_cpu_seconds'])
    def test_two_failures_do_not_stop_next_job(self):
        okay="import os,json;json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))"
        ledger=self.invoke(self.config([self.job('bad','raise RuntimeError("point failed")'),self.job('next',okay,depends_on=['bad'])]))
        self.assertEqual(ledger['jobs']['bad']['status'],'FAILED_CONTINUED');self.assertEqual(len(ledger['jobs']['bad']['attempts']),2)
        self.assertEqual(ledger['jobs']['next']['status'],'COMPLETED')
    def test_global_cpu_guard_has_real_cost_and_stops_queue(self):
        ledger=self.invoke(self.config([self.job('burn','while True: pass'),self.job('later','pass')],cap=2034))
        self.assertEqual(ledger['status'],'CPU_BUDGET_STOPPED')
        self.assertGreater(ledger['payload_cpu_seconds'],2.0);self.assertLess(ledger['consumed_cpu_seconds'],2034)
        self.assertNotIn('later',ledger['jobs'])
        self.assertTrue(ledger['jobs']['burn']['attempts'][0]['all_owned_processes_closed'])
    def test_original_budget_chain_cannot_be_reset(self):
        with self.assertRaisesRegex(ValueError,'original CPU grant'):
            budget.config_load(self.config([self.job('pass','pass')],cap=81000,prior_charged_cpu_seconds=9678.94))
        budget.config_load(self.config([self.job('pass','pass')],cap=80000,prior_charged_cpu_seconds=9678.94))
    def test_missing_input_uses_fallback_and_other_missing_continues(self):
        fallback=self.directory/'fallback.json';fallback.write_text('{"usable":true}')
        missing=str(self.directory/'missing.json')
        code="import os,json,sys;assert json.load(open(sys.argv[1]))['usable'];json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))"
        fallback_job=self.job('fallback',code,output_prerequisites=[{'path':missing,'fallback_paths':[str(fallback)]}]);fallback_job['argv'].append(missing)
        ledger=self.invoke(self.config([self.job('missing','pass',output_prerequisites=[missing]),fallback_job]))
        self.assertEqual(ledger['jobs']['missing']['status'],'MISSING_INPUT_CONTINUED');self.assertEqual(ledger['jobs']['fallback']['status'],'COMPLETED')
    def test_children_reaped_and_cost_counted(self):
        code="import subprocess,sys,os,json;subprocess.run([sys.executable,'-c','sum(i*i for i in range(1000000))']);json.dump({'status':'PASSED'},open(os.environ['R4_JOB_OUTPUT']+'/result.json','w'))"
        ledger=self.invoke(self.config([self.job('child',code)]));self.assertGreater(ledger['jobs']['child']['actual_cpu_seconds'],.04)
    def test_unknown_running_blocks_resume(self):
        p=self.config([self.job('waiting','pass')]);config,directory,digest,_=budget.config_load(p);directory.mkdir()
        budget.save(directory/'ledger.json',{'schema':budget.SCHEMA,'config_sha256':digest,'supervisor_sha256':budget.sha(SCRIPT),'jobs':{'waiting':{'status':'RUNNING'}}})
        with self.assertRaisesRegex(ValueError,'Unclosed'):budget.Queue(config,directory,digest,p).run(resume=True)
    def test_stop_file_contains_worker(self):
        p=self.config([self.job('wait','import time;time.sleep(30)')]);child=subprocess.Popen([sys.executable,str(SCRIPT),'--config',str(p),'--run'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        path=self.directory/'run/ledger.json';deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if path.exists() and json.loads(path.read_text()).get('jobs',{}).get('wait',{}).get('attempts',[{}])[-1].get('worker_identity'):break
            time.sleep(.05)
        (self.directory/'run/STOP').touch();_,err=child.communicate(timeout=10);self.assertEqual(child.returncode,0,err.decode())
        ledger=json.loads(path.read_text());attempt=ledger['jobs']['wait']['attempts'][0]
        self.assertEqual(attempt['stop_reason'],'STOP_FILE');self.assertFalse(budget.alive(attempt['worker_identity']))

if __name__=='__main__':unittest.main()
