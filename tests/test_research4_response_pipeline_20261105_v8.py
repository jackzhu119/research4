"""Pure-stdlib recovery/identity/config tests; no quantum or numerical imports."""
import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v8_response_pipeline_under_test',ROOT/'scripts/research4_response_pipeline_20261105_v8.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def ref(path,sha='0'*64):return {'path':str(path),'sha256':sha}

class FakeQ:
    def __init__(self,ledger=None):self.ledger=ledger or {'quantum_cpu_cap_seconds':m.CAP,'consumed_cpu_seconds':210160.8308359999,'run_deadline_at_utc':m.DEADLINE,'phases':[]};self.verified=[]
    path=staticmethod(lambda x:Path(x))
    checked=staticmethod(lambda x:Path(x['path']))
    sha=staticmethod(lambda x:'d'*64)
    aware=staticmethod(datetime.fromisoformat)
    utc=staticmethod(lambda:datetime(2026,11,5,tzinfo=timezone.utc))
    def read(self,*args):return self.ledger
    def verify_existing(self,*args):self.verified.append(args)
    def process_observation(self,**kw):return {'conflicts':[]}

class FakeGuard:
    def validate_ledger(self,value):pass
    def healthy_closed_row(self,row):return row.get('status')=='EXITED' and row.get('exit_code')==0

def fixture():
    work=ROOT/'reports/v8-stdlib-fixture-not-created'
    cfg={'schema':m.SCHEMA,'quantum_cpu_cap_seconds':m.CAP,'quantum_deadline_at_utc':m.DEADLINE,
        'work_directory':str(work),'state':str(work/'state.json'),'monitor':str(work/'monitor.jsonl'),
        'ledger':str(ROOT/'reports/v8-ledger-fixture-not-created.json'),'snapshot':str(ROOT/'.runtime/snapshot-fixture'),
        'snapshot_manifest_sha256':m.SNAPSHOT,'gradient_result':ref(ROOT/'reports/G.json',m.GRADIENT),
        'threads':2,'pythonpath':['.runtime/quantum-pilot-site','.runtime/quantum-upgrade-site'],
        'python_executable':'.runtime/bin/python','phase_prefix':'v8-stdlib','existing_results':{},
        'partial_result':ref(ROOT/'reports/A.json'),'h1_result':ref(ROOT/'reports/H1.json'),'control_release':ref(ROOT/'reports/control.json'),
        'source_refs':[ref(p,s) for p,s in [(m.QUEUE,m.QUEUE_SHA),(m.GUARD,m.GUARD_SHA),(m.CONSUMER,m.CONSUMER_SHA),(m.ANALYZER,m.ANALYZER_SHA)]],
        'phase_requests':{k:{'cpu_seconds':21600,'wall_seconds':14400} for k in m.RESPONSES},
        'forecast_multiplier':1.2,'global_CPU_reserve_seconds':1800}
    cfg['phase_requests']['global-assemble']={'cpu_seconds':1800,'wall_seconds':3600}
    grant={'schema':m.GRANT_SCHEMA,'config_sha256':'config','driver_sha256':m.sha(m.__file__),
        'no_new_quantum_budget':True,'no_deadline_extension':True,'expires_at_utc':m.DEADLINE,
        'max_invocations':4,'launcher_cpu_seconds_per_invocation':600,'wall_seconds_per_invocation':86400,
        'launcher_rss_stop_bytes':256*1024**2,'conflict_wait_seconds':600,'opaque_process_identity_exceptions':[],
        'root_execution_approved':False,'root_serial_scheduling_confirmed':False}
    return cfg,grant

class DriverTests(unittest.TestCase):
    def test_exact_new_pins_and_finite_10_plus_global_graph(self):
        self.assertEqual(len(m.RESPONSES),10);self.assertEqual(len(m.STAGES),11)
        self.assertEqual(m.GUARD_SHA,'7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65')
        self.assertEqual(m.CONSUMER_SHA,'3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585')
        self.assertEqual(m.ANALYZER_SHA,'da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6')
        self.assertEqual(m.CAP,210160.8308359999+180000)

    def test_disabled_metadata_pass_but_execution_requires_root_and_actual_release(self):
        c,g=fixture();m.validate(FakeQ(),c,g,'config')
        with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config',True)
        g.update(root_execution_approved=True,root_serial_scheduling_confirmed=True);m.validate(FakeQ(),c,g,'config',True)
        c['control_release']=None
        with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config',True)

    def test_old_pool_or_later_deadline_cannot_be_substituted(self):
        for key,value in [('quantum_cpu_cap_seconds',216000),('quantum_deadline_at_utc','2026-11-10T12:30:00+00:00')]:
            c,g=fixture();c[key]=value
            with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config')

    def test_changed_source_config_or_forecast_factor_rejected(self):
        c,g=fixture();g['driver_sha256']='f'*64
        with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config')
        c,g=fixture();c['forecast_multiplier']=1
        with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config')

    def test_original_pilot_ceiling_cannot_be_silently_lowered(self):
        c,g=fixture();c['phase_requests']['response-00']['cpu_seconds']=7200
        with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config')

    def test_only_explicit_two_unique_daemon_exceptions_allowed(self):
        c,g=fixture();one={'pid':125,'start_ticks':574,'uid':0,'comm':'dockerd','boot_id':'real-boot-id'}
        g['opaque_process_identity_exceptions']=[one,dict(one)]
        with self.assertRaises(ValueError):m.validate(FakeQ(),c,g,'config')

    def test_all_ten_B8_arguments_preserve_full_source_and_no_control_flag(self):
        c,_=fixture()
        for i,k in enumerate(m.RESPONSES):
            j=m.stage_job(FakeQ(),c,k);a=j['payload_argv'];self.assertEqual(j['stage'],'response-block')
            self.assertEqual(a[a.index('--block-index')+1],str(i));self.assertNotIn('--control-validation',a)
            self.assertEqual(a[a.index('--h1-result-sha256')+1],c['h1_result']['sha256'])
            self.assertEqual(j['threads'],2);self.assertEqual(j['memory_gib'],24)

    def test_global_needs_actual_ordered_response_index(self):
        c,_=fixture()
        with self.assertRaises(ValueError):m.stage_job(FakeQ(),c,'global-assemble')
        j=m.stage_job(FakeQ(),c,'global-assemble',ref(ROOT/'reports/actual-index.json','e'*64))
        self.assertEqual(j['success_status'],'FREQUENCIES_COMPUTED')
        self.assertEqual(j['payload_argv'][-2:],['--response-index-sha256','e'*64])

    def test_full_wall_window_boundary_does_not_shorten_active_stage(self):
        with patch.object(m.time,'time',return_value=100),patch.object(m.time,'monotonic',return_value=200):
            with self.assertRaises(m.BoundaryReached):m.ensure_full_window({'wall_seconds':14400},14500,14600)
            m.ensure_full_window({'wall_seconds':14400},15000,15100)

    def test_first_pilot_forecast_uses_actual_wait4_and_1p2_margin(self):
        c,_=fixture();state={'stages':{'response-00':{'status':'HEALTHY_CLOSED','materialized_job':{'phase':'actual-pilot'}}}}
        ledger={'quantum_cpu_cap_seconds':m.CAP,'consumed_cpu_seconds':220160.8308359999,
            'phases':[{'phase':'actual-pilot','status':'EXITED','exit_code':0,'actual_cpu_seconds':10000}]}
        got=m.feasibility(c,state,ledger);self.assertEqual(got['pending_response_blocks'],9)
        self.assertEqual(got['forecast_CPU_seconds'],109800);self.assertTrue(got['feasible'])
        ledger['consumed_cpu_seconds']=m.CAP-100000;self.assertFalse(m.feasibility(c,state,ledger)['feasible'])

    def test_forecast_uses_max_observed_successful_cost_not_favorable_block(self):
        c,_=fixture();state={'stages':{k:{'status':'HEALTHY_CLOSED','materialized_job':{'phase':k}} for k in m.RESPONSES[:2]}}
        ledger={'quantum_cpu_cap_seconds':m.CAP,'consumed_cpu_seconds':230000,'phases':[
            {'phase':k,'status':'EXITED','exit_code':0,'actual_cpu_seconds':v} for k,v in zip(m.RESPONSES[:2],[10000,12000])]}
        self.assertEqual(m.feasibility(c,state,ledger)['forecast_CPU_seconds'],117000)
        ledger['phases'][1]['exit_code']=2
        with self.assertRaises(ValueError):m.feasibility(c,state,ledger)

    def test_no_forecast_invented_before_pilot(self):
        c,_=fixture();self.assertIsNone(m.feasibility(c,{'stages':{}},{'phases':[]}))

    def test_healthy_seal_is_verified_and_skipped_without_subprocess(self):
        c,g=fixture();q=FakeQ();job=m.stage_job(q,c,'response-00');state={'stages':{'response-00':{'status':'HEALTHY_CLOSED','seal':[{'actual':'seal'}]}}}
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('No Q in tests')):
            result=m.launch_quantum(q,FakeGuard(),c,g,job,state,0,0,[],lambda *x:None)
        self.assertTrue(q.verified);self.assertEqual(result['path'],job['result'])

    def test_failed_open_checkpoint_never_retries_or_refunds(self):
        c,g=fixture();q=FakeQ();job=m.stage_job(q,c,'response-00')
        for status in ['RUNNING','STARTING','BLOCKED_REQUIRES_ROOT_REVIEW']:
            with patch.object(m.subprocess,'Popen',side_effect=AssertionError('No Q in tests')):
                with self.assertRaises(ValueError):m.launch_quantum(q,FakeGuard(),c,g,job,{'stages':{'response-00':{'status':status}}},0,0,[],lambda *x:None)
        self.assertEqual(q.ledger['consumed_cpu_seconds'],210160.8308359999)

    def test_unsealed_ledger_phase_needs_forensics_not_missing_PID_guess(self):
        c,g=fixture();job=m.stage_job(FakeQ(),c,'response-00');q=FakeQ();q.ledger['phases']=[{'phase':job['phase'],'status':'PARTIAL_OR_FAILED'}]
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('No Q in tests')):
            with self.assertRaises(ValueError):m.launch_quantum(q,FakeGuard(),c,g,job,{'stages':{}},0,0,[],lambda *x:None)

    def test_transient_conflict_waits_without_stopping_any_child(self):
        q=FakeQ();q.process_observation=unittest.mock.Mock(side_effect=[{'conflicts':['app-git']},{'conflicts':[]}]);events=[]
        with patch.object(m.time,'monotonic',return_value=100),patch.object(m.time,'time',return_value=100),patch.object(m.time,'sleep'):
            m.wait_for_clear(q,{'conflict_wait_seconds':600},1000,1000,lambda *v:events.append(v),[])
        self.assertEqual(len(events),1)

    def test_persistent_conflict_has_finite_wait_and_no_subprocess(self):
        q=FakeQ();q.process_observation=lambda **x:{'conflicts':['unknown']}
        with patch.object(m.time,'monotonic',return_value=100),patch.object(m.time,'time',return_value=100),patch.object(m.subprocess,'Popen',side_effect=AssertionError('No Q')):
            with self.assertRaises(ValueError):m.wait_for_clear(q,{'conflict_wait_seconds':0},1000,1000,lambda *v:None,[])

    def test_all_subprocesses_detached_and_stdin_DEVNULL(self):
        tree=ast.parse(Path(m.__file__).read_text());calls=[v for v in ast.walk(tree) if isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr=='Popen']
        self.assertEqual(len(calls),3)
        for call in calls:
            kw={v.arg:v.value for v in call.keywords};self.assertIs(kw['start_new_session'].value,True)
            self.assertEqual(ast.dump(kw['stdin']),ast.dump(ast.Attribute(ast.Name('subprocess',ast.Load()),'DEVNULL',ast.Load())))

    def test_new_control_and_legacy_H1_qualification_happen_before_stage_launch(self):
        src=Path(m.__file__).read_text();start=src.index('def run(args')
        self.assertLess(src.index('consumer.require_control_release',start),src.index('for stage in STAGES:',start))
        self.assertLess(src.index("consumer.qualify_stage(config['h1_result']",start),src.index('for stage in STAGES:',start))
        self.assertNotIn('register(',src)

    def test_signed_negative_modes_are_not_relabelled_as_missing_completion(self):
        src=Path(m.__file__).read_text();self.assertIn('qualified_for_mechanism=checks["qualified_for_mechanism"]',src)
        self.assertNotIn('checks["all_frequencies_positive"]',src)
        self.assertIn('checks["IR_intensities_status"] == "NOT_COMPUTED"',src)

    def test_detached_resume_uses_new_console_and_receipt_per_attempt(self):
        src=Path(m.__file__).read_text();self.assertIn("'driver-'+args.attempt+'.console.log'",src)
        self.assertIn("'detached-launch-'+args.attempt+'.json'",src)
        self.assertIn("'BOUNDARY_CLOSED'",src)
        self.assertIn("'current-launch.json'",src)

if __name__=='__main__':unittest.main()
