"""Tiny fixtures and fake child tests. No research process, advice or Git runs."""
import copy
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parent/'cache_aware_controller.py'
spec=importlib.util.spec_from_file_location('runtime_under_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ORIGINAL_SUPERVISOR=Path('/workspace/research4-relaunch/scripts/species_background_budget.py')


class CacheAwareTests(unittest.TestCase):
    @staticmethod
    def ref(path):return {'path':str(path),'sha256':m.sha(path)}
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cache-aware-controller-tests-')
        self.here=Path(self.tmp.name);self.target=self.here/'science';self.runroot=self.target/'data/research/species_relaunch_20261108_v2'
        self.sup=self.target/'scripts/species_background_budget.py';self.sup.parent.mkdir(parents=True)
        shutil.copyfile(ORIGINAL_SUPERVISOR,self.sup)
        self.wrapper=self.target/'scripts/species_relaunch_resources_20261108.py';self.wrapper.write_text('# historical frozen wrapper\n')
        self.worker=self.target/'scripts/species_revision_background_neural.py';self.worker.write_text('# tiny scientific placeholder, never run\n')
        for name in ['control','resource-control']:(self.runroot/name).mkdir(parents=True)
        (self.here/'evidence').mkdir()
        self.patches=[patch.object(m,'HERE',self.here),patch.object(m,'TARGET',self.target),patch.object(m,'OLD_RUN',self.runroot),patch.object(m,'SUPERVISOR',self.sup)]
        for p in self.patches:p.start()
        self.helper=self.here/'closed_cache_advice.py';self.helper.write_text('def advise_closed():\n    raise AssertionError("tests must not advise real files")\n')
        def job(name):return {'id':name,'kind':'neural','command':[sys.executable,str(self.worker),'--output',str(self.runroot/'runs'/name)],'cwd':str(self.target),'inputs':[str(self.worker)],'outputs':[],'max_retries':1}
        self.original_data={'cpu_budget_seconds':88000,'jobs':[job('done'),job('pending')]}
        self.original=self.runroot/'jobs.json';self.original.write_text(json.dumps(self.original_data))
        self.prior_config=self.here/'evidence/original_config.json'
        self.prior_config.write_text(json.dumps({'deadline_utc':m.DEADLINE.isoformat(),'total_budget_cpu_seconds':90000,
            'inner_budget_cpu_seconds':88000,'manifest_ref':self.ref(self.original),'wrapper_source_sha256':m.sha(self.wrapper)}))
        identity={'pid':424240,'start_ticks':1,'session':424240,'pgrp':424240,'boot_id':'closed-test-boot','uid':os.getuid()}
        self.old_inner={'status':'paused','active':None,'owner':identity,'spent_cpu_seconds':180.,'budget_cpu_seconds':88000,
            'supervisor_code_sha256':m.SUPERVISOR_SHA,'input_hashes':{'done':{str(self.worker):m.sha(self.worker)},'pending':{str(self.worker):m.sha(self.worker)}},
            'jobs':{'done':{'status':'completed'}}}
        self.old_outer={'status':'PAUSED_RESOURCE','active':None,'closure_confirmed':True,'identity':identity,
            'spent_cpu_seconds':200.,'inner_actual_spent_cpu_seconds':185.,'wall_seconds':400.,'config_sha256':m.sha(self.prior_config)}
        self.inner_copy=self.here/'evidence/prior_inner_closed.json';self.inner_copy.write_text(json.dumps(self.old_inner))
        self.outer_copy=self.here/'evidence/prior_outer_closed.json';self.outer_copy.write_text(json.dumps(self.old_outer))
        shutil.copyfile(self.inner_copy,self.runroot/'control/state.json');shutil.copyfile(self.outer_copy,self.runroot/'resource-control/state.json')
        self.cfg={'schema':m.SCHEMA,'controller_source_sha256':m.sha(SOURCE),'target_root':str(self.target),'run_root':str(self.runroot),
            'workdir':str(self.here/'work'),'prior_outer_ref':self.ref(self.outer_copy),'prior_inner_ref':self.ref(self.inner_copy),
            'prior_config_ref':self.ref(self.prior_config),'supervisor_ref':self.ref(self.sup),'manifest_ref':self.ref(self.original),
            'helper_ref':self.ref(self.helper),'total_cpu_cap_seconds':90000,'prior_total_cpu_seconds':200.,
            'deadline_utc':m.DEADLINE.isoformat(),'python':sys.executable,'_config_sha256':'e'*64}
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def b(self):return m.helpers()
    def remaining(self):
        script=self.here/'compact.py';script.write_text('# runtime auxiliary, never run')
        self.remaining_data={'cpu_budget_seconds':88000,'jobs':[copy.deepcopy(self.original_data['jobs'][1]),
            {'id':'new_compact','kind':'summary','command':[sys.executable,str(script)],'inputs':[],'outputs':[],'max_retries':0}]}
        self.remaining_path=self.here/'remaining_jobs.json';self.remaining_path.write_text(json.dumps(self.remaining_data))
        self.cfg.update(execution_mode='remaining_manifest',original_manifest_ref=self.ref(self.original),
                        manifest_ref=self.ref(self.remaining_path),state_dir=str(self.here/'work/control'))
    def test_original_readonly_preflight(self):
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('no child')):
            p=m.validate(self.cfg,self.b());self.assertEqual(p['prior_total'],200.);self.assertEqual(p['action'],'resume')
    def test_remaining_exact_science_and_new_local_ledger(self):
        self.remaining();p=m.validate(self.cfg,self.b())
        self.assertEqual(p['remaining_scientific_jobs'],1);self.assertEqual(p['action'],'run')
        self.assertNotEqual(p['state_dir'],self.runroot/'control')
    def test_remaining_cannot_modify_scientific_command(self):
        self.remaining();self.remaining_data['jobs'][0]['command'].append('--silently-change')
        self.remaining_path.write_text(json.dumps(self.remaining_data));self.cfg['manifest_ref']=self.ref(self.remaining_path)
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_remaining_cannot_rerun_completed_science(self):
        self.remaining();self.remaining_data['jobs'].append(copy.deepcopy(self.original_data['jobs'][0]))
        self.remaining_path.write_text(json.dumps(self.remaining_data));self.cfg['manifest_ref']=self.ref(self.remaining_path)
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_remaining_must_cover_all_pending_science(self):
        self.remaining();self.remaining_data['jobs']=self.remaining_data['jobs'][1:]
        self.remaining_path.write_text(json.dumps(self.remaining_data));self.cfg['manifest_ref']=self.ref(self.remaining_path)
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_remaining_no_reset_of_unknown_local_ledger(self):
        self.remaining();p=Path(self.cfg['state_dir']);p.mkdir(parents=True);(p/'state.json').write_text('{}')
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_actual_prior_total_cannot_be_replaced_with_smaller_inner_cost(self):
        self.cfg['prior_total_cpu_seconds']=180.
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_original_scientific_source_hash_cannot_change(self):
        self.worker.write_text('# changed')
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_original_ledger_must_stay_historical_and_identical(self):
        (self.runroot/'resource-control/state.json').write_text('{}')
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_original_unknown_active_never_adopted(self):
        obj=copy.deepcopy(self.old_inner);obj['active']={'pid':123};(self.runroot/'control/state.json').write_text(json.dumps(obj))
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def test_matching_live_prior_identity_blocks_without_signal(self):
        b=self.b()
        with patch.object(b,'same_process',return_value=True),patch.object(b,'proc_stat',return_value={'state':'S'}):
            with self.assertRaises(RuntimeError):m.validate(self.cfg,b)
    def test_pagecache_above12GiB_is_not_a_stop(self):
        mem={'current_bytes':15*m.GIB,'file_bytes':13*m.GIB,'anon_plus_kernel_bytes':2*m.GIB}
        self.assertIsNone(m.stop_reason(500.,datetime.now(timezone.utc),2*m.GIB,mem))
        self.assertTrue(m.should_advise(mem,None,100.))
    def test_true_anon_kernel_emergency_stops_only_for_real_pressure(self):
        mem={'current_bytes':15*m.GIB,'file_bytes':m.GIB,'anon_plus_kernel_bytes':14*m.GIB}
        self.assertEqual(m.stop_reason(500.,datetime.now(timezone.utc),2*m.GIB,mem),'EMERGENCY_MEMORY')
    def test_no100_second_controller_suballocation_stop(self):
        self.assertIsNone(m.stop_reason(1200.,datetime.now(timezone.utc),2*m.GIB,{'anon_plus_kernel_bytes':m.GIB}))
    def test_total_cap_and_real_disk_floor_still_apply(self):
        mem={'anon_plus_kernel_bytes':m.GIB}
        self.assertEqual(m.stop_reason(89940.,datetime.now(timezone.utc),2*m.GIB,mem),'CPU_BUDGET')
        self.assertEqual(m.stop_reason(500.,datetime.now(timezone.utc),m.GIB-1,mem),'DISK')
    def test_cache_advice_is_limited_to15seconds(self):
        mem={'current_bytes':10*m.GIB,'file_bytes':6*m.GIB}
        self.assertFalse(m.should_advise(mem,100.,114.9));self.assertTrue(m.should_advise(mem,100.,115.))
    def test_total_cap_may_not_expand(self):
        self.cfg['total_cpu_cap_seconds']=90001
        with self.assertRaises(RuntimeError):m.validate(self.cfg,self.b())
    def fake_run(self,*,resume=False,inner_status='completed',returncode=0):
        self.remaining()
        cfg_path=self.here/'controller_config.json';cfg_path.write_text(json.dumps(self.cfg));self.cfg['_config_sha256']=m.sha(cfg_path)
        if resume:
            work=Path(self.cfg['workdir']);work.mkdir()
            (work/'state.json').write_text(json.dumps({'status':'PAUSED_EXTERNAL','active':None,'closure_confirmed':True,
                'config_sha256':self.cfg['_config_sha256'],'spent_cpu_seconds':250.,'wall_seconds':500.}))
            local=Path(self.cfg['state_dir']);local.mkdir();(local/'state.json').write_text(json.dumps({'status':'paused','active':None}))
        b=self.b();actual_stat=b.proc_stat
        class FakeProcess:
            pid=424242
            def __init__(self):self.returncode=returncode
            def poll(self):return self.returncode
            def wait(self,timeout=None):return self.returncode
        def spawn(*args,**kwargs):
            local=Path(self.cfg['state_dir']);local.mkdir(exist_ok=True);(local/'state.json').write_text(json.dumps({'status':inner_status,'active':None}));return FakeProcess()
        def stat(pid):
            if pid==424242:return {'pid':pid,'start_ticks':55,'pgrp':pid,'session':pid,'state':'Z','cpu_seconds':0.}
            return actual_stat(pid)
        with patch.object(m.subprocess,'Popen',side_effect=spawn),patch.object(b,'proc_stat',side_effect=stat),\
             patch.object(b,'owned_members',return_value=[]),patch.object(b,'signal_owned_groups',side_effect=AssertionError('no actual signals')),\
             patch.object(m,'usage',side_effect=lambda kind:3. if kind==m.resource.RUSAGE_SELF else 0.),\
             patch.object(m,'memory_info',return_value={'current_bytes':15*m.GIB,'file_bytes':14*m.GIB,'anon_plus_kernel_bytes':m.GIB}):
            return m.run(self.cfg,cfg_path,b,resume=resume)
    def test_actual_total_carry_keeps_all_prior_failure_cpu(self):
        state=self.fake_run();self.assertEqual(state['status'],'COMPLETED')
        self.assertAlmostEqual(state['spent_cpu_seconds'],203.)
        self.assertEqual(state['invocation_base_total_cpu_seconds'],200.)
    def test_resume_uses_controller_cumulative_not_second_carry(self):
        state=self.fake_run(resume=True);self.assertEqual(state['status'],'COMPLETED')
        self.assertAlmostEqual(state['spent_cpu_seconds'],253.)
        self.assertEqual(state['invocation_base_total_cpu_seconds'],250.)
    def test_failed_stage_not_falsely_complete(self):
        state=self.fake_run(inner_status='completed_with_errors',returncode=1)
        self.assertEqual(state['status'],'COMPLETED_WITH_ERRORS')


if __name__=='__main__':unittest.main()
