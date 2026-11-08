"""Small synthetic resource/accounting checks; no models or background jobs."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/species_relaunch_resources_20261108.py'
spec=importlib.util.spec_from_file_location('resource_wrapper_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Clock:
    fromisoformat=staticmethod(datetime.fromisoformat)
    @staticmethod
    def now(tz):return datetime(2026,11,8,6,tzinfo=timezone.utc)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        m.BASE.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=m.BASE,prefix='unit-resource-')
        self.root=Path(self.temp.name)
        self.manifest=self.root/'jobs.json';self.manifest.write_text(json.dumps({'run_id':'species-relaunch-20261108-v2','cpu_budget_seconds':88000,'jobs':[]}))
        self.prep=self.root/'prep.json';self.prep.write_text(json.dumps({'spent_cpu_seconds':23.8}))
        self.cfg={'schema':m.SCHEMA,'run_id':'species-relaunch-20261108-v2','wrapper_source_sha256':m.sha(SOURCE),
            'supervisor_ref':self.ref(m.SUPERVISOR),'manifest_ref':self.ref(self.manifest),'preparation_ref':self.ref(self.prep),
            'preparation_cpu_seconds':23.8,'total_budget_cpu_seconds':90000,'inner_budget_cpu_seconds':88000,
            'prep_budget_cpu_seconds':1900,'controller_budget_cpu_seconds':100,'wall_cap_seconds':129600,
            'deadline_utc':'2026-11-10T02:50:37.289380+00:00','rss_stop_bytes':10*1024**3,'cgroup_stop_bytes':12*1024**3,
            'minimum_disk_bytes':1024**3,'state_dir':str(self.root/'inner'),'workroot':str(self.root/'outer'),
            'python':sys.executable,'_config_sha256':'e'*64}
        self.clock=patch.object(m,'datetime',Clock);self.clock.start()
    def tearDown(self):self.clock.stop();self.temp.cleanup()
    @staticmethod
    def ref(p):return {'path':str(p),'sha256':m.sha(p)}
    def reason(self,**over):
        args={'controller':2.,'inner':3.,'total':29.,'wall':10.,'cfg':self.cfg,
            'rss':100000,'cgroup':100000,'free_disk':2*1024**3,'now':Clock.now(timezone.utc)}
        args.update(over);return m.resource_reason(**args)
    def test_exact_allocation_is90000(self):self.assertEqual(m.limits(self.cfg)[0],23.8)
    def test_allocation_cannot_expand(self):
        for key in ['total_budget_cpu_seconds','inner_budget_cpu_seconds','prep_budget_cpu_seconds','controller_budget_cpu_seconds']:
            cfg=copy.deepcopy(self.cfg);cfg[key]+=1
            with self.assertRaises(RuntimeError):m.limits(cfg)
    def test_prep_nan_blocked(self):
        self.cfg['preparation_cpu_seconds']=float('nan')
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_prep_overrun_blocked(self):
        self.cfg['preparation_cpu_seconds']=1900.1
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_deadline_extension_blocked(self):
        self.cfg['deadline_utc']='2026-11-11T00:00:00+00:00'
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_wall_extension_blocked(self):
        self.cfg['wall_cap_seconds']=129601
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_disk_floor_cannot_reduce(self):
        self.cfg['minimum_disk_bytes']=1024
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_rss_cap_cannot_expand(self):
        self.cfg['rss_stop_bytes']=11*1024**3
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_cgroup_cap_cannot_expand(self):
        self.cfg['cgroup_stop_bytes']=13*1024**3
        with self.assertRaises(RuntimeError):m.limits(self.cfg)
    def test_normal_resources_do_not_stop(self):self.assertIsNone(self.reason())
    def test_controller_stop_retains_cleanup_margin(self):self.assertEqual(self.reason(controller=90.),'controller_cpu_allocation')
    def test_inner_stop_retains64_cpu_seconds(self):self.assertEqual(self.reason(inner=87936.),'inner_actual_cpu_allocation')
    def test_total_stop_retains64_cpu_seconds(self):self.assertEqual(self.reason(total=89936.),'total_cpu_allocation')
    def test_memory_stop(self):self.assertEqual(self.reason(rss=10*1024**3),'memory_limit')
    def test_cgroup_stop(self):self.assertEqual(self.reason(cgroup=12*1024**3),'memory_limit')
    def test_disk_stop(self):self.assertEqual(self.reason(free_disk=1),'disk_floor')
    def test_wall_stop(self):self.assertEqual(self.reason(wall=129600.),'wall_or_deadline')
    def test_date_stop(self):self.assertEqual(self.reason(now=datetime(2026,11,11,tzinfo=timezone.utc)),'wall_or_deadline')
    def test_fresh_validation_starts_no_process(self):
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch')):m.validate(self.cfg)
    def test_old_runid_cannot_be_reused(self):
        self.cfg['run_id']='species-background-20261108-v1'
        with self.assertRaises(RuntimeError):m.validate(self.cfg)
    def test_source_drift_blocks(self):
        self.cfg['wrapper_source_sha256']='0'*64
        with self.assertRaises(RuntimeError):m.validate(self.cfg)
    def test_manifest_drift_blocks(self):
        self.manifest.write_text('{}')
        with self.assertRaises(RuntimeError):m.validate(self.cfg)
    def test_prep_measurement_mismatch_blocks(self):
        self.cfg['preparation_cpu_seconds']=0
        with self.assertRaises(RuntimeError):m.validate(self.cfg)
    def test_root_consumed_cpu_preparation_field_is_hash_bound(self):
        self.prep.write_text(json.dumps({'consumed_cpu_seconds':23.8}))
        self.cfg['preparation_ref']=self.ref(self.prep)
        m.validate(self.cfg)
    def test_state_outside_new_grant_blocks(self):
        self.cfg['state_dir']=str(ROOT/'data/research/species_background_20261108/control')
        with self.assertRaises(RuntimeError):m.validate(self.cfg)
    def test_existing_inner_state_blocks_fresh_run(self):
        p=Path(self.cfg['state_dir']);p.mkdir();(p/'state.json').write_text('{}')
        with self.assertRaises(RuntimeError):m.validate(self.cfg)
    def test_clean_resume_preserves_configs(self):
        p=Path(self.cfg['state_dir']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None}))
        p=Path(self.cfg['workroot']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None,'closure_confirmed':True,'status':'PAUSED_RESOURCE','config_sha256':'e'*64,'controller_spent_cpu_seconds':4.,'inner_actual_spent_cpu_seconds':10.,'wall_seconds':12.}))
        m.validate(self.cfg,resume=True)
    def test_resume_missing_or_nan_cost_cannot_reset_allocation(self):
        p=Path(self.cfg['state_dir']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None}))
        p=Path(self.cfg['workroot']);p.mkdir()
        base={'active':None,'closure_confirmed':True,'status':'PAUSED_RESOURCE','config_sha256':'e'*64,'controller_spent_cpu_seconds':4.,'inner_actual_spent_cpu_seconds':10.,'wall_seconds':12.}
        for key in ['controller_spent_cpu_seconds','inner_actual_spent_cpu_seconds','wall_seconds']:
            for value in [None,-1,float('nan')]:
                changed=dict(base)
                if value is None:del changed[key]
                else:changed[key]=value
                (p/'state.json').write_text(json.dumps(changed))
                with self.assertRaises(RuntimeError):m.validate(self.cfg,resume=True)
    def test_unclean_outer_resume_blocks(self):
        p=Path(self.cfg['workroot']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':{'pid':123},'closure_confirmed':False}))
        with self.assertRaises(RuntimeError):m.validate(self.cfg,resume=True)
    def test_completed_queue_cannot_be_burned_again(self):
        p=Path(self.cfg['state_dir']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None}))
        p=Path(self.cfg['workroot']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None,'closure_confirmed':True,'status':'COMPLETED','config_sha256':'e'*64}))
        with self.assertRaises(RuntimeError):m.validate(self.cfg,resume=True)
    def test_original_source_is_preserved_when_helpers_imported(self):
        before=m.sha(m.SUPERVISOR);b=m.load_supervisor();self.assertEqual(before,m.SUPERVISOR_SHA)
        self.assertEqual(m.sha(m.SUPERVISOR),before);self.assertEqual(b.TOKEN_KEY,m.TOKEN)
    def run_fake(self, *, controller_cpu=1., before_spawn=None):
        """Exercise controller lifecycle with a fake process, never an OS child."""
        cfgfile=self.root/'config.json';cfgfile.write_text(json.dumps(self.cfg))
        self.cfg['_config_sha256']=m.sha(cfgfile)
        b=m.load_supervisor();actual_stat=b.proc_stat
        class FakeProcess:
            pid=424242;returncode=0
            def poll(self):return 0
            def wait(self,timeout=None):return 0
        def fake_spawn(*args,**kwargs):
            saved=m.read(Path(self.cfg['workroot'])/'state.json')
            self.assertEqual(saved['status'],'STARTING');self.assertFalse(saved['closure_confirmed'])
            if before_spawn:before_spawn(saved)
            return FakeProcess()
        def fake_stat(pid):
            if pid==os.getpid():return actual_stat(pid)
            return {'pid':pid,'start_ticks':123,'pgrp':pid,'session':pid,'cpu_seconds':0.,'state':'Z'}
        with patch.object(m.subprocess,'Popen',side_effect=fake_spawn) as launch, \
             patch.object(m,'usage',side_effect=lambda kind:controller_cpu if kind==m.resource.RUSAGE_SELF else 0.), \
             patch.object(b,'proc_stat',side_effect=fake_stat),patch.object(b,'owned_members',return_value=[]), \
             patch.object(b,'signal_owned_groups',side_effect=AssertionError('no real signals')), \
             patch.object(m,'validate',wraps=m.validate) as validation:
            result=m.run(self.cfg,cfgfile,b,resume=False)
        return result,launch.call_count,validation.call_count
    def test_fake_completed_lifecycle_persists_clean_closure(self):
        result,launches,validations=self.run_fake()
        self.assertEqual((launches,validations),(1,2))
        self.assertEqual(result['status'],'COMPLETED');self.assertIsNone(result['active'])
        self.assertTrue(result['closure_confirmed']);self.assertEqual(result['spent_cpu_seconds'],24.8)
    def test_fake_prelaunch_exhaustion_does_not_spawn(self):
        result,launches,validations=self.run_fake(controller_cpu=90.)
        self.assertEqual((launches,validations),(0,2))
        self.assertEqual(result['status'],'BLOCKED_REQUIRES_ROOT_RECONCILIATION')
        self.assertEqual(result['spent_cpu_seconds'],113.8)


if __name__=='__main__':unittest.main()
