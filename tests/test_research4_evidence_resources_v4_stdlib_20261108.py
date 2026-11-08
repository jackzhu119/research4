"""Closed-cost carry tests using tiny fixtures and fake processes only."""
import copy
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/research4_evidence_resources_20261108_v4.py'
spec=importlib.util.spec_from_file_location('v4_resource_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class CarryTests(unittest.TestCase):
    @staticmethod
    def ref(path):return {'path':str(path),'sha256':m.sha(path)}
    def setUp(self):
        m.BASE.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=m.BASE,prefix='unit-v4-carry-')
        self.base=Path(self.tmp.name);self.patcher=patch.object(m,'BASE',self.base);self.patcher.start()
        self.manifest=self.base/'jobs.json';self.manifest.write_text(json.dumps({'run_id':'evidence-validation-20261108-v4','cpu_budget_seconds':88000,'jobs':[]}))
        self.prep=self.base/'prep.json';self.prep.write_text(json.dumps({'consumed_cpu_seconds':10.}))
        self.cfg={'schema':m.SCHEMA,'run_id':'evidence-validation-20261108-v4','wrapper_source_sha256':m.sha(SOURCE),
            'supervisor_ref':self.ref(m.SUPERVISOR),'manifest_ref':self.ref(self.manifest),'preparation_ref':self.ref(self.prep),
            'preparation_cpu_seconds':10.,'total_budget_cpu_seconds':90000,'inner_budget_cpu_seconds':88000,
            'prep_budget_cpu_seconds':1000,'controller_budget_cpu_seconds':1000,'wall_cap_seconds':172800,
            'authorization_observed_utc':'2026-11-08T08:44:13.376487+00:00','deadline_utc':'2026-11-10T08:44:13.376487+00:00',
            'rss_stop_bytes':int(1.75*1024**3),'cgroup_stop_bytes':int(14.5*1024**3),'minimum_disk_bytes':2*1024**3,
            'state_dir':str(self.base/'control'),'workroot':str(self.base/'outer'),'python':sys.executable,'_config_sha256':'e'*64}
        prior_config={**self.cfg,'run_id':'evidence-validation-20261108-v3','preparation_cpu_seconds':8.,
                      'wrapper_source_sha256':m.sha(ROOT/'scripts/research4_evidence_resources_20261108.py')}
        self.prior_config=self.base/'prior_config.json';self.prior_config.write_text(json.dumps(prior_config))
        identity={'pid':424240,'start_ticks':123,'pgrp':424240,'session':424240,'boot_id':'unit-test-closed-boot','uid':os.getuid()}
        self.outer=self.base/'prior_outer.json';self.outer.write_text(json.dumps({'status':'PAUSED_RESOURCE','active':None,
            'closure_confirmed':True,'run_id':'evidence-validation-20261108-v3','identity':identity,
            'controller_spent_cpu_seconds':.2,'inner_actual_spent_cpu_seconds':17.5,'wall_seconds':20.,
            'preparation_cpu_seconds':8.,'spent_cpu_seconds':25.7,'config_sha256':m.sha(self.prior_config)}))
        self.inner=self.base/'prior_inner.json';self.inner.write_text(json.dumps({'status':'paused','active':None,'owner':identity,'spent_cpu_seconds':17.4}))
        self.carry_path=self.base/'prior_execution_carry.json'
        self.carry={'schema':'research4-closed-execution-carry-v1','grant_id':'evidence-validation-20261108-v3',
            'prior_run_id':'evidence-validation-20261108-v3','prior_closed':True,'controller_spent_cpu_seconds':.2,
            'inner_actual_spent_cpu_seconds':17.5,'wall_seconds':20.,'deadline_utc':self.cfg['deadline_utc'],
            'outer_pause_ref':self.ref(self.outer),'inner_pause_ref':self.ref(self.inner),
            'prior_config_ref':self.ref(self.prior_config),'original_outer_state_sha256':m.sha(self.outer),
            'original_inner_state_sha256':m.sha(self.inner)}
        self.freeze()
    def tearDown(self):self.patcher.stop();self.tmp.cleanup()
    def freeze(self):
        self.carry_path.write_text(json.dumps(self.carry));self.cfg['prior_execution_ref']=self.ref(self.carry_path)
    def change_snapshot(self,path,**changes):
        obj=json.loads(path.read_text());obj.update(changes);path.write_text(json.dumps(obj))
        key='outer' if path==self.outer else 'inner'
        self.carry[key+'_pause_ref']=self.ref(path);self.carry['original_'+key+'_state_sha256']=m.sha(path);self.freeze()
    def test_exact_closed_costs_validate(self):
        self.assertEqual(m.prior_execution(self.cfg)['inner_actual_spent_cpu_seconds'],17.5)
        m.validate(self.cfg)
    def test_known_failure_cost_cannot_be_refunded(self):
        self.carry['inner_actual_spent_cpu_seconds']=17.;self.freeze()
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_negative_nan_and_missing_costs_rejected(self):
        for key in ['controller_spent_cpu_seconds','inner_actual_spent_cpu_seconds','wall_seconds']:
            for value in [-1,float('nan'),float('inf'),None]:
                saved=copy.deepcopy(self.carry);self.carry[key]=value;self.freeze()
                with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
                self.carry=saved;self.freeze()
    def test_prior_closure_required(self):
        self.carry['prior_closed']=False;self.freeze()
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_original_snapshot_sha_must_match(self):
        self.carry['original_outer_state_sha256']='0'*64;self.freeze()
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_carry_file_hash_drift_rejected(self):
        self.carry_path.write_text('{}')
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_outer_unclean_active_rejected_even_with_updated_seals(self):
        self.change_snapshot(self.outer,active={'pid':424241})
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_inner_unclean_active_rejected(self):
        self.change_snapshot(self.inner,active={'pid':424241})
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_outer_cannot_discard_higher_inner_observation(self):
        self.change_snapshot(self.inner,spent_cpu_seconds=19.)
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_deadline_cannot_be_extended(self):
        self.cfg['deadline_utc']='2026-11-10T09:44:13.376487+00:00'
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_authorization_cannot_restart(self):
        self.cfg['authorization_observed_utc']='2026-11-08T09:44:13.376487+00:00'
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_preparation_is_cumulative_not_refunded(self):
        self.cfg['preparation_cpu_seconds']=7.
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_whole_grant_caps_unchanged(self):
        self.cfg['inner_budget_cpu_seconds']=90000
        with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def test_matching_live_prior_identity_blocks_without_signal(self):
        b=m.load_supervisor()
        with patch.object(m,'load_supervisor',return_value=b),patch.object(b,'same_process',return_value=True),\
             patch.object(b,'proc_stat',return_value={'state':'S'}):
            with self.assertRaises(RuntimeError):m.prior_execution(self.cfg)
    def fake_run(self,resume=False):
        cfgpath=self.base/'current_config.json';cfgpath.write_text(json.dumps(self.cfg));self.cfg['_config_sha256']=m.sha(cfgpath)
        if resume:
            p=Path(self.cfg['state_dir']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None,'status':'paused'}))
            p=Path(self.cfg['workroot']);p.mkdir();(p/'state.json').write_text(json.dumps({'active':None,'closure_confirmed':True,
                'status':'PAUSED_RESOURCE','config_sha256':self.cfg['_config_sha256'],'controller_spent_cpu_seconds':3.2,
                'inner_actual_spent_cpu_seconds':21.5,'wall_seconds':50.}))
        b=m.load_supervisor();actual_stat=b.proc_stat
        class Fake:
            pid=424242;returncode=0
            def poll(self):return 0
            def wait(self,timeout=None):return 0
        def spawn(*args,**kwargs):
            p=Path(self.cfg['state_dir']);p.mkdir(exist_ok=True);(p/'state.json').write_text(json.dumps({'active':None,'status':'completed'}));return Fake()
        def stat(pid):
            if pid==424242:return {'pid':pid,'start_ticks':55,'pgrp':pid,'session':pid,'state':'Z','cpu_seconds':0.}
            return actual_stat(pid)
        with patch.object(m.subprocess,'Popen',side_effect=spawn),patch.object(b,'proc_stat',side_effect=stat),\
             patch.object(b,'owned_members',return_value=[]),patch.object(b,'signal_owned_groups',side_effect=AssertionError('No actual signals')) ,\
             patch.object(m,'usage',side_effect=lambda kind:2. if kind==m.resource.RUSAGE_SELF else 0.):
            return m.run(self.cfg,cfgpath,b,resume=resume)
    def test_fresh_v4_starts_from_carried_costs(self):
        state=self.fake_run()
        self.assertAlmostEqual(state['controller_spent_cpu_seconds'],2.2)
        self.assertAlmostEqual(state['inner_actual_spent_cpu_seconds'],17.5)
        self.assertAlmostEqual(state['spent_cpu_seconds'],29.7)
        self.assertGreaterEqual(state['wall_seconds'],20.)
        self.assertTrue(state['carry_applied_this_invocation'])
    def test_v4_resume_uses_live_total_without_second_carry(self):
        state=self.fake_run(resume=True)
        self.assertAlmostEqual(state['controller_spent_cpu_seconds'],5.2)
        self.assertAlmostEqual(state['inner_actual_spent_cpu_seconds'],21.5)
        self.assertAlmostEqual(state['spent_cpu_seconds'],36.7)
        self.assertGreaterEqual(state['wall_seconds'],50.)
        self.assertFalse(state['carry_applied_this_invocation'])
    def test_primary_cap_counts_prior_plus_new_cpu(self):
        reason=m.resource_reason(1.,17.5+87918.5,87960.,100.,self.cfg,rss=1,cgroup=1,
                                 free_disk=3*1024**3,now=datetime.now(timezone.utc))
        self.assertEqual(reason,'inner_actual_cpu_allocation')


if __name__=='__main__':unittest.main()
