"""Stdlib metadata/ordering tests only; no numerical imports or molecular IO."""
from argparse import Namespace
import ast
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/pyscf_calycosin_optfreq_pilot_20261105.py'


class CalycosinPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('calycosin_metadata_tests',SOURCE)
        cls.w=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.w)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.output=self.path/'future_output'
        self.ledger_path=self.path/'ledger.json';self.main_path=self.path/'spinosin/result.json'
        self.main_path.parent.mkdir();self.analysis_path=self.path/'main_checks.json'
        self.auth_path=self.path/'synthetic_root_decision.json'
        self.proof={'cpu_hard_limit_seconds':7200,'wall_limit_seconds':7200,'threads':2,'ledger_path':str(self.ledger_path)}
        self.main={'schema':'research4-staged-full-rks-hessian-result-v6b','stage':'global','phase':'global',
            'status':'FREQUENCIES_COMPUTED','case':'Spinosin','atoms':75,'normal_mode_count':219,
            'complete_global_Hessian':True,'normal_modes_complete':True,
            'source_worker':{'path':str(self.w.MAIN_WORKER),'sha256':self.w.MAIN_WORKER_SHA},
            'resource_supervision':{'proof':{'ledger_path':str(self.ledger_path),'phase':'main_C'}},
            'snapshot_manifest_sha256':'3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311'}
        self.write(self.main_path,self.main)
        self.analysis={'schema':'research4-V6b-independent-staged-modes-analysis-v1',
            'status':'DONE_COMPUTED_MODES_AUDITED','case':'Spinosin','atoms':75,'normal_modes':219,
            'normal_mode_count_complete':True,'source_result_sha256':self.w.sha(self.main_path),
            'analysis_source_sha256':self.w.MAIN_ANALYZER_SHA,'actual_V6b_worker_sha256':self.w.MAIN_WORKER_SHA,
            'actual_V6b_parent_sha256':self.w.GUARD_SHA,'new_quantum_evaluations':0,
            'independent_numpy_replay':{'status':'PASS_INDEPENDENT_NUMPY_TR_REPLAY'},
            'qualified_for_mechanism':False,'created_at_utc':'2026-11-04T00:30:00+00:00'}
        self.write(self.analysis_path,self.analysis)
        self.ledger={'phases':[{'phase':'main_C','ended_at_utc':'2026-11-04T00:00:00+00:00'}]}
        self.write(self.ledger_path,self.ledger)
        self.auth={'schema':self.w.AUTH_SCHEMA,'enable_execution':True,'after_spinosin_only':True,
            'no_budget_reset':True,'worker_sha256':self.w.sha(SOURCE),'supervisor_sha256':self.w.GUARD_SHA,
            'input_manifest_sha256':self.w.INPUT_SHA,'identity_overlay_sha256':self.w.OVERLAY_SHA,
            'output_directory':str(self.output),'selected_threads':2,'optimizer_maxsteps':35,
            'maximum_cpu_seconds':7200,'maximum_wall_seconds':7200,'existing_ledger_path':str(self.ledger_path),
            'completed_spinosin_result':self.w.artifact(self.main_path),
            'completed_spinosin_analysis_checks':self.w.artifact(self.analysis_path),
            'created_at_utc':'2026-11-04T01:00:00+00:00'}
        self.write(self.auth_path,self.auth)
        self.guard=SimpleNamespace(verify_closed_phase=Mock(return_value={'phase':'main_C','actual_cpu_seconds':100.,
            'producer_result_sha256':self.w.sha(self.main_path)}))

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')

    def args(self):
        return Namespace(authorization=self.auth_path,authorization_sha256=self.w.sha(self.auth_path),
            output=self.output,threads=2,maxsteps=35)

    def authorize(self):
        self.write(self.auth_path,self.auth)
        return self.w.authorize_after_spinosin(self.args(),{'proof':self.proof},self.guard)

    def update_main(self,**changes):
        self.main.update(changes);self.write(self.main_path,self.main)
        self.auth['completed_spinosin_result']=self.w.artifact(self.main_path)
        self.analysis['source_result_sha256']=self.w.sha(self.main_path)
        self.update_analysis()

    def update_analysis(self,**changes):
        self.analysis.update(changes);self.write(self.analysis_path,self.analysis)
        self.auth['completed_spinosin_analysis_checks']=self.w.artifact(self.analysis_path)

    def test_real_small_JSON_identity_overlay_preserves_historical_false_flag(self):
        value=self.w.identity_metadata()
        self.assertEqual((value['name'],value['atoms'],value['electrons']),('Calycosin',33,148))
        self.assertTrue(value['old_input_name_pending_flag_is_historical'])
        self.assertTrue(value['later_name_graph_evidence_applied_without_overwriting_history'])
        self.assertEqual(value['full_inchikey'],'ZZAJQOPSWWVMBI-UHFFFAOYSA-N')
        self.assertFalse(value['coordinate_download_bytes_independently_verified'])

    def test_valid_synthetic_post_main_gate_reaches_no_numeric_work(self):
        value=self.authorize()
        self.guard.verify_closed_phase.assert_called_once()
        self.assertTrue(value['existing_budget_only_no_registration'])
        self.assertFalse(value['main_minimum_qualification_separate_from_computation'])
        self.assertFalse(self.output.exists())

    def test_incomplete_main_cannot_yield_resources_to_pilot(self):
        for changes in ({'status':'PARTIAL'},{'stage':'rks-partial'},{'normal_mode_count':93},
                        {'case':'water_software_control'},{'atoms':33},{'complete_global_Hessian':False}):
            original=deepcopy(self.main);self.update_main(**changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.authorize()
            self.main=original;self.update_main()

    def test_main_independent_replay_required(self):
        for changes in ({'status':'NOT_RUN'},{'normal_modes':93},{'new_quantum_evaluations':1},
                        {'analysis_source_sha256':'0'*64},{'independent_numpy_replay':{'status':'FAILED'}}):
            original=deepcopy(self.analysis);self.update_analysis(**changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.authorize()
            self.analysis=original;self.update_analysis()

    def test_real_parent_closure_gate_not_a_self_report(self):
        self.guard.verify_closed_phase.side_effect=ValueError('actual parent failed')
        with self.assertRaisesRegex(ValueError,'actual parent failed'):self.authorize()

    def test_actual_closed_parent_result_hash_must_match_authorization(self):
        self.guard.verify_closed_phase.return_value['producer_result_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'result SHA differs'):self.authorize()

    def test_main_completion_cannot_come_from_a_different_budget_pool(self):
        self.main['resource_supervision']['proof']['ledger_path']=str(self.path/'other_pool.json')
        self.update_main()
        with self.assertRaisesRegex(ValueError,'same cumulative'):self.authorize()

    def test_prep_does_not_supply_enabled_authorization(self):
        for key,value in [('enable_execution',False),('after_spinosin_only',False),('no_budget_reset',False)]:
            original=deepcopy(self.auth);self.auth[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.authorize()
            self.auth=original

    def test_7200_cpu_and_wall_bounds_and_no_silent_clamp(self):
        for key,value in [('maximum_cpu_seconds',7201),('maximum_wall_seconds',7201),
                          ('maximum_cpu_seconds',True),('maximum_wall_seconds',0)]:
            original=deepcopy(self.auth);self.auth[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.authorize()
            self.auth=original
        self.proof['cpu_hard_limit_seconds']=3600
        with self.assertRaisesRegex(ValueError,'insufficient resources'):self.authorize()

    def test_actual_root_decision_must_follow_main_closure_and_replay(self):
        self.auth['created_at_utc']='2026-11-03T23:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'must actually follow'):self.authorize()

    def test_output_threads_input_and_source_bound(self):
        for key,value in [('selected_threads',1),('output_directory',str(self.path/'other')),
                          ('worker_sha256','0'*64),('input_manifest_sha256','0'*64),
                          ('identity_overlay_sha256','0'*64),('existing_ledger_path',str(self.path/'new_budget.json'))]:
            original=deepcopy(self.auth);self.auth[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.authorize()
            self.auth=original

    def test_max35_steps_and_one_or_two_threads(self):
        for changes in ({'maxsteps':36},{'maxsteps':0},{'threads':4},{'threads':True}):
            args=self.args();args.__dict__.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.w.argument_limits(args)

    def test_live_parent_failure_occurs_before_any_numerical_import(self):
        events=[]
        fake=SimpleNamespace(verify_resource_supervision=Mock(side_effect=ValueError('NO_LIVE_PARENT')))
        with patch.object(self.w,'frozen',return_value=fake),patch.object(self.w,'authorize_after_spinosin',side_effect=AssertionError('not reached')):
            with self.assertRaisesRegex(ValueError,'NO_LIVE_PARENT'):self.w.execute(self.args())

    def test_gate_order_parent_then_main_then_identity_before_numeric(self):
        events=[]
        fake=SimpleNamespace(verify_resource_supervision=lambda **kwargs:events.append('parent') or {})
        def authorize(*args):events.append('post_main');return {}
        def identity():events.append('identity');raise ValueError('STOP_BEFORE_NUMERICAL_IMPORT')
        with patch.object(self.w,'frozen',return_value=fake),patch.object(self.w,'authorize_after_spinosin',side_effect=authorize),patch.object(self.w,'identity_metadata',side_effect=identity):
            with self.assertRaisesRegex(ValueError,'STOP_BEFORE_NUMERICAL_IMPORT'):self.w.execute(self.args())
        self.assertEqual(events,['parent','post_main','identity'])

    def test_module_top_level_is_stdlib_only(self):
        tree=ast.parse(SOURCE.read_text())
        banned={'numpy','scipy','pyscf','torch','rdkit','geometric'}
        for node in tree.body:
            if isinstance(node,ast.Import):names=[n.name.split('.')[0] for n in node.names]
            elif isinstance(node,ast.ImportFrom):names=[(node.module or '').split('.')[0]]
            else:continue
            self.assertFalse(banned.intersection(names))
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('pyscf',sys.modules)

    def test_stock_Hessian_and_all93_signed_modes_not_old_case_hack(self):
        text=SOURCE.read_text();tree=ast.parse(text)
        self.assertIn('hobj.kernel(atmlst=list(range(33)))',text)
        self.assertIn('imaginary_freq=False',text)
        self.assertIn("info['normal_mode_count']==93",text)
        self.assertNotIn('case_for_symbols(',text)
        self.assertNotIn('save_snapshot(',text)
        self.assertNotIn('restore_snapshot(',text)
        self.assertNotIn('run_checkpointed_hessian(',text)
        self.assertNotIn('initialize(',text)
        self.assertNotIn('register_continuation(',text)

    def test_pilot_settings_explicitly_noD3_noDF(self):
        settings=self.w.settings_document(2,35)
        self.assertEqual((settings['xc'],settings['basis'],settings['grid_level']),('PBE','def2-svp',1))
        self.assertFalse(settings['density_fitting'])
        self.assertEqual(settings['dispersion'],'NONE_EXPLICIT_PBE_PILOT_NOT_D3')
        self.assertEqual(settings['optimizer_maxsteps'],35)


if __name__=='__main__':unittest.main()
