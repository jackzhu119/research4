"""Stdlib gates/static regression only; no NumPy import or array access."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/analyze_calycosin_pilot_modes_20261105.py'


class CalycosinAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('calycosin_numpy_gate_stdlib_tests',SOURCE)
        cls.a=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.a)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.result_path=self.path/'result.json'
        self.identity={'name':'Calycosin','input_document':{'atom_symbols':['O']*5+['C']*16+['H']*12}}
        self.result={'schema':self.a.RESULT_SCHEMA,'status':'FREQUENCIES_COMPUTED',
            'identity':'Calycosin_CID5280448_user_supplied_conformer_001',
            'atoms':33,'electrons':148,'basis_functions':354,'normal_mode_count':93,'expected_normal_mode_count':93,
            'optimizer_converged':True,'scf_converged':True,'geometry_optimized':True,
            'final_gradient_thresholds_passed':True,'stable_MF_state_unchanged':True,
            'complete_global_Hessian':True,'normal_modes_complete':True,'full_system_CPHF_returned_normally':True,
            'mode_geometry_and_Hessian_same_final_SCF':True,'IR_intensities_status':'NOT_COMPUTED',
            'Hessian_method':'ORIGINAL_STOCK_FULL_RKS_KERNEL_NO_INCREMENTAL_OR_STAGE_CACHE',
            'settings':{'xc':'PBE','basis':'def2-svp','grid_level':1,'density_fitting':False,
                'dispersion':'NONE_EXPLICIT_PBE_PILOT_NOT_D3','threads':2,'optimizer_maxsteps':35},
            'source_worker':{'path':str(self.a.WORKER),'sha256':self.a.WORKER_SHA},
            'resource_supervision':{},'output_directory':str(self.path),'chemical_identity':self.identity,
            'root_post_spinosin_decision':{'authorization':{'path':str(self.path/'root_decision.json'),'sha256':'a'*64}},
            'arrays':{name:{'path':str(self.path/name),'sha256':'b'*64,'bytes':1024} for name in (
                'hessian_hartree_per_bohr2.npy','hessian_used_for_analysis.npy','normal_modes.npy','frequencies_cm1.npy',
                'optimized_coordinates_angstrom.npy','normal_mode_geometry_bohr.npy','masses_amu.npy','gradient_hartree_per_bohr.npy')}}
        self.write()
        self.guard=SimpleNamespace(verify_closed_phase=Mock(side_effect=lambda *args,**kwargs:{'producer_result_sha256':self.a.sha(self.result_path)}))
        self.worker=SimpleNamespace(read_json=lambda path,expected:json.loads(Path(path).read_text()),
            identity_metadata=lambda:self.identity,authorize_after_spinosin=Mock(return_value={}))

    def write(self):self.result_path.write_text(json.dumps(self.result,sort_keys=True)+'\n')

    def qualify(self):
        self.write()
        with patch.object(self.a,'frozen',side_effect=lambda path,*args:self.worker if path==self.a.WORKER else self.guard):
            return self.a.qualify(self.result_path,self.a.sha(self.result_path))

    def test_qualify_only_checks_metadata_and_returns_expected93_shapes(self):
        _,_,_,_,_,shapes=self.qualify()
        self.assertEqual(shapes['normal_modes.npy'],(93,33,3))
        self.assertEqual(shapes['hessian_hartree_per_bohr2.npy'],(33,33,3,3))
        self.guard.verify_closed_phase.assert_called_once()
        self.worker.authorize_after_spinosin.assert_called_once()
        self.assertFalse((self.path/'normal_modes.npy').exists())
        self.assertNotIn('numpy',sys.modules)

    def test_parent_closure_failure_blocks_before_any_array_access(self):
        self.guard.verify_closed_phase.side_effect=ValueError('unhealthy source parent')
        with self.assertRaisesRegex(ValueError,'unhealthy source parent'):self.qualify()
        self.worker.authorize_after_spinosin.assert_not_called()

    def test_actual_producer_result_SHA_mismatch_blocks(self):
        self.guard.verify_closed_phase.side_effect=None
        self.guard.verify_closed_phase.return_value={'producer_result_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'producer result SHA differs'):self.qualify()

    def test_incomplete_or_wrong_molecule_is_not_software_control(self):
        for changes in ({'status':'PARTIAL'},{'atoms':75},{'normal_mode_count':219},
                        {'identity':'water_software_control'},{'geometry_optimized':False},
                        {'stable_MF_state_unchanged':False},{'IR_intensities_status':'FABRICATED'}):
            original=deepcopy(self.result);self.result.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.a.result_header(self.result)
            self.result=original

    def test_root_post_main_authorization_is_revalidated(self):
        self.worker.authorize_after_spinosin.side_effect=ValueError('wrong main lineage')
        with self.assertRaisesRegex(ValueError,'wrong main lineage'):self.qualify()

    def test_array_metadata_cannot_point_outside_producer_directory(self):
        self.result['arrays']['normal_modes.npy']['path']=str(self.path.parent/'external.npy')
        with self.assertRaisesRegex(ValueError,'artifact metadata'):self.qualify()

    def test_changed_identity_overlay_is_rejected_before_numeric(self):
        self.result['chemical_identity']={'name':'unknown'}
        with self.assertRaisesRegex(ValueError,'identity overlay changed'):self.qualify()

    def test_negative_modes_do_not_imply_incomplete_computation(self):
        self.result.update(imaginary_frequency_count=1,all_frequencies_real_positive=False,
            stationary_minimum_verified=False,frequency_validation_passed=False)
        self.a.result_header(self.result)

    def test_top_level_stdlib_and_only_generic_math_helper_entry(self):
        tree=ast.parse(SOURCE.read_text());text=SOURCE.read_text()
        for n in tree.body:
            if isinstance(n,ast.Import):names=[v.name.split('.')[0] for v in n.names]
            elif isinstance(n,ast.ImportFrom):names=[(n.module or '').split('.')[0]]
            else:continue
            self.assertFalse({'numpy','pyscf','scipy','torch','rdkit'}.intersection(names))
        self.assertIn('math_helper.numpy_mode_audit(',text)
        self.assertNotIn('math_helper.analyze(',text)
        self.assertNotIn('case_for_symbols(',text)
        self.assertNotIn('import pyscf',text)
        self.assertNotIn('numpy',sys.modules)

    def test_actual_analysis_first_qualifies_before_numpy(self):
        from contextlib import nullcontext
        with patch.object(self.a,'exclusive_closed_quantum',return_value=nullcontext()), \
             patch.object(self.a,'qualify',side_effect=ValueError('gate failed before numpy')):
            with self.assertRaisesRegex(ValueError,'gate failed before numpy'):
                self.a.analyze(self.result_path,'0'*64,self.path/'analysis')
        self.assertNotIn('numpy',sys.modules)


if __name__=='__main__':unittest.main()
