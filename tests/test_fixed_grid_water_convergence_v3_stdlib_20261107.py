"""No-Q method/provenance tests; fake grids exercise stock setter hazards."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT/"scripts/research4_fixed_grid_water_convergence_20261107_v3.py"
spec = importlib.util.spec_from_file_location("fixed_grid_water_convergence_worker_under_test",WORKER)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class FakeArray:
    def __init__(self,value):
        self.data = copy.deepcopy(value.data if isinstance(value,FakeArray) else value)
        self.flags = SimpleNamespace(writeable=True)


class FakeNP:
    @staticmethod
    def array(value,**kwargs):
        return FakeArray(value)
    @staticmethod
    def array_equal(a,b):
        return a.data == b.data
    @staticmethod
    def shares_memory(a,b):
        return a is b


class FakeGrids:
    level,alignment,cutoff = 1,8,1e-15
    radi_cutoff,radi_method,becke_scheme,prune,radii_adjust,atomic_radii = None,None,None,None,None,None
    RESET_KEYS = {"level","atom_grid","atomic_radii","radii_adjust","radi_method","becke_scheme","prune"}
    def __init__(self,mol):
        self.resets = 0
        self.mol,self.stdout,self.verbose,self.symmetry = mol,mol.stdout,0,False
        self.atom_grid = {}
        self.reset()
    def __setattr__(self,name,value):
        if name in self.RESET_KEYS:
            self.reset()
        object.__setattr__(self,name,value)
    def reset(self,*args,**kwargs):
        self.resets += 1
        for name in M.FIXED_ARRAY_NAMES:
            setattr(self,name,None)
        return self
    def build(self,*args,**kwargs):
        raise AssertionError("Building a fake quadrature is never permitted in tests")
    kernel=gen_atomic_grids=get_partition=gen_partition=prune_by_density_=_select_grids=_add_padding=build


ATTRS = ("level","atom_grid","alignment","cutoff","radi_cutoff","radi_method","becke_scheme","prune","radii_adjust","atomic_radii")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value.data,sort_keys=True).encode()).hexdigest()


def grid_settings(value):
    return {key:copy.deepcopy(getattr(value,key)) for key in ATTRS}


class FixedGridTests(unittest.TestCase):
    def fixture(self):
        source_mol = SimpleNamespace(stdout=None,tag="source")
        moved_mol = SimpleNamespace(stdout=None,tag="moved")
        grids = FakeGrids(source_mol)
        for name,value in (("coords",[[1.,2.,3.],[.0001,.0001,.0001]]),("weights",[2.,0.]),
                           ("non0tab",[[7]]),("screen_index",[[7]]),("atm_idx",[0,-1]),("quadrature_weights",[3.,0.])):
            setattr(grids,name,FakeArray(value))
        original = SimpleNamespace(mol=source_mol,grids=grids,nlc="",disp=None,
            _numint=SimpleNamespace(cutoff=1e-13,omega=None))
        mf = SimpleNamespace(mol=moved_mol,small_rho_cutoff=1e-7,
            _numint=SimpleNamespace(cutoff=1e-4,omega=8),do_nlc=lambda:False)
        calls = []
        def make_mask(mol,coords,cutoff):
            calls.append((mol,coords,cutoff))
            return FakeArray([[7 if mol is source_mol else 9]])
        gen_grid = SimpleNamespace(Grids=FakeGrids,make_mask=make_mask)
        w = SimpleNamespace(GRID_ATTRIBUTES=ATTRS,grid_settings=grid_settings)
        base = SimpleNamespace(array_fingerprint=fingerprint)
        return original,mf,w,base,gen_grid,calls

    def test_exact_copied_grid_and_fresh_both_masks_preserve_native_cutoffs(self):
        original,mf,w,base,gen,calls = self.fixture()
        before = M.grid_payload_fingerprints(original.grids,base)
        proof = M.attach_fixed_quadrature(mf,original,w,base,FakeNP,gen)
        self.assertIs(mf.grids.mol,mf.mol)
        self.assertIsNot(mf.grids,original.grids)
        self.assertIsNot(mf.grids.coords,original.grids.coords)
        self.assertEqual(mf.grids.coords.data,original.grids.coords.data)
        self.assertEqual(mf.grids.weights.data,[2.,0.])
        self.assertIs(calls[0][0],original.mol)
        self.assertIs(calls[1][0],mf.mol)
        self.assertEqual([item[2] for item in calls],[1e-15,1e-15])
        self.assertIs(mf.grids.non0tab,mf.grids.screen_index)
        self.assertEqual(mf.grids.non0tab.data,[[9]])
        self.assertEqual(original.grids.non0tab.data,[[7]])
        self.assertEqual(mf.small_rho_cutoff,0)
        self.assertEqual(mf._numint.cutoff,1e-13)
        self.assertIsNone(mf._numint.omega)
        self.assertGreater(mf.grids.resets,1)
        self.assertEqual(M.grid_payload_fingerprints(original.grids,base),before)
        M.verify_fixed_quadrature(mf,proof,w,base)
        self.assertTrue(proof["fingerprints_unchanged_after_SCF_and_gradient"])

    def test_build_reset_partition_pruning_padding_and_option_mutation_raise(self):
        original,mf,w,base,gen,calls = self.fixture()
        M.attach_fixed_quadrature(mf,original,w,base,FakeNP,gen)
        for name in ("build","kernel","reset","gen_atomic_grids","get_partition","gen_partition",
                     "prune_by_density_","_select_grids","_add_padding"):
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                getattr(mf.grids,name)()
        for name in (*ATTRS,*M.FIXED_ARRAY_NAMES,"mol","symmetry","_research4_fixed_attached"):
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                setattr(mf.grids,name,None)
        self.assertTrue(all(not getattr(mf.grids,name).flags.writeable for name in M.FIXED_ARRAY_NAMES))

    def test_changed_saved_zero_mask_rejected(self):
        original,mf,w,base,gen,calls = self.fixture()
        original.grids.screen_index = FakeArray([[4]])
        with self.assertRaisesRegex(ValueError,"source-geometry screening mask differs"):
            M.attach_fixed_quadrature(mf,original,w,base,FakeNP,gen)

    def test_fixed_grid_fingerprint_change_detected(self):
        original,mf,w,base,gen,calls = self.fixture()
        proof = M.attach_fixed_quadrature(mf,original,w,base,FakeNP,gen)
        mf.grids.coords.data[0][0] += 1
        with self.assertRaisesRegex(ValueError,"quadrature.*changed"):
            M.verify_fixed_quadrature(mf,proof,w,base)



def nested_map(fn,a,b=None):
    if isinstance(a,list):
        return [nested_map(fn,x,b[i] if b is not None else None) for i,x in enumerate(a)]
    return fn(a,b) if b is not None else fn(a)


def flattened(value):
    if isinstance(value,list):
        return [item for row in value for item in flattened(row)]
    return [value]


FakeArray.shape = property(lambda self:(len(self.data),len(self.data[0])) if self.data and isinstance(self.data[0],list) else (len(self.data),))
FakeArray.__sub__ = lambda self,other:FakeArray(nested_map(lambda a,b:a-b,self.data,other.data))
FakeArray.__matmul__ = lambda self,other:FakeArray([[sum(a*b for a,b in zip(row,col)) for col in zip(*other.data)] for row in self.data])


class PolishNP(FakeNP):
    linalg = SimpleNamespace(norm=lambda value:sum(x*x for x in flattened(value.data))**.5)
    @staticmethod
    def asarray(value):
        return value if isinstance(value,FakeArray) else FakeArray(value)
    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda:all(__import__('math').isfinite(x) for x in flattened(value.data)))
    @staticmethod
    def einsum(code,a,b):
        if code != 'pi,pi->i':
            raise AssertionError(code)
        return FakeArray([sum(x*y for x,y in zip(ca,cb)) for ca,cb in zip(zip(*a.data),zip(*b.data))])


class FakePolishMF:
    def __init__(self,scenarios):
        self.scenarios,self.calls,self.veff_calls,self.fock_calls = scenarios,[],[],[]
        self.diis,self.direct_scf,self.direct_scf_tol,self.small_rho_cutoff = True,True,1e-13,0
        self.mol = SimpleNamespace(nao_nr=lambda:1)
        self.grids = SimpleNamespace(mol=self.mol,cutoff=1e-15)
        self._numint = SimpleNamespace(cutoff=1e-13)
        self.mo_coeff,self.mo_energy,self.mo_occ = FakeArray([[1.]]),FakeArray([1.]),FakeArray([2.])
    def get_hcore(self,mol):
        return FakeArray([[1.]])
    def get_ovlp(self,mol):
        return FakeArray([[1.]])
    def kernel(self,dm0):
        self.calls.append(copy.deepcopy(dm0.data))
        self.scenario = self.scenarios[len(self.calls)-1]
        self.previous = FakeArray(self.scenario.get('previous',[[len(self.calls)-.1]]))
        self.current = FakeArray(self.scenario.get('current',[[float(len(self.calls))]]))
        self.e_tot = 1.
        common = {'e_tot':1.,'dm_last':self.previous,'dm':self.current,'cycle':0}
        normal = {**common,'last_hf_e':1.-1e-14,'norm_gorb':5e-11,
                  'conv_tol':self.conv_tol,'conv_tol_grad':self.conv_tol_grad}
        self.check_convergence(normal)
        final = {**common,'last_hf_e':1.-self.scenario.get('final_dE',1e-14),
                 'norm_gorb':self.scenario.get('final_g',5e-11),
                 'conv_tol':self.conv_tol*10,'conv_tol_grad':self.conv_tol_grad*3}
        self.converged = self.check_convergence(final)
        self.callback(final)
        Path(self.chkfile).write_text('Synthetic no-Q checkpoint fixture')
        return self.e_tot
    def make_rdm1(self):
        return FakeArray(self.current)
    def get_veff(self,mol,dm,*,dm_last,vhf_last):
        self.veff_calls.append({'dm':copy.deepcopy(dm.data),'dm_last':dm_last,'vhf_last':vhf_last})
        return FakeArray(dm)
    def energy_tot(self,dm,hcore,vhf):
        if dm.data == self.current.data:
            return self.scenario.get('full_current_E',1.)
        return self.scenario.get('full_previous_E',1.-1e-14)
    def get_fock(self,**kwargs):
        self.fock_calls.append(kwargs)
        return FakeArray([[2.]])
    def get_grad(self,coeff,occ,fock):
        return FakeArray([self.scenario.get('full_g',5e-11)])


def fake_base():
    def atomic_npy(path,value):
        Path(path).write_text(json.dumps(value.data))
    return SimpleNamespace(array_fingerprint=fingerprint,atomic_npy=atomic_npy)


class BoundedPolishTests(unittest.TestCase):
    def run_helper(self,mf,out,state):
        return M.bounded_polished_scf(mf,FakeArray([[0.]]),np=PolishNP,base=fake_base(),out=out,
            emit=lambda *args:None,state=state,final_energy_tolerance=1e-13,
            final_orbital_gradient_tolerance=1e-9,main_orbital_gradient_trigger=1e-10)

    def test_normal_accept_extra_fail_continues_current_density_then_first_pass(self):
        mf = FakePolishMF([{'final_dE':2e-13,'full_g':2e-9},{},{}])
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            result = self.run_helper(mf,Path(tmp),state)
            self.assertTrue(result['qualified'])
            self.assertEqual(len(mf.calls),2)
            self.assertEqual(mf.calls,[[[0.]],[[1.]]])
            self.assertTrue(state['strict_SCF_convergence_checks'][0]['accepted'])
            self.assertFalse(state['strict_SCF_convergence_checks'][1]['accepted'])
            self.assertFalse(state['SCF_polishing_attempts'][0]['qualified'])
            self.assertTrue(state['SCF_polishing_attempts'][1]['qualified'])
            self.assertEqual(state['qualified_SCF_kernel_call_1based'],2)
            self.assertEqual(mf.e_tot,state['final_SCF_endpoint_audit']['current_full_energy_hartree'])
            self.assertEqual(mf.mo_energy.data,[2.])
            self.assertEqual(set(state['qualified_MF_array_files']),{'MO_coeff','MO_energy','MO_occ','density','full_Fock'})

    def test_all_three_native_failed_endpoints_exhaust_without_acceptance(self):
        mf = FakePolishMF([{'final_dE':2e-13,'full_g':2e-9} for _ in range(3)])
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            with self.assertRaisesRegex(ValueError,'bounded three-call'):
                self.run_helper(mf,Path(tmp),state)
            self.assertEqual(len(mf.calls),3)
            self.assertEqual(len(state['SCF_polishing_attempts']),3)
            self.assertTrue(all(not row['qualified'] for row in state['SCF_polishing_attempts']))
            self.assertNotIn('qualified_SCF_kernel_call_1based',state)
            self.assertTrue(all(len(row['arrays']) == 6 for row in state['SCF_polishing_attempts']))

    def test_full_audit_separate_consecutive_density_veff_no_diis_no_shift(self):
        mf = FakePolishMF([{}])
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            self.run_helper(mf,Path(tmp),state)
            self.assertEqual(len(mf.veff_calls),2)
            self.assertNotEqual(mf.veff_calls[0]['dm'],mf.veff_calls[1]['dm'])
            self.assertTrue(all(call['dm_last'] is None and call['vhf_last'] is None for call in mf.veff_calls))
            kwargs = mf.fock_calls[0]
            self.assertEqual(kwargs['cycle'],-1)
            self.assertIsNone(kwargs['diis'])
            self.assertEqual(kwargs['level_shift_factor'],0)
            self.assertEqual(kwargs['damp_factor'],0)
            self.assertEqual(mf.max_cycle,100)

    def test_full_endpoint_audit_failure_continues_even_native_kernel_passed(self):
        mf = FakePolishMF([{'full_g':2e-9},{}])
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            self.run_helper(mf,Path(tmp),state)
            self.assertTrue(state['SCF_polishing_attempts'][0]['native_kernel_converged'])
            self.assertFalse(state['SCF_polishing_attempts'][0]['qualified'])
            self.assertEqual(len(mf.calls),2)

    def test_actual_consecutive_states_can_have_equal_values_without_fake_self_probe(self):
        mf = FakePolishMF([{'previous':[[1.]],'current':[[1.]]}])
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            self.assertTrue(self.run_helper(mf,Path(tmp),state)['qualified'])
            self.assertEqual(len(mf.veff_calls),2)
            self.assertEqual(state['final_SCF_endpoint_audit']['consecutive_density_Frobenius_difference'],0.)
            self.assertEqual(state['strict_SCF_convergence_checks'][-1]['phase'],'FINAL_EXTRA_CYCLE')

    def native_checks(self):
        main = {'phase':'SCF_MAIN_CYCLE','accepted':True,'kernel_call_1based':1,
            'energy_delta_abs_hartree':1e-14,'orbital_gradient_norm':5e-11,
            'energy_tolerance_hartree':1e-13,'orbital_gradient_tolerance':1e-10}
        final = {'phase':'FINAL_EXTRA_CYCLE','accepted':False,'kernel_call_1based':1,
            'energy_delta_abs_hartree':2e-13,'orbital_gradient_norm':5e-11,
            'energy_tolerance_hartree':1e-13,'orbital_gradient_tolerance':1e-9}
        return main,final

    def test_full_native_authority_certifies_while_preserving_false_native_flag(self):
        main,final = self.native_checks()
        options = dict(preceding_main_check=main,native_checks_finite=True,density_change_norm=.1,
            energy_tolerance=1e-13,orbital_gradient_tolerance=1e-9)
        result = M.audited_endpoint_qualification(False,final,1.,1.,1e-11,**options)
        self.assertTrue(result['qualified'])
        self.assertFalse(result['native_kernel_converged'])
        self.assertFalse(result['last_native_check']['accepted'])
        self.assertEqual(result['certification_policy'],'FULL_NATIVE_VEFF_ENDPOINT_CERTIFICATION_NOT_NATIVE_KERNEL_FLAG')
        mf = FakePolishMF([{'final_dE':2e-13}])
        with tempfile.TemporaryDirectory() as tmp:
            state = {}
            self.run_helper(mf,Path(tmp),state)
            self.assertTrue(mf.converged)
            self.assertFalse(state['native_kernel_converged_at_certified_endpoint'])
            self.assertFalse(state['SCF_polishing_attempts'][0]['native_kernel_converged'])
            self.assertFalse(state['strict_SCF_convergence_checks'][-1]['accepted'])
            self.assertEqual(state['SCF_convergence_scope'],M.SCF_CERTIFICATION_POLICY)
            self.assertEqual(len(mf.calls),1)

    def test_main_only_nonfinite_or_wrong_call_cannot_certify(self):
        for mutation in ('main_phase','nan_final','nan_main','wrong_call','main_failed','finite_flag_false'):
            main,final = self.native_checks()
            options = dict(preceding_main_check=main,native_checks_finite=True,density_change_norm=.1,
                energy_tolerance=1e-13,orbital_gradient_tolerance=1e-9)
            if mutation == 'main_phase': final['phase']='SCF_MAIN_CYCLE'
            if mutation == 'nan_final': final['energy_delta_abs_hartree']=float('nan')
            if mutation == 'nan_main': main['orbital_gradient_norm']=float('nan')
            if mutation == 'wrong_call': main['kernel_call_1based']=2
            if mutation == 'main_failed': main['accepted']=False
            if mutation == 'finite_flag_false': options['native_checks_finite']=False
            with self.subTest(mutation=mutation):
                self.assertFalse(M.audited_endpoint_qualification(False,final,1.,1.,1e-11,**options)['qualified'])

    def test_native_true_never_waives_full_physical_errors(self):
        main,final = self.native_checks()
        final['accepted']=True;final['energy_delta_abs_hartree']=1e-14
        options = dict(preceding_main_check=main,native_checks_finite=True,density_change_norm=.1,
            energy_tolerance=1e-13,orbital_gradient_tolerance=1e-9)
        for previous,current,gradient in ((1.,1.,2e-9),(1.,1.+2e-13,1e-11)):
            self.assertFalse(M.audited_endpoint_qualification(True,final,previous,current,gradient,**options)['qualified'])

    def test_extra_cycle_original_precision_and_tighter_main_trigger(self):
        opts = dict(main_energy_tolerance=1e-13,main_orbital_gradient_trigger=1e-10,
            final_energy_tolerance=1e-13,final_orbital_gradient_tolerance=1e-9)
        base_env = {'e_tot':0.,'last_hf_e':1e-14,'norm_gorb':5e-10}
        normal = M.polish_convergence_check({**base_env,'conv_tol':1e-13,'conv_tol_grad':1e-10},**opts)
        final = M.polish_convergence_check({**base_env,'conv_tol':1e-13*10,'conv_tol_grad':1e-10*3},**opts)
        self.assertFalse(normal['accepted'])
        self.assertTrue(final['accepted'])
        self.assertEqual(final['energy_tolerance_hartree'],1e-13)
        self.assertEqual(final['orbital_gradient_tolerance'],1e-9)
        failed = M.polish_convergence_check({**base_env,'last_hf_e':2e-13,
            'conv_tol':1e-13*10,'conv_tol_grad':1e-10*3},**opts)
        self.assertFalse(failed['accepted'])

    def test_fixed_original_scientific_gate_failure_is_preserved(self):
        rows = [{'coordinate_index_0based':c,'step_bohr':h,'pair_completed':True,'fresh_zero_completed':True,
                 'FD_Hq_max_abs_error_Eh_per_Bohr2':1e-6,'central_energy_curvature_abs_error_Eh_per_Bohr2':1e-5}
                for c in M.CONTROL_COORDINATES for h in M.CONTROL_STEPS]
        half = [{'coordinate_index_0based':c,'coarser_step_bohr':M.CONTROL_STEPS[-2],
                 'finer_step_bohr':M.CONTROL_STEPS[-1],'pair_completed':True,'max_abs_difference_Eh_per_Bohr2':1e-6}
                for c in M.CONTROL_COORDINATES]
        self.assertTrue(M.finest_pair_gate(rows,half,measurements_complete=True))
        rows[-1]['central_energy_curvature_abs_error_Eh_per_Bohr2'] = .00011870118011625408
        self.assertFalse(M.finest_pair_gate(rows,half,measurements_complete=True))
        self.assertEqual(rows[-1]['central_energy_curvature_abs_error_Eh_per_Bohr2'],.00011870118011625408)

    def test_old_source_hashes_and_eight_core_functions_preserved(self):
        original = ROOT/'scripts/research4_fixed_grid_water_convergence_20261107.py'
        self.assertEqual(M.sha(ROOT/'scripts/research4_fixed_grid_water_convergence_20261107_v2.py'),
            '2b9de9ec029afffd5afc12a5214e714d99b9f74d050b1b020c7a2b8fc6aeb525')
        self.assertEqual(M.sha(original),'f9658ae6146b3f6ef340771d2916dc8b9d41debc0af88a8c096ee804fcef5fc3')
        self.assertEqual(M.sha(ROOT/'scripts/research4_fixed_quadrature_worker_20261107.py'),
            'cab1b47de0e1b2868acd5a1ac6ee8141a0e57c7f79f6a2755ca0139032157e2a')
        old,new = ast.parse(original.read_text()),ast.parse(WORKER.read_text())
        for name in ('read_schedule','validate_appended_rows','canonical_proof_sha','closed_main',
                     'frozen_grid_type','attach_fixed_quadrature','verify_fixed_quadrature','original_mf_fingerprints',
                     'finest_pair_gate','same_spinosin_step_energy_gate'):
            a = next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name == name)
            b = next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
        self.assertEqual(M.MAX_SCF_KERNEL_CALLS,3)
        self.assertEqual(M.SCF_MAX_CYCLES_PER_CALL,100)
        self.assertEqual(M.CONTROL_THRESHOLDS,{'FD_Hq_max_abs_error_Eh_per_Bohr2':1e-4,
            'h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2':5e-5,'central_energy_curvature_abs_error_Eh_per_Bohr2':1e-4})


if __name__ == '__main__':
    unittest.main()
