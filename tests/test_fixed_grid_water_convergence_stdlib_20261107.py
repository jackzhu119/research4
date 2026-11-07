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
WORKER = ROOT/"scripts/research4_fixed_grid_water_convergence_20261107.py"
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


class WaterConvergenceTests(unittest.TestCase):
    def args(self,**changes):
        args = argparse.Namespace(control_suite=True,control_validation=True,threads=2,
            source_result=Path("x"),source_result_sha256="a"*64,
            main_pipeline_state=Path("y"),main_pipeline_state_sha256="b"*64,
            source_qualification=Path("z"),source_qualification_sha256="c"*64,
            allowed_phase_schedule=None,allowed_phase_schedule_sha256=None)
        for key,value in changes.items():
            setattr(args,key,value)
        return args

    def test_exact_water_only_cli_contract(self):
        M.validate_actual_arguments(self.args())
        for changes in ({"control_suite":False},{"control_validation":False},{"threads":1},{"source_qualification":None}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                M.validate_actual_arguments(self.args(**changes))
        M.source_identity({"case":"water_software_control","atoms":3},True)
        for control in (False,True):
            with self.assertRaises(ValueError):
                M.source_identity({"case":"Spinosin","atoms":75},control)
        source = WORKER.read_text()
        self.assertNotIn('mode.add_argument("--execute"',source)
        self.assertNotIn('p.add_argument("--mode-index"',source)
        self.assertNotIn('p.add_argument("--step-bohr"',source)
        self.assertNotIn('def verify_control_release(',source)

    def test_31_unique_points_exact_five_steps_and_fine_filename_precision(self):
        self.assertEqual(M.CONTROL_COORDINATES,(0,4,8))
        self.assertEqual(M.CONTROL_STEPS,(.00125,.000625,.0003125,.00015625,.000078125))
        names = [M.point_name(c,h,s) for c in M.CONTROL_COORDINATES for h in M.CONTROL_STEPS for s in (-1,1)]
        self.assertEqual(len(names)+1,M.EXPECTED_POINT_COUNT)
        self.assertEqual(len(set(names)),30)
        self.assertIn("c0-h0.00015625-minus",names)
        self.assertIn("c0-h0.00007813-plus",names)

    def test_fresh_zero_requires_real_kernel_and_same_tight_settings_as_every_point(self):
        tree = ast.parse(WORKER.read_text())
        compute = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "compute_point")
        source = ast.get_source_segment(WORKER.read_text(),compute)
        self.assertIn('"SCF_performed":True',source)
        self.assertIn('mf.kernel(dm0=dm0)',source)
        self.assertIn('mf.check_convergence = check_convergence',source)
        self.assertIn('strict_scf_convergence(env)',source)
        self.assertIn('state["strict_SCF_convergence_checks"][-1]["accepted"]',source)
        self.assertNotIn('mf.e_tot,mf.converged = original.e_tot,True',source)
        self.assertNotIn('if zero:',source)
        self.assertEqual(M.TIGHT_SCF_CONV_TOL,1e-13)
        self.assertEqual(M.TIGHT_SCF_CONV_TOL_GRAD,1e-9)

    def test_strict_scf_rejects_relaxed_extra_cycle_or_acceptance(self):
        self.assertTrue(M.strict_scf_convergence({"e_tot":0.,"last_hf_e":5e-14,"norm_gorb":8e-10})["accepted"])
        for energy_drift,gradient in ((5e-13,1e-11),(1e-15,2e-9),(float("inf"),1e-11),(1e-15,float("nan"))):
            with self.subTest(energy_drift=energy_drift,gradient=gradient):
                self.assertFalse(M.strict_scf_convergence({"e_tot":0.,"last_hf_e":energy_drift,"norm_gorb":gradient})["accepted"])

    def test_energy_curvature_uses_fresh_zero_and_accurate_signed_sum(self):
        h,k,linear = .125,.55,.012
        fresh_e0,archived_e0 = -75.,-74.9
        plus = fresh_e0+linear*h+.5*k*h*h
        minus = fresh_e0-linear*h+.5*k*h*h
        self.assertAlmostEqual(M.central_energy_curvature(plus,minus,fresh_e0,h),k,places=10)
        self.assertGreater(abs(M.central_energy_curvature(plus,minus,archived_e0,h)-k),1.)
        self.assertEqual(M.central_energy_curvature(1e16,1.,5e15,1.),1.)

    def comparisons_fixture(self):
        comparisons = []
        halvings = []
        for c in M.CONTROL_COORDINATES:
            for h in M.CONTROL_STEPS:
                fine = h in M.CONTROL_STEPS[-2:]
                comparisons.append({"coordinate_index_0based":c,"step_bohr":h,
                    "pair_completed":True,"fresh_zero_completed":True,
                    "FD_Hq_max_abs_error_Eh_per_Bohr2":1e-6 if fine else .2,
                    "central_energy_curvature_abs_error_Eh_per_Bohr2":1e-5 if fine else .3})
            for coarse,fine in zip(M.CONTROL_STEPS,M.CONTROL_STEPS[1:]):
                halvings.append({"coordinate_index_0based":c,"coarser_step_bohr":coarse,"finer_step_bohr":fine,
                    "pair_completed":True,"max_abs_difference_Eh_per_Bohr2":1e-5 if fine == M.CONTROL_STEPS[-1] else .4})
        return comparisons,halvings

    def test_finest_gate_retains_coarse_failures_without_using_them_for_acceptance(self):
        comparisons,halvings = self.comparisons_fixture()
        before = copy.deepcopy((comparisons,halvings))
        self.assertTrue(M.finest_pair_gate(comparisons,halvings,measurements_complete=True))
        self.assertEqual((comparisons,halvings),before)
        self.assertEqual(len(comparisons),15)
        self.assertEqual(len(halvings),12)
        self.assertGreater(comparisons[0]["FD_Hq_max_abs_error_Eh_per_Bohr2"],M.CONTROL_THRESHOLDS["FD_Hq_max_abs_error_Eh_per_Bohr2"])
        self.assertFalse(M.finest_pair_gate(comparisons,halvings,measurements_complete=False))

    def test_each_finest_full_vector_energy_and_halving_error_can_reject(self):
        for key in ("FD_Hq_max_abs_error_Eh_per_Bohr2","central_energy_curvature_abs_error_Eh_per_Bohr2"):
            comparisons,halvings = self.comparisons_fixture()
            comparisons[-1][key] = .001
            self.assertFalse(M.finest_pair_gate(comparisons,halvings,measurements_complete=True))
        comparisons,halvings = self.comparisons_fixture()
        halvings[-1]["max_abs_difference_Eh_per_Bohr2"] = .00006
        self.assertFalse(M.finest_pair_gate(comparisons,halvings,measurements_complete=True))
        comparisons,halvings = self.comparisons_fixture()
        comparisons[-1]["central_energy_curvature_abs_error_Eh_per_Bohr2"] = float("nan")
        self.assertFalse(M.finest_pair_gate(comparisons,halvings,measurements_complete=True))
        comparisons,halvings = self.comparisons_fixture()
        comparisons[-1] = comparisons[-2]
        self.assertFalse(M.finest_pair_gate(comparisons,halvings,measurements_complete=True))

    def test_same_spinosin_step_energy_gate_independent_and_all_three_coordinates(self):
        comparisons,halvings = self.comparisons_fixture()
        self.assertTrue(M.finest_pair_gate(comparisons,halvings,measurements_complete=True))
        self.assertFalse(M.same_spinosin_step_energy_gate(comparisons,measurements_complete=True))
        for row in comparisons:
            if row["step_bohr"] == .0003125:
                row["central_energy_curvature_abs_error_Eh_per_Bohr2"] = 1e-4
        self.assertTrue(M.same_spinosin_step_energy_gate(comparisons,measurements_complete=True))
        self.assertFalse(M.same_spinosin_step_energy_gate(comparisons,measurements_complete=False))
        row = next(row for row in comparisons if row["step_bohr"] == .0003125)
        row["central_energy_curvature_abs_error_Eh_per_Bohr2"] = .000101
        self.assertFalse(M.same_spinosin_step_energy_gate(comparisons,measurements_complete=True))

    def test_phase_cap_and_earlier_wall_deadline_fail_before_source_or_numerical_work(self):
        for cap,start,wall in ((501,"2026-11-07T10:00:00+00:00",100),(500,"2026-11-09T08:25:00+00:00",100)):
            guard = SimpleNamespace(verify_resource_supervision=lambda **kwargs:{"proof":{
                "cpu_hard_limit_seconds":cap,"started_at_utc":start,"wall_limit_seconds":wall}})
            with self.subTest(cap=cap,start=start),mock.patch.object(M,"load",return_value=guard),\
                    mock.patch.object(M,"sole_phase",side_effect=AssertionError("Must reject before source/point work")):
                with self.assertRaises(ValueError):
                    M.execute(self.args())

    def test_original_workers_unchanged_and_frozen_functions_equivalent(self):
        self.assertEqual(M.sha(ROOT/"scripts/research4_fixed_quadrature_worker_20261107.py"),
            "cab1b47de0e1b2868acd5a1ac6ee8141a0e57c7f79f6a2755ca0139032157e2a")
        self.assertEqual(M.sha(ROOT/"scripts/research4_cloud_followup_worker_20261106_v2.py"),
            "d1755eb103eb5c6bd4100771500fb969e19d0a05c69fe88257ef63668025b1fb")
        old = ast.parse((ROOT/"scripts/research4_fixed_quadrature_worker_20261107.py").read_text())
        new = ast.parse(WORKER.read_text())
        names = ("read_schedule","validate_appended_rows","canonical_proof_sha","closed_main",
                 "frozen_grid_type","attach_fixed_quadrature","verify_fixed_quadrature","original_mf_fingerprints")
        for name in names:
            a = next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name == name)
            b = next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)

    def test_no_top_level_numerical_import_or_old_zero_gradient_gate(self):
        tree = ast.parse(WORKER.read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names = [item.name for item in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(name.startswith(("numpy","scipy","pyscf")) for name in names))
        self.assertEqual(M.CONTROL_THRESHOLDS,{"FD_Hq_max_abs_error_Eh_per_Bohr2":1e-4,
            "h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2":5e-5,"central_energy_curvature_abs_error_Eh_per_Bohr2":1e-4})


if __name__ == "__main__":
    unittest.main()
