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
WORKER = ROOT/"scripts/research4_fixed_quadrature_worker_20261107.py"
spec = importlib.util.spec_from_file_location("fixed_quadrature_worker_under_test",WORKER)
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


class MetadataGates(unittest.TestCase):
    def args(self,**changes):
        args = argparse.Namespace(source_result=Path("x"),source_result_sha256="a"*64,
            main_pipeline_state=Path("y"),main_pipeline_state_sha256="b"*64,
            source_qualification=Path("z"),source_qualification_sha256="c"*64,
            allowed_phase_schedule=None,allowed_phase_schedule_sha256=None,threads=2,
            control_suite=False,control_validation=False,mode_index=1,coordinate_index=None,
            step_bohr=.005,sign=1,mode_array_sha256="d"*64,control_result=Path("control"),control_result_sha256="e"*64)
        for key,value in changes.items():
            setattr(args,key,value)
        return args

    def test_only_one_exact_spinosin_point(self):
        M.validate_actual_arguments(self.args())
        for changes in ({"mode_index":2},{"sign":-1},{"step_bohr":.0025},{"threads":1},
                        {"control_result":None},{"control_validation":True},{"coordinate_index":0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                M.validate_actual_arguments(self.args(**changes))

    def test_control_suite_no_custom_points_and_requires_control_validation(self):
        values = dict(control_suite=True,control_validation=True,mode_index=None,mode_array_sha256=None,
            step_bohr=None,sign=None,control_result=None,control_result_sha256=None)
        M.validate_actual_arguments(self.args(**values))
        values["control_validation"] = False
        with self.assertRaises(ValueError):
            M.validate_actual_arguments(self.args(**values))

    def test_frozen_v2_worker_unchanged(self):
        self.assertEqual(M.sha(ROOT/"scripts/research4_cloud_followup_worker_20261106_v2.py"),
            "d1755eb103eb5c6bd4100771500fb969e19d0a05c69fe88257ef63668025b1fb")
        self.assertEqual(M.GUARD_SHA,"7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65")

    def test_no_top_level_numerical_import_and_fixed_thresholds(self):
        tree = ast.parse(WORKER.read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names = [item.name for item in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(name.startswith(("numpy","scipy","pyscf")) for name in names))
        self.assertEqual(M.CONTROL_THRESHOLDS,{"zero_gradient_max_abs_Eh_per_Bohr":1e-8,
            "FD_Hq_max_abs_error_Eh_per_Bohr2":1e-4,"h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2":5e-5})
        self.assertEqual(M.SPINOSIN_GRADIENT_SHA,"9418727d662aaada4548cf27bbfdec365627126eb1836703096f737a5ad94ad4")

    def test_qualification_preserves_historical_boot_argument_via_original_analyzer(self):
        source = WORKER.read_text()
        self.assertIn('gate=analyzer.qualify(args.source_result.resolve(),args.source_result_sha256,control=args.control_validation)',source)
        self.assertNotIn('qualify_gradient_receipt(',source)
        self.assertIn('historical_boot_id":source["boot_id"]',source)

    def test_phase_caps_and_earlier_wall_deadline_fail_before_numerical_import(self):
        cases = ((False,7001,"2026-11-07T10:00:00+00:00",100),
                 (True,301,"2026-11-07T10:00:00+00:00",100),
                 (False,7000,"2026-11-09T08:25:00+00:00",100))
        for control,cap,start,wall in cases:
            guard = SimpleNamespace(verify_resource_supervision=lambda **kwargs:{"proof":{
                "cpu_hard_limit_seconds":cap,"started_at_utc":start,"wall_limit_seconds":wall}})
            with self.subTest(control=control,cap=cap,start=start),mock.patch.object(M,"load",return_value=guard),\
                    mock.patch.object(M,"sole_phase",side_effect=AssertionError("Must reject before any source/point work")):
                with self.assertRaises(ValueError):
                    M.execute(self.args(control_suite=control))

    def control_fixture(self,root):
        out = root/"reports/control";out.mkdir(parents=True)
        ledger = str(root/"ledger.json")
        result = {"schema":M.CONTROL_SCHEMA,"status":M.CONTROL_SUCCESS,"control_passed":True,
            "method":M.METHOD,"thresholds":M.CONTROL_THRESHOLDS,"case":"water_software_control",
            "source_worker":M.ref(WORKER),"snapshot_manifest_sha256":"s"*64,
            "resource_supervision":{"proof":{"cpu_hard_limit_seconds":300,"ledger_path":ledger}},
            "current_execution_boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "predetermined_coordinates_0based":list(M.CONTROL_COORDINATES),"predetermined_steps_bohr":list(M.CONTROL_STEPS),
            "predetermined_signs":[-1,1],"all_13_prescribed_points_completed":True,
            "zero_gradient_check_passed":True,"zero_gradient_max_abs_difference_Eh_per_Bohr":1e-13,
            "software_control_displaced_SCF_conv_tol_hartree":1e-11,"software_control_displaced_SCF_conv_tol_grad":1e-7,
            "comparisons":[{"pair_completed":True,"FD_Hq_absolute_error_check_passed":True,
                "FD_Hq_max_abs_error_Eh_per_Bohr2":1e-6} for _ in range(6)],
            "h_halving_comparisons":[{"check_passed":True,"max_abs_difference_Eh_per_Bohr2":1e-6} for _ in range(3)],"points":[]}
        cover = [(None,0.,0)]+[(c,h,s) for c in M.CONTROL_COORDINATES for h in M.CONTROL_STEPS for s in (-1,1)]
        for i,(c,h,s) in enumerate(cover):
            p = out/("point%d.json"%i)
            value = {"status":M.POINT_SUCCESS,"method":M.METHOD,"gradient_completed":True,
                "source_worker":M.ref(WORKER),"source_snapshot_manifest_sha256":"s"*64,
                "parent_output_directory":str(out),"SCF_performed":s != 0,
                "settings":{"actual_conv_tol_hartree":1e-11,"actual_conv_tol_grad":1e-7},
                "displacement":{"mode_index_1based":None,"coordinate_index_0based":c,"step_bohr":h,"sign":s}}
            M.save(p,value,exclusive=True);result["points"].append(M.ref(p))
        path = out/"result.json"
        M.save(path,result,exclusive=True)
        args = self.args(control_result=path,control_result_sha256=M.sha(path))
        guard = SimpleNamespace(verify_closed_phase=mock.Mock(return_value={"authentic_closed_producer":True}))
        supervision = {"proof":{"ledger_path":ledger}}
        return path,result,args,guard,supervision

    def test_release_uses_parser_control_result_names_and_authentic_guard(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            path,result,args,guard,supervision = self.control_fixture(Path(tmp))
            value = M.verify_control_release(args,guard,supervision)
            self.assertTrue(value["control_passed"])
            guard.verify_closed_phase.assert_called_once()

    def test_wrong_boot_failed_threshold_incomplete_or_duplicated_controls_rejected(self):
        mutations = (lambda r:r.update(current_execution_boot_id="old-boot"),
            lambda r:r.update(control_passed=False),lambda r:r.update(predetermined_coordinates_0based=[0,1,2]),
            lambda r:r["comparisons"][0].update(FD_Hq_max_abs_error_Eh_per_Bohr2=.1),
            lambda r:r["points"].__setitem__(0,r["points"][1]))
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                path,result,args,guard,supervision = self.control_fixture(Path(tmp))
                mutate(result);M.save(path,result);args.control_result_sha256=M.sha(path)
                with self.assertRaises(ValueError):
                    M.verify_control_release(args,guard,supervision)
                guard.verify_closed_phase.assert_not_called()


if __name__ == "__main__":
    unittest.main()
