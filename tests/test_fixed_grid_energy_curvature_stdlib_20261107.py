"""No-Q tests for authentic controls, strict SCF decisions and fresh energy math."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import math
import struct
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/"scripts/research4_fixed_grid_energy_curvature_20261107.py"
spec=importlib.util.spec_from_file_location("energy_curvature_worker_under_test",WORKER)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class EnergyMathTests(unittest.TestCase):
    def test_fresh_zero_energy_and_compensated_numerator_used(self):
        zero,plus,minus=-2210.,-2209.999999999,-2209.999999999
        value=M.central_energy_curvature(zero,plus,minus)
        self.assertEqual(value["central_energy_curvature_Eh_per_Bohr2"],math.fsum([plus,minus,-2*zero])/M.STEP_BOHR**2)
        self.assertTrue(value["fresh_zero_energy_used"])
        self.assertFalse(value["archived_source_energy_used_in_numerator"])
        self.assertFalse(value["Spinosin_step_convergence_verified"])
        self.assertEqual(value["diagnostic_status"],M.DIAGNOSTIC_STATUS)
        self.assertGreater(value["binary64_ulp_scale_Eh_per_Bohr2"],0)
        # Substituting an old source E0 materially changes the numerator.
        self.assertNotEqual(value["central_energy_curvature_Eh_per_Bohr2"],
            M.central_energy_curvature(zero+1e-6,plus,minus)["central_energy_curvature_Eh_per_Bohr2"])

    def test_missing_nonfinite_or_boolean_energy_never_produces_curvature(self):
        for bad in (None,True,float("nan"),float("inf")):
            for position in range(3):
                values=[0.,1.,2.];values[position]=bad
                with self.subTest(bad=bad,position=position),self.assertRaises(ValueError):
                    M.central_energy_curvature(*values)

    def test_both_strict_checks_required_including_unrelaxed_extra_cycle(self):
        self.assertTrue(M.strict_scf_decision({"e_tot":0.,"last_hf_e":5e-13,"norm_gorb":5e-10})["accepted"])
        for env in ({"e_tot":0.,"last_hf_e":2e-12,"norm_gorb":0.},
                    {"e_tot":0.,"last_hf_e":0.,"norm_gorb":2e-9},
                    {"e_tot":0.,"last_hf_e":0.},{"e_tot":True,"last_hf_e":0.,"norm_gorb":0.}):
            with self.subTest(env=env):
                self.assertFalse(M.strict_scf_decision(env)["accepted"])
        self.assertFalse(M.strict_scf_decision({"e_tot":0.,"last_hf_e":0.,"norm_gorb":0.})["extra_cycle_relaxation_allowed"])

    def test_partial_triplet_preserves_real_completed_energies_without_inventing_minus(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name,energy in (("fresh-zero",-2.),("plus",-1.9)):
                (out/name).mkdir();M.save(out/name/"native_result.json",{
                    "status":M.POINT_SUCCESS,"energy_completed":True,"energy_hartree":energy})
            points=M.completed_energy_points(out)
            self.assertEqual(set(points),{"fresh-zero","plus"})
            self.assertEqual(points["plus"]["energy_hartree"],-1.9)
            (out/"minus").mkdir();M.save(out/"minus/native_result.json",{
                "status":"PARTIAL_RESOURCE_STOP","energy_completed":False,
                "last_iterate_energy_hartree_NOT_COMPLETED":-1.9})
            self.assertIsNone(M.completed_energy_points(out)["minus"]["energy_hartree"])


class ProofAndSourceTests(unittest.TestCase):
    def args(self,**changes):
        value=argparse.Namespace(source_result=Path("source"),source_result_sha256="a"*64,
            main_pipeline_state=Path("main"),main_pipeline_state_sha256="b"*64,
            source_qualification=Path("seal"),source_qualification_sha256="c"*64,
            allowed_phase_schedule=None,allowed_phase_schedule_sha256=None,threads=2,
            mode_index=1,coordinate_index=None,step_bohr=M.STEP_BOHR,sign=1,
            control_suite=False,control_validation=False,mode_array_sha256="d"*64,
            control_result=Path("control"),control_result_sha256="e"*64,control_worker_sha256="f"*64)
        for key,item in changes.items():setattr(value,key,item)
        return value

    def test_exact_fixed_protocol_arguments(self):
        M.validate_actual_arguments(self.args())
        for changes in ({"mode_index":2},{"step_bohr":.005},{"sign":-1},{"threads":1},
                        {"control_worker_sha256":None},{"mode_array_sha256":None},{"control_validation":True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                M.validate_actual_arguments(self.args(**changes))

    def test_proof_requires_actual_started_time_stop_and_entire_stricter_deadline(self):
        now=datetime(2026,11,7,12,tzinfo=timezone.utc)
        good={"cpu_hard_limit_seconds":7500,"actual_cpu_stop_threshold_seconds":7380,
            "started_at_utc":"2026-11-07T11:59:59+00:00","wall_limit_seconds":8000}
        M.validate_execution_envelope(good,now=now)
        for changes in ({"cpu_hard_limit_seconds":7501},{"actual_cpu_stop_threshold_seconds":7381},
                        {"started_at_utc":"2026-11-07T12:00:01+00:00"},
                        {"started_at_utc":"2026-11-07T11:59:59"},
                        {"started_at_utc":"2026-11-09T08:25:00+00:00","wall_limit_seconds":100}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                M.validate_execution_envelope({**good,**changes},now=now)

    def test_old_schema_or_different_worker_source_qualification_rejected(self):
        args=self.args()
        good={"schema":M.QUALIFICATION_SCHEMA,"status":M.QUALIFICATION_SUCCESS,
            "qualification_performed_with_all_ledgers_closed":True,"new_quantum_evaluations":0,
            "source_worker":M.ref(ROOT/"scripts/research4_fixed_quadrature_worker_20261107.py")}
        for seal in ({**good,"schema":"research4-fixed-original-quadrature-source-qualification-v1"},good):
            with self.subTest(seal=seal),mock.patch.object(M,"checked",return_value=Path("seal")),\
                    mock.patch.object(M,"read",return_value=seal),self.assertRaises(ValueError):
                M.source_gate(args,None,None,None)

    def test_original_mf_energy_and_arrays_mutation_detected(self):
        original=SimpleNamespace(mol=object(),e_tot=-2.,converged=True,small_rho_cutoff=0,disp=None,
            mo_coeff="coeff",mo_energy="energy",mo_occ="occ",grids=SimpleNamespace())
        w=SimpleNamespace(molecule_physical_fingerprint=lambda mol:"mol")
        base=SimpleNamespace(array_fingerprint=lambda value:value)
        payload=M.original_mf_fingerprints(original,base)
        M.ensure_original_unchanged(original,"mol",payload,w,base)
        original.e_tot=-1.
        with self.assertRaisesRegex(ValueError,"MF/MO/energy/quadrature changed"):
            M.ensure_original_unchanged(original,"mol",payload,w,base)
        original.e_tot=-2.;original.mo_coeff="mutated"
        with self.assertRaises(ValueError):M.ensure_original_unchanged(original,"mol",payload,w,base)

    def test_source_qualification_bound_to_new_file_and_no_nuclear_gradient_calls(self):
        source=WORKER.read_text();tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(n.startswith(("numpy","scipy","pyscf")) for n in names))
        self.assertIn('seal["source_worker"]==ref(__file__)',source)
        self.assertIn('expected_threads=2',source)
        self.assertNotIn('from pyscf.grad',source)
        self.assertNotIn('gradient_obj.kernel',source)
        self.assertIn('addons.project_dm_nr2nr(fresh_zero.mol,fresh_zero.make_rdm1(),mol)',source)
        self.assertEqual(M.SCF_CONV_TOL,1e-12);self.assertEqual(M.SCF_CONV_TOL_GRAD,1e-9)
        self.assertEqual(M.sha(ROOT/"scripts/research4_fixed_quadrature_worker_20261107.py"),
            "cab1b47de0e1b2868acd5a1ac6ee8141a0e57c7f79f6a2755ca0139032157e2a")


class ControlReleaseTests(unittest.TestCase):
    def fixture(self,root):
        worker=root/"scripts/research4_fixed_grid_water_convergence_20261107.py"
        worker.parent.mkdir();worker.write_text("# frozen synthetic control producer for no-Q tests\n")
        out=root/"reports/water";out.mkdir(parents=True)
        ledger=str(root/"ledger.json");points=[];bykey={}
        cover=[(None,0.,0)]+[(c,h,s) for c in M.CONTROL_COORDINATES for h in M.CONTROL_STEPS for s in (-1,1)]
        for idx,(coordinate,step,sign) in enumerate(cover):
            gradient=[[0.,0.,0.] for _ in range(3)]
            energy=0.
            if coordinate is not None:
                gradient[coordinate//3][coordinate%3]=sign*step
                energy=.5*step*step
            value={"schema":M.CONTROL_POINT_SCHEMA,"status":M.CONTROL_POINT_SUCCESS,"method":M.METHOD,
                "gradient_completed":True,"SCF_performed":True,"scf_converged":True,"source_worker":M.ref(worker),
                "source_snapshot_manifest_sha256":"s"*64,"parent_output_directory":str(out),
                "settings":{"actual_conv_tol_hartree":1e-13,"actual_conv_tol_grad":1e-9},
                "strict_SCF_convergence_checks":[{"energy_delta_abs_hartree":0.,"orbital_gradient_norm":0.,"accepted":True}],
                "energy_hartree":energy,"gradient_hartree_per_bohr":gradient,
                "displacement":{"mode_index_1based":None,"coordinate_index_0based":coordinate,"step_bohr":step,"sign":sign}}
            point_dir=out/("point%d"%idx);point_dir.mkdir()
            hq=[[0.,0.,0.] for _ in range(3)]
            if coordinate is not None:hq[coordinate//3][coordinate%3]=1.
            value["computed_array_files"]={}
            for name in ("direction_cartesian_unit.npy","analytic_H_times_direction.npy","source_coordinates_bohr.npy",
                         "displaced_coordinates_bohr.npy","gradient_hartree_per_bohr.npy"):
                array=gradient if name=="gradient_hartree_per_bohr.npy" else hq
                header=repr({"descr":"<f8","fortran_order":False,"shape":(3,3)}).encode()+b"\n"
                file=point_dir/name
                file.write_bytes(b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+
                    struct.pack("<9d",*(x for row in array for x in row)))
                value["computed_array_files"][name]=M.ref(file)
            path=point_dir/"native_result.json";M.save(path,value,exclusive=True);points.append(M.ref(path));bykey[(coordinate,step,sign)]=value
        comparisons=[];halvings=[]
        for coordinate in M.CONTROL_COORDINATES:
            for step in M.CONTROL_STEPS:
                hq=[[0.,0.,0.] for _ in range(3)];hq[coordinate//3][coordinate%3]=1.
                comparisons.append({"coordinate_index_0based":coordinate,"step_bohr":step,
                    "analytic_Hq_hartree_per_bohr2":hq,"FD_Hq_hartree_per_bohr2":hq,
                    "FD_Hq_max_abs_error_Eh_per_Bohr2":0.,"central_energy_curvature_abs_error_Eh_per_Bohr2":0.,
                    "pair_completed":True,"fresh_zero_completed":True,
                    "plus_energy_hartree":.5*step*step,"minus_energy_hartree":.5*step*step,
                    "fresh_zero_energy_hartree":0.,"analytic_directional_curvature_Eh_per_Bohr2":1.,
                    "central_energy_curvature_Eh_per_Bohr2":1.})
            for coarse,fine in zip(M.CONTROL_STEPS[:-1],M.CONTROL_STEPS[1:]):
                halvings.append({"coordinate_index_0based":coordinate,"coarser_step_bohr":coarse,"finer_step_bohr":fine,
                    "max_abs_difference_Eh_per_Bohr2":0.})
        result={"schema":M.CONTROL_SCHEMA,"status":M.CONTROL_SUCCESS,"method":M.METHOD,
            "finest_pair_passed":True,"same_spinosin_step_energy_passed":True,"same_spinosin_energy_step_bohr":M.STEP_BOHR,
            "thresholds":M.CONTROL_THRESHOLDS,"case":"water_software_control","source_worker":M.ref(worker),
            "current_execution_boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "resource_supervision":{"proof":{"cpu_hard_limit_seconds":500,"ledger_path":ledger}},
            "snapshot_manifest_sha256":"s"*64,"predetermined_coordinates_0based":list(M.CONTROL_COORDINATES),
            "predetermined_steps_bohr":list(M.CONTROL_STEPS),"predetermined_signs":[-1,1],
            "all_31_prescribed_points_completed":True,"finest_pair_selection":list(M.CONTROL_STEPS[-2:]),
            "software_control_SCF_conv_tol_hartree":1e-13,"software_control_SCF_conv_tol_grad":1e-9,
            "all_points_including_zero_perform_fresh_SCF":True,
            "strict_SCF_energy_AND_orbital_gradient_required_including_extra_cycle":True,"points":points,"comparisons":comparisons,"h_halving_comparisons":halvings}
        path=out/"result.json";M.save(path,result,exclusive=True)
        args=SimpleNamespace(control_result=path,control_result_sha256=M.sha(path),control_worker_sha256=M.sha(worker))
        guard=SimpleNamespace(verify_closed_phase=mock.Mock(return_value={"real_closed_phase":True}))
        return worker,path,result,args,guard,{"proof":{"ledger_path":ledger}}

    def test_hash_bound_closed_controls_replayed_and_explicit_worker_bound(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            worker,path,result,args,guard,proof=self.fixture(Path(tmp))
            with mock.patch.object(M,"CONTROL_WORKER",worker):
                value=M.verify_control_release(args,guard,proof)
            self.assertTrue(value["all_31_child_artifacts_and_raw_comparisons_recomputed"])
            self.assertEqual(guard.verify_closed_phase.call_args.kwargs["expected_worker_path"],worker)
            self.assertEqual(guard.verify_closed_phase.call_args.kwargs["expected_worker_sha256"],M.sha(worker))

    def test_fake_pass_old_schema_wrong_boot_missing_energy_and_raw_forgery_rejected(self):
        mutations=[lambda r:r.update(schema="research4-fixed-original-quadrature-water-controls-v1"),
            lambda r:r.update(current_execution_boot_id="archived-boot"),
            lambda r:r.update(same_spinosin_step_energy_passed=False),
            lambda r:r["comparisons"][0].update(FD_Hq_max_abs_error_Eh_per_Bohr2=.01),
            lambda r:r["points"].__setitem__(0,r["points"][1])]
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                worker,path,result,args,guard,proof=self.fixture(Path(tmp));mutate(result);M.save(path,result);args.control_result_sha256=M.sha(path)
                with mock.patch.object(M,"CONTROL_WORKER",worker),self.assertRaises(ValueError):M.verify_control_release(args,guard,proof)
        for field in ("energy_hartree","gradient_hartree_per_bohr","strict_SCF_convergence_checks"):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                worker,path,result,args,guard,proof=self.fixture(Path(tmp))
                point_path=Path(result["points"][0]["path"]);point=M.read(point_path);point.pop(field);M.save(point_path,point)
                result["points"][0]=M.ref(point_path);M.save(path,result);args.control_result_sha256=M.sha(path)
                with mock.patch.object(M,"CONTROL_WORKER",worker),self.assertRaises(ValueError):M.verify_control_release(args,guard,proof)

    def test_child_raw_gradient_forgery_and_missing_sibling_hash_rejected(self):
        for mutation in ("json-gradient","missing-raw","changed-raw"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                worker,path,result,args,guard,proof=self.fixture(Path(tmp))
                point_path=Path(result["points"][0]["path"]);point=M.read(point_path)
                if mutation=="json-gradient":point["gradient_hartree_per_bohr"][0][0]=1.
                elif mutation=="missing-raw":point["computed_array_files"].pop("gradient_hartree_per_bohr.npy")
                else:
                    raw=Path(point["computed_array_files"]["gradient_hartree_per_bohr.npy"]["path"])
                    raw.write_bytes(raw.read_bytes()[:-8]+struct.pack("<d",1.))
                M.save(point_path,point);result["points"][0]=M.ref(point_path)
                M.save(path,result);args.control_result_sha256=M.sha(path)
                with mock.patch.object(M,"CONTROL_WORKER",worker),self.assertRaises(ValueError):M.verify_control_release(args,guard,proof)

    def test_forged_passing_result_cannot_bypass_actual_healthy_guard_closure(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            worker,path,result,args,guard,proof=self.fixture(Path(tmp))
            guard.verify_closed_phase.side_effect=ValueError("Not a healthy actual EXITED/0")
            with mock.patch.object(M,"CONTROL_WORKER",worker),self.assertRaisesRegex(ValueError,"healthy actual"):
                M.verify_control_release(args,guard,proof)


if __name__=="__main__":unittest.main()
