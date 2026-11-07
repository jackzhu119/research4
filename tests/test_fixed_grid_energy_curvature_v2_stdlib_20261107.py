"""No-Q v2 controls, bounded SCF polishing and consecutive-density energy audits."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import hashlib
import math
import struct
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/"scripts/research4_fixed_grid_energy_curvature_20261107_v2.py"
spec=importlib.util.spec_from_file_location("energy_curvature_v2_worker_under_test",WORKER)
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

    def test_main_trigger_and_final_AND_criteria_have_separate_predeclared_limits(self):
        limits={"main_energy_tolerance":1e-12,"main_orbital_gradient_trigger":1e-10,
            "final_energy_tolerance":1e-12,"final_orbital_gradient_tolerance":1e-9}
        main={"e_tot":0.,"last_hf_e":5e-13,"norm_gorb":5e-10,"conv_tol":1e-12,"conv_tol_grad":1e-10}
        extra={**main,"conv_tol":10*1e-12,"conv_tol_grad":3*1e-10}
        self.assertFalse(M.polish_convergence_check(main,**limits)["accepted"])
        self.assertTrue(M.polish_convergence_check(extra,**limits)["accepted"])
        self.assertEqual(M.polish_convergence_check(extra,**limits)["phase"],"FINAL_EXTRA_CYCLE")
        for change in ({"last_hf_e":2e-12,"norm_gorb":0.},{"last_hf_e":0.,"norm_gorb":2e-9},
                       {"norm_gorb":float("nan")},{"e_tot":float("inf")}):
            with self.subTest(change=change):
                self.assertFalse(M.polish_convergence_check({**extra,**change},**limits)["accepted"])

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

    def test_large_state_directory_is_local_separate_and_never_reused_or_redirected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/"queue-host-a-006";job=work/"science/mode1-central-energy"
            with mock.patch.object(M,"ENERGY_WORK_DIRECTORY",work):
                path=M.local_state_directory(job,"fresh-zero")
                self.assertEqual(path,work/"local-checkpoints/mode1-central-energy/fresh-zero")
                self.assertFalse(path.is_relative_to(job))
                for bad_job,bad_point in ((work/"other/mode1-central-energy","fresh-zero"),(job,"custom-point")):
                    with self.subTest(job=bad_job,point=bad_point),self.assertRaises(ValueError):
                        M.local_state_directory(bad_job,bad_point)
                path.mkdir(parents=True);evidence=path/"preserved.chk";evidence.write_bytes(b"must remain local")
                with self.assertRaises(ValueError):M.local_state_directory(job,"fresh-zero")
                self.assertEqual(evidence.read_bytes(),b"must remain local")
                outside=Path(tmp)/"outside";outside.mkdir()
                (work/"local-checkpoints/other-job").symlink_to(outside,target_is_directory=True)
                with self.assertRaises(ValueError):M.local_state_directory(work/"science/other-job","plus")

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
        self.assertEqual(M.SCF_MAIN_ORBITAL_GRADIENT_TRIGGER,1e-10)
        self.assertEqual(M.MAX_SCF_KERNEL_CALLS,3);self.assertEqual(M.SCF_MAX_CYCLES_PER_CALL,100)
        self.assertEqual(M.sha(ROOT/"scripts/research4_fixed_grid_energy_curvature_20261107.py"),
            "5c93bdeacae3e91d78c47ccfc3a73d1e715ab9ec0dd53391e3f7274536d7a6bf")
        self.assertEqual(M.sha(ROOT/"scripts/research4_fixed_quadrature_worker_20261107.py"),
            "cab1b47de0e1b2868acd5a1ac6ee8141a0e57c7f79f6a2755ca0139032157e2a")


class ControlReleaseTests(unittest.TestCase):
    def fixture(self,root):
        worker=root/"scripts/research4_fixed_grid_water_convergence_20261107_v3.py"
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
                "basis_functions":3,"occupied_orbitals":1,
                "settings":{"actual_conv_tol_hartree":1e-13,"actual_conv_tol_grad":1e-10,
                    "actual_scf_max_cycles_per_call":100,"maximum_SCF_kernel_calls":3,
                    "final_acceptance_energy_tolerance_hartree":1e-13,"final_acceptance_orbital_gradient_tolerance":1e-9},
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
            def write_array(name,array):
                shape=(len(array),len(array[0])) if isinstance(array[0],list) else (len(array),)
                flat=[x for row in array for x in row] if len(shape)==2 else array
                header=repr({"descr":"<f8","fortran_order":False,"shape":shape}).encode()+b"\n"
                target=point_dir/name
                target.write_bytes(b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+
                    struct.pack("<"+str(len(flat))+"d",*flat))
                return M.ref(target)
            def fingerprint(array):
                matrix=isinstance(array[0],list)
                shape=[len(array),len(array[0])] if matrix else [len(array)]
                flat=[x for row in array for x in row] if matrix else array
                payload=struct.pack("="+str(len(flat))+"d",*flat)
                return {"shape":shape,"dtype":"float64","payload_bytes":len(payload),
                    "payload_sha256_C_order":hashlib.sha256(payload).hexdigest()}
            last={"phase":"FINAL_EXTRA_CYCLE","kernel_call_1based":1,"accepted":True,
                "energy_delta_abs_hartree":1e-14,"orbital_gradient_norm":1e-10,
                "energy_tolerance_hartree":1e-13,"orbital_gradient_tolerance":1e-9,
                "acceptance_rule":"ENERGY_AND_ORBITAL_GRADIENT_NO_RELAXED_EXTRA_CYCLE_OR"}
            main={**last,"phase":"SCF_MAIN_CYCLE","orbital_gradient_norm":5e-11,
                "orbital_gradient_tolerance":1e-10}
            value["strict_SCF_convergence_checks"]=[main,last]
            previous=[[0.,0.,0.],[0.,2.,0.],[0.,0.,0.]]
            current=[[2.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
            identity=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            audit=M.audited_endpoint_qualification(True,last,energy+5e-14,energy,1e-10,
                preceding_main_check=main,native_checks_finite=True,density_change_norm=math.sqrt(8.),energy_tolerance=1e-13,orbital_gradient_tolerance=1e-9)
            self.assertTrue(audit["qualified"])
            raw={"start-density":previous,"previous-density":previous,"current-density":current,
                "MO-coeff":identity,"MO-energy":[-1.,0.,1.],"MO-occ":[2.,0.,0.]}
            kernel_energy=energy+1e-12
            attempt={"kernel_call_1based":1,"maximum_cycles":100,
                "main_energy_tolerance_hartree":1e-13,"main_orbital_gradient_trigger":1e-10,
                "final_energy_tolerance_hartree":1e-13,"final_orbital_gradient_tolerance":1e-9,
                "VEFF_audit_policy":"TWO_FULL_NATIVE_EVALUATIONS_DM_LAST_NONE_VHF_LAST_NONE",**audit,
                "native_kernel_energy_hartree":kernel_energy,
                "full_current_minus_kernel_energy_hartree":energy-kernel_energy,
                "density_change_Frobenius_norm":math.sqrt(8.),
                "start_density_fingerprint":fingerprint(previous),
                "previous_density_fingerprint":fingerprint(previous),
                "current_density_fingerprint":fingerprint(current),
                "arrays":{name:write_array("scf-call-1-%s.npy"%name,array) for name,array in raw.items()}}
            checkpoint=point_dir/"scf-call-1.chk";checkpoint.write_bytes(b"synthetic local checkpoint")
            attempt["checkpoint"]={**M.ref(checkpoint),"storage_policy":"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"}
            value.update(SCF_polishing_attempts=[attempt],qualified_SCF_kernel_call_1based=1,
                final_SCF_endpoint_audit=audit,SCF_convergence_scope=M.SCF_CERTIFICATION_POLICY,
                native_kernel_converged_at_certified_endpoint=True,
                reported_energy_evaluation="FULL_NONINCREMENTAL_NATIVE_VEFF_AT_AUDITED_RETURNED_MO_DENSITY")
            qualified={"MO_coeff":identity,"MO_energy":[-1.1,.1,1.1],"MO_occ":[2.,0.,0.],
                "density":current,"full_Fock":[[-1.1,5e-11,0.],[5e-11,.1,0.],[0.,0.,1.1]]}
            value["native_checkpoint_scope"]="NATIVE_KERNEL_ENDPOINT_BEFORE_FULL_ENERGY_AND_MO_ENERGY_REFRESH"
            value["final_MO_energy_evaluation"]="DIAGONAL_FULL_CURRENT_FOCK_IN_SAME_MO_BASIS_WITHOUT_ORBITAL_ROTATION"
            value["qualified_MF_array_files"]={name:write_array("qualified-%s.npy"%name,array)
                for name,array in qualified.items()}
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
            "resource_supervision":{"proof":{"cpu_hard_limit_seconds":300,"ledger_path":ledger}},
            "snapshot_manifest_sha256":"s"*64,"predetermined_coordinates_0based":list(M.CONTROL_COORDINATES),
            "predetermined_steps_bohr":list(M.CONTROL_STEPS),"predetermined_signs":[-1,1],
            "all_31_prescribed_points_completed":True,"finest_pair_selection":list(M.CONTROL_STEPS[-2:]),
            "software_control_SCF_conv_tol_hartree":1e-13,"software_control_SCF_conv_tol_grad":1e-9,
            "all_points_including_zero_perform_fresh_SCF":True,
            "full_native_energy_AND_orbital_gradient_required_at_final_extra_endpoint":True,
            "native_kernel_flags_are_preserved_separate_from_custom_full_native_certification":True,
            "SCF_main_orbital_gradient_trigger":1e-10,"maximum_SCF_kernel_calls_per_point":3,
            "maximum_SCF_cycles_per_call":100,
            "final_energy_evaluation":"FRESH_NONINCREMENTAL_NATIVE_VEFF_AT_QUALIFIED_CURRENT_DENSITY",
            "SCF_endpoint_certification_policy":M.SCF_CERTIFICATION_POLICY,"points":points,"comparisons":comparisons,"h_halving_comparisons":halvings}
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

    def test_control_release_keeps_false_native_flags_separate_from_full_native_certification(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            worker,path,result,args,guard,proof=self.fixture(Path(tmp))
            point_path=Path(result["points"][0]["path"]);point=M.read(point_path)
            last=point["strict_SCF_convergence_checks"][-1]
            last.update(accepted=False,energy_delta_abs_hartree=2e-13)
            old=point["final_SCF_endpoint_audit"]
            audit=M.audited_endpoint_qualification(False,last,old["previous_full_energy_hartree"],
                old["current_full_energy_hartree"],old["full_current_orbital_gradient_norm"],
                preceding_main_check=point["strict_SCF_convergence_checks"][-2],native_checks_finite=True,
                density_change_norm=old["consecutive_density_Frobenius_difference"],
                energy_tolerance=1e-13,orbital_gradient_tolerance=1e-9)
            self.assertTrue(audit["qualified"])
            point["final_SCF_endpoint_audit"]=audit
            point["SCF_polishing_attempts"][-1].update(audit)
            point["native_kernel_converged_at_certified_endpoint"]=False
            M.save(point_path,point);result["points"][0]=M.ref(point_path)
            M.save(path,result);args.control_result_sha256=M.sha(path)
            with mock.patch.object(M,"CONTROL_WORKER",worker):
                release=M.verify_control_release(args,guard,proof)
            self.assertTrue(release["control_passed"])
            self.assertFalse(M.read(point_path)["native_kernel_converged_at_certified_endpoint"])
            self.assertFalse(M.read(point_path)["strict_SCF_convergence_checks"][-1]["accepted"])

    def test_endpoint_rejects_fake_scope_missing_lineage_raw_arrays_and_full_Fock_forgery(self):
        for mutation in ("native-scope","main-missing","raw-missing","density-fingerprint","checkpoint-hash",
                         "too-many-calls","too-many-cycles","qualified-density","full-Fock"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                worker,path,result,args,guard,proof=self.fixture(Path(tmp))
                point_path=Path(result["points"][0]["path"]);point=M.read(point_path)
                attempt=point["SCF_polishing_attempts"][-1]
                if mutation=="native-scope":point["SCF_convergence_scope"]="NATIVE_KERNEL_CONVERGED"
                elif mutation=="main-missing":point["strict_SCF_convergence_checks"]=point["strict_SCF_convergence_checks"][-1:]
                elif mutation=="raw-missing":attempt["arrays"].pop("previous-density")
                elif mutation=="density-fingerprint":attempt["previous_density_fingerprint"]["payload_sha256_C_order"]="0"*64
                elif mutation=="checkpoint-hash":attempt["checkpoint"]["sha256"]="0"*64
                elif mutation=="too-many-calls":point["qualified_SCF_kernel_call_1based"]=4
                elif mutation=="too-many-cycles":attempt["maximum_cycles"]=101
                else:
                    name="density" if mutation=="qualified-density" else "full_Fock"
                    record=point["qualified_MF_array_files"][name];raw=Path(record["path"])
                    matrix=M.checked_control_array(record,(3,3))
                    if name=="density":matrix[0][0]+=1.
                    else:matrix[1][0]=matrix[0][1]=5e-8
                    raw.write_bytes(raw.read_bytes()[:-72]+struct.pack("<9d",*(x for row in matrix for x in row)))
                    point["qualified_MF_array_files"][name]=M.ref(raw)
                with self.assertRaises(ValueError):M.verify_control_endpoint(point,point_path)

    def test_forged_passing_result_cannot_bypass_actual_healthy_guard_closure(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            worker,path,result,args,guard,proof=self.fixture(Path(tmp))
            guard.verify_closed_phase.side_effect=ValueError("Not a healthy actual EXITED/0")
            with mock.patch.object(M,"CONTROL_WORKER",worker),self.assertRaisesRegex(ValueError,"healthy actual"):
                M.verify_control_release(args,guard,proof)


class ScalarMatrix:
    """One-by-one fake array; this never imports a numerical package."""
    shape=(1,1)
    def __init__(self,value):self.value=float(value.value if isinstance(value,ScalarMatrix) else value)
    def __sub__(self,other):return ScalarMatrix(self.value-other.value)
    def __matmul__(self,other):return ScalarMatrix(self.value*other.value)


class ScalarNP:
    @staticmethod
    def array(value,**kwargs):return ScalarMatrix(value)
    @staticmethod
    def asarray(value):return value
    @staticmethod
    def array_equal(left,right):return left.value==right.value
    @staticmethod
    def shares_memory(left,right):return left is right
    @staticmethod
    def isfinite(value):return SimpleNamespace(all=lambda:math.isfinite(value.value))
    @staticmethod
    def einsum(pattern,left,right):return ScalarMatrix(left.value*right.value)
    linalg=SimpleNamespace(norm=lambda value:abs(value.value))


class FakeNativeSCF:
    """A deterministic native-SCF stand-in exposing both last-step densities."""
    def __init__(self,specs):
        self.specs=specs;self.calls=0;self.starts=[];self.veff_calls=[];self.veff_objects=[];self.fock_calls=[]
        self.diis=True;self.mol=SimpleNamespace(nao_nr=lambda:1)
        self.grids=SimpleNamespace(mol=self.mol,cutoff=1e-15)
        self.small_rho_cutoff=0;self.direct_scf=True;self.direct_scf_tol=1e-13
        self._numint=SimpleNamespace(cutoff=1e-13)
        self.mo_coeff=ScalarMatrix(1.);self.mo_occ=ScalarMatrix(2.);self.mo_energy=ScalarMatrix(-2.)
    def get_hcore(self,mol):return ScalarMatrix(1.)
    def get_ovlp(self,mol):return ScalarMatrix(1.)
    def kernel(self,dm0):
        self.calls+=1;self.starts.append(dm0.value);self.spec=self.specs[min(self.calls-1,len(self.specs)-1)]
        previous,current=self.spec["previous_dm"],self.spec["current_dm"]
        self.final_density=ScalarMatrix(self.spec.get("returned_dm",current))
        if self.spec.get("suppress_checks"):
            self.converged=True;self.e_tot=self.spec["current_energy"];return
        if self.spec.get("raise_error"):
            raise self.spec["raise_error"]
        env={"e_tot":0.,"last_hf_e":5e-13,"norm_gorb":5e-11,
            "conv_tol":self.conv_tol,"conv_tol_grad":self.conv_tol_grad,
            "dm_last":ScalarMatrix(previous),"dm":ScalarMatrix(current),"cycle":0}
        self.check_convergence(env);self.callback(env)
        env={**env,"conv_tol":10*self.conv_tol,"conv_tol_grad":3*self.conv_tol_grad,"norm_gorb":5e-10,
            "last_hf_e":self.spec.get("native_extra_energy_drift",5e-13)}
        accepted=self.check_convergence(env)
        self.converged=accepted and self.spec.get("native_converged",True)
        self.e_tot=self.spec.get("kernel_energy",self.spec["current_energy"]+1e-8)
        Path(self.chkfile).write_bytes(b"fake local-only checkpoint")
    def make_rdm1(self):return self.final_density
    def get_veff(self,mol,dm,**kwargs):
        self.veff_calls.append((dm.value,dict(kwargs)));self.veff_objects.append(dm);return ScalarMatrix(dm)
    def energy_tot(self,dm,hcore,veff):
        return self.spec["previous_energy"] if dm.value==self.spec["previous_dm"] else self.spec["current_energy"]
    def get_fock(self,**kwargs):self.fock_calls.append(dict(kwargs));return ScalarMatrix(1.)
    def get_grad(self,mo_coeff,mo_occ,fock):return ScalarMatrix(self.spec.get("full_orbital_gradient",5e-10))


class BoundedConsecutiveDensityAuditTests(unittest.TestCase):
    def call(self,mf,out,**changes):
        def save_array(path,value):path.write_text(repr(value.value)+"\n")
        base=SimpleNamespace(array_fingerprint=lambda value:repr(value.value),atomic_npy=save_array)
        state={};events=[]
        value=M.bounded_polished_scf(mf,ScalarMatrix(0.),np=ScalarNP,base=base,out=out,
            emit=lambda name,row:events.append((name,copy.deepcopy(row))),state=state,
            final_energy_tolerance=1e-12,final_orbital_gradient_tolerance=1e-9,
            main_orbital_gradient_trigger=1e-10,**changes)
        return value,state,events

    def spec(self,**changes):
        return {"previous_dm":1.,"current_dm":2.,"previous_energy":-2.,"current_energy":-2.+5e-13,
            "full_orbital_gradient":5e-10,**changes}

    def native_checks(self,*,last_accepted=True):
        main={"phase":"SCF_MAIN_CYCLE","kernel_call_1based":1,"accepted":True,
            "energy_delta_abs_hartree":5e-13,"orbital_gradient_norm":5e-11,
            "energy_tolerance_hartree":1e-12,"orbital_gradient_tolerance":1e-10}
        last={**main,"phase":"FINAL_EXTRA_CYCLE","accepted":last_accepted,
            "energy_delta_abs_hartree":5e-13 if last_accepted else 2e-12,
            "orbital_gradient_norm":5e-10,"orbital_gradient_tolerance":1e-9}
        return main,last

    def test_final_qualification_allows_fixedpoint_zero_delta_but_rejects_invalid_density_delta(self):
        main,last=self.native_checks()
        for difference in (-1.,float("nan"),float("inf")):
            with self.subTest(difference=difference):
                result=M.audited_endpoint_qualification(True,last,-2.,-2.,0.,
                    preceding_main_check=main,native_checks_finite=True,density_change_norm=difference,
                    energy_tolerance=1e-12,orbital_gradient_tolerance=1e-9)
                self.assertFalse(result["qualified"])
        for difference in (0.,.1):
            self.assertTrue(M.audited_endpoint_qualification(True,last,-2.,-2.+5e-13,5e-10,
                preceding_main_check=main,native_checks_finite=True,density_change_norm=difference,
                energy_tolerance=1e-12,orbital_gradient_tolerance=1e-9)["qualified"])

    def test_two_full_nonincremental_energies_and_unshifted_fock_replace_kernel_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf=FakeNativeSCF([self.spec()]);result,state,events=self.call(mf,Path(tmp))
            self.assertTrue(result["qualified"]);self.assertEqual(mf.calls,1)
            self.assertEqual(mf.veff_calls,[(1.,{"dm_last":None,"vhf_last":None}),(2.,{"dm_last":None,"vhf_last":None})])
            self.assertEqual(mf.e_tot,mf.spec["current_energy"])
            self.assertNotEqual(mf.e_tot,mf.spec.get("kernel_energy",mf.spec["current_energy"]+1e-8))
            self.assertEqual(mf.fock_calls[-1]["cycle"],-1);self.assertIsNone(mf.fock_calls[-1]["diis"])
            self.assertEqual(mf.fock_calls[-1]["level_shift_factor"],0);self.assertEqual(mf.fock_calls[-1]["damp_factor"],0)
            self.assertEqual(mf.mo_coeff.value,1.);self.assertEqual(mf.mo_occ.value,2.)
            self.assertEqual(state["qualified_SCF_kernel_call_1based"],1)
            self.assertEqual(state["reported_energy_evaluation"],"FULL_NONINCREMENTAL_NATIVE_VEFF_AT_AUDITED_RETURNED_MO_DENSITY")

    def test_large_MO_density_and_checkpoint_payload_stays_outside_small_job_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"science/point";output.mkdir(parents=True)
            storage=Path(tmp)/"local-checkpoints/job/point";storage.mkdir(parents=True)
            mf=FakeNativeSCF([self.spec()]);audit,state,events=self.call(mf,output,storage_out=storage)
            self.assertTrue(audit["qualified"]);self.assertEqual(list(output.iterdir()),[])
            self.assertEqual(state["large_audit_states_storage_policy"],"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT")
            references=[*state["qualified_MF_array_files"].values()]
            for attempt in state["SCF_polishing_attempts"]:
                references.extend(attempt["arrays"].values());references.append(attempt["checkpoint"])
            self.assertGreaterEqual(len(references),12)
            for record in references:
                self.assertEqual(Path(record["path"]).parent,storage)
                self.assertEqual(M.sha(record["path"]),record["sha256"])
            self.assertEqual(state["native_checkpoint_scope"],"NATIVE_KERNEL_ENDPOINT_BEFORE_FULL_ENERGY_AND_MO_ENERGY_REFRESH")

    def test_only_failed_endpoint_continues_from_immediately_preceding_density(self):
        specs=[self.spec(current_energy=-2.+2e-12),
               self.spec(previous_dm=2.,current_dm=3.,previous_energy=-2.+2e-12,current_energy=-2.+2.5e-12)]
        with tempfile.TemporaryDirectory() as tmp:
            mf=FakeNativeSCF(specs);result,state,events=self.call(mf,Path(tmp))
            self.assertTrue(result["qualified"]);self.assertEqual(mf.calls,2);self.assertEqual(mf.starts,[0.,2.])
            self.assertFalse(state["SCF_polishing_attempts"][0]["qualified"])
            self.assertEqual(state["qualified_SCF_kernel_call_1based"],2)
            self.assertEqual(mf.max_cycle,100)

    def test_three_total_calls_cap_and_no_fourth_call_if_final_audits_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf=FakeNativeSCF([self.spec(full_orbital_gradient=2e-9)])
            with self.assertRaises(ValueError):self.call(mf,Path(tmp))
            self.assertEqual(mf.calls,3)
        for changes in ({"max_calls":4},{"max_cycles":101},{"max_calls":0}):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as tmp:
                mf=FakeNativeSCF([self.spec()])
                with self.assertRaises(ValueError):self.call(mf,Path(tmp),**changes)
                self.assertEqual(mf.calls,0)

    def test_true_fixedpoint_has_two_actual_callback_states_and_two_independent_full_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf=FakeNativeSCF([self.spec(previous_dm=2.,current_dm=2.,previous_energy=-2.,current_energy=-2.)])
            audit,state,events=self.call(mf,Path(tmp))
            self.assertTrue(audit["qualified"])
            self.assertEqual(audit["consecutive_density_Frobenius_difference"],0.)
            self.assertEqual(len(mf.veff_calls),2)
            self.assertIsNot(mf.veff_objects[0],mf.veff_objects[1])
            self.assertEqual([item[0] for item in mf.veff_calls],[2.,2.])

    def test_missing_callback_lineage_or_returned_density_mismatch_rejects_before_full_audit(self):
        for change in ({"suppress_checks":True},{"returned_dm":3.}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                mf=FakeNativeSCF([self.spec(**change)])
                with self.assertRaises(ValueError):self.call(mf,Path(tmp))
                self.assertEqual(mf.calls,1);self.assertEqual(mf.veff_calls,[])

    def test_resource_or_kernel_exception_propagates_without_an_automatic_new_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf=FakeNativeSCF([self.spec(raise_error=RuntimeError("synthetic resource stop"))])
            with self.assertRaisesRegex(RuntimeError,"resource stop"):self.call(mf,Path(tmp))
            self.assertEqual(mf.calls,1);self.assertEqual(mf.veff_calls,[])

    def test_final_audit_requires_actual_extra_after_same_call_main_and_both_full_criteria(self):
        main,last=self.native_checks()
        cases=[({**last,"phase":"SCF_MAIN_CYCLE"},main,True,-2.,-2.+5e-13,5e-10),
            (last,{**main,"accepted":False},True,-2.,-2.+5e-13,5e-10),
            (last,{**main,"kernel_call_1based":2},True,-2.,-2.+5e-13,5e-10),
            (last,main,False,-2.,-2.+5e-13,5e-10),
            ({**last,"orbital_gradient_norm":float("nan")},main,False,-2.,-2.+5e-13,5e-10),
            (last,main,True,-2.,-2.+2e-12,0.),(last,main,True,-2.,-2.,2e-9)]
        for extra,preceding,finite,previous,current,gradient in cases:
            with self.subTest(extra=extra,preceding=preceding,gradient=gradient):
                self.assertFalse(M.audited_endpoint_qualification(extra["accepted"],extra,previous,current,gradient,
                    preceding_main_check=preceding,native_checks_finite=finite,density_change_norm=.1,
                    energy_tolerance=1e-12,orbital_gradient_tolerance=1e-9)["qualified"])

    def test_false_incremental_extra_flag_is_preserved_while_full_native_audit_certifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf=FakeNativeSCF([self.spec(native_extra_energy_drift=2e-12)])
            audit,state,events=self.call(mf,Path(tmp))
            self.assertTrue(audit["qualified"]);self.assertTrue(mf.converged)
            self.assertFalse(audit["native_kernel_converged"])
            self.assertFalse(audit["last_native_check"]["accepted"])
            self.assertGreater(audit["last_native_check"]["energy_delta_abs_hartree"],1e-12)
            self.assertFalse(state["native_kernel_converged_at_certified_endpoint"])
            self.assertEqual(state["SCF_convergence_scope"],M.SCF_CERTIFICATION_POLICY)
            self.assertEqual(audit["certification_policy"],M.SCF_CERTIFICATION_POLICY)
            self.assertTrue(audit["actual_native_final_extra_after_accepted_main_same_kernel"])

    def test_native_diagnostics_must_remain_finite_real_values(self):
        main,last=self.native_checks()
        self.assertTrue(M.finite_native_scf_check(last))
        for key in ("energy_delta_abs_hartree","orbital_gradient_norm","energy_tolerance_hartree","orbital_gradient_tolerance"):
            for bad in (True,None,float("nan"),float("inf"),-1.):
                with self.subTest(key=key,bad=bad):
                    self.assertFalse(M.finite_native_scf_check({**last,key:bad}))


if __name__=="__main__":unittest.main()
