"""Metadata/fake-driver checks only: no PySCF imports, integrals, SCF or Git."""
import argparse
import ast
import copy
from contextlib import redirect_stderr
import importlib.util
import hashlib
import io
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/"scripts/research4_full_native_veff_restart_helpers_20261107.py"
WORKER=ROOT/"scripts/research4_full_native_checkpoint_worker_20261107.py"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


H=load("full_native_restart_helpers_test",HELPER)
M=load("full_native_checkpoint_worker_test",WORKER)


def array_fingerprint(array):
    shape=[len(array),len(array[0])] if isinstance(array[0],list) else [len(array)]
    flat=[value for row in array for value in row] if len(shape)==2 else array
    payload=struct.pack("="+str(len(flat))+"d",*flat)
    return {"shape":shape,"dtype":"float64","payload_bytes":len(payload),
        "payload_sha256_C_order":hashlib.sha256(payload).hexdigest()}


def write_npy(path,array):
    shape=(len(array),len(array[0])) if isinstance(array[0],list) else (len(array),)
    flat=[value for row in array for value in row] if len(shape)==2 else array
    header=repr({"descr":"<f8","fortran_order":False,"shape":shape}).encode()+b"\n"
    path.write_bytes(b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+
        struct.pack("<"+str(len(flat))+"d",*flat))
    return M.ref(path)


class ParentMF:
    def __init__(self):
        self.mol=SimpleNamespace(incore_anyway=False)
        self.grids=SimpleNamespace(mol=self.mol)
        self.direct_scf=True;self._eri=None;self.with_df=None;self.small_rho_cutoff=0
    def get_veff(self,*args,**kwargs):
        raise AssertionError("The parent cached builder must not be monkeypatched or called")


class FullNativeVEFFTests(unittest.TestCase):
    def test_full_build_ignores_incremental_operators_and_preserves_explicit_molecule_dm_hermiticity(self):
        calls=[];events=[];result=object()
        def native(self,mol,dm,**kwargs):calls.append((self,mol,dm,kwargs));return result
        cls=H.full_native_veff_type(ParentMF,native,emit=lambda event,row:events.append((event,row)))
        mf=cls();dm=object();old_dm=object();old_vhf=object()
        self.assertIs(mf.get_veff(mf.mol,dm,dm_last=old_dm,vhf_last=old_vhf,hermi=0),result)
        self.assertEqual(calls,[(mf,mf.mol,dm,{"dm_last":None,"vhf_last":None,"hermi":0})])
        self.assertTrue(events[0][1]["incoming_incremental_arguments_ignored"])
        mf.get_veff(dm=dm)
        self.assertEqual(calls[-1][3],{"dm_last":None,"vhf_last":None,"hermi":1})
        self.assertEqual(mf._research4_full_veff_calls,2)
        self.assertFalse(events[-1][1]["incoming_incremental_arguments_ignored"])
        self.assertEqual(cls._research4_full_native_veff_policy,H.FULL_BUILD_POLICY)

    def test_new_class_does_not_change_parent_source_instance_or_input_arrays(self):
        original=ParentMF();active=ParentMF();parent_method=ParentMF.get_veff
        original_state=original.__dict__.copy();dm=[[1.,0.],[0.,0.]];before=copy.deepcopy(dm)
        cls=H.full_native_veff_type(ParentMF,lambda self,mol,dm,**kwargs:dm)
        active.__class__=cls;self.assertIs(active.get_veff(active.mol,dm),dm)
        self.assertEqual(dm,before);self.assertEqual(original.__dict__,original_state)
        self.assertIs(type(original),ParentMF);self.assertIs(ParentMF.get_veff,parent_method)
        self.assertFalse(hasattr(original,"_research4_full_veff_calls"))

    def test_wrong_molecule_DF_ERI_direct_and_grid_policies_fail_before_any_native_build(self):
        for change in ("wrong-mol","direct-off","eri","df","incore","wrong-grid-mol","rho-prune"):
            with self.subTest(change=change):
                native=mock.Mock();mf=H.full_native_veff_type(ParentMF,native)();mol=mf.mol
                if change=="wrong-mol":mol=object()
                elif change=="direct-off":mf.direct_scf=False
                elif change=="eri":mf._eri=object()
                elif change=="df":mf.with_df=object()
                elif change=="incore":mf.mol.incore_anyway=True
                elif change=="wrong-grid-mol":mf.grids.mol=object()
                else:mf.small_rho_cutoff=1e-8
                with self.assertRaises(ValueError):mf.get_veff(mol,object(),dm_last=object(),vhf_last=object())
                native.assert_not_called()

    def test_imports_are_metadata_only_and_helper_has_exact_three_exported_functions(self):
        expected={"full_native_veff_type","validate_checkpoint_restart_payload","compare_same_density_veff"}
        tree=ast.parse(HELPER.read_text())
        self.assertEqual({node.name for node in tree.body if isinstance(node,ast.FunctionDef)},expected)
        for file in (HELPER,WORKER):
            tree=ast.parse(file.read_text())
            for node in tree.body:
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module]
                    self.assertFalse(any(n.startswith(("pyscf","numpy","h5py","scipy")) for n in names))


class FiniteStageTests(unittest.TestCase):
    def args(self,**changes):
        value=argparse.Namespace(stage="zero",threads=2,max_cycles=30,source_zero_result=None,
            source_zero_result_sha256=None,mode_index=None,steps=list(M.STEPS))
        for key,item in changes.items():setattr(value,key,item)
        return value

    def test_exact_30_cycle_pilot_and_80_cycle_continuation(self):
        M.validate_stage_args(self.args())
        M.validate_stage_args(self.args(max_cycles=80,source_zero_result=Path("pilot.json")))
        for changes in ({"max_cycles":80},{"max_cycles":30,"source_zero_result":Path("pilot.json")},
                        {"max_cycles":100},{"max_cycles":0},{"threads":1}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):M.validate_stage_args(self.args(**changes))

    def test_controls_have_one_bounded_80_cycle_variant_and_no_zero_dependency(self):
        M.validate_stage_args(self.args(stage="controls",max_cycles=80))
        for changes in ({"max_cycles":30},{"max_cycles":81},{"source_zero_result":Path("zero.json")}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                M.validate_stage_args(self.args(**{"stage":"controls","max_cycles":80,**changes}))

    def test_energy_anchors_steps_certified_dependency_and_cycles_are_predeclared(self):
        self.assertEqual(M.MODES,(1,13,25,26));self.assertEqual(M.STEPS,(.005,.0025))
        for anchor in M.MODES:
            M.validate_stage_args(self.args(stage="energies",max_cycles=80,mode_index=anchor,source_zero_result=Path("zero.json")))
        base={"stage":"energies","max_cycles":80,"mode_index":1,"source_zero_result":Path("zero.json")}
        for changes in ({"mode_index":2},{"max_cycles":30},{"source_zero_result":None},
                        {"steps":[.0025,.005]},{"steps":[.005,.0025,.00125]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):M.validate_stage_args(self.args(**{**base,**changes}))

    def test_cli_rejects_unknown_stage_before_engine_work(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            M.parser().parse_args(["--execute","--output","metadata-only","--main-pipeline-state","state.json",
                "--main-pipeline-state-sha256","a"*64,"--stage","unregistered"])


class CompactStoragePreflightTests(unittest.TestCase):
    def test_global_free_floor_and_each_remaining_point_reserve_are_required(self):
        self.assertEqual(M.POINT_STORAGE_RESERVE_BYTES,12*1024**2)
        self.assertEqual(M.GLOBAL_FREE_FLOOR_BYTES,64*1024**2)
        for points in range(5):
            required=64*1024**2+points*12*1024**2
            with self.subTest(points=points),mock.patch.object(M.shutil,"disk_usage",return_value=SimpleNamespace(free=required)):
                result=M.free_space_preflight(Path("metadata-only"),points)
                self.assertEqual(result["required_bytes"],required);self.assertEqual(result["remaining_points"],points)
            with mock.patch.object(M.shutil,"disk_usage",return_value=SimpleNamespace(free=required-1)),self.assertRaises(ValueError):
                M.free_space_preflight(Path("metadata-only"),points)
        for bad in (-1,5,True,1.0):
            with self.subTest(bad=bad),mock.patch.object(M.shutil,"disk_usage") as usage,self.assertRaises(ValueError):
                M.free_space_preflight(Path("metadata-only"),bad)
            usage.assert_not_called()

    def test_insufficient_global_space_preserves_existing_evidence_and_precedes_native_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);evidence=path/"earlier-result.json";evidence.write_bytes(b"preserved failed results")
            with mock.patch.object(M.shutil,"disk_usage",return_value=SimpleNamespace(free=1)),self.assertRaises(ValueError):
                M.free_space_preflight(path,4)
            self.assertEqual(evidence.read_bytes(),b"preserved failed results")
            self.assertEqual(list(path.iterdir()),[evidence])
        tree=ast.parse(WORKER.read_text());body=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="execute_closed")
        calls=[node for node in ast.walk(body) if isinstance(node,ast.Call)]
        preflight=min(node.lineno for node in calls if isinstance(node.func,ast.Name) and node.func.id=="free_space_preflight")
        restore=min(node.lineno for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=="restore_snapshot")
        self.assertLess(preflight,restore)

    def test_only_compact_previous_density_gradient_and_small_MO_energies_are_persisted(self):
        tree=ast.parse(WORKER.read_text());body=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="compute_point")
        selections=[]
        for node in ast.walk(body):
            if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple):
                names=[item.id for item in node.target.elts if isinstance(item,ast.Name)]
                if names==["key","filename"]:selections.append(ast.literal_eval(node.iter))
        self.assertEqual(selections,[(('previous-density','previous-density.npy'),
            ('full-orbital-gradient','full-orbital-gradient.npy'),('MO_energy','refreshed-MO-energy.npy'))])
        source=ast.get_source_segment(WORKER.read_text(),body)
        self.assertNotIn('initial-density.npy',source)
        self.assertIn('all_intermediate_raw_matrices_archived',source)
        self.assertIn('audit_state_fingerprints',source)


class DependencyTests(unittest.TestCase):
    def fixture(self,root):
        out=root/"reports/zero";pointout=out/"fresh-zero";pointout.mkdir(parents=True)
        checkpoint=pointout/"rolling.chk";checkpoint.write_bytes(b"synthetic initial-guess checkpoint")
        point={"schema":M.POINT_SCHEMA,"source_worker":M.ref(WORKER),"status":"FULL_NATIVE_ENDPOINT_CERTIFIED",
            "output_directory":str(pointout),"SCF_endpoint_certified":True,"energy_completed":True,
            "energy_hartree":-2.,"endpoint_audit":{"qualified":True,"current_full_energy_hartree":-2.,
                "previous_full_energy_hartree":-2.+5e-13,"full_current_orbital_gradient_norm":5e-10,
                "full_same_density_energy_repeats_hartree":[-2.,-2.,-2.],"full_same_density_energy_spread_hartree":0.,
                "actual_native_final_extra_after_accepted_main_same_kernel":True,"full_VEFF_applied_every_cycle":True},
            "SCF_convergence_scope":"FULL_NATIVE_EVERY_CYCLE_AND_INDEPENDENT_ENDPOINT_AUDIT",
            "checkpoint":M.ref(checkpoint)}
        states={"previous-density":[[0.,0.],[0.,2.]],"current-density":[[2.,0.],[0.,0.]],
            "MO_coeff":[[1.,0.],[0.,1.]],"MO_occ":[2.,0.],"MO_energy":[-1.,1.],
            "full_Fock":[[-1.,2.5e-10],[2.5e-10,1.]],"full-orbital-gradient":[5e-10]}
        point.update(storage_policy=M.STORAGE_POLICY,all_intermediate_raw_matrices_archived=False,
            basis_functions=2,electrons=2,audit_state_fingerprints={name:array_fingerprint(array) for name,array in states.items()})
        point["audit_state_files"]={name:write_npy(pointout/(name+".npy"),states[name])
            for name in ("previous-density","full-orbital-gradient")}
        point["audit_state_files"]["refreshed-MO-energy"]=write_npy(pointout/"refreshed-MO-energy.npy",states["MO_energy"])
        pointpath=pointout/"native_result.json";M.save(pointpath,point)
        value={"schema":M.SCHEMA,"source_worker":M.ref(WORKER),"stage":"zero","status":"ZERO_PILOT_COMPLETE",
            "output_directory":str(out),"zero_certified":True,"case":"Spinosin",
            "snapshot_manifest_sha256":"3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311",
            "points":{"fresh-zero":{"result":M.ref(pointpath),"status":point["status"],"energy_completed":True,"energy_hartree":-2.}},
            "completed_points":["fresh-zero"],"energies_hartree":{"fresh-zero":-2.},"energies":{"fresh-zero":-2.}}
        path=out/"result.json";M.save(path,value)
        args=SimpleNamespace(source_zero_result=path,source_zero_result_sha256=M.sha(path))
        guard=SimpleNamespace(verify_closed_phase=mock.Mock(return_value={"healthy_closed":True}))
        return args,guard,path,value,pointpath,point

    def update(self,args,path,value,pointpath,point):
        M.save(pointpath,point);value["points"]["fresh-zero"]["result"]=M.ref(pointpath)
        M.save(path,value);args.source_zero_result_sha256=M.sha(path)

    def test_certified_zero_requires_real_closed_producer_and_matching_child(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            args,guard,path,value,pointpath,point=self.fixture(Path(tmp))
            actual,actualpoint,record=M.dependency(args,guard,certified=True)
            self.assertTrue(actual["zero_certified"]);self.assertEqual(actualpoint["energy_hartree"],-2.)
            guard.verify_closed_phase.assert_called_once()
            self.assertEqual(guard.verify_closed_phase.call_args.kwargs["expected_worker_path"],WORKER)
            guard.verify_closed_phase.side_effect=ValueError("No healthy actual producer")
            with self.assertRaisesRegex(ValueError,"healthy actual"):M.dependency(args,guard,certified=True)

    def test_uncertified_pilot_can_supply_only_a_continuation_initial_guess(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            args,guard,path,value,pointpath,point=self.fixture(Path(tmp))
            value["zero_certified"]=False;value["completed_points"]=[];value["energies_hartree"]={};value["energies"]={}
            value["points"]["fresh-zero"].update(status="BOUNDED_SCF_MEASUREMENT_COMPLETE_UNCERTIFIED",energy_completed=False)
            value["points"]["fresh-zero"].pop("energy_hartree")
            point.update(status="BOUNDED_SCF_MEASUREMENT_COMPLETE_UNCERTIFIED",SCF_endpoint_certified=False,energy_completed=False)
            point["endpoint_audit"]["qualified"]=False;point.pop("energy_hartree")
            self.update(args,path,value,pointpath,point)
            M.dependency(args,guard,certified=False)
            with self.assertRaises(ValueError):M.dependency(args,guard,certified=True)

    def test_forged_zero_flag_missing_energy_wrong_source_or_uncertified_child_rejected(self):
        for change in ("child-uncertified","audit-false","energy-incomplete","energy-missing","energy-nonfinite",
                       "parent-energy-missing","wrong-child-source","wrong-parent-stage"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                args,guard,path,value,pointpath,point=self.fixture(Path(tmp))
                if change=="child-uncertified":point["SCF_endpoint_certified"]=False
                elif change=="audit-false":point["endpoint_audit"]["qualified"]=False
                elif change=="energy-incomplete":point["energy_completed"]=False
                elif change=="energy-missing":point.pop("energy_hartree")
                elif change=="energy-nonfinite":point["energy_hartree"]="not-a-real-energy"
                elif change=="parent-energy-missing":value["energies_hartree"]={};value["energies"]={}
                elif change=="wrong-child-source":point["source_worker"]={"path":str(WORKER),"sha256":"0"*64}
                else:value["stage"]="energies"
                self.update(args,path,value,pointpath,point)
                with self.assertRaises(ValueError):M.dependency(args,guard,certified=True)

    def test_certified_scalar_booleans_cannot_override_repeat_energy_residual_or_lineage_limits(self):
        for mutation in ("repeat-spread","full-gradient","consecutive-energy","no-extra","no-full-VEFF"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                args,guard,path,value,pointpath,point=self.fixture(Path(tmp));audit=point["endpoint_audit"]
                if mutation=="repeat-spread":
                    audit["full_same_density_energy_repeats_hartree"]=[-2.,-2.+2e-12,-2.]
                    audit["full_same_density_energy_spread_hartree"]=2e-12
                elif mutation=="full-gradient":audit["full_current_orbital_gradient_norm"]=2e-9
                elif mutation=="consecutive-energy":audit["previous_full_energy_hartree"]=-2.+2e-12
                elif mutation=="no-extra":audit["actual_native_final_extra_after_accepted_main_same_kernel"]=False
                else:audit["full_VEFF_applied_every_cycle"]=False
                with self.assertRaises(ValueError):M.verify_certified_point(point)

    def test_compact_metadata_and_actual_gradient_vector_cannot_be_forged_with_scalar_flags(self):
        for mutation in ("archive-scope","missing-vector","vector-wrong-norm","scalar-wrong-norm","vector-truncated","fingerprint-missing"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
                args,guard,path,value,pointpath,point=self.fixture(Path(tmp))
                if mutation=="archive-scope":point["all_intermediate_raw_matrices_archived"]=True
                elif mutation=="missing-vector":Path(point["audit_state_files"]["full-orbital-gradient"]["path"]).unlink()
                elif mutation=="scalar-wrong-norm":point["endpoint_audit"]["full_current_orbital_gradient_norm"]=1e-11
                elif mutation=="fingerprint-missing":point["audit_state_fingerprints"]["full_Fock"]["payload_sha256_C_order"]="missing"
                else:
                    vector=[2e-9] if mutation=="vector-wrong-norm" else [5e-10,0.]
                    record=point["audit_state_files"]["full-orbital-gradient"]
                    point["audit_state_files"]["full-orbital-gradient"]=write_npy(Path(record["path"]),vector)
                    point["audit_state_fingerprints"]["full-orbital-gradient"]=array_fingerprint(vector)
                with self.assertRaises(ValueError):M.verify_certified_point(point)

    def test_archived_gradient_sign_change_with_same_norm_must_match_computed_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            args,guard,path,value,pointpath,point=self.fixture(Path(tmp))
            record=point["audit_state_files"]["full-orbital-gradient"]
            point["audit_state_files"]["full-orbital-gradient"]=write_npy(Path(record["path"]),[-5e-10])
            # The archive's new byte hash is real, while its claimed computed payload is stale.
            with self.assertRaises(ValueError):M.verify_certified_point(point)


@unittest.skipUnless(importlib.util.find_spec("numpy"),"Metadata-only NumPy algebra unavailable")
class ArrayMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy
        cls.np=numpy

    def payload(self):
        np=self.np
        molecule={"_atm":[[0,1]],"_bas":[[0,1]],"_env":[0.,1.,2.],"charge":0,"spin":0,"basis":"def2-SVP"}
        return {"molecule":json.dumps(molecule),"mo_coeff":np.eye(2),"mo_energy":np.array([-1.,1.]),
            "mo_occ":np.array([2.,0.]),"energy_hartree":-1.},molecule

    def test_checkpoint_is_full_space_copy_and_initial_guess_only_without_cached_J_claim(self):
        payload,molecule=self.payload();before=copy.deepcopy(payload)
        arrays,metadata=H.validate_checkpoint_restart_payload(payload,molecule,np=self.np,atoms=1,basis_functions=2,electrons=2)
        self.assertTrue(metadata["geometry_basis_payload_exact"])
        self.assertEqual(metadata["scope"],"INITIAL_GUESS_ONLY_NOT_CONVERGENCE_CERTIFICATION")
        self.assertFalse(metadata["historical_cached_J_VEFF_Fock_available"])
        self.assertTrue(self.np.array_equal(arrays["density"],self.np.diag([2.,0.])))
        for name in ("mo_coeff","mo_energy","mo_occ"):
            self.assertFalse(self.np.shares_memory(arrays[name],payload[name]))
            self.assertTrue(self.np.array_equal(payload[name],before[name]))
        self.assertEqual(payload["molecule"],before["molecule"])

    def test_real_Mole_dump_repr_attributes_decode_without_changing_plain_source(self):
        payload,molecule=self.payload();stored=json.loads(payload["molecule"])
        stored.update(basis="'def2-svp'",ecp="{}",_ecp="{}")
        payload["molecule"]=json.dumps(stored).encode()
        before=copy.deepcopy(molecule)
        arrays,metadata=H.validate_checkpoint_restart_payload(payload,molecule,np=self.np,
            atoms=1,basis_functions=2,electrons=2)
        self.assertEqual(arrays["mo_coeff"].shape,(2,2));self.assertEqual(molecule,before)
        self.assertEqual(metadata["scope"],"INITIAL_GUESS_ONLY_NOT_CONVERGENCE_CERTIFICATION")

    def test_checkpoint_geometry_space_occupation_complex_and_method_forgery_rejected(self):
        np=self.np
        for change in ("coordinates","atoms","charge","spin","basis","ecp","truncated","fractional","complex","nan","energy"):
            with self.subTest(change=change):
                payload,molecule=self.payload();stored=json.loads(payload["molecule"])
                if change=="coordinates":stored["_env"][1]+=1e-12
                elif change=="atoms":stored["_atm"].append([1,2])
                elif change=="charge":stored["charge"]=1
                elif change=="spin":stored["spin"]=1
                elif change=="basis":stored["basis"]="def2-TZVP"
                elif change=="ecp":stored["ecp"]={"X":"fake"}
                elif change=="truncated":payload["mo_coeff"]=np.ones((2,1))
                elif change=="fractional":payload["mo_occ"]=np.array([1.,1.])
                elif change=="complex":payload["mo_coeff"]=np.eye(2,dtype=complex)
                elif change=="nan":payload["mo_energy"][0]=float("nan")
                else:payload["energy_hartree"]=float("inf")
                payload["molecule"]=json.dumps(stored)
                with self.assertRaises(ValueError):
                    H.validate_checkpoint_restart_payload(payload,molecule,np=np,atoms=1,basis_functions=2,electrons=2)

    def test_forged_certification_cannot_replace_complete_AO_space_with_small_checkpoint_arrays(self):
        np=self.np
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(M,"ROOT",Path(tmp)):
            fixture=DependencyTests();args,guard,path,value,pointpath,point=fixture.fixture(Path(tmp))
            molecule={"_atm":[[0,1]],"_bas":[[0,1]],"_env":[0.,1.,2.]}
            data={"scf/mo_coeff":np.eye(2),"scf/mo_occ":np.array([2.,0.]),"scf/mo_energy":np.array([-1.,1.]),
                "mol":json.dumps(molecule).encode()}
            class Dataset:
                def __init__(self,value):self.value=value
                def __getitem__(self,key):return self.value
            class File:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def __getitem__(self,key):return Dataset(data[key])
            source=SimpleNamespace(mol=SimpleNamespace(**{key:np.asarray(value) for key,value in molecule.items()}),
                make_rdm1=lambda c,occ:(c*occ)@c.T,
                get_ovlp=mock.Mock(side_effect=AssertionError("No overlap evaluation permitted")),
                get_grad=mock.Mock(side_effect=AssertionError("No Fock/orbital evaluation permitted")))
            fake_h5py=SimpleNamespace(File=lambda *args,**kwargs:File())
            with mock.patch.dict(sys.modules,{"h5py":fake_h5py}),self.assertRaisesRegex(ValueError,"fullAO"):
                M.replay_zero_arrays(point,source,np=np)
            source.get_ovlp.assert_not_called();source.get_grad.assert_not_called()

    def test_same_density_probe_builds_new_reference_and_both_current_operators(self):
        np=self.np;calls=[];focks=[]
        class Tagged(np.ndarray):pass
        dm=np.diag([2.,0.]);reference_dm=np.diag([1.5,.5])
        def native(mf,mol,density,dm_last=None,vhf_last=None):
            value=np.asarray(2*density+(np.eye(2)*1e-8 if dm_last is not None else 0)).view(Tagged)
            value.vj=np.asarray(3*density+(np.eye(2)*2e-8 if dm_last is not None else 0))
            calls.append((density,dm_last,vhf_last,value));return value
        mf=SimpleNamespace(mol=object(),get_hcore=lambda mol:np.eye(2),get_ovlp=lambda mol:np.eye(2))
        def fock(**kwargs):focks.append(kwargs);return kwargs["h1e"]+kwargs["vhf"]
        mf.get_fock=fock;mf.energy_tot=lambda density,hcore,vhf:float(np.einsum("ij,ji->",density,hcore+vhf))
        before=dm.copy();result=H.compare_same_density_veff(mf,dm,reference_dm,native_get_veff=native,np=np)
        self.assertEqual(len(calls),3);self.assertIs(calls[0][0],reference_dm)
        self.assertIsNone(calls[0][1]);self.assertIsNone(calls[0][2])
        self.assertIs(calls[1][0],dm);self.assertIs(calls[1][1],reference_dm);self.assertIs(calls[1][2],calls[0][3])
        self.assertIs(calls[2][0],dm);self.assertIsNone(calls[2][1]);self.assertIsNone(calls[2][2])
        for kwargs in focks:
            self.assertEqual(kwargs["cycle"],-1);self.assertIsNone(kwargs["diis"])
            self.assertEqual(kwargs["level_shift_factor"],0);self.assertEqual(kwargs["damp_factor"],0)
        self.assertEqual(result["reference_origin"],"NEW_IN_PROCESS_FULL_REFERENCE_NOT_HISTORICAL_CACHED_OPERATOR")
        self.assertAlmostEqual(result["VEFF_max_abs_difference_hartree"],1e-8)
        self.assertAlmostEqual(result["J_max_abs_difference_hartree"],2e-8)
        self.assertTrue(np.array_equal(dm,before))


if __name__=="__main__":unittest.main()
