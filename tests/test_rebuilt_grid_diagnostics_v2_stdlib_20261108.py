"""No-Q rebuilt-quadrature checks with fake grids, SCFs and metadata only."""
import argparse
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/"scripts/research4_rebuilt_grid_diagnostics_20261108_v2.py"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


# The implementation imports no numerical package at module import.
M=load("rebuilt_grid_diagnostics_v2_test",WORKER)


class SourcePurityTests(unittest.TestCase):
    def test_no_numerical_package_import_at_module_scope(self):
        tree=ast.parse(WORKER.read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(name.startswith(("numpy","pyscf","h5py","scipy")) for name in names))


class FakeArray:
    def __init__(self,data):self.data=copy.deepcopy(data)
    @property
    def ndim(self):return 2 if self.data and isinstance(self.data[0],list) else 1
    @property
    def shape(self):return (len(self.data),len(self.data[0])) if self.ndim==2 else (len(self.data),)
    def __len__(self):return len(self.data)


class FakeNP:
    @staticmethod
    def isfinite(array):
        flat=[x for row in array.data for x in row] if array.ndim==2 else array.data
        return SimpleNamespace(all=lambda:all(math.isfinite(x) for x in flat))


class FakeGrids:
    def __init__(self,mol):
        self.mol=mol;self.level=0;self.cutoff=1e-15;self.atom_grid={};self.alignment=8;self.non0tab=None;self.build_calls=[]
    def build(self,**kwargs):
        self.build_calls.append(kwargs)
        self.coords=FakeArray([[self.mol.offset,1.,2.],[0.,0.,0.]])
        self.weights=FakeArray([2.,0.]);self.non0tab=FakeArray([[9]])
        return self


class FreshGridTests(unittest.TestCase):
    def fixture(self):
        attrs=("level","cutoff","atom_grid","alignment")
        source_mol=SimpleNamespace(offset=99.,stdout=None)
        moved_mol=SimpleNamespace(offset=.01,stdout=None)
        source_grids=SimpleNamespace(mol=source_mol,level=1,cutoff=1e-15,atom_grid={"O":[1,2]},alignment=8,
            verbose=0,coords=FakeArray([[999.,999.,999.]]),weights=FakeArray([999.]))
        original=SimpleNamespace(mol=source_mol,grids=source_grids,nlc="",_numint=SimpleNamespace(cutoff=1e-13,omega=None))
        mf=SimpleNamespace(mol=moved_mol,direct_scf=True,_eri=None,with_df=None,direct_scf_tol=1e-13,
            _numint=SimpleNamespace(cutoff=1.,omega=9),do_nlc=lambda:False)
        w=SimpleNamespace(GRID_ATTRIBUTES=attrs,grid_settings=lambda grids:{key:copy.deepcopy(getattr(grids,key)) for key in attrs})
        base=SimpleNamespace(array_fingerprint=lambda value:json.dumps(value.data,separators=(",",":")))
        return original,mf,w,base,SimpleNamespace(Grids=FakeGrids)

    def test_new_molecule_owned_grid_builds_actual_geometry_and_preserves_source(self):
        for level in (1,3):
            with self.subTest(level=level):
                original,mf,w,base,gen=self.fixture();before=copy.deepcopy(original.grids.coords.data)
                proof=M.attach_rebuilt_quadrature(mf,original,level,w,base,FakeNP,gen)
                self.assertIs(mf.grids.mol,mf.mol);self.assertIsNot(mf.grids,original.grids)
                self.assertEqual(mf.grids.build_calls,[{"with_non0tab":True}]);self.assertEqual(mf.grids.level,level)
                self.assertEqual(mf.grids.coords.data[0][0],.01);self.assertEqual(original.grids.coords.data,before)
                self.assertEqual(original.grids.level,1);self.assertIsNot(mf.grids.atom_grid,original.grids.atom_grid)
                self.assertEqual(mf.small_rho_cutoff,0);self.assertIsNone(mf.disp)
                self.assertEqual(mf._numint.cutoff,1e-13);self.assertIsNone(mf._numint.omega)
                self.assertFalse(proof["dense_grid_arrays_archived"])
                self.assertFalse(proof["original_fixed_grid_coordinates_or_weights_attached"])
                M.verify_rebuilt_quadrature(mf,proof,w,base);self.assertTrue(proof["unchanged_within_geometry"])

    def test_changed_grid_ownership_points_weights_or_level_rejects(self):
        for mutation in ("owner","coordinates","weights","level","pruning"):
            with self.subTest(mutation=mutation):
                original,mf,w,base,gen=self.fixture();proof=M.attach_rebuilt_quadrature(mf,original,1,w,base,FakeNP,gen)
                if mutation=="owner":mf.grids.mol=object()
                elif mutation=="coordinates":mf.grids.coords.data[0][0]+=1.
                elif mutation=="weights":mf.grids.weights.data[0]+=1.
                elif mutation=="level":mf.grids.level=3
                else:mf.small_rho_cutoff=1e-8
                with self.assertRaises(ValueError):M.verify_rebuilt_quadrature(mf,proof,w,base)

    def test_native_direct_cutoff_density_fitting_and_ERI_changes_fail(self):
        for mutation in ("direct","DF","ERI","direct-cutoff","numint"):
            with self.subTest(mutation=mutation):
                original,mf,w,base,gen=self.fixture()
                if mutation=="direct":mf.direct_scf=False
                elif mutation=="DF":mf.with_df=object()
                elif mutation=="ERI":mf._eri=object()
                elif mutation=="direct-cutoff":mf.direct_scf_tol=1e-8
                else:original._numint.cutoff=1e-8
                with self.assertRaises(ValueError):M.attach_rebuilt_quadrature(mf,original,1,w,base,FakeNP,gen)


class ControlRuleTests(unittest.TestCase):
    def energies(self,derivatives):
        return {step:(-1.+derivative*step,-1.-derivative*step) for step,derivative in derivatives.items()}
    def gradient(self,component=.3,total=0.):return [[component,2.,0.],[-component+total,-2.,0.],[0.,0.,0.]]

    def test_nonzero_atomic_forces_with_zero_net_force_validate_translation_balance(self):
        result=M.control_comparison(self.gradient(),self.energies({h:.3 for h in M.CONTROL_STEPS}))
        self.assertTrue(result["passed"]);self.assertEqual(result["sum_atomic_gradient_vector_norm_hartree_per_bohr"],0.)
        self.assertEqual(result["force_balance_scope"],"TRANSLATION_BALANCE_NOT_ATOMIC_STATIONARITY_OR_MINIMUM")
        self.assertGreater(max(abs(x) for row in self.gradient() for x in row),1e-6)

    def test_each_gradient_error_halving_error_total_force_and_trivial_component_can_reject(self):
        cases=[(self.gradient(),{.002:.3+2e-5,.001:.3}),
            (self.gradient(),{.002:.3+7e-6,.001:.3-7e-6}),
            (self.gradient(total=2e-6),{.002:.3,.001:.3}),
            (self.gradient(component=0.),{.002:0.,.001:0.})]
        for gradient,derivatives in cases:
            with self.subTest(gradient=gradient,derivatives=derivatives):
                self.assertFalse(M.control_comparison(gradient,self.energies(derivatives))["passed"])
        halving=M.control_comparison(self.gradient(),self.energies({.002:.3+7e-6,.001:.3-7e-6}))
        self.assertTrue(all(row["absolute_gradient_FD_error_hartree_per_bohr"]<1e-5 for row in halving["comparisons"]))
        self.assertGreater(halving["halving_derivative_abs_difference_hartree_per_bohr"],1e-5)

    def test_missing_or_nonfinite_true_measurements_never_become_zero_defaults(self):
        for bad in (None,True,float("nan"),float("inf")):
            values=self.energies({h:.3 for h in M.CONTROL_STEPS});values[.002]=(bad,-1.)
            with self.subTest(bad=bad),self.assertRaises(ValueError):M.control_comparison(self.gradient(),values)
        with self.assertRaises((KeyError,ValueError)):M.control_comparison(self.gradient(),{})
        gradient=self.gradient();gradient[0][0]=float("nan")
        with self.assertRaises(ValueError):M.control_comparison(gradient,self.energies({h:.3 for h in M.CONTROL_STEPS}))


class StageAndLiveProofTests(unittest.TestCase):
    def args(self,**changes):
        value=argparse.Namespace(stage="controls",threads=2,max_cycles=80,grid_level=1,with_displacements=False,
            source_zero_result=None,source_zero_result_sha256=None,grid_control_result=None,grid_control_result_sha256=None,
            mode_array_sha256=None,output=ROOT/"reports/metadata-only-rebuilt-test")
        for key,item in changes.items():setattr(value,key,item)
        return value

    def spino(self,**changes):
        return self.args(**{"stage":"spinosin","source_zero_result":Path("fixed-zero.json"),"source_zero_result_sha256":"a"*64,
            "grid_control_result":Path("controls.json"),"grid_control_result_sha256":"b"*64,**changes})

    def test_finite_control_grid_and_80cycle_envelope(self):
        M.validate_stage_args(self.args())
        for changes in ({"stage":"unknown"},{"threads":1},{"max_cycles":81},{"max_cycles":30},
                        {"grid_level":2},{"grid_level":3},{"with_displacements":True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):M.validate_stage_args(self.args(**changes))
        self.assertEqual(M.CONTROL_SHIFT_BOHR,.01);self.assertEqual(M.CONTROL_STEPS,(.002,.001))
        self.assertEqual(M.SPINOSIN_STEP,.005)

    def test_spinosin_requires_both_authentic_dependencies_and_only_grid1_or3(self):
        for level in (1,3):
            M.validate_stage_args(self.spino(grid_level=level))
            M.validate_stage_args(self.spino(grid_level=level,with_displacements=True,mode_array_sha256="c"*64))
        for changes in ({"source_zero_result":None},{"source_zero_result_sha256":None},{"grid_control_result":None},
                        {"grid_control_result_sha256":None},{"grid_level":2},{"max_cycles":100},
                        {"with_displacements":True},{"with_displacements":True,"mode_array_sha256":"c"*63}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):M.validate_stage_args(self.spino(**changes))

    def test_live_guard_must_bind_own_worker_exact_output_and_unique_running_phase(self):
        args=self.args();row={"phase":"actual","status":"RUNNING","command":["python",str(WORKER),"--output",str(args.output)]}
        proof={"proof":{"phase":"actual","ledger_path":"ledger.json"}}
        def guard(rows):return SimpleNamespace(read=lambda path:{"phases":rows},root_path=Path,validate_ledger=mock.Mock())
        actual=guard([row]);M.validate_live_phase(args,actual,proof)
        actual.validate_ledger.assert_called_once_with({"phases":[row]},allow_running=True)
        cases=[[],[row,{**row,"phase":"other"}],[{**row,"phase":"other"}],
            [{**row,"command":["python",str(M.FROZEN),"--output",str(args.output)]}],
            [{**row,"command":["python",str(WORKER),"--output",str(args.output.parent/"other")]}]]
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):M.validate_live_phase(args,guard(rows),proof)

    def test_frozen_numerical_source_and_compact_storage_policy_are_explicit(self):
        self.assertEqual(M.sha(M.FROZEN),M.FROZEN_SHA);self.assertEqual(M.sha(M.HELPERS),M.HELPERS_SHA)
        self.assertEqual(M.STORAGE_POLICY,"COMPACT_RESTART_AND_AUDIT_HASHES")
        tree=ast.parse(WORKER.read_text());body=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="compute_point")
        selections=[]
        for node in ast.walk(body):
            if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple):
                if [item.id for item in node.target.elts if isinstance(item,ast.Name)]==["key","filename"]:
                    selections.append(ast.literal_eval(node.iter))
        self.assertEqual(selections,[(('previous-density','previous-density.npy'),
            ('full-orbital-gradient','full-orbital-gradient.npy'),('MO_energy','refreshed-MO-energy.npy'))])
        self.assertNotIn('grid-coordinates.npy',WORKER.read_text());self.assertNotIn('grid-weights.npy',WORKER.read_text())


class RawControlReleaseTests(unittest.TestCase):
    def fixture(self,root):
        out=root/"reports/rebuilt-controls";out.mkdir(parents=True)
        value={"schema":M.SCHEMA,"source_worker":M.ref(WORKER),"status":M.CONTROL_SUCCESS,
            "stage":"controls","controls":{},"points":{}}
        def write(path,payload):path.write_text(json.dumps(payload)+"\n");return M.ref(path)
        for case,atoms in (("water",3),("ethanol",9)):
            gradient=[[.3,0.,0.],[-.3,0.,0.]]+[[0.,0.,0.] for _ in range(atoms-2)]
            energies={step:(-1.+.3*step,-1.-.3*step) for step in M.CONTROL_STEPS}
            names=[case+"-center"]+[case+"-h%.3f-%s"%(step,label) for step in M.CONTROL_STEPS for label in ("plus","minus")]
            for name in names:
                energy=-1.
                for step in M.CONTROL_STEPS:
                    for index,label in enumerate(("plus","minus")):
                        if name==case+"-h%.3f-%s"%(step,label):energy=energies[step][index]
                point={"schema":M.POINT_SCHEMA,"source_worker":M.ref(WORKER),"SCF_performed":True,
                    "energy_hartree":energy,"energy_completed":True,"SCF_endpoint_certified":True,
                    "rebuilt_quadrature":{"grid_level":1,"original_fixed_grid_coordinates_or_weights_attached":False}}
                pointpath=out/(name+".json");record=write(pointpath,point)
                value["points"][name]={"result":record,"energy_hartree":energy}
            value["controls"][case]={"true_gradient":write(out/(case+"-gradient.json"),gradient),
                "point_names":names,"comparison":M.control_comparison(gradient,energies)}
        path=out/"result.json";write(path,value)
        def checked(record):
            actual=Path(record["path"])
            M.require(actual.is_file() and actual.resolve().is_relative_to(root) and M.sha(actual)==record["sha256"],"Bad synthetic control artifact")
            return actual
        def checked_gradient(record,shape):
            array=json.loads(checked(record).read_text())
            M.require(len(array)==shape[0] and all(len(row)==shape[1] for row in array),"Bad synthetic raw gradient shape")
            return array
        # The frozen certification routine has its own separate compact-state suite.
        frozen=SimpleNamespace(checked=checked,read=lambda p:json.loads(Path(p).read_text()),
            old=lambda:SimpleNamespace(checked_control_array=checked_gradient),verify_certified_point=mock.Mock(return_value=True))
        guard=SimpleNamespace(verify_closed_phase=mock.Mock(return_value={"actual_closed_parent":True}))
        args=SimpleNamespace(grid_control_result=path,grid_control_result_sha256=M.sha(path))
        return path,value,args,frozen,guard

    def test_healthy_closed_own_producer_and_raw_energies_release_both_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,value,args,frozen,guard=self.fixture(Path(tmp))
            with mock.patch.object(M,"frozen_worker",return_value=frozen):result=M.grid_control_release(args,guard)
            self.assertTrue(result["raw_control_comparisons_replayed"])
            self.assertEqual(frozen.verify_certified_point.call_count,10)
            self.assertEqual(guard.verify_closed_phase.call_args.kwargs["expected_worker_path"],WORKER)
            guard.verify_closed_phase.side_effect=ValueError("No healthy closed actual producer")
            with mock.patch.object(M,"frozen_worker",return_value=frozen),self.assertRaises(ValueError):M.grid_control_release(args,guard)

    def test_duplicate_geometry_fabricated_energy_stale_raw_gradient_and_wrong_worker_reject(self):
        for mutation in ("duplicate","comparison-energy","child-energy","raw-gradient","wrong-worker"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                path,value,args,frozen,guard=self.fixture(Path(tmp))
                if mutation=="duplicate":value["controls"]["water"]["point_names"][1]=value["controls"]["water"]["point_names"][0]
                elif mutation=="comparison-energy":value["controls"]["water"]["comparison"]["comparisons"][0]["Eplus_hartree"]+=1e-3
                elif mutation=="wrong-worker":value["source_worker"]={"path":str(M.FROZEN),"sha256":M.FROZEN_SHA}
                elif mutation=="child-energy":
                    record=value["points"]["water-h0.002-plus"]["result"];pointpath=Path(record["path"])
                    point=json.loads(pointpath.read_text());point["energy_hartree"]+=1e-3
                    pointpath.write_text(json.dumps(point)+"\n");value["points"]["water-h0.002-plus"]["result"]=M.ref(pointpath)
                else:
                    raw=Path(value["controls"]["water"]["true_gradient"]["path"])
                    raw.write_text("[]\n")
                path.write_text(json.dumps(value)+"\n");args.grid_control_result_sha256=M.sha(path)
                with mock.patch.object(M,"frozen_worker",return_value=frozen),self.assertRaises(ValueError):M.grid_control_release(args,guard)


class BoundedPolishingTests(unittest.TestCase):
    def fixture(self,qualifications):
        pair={};checks=[];events=[];state={"maximum_cycles":80,"SCF_kernel_calls":0}
        np=SimpleNamespace(array=lambda value,copy=True:__import__('copy').deepcopy(value),array_equal=lambda a,b:a==b)
        mf=SimpleNamespace(diis=True,calls=[],current=None,e_tot=-1.,converged=False)
        def kernel(dm0):
            self.assertEqual(pair,{})
            call=len(mf.calls)+1;mf.calls.append(copy.deepcopy(dm0))
            self.assertEqual(state["SCF_kernel_calls"],call)
            self.assertEqual(mf.conv_tol_grad,1e-10);self.assertEqual(mf.conv_tol,1e-12);self.assertEqual(mf.max_cycle,80)
            mf.current=[[call+.25]];mf.e_tot=-1.-call*.001;mf.converged=qualifications[call-1]
            main={"phase":"SCF_MAIN_CYCLE","kernel_call_1based":call,"accepted":True}
            final={"phase":"FINAL_EXTRA_CYCLE","kernel_call_1based":call,"accepted":qualifications[call-1]}
            checks.extend([main,final]);pair.update(previous=copy.deepcopy(dm0),current=copy.deepcopy(mf.current),
                check=final,main_check=main)
        mf.kernel=kernel;mf.make_rdm1=lambda:copy.deepcopy(mf.current)
        def auditor(obj,actual_pair,actual_checks):
            self.assertIs(obj,mf);self.assertIs(actual_pair,pair);self.assertIs(actual_checks,checks)
            return {"qualified":qualifications[len(mf.calls)-1]}, {"current-density":copy.deepcopy(mf.current)}
        def fingerprint(value):return json.dumps(value)
        kwargs={"np":np,"pair":pair,"checks":checks,"emit":lambda event,detail:events.append((event,copy.deepcopy(detail))),
            "check_budget":mock.Mock(),"state":state,"fingerprint":fingerprint,"auditor":auditor}
        return mf,kwargs,state,events

    def test_first_qualified_endpoint_does_not_start_second_kernel(self):
        mf,kwargs,state,events=self.fixture([True])
        audit,arrays=M.bounded_kernel_sequence(mf,[[99.]],**kwargs)
        self.assertTrue(audit["qualified"]);self.assertEqual(mf.calls,[[[99.]]])
        self.assertEqual(state["SCF_kernel_calls"],1);self.assertEqual(len(state["SCF_polishing_attempts"]),1)
        attempt=state["SCF_polishing_attempts"][0]
        self.assertEqual(attempt["status"],"QUALIFIED");self.assertIn("ended_at_utc",attempt)
        self.assertGreaterEqual(attempt["actual_CPU_seconds"],0.)

    def test_second_kernel_uses_immediately_preceding_actual_final_density_and_records_both(self):
        mf,kwargs,state,events=self.fixture([False,True])
        audit,arrays=M.bounded_kernel_sequence(mf,[[99.]],**kwargs)
        self.assertTrue(audit["qualified"]);self.assertEqual(mf.calls,[[[99.]],[[1.25]]])
        attempts=state["SCF_polishing_attempts"];self.assertEqual(len(attempts),2)
        self.assertEqual([a["status"] for a in attempts],["UNQUALIFIED","QUALIFIED"])
        self.assertEqual([a["kernel_call_1based"] for a in attempts],[1,2])
        self.assertEqual(attempts[0]["actual_final_density_fingerprint"],attempts[1]["start_density_fingerprint"])
        self.assertEqual(attempts[1]["initial_guess_policy"],"IMMEDIATELY_PRECEDING_ACTUAL_FINAL_DENSITY")
        self.assertFalse(attempts[0]["native_kernel_converged"]);self.assertTrue(attempts[1]["native_kernel_converged"])
        self.assertEqual(attempts[1]["main_check"]["kernel_call_1based"],2)
        self.assertEqual(attempts[1]["final_check"]["kernel_call_1based"],2)
        self.assertTrue(all("started_at_utc" in a and "ended_at_utc" in a for a in attempts))

    def test_two_unqualified_kernels_never_start_a_third(self):
        mf,kwargs,state,events=self.fixture([False,False])
        audit,arrays=M.bounded_kernel_sequence(mf,[[99.]],**kwargs)
        self.assertFalse(audit["qualified"]);self.assertEqual(len(mf.calls),2)
        self.assertEqual(state["SCF_kernel_calls"],2)
        self.assertEqual([a["status"] for a in state["SCF_polishing_attempts"]],["UNQUALIFIED","UNQUALIFIED"])

    def test_exception_is_recorded_without_blind_second_call(self):
        mf,kwargs,state,events=self.fixture([False,True])
        def fail(dm0):raise RuntimeError("synthetic actual-kernel exception")
        mf.kernel=fail
        with self.assertRaisesRegex(RuntimeError,"synthetic"):
            M.bounded_kernel_sequence(mf,[[99.]],**kwargs)
        attempts=state["SCF_polishing_attempts"];self.assertEqual(len(attempts),1)
        self.assertEqual(attempts[0]["status"],"EXCEPTION");self.assertEqual(attempts[0]["error"]["type"],"RuntimeError")
        self.assertIn("ended_at_utc",attempts[0])

    def test_non_native_DIIS_and_larger_cycle_or_call_envelopes_are_rejected(self):
        for changed in ("diis","cycles","calls"):
            mf,kwargs,state,events=self.fixture([True])
            if changed=="diis":mf.diis=object()
            if changed=="cycles":state["maximum_cycles"]=81
            with mock.patch.object(M,"MAX_SCF_KERNEL_CALLS",3 if changed=="calls" else 2):
                with self.subTest(changed=changed),self.assertRaises(ValueError):
                    M.bounded_kernel_sequence(mf,[[99.]],**kwargs)

    def test_main_trigger_is_tighter_but_final_energy_and_gradient_bounds_are_unchanged(self):
        frozen=M.frozen_worker().old()
        def check(extra,gradient,drift):
            return frozen.polish_convergence_check({"conv_tol":1e-11 if extra else 1e-12,
                "conv_tol_grad":3e-10 if extra else 1e-10,"e_tot":0.,"last_hf_e":-drift,"norm_gorb":gradient},
                main_energy_tolerance=M.ENERGY_TOLERANCE,main_orbital_gradient_trigger=M.MAIN_GRADIENT_TRIGGER,
                final_energy_tolerance=M.ENERGY_TOLERANCE,final_orbital_gradient_tolerance=M.GRADIENT_TOLERANCE)
        self.assertLess(M.MAIN_GRADIENT_TRIGGER,M.GRADIENT_TOLERANCE)
        self.assertFalse(check(False,5e-10,1e-14)["accepted"])
        self.assertTrue(check(True,5e-10,1e-14)["accepted"])
        self.assertFalse(check(True,1.1e-9,1e-14)["accepted"])
        self.assertFalse(check(True,1e-10,1.1e-12)["accepted"])
        self.assertEqual(check(True,5e-10,1e-14)["phase"],"FINAL_EXTRA_CYCLE")
        self.assertEqual(check(True,5e-10,1e-14)["energy_tolerance_hartree"],1e-12)
        self.assertEqual(check(True,5e-10,1e-14)["orbital_gradient_tolerance"],1e-9)

    def test_old_grid_producer_cannot_be_relabelled_as_new_control_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,value,args,frozen,guard=RawControlReleaseTests().fixture(Path(tmp))
            value["schema"]="research4-rebuilt-moving-grid-diagnostics-v1"
            value["source_worker"]=M.ref(ROOT/"scripts/research4_rebuilt_grid_diagnostics_20261107.py")
            path.write_text(json.dumps(value));args.grid_control_result_sha256=M.sha(path)
            with mock.patch.object(M,"frozen_worker",return_value=frozen),self.assertRaisesRegex(ValueError,"producer"):
                M.grid_control_release(args,guard)


if __name__=="__main__":unittest.main()
