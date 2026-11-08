"""Safety/convergence tests without importing numerical packages or running Q."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research4_cloud_orbital_stability_20261106.py"
spec = importlib.util.spec_from_file_location("cloud_orbital_stability_stdlib", SCRIPT)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class NativeInternalStabilitySafety(unittest.TestCase):
    def test_import_and_preflight_never_import_numeric_packages_or_consume_target(self):
        self.assertFalse(any(k.split(".")[0] in {"numpy", "pyscf", "scipy", "torch"} for k in sys.modules))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "prepared"
            value = M.preflight(out)
            self.assertEqual(value["status"], "NOT_RUN")
            self.assertEqual(value["quantum_operator_evaluations"], 0)
            self.assertFalse(value["target_arrays_read"])
            self.assertFalse(value["numerical_packages_imported"])
            self.assertFalse(any(k.split(".")[0] in {"numpy", "pyscf", "scipy", "torch"} for k in sys.modules))
            with self.assertRaises(ValueError): M.preflight(out)

    def test_frozen_native_sources_and_dependencies_are_real(self):
        audit = M.load_dependencies()
        self.assertEqual(audit.sha(M.GUARD), M.GUARD_SHA)
        self.assertEqual(audit.sha(M.CONSUMER), M.CONSUMER_SHA)
        for relative, expected in M.NATIVE_SOURCES.items():
            self.assertEqual(audit.sha(ROOT / ".runtime/quantum-upgrade-site/pyscf" / relative), expected)

    def test_solver_limits_reject_unbounded_policy_or_bool_integer(self):
        M.validate_solver_policy(2, 30, 3, 1e-4)
        for values in [(1,30,3,1e-4),(True,30,3,1e-4),(2,31,3,1e-4),(2,0,3,1e-4),
                       (2,True,3,1e-4),(2,30,1,1e-4),(2,30,True,1e-4),(2,30,3,1e-3)]:
            with self.subTest(values=values), self.assertRaises(ValueError): M.validate_solver_policy(*values)

    def test_convergence_not_stability_flag_qualifies_solver_completion(self):
        record = {"actual_returned_roots_Eh": [0.1,0.2,0.3],
                  "actual_returned_convergence_flags": [True,True,True], "actual_returned_root_count": 3}
        self.assertEqual(M.interpret_native_result(record, True)["status"], M.SUCCESS)
        for flags, roots, count in [([True,True,False],[.1,.2,.3],3),([True],[.1],1),
                                  ([True]*3,[.1,float("nan"),.3],3),([],[],0)]:
            partial = M.interpret_native_result({"actual_returned_roots_Eh":roots,
                "actual_returned_convergence_flags":flags,"actual_returned_root_count":count},True)
            self.assertEqual(partial["status"],"PARTIAL")
            self.assertFalse(partial["native_all_requested_roots_converged"])
        with self.assertRaises(ValueError):M.interpret_native_result(record, False)
        record["actual_returned_roots_Eh"] = [-.1,.2,.3]
        self.assertEqual(M.interpret_native_result(record,False)["internal_orbital_stability"],"UNSTABLE")

    def test_observer_uses_original_native_solver_and_restores_binding(self):
        seen=[]
        def native(aop, x0, precond, tol=1e-12,max_cycle=50,max_space=12,lindep=1e-14,
                   max_memory=4000,dot=None,callback=None,nroots=1,lessio=False,pick=None,
                   verbose=None,follow_state=True,tol_residual=None,fill_heff=None):
            seen.append((max_cycle,nroots,tol,max_memory))
            callback({"icyc":0,"e":[.1],"dx_norm":[.2],"de":[.1],"conv":[False]})
            return [True]*3,[.1,.2,.3],[[1],[2],[3]]
        module=SimpleNamespace(davidson1=native);record={};events=[]
        with M.capture_native_davidson(module,max_cycle=30,nroots=3,tol=1e-4,
                                      emit=lambda e,d:events.append((e,d)),record=record):
            result=module.davidson1(None,None,None,tol=1e-4,nroots=3)
            self.assertEqual(result[1],[.1,.2,.3])
            with self.assertRaises(ValueError):module.davidson1(None,None,None,tol=1e-4,nroots=3)
        self.assertIs(module.davidson1,native)
        self.assertEqual(seen,[(30,3,1e-4,2048)])
        self.assertEqual(record["actual_returned_convergence_flags"],[True]*3)
        self.assertEqual(record["effective_native_solver_policy"]["effective_native_residual_tolerance"],.01)
        self.assertIn("NOT_FINAL",record["intermediate_callbacks"][0]["scope"])
        self.assertIn("NOT_RETURNED",record["effective_native_solver_policy"]["final_residual_norms"])

    def test_observer_restores_original_even_if_native_solver_fails(self):
        def native(aop,x0,precond,tol=1e-12,nroots=1,callback=None,max_cycle=50,max_memory=4000,
                   max_space=12,tol_residual=None):raise RuntimeError("synthetic native failure")
        module=SimpleNamespace(davidson1=native)
        with self.assertRaisesRegex(RuntimeError,"synthetic"), M.capture_native_davidson(module,
                max_cycle=30,nroots=3,tol=1e-4,emit=lambda *x:None,record={}):
            module.davidson1(None,None,None,tol=1e-4,nroots=3)
        self.assertIs(module.davidson1,native)

    def test_guard_is_required_before_metadata_or_numerical_import(self):
        args=SimpleNamespace(threads=2,max_cycle=30,nroots=3,tol=1e-4)
        fake=SimpleNamespace(verify_resource_supervision=Mock(side_effect=ValueError("actual guardian missing")))
        audit=SimpleNamespace(load=Mock(side_effect=[fake,object()]),closed_main_queue=Mock(),metadata_gate=Mock())
        with patch.object(M,"load_dependencies",return_value=audit), self.assertRaisesRegex(ValueError,"guardian"):
            M.execute(args)
        audit.closed_main_queue.assert_not_called();audit.metadata_gate.assert_not_called()

    def test_only_current_bounded_diagnostic_may_be_active(self):
        proof={"ledger_path":"/fixture/ledger.json","phase":"internal-test"}
        row={"phase":"internal-test","status":"RUNNING","command":["python",str(SCRIPT)]}
        ledger={"phases":[row]};guard=SimpleNamespace(read=lambda p:ledger,
            root_path=lambda p:Path(p).resolve(),validate_ledger=Mock())
        self.assertIs(M.sole_diagnostic_phase(guard,{"proof":proof}),ledger)
        for rows in [[],[row,{**row,"phase":"other"}],[{**row,"command":["python","/other.py"]}]]:
            ledger["phases"]=rows
            with self.assertRaises(ValueError):M.sole_diagnostic_phase(guard,{"proof":proof})

    def test_actual_internal_native_entrypoint_and_no_unwanted_scientific_calls(self):
        tree=ast.parse(SCRIPT.read_text())
        calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        self.assertEqual(calls.count("rhf_stability"),1)
        self.assertFalse(set(calls)&{"kernel","solve_mo1","hess_elec","make_h1","nuc_grad_method","optimize"})
        numerical=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))
                   and ((getattr(node,"module","") or "").split(".")[0] in {"pyscf","numpy"}
                   or isinstance(node,ast.Import) and any(x.name in {"numpy","pyscf"} for x in node.names))]
        fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="execute_closed")
        self.assertTrue(numerical)
        self.assertTrue(all(fn.lineno<n.lineno<=fn.end_lineno for n in numerical))


if __name__ == "__main__":unittest.main()
