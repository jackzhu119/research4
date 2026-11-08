"""Saved-response audit safety tests; no NumPy, PySCF, arrays or Q calls."""
import ast
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_pyscf_saved_cphf_residual_20261105.py"
SPEC = importlib.util.spec_from_file_location("saved_cphf_audit_stdlib", SCRIPT)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def closed_state(status="PIPELINE_HEALTHY_CLOSED"):
    return {"invocations": [{"status": status}], "stages": {"response-00": {"status": "HEALTHY_CLOSED"}},
        "analysis": {"status": "HEALTHY_CLOSED"}}


class ResidualAuditSafety(unittest.TestCase):
    def test_import_is_stdlib_and_frozen_sources_are_real(self):
        self.assertEqual(M.sha(M.GUARD), M.GUARD_SHA)
        self.assertEqual(M.sha(M.CONSUMER), M.CONSUMER_SHA)
        self.assertFalse(any(k.split(".")[0] in {"numpy", "scipy", "pyscf", "torch"} for k in sys.modules))

    def test_one_atom_and_whole_completed_block_selection_are_exclusive(self):
        self.assertEqual(M.select_atoms(list(range(8)), 3, False), [3])
        self.assertEqual(M.select_atoms([72,73,74], None, True), [72,73,74])
        for ids, index, whole in (([],0,False),([1,1],1,False),([1],0,False),([1],None,False),([1],1,True),([1],True,False)):
            with self.subTest(ids=ids,index=index,whole=whole), self.assertRaises(ValueError):
                M.select_atoms(ids,index,whole)

    def test_active_unknown_or_unresolved_main_state_is_rejected(self):
        for status in ("PIPELINE_HEALTHY_CLOSED", "BOUNDARY_CLOSED", "BLOCKED_REQUIRES_ROOT_REVIEW_NO_AUTOMATIC_RETRY"):
            M.validate_main_closed_state(closed_state(status))
        for status in ("RUNNING", "UNKNOWN", None):
            with self.assertRaises(ValueError): M.validate_main_closed_state(closed_state(status))
        for stage_status in ("RUNNING", "STARTING", "UNKNOWN", None):
            value=closed_state();value["stages"]["response-00"]["status"]=stage_status
            with self.assertRaises(ValueError): M.validate_main_closed_state(value)
        for change in ({"analysis":{"status":"RUNNING"}}, {"analysis":{"status":"UNKNOWN"}},
                       {"analysis":{"status":"BLOCKED_REQUIRES_ROOT_REVIEW","child_unresolved":True}}, {"stages":{}}):
            with self.assertRaises(ValueError): M.validate_main_closed_state({**closed_state(),**change})
        value=closed_state();value["stages"]["response-00"]["guardian_unresolved"]=True
        with self.assertRaises(ValueError): M.validate_main_closed_state(value)

    def test_only_current_audit_phase_may_be_running(self):
        proof={"ledger_path":"/synthetic/ledger.json","phase":"audit-only"}
        row={"phase":"audit-only","status":"RUNNING","command":["python",str(SCRIPT)]}
        ledger={"phases":[row]}
        guard=SimpleNamespace(read=lambda p:ledger,root_path=lambda p:Path(p).resolve(),validate_ledger=Mock())
        self.assertIs(M.sole_audit_phase(guard,{"proof":proof}),ledger)
        guard.validate_ledger.assert_called_once_with(ledger,allow_running=True)
        for rows in ([],[{**row,"phase":"other"}],[row,{**row,"phase":"other"}],
                     [{**row,"command":["python","/synthetic/not-this-worker.py"]}]):
            ledger["phases"]=rows
            with self.assertRaises(ValueError): M.sole_audit_phase(guard,{"proof":proof})

    def test_main_queue_lock_is_real_and_never_stops_its_owner(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/".runtime").mkdir();lock=root/".runtime/research4-materialized-quantum-queue.lock";lock.touch()
            path=root/"main.json";path.write_text(json.dumps(closed_state()));digest=M.sha(path)
            def checked(ref):
                self.assertEqual(M.sha(Path(ref["path"])),ref["sha256"]);return Path(ref["path"])
            consumer=SimpleNamespace(checked=checked,read=lambda p:json.loads(Path(p).read_text()))
            args=SimpleNamespace(main_pipeline_state=path,main_pipeline_state_sha256=digest)
            with patch.object(M,"ROOT",root):
                with lock.open("a") as owner:
                    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        with M.closed_main_queue(args,consumer):self.fail("Cannot enter live main queue")
                with M.closed_main_queue(args,consumer) as ref:self.assertEqual(ref["sha256"],digest)
                args.main_pipeline_state_sha256="0"*64
                with self.assertRaises(AssertionError):
                    with M.closed_main_queue(args,consumer):pass

    def test_preflight_is_NOT_RUN_and_never_qualifies_or_reads_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"prep"
            with patch.object(M,"metadata_gate",side_effect=AssertionError("No target inputs")):
                value=M.preflight(out)
            self.assertEqual(value["status"],"NOT_RUN")
            self.assertEqual(value["quantum_operator_evaluations"],0)
            self.assertEqual(value["actual_residuals"],"NOT_RUN")
            self.assertFalse(value["target_arrays_read"])
            self.assertFalse(value["numerical_packages_imported"])
            self.assertEqual([p.name for p in out.iterdir()],["preflight.json"])
            with self.assertRaises(ValueError):M.preflight(out)

    def test_numeric_imports_follow_actual_metadata_gate_only(self):
        tree=ast.parse(SCRIPT.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend((a.name.split(".")[0],node.lineno) for a in node.names)
            if isinstance(node,ast.ImportFrom):names.append(((node.module or "").split(".")[0],node.lineno))
        numeric=[line for name,line in names if name in {"numpy","pyscf","scipy","torch"}]
        fn=functions["execute_closed"];gate=next(n.lineno for n in fn.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=="gate" for t in n.targets))
        self.assertTrue(numeric)
        self.assertTrue(all(fn.lineno<line<=fn.end_lineno and line>gate for line in numeric))
        self.assertIsNone(M.__dict__.get("np"))

    def test_physical_binding_fields_match_current_frozen_consumer(self):
        def fields(path, function):
            fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==function)
            assignment=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="physical" for t in n.targets))
            self.assertIsInstance(assignment.value,ast.Dict)
            return [ast.literal_eval(k) for k in assignment.value.keys]
        expected=fields(M.CONSUMER,"execute")
        self.assertEqual(fields(SCRIPT,"execute_closed"),expected)
        self.assertIn("native_krylov_lindep",expected)

    def test_entrypoint_has_no_solver_SCF_gradient_or_Hessian_computation_call(self):
        tree=ast.parse(SCRIPT.read_text());forbidden={"kernel","solve_mo1","hess_elec","make_h1","Hessian_kernel","nuc_grad_method"}
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&forbidden)
        self.assertEqual(calls.count("gen_vind"),1)
        # Exactly one fresh operator action exists; it is fed U reconstructed
        # from saved coefficients, rather than an inferred potential.
        fx_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="fx"]
        self.assertEqual(len(fx_calls),1)
        self.assertEqual(ast.dump(fx_calls[0].args[0]),ast.dump(ast.Name(id="U",ctx=ast.Load())))

    def test_execution_refuses_without_real_guard_before_any_main_or_numeric_gate(self):
        args=SimpleNamespace(threads=2)
        fake=SimpleNamespace(verify_resource_supervision=Mock(side_effect=ValueError("No real guardian")))
        with patch.object(M,"load",side_effect=[fake,object()]),patch.object(M,"metadata_gate") as metadata:
            with self.assertRaisesRegex(ValueError,"real guardian"):M.execute(args)
            metadata.assert_not_called()


if __name__=="__main__":unittest.main()
