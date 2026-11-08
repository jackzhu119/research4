"""Policy/closure tests for finite-displacement worker; no quantum calls."""
import ast
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research4_cloud_followup_worker_20261106.py"
SPEC = importlib.util.spec_from_file_location("cloud_followup_worker_stdlib", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class CloudFollowupPolicyTests(unittest.TestCase):
    def test_import_does_not_import_numerics_or_quantum_and_sources_are_pinned(self):
        self.assertFalse(any(n.split('.')[0] in {'numpy','scipy','pyscf','torch'} for n in sys.modules))
        for path, digest in ((M.GUARD,M.GUARD_SHA),(M.ANALYZER,M.ANALYZER_SHA),(M.AUDITOR,M.AUDITOR_SHA)):
            self.assertEqual(M.sha(path),digest)

    def test_displacements_are_bounded_exclusive_and_correctly_indexed(self):
        M.validate_displacement(1,None,.01,1,75)
        M.validate_displacement(219,None,.005,-1,75)
        M.validate_displacement(None,8,.001,1,3)
        bad=[(None,None,.01,1,75),(1,0,.01,1,75),(0,None,.01,1,75),
             (220,None,.01,1,75),(None,9,.01,1,3),(None,-1,.01,1,3),
             (True,None,.01,1,75),(1,None,0.,1,75),(1,None,.1,1,75),
             (1,None,float('nan'),1,75),(1,None,float('inf'),1,75),(1,None,.01,0,75)]
        for values in bad:
            with self.subTest(values=values),self.assertRaises(ValueError):
                M.validate_displacement(*values)

    def test_guard_requires_unique_live_current_worker_command(self):
        proof={'ledger_path':'/fixture/ledger.json','phase':'point-plus'}
        row={'status':'RUNNING','phase':'point-plus','command':['python',str(SCRIPT)]}
        ledger={'phases':[row]}
        guard=SimpleNamespace(read=lambda p:ledger,root_path=lambda p:Path(p).resolve(),validate_ledger=Mock())
        M.sole_phase(guard,{'proof':proof})
        guard.validate_ledger.assert_called_once_with(ledger,allow_running=True)
        for rows in ([],[row,row],[{**row,'phase':'other'}],[{**row,'command':['python','/fixture/other.py']}]):
            ledger['phases']=rows
            with self.assertRaises(ValueError):M.sole_phase(guard,{'proof':proof})

    def test_closed_main_lock_cannot_be_stolen_and_does_not_signal_owner(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'.runtime').mkdir()
            lock=root/'.runtime/research4-materialized-quantum-queue.lock';lock.touch()
            main=root/'main.json';main.write_text(json.dumps({'closed':True}))
            args=SimpleNamespace(main_pipeline_state=main,main_pipeline_state_sha256=M.sha(main))
            validator=SimpleNamespace(validate_main_closed_state=Mock())
            with patch.object(M,'ROOT',root),patch.object(M,'load',return_value=validator):
                with lock.open('a') as owner:
                    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        with M.closed_main(args,None):self.fail('must not enter owned main queue')
                with M.closed_main(args,None):pass
            validator.validate_main_closed_state.assert_called_once_with({'closed':True})

    def test_new_reports_preserve_existing_outputs_and_checked_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reports').mkdir()
            with patch.object(M,'ROOT',root):
                out=M.fresh_output(root/'reports/new-point')
                M.save(out/'result.json',{'status':'NOT_RUN'},exclusive=True)
                record=M.ref(out/'result.json')
                self.assertEqual(M.checked(record),out/'result.json')
                with self.assertRaises(ValueError):M.fresh_output(out)
                with self.assertRaises(ValueError):M.fresh_output(root/'historical')
                with self.assertRaises(FileExistsError):M.save(out/'result.json',{},exclusive=True)
                (out/'result.json').write_text('{}')
                with self.assertRaises(ValueError):M.checked(record)

    def test_grid_limitation_is_explicit_and_no_global_Q_call_at_import(self):
        self.assertIn('REBUILT_ATOM_CENTERED_GRID',M.GRID_LIMITATION)
        self.assertIn('NOT_EXACT_COMMON_FIXED_QUADRATURE',M.GRID_LIMITATION)
        tree=ast.parse(SCRIPT.read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[x.name.split('.')[0] for x in node.names] if isinstance(node,ast.Import) else [(node.module or '').split('.')[0]]
                self.assertFalse({'numpy','pyscf','scipy','torch'}.intersection(names))
        fn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}['analyze']
        imported=[]
        for node in ast.walk(fn):
            if isinstance(node,ast.Import):imported.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):imported.append(node.module)
        self.assertEqual(imported,['numpy'])


if __name__=='__main__':
    unittest.main()
