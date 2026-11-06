"""Recovery compatibility tests; no SCF, gradients or other quantum kernels run."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parent
ORIGINAL = SCRIPTS / "research4_b_diagnostic_worker_20261106.py"
RECOVERY = SCRIPTS / "research4_b_diagnostic_worker_recovery_20261107.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load(RECOVERY, "diagnostic_recovery_worker_tests")
# Reuse the original fourteen scientific/provenance tests against the recovery
# implementation, without modifying the original frozen test or worker.
original_tests = load(SCRIPTS / "test_research4_b_diagnostic_worker_20261106.py", "diagnostic_original_test_cases")
original_tests.worker = worker


class RecoveryMetadataTests(unittest.TestCase):
    def setUp(self):
        self.old_get = platform._Processor.__dict__["get"]
        self.old_cache = platform._uname_cache

    def tearDown(self):
        platform._Processor.get = self.old_get
        platform._uname_cache = self.old_cache

    def test_processor_resolution_survives_cache_invalidation_without_subprocess(self):
        with patch.object(subprocess, "check_output", side_effect=AssertionError("subprocess forbidden")) as check_output:
            record = worker.initialize_platform_processor_without_fork()
            platform._uname_cache = None
            self.assertEqual(platform.processor(), record["processor_value"])
        check_output.assert_not_called()
        self.assertEqual(record["uname_machine"], os.uname().machine)
        self.assertEqual(record["stdlib_platform_sha256"], worker.sha(platform.__file__))
        self.assertFalse(record["subprocess_used"])
        self.assertIn(record["processor_value_source"], (
            "/proc/cpuinfo:first model name", "os.uname().machine:architecture fallback, not a CPU model"))

    def test_reported_processor_value_is_an_observed_local_value(self):
        record = worker.initialize_platform_processor_without_fork()
        if record["processor_is_cpu_model"]:
            model = next(line.partition(":")[2].strip()
                         for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.partition(":")[0].strip().lower() == "model name")
            self.assertEqual(record["processor_value"], model)
        else:
            self.assertEqual(record["processor_value"], os.uname().machine)

    def test_real_cold_numerical_imports_cannot_start_a_subprocess(self):
        # This subprocess is the test harness. Inside it, every subprocess API
        # is blocked before platform initialization and cold numerical imports.
        code = """
import importlib.util,json,platform,subprocess,sys
from pathlib import Path
from unittest.mock import patch
path=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('cold_recovery_import_test',path)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
calls=[]
def forbidden(*args,**kwargs):
    calls.append(str(args[:1]));raise AssertionError('NO SUBPROCESS IS PERMITTED')
with patch.object(subprocess,'Popen',forbidden), patch.object(subprocess,'check_output',forbidden), patch.object(subprocess,'run',forbidden), patch.object(subprocess,'call',forbidden):
    record=worker.initialize_platform_processor_without_fork()
    platform._uname_cache=None
    import h5py,numpy,pyscf
    platform._uname_cache=None
    assert platform.processor()==record['processor_value']
    assert not calls,calls
    print(json.dumps({'subprocess_calls':calls,'processor_source':record['processor_value_source'],'h5py':h5py.__version__,'numpy':numpy.__version__,'pyscf':pyscf.__version__}))
"""
        result = subprocess.run([sys.executable, "-c", code, str(RECOVERY)],
                                text=True, capture_output=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"subprocess_calls": []', result.stdout)

    def test_failed_proof_never_initializes_platform_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text(json.dumps({"schema": worker.REQUEST_SCHEMA, "operation": "aggregate"}))
            args = argparse.Namespace(request=request, request_sha256=worker.sha(request),
                                      output=root / "reports" / "attempt", execute=True,
                                      point_result=[], point_result_sha256=[])
            with patch.object(worker, "ROOT", root), \
                    patch.object(worker, "supervise", side_effect=ValueError("no live proof")), \
                    patch.object(worker, "initialize_platform_processor_without_fork", side_effect=AssertionError("too early")) as initialize:
                with self.assertRaisesRegex(ValueError, "no live proof"):
                    worker.run(args)
            initialize.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_scientific_and_provenance_functions_have_identical_ast(self):
        before = ast.parse(ORIGINAL.read_text())
        after = ast.parse(RECOVERY.read_text())
        original_functions = {node.name: node for node in before.body if isinstance(node, ast.FunctionDef)}
        recovery_functions = {node.name: node for node in after.body if isinstance(node, ast.FunctionDef)}
        for name, node in original_functions.items():
            if name == "run":
                continue
            self.assertEqual(ast.dump(node, include_attributes=False),
                             ast.dump(recovery_functions[name], include_attributes=False), name)
        added = set(recovery_functions) - set(original_functions)
        self.assertEqual(added, {"initialize_platform_processor_without_fork"})

    def test_run_only_adds_postproof_metadata_initialization_and_record(self):
        before = next(node for node in ast.parse(ORIGINAL.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "run")
        after = next(node for node in ast.parse(RECOVERY.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name == "run")
        class RemoveOnlyCompatibilityMetadata(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id == "platform_processor_metadata"
                       for target in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_Dict(self, node):
                keep = [i for i, key in enumerate(node.keys)
                        if not (isinstance(key, ast.Constant) and key.value == "platform_processor_nofork_setup")]
                node.keys = [node.keys[i] for i in keep]
                node.values = [node.values[i] for i in keep]
                return self.generic_visit(node)
        reduced = RemoveOnlyCompatibilityMetadata().visit(after)
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(reduced, include_attributes=False))


def suite():
    result = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    result.addTests(loader.loadTestsFromTestCase(original_tests.NumericalTests))
    result.addTests(loader.loadTestsFromTestCase(original_tests.ProvenanceTests))
    result.addTests(loader.loadTestsFromTestCase(RecoveryMetadataTests))
    return result


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    raise SystemExit(0 if runner.run(suite()).wasSuccessful() else 1)
