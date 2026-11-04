"""Source-contract checks only: no imports of quantum/numerical code or arrays."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def dict_literal_keys(node):
    return {key.value for value in ast.walk(node) if isinstance(value, ast.Dict)
            for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}


class ThreadContractTests(unittest.TestCase):
    def test_frozen_RHF_and_snapshot_environment_keys_do_not_bind_thread_count(self):
        core = function(ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py", "runtime_state_fingerprint")
        environment = function(ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py", "environment_identity")
        for node in (core, environment):
            self.assertFalse(any("thread" in key.lower() for key in dict_literal_keys(node)))
            self.assertFalse(any(isinstance(n, ast.Attribute) and "thread" in n.attr.lower() for n in ast.walk(node)))
        # Source identity still pins the numerical binaries; thread compatibility
        # does not relax source, MO, geometry, quadrature or screening identity.
        self.assertTrue({"MO_coeff", "MO_energy", "MO_occ", "grid", "settings"} <= dict_literal_keys(core))
        self.assertIn("sources", dict_literal_keys(environment))

    def test_new_stage_physical_key_binds_selected_threads_and_configures_runtime_before_restore(self):
        execute = function(ROOT / "scripts/pyscf_staged_hessian_worker_20261105_v6b.py", "execute")
        physical = next(n.value for n in ast.walk(execute) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "physical" for t in n.targets))
        values = {key.value: value for key, value in zip(physical.keys, physical.values) if isinstance(key, ast.Constant)}
        selected = values["native_threads"]
        self.assertTrue(isinstance(selected, ast.Attribute) and isinstance(selected.value, ast.Name)
                        and selected.value.id == "args" and selected.attr == "threads")
        calls = [n for n in ast.walk(execute) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        configured = next(n for n in calls if n.func.attr == "num_threads")
        restored = next(n for n in calls if n.func.attr == "restore_snapshot")
        self.assertLess(configured.lineno, restored.lineno)
        self.assertEqual(ast.dump(configured.args[0]), ast.dump(selected))
        gate = next(n for n in calls if n.func.attr == "verify_resource_supervision")
        self.assertEqual(ast.dump(next(k.value for k in gate.keywords if k.arg == "expected_threads")), ast.dump(selected))


if __name__ == "__main__":
    unittest.main()
