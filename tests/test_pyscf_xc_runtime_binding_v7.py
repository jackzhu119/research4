"""Stdlib fake-module tests: never import PySCF or evaluate any AO."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py"
SPEC = importlib.util.spec_from_file_location("v7_xc_runtime_binding_pure", PATH)
CORE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(CORE)


def fixture():
    modules = {name: types.ModuleType("pure_fake_" + name) for name in CORE.PINNED}
    for name, module in modules.items(): module.__file__ = str(CORE.PYSCF_ROOT / name)
    ni_module = modules["dft/numint.py"]
    exec(compile("def block_loop(self):\n    return None\n", ni_module.__file__, "exec"), ni_module.__dict__)
    ni_module.NumInt = type("FakeNumIntForNamespaceOnly", (), {"block_loop": ni_module.block_loop})
    ni_module.BLKSIZE = 56
    modules["dft/gen_grid.py"].BLKSIZE = 56
    modules["hessian/rks.py"].numint = ni_module
    modules["hessian/rks.py"].rks_grad = modules["grad/rks.py"]
    modules["grad/rks.py"].numint = ni_module
    return ni_module.NumInt(), {"numint": ni_module, "rks": modules["hessian/rks.py"],
        "grad_rks": modules["grad/rks.py"], "gen_grid": modules["dft/gen_grid.py"]}


class RuntimeBinding(unittest.TestCase):
    def test_same_pinned_paths_and_module_aliases_pass_without_any_numerics(self):
        ni, modules = fixture()
        result = CORE.validate_runtime_sources(ni, **modules)
        self.assertEqual(set(result), set(CORE.PINNED))
        self.assertFalse(any(name.split(".")[0] in {"numpy", "pyscf", "torch", "scipy"} for name in sys.modules))

    def test_another_installed_source_tree_is_rejected(self):
        for key in ("numint", "rks", "grad_rks", "gen_grid"):
            ni, modules = fixture(); modules[key].__file__ = "/untrusted/other-site/" + key + ".py"
            with self.subTest(key=key), self.assertRaises(ValueError): CORE.validate_runtime_sources(ni, **modules)

    def test_cross_site_alias_or_different_function_namespace_is_rejected(self):
        for key in ("rks_numint", "rks_grad", "grad_numint", "globals", "alignment"):
            ni, modules = fixture()
            if key == "rks_numint": modules["rks"].numint = types.ModuleType("other")
            if key == "rks_grad": modules["rks"].rks_grad = types.ModuleType("other")
            if key == "grad_numint": modules["grad_rks"].numint = types.ModuleType("other")
            if key == "globals":
                namespace = {}; exec(compile("def block_loop(self): pass\n", modules["numint"].__file__, "exec"), namespace)
                modules["numint"].NumInt.block_loop = namespace["block_loop"]
            if key == "alignment": modules["gen_grid"].BLKSIZE = 112
            with self.subTest(key=key), self.assertRaises(ValueError): CORE.validate_runtime_sources(ni, **modules)


if __name__ == "__main__": unittest.main()
