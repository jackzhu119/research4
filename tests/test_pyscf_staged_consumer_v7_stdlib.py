"""Pure stdlib metadata/provider tests; fixtures are not quantum calculations."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/pyscf_staged_consumer_20261105_v7.py"
spec = importlib.util.spec_from_file_location("consumer_v7_stdlib_test", PATH)
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


class Providers:
    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, max_memory=4000, verbose=None):
        return ["native partial"]
    def make_h1(self, *args, **kwargs):
        return ["native H1"]
    def solve_mo1(self, *args, **kwargs):
        return ["native response"]
    def hess_nuc(self, mol=None, atmlst=None):
        return ["native nuclear"]


class ConsumerV7Stdlib(unittest.TestCase):
    def binding_fixture(self, td):
        directory = Path(td); (directory / 'hessian').mkdir()
        path = directory / 'hessian/rhf.py'
        text = 'def hess_elec(): pass\ndef solve_mo1(): pass\ndef hess_nuc(): pass\ndef gen_vind(): pass\nclass HessianBase:\n def solve_mo1(self): pass\n'
        path.write_text(text); rhf = types.ModuleType('synthetic_rhf_fixture'); rhf.__file__ = str(path)
        exec(compile(text, str(path), 'exec'), rhf.__dict__)
        rhf.cphf = types.SimpleNamespace()
        cls = type('SyntheticNumIntFixture', (), {})
        return dict(ni=cls(), xc_core=types.SimpleNamespace(PYSCF_ROOT=directory,
            validate_runtime_sources=lambda *a, **kw: {}),
            rhf_core=types.SimpleNamespace(PINNED={'hessian/rhf.py':w.sha(path)}), rhf=rhf,
            rks=types.SimpleNamespace(rhf_hess=rhf), numint=types.SimpleNamespace(NumInt=cls),
            grad_rks=None, gen_grid=None, linalg_helper=types.SimpleNamespace(__file__=str(directory/'WRONG.py')),
            mf=None, dft_rks=None, hf=None, grad_rhf=None, response=None, cphf=None, lib=None, vhf=None)

    def test_no_numerical_top_level_import(self):
        import ast
        tree = ast.parse(PATH.read_text())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        roots = [a.name.split('.')[0] for n in imports for a in n.names]
        self.assertFalse(set(roots) & {"numpy", "pyscf", "scipy", "h5py", "torch"})

    def test_old_sources_unchanged(self):
        self.assertEqual(w.sha(w.GUARD), w.GUARD_SHA)
        self.assertEqual(w.sha(w.HELPER), w.HELPER_SHA)

    def test_exact_response_partitions(self):
        for n in (3, 9, 75):
            chunks = w.response_blocks(n)
            self.assertEqual([i for c in chunks for i in c], list(range(n)))
            self.assertTrue(all(0 < len(c) <= 8 for c in chunks))

    def test_invalid_atom_counts_rejected(self):
        for n in (0, 1, 2, 8, 74, 76, True, 75.0):
            with self.assertRaises(ValueError):
                w.response_blocks(n)

    def test_unfrozen_partial_fails_before_import(self):
        with mock.patch.object(w, "PARTIAL_WORKER_SHA", "NOT_FROZEN"):
            with self.assertRaisesRegex(ValueError, "Unfrozen"):
                w.partial_worker()

    def test_option_requires_one_value(self):
        self.assertEqual(w.option(["--stage", "h1ao"], "--stage"), "h1ao")
        for command in ([], ["--stage"], ["--stage", "h1ao", "--stage", "response-block"]):
            with self.assertRaises(ValueError):
                w.option(command, "--stage")

    def test_partial_has_its_own_source_gate(self):
        with self.assertRaisesRegex(ValueError, "separate genuine XC"):
            w.qualify_stage({}, "rks-partial", {}, 75, 762, None)

    def test_partial_contract_not_translated(self):
        fake = types.SimpleNamespace(SCHEMA="old-v6b", CACHE_SCHEMA=w.CACHE_SCHEMA,
            PHYSICAL_SCHEMA=w.PHYSICAL_SCHEMA, NUMERICAL_POLICY=w.NUMERICAL_POLICY)
        with mock.patch.object(w, "partial_worker", return_value=fake):
            with self.assertRaisesRegex(ValueError, "another physical"):
                w.qualify_partial({}, {}, 75, 762, None)

    def test_partial_requires_complete_lineage_tuple(self):
        fake = types.SimpleNamespace(SCHEMA=w.SCHEMA, CACHE_SCHEMA=w.CACHE_SCHEMA,
            PHYSICAL_SCHEMA=w.PHYSICAL_SCHEMA, NUMERICAL_POLICY=w.NUMERICAL_POLICY,
            qualify_partial=lambda *args: ({}, {}, Path('/tmp'), {}))
        with mock.patch.object(w, "partial_worker", return_value=fake):
            with self.assertRaisesRegex(ValueError, "Complete native XC"):
                w.qualify_partial({}, {}, 75, 762, None)

    def test_fixture_array_hash_and_identity_checked_without_decode(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'fixture.npy'; p.write_bytes(b'SYNTHETIC_NON_NUMERICAL_BYTES')
            row = {"key": "H1AO_atom_000", "file": p.name, "file_sha256": w.sha(p), "shape": [3, 2, 2],
                   "dtype": "float64", "payload": {"fixture": True}}
            w.verify_arrays(Path(td), [row], [{"key": row['key'], "shape": [3, 2, 2]}])
            p.write_bytes(b'CHANGED_SYNTHETIC_BYTES')
            with self.assertRaisesRegex(ValueError, "changed"):
                w.verify_arrays(Path(td), [row], [{"key": row['key'], "shape": [3, 2, 2]}])

    def test_array_missing_or_reordered_shape_rejected(self):
        with self.assertRaises(ValueError):
            w.verify_arrays(Path('/tmp'), [], [{"key": "x", "shape": [1]}])

    def test_symlink_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'real'; p.write_text('fixture')
            q = Path(td) / 'link'; q.symlink_to(p)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                w.checked({"path": str(q), "sha256": w.sha(p)})

    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'real'; d.mkdir(); p = d / 'fixture'; p.write_text('fixture')
            q = Path(td) / 'alias'; q.symlink_to(d, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                w.checked({"path": str(q / 'fixture'), "sha256": w.sha(p)})

    def test_correct_assembly_uses_partial_and_nuclear_once(self):
        obj = Providers(); seen = []
        originals = {k: getattr(obj, k).__func__ for k in ('partial_hess_elec','make_h1','solve_mo1','hess_nuc')}
        source = ['full cached partial']
        with w.AssemblyProviders(obj, source, lambda kind, values: seen.append((kind, values))) as counts:
            loaded = obj.partial_hess_elec(atmlst=[0, 1, 2]); loaded.append('owned mutation')
            self.assertEqual(obj.hess_nuc(atmlst=[0, 1, 2]), ['native nuclear'])
            self.assertEqual(counts, {'partial': 1, 'h1': 0, 'response_solver': 0, 'nuclear': 1})
        self.assertEqual(source, ['full cached partial'])
        self.assertEqual([x[0] for x in seen], ['partial','nuclear'])
        self.assertEqual(seen[0][1]['max_memory'], 4000)
        for k,v in originals.items():
            self.assertIs(getattr(obj,k).__func__, v)

    def test_unexpected_native_h1_rejected_and_methods_restored(self):
        obj = Providers()
        with self.assertRaisesRegex(RuntimeError, "forbids native recomputation"):
            with w.AssemblyProviders(obj, ['partial'], lambda *args: None):
                obj.make_h1()
        self.assertEqual(obj.make_h1(), ['native H1'])

    def test_unexpected_response_solve_rejected_and_methods_restored(self):
        obj = Providers()
        with self.assertRaises(RuntimeError):
            with w.AssemblyProviders(obj, ['partial'], lambda *args: None):
                obj.solve_mo1()
        self.assertEqual(obj.solve_mo1(), ['native response'])

    def test_duplicate_partial_rejected(self):
        obj = Providers()
        with self.assertRaisesRegex(ValueError, "Repeated partial"):
            with w.AssemblyProviders(obj, ['partial'], lambda *args: None):
                obj.partial_hess_elec(); obj.partial_hess_elec()

    def test_duplicate_nuclear_rejected(self):
        obj = Providers()
        with self.assertRaisesRegex(ValueError, "Repeated nuclear"):
            with w.AssemblyProviders(obj, ['partial'], lambda *args: None):
                obj.hess_nuc(); obj.hess_nuc()

    def test_incomplete_assembly_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            with w.AssemblyProviders(Providers(), ['partial'], lambda *args: None):
                pass

    def test_preexisting_instance_override_restored(self):
        obj = Providers(); replacement = types.MethodType(lambda self: ['override'],obj)
        obj.make_h1 = replacement
        with w.AssemblyProviders(obj, ['partial'], lambda *args: None):
            obj.partial_hess_elec(); obj.hess_nuc()
        self.assertIs(obj.make_h1, replacement)

    def test_parent_rejection_precedes_any_numerical_import(self):
        guard = types.SimpleNamespace(verify_resource_supervision=mock.Mock(side_effect=ValueError('NO_REAL_PARENT')))
        with mock.patch.object(w, 'load_source', return_value=guard):
            with self.assertRaisesRegex(ValueError, 'NO_REAL_PARENT'):
                w.execute(types.SimpleNamespace(threads=2))
        self.assertNotIn('pyscf', sys.modules)

    def test_wrong_NumInt_type_rejected_before_native_use(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.binding_fixture(td); args['ni'] = object()
            with self.assertRaisesRegex(ValueError, 'plain NumInt'):
                w.bind_actual_native_modules(**args)

    def test_wrong_actual_rhf_file_site_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.binding_fixture(td); args['rhf'].__file__ = str(Path(td)/'another_site/rhf.py')
            with self.assertRaisesRegex(ValueError, 'Actual RHF module'):
                w.bind_actual_native_modules(**args)

    def test_copied_callable_foreign_globals_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.binding_fixture(td)
            foreign = {}; exec(compile('def hess_elec(): pass', args['rhf'].__file__, 'exec'), foreign)
            args['rhf'].hess_elec = foreign['hess_elec']
            with self.assertRaisesRegex(ValueError, 'callable namespace'):
                w.bind_actual_native_modules(**args)

    def test_wrong_actual_krylov_file_site_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.binding_fixture(td)
            with self.assertRaisesRegex(ValueError, 'solver module differs'):
                w.bind_actual_native_modules(**args)


if __name__ == '__main__':
    unittest.main()
