"""V6 exact forwarding and scoped native thread control; no quantum calls."""
import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/pyscf_spinosin_memory_bounded_direct_worker_20261031.py"
spec=importlib.util.spec_from_file_location("V6_guard",SCRIPT)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class ThreadFixture:
    def __init__(self, value=4):self.value=value;self.events=[]
    def get(self):return self.value
    @contextmanager
    def context(self, value):
        before=self.value;self.events.append(("enter",value));self.value=value
        try:yield
        finally:self.value=before;self.events.append(("exit",before))


class MemoryBoundedDirectTests(unittest.TestCase):
    def make_guard(self, function=None, native=1, outer=4, emit=None):
        threads=ThreadFixture(outer)
        module=SimpleNamespace(nr_direct_drv=function or (lambda *a,**k:None))
        guard=worker.NativeDirectDriverThreadGuard(module,threads.context,threads.get,native,emit)
        return module,threads,guard

    def test_complete_density_batch_and_all_parameters_forward_unchanged(self):
        calls=[];module,threads,guard=self.make_guard()
        densities=np.arange(6*4*4,dtype=float).reshape(6,4,4)
        operators=["ji->s2kl"]*6;atm=object();bas=object();env=object();out=object()
        def original(*args,**kwargs):
            calls.append((args,kwargs,threads.value));return out
        module.nr_direct_drv=original
        args=("int2e_sph","s8",operators,densities,1,atm,bas,env)
        kwargs={"cvhfopt":object(),"cintopt":object(),"out":out,"optimize_sr":False}
        with guard:
            self.assertIs(module.nr_direct_drv(*args,**kwargs),out)
            self.assertEqual(threads.value,4)
        forwarded,forwarded_kwargs,inside=calls[0]
        self.assertEqual(inside,1)
        self.assertTrue(all(a is b for a,b in zip(args,forwarded)))
        self.assertEqual(forwarded_kwargs,kwargs)
        self.assertIs(forwarded[3],densities)
        self.assertEqual(forwarded[3].shape[0],6)
        self.assertIs(module.nr_direct_drv,original)
        self.assertTrue(guard.summary()["original_driver_restored"])

    def test_two_native_threads_are_bounded_and_outer_restored(self):
        module,threads,guard=self.make_guard(native=2)
        seen=[]
        module.nr_direct_drv=lambda *a,**k:seen.append(threads.value)
        with guard:module.nr_direct_drv()
        self.assertEqual(seen,[2]);self.assertEqual(threads.value,4)
        self.assertEqual(guard.summary()["max_observed_native_threads"],2)

    def test_native_exception_keeps_original_exception_and_restores_everything(self):
        class NativeFixtureError(RuntimeError):pass
        def original(*args,**kwargs):raise NativeFixtureError("synthetic, not C computation")
        module,threads,guard=self.make_guard(original)
        with self.assertRaisesRegex(NativeFixtureError,"synthetic"):
            with guard:module.nr_direct_drv()
        self.assertEqual(threads.value,4)
        self.assertIs(module.nr_direct_drv,original)
        self.assertEqual(guard.summary()["native_calls_entered"],1)
        self.assertEqual(guard.summary()["native_calls_returned"],0)

    def test_keyboard_interrupt_is_not_swallowed_and_threads_restore(self):
        def original(*a,**k):raise KeyboardInterrupt()
        module,threads,guard=self.make_guard(original)
        with self.assertRaises(KeyboardInterrupt):
            with guard:module.nr_direct_drv()
        self.assertEqual(threads.value,4);self.assertIs(module.nr_direct_drv,original)

    def test_wrong_omp_effect_refuses_before_native_allocation(self):
        called=[];module,threads,guard=self.make_guard(lambda *a,**k:called.append(True))
        @contextmanager
        def broken(value):yield
        guard.omp_context=broken
        with self.assertRaisesRegex(RuntimeError,"did not take effect"):
            with guard:module.nr_direct_drv()
        self.assertFalse(called)
        self.assertEqual(guard.summary()["native_calls_entered"],0)

    def test_restoration_failure_is_observed_not_hidden(self):
        module,threads,guard=self.make_guard()
        @contextmanager
        def broken_restore(value):
            threads.value=value
            try:yield
            finally:threads.value=2
        guard.omp_context=broken_restore
        with guard:module.nr_direct_drv()
        self.assertEqual(guard.summary()["restoration_failures"],1)
        self.assertTrue(guard.summary()["original_driver_restored"])

    def test_no_double_install_or_nested_driver_guard(self):
        module,threads,guard=self.make_guard()
        guard.install()
        with self.assertRaises(RuntimeError):guard.install()
        other=worker.NativeDirectDriverThreadGuard(module,threads.context,threads.get)
        with self.assertRaises(RuntimeError):other.install()
        guard.restore();guard.restore()
        self.assertTrue(guard.summary()["original_driver_restored"])

    def test_invalid_thread_policies_and_raising_outer_threads_are_rejected(self):
        for native in (0,3,4):
            with self.assertRaises(ValueError):self.make_guard(native=native)
        module,threads,guard=self.make_guard(native=2,outer=1)
        original=module.nr_direct_drv
        with self.assertRaises(ValueError):guard.install()
        self.assertIs(module.nr_direct_drv,original)

    def test_guard_does_not_overwrite_someone_elses_later_hook(self):
        module,threads,guard=self.make_guard();guard.install()
        unrelated=lambda *a,**k:None
        module.nr_direct_drv=unrelated;guard.restore()
        self.assertIs(module.nr_direct_drv,unrelated)
        self.assertEqual(guard.summary()["restoration_failures"],1)

    def test_unique_signature_observations_record_actual_limit_and_as_metadata(self):
        events=[];module,threads,guard=self.make_guard(emit=events.append)
        with guard:
            for _ in range(2):module.nr_direct_drv("i","s",[],np.zeros((6,4,4)),1,None,None,None)
            module.nr_direct_drv("i","s",[],np.zeros((4,4)),3,None,None,None)
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]["native_threads_observed"],1)
        self.assertEqual(events[0]["outer_threads_before_call"],4)
        self.assertIn("address_space_hard_limit_bytes",events[0]["process_resources_before_native_allocation"])
        summary=guard.summary()
        self.assertEqual(summary["native_calls_entered"],3)
        self.assertEqual(summary["native_calls_returned"],3)
        self.assertEqual(summary["density_batch_signature_counts"]["density_count=6;components=1"],2)
        summary["native_calls_entered"]=999
        self.assertEqual(guard.summary()["native_calls_entered"],3)

    def test_keyword_density_and_operator_arguments_are_not_sliced(self):
        calls=[];module,threads,guard=self.make_guard(lambda **kwargs:calls.append(kwargs))
        density=[np.zeros((4,4)) for _ in range(6)]
        with guard:module.nr_direct_drv(dms=density,ncomp=9,intor="derivative",jkscript=["x"]*6)
        self.assertIs(calls[0]["dms"],density)
        self.assertEqual(len(calls[0]["dms"]),6)
        self.assertEqual(guard.summary()["density_batch_signature_counts"],{"density_count=6;components=9":1})

    def test_actual_pyscf_omp_context_restores_without_any_native_integral(self):
        worker.install_sites()
        try:from pyscf import lib
        except ImportError:self.skipTest("Optional PySCF metadata unavailable")
        before=lib.num_threads();lib.num_threads(4)
        try:
            seen=[]
            module=SimpleNamespace(nr_direct_drv=lambda *a,**k:seen.append(lib.num_threads()))
            guard=worker.NativeDirectDriverThreadGuard(module,lib.with_omp_threads,lib.num_threads)
            with guard:module.nr_direct_drv()
            self.assertEqual(seen,[1])
            self.assertEqual(lib.num_threads(),4)
        finally:lib.num_threads(before)

    def test_all_standard_python_direct_routes_use_the_guarded_driver_global(self):
        source=ROOT/".runtime/quantum-upgrade-site/pyscf/scf/_vhf.py"
        if not source.exists():self.skipTest("Optional installed source absent")
        tree=ast.parse(source.read_text())
        functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        for name in ("direct","direct_mapdm","direct_bindm"):
            calls=[n for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="nr_direct_drv"]
            self.assertTrue(calls,name)

    def test_v6_settings_are_engineering_control_not_changed_integrals(self):
        args=SimpleNamespace(grid_level=1,threads=4,maxsteps=35,hessian_block_atoms=2,native_direct_driver_threads=1)
        settings=worker.worker_settings(args)
        self.assertEqual(settings["density_fit_backend"],"SCREENED_DIRECT_RKS_V4")
        self.assertFalse(settings["density_fitting"])
        self.assertEqual(settings["native_direct_driver_threads"],1)
        self.assertEqual(settings["threads"],4)
        self.assertEqual(settings["direct_scf_tol"],1e-13)
        self.assertEqual(settings["scf_conv_tol_hartree"],1e-8)
        self.assertEqual(settings["hessian_conv_tol_cpscf"],1e-8)
        self.assertIn("V6",settings["engineering_worker_version"])

    def test_native_buffer_failure_arithmetic_is_scalar_only(self):
        observed_per_density=533459336
        per_thread=observed_per_density*6
        self.assertEqual(per_thread,3200756016)
        self.assertEqual(per_thread*4,12803024064)
        self.assertEqual(12*1024**3-per_thread*4,81877824)
        self.assertGreater(per_thread*4+167632*1024,12*1024**3)


if __name__=="__main__":unittest.main()
