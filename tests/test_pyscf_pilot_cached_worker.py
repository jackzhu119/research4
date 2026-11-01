"""V2 backend guards without quantum calculations."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("cached_worker", Path(__file__).parents[1] / "scripts/pyscf_pilot_cached_df_worker_20261031.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class CachedWorkerTests(unittest.TestCase):
    def test_cache_report_does_not_scan_or_persist(self):
        cache = np.zeros((11, 15))
        mf = SimpleNamespace(with_df=SimpleNamespace(_cderi=cache))
        result = worker.cache_diagnostics(mf)
        self.assertEqual(result["storage"], "incore_ndarray")
        self.assertEqual(result["nbytes"], 11 * 15 * 8)
        self.assertEqual(result["auxiliary_functions"], 11)
        self.assertFalse(result["array_saved_to_output"])
        self.assertEqual(result["cache_values_finiteness"], "NOT_SCANNED_RESOURCE_POLICY")
        mf.with_df._cderi = None
        self.assertEqual(worker.cache_diagnostics(mf)["storage"], "not_built")
        mf.with_df._cderi = "not-an-incore-cache.h5"
        self.assertEqual(worker.cache_diagnostics(mf)["storage"], "unexpected_disk_or_other")

    def test_release_resets_only_derived_cache(self):
        class FakeDF:
            def __init__(self):
                self._cderi = np.ones((2, 3))
                self.calls = []
            def reset(self, mol):
                self.calls.append(mol)
        df = FakeDF()
        mf = SimpleNamespace(with_df=df, _eri=object(), checkpoint="keep.chk")
        sentinel = object()
        worker.release_cached_integrals(mf, sentinel)
        self.assertIsNone(df._cderi)
        self.assertIsNone(mf._eri)
        self.assertEqual(df.calls, [sentinel])
        self.assertEqual(mf.checkpoint, "keep.chk")

    def test_frequency_gate_binds_execution_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "optimized.xyz"
            path.write_text("fixture")
            settings = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1, "dispersion": "NONE",
                        "density_fitting": True, "auxbasis": "jfit", "density_fit_backend": "CACHED_INCORE_DFJ_V2"}
            result = {"status": "DONE", "geometry_optimized": True, "optimizer_converged": True,
                      "scf_converged": True, "final_gradient_thresholds_passed": True,
                      "optimized_xyz_sha256": worker.sha(path), "settings": settings}
            worker.require_optimized_result(result, path, settings)
            with self.assertRaises(ValueError):
                worker.require_optimized_result(result, path, dict(settings, density_fit_backend="DIRECT_J_V1"))

    def test_gradient_and_mode_guards_retained(self):
        self.assertFalse(worker.stationary_checks(np.ones((3, 3)) * .0003)["final_gradient_thresholds_passed"])
        result = worker.frequencies_payload([-1, 2, 3], np.ones((3, 3, 3)), 3)
        self.assertEqual(result["imaginary_frequency_count"], 1)
        self.assertAlmostEqual(result["frequencies_thz"][0], -.0299792458)


if __name__ == "__main__":
    unittest.main()
