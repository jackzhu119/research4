"""Pure metadata/tiny NumPy fixtures only; never read an active molecular file."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_pyscf_pure_j_results_20261101.py"


class PureJResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("pure_j_results_unit_fixtures", SCRIPT)
        cls.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def healthy_row(self):
        return {"phase": "synthetic_fixture_only", "status": "EXITED", "exit_code": 0,
                "ended_at_utc": datetime.now(timezone.utc).isoformat()}

    def pool(self, values, cap):
        return {"phases": [{"actual_cpu_seconds": value, "status": "PARTIAL_OR_FAILED"}
                           for value in values],
                "consumed_cpu_seconds": sum(values), "quantum_cpu_cap_seconds": cap}

    def rows(self, count=219):
        return [{"phase": "synthetic_not_a_calculation", "identity": "spinosin", "mode": i+1,
                 "frequency_cm1": float(i+1), "frequency_thz": (i+1)*self.audit.CM1_TO_THZ,
                 "imaginary": False, "accepted_for_comparison": False,
                 "IR_intensity_status": "NOT_COMPUTED"} for i in range(count)]

    def valid_result(self):
        rows = self.rows()
        settings = {**self.audit.CORE.METHOD,
            "density_fitting": False, "auxbasis": None, "hessian_auxbasis_response": None,
            "density_fit_backend": "SCREENED_DIRECT_RKS_V4", "direct_scf": True,
            "direct_scf_tol": 1e-13, "global_four_center_ERI_cache_allowed": False,
            "scf_conv_tol_hartree": 1e-8, "optimizer_thresholds": self.audit.CORE.THRESHOLDS,
            "threads": 4, "engineering_worker_version": "PURE_PBE_J_PARTIAL_V9_SAVE_ONLY",
            "partial_hessian_engine": "PURE_PBE_J_PARTIAL_V9_SAVE_ONLY",
            "completed_pure_j_equivalence_evidence": "synthetic_unit_test_marker_not_a_scientific_claim"}
        required = ("geometry_optimized", "optimizer_converged", "scf_converged",
            "final_gradient_thresholds_passed", "frequency_validation_passed",
            "stationary_minimum_verified", "all_frequencies_real_positive", "normal_modes_complete",
            "harmonic_analysis_excludes_translations_and_rotations", "normal_mode_hessian_symmetrized")
        return {**{key: True for key in required}, "identity": "spinosin", "atoms": 75,
            "status": "DONE", "frequencies_status": "DONE", "settings": settings,
            "normal_mode_count": 219, "expected_normal_mode_count": 219,
            "imaginary_frequency_count": 0, "IR_intensities_status": "NOT_COMPUTED",
            "frequencies_cm1": [row["frequency_cm1"] for row in rows],
            "frequencies_thz": [row["frequency_thz"] for row in rows],
            "atom_rms_hartree_per_bohr": 1e-4, "atom_max_hartree_per_bohr": 2e-4,
            "unit_fixture_only_not_an_experimental_or_quantum_result": True}

    def write_csv(self, values):
        path = self.root / "frequencies.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("mode", "frequency_cm1", "frequency_thz", "IR_intensity_status"))
            writer.writeheader()
            writer.writerows(values)
        return path

    def mode_basis(self):
        saved = np.zeros((2, 3, 3))
        saved.reshape(2, -1)[0, 0] = 1.
        saved.reshape(2, -1)[1, 1] = 1.
        return saved

    def authorization_fixture(self):
        stages = {name: {"payload_argv": ["synthetic_never_executed", "--phase", name]}
                  for name in ("water", "ethanol", "spinosin")}
        rows, closed = [], 0.
        for index, (name, value) in enumerate(stages.items(), 1):
            rows.append({**self.healthy_row(), "phase": name, "command": value["payload_argv"],
                "actual_cpu_seconds": float(index), "actual_cpu_closed_batch_cost_seconds": closed})
            closed += index
        ledger = {"schema": "research4-pyscf-actual-cpu-budget-v4", "phases": rows,
                  "quantum_cpu_cap_seconds": 129600, "consumed_cpu_seconds": closed}
        budget = self.root / "cpu_budget.json"
        budget.write_text(json.dumps(ledger))
        lock = {"status": "SOURCE_AND_COMMANDS_FROZEN_BEFORE_QUANTUM",
                "source_sha256": {}, "input_sha256": {}, "phases": stages}
        lock_path = self.root / "execution_lock.json"
        lock_path.write_text(json.dumps(lock))
        entry, supervisor = self.root / "entry_fixture.py", self.root / "supervisor_fixture.py"
        entry.write_text("# Synthetic source-hash fixture, never executed.\n")
        supervisor.write_text("# Synthetic source-hash fixture, never executed.\n")
        authorization_path = self.root / "synthetic_v4_authorization.json"
        authorization_path.write_text('{"synthetic_unit_test_fixture": true}')
        ledger.update(runner_sha256=self.audit.sha(supervisor), authorization_path=str(authorization_path),
                      authorization_sha256=self.audit.sha(authorization_path))
        budget.write_text(json.dumps(ledger))
        release = {"allow_analysis_export": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(self.root), "cpu_budget_sha256": self.audit.sha(budget),
            "execution_lock_sha256": self.audit.sha(lock_path),
            "analyzer_source_sha256": self.audit.sha(SCRIPT),
            "entry_source_sha256": self.audit.sha(entry), "supervisor_source_sha256": self.audit.sha(supervisor)}
        return ledger, budget, lock_path, entry, supervisor, release

    def global_manifest_fixture(self):
        result = {"settings": {"synthetic_fixture_only": True,
                  "partial_hessian_engine": "PURE_PBE_J_PARTIAL_V9_SAVE_ONLY"},
                  "optimization_result_sha256": "a" * 64,
                  "resource_supervision": {"proof_sha256": "b" * 64}}
        gradient, geometry = "1" * 64, "2" * 64
        binding = {"settings": deepcopy(result["settings"]),
            "worker_script_sha256": self.audit.PINNED[self.audit.V8],
            "input_evidence": {"geometry_optimized": True, "final_gradient_thresholds_passed": True,
                "gradient_npy_sha256": gradient, "normal_mode_geometry_npy_sha256": geometry,
                "optimization_result_sha256": result["optimization_result_sha256"],
                "resource_supervision_proof_sha256": result["resource_supervision"]["proof_sha256"]}}
        value = {"status": "DONE_INTERMEDIATE_CAPTURE_ONLY", "native_global_kernel_returned_normally": True,
            "H1AO_complete": True, "global_Hessian_assembly_unchanged": True, "partial_kernel_tiling_used": False,
            "integral_backend": "SCREENED_DIRECT_RKS_V4", "response_block_atoms": 8,
            "native_conv_tol_cpscf": 1e-8, "native_max_cycle": 50, "native_level_shift": 0,
            "independent_CPHF_residual_norm": "NOT_COMPUTED", "requested_tolerance_is_not_a_measured_residual": True,
            "dimensions": {"atoms": 3}, "response_complete_global_atom_ids": [0, 1, 2],
            "H1AO_atoms": [{"atom_id": i} for i in range(3)],
            "partial_hessian": {"native_function_returned_normally": True}, "binding": binding,
            "response_blocks": [{"global_atom_ids": [0, 1, 2], "native_solver_returned_normally": True,
                "arrays": [{"atom_id": i} for i in range(3)], "residual_norm": "NOT_COMPUTED",
                "normal_return_does_not_prove_measured_residual_below_requested_tolerance": True,
                "requested_effective_CPHF_tolerance": 3e-8}]}
        self.refresh_binding(value)
        return value, result, gradient, geometry

    def refresh_binding(self, value):
        value["binding_sha256"] = hashlib.sha256(json.dumps(value["binding"], sort_keys=True).encode()).hexdigest()

    def test_healthy_parent_rejects_every_resource_stop_and_partial_exit(self):
        row = self.healthy_row()
        self.assertTrue(self.audit.healthy_parent(row))
        for key in self.audit.STOP_FLAGS:
            self.assertFalse(self.audit.healthy_parent({**row, key: True}), key)
        for change in ({"status": "PARTIAL_OR_FAILED"}, {"exit_code": 1}, {"ended_at_utc": None},
                       {"actual_cpu_stop_observation": {"sampled": True}},
                       {"resource_stop_reason": "ACTUAL_CPU_GUARD"},
                       {"residual_processes_require_review_after_main_payload_exit": [7001]}):
            self.assertFalse(self.audit.healthy_parent({**row, **change}))

    def test_costs_keep_separate_actual_pools_and_preserve_overrun(self):
        totals = self.audit.cost_totals(self.pool([2., 4.25, 0.], 5), self.pool([10., 20.], 100), self.pool([4.], 10))
        self.assertEqual(totals["V4_actual_CPU_seconds"], 6.25)
        self.assertEqual(totals["V4_pool_overrun_CPU_seconds"], 1.25)
        self.assertEqual(totals["V3_actual_CPU_seconds"], 30.)
        self.assertEqual(totals["V2_actual_CPU_seconds"], 4.)
        self.assertEqual(totals["V2_V3_V4_actual_CPU_seconds_scope_excludes_original_pilot_and_nonquantum_tests"], 40.25)
        self.assertTrue(totals["V4_is_separate_36_CPU_hour_pool_not_continuation_of_original_48_hour_claim"])

    def test_invalid_or_forged_actual_cpu_totals_are_rejected(self):
        for pool in (self.pool([2., -1.], 5), self.pool([float("nan")], 5),
                     {**self.pool([2.], 5), "consumed_cpu_seconds": 3.}):
            with self.assertRaises(ValueError):
                self.audit.cost_totals(pool, self.pool([1.], 5), self.pool([1.], 5))

    def test_missing_frequency_csv_is_empty_evidence_not_zero_absorption(self):
        hashes = {}
        self.assertEqual(self.audit.frequency_rows(self.root, "spinosin", "unit", hashes), [])
        self.assertEqual(hashes, {})
        self.assertFalse((self.root / "frequencies.csv").exists())

    def test_actual_csv_keeps_negative_signs_and_records_hash_without_acceptance(self):
        values = [{"mode": i+1, "frequency_cm1": cm, "frequency_thz": cm*self.audit.CM1_TO_THZ,
                   "IR_intensity_status": "NOT_COMPUTED"} for i, cm in enumerate([-3.5, 20., 80.])]
        path = self.write_csv(values)
        hashes = {}
        rows = self.audit.frequency_rows(self.root, "water", "synthetic_closed_control", hashes)
        self.assertEqual([row["frequency_cm1"] for row in rows], [-3.5, 20., 80.])
        self.assertTrue(rows[0]["imaginary"])
        self.assertFalse(any(row["accepted_for_comparison"] for row in rows))
        self.assertEqual(hashes[str(path)], self.audit.sha(path))

    def test_inconsistent_csv_mode_units_or_invented_ir_status_is_rejected(self):
        baseline = {"mode": 1, "frequency_cm1": 20., "frequency_thz": 20*self.audit.CM1_TO_THZ,
                    "IR_intensity_status": "NOT_COMPUTED"}
        for change in ({"mode": 2}, {"frequency_thz": 20.}, {"frequency_cm1": float("nan")},
                       {"IR_intensity_status": "COMPUTED"}):
            self.write_csv([{**baseline, **change}])
            with self.assertRaises(ValueError):
                self.audit.frequency_rows(self.root, "water", "unit", {})

    def test_qualification_requires_complete219_positive_modes_and_both_real_gates(self):
        result, rows = self.valid_result(), self.rows()
        self.assertTrue(self.audit.qualified_modes(result, rows, True, True))
        self.assertFalse(self.audit.qualified_modes(result, rows[:-1], True, True))
        self.assertFalse(self.audit.qualified_modes(result, rows, False, True))
        self.assertFalse(self.audit.qualified_modes(result, rows, True, False))
        negative = deepcopy(rows); negative[0]["frequency_cm1"] = -1.
        self.assertFalse(self.audit.qualified_modes(result, negative, True, True))
        self.assertFalse(self.audit.qualified_modes({**result, "worker_result_missing": True}, rows, True, True))

    def test_real_common_method_result_ir_and_fresh_gradient_guards_are_required(self):
        rows = self.rows()
        for key, value in (("IR_intensities_status", "COMPUTED"),
                           ("final_gradient_thresholds_passed", False), ("imaginary_frequency_count", 1)):
            result = self.valid_result(); result[key] = value
            self.assertFalse(self.audit.qualified_modes(result, rows, True, True), key)
        result = self.valid_result(); result["settings"]["xc"] = "B3LYP"
        self.assertFalse(self.audit.qualified_modes(result, rows, True, True))

    def test_pure_j_extra_engine_and_evidence_requirements_are_not_only_common_modes(self):
        for key, value in (("engineering_worker_version", "ordinary_control"),
                           ("partial_hessian_engine", "RHF_ONLY_NOT_FULL_PBE"),
                           ("completed_pure_j_equivalence_evidence", ""), ("threads", 1),
                           ("density_fit_backend", "unreviewed_backend")):
            result = self.valid_result(); result["settings"][key] = value
            with patch.object(self.audit.CORE, "valid_modes", return_value=True):
                self.assertFalse(self.audit.qualified_modes(result, self.rows(), True, True), key)

    def test_all20_candidates_cross_every_low_mode_without_nearest_peak_selection(self):
        experimental = [{"candidate_number": i+1, "experimental_frequency_thz": .1+i*.07,
            "experimental_wavenumber_cm1": (.1+i*.07)/self.audit.CM1_TO_THZ} for i in range(20)]
        modes = [{"phase": "synthetic_not_quantum", "mode": i+1, "frequency_thz": thz,
                  "frequency_cm1": thz/self.audit.CM1_TO_THZ} for i, thz in enumerate([.15, .65, 1.45])]
        pairs = self.audit.all_candidate_pairs(experimental, modes)
        self.assertEqual(len(pairs), 60)
        self.assertEqual({(row["candidate_number"], row["theoretical_mode"]) for row in pairs},
                         set(itertools.product(range(1, 21), range(1, 4))))
        self.assertGreater(max(row["absolute_difference_thz"] for row in pairs), 1.)
        self.assertTrue(all(row["chemical_attribution_verified"] is False for row in pairs))
        self.assertTrue(all(row["IR_intensity_status"] == "NOT_COMPUTED" for row in pairs))

    def test_empty_low_modes_do_not_invent_rows_and_experimental_count_is_fixed(self):
        experimental = [{"fixture": i} for i in range(20)]
        self.assertEqual(self.audit.all_candidate_pairs(experimental, []), [])
        with self.assertRaises(ValueError):
            self.audit.all_candidate_pairs(experimental[:-1], [])

    def test_mode_comparison_accepts_arbitrary_sign_and_positive_rescaling(self):
        saved = self.mode_basis()
        replayed = saved.copy(); replayed[0] *= -3.; replayed[1] *= 5.
        result = self.audit.compare_mode_subspaces([20., 80.], saved, replayed)
        self.assertLess(result["maximum_projector_distance"], 1e-12)
        self.assertTrue(result["arbitrary_mode_signs_allowed"])

    def test_degenerate_rotation_compares_the_whole_cluster_subspace(self):
        saved = self.mode_basis()
        rotation = np.asarray([[1., 1.], [-1., 1.]])/np.sqrt(2)
        replayed = (rotation @ saved.reshape(2, -1)).reshape(saved.shape)
        result = self.audit.compare_mode_subspaces([20., 20.+5e-6], saved, replayed)
        self.assertLess(result["maximum_projector_distance"], 1e-12)

    def test_wrong_direction_or_nondegenerate_rotation_is_rejected(self):
        saved = self.mode_basis()
        wrong = saved.copy(); wrong[0] = 0.; wrong.reshape(2, -1)[0, 2] = 1.
        with self.assertRaises(ValueError):
            self.audit.compare_mode_subspaces([20., 80.], saved, wrong)
        rotation = np.asarray([[1., 1.], [-1., 1.]])/np.sqrt(2)
        with self.assertRaises(ValueError):
            self.audit.compare_mode_subspaces([20., 80.], saved,
                (rotation @ saved.reshape(2, -1)).reshape(saved.shape))

    def test_zero_nonfinite_complex_and_rank_deficient_modes_are_rejected(self):
        saved = self.mode_basis()
        zero = saved.copy(); zero[0] = 0.
        nonfinite = saved.copy(); nonfinite[0, 0, 0] = np.nan
        rank_deficient = saved.copy(); rank_deficient[1] = rank_deficient[0]
        for cm, wrong in (([20., 80.], zero), ([20., 80.], nonfinite),
                          ([20., 80.], saved.astype(complex)+1j), ([20., 20.], rank_deficient)):
            with self.assertRaises(ValueError):
                self.audit.compare_mode_subspaces(cm, saved, wrong)

    def test_running_or_unended_stage_is_rejected_before_any_array_or_validator(self):
        ledger, path, lock, entry, supervisor, release = self.authorization_fixture()
        validator = Mock(side_effect=AssertionError("Cannot validate while a stage is active"))
        for change in ({"status": "RUNNING"}, {"ended_at_utc": None}):
            active = deepcopy(ledger); active["phases"][0].update(change)
            path.write_text(json.dumps(active))
            with patch.object(self.audit.np, "load", side_effect=AssertionError("No active molecular array may be read")) as load:
                with self.assertRaises(ValueError):
                    self.audit.authorize(self.root, release, validator=validator)
                load.assert_not_called()
        validator.assert_not_called()

    def test_wrong_execution_lock_or_released_source_is_rejected_before_arrays(self):
        ledger, path, lock, entry, supervisor, release = self.authorization_fixture()
        validator = Mock()
        with patch.object(self.audit, "EXECUTION_LOCK_SHA", self.audit.sha(lock)), \
                patch.object(self.audit, "ENTRY", entry), patch.object(self.audit, "SUPERVISOR", supervisor), \
                patch.object(self.audit, "PINNED", {}), \
                patch.object(self.audit.np, "load", side_effect=AssertionError("No array before full release")) as load:
            for changed in ({**release, "execution_lock_sha256": "0"*64},
                            {**release, "entry_source_sha256": "0"*64}):
                with self.assertRaises(ValueError):
                    self.audit.authorize(self.root, changed, validator=validator)
            load.assert_not_called()
        validator.assert_not_called()

    def test_all_three_frozen_stages_and_exact_commands_must_close_before_arrays(self):
        ledger, path, lock, entry, supervisor, release = self.authorization_fixture()
        validator = Mock()
        cases = []
        missing = deepcopy(ledger); missing["phases"].pop(); cases.append(missing)
        duplicate = deepcopy(ledger); duplicate["phases"][2] = deepcopy(duplicate["phases"][1]); cases.append(duplicate)
        command = deepcopy(ledger); command["phases"][2]["command"] = ["changed_not_executed"]; cases.append(command)
        with patch.object(self.audit, "EXECUTION_LOCK_SHA", self.audit.sha(lock)), \
                patch.object(self.audit, "ENTRY", entry), patch.object(self.audit, "SUPERVISOR", supervisor), \
                patch.object(self.audit, "PINNED", {}), \
                patch.object(self.audit.np, "load", side_effect=AssertionError("No active molecular array may be read")) as load:
            for case in cases:
                path.write_text(json.dumps(case))
                changed = {**release, "cpu_budget_sha256": self.audit.sha(path)}
                with self.assertRaises(ValueError):
                    self.audit.authorize(self.root, changed, validator=validator)
            load.assert_not_called()
        validator.assert_not_called()

    def test_full_synthetic_authorize_returns_seven_bound_metadata_objects_without_arrays(self):
        ledger, budget, lock, entry, supervisor, release = self.authorization_fixture()
        v2 = {**self.pool([14846.81297], 43200), "synthetic_fixture_only": True}
        v2path = self.root / "synthetic_prior_v2.json"
        v2path.write_text(json.dumps(v2))
        v3 = {**self.pool([111023.488913], 172800), "synthetic_fixture_only": True,
              "prior_closed_ledger": {"path": str(v2path), "sha256": self.audit.sha(v2path)}}
        v3path = self.root / "synthetic_prior_v3.json"
        v3path.write_text(json.dumps(v3))
        authorization = {"synthetic_validator_fixture_only": True,
            "prior_closed_ledger_path": str(v3path), "prior_closed_ledger_sha256": self.audit.sha(v3path)}
        validator = Mock(return_value=authorization)
        with patch.object(self.audit, "EXECUTION_LOCK_SHA", self.audit.sha(lock)), \
                patch.object(self.audit, "ENTRY", entry), patch.object(self.audit, "SUPERVISOR", supervisor), \
                patch.object(self.audit, "PINNED", {}), \
                patch.object(self.audit.np, "load", side_effect=AssertionError("Metadata authorization must not read arrays")) as load:
            actual = self.audit.authorize(self.root, release, validator=validator)
            load.assert_not_called()
        validator.assert_called_once_with(ledger)
        self.assertEqual(len(actual), 7)
        self.assertEqual(actual[0], ledger)
        self.assertEqual(actual[1], authorization)
        self.assertEqual(actual[4]["V4_actual_CPU_seconds"], 6.)
        self.assertEqual(actual[4]["V3_actual_CPU_seconds"], 111023.488913)
        self.assertEqual(actual[4]["V2_actual_CPU_seconds"], 14846.81297)
        self.assertTrue(all(not name.endswith((".npy", ".h5", ".chk")) for name in actual[3]))

    def test_complete_global_manifest_validates_small_json_without_loading_intermediates(self):
        value, result, gradient, geometry = self.global_manifest_fixture()
        with patch.object(self.audit.np, "load", side_effect=AssertionError("No intermediate-array read is allowed")) as load:
            checked = self.audit.validate_global_manifest(value, result, gradient, geometry, natom=3)
            load.assert_not_called()
        self.assertEqual(checked["status"], "FULL_GLOBAL_KERNEL_AND_COMPLETE_RESPONSE_METADATA_VERIFIED")
        self.assertEqual(checked["atoms"], 3)
        self.assertFalse(checked["large_intermediate_arrays_loaded"])
        self.assertFalse(checked["CPHF_residual_independently_measured"])

    def test_early_rhf_partial_missing_atoms_and_wrong_fresh_bindings_are_not_global_completion(self):
        value, result, gradient, geometry = self.global_manifest_fixture()
        cases = []
        early = deepcopy(value); early["partial_hessian"] = None
        early["rhf_early_helper_cache"] = {"status": "DONE", "not_full_PBE": True}; cases.append(early)
        missing = deepcopy(value); missing["H1AO_atoms"].pop(); cases.append(missing)
        coverage = deepcopy(value); coverage["response_complete_global_atom_ids"] = [0, 2]; cases.append(coverage)
        native = deepcopy(value); native["native_global_kernel_returned_normally"] = False; cases.append(native)
        canonical = deepcopy(value); canonical["binding_sha256"] = "0"*64; cases.append(canonical)
        for key, bad in (("worker_script_sha256", "0"*64), ("settings", {"changed": True})):
            changed = deepcopy(value); changed["binding"][key] = bad
            self.refresh_binding(changed); cases.append(changed)
        changed = deepcopy(value); changed["binding"]["input_evidence"]["gradient_npy_sha256"] = "0"*64
        self.refresh_binding(changed); cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError):
                self.audit.validate_global_manifest(changed, result, gradient, geometry, natom=3)

    def test_response_block_arrays_tolerance_and_unmeasured_residual_contract_are_strict(self):
        value, result, gradient, geometry = self.global_manifest_fixture()
        for key, bad in (("global_atom_ids", [2, 1, 0]), ("native_solver_returned_normally", False),
                         ("arrays", [{"atom_id": 0}, {"atom_id": 1}]),
                         ("residual_norm", 0.),
                         ("normal_return_does_not_prove_measured_residual_below_requested_tolerance", False),
                         ("requested_effective_CPHF_tolerance", 8e-8)):
            changed = deepcopy(value); changed["response_blocks"][0][key] = bad
            with self.assertRaises(ValueError):
                self.audit.validate_global_manifest(changed, result, gradient, geometry, natom=3)


if __name__ == "__main__":
    unittest.main()
