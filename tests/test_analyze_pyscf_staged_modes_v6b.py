"""Pure stdlib contract tests only: no molecular arrays or quantum engine."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_pyscf_staged_modes_20261105_v6b.py"
SPEC = importlib.util.spec_from_file_location("v6b_staged_analysis_pure_tests", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def header(n=75):
    result = {"schema": M.SCHEMA, "stage": "global", "phase": "global", "status": "FREQUENCIES_COMPUTED",
        "frequencies_status": "DONE", "atoms": n, "normal_mode_count": 3 * n - 6,
        "expected_normal_mode_count": 3 * n - 6, "source_MF_stability_status": "VERIFIED_UNCHANGED",
        "IR_intensities_status": "NOT_COMPUTED", "measured_CPHF_residual_norm": "NOT_COMPUTED",
        "full_global_contribution_calls": {"partial": 1, "h1": 1, "nuclear": 1},
        "V5_analyzer_compatibility": "BLOCKED_V6B_PROOF_SCHEMA_REQUIRES_REVIEWED_V6B_POSTPROCESSOR"}
    for key in ("no_new_SCF_performed", "stable_MF_state_unchanged", "complete_RHF_helper", "complete_RKS_partial",
                "complete_H1AO", "complete_global_Hessian", "normal_modes_complete", "final_gradient_thresholds_passed",
                "full_system_CPHF_returned_normally", "harmonic_analysis_excludes_translations_and_rotations"):
        result[key] = True
    return result


def capture_fixture(n=9):
    a = {"path": "/synthetic/A/result.json", "sha256": "a" * 64}
    b = {"path": "/synthetic/B/result.json", "sha256": "b" * 64}
    result = {"basis_functions": 20, "settings": {"threads": 1}, "snapshot_manifest_sha256": "s" * 64,
        "resource_supervision": {"proof_sha256": "p" * 64}}
    origin = {"partial": "HEALTHY_COMPLETE_V6B_CACHE_COPY", "H1AO": "HEALTHY_COMPLETE_V6B_CACHE_FRESH_LOADS",
        "CPHF": "FRESH_FULL_SYSTEM_STOCK_B8", "nuclear": "FRESH_STOCK_ONCE", "native_make_h1_executed_in_this_phase": False,
        "H1AO_capture_duplicates_closed_source_bytes": 3 * n * 20**2 * 8, "duplicate_H1_capture_IO_counted_not_elided": True}
    result["RKS_partial_and_H1_cache_restore"] = deepcopy(origin)
    value = {"status": "DONE_INTERMEDIATE_CAPTURE_ONLY", "native_global_kernel_returned_normally": True,
        "H1AO_complete": True, "global_Hessian_assembly_unchanged": True, "partial_kernel_tiling_used": False,
        "response_block_atoms": 8, "native_conv_tol_cpscf": 1e-8, "native_max_cycle": 50, "native_level_shift": 0,
        "independent_CPHF_residual_norm": "NOT_COMPUTED", "dimensions": {"atoms": n},
        "response_complete_global_atom_ids": list(range(n)), "H1AO_atoms": [{"atom_id": i} for i in range(n)],
        "partial_hessian": {"native_function_returned_normally": False, "cache_provider_returned_normally": True,
            "native_partial_executed_in_this_phase": False, "healthy_native_partial_source": a},
        "V6B_stage_execution_origin": origin}
    binding = {"settings": result["settings"], "input_evidence": {"source_snapshot_sha256": "s" * 64,
        "actual_entry_source_sha256": M.WORKER_SHA, "consumer_resource_proof_sha256": "p" * 64,
        "partial_stage_result_sha256": a["sha256"], "H1_stage_result_sha256": b["sha256"],
        "geometry_optimized": True, "final_gradient_thresholds_passed": True}}
    value["binding"] = binding; value["binding_sha256"] = M.digest(binding, compact=False)
    value["response_blocks"] = [{"global_atom_ids": list(range(i, min(i + 8, n))),
        "native_solver_returned_normally": True, "residual_norm": "NOT_COMPUTED",
        "arrays": [{"atom_id": j} for j in range(i, min(i + 8, n))]} for i in range(0, n, 8)]
    return value, result, a, b


class PureContracts(unittest.TestCase):
    def test_import_and_metadata_definitions_never_load_numerical_libraries(self):
        state, consumer = M.v5_metadata_validators()
        self.assertEqual(state.RESULT_SCHEMA, "research4-molecular-state-result-v1")
        self.assertEqual(consumer.SCHEMA, "research4-resumable-full-rks-hessian-result-v1")
        self.assertFalse(any(name.split(".")[0] in {"numpy", "scipy", "torch", "pyscf"} for name in sys.modules))
        row = {"status": "EXITED", "exit_code": 0, "ended_at_utc": "2026-11-05T00:00:00Z", "actual_cpu_seconds": 2.}
        row.update({k: False for k in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed")})
        self.assertTrue(consumer.healthy_closed_row(row))
        for key in ("cpu_terminated", "monitor_failed", "hard_killed"):
            changed = {**row, key: True}
            self.assertFalse(consumer.healthy_closed_row(changed))

    def test_metadata_extractor_ignores_top_level_imports_and_statements(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fake.py"
            p.write_text("import forbidden_library\nraise RuntimeError('must not run')\nVALUE=7\ndef one():\n    return VALUE\n")
            subset = M.frozen_metadata_subset(p, M.sha(p), ("one",), ("VALUE",))
            self.assertEqual(subset.one(), 7)
            self.assertNotIn("forbidden_library", sys.modules)
            with self.assertRaises(ValueError):
                M.frozen_metadata_subset(p, "0" * 64, ("one",), ("VALUE",))
            with self.assertRaises(ValueError):
                M.frozen_metadata_subset(p, M.sha(p), ("missing",), ("VALUE",))
            p.write_text("def decorator(x): return x\n@decorator\ndef one(): return 1\n")
            with self.assertRaises(ValueError):
                M.frozen_metadata_subset(p, M.sha(p), ("one",), ())

    def test_default_and_real_control_identity_are_distinct(self):
        self.assertEqual(M.case_for(["C"] * 28 + ["H"] * 32 + ["O"] * 15, False), "Spinosin")
        self.assertEqual(M.case_for(["O", "H", "H"], True), "water_software_control")
        self.assertEqual(M.case_for(["C"] * 2 + ["H"] * 6 + ["O"], True), "ethanol_software_control")
        for symbols, control in ((["O", "H", "H"], False), (["C"] * 28 + ["H"] * 32 + ["O"] * 15, True), (["He"], True)):
            with self.assertRaises(ValueError):
                M.case_for(symbols, control)

    def test_complete_global_header_and_mode_counts(self):
        self.assertEqual(M.global_header(header(), control=False), 75)
        self.assertEqual(M.global_header(header(3), control=True), 3)
        self.assertEqual(M.global_header(header(9), control=True), 9)
        for change in ({"schema": "research4-resumable-full-rks-hessian-result-v1"}, {"complete_H1AO": False},
                {"frequencies_status": "NOT_RUN"}, {"normal_mode_count": 218}, {"IR_intensities_status": "COMPUTED"},
                {"measured_CPHF_residual_norm": 0.}, {"full_global_contribution_calls": {"partial": 1, "h1": 1, "nuclear": 2}},
                {"V5_analyzer_compatibility": "COMPATIBLE"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                M.global_header({**header(), **change}, control=False)
        with self.assertRaises(ValueError):
            M.global_header(header(3), control=False)

    def test_actual_command_options_unique_and_root_relative(self):
        self.assertEqual(M.command_option(["x.py", "--stage", "global"], "--stage"), "global")
        self.assertEqual(M.root_path("reports/test/result.json"), (M.ROOT / "reports/test/result.json").resolve())
        for command in (["--stage"], ["--stage", "global", "--stage", "global"], ["x.py"]):
            with self.assertRaises(ValueError):
                M.command_option(command, "--stage")

    def test_three_actual_stage_commands_bind_source_g_and_b_to_a(self):
        base = {"snapshot": "/synthetic/source", "snapshot_manifest_sha256": "s" * 64,
            "gradient_check_result": "/synthetic/g/result.json", "gradient_check_result_sha256": "g" * 64,
            "partial_result": "/synthetic/a/result.json", "partial_result_sha256": "a" * 64,
            "h1_result": "/synthetic/b/result.json", "h1_result_sha256": "b" * 64, "threads": 1}
        result = {"settings": {"threads": 1}, "snapshot_manifest_sha256": "s" * 64}
        for stage in ("rks-partial", "h1ao", "global"):
            args = {**base, "execute": True, "control_validation": False, "stage": stage, "rhf_cache": "/synthetic/rhf"}
            fields = [("--stage", stage), ("--output", "/synthetic/out"), ("--threads", "1"),
                ("--snapshot", args["snapshot"]), ("--snapshot-manifest-sha256", args["snapshot_manifest_sha256"]),
                ("--gradient-check-result", args["gradient_check_result"]),
                ("--gradient-check-result-sha256", args["gradient_check_result_sha256"])]
            if stage == "rks-partial": fields += [("--rhf-cache", args["rhf_cache"])]
            if stage in ("h1ao", "global"):
                fields += [("--partial-result", args["partial_result"]), ("--partial-result-sha256", args["partial_result_sha256"])]
            if stage == "global": fields += [("--h1-result", args["h1_result"]), ("--h1-result-sha256", args["h1_result_sha256"])]
            command = ["script.py", "--execute"] + [value for pair in fields for value in pair]
            M.stage_command_gate(command, args, result, "/synthetic/out", stage, False, base)
            for flag, wrong in (("--stage", "wrong"), ("--snapshot", "/wrong"),
                    ("--gradient-check-result-sha256", "0" * 64), ("--threads", "2")):
                changed = command.copy(); changed[changed.index(flag) + 1] = wrong
                with self.subTest(stage=stage, flag=flag), self.assertRaises(ValueError):
                    M.stage_command_gate(changed, args, result, "/synthetic/out", stage, False, base)
            if stage == "h1ao":
                with self.assertRaises(ValueError):
                    M.stage_command_gate(command, args, result, "/synthetic/out", stage, False,
                        {**base, "partial_result_sha256": "0" * 64})

    def test_control_requires_authenticated_snapshot_and_recipe_not_formula(self):
        for case, wanted in M.CONTROL_SOURCES.items():
            producer = {"identity": wanted["identity"], "input_manifest": {"sha256": wanted["input_sha256"]}}
            doc = {"molecule_id": wanted["identity"], "formula": wanted["formula"], "connectivity_verified": True}
            M.control_identity_gate(case, wanted["snapshot_sha256"], producer, doc)
            with self.assertRaises(ValueError):
                M.control_identity_gate(case, "0" * 64, producer, doc)
            with self.assertRaises(ValueError):
                M.control_identity_gate(case, wanted["snapshot_sha256"], producer, {**doc, "molecule_id": "same_formula_other_isomer"})
            with self.assertRaises(ValueError):
                M.control_identity_gate(case, wanted["snapshot_sha256"], producer, {**doc, "connectivity_verified": False})

    def test_control_reference_authenticates_genuine_old_stock_before_arrays(self):
        case = "water_software_control"; wanted = M.CONTROL_SOURCES[case]
        source = {"case": case, "binding": {"synthetic": "same-exact-source"}}
        value = {"schema": "research4-resumable-full-rks-hessian-result-v1", "status": "FREQUENCIES_COMPUTED",
            "phase": "stock-hessian-reference", "frequencies_status": "DONE", "complete_global_Hessian": True,
            "normal_modes_complete": True, "source_state_binding": source["binding"],
            "snapshot_manifest_sha256": wanted["snapshot_sha256"], "source_worker": {}, "resource_supervision": {}}
        closure = {"producer_result_sha256": wanted["stock_result_sha256"]}
        v5 = types.SimpleNamespace(SCHEMA=value["schema"], EXECUTION_SCHEMA="genuine-old-v5-execution",
            verify_closed_execution=lambda *args, **kwargs: closure)
        with patch.object(M, "read_json", return_value=value), \
                patch.object(M, "v5_metadata_validators", return_value=(None, v5)), \
                patch.object(M, "retain_v5_closure") as retain:
            ref = M.control_reference_gate(source, {})
            self.assertEqual(ref["closure"], closure)
            retain.assert_called_once()
        for change in ({"schema": M.SCHEMA}, {"phase": "hessian-segment"},
                {"source_state_binding": {"other": "source"}}, {"normal_modes_complete": False}):
            with patch.object(M, "read_json", return_value={**value, **change}), \
                    patch.object(M, "v5_metadata_validators", return_value=(None, v5)), \
                    self.assertRaises(ValueError):
                M.control_reference_gate(source, {})

    def test_no_numerical_runtime_without_gate(self):
        with self.assertRaises(ValueError):
            M.numpy_runtime({})
        self.assertIsNone(M.NUMERICAL)

    def test_rejected_provenance_stops_before_numpy_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not_created"
            with patch.object(M, "qualify", side_effect=ValueError("synthetic malformed closed chain")), \
                    patch.object(M, "numpy_runtime", side_effect=AssertionError("must not be called")) as numerical:
                with self.assertRaisesRegex(ValueError, "synthetic malformed"):
                    M._analyze_closed("unused", "f" * 64, target)
                numerical.assert_not_called()
                self.assertFalse(target.exists())

    def test_public_analysis_always_requests_actual_exclusive_q_lock(self):
        events = []
        class Lock:
            def __enter__(self): events.append("lock")
            def __exit__(self, *exc): events.append("unlock")
        with patch.object(M, "exclusive_closed_quantum", return_value=Lock()), \
                patch.object(M, "_analyze_closed", side_effect=lambda *a, **k: events.append("analysis") or {"test": True}):
            self.assertEqual(M.analyze("unused", "f" * 64, "unused"), {"test": True})
        self.assertEqual(events, ["lock", "analysis", "unlock"])

    def test_signed_frequencies_and_minimum_qualification_are_separate(self):
        for values, positive in (([1., 2., 3.], True), ([-1., 2., 3.], False), ([0., 2., 3.], False)):
            value = {"imaginary_frequency_count": sum(v < 0 for v in values), "all_frequencies_real_positive": positive,
                "stationary_minimum_verified": positive, "frequency_validation_passed": positive}
            self.assertIs(M.validate_minimum_flags(value, values), positive)
            with self.assertRaises(ValueError):
                M.validate_minimum_flags({**value, "stationary_minimum_verified": not positive}, values)
        with self.assertRaises(ValueError):
            M.validate_minimum_flags({}, [float("nan")])
        self.assertTrue(M.mechanism_qualified([1.] * 219, control=False, fresh_gradient_verified=True))
        for values, control, gradient in (([1.] * 3, False, True), ([1.] * 219, True, True),
                ([1.] * 219, False, False), ([-1.] + [1.] * 218, False, True), ([0.] + [1.] * 218, False, True)):
            self.assertFalse(M.mechanism_qualified(values, control=control, fresh_gradient_verified=gradient))

    def test_fixed_eigenvalue_cluster_contract(self):
        self.assertEqual(M.eigenvalue_clusters([-.2, .1, .1 + 1e-10, .9]), [[0], [1, 2], [3]])
        self.assertEqual(M.eigenvalue_clusters([0.]), [[0]])
        for values in ([], [1., -1.], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                M.eigenvalue_clusters(values)

    def test_v6b_capture_accepts_cache_origin_not_old_native_partial_claim(self):
        value, result, a, b = capture_fixture()
        self.assertEqual(M.stage_capture_gate(value, result, a, b, 9)["CPHF_residual_measured"], False)
        changes = [lambda v: v["partial_hessian"].update(native_function_returned_normally=True),
            lambda v: v["partial_hessian"].update(native_partial_executed_in_this_phase=True),
            lambda v: v["partial_hessian"].update(healthy_native_partial_source={"path": "other", "sha256": "0" * 64}),
            lambda v: v["response_blocks"][0].update(residual_norm=0.),
            lambda v: v["response_blocks"][0].update(global_atom_ids=[0]),
            lambda v: v.update(response_complete_global_atom_ids=list(range(8))),
            lambda v: v["V6B_stage_execution_origin"].update(H1AO_capture_duplicates_closed_source_bytes=0),
            lambda v: v["binding"]["input_evidence"].update(actual_entry_source_sha256="0" * 64)]
        for change in changes:
            altered = deepcopy(value); change(altered)
            # Update binding hash so a substituted source still fails the actual semantic binding.
            altered["binding_sha256"] = M.digest(altered["binding"], compact=False)
            with self.subTest(change=change), self.assertRaises(ValueError):
                M.stage_capture_gate(altered, result, a, b, 9)

    def test_actual_source_g_closure_receipts_retained_and_changed_row_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            log = directory / "log"; log.write_text("synthetic closed receipt\n")
            result = directory / "result.json"; result.write_text('{}\n')
            row = {"phase": "source", "log": str(log), "log_sha256": M.sha(log), "status": "EXITED"}
            ledger = directory / "ledger.json"; ledger.write_text(json.dumps({"phases": [row]}))
            proof = {"ledger_path": str(ledger), "phase": "source"}
            proof_path = directory / "proof.json"; proof_path.write_text(json.dumps(proof))
            closure = {"phase": "source", "proof_sha256": M.sha(proof_path), "closed_row_sha256": M.digest(row),
                "closed_result_sha256": M.sha(result), "log_sha256": M.sha(log)}
            supervision = {"proof_path": str(proof_path), "proof": proof}
            hashes = {}
            M.retain_v5_closure(result, supervision, closure, hashes)
            self.assertEqual(len(hashes), 4)
            ledger.write_text(json.dumps({"phases": [{**row, "status": "RUNNING"}]}))
            with self.assertRaises(ValueError):
                M.retain_v5_closure(result, supervision, closure, {})

    def test_metadata_checks_are_written_before_any_numerical_import_in_analysis(self):
        tree = ast.parse(SCRIPT.read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in top_imports:
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            self.assertFalse(any(name and name.split(".")[0] in {"numpy", "scipy", "torch", "pyscf"} for name in names))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        analysis = ast.get_source_segment(SCRIPT.read_text(), functions["_analyze_closed"])
        self.assertLess(analysis.index("gate = qualify("), analysis.index("np, a2 = numpy_runtime(gate)"))
        qualification = ast.get_source_segment(SCRIPT.read_text(), functions["qualify"])
        self.assertLess(qualification.index("source = source_gate("), qualification.index("worker.qualify_stage("))
        self.assertIn("allow_running=False", ast.get_source_segment(SCRIPT.read_text(), functions["closed_command"]))


if __name__ == "__main__":
    unittest.main()
