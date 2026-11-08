"""V7 metadata rejection tests; no numerical imports, payloads or calculations."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v7.py"
SPEC = importlib.util.spec_from_file_location("v7_staged_analysis_stdlib_tests", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
OLD = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v6b_v2.py"
OLD_SHA = "38dde5e185c4fc48d58732982e0164b3a2f5430a26bdfc1076b21103d1d75c6a"


def header(n=75):
    value = {"schema": M.SCHEMA, "stage": "global-assemble", "phase": "global-assemble",
        "status": "FREQUENCIES_COMPUTED", "frequencies_status": "DONE", "atoms": n,
        "normal_mode_count": 3*n-6, "expected_normal_mode_count": 3*n-6,
        "source_MF_stability_status": "VERIFIED_UNCHANGED", "IR_intensities_status": "NOT_COMPUTED",
        "measured_CPHF_residual_norm": "NOT_COMPUTED",
        "full_global_contribution_calls": {"partial": 1, "h1": 0, "response_solver": 0, "nuclear": 1},
        "V5_analyzer_compatibility": "BLOCKED_V7_PRODUCER_REQUIRES_REVIEWED_V7_POSTPROCESSOR",
        "full_system_CPHF_executed_in_this_phase": False,
        "full_system_CPHF_completion_basis": "ALL_HEALTHY_FULL_AO_MO_B8_RESPONSE_PRODUCERS"}
    for key in ("no_new_SCF_performed", "stable_MF_state_unchanged", "complete_RHF_helper", "complete_RKS_partial",
                "complete_H1AO", "complete_global_Hessian", "normal_modes_complete", "final_gradient_thresholds_passed",
                "full_system_CPHF_returned_normally", "harmonic_analysis_excludes_translations_and_rotations"):
        value[key] = True
    return value


def arguments(stage="global-assemble", control=False):
    args = {"execute": True, "stage": stage, "control_validation": control, "threads": 2,
        "snapshot": "/synthetic/source", "snapshot_manifest_sha256": "s"*64,
        "gradient_check_result": "/synthetic/g/result.json", "gradient_check_result_sha256": "g"*64,
        "partial_result": "/synthetic/a/result.json", "partial_result_sha256": "a"*64,
        "h1_result": "/synthetic/b/result.json", "h1_result_sha256": "b"*64,
        "response_index": "/synthetic/response-index.json", "response_index_sha256": "r"*64,
        "control_release": "/synthetic/control-release.json", "control_release_sha256": "c"*64,
        "block_index": 1}
    keys = ("snapshot", "snapshot_manifest_sha256", "gradient_check_result", "gradient_check_result_sha256",
            "partial_result", "partial_result_sha256")
    if stage in ("response-block", "global-assemble"):
        keys += ("h1_result", "h1_result_sha256")
    if stage == "global-assemble":
        keys += ("response_index", "response_index_sha256")
    if not control:
        keys += ("control_release", "control_release_sha256")
    command = ["worker.py", "--execute", "--stage", stage, "--output", "/synthetic/out", "--threads", "2"]
    for key in keys:
        command.extend(["--"+key.replace("_", "-"), args[key]])
    if stage == "response-block":
        command.extend(["--block-index", "1"])
    if control:
        command.append("--control-validation")
    result = {"settings": {"threads": 2}, "snapshot_manifest_sha256": "s"*64, "block_index": 1}
    return command, args, result


def capture_fixture(n=9):
    a = {"path": "/synthetic/A/result.json", "sha256": "a"*64}
    b = {"path": "/synthetic/B/result.json", "sha256": "b"*64}
    index = {"path": "/synthetic/index.json", "sha256": "i"*64}
    refs = [{"path": f"/synthetic/response-{i}/result.json", "sha256": str(i)*64} for i in range((n+7)//8)]
    closures = [{"phase": f"response-{i}", "actual_cpu_seconds": 1.5} for i in range((n+7)//8)]
    value = {"schema": "research4-v7-full-global-assembly-capture-v1", "status": "DONE_COMPLETE_GLOBAL_ASSEMBLY",
        "contribution_calls": {"partial": 1, "h1": 0, "response_solver": 0, "nuclear": 1},
        "global_atom_ids": list(range(n)), "partial_result_dependency": a, "h1_result_dependency": b,
        "response_index_dependency": index, "explicit_full_H1AO_and_MO1_MOE1_supplied": True,
        "stock_hess_elec_returned_normally": True, "stock_nuclear_returned_normally": True,
        "global_cross_atom_contractions_preserved": True, "nuclear_added_once": True,
        "IR_intensities_status": "NOT_COMPUTED", "measured_CPHF_residual_norm": "NOT_COMPUTED",
        "H1AO_source_referenced_without_duplicate_capture":True,"CPHF_executed_in_this_phase":False,
        "all_response_blocks_healthily_closed":True,"response_result_dependencies":refs,
        "physical_binding_sha256":"p"*64, "source_worker":{"path":str(M.WORKER),"sha256":M.WORKER_SHA},
        "response_blocks": [{"block_index": i, "global_atom_ids": list(range(i*8, min((i+1)*8, n))),
            "result": refs[i], "closure": closures[i]} for i in range((n+7)//8)]}
    result = {"full_global_contribution_calls": deepcopy(value["contribution_calls"]),"physical_binding_sha256":"p"*64}
    return value, result, a, b, index, refs, closures, n


class V7MetadataTests(unittest.TestCase):
    def test_import_is_stdlib_only_and_old_source_unchanged(self):
        M.v5_metadata_validators()
        self.assertEqual(M.sha(OLD), OLD_SHA)
        self.assertFalse(any(name.split(".")[0] in {"numpy", "scipy", "torch", "pyscf"} for name in sys.modules))

    def test_actual_result_is_blocked_while_source_is_unfrozen(self):
        with patch.object(M, "WORKER_SHA", "SYNTHETIC_UNFROZEN_WORKER"):
            with self.assertRaisesRegex(ValueError, "not frozen"):
                M.qualify("unused", "0"*64)

    def test_rejected_gate_never_imports_numeric_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "never-written"
            with patch.object(M, "qualify", side_effect=ValueError("synthetic invalid actual chain")), \
                    patch.object(M, "numpy_runtime", side_effect=AssertionError("no numeric allowed")) as numeric:
                with self.assertRaisesRegex(ValueError, "invalid actual"):
                    M._analyze_closed("unused", "0"*64, out)
                numeric.assert_not_called()
                self.assertFalse(out.exists())

    def test_true_v7_header_distinguishes_global_assembly(self):
        self.assertEqual(M.global_header(header(), control=False), 75)
        self.assertEqual(M.global_header(header(3), control=True), 3)
        self.assertEqual(M.global_header(header(9), control=True), 9)
        for change in ({"schema": "research4-staged-full-rks-hessian-result-v6b"}, {"phase": "global"},
                {"normal_mode_count": 218}, {"complete_RKS_partial": False}, {"complete_H1AO": False},
                {"IR_intensities_status": "COMPUTED"}, {"measured_CPHF_residual_norm": 0.},
                {"full_global_contribution_calls": {"partial": 1, "h1": 1, "response_solver": 1, "nuclear": 1}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                M.global_header({**header(), **change}, control=False)

    def test_target_and_control_identities_are_separate(self):
        self.assertEqual(M.case_for(["C"]*28+["H"]*32+["O"]*15, False), "Spinosin")
        for symbols, control in ((["O", "H", "H"], False), (["C"]*28+["H"]*32+["O"]*15, True)):
            with self.assertRaises(ValueError): M.case_for(symbols, control)

    def test_new_actual_argv_bindings(self):
        for stage in ("h1ao", "response-block", "global-assemble"):
            for control in (False, True):
                command, args, result = arguments(stage, control)
                M.stage_command_gate(command, args, result, "/synthetic/out", stage, control, args)
                for flag in ("--snapshot-manifest-sha256", "--partial-result-sha256", "--threads"):
                    changed = command.copy(); changed[changed.index(flag)+1] = "0" if flag == "--threads" else "0"*64
                    with self.subTest(stage=stage, control=control, flag=flag), self.assertRaises(ValueError):
                        M.stage_command_gate(changed, args, result, "/synthetic/out", stage, control, args)
                if not control:
                    changed = command.copy(); changed[changed.index("--control-release-sha256")+1] = "0"*64
                    with self.assertRaises(ValueError):
                        M.stage_command_gate(changed, args, result, "/synthetic/out", stage, control, args)

    def test_A_cannot_use_old_stage_argv_gate(self):
        command, args, result = arguments()
        with self.assertRaisesRegex(ValueError, "separate decomposition"):
            M.stage_command_gate(command, args, result, "/synthetic/out", "rks-partial", False, args)

    def test_response_block_index_is_anchored_to_actual_argv(self):
        command, args, result = arguments("response-block")
        for wrong in (0, 2):
            with self.assertRaises(ValueError):
                M.stage_command_gate(command, args, {**result, "block_index": wrong}, "/synthetic/out", "response-block", False, args)
        with self.assertRaises(ValueError):
            M.stage_command_gate(command+["--block-index", "1"], args, result, "/synthetic/out", "response-block", False, args)

    def test_capture_complete_global_response_and_no_duplicate_h1_or_solver(self):
        fixture = capture_fixture(); value = fixture[0]
        self.assertEqual(M.stage_capture_gate(*fixture)["response_blocks"], 2)
        for key, wrong in (("stock_hess_elec_returned_normally", False), ("nuclear_added_once", False),
                ("global_cross_atom_contractions_preserved", False), ("explicit_full_H1AO_and_MO1_MOE1_supplied", False),
                ("measured_CPHF_residual_norm", 0.)):
            bad = {**value, key: wrong}
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.stage_capture_gate(bad, *fixture[1:])

    def test_response_missing_duplicate_out_of_order_or_invented_closure_rejected(self):
        fixture = capture_fixture()
        for mutate in (lambda v: v["response_blocks"].pop(),
                lambda v: v["response_blocks"].reverse(),
                lambda v: v["response_blocks"][1].update(global_atom_ids=[7,8]),
                lambda v: v["response_blocks"][1].update(closure={"invented": True})):
            value = deepcopy(fixture[0]); mutate(value)
            with self.assertRaises(ValueError): M.stage_capture_gate(value, *fixture[1:])

    def test_chunk_lineage_and_artifact_race_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p = root / "unit.json"; p.write_text('{"synthetic_metadata":true}\n')
            row = {"phase":"synthetic-healthy", "status":"EXITED", "exit_code":0,
                "ended_at_utc":"2026-11-05T00:00:00Z", "actual_cpu_seconds":1.}
            row.update({key:False for key in ("wall_terminated","memory_terminated","cpu_terminated","monitor_failed","hard_killed")})
            ledger = root / "ledger.json"; ledger.write_text(json.dumps({"phases":[row]}))
            value = {"schema": "research4-v7-complete-xc-partial-lineage-v1", "nuclear_or_CPHF_included": False,
                "RHF_producers": [{"synthetic": "healthy"}], "verified_artifacts": [{"path": str(p),
                    "sha256": M.sha(p), "bytes": p.stat().st_size}], "closed_ledger_rows":[{"path":str(ledger),
                        "phase":row["phase"],"closed_row_sha256":M.digest(row,compact=False)}]}
            booleans = ("all_chunk_producers_healthy", "all_required_grid_blocks_covered_exactly_once",
                "RHF_added_once", "XCdiag_once", "XCderiv2_once", "actual_partial_command_verified", "actual_component_commands_verified")
            value.update({key: True for key in booleans})
            with patch.object(M, "ROOT", root):
                hashes = {}; M.validate_extra_chain(value, hashes); self.assertIn(str(p), hashes)
                for key in booleans:
                    with self.subTest(key=key), self.assertRaises(ValueError): M.validate_extra_chain({**value,key:False}, {})
                for changed in ({"nuclear_or_CPHF_included":True}, {"RHF_producers":[]}, {"verified_artifacts":[]},
                        {"verified_artifacts":value["verified_artifacts"]*2}):
                    with self.assertRaises(ValueError): M.validate_extra_chain({**value,**changed}, {})
                for changed in ({"closed_ledger_rows":[]}, {"closed_ledger_rows":[{**value["closed_ledger_rows"][0],"closed_row_sha256":"0"*64}]}):
                    with self.assertRaises(ValueError): M.validate_extra_chain({**value,**changed}, {})
                ledger.write_text(json.dumps({"phases":[{**row,"status":"PARTIAL_OR_FAILED","exit_code":-9}]}))
                with self.assertRaises(ValueError): M.validate_extra_chain(value, {})
                ledger.write_text(json.dumps({"phases":[row]}))
                p.write_text('{"synthetic_metadata":false}\n')
                with self.assertRaises(ValueError): M.validate_extra_chain(value, {})

    def test_signed_modes_and_minimum_qualification_are_not_conflated(self):
        v = {"imaginary_frequency_count":1,"all_frequencies_real_positive":False,
             "stationary_minimum_verified":False,"frequency_validation_passed":False}
        self.assertFalse(M.validate_minimum_flags(v, [-1., 2., 3.]))
        self.assertFalse(M.mechanism_qualified([-1.]+[1.]*218, control=False, fresh_gradient_verified=True))
        self.assertFalse(M.mechanism_qualified([1.]*219, control=True, fresh_gradient_verified=True))
        self.assertTrue(M.mechanism_qualified([1.]*219, control=False, fresh_gradient_verified=True))

    def test_existing_pure_math_algorithms_preserved(self):
        old = {n.name:n for n in ast.parse(OLD.read_text()).body if isinstance(n,ast.FunctionDef)}
        new = {n.name:n for n in ast.parse(SCRIPT.read_text()).body if isinstance(n,ast.FunctionDef)}
        for name in ("eigenvalue_clusters","numpy_runtime","array","independent_subspace_audit",
                     "validate_minimum_flags","compare_actual_control","mechanism_qualified",
                     "v5_metadata_validators","source_gate","control_reference_gate"):
            text = ast.get_source_segment(OLD.read_text(),old[name])
            expected = ast.parse(text.replace("V6b","V7").replace("V6B","V7").replace("v6b","v7")).body[0]
            self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(new[name],include_attributes=False), name)
        with self.assertRaises(ValueError): M.numpy_runtime({})
        self.assertIsNone(M.NUMERICAL)

    def test_actual_native_module_receipt_rejects_wrong_path_hash_or_coverage(self):
        expected={"hessian/rhf.py":"1"*64,"scf/cphf.py":"2"*64}
        receipt={k:{"path":str(M.ROOT/".runtime/quantum-upgrade-site/pyscf"/k),"sha256":v} for k,v in expected.items()}
        with patch.object(M,"expected_native_sources",return_value=expected), patch.object(M,"checked") as check:
            self.assertEqual(M.validate_actual_native_receipt({"actual_imported_native_sources":receipt},None,{})["module_count"],2)
            self.assertEqual(check.call_count,2)
            for bad in ({}, {**receipt,"extra.py":{}}, {**receipt,"scf/cphf.py":{**receipt["scf/cphf.py"],"sha256":"0"*64}},
                        {**receipt,"scf/cphf.py":{**receipt["scf/cphf.py"],"path":"/wrong/scf/cphf.py"}}):
                with self.assertRaises(ValueError):
                    M.validate_actual_native_receipt({"actual_imported_native_sources":bad},None,{})

    def test_occupied_space_comes_from_source_electrons_not_response_claim(self):
        for electrons, expected in ((320,160),(10,5),(26,13)):
            self.assertEqual(M.closed_shell_occupied_space({"nelectron":electrons,"charge":0,"spin":0}), expected)
        for changed in ({"nelectron":319},{"nelectron":True},{"nelectron":0},{"spin":2},{"charge":1}):
            with self.assertRaises(ValueError):
                M.closed_shell_occupied_space({"nelectron":320,"charge":0,"spin":0,**changed})

    def test_runtime_formula_gate_rejects_scientific_changes(self):
        checked=M.numerical_formula_source_gate()
        self.assertEqual(checked["status"],"UNCHANGED_FROZEN_NUMERICAL_FORMULAS_V7_LABELS_ONLY")
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"changed.py"
            p.write_text(SCRIPT.read_text().replace("def eigenvalue_clusters(values, *, atol=1e-8, rtol=1e-6)",
                "def eigenvalue_clusters(values, *, atol=1e-7, rtol=1e-6)"))
            with self.assertRaisesRegex(ValueError,"Unreviewed numerical"):
                M.numerical_formula_source_gate(p)

    def test_real_q_lock_is_always_requested_for_actual_analysis(self):
        events=[]
        class Lock:
            def __enter__(self):events.append("lock")
            def __exit__(self,*exc):events.append("unlock")
        with patch.object(M,"exclusive_closed_quantum",return_value=Lock()), \
                patch.object(M,"_analyze_closed",side_effect=lambda *a,**kw:events.append("analyze")):
            M.analyze("unused","0"*64,"unused")
        self.assertEqual(events,["lock","analyze","unlock"])


if __name__ == "__main__":
    unittest.main()
