"""Stdlib source/proof/argv contracts only; no quantum work or target arrays."""
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

PATH = Path(__file__).resolve().parents[1] / "scripts/pyscf_xc_chunk_worker_20261105_v7.py"
SPEC = importlib.util.spec_from_file_location("v7_XC_worker_pure_contracts", PATH)
WORKER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(WORKER)


class WorkerContracts(unittest.TestCase):
    def test_top_level_and_metadata_dependencies_stay_cold(self):
        core, guard, previous = WORKER.metadata_modules()
        self.assertEqual(WORKER.NUMERICAL_POLICY, previous.NUMERICAL_POLICY)
        self.assertEqual(WORKER.PHYSICAL_SCHEMA, previous.PHYSICAL_SCHEMA)
        self.assertNotEqual(WORKER.SCHEMA, previous.SCHEMA)
        self.assertEqual(core.POLICY["xc"], "PBE")
        for name in ("numpy", "scipy", "pyscf", "torch"):
            self.assertFalse(any(k == name or k.startswith(name + ".") for k in sys.modules))

    def test_actual_parent_gate_precedes_output_and_numerical_import(self):
        class Guard:
            def verify_resource_supervision(self, **kwargs): raise ValueError("synthetic missing actual parent")
        with patch.object(WORKER, "metadata_modules", return_value=(None, Guard(), None)):
            with self.assertRaisesRegex(ValueError, "actual parent"):
                WORKER.execute(types.SimpleNamespace(threads=2))
        self.assertNotIn("numpy", sys.modules)

    def test_spinosin_without_real_control_release_is_blocked_before_import(self):
        with self.assertRaises(ValueError): WORKER.require_control_release(None, None)
        with tempfile.TemporaryDirectory() as directory, patch.object(WORKER, "ROOT", Path(directory)):
            path = Path(directory) / "not_authority.json"
            path.write_text(json.dumps({"schema": "research4-v7-actual-small-control-release-v1",
                "root_execution_authorized": True, "v7_actual_water_ethanol_controls_reviewed": False, "no_budget_increase": True}))
            with self.assertRaises(ValueError): WORKER.require_control_release(path, WORKER.sha(path))
        self.assertNotIn("numpy", sys.modules)

    def command_fixture(self, directory, *, control=True, stage="xc-diag-chunk"):
        root = Path(directory); out = root / "report"; out.mkdir()
        proof = root / "proof.json"; proof.write_text('{}\n')
        log = root / "log.txt"; log.write_text('synthetic receipt only, no scientific calculation\n')
        release = root / "release.json"; release.write_text('{}\n')
        ledger = root / "ledger.json"
        source = str(root / "snapshot"); gradient = str(root / "G.json")
        args = {"execute": True, "stage": stage, "threads": 2, "control_validation": control,
            "snapshot": source, "snapshot_manifest_sha256": "s" * 64,
            "gradient_check_result": gradient, "gradient_check_result_sha256": "g" * 64,
            "plan_result": str(root / "plan/result.json"), "plan_result_sha256": "p" * 64,
            "start_block": 3, "block_count": 2,
            "control_release": str(release), "control_release_sha256": WORKER.sha(release)}
        command = [str(WORKER.SELF), "--execute", "--stage", stage, "--output", str(out), "--threads", "2",
            "--snapshot", source, "--snapshot-manifest-sha256", "s" * 64,
            "--gradient-check-result", gradient, "--gradient-check-result-sha256", "g" * 64]
        if stage in ("xc-diag-chunk", "xc-deriv2-chunk"):
            command += ["--plan-result", args["plan_result"], "--plan-result-sha256", "p" * 64,
                "--start-block", "3", "--block-count", "2"]
        if control: command += ["--control-validation"]
        else: command += ["--control-release", str(release), "--control-release-sha256", WORKER.sha(release)]
        row = {"phase": "synthetic_closed_phase", "status": "EXITED", "command": command, "log": str(log)}
        ledger.write_text(json.dumps({"phases": [row]}))
        value = {"stage": stage, "settings": {"threads": 2}, "snapshot_manifest_sha256": "s" * 64,
            "gradient_gate": {"result": {"path": gradient, "sha256": "g" * 64}},
            "resource_supervision": {"proof_path": str(proof), "proof": {"ledger_path": str(ledger), "phase": row["phase"]}},
            "actual_control_release": WORKER.artifact(release) if not control else None}
        result = out / "result.json"; result.write_text(json.dumps(value))
        (out / "input.json").write_text(json.dumps({"arguments": args}))
        closure = {"producer_result_sha256": WORKER.sha(result), "proof_sha256": WORKER.sha(proof),
            "log_sha256": WORKER.sha(log), "closed_row_sha256": "x" * 64}
        guard = types.SimpleNamespace(healthy_closed_row=lambda r: r["status"] == "EXITED")
        return out, value, closure, guard, ledger

    def test_actual_chunk_command_binds_range_plan_source_and_threads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(WORKER, "ROOT", Path(directory)):
            out, value, closure, guard, ledger = self.command_fixture(directory)
            args, records, row = WORKER.component_command(value, out / "result.json", closure, guard)
            self.assertEqual((args["start_block"], args["block_count"]), (3, 2))
            self.assertEqual(len(records), 4)
            self.assertEqual(row["hash_style"], "json_sort_keys_default")
            original = json.loads(ledger.read_text())
            for flag, wrong in (("--start-block", "4"), ("--block-count", "1"), ("--plan-result-sha256", "0" * 64),
                    ("--gradient-check-result-sha256", "0" * 64), ("--threads", "1")):
                data = deepcopy(original); command = data["phases"][0]["command"]; command[command.index(flag) + 1] = wrong
                ledger.write_text(json.dumps(data))
                with self.subTest(flag=flag), self.assertRaises(ValueError): WORKER.component_command(value, out / "result.json", closure, guard)

    def test_target_control_release_is_actual_argv_bound_and_retained(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(WORKER, "ROOT", Path(directory)):
            out, value, closure, guard, ledger = self.command_fixture(directory, control=False)
            _, records, _ = WORKER.component_command(value, out / "result.json", closure, guard)
            self.assertEqual(records[-1], value["actual_control_release"])
            data = json.loads(ledger.read_text()); command = data["phases"][0]["command"]
            command[command.index("--control-release-sha256") + 1] = "0" * 64
            ledger.write_text(json.dumps(data))
            with self.assertRaises(ValueError): WORKER.component_command(value, out / "result.json", closure, guard)

    def test_final_partial_incomplete_or_legacy_result_never_opens_components(self):
        value = {"complete_RHF_helper": False, "complete_RKS_partial": False}
        with patch.object(WORKER, "qualify_component", return_value=(value, {}, {}, [], {})), \
                patch.object(WORKER, "_assembly_sources", side_effect=AssertionError("must not open components")) as sources:
            with self.assertRaises(ValueError): WORKER.qualify_partial({}, {}, 75, 762, None)
            sources.assert_not_called()

    def test_component_public_status_and_namespace_are_explicitly_v7(self):
        self.assertEqual(WORKER.SUCCESS["rks-partial"], "STAGE_COMPLETE")
        self.assertEqual(WORKER.CACHE_SCHEMA, "research4-complete-full-rks-stage-cache-v7")
        self.assertEqual(WORKER.REPORT_ROOT.name, "pyscf-spinosin-grid-checkpoint-20261105-v7")
        self.assertEqual(WORKER.CACHE_ROOT.name, "pyscf-spinosin-grid-checkpoint-20261105-v7")
        self.assertNotIn("f137", WORKER.SCHEMA)

    def test_source_deferred_imports_and_required_actual_stage_controls(self):
        tree = ast.parse(PATH.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import): self.assertFalse(any(v.name.split(".")[0] in {"numpy", "pyscf", "torch", "scipy"} for v in node.names))
            if isinstance(node, ast.ImportFrom): self.assertNotIn((node.module or "").split(".")[0], {"numpy", "pyscf", "torch", "scipy"})
        functions = {n.name: ast.get_source_segment(PATH.read_text(), n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertLess(functions["execute"].index("guard.verify_resource_supervision"), functions["execute"].index("import numpy as np"))
        self.assertLess(functions["execute"].index("require_control_release(args.control_release"), functions["execute"].index("import numpy as np"))
        self.assertIn("args.threads == 2", functions["execute"])
        self.assertIn("core.grid_binding(hobj, np, base.array_fingerprint) == grid", functions["execute"])
        self.assertIn("not args.control_validation", functions["execute"])
        self.assertIn("max_new_units=1", functions["execute"])


if __name__ == "__main__": unittest.main()
