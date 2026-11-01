"""Pure pipeline state-machine mocks; no supervisor, transfer or QM execution."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pyscf_v3_pipeline", ROOT / "scripts/run_pyscf_v3_pipeline_20261031.py")
pipeline = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(pipeline)


def settings():
    return {**pipeline.PHYSICS, "optimizer_thresholds": pipeline.THRESHOLDS,
            "threads": 4, "native_direct_driver_threads": 4}


def add_closed(ledger, output, phase, command, log, code=0, cpu=1):
    log.parent.mkdir(parents=True, exist_ok=True); log.write_text("Synthetic closed driver, never executed\n")
    proof = Path(str(log) + ".supervisor-proof.json"); pipeline.write(proof, {"unit_fixture_only": True})
    row = {"phase": phase, "status": "EXITED" if code == 0 else "PARTIAL_OR_FAILED", "exit_code": code,
           "ended_at_utc": pipeline.utc(), "actual_cpu_seconds": cpu, "command": command,
           "log": str(log), "log_sha256": pipeline.sha(log), "supervisor_proof_path": str(proof),
           "supervisor_proof_sha256": pipeline.sha(proof), "pid": 12345}
    ledger["phases"].append(row); ledger["consumed_cpu_seconds"] += cpu
    return row


def fixture(project):
    root = project / "run"; root.mkdir(); inputs = project / "inputs"; inputs.mkdir()
    coords = np.asarray([[i * .07, (i % 4) * .2, (i % 3) * .3] for i in range(75)])
    xyz = inputs / "synthetic.xyz"; precise = inputs / "synthetic.npy"
    symbols = ["C"] * 28 + ["H"] * 32 + ["O"] * 15
    xyz.write_text("75\nSynthetic state machine fixture, never a molecular calculation\n" +
        "\n".join(f"{s} {point[0]:.14f} {point[1]:.14f} {point[2]:.14f}" for s, point in zip(symbols, coords)) + "\n")
    np.save(precise, coords); checkpoint = inputs / "source.chk"; checkpoint.write_bytes(b"Not HDF5: synthetic hash-only fixture")
    provenance = inputs / "provenance.json"
    pipeline.write(provenance, {"identity_verified": True, "stereochemistry_verified": True,
        "source_url": "https://example.invalid/unit-test-only", "input_xyz_sha256": pipeline.sha(xyz),
        "input_coordinates_angstrom_npy_sha256": pipeline.sha(precise)})
    auth = root / "authorization.json"; pipeline.write(auth, {"unit_fixture_only": True})
    ledger = {"schema": "research4-pyscf-monitored-budget-v3", "source_base_commit": "synthetic",
        "authorization_path": str(auth), "quantum_cpu_cap_seconds": 172800, "consumed_cpu_seconds": 0.,
        "run_deadline_at_utc": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(), "phases": []}
    common = {"status": "DONE", "settings": settings(), "geometry_optimized": True,
        "optimizer_converged": True, "scf_converged": True, "final_gradient_thresholds_passed": True,
        "stationary_minimum_verified": True, "atom_rms_hartree_per_bohr": 0., "atom_max_hartree_per_bohr": 0.}
    water = root / "water"; water.mkdir()
    owner = add_closed(ledger, water, "water", ["fixture", "--output", str(water)], root / "execution/water.log")
    pipeline.write(water / "result.json", {**common, "identity": "water", "atoms": 3,
        "frequencies_status": "DONE", "frequency_validation_passed": True, "normal_modes_complete": True,
        "all_frequencies_real_positive": True, "normal_mode_count": 3, "imaginary_frequency_count": 0,
        "frequencies_cm1": [100., 200., 300.], "script_sha256": pipeline.sha(pipeline.WORKER),
        "resource_supervision": {"proof_sha256": owner["supervisor_proof_sha256"]}})
    ethanol = root / "ethanol"; ethanol.mkdir()
    owner = add_closed(ledger, ethanol, "ethanol", ["fixture", "--output", str(ethanol)], root / "execution/ethanol.log")
    cases = [{"case": name, "response_block_atoms": block, "native_direct_driver_threads": native,
              "numerical_equivalence_passed": True, "candidate_all21_frequencies_positive": True,
              "all_atom_pair_cross_couplings_compared": True, "checks": {"synthetic": True}}
             for name, block, native in [("standard_full_native1", None, 1), ("response_block4_native4", 4, 4), ("response_block8_native4", 8, 4)]]
    pipeline.write(ethanol / "result.json", {**common, "identity": "validation_ethanol", "validation_only": True,
        "atoms": 9, "electrons": 26, "numerical_equivalence_passed": True, "all_cases_positive21_frequency_minimum": True,
        "script_sha256": pipeline.sha(pipeline.ETHANOL_VALIDATOR), "engineering_helper_source_sha256": pipeline.sha(pipeline.WORKER),
        "resource_supervision": {"proof_sha256": owner["supervisor_proof_sha256"]}, "comparisons": cases})
    ledger_path = root / "cpu_budget.json"; pipeline.write(ledger_path, ledger)
    args = SimpleNamespace(run_root=root, ledger=ledger_path, water_result=water / "result.json", ethanol_result=ethanol / "result.json",
        input_xyz=xyz, input_coordinates_npy=precise, provenance_json=provenance, checkpoint=checkpoint,
        output=root / "controller", protocol_file=[], opt_phase="unit-opt-" + project.name,
        frequency_phase="unit-freq-" + project.name, opt_cpu_seconds=21600, opt_wall_seconds=10800,
        frequency_cpu_seconds=86400, frequency_wall_seconds=43200)
    return args


def fake_validation(ledger):
    if any(row["status"] == "RUNNING" for row in ledger["phases"]): raise RuntimeError("Synthetic budget still RUNNING")
    return {"max_phase_cpu_seconds": 86400, "max_phase_wall_seconds": 43200}


class FakeExecution:
    def __init__(self, args, opt_failure=False, missing_opt=False, bad_gradient=False,
                 negative_freq=False, exhausted_after_opt=False, mutate_source=False):
        self.args = args; self.commands = []; self.prepared = 0
        self.opt_failure, self.missing_opt, self.bad_gradient = opt_failure, missing_opt, bad_gradient
        self.negative_freq, self.exhausted_after_opt, self.mutate_source = negative_freq, exhausted_after_opt, mutate_source

    def launch(self, command, **kwargs):
        self.commands.append(command)
        if command[1] != str(pipeline.SUPERVISOR) or "--initialize" in command:
            raise AssertionError("A quantum payload was launched without formal supervisor")
        marker = command.index("--"); parent, payload = command[:marker], command[marker+1:]
        get = lambda values, option: values[values.index(option) + 1]
        kind = get(payload, "--phase"); output = Path(get(payload, "--output")); output.mkdir()
        if kind == "spinosin-frequency":
            inflight = pipeline.read(self.args.output / "state.json")
            assert inflight["frequency_started"] is None
            assert inflight["frequency_execution_observation"] == "UNKNOWN_UNTIL_LEDGER_OBSERVED"
            assert inflight["frequency_stage_requested"] is True
        code = 1 if (kind == "spinosin-optimize" and self.opt_failure) or (kind == "spinosin-frequency" and self.negative_freq) else 0
        if kind == "spinosin-optimize" and not self.missing_opt:
            gradient = np.full((75, 3), .01 if self.bad_gradient else 0.)
            np.save(output / "gradient_hartree_per_bohr.npy", gradient)
            xyz = output / "optimized.xyz"; xyz.write_bytes(self.args.input_xyz.read_bytes())
            coords = output / "optimized_coordinates_angstrom.npy"; coords.write_bytes(self.args.input_coordinates_npy.read_bytes())
            checkpoint = output / "calculation.chk"; checkpoint.write_bytes(b"Synthetic optimized checkpoint, never loaded")
            result = {"identity": "spinosin", "phase": kind, "atoms": 75, "electrons": 320,
                "status": "PARTIAL" if self.opt_failure else "DONE", "settings": settings(),
                "geometry_optimized": not self.opt_failure, "optimizer_converged": not self.opt_failure,
                "scf_converged": True, "final_gradient_thresholds_passed": True,
                "atom_rms_hartree_per_bohr": 0., "atom_max_hartree_per_bohr": 0.,
                "checkpoint_sha256": pipeline.sha(checkpoint), "optimized_xyz_sha256": pipeline.sha(xyz),
                "optimized_coordinates_angstrom_npy_sha256": pipeline.sha(coords)}
            pipeline.write(output / "result.json", result)
        elif kind == "spinosin-frequency":
            pipeline.write(output / "result.json", {"identity": "spinosin", "status": "PARTIAL" if self.negative_freq else "DONE",
                "frequencies_status": "DONE", "normal_mode_count": 219, "imaginary_frequency_count": int(self.negative_freq),
                "geometry_optimized": True, "scf_converged": True, "final_gradient_thresholds_passed": True,
                "normal_modes_complete": True, "all_frequencies_real_positive": not self.negative_freq,
                "frequency_validation_passed": not self.negative_freq, "stationary_minimum_verified": not self.negative_freq,
                "frequencies_cm1": [-1.] + [1.] * 218 if self.negative_freq else [1.] * 219})
        ledger = pipeline.read(self.args.ledger)
        add_closed(ledger, output, get(parent, "--phase"), payload, Path(get(parent, "--log")), code=code, cpu=3)
        if kind == "spinosin-optimize" and self.exhausted_after_opt:
            ledger["quantum_cpu_cap_seconds"] = ledger["consumed_cpu_seconds"] + 20
        pipeline.write(self.args.ledger, ledger)
        if kind == "spinosin-optimize" and self.mutate_source: self.args.provenance_json.write_text("{}")
        kwargs["stdout"].write(b"Synthetic supervisor state-machine fixture, no subprocess or QM executed\n")
        return SimpleNamespace(returncode=code)

    def prepare(self, source_result, source_output, ledger, output, purpose, release):
        self.prepared += 1; self.assert_closed = not any(row["status"] == "RUNNING" for row in pipeline.read(ledger)["phases"])
        assert purpose == "frequency" and self.assert_closed
        bound = pipeline.read(release)
        assert bound["cpu_budget_sha256"] == pipeline.sha(ledger)
        assert bound["source_result_sha256"] == pipeline.sha(source_result)
        output.mkdir()
        (output / "transfer_geometry.xyz").write_bytes((source_output / "optimized.xyz").read_bytes())
        (output / "transfer_coordinates_angstrom.npy").write_bytes((source_output / "optimized_coordinates_angstrom.npy").read_bytes())
        pipeline.write(output / "transfer_geometry_provenance.json", {"unit_fixture_only": True})


class PipelineTests(unittest.TestCase):
    def run_fixture(self, **options):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        args = fixture(Path(temp.name)); fake = FakeExecution(args, **options)
        state = pipeline.run(args, launch=fake.launch, validate_budget=fake_validation, transfer_prepare=fake.prepare)
        return args, fake, state

    def test_exactly_two_formal_supervisor_stages_and_bound_transfer_on_success(self):
        args, fake, state = self.run_fixture()
        self.assertEqual(state["status"], "DONE"); self.assertEqual(len(fake.commands), 2); self.assertEqual(fake.prepared, 1)
        self.assertTrue(state["frequency_started"]); self.assertFalse(state["worker_result_json_fabricated"])
        self.assertEqual(state["quantum_payloads_directly_launched_by_controller"], 0)
        for command in fake.commands:
            payload = command[command.index("--")+1:]
            for flag, value in [("--threads", "4"), ("--native-direct-threads", "4"), ("--maxsteps", "35"), ("--hessian-block-atoms", "8")]:
                self.assertEqual(payload[payload.index(flag)+1], value)
        self.assertIn("--optimization-result", fake.commands[1])
        self.assertTrue((args.output / "plan.json").exists())

    def test_optimization_failure_stops_without_frequency_or_transfer(self):
        _, fake, state = self.run_fixture(opt_failure=True)
        self.assertEqual(state["status"], "PARTIAL"); self.assertEqual(len(fake.commands), 1); self.assertEqual(fake.prepared, 0)
        self.assertFalse(state["frequency_stage_requested"])

    def test_native_exit_without_final_result_does_not_fabricate_one(self):
        args, fake, state = self.run_fixture(missing_opt=True)
        self.assertEqual(state["status"], "PARTIAL"); self.assertEqual(fake.prepared, 0)
        self.assertFalse((args.run_root / args.opt_phase / "result.json").exists())

    def test_gradient_array_gate_blocks_false_optimized_metadata(self):
        _, fake, state = self.run_fixture(bad_gradient=True)
        self.assertEqual(state["status"], "PARTIAL"); self.assertEqual(len(fake.commands), 1)
        self.assertIn("gradient", state["stop_reason"])

    def test_negative_frequencies_preserved_and_pipeline_partial(self):
        args, fake, state = self.run_fixture(negative_freq=True)
        self.assertEqual(state["status"], "PARTIAL"); self.assertEqual(len(fake.commands), 2)
        self.assertEqual(state["actual_frequency_summary"]["imaginary_frequency_count"], 1)
        self.assertEqual(pipeline.read(args.run_root / args.frequency_phase / "result.json")["frequencies_cm1"][0], -1.)

    def test_budget_exhaustion_never_initializes_or_launches_an_extra_stage(self):
        _, fake, state = self.run_fixture(exhausted_after_opt=True)
        self.assertEqual(state["status"], "PARTIAL"); self.assertEqual(len(fake.commands), 1)
        self.assertFalse(state["frequency_stage_requested"])

    def test_source_tampering_between_stages_stops_frozen_plan(self):
        _, fake, state = self.run_fixture(mutate_source=True)
        self.assertEqual(state["status"], "PARTIAL"); self.assertEqual(len(fake.commands), 1)
        self.assertIn("source changed", state["stop_reason"])

    def test_failed_ethanol_gate_cannot_launch_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp)); bad = pipeline.read(args.ethanol_result)
            bad["numerical_equivalence_passed"] = False; pipeline.write(args.ethanol_result, bad)
            fake = FakeExecution(args)
            with self.assertRaises(ValueError): pipeline.run(args, launch=fake.launch, validate_budget=fake_validation, transfer_prepare=fake.prepare)
            self.assertEqual(fake.commands, [])

    def test_phase_budget_limits_are_clamped_to_original_remaining_resources(self):
        now = datetime.now(timezone.utc)
        ledger = {"phases": [{"actual_cpu_seconds": 100.}], "consumed_cpu_seconds": 100.,
                  "quantum_cpu_cap_seconds": 300., "run_deadline_at_utc": (now + timedelta(seconds=180)).isoformat()}
        result = pipeline.finite_stage_budget(ledger, {"max_phase_cpu_seconds": 86400, "max_phase_wall_seconds": 43200}, 86400, 43200, now)
        self.assertEqual(result["cpu_seconds"], 190); self.assertEqual(result["wall_seconds"], 178)
        self.assertFalse(result["budget_enlarged"])
        with self.assertRaises(ValueError): pipeline.finite_stage_budget(ledger, {}, 86401, 43200, now)
        ledger["quantum_cpu_cap_seconds"] = 110
        with self.assertRaises(RuntimeError): pipeline.finite_stage_budget(ledger, {"max_phase_cpu_seconds": 86400, "max_phase_wall_seconds": 43200}, 86400, 43200, now)


if __name__ == "__main__":
    unittest.main()
