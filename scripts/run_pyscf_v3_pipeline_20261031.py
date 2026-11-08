"""Finite V3 control plane: gated supervised optimization then frequency.

This program never evaluates SCF/gradients/Hessians itself. Only the formal V3
supervisor subprocess launches the two fixed quantum payloads. Geometry transfer
is a read-only, provenance-bound preparation call after optimization has exited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".runtime/bin/python"
SUPERVISOR = ROOT / "scripts/run_pyscf_monitored_budget_20261031.py"
WORKER = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
TRANSFER = ROOT / "scripts/prepare_pyscf_geometry_transfer_20261031.py"
ETHANOL_VALIDATOR = ROOT / "scripts/validate_pyscf_native4_large_block_engineering_20261031.py"
MAX_PHASE_CPU, MAX_PHASE_WALL = 86400, 43200
THRESHOLDS = {"convergence_energy": 1e-6, "convergence_grms": 3e-4,
              "convergence_gmax": 4.5e-4, "convergence_drms": .0012, "convergence_dmax": .0018}
PHYSICS = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1,
           "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "density_fitting": False,
           "density_fit_backend": "SCREENED_DIRECT_RKS_V4", "auxbasis": None,
           "direct_scf": True, "direct_scf_tol": 1e-13, "scf_conv_tol_hartree": 1e-8,
           "global_four_center_ERI_cache_allowed": False, "charge": 0, "spin_2S": 0, "multiplicity": 1}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_hashes(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("Frozen pipeline source changed: " + str(path))


def physical_method(result):
    settings = result.get("settings", {})
    return (all(settings.get(key) == value for key, value in PHYSICS.items())
            and settings.get("optimizer_thresholds") == THRESHOLDS)


def owned_closed_phase(ledger, output):
    if any(row.get("status") == "RUNNING" for row in ledger["phases"]):
        raise ValueError("A quantum stage remains active; do not inspect/launch another")
    matches = []
    for row in ledger["phases"]:
        command = row.get("command", [])
        if "--output" in command and resolved(command[command.index("--output") + 1]) == output.resolve():
            matches.append(row)
    if len(matches) != 1 or not matches[0].get("ended_at_utc"):
        raise ValueError("Exactly one ended supervised phase must own the result")
    row = matches[0]
    if row.get("status") != "EXITED" or row.get("exit_code") != 0:
        raise ValueError("Gate/source phase did not exit successfully under its supervisor")
    if sha(resolved(row["log"])) != row["log_sha256"]:
        raise ValueError("Closed phase driver log changed")
    proof = resolved(row["supervisor_proof_path"])
    if sha(proof) != row["supervisor_proof_sha256"]:
        raise ValueError("Closed phase supervisor proof changed")
    return row


def check_gate(result_path, kind, ledger):
    path = Path(result_path).resolve(); result = read(path)
    phase = owned_closed_phase(ledger, path.parent)
    if result.get("status") != "DONE" or not physical_method(result):
        raise ValueError("Gate result is not completed with the fixed physical method")
    for flag in ("geometry_optimized", "optimizer_converged", "scf_converged",
                 "final_gradient_thresholds_passed", "stationary_minimum_verified"):
        if result.get(flag) is not True: raise ValueError("Gate flag failed: " + flag)
    if (not math.isfinite(float(result.get("atom_rms_hartree_per_bohr", math.inf)))
            or not 0 <= result["atom_rms_hartree_per_bohr"] <= .0003
            or not 0 <= result.get("atom_max_hartree_per_bohr", math.inf) <= .00045):
        raise ValueError("Gate fresh gradient exceeds original numerical thresholds")
    proof = result.get("resource_supervision", {})
    if proof.get("proof_sha256") != phase["supervisor_proof_sha256"]:
        raise ValueError("Gate did not record its actual supervised execution")
    if kind == "water":
        if (result.get("identity") != "water" or result.get("atoms") != 3
                or result.get("frequencies_status") != "DONE"
                or not all(result.get(flag) is True for flag in ("frequency_validation_passed", "normal_modes_complete", "all_frequencies_real_positive"))
                or result.get("normal_mode_count") != 3 or result.get("imaginary_frequency_count") != 0
                or result.get("script_sha256") != sha(WORKER)):
            raise ValueError("Actual V8 water Opt/Freq gate failed")
        frequencies = np.asarray(result.get("frequencies_cm1", []), float)
        if frequencies.shape != (3,) or not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
            raise ValueError("Water gate lacks its three actual positive frequencies")
    elif kind == "ethanol":
        if (result.get("identity") != "validation_ethanol" or result.get("validation_only") is not True
                or result.get("atoms") != 9 or result.get("electrons") != 26
                or result.get("numerical_equivalence_passed") is not True
                or result.get("all_cases_positive21_frequency_minimum") is not True
                or result.get("script_sha256") != sha(ETHANOL_VALIDATOR)
                or result.get("engineering_helper_source_sha256") != sha(WORKER)):
            raise ValueError("Actual native4 ethanol equivalence/positive-minimum gate failed")
        expected = [("standard_full_native1", None, 1), ("response_block4_native4", 4, 4), ("response_block8_native4", 8, 4)]
        comparisons = result.get("comparisons", [])
        if len(comparisons) != 3: raise ValueError("All three predeclared ethanol cases required")
        for row, (name, block, native) in zip(comparisons, expected):
            if (row.get("case") != name or row.get("response_block_atoms") != block
                    or row.get("native_direct_driver_threads") != native
                    or row.get("numerical_equivalence_passed") is not True
                    or row.get("candidate_all21_frequencies_positive") is not True
                    or row.get("all_atom_pair_cross_couplings_compared") is not True
                    or not row.get("checks") or not all(row["checks"].values())):
                raise ValueError("A required full/native1 vs B4/B8/native4 comparison failed")
    else:
        raise ValueError("Unknown fixed pipeline gate")
    return {"result": str(path), "result_sha256": sha(path), "phase": phase["phase"],
            "identity": result["identity"], "status": "GATE_PASSED"}


def input_geometry(args):
    provenance = read(args.provenance_json)
    lines = Path(args.input_xyz).read_text().splitlines(); atoms = [line.split() for line in lines[2:] if line.strip()]
    if int(lines[0]) != 75 or len(atoms) != 75 or any(len(row) != 4 for row in atoms):
        raise ValueError("Explicit complete75-atom optimization XYZ required")
    coordinates = np.load(args.input_coordinates_npy, allow_pickle=False)
    rounded = np.asarray([row[1:] for row in atoms], float)
    if (Counter(row[0] for row in atoms) != Counter({"C": 28, "H": 32, "O": 15})
            or coordinates.shape != (75, 3) or not np.isfinite(coordinates).all()
            or not np.allclose(coordinates, rounded, atol=5e-12, rtol=0)):
        raise ValueError("Source fullprecision coordinates/composition differ from supplied XYZ")
    if (provenance.get("identity_verified") is not True or provenance.get("stereochemistry_verified") is not True
            or not provenance.get("source_url") or provenance.get("input_xyz_sha256") != sha(args.input_xyz)
            or provenance.get("input_coordinates_angstrom_npy_sha256") != sha(args.input_coordinates_npy)):
        raise ValueError("SHA-bound verified prepared optimization provenance required")


def finite_stage_budget(ledger, authorization, cpu_request, wall_request, now=None):
    if type(cpu_request) is not int or not 65 <= cpu_request <= MAX_PHASE_CPU:
        raise ValueError("Finite stage CPU request must remain65..86400seconds")
    if type(wall_request) is not int or not 61 <= wall_request <= MAX_PHASE_WALL:
        raise ValueError("Finite stage wall request must remain61..43200seconds")
    used = [float(row["actual_cpu_seconds"]) for row in ledger["phases"]]
    if not all(math.isfinite(value) and value >= 0 for value in used) or not math.isclose(sum(used), ledger["consumed_cpu_seconds"], abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError("Actual accumulated CPU differs from recorded ledger")
    remaining_cpu = float(ledger["quantum_cpu_cap_seconds"]) - sum(used)
    current = now or datetime.now(timezone.utc)
    remaining_wall = (datetime.fromisoformat(ledger["run_deadline_at_utc"]) - current).total_seconds()
    cpu = min(cpu_request, authorization["max_phase_cpu_seconds"], math.floor(remaining_cpu)-10)
    wall = min(wall_request, authorization["max_phase_wall_seconds"], math.floor(remaining_wall)-2)
    if cpu < 65 or wall < 61:
        raise RuntimeError("PARTIAL: insufficient original remaining CPU/wall budget; no new stage launched")
    return {"cpu_seconds": cpu, "wall_seconds": wall, "original_remaining_cpu_seconds": remaining_cpu,
            "original_remaining_wall_seconds": remaining_wall, "budget_enlarged": False}


def worker_command(phase, xyz, precise, provenance, checkpoint, output, cache, optimization_result=None):
    command = [str(PYTHON), str(WORKER), "--phase", phase, "--input-xyz", str(xyz),
        "--input-coordinates-npy", str(precise), "--expected-coordinates-sha256", sha(precise),
        "--provenance-json", str(provenance), "--expected-input-sha256", sha(xyz),
        "--checkpoint", str(checkpoint), "--output", str(output), "--grid-level", "1",
        "--threads", "4", "--native-direct-threads", "4", "--maxsteps", "35",
        "--hessian-block-atoms", "8", "--hessian-cache-dir", str(cache)]
    if optimization_result is not None: command += ["--optimization-result", str(optimization_result)]
    return command


def supervised_command(ledger, phase, limits, driver_log, payload):
    command = [str(PYTHON), str(SUPERVISOR), "--ledger", str(ledger), "--phase", phase,
        "--cpu-seconds", str(limits["cpu_seconds"]), "--wall-seconds", str(limits["wall_seconds"]),
        "--threads", "4", "--memory-gib", "24", "--log", str(driver_log), "--", *payload]
    if "--initialize" in command or command[1] != str(SUPERVISOR):
        raise ValueError("Controller cannot initialize budgets or directly launch quantum payloads")
    return command


def optimization_gate(result_path):
    result = read(result_path)
    flags = ("geometry_optimized", "optimizer_converged", "scf_converged", "final_gradient_thresholds_passed")
    if (result.get("identity") != "spinosin" or result.get("phase") != "spinosin-optimize"
            or result.get("status") != "DONE" or not all(result.get(key) is True for key in flags)
            or not physical_method(result) or result.get("atoms") != 75 or result.get("electrons") != 320):
        raise ValueError("Actual native4 optimization/fresh-gradient gate failed; frequency NOT_RUN")
    gradient = np.load(result_path.parent / "gradient_hartree_per_bohr.npy", allow_pickle=False)
    if gradient.shape != (75, 3) or not np.isfinite(gradient).all(): raise ValueError("Incomplete final optimization gradient")
    norms = np.linalg.norm(gradient, axis=1); rms, maximum = float(np.sqrt(np.mean(norms**2))), float(norms.max())
    if (rms > .0003 or maximum > .00045 or not math.isclose(rms, result["atom_rms_hartree_per_bohr"], abs_tol=1e-13, rel_tol=1e-11)
            or not math.isclose(maximum, result["atom_max_hartree_per_bohr"], abs_tol=1e-13, rel_tol=1e-11)):
        raise ValueError("Saved actual optimization gradient does not meet original thresholds")
    if result.get("settings", {}).get("native_direct_driver_threads") != 4 or result["settings"].get("threads") != 4:
        raise ValueError("Optimization did not execute the fixed native4/outer4 pipeline")
    return result


def run(args, launch=subprocess.run, validate_budget=None, transfer_prepare=None):
    args.run_root, args.ledger, args.output = resolved(args.run_root), resolved(args.ledger), resolved(args.output)
    if args.ledger.parent != args.run_root or not args.output.is_relative_to(args.run_root):
        raise ValueError("Controller and ledger must remain in the explicit V3 run root")
    if args.output.exists(): raise FileExistsError("Use a new controller directory; never resume/repeat implicitly")
    for name in (args.opt_phase, args.frequency_phase):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name): raise ValueError("Safe fresh phase identifier required")
    if args.opt_phase == args.frequency_phase: raise ValueError("Optimization and frequency phase IDs must differ")
    if validate_budget is None: validate_budget = load_module(SUPERVISOR, "pipeline_official_budget").validate_ledger
    ledger = read(args.ledger); authorization = validate_budget(ledger)
    if ledger.get("schema") != "research4-pyscf-monitored-budget-v3": raise ValueError("Only the new monitored V3 ledger is accepted")
    gate_records = [check_gate(args.water_result, "water", ledger), check_gate(args.ethanol_result, "ethanol", ledger)]
    input_geometry(args)
    outputs = {"optimization": args.run_root / args.opt_phase, "frequency": args.run_root / args.frequency_phase}
    existing_names = [row["phase"] for row in ledger["phases"]]
    if any(name in existing_names for name in (args.opt_phase, args.frequency_phase)) or any(path.exists() for path in outputs.values()):
        raise FileExistsError("Do not reuse a previous quantum phase or its output")
    cache = ROOT / "outputs" / args.run_root.name / "pipeline-Hessian-cache" / args.frequency_phase
    if cache.exists(): raise FileExistsError("Do not overwrite an earlier SAVE_ONLY Hessian cache")
    source_paths = [Path(__file__), SUPERVISOR, WORKER, TRANSFER, ETHANOL_VALIDATOR,
                    Path(args.input_xyz), Path(args.input_coordinates_npy), Path(args.provenance_json), Path(args.checkpoint),
                    resolved(ledger["authorization_path"]), *[Path(path) for path in args.protocol_file]]
    for record in gate_records:
        source_paths.extend(path for path in Path(record["result"]).parent.rglob("*") if path.is_file())
    bindings = {str(path.resolve()): sha(path) for path in source_paths}
    protocol = {"physical_method": PHYSICS, "thresholds": THRESHOLDS, "outer_threads": 4, "native_threads": 4,
        "maxsteps": 35, "Hessian_response_block_atoms": 8, "maximum_quantum_phases": 2,
        "optimization_requested_cpu_seconds": args.opt_cpu_seconds, "optimization_requested_wall_seconds": args.opt_wall_seconds,
        "frequency_requested_cpu_seconds": args.frequency_cpu_seconds, "frequency_requested_wall_seconds": args.frequency_wall_seconds,
        "no_parameter_switch_or_retry": True, "no_budget_initialization_or_enlargement": True}
    args.output.mkdir(parents=True)
    plan = {"created_at_utc": utc(), "controller_source_sha256": sha(__file__), "gates": gate_records,
        "ledger_path": str(args.ledger), "ledger_sha256_at_start": sha(args.ledger),
        "source_base_commit": ledger["source_base_commit"], "physical_backend_protocol": protocol,
        "protocol_sha256": value_sha(protocol), "source_sha256": bindings,
        "optimization_output": str(outputs["optimization"]), "frequency_output": str(outputs["frequency"]),
        "Hessian_cache_root_ignored_SAVE_ONLY": str(cache)}
    write(args.output / "plan.json", plan)
    state = {"status": "RUNNING", "started_at_utc": utc(), "controller_pid": os.getpid(),
             "plan_sha256": sha(args.output / "plan.json"), "quantum_payloads_directly_launched_by_controller": 0,
             "supervisor_stages_requested": [], "frequency_started": False,
             "frequency_stage_requested": False, "frequency_execution_observation": "NOT_REQUESTED",
             "in_flight_worker_status_authority": str(args.ledger), "worker_result_json_fabricated": False}
    write(args.output / "state.json", state)
    def stage(kind, phase_name, payload, cpu_request, wall_request):
        verify_hashes(bindings)
        current = read(args.ledger); auth = validate_budget(current)
        limits = finite_stage_budget(current, auth, cpu_request, wall_request)
        driver_log = args.run_root / "execution" / (phase_name + ".log")
        command = supervised_command(args.ledger, phase_name, limits, driver_log, payload)
        command_record = {"created_at_utc": utc(), "kind": kind, "argv": command, "argv_sha256": value_sha(command),
                          "limits_requested_within_original_budget": limits, "source_plan_sha256": state["plan_sha256"]}
        write(args.output / (kind + "_command.json"), command_record)
        state["supervisor_stages_requested"].append({"kind": kind, "phase": phase_name, "command_sha256": command_record["argv_sha256"]})
        state["current_stage"] = kind
        state["frequency_stage_requested"] = kind == "frequency" or state["frequency_stage_requested"]
        if kind == "frequency":
            state["frequency_started"] = None
            state["frequency_execution_observation"] = "UNKNOWN_UNTIL_LEDGER_OBSERVED"
        write(args.output / "state.json", state)
        with (args.output / (kind + "_supervisor.log")).open("xb") as log:
            completed = launch(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        current = read(args.ledger)
        observed = [row for row in current["phases"] if row["phase"] == phase_name]
        if kind == "frequency":
            state["frequency_started"] = bool(len(observed) == 1 and observed[0].get("pid"))
            state["frequency_execution_observation"] = ("STARTED_PROCESS_OBSERVED_IN_CLOSED_LEDGER" if state["frequency_started"]
                                                         else "NO_WORKER_PROCESS_OBSERVED")
        execution = {"kind": kind, "supervisor_returncode": completed.returncode,
                     "closed_phase": observed[0] if len(observed) == 1 else None,
                     "supervisor_output_sha256": sha(args.output / (kind + "_supervisor.log"))}
        write(args.output / (kind + "_execution.json"), execution)
        if completed.returncode != 0: raise RuntimeError("Actual supervised " + kind + " stage failed or remained PARTIAL")
        owned_closed_phase(current, outputs[kind]); return current
    try:
        payload = worker_command("spinosin-optimize", resolved(args.input_xyz), resolved(args.input_coordinates_npy),
            resolved(args.provenance_json), resolved(args.checkpoint), outputs["optimization"], cache)
        current = stage("optimization", args.opt_phase, payload, args.opt_cpu_seconds, args.opt_wall_seconds)
        result_path = outputs["optimization"] / "result.json"
        if not result_path.is_file(): raise RuntimeError("Optimization has no final worker result; no result.json will be fabricated")
        optimized = optimization_gate(result_path)
        if sha(outputs["optimization"] / "calculation.chk") != optimized.get("checkpoint_sha256"):
            raise ValueError("Closed optimized checkpoint differs from its recorded SHA")
        verify_hashes(bindings)
        release = args.output / "frequency_geometry_release.json"
        write(release, {"created_at_utc": utc(), "allow_geometry_transfer": True,
                        "cpu_budget_sha256": sha(args.ledger), "source_result_sha256": sha(result_path)})
        if transfer_prepare is None:
            for relative in (".runtime/quantum-upgrade-site", ".runtime/quantum-pilot-site"):
                sys.path.insert(0, str(ROOT / relative))
            transfer_prepare = load_module(TRANSFER, "pipeline_geometry_transfer_only").prepare
        prepared = args.output / "frequency-inputs"
        transfer_prepare(result_path, outputs["optimization"], args.ledger, prepared, "frequency", release)
        frequency_files = {str(path): sha(path) for path in prepared.iterdir() if path.is_file()}
        frequency_files[str(result_path)] = sha(result_path)
        frequency_files[str(outputs["optimization"] / "calculation.chk")] = sha(outputs["optimization"] / "calculation.chk")
        write(args.output / "frequency_inputs_manifest.json", {"created_at_utc": utc(), "source_sha256": frequency_files,
            "geometry_preparation_only_no_quantum_evaluations": True})
        verify_hashes(frequency_files)
        payload = worker_command("spinosin-frequency", prepared / "transfer_geometry.xyz",
            prepared / "transfer_coordinates_angstrom.npy", prepared / "transfer_geometry_provenance.json",
            outputs["optimization"] / "calculation.chk", outputs["frequency"], cache, result_path)
        stage("frequency", args.frequency_phase, payload, args.frequency_cpu_seconds, args.frequency_wall_seconds)
        frequency_result = read(outputs["frequency"] / "result.json")
        state["actual_frequency_summary"] = {key: frequency_result.get(key) for key in
            ("status", "frequencies_status", "normal_mode_count", "imaginary_frequency_count", "frequency_validation_passed", "stationary_minimum_verified")}
        if (frequency_result.get("status") != "DONE" or not all(frequency_result.get(key) is True for key in
            ("geometry_optimized", "scf_converged", "final_gradient_thresholds_passed", "normal_modes_complete",
             "all_frequencies_real_positive", "frequency_validation_passed", "stationary_minimum_verified"))
                or frequency_result.get("normal_mode_count") != 219):
            raise RuntimeError("Frequency phase did not validate a complete positive219-mode minimum; original outputs retained")
        state["status"] = "DONE"
    except Exception as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), exception_type=type(exc).__name__)
    finally:
        frequency_result_path = outputs["frequency"] / "result.json"
        if frequency_result_path.is_file() and not any(row.get("status") == "RUNNING" for row in read(args.ledger)["phases"]):
            actual = read(frequency_result_path)
            state["actual_frequency_summary"] = {key: actual.get(key) for key in
                ("status", "frequencies_status", "normal_mode_count", "imaginary_frequency_count", "frequency_validation_passed", "stationary_minimum_verified")}
        state["ended_at_utc"] = utc(); state["ledger_sha256_at_end"] = sha(args.ledger)
        state["supervisor_stage_request_count"] = len(state["supervisor_stages_requested"])
        write(args.output / "state.json", state)
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "ledger", "water-result", "ethanol-result", "input-xyz", "input-coordinates-npy", "provenance-json", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--protocol-file", type=Path, action="append", default=[])
    p.add_argument("--opt-phase", default="spinosin-native4-optimization-001")
    p.add_argument("--frequency-phase", default="spinosin-native4-frequency-001")
    p.add_argument("--opt-cpu-seconds", type=int, default=21600)
    p.add_argument("--opt-wall-seconds", type=int, default=10800)
    p.add_argument("--frequency-cpu-seconds", type=int, default=86400)
    p.add_argument("--frequency-wall-seconds", type=int, default=43200)
    return p


def main():
    args = parser().parse_args(); state = run(args)
    print(json.dumps(state, ensure_ascii=False)); sys.exit(0 if state["status"] == "DONE" else 2)


if __name__ == "__main__":
    main()
