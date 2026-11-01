"""Recover a final-phase geometry and prepare, never execute, a bounded restart.

No SCF, gradient, optimizer, Hessian or frequency call is made by this utility.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "9855fb57cac932c221ac561884562e38a97688b143bf8dbb7c5ad09333eea079"
EXPECTED_KEY = "VGGSULWDCMWZPO-ODEMIOGVSA-N"
BOHR_TO_ANGSTROM = .52917721092


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def isolated_sites():
    for relative in (".runtime/quantum-upgrade-site", ".runtime/quantum-pilot-site"):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def validate_ledger(ledger):
    cap, user_cap = float(ledger["quantum_cpu_cap_seconds"]), float(ledger["user_total_cpu_cap_seconds"])
    reserve = float(ledger.get("non_quantum_reserve_seconds", 0))
    if not np.isfinite([cap, user_cap, reserve]).all() or not 0 < cap <= 3300 or not cap <= user_cap <= 3600 or reserve < 0 or cap + reserve > user_cap:
        raise ValueError("Original authorized CPU caps/reserve must remain bounded")
    phases = ledger["phases"]
    if any(row.get("status") == "RUNNING" for row in phases):
        raise ValueError("A phase is still running; do not read changing checkpoint evidence as final")
    names = [row["phase"] for row in phases]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate recorded phase identities")
    actual = np.asarray([float(row["actual_cpu_seconds"]) for row in phases], float)
    consumed = float(ledger["consumed_cpu_seconds"])
    if not np.isfinite(actual).all() or np.any(actual < 0) or not math.isfinite(consumed) or consumed < 0 or consumed > cap:
        raise ValueError("Invalid or over-budget actual CPU accounting")
    if not math.isclose(float(actual.sum()), consumed, abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError("Phase CPU totals differ from cumulative ledger")
    remaining = cap - consumed
    if "remaining_quantum_cpu_seconds" in ledger and not math.isclose(float(ledger["remaining_quantum_cpu_seconds"]), remaining, abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError("Remaining CPU field conflicts with original cumulative cap")
    return remaining


def budget_decision(ledger, requested_cpu_seconds, resume_phase):
    remaining = validate_ledger(ledger)
    if not isinstance(requested_cpu_seconds, int) or not 0 <= requested_cpu_seconds <= 3300:
        raise ValueError("Requested CPU must be an integer0..3300; no automatic enlargement")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", resume_phase or "") or resume_phase in {row["phase"] for row in ledger["phases"]}:
        raise ValueError("A unique new resume phase name is required")
    grant = max(0, min(requested_cpu_seconds, math.floor(remaining) - 2))
    return {"remaining_original_quantum_cpu_seconds": remaining, "requested_cpu_seconds": requested_cpu_seconds,
            "bounded_cpu_grant_seconds": grant, "status": "READY" if grant >= 5 else "BLOCKED_INSUFFICIENT_ORIGINAL_CPU_BUDGET",
            "ledger_reinitialized": False, "automatic_budget_enlargement": False}


def phase_directory(run_root, row):
    command = row["command"]
    if not isinstance(command, list) or "--phase" not in command or command[command.index("--phase") + 1] != "spinosin-optimize" or "--output" not in command:
        raise ValueError("Recorded completed Spinosin optimization worker command required")
    location = Path(command[command.index("--output") + 1])
    location = location.resolve() if location.is_absolute() else (ROOT / location).resolve()
    if not location.is_relative_to(Path(run_root).resolve()):
        raise ValueError("Recorded source phase lies outside this pilot run")
    return location


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) != 77 or lines[0].strip() != "75":
        raise ValueError("Complete75-atom XYZ required")
    split = [line.split() for line in lines[2:]]
    if any(len(row) != 4 for row in split):
        raise ValueError("XYZ symbol/coordinate columns invalid")
    return [row[0] for row in split], np.asarray([row[1:] for row in split], float)


def latest_completed_geometry(directory):
    valid = []
    for metadata_path in (Path(directory) / "geometry_steps").glob("step_*.json"):
        row = read(metadata_path)
        xyz = metadata_path.with_suffix(".xyz")
        gradient = metadata_path.with_suffix(".gradient.npy")
        if not xyz.is_file() or not gradient.is_file() or row.get("scf_converged") is not True:
            continue
        values = np.load(gradient, allow_pickle=False)
        if values.shape != (75, 3) or not np.isfinite(values).all() or not math.isfinite(float(row["energy_hartree"])):
            raise ValueError("Finished geometry step has invalid gradient/energy")
        symbols, coordinates = read_xyz(xyz)
        valid.append((int(row["cycle"]), symbols, coordinates, {"metadata_path": str(metadata_path),
            "metadata_sha256": sha(metadata_path), "xyz_path": str(xyz), "xyz_sha256": sha(xyz),
            "gradient_path": str(gradient), "gradient_sha256": sha(gradient), "record": row}))
    return max(valid, key=lambda entry: entry[0]) if valid else None


def load_checkpoint_geometry(checkpoint):
    isolated_sites()
    from pyscf.scf.chkfile import load_mol
    molecule = load_mol(str(checkpoint))
    if molecule.charge != 0 or molecule.spin != 0 or molecule.nelectron != 320:
        raise ValueError("Checkpoint charge/spin/electrons differ from verified neutral Spinosin")
    symbols = [molecule.atom_symbol(index) for index in range(molecule.natm)]
    return symbols, np.asarray(molecule.atom_coords(unit="Angstrom"), float), np.asarray(molecule.atom_coords(unit="Bohr"), float)


def verify_coordinates(source_record, symbols, coordinates):
    isolated_sites()
    spec = importlib.util.spec_from_file_location("verified_conformer_graph_for_resume", ROOT / "scripts/prepare_spinosin_conformers_20261031.py")
    graph_helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(graph_helpers)
    from rdkit import Chem
    molecule, identity = graph_helpers.verify_source(source_record)
    values = np.asarray(coordinates, float)
    if values.shape != (75, 3) or not np.isfinite(values).all() or Counter(symbols) != Counter({"C": 28, "H": 32, "O": 15}):
        raise ValueError("Recovered coordinates require finite full C28H32O15 composition")
    if list(symbols) != [atom.GetSymbol() for atom in molecule.GetAtoms()]:
        raise ValueError("Atom order differs from original verified graph; do not guess a remapping")
    d = np.linalg.norm(values[:, None] - values[None, :], axis=2); np.fill_diagonal(d, np.inf)
    if d.min() < .35:
        raise ValueError("Recovered geometry contains near-coincident atoms")
    bonds = [float(np.linalg.norm(values[b.GetBeginAtomIdx()] - values[b.GetEndAtomIdx()])) for b in molecule.GetBonds()]
    if min(bonds) < .6 or max(bonds) > 2.2:
        raise ValueError("Recovered graph bond distances are implausible")
    molecule.RemoveAllConformers(); conformer = Chem.Conformer(75)
    for index, point in enumerate(values):
        conformer.SetAtomPosition(index, tuple(map(float, point)))
    molecule.AddConformer(conformer, assignId=True)
    coordinate_key = graph_helpers.stereo_key_from_coordinates(molecule)
    if coordinate_key != EXPECTED_KEY:
        raise ValueError("Recovered3D stereochemistry differs from trusted Spinosin")
    return {"identity_verified": True, "stereochemistry_verified": True, "source_inchikey": coordinate_key,
            "atom_count": 75, "composition": identity["composition"], "charge": 0, "multiplicity": 1,
            "radical_electrons": 0, "electron_count": 320,
            "minimum_bond_length_angstrom": min(bonds), "maximum_bond_length_angstrom": max(bonds),
            "geometry_stationarity_verified_by_this_utility": False}


def validate_parent_settings(settings):
    expected = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1, "charge": 0, "spin_2S": 0,
                "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3"}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Resume command must preserve parent PBE/def2-SVP/grid1/no-D3/neutral-singlet settings")


def build_worker_command(ledger, phase_name, grant, xyz, provenance, checkpoint, output, log, maxsteps=35):
    if not 5 <= grant <= 3300 or not 1 <= maxsteps <= 35:
        raise ValueError("Bounded meaningful CPU grant and<=35optimizer steps required")
    command = [str(ROOT / ".runtime/bin/python"), str(ROOT / "scripts/run_quantum_budget_20261031.py"),
               "--ledger", str(ledger), "--phase", phase_name, "--cpu-seconds", str(grant),
               "--wall-seconds", str(min(7200, math.ceil(grant * 1.5))), "--memory-gib", "12", "--log", str(log), "--",
               str(ROOT / ".runtime/bin/python"), str(ROOT / "scripts/pyscf_pilot_cached_df_worker_20261031.py"),
               "--phase", "spinosin-optimize", "--input-xyz", str(xyz), "--provenance-json", str(provenance),
               "--expected-input-sha256", sha(xyz), "--output", str(output), "--grid-level", "1", "--maxsteps", str(maxsteps)]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if "--initialize" in command or "--total-cpu-seconds" in command:
        raise AssertionError("Continuation must never reset or enlarge the existing ledger")
    return command


def prepare(run_root, source_phase, output, release_path, resume_phase, requested_cpu=900, maxsteps=35):
    ledger = Path(run_root).resolve() / "cpu_budget.json"
    with ledger.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Budget worker still holds its execution lock; wait for final phase exit") from exc
        return _prepare_locked(run_root, source_phase, output, release_path, resume_phase, requested_cpu, maxsteps)


def _prepare_locked(run_root, source_phase, output, release_path, resume_phase, requested_cpu=900, maxsteps=35):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    release_path = Path(release_path).resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier continuation preparation")
    if not isinstance(maxsteps, int) or not 1 <= maxsteps <= 35:
        raise ValueError("Continuation optimizer steps must remain1..35")
    ledger_path = run_root / "cpu_budget.json"
    ledger, release = read(ledger_path), read(release_path)
    if release.get("allow_resume_preparation") is not True or not release.get("created_at_utc") or release.get("cpu_budget_sha256") != sha(ledger_path) or release.get("source_phase") != source_phase:
        raise ValueError("Dated root release must bind final existing ledger and source phase")
    decision = budget_decision(ledger, requested_cpu, resume_phase)
    if not ledger["phases"] or ledger["phases"][-1]["phase"] != source_phase:
        raise ValueError("Source must be the final recorded ended optimization phase")
    row = ledger["phases"][-1]; source = phase_directory(run_root, row)
    result_path = source / "result.json"
    if result_path.exists():
        if release.get("source_result_sha256") != sha(result_path):
            raise ValueError("Final worker result differs from authorized snapshot")
        parent_result = read(result_path)
    else:
        if release.get("allow_missing_worker_result") is not True:
            raise ValueError("Missing final worker result must be explicitly acknowledged")
        parent_result = {"status": "PARTIAL", "geometry_optimized": False, "result_missing_after_phase_exit": True}
    parent_input = read(source / "input.json")
    validate_parent_settings(parent_result.get("settings", parent_input["settings"]))
    record_path = run_root / "structures/spinosin_verified_2d.json"
    if sha(record_path) != SOURCE_SHA256:
        raise ValueError("Original trusted stereochemical source changed")
    source_record = read(record_path)
    checkpoint = source / "calculation.chk"
    last_completed = latest_completed_geometry(source)
    derivation = "PySCF DFT optimization checkpoint; not certified stationary"
    if checkpoint.exists():
        expected_checkpoint = parent_result.get("checkpoint_sha256") or release.get("source_checkpoint_sha256")
        if not expected_checkpoint or sha(checkpoint) != expected_checkpoint:
            raise ValueError("Checkpoint differs from final worker result or explicit root checkpoint binding")
        try:
            symbols, coordinates, bohr = load_checkpoint_geometry(checkpoint)
        except Exception as exc:
            raise ValueError("Existing checkpoint could not be safely loaded; no silent coordinate fallback") from exc
        coordinate_origin = "checkpoint_fullprecision_molecule"
    elif last_completed is not None:
        _, symbols, coordinates, _ = last_completed
        bohr = coordinates / BOHR_TO_ANGSTROM
        coordinate_origin = "last_completed_gradient_step_xyz_rounded_to12decimals"
        derivation = "Last completed PySCF gradient-step XYZ; original12decimal precision retained; not certified stationary"
        checkpoint = None
    else:
        raise ValueError("No checkpoint and no complete SCF/gradient geometry available")
    identity = verify_coordinates(source_record, symbols, coordinates)
    if sha(ledger_path) != release["cpu_budget_sha256"]:
        raise ValueError("Ledger changed during read-only extraction")
    output.mkdir(parents=True)
    xyz = output / "resume_geometry.xyz"
    xyz.write_text("75\nRecovered Spinosin PBE geometry; NOT a stationarity certificate; fresh SCF/gradient required\n" +
                   "\n".join(f"{symbol} {p[0]:.14f} {p[1]:.14f} {p[2]:.14f}" for symbol, p in zip(symbols, coordinates)) + "\n")
    np.save(output / "resume_coordinates_angstrom.npy", coordinates)
    np.save(output / "resume_coordinates_bohr.npy", bohr)
    dependencies = {str(ledger_path): sha(ledger_path), str(release_path): sha(release_path),
                    str(record_path): SOURCE_SHA256, str(source / "input.json"): sha(source / "input.json")}
    for relative in ("scripts/run_quantum_budget_20261031.py", "scripts/pyscf_pilot_cached_df_worker_20261031.py", "scripts/prepare_spinosin_conformers_20261031.py"):
        dependencies[str(ROOT / relative)] = sha(ROOT / relative)
    if result_path.exists():
        dependencies[str(result_path)] = sha(result_path)
    if checkpoint is not None:
        dependencies[str(checkpoint)] = sha(checkpoint)
    if last_completed is not None:
        for key in ("metadata", "xyz", "gradient"):
            dependencies[last_completed[3][key + "_path"]] = last_completed[3][key + "_sha256"]
    provenance = {**identity, "geometry_origin": "documented_conformer_generation",
        "original_geometry_lineage": "Verified ChEBI/PubChem graph→ETKDGv3/MMFF94s→recorded PySCF optimization",
        "current_geometry_derivation": derivation, "coordinate_origin": coordinate_origin,
        "source_url": source_record["primary_source"], "pubchem_cid": 155692,
        "source_record_sha256": SOURCE_SHA256, "input_xyz_sha256": sha(xyz),
        "checkpoint_sha256": sha(checkpoint) if checkpoint is not None else None,
        "parent_recorded_checkpoint_sha256": parent_result.get("checkpoint_sha256"),
        "parent_worker_result_sha256": sha(result_path) if result_path.exists() else None,
        "parent_status": parent_result.get("status"), "parent_geometry_optimized_claim": parent_result.get("geometry_optimized", False),
        "parent_scf_converged_claim": parent_result.get("scf_converged", False),
        "checkpoint_warning": "PySCF saves the trial molecule before SCF completes. Its geometry can differ from the last completed gradient, and stored MO data may be from a previous cycle. Always run fresh SCF/gradient; do not certify stationarity from checkpoint availability.",
        "last_completed_gradient_step": last_completed[3] if last_completed else None,
        "checkpoint_vs_last_completed_max_abs_angstrom": float(np.max(np.abs(coordinates - last_completed[2]))) if last_completed else None,
        "fresh_SCF_and_gradient_required": True, "new_quantum_computations": 0}
    provenance_path = output / "resume_geometry_provenance.json"; write(provenance_path, provenance)
    worker_output = run_root / "calculations" / resume_phase
    log = run_root / "execution" / (resume_phase + ".log")
    if worker_output.exists() or log.exists():
        raise FileExistsError("Fresh continuation worker output and log required")
    command = None
    continuation_status = decision["status"]
    if decision["status"] == "READY" and parent_result.get("geometry_optimized") is not True:
        command = build_worker_command(ledger_path, resume_phase, decision["bounded_cpu_grant_seconds"], xyz, provenance_path, checkpoint, worker_output, log, maxsteps)
    elif parent_result.get("geometry_optimized") is True:
        continuation_status = "NOT_REQUIRED_PARENT_REPORTS_OPTIMIZED_REVIEW_FREQUENCY_GATES"
    if command is not None:
        verification_command = [str(ROOT / ".runtime/bin/python"), str(Path(__file__).resolve()), "verify", "--output", str(output)]
        env = ["env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
               "PYTHONPATH=" + str(ROOT / ".runtime/quantum-pilot-site") + ":" + str(ROOT / ".runtime/quantum-upgrade-site")]
        (output / "resume_same_budget.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(env + verification_command) + "\n" + shlex.join(env + command) + "\n")
    manifest = {"prepared_at_utc": utc(), "status": continuation_status, "source_phase": source_phase,
        "source_phase_directory": str(source), "resume_phase": resume_phase, "budget": decision,
        "same_existing_ledger": str(ledger_path), "new_quantum_jobs": 0, "command_executed": False,
        "parent_input_dependencies_sha256": dependencies, "worker_command": command,
        "local_files_sha256": {str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file()},
        "resume_script_sha256": sha(__file__)}
    write(output / "continuation_manifest.json", manifest)
    text = "# PySCF优化检查点续算准备\n\n状态：`" + continuation_status + "`。只提取并核验几何与生成命令；本阶段没有执行SCF、梯度或优化。\n\n"
    text += "原累计量子CPU预算剩余" + str(decision["remaining_original_quantum_cpu_seconds"]) + "秒；本次最多可请求" + str(decision["bounded_cpu_grant_seconds"]) + "秒，不重置或扩大账本。\n\n"
    text += "检查点可能包含下一试探几何，并不等于最后一个完成梯度或驻点；已保存完整精度.npy及带新SHA的XYZ。恢复命令始终要求新的SCF和梯度。\n\n"
    text += "执行前必须核对全部冻结父证据及同一CPU账本。" + ("可运行`bash " + str(output / "resume_same_budget.sh") + "`，本准备过程没有执行此命令。" if command else "当前不生成可执行续算命令。若因原预算不足而BLOCKED，另一个预算批次需要另行明确授权，不能自动初始化新账本。") + "\n"
    (output / "README.md").write_text(text)
    write(output / "files_sha256.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "files_sha256.json"})
    return manifest


def verify(output):
    output = Path(output)
    manifest = read(output / "continuation_manifest.json")
    if manifest.get("new_quantum_jobs") != 0 or manifest.get("command_executed") is not False:
        raise ValueError("Unexpected execution claim")
    for path, expected in manifest["parent_input_dependencies_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Parent file or cumulative ledger changed; prepare a new bounded continuation")
    for relative, expected in manifest["local_files_sha256"].items():
        if sha(output / relative) != expected:
            raise ValueError("Prepared geometry or provenance changed")
    if manifest["resume_script_sha256"] != sha(__file__):
        raise ValueError("Continuation helper version changed")
    decision = budget_decision(read(manifest["same_existing_ledger"]), manifest["budget"]["requested_cpu_seconds"], manifest["resume_phase"])
    if decision != manifest["budget"]:
        raise ValueError("Bounded budget decision differs from prepared snapshot")
    return {"status": "FROZEN_CONTINUATION_INPUTS_MATCH", "scientific_stationarity_certified": False, "new_quantum_jobs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run-root", type=Path, required=True); p.add_argument("--source-phase", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--release", type=Path, required=True)
    p.add_argument("--resume-phase", default="spinosin-optimize-resume-001")
    p.add_argument("--requested-cpu-seconds", type=int, default=900); p.add_argument("--maxsteps", type=int, default=35)
    v = sub.add_parser("verify"); v.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output) if args.command == "verify" else prepare(args.run_root, args.source_phase, args.output, args.release, args.resume_phase, args.requested_cpu_seconds, args.maxsteps)
    print(json.dumps({k: result[k] for k in result if k in ("status", "source_phase", "budget", "new_quantum_jobs", "scientific_stationarity_certified")}))


if __name__ == "__main__":
    main()
