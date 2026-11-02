"""Prepare provenance-bound geometry inputs; never run quantum calculations."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRUSTED_GRAPH = ROOT / "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json"
TRUSTED_GRAPH_SHA = "9855fb57cac932c221ac561884562e38a97688b143bf8dbb7c5ad09333eea079"
THRESHOLDS = {"convergence_energy": 1e-6, "convergence_grms": 3e-4,
              "convergence_gmax": 4.5e-4, "convergence_drms": .0012,
              "convergence_dmax": .0018}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def aware_time(value):
    if not isinstance(value, str):
        raise ValueError("Dated timezone-aware release required")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("UTC-offset timestamp required")
    return moment


def source_phase(ledger, source_output):
    if any(row.get("status") == "RUNNING" for row in ledger["phases"]):
        raise ValueError("Quantum phase is running; refuse geometry transfer")
    matches = []
    for row in ledger["phases"]:
        command = row.get("command", [])
        if "--output" not in command:
            continue
        location = Path(command[command.index("--output") + 1])
        location = location if location.is_absolute() else ROOT / location
        if location.resolve() == source_output.resolve():
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("Exactly one recorded source phase must own source output")
    phase = matches[0]
    if phase.get("status") not in ("EXITED", "PARTIAL_OR_FAILED") or "exit_code" not in phase or "actual_cpu_seconds" not in phase:
        raise ValueError("Source phase has no completed supervisor evidence")
    if not math.isfinite(float(phase["actual_cpu_seconds"])) or float(phase["actual_cpu_seconds"]) < 0:
        raise ValueError("Completed phase CPU evidence is invalid")
    aware_time(phase["ended_at_utc"])
    return phase


def gradient_checks(path, record, require_stationary=False):
    values = np.load(path, allow_pickle=False)
    if values.shape != (75, 3) or not np.isfinite(values).all():
        raise ValueError("Finite complete75-atom gradient required")
    norms = np.linalg.norm(values, axis=1)
    stats = {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(values**2))),
             "component_max_hartree_per_bohr": float(np.abs(values).max()),
             "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(norms**2))),
             "atom_max_hartree_per_bohr": float(norms.max()),
             "gradient_norm_hartree_per_bohr": float(np.linalg.norm(values))}
    for key, value in stats.items():
        if key not in record or not math.isclose(value, float(record[key]), abs_tol=1e-13, rel_tol=1e-11):
            raise ValueError("Stored gradient statistics do not match saved array: " + key)
    passed = stats["atom_rms_hartree_per_bohr"] <= 3e-4 and stats["atom_max_hartree_per_bohr"] <= 4.5e-4
    if require_stationary and not passed:
        raise ValueError("Actual fresh gradient does not pass stationarity thresholds")
    return {**stats, "stationary_gradient_thresholds_passed": bool(passed)}


def frequency_source_gate(result):
    if result.get("status") != "DONE" or any(result.get(k) is not True for k in
       ("geometry_optimized", "optimizer_converged", "scf_converged", "final_gradient_thresholds_passed")):
        raise ValueError("Frequency input requires completed V4 optimized/fresh-gradient evidence")
    settings = result["settings"]
    expected = {"density_fit_backend": "SCREENED_DIRECT_RKS_V4", "auxbasis": None,
                "xc": "PBE", "basis": "def2-svp", "grid_level": 1,
                "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "charge": 0,
                "spin_2S": 0, "multiplicity": 1, "direct_scf_tol": 1e-13,
                "scf_conv_tol_hartree": 1e-8, "global_four_center_ERI_cache_allowed": False,
                "optimizer_thresholds": THRESHOLDS}
    if settings.get("density_fitting") is not False or settings.get("direct_scf") is not True:
        raise ValueError("Frequency source must be the distinct no-DF direct RKS method")
    if any(key not in settings or settings[key] != value for key, value in expected.items()):
        raise ValueError("Frequency source method/thresholds differ from reviewed V4")


def selected_geometry(source_output, result, purpose):
    """Never read a checkpoint or a PARTIAL optimizer's returned trial geometry."""
    if purpose == "frequency":
        frequency_source_gate(result)
    optimized = result.get("status") == "DONE" and result.get("geometry_optimized") is True
    if optimized:
        if any(result.get(k) is not True for k in ("optimizer_converged", "scf_converged", "final_gradient_thresholds_passed")):
            raise ValueError("Inconsistent optimized source evidence")
        xyz, coords = source_output / "optimized.xyz", source_output / "optimized_coordinates_angstrom.npy"
        if not result.get("optimized_xyz_sha256") or sha(xyz) != result["optimized_xyz_sha256"]:
            raise ValueError("Source optimized XYZ differs from result SHA")
        if not result.get("optimized_coordinates_angstrom_npy_sha256") or sha(coords) != result["optimized_coordinates_angstrom_npy_sha256"]:
            raise ValueError("Source fullprecision optimized coordinates differ from result SHA")
        gradient = source_output / "gradient_hartree_per_bohr.npy"
        stats = gradient_checks(gradient, result, require_stationary=True)
        record, kind, metadata_path = result, "optimized_fullprecision_with_fresh_gradient", source_output / "result.json"
    elif purpose == "optimization" and result.get("status") == "PARTIAL":
        candidates = []
        for metadata in (source_output / "geometry_steps").glob("step_*.json"):
            row = read(metadata)
            if row.get("scf_converged") is True:
                candidates.append((int(row["cycle"]), metadata, row))
        if not candidates:
            raise ValueError("PARTIAL source has no completed SCF/gradient step")
        _, metadata_path, record = max(candidates, key=lambda row: row[0])
        xyz = metadata_path.with_suffix(".xyz")
        coords = metadata_path.with_suffix(".coordinates_angstrom.npy")
        gradient = metadata_path.with_suffix(".gradient.npy")
        for path, key in ((coords, "coordinates_angstrom_npy_sha256"), (gradient, "gradient_npy_sha256")):
            if not record.get(key) or sha(path) != record[key]:
                raise ValueError("Last completed step coordinate/gradient SHA differs")
        stats = gradient_checks(gradient, record)
        kind = "last_completed_SCF_gradient_step_NOT_verified_optimizer_minimum"
    else:
        raise ValueError("Source is neither verified optimized nor PARTIAL with completed steps")
    energy = float(record["energy_hartree"])
    if not math.isfinite(energy):
        raise ValueError("Finite geometry-paired source energy required")
    values = np.load(coords, allow_pickle=False)
    if values.shape != (75, 3) or not np.isfinite(values).all():
        raise ValueError("Finite fullprecision75-atom Angstrom coordinates required")
    return xyz, coords, gradient, values, {"selection": kind, "metadata_path": str(metadata_path),
        "metadata_sha256": sha(metadata_path), "source_geometry_optimized": bool(optimized),
        "source_cycle": record.get("cycle"), "paired_source_energy_hartree": energy,
        "paired_source_gradient_statistics": stats}


def verify_identity(symbols, coordinates):
    if sha(TRUSTED_GRAPH) != TRUSTED_GRAPH_SHA:
        raise ValueError("Trusted stereochemical source changed")
    spec = importlib.util.spec_from_file_location("trusted_geometry_transfer_verifier", ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.verify_coordinates(read(TRUSTED_GRAPH), symbols, coordinates)


def prepare(source_result, source_output, ledger_path, output, purpose, release_path):
    source_result, source_output = Path(source_result).resolve(), Path(source_output).resolve()
    ledger_path, output, release_path = Path(ledger_path).resolve(), Path(output).resolve(), Path(release_path).resolve()
    if output.exists() or output.is_relative_to(source_output):
        raise ValueError("Fresh output outside the immutable source phase required")
    if source_result != source_output / "result.json" or purpose not in ("optimization", "frequency"):
        raise ValueError("Source result and purpose do not identify the requested phase")
    with ledger_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Quantum execution lock held; do not read its geometry") from exc
        result, ledger, release = read(source_result), read(ledger_path), read(release_path)
        if release.get("allow_geometry_transfer") is not True or release.get("cpu_budget_sha256") != sha(ledger_path) or release.get("source_result_sha256") != sha(source_result):
            raise ValueError("Root release must bind ended ledger and source result")
        released_at = aware_time(release["created_at_utc"])
        phase = source_phase(ledger, source_output)
        if released_at < aware_time(phase["ended_at_utc"]):
            raise ValueError("Release predates source phase completion")
        if result.get("identity") != "spinosin" or result.get("phase") != "spinosin-optimize" or result.get("atoms") != 75 or result.get("electrons") != 320:
            raise ValueError("Source must be a completed Spinosin optimization phase")
        if any(result.get("settings", {}).get(key) != value for key, value in
               (("charge", 0), ("spin_2S", 0), ("multiplicity", 1))):
            raise ValueError("Source electronic-state metadata differ from neutral singlet Spinosin")
        xyz, coords, gradient, values, selection = selected_geometry(source_output, result, purpose)
        lines = xyz.read_text().splitlines()
        rows = [line.split() for line in lines[2:] if line.strip()]
        if len(rows) != 75 or int(lines[0]) != 75 or any(len(row) != 4 for row in rows):
            raise ValueError("Source XYZ atom rows incomplete")
        symbols = [row[0] for row in rows]
        rounded = np.asarray([row[1:] for row in rows], float)
        if not np.isfinite(rounded).all() or not np.allclose(rounded, values, atol=5e-12, rtol=0):
            raise ValueError("Fullprecision source and source XYZ do not describe the same geometry")
        identity = verify_identity(symbols, values)
        if sha(ledger_path) != release["cpu_budget_sha256"] or sha(source_result) != release["source_result_sha256"]:
            raise ValueError("Released source evidence changed during preparation")
        output.mkdir(parents=True)
        new_xyz, new_coords = output / "transfer_geometry.xyz", output / "transfer_coordinates_angstrom.npy"
        shutil.copy2(coords, new_coords)
        if purpose == "frequency":
            # Preserve the optimizer result's exact XYZ byte hash. Changing a
            # title/rounding would correctly fail the frequency worker gate.
            shutil.copy2(xyz, new_xyz)
        else:
            new_xyz.write_text("75\nSource geometry only; fresh target-method SCF/gradient/optimizer required\n" +
                "\n".join(f"{symbol} {x:.14f} {y:.14f} {z:.14f}" for symbol, (x, y, z) in zip(symbols, values)) + "\n")
        provenance = {**identity, "geometry_origin": "documented_conformer_generation",
            "source_url": read(TRUSTED_GRAPH)["primary_source"], "pubchem_cid": 155692,
            "original_geometry_lineage": "Verified ChEBI/PubChem stereo graph -> ETKDGv3/MMFF94s -> recorded DFT source geometry",
            "current_geometry_derivation": selection["selection"], "source_method": result["settings"],
            "source_result_sha256": sha(source_result), "source_xyz_sha256": sha(xyz),
            "source_coordinates_angstrom_npy_sha256": sha(coords), "input_xyz_sha256": sha(new_xyz),
            "input_coordinates_angstrom_npy_sha256": sha(new_coords), "purpose": purpose,
            "parent_status": result["status"], "selection": selection,
            "fresh_target_SCF_and_gradient_required": True,
            "new_geomeTRIC_optimizer_required_for_optimization": purpose == "optimization",
            "optimizer_history_restored": False, "checkpoint_read": False, "new_quantum_jobs": 0}
        write(output / "transfer_geometry_provenance.json", provenance)
        deps = {str(path): sha(path) for path in (ledger_path, release_path, source_result, xyz, coords, gradient, TRUSTED_GRAPH)}
        deps[selection["metadata_path"]] = selection["metadata_sha256"]
        manifest = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "status": "READY_INPUTS_ONLY_NOT_EXECUTED",
                    "purpose": purpose, "source_phase": phase["phase"], "source_output": str(source_output),
                    "source_result_sha256": sha(source_result), "source_method": result["settings"],
                    "selection": selection, "new_quantum_jobs": 0, "worker_launched": False,
                    "checkpoint_read": False, "optimizer_history_restored": False,
                    "parent_input_sha256": deps, "script_sha256": sha(__file__)}
        write(output / "transfer_manifest.json", manifest)
        (output / "README.md").write_text("# 几何与来源转移准备\n\n状态：READY_INPUTS_ONLY_NOT_EXECUTED。新增量子计算0，未启动worker，未读取检查点。\n\n"
            "用途：" + purpose + "；原阶段状态：" + result["status"] + "。选择：" + selection["selection"] + "。\n\n"
            "这是几何和来源转移，源能量／梯度只与选中的源几何配对，不能赋给另一方法。目标方法必须重新SCF与核梯度；优化用途使用新geomeTRIC，不恢复历史。PARTIAL仅取最后完整步，不能视为驻点。频率用途保持原优化XYZ及完整精度NPY的字节SHA，同时仍须目标worker的新SCF／梯度检查。\n")
        write(output / "files_sha256.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-result", "source-output", "ledger", "output", "release"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--purpose", choices=("optimization", "frequency"), required=True)
    args = parser.parse_args()
    result = prepare(args.source_result, args.source_output, args.ledger, args.output, args.purpose, args.release)
    print(json.dumps({key: result[key] for key in ("status", "purpose", "source_phase", "new_quantum_jobs", "worker_launched")}))


if __name__ == "__main__":
    main()
