"""Audit a closed DF optimization while other independent phases may run."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = OUT.parent
SOURCE = RUN / "spinosin-extended-optimization-001"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(gradient):
    values = np.asarray(gradient)
    if values.shape != (75, 3) or not np.isfinite(values).all():
        raise ValueError("Complete finite75-atom gradient required")
    norms = np.linalg.norm(values, axis=1)
    return {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(values**2))),
            "component_max_hartree_per_bohr": float(np.abs(values).max()),
            "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(norms**2))),
            "atom_max_hartree_per_bohr": float(norms.max()),
            "gradient_norm_hartree_per_bohr": float(np.linalg.norm(values))}


def read(path):
    return json.loads(Path(path).read_text())


def match_stats(gradient, record):
    values = stats(gradient)
    for key, value in values.items():
        if not math.isclose(value, float(record[key]), abs_tol=1e-13, rel_tol=1e-11):
            raise ValueError("Gradient statistic does not match saved array: " + key)
    return values


def run():
    target = OUT / "completed_df_optimization_review.json"
    if target.exists():
        raise FileExistsError("Preserve completed phase audit")
    started = datetime.now(timezone.utc).isoformat()
    ledger_bytes = (RUN / "cpu_budget.json").read_bytes()
    ledger = json.loads(ledger_bytes)
    phases = [p for p in ledger["phases"] if p["phase"] == "spinosin_extended_optimization_001"]
    assert len(phases) == 1
    phase = phases[0]
    assert phase["status"] == "EXITED" and phase["exit_code"] == 0 and phase["ended_at_utc"]
    assert phase["actual_cpu_seconds"] <= phase["granted_cpu_seconds"]
    (OUT / "ledger_snapshot_with_closed_source.json").write_bytes(ledger_bytes)
    result_path, checkpoint = SOURCE / "result.json", SOURCE / "calculation.chk"
    result = read(result_path)
    result_sha, checkpoint_sha = sha(result_path), sha(checkpoint)
    assert result["status"] == "DONE" and result["phase"] == "spinosin-optimize"
    for key in ("optimizer_converged", "geometry_optimized", "scf_converged", "final_gradient_thresholds_passed"):
        assert result[key] is True, key
    assert result["settings"]["density_fitting"] is True
    assert result["settings"]["density_fit_backend"] == "INCORE_DFJ_RELEASE_BEFORE_GRAD_V3"
    assert result["frequencies_status"] == "NOT_RUN" and result["IR_intensities_status"] == "NOT_COMPUTED"
    worker = ROOT / "scripts/pyscf_spinosin_continuation_worker_20261031.py"
    assert result["script_sha256"] == sha(worker)
    assert result["checkpoint_sha256"] == checkpoint_sha
    supervisor_log = Path(phase["log"])
    assert sha(supervisor_log) == phase["log_sha256"]
    assert "Converged!" in supervisor_log.read_text()
    spec = importlib.util.spec_from_file_location("df_end_geometry_verifier", ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    graph_path = ROOT / "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json"
    graph = read(graph_path)
    log = (SOURCE / "pyscf.log").read_text()
    starts = list(re.finditer(r"Geometry optimization cycle\s+(\d+)", log))
    chunks = {int(m.group(1)): log[m.end():starts[i+1].start() if i+1 < len(starts) else len(log)] for i,m in enumerate(starts)}
    evidence_paths = [result_path, SOURCE / "input.json", SOURCE / "progress.json", SOURCE / "pyscf.log",
                      SOURCE / "events.jsonl", checkpoint, worker, graph_path,
                      ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py", supervisor_log]
    rows = []
    metadata_files = sorted((SOURCE / "geometry_steps").glob("step_*.json"))
    assert len(metadata_files) == 20
    for expected_cycle, metadata in enumerate(metadata_files, 1):
        row = read(metadata); cycle = int(row["cycle"])
        assert cycle == expected_cycle and row["scf_converged"] is True
        coordinate_path, bohr_path, gradient_path, xyz_path = [metadata.with_suffix(s) for s in
            (".coordinates_angstrom.npy", ".coordinates_bohr.npy", ".gradient.npy", ".xyz")]
        coordinates = np.load(coordinate_path, allow_pickle=False)
        bohr = np.load(bohr_path, allow_pickle=False)
        gradient = np.load(gradient_path, allow_pickle=False)
        symbols, rounded = helper.read_xyz(xyz_path)
        assert coordinates.shape == bohr.shape == (75,3) and np.isfinite(coordinates).all() and np.isfinite(bohr).all()
        assert np.allclose(coordinates, rounded, atol=5e-12, rtol=0)
        assert np.allclose(coordinates, bohr * helper.BOHR_TO_ANGSTROM, atol=1e-12, rtol=0)
        for path, key in ((coordinate_path, "coordinates_angstrom_npy_sha256"),
                          (bohr_path, "coordinates_bohr_npy_sha256"), (gradient_path, "gradient_npy_sha256")):
            assert sha(path) == row[key]
        energy_matches = [float(v) for v in re.findall(r"converged SCF energy\s*=\s*([-+0-9.eE]+)", chunks[cycle])]
        # The last geometry's log chunk also contains the later fresh SCF.
        # The first convergence record is the geometry callback's paired SCF.
        assert energy_matches and math.isclose(energy_matches[0], row["energy_hartree"], abs_tol=5e-9, rel_tol=0)
        values = match_stats(gradient, row)
        identity = helper.verify_coordinates(graph, symbols, coordinates)
        rows.append({"cycle": cycle, "paired_energy_hartree": row["energy_hartree"],
                     "matched_geometry_SCF_log_energy_hartree": energy_matches[0],
                     "recomputed_gradient_statistics": values,
                     "coordinate_and_gradient_file_SHA_passed": True, "identity": identity})
        evidence_paths.extend((metadata, coordinate_path, bohr_path, gradient_path, xyz_path))
    optimized_xyz, optimized_coords = SOURCE / "optimized.xyz", SOURCE / "optimized_coordinates_angstrom.npy"
    assert sha(optimized_xyz) == result["optimized_xyz_sha256"]
    assert sha(optimized_coords) == result["optimized_coordinates_angstrom_npy_sha256"]
    symbols, rounded = helper.read_xyz(optimized_xyz)
    coordinates = np.load(optimized_coords, allow_pickle=False)
    assert coordinates.shape == (75,3) and np.isfinite(coordinates).all() and np.allclose(coordinates, rounded, atol=5e-12, rtol=0)
    final_identity = helper.verify_coordinates(graph, symbols, coordinates)
    fresh_gradient_path = SOURCE / "gradient_hartree_per_bohr.npy"
    final_gradient = match_stats(np.load(fresh_gradient_path, allow_pickle=False), result)
    assert final_gradient["atom_rms_hartree_per_bohr"] <= 3e-4 and final_gradient["atom_max_hartree_per_bohr"] <= 4.5e-4
    assert math.isclose(float(re.findall(r"converged SCF energy\s*=\s*([-+0-9.eE]+)", log)[-1]), result["energy_hartree"], abs_tol=5e-9, rel_tol=0)
    # Only the closed source checkpoint is opened. No other phase checkpoint
    # is searched, inspected or loaded, even if the batch is active.
    helper.isolated_sites()
    from pyscf.scf import chkfile
    molecule = chkfile.load_mol(str(checkpoint))
    checkpoint_scf = chkfile.load(str(checkpoint), "scf")
    assert molecule.natm == 75 and molecule.nelectron == 320 and molecule.charge == molecule.spin == 0
    checkpoint_coordinates = molecule.atom_coords(unit="Angstrom")
    assert np.allclose(checkpoint_coordinates, coordinates, atol=1e-12, rtol=0)
    assert math.isclose(float(checkpoint_scf["e_tot"]), result["energy_hartree"], abs_tol=1e-12, rel_tol=0)
    checkpoint_orbitals = {}
    for name in ("mo_coeff", "mo_energy", "mo_occ"):
        values = np.asarray(checkpoint_scf[name]); assert np.isfinite(values).all()
        checkpoint_orbitals[name] = {"shape": list(values.shape), "dtype": str(values.dtype),
            "C_order_payload_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
    assert checkpoint_scf["mo_coeff"].shape == (762,762)
    assert checkpoint_scf["mo_energy"].shape == checkpoint_scf["mo_occ"].shape == (762,)
    assert np.isin(checkpoint_scf["mo_occ"], [0,2]).all() and checkpoint_scf["mo_occ"].sum() == 320
    events = [json.loads(line) for line in (SOURCE / "events.jsonl").read_text().splitlines()]
    releases = [event for event in events if event["event"] == "gradient_started_after_cache_release"]
    timings = [event for event in events if event["event"] == "gradient_kernel_complete"]
    assert len(releases) == len(timings) == 21
    for event in releases:
        assert event["cache_released"] and event["auxmol_identity_preserved"] and not event["with_df_reset_called"]
        assert event["SCF_cache_before_release"]["nbytes"] == 5718709416
        assert event["current_rss_mb_before_release"] > event["current_rss_mb_after_release"]
    for event in timings:
        assert event["cache_after_gradient"]["storage"] == "not_built"
        assert event["gradient_cpu_seconds"] > 0 and event["gradient_wall_seconds"] > 0
    assert not any((SOURCE / name).exists() for name in ("normal_modes.npy", "frequencies.csv", "hessian_hartree_per_bohr2.npy"))
    evidence_paths.extend((optimized_xyz, optimized_coords, fresh_gradient_path))
    file_hashes = {str(path.relative_to(ROOT)): sha(path) for path in evidence_paths}
    assert sha(result_path) == result_sha and sha(checkpoint) == checkpoint_sha
    reviewed = {"status": "PASS_CLOSED_DF_OPTIMIZATION_STATIONARITY_AUDIT_NOT_A_FREQUENCY_RESULT",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "new_SCF_gradient_integral_Hessian_calls": 0,
        "closed_source_phase": phase, "batch_is_not_asserted_final": True,
        "ledger_snapshot_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "other_batch_phases_running_at_snapshot": [p["phase"] for p in ledger["phases"] if p["status"] == "RUNNING"],
        "only_closed_source_checkpoint_read": str(checkpoint.relative_to(ROOT)),
        "source_result_sha256": result_sha, "checkpoint_sha256": checkpoint_sha,
        "complete_geometry_evaluations": len(rows), "geometries": rows,
        "optimizer_converged_flag_and_driver_log_passed": True,
        "final_fresh_SCF_energy_hartree": result["energy_hartree"],
        "last_geometry_callback_energy_hartree": rows[-1]["paired_energy_hartree"],
        "fresh_gradient_statistics": final_gradient, "optimized_identity": final_identity,
        "checkpoint_same_geometry_energy_electrons_and_MO_arrays_passed": True,
        "checkpoint_MO_array_fingerprints": checkpoint_orbitals,
        "cache_releases_and_completed_gradients": len(timings),
        "total_recorded_gradient_CPU_seconds": sum(e["gradient_cpu_seconds"] for e in timings),
        "total_recorded_gradient_wall_seconds": sum(e["gradient_wall_seconds"] for e in timings),
        "source_quantum_cpu_seconds": phase["actual_cpu_seconds"], "source_wall_seconds": phase["elapsed_wall_seconds"],
        "DF_stationary_geometry_verified": True, "positive_frequency_minimum_verified": False,
        "Spinosin_Hessian_and_frequencies": "NOT_RUN", "IR_intensities": "NOT_COMPUTED",
        "no_DF_method_stationarity_verified_by_this_audit": False,
        "source_files_sha256": file_hashes,
        "limits": ["A converged stationary geometry is not a proven positive-frequency minimum or a global conformer minimum.",
                   "The complete batch may still run independent phases; only this exited source was audited.",
                   "DF optimization evidence must not be relabelled as no-DF optimization or vibrational results."]}
    target.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": reviewed["status"], "complete_geometry_evaluations": len(rows),
                      "final_energy_hartree": result["energy_hartree"], "fresh_atom_rms": final_gradient["atom_rms_hartree_per_bohr"],
                      "fresh_atom_gmax": final_gradient["atom_max_hartree_per_bohr"], "source_CPU_seconds": phase["actual_cpu_seconds"],
                      "Hessian_frequencies": "NOT_RUN", "new_quantum_jobs": 0}))


if __name__ == "__main__":
    run()
