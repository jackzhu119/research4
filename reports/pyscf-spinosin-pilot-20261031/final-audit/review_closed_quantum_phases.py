"""Audit closed saved quantum phases, without evaluating quantum chemistry.

Reads completed files/checkpoints only after the parent confirms worker exit.
No SCF, gradients, Hessian evaluations, optimization or Git mutations.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
PILOT = HERE.parent
BASE = "3fc996e99cce8ff5799b5a2af9b565f982dfc899"
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
import h5py  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.scf import chkfile  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(gradient):
    atom_norms = np.linalg.norm(gradient, axis=1)
    return {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(gradient**2))),
            "component_max_hartree_per_bohr": float(np.abs(gradient).max()),
            "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(atom_norms**2))),
            "atom_max_hartree_per_bohr": float(atom_norms.max()),
            "gradient_norm_hartree_per_bohr": float(np.linalg.norm(gradient))}


def main():
    out = HERE / "closed_phases_independent_review.json"
    if out.exists():
        raise FileExistsError("Preserve audits")
    ledger_path = PILOT / "cpu_budget.json"
    ledger = json.loads(ledger_path.read_text())
    assert all(phase["status"] != "RUNNING" and "ended_at_utc" in phase for phase in ledger["phases"])
    assert all(not Path(f"/proc/{phase['pid']}").exists() for phase in ledger["phases"])
    cpu_sum = sum((Decimal(str(phase["actual_cpu_seconds"])) for phase in ledger["phases"]), Decimal(0))
    cpu_match = abs(cpu_sum - Decimal(str(ledger["consumed_cpu_seconds"]))) < Decimal("1e-9")
    remainder = Decimal(ledger["quantum_cpu_cap_seconds"]) - cpu_sum
    assert cpu_match and remainder > 0
    assert abs(remainder - Decimal(str(ledger["remaining_quantum_cpu_seconds"]))) < Decimal("1e-9")
    logs_valid = []
    for phase in ledger["phases"]:
        log = ROOT / phase["log"]
        assert sha(log) == phase["log_sha256"]
        assert phase["actual_cpu_seconds"] <= phase["granted_cpu_seconds"]
        logs_valid.append({"phase": phase["phase"], "actual_cpu_seconds": phase["actual_cpu_seconds"],
                           "granted_cpu_seconds": phase["granted_cpu_seconds"],
                           "exit_code": phase["exit_code"], "peak_rss_kib": phase["peak_rss_kib"],
                           "log_sha256_matches_ledger": True})

    phase_dir = PILOT / "spinosin-cached-optimization"
    final = json.loads((phase_dir / "result.json").read_text())
    checkpoint = phase_dir / "calculation.chk"
    assert final["checkpoint_sha256"] == sha(checkpoint)
    assert final["script_sha256"] == sha(ROOT / "scripts/pyscf_pilot_cached_df_worker_20261031.py")
    assert final["status"] == "PARTIAL" and not final["geometry_optimized"] and not final["optimizer_converged"]
    assert final["frequencies_status"] == "NOT_RUN" and final["IR_intensities_status"] == "NOT_COMPUTED"
    spino_log = (phase_dir / "pyscf.log").read_text()
    converged_log_energies = [float(value) for value in re.findall(r"converged SCF energy\s*=\s*([-+0-9.eE]+)", spino_log)]
    step_paths = sorted((phase_dir / "geometry_steps").glob("step_*.json"))
    assert len(step_paths) == 4
    steps = []
    for i, step_path in enumerate(step_paths):
        saved = json.loads(step_path.read_text())
        array_path = step_path.with_suffix(".gradient.npy")
        xyz_path = step_path.with_suffix(".xyz")
        gradient = np.load(array_path, allow_pickle=False)
        assert gradient.shape == (75, 3) and np.isfinite(gradient).all()
        recalc = stats(gradient)
        assert saved["cycle"] == i + 1 and saved["scf_converged"]
        assert all(np.isclose(value, saved[key], rtol=1e-13, atol=1e-15) for key, value in recalc.items())
        matches_log = abs(saved["energy_hartree"] - converged_log_energies[i]) < 1e-8
        assert matches_log
        checks_pass = recalc["atom_rms_hartree_per_bohr"] <= 3e-4 and recalc["atom_max_hartree_per_bohr"] <= 4.5e-4
        assert checks_pass == saved["final_gradient_thresholds_passed"]
        steps.append({"cycle": saved["cycle"], "paired_energy_hartree": saved["energy_hartree"],
                      "saved_SCF_converged": saved["scf_converged"],
                      "energy_matches_corresponding_converged_SCF_log": matches_log,
                      "recomputed_gradient_statistics": recalc, "gradient_thresholds_passed": checks_pass,
                      "evidence": {path.name: sha(path) for path in (step_path, array_path, xyz_path)}})
    last = steps[-1]
    assert final["last_geometry_cycle"] == last["cycle"]
    assert final["last_geometry_energy_hartree"] == last["paired_energy_hartree"]
    assert len(converged_log_energies) == 5 and spino_log.rstrip().endswith("grid_response = False")
    with h5py.File(checkpoint, "r") as handle:
        checkpoint_energy = float(handle["scf/e_tot"][()])
        checkpoint_keys = sorted(handle.keys())
    checkpoint_mol = chkfile.load_mol(str(checkpoint))
    last_xyz = (phase_dir / "geometry_steps/step_0004.xyz").read_text().splitlines()
    last_mol = gto.M(atom="\n".join(last_xyz[2:]), basis="def2-svp", unit="Angstrom", verbose=0)
    checkpoint_vs_last = float(np.max(np.abs(checkpoint_mol.atom_coords() - last_mol.atom_coords())))
    assert checkpoint_mol.natm == 75 and checkpoint_mol.nelectron == 320 and checkpoint_mol.nao_nr() == 762
    assert abs(checkpoint_energy - converged_log_energies[-1]) < 1e-8 and checkpoint_vs_last > 1e-6
    spino_frequency_artifacts = [str(path.relative_to(ROOT)) for pattern in ("*hessian*.npy", "normal_modes.npy", "frequencies.csv")
                                for path in phase_dir.glob(pattern)]
    assert not spino_frequency_artifacts

    water = []
    for name in ("water-optfreq", "water-cached-optfreq"):
        directory = PILOT / name
        state = json.loads((directory / "result.json").read_text())
        gradient = np.load(directory / "gradient_hartree_per_bohr.npy", allow_pickle=False)
        hessian = np.load(directory / "hessian_hartree_per_bohr2.npy", allow_pickle=False)
        modes = np.load(directory / "normal_modes.npy", allow_pickle=False)
        grad = stats(gradient)
        assert all(np.isclose(value, state[key], rtol=1e-13, atol=1e-15) for key, value in grad.items())
        flags = {key: state[key] for key in ("optimizer_converged", "geometry_optimized", "scf_converged",
                    "final_gradient_thresholds_passed", "normal_modes_complete", "all_frequencies_real_positive",
                    "frequency_validation_passed", "stationary_minimum_verified")}
        assert state["status"] == "DONE" and all(flags.values())
        assert state["normal_mode_count"] == state["expected_normal_mode_count"] == 3 and state["imaginary_frequency_count"] == 0
        assert np.all(np.asarray(state["frequencies_cm1"]) > 0)
        assert np.allclose(np.asarray(state["frequencies_cm1"]) * .0299792458, state["frequencies_thz"], atol=1e-10)
        assert state["checkpoint_sha256"] == sha(directory / "calculation.chk")
        assert hessian.shape == (3, 3, 3, 3) and modes.shape == (3, 3, 3) and np.isfinite(hessian).all()
        symmetry = float(np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2))))
        assert symmetry < 1e-5 and grad["atom_rms_hartree_per_bohr"] <= 3e-4 and grad["atom_max_hartree_per_bohr"] <= 4.5e-4
        water.append({"phase": name, "all_actual_acceptance_flags": flags,
                      "gradient_statistics_recomputed": grad, "energy_hartree": state["energy_hartree"],
                      "positive_modes": 3, "imaginary_modes": 0, "frequencies_cm1": state["frequencies_cm1"],
                      "frequencies_THz": state["frequencies_thz"], "hessian_max_asymmetry": symmetry,
                      "IR_intensities": "NOT_COMPUTED", "result_sha256": sha(directory / "result.json")})

    helper_path = PILOT / "delivery-preflight/audit_delivery_before_stop.py"
    spec = importlib.util.spec_from_file_location("closed_review_readonly_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    index_checks = [helper.verify_sealed_index(path) for path in sorted(PILOT.rglob("files_sha256*.json"))
                    if HERE not in path.parents]
    assert all(record["all_match"] for record in index_checks)
    changes = subprocess.check_output(["git", "diff", "--name-only", "-z", BASE], cwd=ROOT).decode().split("\0")
    changes = [path for path in changes if path]
    numstat = subprocess.check_output(["git", "diff", "--numstat", BASE], cwd=ROOT).decode().splitlines()
    assert set(changes) <= {"AGENTS.md", "README.md"}
    assert all(line.split("\t")[1] == "0" for line in numstat)
    oversized = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
                 for path in PILOT.rglob("*") if path.is_file() and path.stat().st_size >= 10 * 1024**2]
    assert not oversized
    direct_state = json.loads((PILOT / "spinosin-scf-gradient/result.json").read_text())
    assert direct_state["checkpoint_sha256"] == sha(PILOT / "spinosin-scf-gradient/calculation.chk")
    assert direct_state["status"] == "PARTIAL"
    result = {
        "status": "PASS_INDEPENDENT_EVIDENCE_AUDIT_WITH_SPINOSIN_SCIENTIFIC_STAGE_PARTIAL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)), "new_quantum_jobs": 0,
        "all_recorded_workers_exited_before_checkpoint_read": True,
        "CPU": {"independent_sum_decimal_seconds": str(cpu_sum), "recorded_consumed_seconds": ledger["consumed_cpu_seconds"],
                "matches_ledger": cpu_match, "minutes": float(cpu_sum / 60),
                "remaining_quantum_cpu_seconds": ledger["remaining_quantum_cpu_seconds"],
                "quantum_cap_seconds": ledger["quantum_cpu_cap_seconds"], "ledger_sha256": sha(ledger_path),
                "per_phase": logs_valid,
                "non_quantum_reserved_seconds": 300,
                "non_quantum_reserve_is_not_a_measurement_of_all_preparation_CPU": True},
        "Spinosin": {"scientific_status": "PARTIAL_REAL_SCF_GRADIENT_AND_OPTIMIZATION_ATTEMPT",
            "complete_paired_geometry_steps": steps, "last_complete_SCF_gradient_geometry_cycle": 4,
            "last_complete_paired_energy_hartree": last["paired_energy_hartree"],
            "last_complete_gradient_statistics": last["recomputed_gradient_statistics"],
            "last_complete_gradRMS_threshold_factor": last["recomputed_gradient_statistics"]["atom_rms_hartree_per_bohr"] / 3e-4,
            "last_complete_gradmax_threshold_factor": last["recomputed_gradient_statistics"]["atom_max_hartree_per_bohr"] / 4.5e-4,
            "optimizer_converged": False, "stationary_structure_verified": False,
            "final_result_sha256": sha(phase_dir / "result.json"), "final_checkpoint_sha256": sha(checkpoint),
            "final_checkpoint_matches_worker_declared_sha256": True,
            "raw_result_energy_matches_last_complete_geometry_in_this_run": final["energy_hartree"] == last["paired_energy_hartree"],
            "checkpoint": {"root_keys": checkpoint_keys, "SCF_energy_saved_hartree": checkpoint_energy,
                "atom_count": checkpoint_mol.natm, "electrons": checkpoint_mol.nelectron, "AO": checkpoint_mol.nao_nr(),
                "geometry_differs_from_last_complete_step_max_abs_Bohr": checkpoint_vs_last,
                "matching_latest_logged_converged_SCF_at_next_trial_geometry": True,
                "next_trial_gradient_completed": False, "stationary_geometry_claim": False},
            "next_trial_SCF_log_energy_hartree": converged_log_energies[-1],
            "next_trial_gradient_state": "ENTERED_GRADIENT_SCANNER_THEN_INTERRUPTED_NO_COMPLETED_STEP5",
            "completed_hessian_evaluations": 0, "computed_normal_modes": 0,
            "frequency_status": "NOT_RUN", "IR_intensities": "NOT_COMPUTED",
            "frequency_artifacts": spino_frequency_artifacts,
            "interpretation": "Use only each complete geometry JSON with its matching gradient/XYZ as paired energy-gradient evidence. Fifth-trial SCF energy/checkpoint must not be paired with the step4 gradient or treated as an optimized minimum."},
        "water_acceptance_review": water,
        "historical_preservation": {"base_commit": BASE, "tracked_changes": changes, "numstat": numstat,
            "only_authorized_additive_notes_changed": True, "old_data_models_reports_code_unchanged_by_Git_comparison": True,
            "comparison_scope": "All Git-tracked paths against base; ignored runtime files are outside this assertion."},
        "sealed_indices": index_checks,
        "all_sealed_indices_pass": True,
        "large_file_threshold_10MiB_hits": oversized,
        "direct_J_interrupted_phase_and_checkpoint_preserved": True,
        "remaining_limits": ["No converged Spinosin geometry or Spinosin vibrational result was obtained.",
            "PBE/grid1/noD3 single-conformer pilot is not a solid-state THz assignment or grid/basis convergence study.",
            "A/B complete DFT calculations remain NOT_RUN; existing classification performance is not changed."]}
    with out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "CPU_seconds": ledger["consumed_cpu_seconds"],
                      "complete_Spinosin_geometry_steps": len(steps),
                      "last_complete_energy_hartree": last["paired_energy_hartree"],
                      "last_complete_gradient": last["recomputed_gradient_statistics"],
                      "checkpoint_trial_SCF_energy_hartree": checkpoint_energy,
                      "water_actual_ACCEPT": True,
                      "sealed_indices_checked": len(index_checks),
                      "sealed_files_checked": sum(record["checked_files"] for record in index_checks),
                      "original_data_models_reports_unchanged": True,
                      "review_sha256": sha(out)}, indent=2))


if __name__ == "__main__":
    main()
