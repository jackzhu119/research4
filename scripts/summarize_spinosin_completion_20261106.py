#!/usr/bin/env python3
"""Build a small, source-hashed completion report; never run quantum chemistry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/pyscf-spinosin-96h-continuation-20261105-v8"
WORK = BASE / "global-assemble-recovery-002"
AUDITS = ROOT / "reports/cphf-residual-audit-actual-20261106-v1"
CASES = ("water-atom000", "ethanol-atom000", "spinosin-atom000", "spinosin-atom036")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_file() and path.stat().st_size < 8 * 1024**2, f"Small input required: {path}")
    return json.loads(path.read_text())


def reference(path):
    require(path.stat().st_size < 8 * 1024**2, f"No giant inputs: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh report directory under reports")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports") and not output.exists(), "Fresh reports directory required")

    ledger = read(BASE / "cpu_budget.json")
    require(not any(p["status"] == "RUNNING" for p in ledger["phases"]), "Do not analyze while a Q phase is open")
    state = read(WORK / "state.json")
    require(state["invocations"][-1]["status"] == "PIPELINE_HEALTHY_CLOSED", "Actual terminal invocation required")
    require(state["analysis"]["status"] == "HEALTHY_CLOSED", "Actual independent analysis closure required")
    require(all(v["status"] == "HEALTHY_CLOSED" for v in state["stages"].values()), "Every main stage must be healthy")
    result = read(WORK / "science/global-assemble/result.json")
    checks = read(WORK / "independent-modes/checks.json")
    require(result["status"] == "FREQUENCIES_COMPUTED" and result["complete_global_Hessian"], "Full actual Hessian required")
    frequencies = result["frequencies_cm1"]
    thz = result["frequencies_thz"]
    require(len(frequencies) == len(thz) == result["normal_mode_count"] == 3 * result["atoms"] - 6 == 219, "Complete signed modes required")
    require(all(math.isfinite(v) for v in frequencies + thz), "Finite signed frequencies required")
    negatives = sum(v < 0 for v in frequencies)
    positives = sum(v > 0 for v in frequencies)
    zeros = sum(v == 0 for v in frequencies)
    require(negatives == result["imaginary_frequency_count"] == checks["imaginary_frequency_count"], "Imaginary count mismatch")
    require(checks["independent_numpy_replay"]["status"] == "PASS_INDEPENDENT_NUMPY_TR_REPLAY", "Independent algebra replay required")
    csv_path = WORK / "independent-modes/all_frequencies_cm1_THz.csv"
    with csv_path.open(newline="") as handle:
        frequency_rows = list(csv.DictReader(handle))
    require(len(frequency_rows) == 219, "Actual CSV must contain all 219 rows")
    require({int(r["mode"]) for r in frequency_rows} == set(range(1, 220)), "Unique complete mode IDs required")
    for row in frequency_rows:
        i = int(row["mode"]) - 1
        require(abs(float(row["frequency_cm1"]) - frequencies[i]) < 1e-7, "CSV/result frequency mismatch")

    phase_by_name = {p["phase"]: p for p in ledger["phases"]}
    require(len(phase_by_name) == len(ledger["phases"]), "Duplicate ledger phase")
    residual_rows, residual_cases, source_paths = [], [], []
    for name in CASES:
        path = AUDITS / (name + "-001/result.json")
        data = read(path)
        proof = data["resource_supervision"]["proof"]
        phase = phase_by_name[proof["phase"]]
        require(phase["status"] == "EXITED" and phase["exit_code"] == 0, "Actual audit closure required")
        require(data["fresh_native_vind_calls_completed"] == 1 and data["checked_nuclear_RHS"] == 3, "Exactly one fresh V and three RHS expected")
        require(data["old_producer_modified"] is False and data["no_SCF_gradient_Hessian_or_response_solver_run"] is True, "Saved source must stay unchanged")
        require(data["convergence_threshold_status"] == "NOT_SET_NOT_A_PASS_FAIL_GATE", "Do not invent a numerical gate")
        case_rows = []
        for row in data["metrics_by_nuclear_RHS"]:
            small = {
                "case": name, "atom_index_zero_based": row["atom_id"], "axis": row["axis"],
                "virtual_max_Eh_per_Bohr": row["virtual"]["max_abs"],
                "virtual_frobenius_Eh_per_Bohr": row["virtual"]["frobenius"],
                "preconditioned_max_per_Bohr": row["preconditioned"]["max_abs"],
                "occupied_gauge_max_per_Bohr": row["occupied_gauge"]["max_abs"],
                "relative_term_scaled_residual": row["relative_term_scaled_residual"],
                "source_result_sha256": data["source_response"]["sha256"],
                "new_pass_fail_threshold": "NOT_SET",
            }
            residual_rows.append(small)
            case_rows.append(small)
        residual_cases.append({
            "case": name, "atom_ids_zero_based": data["selected_atom_ids"],
            "checked_RHS": 3, "total_molecular_RHS": data["total_molecular_nuclear_RHS"],
            "all_molecular_RHS_checked": data["all_molecular_RHS_checked"],
            "max_virtual_residual_Eh_per_Bohr": max(x["virtual_max_Eh_per_Bohr"] for x in case_rows),
            "actual_guard_CPU_seconds": phase["actual_cpu_seconds"],
            "status": data["status"], "source": reference(path),
        })
        source_paths.append(path)

    # Only small final geometry/modes are read, after all actual Q phases closed.
    import numpy as np
    geometry_path = WORK / "science/global-assemble/normal_mode_geometry.xyz"
    lines = geometry_path.read_text().splitlines()
    atoms = [row.split() for row in lines[2:] if len(row.split()) >= 4]
    xyz = np.asarray([[float(x) for x in row[1:4]] for row in atoms])
    symbols = np.asarray([row[0] for row in atoms])
    require(xyz.shape == (75, 3), "Full original atom-order geometry required")
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    modes_path = WORK / "science/global-assemble/normal_modes.npy"
    require(modes_path.stat().st_size < 1024**2, "Small final mode array only")
    modes = np.load(modes_path, allow_pickle=False)
    require(modes.shape == (219, 75, 3), "Complete normal-mode array required")
    participation = []
    for i in range(min(3, negatives)):
        norm = np.einsum("ij,ij->i", modes[i], modes[i])
        total = norm.sum()
        top = np.argsort(norm)[-6:][::-1]
        participation.append({
            "mode": i + 1, "frequency_cm1": frequencies[i],
            "element_displacement_norm_fraction": {str(e): float(norm[symbols == e].sum() / total) for e in sorted(set(symbols))},
            "top_atoms_zero_based": [{"atom": int(a), "element": str(symbols[a]), "displacement_norm_fraction": float(norm[a] / total)} for a in top],
        })
    geometry = {
        "source_geometry": reference(geometry_path), "source_modes": reference(modes_path),
        "singular_values_A": singular.tolist(),
        "best_plane_RMS_A": float(singular[-1] / np.sqrt(75)),
        "smallest_largest_singular_ratio": float(singular[-1] / singular[0]),
        "negative_mode_displacement_participation": participation,
        "scope": "Small geometry SVD and Cartesian displacement norms; not IR, a chemical group assignment or a cause of negative modes",
    }

    controls = []
    for name in ("water", "ethanol"):
        directory = BASE / f"controls/v8-control-{name}-independent-modes-003"
        control = read(directory / "checks.json")
        with (directory / "all_frequencies_cm1_THz.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        controls.append({"case": name, "actual_signed_mode_rows": len(rows),
                         "imaginary_modes": control["imaginary_frequency_count"],
                         "all_positive": control["all_frequencies_positive"],
                         "actual_stock_full_control_comparison": control["actual_old_stock_control_comparison"],
                         "source": reference(directory / "checks.json")})

    response_phases = [phase_by_name[f"v8-spinosin-C001-response-pipeline-001-response-{i:02d}"] for i in range(10)]
    response_CPU = sum(p["actual_cpu_seconds"] for p in response_phases)
    legacy_n = ledger["source_v6b"]["phase_count"]
    new_phases = ledger["phases"][legacy_n:]
    recorded_new_CPU = sum(p.get("actual_cpu_seconds", 0) for p in new_phases)
    require(math.isclose(ledger["source_v6b"]["actual_closed_CPU_seconds"] + recorded_new_CPU,
                         ledger["consumed_cpu_seconds"], abs_tol=1e-7), "Cumulative retained-cost mismatch")
    low = [{"mode": i + 1, "frequency_cm1": f, "frequency_THz": thz[i],
            "qualified_for_mechanism": False, "IR_intensity": "NOT_COMPUTED"}
           for i, f in enumerate(frequencies) if f > 0 and 0.1 <= thz[i] <= 1.5]
    source_paths += [WORK / "science/global-assemble/result.json", WORK / "independent-modes/checks.json",
                     WORK / "independent-modes/artifact_index.json", WORK / "state.json", BASE / "cpu_budget.json",
                     BASE / "resource-protocol/actual-completion-independent-review-001.json", csv_path, Path(__file__)]
    summary = {
        "schema": "research4-spinosin-actual-completion-summary-v1",
        "status": "COMPUTATION_DONE_NOT_A_POSITIVE_MINIMUM_OR_CHEMICAL_MECHANISM_RESULT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_quantum_finished_at_utc": result["completed_at_utc"],
        "method": result["settings"], "environment": result["versions"],
        "atoms": result["atoms"], "basis_functions": result["basis_functions"],
        "occupied_orbitals": result["occupied_orbitals"], "energy_hartree": result["energy_hartree"],
        "normal_modes": {"complete": True, "signed": 219, "positive": positives, "imaginary": negatives, "zero": zeros,
                         "negative_range_cm1": [min(frequencies), max(f for f in frequencies if f < 0)],
                         "positive_0p1_to_1p5_THz": low},
        "stationary_minimum_verified": result["stationary_minimum_verified"],
        "frequency_validation_passed": result["frequency_validation_passed"],
        "qualified_for_mechanism": checks["qualified_for_mechanism"],
        "mechanism_pairing_status": checks["mechanism_pairing_status"],
        "IR_intensities_status": result["IR_intensities_status"],
        "MF_stability_flag_scope": checks["MF_stability_flag_scope"],
        "independent_numpy_replay": {k: v for k, v in checks["independent_numpy_replay"].items()
                                     if k != "independent_eigenvectors_and_subspaces"},
        "control_results": controls,
        "residual_audits": residual_cases,
        "Spinosin_selected_RHS_coverage": {"atom_ids_zero_based": [0, 36], "checked": 6, "total": 225,
                                          "all_RHS_verified": False, "new_convergence_threshold": "NOT_SET"},
        "resources": {"fixed_cumulative_CPU_cap_seconds": ledger["quantum_cpu_cap_seconds"],
                      "actual_cumulative_CPU_seconds": ledger["consumed_cpu_seconds"],
                      "remaining_CPU_seconds": ledger["remaining_quantum_cpu_seconds"],
                      "legacy_closed_CPU_carried": ledger["source_v6b"]["actual_closed_CPU_seconds"],
                      "actual_ten_response_CPU_seconds": response_CPU,
                      "new_phase_cost_rows_missing_actual_CPU": sum("actual_cpu_seconds" not in p for p in new_phases),
                      "deadline_utc": ledger["effective_run_deadline_at_utc"], "all_Q_closed": True,
                      "historical_failures_retained": True, "allocation_extended_or_refunded": False},
        "next_full_geometry_recalculation": {"status": "BLOCKED_REQUIRES_NEW_QUALIFIED_PROTOCOL_AND_BUDGET_ASSESSMENT",
                                            "changed_geometry_cannot_reuse_old_derivatives": True,
                                            "current_ten_response_CPU_exceeds_remaining_budget": response_CPU > ledger["remaining_quantum_cpu_seconds"]},
        "negative_mode_cause": "NOT_ESTABLISHED",
        "orbital_stability_finite_difference_curvature_grid_convergence": "NOT_RUN",
        "pure_standard_HPLC_LCMS_new_batches": "LAB_REQUIRED_NOT_RUN",
        "chemical_assignment_verified": checks["chemical_attribution_verified"],
        "theory_or_peaks_enter_classifier": checks["theory_or_peaks_enter_classifier"],
        "original_spectra_labels_historical_models_main_branch": "UNCHANGED",
        "source_artifacts": [reference(p) for p in source_paths],
    }
    require(not summary["stationary_minimum_verified"] and not summary["qualified_for_mechanism"], "Retain actual failed physical qualification")
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    write_json(output / "geometry_diagnostic.json", geometry)
    write_csv(output / "residual_summary.csv", residual_rows, list(residual_rows[0]))
    fields = ["phase", "status", "actual_cpu_seconds", "exit_code", "started_at_utc", "ended_at_utc",
              "cpu_terminated", "wall_terminated", "memory_terminated"]
    write_csv(output / "resource_costs.csv", new_phases, fields)
    (output / "README.md").write_text(
        "# Actual Spinosin computation\n\n"
        "219 signed modes were computed and independently replayed. There are 25 imaginary modes; "
        "the positive-minimum and chemical-mechanism gates are false. IR NOT_COMPUTED.\n\n"
        "The four residual audits measure selected equations, without a new convergence threshold. "
        "Spinosin coverage is only atoms 0 and 36: 6/225 nuclear RHS.\n\n"
        "Source-hashed summary generation only, no DFT is run by this script:\n\n"
        "```bash\nPYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site "
        ".runtime/bin/python -B scripts/summarize_spinosin_completion_20261106.py "
        "--output reports/spinosin-summary-replay-FRESH\n```\n\n"
        "Use the ledger from this Git revision for the same resource snapshot. Future appended phases "
        "change costs, not the immutable frequency result. Large H1/MO1 caches stay local with their "
        "original manifests and hashes; they are not required to rebuild this small summary.\n"
    )
    files = [reference(p) for p in sorted(output.iterdir()) if p.is_file()]
    write_json(output / "artifact_index.json", {"schema": "research4-completion-derived-artifacts-v1", "files": files})
    print(json.dumps({"status": summary["status"], "output": str(output), "signed_modes": 219,
                      "imaginary_modes": negatives, "qualified_for_mechanism": False,
                      "residual_case_count": len(residual_cases), "all_Q_closed": True}))


if __name__ == "__main__":
    main()
