"""Authenticate and replay one completed fixed-grid energy triplet; no quantum work."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/research4_fixed_grid_energy_curvature_20261107_v2.py"
WORKER_SHA = "5f746f94a792b6d653c724774236410dbfea768a382462c6f6b375b79757b592"
SCHEMA = "research4-fixed-original-grid-central-energy-curvature-v2"
POINT_SCHEMA = "research4-fixed-original-grid-energy-point-v2"
POINT_SUCCESS = "FIXED_ORIGINAL_GRID_POLISHED_SCF_ENERGY_COMPUTED"
POLICY = "FULL_NATIVE_VEFF_ENDPOINT_CERTIFICATION_NOT_NATIVE_KERNEL_FLAG"
STEP = 0.0003125
NAMES = ("fresh-zero", "plus", "minus")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked(record):
    path = Path(record["path"]).resolve()
    require(path.is_file() and not Path(record["path"]).is_symlink()
            and path.stat().st_size <= 4 * 1024**2, "Missing, symlink or oversize metadata")
    require(sha(path) == record["sha256"], "Metadata SHA mismatch: " + str(path))
    require("bytes" not in record or record["bytes"] == path.stat().st_size, "Metadata size mismatch")
    return path


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def authenticate_closed(value, result_path):
    """Read the actual original closed phase without importing its executor."""
    require(checked(value["source_worker"]) == WORKER and sha(WORKER) == WORKER_SHA,
            "Unreviewed energy-v2 producer")
    supervision = value["resource_supervision"]
    proof_path = checked({"path": supervision["proof_path"], "sha256": supervision["proof_sha256"]})
    proof = read(proof_path)
    require(proof == supervision["proof"], "Launch proof content differs")
    ledger_path = Path(proof["ledger_path"]).resolve()
    require(ledger_path == ROOT / "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json",
            "Original registered ledger required")
    rows = [r for r in read(ledger_path)["phases"] if r.get("phase") == proof["phase"]]
    require(len(rows) == 1, "Unique actual producer phase required")
    row = rows[0]
    require(row.get("status") == "EXITED" and row.get("exit_code") == 0 and row.get("ended_at_utc"),
            "Complete healthy closed producer required")
    require(all(row.get(k) is False for k in
                ("wall_terminated", "hard_killed", "memory_terminated", "monitor_failed", "cpu_terminated")),
            "Producer resource or monitor failure")
    require(not row.get("residual_processes_require_review_after_main_payload_exit"), "Unresolved producer processes")
    require(row["supervisor_proof_sha256"] == sha(proof_path)
            and Path(row["supervisor_proof_path"]).resolve() == proof_path, "Closed proof binding differs")
    checked({"path": row["log"], "sha256": row["log_sha256"]})
    command = row["command"]
    require(str(WORKER) in command and command.count("--output") == 1, "Actual producer argv required")
    output = Path(command[command.index("--output") + 1])
    output = (output if output.is_absolute() else ROOT / output).resolve()
    require(output == result_path.parent, "Actual producer output differs")
    return {"phase": row["phase"], "actual_CPU_seconds": row["actual_cpu_seconds"],
            "closed_log": ref(Path(row["log"])), "launch_proof": ref(proof_path),
            "closed_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}


def validate_triplet(value, result_path):
    require(value.get("schema") == SCHEMA and value.get("status") == "ENERGY_CURVATURE_COMPUTED",
            "Only complete actual energy-v2 results may be analyzed")
    require(value.get("case") == "Spinosin" and value.get("method") == "FIXED_ORIGINAL_QUADRATURE"
            and value.get("predetermined_mode_index_1based") == 1
            and value.get("predetermined_step_bohr") == STEP, "Exact single mode-1 diagnostic required")
    require(value.get("all_three_fresh_energies_completed") is True
            and set(value.get("points", {})) == set(NAMES)
            and set(value.get("completed_points", [])) == set(NAMES), "Incomplete energy triplet")
    require(value.get("stable_MF_state_unchanged") is True
            and value.get("same_fresh_zero_MF_and_density_preserved_for_both_signs") is True,
            "Original and shared fresh-zero state must remain preserved")
    points = {}; paths = set()
    for name, sign in zip(NAMES, (0, 1, -1)):
        entry = value["points"][name]; path = checked(entry["result"])
        require(path == result_path.parent / name / "native_result.json" and path not in paths,
                "Each real child must be its unique expected sibling")
        paths.add(path); point = read(path)
        require(point.get("schema") == POINT_SCHEMA and point.get("status") == POINT_SUCCESS
                and entry.get("status") == POINT_SUCCESS and entry.get("energy_completed") is True
                and point.get("energy_completed") is True and point.get("SCF_performed") is True
                and point.get("scf_converged") is True, "A child energy is incomplete")
        require(point.get("source_worker") == value["source_worker"]
                and point.get("source_snapshot_manifest_sha256") == value["snapshot_manifest_sha256"]
                and Path(point["parent_output_directory"]).resolve() == result_path.parent,
                "Child producer, snapshot or parent binding differs")
        d = point["displacement"]
        require(d.get("mode_index_1based") == 1 and d.get("coordinate_index_0based") is None
                and d.get("sign") == sign and d.get("step_bohr") == (0. if sign == 0 else STEP)
                and d.get("direction_normalization") == "EUCLIDEAN_CARTESIAN_UNIT_NORM", "Wrong physical child point")
        energy = point.get("energy_hartree"); audit = point.get("final_SCF_endpoint_audit", {})
        require(finite(energy) and energy == entry.get("energy_hartree")
                and energy == value.get("energies_hartree", {}).get(name)
                and energy == value.get("energies", {}).get(name), "Parent energy differs from actual child")
        require(audit.get("qualified") is True and audit.get("certification_policy") == POLICY
                and audit.get("actual_native_final_extra_after_accepted_main_same_kernel") is True
                and audit.get("current_full_energy_hartree") == energy
                and finite(audit.get("full_consecutive_density_energy_delta_abs_hartree"))
                and audit["full_consecutive_density_energy_delta_abs_hartree"] < 1e-12
                and finite(audit.get("full_current_orbital_gradient_norm"))
                and 0 <= audit["full_current_orbital_gradient_norm"] < 1e-9,
                "Child lacks its strict full-native endpoint certificate")
        points[name] = {"energy_hartree": energy, "result": ref(path),
                        "native_kernel_converged": audit.get("native_kernel_converged")}
    e0, ep, em = (points[n]["energy_hartree"] for n in NAMES)
    numerator = math.fsum((ep, em, -2 * e0)); curvature = numerator / STEP**2
    reported = value["central_energy_analysis"]
    for key, actual in (("Ezero_hartree", e0), ("Eplus_hartree", ep), ("Eminus_hartree", em),
                        ("step_bohr", STEP), ("compensated_energy_numerator_hartree", numerator),
                        ("central_energy_curvature_Eh_per_Bohr2", curvature)):
        require(finite(reported.get(key)) and reported[key] == actual, "Reported scalar mismatch: " + key)
    analytic = reported.get("analytic_directional_curvature_Eh_per_Bohr2")
    require(finite(analytic) and reported.get("central_energy_minus_analytic_curvature_Eh_per_Bohr2") == curvature - analytic,
            "Reported analytic comparison mismatch")
    return points, {"step_bohr": STEP, "compensated_energy_numerator_hartree": numerator,
        "central_energy_curvature_Eh_per_Bohr2": curvature,
        "analytic_directional_curvature_Eh_per_Bohr2": analytic,
        "energy_minus_analytic_curvature_Eh_per_Bohr2": curvature - analytic,
        "binary64_ULP_scale_Eh_per_Bohr2": (math.ulp(ep) + math.ulp(em) + 2 * math.ulp(e0)) / STEP**2}


def analyze(result, expected, output):
    path = checked({"path": str(Path(result).resolve()), "sha256": expected})
    value = read(path)
    points, comparison = validate_triplet(value, path)
    closure = authenticate_closed(value, path)
    checked(value["source_result"])
    output = Path(output).resolve()
    require(not output.exists() and not output.is_symlink(), "Fresh output directory required")
    output.mkdir(parents=True)
    with (output / "energies.csv").open("x", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["point", "energy_hartree", "energy_minus_fresh_zero_hartree", "native_kernel_converged"])
        for name in NAMES:
            writer.writerow([name, points[name]["energy_hartree"], points[name]["energy_hartree"] - points["fresh-zero"]["energy_hartree"], points[name]["native_kernel_converged"]])
    with (output / "curvature_comparison.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison)); writer.writeheader(); writer.writerow(comparison)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        ax.bar(["Analytic directional H", "Central energy"], [comparison["analytic_directional_curvature_Eh_per_Bohr2"], comparison["central_energy_curvature_Eh_per_Bohr2"]], color=["#6b7280", "#176bba"])
        ax.axhline(0, color="black", lw=.7); ax.set_ylabel("Curvature (Eh / Bohr²)"); ax.set_title("Spinosin mode 1 · one fixed-grid step")
        fig.text(.5, .015, "Local energy diagnostic; no new Hessian, minimum or frequencies", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, 1))
        with (output / "curvature_comparison.png").open("xb") as handle:
            fig.savefig(handle, format="png", dpi=140)
        plt.close(fig); plot_format = "MATPLOTLIB_PNG"
    except ImportError:
        vals = [comparison["analytic_directional_curvature_Eh_per_Bohr2"], comparison["central_energy_curvature_Eh_per_Bohr2"]]
        text = '<svg xmlns="http://www.w3.org/2000/svg" width="650" height="140"><rect width="100%" height="100%" fill="white"/><text x="15" y="28">STDLIB SVG FALLBACK · one local energy diagnostic</text>'
        text += ''.join('<text x="15" y="%d">%s: %.12g Eh/Bohr²</text>' % (65+i*30, label, val) for i,(label,val) in enumerate(zip(("Analytic", "Central energy"), vals))) + '</svg>'
        with (output / "curvature_comparison.svg").open("x") as handle: handle.write(text + "\n")
        plot_format = "STDLIB_SVG_FALLBACK"
    summary = {"schema": "research4-fixed-grid-local-energy-analysis-v1", "status": "LOCAL_ENERGY_ANALYSIS_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_analyzer": ref(__file__), "source_result": ref(path),
        "closed_producer": closure, "points": points, "comparison": comparison, "plot_format": plot_format,
        "new_quantum_evaluations": 0, "diagnostic_status": "NOT_HESSIAN_NOT_MINIMUM_NOT_NEW_FREQUENCIES",
        "IR_intensities_status": "NOT_COMPUTED", "Spinosin_step_convergence_verified": False,
        "scientific_PASS_FAIL_threshold": "NOT_SET", "single_step_limitation": "One mode direction and one step cannot establish a Hessian, a minimum, or step convergence."}
    with (output / "result.json").open("x") as handle: json.dump(summary, handle, indent=2, allow_nan=False); handle.write("\n")
    with (output / "README.md").open("x") as handle:
        handle.write("# Local fixed-grid energy replay\n\nAuthenticated the completed energy-v2 producer and all three child JSON hashes. Recomputed the compensated energy numerator and curvature exactly.\n\nCentral-energy curvature: **%.12g Eh/Bohr²**. Analytic directional curvature: **%.12g Eh/Bohr²**. Difference: **%.12g Eh/Bohr²**.\n\nThis is one mode-1 direction at h=0.0003125 Bohr. It establishes no new Hessian, minimum, frequencies, IR intensities or step convergence. Binary64 scales are descriptive, not rigorous SCF error bounds.\n" % (comparison["central_energy_curvature_Eh_per_Bohr2"], comparison["analytic_directional_curvature_Eh_per_Bohr2"], comparison["energy_minus_analytic_curvature_Eh_per_Bohr2"]))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = analyze(args.result, args.result_sha256, args.output)
    print(json.dumps({"status": value["status"], "new_quantum_evaluations": 0}))


if __name__ == "__main__":
    main()
