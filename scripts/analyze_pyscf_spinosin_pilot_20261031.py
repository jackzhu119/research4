"""Read-only, gated analysis of actual PySCF pilot evidence; never run DFT."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CM1_TO_THZ = 0.0299792458
LOW, HIGH = .1, 1.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csvwrite(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def valid_modes(result, identity):
    """Computed and accepted modes are different from a completed test molecule."""
    count = 3 if identity == "water" else 219
    atom_count = 3 if identity == "water" else 75
    flags = [result.get("identity") == identity, result.get("status") == "DONE", result.get("atoms") == atom_count,
             result.get("geometry_optimized") is True, result.get("optimizer_converged") is True,
             result.get("scf_converged") is True, result.get("final_gradient_thresholds_passed") is True,
             result.get("frequencies_status") == "DONE", result.get("frequency_validation_passed") is True,
             result.get("all_frequencies_real_positive") is True,
             result.get("normal_modes_complete") is True, result.get("normal_mode_count") == count,
             result.get("expected_normal_mode_count") == count,
             result.get("harmonic_analysis_excludes_translations_and_rotations") is True]
    if not all(flags):
        return False
    values = np.asarray(result.get("frequencies_cm1", []), float)
    thz = np.asarray(result.get("frequencies_thz", []), float)
    return bool(values.shape == thz.shape == (count,) and np.isfinite(values).all()
                and np.all(values > 0) and np.allclose(thz, values * CM1_TO_THZ, atol=1e-12, rtol=1e-12)
                and result.get("imaginary_frequency_count") == 0
                and result.get("IR_intensities_status") == "NOT_COMPUTED")


def experimental_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        frequency, wavenumber = float(row["frequency_thz"]), float(row["wavenumber_cm1"])
        if not np.isfinite([frequency, wavenumber]).all() or not LOW <= frequency <= HIGH:
            raise ValueError("Experimental candidate frequency is outside the declared interval")
        if abs(frequency - wavenumber * CM1_TO_THZ) > 1e-10:
            raise ValueError("Experimental cm-1/THz conversion mismatch")
        rows.append({"outer_fold": int(row["outer_fold"]), "candidate_number": int(row["candidate_number"]),
                     "frequency_index": int(row["frequency_index"]), "experimental_frequency_thz": frequency,
                     "experimental_wavenumber_cm1": wavenumber, "chemical_assignment": "NOT_CONFIRMED"})
    if not rows:
        raise ValueError("No actual experimental candidates")
    return rows


def comparison_rows(experimental, theory):
    """All candidate/mode pairs; no closest-peak cherry-picking or attribution."""
    rows = []
    for observed in experimental:
        for mode in theory or [None]:
            rows.append({**observed,
                "theory_phase": None if mode is None else mode["phase"],
                "theoretical_mode": None if mode is None else mode["mode"],
                "theoretical_frequency_thz": None if mode is None else mode["frequency_thz"],
                "theoretical_wavenumber_cm1": None if mode is None else mode["frequency_cm1"],
                "absolute_difference_thz": None if mode is None else abs(observed["experimental_frequency_thz"] - mode["frequency_thz"]),
                "absolute_difference_cm1": None if mode is None else abs(observed["experimental_wavenumber_cm1"] - mode["frequency_cm1"]),
                "comparison_status": "BLOCKED_NO_VALIDATED_SPINOSIN_MODES_IN_RANGE" if mode is None else "DESCRIPTIVE_FREQUENCY_PAIR_ONLY",
                "IR_intensity_status": "NOT_COMPUTED", "chemical_attribution_verified": False})
    return rows


def xyz(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[0])
    rows = [line.split() for line in lines[2:] if line.strip()]
    if len(rows) != count or any(len(row) != 4 for row in rows):
        raise ValueError("Incomplete real geometry")
    coordinates = np.asarray([[float(x) for x in row[1:]] for row in rows])
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite real geometry")
    return [row[0] for row in rows], coordinates


def authorize(root, release):
    ledger_path = root / "cpu_budget.json"
    ledger = read(ledger_path)
    if not release.get("allow_analysis_export") or not release.get("created_at_utc") or release.get("cpu_budget_sha256") != sha(ledger_path):
        raise ValueError("Dated parent analysis release must bind final CPU ledger")
    if any(row.get("status") == "RUNNING" for row in ledger["phases"]):
        raise ValueError("A quantum phase is still running; do not claim completed evidence")
    used = [float(row["actual_cpu_seconds"]) for row in ledger["phases"]]
    if not np.isfinite(used).all() or any(value < 0 for value in used) or not np.isclose(sum(used), ledger["consumed_cpu_seconds"], atol=1e-7, rtol=1e-12):
        raise ValueError("Actual CPU phase totals do not match final ledger")
    if ledger["consumed_cpu_seconds"] > ledger["user_total_cpu_cap_seconds"]:
        raise ValueError("CPU ledger exceeds user budget")
    return ledger


def source_phase(root, row):
    command = row["command"]
    if "--output" not in command:
        raise ValueError("No worker output in actual recorded command")
    location = Path(command[command.index("--output") + 1])
    location = location if location.is_absolute() else ROOT / location
    if not location.resolve().is_relative_to(root.resolve()):
        raise ValueError("Recorded worker output is outside this pilot")
    path = location / "result.json"
    if path.exists():
        result = read(path)
    else:
        result = read(location / "progress.json") if (location / "progress.json").exists() else {}
        result.update(status="PARTIAL", worker_result_missing=True,
                      stop_reason="Recorded worker ended without a complete result.json")
    return location, result


def modes_from_phase(location, result):
    identity = result.get("identity")
    count = 3 if identity == "water" else 219
    natoms = 3 if identity == "water" else 75
    modes = np.load(location / "normal_modes.npy", allow_pickle=False)
    if modes.shape != (count, natoms, 3) or not np.isfinite(modes).all() or np.any(np.linalg.norm(modes.reshape(count, -1), axis=1) == 0):
        raise ValueError("Normal mode file does not match accepted actual frequencies")
    values = np.asarray(result.get("frequencies_cm1", []), float)
    thz_values = np.asarray(result.get("frequencies_thz", []), float)
    if values.shape != thz_values.shape or values.shape != (count,) or not np.isfinite(values).all() or not np.isfinite(thz_values).all():
        raise ValueError("Computed mode arrays are incomplete or nonfinite")
    hessian = np.load(location / "hessian_hartree_per_bohr2.npy", allow_pickle=False)
    if hessian.shape != (natoms, natoms, 3, 3) or not np.isfinite(hessian).all():
        raise ValueError("Missing/nonfinite Hessian for accepted normal modes")
    if np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2))) > 1e-5:
        raise ValueError("Saved Hessian is asymmetric")
    with (location / "frequencies.csv").open(newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != count:
        raise ValueError("Frequency CSV denominator mismatch")
    rows = []
    for i, (cm, thz, old) in enumerate(zip(result["frequencies_cm1"], result["frequencies_thz"], original), 1):
        if abs(float(old["frequency_cm1"]) - cm) > 1e-10 or abs(float(old["frequency_thz"]) - thz) > 1e-10:
            raise ValueError("Actual frequency CSV and result JSON disagree")
        rows.append({"phase": location.name, "identity": identity, "mode": i,
                     "frequency_cm1": cm, "frequency_thz": thz, "imaginary": bool(cm < 0),
                     "IR_intensity_status": "NOT_COMPUTED", "normal_mode_file_sha256": sha(location / "normal_modes.npy")})
    return rows, modes


def save_figure(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")


def atom_plot(ax, symbols, coordinates, vectors=None):
    colors = {"H": "#dddddd", "C": "#303030", "O": "#c73535"}
    for element in sorted(set(symbols)):
        positions = coordinates[np.asarray(symbols) == element]
        ax.scatter(*positions.T, c=colors.get(element, "#306fa0"), label=element,
                   s=18 if element == "H" else 48, edgecolors="gray", linewidths=.3)
    if vectors is not None:
        ax.quiver(*coordinates.T, *vectors.T, color="#1565b0", linewidths=.8, arrow_length_ratio=.2)
    center = coordinates.mean(0)
    extent = max(np.ptp(coordinates, axis=0).max() / 2, .5) * 1.15
    for setter, value in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(value - extent, value + extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="x / Angstrom", ylabel="y / Angstrom", zlabel="z / Angstrom")
    ax.view_init(20, 35)
    ax.legend()


def run(root, output, release_path):
    if output.exists():
        raise FileExistsError("Use a fresh analysis directory")
    release = read(release_path)
    ledger = authorize(root, release)
    peak_path = ROOT / "reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv"
    experimental = experimental_rows(peak_path)
    input_hashes = {str(root / "cpu_budget.json"): sha(root / "cpu_budget.json"), str(peak_path): sha(peak_path),
                    str(release_path): sha(release_path), str(Path(__file__)): sha(__file__)}
    phases, water, theory, computed_theory, geometries, steps, validated_theory = [], [], [], [], [], [], []
    for row in ledger["phases"]:
        location, result = source_phase(root, row)
        for filename in ["result.json", "progress.json", "input.json", "normal_modes.npy", "hessian_hartree_per_bohr2.npy", "frequencies.csv"]:
            path = location / filename
            if path.exists():
                input_hashes[str(path)] = sha(path)
        identity = result.get("identity", "UNKNOWN")
        phases.append({"phase": row["phase"], "identity": identity, "status": result.get("status", "PARTIAL"),
            "scf_converged": result.get("scf_converged"), "geometry_optimized": result.get("geometry_optimized", False),
            "frequencies_status": result.get("frequencies_status", "NOT_RUN"), "IR_intensity_status": "NOT_COMPUTED",
            "energy_hartree": result.get("energy_hartree"), "actual_cpu_seconds": row["actual_cpu_seconds"],
            "wall_seconds": row["elapsed_wall_seconds"], "exit_code": row["exit_code"],
            "atom_gradient_rms_hartree_per_bohr": result.get("atom_rms_hartree_per_bohr"),
            "atom_gradient_max_hartree_per_bohr": result.get("atom_max_hartree_per_bohr"),
            "stop_reason": result.get("stop_reason"), "result_path": str(location / "result.json")})
        if identity == "spinosin" and result.get("frequencies_status") == "DONE":
            rows, _ = modes_from_phase(location, result)
            computed_theory.extend({**entry, "accepted_for_comparison": valid_modes(result, identity),
                                    "validation_status": "ACCEPTED" if valid_modes(result, identity) else "COMPUTED_NOT_VALIDATED"}
                                   for entry in rows)
        if identity in ("water", "spinosin") and valid_modes(result, identity):
            rows, modes = modes_from_phase(location, result)
            if identity == "water":
                water.extend(rows)
            else:
                theory.extend(rows)
                validated_theory.append((location, result, rows, modes))
        if identity == "spinosin":
            for path in sorted((location / "geometry_steps").glob("step_*.json")):
                step = read(path)
                input_hashes[str(path)] = sha(path)
                steps.append({"phase": location.name, "phase_status": result.get("status", "PARTIAL"),
                              "cycle": step["cycle"], "energy_hartree": step["energy_hartree"],
                              "atom_rms_hartree_per_bohr": step["atom_rms_hartree_per_bohr"],
                              "atom_max_hartree_per_bohr": step["atom_max_hartree_per_bohr"],
                              "scf_converged": step["scf_converged"]})
            for filename, description in [("optimized.xyz", "DFT optimizer returned geometry; inspect convergence"),
                                          ("last_geometry.xyz", "Last actually evaluated DFT geometry; not convergence proof"),
                                          ("input.xyz", "Input computational conformer; not an experimental structure")]:
                path = location / filename
                if path.exists():
                    geometries.append((path, location.name, result.get("geometry_optimized", False), description))
                    break
    low_theory = [r for r in theory if LOW <= r["frequency_thz"] <= HIGH]
    paired = comparison_rows(experimental, low_theory)
    output.mkdir(parents=True)
    csvwrite(output / "phase_status_and_resources.csv", phases, list(phases[0]))
    csvwrite(output / "experimental_candidate_positions.csv", experimental, list(experimental[0]))
    freq_fields = ["phase", "identity", "mode", "frequency_cm1", "frequency_thz", "imaginary", "IR_intensity_status", "normal_mode_file_sha256"]
    csvwrite(output / "water_validated_frequencies.csv", water, freq_fields)
    csvwrite(output / "spinosin_computed_frequencies_including_unvalidated.csv", computed_theory,
             [*freq_fields, "accepted_for_comparison", "validation_status"])
    csvwrite(output / "spinosin_validated_all_frequencies.csv", theory, freq_fields)
    csvwrite(output / "spinosin_validated_0p1_1p5_thz_modes.csv", low_theory, freq_fields)
    csvwrite(output / "experiment_theory_pairs.csv", paired, list(paired[0]))
    csvwrite(output / "actual_optimization_steps.csv", steps, ["phase", "phase_status", "cycle", "energy_hartree", "atom_rms_hartree_per_bohr", "atom_max_hartree_per_bohr", "scf_converged"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot_water = [r for r in water if r["phase"] == water[-1]["phase"]] if water else []
    if plot_water:
        for ax, column, label in zip(axes, ["frequency_cm1", "frequency_thz"], ["Wavenumber / cm$^{-1}$", "Frequency / THz"]):
            values = [r[column] for r in plot_water]
            ax.vlines(values, 0, 1, color="#286090")
            for row, value in zip(plot_water, values):
                ax.text(value, 1.02, str(row["mode"]), ha="center", fontsize=8)
            ax.set(xlabel=label, ylim=(0, 1.15), yticks=[])
        fig.suptitle("Water: actual optimized molecular mode positions; unit heights, no IR intensities")
    else:
        for ax in axes:
            ax.text(.5, .5, "No validated water frequencies", transform=ax.transAxes, ha="center")
    save_figure(fig, output, "water_mode_positions")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for phase in dict.fromkeys(r["phase"] for r in steps):
        rows = [r for r in steps if r["phase"] == phase]
        cycles = [r["cycle"] for r in rows]
        axes[0].plot(cycles, [r["energy_hartree"] for r in rows], ".-", label=phase)
        axes[1].semilogy(cycles, [r["atom_rms_hartree_per_bohr"] for r in rows], ".-", label=phase + " RMS")
        axes[1].semilogy(cycles, [r["atom_max_hartree_per_bohr"] for r in rows], "--", label=phase + " max")
    axes[0].set(xlabel="Actually evaluated optimization cycle", ylabel="Energy / Hartree")
    axes[1].set(xlabel="Actually evaluated optimization cycle", ylabel="Atom-vector gradient / Hartree Bohr$^{-1}$")
    axes[1].axhline(3e-4, color="gray", ls=":", label="RMS convergence threshold")
    axes[1].axhline(4.5e-4, color="black", ls=":", label="max convergence threshold")
    if steps:
        for ax in axes:
            ax.legend(fontsize=6)
    else:
        axes[0].text(.5, .5, "No completed Spinosin geometry evaluation", transform=axes[0].transAxes, ha="center")
    fig.suptitle("Spinosin geometry trajectory: raw energy/gradient evidence, inspect phase completion status")
    save_figure(fig, output, "spinosin_optimization_progress")
    plt.close(fig)
    if geometries:
        path, phase, optimized, description = geometries[-1]
        input_hashes[str(path)] = sha(path)
        symbols, coordinates = xyz(path)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        atom_plot(ax, symbols, coordinates)
        ax.set_title(f"Spinosin C28H32O15: {phase}\nOptimized checks passed: {optimized}; generated model geometry", fontsize=10)
        save_figure(fig, output, "spinosin_model_geometry")
        plt.close(fig)
        (output / "displayed_geometry_provenance.json").write_text(json.dumps({"source": str(path), "sha256": sha(path), "description": description, "geometry_optimized": optimized, "bonds_drawn": False}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for row in experimental:
        axes[0].plot(row["experimental_frequency_thz"], row["outer_fold"], "|", color="#ac5423", ms=16)
    axes[0].set(xlabel="Frequency / THz", ylabel="Historical source fold", xlim=(LOW, HIGH), yticks=range(5), title="Experimental candidate positions")
    if low_theory:
        for row in low_theory:
            axes[1].plot(row["frequency_thz"], row["mode"], "|", color="#286090", ms=10)
        axes[1].set(ylabel="Actual theoretical mode index", title="Validated isolated Spinosin mode positions")
    else:
        axes[1].text(.5, .5, "No validated Spinosin modes in 0.1–1.5 THz\nComparison BLOCKED\nWater modes are not herbal modes", transform=axes[1].transAxes, ha="center", va="center")
        axes[1].set(yticks=[], title="No theoretical absorption curve generated")
    axes[1].set(xlabel="Frequency / THz", xlim=(LOW, HIGH))
    fig.suptitle("Position comparison only: no IR intensities, no chemical attribution")
    save_figure(fig, output, "mechanism_comparison_status")
    plt.close(fig)
    for location, result, rows, modes in validated_theory:
        geometry = location / "input.xyz"
        symbols, coordinates = xyz(geometry)
        for row in rows:
            if not LOW <= row["frequency_thz"] <= HIGH:
                continue
            mode = modes[row["mode"] - 1]
            scale = .4 / max(np.linalg.norm(mode, axis=1).max(), 1e-30)
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            atom_plot(ax, symbols, coordinates, mode * scale)
            ax.set_title(f"Spinosin mode {row['mode']}: {row['frequency_cm1']:.4f} cm-1\nArbitrary display-scaled vectors; no IR intensity", fontsize=10)
            stem = f"{location.name}_mode_{row['mode']:03d}"
            save_figure(fig, output, stem)
            plt.close(fig)
            frames = []
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                display = coordinates + np.sin(angle) * mode * scale
                frames.extend([str(len(symbols)), "Arbitrary display amplitude, not thermal motion or IR intensity"] +
                              [f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(symbols, display)])
            (output / (stem + ".xyz")).write_text("\n".join(frames) + "\n")
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_only": True,
               "new_SCF_gradient_Hessian_jobs": 0, "water_validated_frequency_rows": len(water),
               "computed_spinosin_frequency_rows_including_unvalidated": len(computed_theory),
               "validated_spinosin_frequency_rows": len(theory), "validated_spinosin_modes_in_0p1_1p5_thz": len(low_theory),
               "spinosin_frequency_interpretation_status": "DONE_DESCRIPTIVE_POSITIONS_ONLY" if theory else "BLOCKED_NO_VALIDATED_DFT_MODES",
               "chemical_attribution_verified": False, "classification_modified": False, "IR_intensity_status": "NOT_COMPUTED",
               "experimental_candidate_rows": len(experimental), "comparison_rows": len(paired),
               "all_mode_markers_are_unit_height_not_IR_intensities": True,
               "remaining_limits": ["Isolated molecule is not solid herbal THz spectrum", "No pure-standard THz or content assay confirms assignment", "Frequency proximity does not identify a compound", "Grid/basis/conformer sensitivity and intermolecular/crystal effects remain"],
               "input_sha256": input_hashes}
    for path, expected in input_hashes.items():
        if sha(path) != expected:
            raise ValueError("Input evidence changed during analysis: " + path)
    write(output / "summary.json", summary)
    text = f"""# PySCF实际结果分析与机制边界

本目录为只读后处理，新增SCF、梯度或Hessian计算均为0。所有图来自已结束阶段的真实日志或坐标；不会把水的频率当作斯皮诺素频率。

水分子实际通过检查的频率记录：{len(water)}行（不同后端复测不是新分子）。斯皮诺素通过全部频率检查的记录：{len(theory)}行；其中0.1—1.5 THz：{len(low_theory)}个模式。

斯皮诺素频率解释状态：**{summary['spinosin_frequency_interpretation_status']}**。未通过优化／驻点／完整模式／正频率检查时，比较表的理论频率和差值保持空值；不使用水或合成测试数值填空。

`water_validated_frequencies.csv`为实际技术测试；`phase_status_and_resources.csv`、`actual_optimization_steps.csv`分别保留DONE／PARTIAL阶段与实际梯度轨迹；`experiment_theory_pairs.csv`列出全部实际实验候选及可用的理论模式笛卡尔对照，不按峰位接近挑构象。频率位置图使用等高标记，**不代表IR强度**，IR始终NOT_COMPUTED。模式箭头／动画仅作任意幅度显示。

孤立分子的分子内振动不包含晶格、复杂基质、分子间氢键和集体振动。当前没有纯品THz和化学含量证据，任何频率接近都不能确认化学归属。本次不修改分类模型，也不声称提高96.0234%准确率。

`summary.json`绑定输入摘要和完整条件；`files_sha256.json`覆盖本目录其它产物。
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    write(output / "files_sha256.json", {str(path.relative_to(output)): {"sha256": sha(path), "bytes": path.stat().st_size}
                                         for path in sorted(output.rglob("*")) if path.is_file()})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=ROOT / "reports/pyscf-spinosin-pilot-20261031")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    a = p.parse_args()
    result = run(a.run_root.resolve(), a.output.resolve(), a.release.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
