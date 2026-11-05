"""V6b-only C001 mode-group adapter. Numerical results remain NOT_RUN at prep.

The old V5/a2 gates are not changed or used as V6b gates. This module imports
only stdlib; all numerical libraries and the old participation helper are
deferred until actual closed V6b modes and the new independent export qualify.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v6b_v2.py"
ANALYZER_SHA = "38dde5e185c4fc48d58732982e0164b3a2f5430a26bdfc1076b21103d1d75c6a"
OLD_GROUPS = ROOT / "scripts/analyze_spinosin_group_participation_20261104.py"
OLD_GROUPS_SHA = "b997a51f1d57b29cb2f05b67209bf993d1882af504a4af164c6912ed497c2dcd"
STATIC = ROOT / "reports/external-evidence-20261104-v1/mode-interpretation-preparation"
STATIC_INDEX_SHA = "5ecf8bbe7c5317d5ef0d18070f01307542c366f0700bcbaf43e5e7849a5aa246"
PREPARED_XYZ = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/optimized.xyz"
PREPARED_XYZ_SHA = "f7c69d098159254cf22260b3c11c6b63a865839ce942e5a7ca0982da1ec73e39"
SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
SOURCE_SDF_SHA = "02dc923dce2f225ede414341229f3090967e569f6075165a41685b3a5d8da2f7"
THERMO = ROOT / ".runtime/quantum-upgrade-site/pyscf/hessian/thermo.py"
THERMO_SHA = "8143ebed7653bd79b4b8d7b2f5933671a33285b11aba47cf0d8134c229a031f8"
PARTITION, OVERLAP = "DISJOINT_PARTITION", "OVERLAPPING_GRAPH_SUBGROUP"
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}
CM1_TO_THZ = .0299792458
FORMULA = "sum_group(m_i*sum_xyz(d_i**2))/sum_all(m_i*sum_xyz(d_i**2))"
LIMITATIONS = ("Dimensionless mass-weighted squared Cartesian normal-mode shares, not IR intensity, energy or absorption. "
    "Exactly four source-defined DISJOINT_PARTITION groups form ONE complete atom partition; "
    "the other18 OVERLAPPING_GRAPH_SUBGROUP groups overlap and must not be summed to one. "
    "Sign/global scale invariant; individual-mode fractions can change under rotations inside a degenerate eigenspace. "
    "No phase/torsion/bending or hydrogen-bond mechanism is inferred from these fractions. "
    "Original C001 ordered graph only; later SDF automorphisms do not supply a unique atom map. "
    "One isolated conformer is not solid herbal material or a global-minimum search. "
    "No shares/frequencies enter a classifier or modify historical96.0234%; no chemical peak assignment.")


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def checked(path, hashes, expected, *, maximum=16 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum, "Missing/symlink/oversize input")
    require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected, "Input SHA changed: " + str(path))
    path = path.resolve(); hashes[str(path)] = {"sha256": expected, "bytes": path.stat().st_size}
    return path


def read_json(path, hashes, expected):
    return json.loads(checked(path, hashes, expected).read_text())


def load_frozen(path, expected, name):
    require(sha(path) == expected, "Frozen source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def modes_analyzer():
    # This reviewed analyzer's top level is stdlib only. Never call its
    # numerical_runtime or its actual qualify() from a preflight.
    return load_frozen(ANALYZER, ANALYZER_SHA, "frozen_actual_V6b_modes_analyzer_v2")


def static_package(hashes):
    analyzer = modes_analyzer()
    pure = analyzer.frozen_metadata_subset(OLD_GROUPS, OLD_GROUPS_SHA,
        ("require", "sha", "checked", "read_json", "validate_groups", "static_groups"),
        ("PARTITION", "OVERLAP", "STATIC_INDEX_SHA", "STATIC_HASHES", "PREPARED_XYZ_SHA", "SOURCE_SDF_SHA"),
        {"STATIC": STATIC, "PREPARED_XYZ": PREPARED_XYZ, "Counter": Counter, "csv": csv})
    groups, symbols = pure.static_groups(hashes)
    index = read_json(STATIC / "files_sha256.json", hashes, STATIC_INDEX_SHA)
    for row in index["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe static package path")
        path = checked(STATIC / relative, hashes, row["sha256"])
        require(path.stat().st_size == row["bytes"], "Static package size differs")
    audit = read_json(STATIC / "atom_order_audit.json", hashes, pure.STATIC_HASHES["atom_order_audit.json"])
    # Recheck every sealed SMALL original graph/recipe/XYZ source. No NPY/CHK,
    # current target geometry array, Hessian/cache or quantum engine is used.
    for value in audit["inputs"].values():
        path = (ROOT / value["path"]).resolve()
        require(path.is_relative_to(ROOT) and path.suffix.lower() in (".json", ".sdf", ".xyz", ".py"), "Static graph source is not allowlisted text")
        checked(path, hashes, value["sha256"])
    trusted_record = audit["inputs"]["trusted_source"]
    trusted = read_json(ROOT / trusted_record["path"], hashes, trusted_record["sha256"])
    require(trusted["name"] == "Spinosin" and trusted["formula"] == "C28H32O15"
            and trusted["expected_total_atoms_including_hydrogen"] == 75
            and trusted["source_inchikey"] == audit["trusted_source_and_C001_and_upload_inchikey"]
            == audit["current_prepared_geometry_3D_stereokey"] == "VGGSULWDCMWZPO-ODEMIOGVSA-N"
            and audit["source_and_C001_indexed_graph_signatures_equal"] is True,
            "Original C001 trusted ordered-graph proof differs")
    checked(THERMO, hashes, THERMO_SHA)  # Inspect bytes only, never import PySCF.
    require(len(groups) == 22 and sum(g["family"] == PARTITION for g in groups) == 4
            and sum(g["family"] == OVERLAP for g in groups) == 18, "Source-defined structural group count differs")
    return {"groups": groups, "symbols": symbols, "audit": audit, "trusted": trusted,
            "metadata_only": True, "source_graph_verification": "SEALED_PREVIOUSLY_VALIDATED_ORDERED_GRAPH_AND_ALL_TEXT_SOURCE_HASHES_RECHECKED"}


def validate_analysis_checks(value):
    exact = {"schema": "research4-V6b-independent-staged-modes-analysis-v1", "status": "DONE_COMPUTED_MODES_AUDITED",
        "case": "Spinosin", "analysis_source_sha256": ANALYZER_SHA,
        "actual_V6b_worker_sha256": "f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d",
        "actual_V6b_parent_sha256": "b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5",
        "old_a2_V6b_compatibility_claimed": False, "snapshot_manifest_sha256": SNAPSHOT_SHA,
        "atoms": 75, "normal_modes": 219, "normal_mode_count_complete": True,
        "global_completion_status": "ACTUAL_FULL_HESSIAN_AND_ALL_SIGNED_MODES_AUDITED",
        "all_frequencies_positive": True, "imaginary_frequency_count": 0, "zero_frequency_count": 0,
        "qualified_for_mechanism": True, "minimum_qualification_separate_from_completion": True,
        "metadata_provenance_gates_before_numpy_or_array_access": True,
        "IR_intensities_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
        "theory_or_peaks_enter_classifier": False, "new_quantum_evaluations": 0}
    require(all(type(value.get(key)) is type(wanted) and value[key] == wanted for key, wanted in exact.items()),
            "BLOCKED: only original positive219 Spinosin V6b modes audited by the frozen v2 analyzer qualify")
    require(value["restored_gradient_gate"]["result"]["sha256"] == GRADIENT_SHA
            and value["independent_numpy_replay"]["status"] == "PASS_INDEPENDENT_NUMPY_TR_REPLAY",
            "Fresh-G or actual full normal-mode replay missing")


def validate_positive_frequencies(result):
    cm, thz = result["frequencies_cm1"], result["frequencies_thz"]
    require(len(cm) == len(thz) == 219 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in cm + thz)
            and all(abs(t - c * CM1_TO_THZ) <= 1e-12 for c, t in zip(cm, thz)),
            "BLOCKED: missing/nonpositive/zero actual signed frequencies; no fractions computed")
    require(result["imaginary_frequency_count"] == 0 and all(result.get(k) is True for k in
        ("all_frequencies_real_positive", "stationary_minimum_verified", "frequency_validation_passed")),
        "Original result does not qualify as an all-positive local minimum")
    return cm, thz


def terminal_quantum(ledger_path, expected_sha, hashes, analyzer):
    path = Path(ledger_path).resolve()
    require(path.is_relative_to(ROOT / "reports"), "Explicit real quantum ledger required")
    ledger = read_json(path, hashes, expected_sha)
    guard = analyzer.frozen_module(analyzer.GUARD, analyzer.GUARD_SHA, "group_closed_V6b_quantum_parent")
    guard.validate_ledger(ledger, allow_running=False)
    require(ledger["phases"] and all(p.get("status") != "RUNNING"
        and type(p.get("actual_cpu_seconds")) in (int, float) and math.isfinite(p["actual_cpu_seconds"])
        and p["actual_cpu_seconds"] >= 0 for p in ledger["phases"]), "All quantum phases must be closed with known CPU")
    return {"path": str(path), "sha256": expected_sha, "closed_phases": len(ledger["phases"])}


def qualify_analysis(directory, index_sha, ledger_path, ledger_sha, hashes):
    """Only metadata here; current active Q forbids calling this in preflight."""
    analyzer = modes_analyzer()
    terminal = terminal_quantum(ledger_path, ledger_sha, hashes, analyzer)
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / "reports"), "Explicit closed analysis directory required")
    index = read_json(directory / "artifact_index.json", hashes, index_sha)
    require(index["analysis_source_sha256"] == ANALYZER_SHA
            and index["a2_numerical_helper_sha256"] == analyzer.A2_SHA
            and index["V6b_worker_sha256"] == analyzer.WORKER_SHA, "Actual reviewed v2 analysis source/index required")
    records = index["files"]
    require(len({r["path"] for r in records}) == len(records), "Duplicate export path")
    for row in records:
        path = Path(row["path"])
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe exported artifact path")
    check_rows = [r for r in records if r["path"] == "checks.json"]
    require(len(check_rows) == 1, "Unique anchored checks.json required")
    value = read_json(directory / "checks.json", hashes, check_rows[0]["sha256"])
    validate_analysis_checks(value)  # All-positive gate BEFORE any target array hash/load.
    locations = [Path(p).resolve() for p, r in value["input_hashes"].items()
        if Path(p).name == "result.json" and r["sha256"] == value["source_result_sha256"]]
    require(len(locations) == 1 and locations[0].is_relative_to(ROOT / "reports"), "Ambiguous actual V6b global result")
    result_path = locations[0]
    # This genuine new V6b gate verifies A/B/global parent proof, actual argv,
    # full healthy RHF lineage and original source/G before stage payload hashes.
    gate = analyzer.qualify(result_path, value["source_result_sha256"], control=False, hashes=hashes)
    result = gate["result"]; cm, thz = validate_positive_frequencies(result)
    require(Path(result["resource_supervision"]["proof"]["ledger_path"]).resolve() == Path(terminal["path"])
            and gate["closure"] == value["producer_closure"] and gate["partial_closure"] == value["partial_producer"]
            and gate["H1_closure"] == value["H1_producer"] and gate["RHF_producers"] == value["healthy_RHF_unit_producers"]
            and gate["source"]["gradient_gate"] == value["restored_gradient_gate"]
            and gate["source"]["source_closure"] == value["source_SCF_closed_producer"]
            and result["snapshot_manifest_sha256"] == SNAPSHOT_SHA,
            "Actual A/B/global/source/G closure differs from independent modes export")
    for row in records:
        path = checked(directory / row["path"], hashes, row["sha256"])
        require(path.stat().st_size == row["bytes"], "Export size changed")
    return {"analysis": value, "index_sha": index_sha, "global_gate": gate, "result": result,
        "result_path": result_path, "cm": cm, "thz": thz, "analyzer": analyzer, "terminal_quantum": terminal,
        "qualified_before_numerical_import": True}


def numerical_runtime(gate):
    require(gate.get("qualified_before_numerical_import") is True, "No numerical import without completed positive V6b evidence")
    np, a2 = gate["analyzer"].numpy_runtime(gate["global_gate"])
    old = load_frozen(OLD_GROUPS, OLD_GROUPS_SHA, "frozen_old_group_numeric_helper_only_not_V6b_gate")
    require(old.MASS == MASS and old.FORMULA == FORMULA and old.SNAPSHOT_SHA == SNAPSHOT_SHA
            and old.GRADIENT_SHA == GRADIENT_SHA and old.STATIC_INDEX_SHA == STATIC_INDEX_SHA,
            "Proven group math/mass/index convention changed")
    return np, a2, old


def scalar_participation(mode, masses, groups):
    """Independent stdlib scalar formula; fixtures are not molecular results."""
    require(len(mode) == len(masses) and mode and all(len(v) == 3 for v in mode), "Aligned mode/masses required")
    require(all(type(v) in (int, float) and math.isfinite(v) for row in mode for v in row)
            and all(type(m) in (int, float) and math.isfinite(m) and m > 0 for m in masses), "Finite real displacements/masses required")
    require(groups and all(isinstance(g.get("indices_zero_based"), list) and g["indices_zero_based"]
        and len(g["indices_zero_based"]) == len(set(g["indices_zero_based"]))
        and all(type(i) is int and 0 <= i < len(mode) for i in g["indices_zero_based"]) for g in groups),
        "Explicit in-range unique group atom indices required")
    scale = max(abs(v) for row in mode for v in row); require(scale > 0, "Zero mode")
    weights = [m * math.fsum((v / scale)**2 for v in row) for m, row in zip(masses, mode)]
    total = math.fsum(weights); require(total > 0 and math.isfinite(total), "Invalid mode denominator")
    return [math.fsum(weights[i] for i in g["indices_zero_based"]) / total for g in groups]


def save_json(path, value):
    with Path(path).open("x") as handle: json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def seal(output):
    files = [{"path": str(p.relative_to(output)), "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output / "artifact_index.json", {"participation_source_sha256": sha(__file__),
        "actual_modes_analyzer_source_sha256": ANALYZER_SHA, "numeric_group_helper_sha256": OLD_GROUPS_SHA, "files": files})


def heatmap(output, shares, low_indices, groups, cm, thz, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, max(5, .35 * len(low_indices) + 4)), constrained_layout=True)
    for axis, family, title in zip(axes, (PARTITION, OVERLAP),
        ("ONE complete atom partition: four disjoint groups (sum=1)", "18 overlapping graph groups (do not add to one)")):
        selected = [j for j, g in enumerate(groups) if g["family"] == family]
        if low_indices:
            matrix = shares[np.ix_(low_indices, selected)]
            picture = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            axis.set_xticks(range(len(selected)), [groups[j]["group_id"] for j in selected], rotation=70, ha="right", fontsize=7)
            axis.set_yticks(range(len(low_indices)), [f"{k+1}: {thz[k]:.4f} THz ({cm[k]:.2f} cm^-1)" for k in low_indices], fontsize=7)
            fig.colorbar(picture, ax=axis, label="Dimensionless m*d^2 share (not IR or energy)")
        else:
            axis.text(.5, .5, "No actual modes in fixed 0.1–1.5 THz window", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(title, fontsize=10)
    fig.suptitle("C001 isolated-molecule displacement shares; no chemical peak assignment", fontsize=11)
    for extension in ("png", "pdf"): fig.savefig(output / ("low_mode_group_participation_not_IR." + extension), dpi=180)
    plt.close(fig)


def _analyze_closed(directory, index_sha, ledger_path, ledger_sha, output):
    output = Path(output).resolve(); require(not output.exists(), "Preserve previous outputs; fresh directory required")
    hashes = {}; static = static_package(hashes)
    gate = qualify_analysis(directory, index_sha, ledger_path, ledger_sha, hashes)
    np, a2, helper = numerical_runtime(gate)
    analyzer, value, result, location = gate["analyzer"], gate["analysis"], gate["result"], gate["result_path"].parent
    groups, symbols = static["groups"], static["symbols"]
    require(symbols == gate["global_gate"]["source"]["symbols"], "Static and actual C001 atom symbols/order differ")
    arrays = {}
    shapes = {"normal_modes.npy": (219, 75, 3), "normal_mode_geometry_angstrom.npy": (75, 3),
        "normal_mode_geometry_bohr.npy": (75, 3), "hessian_hartree_per_bohr2.npy": (75, 75, 3, 3),
        "hessian_used_for_analysis.npy": (75, 75, 3, 3)}
    for name, shape in shapes.items():
        path = location / name; row = value["input_hashes"].get(str(path))
        require(row is not None, "Actual mode/Hessian/geometry missing from anchored independent export")
        arrays[name] = analyzer.array(path, shape, hashes, np, row["sha256"])
    modes = arrays["normal_modes.npy"]; geometry = arrays["normal_mode_geometry_angstrom.npy"]
    lines = checked(PREPARED_XYZ, hashes, PREPARED_XYZ_SHA).read_text().splitlines()
    xyz_rows = [line.split() for line in lines[2:] if line.strip()]
    require(int(lines[0]) == len(xyz_rows) == 75 and [r[0] for r in xyz_rows] == symbols
            and np.allclose(np.asarray([r[1:] for r in xyz_rows], dtype=float), geometry, atol=5e-12, rtol=0),
            "Actual geometry/order differs from original trusted C001 prepared source")
    cm, thz = np.asarray(gate["cm"], dtype=float), np.asarray(gate["thz"], dtype=float)
    replay = analyzer.independent_subspace_audit(arrays["hessian_hartree_per_bohr2.npy"], arrays["hessian_used_for_analysis.npy"],
        modes, cm, arrays["normal_mode_geometry_bohr.npy"], symbols, np, a2)
    masses = np.asarray([MASS[s] for s in symbols], dtype=float)
    norms = np.sum(modes**2 * masses[None, :, None], axis=(1, 2))
    require(np.allclose(norms, 1, atol=1e-8, rtol=0), "Actual Cartesian normal-mode mass normalization differs")
    shares, partition_sum = helper.participation(modes, masses, groups)
    independent = np.asarray([scalar_participation(mode, masses.tolist(), groups) for mode in modes.tolist()], dtype=float)
    scalar_error = float(np.max(np.abs(shares - independent)))
    require(scalar_error <= 1e-12, "Proven array helper and independent scalar m*d^2 formula disagree")
    low_indices = [k for k, t in enumerate(gate["thz"]) if .1 <= t <= 1.5]
    low_set = set(low_indices)
    rows = [{"mode": k + 1, "frequency_cm1": float(cm[k]), "frequency_THz": float(thz[k]),
        "within_0p1_to_1p5_THz": k in low_set, "group_id": g["group_id"], "family": g["family"],
        "atom_count": len(g["indices_zero_based"]), "mass_weighted_squared_cartesian_mode_share": float(shares[k, j]),
        "IR_intensity_status": "NOT_COMPUTED", "chemical_peak_assignment": "NOT_ESTABLISHED"}
        for k in range(219) for j, g in enumerate(groups)]
    partitions = [(j, g["group_id"]) for j, g in enumerate(groups) if g["family"] == PARTITION]
    wide = [{"mode": k + 1, "frequency_cm1": float(cm[k]), "frequency_THz": float(thz[k]),
        **{name: float(shares[k, j]) for j, name in partitions}, "one_complete_partition_four_group_sum": float(partition_sum[k])}
        for k in range(219)]
    for path, record in hashes.items(): require(sha(path) == record["sha256"], "Closed input changed during group analysis")
    output.mkdir(parents=True)
    helper.csvwrite(output / "mode_group_participation.csv", rows, list(rows[0]))
    helper.csvwrite(output / "partition_participation.csv", wide, list(wide[0]))
    helper.csvwrite(output / "low_frequency_mode_group_participation.csv", [r for r in rows if r["within_0p1_to_1p5_THz"]], list(rows[0]))
    heatmap(output, shares, low_indices, groups, gate["cm"], gate["thz"], np)
    require(not any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules), "Never import a quantum engine")
    result_checks = {"schema": "research4-Spinosin-C001-staged-V6b-group-participation-v1", "status": "DONE_DESCRIPTIVE_MODE_SHARES",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "participation_source_sha256": sha(__file__),
        "actual_modes_analyzer_source_sha256": ANALYZER_SHA, "numeric_group_helper_sha256": OLD_GROUPS_SHA,
        "old_V5_group_gate_modified_or_claimed_V6b_compatible": False, "analysis_artifact_index_sha256": index_sha,
        "source_result_sha256": value["source_result_sha256"], "snapshot_manifest_sha256": SNAPSHOT_SHA,
        "fresh_gradient_result_sha256": GRADIENT_SHA, "terminal_quantum": gate["terminal_quantum"],
        "global_producer": gate["global_gate"]["closure"], "partial_producer": gate["global_gate"]["partial_closure"],
        "H1_producer": gate["global_gate"]["H1_closure"], "static_package_index_sha256": STATIC_INDEX_SHA,
        "original_ordered_graph_status": static["source_graph_verification"], "formula": FORMULA,
        "normal_modes_are_Cartesian_not_already_mass_weighted": True, "mass_convention_amu": MASS,
        "atoms": 75, "modes": 219, "groups": 22, "complete_atom_partitions": 1,
        "disjoint_groups_in_complete_partition": 4, "overlapping_graph_subgroups": 18,
        "rows": len(rows), "low_frequency_modes": len(low_indices), "low_group_rows": len(low_indices) * 22,
        "maximum_disjoint_group_sum_error": float(np.max(np.abs(partition_sum - 1))),
        "maximum_mass_normalization_error": float(np.max(np.abs(norms - 1))),
        "independent_scalar_formula_max_difference": scalar_error, "full_TR_mode_replay": replay,
        "qualified_for_descriptive_mode_interpretation": True, "IR_intensities_status": "NOT_COMPUTED",
        "chemical_attribution_verified": False, "theory_or_shares_enter_classifier": False,
        "new_quantum_evaluations": 0, "new_XYZ_animations_created": 0,
        "animations_remain_the_actual_independent_analyzer_outputs": True,
        "limitations": LIMITATIONS, "input_hashes": hashes,
        "peak_RSS_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save_json(output / "checks.json", result_checks)
    (output / "README.md").write_text(LIMITATIONS + "\nAll219 actual positive modes retained;4818 group rows. "
        "Only four disjoint group shares form one sum-to-one partition. The18 overlapping labels are separate descriptors. "
        "Fixed0.1-1.5THz view; no threshold-based peak matching or chemical assignments; no new animations.\n")
    seal(output); return result_checks


def analyze(directory, index_sha, ledger_path, ledger_sha, output):
    analyzer = modes_analyzer()
    with analyzer.exclusive_closed_quantum():
        return _analyze_closed(directory, index_sha, ledger_path, ledger_sha, output)


def preflight(output):
    output = Path(output).resolve(); require(not output.exists(), "Preserve earlier preflight")
    hashes = {}; static = static_package(hashes)
    checked(ANALYZER, hashes, ANALYZER_SHA); checked(OLD_GROUPS, hashes, OLD_GROUPS_SHA)
    require(not any(k.split(".")[0] in {"numpy", "scipy", "torch", "pyscf", "rdkit"} for k in sys.modules), "Preflight must remain stdlib only")
    output.mkdir(parents=True)
    value = {"status": "NOT_RUN", "group_adapter_source_sha256": sha(__file__), "actual_modes_analyzer_source_sha256": ANALYZER_SHA,
        "numeric_group_helper_sha256": OLD_GROUPS_SHA, "static_package_index_sha256": STATIC_INDEX_SHA,
        "metadata_only_static_groups_checked": len(static["groups"]), "complete_atom_partitions": 1,
        "disjoint_groups_in_complete_partition": 4, "overlapping_graph_subgroups": 18,
        "real_quantum_ledger_or_current_stage_result_read": False, "target_arrays_or_frequencies_read": False,
        "target_participation_values_computed": False, "old_NumPy_participation_helper_rerun": False,
        "actual_Spinosin_participation": "NOT_RUN_REQUIRES_TRUE_CLOSED_219_POSITIVE_MODES",
        "actual_budget_registered": False, "old_V5_group_gate_changed": False,
        "IR_intensities_status": "NOT_COMPUTED", "new_quantum_evaluations": 0,
        "formula": FORMULA, "limitations": LIMITATIONS, "static_input_hashes": hashes}
    save_json(output / "preflight.json", value)
    (output / "README.md").write_text("NOT_RUN. Only frozen small source/static graph and atom-map files verified. "
        "No real Q ledger, target mode, geometry array, Hessian, frequency or group fraction read/calculated.\n" + LIMITATIONS + "\n")
    seal(output); return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path); parser.add_argument("--analysis-index-sha256")
    parser.add_argument("--quantum-ledger", type=Path); parser.add_argument("--quantum-ledger-sha256")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cpu-seconds", type=int, default=180); parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.preflight:
        require(args.analysis is None and args.analysis_index_sha256 is None and args.quantum_ledger is None
                and args.quantum_ledger_sha256 is None, "Preflight never consumes actual mode/Q results")
        value = preflight(args.output)
    else:
        require(args.analysis is not None and args.analysis_index_sha256 and args.quantum_ledger is not None
                and args.quantum_ledger_sha256 and 1 <= args.cpu_seconds <= 300 and 1 <= args.wall_seconds <= 900,
                "Explicit actual index/terminal-ledger SHAs and bounded postprocessing limits required")
        if hasattr(os, "sched_getaffinity"): os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds + 5))
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[key] = "1"
        def timeout(signum, frame): raise TimeoutError("Finite group postprocessing resource limit")
        signal.signal(signal.SIGALRM, timeout); signal.alarm(args.wall_seconds); signal.signal(signal.SIGXCPU, timeout)
        value = analyze(args.analysis, args.analysis_index_sha256, args.quantum_ledger, args.quantum_ledger_sha256, args.output)
        signal.alarm(0)
    print(json.dumps({k: value.get(k) for k in ("status", "modes", "groups", "IR_intensities_status")}))


if __name__ == "__main__": main()
