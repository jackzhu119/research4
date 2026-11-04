"""Describe atom-group shares of already closed, qualified Spinosin modes.

No quantum engine, training, chemical assignment or IR intensity. Frozen C001
indices are essential: the newly uploaded SDF has a different atom ordering.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_spinosin_v5_modes_20261104.py"
ANALYZER_SHA = "a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154"
CONSUMER = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
CONSUMER_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
GRADIENT_SHA = "6b143defc238471bc815857df08fc4034a9d8454a00567f3b084907a54a5c403"
STATIC = ROOT / "reports/external-evidence-20261104-v1/mode-interpretation-preparation"
STATIC_INDEX_SHA = "5ecf8bbe7c5317d5ef0d18070f01307542c366f0700bcbaf43e5e7849a5aa246"
STATIC_HASHES = {"atom_groups_C001.json": "12943fde895ccb715575434d633fa6bc5d295fdcbc50f111edc278592326a516",
                 "C001_atom_map.csv": "3888e50f50509b5e453faf1eb2be21e072b1a20964c0a2c9f0199060e53b14c8",
                 "atom_order_audit.json": "3f5d9011d86c36ae8290bf0f1a0ce4e2de06b37101542de27ebb7938c37b21f2"}
PREPARED_XYZ = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/optimized.xyz"
PREPARED_XYZ_SHA = "f7c69d098159254cf22260b3c11c6b63a865839ce942e5a7ca0982da1ec73e39"
SOURCE_SDF_SHA = "02dc923dce2f225ede414341229f3090967e569f6075165a41685b3a5d8da2f7"
THERMO = ROOT / ".runtime/quantum-upgrade-site/pyscf/hessian/thermo.py"
THERMO_SHA = "8143ebed7653bd79b4b8d7b2f5933671a33285b11aba47cf0d8134c229a031f8"
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}
CM1_TO_THZ = .0299792458
PARTITION = "DISJOINT_PARTITION"
OVERLAP = "OVERLAPPING_GRAPH_SUBGROUP"
FORMULA = "sum_group(m_i*sum_xyz(d_i**2))/sum_all(m_i*sum_xyz(d_i**2))"
LIMITATIONS = ("Mass-weighted squared Cartesian eigenvector shares, dimensionless; not energy, IR intensity, "
               "absorption or chemical peak assignment. Overlapping subgroups must not be added as a partition. "
               "Sign/global scale invariant; individual-mode shares depend on basis choice within degenerate subspaces. "
               "Shares omit displacement phase, torsion/bending identity and intermolecular effects. OH participation "
               "is not evidence of hydrogen bonding. One isolated conformer is not a solid herb spectrum or a global minimum. "
               "No frequencies or shares enter any classifier; the existing 96.0234% classification result is unchanged.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, hashes, expected):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 16*1024**2, "Missing/symlink/oversize input")
    path = path.resolve()
    require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected, "Input SHA differs: " + str(path))
    hashes[str(path)] = {"sha256": expected, "bytes": path.stat().st_size}
    return path


def read_json(path, hashes, expected):
    return json.loads(checked(path, hashes, expected).read_text())


def load_frozen(path, expected, name):
    require(sha(path) == expected, "Frozen code differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_groups(groups, n, *, expected_counts=None):
    require(isinstance(groups, list) and groups, "Missing structural groups")
    ids, covered, counts = set(), [], Counter()
    for group in groups:
        identity, family = group["group_id"], group["family"]
        indices = group["indices_zero_based"]
        require(isinstance(identity, str) and identity and identity not in ids, "Duplicate/invalid group ID")
        ids.add(identity)
        require(family in (PARTITION, OVERLAP), "Unknown group family")
        require(isinstance(indices, list) and indices and all(type(i) is int and 0 <= i < n for i in indices)
                and len(set(indices)) == len(indices), "Duplicate/out-of-range/noninteger atom index")
        require(group["indices_one_based"] == [i+1 for i in indices], "One-based atom map differs")
        counts[family] += 1
        if family == PARTITION:
            covered.extend(indices)
    require(sorted(covered) == list(range(n)), "Disjoint partitions must cover every atom exactly once")
    require(expected_counts is None or dict(counts) == expected_counts, "Unexpected structural group counts")
    return groups


def static_groups(hashes):
    index = read_json(STATIC / "files_sha256.json", hashes, STATIC_INDEX_SHA)
    for name, expected in STATIC_HASHES.items():
        records = [r for r in index["files"] if r["path"] == name]
        require(len(records) == 1 and records[0]["sha256"] == expected, "Static package index differs")
    data = read_json(STATIC / "atom_groups_C001.json", hashes, STATIC_HASHES["atom_groups_C001.json"])
    audit = read_json(STATIC / "atom_order_audit.json", hashes, STATIC_HASHES["atom_order_audit.json"])
    require(data["schema"] == "research4.C001.graph-functional-groups.v1"
            and data["current_geometry_anchor"] == str(PREPARED_XYZ.relative_to(ROOT))
            and data["partition_coverage_75_atoms_exactly_once"] is True
            and data["overlapping_subgroups_must_not_be_summed_as_disjoint"] is True,
            "Static group schema/anchor differs")
    require(audit["schema"] == "research4.static-C001-atom-order-audit.v1" and audit["atom_count"] == 75
            and audit["C001_to_current_zero_based_index_map"] == list(range(75))
            and audit["source_and_C001_indexed_graph_signatures_equal"] is True
            and audit["inputs"]["C001_source_sdf"]["sha256"] == SOURCE_SDF_SHA
            and audit["inputs"]["v5_prepared_xyz"]["sha256"] == PREPARED_XYZ_SHA,
            "Trusted C001 atom-order proof differs")
    with checked(STATIC / "C001_atom_map.csv", hashes, STATIC_HASHES["C001_atom_map.csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 75, "Atom map must contain 75 rows")
    for i, row in enumerate(rows):
        require(all(int(row[key]) == i for key in ("C001_SDF_index_zero_based", "current_PySCF_index_zero_based"))
                and all(int(row[key]) == i+1 for key in ("C001_SDF_atom_number_one_based", "current_XYZ_atom_number_one_based")),
                "Atom map is not the trusted C001 identity order")
    symbols = [row["element"] for row in rows]
    require(dict(Counter(symbols)) == {"C": 28, "H": 32, "O": 15}, "Static atom composition differs")
    groups = validate_groups(data["groups"], 75, expected_counts={PARTITION: 4, OVERLAP: 18})
    for group in groups:
        if group["family"] == PARTITION:
            require(all(rows[i]["disjoint_partition"] == group["group_id"] for i in group["indices_zero_based"]), "CSV/JSON partitions differ")
    return groups, symbols


def validate_analysis_checks(value):
    exact = {"schema": "research4-V5-independent-modes-analysis-v1", "status": "DONE_COMPUTED_MODES_AUDITED",
             "case": "Spinosin", "atoms": 75, "normal_modes": 219, "qualified_for_mechanism": True,
             "all_frequencies_positive": True, "imaginary_frequency_count": 0, "zero_frequency_count": 0,
             "Spinosin_target_status": "ACTUAL_CLOSED_FREQUENCIES_AUDITED", "analysis_source_sha256": ANALYZER_SHA,
             "consumer_sha256": CONSUMER_SHA, "snapshot_manifest_sha256": SNAPSHOT_SHA,
             "IR_intensities_status": "NOT_COMPUTED", "PySCF_imported": False, "new_quantum_evaluations": 0,
             "chemical_attribution_verified": False, "theory_or_source_peaks_enter_classifier": False,
             "source_MF_unchanged_verified": True}
    require(all(type(value.get(k)) is type(v) and value.get(k) == v for k, v in exact.items()),
            "BLOCKED: incomplete/control/nonpositive or unqualified frozen analysis; do not read molecular arrays")


def qualify_analysis(directory, index_sha, hashes):
    """All closure/gradient metadata checks precede any target NPY hash or load."""
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / "reports"), "Analysis must be a closed report directory")
    index = read_json(directory / "artifact_index.json", hashes, index_sha)
    require(index.get("analysis_source_sha256") == ANALYZER_SHA and sha(ANALYZER) == ANALYZER_SHA, "Unreviewed modes analyzer")
    records = index["files"]
    require(len({r["path"] for r in records}) == len(records), "Duplicate exported artifact path")
    for row in records:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe exported artifact path")
    checks_records = [r for r in records if r["path"] == "checks.json"]
    require(len(checks_records) == 1, "Missing unique analysis checks")
    value = read_json(directory / "checks.json", hashes, checks_records[0]["sha256"])
    validate_analysis_checks(value)
    locations = [Path(p) for p, r in value["input_hashes"].items()
                 if Path(p).name == "result.json" and r["sha256"] == value["source_result_sha256"]]
    require(len(locations) == 1 and locations[0].resolve().is_relative_to(ROOT / "reports"), "Ambiguous original result")
    result_path = locations[0].resolve()
    result = read_json(result_path, hashes, value["source_result_sha256"])
    require(result.get("schema") == "research4-resumable-full-rks-hessian-result-v1"
            and result.get("status") == "FREQUENCIES_COMPUTED" and result.get("frequencies_status") == "DONE"
            and result.get("snapshot_manifest_sha256") == SNAPSHOT_SHA and result.get("atoms") == 75
            and result.get("normal_mode_count") == result.get("expected_normal_mode_count") == 219
            and all(result.get(k) is True for k in ("complete_global_Hessian", "complete_RHF_helper", "normal_modes_complete",
                "stable_MF_state_unchanged", "no_new_SCF_performed", "all_frequencies_real_positive", "stationary_minimum_verified"))
            and result.get("IR_intensities_status") == "NOT_COMPUTED", "Incomplete original Spinosin frequency result")
    cm = np.asarray(result["frequencies_cm1"], float)
    thz = np.asarray(result["frequencies_thz"], float)
    require(cm.shape == thz.shape == (219,) and np.isfinite(cm).all() and np.all(cm > 0)
            and np.allclose(thz, cm*CM1_TO_THZ, atol=1e-12, rtol=0), "BLOCKED: nonpositive/missing actual frequencies")
    consumer = load_frozen(CONSUMER, CONSUMER_SHA, "group_participation_closed_V5_consumer")
    evidence = {"schema": consumer.EXECUTION_SCHEMA, "source_worker": result["source_worker"],
                "resource_supervision": result["resource_supervision"], "output_directory": str(result_path.parent),
                "snapshot_manifest_sha256": SNAPSHOT_SHA, "phase": result["phase"]}
    closure = consumer.verify_closed_execution(evidence, expected_snapshot_sha=SNAPSHOT_SHA, expected_output=result_path.parent)
    require(closure == value["producer_closure"] and closure["producer_result_sha256"] == value["source_result_sha256"], "Original healthy closed producer changed")
    gate = result["gradient_gate"]["result"]
    require(gate["sha256"] == GRADIENT_SHA, "Unexpected Spinosin fresh-gradient gate")
    gradient = consumer.qualify_gradient_receipt(gate["path"], GRADIENT_SHA, SNAPSHOT_SHA, result["boot_id"])
    require(gradient == result["gradient_gate"] == value["restored_gradient_gate"], "Closed same-state fresh-gradient evidence changed")
    snapshot_path = Path(result["snapshot_directory"]) / "manifest.json"
    manifest = read_json(snapshot_path, hashes, SNAPSHOT_SHA)
    producer = consumer.state_module().validate_closed_producer(manifest, snapshot_path)
    require(producer == value["source_SCF_closed_producer"], "Closed exact source SCF producer changed")
    for row in records:
        path = checked(directory / row["path"], hashes, row["sha256"])
        require(path.stat().st_size == row["bytes"], "Closed export size changed")
    require(not any(k == "pyscf" or k.startswith("pyscf.") for k in sys.modules), "Postprocessor must not import PySCF")
    return value, result_path, result, cm, thz


def participation(modes, masses, groups):
    """Generic pure formula; synthetic fixtures are not Spinosin results."""
    modes, masses = np.asarray(modes), np.asarray(masses)
    require(modes.ndim == 3 and modes.shape[2] == 3 and modes.shape[0] > 0
            and masses.shape == (modes.shape[1],) and not np.iscomplexobj(modes) and not np.iscomplexobj(masses)
            and np.issubdtype(modes.dtype, np.number) and np.issubdtype(masses.dtype, np.number), "Invalid real mode/mass shape or type")
    require(np.isfinite(modes).all() and np.isfinite(masses).all() and np.all(masses > 0), "Nonfinite modes or invalid masses")
    validate_groups(groups, len(masses))
    # Scale first to avoid overflow/underflow for otherwise valid display scaling.
    scale = np.max(np.abs(modes), axis=(1, 2))
    require(np.all(scale > 0), "Zero normal-mode vector")
    weights = np.sum((modes/scale[:, None, None])**2, axis=2)*masses[None, :]
    denominator = np.sum(weights, axis=1)
    require(np.isfinite(denominator).all() and np.all(denominator > 0), "Invalid mode normalization")
    shares = np.column_stack([np.sum(weights[:, g["indices_zero_based"]], axis=1)/denominator for g in groups])
    partition_sum = shares[:, [i for i, g in enumerate(groups) if g["family"] == PARTITION]].sum(axis=1)
    require(np.isfinite(shares).all() and np.all(shares >= 0) and np.all(shares <= 1+1e-14)
            and np.allclose(partition_sum, 1, atol=1e-12, rtol=0), "Disjoint participation shares do not sum to one")
    return shares, partition_sum


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def csvwrite(path, rows, fields):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def seal(output):
    records = [{"path": str(p.relative_to(output)), "sha256": sha(p), "bytes": p.stat().st_size}
               for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "artifact_index.json", {"participation_source_sha256": sha(__file__), "files": records})


def analyze(directory, index_sha, output):
    output = Path(output).resolve()
    require(not output.exists(), "Refuse overwrite of existing participation report")
    hashes = {}
    groups, symbols = static_groups(hashes)
    checked(THERMO, hashes, THERMO_SHA)  # Source inspection only; never import the quantum engine.
    value, result_path, result, cm, thz = qualify_analysis(directory, index_sha, hashes)
    # No target array or target coordinate read has occurred above this line.
    arrays = []
    for name, shape in (("normal_modes.npy", (219, 75, 3)), ("normal_mode_geometry_angstrom.npy", (75, 3))):
        path = result_path.parent / name
        record = value["input_hashes"].get(str(path))
        require(record is not None, "Mode/geometry absent from anchored analysis")
        array = np.load(checked(path, hashes, record["sha256"]), allow_pickle=False)
        require(array.shape == shape and array.dtype == np.float64 and np.isfinite(array).all(), "Invalid complete real float64 mode/geometry array")
        arrays.append(array)
    modes, geometry = arrays
    lines = checked(PREPARED_XYZ, hashes, PREPARED_XYZ_SHA).read_text().splitlines()
    atom_rows = [line.split() for line in lines[2:]]
    require(int(lines[0]) == len(atom_rows) == 75 and [row[0] for row in atom_rows] == symbols
            and np.allclose(np.array([row[1:] for row in atom_rows], float), geometry, atol=5e-12, rtol=0), "Normal-mode geometry does not match the trusted C001 source ordering")
    masses = np.array([MASS[s] for s in symbols])
    norms = np.sum(modes**2*masses[None, :, None], axis=(1, 2))
    require(np.allclose(norms, 1, atol=1e-8, rtol=0), "Saved PySCF Cartesian normal-mode mass normalization differs")
    shares, partition_sum = participation(modes, masses, groups)
    low = (.1 <= thz) & (thz <= 1.5)
    rows = [{"mode": k+1, "frequency_cm1": float(cm[k]), "frequency_THz": float(thz[k]),
             "within_0p1_to_1p5_THz": bool(low[k]), "group_id": group["group_id"], "family": group["family"],
             "atom_count": len(group["indices_zero_based"]), "mass_weighted_squared_cartesian_mode_share": float(shares[k, j]),
             "IR_intensity_status": "NOT_COMPUTED", "chemical_peak_assignment": "NOT_ESTABLISHED"}
            for k in range(219) for j, group in enumerate(groups)]
    partitions = [(j, group["group_id"]) for j, group in enumerate(groups) if group["family"] == PARTITION]
    wide = [{"mode": k+1, "frequency_cm1": float(cm[k]), "frequency_THz": float(thz[k]),
             **{name: float(shares[k, j]) for j, name in partitions}, "four_partition_sum": float(partition_sum[k])}
            for k in range(219)]
    for path, record in hashes.items():
        require(sha(path) == record["sha256"], "Closed input changed during analysis: " + path)
    output.mkdir(parents=True)
    csvwrite(output / "mode_group_participation.csv", rows, list(rows[0]))
    csvwrite(output / "partition_participation.csv", wide, list(wide[0]))
    csvwrite(output / "low_frequency_mode_group_participation.csv", [r for r in rows if r["within_0p1_to_1p5_THz"]], list(rows[0]))
    checks = {"schema": "research4-Spinosin-C001-group-participation-v1", "status": "DONE_DESCRIPTIVE_MODE_SHARES",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "participation_source_sha256": sha(__file__),
              "analysis_artifact_index_sha256": index_sha, "analysis_source_sha256": ANALYZER_SHA,
              "source_result_sha256": value["source_result_sha256"], "snapshot_manifest_sha256": SNAPSHOT_SHA,
              "source_producer_closure": value["producer_closure"], "static_package_index_sha256": STATIC_INDEX_SHA,
              "formula": FORMULA, "normal_mode_definition": "PySCF thermo.py: norm_mode = mass**(-0.5)*mass_weighted_orthonormal_eigenvector",
              "PySCF_thermo_source_sha256": THERMO_SHA, "atoms": 75, "modes": 219, "groups": 22,
              "disjoint_partitions": 4, "overlapping_subgroups": 18, "rows": len(rows), "low_frequency_modes": int(low.sum()),
              "maximum_four_partition_sum_error": float(np.max(np.abs(partition_sum-1))),
              "maximum_mass_normalization_error": float(np.max(np.abs(norms-1))), "qualified_for_mechanism": True,
              "IR_intensities_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
              "theory_or_shares_enter_classifier": False, "new_quantum_evaluations": 0,
              "limitations": LIMITATIONS, "input_hashes": hashes}
    write_json(output / "checks.json", checks)
    (output / "README.md").write_text(LIMITATIONS+"\nFormula: "+FORMULA+". Only the four DISJOINT_PARTITION groups sum to one; the other 18 labels overlap. All 219 actual modes are retained, with a separate 0.1–1.5 THz view. C001 zero-based indices, not IUPAC atom positions or the newly uploaded SDF indices.\n")
    seal(output)
    return checks


def preflight(output):
    output = Path(output).resolve()
    require(not output.exists(), "Refuse overwrite of existing preflight")
    hashes = {}
    groups, _ = static_groups(hashes)
    for path, expected in ((ANALYZER, ANALYZER_SHA), (CONSUMER, CONSUMER_SHA), (THERMO, THERMO_SHA)):
        checked(path, hashes, expected)
    output.mkdir(parents=True)
    result = {"status": "NOT_RUN", "target": "Spinosin C001, 75 atoms,219 modes", "formula": FORMULA,
              "participation_source_sha256": sha(__file__), "analysis_source_sha256": ANALYZER_SHA,
              "static_package_index_sha256": STATIC_INDEX_SHA, "structural_groups": len(groups),
              "target_arrays_read": False, "target_frequencies_read": False, "participation_values_computed": False,
              "qualified_for_mechanism": False, "IR_intensities_status": "NOT_COMPUTED", "new_quantum_evaluations": 0,
              "input_hashes": hashes, "limitations": LIMITATIONS}
    write_json(output / "preflight.json", result)
    (output / "README.md").write_text("NOT_RUN: only static atom maps and frozen code were checked. No target arrays, frequencies or participation values read/generated. Run explicitly after the frozen a2 analyzer yields complete healthy positive Spinosin modes. No empty results table is presented as actual target data.\n"+LIMITATIONS+"\n")
    seal(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--analysis-index-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (min(soft, 1024**3) if soft > 0 else 1024**3, min(hard, 1024**3) if hard > 0 else 1024**3))
    if args.preflight:
        require(args.analysis is None and args.analysis_index_sha256 is None, "Preflight must not consume target analysis")
        result = preflight(args.output)
    else:
        require(args.analysis is not None and args.analysis_index_sha256, "Explicit closed analysis and index SHA required")
        result = analyze(args.analysis, args.analysis_index_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("status", "qualified_for_mechanism", "IR_intensities_status")}))


if __name__ == "__main__":
    main()
