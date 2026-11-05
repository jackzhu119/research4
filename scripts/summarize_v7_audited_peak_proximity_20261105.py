"""Optional stdlib table from a real audited V7 spectrum; default is NOT_RUN.

This computes positional distances only, never IR intensity or chemical
assignment. Existing frozen analyzer owns all frequency/PNG/PDF/XYZ exports.
No NPY, Hessian, wavefunction, quantum engine or classifier is loaded here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v7_v2.py"
ANALYZER_SHA = "76e2eddebf681f9adf2ab7c5f9fa0bdc126b0c93f081cc93a606009f31c94705"
WORKER_SHA = "6e9b835f4b49329cd52162b5862da14b3dcd87c80390c1e09b3cf62e14c07dbe"
PEAKS = ROOT / "reports/upgrade-20261031-v1/source-peaks/experimental_candidates_by_fold.csv"
PEAK_SHA = "b3a5f59157f42859f68edf0a8760d9f811f83ea7947c978cc0a9cbc98a4ba8b4"
CM1_TO_THZ = .0299792458
TIE_ATOL_THZ = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected, maximum=2 * 1024**2):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and not any(q.is_symlink() for q in p.parents), "Missing/symlink source")
    require(p.stat().st_size <= maximum and sha(p) == expected, "Bound small source SHA/size differs")
    return p.resolve()


def boolean(value):
    require(str(value).lower() in ("true", "false"), "Explicit boolean required")
    return str(value).lower() == "true"


def finite(value):
    result = float(value); require(math.isfinite(result), "Finite actual frequency required")
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def frequency_rows(rows, checks):
    require(len(rows) == 219 and [int(r["mode"]) for r in rows] == list(range(1, 220)), "Preserve all219 original signed modes")
    result = []
    for row in rows:
        cm, thz = finite(row["frequency_cm1"]), finite(row["frequency_THz"])
        require(abs(thz - cm * CM1_TO_THZ) <= 1e-10 and boolean(row["imaginary"]) == (cm < 0)
                and boolean(row["qualified_for_mechanism"]) is checks["qualified_for_mechanism"]
                and row["case"] == "Spinosin" and row["IR_intensity_status"] == "NOT_COMPUTED",
                "Frequency unit/sign/source/IR metadata differs")
        result.append({"mode": int(row["mode"]), "frequency_cm1": cm, "frequency_THz": thz})
    negative, zero = sum(r["frequency_cm1"] < 0 for r in result), sum(r["frequency_cm1"] == 0 for r in result)
    require(checks["imaginary_frequency_count"] == negative and checks["zero_frequency_count"] == zero
            and checks["all_frequencies_positive"] is (negative == zero == 0), "Signed-mode counts disagree with real audit")
    return result


def experimental_rows(rows):
    require(len(rows) == 20, "Retain20 overlapping-fold records, not deduplicated chemical peaks")
    result = []
    for row in rows:
        thz, cm = finite(row["frequency_thz"]), finite(row["wavenumber_cm1"])
        require(.1 <= thz <= 1.5 and abs(thz - cm * CM1_TO_THZ) <= 1e-10
                and row["classification_prior_used"] == "False" and not row["chemical_assignment"],
                "Experimental peak unit/prior/assignment differs")
        result.append({"outer_fold": int(row["outer_fold"]), "candidate_number": int(row["candidate_number"]),
                       "frequency_index": int(row["frequency_index"]), "experimental_frequency_THz": thz,
                       "experimental_wavenumber_cm1": cm})
    require({(r["outer_fold"], r["candidate_number"]) for r in result} == {(i, j) for i in range(5) for j in range(1, 5)},
            "Missing or duplicate fold/candidate identifiers")
    return result


def nearest_rows(experimental, modes):
    """Keep every tied closest position; fixed interval and tolerance, no fitting."""
    low = [r for r in modes if r["frequency_cm1"] > 0 and .1 <= r["frequency_THz"] <= 1.5]
    output = []
    if not low:
        return output
    for peak in experimental:
        distance = min(abs(m["frequency_THz"] - peak["experimental_frequency_THz"]) for m in low)
        tied = [m for m in low if abs(abs(m["frequency_THz"] - peak["experimental_frequency_THz"]) - distance) <= TIE_ATOL_THZ]
        for mode in tied:
            output.append({**peak, "theoretical_mode": mode["mode"], "theoretical_frequency_THz": mode["frequency_THz"],
                "theoretical_wavenumber_cm1": mode["frequency_cm1"],
                "signed_delta_THz": mode["frequency_THz"] - peak["experimental_frequency_THz"],
                "absolute_delta_THz": abs(mode["frequency_THz"] - peak["experimental_frequency_THz"]),
                "nearest_tie_count": len(tied), "tie_atol_THz": TIE_ATOL_THZ,
                "IR_intensity_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
                "comparison": "NEAREST_POSITION_ONLY_NOT_CHEMICAL_ASSIGNMENT",
                "independent_chemical_peak": False})
    return output


@contextmanager
def closed_quantum_lock():
    p = ROOT / ".runtime/pyscf-v4-global.lock"
    require(p.is_file(), "Use the actual existing Q lock")
    with p.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(index_path, expected_sha, global_sha, output):
    output = Path(output).resolve(); require(output.is_relative_to(ROOT / "reports") and not output.exists(), "Fresh separate report output required")
    with closed_quantum_lock():
        checked(ANALYZER, ANALYZER_SHA, 128 * 1024)
        index_path = checked(index_path, expected_sha)
        require(index_path.name == "artifact_index.json" and index_path.is_relative_to(ROOT / "reports"), "Actual sealed independent report index required")
        index = json.loads(index_path.read_text()); directory = index_path.parent
        require(index["analysis_source_sha256"] == ANALYZER_SHA and index["V7_worker_sha256"] == WORKER_SHA,
                "Require real frozen V7 independent audit source")
        files = {r["path"]: r for r in index["files"]}; require(len(files) == len(index["files"]), "Duplicate sealed audit paths")
        def source(name):
            record = files[name]; require(Path(name).name == name, "Only known small audit files")
            return checked(directory / name, record["sha256"])
        check_path = source("checks.json"); checks = json.loads(check_path.read_text())
        require(checks["schema"] == "research4-V7-independent-staged-modes-analysis-v1"
                and checks["status"] == "DONE_COMPUTED_MODES_AUDITED" and checks["case"] == "Spinosin"
                and checks["atoms"] == 75 and checks["normal_modes"] == 219 and checks["normal_mode_count_complete"] is True
                and checks["source_result_sha256"] == global_sha and checks["analysis_source_sha256"] == ANALYZER_SHA
                and checks["actual_V7_worker_sha256"] == WORKER_SHA and checks["IR_intensities_status"] == "NOT_COMPUTED"
                and checks["chemical_attribution_verified"] is False, "Only a complete actual Spinosin independent audit is eligible")
        frequencies_path = source("all_frequencies_cm1_THz.csv")
        modes = frequency_rows(read_csv(frequencies_path), checks)
        qualified = checks["qualified_for_mechanism"] is True and checks["all_frequencies_positive"] is True
        rows = []
        if qualified:
            require(checks["mechanism_pairing_status"] == "ALL_PAIRS_DESCRIPTIVE_ONLY"
                    and checks["experimental_fold_records"] == 20
                    and checks["input_hashes"][str(PEAKS)]["sha256"] == PEAK_SHA,
                    "Experimental source must already be part of real eligible audit")
            peaks_path = checked(PEAKS, PEAK_SHA); rows = nearest_rows(experimental_rows(read_csv(peaks_path)), modes)
        # Recheck all sources actually read; do not re-open raw modes/Hessian.
        checked(index_path, expected_sha); source("checks.json"); source("all_frequencies_cm1_THz.csv")
        if qualified:
            checked(PEAKS, PEAK_SHA)
        output.mkdir(parents=True)
        if rows:
            with (output / "nearest_position_pairs_not_assignments.csv").open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        summary = {"status": "DONE_DESCRIPTIVE_NEAREST_POSITIONS" if rows else
                    ("DONE_NO_POSITIVE_MODE_IN_FIXED_INTERVAL" if qualified else "BLOCKED_NONPOSITIVE_OR_UNQUALIFIED_MINIMUM"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_analysis_index_sha256": expected_sha,
            "source_global_result_sha256": global_sha, "source_analyzer_sha256": ANALYZER_SHA,
            "source_helper_sha256": sha(__file__), "all219_signed_modes_retained_in_source_csv": True,
            "negative_modes": checks["imaginary_frequency_count"], "zero_modes": checks["zero_frequency_count"],
            "comparison_rows": len(rows), "fixed_positive_frequency_range_THz": [.1, 1.5], "all_nearest_ties_retained": True,
            "experimental_records_not_independent_chemical_peaks": 20 if qualified else "NOT_READ",
            "IR_intensities_status": "NOT_COMPUTED", "chemical_attribution_verified": False,
            "classification_modified": False, "quantum_or_array_decode": False,
            "new_PNG_PDF_XYZ_rendered": False,
            "existing_PNG_PDF_XYZ_exporter": "FROZEN_V7_V2_ANALYZER_NOT_REIMPLEMENTED"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        index = {"source_helper_sha256": sha(__file__), "source_audit_index_sha256": expected_sha,
            "files": [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]}
        (output / "artifact_index.json").write_text(json.dumps(index, indent=2) + "\n")
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--execute", action="store_true")
    p.add_argument("--audit-index", type=Path); p.add_argument("--audit-index-sha256"); p.add_argument("--global-result-sha256")
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if not a.execute:
        require(not a.audit_index and not a.audit_index_sha256 and not a.global_result_sha256, "Preparation consumes no actual data")
        require(not a.output.exists(), "Fresh NOT_RUN preparation only"); a.output.mkdir(parents=True)
        result = {"status": "NOT_RUN_PREPARATION_ONLY", "source_sha256": sha(__file__), "numeric_imports": False,
                  "scientific_payloads_read": False, "future_comparison_rows_generated": 0, "IR_status": "NOT_COMPUTED"}
        (a.output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        require(a.audit_index and a.audit_index_sha256 and a.global_result_sha256, "Explicit real audit index/global SHA required")
        result = execute(a.audit_index, a.audit_index_sha256, a.global_result_sha256, a.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
