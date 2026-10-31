"""Read-only compatibility/duplicate qualification; never fits or scores a model."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
THRESHOLD = 0.99999


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def values_summary(x):
    return {
        "shape": list(x.shape), "dtype": str(x.dtype),
        "finite_cells": int(np.isfinite(x).sum()),
        "nonfinite_cells": int((~np.isfinite(x)).sum()),
        "negative_cells": int((x < 0).sum()),
        "minimum": float(np.min(x)), "maximum": float(np.max(x)),
    }


def regrid(x, axis, grid):
    if grid[0] < axis[0] or grid[-1] > axis[-1]:
        raise ValueError("No extrapolation permitted")
    return np.asarray([np.interp(grid, axis, row) for row in x])


def row_shapes(x):
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Constant spectrum cannot pass shape-correlation QC")
    return centered / norms


def main():
    paths = {
        "zaoren": "data/multiplant_database/zaoren/aligned_modalities.npz",
        "huangqi": "data/multiplant_database/huangqi/alpha.npz",
        "tianma": "data/multiplant_database/tianma/alpha.npz",
        "ginseng": "data/external/public_thz_20261031_v1/ginseng/observations.npz",
    }
    quantities = {}
    native = {}
    for dataset, rel in paths.items():
        with np.load(ROOT / rel, allow_pickle=False) as pack:
            if dataset == "zaoren":
                x, axis, labels = pack["X_alpha"], pack["axis_alpha"], pack["y"]
                group = labels
            elif dataset == "ginseng":
                x, axis, labels = pack["X_alpha"], pack["axis"], pack["y"]
                group = pack["export_groups"]
            else:
                x, axis, labels = pack["X"], pack["axis"], pack["labels"]
                group = pack["filename_groups"]
            ids = pack["record_ids"]
            assert len(set(ids)) == len(ids) == len(x) == len(labels) == len(group)
            assert x.shape[1] == len(axis) and np.all(np.isfinite(x))
            assert np.all(np.isfinite(axis)) and np.all(np.diff(axis) > 0)
            native[dataset] = (x.copy(), axis.copy(), ids.copy())
            quantities[dataset] = {
                "source": rel, "source_sha256": sha(ROOT / rel),
                "records": len(ids), "unique_record_ids": len(set(ids)),
                "alpha": values_summary(x),
                "axis_range": [float(axis[0]), float(axis[-1])],
                "axis_points": len(axis), "axis_unit_basis": "THz explicitly in original table heading / public metadata",
                "label_counts_as_stored": dict(sorted(Counter(labels.tolist()).items())),
                "source_group_counts": dict(sorted(Counter(group.tolist()).items())),
                "alpha_length_unit": "cm^-1, official source metadata" if dataset == "ginseng" else "UNKNOWN",
                "verified_independent_batch_count": None,
                "physical_independence_basis": "UNKNOWN" if dataset == "ginseng" else "USER_DECLARATION_ONLY",
            }
    assert {k: v[0].shape[0] for k, v in native.items()} == {"zaoren": 855, "huangqi": 403, "tianma": 405, "ginseng": 42}

    target_axis = native["zaoren"][1]
    teacher_axes_match = all(np.array_equal(native[key][1][:len(target_axis)], target_axis) for key in ("huangqi", "tianma"))
    assert teacher_axes_match
    ginseng_axis = native["ginseng"][1]
    grid = ginseng_axis[(ginseng_axis >= target_axis[0]) & (ginseng_axis <= target_axis[-1])]
    assert len(grid) == 60
    fixed128 = np.linspace(0.21, 1.49, 128)
    assert all(axis[0] <= fixed128[0] and axis[-1] >= fixed128[-1] for _, axis, _ in native.values())

    common = {name: regrid(x, axis, grid) for name, (x, axis, _) in native.items()}
    shapes = {name: row_shapes(x) for name, x in common.items()}
    cross = []
    for left, right in itertools.combinations(native, 2):
        a, b = common[left], common[right]
        a_hashes = [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in a]
        b_hashes = [hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in b]
        exact = sum(ha == hb for ha in a_hashes for hb in b_hashes)
        corr = shapes[left] @ shapes[right].T
        at = np.unravel_index(int(np.argmax(corr)), corr.shape)
        cross.append({
            "datasets": [left, right], "comparisons": int(corr.size),
            "common_grid_exact_row_matches": exact,
            "positive_centered_cosine_at_least_threshold": int((corr >= THRESHOLD).sum()),
            "maximum_positive_centered_cosine": float(corr[at]),
            "maximum_pair_record_ids": [str(native[left][2][at[0]]), str(native[right][2][at[1]])],
        })
        quantities[left]["alpha_on_common60"] = values_summary(a)
        quantities[right]["alpha_on_common60"] = values_summary(b)

    n_inventory = []
    for dataset in ("zaoren", "huangqi", "tianma", "ginseng"):
        p = ROOT / (paths[dataset] if dataset in ("zaoren", "ginseng") else f"data/multiplant_database/{dataset}/n.npz")
        with np.load(p, allow_pickle=False) as pack:
            x = pack["X_n"] if dataset in ("zaoren", "ginseng") else pack["X"]
            axis = pack["axis_n"] if dataset == "zaoren" else pack["axis"]
            ids = pack["record_ids"]
            alpha_ids = native[dataset][2]
            assert len(set(ids)) == len(ids) and set(ids).issubset(set(alpha_ids))
            n_inventory.append({
                "dataset": dataset, "source": str(p.relative_to(ROOT)), "sha256": sha(p),
                "records": len(ids), "missing_alpha_ids": sorted(set(alpha_ids) - set(ids)),
                "quantity": "refractive index; dimensionless by definition; public metadata RIU for ginseng",
                "instrument_and_inversion_equivalence_verified": False,
                "values_native": values_summary(x),
                "values_common60": values_summary(regrid(x, axis, grid)),
            })

    # Verify the actual workbook versions behind the frozen array database.
    workbook_manifest = ROOT / "data/multiplant_database/data_quality/source_workbooks.json"
    original_sources = []
    for source in json.loads(workbook_manifest.read_text()):
        if source["dataset"] not in native or source["dataset"] == "ginseng":
            continue
        p = ROOT / source["source_file"]
        actual = sha(p)
        assert actual == source["sha256"] == source["expected_sha256"]
        original_sources.append({
            "dataset": source["dataset"], "source_id": source["source_id"],
            "source_file": source["source_file"], "bytes": p.stat().st_size,
            "sha256": actual,
            "table_axis_metadata": [
                {"modality": s["modality"], "status": s["status"],
                 "axis_header": s.get("profile", {}).get("axis_header"),
                 "axis_unit": s.get("profile", {}).get("axis_unit")}
                for s in source["sheets"] if s["modality"] in ("alpha", "n")
            ],
        })
    assert len(original_sources) == 15

    overlay_path = ROOT / "data/multiplant_database/metadata/user_declaration_overlays/20261031_huangqi_tianma_v1/overlay_manifest.json"
    overlay = json.loads(overlay_path.read_text())
    assert overlay["cross_plant_transfer_status"] == "BLOCKED"
    proposals = {
        "teacher_native_612": {"points": 612, "minimum_thz": float(target_axis[0]), "maximum_thz": float(target_axis[-1]),
                               "teacher_axes_exact_first612_match": teacher_axes_match, "source_pool": "HQ403 + TM405 =808 observations", "ginseng_excluded": "insufficient coverage below0.2THz"},
        "public_native_overlap60": {"points": len(grid), "axis": grid.tolist(), "rule": "Keep original ginseng frequency coordinates inside current target axis endpoints; interpolate target/teacher linearly, ginseng direct subset; no extrapolation", "minimum_thz": float(grid[0]), "maximum_thz": float(grid[-1])},
        "fixed_uniform128": {"points": 128, "minimum_thz": 0.21, "maximum_thz": 1.49, "rule": "np.linspace(0.21,1.49,128), pointwise linear interpolation only; no data-driven choice or extrapolation", "caveat": "Interpolating65-point public data does not create128 independent physical frequency measurements."},
        "chosen_training_grid": "ROOT_PROTOCOL_TO_FIX; this qualification audit does not choose based on labels or predictions",
    }
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "cross-plant-deep-transfer-20261031-v1-data-qualification",
        "code_sha256": sha(__file__), "model_fit_operations": 0, "prediction_or_scoring_operations": 0,
        "all_native_input_hashes_unchanged": all(sha(ROOT / quantities[k]["source"]) == quantities[k]["source_sha256"] for k in quantities),
        "scientific_independence_or_calibration_certified": False,
        "overall_status": "EXPLORATORY_ENGINEERING_TRANSFER_CANDIDATE_ONLY_NOT_PHYSICAL_COMPATIBILITY_PASS",
        "alpha_source_observations": 850, "teacher_only_alpha_source_observations": 808,
        "alpha_source_and_target_observations_not_pooled_as_ZY_classes": True,
        "quantities": quantities, "n_inventory": n_inventory,
        "cross_dataset_common60_quality_control": {"grid": "public_native_overlap60", "threshold": THRESHOLD, "comparisons": cross,
            "total_comparisons": sum(x["comparisons"] for x in cross), "records_removed_or_relabelled": 0,
            "not_independence_proof": True, "not_used_to_choose_features_or_model": True},
        "preexisting_readiness_snapshot": {
            "overlay_manifest": str(overlay_path.relative_to(ROOT)), "sha256": sha(overlay_path),
            "cross_plant_transfer_status": overlay["cross_plant_transfer_status"],
            "reason": "Units/instrument compatibility unknown; current experiment must retain this snapshot and explicitly introduce exploratory engineering qualification in a new protocol."},
        "original_workbook_sources_verified": original_sources,
        "source_head_rules": {
            "zaoren": "ZY01–ZY07 only current target labels",
            "huangqi": "HQ01–HQ04 stored filename_groups only; scientific labels array remains blank, no named-origin verification",
            "tianma": "TM01–TM04 stored filename_groups only; scientific labels array remains blank, no named-origin verification",
            "ginseng": "0American/1Asian species labels in independent source head, not ZY origins",
        },
        "limitations": [
            "Matching frequency coordinates does not prove matching instruments, inversion, thickness, matrix or calibration.",
            "Teacher alpha length units are unknown; no unit conversion factor invented. Row SNV is fixed within each spectrum and does not establish physical equivalence.",
            "Refractive index is dimensionless but acquisition/inversion equivalence and valid physical range are not verified; negative/nonphysical values retained as quality flags.",
            "HQ n is missing100HQ03 observations; do not fill them or silently compare a303-row source pool to403 as if identical.",
            "Ginseng roots and biological batches unknown;6export groups are not proven batches.",
            "All855ZY remain historical exposed development samples; pretraining must never use current target inner validation or outer query, including unlabeled copies.",
            "Common-grid quality checks use existing arrays solely to inspect finite values and duplicate risk; they do not fit any transform, classifier or transfer model.",
            "Fruit unknown physical sample-index/time/frequency relation and leaf unknown independent units block their use as medicinal frequency-domain transfer examples.",
        ],
    }
    write("common_axis_proposals.json", proposals)
    write("audit.json", report)
    with (OUT / "source_inputs.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "source_id", "source_file", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in original_sources)
    print(json.dumps({"status": report["overall_status"], "alpha_source_observations": 850,
                      "cross_dataset_comparisons": sum(x["comparisons"] for x in cross),
                      "exact_matches": sum(x["common_grid_exact_row_matches"] for x in cross),
                      "high_shape_matches": sum(x["positive_centered_cosine_at_least_threshold"] for x in cross),
                      "fit_operations": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
