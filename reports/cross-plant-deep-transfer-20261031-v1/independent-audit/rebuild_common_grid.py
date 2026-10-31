"""Independently derive fixed engineering alpha grid; no learned operations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes()).hexdigest()


def reconstruct():
    grid = np.linspace(.21, 1.49, 128, dtype=np.float64)
    paths = {
        "zaoren": "data/multiplant_database/zaoren/aligned_modalities.npz",
        "huangqi": "data/multiplant_database/huangqi/alpha.npz",
        "tianma": "data/multiplant_database/tianma/alpha.npz",
        "ginseng": "data/external/public_thz_20261031_v1/ginseng/observations.npz",
    }
    result = {}
    for name, relative in paths.items():
        with np.load(ROOT / relative, allow_pickle=False) as data:
            x = data["X_alpha" if name in {"zaoren", "ginseng"} else "X"]
            axis = data["axis_alpha" if name == "zaoren" else "axis"]
            assert grid[0] >= axis[0] and grid[-1] <= axis[-1]
            aligned = np.vstack([np.interp(grid, axis, row) for row in x])
            sigma = aligned.std(axis=1, ddof=1, keepdims=True)
            assert np.isfinite(aligned).all() and np.all(sigma > np.finfo(float).eps)
            snv = (aligned - aligned.mean(axis=1, keepdims=True)) / sigma
            assert np.allclose(snv.mean(axis=1), 0, atol=1e-12)
            assert np.allclose(snv.std(axis=1, ddof=1), 1, atol=1e-12)
            result[name] = {
                "raw_common": aligned, "snv_common": snv,
                "record_ids": data["record_ids"].astype(str),
                "source_labels": data["y" if name in {"zaoren", "ginseng"} else "filename_groups"].astype(str),
                "original_sha256": sha(ROOT / relative),
            }
    return grid, result


def main():
    grid, arrays = reconstruct()
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "new_training_operations": 0,
        "operation": "np.interp per observation to fixed128 [.21,1.49] THz; per-row SNV ddof1",
        "axis": grid.tolist(), "axis_sha256": array_sha(grid),
        "all_points_within_each_native_axis": True,
        "scientific_status": "EXPLORATORY_ENGINEERED_SHAPE_TRANSFER_UNVERIFIED_ABSORPTION_UNITS",
        "physical_calibration_certified": False,
        "interpolation_is_new_independent_measurement": False,
        "datasets": {
            name: {
                "records": len(data["record_ids"]),
                "original_sha256": data["original_sha256"],
                "raw_common_sha256": array_sha(data["raw_common"]),
                "snv_common_sha256": array_sha(data["snv_common"]),
                "raw_negative_cells": int((data["raw_common"] < 0).sum()),
                "minimum_snv_row_sd": float(data["snv_common"].std(axis=1, ddof=1).min()),
                "maximum_snv_row_mean_abs": float(np.abs(data["snv_common"].mean(axis=1)).max()),
            } for name, data in arrays.items()
        },
    }
    output = HERE / "common_grid_rebuild.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output),
                      "numeric_values_reconstructed": int(sum(d["raw_common"].size for d in arrays.values())),
                      "new_training_operations": 0}))


if __name__ == "__main__":
    main()
