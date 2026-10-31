"""Read-only canonical arrays and historical-fold audit; no model fitting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_sha(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    paths = {
        "zaoren": "data/multiplant_database/zaoren/aligned_modalities.npz",
        "huangqi": "data/multiplant_database/huangqi/alpha.npz",
        "tianma": "data/multiplant_database/tianma/alpha.npz",
        "ginseng": "data/external/public_thz_20261031_v1/ginseng/observations.npz",
    }
    rows = {}
    zaoren_labels = {}
    for dataset, relative in paths.items():
        with np.load(ROOT / relative, allow_pickle=False) as arrays:
            xkey = "X_alpha" if dataset in {"zaoren", "ginseng"} else "X"
            axiskey = "axis_alpha" if dataset == "zaoren" else "axis"
            labelkey = "y" if dataset in {"zaoren", "ginseng"} else "filename_groups"
            values, axis = arrays[xkey], arrays[axiskey]
            ids, labels = arrays["record_ids"].astype(str), arrays[labelkey].astype(str)
            assert len(set(ids)) == len(ids)
            assert values.shape == (len(ids), len(axis))
            assert np.isfinite(values).all() and np.all(np.diff(axis) > 0)
            unique, counts = np.unique(labels, return_counts=True)
            if dataset == "zaoren":
                zaoren_labels = dict(zip(ids, labels))
            rows[dataset] = {
                "path": relative, "sha256": sha(ROOT / relative),
                "feature_field": xkey, "axis_field": axiskey,
                "source_label_field": labelkey, "records": len(ids),
                "unique_record_ids": len(set(ids)), "columns": values.shape[1],
                "all_finite": True, "axis_strictly_increasing": True,
                "axis_range": [float(axis.min()), float(axis.max())],
                "labels": dict(zip(unique.tolist(), counts.tolist())),
                "X_fingerprint": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                "ids_fingerprint": id_sha(ids),
            }
    split_relative = "reports/staged-nested-optimization/2026-10-31/core/splits.json"
    folds = []
    all_query = []
    for fold in json.loads((ROOT / split_relative).read_text()):
        query, source = set(fold["test_sample_ids"]), set(fold["train_sample_ids"])
        assert not query & source and query | source == set(zaoren_labels)
        all_query.extend(query)
        inner = []
        for part in fold["inner_folds"]:
            train, val = set(part["train_sample_ids"]), set(part["validation_sample_ids"])
            assert not train & val and not (train | val) & query and train | val == source
            classes, counts = np.unique([zaoren_labels[i] for i in train], return_counts=True)
            assert classes.tolist() == [f"ZY{i:02d}" for i in range(1, 8)]
            inner.append({
                "source": len(train), "validation": len(val),
                "source_set_sha256": id_sha(train), "validation_set_sha256": id_sha(val),
                "source_class_counts": dict(zip(classes.tolist(), counts.tolist())),
            })
        folds.append({
            "fold": fold["fold"], "source": len(source), "query": len(query),
            "source_set_sha256": id_sha(source), "query_set_sha256": id_sha(query),
            "inner": inner,
        })
    assert len(all_query) == len(set(all_query)) == 855
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INDEPENDENT_CANONICAL_ARRAY_AND_HISTORICAL_PARTITION_SNAPSHOT",
        "new_training_operations": 0, "new_model_scoring_operations": 0,
        "script_sha256": sha(__file__), "input_arrays": rows,
        "historical_split_path": split_relative,
        "historical_split_sha256": sha(ROOT / split_relative), "folds": folds,
        "independence_or_units_certified": False,
        "limitations": [
            "All acid855 are exposed development; independence user-declared and one source batch/class.",
            "Ginseng42 upstream plants unknown; export groups are not biological batches.",
            "HQ/TM labels are anonymous file-origin groups, not verified named origins.",
            "Teacher absorption length units are unverified; shape compatibility is not physical calibration.",
            "Fruit unknown physical axis/leaf curves unknown specimen units block physical THz transfer.",
        ],
    }
    path = HERE / "canonical_snapshot.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(path.relative_to(ROOT)), "sha256": sha(path),
                      "contexts": 15, "new_training_operations": 0}))


if __name__ == "__main__":
    main()
