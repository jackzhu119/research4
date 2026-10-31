#!/usr/bin/env python3
"""Prepare a LOCAL-ONLY, fixed time-index waveform regression interface.

No model fits or data-driven feature selection occur here. Upstream numerical
data have no detected license: the NPZ must remain in the ignored acquisition
directory. Do not call these waveforms absorption coefficients or construct a
THz axis from the unverified 0.1 calibration interval.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import loadmat

from audit_princeton_fruit_20261031 import mat_identity, sha256, workbook_targets

EXPECTED_ARCHIVE_SHA256 = "902cca56a744187dd3eb3bbed41aa1ce28eb5e75e5b003946d29007a7fed603f"
# Fixed before any model evaluation: four equally spaced scheduled observations
# per fruit. Other scheduled observations remain available, without relabeling.
TIMEPOINT_INDICES = (0, 10, 20, 30)


def prepare(archive: Path, output: Path, manifest: Path):
    if output.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite a prepared interface or manifest")
    archive_hash = sha256(archive.read_bytes())
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive hash differs from the independent audit")
    expected_parent = archive.resolve().parent
    if not output.resolve().is_relative_to(expected_parent):
        raise ValueError("Numeric output must stay under the ignored raw archive directory")
    with zipfile.ZipFile(archive) as z:
        workbook = next(n for n in z.namelist() if n.endswith(".xlsx"))
        lookup, summaries = workbook_targets(z.read(workbook))
        times = sorted({timestamp for _, timestamp in lookup})
        chosen_times = {times[i] for i in TIMEPOINT_INDICES}
        sources = {}
        for member in z.namelist():
            if not member.endswith(".mat"):
                continue
            ident = mat_identity(member)
            if ident["role"] != "Fruit" or ident["condition"] != "Tag" or ident["timestamp"] not in chosen_times:
                continue
            key = (ident["fruit_id"], ident["timestamp"])
            # Prefer explicitly T-named canonical files. The audit established
            # byte-identical arrays for the May13 duplicate aliases.
            if key in sources and not ident["canonical_name"]:
                continue
            sources[key] = (member, ident)
        arrays, brix, dm, record_ids, groups, fruit_types, timestamps, members = ([] for _ in range(8))
        for key, (member, ident) in sorted(sources.items()):
            targets = lookup[key]
            if targets is None:
                raise ValueError("A prelocked observation has missing targets; no silent sample replacement")
            parsed = loadmat(io.BytesIO(z.read(member)))
            values = [v for k, v in parsed.items() if not k.startswith("__")]
            if len(values) != 1 or values[0].shape != (1, 7000) or not np.isfinite(values[0]).all():
                raise ValueError(f"Invalid waveform {member}")
            arrays.append(values[0].ravel())
            brix.append(targets["Brix"])
            dm.append(targets.get("DM", np.nan))
            groups.append(ident["fruit_id"])
            fruit_types.append(ident["fruit_type"])
            timestamps.append(ident["timestamp"])
            record_ids.append(ident["fruit_id"] + "__" + ident["timestamp"] + "__Tag")
            members.append(member)
    X = np.stack(arrays)
    if X.shape != (120, 7000) or len(set(groups)) != 30 or len(set(record_ids)) != 120:
        raise ValueError("Unexpected prelocked cohort size")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=X, y_Brix=np.array(brix), y_DM=np.array(dm),
                        record_id=np.array(record_ids), groups=np.array(groups),
                        fruit_type=np.array(fruit_types), timestamp=np.array(timestamps),
                        condition=np.repeat("Tag", len(groups)), sample_index=np.arange(7000),
                        source_member=np.array(members))
    info = {
        "protocol": "princeton-fruit-fixed-Tag-time-index-20261031-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_sha256": archive_hash, "npz_path": str(output),
        "npz_sha256": sha256(output.read_bytes()), "npz_bytes": output.stat().st_size,
        "records": 120, "waveform_features": 7000, "physical_fruit_ids": 30,
        "per_fruit_type": {ft: {"records": fruit_types.count(ft),
                                      "fruit_ids": len({g for g, f in zip(groups, fruit_types) if f == ft})}
                           for ft in ("MA", "PR", "PM")},
        "condition": "Tag",
        "timepoint_indices_zero_based": list(TIMEPOINT_INDICES),
        "timestamps": sorted(chosen_times),
        "input_quantity": "reported time-domain reflection waveform amplitude, arbitrary instrument units",
        "feature_axis": "integer sample index 0..6999; time unit and frequency axis unverified",
        "preparation": "verbatim finite float64 waveform, canonical exact-duplicate alias chosen, no scaling/FFT/selection",
        "targets": {
            "Brix": {"source": "Excel Brix column/value", "availability": "all fruit types",
                     "unit": "reported Brix reading; degree-Brix convention not explicitly documented in archive"},
            "DM": {"source": "first value in Mango DM, Brix cells", "availability": "MA only; NaN for PR/PM",
                   "unit": "reported DM reading; percentage/method not explicitly documented in archive"}},
        "measurement_method": "not specified in this archive; later Nature Food article uses NIR F-750 but same-cohort association remains unverified",
        "features_exclude": ["labels", "fruit ID", "time/date", "fruit type", "source filename"],
        "evaluation": "separate per-fruit-type regression; grouped outer and inner CV by persistent fruit_id; mean-baseline and train-only PCA/SVR/ridge comparisons; no row-wise split",
        "redistribution": "LOCAL_ONLY_NO_LICENSE_DETECTED_IN_UPSTREAM_REPOSITORY",
        "models_trained": 0,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"records":120,"groups":30,"X_shape":list(X.shape),"output":str(output),
                      "manifest":str(manifest),"models_trained":0}, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.archive,args.output,args.manifest)


if __name__ == "__main__":
    main()
