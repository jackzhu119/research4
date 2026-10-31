#!/usr/bin/env python3
"""Inspect original TikZ numeric coordinates as data, without executing TeX."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data/external/public_thz_20261031_v1/acquired/living_leaves_author_plot_data"
OUT = ROOT / "reports/public-thz-20261031-v1/acquisition"


def inspect(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    curves = []
    for block in text.split("\\addplot")[1:]:
        points = re.findall(r"^([+-]?[\d.]+(?:e[+-]?\d+)?)\s+([+-]?[\d.]+(?:e[+-]?\d+)?)\\\\", block, re.M | re.I)
        day = re.search(r"\\addlegendentry\{([^}]+)\}", block)
        curves.append({"legend": day.group(1) if day else None, "n_frequency_points": len(points),
                       "frequency_min_thz": min((float(x) for x, _ in points), default=None),
                       "frequency_max_thz": max((float(x) for x, _ in points), default=None)})
    return {"file": path.name, "species_folder_name": path.name.split("_leaf_")[0],
            "quantity": "real_refractive_index_dimensionless" if "n_idx" in path.name else "real_permittivity_dimensionless",
            "x_axis_explicitly_thz": "Frequency (THz)" in text, "curves": curves,
            "specimen_ids_available": False, "supervised_independent_cohort_eligible": False,
            "reason": "Original author plotted curves over days; no mapping to independent leaves or per-leaf outcomes."}


def main() -> None:
    records = [inspect(path) for path in sorted(LOCAL.glob("*.tikz"))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leaf_numeric_curve_audit.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    rows = []
    for rec in records:
        for curve in rec["curves"]:
            rows.append({"file": rec["file"], "species_folder_name": rec["species_folder_name"],
                         "quantity": rec["quantity"], **curve, "independent_specimen": "NOT_ESTABLISHED"})
    with (OUT / "leaf_numeric_curve_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"files_read": len(records), "plotted_curves": len(rows),
                      "independent_sample_count": None, "eligible_for_classification": False}))


if __name__ == "__main__":
    main()
