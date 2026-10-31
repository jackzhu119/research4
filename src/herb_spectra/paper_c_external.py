"""Separate external THz tasks with explicit provenance and evaluation gates.

Reading spectra never establishes biological independence or an origin label.
No fitting, label inference from filenames, extrapolation or imputation occurs.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .dataset_registry import select_workbooks
from .io import read_spectra


@dataclass(frozen=True)
class ExternalTask:
    task_id: str
    axis: np.ndarray
    spectra: np.ndarray
    record_ids: tuple[str, ...]
    labels: tuple[str, ...] | None
    physical_ids: tuple[str, ...]
    axis_unit: str | None
    quantity: str
    value_unit: str | None
    independent_units_verified: bool
    labels_verified: bool
    provenance: dict

    def validate(self):
        x, axis = np.asarray(self.spectra), np.asarray(self.axis)
        if x.ndim != 2 or axis.ndim != 1 or x.shape[1] != len(axis):
            raise ValueError("Spectral dimensions do not match the coordinate axis")
        if not np.isfinite(x).all() or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
            raise ValueError("Nonfinite data or nonincreasing coordinates")
        if len(self.record_ids) != len(x) or len(set(self.record_ids)) != len(x):
            raise ValueError("Record IDs must be unique and aligned with rows")
        if len(self.physical_ids) != len(x):
            raise ValueError("Physical identity field must explicitly align with every row")
        if self.labels is not None and len(self.labels) != len(x):
            raise ValueError("Labels are not aligned with rows")
        return self

    def scientific_blocks(self):
        self.validate()
        blocks = []
        if not self.independent_units_verified or not all(self.physical_ids):
            blocks.append("physical_sample_independence_or_repeat_mapping_unverified")
        if self.labels is None or not self.labels_verified:
            blocks.append("scientific_task_labels_unverified_or_missing")
        if not self.axis_unit or not self.value_unit:
            blocks.append("physical_units_unverified")
        return blocks

    def require_independent_evaluation(self):
        blocks = self.scientific_blocks()
        if blocks:
            raise ValueError("Independent task evaluation blocked: " + "; ".join(blocks))
        return self


def require_transfer_compatibility(source, target):
    """Require verified units and identities before defining a shared THz task.

    This gate does not establish compatible instruments, pellets or inversion
    settings; those conditions still require scientific metadata review.
    """
    source.require_independent_evaluation()
    target.require_independent_evaluation()
    if source.axis_unit != target.axis_unit:
        raise ValueError("Coordinate units differ")
    if source.quantity != target.quantity or source.value_unit != target.value_unit:
        raise ValueError("Physical quantities or value units differ")
    if max(source.axis[0], target.axis[0]) >= min(source.axis[-1], target.axis[-1]):
        raise ValueError("No common measured frequency interval")
    return True


def interpolate_within_measured_support(task, new_axis):
    """Numerical alignment only; never adds physical resolution or independence."""
    task.validate()
    axis = np.asarray(new_axis, dtype=float)
    if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
            or not np.all(np.diff(axis) > 0)):
        raise ValueError("Target frequency axis must be finite and increasing")
    if axis[0] < task.axis[0] or axis[-1] > task.axis[-1]:
        raise ValueError("Extrapolation outside measured support is forbidden")
    return np.stack([np.interp(axis, task.axis, row) for row in task.spectra])


def load_huangqi_alpha_records(repo):
    repo = Path(repo)
    selected = [s for s in select_workbooks(repo=repo) if s.dataset == "huangqi"]
    rows, names, file_groups, sources = [], [], [], []
    common_axis = None
    for index, source in enumerate(selected, 1):
        spectrum = read_spectra(source.path, bounds=(0.1, 1.5))
        if common_axis is not None and not np.array_equal(spectrum.axis, common_axis):
            raise ValueError("Huangqi active workbooks have different frequency axes")
        common_axis = spectrum.axis
        if len(spectrum.values) != source.record_count:
            raise ValueError("Huangqi registry count mismatch")
        rows.extend(spectrum.values)
        names.extend("HQ_RECORD_" + hashlib.sha256(
            f"{source.relative_path}:{col}".encode()).hexdigest()[:20]
            for col in spectrum.column_numbers)
        file_groups.extend([f"HQ_FILE_{index:02d}"] * len(spectrum.values))
        sources.append({"path": source.relative_path, "sha256": source.sha256,
                        "records": len(spectrum.values), "filename_group": f"HQ_FILE_{index:02d}"})
    return ExternalTask(
        "huangqi_alpha_records", common_axis, np.asarray(rows), tuple(names), None,
        ("",) * len(rows), "THz", "absorption_coefficient", None, False, False,
        {"sources": sources, "file_group_ids": file_groups,
         "file_groups_are_not_verified_origin_labels": True,
         "axis_unit_basis": "Workbook header Thz; instrument and inversion metadata not supplied",
         "value_unit_basis": "Absorption coefficient worksheet has no length-unit declaration",
         "license": "user_supplied; no separate public redistribution license supplied"}).validate()


def load_ginseng_alpha_pellets(repo):
    root = Path(repo) / "data/external/ginseng_hvbp5mdtxx_v2/original"
    rows, names, labels, sources = [], [], [], []
    common_axis = None
    for group in range(1, 7):
        spectrum_path, label_path = root / f"data_a{group}.csv", root / f"label_{group}.csv"
        data = np.loadtxt(spectrum_path, delimiter=",", skiprows=1, encoding="utf-8-sig")
        group_labels = np.loadtxt(label_path, delimiter=",", dtype=int, encoding="utf-8-sig").reshape(-1)
        if data.shape[1] - 1 != len(group_labels) or set(group_labels) != {0, 1}:
            raise ValueError("Invalid ginseng species labels or column alignment")
        if common_axis is not None and not np.array_equal(common_axis, data[:, 0]):
            raise ValueError("Ginseng files have different frequency axes")
        common_axis = data[:, 0]
        rows.extend(data[:, 1:].T)
        names.extend(f"G{group}_S{i:02d}" for i in range(1, len(group_labels) + 1))
        labels.extend("Panax_ginseng" if y == 1 else "Panax_quinquefolius" for y in group_labels)
        for path in (spectrum_path, label_path):
            sources.append({"path": str(path.relative_to(repo)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return ExternalTask(
        "ginseng_species_pellet_records", common_axis, np.asarray(rows), tuple(names), tuple(labels),
        ("",) * len(rows), "THz", "absorption_coefficient", "cm^-1", False, True,
        {"sources": sources, "doi": "10.17632/hvbp5mdtxx.2",
         "source_url": "https://data.mendeley.com/datasets/hvbp5mdtxx/2", "license": "CC BY 4.0",
         "units_and_species_code_basis": "Publisher description, rechecked 2026-10-31",
         "pairing_basis": "Same file suffix and sample column order; no parent plant identity table",
         "biological_independence_verified": False,
         "download_basis": "Previously uploaded archive; not newly downloaded or byte-matched to repository"}).validate()
