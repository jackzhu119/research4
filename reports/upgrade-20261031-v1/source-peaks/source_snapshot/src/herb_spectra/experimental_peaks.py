"""Descriptive experimental candidates from one permitted source fold only.

No labels, target spectra or theoretical assignments enter this interface.
Candidates are not a classifier mask, chemical identity or concentration.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .leakage_guard import LeakageError, source_fingerprint


THZ_TO_WAVENUMBER_CM1 = 33.3564095198


def thz_to_cm1(frequency):
    return np.asarray(frequency,dtype=float) * THZ_TO_WAVENUMBER_CM1


def cm1_to_thz(wavenumber):
    return np.asarray(wavenumber,dtype=float) / THZ_TO_WAVENUMBER_CM1


@dataclass(frozen=True)
class SourcePeakProtocol:
    sg_window: int = 15
    sg_polyorder: int = 3
    prominence_fraction_of_raw_mean_ptp: float = .05
    minimum_frequency_distance_thz: float = .03
    axis_uniform_rtol: float = 1e-5
    axis_uniform_atol: float = 1e-10


LOCKED_PEAK_PROTOCOL = SourcePeakProtocol()


def analyze_source_peaks(X_source, axis_thz, source_ids, scope, *, protocol=LOCKED_PEAK_PROTOCOL):
    """Only the current complete source set is accepted; y/query are not inputs."""
    if protocol != LOCKED_PEAK_PROTOCOL:
        raise LeakageError("Use a separately recorded version for changed peak parameters")
    values = np.asarray(X_source,dtype=float)
    axis = np.asarray(axis_thz,dtype=float)
    ids = tuple(str(v) for v in source_ids)
    if (values.ndim!=2 or not len(values) or values.shape[1]<protocol.sg_window
            or axis.shape!=(values.shape[1],) or not np.isfinite(values).all()
            or not np.isfinite(axis).all() or not np.all(np.diff(axis)>0)
            or len(ids)!=len(values) or len(set(ids))!=len(ids)):
        raise LeakageError("Finite source spectra, increasing axis and unique aligned IDs required")
    if set(ids)!=set(scope.allowed_ids):
        raise LeakageError("Peak analysis must use exactly its current source scope")
    steps = np.diff(axis)
    step = float(np.median(steps))
    if not np.allclose(steps,step,rtol=protocol.axis_uniform_rtol,atol=protocol.axis_uniform_atol):
        raise LeakageError("Locked SG analysis requires a uniform measured grid")
    before = source_fingerprint(values)
    with scope.operation("source_mean_sg15_candidate_peaks",ids,X=values) as proof:
        mean = values.mean(axis=0)
        smoothed = savgol_filter(mean,window_length=protocol.sg_window,
                                 polyorder=protocol.sg_polyorder,mode="interp")
        span = float(np.ptp(mean))
        prominence = protocol.prominence_fraction_of_raw_mean_ptp*span
        distance = max(1,int(math.ceil(protocol.minimum_frequency_distance_thz/step)))
        numerical_floor = 64*np.finfo(float).eps*max(1.,float(np.max(np.abs(mean))))
        if span <= numerical_floor:
            indices = np.array([],dtype=int)
            properties = {"prominences":[],"left_bases":[],"right_bases":[]}
        else:
            indices,properties = find_peaks(smoothed,prominence=prominence,distance=distance)
        candidates = []
        for number,index in enumerate(indices):
            candidates.append({"candidate_number":number+1,"frequency_index":int(index),
                "frequency_thz":float(axis[index]),"wavenumber_cm1":float(thz_to_cm1(axis[index])),
                "smoothed_source_mean_height":float(smoothed[index]),"raw_source_mean_height":float(mean[index]),
                "prominence":float(properties["prominences"][number]),
                "left_base_frequency_thz":float(axis[int(properties["left_bases"][number])]),
                "right_base_frequency_thz":float(axis[int(properties["right_bases"][number])]),
                "chemical_assignment":"","theoretical_peak_cm1":"","classification_prior_used":False})
    if source_fingerprint(values)!=before:
        raise LeakageError("Source-only analysis unexpectedly modified input values")
    return {"status":"CANDIDATES_FOUND" if candidates else "NO_PEAKS",
            "parameters":asdict(protocol),"source_sample_ids":list(ids),"source_samples":len(ids),
            "source_data_fingerprint":before,"fit_provenance":dict(proof),
            "raw_source_mean_ptp":span,"required_prominence":prominence,
            "minimum_distance_samples":distance,"median_axis_step_thz":step,
            "thz_to_cm1_factor":THZ_TO_WAVENUMBER_CM1,"candidate_count":len(candidates),
            "candidates":candidates,"raw_source_mean":mean,"smoothed_source_mean":smoothed,
            "labels_used":False,"target_spectra_received":False,"classification_prior_used":False,
            "chemical_or_concentration_assignment_established":False}
