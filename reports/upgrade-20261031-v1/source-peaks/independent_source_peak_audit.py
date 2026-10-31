"""Read-only numerical checks of saved source candidates, not chemical assignment."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks,savgol_filter

from herb_spectra.leakage_guard import source_fingerprint


ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent


def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    with np.load(ROOT/"outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz",allow_pickle=False) as d:
        X,axis,ids=d["X_alpha"],d["axis"],d["record_ids"]
    index={str(value):i for i,value in enumerate(ids)}
    splits=json.loads((ROOT/"reports/staged-nested-optimization/2026-10-31/core/splits.json").read_text())
    events=json.loads((DEST/"fit_scope_manifest.json").read_text())["fit_events"]
    lock=json.loads((DEST/"protocol_lock.json").read_text())
    checks=0;maximum=0.;peaks=0
    for p,h in lock["source_sha256"].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h;checks+=1
    for fold in splits:
        number=fold["fold"];p=DEST/f"outer_{number:02d}"
        metadata=json.loads((p/"analysis.json").read_text())
        source_ids=metadata["source_sample_ids"]
        assert source_ids==fold["train_sample_ids"] and len(source_ids)==684;checks+=2
        assert not set(source_ids)&set(fold["test_sample_ids"]);checks+=1
        source=X[[index[rid] for rid in source_ids]]
        assert source_fingerprint(source)==metadata["source_data_fingerprint"];checks+=1
        raw=source.mean(0);smooth=savgol_filter(raw,15,3,mode="interp")
        source_rows=rows(p/"source_mean_spectrum.csv")
        for field,expected in [("frequency_thz",axis),("wavenumber_cm1",axis*33.3564095198),
                               ("raw_source_mean",raw),("smoothed_source_mean",smooth)]:
            actual=np.asarray([float(row[field]) for row in source_rows]);difference=np.max(np.abs(actual-expected))
            maximum=max(maximum,float(difference));np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
            checks+=len(expected)
        threshold=.05*float(np.ptp(raw));distance=int(np.ceil(.03/np.median(np.diff(axis))))
        found,props=find_peaks(smooth,prominence=threshold,distance=distance)
        candidates=rows(p/"candidate_peaks.csv")
        assert len(found)==len(candidates)==metadata["candidate_count"];checks+=1
        for j,(candidate,position) in enumerate(zip(candidates,found,strict=True)):
            assert int(candidate["frequency_index"])==int(position);checks+=1
            for field,value in [("frequency_thz",axis[position]),("wavenumber_cm1",axis[position]*33.3564095198),
                                ("prominence",props["prominences"][j]),("smoothed_source_mean_height",smooth[position])]:
                difference=abs(float(candidate[field])-float(value));maximum=max(maximum,difference)
                assert difference<1e-12;checks+=1
            assert candidate["chemical_assignment"]==candidate["theoretical_peak_cm1"]=="";checks+=1
        assert metadata["required_prominence"]==threshold and metadata["minimum_distance_samples"]==distance;checks+=2
        subset=[e for e in events if e["split_id"]==f"outer{number}.source_peaks"]
        assert len(subset)==2 and {e["event"] for e in subset}=={"fit_start","fit_completed"};checks+=1
        for event in subset:
            assert event["fit_sample_ids"]==source_ids and event["data_fingerprint"]==source_fingerprint(source);checks+=2
        peaks+=len(candidates)
    pairings=rows(DEST/"theory_pairing_NOT_RUN.csv")
    assert len(pairings)==15;checks+=1
    for entry in pairings:
        assert entry["full_molecule_dft_status"]=="NOT_RUN";checks+=1
        assert entry["experimental_frequency_thz"]==entry["theoretical_frequency_cm1"]==entry["assignment"]=="";checks+=1
    report={"status":"TECHNICAL_CHECK_PASS_FOR_SOURCE_DESCRIPTIVE_CANDIDATES_ONLY","checks":checks,
            "maximum_numeric_difference":maximum,"folds":5,"source_samples_per_fold":684,
            "candidate_entries_across_overlapping_folds":peaks,"distinct_independent_chemical_peaks_claimed":False,
            "labels_read":False,"new_classifier_fits":0,"theoretical_assignments":0,
            "classifiers_or_selection_masks_modified":False,"biological_independence_confirmed":False,
            "audit_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (DEST/"independent_source_peak_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
