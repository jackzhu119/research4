"""Run locked, descriptive peak analysis separately on five historical sources."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from herb_spectra.experimental_peaks import (LOCKED_PEAK_PROTOCOL, analyze_source_peaks,
                                            thz_to_cm1,cm1_to_thz)
from herb_spectra.leakage_guard import FitScope


ROOT=Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHA="a3fac9b2f0ee6b8e51c835c7c74354f80e2fac2bc66923e76647e35d3c6bec9d"
EXPECTED_SPLITS_SHA="fcf14231b4dc1a1e15af8c46310ee6f39e71520b76a7b62fe0e176485f6f5cfe"
EXPECTED_COMPOUNDS_SHA="f05301abafa7c16e254a50689d6b1ebc63a55bd32deb36b31d246f2212888666"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def csv_write(path,rows,fields):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz")
    parser.add_argument("--splits",type=Path,default=ROOT/"reports/staged-nested-optimization/2026-10-31/core/splits.json")
    parser.add_argument("--candidate-registry",type=Path,default=ROOT/"data/research/quantum_upgrade_20261031/candidate_registry.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Descriptive evidence must use a fresh directory")
    assert sha(args.input)==EXPECTED_INPUT_SHA,"Unexpected source dataset version"
    assert sha(args.splits)==EXPECTED_SPLITS_SHA,"Original five outer folds changed"
    legacy=ROOT/"reports/approved-zaoren/2026-10-31/experiments/alpha_primary/splits.json"
    assert args.splits.read_bytes()==legacy.read_bytes(),"Split records differ from historical approved study"
    assert sha(args.candidate_registry)==EXPECTED_COMPOUNDS_SHA,"Candidate scope changed"
    compounds=json.loads(args.candidate_registry.read_text())["compounds"]
    assert [row["candidate_id"] for row in compounds]==["spinosin","jujuboside_a","jujuboside_b"]
    assert all(row["full_molecule_dft_status"]=="NOT_RUN" and row["verified_3d_geometry"] is None for row in compounds)
    args.output.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),ROOT/"src/herb_spectra/experimental_peaks.py",ROOT/"src/herb_spectra/leakage_guard.py"]
    code={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    protocol={"protocol_id":"upgrade-source-descriptive-peaks-20261031-v1",
              "recorded_at_utc":datetime.now(timezone.utc).isoformat(),"parameters":asdict(LOCKED_PEAK_PROTOCOL),
              "prominence_definition":"0.05 times raw current-source mean spectrum peak-to-peak range",
              "scope":"Each historical outer684 source separately; no target numerical values or labels passed to analysis",
              "input_sha256":sha(args.input),"old_split_sha256":sha(args.splits),
              "candidate_registry_sha256":sha(args.candidate_registry),"source_sha256":code,
              "versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","matplotlib")},
              "cross_fold_frequency_union_used":False,"used_for_current_classifier_selection":False,
              "theory_computations":"NOT_RUN for all three full compounds",
              "access_boundary":"API-level source slicing; loader can access original full NPZ, no OS sandbox claimed",
              "labels_read":False,"random_seed_for_peak_detection":None}
    write(args.output/"protocol_lock.json",protocol)
    for path in source_paths:
        target=args.output/"source_snapshot"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    shutil.copyfile(args.candidate_registry,args.output/"candidate_registry_snapshot.json")
    folds=json.loads(args.splits.read_text())
    with np.load(args.input,allow_pickle=False) as archive:
        axis=np.asarray(archive["axis"]).copy()
        identifiers=np.asarray(archive["record_ids"]).copy()
        # The independent loader materializes compressed X once, then passes only
        # named source slices to the analytical interface. It never reads y.
        X=np.asarray(archive["X_alpha"])
    assert X.shape==(855,612) and len(set(identifiers.tolist()))==855 and len(folds)==5
    index={str(value):i for i,value in enumerate(identifiers)}
    events=[];all_candidates=[];pairings=[];summary=[]
    fields=["outer_fold","candidate_number","frequency_index","frequency_thz","wavenumber_cm1",
            "smoothed_source_mean_height","raw_source_mean_height","prominence","left_base_frequency_thz",
            "right_base_frequency_thz","chemical_assignment","theoretical_peak_cm1","classification_prior_used"]
    for fold in folds:
        number=fold["fold"];source_ids=fold["train_sample_ids"];target_ids=fold["test_sample_ids"]
        assert len(source_ids)==684 and len(target_ids)==171 and not set(source_ids)&set(target_ids)
        assert set(source_ids)|set(target_ids)==set(index)
        source=X[[index[value] for value in source_ids]].copy()
        scope=FitScope(source_ids,f"outer{number}.source_peaks","2026-10-31-zaoren-numbered-v1",
                       args.output/"fit_scope_manifest.jsonl",code_version=code["src/herb_spectra/experimental_peaks.py"])
        result=analyze_source_peaks(source,axis,source_ids,scope)
        directory=args.output/f"outer_{number:02d}";directory.mkdir()
        rows=[{"outer_fold":number,**p} for p in result["candidates"]]
        csv_write(directory/"candidate_peaks.csv",rows,fields)
        spectra=[{"frequency_thz":float(f),"wavenumber_cm1":float(thz_to_cm1(f)),
                  "raw_source_mean":float(raw),"smoothed_source_mean":float(smooth)}
                 for f,raw,smooth in zip(axis,result["raw_source_mean"],result["smoothed_source_mean"],strict=True)]
        csv_write(directory/"source_mean_spectrum.csv",spectra,list(spectra[0]))
        metadata={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        metadata.update(outer_fold=number,classification_evaluation_performed=False)
        write(directory/"analysis.json",metadata)
        fig,ax=plt.subplots(figsize=(8.4,4.8))
        ax.plot(axis,result["raw_source_mean"],color="#93A9B8",lw=1.1,label="Raw source mean (n=684)")
        ax.plot(axis,result["smoothed_source_mean"],color="#155E75",lw=1.7,label="Fixed SG15 / polynomial3")
        if rows:
            ax.scatter([r["frequency_thz"] for r in rows],[r["smoothed_source_mean_height"] for r in rows],
                       c="#B45309",marker="v",s=40,zorder=3,label="Descriptive candidates")
        else:
            ax.text(.04,.93,"NO_PEAKS under the locked rule",transform=ax.transAxes,va="top",color="#9A3412")
        ax.set(xlabel="Frequency (THz, inherited axis)",ylabel="Absorption coefficient (exported units)",
               title=f"Outer source {number+1}: source-only descriptive peaks")
        secondary=ax.secondary_xaxis("top",functions=(thz_to_cm1,cm1_to_thz))
        secondary.set_xlabel(r"Wavenumber (cm$^{-1}$; coordinate conversion only)")
        ax.legend(loc="best",fontsize=8);ax.grid(alpha=.15)
        fig.text(.5,.015,"No target labels; no chemical assignment; no classifier prior",ha="center",fontsize=8)
        fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(directory/"source_candidates.png",dpi=220)
        fig.savefig(directory/"source_candidates.pdf");plt.close(fig)
        summary.append({"outer_fold":number,"source_samples":len(source_ids),"source_status":result["status"],
                        "candidate_count":len(rows),"required_prominence":result["required_prominence"],
                        "minimum_distance_samples":result["minimum_distance_samples"],"raw_source_mean_ptp":result["raw_source_mean_ptp"]})
        for compound in compounds:
            pairings.append({"outer_fold":number,"candidate_id":compound["candidate_id"],"molecule_name":compound["name"],
                             "full_molecule_dft_status":"NOT_RUN","structure_status":compound["structure_geometry_status"],
                             "experimental_candidates_file":f"outer_{number:02d}/candidate_peaks.csv",
                             "experimental_frequency_thz":"","theoretical_frequency_cm1":"","assignment":"",
                             "pairing_status":"NO_THEORETICAL_PEAKS_NO_ASSIGNMENT"})
        events.extend(scope.entries);all_candidates.extend(rows)
    csv_write(args.output/"experimental_candidates_by_fold.csv",all_candidates,fields)
    csv_write(args.output/"theory_pairing_NOT_RUN.csv",pairings,list(pairings[0]))
    csv_write(args.output/"source_fold_summary.csv",summary,list(summary[0]))
    write(args.output/"fit_scope_manifest.json",{"fit_events":events,"source_only_descriptive_operations":5,
                                               "new_classifier_fits":0,"labels_read":False})
    assert all(sha(ROOT/p)==h for p,h in code.items())
    assert sha(args.input)==EXPECTED_INPUT_SHA and sha(args.splits)==EXPECTED_SPLITS_SHA
    write(args.output/"completed.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),
          "status":"EXPLORATORY_SOURCE_DESCRIPTIVE_ANALYSIS","folds":summary,
          "source_operations":5,"labels_read":False,"new_classifier_fits":0,
          "theory_assignment_count":0,"current_model_modified":False,"classifier_frequency_mask_created":False,
          "protocol_lock_sha256":sha(args.output/"protocol_lock.json")})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
