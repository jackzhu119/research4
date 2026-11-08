#!/usr/bin/env python3
"""Plot a closed DF optimization from original JSON/NPY only; never runs QM.

The final fresh SCF/gradient is a repeat evaluation at geometry20, not a 21st
geometry or a new conformer. No checkpoint is read by this utility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, rdBase


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stats(g):
    norms = np.linalg.norm(g, axis=1)
    return {"atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(norms**2))),
            "atom_max_hartree_per_bohr": float(norms.max()),
            "component_rms_hartree_per_bohr": float(np.sqrt(np.mean(g**2))),
            "component_max_hartree_per_bohr": float(np.abs(g).max()),
            "gradient_norm_hartree_per_bohr": float(np.linalg.norm(g))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    repo=args.repo.resolve(); out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"metadata.json").exists():
        raise FileExistsError("Refusing to overwrite existing figures; use a fresh --output")
    started=datetime.now(timezone.utc).isoformat(); t0=time.monotonic()
    run=repo/"reports/pyscf-spinosin-continuation-20261031-v2"
    source=run/"spinosin-extended-optimization-001"
    audit_path=run/"df-optimization-audit/completed_df_optimization_review.json"
    audit=json.loads(audit_path.read_text())
    assert audit["closed_source_phase"]["status"]=="EXITED"
    assert audit["closed_source_phase"]["exit_code"]==0
    assert audit["DF_stationary_geometry_verified"] is True
    hashes={}
    def consume(path):
        key=str(path.relative_to(repo)); actual=sha256(path)
        expected=audit.get("source_files_sha256",{}).get(key)
        if expected is not None:
            assert actual==expected, f"Source changed after closed-source audit: {key}"
        hashes[key]=actual
        return path
    consume(audit_path)
    result_path=consume(source/"result.json")
    result=json.loads(result_path.read_text())
    assert sha256(result_path)==audit["source_result_sha256"]
    assert result["status"]=="DONE" and result["optimizer_converged"]
    settings=result["settings"]
    assert settings["xc"]=="PBE" and settings["density_fitting"] is True
    assert settings["dispersion"]=="NONE_EXPLICIT_PBE_PILOT_NOT_D3"
    assert result["frequencies_status"]=="NOT_RUN"
    assert result["IR_intensities_status"]=="NOT_COMPUTED"
    thresholds=settings["optimizer_thresholds"]
    grms=thresholds["convergence_grms"]; gmax=thresholds["convergence_gmax"]
    etol=thresholds["convergence_energy"]
    records=sorted((source/"geometry_steps").glob("step_*.json"))
    assert len(records)==audit["complete_geometry_evaluations"]==20
    rows=[]; coordinate_hashes=[]
    for n,p in enumerate(records,1):
        data=json.loads(consume(p).read_text())
        assert data["cycle"]==n and data["scf_converged"] is True
        prefix=p.name.removesuffix(".json")
        gp=consume(p.with_name(prefix+".gradient.npy"))
        cp=consume(p.with_name(prefix+".coordinates_angstrom.npy"))
        assert sha256(gp)==data["gradient_npy_sha256"]
        assert sha256(cp)==data["coordinates_angstrom_npy_sha256"]
        g=np.load(gp,allow_pickle=False); c=np.load(cp,allow_pickle=False)
        assert g.shape==c.shape==(75,3) and np.isfinite(g).all() and np.isfinite(c).all()
        computed=stats(g)
        for key,val in computed.items():
            assert np.isclose(val,data[key],atol=1e-14,rtol=1e-12)
        energy=float(data["energy_hartree"])
        assert np.isfinite(energy)
        delta=None if not rows else energy-rows[-1]["energy_hartree"]
        rows.append({"complete_geometry_evaluation":n,"energy_hartree":energy,
                     "delta_energy_from_previous_hartree":delta,
                     **computed,"atom_rms_threshold_hartree_per_bohr":grms,
                     "atom_max_threshold_hartree_per_bohr":gmax,
                     "energy_change_threshold_hartree":etol,
                     "gradient_thresholds_passed":computed["atom_rms_hartree_per_bohr"]<=grms and computed["atom_max_hartree_per_bohr"]<=gmax,
                     "energy_change_threshold_passed":None if delta is None else abs(delta)<=etol,
                     "scf_converged":True,"record_sha256":sha256(p),"gradient_sha256":sha256(gp),"coordinates_sha256":sha256(cp)})
        coordinate_hashes.append(sha256(cp))
    # Coordinate uniqueness here is for recorded numerical evaluations, not for
    # independent chemical conformers, plant samples or conformational sampling.
    assert len(set(coordinate_hashes))==20
    energies=np.array([r["energy_hartree"] for r in rows]); e_ref=energies[-1]
    for r in rows:
        r["energy_relative_to_evaluation20_millihartree"]=(r["energy_hartree"]-e_ref)*1000
    with (out/"df_optimization_evaluations.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    fresh_g=np.load(consume(source/"gradient_hartree_per_bohr.npy"),allow_pickle=False)
    assert fresh_g.shape==(75,3) and np.isfinite(fresh_g).all()
    fresh_stats=stats(fresh_g)
    for k,v in fresh_stats.items():
        assert np.isclose(v,result[k],atol=1e-14,rtol=1e-12)
        assert np.isclose(v,audit["fresh_gradient_statistics"][k],atol=1e-14,rtol=1e-12)
    fresh={"same_complete_geometry_evaluation":20,"additional_geometry_or_conformer":False,
           "energy_hartree":result["energy_hartree"],**fresh_stats,
           "atom_rms_threshold_hartree_per_bohr":grms,"atom_max_threshold_hartree_per_bohr":gmax,
           "gradient_thresholds_passed":fresh_stats["atom_rms_hartree_per_bohr"]<=grms and fresh_stats["atom_max_hartree_per_bohr"]<=gmax}
    with (out/"final_geometry_fresh_validation.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(fresh));w.writeheader();w.writerow(fresh)
    coords=np.load(consume(source/"optimized_coordinates_angstrom.npy"),allow_pickle=False)
    final_xyz=consume(source/"optimized.xyz").read_text().splitlines()
    elements=[line.split()[0] for line in final_xyz[2:]]
    assert coords.shape==(75,3) and len(elements)==75
    assert Counter(elements)=={"C":28,"H":32,"O":15}
    assert np.array_equal(coords,np.load(source/"geometry_steps/step_0020.coordinates_angstrom.npy",allow_pickle=False))

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"savefig.facecolor":"white"})
    x=np.arange(1,21)
    fig,axs=plt.subplots(3,1,figsize=(8.2,8.0),sharex=True,gridspec_kw={"height_ratios":[1.2,1,1]})
    axs[0].plot(x,(energies-e_ref)*1000,"o-",color="#135f8c",ms=4,lw=1.5)
    axs[0].set_ylabel("Energy relative to evaluation20\n(mEh)")
    axs[0].text(.98,.94,f"Evaluation20: {e_ref:.10f} Eh",ha="right",va="top",transform=axs[0].transAxes,fontsize=9)
    for ax,key,title,limit,color in [(axs[1],"atom_rms_hartree_per_bohr","Atomic gradient RMS",grms,"#13776b"),(axs[2],"atom_max_hartree_per_bohr","Maximum atomic gradient norm",gmax,"#b76419")]:
        ax.semilogy(x,[r[key] for r in rows],"o-",color=color,ms=4,lw=1.5,label="Original optimization evaluations")
        ax.axhline(limit,ls="--",color="#7d4343",lw=1,label=f"Gradient gate {limit:g} Eh/bohr")
        ax.scatter([20],[fresh_stats[key]],marker="D",facecolors="white",edgecolors="black",s=45,zorder=5,label="Fresh recheck at geometry20")
        ax.set_ylabel(title+"\n(Eh/bohr)")
        ax.legend(fontsize=8,loc="best",frameon=False)
    for ax in axs:
        ax.grid(alpha=.2);ax.set_xlim(.6,20.4)
    axs[-1].set_xticks(range(1,21));axs[-1].set_xlabel("Complete geometry evaluation within this continuation (1–20)")
    fig.suptitle("Spinosin: closed density-fitted PBE/def2-SVP optimization",fontsize=13,y=.985)
    fig.text(.5,.017,"Gas phase • grid level1 • no D3 • provisional geometry; frequencies NOT_RUN, IR NOT_COMPUTED",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(out/"spinosin_df_optimization_progress.png",dpi=220)
    fig.savefig(out/"spinosin_df_optimization_progress.pdf")
    plt.close(fig)

    # Take topology from the verified starting SDF, retaining atom order. Do not
    # infer chemistry/bonds from drawing distances or change optimized positions.
    sdf_path=consume(repo/"reports/pyscf-spinosin-pilot-20261031/conformers/C001_mmff94s.sdf")
    mol=next(Chem.SDMolSupplier(str(sdf_path),removeHs=False))
    assert mol is not None and mol.GetNumAtoms()==75
    assert [a.GetSymbol() for a in mol.GetAtoms()]==elements
    assert Chem.MolToInchiKey(Chem.RemoveHs(mol))=="VGGSULWDCMWZPO-ODEMIOGVSA-N"
    topology=[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        distance=float(np.linalg.norm(coords[i]-coords[j]))
        assert .7<distance<2.0, "Atom order/topology inconsistent with optimized geometry"
        topology.append({"atom_indices_zero_based":[i,j],"order":bond.GetBondTypeAsDouble(),"optimized_length_angstrom":distance})
    write_json(out/"display_topology.json",{"source_sdf":str(sdf_path.relative_to(repo)),"source_sdf_sha256":sha256(sdf_path),"bond_order_for_display_from_verified_starting_graph":True,"bonds":topology})
    # A rigid PCA rotation improves view only; all internal coordinates unchanged.
    centered=coords-coords.mean(axis=0);_,_,vh=np.linalg.svd(centered,full_matrices=False)
    displayed=centered@vh.T
    if np.linalg.det(vh)<0:
        displayed[:,2]*=-1
    fig=plt.figure(figsize=(8.5,6.8));ax=fig.add_subplot(111,projection="3d")
    for b in topology:
        p=displayed[b["atom_indices_zero_based"]]
        ax.plot(*p.T,color="#9198a1",lw=1.7,alpha=.85,zorder=1)
    colors={"C":"#505e6e","O":"#c63740","H":"#e7ebef"};sizes={"C":70,"O":86,"H":26}
    for el in ("C","O","H"):
        mask=np.array([s==el for s in elements]);p=displayed[mask]
        ax.scatter(*p.T,c=colors[el],s=sizes[el],edgecolors="#3e4b5a",linewidths=.35,depthshade=True,label=f"{el}: {sum(mask)}")
    radius=float(np.max(np.ptp(displayed,axis=0))/2)+.5
    for setter in (ax.set_xlim,ax.set_ylim,ax.set_zlim):setter(-radius,radius)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=29,azim=-68);ax.set_axis_off()
    ax.legend(loc="upper right",frameon=False,fontsize=9)
    ax.set_title("Spinosin: optimized 75-atom gas-phase model\nDF-PBE/def2-SVP, grid level1, no D3",fontsize=12,pad=7)
    fig.text(.5,.035,"Provisional stationary geometry, not a verified positive-frequency minimum.\nNo vibrational assignment; IR NOT_COMPUTED. Bonds show source connectivity, not bond-order analysis.",ha="center",fontsize=8.5)
    fig.subplots_adjust(left=0,right=1,top=.90,bottom=.10)
    fig.savefig(out/"spinosin_df_optimized_geometry.png",dpi=220)
    fig.savefig(out/"spinosin_df_optimized_geometry.pdf")
    plt.close(fig)
    metadata={"status":"DONE_POSTPROCESSING_CLOSED_DF_OPTIMIZATION_ONLY","started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_wall_seconds":time.monotonic()-t0,
              "script_sha256":sha256(Path(__file__)),"source_phase":str(source.relative_to(repo)),"source_result_status":result["status"],"source_phase_exited":True,
              "method_settings":settings,"complete_geometry_evaluations":20,"distinct_recorded_coordinate_arrays":20,
              "fresh_final_recheck_at_evaluation":20,"fresh_recheck_is_new_geometry_or_conformer":False,
              "energy_reference_for_plot_hartree":float(e_ref),"first_callback_energy_hartree":float(energies[0]),"last_callback_energy_hartree":float(energies[-1]),"fresh_final_validation":fresh,
              "gradient_definitions":{"atom_rms":"sqrt(mean_i(sum_a(g[i,a]**2)))","atom_max":"max_i(sqrt(sum_a(g[i,a]**2)))","component_rms":"sqrt(mean_i,a(g[i,a]**2))"},
              "gradient_thresholds_only_are_not_all_optimizer_gates":True,"positive_frequency_minimum_verified":False,"frequencies":"NOT_RUN","IR_intensities":"NOT_COMPUTED","new_quantum_computations":0,"checkpoints_opened":0,
              "geometry_display":{"coordinates_source":"optimized_coordinates_angstrom.npy","units":"angstrom","rigid_rotation_for_display_only":True,"topology_source":str(sdf_path.relative_to(repo)),"source_inchikey":"VGGSULWDCMWZPO-ODEMIOGVSA-N"},
              "environment":{"python":sys.version,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"rdkit":rdBase.rdkitVersion,"platform":platform.platform()},
              "limitations":["Energy and gradient traces are optimization evaluations, not independent samples or conformational sampling.","Density-fitted and no-DF methods must not be relabelled as one another.","Gas-phase uncorrected-PBE provisional geometry is not a solid-state mechanism or vibrational assignment.","No theoretical absorption spectrum or IR intensity was computed."]}
    write_json(out/"source_hashes.json",hashes)
    write_json(out/"metadata.json",metadata)
    print(json.dumps({k:metadata[k] for k in ("status","complete_geometry_evaluations","fresh_final_validation","elapsed_wall_seconds","new_quantum_computations","checkpoints_opened")},indent=2))


if __name__=="__main__":main()
