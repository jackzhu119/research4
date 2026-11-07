"""Independently recompute modes from saved Hessian; no SCF/gradient jobs."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
from pyscf import gto  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402
from pyscf.scf import chkfile  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = Path(__file__).parent.parent / "water-optfreq"
    out = Path(__file__).parent / "water_saved_result_review.json"
    if out.exists():
        raise FileExistsError("Preserve saved audit")
    state = json.loads((source / "result.json").read_text())
    assert state["status"] == "DONE" and state["optimizer_converged"] and state["geometry_optimized"]
    assert state["scf_converged"] and state["final_gradient_thresholds_passed"]
    assert state["script_sha256"] == sha(ROOT / "scripts/pyscf_pilot_worker_20261031.py")
    assert state["optimized_xyz_sha256"] == sha(source / "optimized.xyz")
    lines = (source / "optimized.xyz").read_text().splitlines()
    assert int(lines[0]) == 3
    xyz_mol = gto.M(atom="\n".join(lines[2:]), basis=state["settings"]["basis"], unit="Angstrom", verbose=0)
    mol = chkfile.load_mol(str(source / "calculation.chk"))
    coordinate_roundtrip_error = float(np.max(np.abs(mol.atom_coords() - xyz_mol.atom_coords())))
    assert coordinate_roundtrip_error < 1e-11
    gradient = np.load(source / "gradient_hartree_per_bohr.npy", allow_pickle=False)
    atom_norms = np.linalg.norm(gradient, axis=1)
    atom_rms, atom_max = float(np.sqrt(np.mean(atom_norms**2))), float(atom_norms.max())
    assert np.isclose(atom_rms, state["atom_rms_hartree_per_bohr"], atol=1e-14)
    assert np.isclose(atom_max, state["atom_max_hartree_per_bohr"], atol=1e-14)
    assert atom_rms <= 3e-4 and atom_max <= 4.5e-4
    hessian = np.load(source / "hessian_hartree_per_bohr2.npy", allow_pickle=False)
    assert hessian.shape == (3, 3, 3, 3) and np.isfinite(hessian).all()
    symmetry_error = float(np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2))))
    assert symmetry_error < 1e-5
    analysis = thermo.harmonic_analysis(mol, hessian, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
    xyz_analysis = thermo.harmonic_analysis(xyz_mol, hessian, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
    frequencies = np.asarray(analysis["freq_wavenumber"])
    assert frequencies.shape == (3,) and np.all(frequencies > 0)
    assert np.allclose(frequencies, state["frequencies_cm1"], atol=1e-7, rtol=1e-9)
    assert np.allclose(frequencies * .0299792458, state["frequencies_thz"], atol=1e-8)
    modes = np.load(source / "normal_modes.npy", allow_pickle=False)
    masses = mol.atom_mass_list(isotope_avg=True)
    gram = np.einsum("iar,a,jar->ij", modes, masses, modes)
    assert modes.shape == (3, 3, 3) and np.allclose(gram, np.eye(3), atol=1e-10)
    overlaps = np.einsum("iar,a,iar->i", modes, masses, analysis["norm_mode"])
    assert np.allclose(np.abs(overlaps), 1, atol=1e-8)
    assert np.allclose(np.load(source / "reduced_mass_amu.npy", allow_pickle=False), analysis["reduced_mass"], atol=1e-8)
    with (source / "frequencies.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and all(row["IR_intensity_status"] == "NOT_COMPUTED" for row in rows)
    assert all(row["imaginary"] == "False" for row in rows)
    assert np.allclose([float(row["frequency_cm1"]) for row in rows], frequencies, atol=1e-7)
    evidence = ["result.json", "input.json", "pyscf.log", "optimized.xyz", "calculation.chk", "gradient_hartree_per_bohr.npy",
                "hessian_hartree_per_bohr2.npy", "normal_modes.npy", "reduced_mass_amu.npy", "frequencies.csv", "events.jsonl"]
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SAVED_ACTUAL_WATER_OPT_FREQ_REVIEW_PASS",
        "new_SCF_gradient_Hessian_or_optimization_jobs": 0,
        "source_calculation_is_actual": True, "molecule": "water, not a medicinal compound",
        "authoritative_geometry": "Full-precision molecule from saved SCF checkpoint",
        "XYZ_12decimal_roundtrip_max_coordinate_difference_Bohr": coordinate_roundtrip_error,
        "XYZ_12decimal_reprojection_max_frequency_difference_cm1": float(np.max(np.abs(xyz_analysis["freq_wavenumber"] - frequencies))),
        "preserved_auditor_attempt": "water_review_attempt01_xyz_precision: strict frequency comparison after roundedXYZ import; no scientific artifact changed",
        "input_evidence_sha256": {name: sha(source / name) for name in evidence},
        "script_sha256": sha(__file__), "optimizer_and_fresh_SCF_converged": True,
        "final_atom_gradient_RMS_Eh_per_Bohr": atom_rms,
        "final_atom_gradient_max_Eh_per_Bohr": atom_max,
        "hessian_max_asymmetry_Eh_per_Bohr2": symmetry_error,
        "projected_vibrational_modes": 3, "imaginary_modes": 0,
        "all_frequencies_strictly_positive": True,
        "independently_recomputed_frequencies_cm1": frequencies.tolist(),
        "independently_recomputed_frequencies_THz": (frequencies * .0299792458).tolist(),
        "frequency_recompute_max_difference_cm1": float(np.max(np.abs(frequencies - state["frequencies_cm1"]))),
        "normal_mode_mass_Gram_max_error": float(np.max(np.abs(gram - np.eye(3)))),
        "normal_mode_sign_independent_mass_overlaps": np.abs(overlaps).tolist(),
        "IR_intensities_status": "NOT_COMPUTED",
        "direct_J_no_stored_cderi": state["density_fit_direct_J_no_stored_cderi"],
        "actual_worker_CPU_seconds": state["process_cpu_seconds"],
        "interpretation_limit": "The PBE/def2-SVP grid1 DF-J workflow completed on H2O. This does not complete Spinosin DFT, establish grid convergence for THz torsions, or assign experimental herbal peaks."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "frequencies_cm1": frequencies.tolist(),
                      "review_sha256": sha(out)}))


if __name__ == "__main__":
    main()
