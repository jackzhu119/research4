"""Compare completed direct/cached DF-J water artifacts; zero quantum jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
from pyscf.hessian import thermo  # noqa: E402
from pyscf.scf import chkfile  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).parent.parent
    out = Path(__file__).parent / "water_backend_comparison.json"
    if out.exists():
        raise FileExistsError("Keep completed audit")
    data = {}
    files = ["result.json", "calculation.chk", "gradient_hartree_per_bohr.npy",
             "hessian_hartree_per_bohr2.npy", "normal_modes.npy", "frequencies.csv"]
    for key, dirname in [("direct", "water-optfreq"), ("cached", "water-cached-optfreq")]:
        source = root / dirname
        result = json.loads((source / "result.json").read_text())
        assert all(result[k] is True for k in ["scf_converged", "optimizer_converged", "geometry_optimized", "final_gradient_thresholds_passed", "all_frequencies_real_positive"])
        assert result["status"] == "DONE" and result["imaginary_frequency_count"] == 0
        assert result["IR_intensities_status"] == "NOT_COMPUTED"
        mol = chkfile.load_mol(str(source / "calculation.chk"))
        gradient = np.load(source / "gradient_hartree_per_bohr.npy", allow_pickle=False)
        hessian = np.load(source / "hessian_hartree_per_bohr2.npy", allow_pickle=False)
        modes = np.load(source / "normal_modes.npy", allow_pickle=False)
        analysis = thermo.harmonic_analysis(mol, hessian, imaginary_freq=False)
        assert np.allclose(analysis["freq_wavenumber"], result["frequencies_cm1"], atol=1e-8)
        data[key] = {"result": result, "gradient": gradient, "hessian": hessian, "modes": modes,
                     "coordinate": mol.atom_coords(), "mol": mol,
                     "evidence_sha256": {name: sha(source / name) for name in files}}
    a, b = data["direct"], data["cached"]
    for key in ["xc", "basis", "grid_level", "dispersion", "density_fitting", "auxbasis", "charge", "multiplicity"]:
        assert a["result"]["settings"][key] == b["result"]["settings"][key]
    tolerances = {"energy_hartree": 1e-9, "gradient_hartree_per_Bohr": 1e-8,
                  "Hessian_hartree_per_Bohr2": 1e-8, "frequency_cm1": 1e-3}
    observed = {"energy_hartree": abs(b["result"]["energy_hartree"] - a["result"]["energy_hartree"]),
                "gradient_hartree_per_Bohr": float(np.max(np.abs(b["gradient"] - a["gradient"]))),
                "Hessian_hartree_per_Bohr2": float(np.max(np.abs(b["hessian"] - a["hessian"]))),
                "frequency_cm1": float(np.max(np.abs(np.asarray(b["result"]["frequencies_cm1"]) - a["result"]["frequencies_cm1"])))}
    assert all(observed[key] < tolerances[key] for key in observed)
    assert b["result"]["density_fit_cache"]["storage"] == "incore_ndarray"
    assert b["result"]["density_fit_cache"]["shape"] == [71, 300]
    assert a["result"]["density_fit_direct_J_no_stored_cderi"] is True
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_WATER_DIRECT_VS_CACHED_DF_NUMERICAL_COMPARISON_PASS",
        "new_quantum_jobs": 0, "script_sha256": sha(__file__),
        "tolerance_scope": "Numerical implementation smoke comparison, not a claimed physical accuracy or experimental uncertainty; not a Spinosin frequency validation",
        "comparison_tolerances": tolerances, "observed_max_absolute_differences": observed,
        "coordinates_max_difference_Bohr": float(np.max(np.abs(b["coordinate"] - a["coordinate"]))),
        "normal_mode_mass_overlap_absolute": np.abs(np.einsum("iar,a,iar->i", a["modes"], a["mol"].atom_mass_list(isotope_avg=True), b["modes"])).tolist(),
        "direct_frequencies_cm1": a["result"]["frequencies_cm1"],
        "cached_frequencies_cm1": b["result"]["frequencies_cm1"],
        "cached_tensor_metadata": b["result"]["density_fit_cache"],
        "both_strictly_positive_three_projected_modes": True,
        "IR_intensities_status": "NOT_COMPUTED",
        "evidence_sha256": {key: value["evidence_sha256"] for key, value in data.items()},
        "actual_worker_CPU_seconds": {key: value["result"]["process_cpu_seconds"] for key, value in data.items()},
        "source_worker_SHA256": {key: value["result"]["script_sha256"] for key, value in data.items()}}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "differences": observed, "comparison_sha256": sha(out)}))


if __name__ == "__main__":
    main()
