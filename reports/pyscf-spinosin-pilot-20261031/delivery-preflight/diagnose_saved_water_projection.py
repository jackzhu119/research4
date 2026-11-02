"""Diagnose small-matrix projection sensitivity of an existing water Hessian.

No electronic calculation, gradient, Hessian evaluation or optimization. Read
only the completed water calculation; never open active Spinosin checkpoints.
Symmetrization is diagnostic only and does not replace the original outputs.
"""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
import pyscf  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.scf import chkfile  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).parent
    out = here / "water_projection_numerical_diagnostic.json"
    if out.exists():
        raise FileExistsError("Preserve diagnostic and all original calculation artifacts")
    source = here.parent / "water-optfreq"
    state = json.loads((source / "result.json").read_text())
    assert state["status"] == "DONE" and state["geometry_optimized"]
    evidence = [source / name for name in ("result.json", "optimized.xyz", "calculation.chk",
                                          "hessian_hartree_per_bohr2.npy", "frequencies.csv")]
    before = {path.name: sha(path) for path in evidence}
    mol = chkfile.load_mol(str(source / "calculation.chk"))
    xyz_mol = gto.M(atom="\n".join((source / "optimized.xyz").read_text().splitlines()[2:]),
                    basis="def2-svp", unit="Angstrom", verbose=0)
    hessian = np.load(source / "hessian_hartree_per_bohr2.npy", allow_pickle=False)
    assert hessian.shape == (3, 3, 3, 3)
    symmetrized_hessian = (hessian + hessian.transpose(1, 0, 3, 2)) / 2

    def frequencies(geometry, array):
        return np.asarray(thermo.harmonic_analysis(
            geometry, array, exclude_trans=True, exclude_rot=True,
            imaginary_freq=False)["freq_wavenumber"])

    original_checkpoint = frequencies(mol, hessian)
    original_xyz = frequencies(xyz_mol, hessian)
    symmetrized_checkpoint = frequencies(mol, symmetrized_hessian)
    symmetrized_xyz = frequencies(xyz_mol, symmetrized_hessian)
    assert np.array_equal(original_checkpoint, np.asarray(state["frequencies_cm1"]))
    original_difference = float(np.max(np.abs(original_xyz - original_checkpoint)))
    symmetrized_difference = float(np.max(np.abs(symmetrized_xyz - symmetrized_checkpoint)))
    assert original_difference < 1e-3 and symmetrized_difference < 1e-8
    after = {path.name: sha(path) for path in evidence}
    assert before == after
    result = {
        "status": "DONE_SAVED_SMALL_MATRIX_NUMERICAL_DIAGNOSTIC_ONLY",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_SCF_gradient_Hessian_or_optimization_evaluations": 0,
        "active_Spinosin_checkpoint_read": False,
        "versions": {"pyscf": pyscf.__version__, "numpy": np.__version__},
        "script_sha256": sha(Path(__file__)),
        "input_evidence_sha256": before,
        "input_artifacts_unchanged_after_diagnostic": before == after,
        "saved_water_result_checkpoint_sha256_matches_actual": before["calculation.chk"] == state["checkpoint_sha256"],
        "source_harmonic_analysis": {"path": inspect.getsourcefile(thermo.harmonic_analysis),
                                     "sha256": sha(Path(inspect.getsourcefile(thermo.harmonic_analysis)))},
        "original_hessian_max_asymmetry_hartree_per_bohr2": float(np.max(np.abs(hessian - hessian.transpose(1, 0, 3, 2)))),
        "xyz_vs_checkpoint_max_coordinate_difference_bohr": float(np.max(np.abs(xyz_mol.atom_coords() - mol.atom_coords()))),
        "original_saved_frequencies_cm1": state["frequencies_cm1"],
        "original_hessian_checkpoint_recomputed_cm1": original_checkpoint.tolist(),
        "original_hessian_xyz_reprojected_cm1": original_xyz.tolist(),
        "original_hessian_xyz_vs_checkpoint_max_frequency_difference_cm1": original_difference,
        "diagnostic_only_symmetrized_checkpoint_cm1": symmetrized_checkpoint.tolist(),
        "diagnostic_only_symmetrized_xyz_cm1": symmetrized_xyz.tolist(),
        "symmetrized_hessian_xyz_vs_checkpoint_max_frequency_difference_cm1": symmetrized_difference,
        "original_vs_symmetrized_checkpoint_max_frequency_difference_cm1": float(np.max(np.abs(original_checkpoint - symmetrized_checkpoint))),
        "interpretation": [
            "Original water frequencies are the actual PySCF outputs and remain unchanged; this diagnostic does not replace any historical array or reported frequency.",
            "The original Hessian is slightly non-symmetric. PySCF projects it into a geometry-dependent basis before numpy.linalg.eigh, which assumes symmetry and by default consumes the lower triangle.",
            "Removing the antisymmetric part suppresses the geometry-roundtrip reprojection discrepancy from about1.36e-4 to about1e-12cm^-1 in this saved matrix.",
            "This supports a numerical projection/basis and Hessian-asymmetry explanation; do not describe it as evidence that physical water frequencies are sensitive to1e-12Bohr coordinate rounding.",
            "No finite-difference Hessian, grid-convergence study, physical frequency uncertainty or new Spinosin vibrational calculation was performed.",
        ],
        "IR_intensities": "NOT_COMPUTED",
    }
    with out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"],
        "original_reprojection_max_difference_cm1": original_difference,
        "symmetrized_reprojection_max_difference_cm1": symmetrized_difference,
        "original_data_unchanged": before == after,
        "diagnostic_sha256": sha(out)}, indent=2))


if __name__ == "__main__":
    main()
