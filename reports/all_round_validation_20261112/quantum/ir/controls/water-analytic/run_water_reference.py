"""Real water-only CPHF cross-check, using pinned official PySCF properties.

This audit reuses archived finite-difference water coordinates and mode basis;
the electronic SCF and analytic dipole response are recomputed independently.
It does not replace finite-difference production results or certify targets.
"""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import time

import numpy as np
import scipy
import pyscf
from pyscf import dft, gto, hessian, lib


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.time()
    lib.num_threads(2)
    # Works from the scratch bundle and after archiving under repo/reports/.
    quality_path = next((parent / "scripts" / "pyscf_ir_quality_20261111.py"
                         for parent in ROOT.parents
                         if (parent / "scripts" / "pyscf_ir_quality_20261111.py").is_file()), None)
    if quality_path is None:
        raise FileNotFoundError("Cannot locate independent IR quality module")
    quality = load("ir_quality_reference", quality_path)
    official = load("official_pinned_ir_rhf", ROOT / "rhf.py")
    source_path = ROOT / "source_water_result.json"
    array_path = ROOT / "source_water_analysis_arrays.npz"
    source = json.loads(source_path.read_text())
    expected_hash = source["analysis"]["artifacts"]["analysis_arrays.npz"]["sha256"]
    if sha(array_path) != expected_hash:
        raise ValueError("Archived finite-difference array hash mismatch")
    arrays = np.load(array_path)
    coords = np.asarray(source["coordinates_bohr"])
    np.testing.assert_allclose(coords, arrays["coordinates_bohr"], rtol=0, atol=0)
    mol = gto.M(atom=list(zip(["O", "H", "H"], coords)), unit="Bohr",
                basis="def2-SVP", charge=0, spin=0, verbose=4,
                output=str(ROOT / "water_pyscf.log"), max_memory=4000)
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 5
    mf.small_rho_cutoff = 0
    mf.direct_scf = False
    mf.direct_scf_tol = 1e-14
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 2e-9
    mf.conv_tol_cpscf = 1e-12
    mf.max_cycle = 200
    mf.chkfile = str(ROOT / "water_center.chk")
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent water SCF did not converge")
    density = mf.make_rdm1()
    orbital_gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    # Official rks.py only subclasses Infrared and sets this Hessian class.
    class OfficialRKS(official.Infrared):
        hess_cls = hessian.rks.Hessian
    ir = OfficialRKS(mf)
    ir.mf_hess = hessian.rks.Hessian(mf)
    ir.mf_hess.grid_response = True
    analytic = ir.kernel_dipderiv()
    modes = arrays["step_1_cartesian_mass_normalized_modes"].T
    masses = arrays["masses_amu"]
    ir.vib_dict = {"norm_mode": modes.reshape(3, 3, 3)}
    official_intensities = ir.kernel_ir()
    analytic_projected = modes @ analytic.reshape(9, 3)
    own_intensities = quality.prefactor_si() * np.sum(analytic_projected**2, axis=1)
    finite_series = np.stack([arrays[f"step_{i}_dipole_derivative_au"].T
                              for i in range(2)])
    finite_projected = np.einsum("mk,skd->smd", modes, finite_series)
    mu = mf.dip_moment(unit="AU", verbose=0)
    finite_check = quality.common_frame_derivative_comparison(
        finite_series, [0.002, 0.001], modes, masses)
    vector_differences = np.linalg.norm(analytic_projected - finite_projected[-1], axis=1)
    vector_tolerances = 2e-4 + 0.02 * np.maximum(np.linalg.norm(analytic_projected, axis=1),
                                               np.linalg.norm(finite_projected[-1], axis=1))
    finite_intensities = quality.prefactor_si() * np.sum(finite_projected[-1]**2, axis=1)
    intensity_differences = np.abs(own_intensities - finite_intensities)
    intensity_tolerances = 0.2 + 0.03 * np.maximum(own_intensities, finite_intensities)
    rigid = quality.rigid_body_residuals(analytic, coords, mu, 0)
    passed = bool(np.all(vector_differences <= vector_tolerances) and
                  np.all(intensity_differences <= intensity_tolerances) and
                  rigid["qualified"] and finite_check["qualified"] and
                  orbital_gradient <= 1e-8)
    np.savez(ROOT / "water_analytic_arrays.npz", coordinates_bohr=coords,
             masses_amu=masses, analytic_cartesian_dipole_derivatives_au=analytic,
             source_mode_rows=modes, analytic_projected_dipole_derivatives=analytic_projected,
             source_finite_difference_series=finite_series,
             source_finite_projected_dipole_derivatives=finite_projected,
             analytic_intensities_km_mol=own_intensities,
             official_intensities_km_mol=official_intensities,
             source_finite_intensities_km_mol=finite_intensities,
             analytic_hessian=ir.mf_hess.de, density=density)
    pin = json.loads((ROOT / "official_pin.json").read_text())
    result = {
        "schema": "research4-ir-water-independent-cphf-reference-v1",
        "runner_script_sha256": sha(Path(__file__).resolve()),
        "quality_module_sha256": sha(quality_path),
        "qualified_water_control_only": passed,
        "target_molecules_certified": False,
        "reference_only_not_production_replacement": True,
        "common_mode_basis_from_source": True,
        "official_code": {**pin, "rhf_sha256": sha(ROOT / "rhf.py"),
                          "rks_sha256": sha(ROOT / "rks.py")},
        "source_branch": "research/pyscf-publication-recovery-20261111",
        "source_water_result_sha256": sha(source_path),
        "source_arrays_sha256": sha(array_path),
        "source_arrays_authenticated_against_result": True,
        "method": {"xc": "PBE", "basis": "def2-SVP", "grid_level": 5,
                   "hessian_grid_response": True, "density_fitting": False,
                   "full_veff_every_cycle": True, "charge": 0, "spin": 0,
                   "conv_tol": 1e-12, "conv_tol_grad": 2e-9,
                   "conv_tol_cpscf": 1e-12, "threads": 2,
                   "dipole_origin_bohr": [0, 0, 0]},
        "runtime": {"python": platform.python_version(), "pyscf": pyscf.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "energy_hartree": float(energy),
        "source_energy_difference_hartree": float(energy - source["energy_hartree"]),
        "fresh_orbital_gradient_norm": orbital_gradient,
        "prefactor_si": quality.prefactor_si(),
        "prefactor_atomic": quality.prefactor_atomic(),
        "mass_normalization": quality.validate_mode_normalization(modes, masses),
        "source_finite_difference_quality": finite_check,
        "analytic_rigid_body_quality": rigid,
        "analytic_intensities_km_mol": own_intensities.tolist(),
        "official_intensities_km_mol": official_intensities.tolist(),
        "source_finite_intensities_km_mol": finite_intensities.tolist(),
        "projected_vector_differences": vector_differences.tolist(),
        "projected_vector_tolerances": vector_tolerances.tolist(),
        "intensity_absolute_differences_km_mol": intensity_differences.tolist(),
        "intensity_tolerances_km_mol": intensity_tolerances.tolist(),
        "analytic_cartesian_max_difference_from_finite": float(np.max(np.abs(analytic.reshape(9, 3) - finite_series[-1]))),
        "checkpoint_sha256": sha(ROOT / "water_center.chk"),
        "analytic_arrays_sha256": sha(ROOT / "water_analytic_arrays.npz"),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_wall_seconds": time.time() - started,
        "scope": "Isolated water control at archived geometry. Analytic orbital response vs finite displacement dipoles in identical fixed source modes. No claims for global conformers, solid THz assignments, or target-molecule qualification.",
    }
    (ROOT / "water_analytic_reference.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ["qualified_water_control_only", "fresh_orbital_gradient_norm", "analytic_intensities_km_mol", "source_finite_intensities_km_mol", "projected_vector_differences", "intensity_absolute_differences_km_mol", "elapsed_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
