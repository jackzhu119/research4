"""Inspect the actual installed implementation; run no quantum calculations.

The harmonic-analysis checks use deliberately synthetic Hessian matrices.
They are not molecular frequencies and cannot be used in a research spectrum.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT / ".runtime/quantum-upgrade-site"
sys.path.insert(0, str(SITE))
from pyscf import dft, gto  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402
from pyscf.df.hessian import rks as df_rks_hessian  # noqa: E402
from pyscf.hessian import rks as rks_hessian  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(__file__).parent
    result_path = out / "method_audit.json"
    if result_path.exists():
        raise FileExistsError("Keep the previous audit snapshot unchanged")
    import pyscf
    base = Path(pyscf.__file__).parent
    dependency = {}
    for module, distribution in [("pyscf", "pyscf"), ("h5py", "h5py"),
            ("geometric", "geometric"), ("pyscf.dispersion", "pyscf-dispersion"),
            ("dftd3", "dftd3"), ("rdkit", "rdkit")]:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependency[module] = {"found": spec is not None, "version": version,
                              "origin": None if spec is None else spec.origin}
    source_paths = ["hessian/thermo.py", "geomopt/geometric_solver.py",
                    "hessian/rks.py", "df/hessian/rks.py", "df/hessian/rhf.py",
                    "scf/dispersion.py", "grad/dispersion.py", "hessian/dispersion.py"]
    sources = {p: {"path": str(base / p), "sha256": sha(base / p)} for p in source_paths}
    optimizer_source = (base / "geomopt/geometric_solver.py").read_text()
    dispersion_source = (base / "hessian/dispersion.py").read_text()
    harmonic_source = inspect.getsource(thermo.harmonic_analysis)
    assert "return conv, engine.mol" in optimizer_source
    assert "kwargs)[1]" in optimizer_source
    assert "DFTD3 does not support analytical Hessian" in dispersion_source
    assert "eps = 1e-5" in dispersion_source
    assert "#TODO: IR intensity" in harmonic_source

    # Construction only: no SCF, gradients, Hessian integrals or optimizer calls.
    mol = gto.M(atom="O 0 0 0; H 0 .8 .6; H 0 -.8 .6", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = "PBE"
    gradient_object = mf.nuc_grad_method()
    hessian_object = mf.Hessian()
    assert type(hessian_object).__module__ == "pyscf.df.hessian.rks"
    assert hessian_object.auxbasis_response == 2
    assert gradient_object.auxbasis_response is True

    n = mol.natm
    synthetic = (np.eye(3 * n) * .1).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    positive = thermo.harmonic_analysis(mol, synthetic, imaginary_freq=False)
    negative = thermo.harmonic_analysis(mol, -synthetic, imaginary_freq=False)
    modes = positive["norm_mode"]
    mass_gram = np.einsum("iar,a,jar->ij", modes, mol.atom_mass_list(isotope_avg=True), modes)
    assert len(positive["freq_wavenumber"]) == 3
    assert modes.shape == (3, 3, 3)
    assert np.all(positive["freq_wavenumber"] > 0)
    assert np.all(negative["freq_wavenumber"] < 0)
    assert int(negative["freq_error"]) == 3
    assert np.allclose(mass_gram, np.eye(3), atol=1e-12)
    assert not any("intens" in key.lower() for key in positive)

    basis_counts = {}
    for symbol in ("H", "C", "O"):
        basis_counts[symbol] = sum((2 * shell[0] + 1) * (len(shell[1]) - 1)
                                  for shell in gto.basis.load("def2-svp", symbol))
    candidates = {}
    for name, composition in [("Spinosin", {"C": 28, "H": 32, "O": 15}),
            ("Jujuboside A", {"C": 58, "H": 94, "O": 26}),
            ("Jujuboside B", {"C": 52, "H": 84, "O": 21})]:
        atoms = sum(composition.values())
        electrons = sum({"C": 6, "H": 1, "O": 8}[el] * count for el, count in composition.items())
        candidates[name] = {"formula_composition": composition, "expected_atoms": atoms,
            "neutral_electron_count": electrons, "closed_shell_occupation_if_singlet": electrons // 2,
            "expected_nonlinear_vibrational_modes": 3 * atoms - 6,
            "spherical_def2_svp_AO_count": sum(basis_counts[el] * count for el, count in composition.items()),
            "status": "FORMULA_AND_BASIS_DERIVED_SIZE_INDICATORS_ONLY_NOT_TIMING_OR_STRUCTURE_VALIDATION"}
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "IMPLEMENTATION_AND_SYNTHETIC_MATH_REVIEW_PASS_NO_QUANTUM_JOBS",
        "quantum_scf_gradient_hessian_optimization_jobs": 0,
        "script_sha256": sha(__file__), "dependencies_snapshot": dependency,
        "source_files": sources,
        "density_fitted_PBE": {"gradient_class": type(gradient_object).__module__ + "." + type(gradient_object).__name__,
            "hessian_class": type(hessian_object).__module__ + "." + type(hessian_object).__name__,
            "gradient_auxbasis_response": gradient_object.auxbasis_response,
            "hessian_auxbasis_response": hessian_object.auxbasis_response,
            "RKS_hessian_grid_response_default": rks_hessian.Hessian(mf).grid_response,
            "hessian_implementation_available": True, "scientific_validation_on_a_real_optimized_molecule": "ROOT_RUN_REQUIRED"},
        "optimizer": {"success_aware_API": "geometric_solver.kernel -> (converged_bool, mol)",
            "unsafe_success_inference": "geometric_solver.optimize returns only mol even when maxsteps is exceeded",
            "assert_convergence": True,
            "documented_defaults": {"energy_Eh": 1e-6, "gradient_RMS_Eh_per_Bohr": 3e-4,
                "gradient_max_Eh_per_Bohr": 4.5e-4, "displacement_RMS_Angstrom": 1.2e-3,
                "displacement_max_Angstrom": 1.8e-3}},
        "D3_BJ": {"energy_and_gradient_source_present": True,
            "plugin_required": "pyscf-dispersion", "plugin_snapshot_found": dependency["pyscf.dispersion"]["found"],
            "Hessian_implementation": "central finite difference of dispersion gradients, not analytic",
            "displacement_Bohr": 1e-5, "dispersion_gradient_calls_for_full_75atom_Hessian": 450,
            "actual_dispersion_job_run_by_this_audit": False,
            "cannot_silently_drop_term": True},
        "harmonic_analysis": {"signature": str(inspect.signature(thermo.harmonic_analysis)),
            "input_hessian_units": "Hartree/Bohr^2; axes(atom,atom,cartesian,cartesian)",
            "flattening": "hess.transpose(0,2,1,3).reshape(3N,3N)",
            "mass_default": "isotope-average atomic masses in atomic mass units",
            "translation_rotation_projected": True,
            "signed_imaginary_frequency_option": "imaginary_freq=False",
            "normal_mode_axes": "(mode,atom,cartesian)",
            "normal_mode_mass_orthonormal": True,
            "IR_intensities": "NOT_COMPUTED; PySCF standard routine has TODO only",
            "small_optimized_molecule_actual_workflow_required_before_interpretation": True},
        "synthetic_checks": {"nonphysical_matrices_only": True, "positive_projected_modes": 3,
            "negative_projected_modes": 3, "mass_orthogonality_max_abs_error": float(np.max(np.abs(mass_gram - np.eye(3)))),
            "no_synthetic_frequencies_exported_as_molecular_results": True},
        "unit_conversion": {"cm_inverse_to_THz": .0299792458,
            "THz_to_cm_inverse": 1 / .0299792458, "THz_range": [.1, 1.5],
            "same_range_cm_inverse": [.1 / .0299792458, 1.5 / .0299792458]},
        "basis_AO_counts_per_element": basis_counts, "candidate_size_indicators": candidates,
        "resource_and_interpretation_limits": [
            "Size indicators are not measured runtime estimates; no large-molecule DFT job was launched by audit.",
            "Use the same declared XC, basis, density-fitting, dispersion and integration grid for stationary-point and frequency validation.",
            "Re-run final SCF and nuclear gradient at returned optimized geometry; optimizer return alone is insufficient.",
            "An imaginary mode must be preserved and reported, not deleted to improve a frequency match.",
            "Low-frequency modes are grid/conformer/environment sensitive; coarse-grid isolated-molecule pilot is not a solid-state peak assignment.",
            "Do not scale frequencies to experimental THz candidate peaks or imply IR intensities from eigenvectors."]}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "pyscf": pyscf.__version__,
                      "AO_counts": {k: v["spherical_def2_svp_AO_count"] for k, v in candidates.items()},
                      "method_audit_sha256": sha(result_path)}))


if __name__ == "__main__":
    main()
