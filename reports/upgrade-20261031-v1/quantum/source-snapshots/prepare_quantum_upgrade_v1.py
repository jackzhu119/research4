"""Validate quantum inputs and run an isolated, non-herbal PySCF smoke test.

The water calculation is software validation only. It does not generate herbal
training samples, an optimized candidate structure, or an assigned THz peak.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CM1_TO_THZ = 0.0299792458
ATOMIC_NUMBERS = {"H": 1, "C": 6, "O": 8}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def formula_counts(formula):
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if not tokens or "".join(s + n for s, n in tokens) != formula:
        raise ValueError("Only an explicit elemental molecular formula is accepted")
    counts = Counter()
    for symbol, number in tokens:
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError("Element not covered by this candidate validator")
        amount = int(number or "1")
        if amount < 1:
            raise ValueError("Formula count must be positive")
        counts[symbol] += amount
    return dict(counts)


def validated_xyz(text, formula, provenance, charge, multiplicity):
    """Reject guessed/unattributed coordinates and incompatible composition."""
    if not provenance.get("structure_verified") or not provenance.get("source_url"):
        raise ValueError("Verified structure and its source are required")
    if provenance.get("stereochemistry_status") != "verified":
        raise ValueError("Stereochemistry must be resolved before candidate preparation")
    if provenance.get("geometry_origin") not in {"experimental", "database_3d", "documented_conformer_generation"}:
        raise ValueError("Geometry provenance is required")
    if provenance.get("xyz_sha256") != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError("Coordinate content changed from the reviewed source")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("Complete XYZ data required")
    try:
        declared = int(lines[0].strip())
        atoms = [(p[0], *(float(v) for v in p[1:])) for line in lines[2:] if (p := line.split())]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Malformed XYZ coordinates") from exc
    if len(atoms) != declared or any(len(a) != 4 for a in atoms):
        raise ValueError("XYZ atom count or coordinate columns inconsistent")
    if dict(Counter(a[0] for a in atoms)) != formula_counts(formula):
        raise ValueError("XYZ atom composition does not match reviewed formula")
    if not np.isfinite(np.asarray([a[1:] for a in atoms])).all():
        raise ValueError("Coordinates must be finite")
    if not isinstance(charge, int) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError("Explicit integer charge and positive multiplicity required")
    electrons = sum(ATOMIC_NUMBERS[a[0]] for a in atoms) - charge
    if electrons < multiplicity - 1 or (electrons - (multiplicity - 1)) % 2:
        raise ValueError("Charge/multiplicity incompatible with electron parity")
    return atoms


def gaussian_input(xyz, formula, provenance, charge, multiplicity, title):
    atoms = validated_xyz(xyz, formula, provenance, charge, multiplicity)
    if "\n" in title or "\r" in title:
        raise ValueError("One-line title required")
    route = "#p B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ Opt=(Tight,CalcFC) Freq Int=UltraFine SCF=(Tight,XQC) NoSymm"
    coordinates = "\n".join(f"{a[0]} {a[1]:.10f} {a[2]:.10f} {a[3]:.10f}" for a in atoms)
    return f"%NProcShared=1\n%Mem=2GB\n{route}\n\n{title}\n\n{charge} {multiplicity}\n{coordinates}\n\n"


def parse_gaussian_frequency_log(text):
    frequencies = []
    intensities = []
    for line in text.splitlines():
        for prefix, output in (("Frequencies --", frequencies), ("IR Inten    --", intensities)):
            if prefix in line:
                try:
                    output.extend(float(v.replace("D", "E")) for v in line.split("--", 1)[1].split())
                except ValueError as exc:
                    raise ValueError("Unparseable frequency/intensity value") from exc
    normal = "Normal termination of Gaussian" in text and "Error termination" not in text
    optimized = "Optimization completed." in text
    return {"normal_termination":normal,"optimization_completed":optimized,
            "frequencies_cm1":frequencies,"frequencies_thz":[v * CM1_TO_THZ for v in frequencies],
            "imaginary_frequency_count":sum(v < 0 for v in frequencies),"ir_intensities":intensities,
            "candidate_ready_for_interpretation":bool(normal and optimized and frequencies and
                len(intensities) == len(frequencies) and all(v >= 0 for v in frequencies)),
            "experimental_peak_assignment_verified":False,
            "parser_status":"parsed_metadata_only_not_a_Gaussian_run"}


def validate_theory_prior(prior, allowed_train_ids, forbidden_ids, protocol_sha256):
    """A limited provenance gate, not proof of complete leakage absence."""
    if prior.get("locked_protocol_sha256") != protocol_sha256:
        raise ValueError("Theory prior belongs to a different frozen protocol")
    if prior.get("query_labels_used") or prior.get("test_score_selected") or prior.get("label_conditioned"):
        raise ValueError("Test-selected or origin-conditioned prior is forbidden")
    fit_ids = set(prior.get("fit_sample_ids", []))
    if fit_ids & set(forbidden_ids) or not fit_ids <= set(allowed_train_ids):
        raise ValueError("Theory prior used forbidden or out-of-scope records")
    if prior.get("scope") == "preregistered_external":
        if fit_ids or not prior.get("locked_before_outer_evaluation"):
            raise ValueError("External theory must be fixed before evaluation and use no study records")
    elif prior.get("scope") != "training_only":
        raise ValueError("Unknown prior-selection scope")
    if prior.get("calculation_status") != "COMPLETED" or not prior.get("source_hash"):
        raise ValueError("Unrun/unattributed calculation cannot become model input")
    return {"status":"limited_provenance_checks_passed","scope":prior["scope"],
            "absolute_zero_leakage_proven":False,"experimental_mechanism_proven":False}


def water_smoke(output, site):
    output, site = Path(output), Path(site)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh smoke-test directory")
    output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(site.resolve()))
    from pyscf import gto, dft, lib
    import pyscf
    import h5py
    lib.num_threads(1)
    source = ROOT / "examples/gaussian/water.xyz"
    lines = source.read_text().splitlines()
    atom_text = "\n".join(lines[2:])
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    mol = gto.M(atom=atom_text,unit="Angstrom",basis="sto-3g",charge=0,spin=0,
                max_memory=1000,verbose=4,output=str(output / "pyscf-water.log"))
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.chkfile = str(output / "water.chk")
    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("Technical SCF did not converge; no success claimed")
    gradient = mf.nuc_grad_method().kernel()
    hessian = mf.Hessian().kernel()
    if not np.isfinite([energy]).all() or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise RuntimeError("Nonfinite technical outputs")
    np.savez_compressed(output / "water_derivatives.npz",gradient=gradient,hessian=hessian,
                        geometry_angstrom=mol.atom_coords(unit="Angstrom"))
    result = {"status":"COMPLETED_TECHNICAL_WATER_DFT_SMOKE_ONLY","started_at_utc":started,
              "completed_at_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.perf_counter()-before,
              "python":platform.python_version(),"pyscf":pyscf.__version__,"h5py":h5py.__version__,
              "numpy":np.__version__,"pyscf_import_path":pyscf.__file__,"h5py_import_path":h5py.__file__,
              "numpy_import_path":np.__file__,"threads":lib.num_threads(),"basis":"STO-3G","xc":"PBE",
              "grid_level":3,"charge":0,"multiplicity":1,"electrons":mol.nelectron,"atoms":mol.natm,
              "basis_functions":mol.nao_nr(),"scf_converged":bool(mf.converged),"energy_hartree":energy,
              "gradient_hartree_per_bohr":gradient.tolist(),"gradient_norm":float(np.linalg.norm(gradient)),
              "hessian_shape":list(hessian.shape),
              "hessian_max_symmetry_difference":float(np.max(np.abs(hessian-hessian.transpose(1,0,3,2)))),
              "geometry_source":str(source.relative_to(ROOT)),"geometry_source_sha256":sha(source),
              "geometry_optimized":False,"frequency_validation_performed":False,"herbal_candidate_calculations":0,
              "Gaussian_engine_calculations":0,"added_real_samples":0,"candidate_peak_assignment":None,
              "script_sha256":sha(__file__),"argv":sys.argv}
    (output / "result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    smoke = commands.add_parser("water-smoke")
    smoke.add_argument("--output",type=Path,required=True)
    smoke.add_argument("--package-site",type=Path,default=ROOT / ".runtime/quantum-upgrade-site")
    parse = commands.add_parser("parse-log")
    parse.add_argument("--input",type=Path,required=True)
    parse.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "water-smoke":
        water_smoke(args.output,args.package_site)
    else:
        if args.output.exists():
            raise FileExistsError("Do not overwrite prior parsed evidence")
        value = parse_gaussian_frequency_log(args.input.read_text(errors="replace"))
        value["source_log_sha256"] = sha(args.input)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
