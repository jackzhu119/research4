#!/usr/bin/env python3
"""Generate a documented ethanol *software-validation* fixture, never a plant sample.

No electronic-structure or other quantum calculations are performed.
The input graph is transcribed from the official ChEBI entry and cross-checked
independently from its Standard InChI; coordinates are RDKit-generated models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_xyz(path: Path, mol, comment: str) -> None:
    pos = np.asarray(mol.GetConformer().GetPositions())
    lines = [str(mol.GetNumAtoms()), comment]
    for atom, xyz in zip(mol.GetAtoms(), pos):
        lines.append(f"{atom.GetSymbol():2s} {xyz[0]: .15f} {xyz[1]: .15f} {xyz[2]: .15f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Fail closed on accidental rerun into an already generated artifact directory.
    for name in ("ethanol_mmff94s.xyz", "provenance.json", "files_sha256.json"):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite {out / name}; use a fresh --output directory")

    start = datetime.now(timezone.utc).isoformat()
    start_wall = time.monotonic()
    smiles = "CCO"
    source_inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    source_key = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    from_smiles = Chem.MolFromSmiles(smiles)
    from_inchi = Chem.MolFromInchi(source_inchi)
    assert from_smiles is not None and from_inchi is not None
    can_smiles = Chem.MolToSmiles(from_smiles, isomericSmiles=True)
    assert Chem.MolToSmiles(from_inchi, isomericSmiles=True) == can_smiles == smiles
    assert Chem.MolToInchi(from_smiles) == source_inchi
    assert Chem.MolToInchiKey(from_smiles) == source_key
    assert rdMolDescriptors.CalcMolFormula(from_smiles) == "C2H6O"
    assert not Chem.FindMolChiralCenters(from_smiles, includeUnassigned=True)

    mol = Chem.AddHs(from_smiles)
    composition = dict(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms()).items()))
    electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - Chem.GetFormalCharge(mol)
    radical_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    assert composition == {"C": 2, "H": 6, "O": 1}
    assert mol.GetNumAtoms() == 9 and electrons == 26
    assert Chem.GetFormalCharge(mol) == 0 and radical_electrons == 0
    assert len(Chem.GetMolFrags(mol)) == 1

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 20261031
    parameters.numThreads = 1
    parameters.useRandomCoords = False
    embed_status = int(AllChem.EmbedMolecule(mol, parameters))
    if embed_status != 0:
        raise RuntimeError(f"ETKDGv3 embedding failed: {embed_status}")
    write_xyz(out / "ethanol_etkdgv3.xyz", mol, "RDKit ETKDGv3 generated; seed=20261031; NOT experimental coordinates")
    assert AllChem.MMFFHasAllMoleculeParams(mol)
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    ff = AllChem.MMFFGetMoleculeForceField(mol, properties)
    initial_energy = float(ff.CalcEnergy())
    status = int(ff.Minimize(maxIts=750, forceTol=1e-4, energyTol=1e-6))
    final_energy = float(ff.CalcEnergy())
    if status != 0:
        raise RuntimeError(f"MMFF94s did not converge, status={status}")
    coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    assert coordinates.shape == (9, 3) and np.isfinite(coordinates).all()
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    offdiag = distances[np.triu_indices(9, 1)]
    assert float(offdiag.min()) > 0.6
    assert Chem.MolToInchiKey(Chem.RemoveHs(mol)) == source_key
    assert np.isfinite(final_energy) and final_energy <= initial_energy

    write_xyz(out / "ethanol_mmff94s.xyz", mol, "RDKit ETKDGv3 + MMFF94s generated validation model; Angstrom; NOT experimental or official downloaded 3D")
    np.save(out / "ethanol_coordinates_angstrom.npy", coordinates)
    mol.SetProp("_Name", "Ethanol: software/numerical-validation fixture only")
    mol.SetProp("source_2d", "https://www.ebi.ac.uk/chebi/CHEBI:16236")
    mol.SetProp("geometry_origin", "RDKit ETKDGv3 randomSeed20261031 + MMFF94s; NOT experimental")
    mol.SetProp("usage_scope", "Software and numerical Hessian validation only; not an acid-jujube constituent claim or independent plant sample")
    mol.SetProp("mmff94s_energy_kcal_per_mol", repr(final_energy))
    writer = Chem.SDWriter(str(out / "ethanol_mmff94s.sdf"))
    writer.write(mol)
    writer.close()
    atom_metadata = []
    for atom, xyz in zip(mol.GetAtoms(), coordinates):
        atom_metadata.append({"index_zero_based": atom.GetIdx(), "element": atom.GetSymbol(), "atomic_number": atom.GetAtomicNum(), "xyz_angstrom": xyz.tolist()})
    bonds = [{"atom_indices_zero_based": [b.GetBeginAtomIdx(), b.GetEndAtomIdx()], "bond_order": b.GetBondTypeAsDouble(), "length_angstrom": float(distances[b.GetBeginAtomIdx(), b.GetEndAtomIdx()])} for b in mol.GetBonds()]
    write_json(out / "atom_order_and_bonds.json", {"atoms": atom_metadata, "bonds": bonds})

    source_snapshots = {}
    for name in ("official_structure_web_snapshot.txt", "source_license_and_access_web_snapshot.txt"):
        if (out / name).exists():
            source_snapshots[name] = sha256(out / name)
    identity = {
        "molecule": "ethanol", "primary_source_id": "CHEBI:16236",
        "primary_source_url": "https://www.ebi.ac.uk/chebi/CHEBI:16236",
        "source_license": "CC BY 4.0", "license_source_url": "https://www.ebi.ac.uk/chebi/about",
        "pubchem_cid_reference": 702,
        "pubchem_access_status": "Compound web page returned reCAPTCHA; PUG REST URL was inaccessible via web connector. Identity is verified from official ChEBI, not a successfully retrieved PubChem record.",
        "formula": "C2H6O", "smiles": smiles, "standard_inchi": source_inchi, "inchikey": source_key,
        "charge": 0, "multiplicity": 1, "explicit_atom_count": 9,
        "electron_count": electrons, "radical_electrons": radical_electrons,
        "composition": composition,
        "checks": {"smiles_inchi_graph_agreement": True, "formula_matches_source": True, "inchikey_matches_source": True, "single_connected_component": True, "all_hydrogens_explicit": True, "no_unspecified_or_defined_atom_stereocenters": True},
        "spin_state_basis": "Neutral nonradical closed-shell singlet model selected from verified molecular graph; not an electronic spin-stability calculation.",
        "source_snapshot_sha256": source_snapshots,
    }
    write_json(out / "ethanol_verified_2d.json", identity)
    elapsed = time.monotonic() - start_wall
    generation = {
        "started_utc": start, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_seconds": elapsed,
        "method": "RDKit ETKDGv3 embedding followed by MMFF94s force-field minimization",
        "random_seed": 20261031, "num_threads": 1, "conformer_count": 1,
        "use_random_coordinates": False, "embed_status": embed_status,
        "mmff_variant": "MMFF94s", "max_iterations": 750, "force_tolerance": 1e-4,
        "energy_tolerance": 1e-6, "mmff_status": status, "mmff_converged": status == 0,
        "initial_energy_kcal_per_mol": initial_energy, "final_energy_kcal_per_mol": final_energy,
        "energy_interpretation": "Force-field screening energy, NOT electronic/DFT energy",
        "minimum_interatomic_distance_angstrom": float(offdiag.min()),
        "python_version": sys.version, "platform": platform.platform(),
        "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__,
        "quantum_computation_performed": False,
    }
    write_json(out / "generation.json", generation)
    provenance = {
        "status": "DONE_STRUCTURE_PREPARATION_ONLY",
        "identity": "validation_ethanol", "validation_only": True,
        "purpose": "9-atom ethanol software/numerical Hessian-response-block validation fixture ONLY",
        "not_a_plant_or_acid_jujube_sample": True,
        "geometry_origin": "DOCUMENTED_RDKIT_CONFORMER_GENERATION",
        "geometry_is_experimental": False, "geometry_is_downloaded_official_3d": False,
        "identity_verified": True, "stereochemistry_verified": True,
        "molecule": "ethanol", "formula": "C2H6O", "atom_count": 9,
        "smiles": smiles, "standard_inchi": source_inchi, "inchikey": source_key,
        "composition": composition, "charge": 0, "multiplicity": 1,
        "electron_count": electrons, "coordinate_units": "angstrom",
        "xyz_file": "ethanol_mmff94s.xyz", "input_xyz_sha256": sha256(out / "ethanol_mmff94s.xyz"),
        "coordinates_file": "ethanol_coordinates_angstrom.npy", "coordinates_sha256": sha256(out / "ethanol_coordinates_angstrom.npy"),
        "sdf_file": "ethanol_mmff94s.sdf", "sdf_sha256": sha256(out / "ethanol_mmff94s.sdf"),
        "source_url": identity["primary_source_url"], "source_license": "CC BY 4.0",
        "source_snapshot_sha256": source_snapshots,
        "identity_record": "ethanol_verified_2d.json", "generation_record": "generation.json",
        "quantum_status": {"SCF": "NOT_RUN", "gradient": "NOT_RUN", "geometry_optimization": "NOT_RUN", "Hessian": "NOT_RUN", "frequency_analysis": "NOT_RUN", "IR_intensities": "NOT_COMPUTED"},
        "scope_warning": "Structure generation does not complete the planned PySCF full Opt/Freq or B4/B8 numerical validation. No chemical or biological conclusions follow from this fixture.",
    }
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"status": provenance["status"], "path": str(out), "mmff_converged": status == 0, "mmff_energy_kcal_per_mol": final_energy, "input_xyz_sha256": provenance["input_xyz_sha256"], "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
