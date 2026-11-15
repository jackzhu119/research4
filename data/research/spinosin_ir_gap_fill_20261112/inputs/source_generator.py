"""Generate explicit modeled conformers from the independently verified Spinosin graph.

No quantum calculation, experimental geometry, or global-minimum claim is made.
RDKit is loaded only from the caller's Python environment / explicitly chosen site.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np


SOURCE_SHA256 = "9855fb57cac932c221ac561884562e38a97688b143bf8dbb7c5ad09333eea079"
FORMULA = "C28H32O15"
KEY = "VGGSULWDCMWZPO-ODEMIOGVSA-N"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def rdkit_modules():
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdMolDescriptors
    return Chem, AllChem, rdMolDescriptors, rdBase


def verify_source(record):
    Chem, _, descriptor, _ = rdkit_modules()
    if record.get("pubchem_cid") != 155692 or record.get("formula") != FORMULA:
        raise ValueError("Verified Spinosin CID155692 / C28H32O15 source required")
    if record.get("source_inchikey") != KEY or not record.get("primary_source", "").startswith("https://"):
        raise ValueError("Verified source identity / provenance required")
    molecule = Chem.MolFromSmiles(record["source_isomeric_smiles"])
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("A valid single connected graph is required")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True, useLegacyImplementation=False)
    if len(centers) != 10 or any(label == "?" for _, label in centers):
        raise ValueError("All ten source tetrahedral stereocenters must be specified")
    key = Chem.MolToInchiKey(molecule)
    if key != KEY or descriptor.CalcMolFormula(molecule) != FORMULA:
        raise ValueError("Toolkit formula / stereochemical InChIKey differs from trusted record")
    molecule = Chem.AddHs(molecule)
    composition = dict(sorted(Counter(atom.GetSymbol() for atom in molecule.GetAtoms()).items()))
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    radicals = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    electrons = sum(atom.GetAtomicNum() for atom in molecule.GetAtoms()) - charge
    if composition != {"C": 28, "H": 32, "O": 15} or molecule.GetNumAtoms() != 75:
        raise ValueError("Complete neutral Spinosin must contain 75 atoms including explicit H")
    if charge != 0 or radicals != 0 or electrons != 320:
        raise ValueError("Neutral, zero-radical, 320-electron graph required")
    return molecule, {"formula": descriptor.CalcMolFormula(molecule), "composition": composition,
        "atoms": molecule.GetNumAtoms(), "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "formal_charge": charge, "radical_electrons": radicals, "electron_count": electrons,
        "spin_2S": 0, "multiplicity": 1,
        "spin_basis": "Neutral even-electron, zero-radical valence graph; closed-shell singlet selected. No spin-state search.",
        "source_stereocenters": [{"atom_index": i, "CIP": label} for i, label in centers],
        "computed_inchi": Chem.MolToInchi(molecule), "computed_inchikey": key,
        "computed_isomeric_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True),
        "rotatable_bonds_toolkit": int(descriptor.CalcNumRotatableBonds(molecule))}


def stereo_key_from_coordinates(molecule):
    Chem, _, _, _ = rdkit_modules()
    probe = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(probe)
    Chem.AssignStereochemistryFrom3D(probe, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    return Chem.MolToInchiKey(probe)


def xyz_text(molecule, comment):
    coordinates = np.asarray(molecule.GetConformer(0).GetPositions(), float)
    if coordinates.shape != (75, 3) or not np.isfinite(coordinates).all():
        raise ValueError("A finite full75-atom geometry is required")
    d = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if float(d.min()) <= .45:
        raise ValueError("Unphysical near-coincident atom positions")
    return f"75\n{comment}\n" + "\n".join(
        f"{atom.GetSymbol():2s} {p[0]: .14f} {p[1]: .14f} {p[2]: .14f}"
        for atom, p in zip(molecule.GetAtoms(), coordinates)) + "\n"


def heavy_atom_rmsd(a, b):
    """Proper-rotation Kabsch RMSD with the identical verified atom mapping."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Paired coordinate arrays required")
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=1))))


def run(source, output, source_sha, count=12, seed=20261031, max_iterations=750):
    if sha(source) != source_sha:
        raise ValueError("Source record bytes differ from reviewed SHA256")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve earlier generation attempts; choose a new directory")
    record = json.loads(Path(source).read_text())
    molecule, verification = verify_source(record)
    Chem, AllChem, _, rdBase = rdkit_modules()
    if not 2 <= count <= 12 or not 1 <= max_iterations <= 1000:
        raise ValueError("Bounded2..12 conformers and<=1000 FF iterations required")
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("MMFF94s parameters missing: do not silently substitute another FF")
    output.mkdir(parents=True)
    start, cpu_start = time.perf_counter(), time.process_time()
    parameters = {"started_at_utc": utc(), "method": "ETKDGv3", "num_conformers_requested": count,
        "base_seed": seed, "seed_rule": "base_seed + candidate_index", "num_threads": 1,
        "enforce_chirality": True, "embedding_max_iterations": 300,
        "forcefield": "MMFF94s", "forcefield_max_iterations": max_iterations,
        "forcefield_force_tolerance": .001, "forcefield_energy_tolerance": .000001,
        "energy_unit": "kcal/mol; MMFF force-field energy, NOT DFT electronic energy",
        "selection": "Lowest MMFF94s energy among valid converged screened conformers; not global minimum",
        "source_record_path": str(source), "source_record_sha256": source_sha,
        "RDKit": rdBase.rdkitVersion, "numpy": np.__version__,
        "coordinates_provenance": "Algorithm-generated computational conformers from verified full stereochemical connectivity; not official, measured or PubChem3D coordinates"}
    write(output / "parameters.json", parameters)
    write(output / "source_graph_verification.json", verification)
    rows, geometries = [], {}
    for index in range(count):
        current = Chem.Mol(molecule)
        params = AllChem.ETKDGv3()
        params.randomSeed, params.numThreads, params.enforceChirality = seed + index, 1, True
        params.maxIterations = 300
        if hasattr(params, "timeout"):
            params.timeout = 5
        row = {"candidate": f"C{index:03d}", "seed": seed + index, "embedding_status": None,
            "MMFF_status": None, "MMFF_energy_kcal_mol": None, "stereo_inchikey": None, "eligible_for_DFT": False}
        try:
            result = int(AllChem.EmbedMolecule(current, params))
            row["embedding_status"] = result
            if result < 0:
                raise ValueError("ETKDG embedding failed")
            pre_key = stereo_key_from_coordinates(current)
            if pre_key != KEY:
                raise ValueError("Generated coordinates do not reproduce trusted stereochemistry")
            (output / f"{row['candidate']}_embedded.xyz").write_text(xyz_text(current, "Spinosin generated ETKDGv3 model; not measured coordinates"))
            properties = AllChem.MMFFGetMoleculeProperties(current, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(current, properties, confId=0)
            row["MMFF_initial_energy_kcal_mol"] = float(ff.CalcEnergy())
            row["MMFF_status"] = int(ff.Minimize(maxIts=max_iterations, forceTol=.001, energyTol=.000001))
            row["MMFF_energy_kcal_mol"] = float(ff.CalcEnergy())
            row["stereo_inchikey"] = stereo_key_from_coordinates(current)
            if row["stereo_inchikey"] != KEY:
                raise ValueError("Force-field geometry changed stereochemistry")
            xyz = output / f"{row['candidate']}_mmff94s.xyz"
            xyz.write_text(xyz_text(current, f"Spinosin modeled ETKDGv3/MMFF94s {row['candidate']}; charge0 multiplicity1; not a DFT result"))
            sdf = output / f"{row['candidate']}_mmff94s.sdf"
            current.SetProp("_Name", "Spinosin_" + row["candidate"])
            for key, value in {"SOURCE_RECORD_SHA256": source_sha, "SOURCE_INCHIKEY": KEY,
                               "COORDINATE_PROVENANCE": parameters["coordinates_provenance"],
                               "MMFF94S_STATUS": str(row["MMFF_status"]),
                               "MMFF94S_ENERGY_KCAL_MOL": repr(row["MMFF_energy_kcal_mol"])}.items():
                current.SetProp(key, value)
            writer = Chem.SDWriter(str(sdf)); writer.write(current, confId=0); writer.close()
            row.update(xyz_path=str(xyz), xyz_sha256=sha(xyz), sdf_path=str(sdf), sdf_sha256=sha(sdf),
                       eligible_for_DFT=row["MMFF_status"] == 0)
            heavy = [atom.GetIdx() for atom in current.GetAtoms() if atom.GetAtomicNum() != 1]
            geometries[row["candidate"]] = np.asarray(current.GetConformer().GetPositions())[heavy].copy()
        except Exception as exc:
            row["failure"] = type(exc).__name__ + ": " + str(exc)
        rows.append(row)
        write(output / "progress.json", {"completed": len(rows), "requested": count, "rows": rows,
             "cpu_seconds": time.process_time() - cpu_start, "wall_seconds": time.perf_counter() - start})
        print(json.dumps(row), flush=True)
    valid = sorted([r for r in rows if r["eligible_for_DFT"]], key=lambda r: (r["MMFF_energy_kcal_mol"], r["candidate"]))
    ranks = []
    for rank, row in enumerate(valid):
        row = dict(row, rank=rank + 1, MMFF_relative_energy_kcal_mol=row["MMFF_energy_kcal_mol"] - valid[0]["MMFF_energy_kcal_mol"])
        ranks.append(row)
    pairs = [{"candidate_a": a, "candidate_b": b, "heavy_atom_mapped_RMSD_A": heavy_atom_rmsd(geometries[a], geometries[b])}
             for i, a in enumerate(geometries) for b in list(geometries)[i + 1:]]
    write(output / "heavy_atom_pair_RMSD.json", pairs)
    result = {"status": "DONE" if len(valid) >= 2 and len(rows) == count else "PARTIAL" if valid else "BLOCKED",
        "started_at_utc": parameters.get("started_at_utc"), "completed_at_utc": utc(),
        "requested_conformers": count, "attempted": len(rows), "eligible_converged_conformers": len(valid),
        "all_rows": rows, "ranked_conformers": ranks, "selected_for_initial_DFT": ranks[0] if ranks else None,
        "geometry_status": "GENERATED_FROM_VERIFIED_STEREOCHEMICAL_2D_NOT_MEASURED",
        "global_conformer_minimum_established": False, "DFT_calculations": 0, "IR_intensities": "NOT_COMPUTED",
        "cpu_seconds": time.process_time() - cpu_start, "wall_seconds": time.perf_counter() - start}
    write(output / "conformer_summary.json", result)
    if ranks:
        keys = ["rank", "candidate", "seed", "MMFF_status", "MMFF_energy_kcal_mol", "MMFF_relative_energy_kcal_mol", "stereo_inchikey", "xyz_path", "xyz_sha256", "sdf_path", "sdf_sha256"]
        with (output / "conformer_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, keys, extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(ranks)
    write(output / "files_sha256.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "files_sha256.json"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--max-iterations", type=int, default=750)
    parser.add_argument("--cpu-budget-seconds", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.cpu_budget_seconds <= 60:
        parser.error("This conformer stage permits at most60 CPU seconds")
    if args.site:
        sys.path.insert(0, str(args.site.resolve()))
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_budget_seconds, args.cpu_budget_seconds + 5))
    result = run(args.source, args.output, args.source_sha256, args.count, args.seed, args.max_iterations)
    print(json.dumps({k: result[k] for k in ("status", "eligible_converged_conformers", "selected_for_initial_DFT", "cpu_seconds", "wall_seconds")}))


if __name__ == "__main__":
    main()
