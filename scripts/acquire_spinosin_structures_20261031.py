"""Record ordinary official structure-download attempts; never bypass network policy.

The connected-web ChEBI/PubChem snapshots are separately archived evidence.
This command does not invent 3D coordinates or run quantum chemistry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ENDPOINTS = {
    "pubchem_spinosin_3d": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/155692/SDF?record_type=3d",
    "chebi_spinosin_molfile": "https://www.ebi.ac.uk/chebi/backend/api/public/molfile/81360",
    "pubchem_spinosin_props": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/155692/property/MolecularFormula,IsomericSMILES,InChI,InChIKey,Charge/JSON",
}


def verify_graphs(source, output):
    """Rebuild three database graphs and verify identity without 3D or DFT."""
    from collections import Counter
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdMolDescriptors
    results = []
    inputs = [("spinosin", "spinosin_verified_2d.json"),
              ("jujuboside_a", "jujuboside_a_source_record.json"),
              ("jujuboside_b", "jujuboside_b_source_record.json")]
    for name, filename in inputs:
        path = source / filename
        record = json.loads(path.read_text())
        molecule = Chem.MolFromSmiles(record["source_isomeric_smiles"])
        source_graph = Chem.MolFromInchi(record["source_inchi"])
        if molecule is None or source_graph is None:
            raise ValueError("Invalid authoritative SMILES/InChI: " + name)
        Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
        full = Chem.AddHs(molecule)
        composition = dict(Counter(atom.GetSymbol() for atom in full.GetAtoms()))
        key = Chem.MolToInchiKey(molecule)
        if key != record["source_inchikey"]:
            raise ValueError("Source InChIKey mismatch: " + name)
        if rdMolDescriptors.CalcMolFormula(molecule) != record["formula"]:
            raise ValueError("Formula mismatch: " + name)
        if Chem.MolToSmiles(molecule, isomericSmiles=True) != Chem.MolToSmiles(source_graph, isomericSmiles=True):
            raise ValueError("Source SMILES and InChI stereochemical graphs differ: " + name)
        if composition != record.get("composition", record.get("expected_composition")):
            raise ValueError("Atomic composition mismatch: " + name)
        centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True, useLegacyImplementation=False)
        if any(label == "?" for _, label in centers):
            raise ValueError("Selected record has unresolved stereocenters: " + name)
        charge = Chem.GetFormalCharge(molecule)
        radicals = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
        if charge or radicals:
            raise ValueError("Unexpected charge/radical state: " + name)
        Chem.MolToMolFile(molecule, str(output / (name + "_connectivity.mol")), includeStereo=True)
        results.append({"compound": name, "source_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "formula": rdMolDescriptors.CalcMolFormula(molecule), "composition": composition,
                        "total_atoms_with_hydrogens": full.GetNumAtoms(), "heavy_atoms": molecule.GetNumHeavyAtoms(),
                        "neutral_electrons": sum(atom.GetAtomicNum() for atom in full.GetAtoms()),
                        "formal_charge": charge, "radical_electrons": radicals,
                        "closed_shell_model_multiplicity": 1, "rdkit_inchikey": key,
                        "canonical_isomeric_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                        "stereocenters_atom_indices_CIP": centers, "assigned_stereocenters": len(centers),
                        "unassigned_stereocenters": 0,
                        "rotatable_bonds_strict": rdMolDescriptors.CalcNumRotatableBonds(molecule, rdMolDescriptors.NumRotatableBondsOptions.Strict),
                        "MMFF_all_parameters_available": bool(AllChem.MMFFHasAllMoleculeParams(full)),
                        "fragments": len(Chem.GetMolFrags(molecule)),
                        "molfile_coordinate_type": "RDKit 2D depiction only; NOT a 3D conformer or DFT geometry"})
    partial = json.loads((source / "jujuboside_a_source_record.json").read_text())["source_inchi"]
    partial = partial.replace("51-,52-", "51?,52-").replace("57-,58-", "57?,58-")
    molecule = Chem.MolFromInchi(partial)
    unknown = [item for item in Chem.FindMolChiralCenters(molecule, includeUnassigned=True, useLegacyImplementation=False) if item[1] == "?"]
    partial_key = Chem.MolToInchiKey(molecule)
    if partial_key != "KVKRFLVYJLIZFD-LQRLABOESA-N" or len(unknown) != 2:
        raise ValueError("Partial PubChem171446 comparison failed")
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "rdkit_version": rdBase.rdkitVersion,
              "status": "SELECTED_DATABASE_GRAPH_IDENTITY_VERIFIED", "molecules": results,
              "partial_jujuboside_a": {"cid": 171446, "source_inchi": partial, "inchikey": partial_key,
                                      "unassigned_centers": unknown, "do_not_equate_with_full_51346169": True},
              "quantum_computations": 0, "geometries_3d_generated": 0,
              "limitations": ["Database identity only; no current855sample constituent or standard assay verified",
                              "Singlet is a neutral fully-valent nonradical model assignment, not a spin-state energy search",
                              "MMFF parameter availability is not a conformer or force-field convergence result"]}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified_graphs": len(results), "quantum_computations": 0,
                      "atoms": [row["total_atoms_with_hydrogens"] for row in results]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--verify-graphs", action="store_true")
    parser.add_argument("--structures", type=Path,
                        default=Path("reports/pyscf-spinosin-pilot-20261031/structures"))
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("Timeout must be positive and at most 60 seconds per endpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.verify_graphs:
        verify_graphs(args.structures, args.output)
        return
    receipts = []
    for name, url in ENDPOINTS.items():
        start = time.monotonic()
        record = {"name": name, "url": url,
                  "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
                  "network_policy": "Inherited mandatory proxy unchanged; no destination bypass"}
        try:
            with urllib.request.urlopen(url, timeout=args.timeout) as response:
                content = response.read()
                path = args.output / (name + ".raw")
                path.write_bytes(content)
                record.update(status="DOWNLOADED_UNVALIDATED", http_status=response.status,
                              content_type=response.headers.get("Content-Type"),
                              bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                              path=str(path))
        except Exception as exc:
            record.update(status="BLOCKED_OR_FAILED", error_type=type(exc).__name__, error=str(exc))
        record["elapsed_seconds"] = time.monotonic() - start
        receipts.append(record)
    (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps({"attempts": len(receipts),
                      "downloaded_unvalidated": sum(row["status"] == "DOWNLOADED_UNVALIDATED" for row in receipts),
                      "quantum_computations": 0}))


if __name__ == "__main__":
    main()
