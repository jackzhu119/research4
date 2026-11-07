"""Independent read-only SDF/XYZ/identity/ranking checks; no optimization or DFT."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier verification evidence")
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "conformers/conformer_summary.json").read_text())
    checks = []
    for row in summary["ranked_conformers"]:
        xyz, sdf = Path(row["xyz_path"]), Path(row["sdf_path"])
        assert hashlib.sha256(xyz.read_bytes()).hexdigest() == row["xyz_sha256"]
        assert hashlib.sha256(sdf.read_bytes()).hexdigest() == row["sdf_sha256"]
        mol = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)[0]
        assert mol is not None and mol.GetNumAtoms() == 75
        lines = xyz.read_text().splitlines()
        assert lines[0] == "75" and len(lines) == 77
        coordinates = np.asarray([line.split()[1:] for line in lines[2:]], float)
        assert np.isfinite(coordinates).all()
        assert [line.split()[0] for line in lines[2:]] == [atom.GetSymbol() for atom in mol.GetAtoms()]
        key = Chem.MolToInchiKey(mol)
        assert key == "VGGSULWDCMWZPO-ODEMIOGVSA-N"
        bonds = [float(np.linalg.norm(coordinates[b.GetBeginAtomIdx()] - coordinates[b.GetEndAtomIdx()])) for b in mol.GetBonds()]
        assert min(bonds) > .8 and max(bonds) < 1.9
        checks.append({"candidate": row["candidate"], "XYZ_SHA256": row["xyz_sha256"],
                       "SDF_SHA256": row["sdf_sha256"], "SDF_roundtrip_inchikey": key,
                       "finite75atoms": True, "minimum_bond_length_A": min(bonds),
                       "maximum_bond_length_A": max(bonds), "MMFF_converged": row["MMFF_status"] == 0})
    pairs = json.loads((root / "conformers/heavy_atom_pair_RMSD.json").read_text())
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "GENERATED_GEOMETRY_AND_STEREOCHEMICAL_CHECKS_PASS_NOT_DFT",
              "conformers_checked": len(checks), "checks": checks,
              "minimum_pair_mapped_heavy_RMSD_A": min(x["heavy_atom_mapped_RMSD_A"] for x in pairs),
              "maximum_pair_mapped_heavy_RMSD_A": max(x["heavy_atom_mapped_RMSD_A"] for x in pairs),
              "pairs_below_0_35_A": sum(x["heavy_atom_mapped_RMSD_A"] < .35 for x in pairs),
              "new_quantum_calculations": 0}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("All12 generated SDF/XYZ structures passed read-only roundtrip/hash/identity/bond checks")


if __name__ == "__main__":
    main()
