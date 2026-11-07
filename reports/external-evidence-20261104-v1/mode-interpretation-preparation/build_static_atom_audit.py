"""Reconstruct C001 graph labels from static SDF/XYZ, without QM or Hessian I/O."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import datetime
import hashlib
import json
import math

from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_xyz(path):
    lines = path.read_text().splitlines()
    rows = [line.split() for line in lines[2:] if line.strip()]
    assert int(lines[0]) == len(rows) == 75
    return [r[0] for r in rows], [[float(x) for x in r[1:]] for r in rows]


def signature(mol):
    atoms = [(a.GetSymbol(), a.GetFormalCharge(), str(a.GetChiralTag()))
             for a in mol.GetAtoms()]
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
                   for b in mol.GetBonds())
    return atoms, bonds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    paths = {
        "trusted_source": "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json",
        "C001_source_sdf": "reports/pyscf-spinosin-pilot-20261031/conformers/C001_mmff94s.sdf",
        "C001_source_xyz": "reports/pyscf-spinosin-pilot-20261031/conformers/C001_mmff94s.xyz",
        "v3_optimized_xyz": "reports/pyscf-spinosin-continuation-20261031-v3/spinosin-native4-optimization-001/optimized.xyz",
        "v5_prepared_xyz": "reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/optimized.xyz",
        "v5_input_json": "reports/pyscf-spinosin-resumable-20261104-v5/v5-spinosin-C001-prepare-001/input.json",
        "user_uploaded_sdf": "data/incoming/molecular_structures_20261103_v1/枣仁CID_155692.sdf",
        "generator_script": "scripts/prepare_spinosin_conformers_20261031.py",
        "state_worker_script": "scripts/pyscf_molecular_state_worker_20261104_v5.py",
    }
    # Explicit allowlist. No globbing into live Hessian/cache/checkpoint directories.
    source = json.loads((repo / paths["trusted_source"]).read_text())
    expected_key = source["source_inchikey"]
    trusted = Chem.AddHs(Chem.MolFromSmiles(source["source_isomeric_smiles"]))
    c001 = next(m for m in Chem.SDMolSupplier(str(repo / paths["C001_source_sdf"]),
                                            removeHs=False) if m is not None)
    upload = next(m for m in Chem.SDMolSupplier(str(repo / paths["user_uploaded_sdf"]),
                                              removeHs=False) if m is not None)
    assert c001.GetNumAtoms() == trusted.GetNumAtoms() == upload.GetNumAtoms() == 75
    assert signature(trusted) == signature(c001)
    assert Chem.MolToInchiKey(trusted) == Chem.MolToInchiKey(c001) == expected_key
    assert Chem.MolToInchiKey(upload) == expected_key
    xyzs = {k: load_xyz(repo / paths[k]) for k in
            ("C001_source_xyz", "v3_optimized_xyz", "v5_prepared_xyz")}
    symbols = [a.GetSymbol() for a in c001.GetAtoms()]
    assert all(data[0] == symbols for data in xyzs.values())
    v3, current = xyzs["v3_optimized_xyz"][1], xyzs["v5_prepared_xyz"][1]
    coord_delta = max(abs(a - b) for rowa, rowb in zip(v3, current)
                      for a, b in zip(rowa, rowb))
    assert coord_delta == 0
    manifest = json.loads((repo / paths["v5_input_json"]).read_text())
    assert manifest["molecular_input"]["atom_symbols"] == symbols
    probe = Chem.Mol(c001)
    probe.RemoveAllConformers()
    conf = Chem.Conformer(75)
    for i, xyz in enumerate(current):
        conf.SetAtomPosition(i, Point3D(*xyz))
    probe.AddConformer(conf, assignId=True)
    Chem.RemoveStereochemistry(probe)
    Chem.AssignStereochemistryFrom3D(probe, replaceExistingTags=True)
    current_key = Chem.MolToInchiKey(probe)
    assert current_key == expected_key
    bond_lengths = [math.dist(current[b.GetBeginAtomIdx()], current[b.GetEndAtomIdx()])
                    for b in c001.GetBonds()]
    assert min(bond_lengths) > 0.8 and max(bond_lengths) < 1.9
    ch, uh = Chem.RemoveHs(c001), Chem.RemoveHs(upload)
    heavy_matches = uh.GetSubstructMatches(ch, uniquify=False, useChirality=True,
                                          maxMatches=10001)
    all_matches = upload.GetSubstructMatches(c001, uniquify=False, useChirality=True,
                                            maxMatches=10001)
    assert len(heavy_matches) == 2 and len(all_matches) == 48
    partitions = [
        ("flavone_aglycone_with_methoxy_and_phenolic_OH", list(range(21)), "黄酮苷元区，含甲氧基和酚羟基"),
        ("C_linked_sugar_region", list(range(21, 31)), "与苷元C连接的糖基区，不含连接两糖的O31"),
        ("inter_sugar_oxygen_bridge", [31], "两糖之间的氧桥"),
        ("O_linked_terminal_sugar_region", list(range(32, 43)), "通过O31连接的末端糖基区"),
    ]
    groups, atom_partition = [], {}
    for name, heavy, zh in partitions:
        hydrogens = sorted(n.GetIdx() for i in heavy for n in c001.GetAtomWithIdx(i).GetNeighbors()
                           if n.GetSymbol() == "H")
        indices = sorted(heavy + hydrogens)
        for i in indices:
            assert i not in atom_partition
            atom_partition[i] = name
        groups.append({"group_id": name, "family": "DISJOINT_PARTITION", "description_zh": zh,
                       "indices_zero_based": indices, "indices_one_based": [i + 1 for i in indices],
                       "heavy_indices_zero_based": heavy, "H_indices_zero_based": hydrogens,
                       "composition": dict(Counter(c001.GetAtomWithIdx(i).GetSymbol() for i in indices))})
    assert sorted(atom_partition) == list(range(75))
    details = [
        ("flavone_A_ring", [2, 3, 4, 17, 18, 20], "苷元稠合苯环A"),
        ("flavone_heterocyclic_ring", [4, 5, 6, 14, 15, 17], "苷元含氧杂环"),
        ("flavone_B_phenyl_ring", [7, 8, 9, 10, 12, 13], "苷元侧苯环B"),
        ("methoxy", [0, 1, 43, 44, 45], "甲氧基"),
        ("carbonyl_bond", [15, 16], "羰基C=O"),
        ("aglycone_sugar_C_C_link", [20, 21], "苷元与糖的C—C连接"),
        ("inter_sugar_C_O_C_link", [30, 31, 32], "糖间C—O—C连接"),
        ("C_linked_sugar_ring", [21, 22, 23, 26, 28, 30], "近端糖环"),
        ("O_linked_terminal_sugar_ring", [32, 33, 34, 37, 39, 41], "末端糖环"),
    ]
    oh_pairs = []
    for atom in c001.GetAtoms():
        if atom.GetSymbol() != "O":
            continue
        hydrogen = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
        carbon = [n for n in atom.GetNeighbors() if n.GetSymbol() == "C"]
        if len(hydrogen) == len(carbon) == 1:
            kind = "phenolic_OH" if carbon[0].GetIsAromatic() else "sugar_alcohol_OH"
            oh_pairs.append({"O_index_zero_based": atom.GetIdx(), "H_index_zero_based": hydrogen[0],
                             "bonded_C_index_zero_based": carbon[0].GetIdx(), "type": kind})
            details.append((f"{kind}_O{atom.GetIdx()}", [atom.GetIdx(), hydrogen[0]], kind))
    assert len(oh_pairs) == 9
    for name, indices, zh in details:
        groups.append({"group_id": name, "family": "OVERLAPPING_GRAPH_SUBGROUP",
                       "description_zh": zh, "indices_zero_based": indices,
                       "indices_one_based": [i + 1 for i in indices]})
    output_names = ["atom_order_audit.json", "atom_groups_C001.json", "C001_atom_map.csv",
                    "uploaded_SDF_mapping_ambiguity.json"]
    out.mkdir(parents=True, exist_ok=True)
    assert not any((out / name).exists() for name in output_names), "Refuse to overwrite audit"
    def save(name, document):
        (out / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    audit = {
        "schema": "research4.static-C001-atom-order-audit.v1",
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS_C001_LOCAL_ATOM_ORDER_NOT_UNIQUE_EXTERNAL_SDF_MAP",
        "RDKit_version": rdBase.rdkitVersion, "atom_count": 75,
        "source_and_C001_indexed_graph_signatures_equal": True,
        "trusted_source_and_C001_and_upload_inchikey": expected_key,
        "current_prepared_geometry_3D_stereokey": current_key,
        "C001_and_v3_and_v5_symbol_sequences_equal": True,
        "v3_to_v5_coordinate_max_abs_delta_angstrom": coord_delta,
        "C001_original_bonds_in_current_geometry_min_max_angstrom": [min(bond_lengths), max(bond_lengths)],
        "source_C001_SDF_XYZ_rounding_max_abs_delta_angstrom": max(
            abs(c001.GetConformer().GetAtomPosition(i)[j] - xyzs["C001_source_xyz"][1][i][j])
            for i in range(75) for j in range(3)),
        "C001_to_current_zero_based_index_map": list(range(75)),
        "inputs": {key: {"path": value, "sha256": sha(repo / value)} for key, value in paths.items()},
        "read_active_Hessian_NPY_or_CHK": False, "quantum_jobs_executed": 0,
        "modes_or_frequencies_read": False, "chemical_peak_assignment_established": False,
        "cautions": ["Matching chemical identity does not by itself prove atom order; the original C001 generation and coordinate-preserving lineage provide the local order anchor.",
                     "Identical element sequences alone cannot detect swaps of equal elements; indexed graph, script lineage and 3D stereochemistry checks are complementary.",
                     "This graph annotation is not a frequency/mode result, peak assignment, experimental identification or a global conformer claim."]}
    save("atom_order_audit.json", audit)
    save("atom_groups_C001.json", {
        "schema": "research4.C001.graph-functional-groups.v1", "atom_order_anchor": paths["C001_source_sdf"],
        "current_geometry_anchor": paths["v5_prepared_xyz"], "atom_order": "C001 SDF/PySCF input; zero-based and one-based both explicit",
        "groups": groups, "hydroxyl_pairs": oh_pairs,
        "partition_coverage_75_atoms_exactly_once": True,
        "overlapping_subgroups_must_not_be_summed_as_disjoint": True,
        "atom_numbers_are_file_indices_not_IUPAC_positions": True,
        "mode_interpretation_not_executed": True,
        "molecular_intra_mode_group_labels_not_herbal_peak_chemical_assignment": True})
    save("uploaded_SDF_mapping_ambiguity.json", {
        "uploaded_SDF": paths["user_uploaded_sdf"], "C001_source_SDF": paths["C001_source_sdf"],
        "same_verified_full_stereo_identity": True,
        "heavy_graph_mappings_with_chirality_and_no_uniquification": len(heavy_matches),
        "all_75_graph_mappings_with_chirality_and_no_uniquification": len(all_matches),
        "enumeration_capped": False,
        "heavy_candidate_maps_C001_index_to_uploaded_index_zero_based": [list(m) for m in heavy_matches],
        "unique_historical_C001_to_uploaded_75_atom_mapping_established": False,
        "arbitrary_first_match_selected_for_mode_labels": False,
        "explanation": "Equivalent phenyl paths give two heavy-atom graph mappings; equivalent attached hydrogens multiply possible full maps. Do not infer a unique atom history or use user-SDF indices directly for C001 modes. C001 came from the trusted stereo SMILES, independently of the later SDF upload."})
    fields = ["C001_SDF_index_zero_based", "C001_SDF_atom_number_one_based", "current_PySCF_index_zero_based",
              "current_XYZ_atom_number_one_based", "element", "disjoint_partition", "overlapping_subgroups"]
    with (out / "C001_atom_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, atom in enumerate(c001.GetAtoms()):
            writer.writerow(dict(zip(fields, [i, i + 1, i, i + 1, atom.GetSymbol(), atom_partition[i],
                                  ";".join(g["group_id"] for g in groups if g["family"] == "OVERLAPPING_GRAPH_SUBGROUP"
                                           and i in g["indices_zero_based"])])))
    print(json.dumps({"status": audit["status"], "group_count": len(groups), "OH_count": len(oh_pairs),
                      "uploaded_heavy_graph_maps": len(heavy_matches), "uploaded_all_graph_maps": len(all_matches),
                      "quantum_or_Hessian_io": False}))


if __name__ == "__main__":
    main()
