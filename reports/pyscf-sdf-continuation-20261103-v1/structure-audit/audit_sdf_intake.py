"""Read-only molecular parsing and additive SDF intake. No force field or QM.

SDF property text is metadata only. No properties are executed as instructions.
The two target CIDs identify uploaded records; they do not establish provenance
of a medicinal material, independent biological samples, or measured spectra.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json"
REFERENCE_INDEX = ROOT / "reports/pyscf-spinosin-pilot-20261031/artifact_index.json"
REFERENCE_INDEX_SHA = "8fbd6a0c262a12beffb6320458713c45ae381bbd8adb76f062bea1d882ee8f5b"
OLD_OPT_XYZ = ROOT / "reports/pyscf-spinosin-continuation-20261031-v3/spinosin-native4-optimization-001/optimized.xyz"
RECORDS = [(5280448, "黄芪CID_5280448.sdf"), (155692, "枣仁CID_155692.sdf")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def trusted_spinosin_reference():
    if sha(REFERENCE_INDEX) != REFERENCE_INDEX_SHA:
        raise ValueError("The earlier immutable pilot index differs")
    index = json.loads(REFERENCE_INDEX.read_text())
    key = str(REFERENCE.relative_to(ROOT))
    if sha(REFERENCE) != index["files"][key]["sha256"]:
        raise ValueError("The earlier trusted Spinosin record differs")
    value = json.loads(REFERENCE.read_text())
    if value["pubchem_cid"] != 155692 or value["chebi_id"] != "CHEBI:81360":
        raise ValueError("Wrong trusted reference")
    return value


def geometry_metrics(mol):
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
        raise ValueError("Finite complete Cartesian coordinates required")
    table = Chem.GetPeriodicTable()
    radii = np.asarray([table.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
    bonds = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    lengths = [float(np.linalg.norm(xyz[i] - xyz[j])) for i, j in bonds]
    ratios = [float(np.linalg.norm(xyz[i] - xyz[j]) / (radii[i] + radii[j])) for i, j in bonds]
    nonbonded = [(i, j, float(np.linalg.norm(xyz[i] - xyz[j])), float(np.linalg.norm(xyz[i] - xyz[j]) / (radii[i] + radii[j])))
                 for i in range(len(xyz)) for j in range(i) if (j, i) not in bonds]
    severe = [{"atom_indices_zero_based": [i, j], "distance_angstrom": d, "covalent_radius_sum_ratio": r}
              for i, j, d, r in nonbonded if r < 0.65]
    rank = int(np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-6))
    return xyz, {
        "toolkit_conformer_is_3d": bool(mol.GetConformer().Is3D()),
        "centered_coordinate_rank": rank,
        "maximum_abs_z_angstrom": float(np.abs(xyz[:, 2]).max()),
        "minimum_bond_angstrom": min(lengths), "maximum_bond_angstrom": max(lengths),
        "minimum_bond_covalent_ratio": min(ratios), "maximum_bond_covalent_ratio": max(ratios),
        "minimum_nonbonded_distance_angstrom": min(x[2] for x in nonbonded) if nonbonded else None,
        "minimum_nonbonded_covalent_ratio": min(x[3] for x in nonbonded) if nonbonded else None,
        "severe_nonbonded_overlaps_below_0p65_covalent_sum": severe,
        "loose_bond_geometry_screen_passed": bool(min(ratios) >= 0.6 and max(ratios) <= 1.45 and not severe),
        "screen_scope": "Loose plausibility only, not optimization, gradient stationarity, or minimum verification",
    }


def analyze_sdf(path, expected_cid, reference=None):
    path = Path(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError("Exactly one successfully parsed SDF molecule required")
    mol = supplier[0]
    if not mol.HasProp("PUBCHEM_COMPOUND_CID") or mol.GetProp("PUBCHEM_COMPOUND_CID").strip() != str(expected_cid):
        raise ValueError("Uploaded CID claim differs from the requested record")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("One molecular component required")
    heavy = Chem.RemoveHs(mol)
    xyz, geometry = geometry_metrics(mol)
    # Re-perceive stereochemistry from the coordinates after discarding tags.
    from3d = Chem.Mol(mol)
    Chem.RemoveStereochemistry(from3d)
    Chem.AssignStereochemistryFrom3D(from3d, replaceExistingTags=True)
    from3d = Chem.RemoveHs(from3d)
    stereo = Chem.FindMolChiralCenters(heavy, includeUnassigned=True, useLegacyImplementation=False)
    smiles = Chem.MolToSmiles(heavy, isomericSmiles=True)
    key = Chem.MolToInchiKey(heavy)
    key3d = Chem.MolToInchiKey(from3d)
    charge = int(Chem.GetFormalCharge(mol))
    composition = dict(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms()).items()))
    properties = {name: mol.GetProp(name) for name in mol.GetPropNames()}
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    trusted_match = None
    mapping = None
    name = "CID5280448 用户上传结构（名称-CID映射待独立复核）"
    if expected_cid == 155692:
        if reference is None:
            raise ValueError("Trusted Spinosin graph reference is required")
        refmol = Chem.MolFromSmiles(reference["source_isomeric_smiles"])
        trusted_match = (key == reference["source_inchikey"] and key3d == reference["source_inchikey"]
                         and smiles == Chem.MolToSmiles(refmol, isomericSmiles=True))
        matches = heavy.GetSubstructMatches(refmol, useChirality=True, uniquify=False, maxMatches=1000)
        mapping = {"isomorphic_stereo_heavy_graph_match_count": len(matches),
                   "one_reference_to_uploaded_heavy_atom_mapping_zero_based": list(matches[0]) if matches else None,
                   "mapping_scope": "Heavy graph only, includes symmetry alternatives; not a hydrogen/AO/orbital permutation"}
        name = "Spinosin（斯皮诺素）" if trusted_match else "CID155692 claim failed trusted Spinosin comparison"
    ready = (geometry["toolkit_conformer_is_3d"] and geometry["loose_bond_geometry_screen_passed"]
             and key == key3d and not any(label == "?" for _, label in stereo)
             and charge == 0 and radicals == 0 and trusted_match is not False)
    old_symbols = [line.split()[0] for line in OLD_OPT_XYZ.read_text().splitlines()[2:]] if expected_cid == 155692 else None
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    value = {
        "claimed_pubchem_cid": expected_cid, "record_type": "molecular_structure_not_THz_sample_dataset",
        "name": name, "name_CID_mapping_independently_verified": trusted_match is True,
        "identity_evidence": "SHA-bound previously acquired official ChEBI/PubChem graph" if trusted_match is True else "User SDF structure and internally consistent CID declaration only",
        "formula_computed": rdMolDescriptors.CalcMolFormula(mol), "explicit_atom_count": mol.GetNumAtoms(),
        "heavy_atom_count": mol.GetNumHeavyAtoms(), "explicit_hydrogen_atoms": sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()),
        "implicit_hydrogen_count": sum(a.GetNumImplicitHs() for a in mol.GetAtoms()),
        "composition": composition, "formal_charge": charge, "radical_electrons": radicals,
        "electron_count_from_graph": sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge,
        "closed_shell_singlet_input_supported_by_graph": charge == 0 and radicals == 0,
        "spin_state_search_performed": False,
        "canonical_isomeric_smiles_computed": smiles, "inchi_computed": Chem.MolToInchi(heavy),
        "inchikey_computed": key, "inchikey_rebuilt_from_3d": key3d, "3d_stereo_matches_parsed_graph": key == key3d,
        "stereocenters": [{"heavy_atom_index_zero_based": i, "CIP": label} for i, label in stereo],
        "specified_stereocenter_count": sum(label != "?" for _, label in stereo),
        "unassigned_stereocenter_count": sum(label == "?" for _, label in stereo),
        "trusted_Spinosin_graph_and_stereo_match": trusted_match,
        "geometry": geometry, "coordinates_units": "Angstrom by MDL molfile convention; not measured instrumental data",
        "source_coordinate_decimal_places": 4,
        "coordinate_origin": "User-provided PubChem-style computed-conformer export; download channel not independently verified",
        "experimental_or_crystal_coordinates_verified": False,
        "PubChem_export_claims": properties,
        "inferred_record_URL_not_retrieved": f"https://pubchem.ncbi.nlm.nih.gov/compound/{expected_cid}",
        "live_official_metadata_retrieval": "NOT_RUN_NETWORK_ALLOWLIST_DOES_NOT_INCLUDE_PUBCHEM_OR_CHEBI",
        "geometric_start_candidate_eligible": bool(ready), "DFT_optimized": False,
        "DFT_input_status": "GEOMETRY_CANDIDATE_REQUIRES_NEW_OPTIMIZATION" if trusted_match is True else "GEOMETRY_CANDIDATE_NAMED_IDENTITY_PENDING",
        "vibrational_frequencies": "NOT_COMPUTED", "IR_intensities": "NOT_COMPUTED",
        "biological_sample_count_added": 0, "measured_THz_spectra_added": 0,
        "new_force_field_or_quantum_evaluations": 0,
        "old_C001_coordinate_or_label_replaced": False,
    }
    if old_symbols is not None:
        value.update(old_C001_XYZ_source=str(OLD_OPT_XYZ.relative_to(ROOT)), old_C001_XYZ_sha256=sha(OLD_OPT_XYZ),
                     uploaded_symbol_order_matches_old_C001=symbols == old_symbols,
                     old_checkpoint_AO_or_Hessian_indexwise_reuse_allowed=False,
                     atom_order_warning="Uploaded atom order differs; use a fresh SCF if using this conformer. No validated all-H/AO mapping or orbital projection is provided.",
                     heavy_graph_mapping=mapping)
    return value, symbols, xyz


def run_intake(sources, intake, report):
    intake, report = Path(intake), Path(report)
    if intake.exists():
        raise FileExistsError("Preserve the previous raw intake")
    expected_outputs = [report / x for x in ["audit.json", "summary.csv", "sources.json", "derived", "README.md"]]
    if any(p.exists() for p in expected_outputs):
        raise FileExistsError("Preserve the previous audit outputs")
    reference = trusted_spinosin_reference()
    report.mkdir(parents=True, exist_ok=True)
    intake.mkdir(parents=True, exist_ok=False)
    (report / "derived").mkdir()
    results, receipts = [], []
    for cid, filename in RECORDS:
        source = Path(sources[cid]); destination = intake / filename
        raw = source.read_bytes()
        with destination.open("xb") as handle:
            handle.write(raw)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError("Raw intake copy differs")
        receipt = {"claimed_cid": cid, "file": filename, "bytes": len(raw), "sha256": sha(destination),
                   "provided_via": "User uploaded file", "attachment_source_path": str(source),
                   "download_origin_independently_verified": False}
        value, symbols, xyz = analyze_sdf(destination, cid, reference if cid == 155692 else None)
        coords = report / "derived" / f"CID_{cid}_coordinates_angstrom.npy"
        with coords.open("xb") as handle:
            np.save(handle, xyz)
        xyzfile = report / "derived" / f"CID_{cid}.xyz"
        with xyzfile.open("x") as handle:
            handle.write(f"{len(symbols)}\nUser uploaded computed-conformer candidate CID{cid}; no optimization in this intake\n")
            handle.writelines(f"{s} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n" for s, p in zip(symbols, xyz))
        value["raw_input"] = {"path": str(destination.relative_to(ROOT)) if destination.is_relative_to(ROOT) else str(destination), "sha256": sha(destination)}
        value["derived_coordinates"] = {"NPY": str(coords), "NPY_sha256": sha(coords), "XYZ": str(xyzfile), "XYZ_sha256": sha(xyzfile), "atom_order_preserved": True, "new_coordinates_generated": False}
        results.append(value);receipts.append(receipt)
    write_new(intake / "manifest.json", {"received_at_utc": utc(), "client_intake_date_America_Los_Angeles": "2026-11-03", "files": receipts, "active_spectral_registry_changed": False, "molecular_structures_only": True})
    write_new(report / "audit.json", {"created_at_utc": utc(), "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__, "source_script_sha256": sha(__file__), "new_force_field_or_quantum_evaluations": 0, "records": results})
    write_new(report / "sources.json", {"trusted_reference_path": str(REFERENCE.relative_to(ROOT)), "trusted_reference_sha256": sha(REFERENCE), "original_pilot_index_sha256": sha(REFERENCE_INDEX), "reference_scope": "Previously verified official graph; no new online verification", "user_provided_raw_files": receipts, "network_policy_path": "/etc/codex/network-policy.json", "network_policy_sha256": sha('/etc/codex/network-policy.json')})
    with (report / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        keys = ["claimed_pubchem_cid", "name", "formula_computed", "explicit_atom_count", "electron_count_from_graph", "formal_charge", "radical_electrons", "specified_stereocenter_count", "inchikey_computed", "geometric_start_candidate_eligible", "name_CID_mapping_independently_verified"]
        writer = csv.DictWriter(handle, fieldnames=keys);writer.writeheader();writer.writerows({k: v[k] for k in keys} for v in results)
    return results


def verify_existing(intake, report):
    intake, report = Path(intake), Path(report)
    receipts = json.loads((intake / "manifest.json").read_text())["files"]
    audit = json.loads((report / "audit.json").read_text())
    reference = trusted_spinosin_reference()
    for receipt, saved in zip(receipts, audit["records"]):
        p = intake / receipt["file"]
        if sha(p) != receipt["sha256"] or p.stat().st_size != receipt["bytes"]:
            raise ValueError("Archived raw source hash mismatch")
        current, _, xyz = analyze_sdf(p, receipt["claimed_cid"], reference if receipt["claimed_cid"] == 155692 else None)
        for k in ["formula_computed", "explicit_atom_count", "inchi_computed", "inchikey_computed", "3d_stereo_matches_parsed_graph", "trusted_Spinosin_graph_and_stereo_match", "geometric_start_candidate_eligible"]:
            if current[k] != saved[k]:raise ValueError("Recomputed audit differs: " + k)
        q = Path(saved["derived_coordinates"]["NPY"])
        if sha(q) != saved["derived_coordinates"]["NPY_sha256"] or not np.array_equal(np.load(q, allow_pickle=False), xyz):
            raise ValueError("Derived coordinate payload differs from original SDF")
    return {"status": "PASS", "raw_records": len(receipts), "new_force_field_or_quantum_evaluations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huangqi-sdf", type=Path)
    parser.add_argument("--spinosin-sdf", type=Path)
    parser.add_argument("--intake-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing(args.intake_output, args.report_output)))
    else:
        if not args.huangqi_sdf or not args.spinosin_sdf:parser.error("Both uploaded SDF paths are required")
        values = run_intake({5280448: args.huangqi_sdf, 155692: args.spinosin_sdf}, args.intake_output, args.report_output)
        print(json.dumps([{k: v[k] for k in ["claimed_pubchem_cid", "formula_computed", "explicit_atom_count", "inchikey_computed", "geometric_start_candidate_eligible", "name_CID_mapping_independently_verified"]} for v in values], ensure_ascii=False))


if __name__ == "__main__":
    main()
