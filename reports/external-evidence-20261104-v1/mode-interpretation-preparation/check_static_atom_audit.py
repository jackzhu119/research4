"""Validate graph-label metadata and static hashes; no QM, NPY, CHK or Hessian I/O."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    audit = json.loads((here / "atom_order_audit.json").read_text())
    groups = json.loads((here / "atom_groups_C001.json").read_text())
    ambiguity = json.loads((here / "uploaded_SDF_mapping_ambiguity.json").read_text())
    with (here / "C001_atom_map.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 75
    for i, row in enumerate(rows):
        assert int(row["C001_SDF_index_zero_based"]) == i
        assert int(row["current_PySCF_index_zero_based"]) == i
        assert int(row["C001_SDF_atom_number_one_based"]) == i + 1
        assert int(row["current_XYZ_atom_number_one_based"]) == i + 1
    assert Counter(x["element"] for x in rows) == {"C": 28, "H": 32, "O": 15}
    coverage = []
    for group in groups["groups"]:
        zero = group["indices_zero_based"]
        assert group["indices_one_based"] == [i + 1 for i in zero]
        assert len(set(zero)) == len(zero)
        assert all(0 <= i < 75 for i in zero)
        if group["family"] == "DISJOINT_PARTITION":
            coverage.extend(zero)
            assert all(rows[i]["disjoint_partition"] == group["group_id"] for i in zero)
    assert sorted(coverage) == list(range(75))
    assert len(groups["hydroxyl_pairs"]) == 9
    assert Counter(x["type"] for x in groups["hydroxyl_pairs"]) == {
        "phenolic_OH": 2, "sugar_alcohol_OH": 7}
    assert audit["v3_to_v5_coordinate_max_abs_delta_angstrom"] == 0
    assert audit["read_active_Hessian_NPY_or_CHK"] is False
    assert audit["modes_or_frequencies_read"] is False
    assert ambiguity["unique_historical_C001_to_uploaded_75_atom_mapping_established"] is False
    assert ambiguity["arbitrary_first_match_selected_for_mode_labels"] is False
    assert ambiguity["heavy_graph_mappings_with_chirality_and_no_uniquification"] == 2
    assert ambiguity["all_75_graph_mappings_with_chirality_and_no_uniquification"] == 48
    for bound in audit["inputs"].values():
        path = repo / bound["path"]
        assert path.suffix.lower() not in (".npy", ".npz", ".chk", ".h5", ".hdf5")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bound["sha256"]
    index = here / "files_sha256.json"
    if index.exists():
        for row in json.loads(index.read_text())["files"]:
            assert hashlib.sha256((here / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    print(json.dumps({"status": "PASS_STATIC_ATOM_METADATA", "atoms": 75,
                      "disjoint_partition_count": 4, "hydroxyl_pairs": 9,
                      "mapping_basis": "C001_LOCAL_ORDER", "unique_uploaded_75_map": False,
                      "quantum_or_live_Hessian_IO": False}))


if __name__ == "__main__":
    main()
