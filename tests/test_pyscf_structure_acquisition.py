"""Chemical-identity guards; these checks do not run quantum chemistry."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/pyscf-spinosin-pilot-20261031/structures"
SPEC = importlib.util.spec_from_file_location("pilot_structure_acquire", ROOT / "scripts/acquire_spinosin_structures_20261031.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
try:
    from rdkit import Chem
except ImportError:
    Chem = None


@unittest.skipIf(Chem is None, "Use the recorded quantum-pilot PYTHONPATH for RDKit chemistry guards")
class MolecularIdentityGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "sources"
        self.source.mkdir()
        self.output = Path(self.tmp.name) / "output"
        self.output.mkdir()
        for name in ["spinosin_verified_2d.json", "jujuboside_a_source_record.json", "jujuboside_b_source_record.json"]:
            shutil.copyfile(SOURCE / name, self.source / name)

    def mutate(self, key, value):
        path = self.source / "spinosin_verified_2d.json"
        data = json.loads(path.read_text())
        data[key] = value
        path.write_text(json.dumps(data))

    def test_real_three_graphs_match_independent_identifiers_without_3d_or_dft(self):
        MODULE.verify_graphs(self.source, self.output)
        record = json.loads((self.output / "verification.json").read_text())
        self.assertEqual([row["total_atoms_with_hydrogens"] for row in record["molecules"]], [75, 178, 157])
        self.assertEqual([row["assigned_stereocenters"] for row in record["molecules"]], [10, 34, 29])
        self.assertEqual(record["quantum_computations"], 0)
        self.assertEqual(record["geometries_3d_generated"], 0)
        self.assertEqual(len(record["partial_jujuboside_a"]["unassigned_centers"]), 2)
        for row in record["molecules"]:
            self.assertTrue(row["MMFF_all_parameters_available"])
            self.assertIn("NOT a 3D", row["molfile_coordinate_type"])

    def test_wrong_formula_is_rejected(self):
        self.mutate("formula", "C28H34O15")
        with self.assertRaisesRegex(ValueError, "Formula mismatch"):
            MODULE.verify_graphs(self.source, self.output)

    def test_wrong_inchikey_is_rejected(self):
        self.mutate("source_inchikey", "VGGSULWDCMWZPO-UHFFFAOYSA-N")
        with self.assertRaisesRegex(ValueError, "InChIKey mismatch"):
            MODULE.verify_graphs(self.source, self.output)

    def test_same_formula_changed_stereo_cannot_pass_original_inchi(self):
        original = json.loads((self.source / "spinosin_verified_2d.json").read_text())
        altered = original["source_isomeric_smiles"].replace("[C@@H]1O", "[C@H]1O", 1)
        self.mutate("source_isomeric_smiles", altered)
        self.mutate("source_inchikey", Chem.MolToInchiKey(Chem.MolFromSmiles(altered)))
        with self.assertRaisesRegex(ValueError, "stereochemical graphs differ"):
            MODULE.verify_graphs(self.source, self.output)

    def test_unspecified_centers_rejected_even_if_two_identifiers_agree(self):
        original = json.loads((self.source / "spinosin_verified_2d.json").read_text())
        molecule = Chem.MolFromSmiles(original["source_isomeric_smiles"])
        Chem.RemoveStereochemistry(molecule)
        self.mutate("source_isomeric_smiles", Chem.MolToSmiles(molecule))
        self.mutate("source_inchikey", Chem.MolToInchiKey(molecule))
        self.mutate("source_inchi", Chem.MolToInchi(molecule))
        with self.assertRaisesRegex(ValueError, "unresolved stereocenters"):
            MODULE.verify_graphs(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
