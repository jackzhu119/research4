"""Structural guards and geometric checks; no SCF/DFT is executed."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spinosin_conformer_helpers", ROOT / "scripts/prepare_spinosin_conformers_20261031.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def load_rdkit():
    added = []
    for site in (ROOT / ".runtime/quantum-pilot-site", ROOT / ".runtime/quantum-upgrade-site"):
        if site.exists() and str(site) not in sys.path:
            sys.path.insert(0, str(site)); added.append(str(site))
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except ImportError:
        return None
    finally:
        for item in added:
            sys.path.remove(item)


RDKIT = load_rdkit()


class SpinosinPureGuards(unittest.TestCase):
    def test_kabsch_removes_translation_and_proper_rotation(self):
        xyz = np.asarray([[0, 0, 0], [1, 0, 0], [.2, 1.3, .1], [.3, .5, 1.7]], float)
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        self.assertLess(helpers.heavy_atom_rmsd(xyz, xyz @ rotation + 5), 1e-12)

    def test_kabsch_does_not_identify_a_reflection_as_same_geometry(self):
        xyz = np.asarray([[0, 0, 0], [1, 0, 0], [.2, 1.3, .1], [.3, .5, 1.7]], float)
        mirrored = xyz * np.asarray([-1, 1, 1])
        self.assertGreater(helpers.heavy_atom_rmsd(xyz, mirrored), .1)

    def test_source_hash_rejected_before_any_chemistry_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"; source.write_text("{}")
            output = Path(directory) / "new"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                helpers.run(source, output, "0" * 64)
            self.assertFalse(output.exists())

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"; source.write_text("{}")
            output = Path(directory) / "existing"; output.mkdir()
            marker = output / "keep"; marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                helpers.run(source, output, helpers.sha(source))
            self.assertEqual(marker.read_text(), "preserve")


@unittest.skipUnless(RDKIT, "Isolated RDKit installation is required")
class SpinosinChemistryGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((ROOT / "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json").read_text())

    def test_source_graph_has_correct_formula_electrons_and_stereo(self):
        molecule, checked = helpers.verify_source(self.record)
        self.assertEqual(molecule.GetNumAtoms(), 75)
        self.assertEqual(checked["composition"], {"C": 28, "H": 32, "O": 15})
        self.assertEqual(checked["electron_count"], 320)
        self.assertEqual(checked["radical_electrons"], 0)
        self.assertEqual(checked["formal_charge"], 0)
        self.assertEqual(checked["multiplicity"], 1)
        self.assertEqual(len(checked["source_stereocenters"]), 10)
        self.assertEqual(checked["computed_inchikey"], helpers.KEY)
        self.assertEqual(checked["computed_inchi"], self.record["source_inchi"])

    def test_wrong_formula_or_cid_is_rejected(self):
        for field, value in [("formula", "C28H30O15"), ("pubchem_cid", 1)]:
            changed = dict(self.record, **{field: value})
            with self.assertRaises(ValueError):
                helpers.verify_source(changed)

    def test_missing_anomeric_stereochemistry_is_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["source_isomeric_smiles"] = changed["source_isomeric_smiles"].replace("[C@@H]", "[CH]", 1)
        with self.assertRaisesRegex(ValueError, "stereocenters"):
            helpers.verify_source(changed)

    def test_opposite_anomeric_stereochemistry_cannot_match_source_key(self):
        changed = copy.deepcopy(self.record)
        changed["source_isomeric_smiles"] = changed["source_isomeric_smiles"].replace("[C@@H]", "[C@H]", 1)
        with self.assertRaisesRegex(ValueError, "InChIKey"):
            helpers.verify_source(changed)

    def test_coordinate_derived_stereo_and_xyz_75_rows(self):
        _, AllChem = RDKIT
        molecule, _ = helpers.verify_source(self.record)
        params = AllChem.ETKDGv3(); params.randomSeed = 20261031; params.numThreads = 1
        self.assertGreaterEqual(AllChem.EmbedMolecule(molecule, params), 0)
        self.assertEqual(helpers.stereo_key_from_coordinates(molecule), helpers.KEY)
        xyz = helpers.xyz_text(molecule, "Structural fixture only; not DFT")
        self.assertEqual(xyz.splitlines()[0], "75")
        self.assertEqual(len(xyz.splitlines()), 77)
        self.assertTrue(np.isfinite(np.asarray([row.split()[1:] for row in xyz.splitlines()[2:]], float)).all())

    def test_standard_and_explicit_h_rotor_conventions_are_distinct(self):
        Chem, _ = RDKIT
        from rdkit.Chem import rdMolDescriptors
        molecule = Chem.MolFromSmiles(self.record["source_isomeric_smiles"])
        self.assertEqual(rdMolDescriptors.CalcNumRotatableBonds(molecule), 7)
        self.assertEqual(rdMolDescriptors.CalcNumRotatableBonds(Chem.AddHs(molecule)), 16)

    def test_embedding_budget_cannot_silently_expand(self):
        source = ROOT / "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Bounded"):
                helpers.run(source, Path(directory) / "never-created", helpers.SOURCE_SHA256, count=13)
            self.assertFalse((Path(directory) / "never-created").exists())


if __name__ == "__main__":
    unittest.main()
