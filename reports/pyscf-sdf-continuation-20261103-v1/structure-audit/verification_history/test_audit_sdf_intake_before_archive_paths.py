"""No-QM integrity, stereochemistry, geometry and immutable-intake controls."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

SOURCE = Path(__file__).with_name("audit_sdf_intake.py")
SPEC = importlib.util.spec_from_file_location("new_sdf_intake_audit", SOURCE)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
SPINOSIN = Path("/workspace/attachments/69a91b64-1d07-416d-95a8-4ba0d4636e89/枣仁CID_155692.sdf")
HUANGQI = Path("/workspace/attachments/3d965bd3-ba2e-40d4-b7c3-8b0e07235d7e/黄芪CID_5280448.sdf")


class SDFIntakeTests(unittest.TestCase):
    def test_real_spinosin_matches_previously_verified_full_stereo_graph(self):
        d, _, x = A.analyze_sdf(SPINOSIN, 155692, A.trusted_spinosin_reference())
        self.assertEqual(d["formula_computed"], "C28H32O15")
        self.assertEqual(d["electron_count_from_graph"], 320)
        self.assertEqual(x.shape, (75, 3))
        self.assertEqual(d["specified_stereocenter_count"], 10)
        self.assertTrue(d["trusted_Spinosin_graph_and_stereo_match"])
        self.assertFalse(d["uploaded_symbol_order_matches_old_C001"])
        self.assertFalse(d["old_checkpoint_AO_or_Hessian_indexwise_reuse_allowed"])

    def test_new_cid_is_one_structure_not_a_named_plant_spectrum(self):
        d, _, x = A.analyze_sdf(HUANGQI, 5280448)
        self.assertEqual(d["formula_computed"], "C16H12O5")
        self.assertEqual(d["electron_count_from_graph"], 148)
        self.assertEqual(x.shape, (33, 3))
        self.assertEqual(d["inchikey_computed"], "ZZAJQOPSWWVMBI-UHFFFAOYSA-N")
        self.assertFalse(d["name_CID_mapping_independently_verified"])
        self.assertEqual(d["measured_THz_spectra_added"], 0)

    def test_cid_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "CID claim"):
            A.analyze_sdf(SPINOSIN, 5280448)

    def test_multiple_records_rejected(self):
        with TemporaryDirectory() as temporary:
            p = Path(temporary) / "multiple.sdf"
            p.write_bytes(HUANGQI.read_bytes() * 2)
            with self.assertRaisesRegex(ValueError, "Exactly one"):
                A.analyze_sdf(p, 5280448)

    def test_mirrored_spinosin_rejected_against_trusted_stereo(self):
        before = A.sha(SPINOSIN)
        m = Chem.SDMolSupplier(str(SPINOSIN), removeHs=False)[0]
        conf = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (-p.x, p.y, p.z))
        with TemporaryDirectory() as temporary:
            p = Path(temporary) / "synthetic_mirror_test_only.sdf"
            with Chem.SDWriter(str(p)) as writer:
                writer.write(m)
            d, _, _ = A.analyze_sdf(p, 155692, A.trusted_spinosin_reference())
            self.assertFalse(d["trusted_Spinosin_graph_and_stereo_match"])
            self.assertFalse(d["geometric_start_candidate_eligible"])
        self.assertEqual(A.sha(SPINOSIN), before)

    def test_two_dimensional_depiction_is_not_silently_promoted_to_3d(self):
        m = Chem.AddHs(Chem.MolFromSmiles("O"))
        rdDepictor.Compute2DCoords(m)
        m.SetProp("PUBCHEM_COMPOUND_CID", "5280448")
        with TemporaryDirectory() as temporary:
            p = Path(temporary) / "synthetic_2d_test_only.sdf"
            with Chem.SDWriter(str(p)) as writer:
                writer.write(m)
            d, _, _ = A.analyze_sdf(p, 5280448)
            self.assertFalse(d["geometry"]["toolkit_conformer_is_3d"])
            self.assertFalse(d["geometric_start_candidate_eligible"])

    def test_duplicate_atomic_positions_fail_geometry_screen(self):
        m = Chem.AddHs(Chem.MolFromSmiles("O"))
        c = Chem.Conformer(m.GetNumAtoms())
        c.Set3D(True)
        m.AddConformer(c)
        m.SetProp("PUBCHEM_COMPOUND_CID", "5280448")
        with TemporaryDirectory() as temporary:
            p = Path(temporary) / "synthetic_overlap_test_only.sdf"
            with Chem.SDWriter(str(p)) as writer:
                writer.write(m)
            d, _, _ = A.analyze_sdf(p, 5280448)
            self.assertFalse(d["geometry"]["loose_bond_geometry_screen_passed"])

    def test_intake_is_byte_exact_refuses_overwrite_and_detects_derived_tamper(self):
        with TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            report = Path(temporary) / "report"
            d = A.run_intake({5280448: HUANGQI, 155692: SPINOSIN}, raw, report)
            self.assertEqual((raw / HUANGQI.name).read_bytes(), HUANGQI.read_bytes())
            self.assertEqual((raw / SPINOSIN.name).read_bytes(), SPINOSIN.read_bytes())
            self.assertEqual(A.verify_existing(raw, report)["status"], "PASS")
            with self.assertRaises(FileExistsError):
                A.run_intake({5280448: HUANGQI, 155692: SPINOSIN}, raw, report)
            p = Path(d[0]["derived_coordinates"]["NPY"])
            xyz = np.load(p, allow_pickle=False)
            xyz[0, 0] += 0.1
            np.save(p, xyz)
            with self.assertRaisesRegex(ValueError, "Derived coordinate"):
                A.verify_existing(raw, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
