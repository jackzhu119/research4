import datetime as dt
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import openpyxl
from scipy.io import loadmat

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_princeton_fruit_20261031 import mat_identity, parse_timepoint, workbook_targets
from prepare_princeton_fruit_exploration_20261031 import prepare, TIMEPOINT_INDICES


class PrincetonFruitInterfaceTests(unittest.TestCase):
    def test_am_pm_join_and_persistent_group_excludes_condition_and_time(self):
        a = mat_identity("root/Sensys Data/May132023/9PM/MA/Tag/MA1_T_0513_pm.mat")
        b = mat_identity("root/Sensys Data/May142023/9AM/MA/NT/MA1_NT_0514_am.mat")
        self.assertEqual(a["fruit_id"], b["fruit_id"])
        self.assertNotEqual(a["timestamp"], b["timestamp"])
        self.assertEqual(parse_timepoint("May13_9pm"), dt.datetime(2023, 5, 13, 21))
        self.assertEqual(parse_timepoint("Jun02_9am"), dt.datetime(2023, 6, 2, 9))

    def test_filename_path_disagreement_fails_instead_of_misjoining_labels(self):
        for path in ["root/Sensys Data/May132023/9PM/PR/Tag/MA1_T_0513_pm.mat",
                     "root/Sensys Data/May132023/9PM/MA/Tag/MA1_T_0514_pm.mat",
                     "root/Sensys Data/May132023/9PM/MA/Tag/MA1_NT_0513_pm.mat",
                     "root/Sensys Data/May132023/9AM/MA/Tag/MA1_T_0513_pm.mat"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                mat_identity(path)

    def test_missing_targets_preserved_and_mango_pair_order_explicit(self):
        book = openpyxl.Workbook(); book.remove(book.active)
        for sheet, prefix in [("Mango DM, Brix", "MA"), ("Pear Brix", "PR"), ("Persimmon Brix", "PM")]:
            ws = book.create_sheet(sheet)
            ws.append([prefix] + [prefix + str(i) for i in range(1, 11)])
            ws.append(["May13_9pm"] + (["13.6, 9.2"] * 10 if prefix == "MA" else [9.2] * 10))
            ws.append(["May14_9am"] + [None] * 10)
        blob = io.BytesIO(); book.save(blob)
        lookup, _ = workbook_targets(blob.getvalue())
        self.assertEqual(lookup[("MA1", "2023-05-13T21:00:00")], {"DM":13.6, "Brix":9.2})
        self.assertIsNone(lookup[("MA1", "2023-05-14T09:00:00")])
        self.assertEqual(lookup[("PR1", "2023-05-13T21:00:00")], {"Brix":9.2})

    def test_tampered_source_rejected_before_parse_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); raw = root / "source.zip"; raw.write_bytes(b"changed raw bytes")
            with self.assertRaisesRegex(ValueError, "Archive hash differs"):
                prepare(raw, root/"out.npz", root/"manifest.json")
            self.assertFalse((root/"out.npz").exists())

    def test_real_local_interface_is_raw_waveform_with_30_persistent_groups(self):
        root = SCRIPTS.parent
        archive = root/"data/external/public_thz_20261031_v1/acquired/princeton_metasticker_fruit/author_repository.zip"
        output = archive.parent/"prepared/fruit_fixed_Tag_time_index_v1.npz"
        if not archive.exists() or not output.exists():
            self.skipTest("Local-only upstream archive/interface unavailable; no download or invented fixture")
        with np.load(output, allow_pickle=False) as d:
            self.assertEqual(d["X"].shape, (120,7000))
            self.assertEqual(set(d["condition"]), {"Tag"})
            self.assertEqual(len(set(d["groups"])), 30)
            self.assertEqual(len(set(d["record_id"])), 120)
            self.assertEqual(TIMEPOINT_INDICES, (0,10,20,30))
            for ft in ("MA","PR","PM"):
                selected = d["fruit_type"] == ft
                self.assertEqual(selected.sum(),40)
                self.assertEqual(len(set(d["groups"][selected])),10)
                self.assertTrue(np.isfinite(d["y_Brix"][selected]).all())
                self.assertTrue(np.isfinite(d["y_DM"][selected]).all() if ft=="MA" else np.isnan(d["y_DM"][selected]).all())
            with zipfile.ZipFile(archive) as z:
                targets,_=workbook_targets(z.read(next(n for n in z.namelist() if n.endswith('.xlsx'))))
                for i, member in enumerate(d["source_member"]):
                    arrays=[v for k,v in loadmat(io.BytesIO(z.read(str(member)))).items() if not k.startswith('__')]
                    np.testing.assert_array_equal(d["X"][i],arrays[0].ravel())
                    target=targets[(str(d["groups"][i]),str(d["timestamp"][i]))]
                    self.assertEqual(d["y_Brix"][i], target["Brix"])
                    if "DM" in target:self.assertEqual(d["y_DM"][i],target["DM"])


if __name__ == "__main__":
    unittest.main()
