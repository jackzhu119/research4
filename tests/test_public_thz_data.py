import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from herb_spectra.public_thz_data import (FIELDS,GINSENG_ORIGINAL,STATUS,PublicTHzDataError,
    duplicate_diagnostics,expected_ginseng_hashes,load_ginseng,prepare_ginseng,sha256,validate_ginseng_arrays)


class PublicTHzDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def copy_source(self):
        folder=self.root/'source';folder.mkdir()
        for name in expected_ginseng_hashes():shutil.copyfile(GINSENG_ORIGINAL/name,folder/name)
        return folder
    def test_actual_source_label_pairing_axis_and_no_metadata_features(self):
        d=load_ginseng();self.assertEqual(set(d),set(FIELDS));self.assertEqual(d['X_alpha'].shape,(42,65));self.assertEqual(d['X_n'].shape,(42,65))
        self.assertEqual(d['record_ids'][0],'G1_S01');self.assertEqual(d['record_ids'][-1],'G6_S07')
        self.assertEqual(np.unique(d['y'],return_counts=True)[1].tolist(),[22,20]);self.assertEqual((d['X_alpha']<0).sum(),141)
        for g in range(1,7):
            labels=np.loadtxt(GINSENG_ORIGINAL/f'label_{g}.csv',delimiter=',',encoding='utf-8-sig',dtype=int).astype(str)
            np.testing.assert_array_equal(d['y'][(g-1)*7:g*7],labels)
            for modality,prefix in [('X_alpha','a'),('X_n','n')]:
                raw=np.loadtxt(GINSENG_ORIGINAL/f'data_{prefix}{g}.csv',delimiter=',',skiprows=1,encoding='utf-8-sig')
                np.testing.assert_array_equal(d[modality][(g-1)*7:g*7],raw[:,1:].T)
        self.assertAlmostEqual(d['axis'][0],.2);self.assertAlmostEqual(d['axis'][-1],1.6)
    def test_raw_tamper_rejected_before_parser(self):
        folder=self.copy_source();p=folder/'data_a1.csv';p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(PublicTHzDataError,'bytes changed'):load_ginseng(folder)
    def test_explicit_fixture_manifest_cannot_legalize_nonbinary_labels(self):
        folder=self.copy_source();(folder/'label_1.csv').write_text('2,1,1,0,0,0,0\n')
        expected={name:sha256(folder/name) for name in expected_ginseng_hashes()}
        with self.assertRaisesRegex(PublicTHzDataError,'binary'):load_ginseng(folder,expected_files_sha256=expected)
    def test_explicit_fixture_changed_axis_is_rejected_without_interpolation(self):
        folder=self.copy_source();p=folder/'data_n2.csv';raw=np.loadtxt(p,delimiter=',',skiprows=1,encoding='utf-8-sig');raw[5,0]+=.001
        np.savetxt(p,raw,delimiter=',',header='Frequency (THz),,,,,,,',comments='')
        expected={name:sha256(folder/name) for name in expected_ginseng_hashes()}
        with self.assertRaisesRegex(PublicTHzDataError,'axes differ'):load_ginseng(folder,expected_files_sha256=expected)
    def test_unknown_metadata_permuted_pairing_nonfinite_and_labels_rejected(self):
        d=load_ginseng()
        for mutate in ['extra','ids','groups','nonfinite','labels','bool','shape']:
            c={k:v.copy() for k,v in d.items()}
            if mutate=='extra':c['plant_id']=d['record_ids'].copy()
            if mutate=='ids':c['record_ids'][0]=c['record_ids'][1]
            if mutate=='groups':c['export_groups'][0]='G2'
            if mutate=='nonfinite':c['X_alpha'][0,0]=np.nan
            if mutate=='labels':c['y'][0]='unverified-origin'
            if mutate=='bool':c['X_n']=c['X_n']>0
            if mutate=='shape':c['X_alpha']=c['X_alpha'][:41]
            with self.subTest(mutate=mutate):
                with self.assertRaises(PublicTHzDataError):validate_ginseng_arrays(c)
    def test_qc_detects_cross_export_exact_pair_without_removing_observations(self):
        d=load_ginseng();d['X_alpha'][7]=d['X_alpha'][0];d['X_n'][7]=d['X_n'][0]
        q=duplicate_diagnostics(d)
        self.assertTrue(q['no_observations_removed']);self.assertFalse(q['absence_of_duplicates_proves_independence'])
        for modality in ['alpha','n','paired']:
            self.assertTrue(any(p['record_a']=='G1_S01' and p['record_b']=='G2_S01' and p['different_export_groups'] for p in q['modalities'][modality]['exact_pairs']))
        self.assertEqual(d['X_alpha'].shape,(42,65))
    def test_actual_qc_has_no_exact_or_predeclared_near_pairs(self):
        q=duplicate_diagnostics(load_ginseng())
        self.assertEqual(q['threshold_predefined'],.99999)
        for m in q['modalities'].values():
            self.assertEqual(m['pairs_checked'],861);self.assertEqual(m['exact_pairs'],[]);self.assertEqual(m['near_pairs'],[])
        self.assertEqual(q['modalities']['alpha']['negative_cells'],141)
    def test_prepare_is_lossless_no_independence_inference_or_old_source_change(self):
        before={n:sha256(GINSENG_ORIGINAL/n) for n in expected_ginseng_hashes()};out=self.root/'prepared'
        m=prepare_ginseng(out)
        self.assertEqual(m['scientific_status'],STATUS);self.assertFalse(m['export_group_is_biological_batch'])
        self.assertIsNone(m['verified_independent_plants']);self.assertIsNone(m['verified_independent_batches']);self.assertEqual(m['new_model_fits'],0)
        self.assertEqual(m['new_independent_samples_added'],0)
        with (out/'observation_lineage.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),84);self.assertEqual(len({r['record_id'] for r in rows}),42)
        self.assertTrue(all(not r['physical_sample_id'] and not r['parent_sample_id'] and not r['plant_id'] and not r['batch_id'] for r in rows))
        actual=load_ginseng()
        with np.load(out/'observations.npz',allow_pickle=False) as z:
            for k in FIELDS:np.testing.assert_array_equal(z[k],actual[k])
        self.assertEqual(before,{n:sha256(GINSENG_ORIGINAL/n) for n in expected_ginseng_hashes()})
        with self.assertRaises(FileExistsError):prepare_ginseng(out)

if __name__=='__main__':unittest.main()
