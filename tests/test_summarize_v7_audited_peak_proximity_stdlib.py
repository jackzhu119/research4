"""Synthetic stdlib-only unit fixtures, not simulated or measured spectra."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy

P = Path(__file__).resolve().parents[1] / 'scripts/summarize_v7_audited_peak_proximity_20261105.py'
S = importlib.util.spec_from_file_location('proximity_stdlib_tests', P)
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


def frequencies():
    return [{'case':'Spinosin','mode':i+1,'frequency_cm1':(i+1)/M.CM1_TO_THZ,
             'frequency_THz':float(i+1),'imaginary':False,'qualified_for_mechanism':True,'IR_intensity_status':'NOT_COMPUTED'} for i in range(219)]


def checks(negative=0, zero=0):
    return {'imaginary_frequency_count':negative,'zero_frequency_count':zero,
            'all_frequencies_positive':negative==zero==0,'qualified_for_mechanism':negative==zero==0}


def peaks():
    return [{'outer_fold':i,'candidate_number':j,'frequency_index':i*4+j,'frequency_thz':.6,
             'wavenumber_cm1':.6/M.CM1_TO_THZ,'classification_prior_used':'False','chemical_assignment':''}
            for i in range(5) for j in range(1,5)]


class ProximityTests(unittest.TestCase):
    def test_import_has_no_numeric_runtime(self):
        self.assertFalse(any(n.split('.')[0] in ('numpy','scipy','pyscf','torch','h5py') for n in sys.modules))

    def test_all219_signed_modes_preserved(self):
        rows=frequencies(); rows[0].update(frequency_cm1=-1/M.CM1_TO_THZ,frequency_THz=-1.,imaginary=True)
        rows[1].update(frequency_cm1=0.,frequency_THz=0.,imaginary=False)
        for row in rows:row['qualified_for_mechanism']=False
        result=M.frequency_rows(rows,checks(1,1))
        self.assertEqual(len(result),219); self.assertEqual([r['frequency_THz'] for r in result[:2]],[-1.,0.])

    def test_missing_mode_is_rejected(self):
        with self.assertRaises(ValueError): M.frequency_rows(frequencies()[:-1],checks())

    def test_duplicate_or_reordered_mode_rejected(self):
        rows=frequencies();rows[1]['mode']=1
        with self.assertRaises(ValueError):M.frequency_rows(rows,checks())

    def test_units_and_sign_metadata_rejected(self):
        for key,value in (('frequency_THz',-1),('frequency_cm1',0),('imaginary',True),('IR_intensity_status','0')):
            rows=frequencies();rows[0][key]=value
            with self.assertRaises(ValueError):M.frequency_rows(rows,checks())

    def test_nonfinite_frequency_rejected(self):
        rows=frequencies();rows[0]['frequency_cm1']='nan'
        with self.assertRaises(ValueError):M.frequency_rows(rows,checks())

    def test_row_qualification_must_match_actual_audit(self):
        rows=frequencies();rows[0]['qualified_for_mechanism']=False
        with self.assertRaises(ValueError):M.frequency_rows(rows,checks())

    def test_audit_sign_counts_must_match(self):
        with self.assertRaises(ValueError):M.frequency_rows(frequencies(),checks(1,0))

    def test_retains_all_overlapping_fold_records(self):
        result=M.experimental_rows(peaks());self.assertEqual(len(result),20)
        self.assertEqual(len({r['experimental_frequency_THz'] for r in result}),1)

    def test_unknown_or_modified_experiment_rejected(self):
        for key,value in (('classification_prior_used','True'),('chemical_assignment','spinosin'),('frequency_thz',2.)):
            rows=peaks();rows[0][key]=value
            with self.assertRaises(ValueError):M.experimental_rows(rows)

    def test_fold_duplicate_rejected(self):
        rows=peaks();rows[0]=deepcopy(rows[1])
        with self.assertRaises(ValueError):M.experimental_rows(rows)

    def test_every_nearest_tie_is_retained(self):
        p=M.experimental_rows(peaks())
        modes=[{'mode':1,'frequency_THz':.5,'frequency_cm1':.5/M.CM1_TO_THZ},
               {'mode':2,'frequency_THz':.7,'frequency_cm1':.7/M.CM1_TO_THZ}]
        rows=M.nearest_rows(p,modes);self.assertEqual(len(rows),40)
        self.assertTrue(all(r['nearest_tie_count']==2 and r['chemical_attribution_verified'] is False for r in rows))
        self.assertTrue(all(r['IR_intensity_status']=='NOT_COMPUTED' for r in rows))

    def test_negative_zero_out_of_range_are_not_converted_to_positive_modes(self):
        modes=[{'mode':i,'frequency_THz':f,'frequency_cm1':f/M.CM1_TO_THZ} for i,f in enumerate((-.6,0.,2.),1)]
        self.assertEqual(M.nearest_rows(M.experimental_rows(peaks()),modes),[])

    def test_empty_mode_range_invents_nothing(self):
        self.assertEqual(M.nearest_rows(M.experimental_rows(peaks()),[]),[])

    def test_bound_small_file_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture';p.write_text('SYNTHETIC_METADATA');h=M.sha(p);M.checked(p,h)
            p.write_text('CHANGED_METADATA')
            with self.assertRaises(ValueError):M.checked(p,h)

    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture';p.write_text('SYNTHETIC_METADATA');q=Path(td)/'alias';q.symlink_to(p)
            with self.assertRaises(ValueError):M.checked(q,M.sha(p))


if __name__=='__main__':unittest.main()
