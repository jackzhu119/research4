import csv
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from openpyxl import Workbook

from herb_spectra.multiplant_database import (
    RawSheet, align_sheets, assert_model_input_arrays, base_header,
    numeric_row_hash, read_raw_sheet, validate_zaoren_row_bindings, workbook_health,
)


def raw(headers, rows):
    return RawSheet('吸收系数','Thz',np.array([.1,.2]),np.array(rows,dtype=float),tuple(headers),
                    tuple('BC'[i] for i in range(len(headers))),(),np.empty((0,2)),'A1:A1',0)


class MultiplantDatabaseTests(unittest.TestCase):
    def test_pair_by_explicit_base_id_and_reorder_values(self):
        a=raw(['ZY01-001','ZY01-002'],[[1,2],[3,4]])
        n=raw(['ZY01-002_1mm','ZY01-001_1.5mm'],[[30,40],[10,20]])
        ids,values,_=align_sheets({'alpha':a,'n':n},'zaoren','ZY01')
        self.assertEqual(ids,['ZY01-001','ZY01-002'])
        np.testing.assert_array_equal(values['n'],[[10,20],[30,40]])

    def test_same_ancestor_preparation_not_silently_counted_twice(self):
        a=raw(['ZY01-001_1mm','ZY01-001_2mm'],[[1,2],[3,4]])
        with self.assertRaisesRegex(ValueError,'Duplicate base'):
            align_sheets({'alpha':a},'zaoren','ZY01')

    def test_missing_modality_sample_is_not_deleted_for_pairing(self):
        a=raw(['ZY01-001','ZY01-002'],[[1,2],[3,4]])
        n=raw(['ZY01-001'],[[3,4]])
        with self.assertRaisesRegex(ValueError,'sets differ'):
            align_sheets({'alpha':a,'n':n},'zaoren','ZY01')

    def test_different_column_names_cannot_invent_id(self):
        with self.assertRaisesRegex(ValueError,'Unrecognized'):
            base_header('some_mean_curve','zaoren')

    def test_actual_sheet_dimensions_and_missing_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'test.xlsx';book=Workbook();s=book.active;s.title='吸收系数'
            s.append(['Thz','参考值','样品1','样品2']);s.append([.1,1,2,3]);s.append([.2,1,None,4]);book.save(p)
            with zipfile.ZipFile(p) as z:members={n:z.read(n) for n in z.namelist()}
            members['xl/worksheets/sheet1.xml']=members['xl/worksheets/sheet1.xml'].replace(b'ref="A1:D3"',b'ref="A1:A1"')
            with zipfile.ZipFile(p,'w') as z:
                for n,b in members.items():z.writestr(n,b)
            result=read_raw_sheet(p,'吸收系数')
            self.assertEqual(result.X.shape,(2,2));self.assertTrue(np.isnan(result.X[0,1]))
            self.assertEqual(result.references,('参考值',));self.assertTrue(workbook_health(p)['container_crc_pass'])

    def test_metadata_strings_are_not_model_features(self):
        with self.assertRaises(ValueError):
            assert_model_input_arrays({'X':np.array([['ZY01']]),'record_ids':np.array(['ZY01-001']),'axis':np.array([.1])})

    def test_binding_rejects_values_swapped_under_unchanged_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'binding.csv';payload={'record_ids':np.array(['ZY01-001','ZY01-002'])};rows=[]
            for mode in ['alpha','n','time','frequency']:
                payload['X_'+mode]=np.array([[1.,2.],[3.,4.]])
                payload['axis_'+mode]=np.array([.1,.2])
                for rid,row in zip(payload['record_ids'],payload['X_'+mode]):
                    rows.append({'modality':mode,'record_id':rid,'model_matrix_row_sha256':numeric_row_hash(row),
                                 'model_axis_sha256':numeric_row_hash(payload['axis_'+mode])})
            with p.open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            self.assertEqual(validate_zaoren_row_bindings(payload,p)['record_count'],2)
            payload['X_n']=payload['X_n'][::-1]
            with self.assertRaisesRegex(ValueError,'values do not match'):
                validate_zaoren_row_bindings(payload,p)


if __name__=='__main__':
    unittest.main()
