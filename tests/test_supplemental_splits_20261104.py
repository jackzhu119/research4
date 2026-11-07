"""Meaningful identity, role and exact-value export checks; no model training."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/prepare_supplemental_splits_20261104.py'
spec = importlib.util.spec_from_file_location('supplemental_split_test_module', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.path.insert(0,str(PATH.parent))
from bind_supplemental_protocol_20261104 import check_source_partitions


def rows(count=10, classes=2):
    return [{'record_id':f'C{c}-S{i:03d}', 'entity_id':f'ENTITY-C{c}-S{i:03d}',
             'label':f'C{c}', 'legacy_match_ids':[], 'species':'test', 'source_id':f'TEST{c}',
             'batch_id':f'BATCH{c}', 'stratum':'synthetic_test_only', 'input_source':'synthetic',
             'alpha_fingerprint':f'FP-C{c}-S{i:03d}'} for c in range(classes) for i in range(count)]


class SupplementalSplitTests(unittest.TestCase):
    def test_three_records_fill_each_role_once(self):
        records=rows(3,3)
        result=mod.allocate_roles(records,20261104)
        self.assertEqual(result,mod.allocate_roles(records,20261104))
        for c in range(3):
            self.assertEqual(sorted(result[r['record_id']] for r in records if r['label']==f'C{c}'),sorted(mod.ROLES))

    def test_two_records_block_instead_of_dropping_class(self):
        with self.assertRaisesRegex(ValueError,'fewer than three'):
            mod.allocate_roles(rows(2),20261104)

    def test_cross_namespace_record_legacy_alias_is_rejected(self):
        records=rows(3)
        records[1]['legacy_match_ids']=[records[0]['record_id']]
        assignment={r['record_id']:'source' for r in records}
        assignment[records[1]['record_id']]='query'
        with self.assertRaisesRegex(ValueError,'component spans roles'):
            mod.assert_partition_identity(records,assignment)

    def test_transitive_identity_chain_is_rejected(self):
        records=rows(3)
        records[0]['legacy_match_ids']=['ANCESTOR']
        records[1]['entity_id']='ANCESTOR'
        records[1]['legacy_match_ids']=[records[2]['record_id']]
        assignment={r['record_id']:'source' for r in records}
        assignment[records[2]['record_id']]='support'
        with self.assertRaisesRegex(ValueError,'component spans roles'):
            mod.assert_partition_identity(records,assignment)

    def test_numeric_identity_cannot_cross_roles(self):
        records=rows(3)
        records[1]['alpha_fingerprint']=records[0]['alpha_fingerprint']
        assignment={r['record_id']:'source' for r in records};assignment[records[1]['record_id']]='query'
        with self.assertRaisesRegex(ValueError,'Identical alpha inputs'):
            mod.assert_partition_identity(records,assignment)

    def test_one_identity_with_different_labels_is_rejected(self):
        records=rows(3)
        records[3]['entity_id']=records[0]['entity_id']
        assignment={r['record_id']:'source' for r in records}
        with self.assertRaisesRegex(ValueError,'conflicting anonymous labels'):
            mod.assert_partition_identity(records,assignment)

    def test_missing_role_record_is_rejected(self):
        records=rows(3); assignment={r['record_id']:'source' for r in records[:-1]}
        with self.assertRaisesRegex(ValueError,'cover every record'):
            mod.assert_partition_identity(records,assignment)

    def test_source_singletons_are_not_faked_into_cv(self):
        records=rows(3)
        assignments=mod.allocate_roles(records,20261104)
        with self.assertRaisesRegex(ValueError,'source singleton'):
            mod.make_folds(records,assignments,20261104)

    def test_cv_is_source_only_and_validates_each_source_once(self):
        records=rows(10)
        assignment=mod.allocate_roles(records,20261104)
        folds=mod.make_folds(records,assignment,20261104)
        source={r['record_id'] for r in records if assignment[r['record_id']]=='source'}
        heldout={r['record_id'] for r in records if assignment[r['record_id']]!='source'}
        validated=[]
        for fold in folds:
            self.assertFalse(set(fold['train_ids'])&set(fold['valid_ids']))
            self.assertEqual(set(fold['train_ids'])|set(fold['valid_ids']),source)
            self.assertFalse((set(fold['train_ids'])|set(fold['valid_ids']))&heldout)
            validated+=fold['valid_ids']
        self.assertEqual(set(validated),source);self.assertEqual(len(validated),len(source))

    def test_source_pseudo_tasks_use_six_real_permutations(self):
        records=rows(3,3)
        assignment={r['record_id']:'source' for r in records}
        episodes=mod.make_episodes(records,assignment,20261204,[1])
        self.assertEqual(len(episodes),6)
        self.assertEqual([e['stage'] for e in episodes],['selection']*3+['gate']*3)
        for c in range(3):
            permutations=set()
            own={r['record_id'] for r in records if r['label']==f'C{c}'}
            for e in episodes:
                own_roles=tuple(next(i for i in e[k] if i in own) for k in ['base_ids','support_ids','query_ids'])
                self.assertEqual(len(set(own_roles)),3);permutations.add(own_roles)
            self.assertEqual(len(permutations),6)

    def test_support_draws_are_nested_and_not_oversampled(self):
        records=rows(30,3);assignment=mod.allocate_roles(records,20261104)
        draws,shots,blocked=mod.make_draws(records,assignment,20261104)
        self.assertEqual(shots,[1,3,5]);self.assertFalse(blocked);self.assertEqual(len(draws),10)
        for draw in draws:
            self.assertTrue(set(draw['shots']['1'])<set(draw['shots']['3'])<set(draw['shots']['5']))
            self.assertEqual(len(draw['shots']['5']),len(set(draw['shots']['5'])))
            self.assertTrue(all(assignment[i]=='support' for i in draw['shots']['5']))

    def test_one_support_per_class_blocks_three_and_five_shot(self):
        records=rows(5);assignment=mod.allocate_roles(records,20261104)
        _,shots,blocked=mod.make_draws(records,assignment,20261104)
        self.assertEqual(shots,[1]);self.assertEqual(set(blocked),{'3','5'})

    def test_query_pack_has_no_labels_and_truth_is_separate(self):
        records=rows(10)
        assignment=mod.allocate_roles(records,20261104)
        recipe={'split_seed':20261104,'calibration':{'support_draw_seed':20261104,'support_draws':10,
                  'conservative':{'source_episode_seed_base':20261204}}}
        data={'records':records,'X':np.arange(len(records)*7,dtype=float).reshape(len(records),7),
              'axis':np.arange(7,dtype=float),'input_refs':[]}
        task={'id':'synthetic_role_test','classes_expected':2,'inherit_selection_from':None}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'roles';out.mkdir()
            descriptor,_=mod.write_task(root,out,task,data,assignment,recipe)
            with np.load(root/descriptor['query_pack']['path'],allow_pickle=False) as q:
                self.assertFalse({'y','labels','y_true','origin_group_ids'}&set(q.files))
            self.assertTrue((root/descriptor['truth']['path']).is_file())
            with np.load(root/descriptor['source_pack']['path'],allow_pickle=False) as s:
                self.assertIn('y',s.files)

    def test_exact_float_xlsx_roundtrip_and_no_overwrite(self):
        values=np.array([[np.nextafter(1.,2.),-0.,np.pi],[1.e-300,1.e300,np.nan]])
        sheets=[{'name':'吸收系数','axis_header':'THz','axis':np.array([.1,.2,.3]),
                 'reference_headers':['参考值'],'reference_values':np.array([[np.nextafter(.5,1.),0.,-3.]]),
                 'record_ids':['ANON-001','ANON-002'],'X':values}]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'exact.xlsx'
            receipt=mod.write_exact_xlsx(path,sheets)
            self.assertEqual(receipt['nonfinite_cells_exported_as_blank'],1)
            self.assertEqual(mod.verify_xlsx(path,sheets)['status'],'PASS_INDEPENDENT_OPENPYXL_ALL_VALUES')
            with self.assertRaisesRegex(ValueError,'Never overwrite'):
                mod.write_exact_xlsx(path,sheets)

    def test_axis_is_part_of_measurement_identity(self):
        row=np.array([1.,2.,3.]);a=np.array([.1,.2,.3]);b=a.copy();b[-1]=.4
        self.assertNotEqual(mod.identity_fingerprint(a,row),mod.identity_fingerprint(b,row))

    def test_binding_rejects_inner_fold_cross_namespace_ancestor(self):
        records=rows(3)
        records[1]['legacy_match_ids']=[records[0]['record_id']]
        all_ids=[r['record_id'] for r in records]
        task={'inner_folds':[{'train_ids':all_ids[1:],'valid_ids':all_ids[:1]}]}
        with self.assertRaisesRegex(ValueError,'component spans roles'):
            check_source_partitions(records,task)

    def test_binding_rejects_repeated_source_oof_validation(self):
        records=rows(3)
        all_ids=[r['record_id'] for r in records]
        task={'inner_folds':[{'train_ids':all_ids[1:],'valid_ids':all_ids[:1]}]*2}
        with self.assertRaisesRegex(ValueError,'one OOF'):
            check_source_partitions(records,task)

    def test_binding_checks_original_preparation_script_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prepare.py';path.write_text('original')
            ref=mod.reference(path,Path(temp));path.write_text('modified')
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                mod.checked(Path(temp),ref)


if __name__=='__main__':
    unittest.main()
