"""Meaningful invariants of the locked specimen pools, independent of models."""
import csv
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('freeze', ROOT / 'scripts/freeze_species_revision_split.py')
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


class FrozenSplitTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / 'data/metadata_template.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.rows = list(csv.DictReader(handle))

    def test_rare_groups_keep_three_pools(self):
        self.assertEqual(freeze.counts(3), (1, 1, 1))
        self.assertEqual(freeze.counts(5), (3, 1, 1))
        self.assertEqual(freeze.counts(100), (60, 20, 20))
        self.assertEqual(freeze.counts(350), (210, 70, 70))
        with self.assertRaises(ValueError):
            freeze.counts(2)

    def test_input_order_does_not_change_membership(self):
        self.assertEqual(freeze.make_split(self.rows), freeze.make_split(self.rows[::-1]))

    def test_no_physical_overlap_and_all_original_records_preserved(self):
        split = freeze.make_split(self.rows)
        self.assertEqual(len(split), 1672)
        self.assertEqual({x['record_id'] for x in split}, {x['record_id'] for x in self.rows})
        pools = {role: {row['physical_sample_id'] for row in split if row['role'] == role}
                 for role in freeze.ROLES}
        self.assertEqual(sum(map(len, pools.values())), 1672)
        for left, right in [('train', 'calibration'), ('train', 'prediction'), ('calibration', 'prediction')]:
            self.assertFalse(pools[left] & pools[right])
        for group in {row['origin_group_id'] for row in split}:
            selected = [row for row in split if row['origin_group_id'] == group]
            self.assertEqual(tuple(sum(row['role'] == role for row in selected) for role in freeze.ROLES),
                             freeze.counts(len(selected)))

    def test_repeated_specimen_fails_closed(self):
        rows = [dict(x) for x in self.rows]
        rows[0]['physical_sample_id'] = rows[1]['physical_sample_id']
        with self.assertRaises(ValueError):
            freeze.make_split(rows)

    def test_every_historical_role_is_preserved(self):
        import json
        original = json.loads(freeze.HISTORICAL.read_text())['records']
        split = {row['record_id']: row for row in freeze.make_split(self.rows)}
        self.assertEqual(len(original), 855)
        for row in original:
            expected = 'train' if row['role'] == 'training' else row['role']
            self.assertEqual(split[row['record_id']]['role'], expected)
            self.assertEqual(split[row['record_id']]['physical_sample_id'], row['physical_sample_id'])


if __name__ == '__main__':
    unittest.main()
