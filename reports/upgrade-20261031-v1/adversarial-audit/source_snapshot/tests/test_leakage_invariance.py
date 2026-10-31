"""Negative controls for the bounded real-data invariance audit.

The full C/D/H real fits are run by audit_upgrade_leakage.py after a root gate;
these tests verify its failure paths and scorer separation without a search.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("upgrade_adversarial_audit",ROOT/"scripts/audit_upgrade_leakage.py")
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


class LeakageInvarianceTests(unittest.TestCase):
    def test_group_permutation_never_splits_descendants(self):
        groups=np.array(["p1","p1","p2","p2","p3","p3","p4","p4"])
        labels=np.array(["a","a","b","b","a","a","b","b"])
        changed=AUDIT.block_group_permutation(labels,groups,[np.arange(4),np.arange(4,8)],17)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(changed[groups==group])),1)
        for block in (np.arange(4),np.arange(4,8)):
            self.assertEqual(sorted(changed[block].tolist()),sorted(labels[block].tolist()))

    def test_permutation_refuses_ancestor_crossing_blocks(self):
        with self.assertRaisesRegex(ValueError,"crosses permutation blocks"):
            AUDIT.block_group_permutation(np.array(["a","a","b","b"]),
                np.array(["p1","p1","p2","p2"]),[np.array([0,2]),np.array([1,3])],42)

    def test_permutation_refuses_conflicting_physical_labels(self):
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            AUDIT.block_group_permutation(np.array(["a","b"]),np.array(["p1","p1"]),[np.array([0,1])],42)

    def test_intentional_artifact_contamination_is_detected(self):
        with self.assertRaisesRegex(AssertionError,"invariant failed"):
            AUDIT.assert_unchanged({"selector":[1,4],"weight":.2},{"selector":[1,5],"weight":.2},"bad_test_based_selector")

    def test_scorer_changes_only_metrics_when_truth_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"pred.npz"
            np.savez_compressed(path,record_ids=np.array(["p1","p2","p3"]),predicted=np.array(["a","b","a"]),
                                class_order=np.array(["a","b"]),scores=np.array([[1.,0.],[0.,1.],[1.,0.]]))
            before=AUDIT.sha(path)
            first=AUDIT.score_saved_predictions(path,{"p1":"a","p2":"b","p3":"a"})
            second=AUDIT.score_saved_predictions(path,{"p1":"b","p2":"a","p3":"a"})
            self.assertEqual(first["correct"],3);self.assertEqual(second["correct"],1)
            self.assertEqual(AUDIT.sha(path),before)

    def test_duplicate_oof_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"pred.npz"
            np.savez_compressed(path,record_ids=np.array(["same","same"]),predicted=np.array(["a","a"]),
                                class_order=np.array(["a","b"]),scores=np.ones((2,2)))
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                AUDIT.load_predictions(path)

    def test_training_api_cannot_receive_query_truth(self):
        self.assertEqual(list(inspect.signature(AUDIT.train_nested).parameters),["train","protocol","log"])
        with self.assertRaises(TypeError):
            AUDIT.train_nested({}, {}, None, query_truth=np.array([1]))

    def test_real_worker_hook_refuses_deliberate_truth_file_read(self):
        with tempfile.TemporaryDirectory() as directory:
            truth=Path(directory)/"truth.json";truth.write_text('{"p1":"secret_test_label"}')
            code=("import importlib.util,pathlib; "
                  f"s=importlib.util.spec_from_file_location('audit',{str(ROOT/'scripts/audit_upgrade_leakage.py')!r}); "
                  "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
                  f"m.forbid_reads([{str(truth)!r}]); pathlib.Path({str(truth)!r}).read_text()")
            result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("PermissionError",result.stderr)
            self.assertNotIn("secret_test_label",result.stdout)


if __name__=="__main__":unittest.main()
