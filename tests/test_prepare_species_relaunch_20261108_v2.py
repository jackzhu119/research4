"""Grant, role and complete-group checks without fitting any scientific model."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "scripts/prepare_species_relaunch_20261108_v2.py"
spec = importlib.util.spec_from_file_location("relaunch_prepare", MODULE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def rows():
    result = []
    kinds = ["枣仁"] * 216 + ["酸枣仁"] * 300 + ["黄芪"] * 241 + ["天麻"] * 242
    for i in range(1672):
        result.append({"record_id": f"sample-{i:04d}", "physical_sample_id": f"physical-{i:04d}",
                       "role": "train" if i < 999 else "calibration" if i < 1333 else "prediction",
                       "species": kinds[i] if i < 999 else "unused-holdout-kind"})
    return result


def matrix():
    jobs = []
    for task in ("acid_origins", "huangqi_origins", "tianma_origins", "binary_kinds", "four_kinds"):
        for view in ("alpha", "n", "n_alpha"):
            for repeat in (20261031, 20261108, 42):
                for fold in range(5):
                    jobs.append({"id": f"c-{task}-{view}-{repeat}-{fold}", "kind": "classical",
                                 "command": ["python", "worker.py", "--task", task, "--representation", view,
                                             "--repeat-seed", str(repeat), "--outer-fold", str(fold)]})
    for target in ("suanzaoren", "huangqi", "tianma"):
        for steps in (120, 480, 1920):
            for profile in ("p12_e96_full", "p12_e192_full", "p12_e384_full", "p24_e96_full",
                            "p24_e192_full", "p24_e384_full", "p12_e192_frozen", "p24_e192_frozen"):
                for fold in range(5):
                    for seed in (42, 43, 44):
                        jobs.append({"id": f"n-{target}-{steps}-{profile}-{fold}-{seed}", "kind": "neural",
                                     "command": ["python", "worker.py", "--target", target, "--pretrain-steps", str(steps),
                                                 "--profile", profile, "--outer-fold", str(fold), "--seed", str(seed)]})
    return list(reversed(jobs))


def argument(job, name):
    command = job["command"]
    return command[command.index(name) + 1]


class AuthorizationTests(unittest.TestCase):
    def test_explicit_new_90000_grant_and_finite_allocations(self):
        self.assertEqual(prep.RUN_ID, "species-relaunch-20261108-v2")
        prep.validate_authorization({"authorized_total_cpu_seconds": 90000})
        prep.validate_authorization({"authorized_cpu_seconds": 90000,
                                     "authorized_total_cpu_seconds": 90000})
        self.assertEqual(prep.INNER_CAP, 88000)
        self.assertEqual(prep.INNER_CAP + prep.PREPARATION_RESERVE + prep.WATCH_RESERVE, 90000)

    def test_missing_old_or_conflicting_grants_fail(self):
        for record in ({}, {"authorized_cpu_seconds": 88000}, {"authorized_cpu_seconds": True},
                       {"authorized_cpu_seconds": "90000"},
                       {"authorized_cpu_seconds": 90000, "authorized_total_cpu_seconds": 90001}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                prep.validate_authorization(record)


class IsolationTests(unittest.TestCase):
    def test_exact_training_only_pool_sorted_without_mutation(self):
        source = rows()
        before = copy.deepcopy(source)
        training = prep.select_training(list(reversed(source)))
        self.assertEqual(len(training), 999)
        self.assertEqual(source, before)
        self.assertTrue(all(row["role"] == "train" for row in training))
        self.assertEqual([r["record_id"] for r in training], sorted(r["record_id"] for r in training))
        self.assertFalse({r["record_id"] for r in training} &
                         {r["record_id"] for r in source if r["role"] != "train"})

    def test_master_role_changes_fail(self):
        source = rows()
        source[0]["role"] = "prediction"
        with self.assertRaises(ValueError):
            prep.select_training(source)

    def test_duplicated_master_identity_fails(self):
        source = rows()
        source[-1]["record_id"] = source[0]["record_id"]
        with self.assertRaises(ValueError):
            prep.select_training(source)

    def test_duplicate_physical_training_specimen_fails(self):
        source = rows()
        source[1]["physical_sample_id"] = source[0]["physical_sample_id"]
        with self.assertRaises(ValueError):
            prep.select_training(source)

    def test_reintroducing_kind_mislabel_fails(self):
        source = rows()
        source[216]["species"] = "枣仁"
        with self.assertRaises(ValueError):
            prep.select_training(source)

    def test_array_fingerprints_are_content_and_shape_sensitive(self):
        array = np.arange(12, dtype=np.float64).reshape(3, 4)
        before = array.copy()
        self.assertEqual(prep.fingerprint(array), prep.fingerprint(array.copy()))
        self.assertNotEqual(prep.fingerprint(array), prep.fingerprint(array + 1))
        self.assertNotEqual(prep.fingerprint(array), prep.fingerprint(array.reshape(4, 3)))
        np.testing.assert_array_equal(array, before)
        with self.assertRaises(ValueError):
            prep.fingerprint(np.asarray([object()], dtype=object))


class MatrixTests(unittest.TestCase):
    def test_exact_matrix_retained_no_mutation(self):
        source = matrix()
        before = copy.deepcopy(source)
        ordered = prep.reorder_jobs(source)
        self.assertEqual(source, before)
        self.assertEqual(len(ordered), 1305)
        self.assertEqual({job["id"] for job in source}, {job["id"] for job in ordered})
        self.assertEqual(sum(job["kind"] == "classical" for job in ordered), 225)
        self.assertEqual(sum(job["kind"] == "neural" for job in ordered), 1080)

    def test_initial_groups_complete_and_acid_first(self):
        ordered = prep.reorder_jobs(matrix())
        first = ordered[:5]
        self.assertTrue(all(job["kind"] == "classical" for job in first))
        self.assertEqual({argument(j, "--task") for j in first}, {"acid_origins"})
        self.assertEqual({argument(j, "--representation") for j in first}, {"alpha"})
        self.assertEqual({argument(j, "--repeat-seed") for j in first}, {"20261031"})
        self.assertEqual({argument(j, "--outer-fold") for j in first}, set(map(str, range(5))))
        smoke = ordered[5:20]
        self.assertTrue(all(job["kind"] == "neural" for job in smoke))
        self.assertEqual({argument(j, "--target") for j in smoke}, {"suanzaoren"})
        self.assertEqual({argument(j, "--profile") for j in smoke}, {"p12_e96_full"})
        self.assertEqual({argument(j, "--pretrain-steps") for j in smoke}, {"120"})
        self.assertEqual({(argument(j, "--outer-fold"), argument(j, "--seed")) for j in smoke},
                         {(str(f), str(s)) for f in range(5) for s in (42, 43, 44)})

    def test_every_remaining_group_keeps_all_folds(self):
        ordered = prep.reorder_jobs(matrix())
        classic = [j for j in ordered if j["kind"] == "classical"]
        for start in range(0, len(classic), 5):
            group = classic[start:start + 5]
            self.assertEqual(len({(argument(j, "--task"), argument(j, "--representation"),
                                   argument(j, "--repeat-seed")) for j in group}), 1)
            self.assertEqual({argument(j, "--outer-fold") for j in group}, set(map(str, range(5))))
        neural = [j for j in ordered if j["kind"] == "neural"]
        for start in range(0, len(neural), 15):
            group = neural[start:start + 15]
            self.assertEqual(len({(argument(j, "--target"), argument(j, "--profile"),
                                   argument(j, "--pretrain-steps")) for j in group}), 1)
            self.assertEqual({(argument(j, "--outer-fold"), argument(j, "--seed")) for j in group},
                             {(str(f), str(s)) for f in range(5) for s in (42, 43, 44)})

    def test_missing_duplicate_and_invented_parameters_fail(self):
        source = matrix()
        with self.assertRaises(ValueError):
            prep.reorder_jobs(source[:-1])
        duplicate = copy.deepcopy(source)
        duplicate[1] = duplicate[0]
        with self.assertRaises(ValueError):
            prep.reorder_jobs(duplicate)
        changed = copy.deepcopy(source)
        changed[0]["command"][changed[0]["command"].index("--seed") + 1] = "45"
        with self.assertRaises(ValueError):
            prep.reorder_jobs(changed)

    def test_original_preparer_source_is_hash_checked_without_fitting(self):
        original = prep.original_preparer()
        self.assertEqual(original.BASE, prep.BASE)
        self.assertEqual(prep.digest(prep.ORIGINAL), prep.ORIGINAL_SHA)


if __name__ == "__main__":
    unittest.main()
