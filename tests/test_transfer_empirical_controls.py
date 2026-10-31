"""Synthetic isolation fixtures only; no research data or neural fitting."""
from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_cross_plant_transfer_empirical.py"
SPEC = importlib.util.spec_from_file_location("transfer_empirical_controls_under_test", SCRIPT)
adv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adv)


def metrics_fixture():
    return {arm: {profile: {key: .5 for key in adv.METRICS} for profile in adv.PROFILE_IDS} for arm in adv.ARMS}


def complete_score_fixture(out):
    """Synthetic locked evidence files, not fitted research estimators."""
    axis = np.linspace(.21, 1.49, 128)
    source = {"record_ids": np.array([f"S{i:03d}" for i in range(684)]),
              "X": np.zeros((684, 128)), "axis": axis,
              "source_labels": np.asarray([adv.CLASSES[i % 7] for i in range(684)])}
    np.savez_compressed(out / "source.npz", **source)
    truth_dir = out / "scoring_truth"
    truth_dir.mkdir()
    truth = truth_dir / "outer0_original.csv"
    rows = [f"Q{i:03d},{adv.CLASSES[i % 7]}\n" for i in range(171)]
    truth.write_text("record_id,label\n" + "".join(rows))
    adv.write(out / "plan.json", adv.protocol())
    adv.write(out / "preparation.json", {"contexts": [{"stage": "outer", "source_pack": "source.npz"}]})
    paths = ["source.npz", "plan.json", "preparation.json", "scoring_truth/outer0_original.csv"]
    adv.write(out / "frozen_manifest.json", {"protocol_id": adv.PROTOCOL_ID,
        "files_sha256": {name: adv.sha(out / name) for name in paths}, "source_sha256": {}})
    adv.write(out / "root_fit_release_snapshot.json", {"fixture": "No research fits performed"})
    names = ["observed", "query_truth_permuted_trap", *[f"null_{i:02d}_seed{seed}" for i, seed in enumerate(range(1000, 1010))]]
    for name in names:
        run_dir = out / "fits" / name
        inner, finals, signatures = [], {}, {}
        for index in range(20):
            model_dir = run_dir / "models" / f"model{index:02d}"
            model_dir.mkdir(parents=True)
            for filename in ("pipeline.joblib", "training.json", "batch_trace.jsonl.gz", "batch_schedule.json", "fit_scope.jsonl"):
                (model_dir / filename).write_bytes(b"synthetic non-model evidence\n")
            record = {"model_directory": str(model_dir.relative_to(out)),
                "pipeline_sha256": adv.sha(model_dir / "pipeline.joblib"),
                "training_sha256": adv.sha(model_dir / "training.json"),
                "evidence_sha256": adv.fitted_evidence(model_dir)}
            signatures[str(index)] = {"synthetic": True}
            if index < 16:
                inner.append(record)
            else:
                record["predictions"] = [{"record_id": f"Q{i:03d}", "y_predicted": "ZY01"} for i in range(171)]
                finals[adv.ARMS[index - 16]] = record
        adv.write(run_dir / "source_selection_lock.json", {"inner_records": inner, "source_rows": [None] * (684 * 4 * 2)})
        adv.write(run_dir / "round_prediction_lock.json", {"pipeline_fits": 20, "outer_truth_read": False,
            "signatures": signatures, "finals": finals, "selection": {"policy_arm": "A"},
            "source_selection_lock_sha256": adv.sha(run_dir / "source_selection_lock.json")})
    lock = {"outer_truth_parsed": False, "pipeline_fits": 240,
        "frozen_manifest_sha256": adv.sha(out / "frozen_manifest.json"),
        "root_fit_release_sha256": adv.sha(out / "root_fit_release_snapshot.json"),
        "observed_round_lock_sha256": adv.sha(out / "fits/observed/round_prediction_lock.json"),
        "query_truth_trap_round_lock_sha256": adv.sha(out / "fits/query_truth_permuted_trap/round_prediction_lock.json"),
        "nulls": [{"permutation_seed": seed, "run": names[i + 2],
                   "round_lock_sha256": adv.sha(out / "fits" / names[i + 2] / "round_prediction_lock.json")}
                  for i, seed in enumerate(range(1000, 1010))],
        "query_truth_invariance": {"synthetic_fixture": True}, "D_postfit_single_query": {}}
    adv.write(out / "diagnostic_prediction_lock.json", lock)
    return out / "fits" / names[-1] / "models/model19/batch_trace.jsonl.gz"


class TransferEmpiricalBoundaryTests(unittest.TestCase):
    def test_named_real_truth_reads_blocked_but_source_read_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            truth, source = Path(directory) / "truth.csv", Path(directory) / "source.csv"
            truth.write_text("truth\n")
            source.write_text("allowed\n")
            events = []
            with adv.truth_boundary([truth], events):
                self.assertEqual(source.read_text(), "allowed\n")
                for reader in (lambda: open(truth).read(), lambda: truth.read_text(),
                               lambda: io.open(truth).read(), lambda: os.open(truth, os.O_RDONLY)):
                    with self.assertRaises(adv.LeakageError):
                        reader()
            self.assertEqual(len(events), 4)
            self.assertEqual(truth.read_text(), "truth\n")

    def test_scoring_reads_allowed_named_writes_still_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.csv"
            truth.write_text("frozen\n")
            events = []
            with adv.truth_boundary([truth], events, allow_scoring_read=True):
                self.assertEqual(truth.read_text(), "frozen\n")
                for writer in (lambda: truth.write_text("changed"), lambda: open(truth, "a"),
                               lambda: os.open(truth, os.O_RDWR)):
                    with self.assertRaises(adv.LeakageError):
                        writer()
            self.assertEqual(truth.read_text(), "frozen\n")
            self.assertEqual(sum(row["event"] == "QUERY_TRUTH_ACCESS_REJECTED" for row in events), 3)

    def test_hooks_restore_after_exception(self):
        original = (open, io.open, os.open)
        with self.assertRaises(RuntimeError):
            with adv.truth_boundary([], []):
                raise RuntimeError("fixture")
        self.assertEqual((open, io.open, os.open), original)

    def test_query_schema_rejects_label_or_metadata_fields(self):
        p = {"record_ids": np.array(["q1", "q2"]), "X": np.ones((2, 3)), "axis": np.array([.21, .8, 1.49])}
        self.assertIs(adv.strict_query(p), p)
        for key in ("y", "source_labels", "batch", "filename", "origin"):
            with self.subTest(key=key), self.assertRaises(adv.LeakageError):
                adv.strict_query({**p, key: np.array(["0", "1"])})
        with self.assertRaises(adv.LeakageError):
            adv.strict_query({**p, "X": p["X"].astype(object)})
        with self.assertRaises(adv.LeakageError):
            adv.strict_query({**p, "axis": p["axis"][::-1]})

    def test_null_preserves_block_membership_counts_and_rare_classes(self):
        ids = np.array([f"S{i:02d}" for i in range(14)])
        labels = np.array(list(adv.CLASSES) * 2)
        blocks = [ids[:7].tolist(), ids[7:].tolist()]
        original = labels.copy()
        for seed in range(1000, 1010):
            p = adv.permute_source_labels(ids, labels, blocks, seed)
            self.assertEqual(sorted(p[:7]), sorted(labels[:7]))
            self.assertEqual(sorted(p[7:]), sorted(labels[7:]))
            self.assertFalse(np.array_equal(p, labels))
            np.testing.assert_array_equal(p, adv.permute_source_labels(ids, labels, blocks, seed))
        np.testing.assert_array_equal(labels, original)
        with self.assertRaises(adv.LeakageError):
            adv.permute_source_labels(ids, labels, [blocks[0], blocks[0]], 1000)
        with self.assertRaises(adv.LeakageError):
            adv.permute_source_labels(ids, labels, [blocks[0][1:], blocks[1]], 1000)

    def test_common_profile_selected_from_A_only_and_guard_has_real_fallback(self):
        m = metrics_fixture()
        m["B"]["diagnostic4"] = {key: .99 for key in adv.METRICS}
        result = adv.choose_common_profile_and_policy(m)
        self.assertEqual(result["common_profile"], "diagnostic2")
        self.assertEqual(result["policy_arm"], "A")
        m["A"]["diagnostic4"] = {key: .6 for key in adv.METRICS}
        result = adv.choose_common_profile_and_policy(m)
        self.assertEqual(result["common_profile"], "diagnostic4")
        self.assertEqual(result["policy_arm"], "B")
        m["B"]["diagnostic4"]["balanced_accuracy"] = .59
        self.assertEqual(adv.choose_common_profile_and_policy(m)["policy_arm"], "A")

    def test_pooled_source_not_fold_means_and_duplicate_record_rejected(self):
        ids = [f"S{i}" for i in range(7)]
        rows = [{"arm": arm, "profile_id": profile, "record_id": rid, "truth": label, "prediction": label}
                for arm in adv.ARMS for profile in adv.PROFILE_IDS for rid, label in zip(ids, adv.CLASSES, strict=True)]
        result = adv.pooled_source_metrics(rows, ids)
        self.assertTrue(all(result[a][p]["macro_f1"] == 1 for a in adv.ARMS for p in adv.PROFILE_IDS))
        with self.assertRaises(adv.LeakageError):
            adv.pooled_source_metrics(rows + [rows[0]], ids)

    def test_budget_no_fake_samples_and_gate_before_learning(self):
        p = adv.protocol()
        self.assertEqual(p["pipeline_budget"]["total"], 240)
        self.assertEqual(p["pretrain_optimizer_steps"], 720)
        self.assertFalse(p["fresh_confirmation"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            adv.write(out / "frozen_manifest.json", {})
            adv.write(out / "bad_gate.json", {"allow_empirical_training": False})
            with self.assertRaises(adv.LeakageError):
                adv.check_release(out, out / "bad_gate.json")

    def test_last_null_trace_corruption_rejected_before_any_truth_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            last_trace = complete_score_fixture(out)
            last_trace.write_bytes(b"corrupted last evidence file\n")
            with patch.object(adv.csv, "DictReader", side_effect=AssertionError("No truth parse may occur")):
                with self.assertRaisesRegex(adv.LeakageError, "Optimizer traces"):
                    adv.score(out)

    def test_complete_synthetic_score_uses_171_denominator_and_all_ten_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            complete_score_fixture(out)
            result = adv.score(out)
            self.assertEqual(result["score_new_fits"], 0)
            self.assertEqual(result["pipeline_fits"], 240)
            self.assertEqual(len(result["observed_metrics"]), 5)
            self.assertEqual(len(result["null_metrics"]["C"]), 10)
            self.assertAlmostEqual(result["observed_metrics"]["A"]["accuracy"], 25 / 171)
            self.assertEqual(result["null_summary"]["A"]["accuracy"]["empirical_tail_plus_one"], 1)


if __name__ == "__main__":
    unittest.main()
