"""Independent no-fit replay of main source pipelines and frozen selection math.

Only inner truth is parsed. No production selector, training function, optimizer
step or outer truth is used. Completed pipelines may be reviewed while the
frozen source experiment continues; final PASS requires its complete lock.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from replay_helpers import ARMS, CLASSES, Review, metric, read, sha


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT / "reports/cross-plant-deep-transfer-20261031-v1/neural"
PROGRESS = ROOT / "outputs/cross-plant-deep-transfer-main-source-review-progress.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def csv_rows(path):
    opening = gzip.open if str(path).endswith(".gz") else open
    with opening(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def pack(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def recompute(y, predicted):
    result = metric(y, predicted)
    result.update(correct=int(np.sum(np.asarray(y) == np.asarray(predicted))), total=len(y),
                  class_order=list(CLASSES),
                  class_counts={c: int(np.sum(np.asarray(y) == c)) for c in CLASSES},
                  class_recall={c: result["per_class"][c]["recall"] for c in CLASSES})
    return result


def nested(review, actual, expected, message):
    if isinstance(actual, dict):
        review.check(isinstance(expected, dict) and set(actual) == set(expected), message + " keys")
        for key in actual:
            nested(review, actual[key], expected[key], message + "." + str(key))
    elif isinstance(actual, list):
        review.check(isinstance(expected, list) and len(actual) == len(expected), message + " list length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            nested(review, a, b, message + f"[{i}]")
    elif isinstance(actual, bool) or actual is None or isinstance(actual, str):
        review.check(actual == expected, message)
    else:
        review.close(actual, expected, message)


def binding_keys():
    return {"training_sha256": "training.json", "pipeline_sha256": "pipeline.joblib",
            "prediction_sha256": "predictions.csv.gz", "inference_audit_sha256": "inference_audit.json",
            "truth_access_audit_sha256": "truth_access_audit.json"}


def inference_close(review, a, b, message):
    a, b = np.asarray(a), np.asarray(b)
    review.check(a.shape == b.shape, message + " shape")
    review.values += a.size
    if a.size:
        review.maximum = max(review.maximum, float(np.max(np.abs(a - b))))
    review.check(np.allclose(a, b, atol=1e-5, rtol=1e-5), message)


def independently_select(table, profiles):
    # Reimplemented from the locked written rule, without calling production.
    order = lambda s: (-s["mean"]["macro_f1"], -s["mean"]["balanced_accuracy"],
                       -s["mean"]["accuracy"], s["profile"]["id"])
    baseline = sorted((table["A", p["id"]] for p in profiles), key=order)[0]
    selected_profile = baseline["profile"]
    eligible = []
    for arm in ("Aextra", "B", "C"):
        row = table[arm, selected_profile["id"]]
        m, b = row["mean"], baseline["mean"]
        if (m["macro_f1"] + 1e-12 >= b["macro_f1"] + .005
                and m["balanced_accuracy"] + 1e-12 >= b["balanced_accuracy"]
                and m["accuracy"] + 1e-12 >= b["accuracy"]):
            eligible.append(row)
    winner = sorted(eligible, key=lambda s: (*order(s)[:3], ARMS.index(s["arm"])))[0] if eligible else baseline
    return {"shared_profile": selected_profile, "profile_chosen_only_from_A": True,
            "policy_arm": winner["arm"],
            "source_means_at_shared_profile": {arm: table[arm, selected_profile["id"]]["mean"] for arm in ARMS},
            "outer_used_for_selection": False}


def main(watch):
    torch.set_num_threads(1)
    review = Review()
    preparation, protocol, manifest = (read(MAIN / f) for f in ("preparation.json", "protocol.json", "run_manifest.json"))
    review.check(sha(MAIN / "protocol.json") == manifest["protocol_sha256"], "frozen protocol bytes")
    review.check(sha(MAIN / "preparation.json") == manifest["preparation_sha256"], "frozen preparation bytes")
    for name, expected in manifest["source_sha256"].items():
        review.check(sha(ROOT / name) == expected, "frozen actual code " + name)
    review.check(preparation["unit_compatibility_verified"] is False, "unknown physical units not upgraded")
    review.check(preparation["independent_confirmation_established"] is False, "exposed target not confirmation")
    seeds, profiles = protocol["seeds"], protocol["profiles"]
    review.check(seeds == [42, 43, 44] and protocol["arms"] == list(ARMS), "all fixed seeds/arms")
    review.check(protocol["policy_guard"] == {"macro_f1_margin": .005, "balanced_accuracy_not_lower": True,
                                             "accuracy_not_lower": True, "tolerance": 1e-12,
                                             "fixed_arm_order": ["Aextra", "B", "C"]}, "fixed conservative guard")
    contexts = [c for c in preparation["contexts"] if c["stage"] == "inner"]
    review.check(len(contexts) == 10 and len(profiles) == 2, "ten inner contexts/two profiles")
    all_truth_paths = {str((MAIN / c["truth_path"]).resolve()) for c in preparation["contexts"]}
    inner_inputs, truth, allowed = {}, {}, {}
    for context in contexts:
        split = context["split_id"]
        for key in ("source", "query"):
            review.check(sha(MAIN / context[key + "_pack"]) == context[key + "_sha256"], "current inner " + key + " bytes")
        source, query = pack(MAIN / context["source_pack"]), pack(MAIN / context["query_pack"])
        review.check(set(source) == {"X", "axis", "record_ids", "source_labels"}, "strict source schema")
        review.check(set(query) == {"X", "axis", "record_ids"}, "strict query schema excludes labels")
        review.check(source["record_ids"].tolist() == context["source_ids"], "source membership/order")
        review.check(query["record_ids"].tolist() == context["query_ids"], "query membership/order")
        review.check(len(source["record_ids"]) == len(query["record_ids"]) == 342, "source/query342 each")
        review.check(not set(source["record_ids"]) & set(query["record_ids"]), "current source/query disjoint")
        review.check(sha(MAIN / context["truth_path"]) == context["truth_sha256"], "inner truth unchanged before parse")
        inner_truth = csv_rows(MAIN / context["truth_path"])
        truth[split] = {r["record_id"]: r["label"] for r in inner_truth}
        review.check(len(truth[split]) == len(inner_truth) == 342 and set(truth[split]) == set(query["record_ids"]), "inner truth precise routing")
        inner_inputs[split] = (source, query)
        allowed[split] = context
    for outer in range(5):
        subset = [c for c in contexts if c["outer_fold"] == outer]
        outer_source = next(c["source_ids"] for c in preparation["contexts"] if c["stage"] == "outer" and c["outer_fold"] == outer)
        review.check(set(subset[0]["query_ids"]).isdisjoint(subset[1]["query_ids"]), "two inner query sets disjoint")
        review.check(set(subset[0]["query_ids"]) | set(subset[1]["query_ids"]) == set(outer_source), "both inner queries equal684 current outer source")
    auxiliary = {}
    for name, entry in preparation["auxiliary"].items():
        review.check(sha(MAIN / entry["path"]) == entry["sha256"], "external pack bytes " + name)
        auxiliary[name] = pack(MAIN / entry["path"])
    inventory = [(c, arm, profile, seed) for c in contexts for arm in ARMS for profile in profiles for seed in seeds]
    review.check(len(inventory) == 240, "complete240 inventory")
    completed, metadata_by_key, metrics_by_key, rows_by_key, hashes_by_key = set(), {}, {}, {}, {}
    signatures = {}
    helper_hash = sha(HERE / "replay_helpers.py")
    start = utc()
    while len(completed) < 240:
        prior = len(completed)
        for context, arm, profile, seed in inventory:
            key = (context["split_id"], arm, profile["id"], seed)
            if key in completed:
                continue
            directory = MAIN / "source_models" / context["split_id"] / arm / profile["id"] / f"seed{seed}"
            if not (directory / "truth_access_audit.json").exists():
                continue  # The final flag is written only after weights and predictions.
            source, query = inner_inputs[context["split_id"]]
            rows = csv_rows(directory / "predictions.csv.gz")
            native_rows = [{**row, **{"score_" + c: float(row["score_" + c]) for c in CLASSES}} for row in rows]
            fitted, metadata, signature, (predicted, logits) = review.pipeline(directory, source, auxiliary, query, native_rows)
            review.check(metadata["arm"] == arm and metadata["profile"] == profile and metadata["seed"] == seed, "expected locked candidate identity")
            review.check(metadata["split_id"] == context["split_id"] and metadata["query_received_by_training"] is False, "current training context/no query API")
            review.check(metadata["source_versions"] == manifest["source_sha256"], "actual pipeline frozen dependency hashes")
            inf = read(directory / "inference_audit.json")
            review.check(inf["status"] == "TECHNICAL_CHECK_PASS" and inf["parameters_and_buffers_unchanged"] is True, "saved inference and BN evidence")
            review.check(inf["samples_checked"] == 7 and inf["atol"] == inf["rtol"] == 1e-5, "predeclared finite source inference scope")
            access = read(directory / "truth_access_audit.json")
            review.check(access["named_paths"] == len(all_truth_paths) == 15, "all15 named truth traps")
            review.check({r["path"] for r in access["denied_attempts"]} == all_truth_paths and all(r["denied"] for r in access["denied_attempts"]), "actual named-file read refusals")
            review.check(access["OS_sandbox"] is False, "named Python hooks not OS isolation")
            # Independently cover the whole query in reverse and a single row.
            reverse = {"X": query["X"][::-1].copy(), "axis": query["axis"], "record_ids": query["record_ids"][::-1].copy()}
            reverse_pred, reverse_logits = review.native(fitted, reverse)
            inference_close(review, logits, reverse_logits[::-1], "whole-query reverse invariance")
            review.check(predicted.tolist() == reverse_pred[::-1].tolist(), "whole-query reverse labels")
            single = {"X": query["X"][:1], "axis": query["axis"], "record_ids": query["record_ids"][:1]}
            single_pred, single_logits = review.native(fitted, single)
            inference_close(review, logits[:1], single_logits, "single-row inference invariance")
            review.check(predicted[:1].tolist() == single_pred.tolist(), "single-row labels")
            current_truth = truth[context["split_id"]]
            metrics_by_key[key] = recompute([current_truth[r["record_id"]] for r in rows], predicted)
            rows_by_key[key] = rows
            metadata_by_key[key], signatures[key] = metadata, signature
            hashes_by_key[key] = {field: sha(directory / name) for field, name in binding_keys().items()}
            completed.add(key)
            del fitted
        if len(completed) != prior:
            PROGRESS.parent.mkdir(exist_ok=True)
            PROGRESS.write_text(json.dumps({"utc": utc(), "source_pipeline_weights_replayed": len(completed),
                "planned": 240, "checks": review.checks, "numeric_values": review.values,
                "maximum_difference": review.maximum, "new_audit_fits": 0, "main_outer_truth_read": False}) + "\n")
            print(json.dumps({"replayed_source_pipelines": len(completed), "planned": 240, "new_audit_fits": 0}), flush=True)
        if len(completed) < 240:
            if not watch:
                raise FileNotFoundError("Incomplete source weights; run --watch while source experiment continues")
            time.sleep(5)
    while not (MAIN / "selection_lock.json").exists():
        if not watch:
            raise FileNotFoundError("All weights replayed but full source lock not yet present")
        time.sleep(1)
    review.check(sha(HERE / "replay_helpers.py") == helper_hash, "frozen no-fit replay helper")
    lock = read(MAIN / "selection_lock.json")
    review.check(lock["source_pipeline_fits"] == 240 and lock["outer_used_for_selection"] is False, "complete source-only selection declaration")
    for key, filename in (("source_release_sha256", "source_release_snapshot.json"), ("run_manifest_sha256", "run_manifest.json"),
                          ("source_records_sha256", "source_records.json"), ("source_predictions_sha256", "source_predictions.csv.gz"),
                          ("candidate_summaries_sha256", "candidate_summaries.json")):
        review.check(lock[key] == sha(MAIN / filename), "selection lock binding " + filename)
    records = read(MAIN / "source_records.json")
    review.check(len(records) == 240, "actual240 source records")
    record_keys = set()
    for record in records:
        key = (record["split_id"], record["arm"], record["profile"], record["seed"])
        review.check(key in completed and key not in record_keys, "unique known actual source record")
        record_keys.add(key)
        for field, name in binding_keys().items():
            actual = sha(MAIN / record["model_directory"] / name)
            review.check(actual == record[field] == hashes_by_key[key][field], "incremental-to-lock stable evidence " + name)
        review.close(record["elapsed_seconds"], metadata_by_key[key]["elapsed_seconds"], "actual timing metadata")
        review.check(record["target_steps"] == metadata_by_key[key]["target_steps"] and record["pretrain_steps"] == metadata_by_key[key]["pretrain_steps"], "actual source update counts")
        nested(review, metrics_by_key[key], record["metrics"], "independent inner metric")
    combined = csv_rows(MAIN / "source_predictions.csv.gz")
    review.check(len(combined) == 82080, "all240x342 saved source prediction rows")
    grouped = defaultdict(list)
    for row in combined:
        split = f"outer{int(row['outer_fold']):02d}.inner{int(row['inner_fold']):02d}"
        key = (split, row["arm"], row["profile"], int(row["seed"]))
        grouped[key].append(row)
        review.check(row["y_true"] == truth[split][row["record_id"]], "combined inner ground truth")
    review.check(set(grouped) == completed, "combined prediction exact240 group coverage")
    for key in completed:
        rows = grouped[key]
        expected = rows_by_key[key]
        review.check(len(rows) == len(expected) == 342, "complete saved group342")
        for row, prior in zip(rows, expected):
            review.check({k: row[k] for k in prior} == prior, "combined row exact native prediction provenance")
    # Strong pretraining and initialization invariance across contexts/arms.
    fixed_pretrain = defaultdict(set)
    source_only_pretrain = defaultdict(set)
    optimizer_initial, zy_initial, orders = defaultdict(set), defaultdict(set), defaultdict(set)
    for (split, arm, profile_id, seed), m in metadata_by_key.items():
        if arm in ("B", "C"):
            fixed_pretrain[arm, seed].add(m["pretrained_encoder_sha256"])
        if arm == "Aextra":
            source_only_pretrain[split, seed].add(m["pretrained_encoder_sha256"])
        optimizer_initial[seed].add(m["fresh_target_optimizer_initial_sha256"])
        zy_initial[seed].add(m["initial_zy_head_sha256"])
        orders[split, profile_id, seed].add(m["target_order_sha256"])
    for family, entries in (("B/C fixed external pretrain", fixed_pretrain), ("Aextra same-context across profiles", source_only_pretrain),
                            ("fresh optimizer per seed", optimizer_initial), ("fresh ZY head per seed", zy_initial),
                            ("paired target orders", orders)):
        for key, value in entries.items():
            review.check(len(value) == 1, family + " " + str(key))
    computed_summaries, computed_choices = [], []
    expected_summaries = {(r["outer_fold"], r["arm"], r["profile"]["id"]): r for r in read(MAIN / "candidate_summaries.json")}
    for outer in range(5):
        table = {}
        outer_contexts = [c for c in contexts if c["outer_fold"] == outer]
        for arm in ARMS:
            for profile in profiles:
                per_seed = {}
                for seed in seeds:
                    rows = [r for c in outer_contexts for r in grouped[c["split_id"], arm, profile["id"], seed]]
                    review.check(len(rows) == len({r["record_id"] for r in rows}) == 684, "pooled current-source OOF684")
                    per_seed[str(seed)] = recompute([r["y_true"] for r in rows], [r["y_predicted"] for r in rows])
                summary = {"outer_fold": outer, "arm": arm, "profile": profile, "per_seed": per_seed,
                           "mean": {m: float(np.mean([per_seed[str(seed)][m] for seed in seeds]))
                                    for m in ("macro_f1", "balanced_accuracy", "accuracy")}}
                nested(review, summary, expected_summaries[outer, arm, profile["id"]], "independent pooled source means")
                computed_summaries.append(summary)
                table[arm, profile["id"]] = summary
        choice = {"outer_fold": outer, **independently_select(table, profiles)}
        nested(review, choice, lock["outer_folds"][outer], "independent A-only shared-profile and guarded policy")
        computed_choices.append(choice)
    review.check(len(computed_summaries) == 40 and len(expected_summaries) == 40 and len(lock["outer_folds"]) == 5, "complete40profile summaries/5choices")
    events = [json.loads(line) for line in (MAIN / "events.jsonl").read_text().splitlines()]
    review.check(not any(e["event"] in {"outer_prediction_start", "all_outer_predictions_locked", "scoring_complete"} for e in events), "main outer not released before source review")
    report = {"started_at_utc": start, "completed_at_utc": utc(),
              "status": "SOURCE_SELECTION_NATIVE_REPLAY_PASS_EXPLORATORY_ONLY", "new_audit_fits": 0,
              "main_outer_truth_read": False, "checks": review.checks, "numeric_values": review.values,
              "maximum_numeric_difference": review.maximum, "actual_source_pipeline_weights": 240,
              "saved_source_prediction_rows": 82080, "pooled684_cases": 120,
              "counts": dict(review.counts), "source_shared_profile_and_policy": computed_choices,
              "strong_metadata_invariance_groups": {"external_pretraining_arm_seed": len(fixed_pretrain),
                  "Aextra_context_seed_across_profiles": len(source_only_pretrain), "fresh_optimizer_seed": len(optimizer_initial),
                  "fresh_ZYhead_seed": len(zy_initial), "paired_target_order_context_profile_seed": len(orders)},
              "selection_lock_sha256": sha(MAIN / "selection_lock.json"), "source_records_sha256": sha(MAIN / "source_records.json"),
              "source_predictions_sha256": sha(MAIN / "source_predictions.csv.gz"),
              "candidate_summaries_sha256": sha(MAIN / "candidate_summaries.json"),
              "reviewer_script_sha256": sha(__file__), "replay_helper_sha256": helper_hash,
              "frozen_source_sha256": manifest["source_sha256"],
              "limitations": ["This replays saved artifacts; it does not rerun optimization or externally certify provenance.",
                  "Named Python file hooks and finite inference checks are not an OS sandbox or absolute zero-leakage proof.",
                  "All target samples are historically exposed single-source-batch development data; external ancestry/calibration remain uncertain."]}
    with (HERE / "main_selection_review.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": review.checks, "maximum_numeric_difference": review.maximum,
                      "new_audit_fits": 0, "selection_lock_sha256": report["selection_lock_sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    main(parser.parse_args().watch)
