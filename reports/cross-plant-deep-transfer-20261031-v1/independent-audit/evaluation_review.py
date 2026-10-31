"""Replay frozen outer transfer models/aliases and all metrics without fitting."""
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from replay_helpers import Review, ARMS, CLASSES, sha, read, metric, compare_metric, tensor_state

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "neural"
ROLES = (*ARMS, "source_policy")


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def csvread(path):
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(1)
    review = Review()
    def pointwise_close(a, b, message):
        a, b = np.asarray(a), np.asarray(b)
        review.check(a.shape == b.shape, message + " shape")
        review.values += a.size
        if a.size:
            review.maximum = max(review.maximum, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
        review.check(np.allclose(a, b, atol=1e-5, rtol=1e-5), message)
    manifest, prep = read(RUN / "run_manifest.json"), read(RUN / "preparation.json")
    selection, lock, scoring = read(RUN / "selection_lock.json"), read(RUN / "prediction_lock.json"), read(RUN / "scoring_manifest.json")
    review.check(manifest["protocol_sha256"] == sha(RUN / "protocol.json") and manifest["preparation_sha256"] == sha(RUN / "preparation.json"), "frozen protocol and prepared arrays")
    for relative, expected in manifest["source_sha256"].items():
        review.check(sha(ROOT / relative) == expected, "frozen implementation")
    for key, filename in [("prediction_sha256", "outer_predictions.csv.gz"), ("outer_model_records_sha256", "outer_model_records.json"),
                          ("outer_role_aliases_sha256", "outer_role_aliases.json"), ("selection_lock_sha256", "selection_lock.json"),
                          ("outer_release_sha256", "outer_release_snapshot.json"), ("run_manifest_sha256", "run_manifest.json"), ("protocol_sha256", "protocol.json")]:
        review.check(lock[key] == sha(RUN / filename), "prediction lock dependency " + key)
    review.check(lock["actual_outer_models"] == 60 and lock["alias_cases"] == 15 and lock["prediction_rows"] == 12825 and lock["outer_truth_parsed"] is False, "60 actual /15 aliases /12825 rows")
    review.check(scoring["new_training_fits"] == 0 and scoring["real_target_observations"] == 855 and scoring["three_seeds_are_not_new_specimens"] is True, "score nofit and correct independent unit")
    review.check(scoring["prediction_lock_sha256"] == sha(RUN / "prediction_lock.json"), "score after prediction lock")
    for key, filename in [("source_release_sha256", "source_release_snapshot.json"), ("run_manifest_sha256", "run_manifest.json"),
                          ("source_records_sha256", "source_records.json"), ("source_predictions_sha256", "source_predictions.csv.gz"),
                          ("candidate_summaries_sha256", "candidate_summaries.json")]:
        review.check(selection[key] == sha(RUN / filename), "original source-selection bytes unchanged")
    release = read(RUN / "outer_release_snapshot.json")
    review.check(release["allow_outer_evaluation"] is True and bool(release["created_at_utc"]), "dated root release")
    for key, filename in [("selection_lock_sha256", "selection_lock.json"), ("protocol_sha256", "protocol.json"), ("run_manifest_sha256", "run_manifest.json")]:
        review.check(release[key] == sha(RUN / filename), "outer release bound source choices")
    profiles = {fold["outer_fold"]: fold for fold in selection["outer_folds"]}
    contexts = {c["split_id"]: c for c in prep["contexts"] if c["stage"] == "outer"}
    auxiliary = {}
    for name, spec in prep["auxiliary"].items():
        review.check(sha(RUN / spec["path"]) == spec["sha256"], "frozen auxiliary arrays")
        auxiliary[name] = load(RUN / spec["path"])
    rows, records, aliases = csvread(RUN / "outer_predictions.csv.gz"), read(RUN / "outer_model_records.json"), read(RUN / "outer_role_aliases.json")
    review.check(len(rows) == 12825 and len(records) == 60 and len(aliases) == 15, "actual role inventory")
    native_rows, paired_fields = [], {}
    for record in records:
        context = contexts[record["split_id"]]
        source, query = load(RUN / context["source_pack"]), load(RUN / context["query_pack"])
        for role in ("source", "query"):
            review.check(sha(RUN / context[role + "_pack"]) == context[role + "_sha256"], "outer fixed input bytes")
        review.check(source["record_ids"].tolist() == context["source_ids"] and query["record_ids"].tolist() == context["query_ids"], "outer fixed membership/order")
        review.check(not set(source["record_ids"]) & set(query["record_ids"]), "actual source/query disjoint")
        directory = RUN / record["model_directory"]
        for key, filename in [("training_sha256", "training.json"), ("pipeline_sha256", "pipeline.joblib"), ("prediction_sha256", "predictions.csv.gz"),
                              ("inference_audit_sha256", "inference_audit.json"), ("truth_access_audit_sha256", "truth_access_audit.json")]:
            review.check(record[key] == sha(directory / filename), "actual outer evidence dependencies")
        fold, arm, seed = record["outer_fold"], record["arm"], record["seed"]
        stored = [r for r in rows if int(r["outer_fold"]) == fold and r["role"] == arm and int(r["seed"]) == seed]
        native_stored = [{key: float(value) if key.startswith("score_") else value for key, value in r.items()} for r in stored]
        fitted, meta, signature, (predicted, logits) = review.pipeline(directory, source, auxiliary, query, native_stored)
        review.check(meta["profile"] == profiles[fold]["shared_profile"] and record["profile"] == profiles[fold]["shared_profile"]["id"], "same A-selected target epoch for every arm")
        review.check(meta["arm"] == arm and meta["seed"] == seed, "model role bound")
        native_rows.extend({"outer_fold": fold, "role": arm, "seed": seed, "record_id": rid,
                            "prediction": str(pred)} for rid, pred in zip(query["record_ids"], predicted))
        fields = tuple(meta[key] for key in ("initial_encoder_sha256", "initial_zy_head_sha256", "fresh_target_optimizer_initial_sha256", "target_order_sha256"))
        paired_fields.setdefault((fold, seed), []).append(fields)
        before = tensor_state(fitted.model)
        reverse = {key: value[::-1].copy() if key != "axis" else value.copy() for key, value in query.items()}
        reversed_predictions, reversed_scores = review.native(fitted, reverse)
        review.check(predicted.tolist() == reversed_predictions[::-1].tolist(), "reverse query order labels invariant")
        pointwise_close(logits, reversed_scores[::-1], "reverse query order logits invariant")
        single = {"X": query["X"][:1], "axis": query["axis"], "record_ids": query["record_ids"][:1]}
        one_prediction, one_logits = review.native(fitted, single)
        review.check(one_prediction[0] == predicted[0], "single query native label invariant")
        pointwise_close(one_logits, logits[:1], "single query native logits invariant")
        review.check(tensor_state(fitted.model) == before, "inference preserves all learned state")
        truth_probe = read(directory / "truth_access_audit.json")
        review.check(truth_probe["named_paths"] == 15 and len(truth_probe["denied_attempts"]) >= 15 and truth_probe["OS_sandbox"] is False, "actual named truth I/O probes")
    for key, values in paired_fields.items():
        review.check(len(values) == 4 and len(set(values)) == 1, "paired init/head/optimizer/minibatch schedule all4arms")
    for alias in aliases:
        fold, seed = alias["outer_fold"], alias["seed"]
        arm = profiles[fold]["policy_arm"]
        review.check(alias["arm"] == arm and alias["role"] == "source_policy", "policy alias chosen source-only")
        corresponding = next(r for r in records if (r["outer_fold"], r["seed"], r["arm"]) == (fold, seed, arm))
        review.check(alias["model_directory"] == corresponding["model_directory"], "policy reuses exact actual model")
        actual = [r for r in rows if int(r["outer_fold"]) == fold and int(r["seed"]) == seed and r["role"] == "source_policy"]
        base = [r for r in rows if int(r["outer_fold"]) == fold and int(r["seed"]) == seed and r["role"] == arm]
        review.check([{k: v for k, v in r.items() if k != "role"} for r in actual] == [{k: v for k, v in r.items() if k != "role"} for r in base], "exact policy prediction/logit alias")
        native_rows.extend({**r, "role": "source_policy"} for r in native_rows.copy() if r["outer_fold"] == fold and r["seed"] == seed and r["role"] == arm)
    truth = {}
    for context in contexts.values():
        review.check(sha(RUN / context["truth_path"]) == context["truth_sha256"], "all fixed outer truth bytes")
        with (RUN / context["truth_path"]).open(newline="") as handle:
            part = list(csv.DictReader(handle))
        review.check(not set(truth) & {r["record_id"] for r in part}, "one outer truth per target observation")
        truth.update({r["record_id"]: r["label"] for r in part})
    review.check(len(truth) == 855 and len(native_rows) == 12825, "unique855 denominator per role/seed")
    saved = {(r["role"], r["seed"]): r["metrics"] for r in read(RUN / "pooled_metrics_by_seed.json")}
    saved_fold = {(r["role"], r["seed"], r["outer_fold"]): r["metrics"] for r in read(RUN / "fold_metrics.json")}
    saved_delta = {(r["role"], r["seed"]): r for r in read(RUN / "paired_scratch_deltas.json")}
    independent, deltas = [], []
    for role in ROLES:
        for seed in (42, 43, 44):
            part = [r for r in native_rows if r["role"] == role and r["seed"] == seed]
            ids = [r["record_id"] for r in part]
            review.check(len(ids) == len(set(ids)) == 855 and set(ids) == set(truth), "every855 once per role/seed")
            metrics = metric([truth[rid] for rid in ids], [r["prediction"] for r in part])
            compare_metric(review, saved[role, seed], metrics, "pooled native metric")
            independent.append({"role": role, "seed": seed, "metrics": metrics})
            for fold in range(5):
                fpart = [r for r in part if r["outer_fold"] == fold]
                m = metric([truth[r["record_id"]] for r in fpart], [r["prediction"] for r in fpart])
                compare_metric(review, saved_fold[role, seed, fold], m, "fold native metric")
            if role != "A":
                base = {r["record_id"]: r["prediction"] for r in native_rows if r["role"] == "A" and r["seed"] == seed}
                corrected = sum(base[r["record_id"]] != truth[r["record_id"]] and r["prediction"] == truth[r["record_id"]] for r in part)
                broken = sum(base[r["record_id"]] == truth[r["record_id"]] and r["prediction"] != truth[r["record_id"]] for r in part)
                delta = saved_delta[role, seed]
                review.check(delta["corrected"] == corrected and delta["broken"] == broken, "corrected/broken actual counts")
                for key in ("accuracy", "macro_f1", "balanced_accuracy"):
                    review.close(delta[key + "_delta"], metrics[key] - saved["A", seed][key], "paired metric delta")
                review.close((corrected - broken) / 855, delta["accuracy_delta"], "net corrected equals accuracy change")
                deltas.append({**delta, "independently_verified": True})
    summaries = read(RUN / "seed_summary.json")
    for summary in summaries:
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [r["metrics"][key] for r in independent if r["role"] == summary["role"]]
            review.close(summary[key + "_mean"], np.mean(values), "three seed mean")
            review.close(summary[key + "_sample_sd"], np.std(values, ddof=1), "three seed sample SD")
    step_matched = []
    independent_lookup = {(r["role"], r["seed"]): r["metrics"] for r in independent}
    for baseline_role, candidate_role in (("Aextra", "B"), ("Aextra", "C"), ("B", "C")):
        for seed in (42, 43, 44):
            baseline = {r["record_id"]: r["prediction"] for r in native_rows if r["role"] == baseline_role and r["seed"] == seed}
            candidate = {r["record_id"]: r["prediction"] for r in native_rows if r["role"] == candidate_role and r["seed"] == seed}
            review.check(set(baseline) == set(candidate) == set(truth), "matched120-step contrast denominator")
            corrected = sum(baseline[rid] != truth[rid] and candidate[rid] == truth[rid] for rid in truth)
            broken = sum(baseline[rid] == truth[rid] and candidate[rid] != truth[rid] for rid in truth)
            contrast = {"baseline": baseline_role, "candidate": candidate_role, "seed": seed,
                        "corrected": corrected, "broken": broken, "target_observations": 855,
                        **{key + "_delta": independent_lookup[candidate_role, seed][key] - independent_lookup[baseline_role, seed][key]
                           for key in ("accuracy", "macro_f1", "balanced_accuracy")}}
            review.close((corrected - broken) / 855, contrast["accuracy_delta"], "matchedprestep corrected/broken difference")
            step_matched.append(contrast)
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OUTER_NATIVE_MODEL_SCOPE_ALIAS_METRIC_AND_DELTA_REPLAY_PASS_EXPLORATORY_ONLY",
        "new_audit_fits": 0, "checks": review.checks, "numeric_values": review.values, "maximum_difference": review.maximum,
        "pointwise_inference_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "all_native_saved_prediction_labels_equal": True,
        "actual_counts": dict(review.counts), "actual_outer_models": 60, "exact_policy_alias_cases": 15,
        "unique_observations_per_role_seed": 855, "independently_recomputed_metrics": independent,
        "independently_verified_paired_scratch_deltas": deltas,
        "independently_recomputed_step_matched_contrasts": step_matched,
        "selection_lock_sha256": sha(RUN / "selection_lock.json"), "prediction_lock_sha256": sha(RUN / "prediction_lock.json"),
        "scoring_manifest_sha256": sha(RUN / "scoring_manifest.json"), "helper_sha256": sha(HERE / "replay_helpers.py"), "script_sha256": sha(__file__),
        "limitations": ["Exposed855 internal development; one source batch/class, no new independent confirmation",
                        "Shape-only interpolation transfer with unresolved teacher absorption units/ginseng ancestors",
                        "All repeated seeds share855 samples; sampleSD is training variability, not biological replicate CI",
                        "Historical96.02MC-UVE/SVM differs representation/architecture and is not the causal transfer baseline",
                        "B/C versusAextra separates external-source pretraining from target-source extra updates at equal120 presteps; heads differ and runtime/FLOPs are not exactly equal",
                        "Audit replays weights and actual logged optimizer inputs without retraining gradients; finite guards, not OS sandbox"]}
    output = HERE / "evaluation_review.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output), "checks": review.checks,
                      "numeric_values": review.values, "maximum_difference": review.maximum, "new_audit_fits": 0}))


if __name__ == "__main__":
    main()
