"""Replay every empirical model, optimizer-input trace and metric without fit."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import numpy as np
import torch

from replay_helpers import Review, ARMS, CLASSES, sha, read, metric, compare_metric

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "empirical/prepared"


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def choose(table):
    order = ("macro_f1", "balanced_accuracy", "accuracy")
    profile = min(("diagnostic2", "diagnostic4"), key=lambda p: tuple(-table["A", p][k] for k in order) + (p,))
    baseline = table["A", profile]
    eligible = [arm for arm in ARMS[1:] if table[arm, profile]["macro_f1"] + 1e-12 >= baseline["macro_f1"] + .005
                and table[arm, profile]["balanced_accuracy"] + 1e-12 >= baseline["balanced_accuracy"]
                and table[arm, profile]["accuracy"] + 1e-12 >= baseline["accuracy"]]
    winner = min(eligible, key=lambda arm: tuple(-table[arm, profile][k] for k in order) + (ARMS.index(arm),)) if eligible else "A"
    return profile, winner, eligible


def main():
    torch.set_num_threads(1)
    review = Review()
    manifest = read(RUN / "frozen_manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        review.check(sha(RUN / relative) == expected, "frozen empirical inputs")
    for relative, expected in manifest["source_sha256"].items():
        review.check(sha(ROOT / relative) == expected, "frozen empirical code")
    lock, saved_result, prep = read(RUN / "diagnostic_prediction_lock.json"), read(RUN / "results.json"), read(RUN / "preparation.json")
    outer = next(c for c in prep["contexts"] if c["stage"] == "outer")
    outer_source, query = load(RUN / outer["source_pack"]), load(RUN / outer["query_pack"])
    context_by_split = {c["split_id"]: c for c in prep["contexts"]}
    auxiliary = {name: load(RUN / pool["path"]) for name, pool in prep["auxiliary"].items()}
    positions = {rid: i for i, rid in enumerate(outer_source["record_ids"])}
    original_labels = dict(zip(outer_source["record_ids"].tolist(), outer_source["source_labels"].tolist()))
    with (RUN / "scoring_truth/outer0_original.csv").open(newline="") as handle:
        truth = {r["record_id"]: r["label"] for r in csv.DictReader(handle)}
    review.check(len(truth) == 171 and set(truth) == set(query["record_ids"]), "171 unique query observations")
    names = ["observed", "query_truth_permuted_trap", *[item["run"] for item in lock["nulls"]]]
    null_seeds = {item["run"]: item["permutation_seed"] for item in lock["nulls"]}
    summaries, run_models, paired = {}, {}, {}
    for name in names:
        labels = dict(original_labels)
        if name in null_seeds:
            rng = np.random.default_rng(null_seeds[name])
            for block in prep["inner_query_blocks"]:
                values = rng.permutation(np.asarray([original_labels[rid] for rid in block])).tolist()
                labels.update(zip(block, values))
        round_lock = read(RUN / "fits" / name / "round_prediction_lock.json")
        selection = read(RUN / "fits" / name / "source_selection_lock.json")
        review.check(round_lock["source_selection_lock_sha256"] == sha(RUN / "fits" / name / "source_selection_lock.json"), "source choice dependency")
        review.check(len(selection["inner_records"]) == 16 and len(round_lock["finals"]) == 4, "all20 actual models")
        native_source = []
        for record in [*selection["inner_records"], *round_lock["finals"].values()]:
            directory = RUN / record["model_directory"]
            metadata = read(directory / "training.json")
            context = context_by_split[metadata["split_id"]]
            identifiers = context["source_ids"]
            source = {"X": outer_source["X"][[positions[rid] for rid in identifiers]],
                      "axis": outer_source["axis"].copy(), "record_ids": np.asarray(identifiers),
                      "source_labels": np.asarray([labels[rid] for rid in identifiers], dtype=outer_source["source_labels"].dtype)}
            qp = load(RUN / context["query_pack"])
            is_inner = context["stage"] == "inner"
            if is_inner:
                stored = [r for r in selection["source_rows"] if r["arm"] == record["arm"]
                          and r["profile_id"] == record["profile_id"] and r["record_id"] in set(context["query_ids"])]
                rows = [{**r, "y_predicted": r["prediction"]} for r in stored]
                review.check([r["truth"] for r in stored] == [labels[rid] for rid in context["query_ids"]], "source OOF randomized/current truth")
            else:
                rows = record["predictions"]
            fitted, meta, signature, (predicted, scores) = review.pipeline(directory, source, auxiliary, qp, rows)
            for filename, expected in record["evidence_sha256"].items():
                review.check(sha(directory / filename) == expected, "round lock binds full actual evidence")
            if is_inner:
                native_source.extend({"arm": record["arm"], "profile_id": record["profile_id"], "record_id": rid,
                                      "prediction": str(pred)} for rid, pred in zip(qp["record_ids"], predicted))
            else:
                run_models[name, record["arm"]] = (predicted, scores)
            if name in {"observed", "query_truth_permuted_trap"}:
                key = record.get("key", "outer/" + record["arm"])
                paired[name, key] = {"signature": signature, "paired": {field: meta[field] for field in
                     ("initial_encoder_sha256", "initial_zy_head_sha256", "fresh_target_optimizer_initial_sha256", "target_order_sha256")}}
        table = {}
        for arm in ARMS:
            for profile in ("diagnostic2", "diagnostic4"):
                rows = [r for r in native_source if r["arm"] == arm and r["profile_id"] == profile]
                ids = [r["record_id"] for r in rows]
                review.check(len(ids) == len(set(ids)) == 684 and set(ids) == set(original_labels), "each source observation once per arm/profile")
                table[arm, profile] = metric([labels[rid] for rid in ids], [r["prediction"] for r in rows])
                compare_metric(review, selection["pooled_source_metrics"][arm][profile], table[arm, profile], "native pooled source metric")
        profile, winner, eligible = choose(table)
        review.check((selection["selection"]["common_profile"], selection["selection"]["policy_arm"]) == (profile, winner), "independent common A profile and guarded policy")
        review.check(selection["selection"]["eligible_nonbaseline_arms"] == eligible, "exact eligibility/fallback")
        review.check(round_lock["selection"] == selection["selection"], "round aliases exact source selection")
        summaries[name] = {"common_profile": profile, "policy_arm": winner, "metrics": {}}
        for role in (*ARMS, "source_policy"):
            arm = role if role in ARMS else winner
            predicted, _ = run_models[name, arm]
            m = metric([truth[rid] for rid in query["record_ids"]], predicted)
            summaries[name]["metrics"][role] = m
    for name, key in paired:
        if name == "observed":
            review.check(paired[name, key] == paired["query_truth_permuted_trap", key], "all C native objects and stochastic schedules invariant")
    for role in (*ARMS, "source_policy"):
        compare_metric(review, saved_result["observed_metrics"][role], summaries["observed"]["metrics"][role], "observed actual metric")
        for index, name in enumerate(names[2:]):
            compare_metric(review, saved_result["null_metrics"][role][index], summaries[name]["metrics"][role], "null actual metric")
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [summaries[name]["metrics"][role][key] for name in names[2:]]
            observed = summaries["observed"]["metrics"][role][key]
            expected = saved_result["null_summary"][role][key]
            review.close(expected["null_mean"], np.mean(values), "null mean")
            review.close(expected["empirical_tail_plus_one"], (1 + sum(v >= observed for v in values)) / 11, "finite null tail")
    counts = {c: list(original_labels.values()).count(c) for c in CLASSES}
    majority = min(CLASSES, key=lambda c: (-counts[c], c))
    majority_metric = metric([truth[rid] for rid in query["record_ids"]], np.full(171, majority))
    compare_metric(review, saved_result["majority_metrics"], majority_metric, "source-only majority baseline")
    review.check(review.counts["pipeline_weights"] == 240, "all240 weights independently replayed")
    review.check(review.counts["pretrained_trajectories"] == 180, "actual180 pretraining trajectories")
    review.check(review.counts["optimizer_step_pretrain"] == 720 and review.counts["optimizer_step_target"] == 5040, "all5760 actual optimizer minibatches independently rebound")
    events = [json.loads(line) for line in (HERE.parent / "empirical/commands.jsonl").read_text().splitlines()]
    execute_end = next(e["utc"] for e in events if e["event"] == "command_completed" and e["label"] == "transfer-empirical-actual-isolation-execute")
    score_start = next(e["utc"] for e in events if e["event"] == "command_start" and e["label"] == "transfer-empirical-separate-prediction-score")
    review.check(execute_end < score_start, "fresh score starts after all prediction locks completed")
    review.check(saved_result["score_new_fits"] == 0 and saved_result["main_selection_or_budget_changed"] is False, "score nofit/no main change")
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FULL_EMPIRICAL_NATIVE_PREDICTION_SOURCE_SCOPE_AND_METRIC_REPLAY_PASS_LIMITED_DIAGNOSTIC",
        "new_audit_fits": 0, "checks": review.checks, "numeric_values": review.values, "maximum_difference": review.maximum,
        "actual_counts": dict(review.counts), "independently_recomputed": summaries,
        "independently_recomputed_majority": majority_metric,
        "frozen_manifest_sha256": sha(RUN / "frozen_manifest.json"), "prediction_lock_sha256": sha(RUN / "diagnostic_prediction_lock.json"),
        "results_sha256": sha(RUN / "results.json"), "helper_sha256": sha(HERE / "replay_helpers.py"), "script_sha256": sha(__file__),
        "D_native_replay_evidence_sha256": sha(HERE / "empirical_runtime_readiness.json"),
        "limitations": ["One outer fold/seed42/2-or4epochs/4presteps only; cannot choose main models or establish main results",
                        "C alternate actual frozen truth files, D postfit-only perturbation, H target labels permuted by inner blocks with auxiliary labels fixed",
                        "Every saved gradient-input ID/value/target fingerprint checked; gradients not retrained during this zero-fit audit",
                        "Teacher absorption units/ginseng ancestor independence unknown; all main855 exposed, no new confirmation",
                        "Null tail resolution1/11; small-budget significance is diagnostic, not transfer generalization proof"]}
    output = HERE / "empirical_review.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output), "checks": review.checks,
                      "numeric_values": review.values, "maximum_difference": review.maximum, "new_audit_fits": 0}))


if __name__ == "__main__":
    main()
