"""Independently review completed inner training before any new outer score."""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from audit_common import Auditor, ROOT, RUN, DEST, load, sha, predictions


def key(row):
    return (int(row["outer_fold"]), int(row["inner_fold"]), row["family"], row["profile"], int(row["seed"]))


def objective(row):
    return (-row["mean_inner_metrics"]["macro_f1"], -row["mean_inner_metrics"]["balanced_accuracy"],
            -row["mean_inner_metrics"]["accuracy"], row["family"], row["profile"]["id"])


def main():
    target = DEST/"selection_review.json"
    if target.exists():
        raise FileExistsError("Preserve the completed review")
    a = Auditor()
    protocol = load(ROOT/"data/research/upgrade_neural_protocol_20261031.json")
    prep, lock, manifest = (load(RUN/p) for p in ("preparation.json","selection_lock.json","run_manifest.json"))
    a.require(not (RUN/"outer_predictions.csv.gz").exists(), "Review must precede new outer inference")
    a.require(sha(ROOT/"data/research/upgrade_neural_protocol_20261031.json") == lock["protocol_sha256"] == manifest["protocol_sha256"], "Frozen protocol")
    a.require(sha(RUN/"run_manifest.json") == lock["manifest_sha256"], "Frozen manifest")
    a.require(sha(RUN/"inner_metrics.json") == lock["inner_metrics_sha256"], "Frozen inner records")
    a.require(sha(RUN/"inner_predictions.csv.gz") == lock["inner_predictions_sha256"], "Frozen inner predictions")
    for path, expected in manifest["source_sha256"].items():
        a.require(sha(ROOT/path) == expected, "Code changed " + path)
    trials = load(RUN/"inner_metrics.json")
    a.require(len(trials) == len({key(r) for r in trials}) == 480, "Expected complete inner budget")
    contexts = {c["split_id"]:c for c in prep["contexts"] if c["stage"] == "inner"}
    grouped = defaultdict(list)
    for row in predictions(RUN/"inner_predictions.csv.gz"):
        grouped[key(row)].append(row)
    a.require(len(grouped) == 480 and sum(map(len, grouped.values())) == 164160, "Inner prediction budget")
    recalculated = {}
    for index, trial in enumerate(trials):
        context = contexts[trial["split_id"]]
        a.require(trial["source_ids"] == context["source_ids"] and trial["inference_ids"] == context["inference_ids"], "Trial context identity")
        own_rows = predictions(ROOT/trial["model_directory"]/"predictions.csv.gz")
        a.require(own_rows == grouped[key(trial)], "Combined predictions differ from immutable trial predictions")
        value = a.model(trial,context,own_rows)
        a.metric_match(value,trial["metrics"],"Inner trial independent metrics")
        recalculated[key(trial)] = value
        if (index+1)%48 == 0:
            print(json.dumps({"independent_replayed_inner_models":index+1}),flush=True)
    summaries = []
    for fold in range(5):
        locked = lock["outer_folds"][fold]
        a.require(locked["outer_fold"] == fold, "Outer choice index")
        all_candidates = []
        for family in protocol["families"]:
            candidates = []
            for profile in protocol["profiles"]:
                values = [recalculated[(fold,inner,family,profile["id"],seed)] for inner in range(2) for seed in protocol["seeds"]]
                means = {m:float(np.mean([r[m] for r in values])) for m in ("macro_f1","balanced_accuracy","accuracy")}
                candidate = {"family":family,"profile":profile,"mean_inner_metrics":means}
                candidates.append(candidate);all_candidates.append(candidate)
                saved = next(v for v in locked["choices"][family]["means"] if v["profile"]["id"] == profile["id"])
                a.require(saved["selection_case_count"] == 6, "Complete seed/fold selection")
                for m in means:
                    a.close(saved["mean_inner_metrics"][m],means[m],"Inner mean selection metric",atol=1e-14,rtol=1e-13)
            selected = min(candidates,key=objective)
            a.require(locked["choices"][family]["selected"] == selected["profile"], "Family profile not chosen from current inner data")
            a.require(locked["choices"][family]["outer_test_used_to_select"] is False, "Outer score used to select")
        primary = min(all_candidates,key=objective)
        strategy = locked["neural_strategy"]
        a.require(strategy["selected_family"] == primary["family"] and strategy["selected_profile"] == primary["profile"], "Overall architecture/attention/fusion selection")
        a.require(strategy["selection_case_count"] == 6 and strategy["outer_test_used_to_select"] is False, "Overall inner-only selection scope")
        summaries.append({"outer_fold":fold,"selected_family":primary["family"],"selected_profile":primary["profile"]["id"],"mean_inner_metrics":primary["mean_inner_metrics"]})
    scope_path = RUN/"fit_scope_events.jsonl"
    scope_summary = a.scope_log(scope_path,trials)
    raw_scope = scope_path.read_bytes()
    snapshot = DEST/"selection_fit_scope_prefix.jsonl.gz"
    with gzip.open(snapshot,"wb") as handle:
        handle.write(raw_scope)
    result = {"status":"TECHNICAL_INNER_REVIEW_PASS_EXPLORATORY_DATA","allow_outer_evaluation":True,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":a.checks,
        "maximum_numeric_difference":a.max_difference,"models_replayed":a.models_replayed,
        "MC_selectors_recomputed_from_saved_coefficients":a.mc_selectors_recomputed,
        "inner_cases":len(trials),"inner_prediction_rows":164160,"family_profile_choices_checked":40,
        "overall_neural_policy_choices_checked":5,"policy":summaries,"fit_scope":scope_summary,
        "new_estimator_fits":0,"new_calibrator_fits":0,"outer_predictions_scored":False,
        "selection_lock_sha256":sha(RUN/"selection_lock.json"),"run_manifest_sha256":sha(RUN/"run_manifest.json"),
        "protocol_sha256":sha(ROOT/"data/research/upgrade_neural_protocol_20261031.json"),
        "scope_prefix_bytes":len(raw_scope),"scope_prefix_sha256":sha(scope_path),"scope_prefix_gzip_sha256":sha(snapshot),
        "audit_script_sha256":sha(Path(__file__)),"audit_common_sha256":sha(DEST/"audit_common.py"),
        "limits":["Retrospective exposed855; not new-batch confirmation", "Only explicitly instrumented APIs, not arbitrary code or OS sandbox", "10-seed conditional permutation audit is a minimum diagnostic, not absolute zero-leakage proof"]}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},ensure_ascii=False))


if __name__ == "__main__":
    main()
