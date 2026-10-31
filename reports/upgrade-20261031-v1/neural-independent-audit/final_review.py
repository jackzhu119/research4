"""Independently replay new outer neural models and recompute saved predictions."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_common import Auditor, ROOT, RUN, DEST, CLASSES, load, sha, predictions, metrics


def case_key(row):
    return int(row["outer_fold"]), row["family"], int(row["seed"])


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z","+00:00"))


def main():
    target = DEST/"final_review.json"
    if target.exists():
        raise FileExistsError("Preserve completed final review")
    a = Auditor()
    protocol = load(ROOT/"data/research/upgrade_neural_protocol_20261031.json")
    prep, lock, manifest = (load(RUN/p) for p in ("preparation.json","selection_lock.json","run_manifest.json"))
    before = load(DEST/"selection_review.json")
    a.require(before["status"] == "TECHNICAL_INNER_REVIEW_PASS_EXPLORATORY_DATA", "Inner review missing")
    a.require(sha(RUN/"selection_lock.json") == before["selection_lock_sha256"], "Selected architecture/profile changed")
    a.require(sha(RUN/"run_manifest.json") == before["run_manifest_sha256"], "Run manifest changed")
    a.require(sha(ROOT/"data/research/upgrade_neural_protocol_20261031.json") == before["protocol_sha256"], "Protocol changed")
    a.require(sha(DEST/"audit_common.py") == before["audit_common_sha256"], "Shared independent audit code changed")
    for relative, expected in manifest["source_sha256"].items():
        a.require(sha(ROOT/relative) == expected, "Implementation changed "+relative)
    release = load(RUN/"outer_release_snapshot.json")
    a.require(release["allow_outer_evaluation"] is True and release["selection_lock_sha256"] == sha(RUN/"selection_lock.json"), "Outer release missing or stale")
    a.require(release["manifest_sha256"] == sha(RUN/"run_manifest.json") and release["protocol_sha256"] == sha(ROOT/"data/research/upgrade_neural_protocol_20261031.json"), "Outer release binding")
    prediction_lock = load(RUN/"prediction_lock.json")
    a.require(sha(RUN/"outer_predictions.csv.gz") == prediction_lock["outer_predictions_sha256"], "Outer predictions changed")
    a.require(prediction_lock["truth_read_during_inference"] is False, "Truth permitted during inference")
    scoring = load(RUN/"scoring_manifest.json")
    a.require(scoring["prediction_lock_sha256"] == sha(RUN/"prediction_lock.json"), "Scoring did not bind frozen predictions")
    events = [json.loads(line) for line in (RUN/"stage_events.jsonl").read_text().splitlines()]
    times = {r["event"]:parse_time(r["utc"]) for r in events}
    release_time = parse_time(release["released_at_utc"])
    a.require(times["selection_locked"] < release_time <= times["outer_inference_started"] < times["outer_predictions_locked"] <= times["independent_scoring_completed"], "Actual execution/selection/inference/scoring order")
    contexts = {c["split_id"]:c for c in prep["contexts"] if c["stage"] == "outer"}
    trials = load(RUN/"outer_inference_manifest.json")
    a.require(len(trials) == len({case_key(r) for r in trials}) == 120, "Actual outer model budget")
    rows = predictions(RUN/"outer_predictions.csv.gz")
    grouped = defaultdict(list)
    for row in rows:
        a.require(not set(row) & {"y","label","y_true","truth","is_correct"}, "Inference file carries true labels")
        grouped[case_key(row)].append(row)
    a.require(len(rows) == 23085 and len(grouped) == 135, "Role aliases must not be mistaken for more specimens or fits")
    truth_all, fold_truth, independent_fold = {}, {}, []
    for context in contexts.values():
        _, _, mapping = a.source(context)
        a.require(not set(mapping) & set(truth_all), "Repeated OOF truth specimen")
        truth_all.update(mapping); fold_truth[context["outer_fold"]] = mapping
    a.require(len(truth_all) == 855, "OOF population changed")
    for index, trial in enumerate(trials):
        context = contexts[trial["split_id"]]
        a.require(trial["source_ids"] == context["source_ids"] and trial["inference_ids"] == context["inference_ids"], "Outer model context")
        selected = lock["outer_folds"][trial["outer_fold"]]["choices"][trial["family"]]["selected"]
        a.require(trial["profile"] == selected["id"], "Outer model profile differs from its inner selection")
        own_rows = predictions(ROOT/trial["model_directory"]/"predictions.csv.gz")
        a.require(own_rows == grouped[case_key(trial)], "Combined file differs from independently saved role")
        result = a.model(trial,context,own_rows)
        independent_fold.append({"family":trial["family"],"seed":trial["seed"],"outer_fold":trial["outer_fold"],"metrics":result})
        if (index+1)%24 == 0:
            print(json.dumps({"independent_replayed_outer_models":index+1}),flush=True)
    policy = []
    for fold_lock in lock["outer_folds"]:
        fold, decision = fold_lock["outer_fold"], fold_lock["neural_strategy"]
        policy.append({"outer_fold":fold,"selected_family":decision["selected_family"],"selected_profile":decision["selected_profile"]["id"]})
        for seed in protocol["seeds"]:
            alias = grouped[(fold,"inner_selected_neural",seed)]
            source = grouped[(fold,decision["selected_family"],seed)]
            expected = [{**r,"family":"inner_selected_neural"} for r in source]
            a.require(alias == expected, "Main neural policy predictions are not exact aliases of the source-selected family")
            independent_fold.append({"family":"inner_selected_neural","seed":seed,"outer_fold":fold,"metrics":metrics(alias,fold_truth[fold])})
    saved = load(RUN/"pooled_metrics_by_seed.json")
    saved_folds = {(r["family"],int(r["seed"]),int(r["outer_fold"])):r["metrics"] for r in load(RUN/"fold_metrics.json")}
    sensitivity = load(RUN/"seed_sensitivity.json")
    independent, table, classes_table = {}, [], []
    for family in [*protocol["families"],"inner_selected_neural"]:
        independent[family] = {}
        for seed in protocol["seeds"]:
            selected = [r for r in rows if r["family"] == family and int(r["seed"]) == seed]
            a.require(len(selected) == len({r["record_id"] for r in selected}) == 855, "OOF independent denominator")
            value = metrics(selected,truth_all)
            a.metric_match(value,saved[family][str(seed)],"Independent pooled metrics")
            independent[family][str(seed)] = value
            table.append({"family":family,"seed":seed,"correct":value["correct"],"samples":855,
                          "accuracy":value["accuracy"],"macro_f1":value["macro_f1"],"balanced_accuracy":value["balanced_accuracy"]})
            for cls in CLASSES:
                classes_table.append({"family":family,"seed":seed,"class":cls,**value["per_class"][cls]})
        for metric in ("accuracy","macro_f1","balanced_accuracy"):
            values = [independent[family][str(seed)][metric] for seed in protocol["seeds"]]
            a.close(sensitivity[family][metric]["mean"],np.mean(values),"Three-seed mean",atol=1e-14,rtol=1e-13)
            a.close(sensitivity[family][metric]["sample_sd"],np.std(values,ddof=1),"Three-seed descriptive sample SD",atol=1e-14,rtol=1e-13)
    for row in independent_fold:
        a.metric_match(row["metrics"],saved_folds[(row["family"],row["seed"],row["outer_fold"])],"Independent fold metrics")
    scope_bytes = (RUN/"fit_scope_events.jsonl").read_bytes()
    prefix = scope_bytes[:before["scope_prefix_bytes"]]
    a.require(hashlib.sha256(prefix).hexdigest() == before["scope_prefix_sha256"], "Inner actual-fit provenance prefix changed")
    suffix = scope_bytes[before["scope_prefix_bytes"]:]
    with gzip.open(DEST/"outer_fit_scope_suffix.jsonl.gz","wb") as handle:
        handle.write(suffix)
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as handle:
        handle.write(suffix); handle.flush()
        scope_summary = a.scope_log(Path(handle.name),trials)
    for name, values in (("model_metrics_by_seed.csv",table),("class_metrics_by_seed.csv",classes_table)):
        with (DEST/name).open("w",encoding="utf-8-sig",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(values[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(values)
    result = {"status":"TECHNICAL_OUTER_REPLAY_PASS_EXPLORATORY_DATA","completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "checks":a.checks,"maximum_numeric_difference":a.max_difference,"actual_outer_models_replayed":a.models_replayed,
        "outer_MC_selectors_recomputed_from_saved_coefficients":a.mc_selectors_recomputed,
        "actual_model_roles":120,"aliased_policy_roles":15,"raw_prediction_rows":len(rows),
        "pooled_method_seed_cases":27,"real_specimens_per_method_seed":855,"seeds_expand_real_sample_count":False,
        "inner_fit_scope_prefix_unchanged":True,"outer_fit_scope":scope_summary,"new_estimator_fits":0,"new_calibrator_fits":0,
        "selection_lock_sha256":sha(RUN/"selection_lock.json"),"prediction_lock_sha256":sha(RUN/"prediction_lock.json"),
        "outer_predictions_sha256":sha(RUN/"outer_predictions.csv.gz"),"outer_release_sha256":sha(RUN/"outer_release_snapshot.json"),
        "scoring_manifest_sha256":sha(RUN/"scoring_manifest.json"),"audit_script_sha256":sha(Path(__file__)),
        "audit_common_sha256":sha(DEST/"audit_common.py"),"primary_policy":policy,"independent_metrics":independent,
        "not_proven":["Absolute zero leakage", "Independent biological origins or new-batch generalization", "Chemical peak assignment or theory-guided efficacy", "Uniform superiority of deep learning under other architectures or budgets"],
        "limits":["Historical855 and prior results already exposed", "Only declared code paths and saved-state assertions verified", "Neural file/API isolation is not OS access control", "Different branches and profiles have different parameters/compute", "Three seed SDs describe sensitivity, not independent material samples"]}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},ensure_ascii=False))


if __name__ == "__main__":
    main()
