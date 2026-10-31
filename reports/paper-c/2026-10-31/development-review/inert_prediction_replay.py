"""Supplemental score/prediction audit without fitting or final-query scoring."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "outputs/paper-c-20261031/calibration/run"
OUT = Path(__file__).resolve().parent


def main():
    protocol = json.loads((ROOT / "data/research/paper_c_calibration_protocol_20261031.json").read_text())
    candidates = {candidate["id"]: candidate for candidate in protocol["candidates"]}
    budgets = json.loads((RUN / "development_budget.json").read_text())
    expected, raw_rows, checks, cap_checks = {}, 0, 0, 0
    for entry in budgets:
        selector, episode = entry["selector"], entry["episode"]
        with np.load(RUN / "development" / selector / f"episode_{episode:02d}_scores.npz", allow_pickle=False) as scores:
            classes = scores["classes"]
            ordered = np.sort(scores["source_scores"], axis=1)
            margins = ordered[:, -1] - ordered[:, -2]
            baseline = dict(zip(scores["query_ids"].tolist(), classes[scores["query_scores"].argmax(axis=1)].tolist(), strict=True))
            pool_prediction = dict(zip(scores["pool_ids"].tolist(), classes[scores["pool_scores"].argmax(axis=1)].tolist(), strict=True))
            pool_truth = dict(zip(scores["pool_ids"].tolist(), scores["pool_y"].tolist(), strict=True))
            for draw in entry["draws"]:
                errors = sum(pool_prediction[rid] != pool_truth[rid] for rid in draw["ids"])
                expected[(selector, episode, draw["repeat"])] = (errors, float(np.median(margins)),
                    {q: float(np.quantile(margins, q)) for q in [.5, .75]})
        with gzip.open(RUN / "development" / selector / f"episode_{episode:02d}_predictions.csv.gz", "rt") as handle:
            for row in csv.DictReader(handle):
                assert row["y_predicted"] == baseline[row["record_id"]]
                checks += 1
                raw_rows += 1
    for selector in protocol["selectors"]:
        with gzip.open(RUN / "development" / selector / "metrics.csv.gz", "rt") as handle:
            for row in csv.DictReader(handle):
                errors, median, quantiles = expected[(selector, int(row["episode"]), int(row["repeat"]))]
                candidate = candidates[row["candidate_id"]]
                assert int(row["support_error_count"]) == errors
                assert (row["support_rejected"] == "True") == (errors < candidate["minimum_support_errors"])
                assert abs(float(row["cap"]) - candidate["cap_fraction"] * max(median, np.finfo(float).eps)) < 1e-14
                assert abs(float(row["gate_margin"]) - quantiles[candidate["source_gate_quantile"]]) < 1e-14
                assert row["stage"] == ("selection" if int(row["episode"]) < 24 else "gate")
                cap_checks += 5
    result = {"status": "pass", "development_prediction_rows_exactly_equal_to_source_argmax": raw_rows,
              "raw_prediction_checks": checks, "support_cap_gate_checks": cap_checks, "total_checks": checks + cap_checks,
              "new_estimator_fits": 0, "new_calibrator_fits": 0, "final_query_scored": False,
              "review_sha256": hashlib.sha256((OUT / "review.json").read_bytes()).hexdigest(),
              "meaning": "All raw predictions, including wrong-to-other-wrong labels, equal source argmax. Saved support-error evidence and cap/gate values use only source scores and support labels."}
    (OUT / "inert_prediction_replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
