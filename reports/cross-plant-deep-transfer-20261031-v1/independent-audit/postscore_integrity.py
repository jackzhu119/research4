"""Check scored-label join, hashes and phase order; no estimator operations."""
from datetime import datetime, timezone
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "neural"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    scoring, preparation = read(RUN / "scoring_manifest.json"), read(RUN / "preparation.json")
    checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)
    bindings = [("prediction_lock_sha256", "prediction_lock.json"), ("prediction_sha256", "outer_predictions.csv.gz"),
                ("metrics_sha256", "pooled_metrics_by_seed.json"), ("summary_sha256", "seed_summary.json"),
                ("paired_deltas_sha256", "paired_scratch_deltas.json"), ("scored_predictions_sha256", "scored_outer_predictions.csv.gz")]
    for key, name in bindings:
        check(scoring[key] == sha(RUN / name), "scoring dependency " + key)
    truth = {}
    for context in preparation["contexts"]:
        if context["stage"] == "outer":
            check(sha(RUN / context["truth_path"]) == context["truth_sha256"], "fixed outer truth bytes")
            with (RUN / context["truth_path"]).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    check(row["record_id"] not in truth, "one canonical truth per target")
                    truth[row["record_id"]] = row["label"]
    raw, scored = rows(RUN / "outer_predictions.csv.gz"), rows(RUN / "scored_outer_predictions.csv.gz")
    check(len(raw) == len(scored) == 12825 and len(truth) == 855, "actual denominators")
    for prediction, joined in zip(raw, scored):
        check({key: value for key, value in joined.items() if key != "y_true"} == prediction, "scored row preserves native prediction and routing")
        check(joined["y_true"] == truth[prediction["record_id"]], "scored label joins correct canonical sample")
    timestamps = {"selection": read(RUN / "selection_lock.json")["created_at_utc"],
                  "outer_release": read(RUN / "outer_release_snapshot.json")["created_at_utc"],
                  "prediction": read(RUN / "prediction_lock.json")["created_at_utc"],
                  "scoring": scoring["created_at_utc"]}
    check(timestamps["selection"] < timestamps["outer_release"] < timestamps["prediction"] < timestamps["scoring"], "all source locks precede outer release/prediction/separate score")
    audit_sources = {}
    for path in sorted(HERE.glob("*.py")):
        tree = ast.parse(path.read_text())
        forbidden = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "partial_fit", "backward", "step"}]
        check(not forbidden, "no fit/backward/optimizer step calls in auditor " + path.name)
        audit_sources[path.name] = sha(path)
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": "SCORED_LABEL_JOIN_HASH_AND_PHASE_ORDER_PASS",
              "checks": checks, "new_audit_fits": 0, "scored_rows_verified": 12825, "unique_target_observations": 855,
              "timestamps": timestamps, "scoring_manifest_sha256": sha(RUN / "scoring_manifest.json"),
              "auditor_source_sha256_with_zero_fit_calls": audit_sources, "script_sha256": sha(__file__),
              "scope": "Actual scored output joins exactly the correct anonymous canonical labels; hashes and declared phase timestamps verified, not OS sandbox or independent confirmation"}
    output = HERE / "postscore_integrity.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha(output), "checks": checks, "new_audit_fits": 0}))


if __name__ == "__main__":
    main()
