"""Compact source-search evidence without fitting or loading a model.

Original checkpoint JSON text is retained exactly inside JSONL records. Original
prediction CSVs can be reconstructed byte-for-byte by grouping the consolidated
rows, dropping candidate_id/episode and writing the fixed UTF8-BOM/LF format.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil


CSV_FIELDS = ["record_id", "y_true", "y_predicted", "is_correct"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False)+"\n", encoding="utf-8")


def original_csv_bytes(rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return text.getvalue().encode("utf-8-sig")


def build_archive(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads((source/"source_status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Only a completed source search can be archived")
    originals, metadata_count, prediction_count = {}, 0, 0
    warning_models, warning_types = 0, Counter()
    rows_for_metrics = []
    json_path = output/"source_checkpoints.jsonl.gz"
    csv_path = output/"source_development_predictions.csv.gz"
    with gzip.open(json_path,"wt",encoding="utf-8",newline="") as metadata_handle, \
            gzip.open(csv_path,"wt",encoding="utf-8",newline="") as prediction_handle:
        writer = csv.DictWriter(prediction_handle,
            fieldnames=["candidate_id","episode",*CSV_FIELDS],lineterminator="\n")
        writer.writeheader()
        for meta_path in sorted((source/"development").glob("*/episode_*.json")):
            raw = meta_path.read_bytes()
            meta = json.loads(raw)
            candidate = meta["candidate"]
            metadata_sha = hashlib.sha256(raw).hexdigest()
            relative = str(meta_path.relative_to(source))
            originals[relative] = {"sha256":metadata_sha,"bytes":len(raw),
                                  "storage":"source_checkpoints.jsonl.gz: raw_json_utf8"}
            entry = {"original_path":relative,"sha256":metadata_sha,
                     "candidate_id":candidate["id"],"episode":meta["episode"],
                     "raw_json_utf8":raw.decode("utf-8")}
            metadata_handle.write(json.dumps(entry,ensure_ascii=False,separators=(",",":"))+"\n")
            assert entry["raw_json_utf8"].encode("utf-8") == raw
            metadata_count += 1
            warnings = meta.get("warnings",[])
            warning_models += bool(warnings)
            warning_types.update(warnings)
            base = {"candidate_id":candidate["id"],"episode":meta["episode"],
                    "stage":meta["stage"],"status":meta["status"],
                    "preprocessing":candidate["preprocessing"],"model":candidate["model"],
                    "candidate_json":json.dumps(candidate,sort_keys=True,separators=(",",":")),
                    "source_sample_count":len(meta["partition"]["source"]),
                    "developer_query_count":len(meta["partition"]["query"]),
                    "accuracy":"","balanced_accuracy":"","macro_f1":"",
                    "correct":"","total":"","warnings_count":len(warnings),
                    "seconds":meta["seconds"],"model_sha256":meta.get("model_sha256",""),
                    "predictions_sha256":meta.get("predictions_sha256","")}
            if meta["status"] == "pass":
                metrics = meta["metrics"]
                base.update({key:metrics[key] for key in
                    ["accuracy","balanced_accuracy","macro_f1","correct","total"]})
                stem = meta_path.with_suffix("")
                model_path = stem.with_suffix(".joblib")
                prediction_path = stem.with_suffix(".predictions.csv")
                assert file_hash(model_path) == meta["model_sha256"]
                assert file_hash(prediction_path) == meta["predictions_sha256"]
                assert meta["source_fingerprint_before"] == meta["source_fingerprint_after"]
                with prediction_path.open(encoding="utf-8-sig",newline="") as handle:
                    original_rows = list(csv.DictReader(handle))
                recovered = original_csv_bytes(original_rows)
                assert recovered == prediction_path.read_bytes()
                assert hashlib.sha256(recovered).hexdigest() == meta["predictions_sha256"]
                assert len(original_rows) == metrics["total"]
                assert sum(int(row["is_correct"]) for row in original_rows) == metrics["correct"]
                for row in original_rows:
                    assert int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"])
                    writer.writerow({"candidate_id":candidate["id"],"episode":meta["episode"],**row})
                prediction_count += len(original_rows)
                originals[str(model_path.relative_to(source))] = {
                    "sha256":meta["model_sha256"],"bytes":model_path.stat().st_size,
                    "storage":"Not copied; runtime cache only. Refit with production CLI to reproduce."}
                originals[str(prediction_path.relative_to(source))] = {
                    "sha256":meta["predictions_sha256"],"bytes":prediction_path.stat().st_size,
                    "storage":"source_development_predictions.csv.gz; exact UTF8-BOM/LF reconstruction verified"}
            rows_for_metrics.append(base)
    with (output/"source_metrics.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows_for_metrics[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows_for_metrics)
    copied = ["source_candidate_grid.json","development_partitions.json",
              "source_development_aggregate.json","source_preparation.json",
              "source_run_manifest.json","source_selection_lock.json",
              "source_selection_lock_sha256.json","source_status.json"]
    for name in copied:
        shutil.copy2(source/name,output/name)
        originals[name]={"sha256":file_hash(source/name),"bytes":(source/name).stat().st_size,
                         "storage":"copied byte-for-byte"}
    for name in ["source_run.log","source_run_invocations.jsonl","source_resume.log",
                 "source-truth-isolation-test.log"]:
        path=source.parent/name
        if path.exists():
            shutil.copy2(path,output/name)
            originals["../"+name]={"sha256":file_hash(path),"bytes":path.stat().st_size,
                                     "storage":"copied byte-for-byte"}
    write_json(output/"original_files_sha256.json",originals)
    # Read the compressed metadata back and verify exact recoverability.
    restored = 0
    with gzip.open(json_path,"rt",encoding="utf-8") as handle:
        for line in handle:
            entry=json.loads(line);raw=entry["raw_json_utf8"].encode("utf-8")
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
            assert raw == (source/entry["original_path"]).read_bytes()
            restored += 1
    expected=status["source_fits_recorded"]
    assert metadata_count == restored == len(rows_for_metrics) == expected
    assert prediction_count == sum(row["total"] for row in rows_for_metrics if row["status"]=="pass")
    validation={"status":"passed","models_fitted":0,
        "metadata_records":metadata_count,"metadata_raw_bytes_exactly_recoverable":True,
        "source_metrics_rows":len(rows_for_metrics),"developer_prediction_rows":prediction_count,
        "original_prediction_CSV_bytes_exactly_reconstructable":True,
        "original_joblib_hashes_checked_without_loading":sum(path.endswith('.joblib') for path in originals),
        "source_models_copied":0,"source_training_samples_per_task":307,
        "developer_query_samples_per_task":103,"outer_prediction_role_scored":False,
        "source_model_warnings_recorded":warning_models,"warning_message_counts":dict(warning_types),
        "original_file_entries":len(originals)}
    write_json(output/"archive_validation.json",validation)
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    repo=Path(__file__).resolve().parents[4]
    parser.add_argument("--source",type=Path,default=repo/"outputs/representation-optimization-20261031/source")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    result=build_archive(args.source,args.output)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
