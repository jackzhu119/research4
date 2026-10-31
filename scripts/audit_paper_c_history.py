"""Read-only independent audit of historical staged OOF evidence.

No fit, tuning, download or historical write is performed. Integer confusion
matrices are rebuilt without the research modules' metric helpers.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import warnings

import joblib
import numpy as np
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "reports/staged-nested-optimization/2026-10-31"
OUT = ROOT / "reports/paper-c/2026-10-31/history-audit"
INPUT = ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"
CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
INDEX = {c: i for i, c in enumerate(CLASSES)}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
checks = 0
max_difference = 0.0


def require(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-12):
    global max_difference
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    max_difference = max(max_difference, difference)
    require(difference <= tolerance, f"{message}: {difference}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def calculate(rows):
    matrix = np.zeros((7, 7), dtype=np.int64)
    for row in rows:
        matrix[INDEX[row["y_true"]], INDEX[row["y_predicted"]]] += 1
    true_count, predicted_count, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    precision = np.divide(correct, predicted_count, out=np.zeros(7), where=predicted_count != 0)
    recall = np.divide(correct, true_count, out=np.zeros(7), where=true_count != 0)
    f1 = np.divide(2 * correct, true_count + predicted_count, out=np.zeros(7), where=true_count + predicted_count != 0)
    return {"samples": int(matrix.sum()), "correct": int(correct.sum()),
            "accuracy": float(correct.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "confusion_matrix": matrix.tolist(),
            "class_order": list(CLASSES), "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "support": true_count.tolist(), "predicted_count": predicted_count.tolist()}


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    started = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    evidence_hashes = {}
    # Exact archived byte indices verify old evidence without modifying it.
    for directory in [HIST, ROOT / "models/staged_nested_20261031"]:
        for relative, expected in load_json(directory / "files_sha256.json").items():
            path = directory / relative
            require(sha(path) == expected, f"Frozen evidence changed: {path}")
        evidence_hashes[str((directory / "files_sha256.json").relative_to(ROOT))] = sha(directory / "files_sha256.json")
    core_manifest = load_json(HIST / "core/run_manifest.json")
    require(sha(INPUT) == core_manifest["input_sha256"], "Input NPZ bytes changed")
    dataset = np.load(INPUT, allow_pickle=False)
    X, y, ids, axis = (dataset[k] for k in ["X_alpha", "y", "record_ids", "axis"])
    truth = dict(zip(ids.tolist(), y.tolist(), strict=True))
    by_id = {name: i for i, name in enumerate(ids)}
    require(X.shape == (855, 612) and len(truth) == 855, "Dataset shape/uniqueness")
    require(np.isfinite(X).all() and np.all(np.diff(axis) > 0), "Numerical input validity")
    require(set(dataset["groups"]) == set(ids), "Physical sample declaration mapping")
    spectral_hash_groups = defaultdict(list)
    for name, spectrum in zip(ids, X, strict=True):
        spectral_hash_groups[hashlib.sha256(np.ascontiguousarray(spectrum).tobytes()).hexdigest()].append(str(name))
    exact_duplicate_spectra = [group for group in spectral_hash_groups.values() if len(group) > 1]
    require(not exact_duplicate_spectra, "Exact duplicate input spectra require investigation; never silently delete")
    registry = load_json(ROOT / "data/active_workbooks.json")
    workbook_evidence = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for record in registry["workbooks"]:
            if record["dataset"] != "zaoren":
                continue
            path = ROOT / record["path"]
            require(sha(path) == record["sha256"], f"Active workbook SHA: {path}")
            book = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet = book["吸收系数"]
            sheet.reset_dimensions()
            header = next(sheet.values)
            sample_ids = list(header[1:])
            require(len(sample_ids) == record["record_count"], "Workbook sample count")
            for sample in sample_ids:
                require(sample in truth and truth[sample] == record["origin_group_id"], "Workbook/NPZ label mismatch")
            workbook_evidence.append({"path": record["path"], "sha256": record["sha256"],
                                      "origin_group": record["origin_group_id"], "samples": len(sample_ids)})
            book.close()
    splits = load_json(HIST / "core/splits.json")
    old_splits = ROOT / "outputs/approved-zaoren-20261031/experiments/alpha_primary/splits.json"
    require(sha(old_splits) == load_json(HIST / "core/protocol_snapshot.json")["historical_split_sha256"], "Legacy split SHA")
    require(load_json(old_splits) == splits, "Exact historical split identity")
    test_membership = {}
    for fold in splits:
        train, test = set(fold["train_sample_ids"]), set(fold["test_sample_ids"])
        require(len(train) == 684 and len(test) == 171 and not train & test and train | test == set(ids), "Outer source/test partition")
        for name in test:
            require(name not in test_membership, "Duplicate outer query")
            test_membership[name] = fold["fold"]
        inner_query_count = Counter()
        for inner in fold["inner_folds"]:
            source, query = set(inner["train_sample_ids"]), set(inner["validation_sample_ids"])
            require(len(source) == len(query) == 342 and source | query == train and not source & query, "Inner source/query isolation")
            require(not query & test and not source & test, "Outer query used internally")
            inner_query_count.update(query)
        require(set(inner_query_count) == train and set(inner_query_count.values()) == {1}, "Each source validated exactly once internally")
    require(len(test_membership) == 855, "All IDs have one outer query fold")

    # All original raw inner predictions are independently tallied, not trusted
    # from the all_inner_metrics summaries alone.
    expected_inner = {}
    for branch in ["core", "combinations"]:
        metric_path = HIST / branch / ("all_inner_metrics.json.gz" if branch == "core" else "all_inner_metrics.json")
        with (gzip.open(metric_path, "rt") if metric_path.suffix == ".gz" else metric_path.open()) as handle:
            records = json.load(handle)
        for row in records:
            key = (int(row["outer_fold"]), row["stage"], int(row["inner_fold"]), row["candidate_id"])
            require(key not in expected_inner, "Duplicate inner metric key")
            expected_inner[key] = row
            if "source_ids" in row:
                require(row["source_ids"] == splits[key[0]]["inner_folds"][key[2]]["train_sample_ids"], "Inner source IDs match split")
    inner_counts = 0
    inner_cases = 0
    audited_inner = {}
    archives = [HIST / "core" / f"outer_{i:02d}_selection_evidence.tar.gz" for i in range(5)]
    archives.append(HIST / "combinations/selection_evidence.tar.gz")
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.name.endswith("predictions.csv.gz"):
                    continue
                groups = defaultdict(list)
                with gzip.open(tar.extractfile(member), "rt", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        key = (int(row["outer_fold"]), row["stage"], int(row["inner_fold"]), row["candidate_id"])
                        require(row["record_id"] in truth and row["y_true"] == truth[row["record_id"]], "Inner label aligned to canonical workbook")
                        require(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "Inner is_correct not rewritten")
                        groups[key].append(row)
                for key, rows in groups.items():
                    require(key not in audited_inner, "Duplicate raw inner case")
                    query = splits[key[0]]["inner_folds"][key[2]]["validation_sample_ids"]
                    require(len(rows) == 342 and set(r["record_id"] for r in rows) == set(query), "Raw inner query IDs match source-only validation")
                    metrics = calculate(rows)
                    for metric in METRICS:
                        close(metrics[metric], expected_inner[key]["metrics"][metric], "Raw inner metric")
                    require(metrics["confusion_matrix"] == expected_inner[key]["metrics"]["confusion_matrix"], "Raw inner confusion matrix")
                    audited_inner[key] = metrics
                    inner_counts += len(rows)
                    inner_cases += 1
    require(set(audited_inner) == set(expected_inner), "Every inner metric has raw predictions")
    require((inner_cases, inner_counts) == (1540, 526680), "Complete inner audit counts")

    core_lock = load_json(HIST / "core/selection_lock.json")
    combo_lock = load_json(HIST / "combinations/selection_lock.json")
    selections = []
    for lock, branch in [(core_lock, "core"), (combo_lock, "combinations")]:
        require(lock["outer_scores_used_to_select"] is False, "No declared outer selection")
        for fold in lock["outer_folds"]:
            choices = fold["choices"] if branch == "core" else {"R4": fold["choice"]}
            for stage, choice in choices.items():
                require(choice["selected_using_outer_test"] is False, "Choice declares outer selection")
                outer = int(fold["outer_fold"])
                means = choice["all_candidate_means"]
                for candidate, mean in means.items():
                    for metric in METRICS:
                        reconstructed = float(np.mean([audited_inner[(outer, stage, inner, candidate)][metric] for inner in [0, 1]]))
                        close(reconstructed, mean[metric], "Inner means reconstructed")
                chosen_id = choice["selected"]["id"]
                require(means[chosen_id]["macro_f1"] >= max(m["macro_f1"] for m in means.values()) - 1e-12, "Chosen candidate maximizes inner Macro-F1")
                selections.append({"outer_fold": outer, "stage": stage, "selected": choice["selected"], "mean_inner_metrics": means[chosen_id]})

    # Verify the original release precedes actual evaluation invocations and
    # current source bytes still equal the executed source snapshot.
    release = load_json(HIST / "outer_evaluation_release.json")
    released = datetime.fromisoformat(release["released_at_utc"])
    source_hashes = {}
    for record in release["rounds"]:
        branch = record["name"]
        for filename, field in [("selection_lock.json", "selection_lock_sha256"), ("protocol_snapshot.json", "protocol_sha256"), ("run_manifest.json", "run_manifest_sha256")]:
            require(sha(HIST / branch / filename) == record[field], "Released evidence SHA")
        require(record["outer_predictions_existed_at_release"] is False, "Current outer predictions pre-existed lock release")
        events = [json.loads(line) for line in (HIST / branch / "stage_events.jsonl").read_text().splitlines()]
        starts = [datetime.fromisoformat(e["utc"]) for e in events if e["event"] in {"stage_start", "start"} and e["stage"] == "evaluation"]
        require(starts and min(starts) > released, "Actual evaluation starts after global release")
        manifest = load_json(HIST / branch / "run_manifest.json")
        for path, digest in manifest["source_sha256"].items():
            require(sha(ROOT / path) == digest, "Executed ML source changed")
            source_hashes[path] = digest
    invocations = [json.loads(line) for line in (HIST / "evaluation_invocations.jsonl").read_text().splitlines()]
    for label in ["core-evaluation", "combinations-evaluation"]:
        starts = [e for e in invocations if e["label"] == label and e["event"] == "command_start"]
        completed = [e for e in invocations if e["label"] == label and e["event"] == "command_completed"]
        require(len(starts) == len(completed) == 1 and completed[0]["returncode"] == 0, "External actual evaluation driver completed")
        require(released < datetime.fromisoformat(starts[0]["utc"]) < datetime.fromisoformat(completed[0]["utc"]), "External actual evaluation after release")

    by_method = defaultdict(list)
    summary_map = {}
    outer_metric_map = {}
    for branch in ["core", "combinations"]:
        for row in csv_rows(HIST / branch / "outer_predictions.csv.gz"):
            require(row["record_id"] in truth and row["y_true"] == truth[row["record_id"]], "Outer label agrees with canonical workbook")
            require(test_membership[row["record_id"]] == int(row["outer_fold"]), "Outer record appears only in own frozen fold")
            require(int(row["is_correct"]) == int(row["y_true"] == row["y_predicted"]), "Outer correctness flag")
            by_method[row["method"]].append(row)
        for row in load_json(HIST / branch / "summary.json"):
            summary_map[row["method"]] = row
        for row in load_json(HIST / branch / "outer_metrics.json"):
            outer_metric_map[(row["method"], int(row["outer_fold"]))] = row
    summaries, class_rows, fold_rows = [], [], []
    predictions = {}
    for method, rows in by_method.items():
        require(len(rows) == len({r["record_id"] for r in rows}) == 855 and set(r["record_id"] for r in rows) == set(ids), "Method has all855 unique OOF records")
        metrics = calculate(rows)
        for metric in METRICS:
            close(metrics[metric], summary_map[method][metric], "Outer summary recomputation")
        require(metrics["confusion_matrix"] == summary_map[method]["confusion_matrix"], "OOF confusion matrix")
        row = {"method": method, **metrics}
        summaries.append(row)
        predictions[method] = {r["record_id"]: r for r in rows}
        for i, name in enumerate(CLASSES):
            class_rows.append({"method": method, "class": name, **{k: metrics[k][i] for k in ["precision", "recall", "f1", "support", "predicted_count"]}})
        for outer in range(5):
            subset = [r for r in rows if int(r["outer_fold"]) == outer]
            computed = calculate(subset)
            for metric in METRICS:
                close(computed[metric], outer_metric_map[(method, outer)][metric], "Fold-level outer metric")
            fold_rows.append({"method": method, "outer_fold": outer, **{k: computed[k] for k in ["samples", "correct", *METRICS]}})
    require(len(summaries) == 20 and sum(len(x) for x in by_method.values()) == 17100, "All20 method-role outer records")

    # Reload and replay all five main stages, rather than only reading scores.
    model_manifest = load_json(ROOT / "models/staged_nested_20261031/core/manifest.json")["models"]
    replays, r2_parameters = [], []
    model_sha_before = {}
    for outer in range(5):
        fold = splits[outer]
        train_idx = [by_id[name] for name in fold["train_sample_ids"]]
        for method in ["R0", "R1", "R2", "R3", "R4"]:
            expected = outer_metric_map[(method, outer)]
            if method == "R4":
                path = ROOT / "models/staged_nested_20261031/combinations/evaluation" / f"outer_{outer:02d}" / f"{expected['family']}.joblib"
            else:
                entry = next(m for m in model_manifest if m["path"].startswith(f"outer_{outer:02d}/") and method in m["method_roles"])
                path = ROOT / "models/staged_nested_20261031/core" / entry["path"]
            digest = sha(path)
            require(digest == expected["model_sha256"], "Model bytes equal actual evaluated model")
            model_sha_before[path] = digest
            model = joblib.load(path)
            test_ids = sorted(fold["test_sample_ids"])
            test_X = X[[by_id[name] for name in test_ids]]
            replay_labels = model.predict(test_X)
            replay_scores = model.predict_scores(test_X)
            stored_labels = [predictions[method][name]["y_predicted"] for name in test_ids]
            require(np.array_equal(replay_labels, stored_labels), "Frozen model label replay")
            stored_scores = np.asarray([[float(predictions[method][name][f"score_{c}"]) for c in CLASSES] for name in test_ids])
            close(replay_scores, stored_scores, "Frozen model score replay")
            prep = model.preprocessor if method == "R4" else model.bundle.preprocessor
            scaler = model.scaler if method == "R4" else model.bundle.scaler
            transformed = prep.transform(X[train_idx])
            close(scaler.mean_, transformed.mean(0), "Source-only scaler mean")
            close(scaler.var_, transformed.var(0), "Source-only scaler variance")
            if method == "R4":
                require(model.metadata["source_ids"] == fold["train_sample_ids"], "R4 actual source identities")
            else:
                require(model.metadata["source_samples"] == 684, "Actual source count")
                require(model.metadata["source_value_fingerprint"] == joblib.hash(X[train_idx]), "Actual source numeric fingerprint")
                require(model.metadata["source_label_fingerprint"] == joblib.hash(y[train_idx]), "Actual source label fingerprint")
                require(model.estimator.random_state == 42 and model.estimator.break_ties is False, "Model seed and original vote")
            replay = {"method": method, "outer_fold": outer, "model_path": str(path.relative_to(ROOT)),
                      "sha256": digest, "source_samples": 684, "query_samples": 171,
                      "predictions_equal": True, "scores_equal": True, "source_scaler_verified": True}
            replays.append(replay)
            if method == "R2":
                require(model.config == core_lock["outer_folds"][outer]["choices"]["R2"]["selected"], "R2 matches pre-evaluation selection")
                require(model.config["preprocessing"] == "snv", "R2 uses SNV in all five folds")
                require(len(expected["selected_indices"]) == model.metadata["selected_features"], "R2 selected feature count")
                require(expected["selected_indices"] == model.metadata["selected_indices"], "R2 exact selected indices")
                r2_parameters.append({"outer_fold": outer, "model_seed": 42, "split_seed": 20261031,
                    "config": model.config, "selected_features": model.metadata["selected_features"],
                    "effective_gamma": model.metadata["effective_gamma"], "training_class_weights": model.metadata["training_class_weights"],
                    "selected_indices": expected["selected_indices"], "selected_frequencies": expected["selected_frequencies"],
                    "support_vectors": int(model.estimator.n_support_.sum()), "model_bytes": path.stat().st_size,
                    "correct": int(sum(predictions[method][name]["y_true"] == predictions[method][name]["y_predicted"] for name in test_ids)), **replay})
    for path, digest in model_sha_before.items():
        require(sha(path) == digest, "Model was changed during read-only replay")
    old_predictions = csv_rows(ROOT / "outputs/approved-zaoren-20261031/experiments/alpha_primary/alpha_mcuve20_svc/predictions.csv")
    require(len(old_predictions) == 855, "Historical first baseline count")
    for row in old_predictions:
        name = row["sample_id"]
        require(row["y_true"] == truth[name] and row["y_predicted"] == predictions["R0"][name]["y_predicted"], "Exact initial baseline per-sample reproduction")
        require(int(row["outer_fold"]) == test_membership[name], "Historical baseline same outer partition")

    paired, changed_rows, fold_changes = [], [], []
    for before, after in [("R0", "R1"), ("R0", "R2"), ("R0", "R3"), ("R0", "R4"), ("R1", "R2"), ("R2", "R3"), ("R3", "R4")]:
        for outer in [None, 0, 1, 2, 3, 4]:
            fixed = harmed = 0
            for name in ids:
                if outer is not None and test_membership[name] != outer:
                    continue
                a, b = predictions[before][name], predictions[after][name]
                ac, bc = a["y_true"] == a["y_predicted"], b["y_true"] == b["y_predicted"]
                fixed += int(not ac and bc)
                harmed += int(ac and not bc)
                if outer is None and a["y_predicted"] != b["y_predicted"]:
                    changed_rows.append({"before": before, "after": after, "record_id": name, "outer_fold": test_membership[name],
                                         "y_true": truth[name], "before_prediction": a["y_predicted"], "after_prediction": b["y_predicted"],
                                         "corrected": int(not ac and bc), "harmed": int(ac and not bc)})
            entry = {"before": before, "after": after, "outer_fold": outer,
                     "corrected": fixed, "harmed": harmed, "net_correct": fixed - harmed}
            (paired if outer is None else fold_changes).append(entry)
    primary = [next(r for r in summaries if r["method"] == m) for m in ["R0", "R1", "R2", "R3", "R4"]]
    require([r["correct"] for r in primary] == [773, 791, 821, 816, 812], "Five claimed main counts")
    for row in primary:
        row.update({"delta_accuracy_pp_vs_R0": 100 * (row["correct"] - 773) / 855,
                    "relative_accuracy_gain_percent_vs_R0": 100 * (row["correct"] - 773) / 773,
                    "protocol": "same855 legacy5outer/2inner; pooled OOF; retrospective within-batch",
                    "evidence_grade": "E2: raw labels+splits+inner predictions+locks+model replay; not E3 external confirmation"})
    old_threeway = load_json(ROOT / "data/processed/zaoren_threeway_20261031_v1/split_plan.json")
    result = {"status": "pass", "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "new_model_fits": 0, "checks": checks, "maximum_numeric_difference": max_difference,
              "input_sha256": sha(INPUT), "dataset_version": str(dataset["dataset_version"]),
              "exact_duplicate_spectrum_groups": exact_duplicate_spectra,
              "duplicate_check_scope": "Exact numerical equality only; spectral similarity is not proof of physical independence.",
              "class_counts": dict(Counter(y.tolist())), "workbook_evidence": workbook_evidence,
              "current_head_at_audit": git("rev-parse", "HEAD"),
              "local_branches_at_audit": git("branch", "-avv"), "recent_commits_at_audit": git("log", "-20", "--format=%h %ad %s", "--date=iso-strict"),
              "fetched_remote_branch_snapshot": "reports/paper-c/2026-10-31/repository-state/repository_snapshot.json",
              "outer_release_at_utc": release["released_at_utc"], "frozen_evidence_indices": evidence_hashes,
              "executed_source_sha256": source_hashes, "inner_cases": inner_cases, "inner_prediction_rows": inner_counts,
              "outer_method_roles": len(summaries), "outer_prediction_rows": sum(len(r) for r in by_method.values()),
              "main_results": primary, "all_results": summaries, "r2_parameters": r2_parameters,
              "main_model_replays": replays, "paired_changes": paired, "fold_changes": fold_changes,
              "selections": selections, "old_threeway_protocol": {k: old_threeway[k] for k in ["split_id", "split_seed", "role_counts", "scope", "prior_exposure"]},
              "findings": [
                  "90.41% is773/855 nested-OOF, not the171-prediction-sample experiment.",
                  "96.02% is821/855; all five R2 models use SNV but select fold-specific MC/SVM settings. There is no single96.02% model trained on855 in these historical weights.",
                  "R2 improves Macro-F1 and Balanced Accuracy together relative toR0, but ZY02 remains3/5 correct.",
                  "Raw predictions, source fingerprints, scaler statistics, split isolation and release order pass. No direct outer-test fit/selection was found in this audited implementation.",
                  "All855 and historical test results were already exposed. Method design can still overfit development history; retrospective nested results are not an untouched confirmation.",
                  "R3 changes two folds toSG1/MSC; R4 chooses budget/ordinary/stableMC using inner Macro-F1. Inner selection does not guarantee outer improvement; paired counts quantify the net decline, without inventing a physical cause.",
                  "User-declared sample independence and anonymous origin groups are preserved. Named geographical origins, physical chemical attribution and cross-batch generalization remain unverified.",
                  "The old90.25%MLP is a three-seed mean on the same171query specimens; oldLDA92.40% and SVM88.89% share that171protocol and cannot be ranked directly against855OOF.",
              ]}
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_csv(OUT / "main_results.csv", [{k: row[k] for k in ["method", "samples", "correct", *METRICS, "delta_accuracy_pp_vs_R0", "relative_accuracy_gain_percent_vs_R0", "protocol", "evidence_grade"]} for row in primary])
    save_csv(OUT / "all_method_metrics.csv", [{k: row[k] for k in ["method", "samples", "correct", *METRICS]} for row in summaries])
    save_csv(OUT / "per_class_metrics.csv", class_rows)
    save_csv(OUT / "outer_fold_metrics.csv", fold_rows)
    save_csv(OUT / "paired_changes.csv", paired)
    save_csv(OUT / "fold_changes.csv", fold_changes)
    save_csv(OUT / "changed_predictions.csv", changed_rows)
    rows = ["# 历史阶段准确率：独立只读审计", "", "本次重新计算原始内外预测，重载五个阶段共25个外折模型角色。无新增拟合，不更改标签、划分或历史文件。", "", "| 阶段 | 正确数／855 | Accuracy | Macro-F1 | Balanced Accuracy | 比R0增加（百分点） |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in primary:
        rows.append(f"| {row['method']} | {row['correct']} | {100*row['accuracy']:.4f}% | {row['macro_f1']:.4f} | {100*row['balanced_accuracy']:.4f}% | {row['delta_accuracy_pp_vs_R0']:+.4f} |")
    rows += ["", f"完成{inner_cases}次内层评价／{inner_counts}条原始预测、20条方法角色／17100条OOF的独立计数；{checks}项检查通过，最大数值差{max_difference:.3g}。五阶段共25个外折角色的标签和分数均按冻结模型重放通过，源缩放统计及源身份通过。", "", "## 下降来自哪些改动", "", "| 比较 | 原错误被纠正 | 原正确变错误 | 净正确数变化 |", "| --- | ---: | ---: | ---: |"]
    for row in paired:
        rows.append(f"| {row['before']}→{row['after']} | {row['corrected']} | {row['harmed']} | {row['net_correct']:+d} |")
    rows += ["", "R3外折3改用SG一阶＋SNV，外折4改用MSC；其余外折仍用SNV。R4外折1—5选择预算MC／普通MC／稳定MC／预算MC／普通MC。逐折改对与改错见fold_changes.csv，不把内层最高分当作外层必然改进。序贯继承参数不是所有预处理与参数的全组合联合最优。", "", "## 证据等级与边界", "", "E0仅有汇总；E1有原预测、真实标签和划分；E2再有训练来源、冻结锁和模型重放；E3另有未用于开发的独立批次确认。五阶段在本次审计均达到E2，均未达到E3。", "", "原90.41%明确为855份五外折OOF；此前MLP约90.25%是三种子均值，LDA约92.40%和SVM约88.89%均属于513训练／171校准池／171固定预测协议，不能混排。", "", "未在被审计实现发现直接把外层样品用于拟合或选参；源统计、索引及实际放行时间通过。但历史全部855份和测试成绩已曝光，整个方法开发仍可能适应已见数据，不可称新独立确认。编号和标签只用于监督与评价，特征输入是612个数值频点。", "", "样品独立性依据用户声明，真实地名待补；每组仅一个来源批次，产地与该批次效应无法分离。ZY02仅5份，R2仍为3/5，整体96.02%不能替代该组召回率。", "", "[机器审计与全部混淆矩阵](audit.json) · [五阶段台账](main_results.csv) · [逐类指标](per_class_metrics.csv) · [逐样品改动](changed_predictions.csv)", "", "重现：`.runtime/bin/python scripts/audit_paper_c_history.py`。只读审计会重新生成本审计目录，不会训练模型或改动旧实验。"]
    (OUT / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "checks": checks, "maximum_numeric_difference": max_difference,
                      "inner_cases": inner_cases, "outer_rows": 17100, "model_replay_roles": len(replays),
                      "new_model_fits": 0, "main_correct": [r["correct"] for r in primary]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
