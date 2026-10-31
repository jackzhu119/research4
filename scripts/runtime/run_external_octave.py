"""Run archived 0308 PLS/CARS/MC-UVE cores on held-out external regressions.

The external NPZ must contain X_train, y_train, X_test, y_test. Only X_train,
y_train and X_test are exported to Octave; Python alone opens y_test for final
metrics. Inputs may instead use X/y with train_idx/test_idx. Dataset specs are
a list or a dictionary with a ``datasets`` list, with id/task/data_path fields.

These are fixed-budget archived-core checks, not retuned comparisons or native
MATLAB runs. CARS uses its declared simplified originalVersion=0 branch.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO = Path(__file__).resolve().parents[2]
METHODS = {
    "pls": "archived_pls_lv2_center",
    "mcuve": "archived_mcuve_pls_lv2_center",
    "cars": "archived_cars_simple_pls_lv2_center",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not name:
        raise ValueError("Dataset id needs at least one filename-safe character")
    return name


def resolve_data_path(value, spec_path):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [spec_path.parent / path, REPO / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve external dataset path: {value}")


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(data.files):
            X_train, X_test = np.asarray(data["X_train"], dtype=float), np.asarray(data["X_test"], dtype=float)
            y_train, y_test = np.asarray(data["y_train"], dtype=float).reshape(-1), np.asarray(data["y_test"], dtype=float).reshape(-1)
        elif {"X", "y", "train_idx", "test_idx"}.issubset(data.files):
            X, y = np.asarray(data["X"], dtype=float), np.asarray(data["y"], dtype=float).reshape(-1)
            train, test = np.asarray(data["train_idx"], dtype=int), np.asarray(data["test_idx"], dtype=int)
            if set(train) & set(test):
                raise ValueError("Training and test row indices overlap")
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        else:
            raise ValueError("NPZ needs X_train/y_train/X_test/y_test or X/y/train_idx/test_idx")
        train_ids = np.asarray(data["train_ids"]).astype(str) if "train_ids" in data else np.asarray([f"train:{i}" for i in range(len(y_train))])
        test_ids = np.asarray(data["test_ids"]).astype(str) if "test_ids" in data else np.asarray([f"test:{i}" for i in range(len(y_test))])
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training/test inputs must use one common two-dimensional feature grid")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test) or min(len(y_train), len(y_test)) < 2:
        raise ValueError("Invalid or undersized external regression arrays")
    if len(y_train) < 6 or X_train.shape[1] < 2:
        raise ValueError("Fixed two-LV, three-fold checks need >=6 training rows and >=2 features")
    if not all(np.isfinite(x).all() for x in [X_train, X_test, y_train, y_test]):
        raise ValueError("External inputs and measured targets must be finite")
    if np.std(y_train) <= np.finfo(float).eps:
        raise ValueError("Training target has no variation")
    if train_ids.ndim != 1 or test_ids.ndim != 1 or len(train_ids) != len(y_train) or len(test_ids) != len(y_test):
        raise ValueError("IDs must have one entry per row")
    if set(train_ids) & set(test_ids):
        raise ValueError("Training/test sample IDs overlap")
    train_hashes = {hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() for row in X_train}
    if any(hashlib.sha256(row.astype("<f8").tobytes()).hexdigest() in train_hashes for row in X_test):
        raise ValueError("An exactly identical spectrum crosses the training/test boundary")
    return X_train, y_train, X_test, y_test, train_ids, test_ids


def read_source_inventory(path):
    records = json.loads(path.read_text())
    matches = [row for row in records if Path(row["archive"]).name == "0308 CARS+UVE.rar"]
    if len(matches) != 1:
        raise ValueError("Source inventory must uniquely identify 0308 CARS+UVE.rar")
    record = matches[0]
    archive = Path(record["archive"])
    if not archive.is_absolute():
        archive = REPO / archive
    if not archive.is_file() or sha256(archive) != record["source_sha256"]:
        raise ValueError("Archived original hash does not match source inventory")
    members = {}
    for name in ["pls.m", "mcuvepls.m", "carspls.m"]:
        options = [m for m in record["members"] if Path(m["name"]).name == name]
        if len(options) != 1:
            raise ValueError(f"Source member is not unique: {name}")
        member = options[0]
        if sha256(member["path"]) != member["sha256"]:
            raise ValueError(f"Extracted source hash mismatch: {name}")
        members[name] = member
    return record, members


def prepare_work_copy(record, members, work):
    source_root = Path(members["pls.m"]["path"]).parent
    known = {Path(m["path"]).resolve(): m for m in record["members"]}
    provenance = []
    for path in sorted(source_root.rglob("*.m")):
        raw = path.read_bytes()
        original_hash = hashlib.sha256(raw).hexdigest()
        entry = known.get(path.resolve())
        if entry is None or entry["sha256"] != original_hash:
            raise ValueError(f"Source dependency missing from verified inventory: {path.name}")
        for encoding in ["utf-8-sig", "gb18030", "latin-1"]:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeError:
                continue
        target = work / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(text.splitlines()) + "\n")
        provenance.append({"source_member": entry["name"], "source_sha256": original_hash,
                           "work_path": str(target.relative_to(work)), "work_sha256": sha256(target),
                           "decoded_encoding": encoding})
    helper = work / "sumsqr.m"
    if helper.exists():
        raise ValueError("Refusing to overwrite a supplied sumsqr implementation")
    helper.write_text("function s=sumsqr(x)\n s=sum(x(:).^2);\nend\n")
    return provenance


def run_process(command, log_path, timeout=120):
    """A timeout or Python interruption kills the complete spawned process group."""
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            log.write("\nTIMEOUT: complete Octave process group killed\n")
            return process.returncode, True
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def harness_text(method, work, inputs, predictions, retained, seed):
    prelude = ("warning('off','Octave:shadowed-function'); pkg load statistics;\n"
               f"rand('seed',{seed}); randn('seed',{seed});\n"
               f"cd({quote(work)}); addpath(pwd); load({quote(inputs)});\ntry\n"
               "A=2; selected=1:columns(X_train); selection_cv=NaN; nonfinite_stability=0;\n")
    if method == "mcuve":
        selection = ("U=mcuvepls(X_train,y_train,A,'center',20,0.8);\n"
                     f"selected=U.SortedVariable(1:{retained});\n"
                     "nonfinite_stability=sum(~isfinite(U.RI));\n"
                     "assert(all(isfinite(U.RI(selected))));\n")
    elif method == "cars":
        selection = ("C=carspls(X_train,y_train,A,3,'center',12,0,0,1);\n"
                     "selected=C.vsel; selection_cv=C.RMSECV_min;\n")
    else:
        selection = ""
    final = ("assert(numel(selected)>=A);\n"
             "P=pls(X_train(:,selected),y_train,A,'center');\n"
             "pred=[X_test(:,selected) ones(rows(X_test),1)]*P.coef_origin(:,end);\n"
             "assert(numel(pred)==rows(X_test)); assert(all(isfinite(pred)));\n"
             f"save('-mat7-binary',{quote(predictions)},'pred','selected','selection_cv','nonfinite_stability');\n"
             "fprintf('CHECK_PASSED\\n');\n"
             "catch err\n fprintf(2,'CHECK_FAILED: %s\\n',err.message);\n"
             "for k=1:numel(err.stack); fprintf(2,'  %s:%d\\n',err.stack(k).file,err.stack(k).line); end; exit(1);\nend;\n")
    return prelude + selection + final


def regression_metrics(y, predicted):
    rmse = float(np.sqrt(mean_squared_error(y, predicted)))
    target_sd = float(np.std(y, ddof=1))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y, predicted)),
            "r2": float(r2_score(y, predicted)) if target_sd > 0 else None,
            "bias": float(np.mean(predicted - y)),
            "rpd": target_sd / rmse if rmse > 0 else None,
            "test_target_sd_ddof1": target_sd}


def write_outputs(out, results):
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    columns = ["dataset_id", "fold", "method", "status", "n_train", "n_test", "n_features",
               "n_selected", "duration_seconds", "rmse", "mae", "r2", "bias", "rpd", "error"]
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get("metrics", {}).get(key, row.get(key, "")) for key in columns})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--octave", type=Path, default=REPO / ".runtime/bin/octave")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", help="Only exact dataset ids")
    parser.add_argument("--fold", type=int, help="Only the stated official fold")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    args = parser.parse_args()
    args.spec, args.inventory, args.octave, args.output = (p.resolve() for p in [args.spec, args.inventory, args.octave, args.output])
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty/new output directory")
    spec = json.loads(args.spec.read_text())
    datasets = spec if isinstance(spec, list) else spec["datasets"]
    datasets = [d for d in datasets if d["task"] == "regression"
                and (not args.datasets or d["id"] in args.datasets)
                and (args.fold is None or int(d.get("fold", -1)) == args.fold)]
    if not datasets:
        parser.error("No matching regression datasets")
    if len({d["id"] for d in datasets}) != len(datasets):
        parser.error("Dataset ids must be unique")
    if not args.octave.is_file():
        parser.error("Octave wrapper does not exist")
    record, members = read_source_inventory(args.inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"started_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "spec": str(args.spec), "spec_sha256": sha256(args.spec),
                "runner_sha256": sha256(__file__), "inventory_sha256": sha256(args.inventory),
                "source_archive": record["archive"], "source_archive_sha256": record["source_sha256"],
                "scope": "archived core with explicit compatibility adaptations; fixed budgets; not native MATLAB",
                "matlab_native_execution_verified": False,
                "caveat": "External LC-MS values retain author-provided annotations/relative units; no absolute-content claim",
                "budgets": {"final_pls_components": 2, "mcuve_runs": 20,
                            "mcuve_retained": "min(p, max(2, min(100, ceil(0.1*p))))",
                            "cars_runs": 12, "cars_cv": 3, "cars_originalVersion": 0,
                            "seed_per_case": 42, "timeout_seconds_per_case": 120}}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    results = []
    for item in datasets:
        data_path = resolve_data_path(item["data_path"], args.spec)
        X_train, y_train, X_test, y_test, train_ids, test_ids = load_dataset(data_path)
        retained = min(X_train.shape[1], max(2, min(100, int(np.ceil(0.1 * X_train.shape[1])))))
        for method in args.methods:
            folder = args.output / f"{safe_name(item['id'])}__{method}"
            work = folder / "work"
            work.mkdir(parents=True)
            start = time.monotonic()
            row = {"id": f"{item['id']}__{method}", "dataset_id": item["id"],
                   "fold": item.get("fold"), "method": METHODS[method], "status": "fail",
                   "task": "regression", "source_relation": "archived_core_with_explicit_compatibility_adaptations",
                   "data_path": str(data_path), "data_sha256": sha256(data_path),
                   "target": item.get("target", item.get("target_name")), "target_unit": item.get("target_unit", item.get("unit")),
                   "n_train": len(y_train), "n_test": len(y_test), "n_features": X_train.shape[1],
                   "seed": 42, "fixed_pls_components": 2, "timeout_seconds": 120,
                   "log": str((folder / "run.log").relative_to(args.output)), "metrics": {}}
            try:
                provenance = prepare_work_copy(record, members, work)
                adaptation = {"source_archive": record["archive"], "source_archive_sha256": record["source_sha256"],
                              "source_files": provenance,
                              "declared_adaptations": ["Decode work copies to UTF-8/LF; preserve original archives and extracted source bytes",
                                                       "Add sumsqr compatibility helper: sum(x(:).^2)",
                                                       "Use fixed seed42; called PLS/CARS/MC-UVE files do not reseed from clock",
                                                       "Fit all mean-centering and variable selection on X_train/y_train only",
                                                       "No y_test array is supplied to the Octave process",
                                                       "Final archived PLS always uses two latent variables and center preprocessing"],
                              "method_settings": {"method": method, "pls_components": 2,
                                                  "mcuve_runs": 20 if method == "mcuve" else None,
                                                  "mcuve_sample_fraction": 0.8 if method == "mcuve" else None,
                                                  "mcuve_features": retained if method == "mcuve" else None,
                                                  "cars_runs": 12 if method == "cars" else None,
                                                  "cars_cv": 3 if method == "cars" else None,
                                                  "cars_selectLV": 0 if method == "cars" else None,
                                                  "cars_originalVersion": 0 if method == "cars" else None,
                                                  "cars_order": 1 if method == "cars" else None},
                              "notes": ["CARS uses the archived simplified coefficient-pruning branch, not its random-resampling branch",
                                        "CARS internal RMSECV is a selection diagnostic, not the reported external RMSE",
                                        "Fixed two-LV final models ignore CARS.optLV; budgets are not equal to tuned Python models"]}
                (folder / "adaptation.json").write_text(json.dumps(adaptation, ensure_ascii=False, indent=2) + "\n")
                inputs, predictions = folder / "inputs.mat", folder / "predictions.mat"
                savemat(inputs, {"X_train": X_train, "y_train": y_train[:, None], "X_test": X_test})
                harness = folder / "check.m"
                harness.write_text(harness_text(method, work, inputs, predictions, retained, 42))
                command = [str(args.octave), str(harness)]
                row["command"] = command
                rc, timeout = run_process(command, folder / "run.log", timeout=120)
                row["returncode"] = rc
                log = (folder / "run.log").read_text(errors="replace")
                if timeout:
                    row["status"], row["error"] = "timeout", "120-second budget exceeded; complete process group killed"
                elif rc or "CHECK_PASSED" not in log:
                    error = re.search(r"CHECK_FAILED: ([^\n]+)", log)
                    row["error"] = error.group(1) if error else "Octave did not complete successfully; see run.log"
                else:
                    saved = loadmat(predictions)
                    predicted = np.asarray(saved["pred"]).reshape(-1)
                    selected = np.asarray(saved["selected"]).reshape(-1).astype(int)
                    if len(predicted) != len(y_test) or not np.isfinite(predicted).all():
                        raise ValueError("External predictions are missing or nonfinite")
                    row["n_selected"] = len(selected)
                    row["selected_indices_zero_based"] = (selected - 1).tolist()
                    row["nonfinite_stability_count"] = int(np.asarray(saved["nonfinite_stability"]).item())
                    selection_cv = float(np.asarray(saved["selection_cv"]).item())
                    row["internal_selection_rmsecv_diagnostic"] = selection_cv if np.isfinite(selection_cv) else None
                    row["metrics"], row["status"] = regression_metrics(y_test, predicted), "pass"
                    with (folder / "predictions.csv").open("w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["test_id", "y_true", "y_pred", "residual_pred_minus_true"])
                        writer.writerows(zip(test_ids, y_test, predicted, predicted - y_test))
            except Exception as exc:
                row["status"], row["error"] = "fail", f"{type(exc).__name__}: {exc}"
            row["duration_seconds"] = round(time.monotonic() - start, 4)
            results.append(row)
            write_outputs(args.output, results)
            print(json.dumps({k: row.get(k) for k in ["id", "status", "duration_seconds", "metrics", "error"]}, ensure_ascii=False), flush=True)
    return 0 if all(row["status"] == "pass" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
