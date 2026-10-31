"""Read-only source-set diagnostics of the preceding neural pilot.

This script loads existing source state_dicts and never calls a fit, optimizer,
adaptation routine, or held-out query scoring routine. It writes only its JSON
report. Saved ``supervised.pt`` is the continued model, not the 40-epoch common
initializer; the common initializer was not saved in the preceding run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(truth, prediction, count):
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    support = matrix.sum(axis=1)
    tp = np.diag(matrix)
    precision_support = matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(count, dtype=float), where=support != 0)
    f1 = np.divide(2*tp, support+precision_support, out=np.zeros(count, dtype=float),
                   where=(support+precision_support) != 0)
    return {"accuracy": float(tp.sum()/matrix.sum()),
            "balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist(),
            "recall_by_class": recall.tolist(),
            "source_samples": int(matrix.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    output = args.output or root/"outputs/multiround-20261031/previous_neural_source_diagnostic.json"
    sys.path.insert(0, str(root/"src"))
    from herb_spectra.paper_fewshot_classification import SmallResNet1D, snv_tensor, model_scores

    torch.set_num_threads(1)
    npz_path = root/"outputs/approved-zaoren-20261031/data-preparation/dataset.npz"
    split_path = root/"data/processed/zaoren_threeway_20261031_v1/split_plan.json"
    split = json.loads(split_path.read_text())
    expected_source = sorted(r["record_id"] for r in split["records"] if r["role"] == "training")
    forbidden = {r["record_id"] for r in split["records"] if r["role"] != "training"}
    assert len(expected_source) == 513 and len(forbidden) == 342
    with np.load(npz_path, allow_pickle=False) as data:
        identifiers = data["record_ids"].tolist()
        lookup = {v: i for i, v in enumerate(identifiers)}
        assert len(lookup) == len(identifiers)
        source_rows = np.asarray([lookup[v] for v in expected_source])
        axis = np.asarray(data["axis"])
        keep = (axis >= .1) & (axis <= 1.5)
        assert int(keep.sum()) == 612
        # Only the declared source rows are passed to normalization/inference.
        source_raw = np.asarray(data["X_alpha"][source_rows][:, keep])
        source_labels = data["y"][source_rows].tolist()
    classes = sorted(set(source_labels))
    assert classes == [f"ZY{i:02d}" for i in range(1, 8)]
    truth = np.asarray([classes.index(label) for label in source_labels], dtype=np.int64)
    features = snv_tensor(source_raw)
    frequency_std = features[:, 0, :].numpy().std(axis=0, ddof=1)
    report = {
        "status": "pass",
        "scope": "Read-only existing source-weight inference on 513 source-training specimens only",
        "new_model_fits": 0,
        "new_adaptation_fits": 0,
        "final_query_scored": False,
        "calibration_samples_scored": False,
        "dataset_version": split["dataset_version"],
        "classes": classes,
        "source_record_ids": expected_source,
        "source_role_verified_against_split": True,
        "source_records_disjoint_from_other_roles": not bool(set(expected_source) & forbidden),
        "frequency": {"lower_inclusive": .1, "upper_inclusive": 1.5,
                      "points": int(keep.sum()), "actual_first": float(axis[keep][0]),
                      "actual_last": float(axis[keep][-1])},
        "source_snv_per_frequency_standard_deviation": {
            "specimens": 513, "ddof": 1,
            "quantiles": {str(q): float(v) for q, v in zip(
                [0, .01, .1, .5, .9, .99, 1],
                np.quantile(frequency_std, [0, .01, .1, .5, .9, .99, 1]))},
            "max_to_min_ratio": float(frequency_std.max()/frequency_std.min()),
            "interpretation": "Frequency-wise source variation differs; does not establish a causal benefit from StandardScaler"
        },
        "by_seed": [],
        "provenance": {"dataset_npz_sha256": sha256(npz_path),
                       "split_plan_sha256": sha256(split_path),
                       "script_sha256": sha256(Path(__file__)),
                       "source_code_sha256": {
                           "src/herb_spectra/paper_fewshot_classification.py": sha256(
                               root/"src/herb_spectra/paper_fewshot_classification.py")},
                       "python": platform.python_version(), "numpy": np.__version__,
                       "torch": torch.__version__},
        "limits": [
            "supervised.pt is the 40-epoch supervised model after 100 source-task continuation updates, not the common initializer",
            "The common 40-epoch initializer state_dict was not saved, so its source accuracy cannot be reconstructed here",
            "The saved supervised_loss curve belongs to common initialization before continuation",
            "Low source-training performance motivates stronger supervised baselines; it does not prove pooling or normalization caused the problem",
            "No source-label or spectrum modification, no selection by fixed final-query outcomes",
            "The collection remains same-batch and retrospective; these are training-fit diagnostics, not generalization accuracy"
        ]
    }
    for seed in [42, 43, 44]:
        metadata_rel = f"reports/paper-fewshot-classification/2026-10-31/final/seed_{seed}/training.json"
        metadata = json.loads((root/metadata_rel).read_text())
        assert sorted(metadata["source_ids"]) == expected_source
        assert metadata["classes"] == classes and metadata["seed"] == seed
        losses = [float(row["mean_weighted_cross_entropy"]) for row in metadata["supervised_loss"]]
        assert len(losses) == 40
        row = {"seed": seed, "training_metadata_path": metadata_rel,
               "training_metadata_sha256": sha256(root/metadata_rel),
               "parameter_count": int(metadata["parameter_count"]),
               "supervised_initialization_curve": {"epochs": len(losses),
                    "first_loss": losses[0], "last_loss": losses[-1],
                    "minimum_loss": min(losses), "last10_mean_loss": float(np.mean(losses[-10:])),
                    "curve_is_before_100_continuation_updates": True},
               "saved_model_source_fit": {}}
        for strategy in ["supervised", "fomaml"]:
            relative = f"models/paper_fewshot_20261031/final/seed_{seed}/{strategy}.pt"
            model = SmallResNet1D(7)
            model.load_state_dict(torch.load(root/relative, weights_only=True, map_location="cpu"))
            prediction = np.argmax(model_scores(model, features), axis=1)
            row["saved_model_source_fit"][strategy] = {
                "source_weight_path": relative, "source_weight_sha256": sha256(root/relative),
                "model_scope": "supervised initialization followed by 100 source-task updates",
                **metrics(truth, prediction, 7)}
        report["by_seed"].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "new_model_fits": 0,
                      "source_samples": 513, "models_read": 6, "output": str(output)}))


if __name__ == "__main__":
    main()
