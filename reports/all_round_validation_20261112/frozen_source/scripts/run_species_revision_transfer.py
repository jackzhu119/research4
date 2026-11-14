"""Thin, frozen five-class reuse of the historical cross-plant experiment.

Fit/prediction accepts source and calibration labels only. Prediction truth is
joined in a separate score process after all model/prediction bytes are locked.
Historical Encoder, SNV and score-calibration implementations are reused.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(features_path, split_path):
    with np.load(features_path, allow_pickle=False) as data:
        identifiers = data["record_ids"].astype(str)
        values = data["X_alpha_transfer"].astype(float)
        axis = data["axis_transfer"].astype(float)
    rows = read_csv(split_path)
    lookup = {str(rid): i for i, rid in enumerate(identifiers)}
    if len(lookup) != len(identifiers) or len(rows) != len(identifiers):
        raise ValueError("Unique one-to-one features/split identities required")
    if set(lookup) != {r["record_id"] for r in rows}:
        raise ValueError("Feature and frozen split identities differ")
    if values.shape != (len(identifiers), 128) or axis.shape != (128,):
        raise ValueError("Historical 128-point transfer representation required")
    if not np.isfinite(values).all() or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("Finite increasing axis and finite spectra required")
    if len({r["physical_sample_id"] for r in rows}) != len(rows):
        raise ValueError("Exactly one independently declared physical sample per row required")
    if {r["role"] for r in rows} != {"train", "calibration", "prediction"}:
        raise ValueError("Frozen three-role split required")

    def pack(species, role):
        selected = sorted((r for r in rows if r["species"] == species and r["role"] == role), key=lambda r: r["record_id"])
        ids = np.asarray([r["record_id"] for r in selected])
        # Prediction pack deliberately contains no truth labels.
        result = {"X": values[[lookup[rid] for rid in ids]], "record_ids": ids}
        if role != "prediction":
            result["labels"] = np.asarray([r["origin_group_id"] for r in selected])
        return result

    return pack, axis


def protocol(features_path, split_path, axis):
    return {
        "protocol_id": "species-revision-cross-plant-20261108-v1",
        "study_status": "retrospective_internal_holdout_and_shape_transfer_not_cross_batch",
        "target_species": "酸枣仁", "target_classes": [f"ZY{i:02d}" for i in range(3, 8)],
        "target_roles": {"train": 300, "calibration": 100, "prediction": 100},
        "auxiliary_tasks": {"黄芪": "own four origin-group labels, train role only", "天麻": "own four origin-group labels, train role only"},
        "arms": {"A": "target train only", "Aextra": "120 target-source pretraining steps then fresh target head", "B": "120 alternating Huangqi/Tianma pretraining steps with independent task heads then fresh target head"},
        "seeds": [42, 43, 44],
        "historical_basis": ["docs/CROSS_PLANT_DEEP_TRANSFER_PROTOCOL_20261031.md", "docs/CONSERVATIVE_CROSS_PLANT_BUDGET_ABLATION_20261104.md", "docs/CALIBRATION_OPTIMIZATION_REPORT.md"],
        "reuse": ["upgrade_neural.Encoder", "upgrade_neural.row_snv", "conservative_calibration.build_context", "conservative_calibration.fit_candidate"],
        "architecture": {"width": 12, "blocks": 2, "dropout": .35, "residual": True, "attention": False, "embedding": 16, "target_head_outputs": 5},
        "training": {"target_epochs": 96, "epoch_policy": "fixed historical upper budget, no current prediction-dependent tuning", "optimizer": "Adam", "learning_rate": .0005, "weight_decay": .001, "target_batch_size": 64, "pretrain_steps": 120, "pretrain_batch_size": 32, "class_weight": "source balanced", "l2_sp": False, "batchnorm_statistics_reset_before_target_training": True},
        "preprocessing": {"axis": axis.tolist(), "snv": "historical per-row ddof=1", "scaler": "each task fits own StandardScaler using train role only", "no_query_statistics": True},
        "calibration": {"shots_per_class": [1, 3, 5, 10], "repeats": 20, "draw_seed": 20261108, "nested": True, "methods": [{"id": "legacy_lambda1", "family": "legacy_anchor", "regularization": 1.}, {"id": "weak_lambda1000_blend005", "family": "damped_anchor", "regularization": 1000., "blend": .05}], "parameters_fixed_from_history": True, "unused_calibration_rows_never_enter_prediction": True, "source_only_control": True, "score_calibration_is_not_probability_calibration": True},
        "safeguards": ["source/auxiliary/calibration/prediction physical IDs disjoint", "prediction labels absent from fit/predict interface", "same target initial head, batch order and dropout RNG across arms", "all three seeds and every arm retained", "no old target-finetuned weights reused", "prediction bytes frozen before separate truth join", "same prediction set for all shots/draws", "no real batch IDs inferred"],
        "features_path": str(features_path.relative_to(ROOT)), "features_sha256": sha(features_path),
        "split_path": str(split_path.relative_to(ROOT)), "split_sha256": sha(split_path),
        "code_sha256": sha(__file__),
        "reused_code_sha256": {name: sha(ROOT / "src/herb_spectra" / name) for name in ["upgrade_neural.py", "conservative_calibration.py", "calibration.py"]},
        "old_target_exposure": "The original 855 records participated in historical development; this is not a new external confirmation set.",
    }


def class_targets(labels, classes):
    encoded = np.asarray([classes.index(str(label)) for label in labels], dtype=np.int64)
    counts = np.bincount(encoded, minlength=len(classes))
    if not counts.all():
        raise ValueError("Source lacks a required class")
    return encoded, torch.as_tensor(len(labels) / (len(classes) * counts), dtype=torch.float32)


def train_one(source, auxiliary, arm, seed, config, destination):
    from herb_spectra.upgrade_neural import Encoder, row_snv, state_hash

    destination.mkdir(parents=True, exist_ok=False)
    params = config["training"]
    classes = config["target_classes"]
    allowed_source = set(source["record_ids"].tolist())
    all_seen = set(allowed_source)
    for pool in auxiliary.values():
        ids = set(pool["record_ids"].tolist())
        if all_seen & ids:
            raise ValueError("Cross-task physical identity overlap")
        all_seen |= ids
    scaler = StandardScaler().fit(row_snv(source["X"]))
    z = scaler.transform(row_snv(source["X"])).astype(np.float32)
    y, weights = class_targets(source["labels"], classes)
    torch.manual_seed(seed)
    encoder = Encoder(12, 2, .35, residual=True, attention=False)
    with torch.random.fork_rng():
        torch.manual_seed(seed + 10000)
        head = nn.Linear(16, len(classes))
    initial_encoder, initial_head = state_hash(encoder), state_hash(head)
    pools = [] if arm == "A" else ["target"] if arm == "Aextra" else ["黄芪", "天麻"]
    prepared = {}
    auxiliary_heads = nn.ModuleDict()
    auxiliary_scalers = {}
    for ordinal, name in enumerate(pools):
        pool = source if name == "target" else auxiliary[name]
        own_classes = classes if name == "target" else sorted(set(pool["labels"].tolist()))
        if name == "target":
            current_scaler, values = scaler, z
        else:
            current_scaler = StandardScaler().fit(row_snv(pool["X"]))
            values = current_scaler.transform(row_snv(pool["X"])).astype(np.float32)
            auxiliary_scalers[name] = current_scaler
        targets, own_weights = class_targets(pool["labels"], own_classes)
        prepared[name] = {"X": torch.as_tensor(values), "y": torch.as_tensor(targets), "weights": own_weights, "ids": pool["record_ids"], "generator": torch.Generator().manual_seed(seed + 30000 + ordinal)}
        with torch.random.fork_rng():
            torch.manual_seed(seed + 20000 + ordinal)
            auxiliary_heads[name] = nn.Linear(16, len(own_classes))
    start = time.perf_counter()
    pretrain_losses, batch_audit = [], []
    if pools:
        optimizer = torch.optim.Adam([*encoder.parameters(), *auxiliary_heads.parameters()], lr=params["learning_rate"], weight_decay=params["weight_decay"])
        torch.manual_seed(seed + 40000)
        for step in range(params["pretrain_steps"]):
            name = pools[step % len(pools)]
            pool = prepared[name]
            selected = torch.randperm(len(pool["X"]), generator=pool["generator"])[:params["pretrain_batch_size"]]
            actual_ids = pool["ids"][selected.numpy()].tolist()
            if len(set(actual_ids)) != params["pretrain_batch_size"]:
                raise ValueError("Pretraining duplicated an observation inside one batch")
            encoder.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(auxiliary_heads[name](encoder(pool["X"][selected])), pool["y"][selected], weight=pool["weights"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretraining loss")
            loss.backward()
            optimizer.step()
            pretrain_losses.append(float(loss.detach()))
            batch_audit.append({"step": step, "head": name, "actual_ids": actual_ids})
    pretrain_seconds = time.perf_counter() - start
    pretrained_state = copy.deepcopy(encoder.state_dict())
    for layer in encoder.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.reset_running_stats()
    if state_hash(head) != initial_head:
        raise ValueError("Pretraining changed the paired fresh target head")
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=params["learning_rate"], weight_decay=params["weight_decay"])
    torch.manual_seed(seed + 50000)
    generator = torch.Generator().manual_seed(seed + 60000)
    features, target = torch.as_tensor(z), torch.as_tensor(y)
    losses, orders = [], []
    start = time.perf_counter()
    for epoch in range(params["target_epochs"]):
        order = torch.randperm(len(z), generator=generator)
        orders.append(order.numpy().tolist())
        total = 0.
        encoder.train()
        head.train()
        for left in range(0, len(z), params["target_batch_size"]):
            indices = order[left:left + params["target_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(encoder(features[indices])), target[indices], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite target training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(z))
    train_seconds = time.perf_counter() - start
    encoder.eval()
    head.eval()
    metadata = {"arm": arm, "seed": seed, "classes": classes, "source_ids": source["record_ids"].tolist(), "auxiliary_ids": {name: pool["record_ids"].tolist() for name, pool in auxiliary.items()} if arm == "B" else {}, "initial_encoder_sha256": initial_encoder, "initial_target_head_sha256": initial_head, "final_encoder_sha256": state_hash(encoder), "final_head_sha256": state_hash(head), "target_order_sha256": hashlib.sha256(json.dumps(orders).encode()).hexdigest(), "pretrain_losses": pretrain_losses, "target_losses": losses, "pretrain_seconds": pretrain_seconds, "target_train_seconds": train_seconds}
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "pretrained_encoder": pretrained_state, "classes": classes}, destination / "weights.pt")
    joblib.dump({"target_scaler": scaler, "auxiliary_scalers": auxiliary_scalers}, destination / "scalers.joblib")
    write_json(destination / "training.json", metadata)
    write_json(destination / "pretrain_batch_ids.json", batch_audit)

    def score(values):
        transformed = scaler.transform(row_snv(values)).astype(np.float32)
        before = (state_hash(encoder), state_hash(head))
        with torch.no_grad():
            logits = head(encoder(torch.as_tensor(transformed))).numpy()
        if before != (state_hash(encoder), state_hash(head)) or not np.isfinite(logits).all():
            raise ValueError("Inference mutated model or returned nonfinite scores")
        return transformed, logits

    return score, metadata


def fit_predict(features_path, split_path, output):
    from herb_spectra.conservative_calibration import build_context, fit_candidate

    pack, axis = load_inputs(features_path, split_path)
    cfg = protocol(features_path, split_path, axis)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Run already initialized; preserve its frozen protocol")
    write_json(protocol_path, cfg)
    write_json(output / "protocol_lock.json", {"protocol_sha256": sha(protocol_path), "created_before_model_fits": True})
    source, calibration, query = [pack("酸枣仁", role) for role in ["train", "calibration", "prediction"]]
    auxiliary = {name: pack(name, "train") for name in ["黄芪", "天麻"]}
    if [len(p["X"]) for p in [source, calibration, query]] != [300, 100, 100]:
        raise ValueError("Frozen target support changed")
    for left, right in [(source, calibration), (source, query), (calibration, query)]:
        if set(left["record_ids"]) & set(right["record_ids"]):
            raise ValueError("Target roles overlap")
    classes = np.asarray(cfg["target_classes"])
    rng = np.random.default_rng(cfg["calibration"]["draw_seed"])
    draws = []
    for repeat in range(cfg["calibration"]["repeats"]):
        ordered = {label: rng.permutation(np.flatnonzero(calibration["labels"] == label)) for label in classes}
        for k in cfg["calibration"]["shots_per_class"]:
            chosen = np.concatenate([ordered[label][:k] for label in classes])
            if len(chosen) != len(classes) * k:
                raise ValueError("Insufficient independent calibration samples")
            draws.append({"repeat": repeat, "shots": k, "indices": chosen.tolist(), "support_ids": calibration["record_ids"][chosen].tolist()})
    write_json(output / "support_draws.json", draws)
    predictions, training = [], []
    for seed in cfg["seeds"]:
        paired = []
        for arm in cfg["arms"]:
            score, metadata = train_one(source, auxiliary, arm, seed, cfg, output / "models" / f"seed{seed}_{arm}")
            training.append(metadata)
            paired.append(metadata)
            train_features, train_scores = score(source["X"])
            cal_features, cal_scores = score(calibration["X"])
            query_features, query_scores = score(query["X"])
            context = build_context(train_features, train_scores, source["labels"], classes, seed=42)

            def add_rows(scores, method, repeat=-1, shots=0):
                predicted = classes[scores.argmax(axis=1)]
                predictions.extend({"arm": arm, "seed": seed, "method": method, "repeat": repeat, "shots": shots, "record_id": str(rid), "predicted": str(label), **{f"score_{c}": float(v) for c, v in zip(classes, logits)}} for rid, label, logits in zip(query["record_ids"], predicted, scores))

            add_rows(query_scores, "source_only")
            for draw in draws:
                selected = np.asarray(draw["indices"], dtype=int)
                for candidate in cfg["calibration"]["methods"]:
                    fitted = fit_candidate(context, candidate, cal_features[selected], cal_scores[selected], calibration["labels"][selected])
                    corrected = fitted.predict_scores(query_features, query_scores)
                    add_rows(corrected, candidate["id"], draw["repeat"], draw["shots"])
            print(json.dumps({"seed": seed, "arm": arm, "fit_seconds": metadata["pretrain_seconds"] + metadata["target_train_seconds"], "prediction_rows": len(predictions)}, ensure_ascii=False), flush=True)
        for key in ["initial_encoder_sha256", "initial_target_head_sha256", "target_order_sha256"]:
            if len({m[key] for m in paired}) != 1:
                raise ValueError("Paired arms differ in " + key)
    csv_write(output / "predictions_without_truth.csv.gz", predictions)
    write_json(output / "training_summary.json", training)
    write_json(output / "prediction_lock.json", {"protocol_sha256": sha(protocol_path), "predictions_sha256": sha(output / "predictions_without_truth.csv.gz"), "rows": len(predictions), "prediction_sample_count": len(query["record_ids"]), "all_predeclared_arms_and_seeds_complete": True, "scoring_started": False})


def score_frozen(split_path, output):
    from collections import defaultdict

    lock = json.loads((output / "prediction_lock.json").read_text())
    cfg = json.loads((output / "protocol.json").read_text())
    if lock["protocol_sha256"] != sha(output / "protocol.json") or lock["predictions_sha256"] != sha(output / "predictions_without_truth.csv.gz"):
        raise ValueError("Frozen predictions/protocol bytes changed")
    if sha(split_path) != cfg["split_sha256"]:
        raise ValueError("Frozen truth/split changed")
    truth = {r["record_id"]: r["origin_group_id"] for r in read_csv(split_path) if r["species"] == "酸枣仁" and r["role"] == "prediction"}
    with gzip.open(output / "predictions_without_truth.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != lock["rows"]:
        raise ValueError("Prediction row count changed")
    groups = defaultdict(list)
    for row in rows:
        row["truth"] = truth[row["record_id"]]
        row["correct"] = row["truth"] == row["predicted"]
        groups[(row["arm"], int(row["seed"]), row["method"], int(row["repeat"]), int(row["shots"]))].append(row)
    results = []
    for (arm, seed, method, repeat, shots), subset in groups.items():
        if len(subset) != len(truth) or {r["record_id"] for r in subset} != set(truth):
            raise ValueError("Every setting must predict the identical frozen query")
        y = [r["truth"] for r in subset]
        predicted = [r["predicted"] for r in subset]
        matrix = confusion_matrix(y, predicted, labels=cfg["target_classes"])
        results.append({"arm": arm, "seed": seed, "method": method, "repeat": repeat, "shots": shots, "n": len(subset), "correct": sum(r["correct"] for r in subset), "accuracy": accuracy_score(y, predicted), "macro_f1": f1_score(y, predicted, labels=cfg["target_classes"], average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(y, predicted), "confusion_matrix": matrix.tolist()})
    grouped = defaultdict(list)
    for record in results:
        grouped[(record["arm"], record["method"], record["shots"])].append(record)
    summary = []
    for (arm, method, shots), values in grouped.items():
        summary.append({"arm": arm, "method": method, "shots": shots, "seed_count": len({r["seed"] for r in values}), "support_repeats": 1 if method == "source_only" else cfg["calibration"]["repeats"], "query_samples": len(truth), **{metric + "_mean": float(np.mean([r[metric] for r in values])) for metric in ["accuracy", "macro_f1", "balanced_accuracy"]}})
    baseline = {r["seed"]: r for r in results if r["arm"] == "A" and r["method"] == "source_only"}
    paired = []
    for row in results:
        if row["method"] == "source_only":
            paired.append({"arm": row["arm"], "seed": row["seed"], "correct": row["correct"], "n": row["n"], "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "balanced_accuracy": row["balanced_accuracy"], "accuracy_gain_vs_A_pp": 100 * (row["accuracy"] - baseline[row["seed"]]["accuracy"]), "macro_f1_gain_vs_A": row["macro_f1"] - baseline[row["seed"]]["macro_f1"]})
    csv_write(output / "scored_predictions.csv.gz", rows)
    write_json(output / "metrics_by_setting.json", results)
    csv_write(output / "summary.csv", summary)
    csv_write(output / "paired_source_transfer.csv", paired)
    write_json(output / "result.json", {"status": "complete", "scope": cfg["study_status"], "protocol_sha256": lock["protocol_sha256"], "prediction_lock_sha256": sha(output / "prediction_lock.json"), "all_three_seeds": True, "summary": summary, "paired_transfer": paired, "limits": ["historically exposed target records", "one declared historical source lot per target group; no new batch test", "shape transfer with unverified absolute physical compatibility", "support resampling does not increase independent query samples or batch count"]})
    print(json.dumps({"status": "scored", "settings": len(results), "predictions": len(rows), "source_transfer": paired}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/research/species_revision_20261108/features.npz")
    parser.add_argument("--split", type=Path, default=ROOT / "data/research/species_revision_20261108/split_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/species_revision_20261108/transfer")
    parser.add_argument("--phase", choices=["all", "fit", "score"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase in {"all", "fit"}:
        fit_predict(args.features.resolve(), args.split.resolve(), args.output.resolve())
    if args.phase == "all":
        subprocess.run([sys.executable, __file__, "--phase", "score", "--split", str(args.split.resolve()), "--output", str(args.output.resolve())], check=True)
    elif args.phase == "score":
        score_frozen(args.split.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
