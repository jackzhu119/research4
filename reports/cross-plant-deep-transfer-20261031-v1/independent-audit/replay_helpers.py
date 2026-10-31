"""No-fit independent tensor, normalization, minibatch and native replay helpers."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from herb_spectra.leakage_guard import array_fingerprint, source_fingerprint
from herb_spectra.upgrade_neural import Encoder

CLASSES = tuple(f"ZY{i:02d}" for i in range(1, 8))
ARMS = ("A", "Aextra", "B", "C")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snv(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def tensor_state(value):
    values = value.state_dict() if hasattr(value, "state_dict") else value
    h = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def metric(y, p):
    y, p = np.asarray(y), np.asarray(p)
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "confusion_matrix": confusion_matrix(y, p, labels=CLASSES).tolist(),
            "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])} for i, c in enumerate(CLASSES)}}


class Review:
    def __init__(self):
        self.checks, self.values, self.maximum = 0, 0, 0.0
        self.counts = Counter()
        self.initials = {}

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def close(self, a, b, message, atol=1e-10):
        a, b = np.asarray(a), np.asarray(b)
        self.check(a.shape == b.shape, message + " shape")
        self.values += a.size
        if a.size:
            self.maximum = max(self.maximum, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
        self.check(np.allclose(a, b, atol=atol, rtol=1e-10), message)

    def native(self, fitted, query, rows=None):
        z = fitted.scaler.transform(snv(query["X"])).astype(np.float32)
        fitted.model.eval()
        before = tensor_state(fitted.model)
        with torch.no_grad():
            logits = fitted.model({"alpha": torch.as_tensor(z)}).cpu().numpy()
        self.check(before == tensor_state(fitted.model), "native inference leaves state unchanged")
        predicted = np.asarray(CLASSES)[logits.argmax(axis=1)]
        if rows is not None:
            self.check([r["record_id"] for r in rows] == query["record_ids"].tolist(), "native replay routing order")
            self.check(predicted.tolist() == [r["y_predicted"] for r in rows], "native replay labels")
            self.close(logits, [[r["score_" + c] for c in CLASSES] for r in rows], "native logits", atol=1e-6)
        self.counts["native_prediction_rows"] += len(predicted)
        return predicted, logits

    def _scaler(self, scaler, x, message):
        normalized = snv(x)
        self.close(scaler.mean_, normalized.mean(axis=0), message + " mean")
        self.close(scaler.var_, normalized.var(axis=0), message + " variance")
        self.check(int(scaler.n_samples_seen_) == len(x), message + " current source count")
        return scaler.transform(normalized).astype(np.float32), normalized

    def pipeline(self, directory, source, auxiliary, query=None, rows=None):
        directory = Path(directory)
        metadata = read(directory / "training.json")
        for key, name in [("pipeline_sha256", "pipeline.joblib"), ("batch_trace_sha256", "batch_trace.jsonl.gz"),
                          ("batch_schedule_sha256", "batch_schedule.json"), ("fit_scope_sha256", "fit_scope.jsonl")]:
            self.check(metadata[key] == sha(directory / name), "pipeline evidence hash " + name)
        fitted = joblib.load(directory / "pipeline.joblib")
        for key, value in fitted.metadata.items():
            self.check(metadata[key] == value, "persisted metadata " + key)
        arm, profile, seed = metadata["arm"], metadata["profile"], metadata["seed"]
        ids, labels = source["record_ids"], source["source_labels"]
        self.check(metadata["source_ids"] == ids.tolist(), "current target source membership/order")
        self.check(metadata["raw_target_source_fingerprint"] == source_fingerprint(source["X"], labels), "target source raw values/labels")
        self.check(metadata["axis_sha256"] == array_fingerprint(source["axis"]), "frequency axis provenance")
        z, normalized = self._scaler(fitted.scaler, source["X"], "target scaler")
        encoded = np.asarray([CLASSES.index(str(v)) for v in labels], dtype=np.int64)
        counts = np.bincount(encoded, minlength=7)
        self.check(metadata["source_class_counts"] == dict(zip(CLASSES, counts.tolist())), "target class counts")
        self.close(metadata["target_class_weights"], len(ids) / (7 * counts), "source-only target class weights")
        initial_key = (seed, profile["width"], profile["blocks"], profile["dropout"], profile["residual"], profile["attention"])
        if initial_key not in self.initials:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                encoder = Encoder(profile["width"], profile["blocks"], profile["dropout"], profile["residual"], profile["attention"])
                torch.manual_seed(seed + 10000)
                head = nn.Linear(16, 7)
            self.initials[initial_key] = (tensor_state(encoder), tensor_state(head))
        self.check((metadata["initial_encoder_sha256"], metadata["initial_zy_head_sha256"]) == self.initials[initial_key], "paired independent constructor initialization")
        checkpoint = fitted.pretrained_state
        self.check(tensor_state(checkpoint["fresh_zy_head"]) == metadata["initial_zy_head_sha256"], "fresh ZY head untouched by pretraining")
        self.check(tensor_state(checkpoint["encoder"]) == metadata["pretrained_encoder_sha256"], "saved pretrained encoder")
        reset = {name: value.detach().clone() for name, value in checkpoint["encoder"].items()}
        for name, values in metadata["BN_before_reset"].items():
            self.close(values, checkpoint["encoder"][name].numpy(), "BN before reset", atol=0)
            reset[name].fill_(1 if name.endswith("running_var") else 0)
            self.close(metadata["BN_after_reset"][name], reset[name].numpy(), "BN reset value", atol=0)
        self.check(tensor_state(reset) == metadata["reset_encoder_sha256"], "BN reset preserves pretrained parameters")
        self.check(tensor_state(fitted.model) == metadata["fitted_state_sha256"], "native fitted model state")
        pools = [] if arm == "A" else ["target"] if arm == "Aextra" else ["ginseng"] if arm == "B" else ["huangqi", "tianma", "ginseng"]
        self.check(set(fitted.auxiliary_scalers) == set(pools) - {"target"}, "separate auxiliary scaler domains")
        if pools:
            self.check(set(checkpoint["heads"]) == set(pools) == set(checkpoint["classes"]), "separate supervised heads")
        transformed, targets, packs, class_orders = {"target": z}, {"target": encoded}, {"target": source}, {"target": list(CLASSES)}
        expected_operations = {"target_source_scaler": [(ids, source_fingerprint(normalized))],
                               "target_torch_training": [(ids, source_fingerprint(z, encoded))]}
        for name in pools:
            pool = source if name == "target" else auxiliary[name]
            classes = list(CLASSES) if name == "target" else np.unique(pool["source_labels"]).tolist()
            self.check(checkpoint["classes"][name] == classes, "source head native vocabulary")
            self.check(checkpoint["heads"][name]["weight"].shape == (len(classes), 16), "source head shape")
            if name != "target":
                values, norm = self._scaler(fitted.auxiliary_scalers[name], pool["X"], "auxiliary " + name)
                transformed[name] = values
                targets[name] = np.asarray([classes.index(str(v)) for v in pool["source_labels"]], dtype=np.int64)
                packs[name], class_orders[name] = pool, classes
                expected_operations.setdefault("external_source_scaler", []).append((pool["record_ids"], source_fingerprint(norm)))
            expected_operations.setdefault("source_head_pretraining_trajectory", []).append((pool["record_ids"], source_fingerprint(transformed[name], targets[name])))
            identity = metadata["auxiliary_identity"][name]
            self.check(identity["record_ids"] == pool["record_ids"].tolist() and identity["classes"] == classes, "auxiliary pool identity")
            self.check(identity["axis_sha256"] == array_fingerprint(pool["axis"]), "auxiliary coordinate identity")
            self.check(identity["data_fingerprint"] == source_fingerprint(pool["X"], pool["source_labels"]), "auxiliary data/label identity")
        events = [json.loads(line) for line in (directory / "fit_scope.jsonl").read_text().splitlines()]
        self.check(not any(e["event"] in {"fit_rejected", "fit_failed"} for e in events), "no rejected/failed hidden learning operation")
        starts = [e for e in events if e["event"] == "fit_start"]
        self.check(len(starts) == sum(map(len, expected_operations.values())), "actual learned-operation count")
        unmatched = {key: list(value) for key, value in expected_operations.items()}
        for event in starts:
            key = event["component"]
            self.check(key in unmatched, "known learning component")
            candidates = unmatched[key]
            found = [i for i, (allowed, fingerprint) in enumerate(candidates)
                     if event["fit_sample_ids"] == allowed.tolist() and event["data_fingerprint"] == fingerprint]
            self.check(len(found) == 1, "actual learned source values/labels/IDs")
            candidates.pop(found[0]); self.counts["fit_start_" + key] += 1
        schedule = read(directory / "batch_schedule.json")
        presteps = 0 if not pools else profile["pretrain_steps"]
        self.check(len(schedule["pretrain"]) == presteps == metadata["pretrain_steps"], "pretraining update count")
        generators = {name: torch.Generator().manual_seed(seed + 30000 + ordinal) for ordinal, name in enumerate(pools)}
        expected_batches = []
        for step, entry in enumerate(schedule["pretrain"]):
            name = pools[step % len(pools)]
            self.check(entry["pool"] == name, "pretraining task order")
            selected = torch.randperm(len(packs[name]["record_ids"]), generator=generators[name])[:profile["pretrain_batch_size"]].numpy()
            self.check(entry["indices"] == selected.tolist(), "actual pretraining draw RNG")
            expected_batches.append(("pretrain", name, step, selected))
        target_generator = torch.Generator().manual_seed(seed + 60000)
        self.check(len(schedule["target_epochs"]) == profile["epochs"], "target epoch count")
        target_step = 0
        for epoch, order in enumerate(schedule["target_epochs"]):
            expected = torch.randperm(len(ids), generator=target_generator).numpy()
            self.check(order == expected.tolist(), "paired target minibatch RNG/order")
            for left in range(0, len(ids), profile["target_batch_size"]):
                expected_batches.append(("target", "ZY", target_step, expected[left:left + profile["target_batch_size"]]))
                target_step += 1
        self.check(metadata["target_order_sha256"] == object_sha(schedule["target_epochs"]), "target order hash")
        self.check(metadata["target_steps"] == target_step, "actual target optimizer steps")
        self.check(metadata["pretrain_pool_steps"] == dict(Counter(e["pool"] for e in schedule["pretrain"])), "source task update counts")
        with gzip.open(directory / "batch_trace.jsonl.gz", "rt") as handle:
            trace = [json.loads(line) for line in handle]
        self.check(len(trace) == len(expected_batches), "all actual optimizer steps traced")
        target_losses, prelosses = [], []
        for row, (stage, head, step, selected) in zip(trace, expected_batches):
            name = "target" if stage == "target" else head
            pack = packs[name]
            actual_ids = pack["record_ids"][selected].tolist()
            self.check((row["stage"], row["head"], row["step"]) == (stage, head, step), "optimizer stage/head/step")
            self.check(row["fit_sample_ids"] == actual_ids and len(set(actual_ids)) == len(actual_ids), "actual optimizer sample IDs")
            self.check(row["allowed_source_ids_sha256"] == object_sha(pack["record_ids"].tolist()), "allowed source fingerprint")
            self.check(row["X_fingerprint"] == array_fingerprint(transformed[name][selected]), "actual optimizer X values")
            self.check(row["y_fingerprint"] == array_fingerprint(targets[name][selected]), "actual optimizer target values")
            self.check(np.isfinite(row["loss"]), "finite actual optimizer loss")
            self.counts["optimizer_step_" + stage] += 1
            if stage == "pretrain":
                prelosses.append(row["loss"])
            else:
                target_losses.append((row["loss"], len(selected)))
        self.close(prelosses, metadata["pretrain_loss"], "pretraining loss trajectory", atol=0)
        batches_per_epoch = (len(ids) + profile["target_batch_size"] - 1) // profile["target_batch_size"]
        means = [sum(loss * count for loss, count in target_losses[e * batches_per_epoch:(e + 1) * batches_per_epoch]) / len(ids)
                 for e in range(profile["epochs"])]
        self.close(means, metadata["target_epoch_loss"], "actual target loss trajectory")
        with gzip.open(directory / "batch_trace.jsonl.gz", "rb") as handle:
            decoded_sha = hashlib.sha256(handle.read()).hexdigest()
        signature = {"state": tensor_state(fitted.model), "target_mean": array_fingerprint(fitted.scaler.mean_),
                     "target_var": array_fingerprint(fitted.scaler.var_), "pretrained": tensor_state(checkpoint["encoder"]),
                     "head": tensor_state(checkpoint["fresh_zy_head"]), "decoded_trace": decoded_sha,
                     "schedule": sha(directory / "batch_schedule.json")}
        self.counts["pipeline_weights"] += 1
        if pools:
            self.counts["pretrained_trajectories"] += 1
        prediction = None
        if query is not None:
            prediction = self.native(fitted, query, rows)
        return fitted, metadata, signature, prediction


def compare_metric(review, actual, expected, message):
    for key in ("accuracy", "macro_f1", "balanced_accuracy"):
        review.close(actual[key], expected[key], message + " " + key)
    if "confusion_matrix" in actual:
        review.check(actual["confusion_matrix"] == expected["confusion_matrix"], message + " confusion matrix")
