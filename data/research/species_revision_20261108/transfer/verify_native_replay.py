"""Read-only numerical replay of the nine saved target models; no training."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from herb_spectra.upgrade_neural import Encoder, row_snv


def main():
    torch.set_num_threads(1)
    out = Path(__file__).resolve().parent
    with np.load(out.parent / "features.npz", allow_pickle=False) as pack:
        ids = pack["record_ids"].astype(str)
        x = pack["X_alpha_transfer"]
    with (out.parent / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        query_ids = sorted(r["record_id"] for r in csv.DictReader(handle) if r["role"] == "prediction" and r["species"] == "酸枣仁")
    lookup = {rid: i for i, rid in enumerate(ids)}
    values = x[[lookup[rid] for rid in query_ids]]
    with gzip.open(out / "predictions_without_truth.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        expected = {(int(r["seed"]), r["arm"], r["record_id"]): r for r in csv.DictReader(handle) if r["method"] == "source_only"}
    reports = []
    for seed in [42, 43, 44]:
        for arm in ["A", "Aextra", "B"]:
            folder = out / "models" / f"seed{seed}_{arm}"
            state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
            encoder = Encoder(12, 2, .35, residual=True, attention=False)
            head = nn.Linear(16, 5)
            encoder.load_state_dict(state["encoder"])
            head.load_state_dict(state["head"])
            encoder.eval()
            head.eval()
            scaler = joblib.load(folder / "scalers.joblib")["target_scaler"]
            z = torch.as_tensor(scaler.transform(row_snv(values)).astype(np.float32))
            with torch.no_grad():
                scores = head(encoder(z)).numpy()
                individual = np.concatenate([head(encoder(z[i:i+1])).numpy() for i in range(len(z))])
            stored = np.asarray([[float(expected[seed, arm, rid]["score_" + label]) for label in state["classes"]] for rid in query_ids])
            difference = float(np.max(np.abs(scores - stored)))
            individual_difference = float(np.max(np.abs(scores - individual)))
            predicted = np.asarray(state["classes"])[scores.argmax(1)]
            if difference > 1e-6 or not np.array_equal(predicted, np.asarray([expected[seed, arm, rid]["predicted"] for rid in query_ids])):
                raise AssertionError("Native saved model replay differs")
            if individual_difference > 1e-5 or not np.array_equal(scores.argmax(1), individual.argmax(1)):
                raise AssertionError("Prediction depends on query batch composition")
            reports.append({"seed": seed, "arm": arm, "n": len(z), "saved_logit_max_abs_difference": difference, "single_row_vs_batch_max_abs_difference": individual_difference, "predictions_match": True})
    result = {"status": "passed", "models": reports, "models_count": len(reports), "new_model_fits": 0}
    (out / "native_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
