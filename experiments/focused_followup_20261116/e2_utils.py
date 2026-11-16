"""Shared, source-specific utilities for the focused E2/E3 diagnostics."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("FOCUSED_SOURCE_ROOT", ROOT / ".cache/focused_followup_20261116"))
REPORT = Path(os.environ.get("FOCUSED_RESULT_ROOT", ROOT / "reports/focused_followup_20261116"))
BACKGROUND = SOURCE / "background"


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_write(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def receipt(status, detail, **more):
    task = os.environ.get("FOCUSED_TASK_ID", "E2_E3")
    if "artifacts" in more:
        normalized = []
        for item in more["artifacts"]:
            path = Path(item)
            if not path.is_absolute():
                path = ROOT / path if path.parts[:1] == ("reports",) else REPORT / path
            normalized.append(str(path.relative_to(ROOT)))
        more["artifacts"] = normalized
    dump(REPORT / "task_results" / (task + ".json"), {"status": status, "detail": detail, **more})


def contexts():
    path = REPORT / "e0/common_contexts.json"
    if not path.exists():
        raise FileNotFoundError("E0 exact paired context artifact missing: " + str(path))
    value = read_json(path)
    rows = value["contexts"]
    if len(rows) < 15:
        raise FileNotFoundError("Fewer than three common complete five-fold repeats")
    return rows[:15]


def job_dir(job_id):
    path = BACKGROUND / "jobs" / str(job_id)
    if not path.is_dir():
        raise FileNotFoundError("Archived job missing: " + str(job_id))
    return path


def load_job(job_id):
    path = job_dir(job_id)
    result = read_json(path / "result.json")
    split_path = next((path / name for name in ("split_frozen.json", "split_frozen.json.gz") if (path / name).exists()), None)
    if split_path is None:
        raise FileNotFoundError("Actual split IDs missing for job " + str(job_id))
    return result, read_json(split_path), path


def assert_context_split(context, split):
    for role in ("outer_train_ids", "outer_test_ids"):
        if list(map(str, context[role])) != list(map(str, split[role])):
            raise ValueError("Archived actual ID order differs for " + role)
    if set(split["outer_train_ids"]) & set(split["outer_test_ids"]):
        raise ValueError("Training/test entity overlap")


def core_data():
    path = BACKGROUND / "inputs/current1672_paired612.npz"
    if not path.exists():
        raise FileNotFoundError("Pinned paired612 source NPZ missing")
    expected = "76bde1b77584f9a6aba0e97cbb4fee2cb702f2da727fd26084c1f16209217b73"
    if file_hash(path) != expected:
        raise ValueError("Pinned background input SHA256 differs")
    with np.load(path, allow_pickle=False) as source:
        z = {k: source[k].copy() for k in source.files}
    ix = np.flatnonzero((z["material_labels"] == "酸枣仁") & (z["roles"] == "train"))
    ids = np.asarray(z["record_ids"][ix]).astype(str)
    classes, y = np.unique(z["group_ids"][ix], return_inverse=True)
    if len(ids) != 300 or len(set(ids)) != 300 or len(classes) != 5 or not np.all(np.bincount(y) == 60):
        raise ValueError("Wrong corrected core300 population")
    alpha, n = z["X_alpha"][ix], z["X_n"][ix]
    if alpha.shape != (300, 612) or n.shape != alpha.shape or not np.isfinite(alpha).all() or not np.isfinite(n).all():
        raise ValueError("Invalid paired feature matrices")
    frequency = None
    for key in ("axis", "frequency", "frequencies", "frequency_THz", "freq_thz", "frequency_thz"):
        if key in z and np.asarray(z[key]).size == 612:
            frequency = np.asarray(z[key]).reshape(-1)
            break
    if frequency is None or len(frequency) != 612 or not np.isfinite(frequency).all() or not np.all(np.diff(frequency) > 0):
        raise ValueError("Pinned actual frequency grid missing, nonfinite or not strictly increasing")
    return {"ids": ids, "classes": classes, "y": y, "alpha": alpha, "n": n,
            "frequency": frequency, "input_sha256": expected, "all_npz_keys": list(z)}


def indices_for(ids, wanted):
    index = {str(value): i for i, value in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate input IDs")
    try:
        return np.array([index[str(value)] for value in wanted], dtype=int)
    except KeyError as error:
        raise ValueError("Frozen split contains unknown input ID") from error


def snv(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, ddof=1, keepdims=True), np.finfo(float).eps)


def alpha_mask(result):
    selected = np.asarray(result["selected_indices"], dtype=int)
    if np.any((selected < 612) | (selected >= 1224)):
        raise ValueError("Expected alpha-only global [612,1224) coordinates")
    local = selected - 612
    if len(np.unique(local)) != len(local):
        raise ValueError("Repeated alpha coordinate")
    return np.sort(local)


def numeric_parameters(result):
    p = result["chosen_parameters"]
    c, gamma = float(p["C"]), float(result["numeric_gamma"])
    if not np.isfinite(c) or not np.isfinite(gamma) or c <= 0 or gamma <= 0:
        raise ValueError("Invalid locked numerical SVM parameter")
    return {"C": c, "gamma": gamma}


def pair_summary(masks, p):
    """All unique pairs; no independence assumption or performance threshold."""
    masks = np.asarray(masks, dtype=np.uint16)
    if masks.ndim != 2 or masks.shape[1] != p:
        raise ValueError("Mask universe differs")
    count = len(masks)
    k = masks.sum(1).astype(float)
    prob = masks.mean(0) if count else np.zeros(p)
    denominator = float((k.mean() / p) * (1 - k.mean() / p)) if count else 0.
    phi = None if count < 2 or denominator == 0 else float(1 - (count / (count - 1) * prob * (1 - prob)).mean() / denominator)
    intersections, jaccards = [], []
    for start in range(0, count, 128):
        stop = min(start + 128, count)
        matrix = masks[start:stop] @ masks.T
        for offset, i in enumerate(range(start, stop)):
            inter = matrix[offset, i + 1:].astype(float)
            union = k[i] + k[i + 1:] - inter
            # Empty/empty has no selected frequency and no defined Jaccard.
            defined = union > 0
            intersections.extend(inter.tolist())
            jaccards.extend((inter[defined] / union[defined]).tolist())
    def describe(values):
        values = np.asarray(values, float)
        if not len(values):
            return {"count": 0, "mean": None, "min": None, "q025": None, "median": None, "q975": None, "max": None}
        return {"count": len(values), "mean": float(values.mean()), "min": float(values.min()),
                "q025": float(np.quantile(values, .025)), "median": float(np.median(values)),
                "q975": float(np.quantile(values, .975)), "max": float(values.max())}
    return {"mask_count": count, "universe_p": p, "K": describe(k),
            "Nogueira_Phi": phi, "Nogueira_denominator": denominator,
            "Nogueira_definition": "Definition 4: 1 - mean unbiased Bernoulli sample variance / ((mean K/p)*(1-mean K/p))",
            "intersection": describe(intersections), "exact_Jaccard": describe(jaccards),
            "empty_empty_Jaccard": "undefined; omitted from defined Jaccard denominator",
            "dependence": "Overlapping training subsets; pair distributions descriptive, not independent observations"}, prob


def spaced(scores, k, gap):
    """Exact source DP; ties keep the earlier stored solution."""
    scores = np.asarray(scores, float)
    if gap < 1 or (k - 1) * gap >= len(scores):
        raise ValueError("Infeasible exact-cardinality spacing")
    p = len(scores)
    dp = np.full((p + 1, k + 1), -np.inf)
    dp[:, 0] = 0
    take = np.zeros((p + 1, k + 1), bool)
    for i in range(1, p + 1):
        dp[i] = dp[i - 1]
        value = scores[i - 1] + dp[max(0, i - gap), :-1]
        better = value > dp[i, 1:]
        dp[i, 1:][better] = value[better]
        take[i, 1:] = better
    i, j, out = p, k, []
    while j:
        if take[i, j]:
            out.append(i - 1)
            j -= 1
            i = max(0, i - gap)
        else:
            i -= 1
    return np.array(out[::-1], dtype=int)
