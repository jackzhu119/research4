#!/usr/bin/env python3
"""E2: existing selection masks and fixed-training-set MC randomness.

No classification is performed. The seed task emits both rank settings using
the same actual stratified subsample IDs, retaining all negative results.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from e2_utils import (BACKGROUND, REPORT, SOURCE, alpha_mask, assert_context_split,
                      contexts, core_data, csv_write, dump, file_hash, ids_hash,
                      indices_for, load_job, pair_summary, read_json, receipt,
                      snv, spaced)

OUT = REPORT / "e2"


def selection_views(result):
    selected = np.asarray(result.get("selected_indices", []), int)
    family, variant = result.get("family"), result.get("variant")
    if family not in ("selection", "spacing", "rank_context", "retain_drop", "coverage"):
        return []
    if not len(selected) or np.any((selected < 0) | (selected >= 1224)):
        return []
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Duplicate global selected coordinate")
    method = str(family) + "/" + str(variant)
    if family in ("selection", "retain_drop") and str(variant).startswith("joint"):
        views = [(method, "joint", 1224, selected)]
        views.append((method, "alpha_projection", 612, selected[(selected >= 612) & (selected < 1224)] - 612))
        return views
    if np.any(selected < 612):
        return []
    return [(method, "alpha", 612, selected - 612)]


def describe_groups(records, prefix, protocol):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["space"], record["p"], record.get("draws", "unknown"))].append(record)
    summaries, frequencies = [], []
    data_path = BACKGROUND / "inputs/current1672_paired612.npz"
    axis = core_data()["frequency"] if data_path.exists() else None
    for (method, space, p, draws), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        masks = np.zeros((len(rows), p), np.uint16)
        for i, row in enumerate(rows):
            masks[i, row["indices"]] = 1
        summary, probability = pair_summary(masks, p)
        summaries.append({"protocol": protocol, "method": method, "space": space, "mc_draws": draws,
                          "stability_source": "training-subset variation within same pipeline; MC seed fixed in archived worker",
                          **summary})
        for i, value in enumerate(probability):
            frequencies.append({"protocol": protocol, "method": method, "space": space, "mc_draws": draws,
                                "local_index": i, "global_index": i if space in ("joint", "n") else i + 612,
                                "frequency_export_value": float(axis[i % 612]) if axis is not None else "unknown",
                                "frequency_unit_status": "nominal THz export label; physical confirmation separate",
                                "selection_probability": float(value), "mask_count": len(rows)})
        # Grid groups are sensitivity analyses, never claimed physical resolution.
        for width in (4, 8, 16):
            if space == "joint":
                groups_per_block = (612 + width - 1) // width
                group_ids = np.concatenate([np.arange(612) // width, np.arange(612) // width + groups_per_block])
                group_p = 2 * groups_per_block
            else:
                group_ids = np.arange(p) // width
                group_p = int(group_ids.max()) + 1
            occupied = np.zeros((len(rows), group_p), np.uint16)
            for i, row in enumerate(rows):
                occupied[i, np.unique(group_ids[row["indices"]])] = 1
            group_summary, group_probability = pair_summary(occupied, group_p)
            summaries.append({"protocol": protocol, "method": method, "space": space,
                              "mc_draws": draws, "grid_group_width": width,
                              "tail_group_size_per_612_block": 612 % width or width,
                              "grid_origin": "first frequency in each n/alpha block",
                              "interpretation": "continuous grid-point grouping; instrument resolution and chemical peak width unconfirmed",
                              **group_summary})
            for i, value in enumerate(group_probability):
                frequencies.append({"protocol": protocol, "method": method,
                                    "space": space + "_grid" + str(width), "mc_draws": draws,
                                    "local_index": i, "global_index": "",
                                    "frequency_export_value": "grid group",
                                    "frequency_unit_status": "fixed grid grouping, no instrument resolution claim",
                                    "selection_probability": float(value), "mask_count": len(rows)})
    dump(OUT / (prefix + "_stability.json"), summaries)
    csv_write(OUT / (prefix + "_inclusion_probability.csv"), frequencies)
    return summaries


def same_context_comparisons(records, prefix):
    by_context = defaultdict(dict)
    for row in records:
        by_context[(str(row["seed"]), int(row["fold"]), row.get("draws", 20))][(row["method"], row["space"], row["p"])] = row
    comparison_rows = []
    for context, methods in sorted(by_context.items(), key=lambda item: str(item[0])):
        keys = sorted(methods)
        for ia, a in enumerate(keys):
            for b in keys[ia + 1:]:
                if a[2] != b[2] or a[1] == "joint" or b[1] == "joint" or ((a[1] == "n") != (b[1] == "n")):
                    continue
                ra, rb = methods[a], methods[b]
                if not ra.get("train_ids_hash") or ra.get("train_ids_hash") != rb.get("train_ids_hash"):
                    continue
                sa, sb = set(ra["indices"]), set(rb["indices"])
                union, inter = len(sa | sb), len(sa & sb)
                comparison_rows.append({"seed": context[0], "fold": context[1], "mc_draws": context[2],
                                        "method_A": a[0], "space_A": a[1], "method_B": b[0], "space_B": b[1],
                                        "universe_p": a[2], "K_A": len(sa), "K_B": len(sb),
                                        "intersection": inter, "Jaccard": inter / union if union else "undefined",
                                        "train_ids_sha256": ra.get("train_ids_hash"),
                                        "stability_source": "between pipelines on the identical archived training set"})
    csv_write(OUT / (prefix + "_same_training_set_pairs.csv"), comparison_rows)
    return comparison_rows


def archived_background():
    context_document = read_json(REPORT / "e0/common_contexts.json")
    common = set(map(str, context_document.get("common_repeats", [])))
    common_seeds = set()
    for row in context_document.get("contexts", []):
        common_seeds.add(str(row["seed"]))
    # E0 may preserve only first3 contexts; primary repeated-mask analysis still
    # uses its complete common repeat list, joined by job.repeat or seed.
    records, excluded = [], []
    mc_methods = {"selection/alpha_snv", "spacing/alpha_inner_gap_k62_c2", "selection/alpha_snv_c2_k62",
                  "rank_context/true_n", "rank_context/shuffle_n"}
    cached_results = []
    complete_folds = defaultdict(lambda: defaultdict(set))
    for path in sorted((BACKGROUND / "jobs").glob("*/result.json")):
        result = read_json(path)
        if not result.get("complete") or result.get("testing_only"):
            continue
        repeat = str(result.get("job", {}).get("repeat", result.get("seed")))
        seed = str(result.get("seed"))
        if common and repeat not in common and seed not in common:
            continue
        draws = result.get("mc_draws", result.get("job", {}).get("mc_draws", 20))
        cached_results.append((path, result, repeat, int(draws)))
        complete_folds[(int(draws), repeat)][str(result.get("family")) + "/" + str(result.get("variant"))].add(int(result["fold"]))
    common_by_draw = {}
    for draws in (100, 500):
        common_by_draw[draws] = {repeat for (count, repeat), method_folds in complete_folds.items()
                                if count == draws and all(method_folds.get(method) == set(range(5)) for method in mc_methods)}
    for path, result, repeat, draws in cached_results:
        if draws != 20 and repeat not in common_by_draw.get(draws, set()):
            continue
        method_key = str(result.get("family")) + "/" + str(result.get("variant"))
        if complete_folds[(draws, repeat)].get(method_key) != set(range(5)):
            excluded.append({"job_id": path.parent.name, "reason": "method repeat lacks all five completed outer folds"})
            continue
        try:
            _, split, _ = load_job(path.parent.name)
            train_hash = ids_hash(split["outer_train_ids"])
        except FileNotFoundError as error:
            excluded.append({"job_id": path.parent.name, "reason": str(error)})
            continue
        for method, space, p, selected in selection_views(result):
            records.append({"method": method, "space": space, "p": p, "indices": selected.tolist(),
                            "seed": result["seed"], "fold": result["fold"], "draws": draws,
                            "train_ids_hash": train_hash, "job_id": path.parent.name,
                            "components": result.get("chosen_parameters", {}).get("components"),
                            "gap": result.get("chosen_parameters", {}).get("gap"),
                            "K": result.get("chosen_parameters", {}).get("K")})
    if not records:
        raise FileNotFoundError("No verifiable common MC20 background selection masks")
    dump(OUT / "background_mask_registry.json", {"records": records, "excluded": excluded,
                                               "protocol": "background inner5x2", "common_repeats": sorted(common)})
    summary, pairs = [], []
    for draws in (20, 100, 500):
        rows = [row for row in records if row["draws"] == draws]
        if rows:
            summary.extend(describe_groups(rows, "background_mc" + str(draws), "background-inner5x2"))
            pairs.extend(same_context_comparisons(rows, "background_mc" + str(draws)))
    convergence_by_context = defaultdict(dict)
    for row in records:
        convergence_by_context[(row["method"], row["space"], row["seed"], row["fold"])][row["draws"]] = row
    convergence = []
    for key, draw_rows in sorted(convergence_by_context.items(), key=lambda item: str(item[0])):
        counts = sorted(draw_rows)
        for ia, draw_a in enumerate(counts):
            for draw_b in counts[ia + 1:]:
                a, b = draw_rows[draw_a], draw_rows[draw_b]
                if a["train_ids_hash"] != b["train_ids_hash"]:
                    raise ValueError("Different MC counts do not share actual outer training IDs")
                sa, sb = set(a["indices"]), set(b["indices"])
                convergence.append({"method": key[0], "space": key[1], "seed": key[2], "fold": key[3],
                                    "MC_A": draw_a, "MC_B": draw_b, "K_A": len(sa), "K_B": len(sb),
                                    "intersection": len(sa & sb), "Jaccard": len(sa & sb) / len(sa | sb) if sa | sb else "undefined",
                                    "train_ids_sha256": a["train_ids_hash"]})
    csv_write(OUT / "background_mc_convergence_pairs.csv", convergence)
    parameter_groups = defaultdict(list)
    for row in records:
        parameter_groups[(row["method"], row["space"], row["draws"], row.get("components"), row.get("gap"), row.get("K"))].append(row)
    conditional = []
    for key, rows in sorted(parameter_groups.items(), key=lambda item: str(item[0])):
        masks = np.zeros((len(rows), rows[0]["p"]), np.uint16)
        for i, row in enumerate(rows):
            masks[i, row["indices"]] = 1
        item, _ = pair_summary(masks, masks.shape[1])
        conditional.append({"method": key[0], "space": key[1], "MC_draws": key[2], "PLS_components": key[3],
                            "gap": key[4], "K_parameter": key[5], **item})
    dump(OUT / "background_parameter_conditions.json", conditional)
    return {"mask_count": len(records), "summary_count": len(summary), "same_training_set_pairs": len(pairs),
            "MC100_common_repeats": sorted(common_by_draw[100]), "MC500_common_repeats": sorted(common_by_draw[500]),
            "different_MC_counts_never_pooled": True}


def archived_v4():
    base = SOURCE / "v4/runs"
    records, missing = [], []
    data_path = BACKGROUND / "inputs/current1672_paired612.npz"
    axis = core_data()["frequency"] if data_path.exists() else None
    for seed in (42, 20261031, 20261108):
        for fold in range(5):
            directory = base / ("matched_r" + str(seed) + "_f" + str(fold))
            path = next((directory / name for name in ("selected_features.csv.gz", "selected_features.csv") if (directory / name).exists()), None)
            if path is None:
                missing.append(str(directory / "selected_features.csv.gz"))
                continue
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if axis is not None:
                for row in rows:
                    if row.get("frequency_THz_export_label") and not np.isclose(float(row["frequency_THz_export_label"]), axis[int(row["global_index"]) % 612], rtol=0, atol=1e-12):
                        raise ValueError("V4/background actual frequency coordinate differs")
            grouped = defaultdict(list)
            for row in rows:
                primary = str(row.get("is_primary_k", "")).lower() in ("true", "1", "1.0")
                if primary:
                    grouped[(row["family"], int(row["k"]), "inner_selected_primary_K")].append(row)
                if int(row["k"]) == 62:
                    grouped[(row["family"], int(row["k"]), "fixedK62_ablation")].append(row)
            manifest = directory / "job_manifest.json"
            train_hash = None
            if manifest.exists():
                value = read_json(manifest)
                train = value.get("outer_train_ids", value.get("source_record_ids", value.get("train_ids")))
                if train is not None:
                    train_hash = ids_hash(train)
            for (family, k, condition), rows in grouped.items():
                selected = np.array([int(row["global_index"]) for row in rows], int)
                if np.any((selected < 0) | (selected >= 1224)):
                    raise ValueError("V4 global coordinate outside paired universe")
                joint = not family.startswith(("alpha_", "n_"))
                if joint:
                    views = [("joint", 1224, selected), ("alpha_projection", 612, selected[selected >= 612] - 612)]
                elif family.startswith("n_"):
                    views = [("n", 612, selected)]
                else:
                    views = [("alpha", 612, selected - 612)]
                draws = "5x20-rankvote" if "stability5x20" in family else (100 if family.endswith("100") else 20)
                for space, p, indices in views:
                    records.append({"method": "V4/" + family + "/" + condition, "space": space, "p": p,
                                    "indices": sorted(set(indices.tolist())), "seed": seed, "fold": fold,
                                    "draws": draws, "train_ids_hash": train_hash, "source_path": str(path.relative_to(SOURCE)), "K": k})
    dump(OUT / "v4_mask_registry.json", {"records": records, "missing": missing})
    if records:
        describe_groups(records, "v4", "V4-inner2")
        same_context_comparisons(records, "v4")
    return {"status": "REUSED_VERIFIED" if records and not missing else "BLOCKED_MISSING_ARTIFACT",
            "mask_count": len(records), "missing": missing,
            "review_values": {"within_pipeline_Jaccard_approx": .811, "cross_pipeline_Jaccard_approx": .011,
                              "interpretation": "unverified review values; never program targets. Compare scope, K, alpha projection and actual training IDs before interpreting discrepancies"}}


def existing():
    ctx = contexts()
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "fixed_contexts.json", {"contexts": [row for row in ctx if int(row["fold"]) == 0],
                                      "seeds": list(range(2026111600, 2026111630)),
                                      "selection_rule": "first three common complete repeats in frozen manifest order; fold0; never selected by score",
                                      "settings": {"MC_draws": 20, "PLS_components": 2, "K": 62,
                                                   "A": "alpha-only ranking", "B": "true SNV n ranking context, final alpha-only K62"}})
    background = archived_background()
    v4 = archived_v4()
    status = "COMPLETED" if v4["status"] == "REUSED_VERIFIED" else "BLOCKED_MISSING_ARTIFACT"
    dump(OUT / "existing_summary.json", {"status": status, "background": background, "V4": v4,
                                        "physical_frequency_resolution": "unknown",
                                        "no_accuracy_inference_from_mask_stability": True,
                                        "stability_sources_kept_separate": ["between pipelines on same train IDs", "within pipeline across training subsets", "fixed train IDs across MC seeds"]})
    receipt(status, "Existing-mask stability and fixed MC context manifest saved; V4 scope explicitly recorded", new_pls_fits=0, new_classifier_fits=0, V4_status=v4["status"],
            artifacts=[str(path.relative_to(REPORT)) for path in sorted(OUT.glob("*.json"))] + [str(path.relative_to(REPORT)) for path in sorted(OUT.glob("*.csv"))])


def paired_draws(y, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    draws = []
    for _ in range(20):
        priority = rng.random(len(y))
        ix = np.sort(np.concatenate([np.flatnonzero(y == c)[np.argsort(priority[y == c], kind="stable")[:int(np.ceil(.8 * np.sum(y == c)))]] for c in classes]))
        draws.append(ix)
    return draws


def rank_with_draws(z, y, draws):
    classes = np.unique(y)
    onehot = (y[:, None] == classes).astype(float)
    coefficients, fits = [], 0
    for ix in draws:
        value = z[ix]
        scale = value.std(0)
        scale[scale <= np.finfo(float).eps] = 1.
        value = (value - value.mean(0)) / scale
        current = []
        for target in onehot[ix].T:
            sd = target.std()
            if sd <= np.finfo(float).eps:
                current.append(np.zeros(z.shape[1]))
                continue
            model = PLSRegression(n_components=2, scale=False, max_iter=500, tol=1e-6)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y residual is constant.*")
                model.fit(value, (target - target.mean()) / sd)
            current.append(np.asarray(model.coef_).reshape(-1))
            fits += 1
        coefficients.append(current)
    coefficients = np.asarray(coefficients)
    mu, sd = coefficients.mean(0), coefficients.std(0, ddof=1)
    floor = max(1., float(sd.max())) * 1e-12
    importance = np.linalg.norm(abs(mu / np.maximum(sd, floor)), axis=0)
    if not np.isfinite(importance).all():
        raise ValueError("Nonfinite ranking scores")
    return importance, fits


def mc(context_number, seed):
    if context_number not in range(3) or seed not in range(2026111600, 2026111630):
        raise ValueError("Context or MC seed outside frozen queue")
    all_contexts = contexts()
    context = [row for row in all_contexts if int(row["fold"]) == 0][context_number]
    data = core_data()
    train = indices_for(data["ids"], context["outer_train_ids"])
    test = indices_for(data["ids"], context["outer_test_ids"])
    if set(train) & set(test):
        raise ValueError("Entity overlap")
    # This transform consults training samples only; per-spectrum SNV is local.
    alpha, n = snv(data["alpha"]), snv(data["n"])
    a = StandardScaler().fit_transform(alpha[train])
    b = StandardScaler().fit_transform(np.concatenate([n[train], alpha[train]], axis=1))
    if not np.array_equal(a, b[:, 612:]):
        # Different strides can change rounding very slightly, but a scientific
        # deviation larger than machine rounding must not silently pass.
        if not np.allclose(a, b[:, 612:], rtol=1e-13, atol=1e-13):
            raise ValueError("Alpha preprocessing differs between ranking contexts")
    draws = paired_draws(data["y"][train], seed)
    relative = "mc/context" + str(context_number) + "_seed" + str(seed)
    target = OUT / relative
    target.mkdir(parents=True, exist_ok=True)
    # Seal actual rows before ranking; both settings consume this same list.
    sampled_ids = [[str(v) for v in data["ids"][train[ix]]] for ix in draws]
    draw_sha = ids_hash(sampled_ids)
    draw_path = target / "paired_draw_ids.json.gz"
    with gzip.open(draw_path, "wt", encoding="utf-8") as stream:
        json.dump({"MC_seed": seed, "class_sampling": "source default_rng priority: ceil(0.8*N_class) each draw",
                   "paired_settings": ["alpha_only", "true_n_context"], "draw_ids": sampled_ids,
                   "draw_ids_sha256": draw_sha}, stream, ensure_ascii=False)
    cpu0, wall0 = time.process_time(), time.monotonic()
    with threadpool_limits(limits=1):
        rank_a, fits_a = rank_with_draws(a, data["y"][train], draws)
        rank_b, fits_b = rank_with_draws(b, data["y"][train], draws)
    gap_job = context["methods"]["spacing/alpha_inner_gap_k62_c2"]
    locked, split, _ = load_job(gap_job)
    assert_context_split(context, split)
    locked_gap = int(locked["chosen_parameters"]["gap"])
    masks = {}
    for label, scores in (("alpha_only", rank_a), ("true_n_context", rank_b[612:])):
        for variant, gap in (("no_gap", 1), ("fixed_gap4", 4), ("locked_gap", locked_gap)):
            chosen = np.sort(np.argsort(-scores, kind="stable")[:62]) if gap == 1 else spaced(scores, 62, gap)
            masks[label + "/" + variant] = chosen.tolist()
    joint = np.sort(np.argsort(-rank_b, kind="stable")[:62])
    masks["true_n_context/joint_totalK62"] = joint.tolist()
    masks["true_n_context/joint_totalK62_alpha_projection"] = (joint[joint >= 612] - 612).tolist()
    np.savez_compressed(target / "importance.npz", alpha_only=rank_a, true_n_context=rank_b)
    result = {"status": "COMPLETED", "context": context_number, "source_repeat": context["repeat"],
              "outer_seed": context["seed"], "outer_fold": 0, "MC_seed": seed,
              "input_sha256": data["input_sha256"], "outer_train_ids_sha256": ids_hash(context["outer_train_ids"]),
              "outer_train_ids": context["outer_train_ids"], "paired_draw_ids_sha256": draw_sha,
              "MC_draws": 20, "PLS_components": 2, "K": 62,
              "locked_gap": locked_gap, "locked_gap_source_job": gap_job,
              "locked_gap_source": "pre-existing inner-selected spacing gap on these exact training IDs, held fixed for both new ranking settings",
              "masks_local_indices": masks,
              "coordinates": {"alpha_only/*": "local alpha0..611", "true_n_context/no_gap,fixed_gap4,locked_gap": "local alpha0..611",
                              "joint_totalK62": "global n0..611,alpha612..1223", "joint_totalK62_alpha_projection": "local alpha0..611, variable K"},
              "new_rankings": 2, "new_pls_coefficient_fits": fits_a + fits_b,
              "new_classifier_fits": 0, "ranking_cpu_seconds": time.process_time() - cpu0,
              "ranking_wall_seconds": time.monotonic() - wall0,
              "code_sha256": file_hash(__file__), "shared_code_sha256": file_hash(Path(__file__).with_name("e2_utils.py")),
              "interpretation": "fixed training subset MC randomness; no classification accuracy claim"}
    dump(target / "result.json", result)
    receipt("COMPLETED", "Paired fixed-training MC ranks and all derived masks saved", new_pls_fits=fits_a + fits_b, new_classifier_fits=0, result=str((target / "result.json").relative_to(REPORT)), artifacts=[str(target.relative_to(REPORT))])


def summarize():
    records, paired, missing = [], [], []
    expected = [(c, seed) for c in range(3) for seed in range(2026111600, 2026111630)]
    fits = 0
    for context, seed in expected:
        path = OUT / ("mc/context" + str(context) + "_seed" + str(seed)) / "result.json"
        if not path.exists():
            missing.append({"context": context, "seed": seed})
            continue
        item = read_json(path)
        fits += int(item["new_pls_coefficient_fits"])
        for method, indices in item["masks_local_indices"].items():
            p = 1224 if method.endswith("/joint_totalK62") else 612
            records.append({"method": method + "/train_context" + str(context), "space": "joint" if p == 1224 else "alpha",
                            "p": p, "indices": indices, "seed": seed, "fold": context,
                            "draws": 20, "train_ids_hash": item["outer_train_ids_sha256"]})
        masks = item["masks_local_indices"]
        for variant in ("no_gap", "fixed_gap4", "locked_gap"):
            a, b = set(masks["alpha_only/" + variant]), set(masks["true_n_context/" + variant])
            paired.append({"context": context, "MC_seed": seed, "variant": variant, "K_A": len(a), "K_B": len(b),
                           "intersection": len(a & b), "Jaccard": len(a & b) / len(a | b),
                           "paired_draw_ids_sha256": item["paired_draw_ids_sha256"]})
    summaries = describe_groups(records, "fixed_train_mc", "background-fixed-train-MC-randomness") if records else []
    csv_write(OUT / "fixed_train_paired_context_effects.csv", paired)
    status = "COMPLETED" if not missing else "BLOCKED_MISSING_ARTIFACT"
    dump(OUT / "fixed_train_summary.json", {"status": status, "completed_paired_rank_tasks": len(expected) - len(missing),
                                           "expected_paired_tasks": 90, "expected_rankings": 180,
                                           "new_pls_coefficient_fits": fits, "new_classifier_fits": 0,
                                           "missing": missing, "summaries": summaries,
                                           "no_accuracy_inference_from_masks": True})
    receipt(status, "Fixed-training-set MC stability summary saved", new_pls_fits=0, new_classifier_fits=0, cumulative_E2_pls_fits=fits, missing_tasks=missing,
            artifacts=["e2/fixed_train_summary.json"] + (["e2/fixed_train_mc_stability.json", "e2/fixed_train_mc_inclusion_probability.csv", "e2/fixed_train_paired_context_effects.csv"] if records else []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("existing", "mc", "summarize"))
    parser.add_argument("--context", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        if args.phase == "existing":
            existing()
        elif args.phase == "mc":
            if args.context is None or args.seed is None:
                parser.error("mc requires --context and --seed")
            mc(args.context, args.seed)
        else:
            summarize()
    except FileNotFoundError as error:
        receipt("BLOCKED_MISSING_ARTIFACT", str(error), new_pls_fits=0, new_classifier_fits=0)
        print(str(error), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
