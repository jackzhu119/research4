"""Two fixed source-only spectral-shift augmentations; no model selection.

Retrospective development motivated by the observed frozen-model axis sensitivity.
All 855 records are historically exposed. Only new SVCs are fitted: historical
source-trained preprocessing, MC features, C, numeric gamma and class weights
remain frozen. Both recipes and all six evaluation cases must be reported.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import resource
import signal
import time
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("frozen_robustness_20261107", ROOT / "scripts/run_frozen_oof_robustness_20261107.py")
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)
Blocked = r.Blocked
SCHEMA = "fixed-source-shift-augmentation-20261107-v1"
PRIOR = "reports/research72h-20261107-v1/frozen-robustness"
REVIEW = "reports/research72h-20261107-v1/independent-ML-results-review-001.json"
TEST = "tests/test_source_shift_augmentation_20261107.py"
RECIPES = (("half_median_bin", .5), ("one_median_bin", 1.))
EVALUATIONS = (("clean", "baseline", 0.), ("axis_minus_0p002", "axis_shift_-0.002_THz", -.002),
               ("axis_plus_0p002", "axis_shift_0.002_THz", .002))
LIMITATIONS = [
    "Retrospective exposed-development experiment explicitly motivated by the observed axis sensitivity, not independent confirmation.",
    "Both fixed recipes and all three cases are reported without ranking, hyperparameter search, seed selection or query-guided choice.",
    "Only 684 user-declared physical source records per fold; 2052 augmented rows are dependent variants, not additional specimens.",
    "One source lot per anonymous origin; no independent batch, true instrument-shift, named-origin or chemical-causal validation.",
    "Original R2 remains the historical strong baseline irrespective of any one new case score.",
    "Preparation reads historically exposed labels solely to create each fold's 684-ID source label projection; runtime fit receives only that source projection.",
    "Prediction locks precede new scoring truth access; this is logical phase separation, not an OS sandbox or never-exposed-test guarantee.",
    "Gamma, C, original seven numeric class weights, preprocessing and MC selector are frozen; no refit or recalculation of those quantities.",
    "Frequency shifts use full raw-alpha interpolation and endpoint clamp before the historical 612-point mask; no clipping of original negative alpha.",
]


def evaluation_specs():
    return [{"recipe": recipe, "case": case, "baseline_case": baseline, "delta_THz": delta}
            for recipe, _ in RECIPES for case, baseline, delta in EVALUATIONS]


def source_projection(candidate_path, splits):
    with np.load(candidate_path, allow_pickle=False) as z:
        ids, y = z["groups"].astype(str), z["y"].astype(str)
    if len(ids) != 855 or len(set(ids)) != 855:
        raise Blocked("No unique original label-to-source identity mapping")
    labels = dict(zip(ids, y, strict=True))
    return [{"fold": s["fold"], "source_ids": s["train_sample_ids"],
             "source_y": [labels[i] for i in s["train_sample_ids"]],
             "scope": "684 current-source identities only; query identities excluded"} for s in splits]


def prepare(output):
    output = Path(output).resolve()
    if output.exists(): raise Blocked("Never overwrite an existing frozen preparation")
    prior = r.read_json(ROOT / PRIOR / "freeze/protocol.json")
    summary = r.read_json(ROOT / PRIOR / "results-v1/summary.json")
    review = r.read_json(ROOT / REVIEW)
    if prior["axis_shift_status"] != "VERIFIED_SOURCE_EXPORT_HEADER_THZ" or summary["historical_baseline_correct"] != 821:
        raise Blocked("Sealed native baseline or source THz unit not verified")
    if summary["baseline_replay"]["scores_max_abs_difference"] != 0 or summary["baseline_replay"]["identical_predictions"] != 855:
        raise Blocked("Exact sealed 855 native baseline required")
    if not review["status"].startswith("PASS_"):
        raise Blocked("Independent actual-result review missing")
    for rel in (PRIOR + "/freeze/protocol.json", PRIOR + "/results-v1/summary.json", PRIOR + "/results-v1/prediction_lock.json"):
        if review['proof_artifact_sha256'].get(rel) != r.sha(ROOT / rel):
            raise Blocked("Prior sealed baseline artifacts differ from independent actual-result review")
    for rel, h in prior["input_code_hashes"].items():
        if r.sha(ROOT / rel) != h: raise Blocked("Prior frozen input/model changed: " + rel)
    for rel, h in r.read_json(ROOT / PRIOR / "results-v1/prediction_lock.json")["files_sha256"].items():
        if r.sha(ROOT / PRIOR / "results-v1" / rel) != h: raise Blocked("Sealed prior predictions changed")
    x, axis, ids = r.raw_arrays(ROOT / prior["full_input"])
    mask = (axis >= .1) & (axis <= 1.5)
    bin_width = float(np.median(np.diff(axis[mask])))
    if not np.isfinite(bin_width) or bin_width <= 0 or int(mask.sum()) != 612:
        raise Blocked("Invalid original model-band axis")
    splits = r.read_json(ROOT / prior["historical_core"] / "splits.json")
    r.validate_splits(splits, ids)
    output.mkdir(parents=True)
    projections = source_projection(ROOT / prior["candidate_input"], splits)
    packs = []
    for p in projections:
        path = output / f"source_labels_fold{p['fold']}.json"; r.write_json(path, p)
        packs.append({"fold": p["fold"], "file": path.name, "sha256": r.sha(path)})
    rels = [str(Path(__file__).resolve().relative_to(ROOT)), TEST, 'src/herb_spectra/leakage_guard.py',
            PRIOR + "/freeze/protocol.json", PRIOR + "/results-v1/prediction_lock.json",
            PRIOR + "/results-v1/summary.json", PRIOR + "/results-v1/baseline_replay.json", REVIEW]
    for _, baseline, _ in EVALUATIONS: rels.append(PRIOR + "/results-v1/predictions/" + baseline + ".csv.gz")
    hashes = {**prior["input_code_hashes"], **{rel: r.sha(ROOT / rel) for rel in rels}}
    plan = {"schema": SCHEMA, "status": "FROZEN_NOT_RUN", "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "prior_plan": prior, "input_code_hashes": hashes, "source_label_packs": packs,
            "median_model_band_bin_THz": bin_width,
            "recipes": [{"recipe": name, "h_THz": fraction * bin_width, "multiplier": fraction} for name, fraction in RECIPES],
            "evaluations": evaluation_specs(), "original_samples": 855, "source_per_fold": 684, "query_per_fold": 171,
            "source_variants_per_physical": [0, 1, -1], "variant_sample_weight": 1 / 3,
            "fit_budget": 10, "fit_type": "new SVC only; original frozen bundle.transform API",
            "C_gamma_class_weights": "Exact original estimator C, numeric _gamma and seven metadata training_class_weights, verified against original estimator.class_weight_",
            "prep_label_bearing_file_access": True, "prep_label_bearing_file_access_purpose": "source_projection_only; no label-driven recipe or parameter selection",
            "limits": {"cpu_seconds": 2000, "wall_seconds": 3600, "threads": 2, "maximum_new_weights_bytes": 20000000},
            "limitations": LIMITATIONS}
    r.write_json(output / "protocol.json", plan)
    return {"status": "FROZEN_NOT_RUN", "protocol_sha256": r.sha(output / "protocol.json"), "fits": 10, "evaluation_cases": 6}


def frozen_parameters(model):
    classes = model.classes_.astype(str)
    weights = {str(c): float(model.metadata["training_class_weights"][str(c)]) for c in classes}
    if not np.array_equal(np.asarray([weights[c] for c in classes]), model.estimator.class_weight_):
        raise Blocked("Original explicit numeric class weights disagree")
    gamma = float(model.estimator._gamma)
    if gamma != float(model.metadata["effective_gamma"]): raise Blocked("Original effective gamma metadata mismatch")
    return {"C": float(model.estimator.C), "gamma": gamma, "class_weight": weights,
            "kernel": "rbf", "break_ties": False, "decision_function_shape": "ovr", "random_state": 42}


def augmented_source(raw_source, axis, mask, h, source_ids, source_y):
    if len(source_ids) != 684 or len(set(source_ids)) != 684 or len(source_y) != 684:
        raise Blocked("Augmentation needs exactly 684 original source groups")
    x, provenance = [], []
    for variant, delta in ((0, 0.), (1, h), (-1, -h)):
        shifted = raw_source if variant == 0 else r.corrupt(raw_source, axis, 1., {"kind": "axis_shift", "level": delta}, 0)
        x.append(r.model_band(shifted, mask))
        provenance.extend({"physical_id": identifier, "variant_id": f"{identifier}::dependent_shift_{variant:+d}", "variant": variant, "delta_THz": delta, "sample_weight": 1 / 3} for identifier in source_ids)
    weights = np.full(2052, 1 / 3)
    return np.vstack(x), np.tile(np.asarray(source_y, dtype=str), 3), weights, provenance


def assert_fit_scope(source_ids, query_ids, provenance, labels, source_y, weights):
    if set(source_ids) & set(query_ids) or len(set(query_ids)) != 171: raise Blocked("Source/query overlap or wrong query population")
    if len(provenance) != 2052 or len(labels) != 2052 or weights.shape != (2052,): raise Blocked("Wrong augmented fit population")
    grouped = {i: [] for i in source_ids}
    expected = dict(zip(source_ids, source_y, strict=True))
    for row, label, weight in zip(provenance, labels, weights, strict=True):
        identifier = row["physical_id"]
        if identifier not in grouped or identifier in set(query_ids) or label != expected[identifier] or weight != 1 / 3 or row['variant_id'] != f"{identifier}::dependent_shift_{row['variant']:+d}":
            raise Blocked("Actual fit variant identity/label/weight not current source")
        grouped[identifier].append((row["variant"], weight))
    if any(sorted(v for v, _ in items) != [-1, 0, 1] or sum(w for _, w in items) != 1. for items in grouped.values()):
        raise Blocked("Each physical source must contribute exactly three variants with total weight one")
    if len({p['variant_id'] for p in provenance}) != 2052: raise Blocked('Dependent variant IDs must be unique without inventing independent samples')


def prevent_preprocessor_fits():
    from herb_spectra import staged_nested_search as staged
    stack = ExitStack()
    for obj, name in ((staged, "fit_source_candidate"), (staged, "fit_transform_bundle"),
                      (staged, "_prefix_selector"), (staged.StandardScaler, "fit"), (staged.MCUVESelector, "fit"), (staged.ScatterPreprocessor, "fit")):
        stack.enter_context(patch.object(obj, name, r.forbid_fit))
    return stack


def fit_new_svc(model, processed, labels, sample_weight, scope, variant_ids, svc_type=SVC):
    estimator = svc_type(**frozen_parameters(model))
    scope.fit(estimator, processed, labels, sample_ids=variant_ids,
              component='frozen_numeric_SVC_dependent_source_shift', sample_weight=sample_weight)
    return estimator


def save_capped_weights(estimator, path, cap):
    buffer = io.BytesIO(); joblib.dump(estimator, buffer, compress=3)
    payload = buffer.getvalue()
    current = sum(p.stat().st_size for p in Path(path).parent.glob('*.joblib'))
    if current + len(payload) > cap:
        raise Blocked('New SVC weights would exceed the 20,000,000-byte ceiling')
    with Path(path).open('xb') as f: f.write(payload)


def score(output, plan, protocol_sha):
    lock = r.read_json(output / "prediction_lock.json")
    expected = {f"predictions/{s['recipe']}__{s['case']}.csv.gz" for s in plan["evaluations"]}
    expected_evidence = {'source_fit_records.json','variant_provenance.csv.gz','fit_scope_events.jsonl'} | {f"weights/{recipe}__fold{fold}.joblib" for recipe,_ in RECIPES for fold in range(5)}
    if lock.get('status') != 'ALL_SIX_CASES_LOCKED_BEFORE_NEW_SCORING_TRUTH_ACCESS' or lock["protocol_sha256"] != protocol_sha or set(lock["predictions_sha256"]) != expected or lock["fit_count"] != 10 or set(lock['evidence_sha256']) != expected_evidence:
        raise Blocked("All six prediction cases and ten source-only fits must be locked before scoring")
    for rel, h in {**lock["predictions_sha256"], **lock["evidence_sha256"]}.items():
        if r.sha(output / rel) != h: raise Blocked("Locked augmentation evidence changed")
    # Read and validate every case identity before opening the historical truth.
    core = ROOT / plan["prior_plan"]["historical_core"]
    splits = r.read_json(core / "splits.json")
    fold_map = {i: s["fold"] for s in splits for i in s["test_sample_ids"]}
    cases = {}
    classes = [f"ZY{i:02d}" for i in range(1, 8)]
    expected_fields = {'record_id','outer_fold','y_predicted',*('score_'+c for c in classes)}
    for spec in plan["evaluations"]:
        key = spec["recipe"] + "__" + spec["case"]
        with gzip.open(output / "predictions" / (key + ".csv.gz"), "rt", newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames or []) != expected_fields: raise Blocked('Locked prediction schema/class fields changed')
            rows = list(reader)
        if len(rows) != 855 or len({v['record_id'] for v in rows}) != 855 or {v['record_id'] for v in rows} != set(fold_map): raise Blocked("Case identity mismatch")
        if any(int(v['outer_fold']) != fold_map[v['record_id']] for v in rows): raise Blocked("OOF role changed")
        if any(v['y_predicted'] not in classes or not all(np.isfinite(float(v['score_'+c])) for c in classes) for v in rows): raise Blocked('Invalid prediction class or nonfinite score')
        cases[key] = rows
    with gzip.open(core / "outer_predictions.csv.gz", "rt", newline="") as f:
        truth = {v["record_id"]: v["y_true"] for v in csv.DictReader(f) if v["method"] == "R2"}
    table, per_class, confusions = [], [], {}
    for spec in plan["evaluations"]:
        key = spec["recipe"] + "__" + spec["case"]; rows = cases[key]
        computed = r.metric_counts([truth[v['record_id']] for v in rows], [v['y_predicted'] for v in rows], classes)
        with gzip.open(ROOT / PRIOR / "results-v1/predictions" / (spec["baseline_case"] + ".csv.gz"), "rt", newline="") as f: base = list(csv.DictReader(f))
        baseline = r.metric_counts([truth[v['record_id']] for v in base], [v['y_predicted'] for v in base], classes)
        table.append({**spec, **{k: computed[k] for k in (*r.METRICS, 'correct', 'samples')},
            **{'paired_change_vs_frozen_same_case_' + m: computed[m] - baseline[m] for m in r.METRICS}})
        per_class.extend({'recipe': spec['recipe'], 'case': spec['case'], **v} for v in computed['per_class'])
        confusions[key] = computed
    r.write_csv(output / 'metrics.csv', table); r.write_csv(output / 'per_class_metrics.csv', per_class)
    r.write_json(output / 'confusions_and_metrics.json', {'class_order': classes, 'cases': confusions})
    return table


def run(protocol_path, release_path, output):
    output = Path(output).resolve(); protocol_path = Path(protocol_path).resolve()
    plan, release = r.read_json(protocol_path), r.read_json(release_path)
    if plan['schema'] != SCHEMA or release.get('approved') is not True or release.get('protocol_sha256') != r.sha(protocol_path): raise Blocked('Root-reviewed frozen release required')
    if plan['evaluations'] != evaluation_specs() or [p['recipe'] for p in plan['recipes']] != [n for n, _ in RECIPES]: raise Blocked('Fixed recipes/cases changed')
    for recipe, (_, multiplier) in zip(plan['recipes'], RECIPES, strict=True):
        if recipe['multiplier'] != multiplier or recipe['h_THz'] != multiplier * plan['median_model_band_bin_THz']: raise Blocked('Fixed augmentation magnitude changed')
    for rel, h in plan['input_code_hashes'].items():
        if r.sha(ROOT / rel) != h: raise Blocked('Frozen input/code changed: ' + rel)
    for pack in plan['source_label_packs']:
        if r.sha(protocol_path.parent / pack['file']) != pack['sha256']: raise Blocked('Source-only label projection changed')
    for name, version in plan['prior_plan']['historical_versions'].items():
        if importlib.metadata.version(name) != version: raise Blocked('Historical native pipeline dependency changed')
    if output.exists(): raise Blocked('Never overwrite augmentation results')
    output.mkdir(parents=True); (output/'weights').mkdir(); (output/'predictions').mkdir()
    start, cpu0 = time.monotonic(), time.process_time()
    cpu_limit = int(sum(resource.getrusage(resource.RUSAGE_SELF)[:2])) + 2000
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    def budget_stop(*_): raise Blocked('Finite augmentation CPU/wall budget exceeded')
    old_alarm = signal.signal(signal.SIGALRM, budget_stop); old_cpu = signal.signal(signal.SIGXCPU, budget_stop); signal.alarm(3600)
    try:
        from herb_spectra.staged_nested_search import StagedSVM
        from herb_spectra.leakage_guard import FitScope
        prior = plan['prior_plan']; x, axis, ids = r.raw_arrays(ROOT / prior['full_input']); mask = (axis >= .1) & (axis <= 1.5)
        index = {i: n for n, i in enumerate(ids)}; splits = r.read_json(ROOT / prior['historical_core'] / 'splits.json'); r.validate_splits(splits, ids)
        predictions = {s['recipe']+'__'+s['case']: [] for s in plan['evaluations']}; fits, provenance = [], []
        with threadpool_limits(limits=2), prevent_preprocessor_fits():
            for model_spec in prior['models']:
                fold = model_spec['fold']; roles = splits[fold]; source_ids, query_ids = roles['train_sample_ids'], roles['test_sample_ids']
                source_ix, query_ix = np.array([index[i] for i in source_ids]), np.array([index[i] for i in query_ids])
                pack_spec = next(v for v in plan['source_label_packs'] if v['fold'] == fold); pack = r.read_json(protocol_path.parent / pack_spec['file'])
                if pack['source_ids'] != source_ids: raise Blocked('Actual source label identities changed')
                y = np.asarray(pack['source_y'], dtype=str)
                model = joblib.load(ROOT / model_spec['model']); r.validate_native_model(model, model_spec['config'], r.model_band(x[source_ix], mask))
                if joblib.hash(y) != model.metadata['source_label_fingerprint']: raise Blocked('Source labels differ from native training metadata')
                bundle_sha = joblib.hash(model.bundle)
                for recipe in plan['recipes']:
                    raw, labels, sample_weight, prov = augmented_source(x[source_ix], axis, mask, recipe['h_THz'], source_ids, y)
                    assert_fit_scope(source_ids, query_ids, prov, labels, y, sample_weight)
                    processed = model.bundle.transform(raw); fit_start, fit_cpu = time.monotonic(), time.process_time()
                    variant_ids = [p['variant_id'] for p in prov]
                    scope = FitScope(variant_ids, f"historical_outer{fold}:{recipe['recipe']}",
                        '2026-10-31-zaoren-numbered-v1:dependent-source-variants', output/'fit_scope_events.jsonl', code_version=r.sha(Path(__file__)))
                    estimator = fit_new_svc(model, processed, labels, sample_weight, scope, variant_ids)
                    fit_record = {'recipe': recipe['recipe'], 'outer_fold': fold, 'source_physical_count': 684, 'dependent_variant_rows': 2052,
                        'query_ids_excluded': True, 'source_ids': source_ids, 'source_labels_sha256': joblib.hash(y), 'processed_X_fingerprint': joblib.hash(processed),
                        'actual_augmented_y_fingerprint':joblib.hash(labels), 'actual_sample_weight_fingerprint':joblib.hash(sample_weight),
                        'actual_parent_variant_provenance_fingerprint':joblib.hash(prov),
                        'fit_guard':'Custom source-parent/weight guard plus library FitScope wrapping the actual SVC.fit; variant IDs denote dependent rows, not independent specimens',
                        'library_fit_proof':{k:v for k,v in estimator._leakage_fit_provenance.items() if k!='fit_sample_ids'},
                        'variant_sample_weight': 1/3, 'physical_total_sample_weight': 1., 'frozen_C': float(estimator.C), 'frozen_numeric_gamma': float(estimator._gamma),
                        'frozen_class_weights': frozen_parameters(model)['class_weight'], 'fit_cpu_seconds': time.process_time()-fit_cpu, 'fit_wall_seconds': time.monotonic()-fit_start}
                    fits.append(fit_record); provenance.extend({'recipe': recipe['recipe'], 'outer_fold': fold, **v} for v in prov)
                    r.write_json(output/'source_fit_records.json', fits)
                    path = output/'weights'/f"{recipe['recipe']}__fold{fold}.joblib"; save_capped_weights(estimator,path,plan['limits']['maximum_new_weights_bytes'])
                    adapted = StagedSVM(model.config, model.bundle, estimator, model.metadata)
                    for evaluation in [s for s in plan['evaluations'] if s['recipe'] == recipe['recipe']]:
                        q = x[query_ix] if evaluation['delta_THz'] == 0 else r.corrupt(x[query_ix], axis, 1., {'kind':'axis_shift','level':evaluation['delta_THz']}, fold)
                        predictions[recipe['recipe']+'__'+evaluation['case']].extend(r.native_rows(adapted, r.model_band(q, mask), query_ids, fold))
                    if joblib.hash(model.bundle) != bundle_sha: raise Blocked('Frozen preprocessing/selector learned state changed')
                    if time.process_time()-cpu0 > 2000 or time.monotonic()-start > 3600: raise Blocked('Finite execution budget exceeded')
                if r.sha(ROOT/model_spec['model']) != model_spec['model_sha256']: raise Blocked('Historical baseline model bytes changed')
        if len(fits) != 10: raise Blocked('Exactly ten fixed SVC fits required')
        r.write_json(output/'source_fit_records.json', fits); r.write_csv(output/'variant_provenance.csv.gz', provenance)
        for key, values in predictions.items(): r.write_csv(output/'predictions'/(key+'.csv.gz'), values)
        pred_hash = {str(p.relative_to(output)):r.sha(p) for p in sorted((output/'predictions').glob('*.csv.gz'))}
        evidence_files = [output/'source_fit_records.json', output/'variant_provenance.csv.gz', output/'fit_scope_events.jsonl', *sorted((output/'weights').glob('*.joblib'))]
        r.write_json(output/'prediction_lock.json', {'status':'ALL_SIX_CASES_LOCKED_BEFORE_NEW_SCORING_TRUTH_ACCESS','protocol_sha256':r.sha(protocol_path),
            'fit_count':len(fits),'predictions_sha256':pred_hash,'evidence_sha256':{str(p.relative_to(output)):r.sha(p) for p in evidence_files}})
        table = score(output, plan, r.sha(protocol_path))
        for rel, h in plan['input_code_hashes'].items():
            if r.sha(ROOT/rel) != h: raise Blocked('Frozen historical input changed after run: '+rel)
        summary = {'schema':SCHEMA,'status':'DONE_RETROSPECTIVE_FIXED_SOURCE_AUGMENTATION','protocol_sha256':r.sha(protocol_path),'release_sha256':r.sha(release_path),
            'fit_count':10,'fits_per_recipe':5,'new_independent_samples':0,'original_samples':855,'evaluation_cases':6,'reused_sample_prediction_rows':5130,
            'preprocessor_selector_refits':0,'historical_baseline_changed':False,'new_weights_bytes':sum(p.stat().st_size for p in (output/'weights').glob('*.joblib')),
            'cpu_seconds':time.process_time()-cpu0,'wall_seconds':time.monotonic()-start,'all_fixed_results_unranked':table,'limitations':LIMITATIONS}
        r.write_json(output/'summary.json',summary)
        r.write_json(output/'artifact_index.json',{str(p.relative_to(output)):r.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        return summary
    except Exception as exc:
        r.write_json(output/'status.json',{'status':'BLOCKED','reason':str(exc),'no_partial_results_are_confirmation':True});raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGXCPU,old_cpu)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('prepare','run'),default='prepare');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path);parser.add_argument('--release',type=Path);args=parser.parse_args()
    if args.stage=='run' and (args.protocol is None or args.release is None):parser.error('run requires frozen protocol and root-reviewed release')
    print(json.dumps(prepare(args.output) if args.stage=='prepare' else run(args.protocol,args.release,args.output),ensure_ascii=False,allow_nan=False),flush=True)
