"""Smoke-check independent selector implementations on example/synthetic inputs.

No medicinal-plant targets are inferred. Internal selection errors are not
reported as independent prediction scores. This is not MATLAB reproduction.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from herb_spectra.selection import BoundedIRIVInspiredSelector, CARSStyleSelector, MCUVESelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, help="Optional extracted source inventory for the supplied CARS example")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    def run(case_id, function, implementation, source_relation):
        started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        before = time.monotonic()
        try:
            with threadpool_limits(limits=1):
                metrics = function()
            status, error = "pass", None
        except Exception as exc:
            metrics = {}
            status, error = "fail", f"{type(exc).__name__}: {exc}"
        row = dict(id=case_id, status=status, error=error, metrics=metrics,
                   implementation=implementation, source_relation=source_relation,
                   started_at_shanghai=started, duration_seconds=round(time.monotonic() - before, 5))
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 12))
    y = -8 * X[:, 0] + 3 * X[:, 4] + rng.normal(scale=0.05, size=len(X))

    def mc_negative():
        model = MCUVESelector(n_runs=35, n_components=4, n_features_to_select=2).fit(X, y)
        assert {0, 4}.issubset(model.selected_indices_)
        assert model.signed_stability_[0, 0] < 0
        assert np.isfinite(model.importances_).all()
        return dict(selected=model.selected_indices_.tolist(), negative_effect_feature=0,
                    retained_negative_effect=True, training_sample_units=len(X))
    run("mcuve_absolute_negative_effect", mc_negative, "MCUVESelector",
        "Independent MC-UVE implementation: absolute coefficient stability, fixed seed, standardized OVR-PLS1; not the archived signed ranking")

    def small_iriv():
        model = BoundedIRIVInspiredSelector(n_components=2, n_masks=4, max_rounds=1).fit(X[:, :3], y)
        assert len(model.selected_indices_) >= 1 and model.rounds_completed_ == 0
        assert model.evaluation_count_ <= 7
        return dict(input_features=3, selected=model.selected_indices_.tolist(),
                    candidate_evaluations=model.evaluation_count_, random_rejection_loop=False)
    run("bounded_iriv_three_features", small_iriv, "BoundedIRIVInspiredSelector",
        "Independent IRIV-inspired replacement: bounded paired inclusion/exclusion and backward elimination; not an exact IRIV recreation")

    def group_resampling():
        groups = np.arange(len(y))
        repeated = np.r_[groups, np.repeat(0, 9)]
        a = MCUVESelector(n_runs=20, n_features_to_select=3).fit(X, y, groups=groups)
        b = MCUVESelector(n_runs=20, n_features_to_select=3).fit(X[repeated], y[repeated], groups=groups[repeated])
        np.testing.assert_allclose(a.importances_, b.importances_, rtol=1e-10)
        support = b.support_.copy()
        transformed = b.transform(np.full((3, X.shape[1]), 1e6))
        np.testing.assert_array_equal(support, b.support_)
        return dict(input_scans=len(repeated), fitted_physical_sample_units=b.n_training_sample_units_,
                    sample_unit=b.sample_unit_, query_transform_shape=list(transformed.shape), fit_state_unchanged=True)
    run("selector_sample_group_aggregation", group_resampling, "MCUVESelector(groups=...)",
        "Repeat scans are averaged within the training physical sample before resampling; this is an explicit new implementation choice")

    def classification():
        labels = np.repeat([0, 1, 2], 20)
        Xc = rng.normal(size=(60, 8)); Xc[:, 0] += 3 * labels
        permuted = np.choose(labels, [91, -4, 7])
        a = CARSStyleSelector(n_runs=5, task="classification", reweight=False).fit(Xc, labels)
        b = CARSStyleSelector(n_runs=5, task="classification", reweight=False).fit(Xc, permuted)
        np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)
        return dict(classes=3, selected=a.selected_indices_.tolist(), class_id_permutation_invariant=True,
                    target_encoding="one-hot OVR coefficient norms")
    run("cars_multiclass_not_ordinal", classification, "CARSStyleSelector(task='classification')",
        "Independent multiclass adaptation uses one-hot OVR PLS coefficient norms; arbitrary class IDs are not treated as concentrations")

    def cars_example():
        if args.inventory:
            inventory = json.loads(args.inventory.read_text())
            archive = next(r for r in inventory if Path(r["archive"]).name == "0308 CARS+UVE.rar")
            member = next(m for m in archive["members"] if m["name"].endswith("corn_m5_starch.mat"))
            data = loadmat(member["path"])
            train, target, query = data["Xcal"], data["ycal"].ravel(), data["Xtest"]
            input_kind = "supplied example calibration data"
        else:
            train, target, query = X, y, X[:4] + 1e4
            input_kind = "synthetic engineering fixture"
        selector = CARSStyleSelector(n_runs=10, n_components=4, cv=3).fit(train, target)
        assert len(selector.selected_indices_) >= 1
        before = selector.selected_indices_.copy()
        transformed = selector.transform(query)
        np.testing.assert_array_equal(before, selector.selected_indices_)
        return dict(input_kind=input_kind, training_rows=len(train), input_features=train.shape[1],
                    selected_features=len(before), query_shape=list(transformed.shape),
                    query_used_in_fitting=False, internal_mse_is_not_external_validation=True)
    run("cars_style_training_only_example", cars_example, "CARSStyleSelector",
        "Independent CARS-style exponential pruning/weighted sampling with fixed PLS components; no exact numerical equivalence or novelty claim")

    if any(row["status"] == "fail" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
