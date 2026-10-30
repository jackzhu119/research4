"""Exercise independent legacy-family replacements, with explicit provenance.

Examples and synthetic fixtures prove software execution only. This script
never infers herb origins, assay values, or physical-sample IDs from filenames.
"""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from herb_spectra.algorithms import (ImagePCA, LSSVMRegressor, glcm_texture,
                                    make_bp_classifier, smooth_spectra, sobel_edges,
                                    stack_scan_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/replacements'),
                        help='Directory receiving results.json, versions.json and warnings/errors')
    parser.add_argument('--inventory', type=Path,
                        help='Optional inventory produced by scripts/runtime/prepare_sources.py')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(args.inventory.read_text()) if args.inventory else []
    by_name = {Path(r['archive']).name: r for r in inventory}
    results = []

    def source(archive, suffix):
        if archive not in by_name:
            return None
        matches = [m for m in by_name[archive]['members'] if m['name'].endswith(suffix)]
        if len(matches) != 1 or 'path' not in matches[0]:
            raise ValueError(f'Inventory does not identify one readable {archive}/{suffix}')
        path = Path(matches[0]['path'])
        if not path.is_file():
            raise FileNotFoundError(f'Extracted source data missing: {path}')
        return path

    def run(test_id, function, implementation, source_relation):
        started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        start = time.monotonic()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                metrics = function()
            warning_messages = [f'{w.category.__name__}: {w.message}' for w in captured]
            status, error = 'pass', None
            if warning_messages:
                (args.output / (test_id + '.warnings.log')).write_text('\n'.join(warning_messages) + '\n')
        except Exception as exc:
            metrics = {}; status = 'fail'; error = f'{type(exc).__name__}: {exc}'
            warning_messages = []
            (args.output / (test_id + '.error.log')).write_text(traceback.format_exc())
        row = dict(id=test_id, status=status, error=error, metrics=metrics,
                   implementation=implementation, source_relation=source_relation,
                   scope='engineering/example execution, not validated medicinal-plant performance',
                   warning_count=len(warning_messages), started_at_shanghai=started,
                   duration_seconds=round(time.monotonic() - start, 6))
        results.append(row)
        (args.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def lssvm():
        path = source('0308 CARS+UVE.rar', 'corn_m5_starch.mat')
        if path:
            data = loadmat(path)
            X, y = data['Xcal'], data['ycal'].ravel()
            query, truth = data['Xtest'], data['ytest'].ravel()
            dataset = 'supplied corn_m5_starch.mat fixed example split; y units unverified'
        else:
            rng = np.random.default_rng(42)
            data = rng.normal(size=(80, 4)); target = np.sin(data[:, 0]) + data[:, 1] + 5
            X, query, y, truth = data[:60], data[60:], target[:60], target[60:]
            dataset = 'synthetic nonlinear regression fixture'
        model = make_pipeline(StandardScaler(), LSSVMRegressor(C=10))
        model.fit(X, y); predictions = model.predict(query)
        estimator = model.named_steps['lssvmregressor']
        assert len(predictions) == len(query) and np.isfinite(predictions).all()
        assert estimator.system_residual_max_ < 1e-8
        assert abs(estimator.dual_coef_.sum()) < 1e-8
        return dict(dataset=dataset, train_rows=len(X), test_rows=len(query), features=X.shape[1],
                    example_rmsep=float(np.sqrt(mean_squared_error(truth, predictions))),
                    block_system_residual_max=estimator.system_residual_max_,
                    intercept=float(estimator.intercept_))

    run('replacement_lssvm_regression', lssvm,
        'independent RBF LS-SVM block-system solver with unpenalized intercept',
        'fills missing trainlssvm capability; not the original LS-SVM toolbox or SVR renamed')

    def image_pca():
        rng = np.random.default_rng(42)
        image = rng.normal(size=(24, 20, 6)) + np.arange(6)
        model = ImagePCA(3).fit(image); reference_mean = model.pca_.mean_.copy()
        scores = model.transform(image); query = rng.normal(size=(12, 10, 6)) + 10
        query_scores = model.transform(query)
        assert scores.shape == (24, 20, 3) and query_scores.shape == (12, 10, 3)
        assert np.array_equal(model.pca_.mean_, reference_mean)
        assert np.isfinite(scores).all()
        return dict(dataset='synthetic image cubes', training_shape=list(image.shape),
                    score_shape=list(scores.shape), query_score_shape=list(query_scores.shape),
                    training_mean_unchanged=True,
                    explained_variance_ratio=model.pca_.explained_variance_ratio_.tolist())

    run('replacement_image_pca', image_pca,
        'independent pixel/band PCA helper with stored training centering',
        'fills missing my_pca functionality through a documented Python API, not a MATLAB interface clone')

    def texture():
        rng = np.random.default_rng(42)
        constant = glcm_texture(np.zeros((32, 32), dtype=np.uint8))
        random = glcm_texture(rng.integers(0, 256, size=(32, 32), dtype=np.uint8))
        difference = float(np.max(np.abs(constant.features - random.features)))
        assert difference > 0
        assert np.isfinite(random.features).all()
        assert np.all(np.abs(random.per_direction[:, 3]) <= 1)
        return dict(dataset='synthetic constant and random grayscale images',
                    max_feature_difference=difference, constant_features=constant.features.tolist(),
                    random_features=random.features.tolist(),
                    feature_names=list(random.feature_names),
                    maximum_absolute_directional_correlation=float(np.max(np.abs(random.per_direction[:, 3]))))

    run('replacement_glcm_texture', texture,
        'independent normalized symmetric GLCM with actual pixel-neighbour comparisons',
        'corrected replacement for incomplete conditions and correlation denominator in the stored Texture document')

    def sobel():
        image = np.zeros((32, 32)); image[:, 16:] = 255
        edges = sobel_edges(image)
        assert edges.shape == (34, 34) and np.isfinite(edges).all()
        assert edges[16, 16] > 0 and edges[16, 5] == 0
        return dict(dataset='synthetic step image', input_shape=[32, 32],
                    output_shape=list(edges.shape), maximum_magnitude=float(edges.max()))

    run('replacement_sobel_edges', sobel,
        'SciPy full-convolution Sobel magnitude with zero boundary padding',
        'same standard Sobel operation family as the supplied edge-detection document; real images required for research')

    def smoothing():
        x = np.arange(101, dtype=float)
        polynomial = 2 * x ** 2 + x + 3
        filtered = smooth_spectra(polynomial)
        error = float(np.max(np.abs(filtered - polynomial)))
        assert error < 1e-8
        return dict(dataset='synthetic quadratic signal', points=101, window=15, degree=2,
                    polynomial_preservation_max_error=error,
                    axis_note='scans x frequency; coordinate columns excluded by caller')

    run('replacement_sg_filter', smoothing,
        'SciPy Savitzky-Golay filtering with interp endpoints',
        'replaces legacy smooth dependency; does not assert all MATLAB smooth endpoints are numerically identical')

    def bp():
        paths = [source('BPNN.zip', f'data{i}.mat') for i in range(1, 5)]
        if all(paths):
            arrays = [loadmat(p)[f'c{i}'][:500] for i, p in enumerate(paths, 1)]
            data = np.vstack(arrays); X = data[:, 1:25]
            y = np.asarray([f'class-{int(v)}' for v in data[:, 0]])
            order = np.random.default_rng(42).permutation(len(X))
            train, test = order[:1500], order[1500:]
            dataset = 'supplied four-class voice teaching example; independent NumPy random split'
        else:
            rng = np.random.default_rng(42)
            X = np.vstack([rng.normal(i * 3, .5, size=(40, 6)) for i in range(4)])
            y = np.repeat(['class-A', 'class-B', 'class-C', 'class-D'], 40)
            order = rng.permutation(len(X)); train, test = order[:120], order[120:]
            dataset = 'synthetic four-class engineering fixture'
        model = make_bp_classifier(max_iter=200).fit(X[train], y[train])
        training_mean = model.named_steps['standardscaler'].mean_.copy()
        predicted = model.predict(X[test]); probabilities = model.predict_proba(X[test])
        assert np.isfinite(probabilities).all()
        assert set(predicted) <= set(y[train])
        assert np.array_equal(model.named_steps['standardscaler'].mean_, training_mean)
        return dict(dataset=dataset, train_rows=len(train), test_rows=len(test), features=X.shape[1],
                    example_accuracy=float(accuracy_score(y[test], predicted)),
                    classes=model.classes_.tolist(), max_iterations=200,
                    objective='cross-entropy; sigmoid hidden layer; L-BFGS solver',
                    reproducibility_note='not original manual MSE update, newff behavior or archived weights')

    run('replacement_bp_classifier', bp,
        'independent scikit-learn sigmoid MLP classifier with training-only scaling',
        'runnable BP family replacement; not original newff or numerical reproduction of manual BP')

    def row_adapter():
        arrays = [np.full((300, 92), i, dtype=float) + np.arange(92) for i in range(91)]
        stacked = stack_scan_rows(arrays, 299, slice(1, 92))
        assert stacked.shape == (91, 91)
        assert np.array_equal(stacked[0], arrays[0][299, 1:92])
        try:
            stack_scan_rows([np.zeros((1, 90)), np.zeros((1, 91))], 0)
        except ValueError:
            mismatch_rejected = True
        else:
            mismatch_rejected = False
        assert mismatch_rejected
        return dict(dataset='91 synthetic 300x92 matrices', stacked_shape=list(stacked.shape),
                    row_index_zero_based=299, column_slice_zero_based='1:92',
                    mismatched_90_91_lengths_rejected=True,
                    scope_note='data adapter only; caller must verify matching physical coordinates')

    run('replacement_scan_row_adapter', row_adapter,
        'parameterized equal-length row extraction and stacking',
        'replaces hard-coded syqpcx file paths and inconsistent column slices; not a classifier')
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn']}
    (args.output / 'versions.json').write_text(json.dumps(dict(python=sys.version, **versions), indent=2) + '\n')
    return 1 if any(row['status'] != 'pass' for row in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
