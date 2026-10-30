"""Independent, executable replacements for missing legacy algorithm APIs.

These implement standard mathematical methods using NumPy/SciPy/scikit-learn.
They do not copy or emulate proprietary MathWorks internals, and do not claim
numerical reproduction of missing ``newff``, ``trainlssvm`` or ``my_pca`` code.
Model selection and preprocessing must still remain inside training partitions.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.signal import convolve2d, savgol_filter
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class LSSVMRegressor(RegressorMixin, BaseEstimator):
    """RBF least-squares SVM regression with an unpenalized intercept.

    Fit solves ``[0, 1'; 1, K + I/C] [b; alpha] = [0; y]``. Thus ``C`` is
    the error-penalty regularization parameter, while ``kernel_gamma`` is the
    RBF parameter in ``exp(-kernel_gamma * squared_distance)``. The latter is
    ``1 / (2 * sig2)`` for an RBF written ``exp(-d²/(2*sig2))``. An external
    training-only StandardScaler is usually appropriate for spectral inputs.
    This dense solver needs O(n²) memory and is intended for modest datasets.
    """

    def __init__(self, C=10.0, kernel_gamma=None):
        self.C = C
        self.kernel_gamma = kernel_gamma

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be finite and positive")
        gamma = 1.0 / X.shape[1] if self.kernel_gamma is None else self.kernel_gamma
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("kernel_gamma must be finite and positive")
        self.n_features_in_ = X.shape[1]
        self.kernel_gamma_ = float(gamma)
        self.X_fit_ = X.copy()
        n = len(X)
        K = rbf_kernel(X, gamma=self.kernel_gamma_)
        system = np.empty((n + 1, n + 1), dtype=float)
        system[0, 0] = 0
        system[0, 1:] = 1
        system[1:, 0] = 1
        system[1:, 1:] = K + np.eye(n) / self.C
        rhs = np.r_[0.0, y]
        solution = solve(system, rhs, assume_a="sym")
        self.intercept_ = float(solution[0])
        self.dual_coef_ = solution[1:]
        self.system_residual_max_ = float(np.max(np.abs(system @ solution - rhs)))
        return self

    def predict(self, X):
        check_is_fitted(self, ["X_fit_", "dual_coef_", "intercept_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature grid differs from training")
        return rbf_kernel(X, self.X_fit_, gamma=self.kernel_gamma_) @ self.dual_coef_ + self.intercept_


def _image_cube(image):
    image = np.asarray(image, dtype=float)
    if image.ndim != 3 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError("A finite height x width x bands image cube is required")
    return image


class ImagePCA(TransformerMixin, BaseEstimator):
    """PCA over image pixels with training means retained for future images.

    Input/output are H x W x bands and H x W x components, respectively. Image
    dimensions may change at inference, but the spectral band grid must match.
    This is a documented independent helper, not a drop-in ``my_pca`` clone.
    """

    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, image, y=None):
        image = _image_cube(image)
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer")
        pixels = image.reshape(-1, image.shape[-1])
        if self.n_components > min(pixels.shape):
            raise ValueError("n_components exceeds the available pixel/band dimensions")
        if np.sum(np.var(pixels, axis=0)) == 0:
            raise ValueError("PCA requires variation across image pixels")
        self.n_features_in_ = pixels.shape[1]
        self.pca_ = PCA(n_components=self.n_components, svd_solver="full").fit(pixels)
        return self

    def transform(self, image):
        check_is_fitted(self, "pca_")
        image = _image_cube(image)
        if image.shape[-1] != self.n_features_in_:
            raise ValueError("Image bands differ from training")
        values = self.pca_.transform(image.reshape(-1, image.shape[-1]))
        return values.reshape(*image.shape[:2], self.n_components)

    def inverse_transform(self, scores):
        check_is_fitted(self, "pca_")
        scores = _image_cube(scores)
        if scores.shape[-1] != self.n_components:
            raise ValueError("PCA score dimensions do not match fitted components")
        image = self.pca_.inverse_transform(scores.reshape(-1, self.n_components))
        return image.reshape(*scores.shape[:2], self.n_features_in_)


def grayscale(image):
    """Return finite 2D intensities; RGB uses fixed luminance coefficients."""
    image = np.asarray(image, dtype=float)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image @ np.array([0.299, 0.587, 0.114])
    if image.ndim != 2 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError("A finite grayscale or RGB image is required")
    return image


@dataclass(frozen=True)
class TextureFeatures:
    features: np.ndarray  # mean/std for energy, entropy, contrast, correlation
    per_direction: np.ndarray  # directions x 4 properties
    matrices: np.ndarray  # directions x levels x levels
    feature_names: tuple = ("energy_mean", "energy_std", "entropy_mean", "entropy_std",
                            "contrast_mean", "contrast_std", "correlation_mean", "correlation_std")


def glcm_texture(image, levels=16, value_range=(0.0, 255.0)):
    """Eight GLCM texture features from four actual pixel-neighbour directions.

    Matrices are symmetric and normalized. Energy here means sum(P²), matching
    the uploaded document's definition (often called ASM), entropy uses natural
    logs, and directional standard deviations use ddof=1. Correlation divides
    by standard deviations; constant marginals conventionally yield 1.
    value_range is fixed explicitly, never estimated independently per image.
    """
    gray = grayscale(image)
    if min(gray.shape) < 2:
        raise ValueError("GLCM requires at least a 2 x 2 image")
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or not 2 <= levels <= 256:
        raise ValueError("levels must be an integer from 2 to 256")
    low, high = value_range
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("value_range must be finite and increasing")
    tolerance = 1e-12 * max(1.0, abs(low), abs(high))
    if gray.min() < low - tolerance or gray.max() > high + tolerance:
        raise ValueError("Image values lie outside the declared value_range")
    bins = np.minimum(np.floor((np.clip(gray, low, high) - low) * levels / (high - low)), levels - 1).astype(np.int64)
    h, w = bins.shape
    matrices = []
    properties = []
    grid_i, grid_j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    for dy, dx in [(0, 1), (-1, 1), (-1, 0), (-1, -1)]:
        source = bins[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
        neighbour = bins[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
        counts = np.bincount((source * levels + neighbour).ravel(), minlength=levels ** 2).reshape(levels, levels)
        counts = counts + counts.T
        P = counts / counts.sum()
        marginal_i, marginal_j = P.sum(axis=1), P.sum(axis=0)
        mean_i = np.arange(levels) @ marginal_i
        mean_j = np.arange(levels) @ marginal_j
        variance_i = ((np.arange(levels) - mean_i) ** 2) @ marginal_i
        variance_j = ((np.arange(levels) - mean_j) ** 2) @ marginal_j
        denominator = np.sqrt(variance_i * variance_j)
        correlation = 1.0 if denominator <= np.finfo(float).eps else np.sum((grid_i - mean_i) * (grid_j - mean_j) * P) / denominator
        positive = P[P > 0]
        properties.append([np.sum(P ** 2), -np.sum(positive * np.log(positive)),
                           np.sum((grid_i - grid_j) ** 2 * P), float(np.clip(correlation, -1, 1))])
        matrices.append(P)
    properties = np.asarray(properties)
    features = np.column_stack([properties.mean(axis=0), properties.std(axis=0, ddof=1)]).ravel()
    return TextureFeatures(features, properties, np.asarray(matrices))


def sobel_edges(image, mode="full"):
    """Sobel magnitude with zero boundary padding, matching full-convolution size."""
    if mode not in {"full", "same", "valid"}:
        raise ValueError("mode must be full, same or valid")
    gray = grayscale(image)
    kernel = np.array([[1.0, 2.0, 1.0], [0, 0, 0], [-1.0, -2.0, -1.0]])
    a = convolve2d(gray, kernel, mode=mode, boundary="fill", fillvalue=0)
    b = convolve2d(gray, kernel.T, mode=mode, boundary="fill", fillvalue=0)
    return np.hypot(a, b)


def smooth_spectra(values, window_length=15, polyorder=2, axis=-1):
    """SG filtering along a specified signal axis; coordinate columns stay outside.

    Uses SciPy's interp endpoints. This is not a claim of exact MATLAB smooth
    endpoint equivalence. Pass scans x frequencies with the default axis=-1.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("Finite signal values are required")
    if isinstance(window_length, bool) or not isinstance(window_length, (int, np.integer)) or window_length < 1 or window_length % 2 != 1:
        raise ValueError("window_length must be a positive odd integer")
    if isinstance(polyorder, bool) or not isinstance(polyorder, (int, np.integer)) or not 0 <= polyorder < window_length:
        raise ValueError("polyorder must be nonnegative and below window_length")
    return savgol_filter(values, window_length, polyorder, axis=axis, mode="interp")


def stack_scan_rows(arrays, row_index, columns=slice(None)):
    """Stack one explicitly selected row from each numeric scan matrix.

    row_index and column selections are zero-based. The same column selection
    is applied to every matrix; incompatible lengths are rejected, never padded
    or silently truncated. The caller must confirm that these columns represent
    the same coordinates. This is a data adapter, not an image classifier.
    """
    if isinstance(row_index, bool) or not isinstance(row_index, (int, np.integer)) or row_index < 0:
        raise ValueError("row_index must be a nonnegative zero-based integer")
    selected_rows = []
    width = None
    for index, array in enumerate(arrays):
        array = np.asarray(array)
        if array.ndim != 2 or row_index >= len(array):
            raise ValueError(f"Matrix {index} does not contain the selected row")
        try:
            selected = np.asarray(array[row_index, columns], dtype=float)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid column selection or numeric values in matrix {index}") from exc
        if selected.ndim != 1 or len(selected) == 0 or not np.isfinite(selected).all():
            raise ValueError("Each selection must be a nonempty finite numeric row")
        if width is not None and len(selected) != width:
            raise ValueError(f"Selected row lengths differ: expected {width}, found {len(selected)} in matrix {index}")
        width = len(selected)
        selected_rows.append(selected)
    if not selected_rows:
        raise ValueError("At least one scan matrix is required")
    return np.stack(selected_rows)


def make_bp_classifier(hidden_layer_sizes=(25,), max_iter=300, random_state=42):
    """Training-scaled sigmoid MLP classification baseline using scikit-learn.

    Uses L-BFGS and cross-entropy. It is an independent BP/MLP implementation,
    not original newff, original hand-written MSE updates, or matching weights.
    """
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                         activation="logistic", solver="lbfgs", alpha=1e-4,
                         max_iter=max_iter, tol=1e-6, random_state=random_state))
