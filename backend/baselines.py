"""
Baseline anomaly detection methods for comparative evaluation.

Implements classical unsupervised methods to serve as baselines
against the deep learning ensemble approach.
"""

import numpy as np


def _create_windows(values: np.ndarray, window_size: int = 32) -> np.ndarray:
    """Create overlapping sliding windows from a 1D array."""
    n = len(values)
    if n < window_size:
        padded = np.pad(values, (0, window_size - n), mode="edge")
        return np.expand_dims(padded, axis=0)
    shape = (n - window_size + 1, window_size)
    strides = (values.strides[0], values.strides[0])
    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides).copy()


def _window_scores_to_points(scores: np.ndarray, n_points: int, window_size: int = 32) -> np.ndarray:
    """Map per-window anomaly scores back to per-point scores."""
    point_scores = np.zeros(n_points, dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.float64)
    if n_points < window_size:
        point_scores[:] = float(scores[0])
        return point_scores
    for i, s in enumerate(scores):
        point_scores[i:i + window_size] += s
        counts[i:i + window_size] += 1.0
    counts[counts == 0] = 1.0
    return point_scores / counts


class IsolationForestDetector:
    """Isolation Forest anomaly detector.

    Builds an ensemble of random isolation trees. Anomalies are isolated
    in fewer splits on average, yielding shorter path lengths.
    """

    def __init__(self, n_trees: int = 100, max_samples: int = 256, seed: int = 42):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.seed = seed
        self.trees = []

    def _build_tree(self, data: np.ndarray, rng: np.random.RandomState, depth: int = 0, max_depth: int = 10):
        n, d = data.shape
        if n <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": n}
        feat = rng.randint(0, d)
        min_val = np.min(data[:, feat])
        max_val = np.max(data[:, feat])
        if abs(max_val - min_val) < 1e-12:
            return {"type": "leaf", "size": n}
        split = rng.uniform(min_val, max_val)
        left_mask = data[:, feat] < split
        right_mask = ~left_mask
        return {
            "type": "split",
            "feature": feat,
            "split_value": split,
            "left": self._build_tree(data[left_mask], rng, depth + 1, max_depth),
            "right": self._build_tree(data[right_mask], rng, depth + 1, max_depth),
        }

    def _path_length(self, x: np.ndarray, node: dict, depth: int = 0) -> float:
        if node["type"] == "leaf":
            n = node["size"]
            if n <= 1:
                return float(depth)
            # Average path length correction for unexamined data
            c = 2.0 * (np.log(max(1, n - 1)) + 0.5772156649) - 2.0 * (n - 1) / max(1, n)
            return float(depth) + c
        if x[node["feature"]] < node["split_value"]:
            return self._path_length(x, node["left"], depth + 1)
        return self._path_length(x, node["right"], depth + 1)

    def fit(self, data: np.ndarray):
        rng = np.random.RandomState(self.seed)
        n = len(data)
        self.trees = []
        max_depth = int(np.ceil(np.log2(max(2, min(self.max_samples, n)))))
        for _ in range(self.n_trees):
            sample_idx = rng.choice(n, size=min(self.max_samples, n), replace=False)
            tree = self._build_tree(data[sample_idx], rng, max_depth=max_depth)
            self.trees.append(tree)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        avg_path = np.zeros(n, dtype=np.float64)
        for tree in self.trees:
            for i in range(n):
                avg_path[i] += self._path_length(data[i], tree)
        avg_path /= len(self.trees)
        c = 2.0 * (np.log(max(1, self.max_samples - 1)) + 0.5772156649) - 2.0 * (self.max_samples - 1) / max(1, self.max_samples)
        # Anomaly score: higher = more anomalous
        scores = 2.0 ** (-avg_path / max(c, 1e-12))
        return scores


class LocalOutlierFactorDetector:
    """Local Outlier Factor (LOF) anomaly detector.

    LOF measures the local deviation of density of a given sample
    with respect to its neighbors. Points with substantially lower
    density than their neighbors are considered outliers.
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        k = min(self.n_neighbors, n - 1)
        if k <= 0:
            return np.ones(n)

        # Compute pairwise distances (squared Euclidean)
        distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            diff = data[i] - data
            distances[i] = np.sqrt(np.sum(diff ** 2, axis=1))

        # For each point, find k nearest neighbors
        knn_distances = np.zeros((n, k), dtype=np.float64)
        knn_indices = np.zeros((n, k), dtype=int)
        for i in range(n):
            sorted_idx = np.argsort(distances[i])
            # Skip self (index 0 in sorted)
            neighbors = sorted_idx[1:k + 1]
            knn_indices[i] = neighbors
            knn_distances[i] = distances[i, neighbors]

        # k-distance for each point
        k_distance = knn_distances[:, -1]

        # Reachability distance
        reach_dist = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            for j_idx in range(k):
                j = knn_indices[i, j_idx]
                reach_dist[i, j_idx] = max(k_distance[j], distances[i, j])

        # Local reachability density
        lrd = np.zeros(n, dtype=np.float64)
        for i in range(n):
            mean_reach = np.mean(reach_dist[i])
            lrd[i] = 1.0 / max(mean_reach, 1e-12)

        # LOF score
        lof_scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            neighbor_lrd = lrd[knn_indices[i]]
            lof_scores[i] = np.mean(neighbor_lrd) / max(lrd[i], 1e-12)

        return lof_scores


class OneClassSVMDetector:
    """Simplified One-Class SVM using RBF kernel approximation.

    Uses distance from the centroid in RBF feature space as
    an anomaly score. A full SVM solver is too heavy without sklearn,
    so this is a kernel density-inspired approximation.
    """

    def __init__(self, gamma: float = 0.1, nu: float = 0.05):
        self.gamma = gamma
        self.nu = nu
        self.centroid = None
        self.train_scores = None

    def fit(self, data: np.ndarray):
        n = len(data)
        # Compute RBF kernel matrix
        kernel = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            diff = data[i] - data
            sq_dist = np.sum(diff ** 2, axis=1)
            kernel[i] = np.exp(-self.gamma * sq_dist)

        # Mean kernel values (how similar each point is to all others)
        self.train_kernel_mean = np.mean(kernel, axis=1)
        self.centroid = np.mean(data, axis=0)
        self.train_data = data.copy()

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            diff = data[i] - self.train_data
            sq_dist = np.sum(diff ** 2, axis=1)
            kernel_vals = np.exp(-self.gamma * sq_dist)
            # Lower mean kernel = more anomalous
            scores[i] = 1.0 - np.mean(kernel_vals)
        return scores


class StatisticalZScoreDetector:
    """Simple Z-score-based anomaly detector.

    Computes z-score for each window based on training data statistics.
    Points with high absolute z-scores are flagged as anomalies.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std < 1e-12] = 1.0

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        z_scores = np.abs((data - self.mean) / self.std)
        return np.mean(z_scores, axis=1)


def run_baseline(
    method_name: str,
    flux_values: np.ndarray,
    window_size: int = 32,
    threshold_percentile: float = 95.0,
) -> dict:
    """Run a baseline detector on flux values and return per-point scores.

    Args:
        method_name: One of 'isolation_forest', 'lof', 'ocsvm', 'zscore'
        flux_values: Normalized flux array
        window_size: Sliding window size
        threshold_percentile: Percentile for anomaly threshold

    Returns:
        Dict with scores, predictions, threshold
    """
    values = np.asarray(flux_values, dtype=np.float64)
    windows = _create_windows(values, window_size)

    if method_name == "isolation_forest":
        detector = IsolationForestDetector(n_trees=100, max_samples=min(256, len(windows)))
        detector.fit(windows)
        window_scores = detector.score_samples(windows)
    elif method_name == "lof":
        # Subsample for LOF to keep O(n^2) manageable
        max_lof = min(500, len(windows))
        stride = max(1, len(windows) // max_lof)
        sub_windows = windows[::stride]
        detector = LocalOutlierFactorDetector(n_neighbors=min(20, len(sub_windows) - 1))
        window_scores_sub = detector.score_samples(sub_windows)
        # Interpolate back to full size
        window_scores = np.interp(
            np.arange(len(windows)),
            np.arange(len(windows))[::stride],
            window_scores_sub,
        )
    elif method_name == "ocsvm":
        detector = OneClassSVMDetector(gamma=1.0 / max(1, windows.shape[1]))
        detector.fit(windows)
        window_scores = detector.score_samples(windows)
    elif method_name == "zscore":
        detector = StatisticalZScoreDetector()
        detector.fit(windows)
        window_scores = detector.score_samples(windows)
    else:
        raise ValueError(f"Unknown baseline method: {method_name}")

    point_scores = _window_scores_to_points(window_scores, len(flux_values), window_size)
    threshold = float(np.percentile(point_scores, threshold_percentile))
    predictions = (point_scores >= threshold).astype(int)

    return {
        "method": method_name,
        "scores": point_scores,
        "predictions": predictions,
        "threshold": threshold,
    }
