"""
Evaluation metrics for anomaly detection benchmark.

Computes Precision, Recall, F1-Score, AUC-ROC, AUC-PR, and confusion
matrices. Includes k-fold cross-validation support and statistical
significance testing.
"""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute TP, FP, TN, FN from binary arrays."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    if p + r < 1e-12:
        return 0.0
    return 2.0 * p * r / (p + r)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration for AUC calculation."""
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    return float(np.sum(0.5 * (y_sorted[1:] + y_sorted[:-1]) * np.diff(x_sorted)))


def roc_curve(y_true: np.ndarray, scores: np.ndarray, n_thresholds: int = 200) -> dict:
    """Compute ROC curve (FPR vs TPR) and AUC-ROC."""
    min_score, max_score = np.min(scores), np.max(scores)
    # Ensure thresholds cover the full range
    thresholds = np.linspace(max_score, min_score, n_thresholds)
    fprs = []
    tprs = []
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    auc = abs(_trapz(tprs, fprs))
    return {"fpr": fprs, "tpr": tprs, "auc_roc": min(1.0, auc)}


def pr_curve(y_true: np.ndarray, scores: np.ndarray, n_thresholds: int = 200) -> dict:
    """Compute Precision-Recall curve and AUC-PR."""
    min_score, max_score = np.min(scores), np.max(scores)
    thresholds = np.linspace(min_score, max_score, n_thresholds)
    precisions = []
    recalls = []
    positives = np.sum(y_true == 1)

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / positives if positives > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    auc = abs(_trapz(precisions, recalls))
    return {"precision": precisions, "recall": recalls, "auc_pr": min(1.0, auc)}


def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_percentile: float = 95.0,
    include_curves: bool = False,
) -> dict:
    """Compute all evaluation metrics and optionally full curve data."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    threshold = float(np.percentile(scores, threshold_percentile))
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    roc = roc_curve(y_true, scores)
    pr = pr_curve(y_true, scores)

    accuracy = (cm["tp"] + cm["tn"]) / max(1, cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])

    results = {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "auc_roc": round(roc["auc_roc"], 4),
        "auc_pr": round(pr["auc_pr"], 4),
        "threshold": round(threshold, 6),
        "confusion_matrix": cm,
        "predictions_positive": int(np.sum(y_pred)),
        "actual_positive": int(np.sum(y_true)),
    }

    if include_curves:
        results["roc_curve"] = {"fpr": roc["fpr"].tolist(), "tpr": roc["tpr"].tolist()}
        results["pr_curve"] = {"precision": pr["precision"].tolist(), "recall": pr["recall"].tolist()}

    return results


def cross_validate(
    run_fn,
    flux_values: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Run k-fold cross-validation for an anomaly detection method.

    Args:
        run_fn: Callable(train_flux, test_flux) -> anomaly_scores for test set
        flux_values: Full flux array
        labels: Ground-truth binary labels
        n_folds: Number of CV folds
        seed: Random seed for reproducibility

    Returns:
        Dict with mean ± std for all metrics across folds
    """
    rng = np.random.RandomState(seed)
    n = len(flux_values)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    fold_metrics = []
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_flux = flux_values[train_idx]
        test_flux = flux_values[test_idx]
        test_labels = labels[test_idx]

        try:
            test_scores = run_fn(train_flux, test_flux)
            metrics = compute_all_metrics(test_labels, test_scores)
            fold_metrics.append(metrics)
        except Exception as exc:
            fold_metrics.append({
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "accuracy": 0.0, "auc_roc": 0.5, "auc_pr": 0.0,
                "error": str(exc),
            })

    summary = {}
    for key in ["precision", "recall", "f1_score", "accuracy", "auc_roc", "auc_pr"]:
        values = [m.get(key, 0.0) for m in fold_metrics]
        summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{key}_std"] = round(float(np.std(values)), 4)
    summary["n_folds"] = n_folds
    summary["fold_details"] = fold_metrics
    return summary


def paired_t_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired t-test to compare two methods across folds.

    Returns t-statistic and p-value.
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diff = a - b
    n = len(diff)
    if n < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-12:
        return {"t_statistic": float("inf") if mean_diff != 0 else 0.0, "p_value": 0.0, "significant": mean_diff != 0}

    t_stat = mean_diff / (std_diff / np.sqrt(n))
    # Approximate two-tailed p-value using normal distribution for simplicity
    z = abs(t_stat)
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + _erf(z / np.sqrt(2.0))))
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
    }


def _erf(x: float) -> float:
    """Approximate error function for p-value computation."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
