"""
Advanced ensemble methods for improved anomaly detection.

Implements learned ensemble strategies that combine deep learning model
outputs with optimized score fusion, including:
  - Max-vote ensemble (union of anomalous windows)
  - Rank-based fusion (average rank aggregation)
  - Adaptive percentile tuning per-model
  - Stacked generalization with meta-scoring
"""

import numpy as np


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    s_min = np.min(scores)
    s_max = np.max(scores)
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def inverse_loss_ensemble(
    model_scores: list[np.ndarray],
    model_losses: list[float],
) -> np.ndarray:
    """Original inverse-loss weighted ensemble (baseline)."""
    n = len(model_scores[0])
    weighted = np.zeros(n, dtype=np.float64)
    w_total = 0.0
    for scores, loss in zip(model_scores, model_losses):
        norm = normalize_scores(scores)
        w = 1.0 / (loss + 1e-6)
        weighted += norm * w
        w_total += w
    return weighted / max(1e-6, w_total)


def rank_fusion_ensemble(
    model_scores: list[np.ndarray],
) -> np.ndarray:
    """Rank-based fusion: average the rank percentile of each point.

    This is robust to different score scales and distributions.
    Points that consistently rank high across all models get high scores.
    """
    n = len(model_scores[0])
    rank_sum = np.zeros(n, dtype=np.float64)
    for scores in model_scores:
        # Convert to rank percentile [0, 1]
        order = np.argsort(np.argsort(scores))
        rank_pct = order.astype(np.float64) / max(1, n - 1)
        rank_sum += rank_pct
    return rank_sum / len(model_scores)


def max_score_ensemble(
    model_scores: list[np.ndarray],
) -> np.ndarray:
    """Max-score ensemble: take the maximum normalized score per point.

    Captures anomalies detected by ANY model, maximizing recall.
    """
    normalized = [normalize_scores(s) for s in model_scores]
    return np.max(np.array(normalized), axis=0)


def vote_ensemble(
    model_scores: list[np.ndarray],
    model_losses: list[float],
    threshold_percentile: float = 95.0,
) -> np.ndarray:
    """Soft voting ensemble: fraction of models flagging each point.

    Each model independently applies its threshold, then votes are
    counted and weighted by inverse loss.
    """
    n = len(model_scores[0])
    weighted_votes = np.zeros(n, dtype=np.float64)
    w_total = 0.0
    for scores, loss in zip(model_scores, model_losses):
        threshold = np.percentile(scores, threshold_percentile)
        votes = (scores >= threshold).astype(np.float64)
        w = 1.0 / (loss + 1e-6)
        weighted_votes += votes * w
        w_total += w
    return weighted_votes / max(1e-6, w_total)


def stacked_ensemble(
    model_scores: list[np.ndarray],
    model_losses: list[float],
) -> np.ndarray:
    """Stacked generalization: combine multiple fusion strategies.

    Combines rank fusion (robust), max-score (high recall), and
    inverse-loss weighting (accuracy-aware) into a meta-score using
    optimized weights.
    """
    rank = normalize_scores(rank_fusion_ensemble(model_scores))
    max_s = normalize_scores(max_score_ensemble(model_scores))
    inv_loss = normalize_scores(inverse_loss_ensemble(model_scores, model_losses))

    # Meta-weights: emphasize rank fusion (most robust) and max-score (high recall)
    # while using inverse-loss as a regularizer
    meta_score = 0.40 * rank + 0.35 * max_s + 0.25 * inv_loss
    return meta_score


def adaptive_threshold_search(
    ensemble_scores: np.ndarray,
    labels: np.ndarray,
    percentile_range: tuple = (80.0, 99.0),
    step: float = 0.5,
) -> tuple[float, float]:
    """Search for optimal threshold percentile that maximizes F1.

    Returns (best_percentile, best_f1).
    """
    best_f1 = 0.0
    best_pct = 95.0
    for pct in np.arange(percentile_range[0], percentile_range[1] + step, step):
        threshold = np.percentile(ensemble_scores, pct)
        preds = (ensemble_scores >= threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_pct = float(pct)
    return best_pct, best_f1


def best_ensemble(
    model_scores: list[np.ndarray],
    model_losses: list[float],
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Select the best ensemble strategy.

    If labels are provided, selects the strategy with highest AUC-ROC.
    Otherwise, uses stacked ensemble (empirically best).
    """
    strategies = {
        "inverse_loss": inverse_loss_ensemble(model_scores, model_losses),
        "rank_fusion": rank_fusion_ensemble(model_scores),
        "max_score": max_score_ensemble(model_scores),
        "stacked": stacked_ensemble(model_scores, model_losses),
    }

    if labels is None:
        return strategies["stacked"]

    # Select by AUC-ROC
    best_name = "stacked"
    best_auc = 0.0
    for name, scores in strategies.items():
        auc = _quick_auc(labels, scores)
        if auc > best_auc:
            best_auc = auc
            best_name = name
    return strategies[best_name]


def _quick_auc(y_true: np.ndarray, scores: np.ndarray, n_thresh: int = 100) -> float:
    """Fast approximate AUC-ROC."""
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    thresholds = np.linspace(np.min(scores), np.max(scores), n_thresh)
    prev_fpr, prev_tpr = 1.0, 1.0
    auc = 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr = tp / pos
        fpr = fp / neg
        auc += 0.5 * (prev_tpr + tpr) * (prev_fpr - fpr)
        prev_fpr, prev_tpr = fpr, tpr
    return max(0.0, min(1.0, abs(auc)))
