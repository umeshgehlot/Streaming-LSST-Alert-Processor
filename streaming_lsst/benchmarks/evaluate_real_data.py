"""
Evaluate the Streaming LSST pipeline on real labeled ZTF data.

Usage:
    py benchmarks/evaluate_real_data.py
    py benchmarks/evaluate_real_data.py --max-alerts 5000
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streaming_lsst.processor import StreamingLSSTProcessor


def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def run_pipeline(alerts, device="cpu", enable_gnn=True, seed=42):
    """Run our pipeline on alerts and return predictions + ground truth."""
    np.random.seed(seed)
    processor = StreamingLSSTProcessor(
        device=device,
        enable_transformer=True,
        enable_autoencoder=True,
        enable_gnn=enable_gnn,
    )
    
    y_true, y_pred, y_scores, latencies = [], [], [], []
    indices = np.random.permutation(len(alerts))
    
    for idx in indices:
        alert = alerts[idx]
        gt = alert.get("is_ground_truth_anomaly", False)
        
        start = time.perf_counter()
        result = processor.process_alert(alert)
        lat = (time.perf_counter() - start) * 1000
        
        y_true.append(int(gt))
        y_pred.append(int(result.get("is_anomaly", False)))
        y_scores.append(float(result.get("anomaly_score", 0)))
        latencies.append(lat)
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores), latencies


def compute_metrics(y_true, y_pred, y_scores):
    """Compute precision, recall, F1, AUC-PR."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        precision_recall_curve, auc, accuracy_score,
    )
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        pr_curve, re_curve, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(re_curve, pr_curve)
    except Exception:
        auc_pr = 0.0
    
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "auc_pr": auc_pr}


def run_baselines(alerts, device="cpu"):
    """Run baseline methods on the same data."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Extract features using our pipeline
    processor = StreamingLSSTProcessor(device=device)
    X, y_true = [], []
    for alert in alerts:
        features, _, _ = processor.pipeline.process_alert(alert)
        X.append(features.numpy())
        y_true.append(int(alert.get("is_ground_truth_anomaly", False)))
    
    X = np.array(X)
    y_true = np.array(y_true)
    results = {}
    
    # 1. Isolation Forest
    print("  Running Isolation Forest...")
    t0 = time.perf_counter()
    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    pred = iso.fit_predict(X)
    lat = (time.perf_counter() - t0) * 1000 / len(X)
    y_iso = (pred == -1).astype(int)
    results["Isolation Forest"] = {
        "precision": float(precision_score(y_true, y_iso, zero_division=0)),
        "recall": float(recall_score(y_true, y_iso, zero_division=0)),
        "f1": float(f1_score(y_true, y_iso, zero_division=0)),
        "latency_ms": lat,
    }
    
    # 2. Local Outlier Factor
    print("  Running LOF...")
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    pred = lof.fit_predict(X)
    lat = (time.perf_counter() - t0) * 1000 / len(X)
    y_lof = (pred == -1).astype(int)
    results["LOF"] = {
        "precision": float(precision_score(y_true, y_lof, zero_division=0)),
        "recall": float(recall_score(y_true, y_lof, zero_division=0)),
        "f1": float(f1_score(y_true, y_lof, zero_division=0)),
        "latency_ms": lat,
    }
    
    # 3. One-Class SVM (on subset for speed)
    print("  Running One-Class SVM...")
    try:
        from sklearn.svm import OneClassSVM
        subset = min(3000, len(X))
        idx = np.random.choice(len(X), subset, replace=False)
        t0 = time.perf_counter()
        svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
        pred = svm.fit_predict(X[idx])
        lat = (time.perf_counter() - t0) * 1000 / subset
        y_svm = (pred == -1).astype(int)
        results["One-Class SVM"] = {
            "precision": float(precision_score(y_true[idx], y_svm, zero_division=0)),
            "recall": float(recall_score(y_true[idx], y_svm, zero_division=0)),
            "f1": float(f1_score(y_true[idx], y_svm, zero_division=0)),
            "latency_ms": lat,
        }
    except Exception as e:
        print(f"    SVM failed: {e}")
    
    # 4. Rule-based ZTF filter
    print("  Running Rule-based filter...")
    y_rule = []
    t0 = time.perf_counter()
    for alert in alerts:
        c = alert.get("candidate", {})
        mag = c.get("magpsf", 19.0)
        ndet = c.get("ndethist", 50)
        isdp = c.get("isdiffpos", "f")
        y_rule.append(int(mag < 17.5 and ndet < 20 and isdp == "t"))
    lat = (time.perf_counter() - t0) * 1000 / len(alerts)
    y_rule = np.array(y_rule)
    results["ZTF Rule-based"] = {
        "precision": float(precision_score(y_true, y_rule, zero_division=0)),
        "recall": float(recall_score(y_true, y_rule, zero_division=0)),
        "f1": float(f1_score(y_true, y_rule, zero_division=0)),
        "latency_ms": lat,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on real ZTF data")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max-alerts", type=int, default=5000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    args = parser.parse_args()
    
    # Find data
    data_path = args.data
    if not data_path:
        default = os.path.join(
            str(PROJECT_ROOT), "streaming_lsst", "data", "real_alerts", "unified_real_alerts.json"
        )
        if os.path.exists(default):
            data_path = default
    
    if not data_path or not os.path.exists(data_path):
        print("ERROR: No data file found. Run fetch_real_data.py first.")
        return
    
    # Load
    print(f"Loading data from {data_path}...")
    alerts = load_data(data_path)
    if args.max_alerts and args.max_alerts < len(alerts):
        np.random.seed(42)
        idx = np.random.choice(len(alerts), args.max_alerts, replace=False)
        alerts = [alerts[i] for i in idx]
    
    anomalies = sum(1 for a in alerts if a.get("is_ground_truth_anomaly", False))
    print(f"Dataset: {len(alerts)} alerts, {anomalies} anomalies ({100*anomalies/len(alerts):.1f}%)")
    
    # 1. Full pipeline (multiple seeds for error bars)
    print("\n--- Our Pipeline (Full: Transformer + AE + GNN) ---")
    full_metrics = []
    full_latencies = []
    for seed in range(args.num_seeds):
        print(f"  Seed {seed+1}/{args.num_seeds}...")
        yt, yp, ys, lats = run_pipeline(alerts, enable_gnn=True, seed=seed)
        m = compute_metrics(yt, yp, ys)
        full_metrics.append(m)
        full_latencies.extend(lats)
        print(f"    P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    
    # 2. Ablation (no GNN)
    print("\n--- Ablation (Transformer + AE, no GNN) ---")
    ablation_metrics = []
    ablation_latencies = []
    for seed in range(args.num_seeds):
        print(f"  Seed {seed+1}/{args.num_seeds}...")
        yt, yp, ys, lats = run_pipeline(alerts, enable_gnn=False, seed=seed)
        m = compute_metrics(yt, yp, ys)
        ablation_metrics.append(m)
        ablation_latencies.extend(lats)
        print(f"    P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    
    # 3. Baselines
    print("\n--- Baselines ---")
    baseline_results = run_baselines(alerts)
    
    # Print comparison table
    print("\n" + "=" * 95)
    print("REAL DATA EVALUATION RESULTS (on {} ZTF alerts)".format(len(alerts)))
    print("=" * 95)
    print(f"{'Method':<30s} | {'Precision':>10s} | {'Recall':>10s} | {'F1-Score':>10s} | {'Latency(ms)':>12s}")
    print("-" * 95)
    
    # Full pipeline
    def avg_std(metrics_list, key):
        vals = [m[key] for m in metrics_list]
        return np.mean(vals), np.std(vals)
    
    p_m, p_s = avg_std(full_metrics, "precision")
    r_m, r_s = avg_std(full_metrics, "recall")
    f_m, f_s = avg_std(full_metrics, "f1")
    l_m = np.mean(full_latencies)
    print(f"{'Our Pipeline (Full)':<30s} | {p_m:>7.4f}+/-{p_s:.3f} | {r_m:>7.4f}+/-{r_s:.3f} | {f_m:>7.4f}+/-{f_s:.3f} | {l_m:>9.2f} ms")
    
    p_m, p_s = avg_std(ablation_metrics, "precision")
    r_m, r_s = avg_std(ablation_metrics, "recall")
    f_m, f_s = avg_std(ablation_metrics, "f1")
    l_m = np.mean(ablation_latencies)
    print(f"{'Our Pipeline (No GNN)':<30s} | {p_m:>7.4f}+/-{p_s:.3f} | {r_m:>7.4f}+/-{r_s:.3f} | {f_m:>7.4f}+/-{f_s:.3f} | {l_m:>9.2f} ms")
    
    print("-" * 95)
    for name, b in baseline_results.items():
        print(f"{name:<30s} | {b['precision']:>10.4f} | {b['recall']:>10.4f} | {b['f1']:>10.4f} | {b['latency_ms']:>9.2f} ms")
    print("=" * 95)
    
    # Save results
    out_dir = os.path.join(str(PROJECT_ROOT), "streaming_lsst", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_alerts": len(alerts),
        "num_anomalies": anomalies,
        "pipeline_full": {k: {"mean": float(np.mean([m[k] for m in full_metrics])),
                              "std": float(np.std([m[k] for m in full_metrics]))}
                         for k in full_metrics[0]},
        "pipeline_full_latency_ms": {"mean": float(np.mean(full_latencies)),
                                     "p95": float(np.percentile(full_latencies, 95))},
        "pipeline_no_gnn": {k: {"mean": float(np.mean([m[k] for m in ablation_metrics])),
                                "std": float(np.std([m[k] for m in ablation_metrics]))}
                           for k in ablation_metrics[0]},
        "baselines": baseline_results,
    }
    
    out_file = os.path.join(out_dir, "real_data_evaluation.json")
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {out_file}")
    print("[DONE] Real-data evaluation complete!")


if __name__ == "__main__":
    main()
