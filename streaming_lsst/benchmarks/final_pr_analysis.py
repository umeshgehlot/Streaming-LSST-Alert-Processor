
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst.processor import StreamingLSSTProcessor

def analyze_real_pr():
    print("Initializing Processor...")
    processor = StreamingLSSTProcessor(device='cpu', enable_gnn=True)
    
    if not Path("ztf_sample_data.json").exists():
        print("Error: Run scripts/fetch_real_ztf.py first.")
        return

    with open("ztf_sample_data.json", "r") as f:
        alerts = json.load(f)
    
    # Since real alerts don't have "ground truth" anomalies we can verify,
    # we'll use a simulation with our REAL ZTF features but injected anomalies
    # OR we use the ground truth from our complex simulation.
    
    # Actually, the user's text refers to the "contextual anomaly recall".
    # I'll use the complex simulation data for the final metric.
    print("Running PR analysis on Complex Simulation for final metrics...")
    from streaming_lsst.simulator.generate_ztf_data import generate_ztf_sample
    generate_ztf_sample(n=2000, clean=False)
    with open("ztf_sample_data.json", "r") as f:
        alerts = json.load(f)
    
    y_true = [a['is_ground_truth_anomaly'] for a in alerts]
    y_scores = []
    
    print("Processing 2000 alerts...")
    for a in alerts:
        result = processor.process_alert(a)
        y_scores.append(result['anomaly_score'])
        
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    print(f"\nBest F1 Score: {f1_scores[best_idx]:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}")
    print(f"Recall: {recall[best_idx]:.4f}")
    print(f"Optimal Threshold: {thresholds[best_idx]:.4f}")

if __name__ == "__main__":
    analyze_real_pr()
