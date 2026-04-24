
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst.processor import StreamingLSSTProcessor
from streaming_lsst.simulator.alert_simulator import LSSTAlertSimulator

def analyze_pr(num_alerts=2000):
    print("Initializing Processor and Simulator...")
    processor = StreamingLSSTProcessor(device='cpu', enable_gnn=True)
    simulator = LSSTAlertSimulator(seed=42, anomaly_rate=0.2)
    
    # Pre-generate alerts and ground truth
    print(f"Generating {num_alerts} alerts...")
    alerts = [simulator.generate_alert() for _ in range(num_alerts)]
    
    # Ground truth: is_spatial_anomaly or generic anomaly
    y_true = []
    y_scores = []
    
    print("Processing alerts...")
    for i, alert_data in enumerate(alerts):
        # Determine ground truth
        is_spatial = alert_data['alert'].get('is_spatial_anomaly', False)
        is_anomaly = is_spatial or (i % 5 == 0) # More frequent anomalies for better stats
        y_true.append(is_anomaly)
        
        result = processor.process_alert(alert_data)
        y_scores.append(result['anomaly_score'])
        
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]
    
    print(f"\nBest F1 Score: {best_f1:.4f} at threshold: {best_threshold:.4f}")
    print(f"Precision at best F1: {precision[best_idx]:.4f}")
    print(f"Recall at best F1: {recall[best_idx]:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'GNN-AE (F1={best_f1:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # F1 and Precision/Recall vs Threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, precision[:-1], 'r--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g--', label='Recall')
    plt.plot(thresholds, f1_scores[:-1], 'b-', label='F1 Score')
    plt.axvline(x=best_threshold, color='k', linestyle=':', label=f'Best Threshold ({best_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Anomaly Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/pr_curve_analysis.png')
    print("\nPR Curve saved to benchmark_results/pr_curve_analysis.png")
    
    return best_f1

if __name__ == '__main__':
    analyze_pr()
