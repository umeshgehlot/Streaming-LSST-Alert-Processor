"""
Ablation Study for Streaming LSST Alert Processor.
Compares:
1. Transformer + Autoencoder (No GNN)
2. Transformer + Autoencoder + GNN (Full context)
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
from typing import Dict, List

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst.processor import StreamingLSSTProcessor
from streaming_lsst.simulator.alert_simulator import LSSTAlertSimulator
from streaming_lsst.benchmarks.benchmark_suite import StreamingBenchmarkSuite

def run_ablation(num_alerts=2000):
    print("\n" + "="*80)
    print("STREAMING LSST ALERT PROCESSOR - GNN ABLATION STUDY")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Models
    proc_no_gnn = StreamingLSSTProcessor(device=device, enable_gnn=False)
    proc_with_gnn = StreamingLSSTProcessor(device=device, enable_gnn=True)
    
    configs = {
        "Transformer + AE (No GNN)": proc_no_gnn,
        "Full Pipeline (With GNN)": proc_with_gnn
    }
    
    # 2. Simulator
    simulator = LSSTAlertSimulator(seed=42, anomaly_rate=0.1)
    
    # 3. Benchmark
    results = {}
    
    for name, processor in configs.items():
        print(f"\nEvaluating {name}...")
        suite = StreamingBenchmarkSuite(name)
        suite.start()
        suite.start_phase("evaluation")
        
        # Reset simulator
        simulator.alert_counter = 0
        np.random.seed(42)
        
        # Pre-generate alerts to ensure identical input
        alerts = [simulator.generate_alert() for _ in range(num_alerts)]
        
        correct_detections = 0
        spatial_anomalies_found = 0
        total_spatial_anomalies = sum(1 for a in alerts if a['alert'].get('is_spatial_anomaly'))
        
        for i, alert in enumerate(alerts):
            # For GNN, we simulate spatial relations (alerts in same cluster are connected)
            if "With GNN" in name:
                # Find other alerts in the same cluster that were processed recently
                # (Simplified simulation: connect to previous 2 alerts if same cluster)
                # In real code, we'd use RA/Dec cross-match
                pass # processor.graph_processor.add_relation(...) is called internally in real use
            
            start = time.perf_counter()
            result = processor.process_alert(alert)
            end = time.perf_counter()
            
            suite.record_latency(start, end)
            suite.record_throughput(start)
            
            # Check accuracy against spatial anomalies
            is_spatial = alert['alert'].get('is_spatial_anomaly', False)
            if is_spatial and result.get('is_anomaly'):
                spatial_anomalies_found += 1
            
            # Generic anomaly tracking
            is_any_anomaly = is_spatial or (i % 10 == 0) # Mock ground truth
            suite.record_detection(result['is_anomaly'], is_any_anomaly)
            
        suite.end_phase()
        suite.end()
        
        report = suite.get_comprehensive_report()
        report['spatial_recall'] = spatial_anomalies_found / (total_spatial_anomalies + 1e-8)
        results[name] = report
        
        print(f"  F1-Score: {report['f1_score']:.4f}")
        print(f"  Spatial Anomaly Recall: {report['spatial_recall']:.2%}")
        print(f"  Mean Latency: {report['latency_mean_ms']:.3f} ms")

    # 4. Summary Table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Configuration':30s} | {'F1-Score':10s} | {'Spatial Recall':15s} | {'Latency (ms)':12s}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:30s} | {r['f1_score']:10.4f} | {r['spatial_recall']:15.2%} | {r['latency_mean_ms']:12.3f}")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_ablation()
