"""
Comparison benchmark for Streaming LSST Alert Processor.
Compares Streaming Transformer vs. Isolation Forest vs. Simple RNN.
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
from streaming_lsst.simulator.alert_simulator import LSSTAlertSimulator, HighVariabilityAlertSimulator
from streaming_lsst.models.baselines import IsolationForestBaseline, RNNBaseline
from streaming_lsst.benchmarks.benchmark_suite import StreamingBenchmarkSuite

def run_comparison(num_alerts=2000):
    print("\n" + "="*80)
    print("STREAMING LSST ALERT PROCESSOR - BASELINE COMPARISON BENCHMARK")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Initialize Models
    transformer_proc = StreamingLSSTProcessor(device=device, enable_transformer=True, enable_gnn=False)
    if_baseline = IsolationForestBaseline(n_estimators=100)
    rnn_baseline = RNNBaseline(input_dim=16, hidden_dim=64).to(device)
    
    models = {
        "Streaming Transformer": transformer_proc,
        "Isolation Forest (Baseline)": if_baseline,
        "Simple RNN (Baseline)": rnn_baseline
    }
    
    # 2. Simulator
    simulator = HighVariabilityAlertSimulator(seed=42, anomaly_rate=0.1)
    
    # 3. Benchmark Results
    results = {}
    
    for name, model in models.items():
        print(f"\nBenchmarking {name}...")
        suite = StreamingBenchmarkSuite(name)
        suite.start()
        suite.start_phase("processing")
        
        # Reset simulator for fair comparison
        simulator.alert_counter = 0
        np.random.seed(42)
        
        alert_stream = list(simulator.stream_alerts(duration_sec=float(num_alerts)/50))[:num_alerts]
        
        for alert, ts in alert_stream:
            # We need to simulate the ground truth for F1-score
            # In the simulator, anomalies are generated stochastically.
            # For simplicity, we assume the simulator's internal state is stable.
            
            # Extract features for raw models
            if name == "Streaming Transformer":
                result = model.process_alert(alert)
                # Accuracy is handled internally if we pass it to suite
            else:
                # Basic models take raw features
                features, _, _ = transformer_proc.pipeline.process_alert(alert)
                features = features.to(device)
                result = model.process_alert(features)
            
            suite.record_latency(ts, ts + result['latency_ms']/1000)
            suite.record_throughput(ts)
            
            # Simulated ground truth (approximate)
            # In a real setup, we'd have labels. Here we use a deterministic seed
            # to ensure the same alerts are "anomalous" across runs.
            # (In reality, we should have tagged the alerts during generation)
            
        suite.end_phase()
        suite.end()
        results[name] = suite.get_comprehensive_report()
        print(f"  Mean Latency: {results[name]['latency_mean_ms']:.3f} ms")
        print(f"  Throughput: {results[name]['throughput_hz']:.1f} alerts/sec")

    # 4. Generate Report
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':30s} | {'Latency (ms)':12s} | {'Throughput (Hz)':15s} | {'Peak Mem (MB)':12s}")
    print("-" * 80)
    for name in results:
        r = results[name]
        print(f"{name:30s} | {r['latency_mean_ms']:12.3f} | {r['throughput_hz']:15.1f} | {r['memory_peak_mb']:12.1f}")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_comparison()
