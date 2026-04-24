"""
Quick start example script for Streaming LSST Alert Processor.
Demonstrates basic usage and integration of all components.
"""

import sys
from pathlib import Path
import time
import torch

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst import (
    StreamingLSSTProcessor,
    LSSTAlertSimulator,
    HighVariabilityAlertSimulator,
    StreamingBenchmarkSuite,
)


def example_1_basic_processing():
    """Example 1: Basic alert processing."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Alert Processing")
    print("="*70 + "\n")
    
    # Initialize processor
    processor = StreamingLSSTProcessor(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_transformer=True,
        enable_autoencoder=True,
        enable_gnn=True,
    )
    
    # Create alert simulator
    simulator = LSSTAlertSimulator(seed=42, anomaly_rate=0.1)
    
    # Process 10 alerts
    print("Processing 10 alerts...\n")
    for i, (alert, _) in enumerate(simulator.stream_alerts(duration_sec=5.0, rate_hz=50)):
        if i >= 10:
            break
        
        result = processor.process_alert(alert)
        
        print(f"Alert {i+1}: ID={result['alert_id']}")
        print(f"  |-- Latency: {result['latency_ms']:.2f} ms")
        print(f"  |-- Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  |-- Is Anomaly: {result['is_anomaly']}")
        print(f"  +-- Transformer Embedding Shape: {result['transformer_embedding'].shape}")
    
    # Get pipeline statistics
    stats = processor.get_pipeline_stats()
    print(f"\nPipeline Statistics:")
    print(f"  Average Latency: {stats.get('avg_latency_ms', 0):.2f} ms")
    print(f"  Throughput: {stats.get('throughput_alerts_per_sec', 0):.1f} alerts/sec")
    print(f"  Buffer Occupancy: {stats.get('buffer_occupancy', 0):.1%}")


def example_2_anomaly_detection():
    """Example 2: Focused on anomaly detection."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Anomaly Detection on High-Variability Sources")
    print("="*70 + "\n")
    
    processor = StreamingLSSTProcessor(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_autoencoder=True,
    )
    
    # Use high-variability simulator (more anomalies)
    simulator = HighVariabilityAlertSimulator(seed=42, anomaly_rate=0.2)
    
    anomalies_detected = 0
    normal_alerts = 0
    
    print("Processing 100 alerts and counting anomalies...\n")
    for i, (alert, _) in enumerate(simulator.stream_alerts(duration_sec=10.0)):
        if i >= 100:
            break
        
        result = processor.process_alert(alert)
        
        if result['is_anomaly']:
            anomalies_detected += 1
            print(f"WARNING: ANOMALY DETECTED: Alert {result['alert_id']} "
                  f"(score: {result['anomaly_score']:.4f})")
        else:
            normal_alerts += 1
    
    print(f"\nResults:")
    print(f"  Normal Alerts: {normal_alerts}")
    print(f"  Anomalies Detected: {anomalies_detected}")
    print(f"  Anomaly Rate: {100*anomalies_detected/(normal_alerts+anomalies_detected):.1f}%")


def example_3_latency_measurement():
    """Example 3: Detailed latency measurement."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Latency Profiling")
    print("="*70 + "\n")
    
    processor = StreamingLSSTProcessor(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_transformer=True,
        enable_autoencoder=True,
    )
    
    simulator = LSSTAlertSimulator(seed=42)
    
    latencies = []
    
    print("Processing 100 alerts for latency analysis...\n")
    for i, (alert, _) in enumerate(simulator.stream_alerts(duration_sec=10.0)):
        if i >= 100:
            break
        
        result = processor.process_alert(alert)
        latencies.append(result['latency_ms'])
    
    # Compute statistics
    import numpy as np
    latencies = np.array(latencies)
    
    print(f"Latency Statistics (ms):")
    print(f"  Min:    {latencies.min():.3f}")
    print(f"  P50:    {np.percentile(latencies, 50):.3f}")
    print(f"  P95:    {np.percentile(latencies, 95):.3f}")
    print(f"  P99:    {np.percentile(latencies, 99):.3f}")
    print(f"  Max:    {latencies.max():.3f}")
    print(f"  Mean:   {latencies.mean():.3f}")
    print(f"  Std:    {latencies.std():.3f}")


def example_4_custom_benchmark():
    """Example 4: Custom benchmark with detailed profiling."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Benchmarking")
    print("="*70 + "\n")
    
    processor = StreamingLSSTProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
    simulator = LSSTAlertSimulator(seed=42, anomaly_rate=0.05)
    
    # Create benchmark suite
    suite = StreamingBenchmarkSuite("Custom Benchmark")
    suite.start()
    suite.start_phase("stream_processing")
    
    print("Running custom benchmark for 30 seconds...\n")
    
    start = time.perf_counter()
    for alert, ts in simulator.stream_alerts(duration_sec=30, rate_hz=100):
        result = processor.process_alert(alert)
        
        suite.record_latency(ts, ts + result['latency_ms']/1000)
        suite.record_throughput(ts)
        
        if result['is_anomaly']:
            suite.record_detection(True, True)
        else:
            suite.record_detection(False, False)
    
    suite.end_phase()
    suite.end()
    
    # Print report
    suite.print_report(verbose=True)


def example_5_gnn_graph_processing():
    """Example 5: GNN-based graph processing."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Graph Neural Network Processing")
    print("="*70 + "\n")
    
    processor = StreamingLSSTProcessor(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_gnn=True,
    )
    
    simulator = LSSTAlertSimulator(seed=42)
    
    print("Processing alerts and building dynamic alert graph...\n")
    
    for i, (alert, _) in enumerate(simulator.stream_alerts(duration_sec=5)):
        if i >= 20:
            break
        
        result = processor.process_alert(alert)
        
        # Create pseudo-relationships between alerts
        if i > 0 and i % 3 == 0:
            # Add edge between current and previous alert
            prev_id = str(i - 1)
            curr_id = result['alert_id']
            processor.graph_processor.add_relation(prev_id, curr_id, weight=0.5)
        
        if i > 0 and i % 5 == 0:
            embeddings = processor.graph_processor.get_embeddings()
            print(f"Step {i}: Graph has {len(processor.graph_processor.node_features)} nodes, "
                  f"embeddings shape: {embeddings.shape}")


def main():
    """Run all examples."""
    print("\n" + "== "*20)
    print("STREAMING LSST ALERT PROCESSOR - QUICK START EXAMPLES")
    print("== "*20)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device.upper()}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        example_1_basic_processing()
        example_2_anomaly_detection()
        example_3_latency_measurement()
        example_4_custom_benchmark()
        example_5_gnn_graph_processing()
        
        print("\n" + "="*70)
        print("SUCCESS: ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
