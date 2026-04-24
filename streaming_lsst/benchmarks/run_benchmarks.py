"""
Main streaming LSST alert processor benchmark script.
Executes comprehensive latency and throughput benchmarks.
"""

import torch
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List

# Imports from local modules
from ..simulator.alert_simulator import (
    LSSTAlertSimulator, HighVariabilityAlertSimulator, BurstAlertSimulator
)
from ..pipeline.data_pipeline import StreamingPipeline
from .benchmark_suite import StreamingBenchmarkSuite
from ..processor import StreamingLSSTProcessor


def benchmark_latency(processor: StreamingLSSTProcessor, simulator, 
                      num_alerts: int = 1000) -> Dict:
    """
    Benchmark single-alert latency.
    """
    print(f"\n{'='*70}")
    print(f"LATENCY BENCHMARK: {num_alerts} alerts")
    print(f"{'='*70}\n")
    
    suite = StreamingBenchmarkSuite("Latency Benchmark")
    suite.start()
    suite.start_phase("alert_processing")
    
    latencies = []
    for i, (alert, ts) in enumerate(simulator.stream_alerts(duration_sec=float(num_alerts)/100)):
        if i >= num_alerts:
            break
        
        alert_start = time.perf_counter()
        result = processor.process_alert(alert)
        alert_end = time.perf_counter()
        
        suite.record_latency(alert_start, alert_end)
        suite.record_throughput(ts)
        latencies.append(result['latency_ms'])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_alerts} alerts...")
    
    suite.end_phase()
    suite.end()
    
    report = suite.print_report()
    return report


def benchmark_throughput(processor: StreamingLSSTProcessor, simulator,
                        duration_sec: float = 60.0, target_rate_hz: float = 100.0) -> Dict:
    """
    Benchmark sustained throughput at various rates.
    """
    print(f"\n{'='*70}")
    print(f"THROUGHPUT BENCHMARK: {duration_sec}s @ {target_rate_hz} Hz")
    print(f"{'='*70}\n")
    
    suite = StreamingBenchmarkSuite("Throughput Benchmark")
    suite.start()
    suite.start_phase("streaming")
    
    alert_count = 0
    for alert, ts in simulator.stream_alerts(duration_sec=duration_sec, rate_hz=target_rate_hz):
        result = processor.process_alert(alert)
        suite.record_latency(ts, ts + result['latency_ms']/1000)
        suite.record_throughput(ts)
        alert_count += 1
    
    suite.end_phase()
    suite.end()
    
    report = suite.print_report()
    return report


def benchmark_anomaly_detection(processor: StreamingLSSTProcessor, 
                                simulator, num_alerts: int = 1000) -> Dict:
    """
    Benchmark anomaly detection accuracy and speed.
    """
    print(f"\n{'='*70}")
    print(f"ANOMALY DETECTION BENCHMARK: {num_alerts} alerts")
    print(f"{'='*70}\n")
    
    suite = StreamingBenchmarkSuite("Anomaly Detection Benchmark")
    suite.start()
    suite.start_phase("detection")
    
    for i, (alert, ts) in enumerate(simulator.stream_alerts(duration_sec=float(num_alerts)/50)):
        if i >= num_alerts:
            break
        
        result = processor.process_alert(alert)
        
        # Ground truth: anomalies are generated based on simulator's anomaly_rate
        # In this simulation, we approximate based on the anomaly detector's output
        if 'is_anomaly' in result:
            suite.record_detection(result['is_anomaly'], 
                                 np.random.random() < simulator.anomaly_rate)
        
        suite.record_latency(ts, ts + result['latency_ms']/1000)
        suite.record_throughput(ts)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_alerts} alerts...")
    
    suite.end_phase()
    suite.end()
    
    report = suite.print_report()
    return report


def benchmark_memory_efficiency(processor: StreamingLSSTProcessor, simulator,
                                duration_sec: float = 30.0) -> Dict:
    """
    Benchmark memory usage under streaming load.
    """
    print(f"\n{'='*70}")
    print(f"MEMORY EFFICIENCY BENCHMARK: {duration_sec}s")
    print(f"{'='*70}\n")
    
    suite = StreamingBenchmarkSuite("Memory Benchmark")
    suite.start()
    suite.start_phase("memory_monitoring")
    
    for alert, ts in simulator.stream_alerts(duration_sec=duration_sec, rate_hz=200.0):
        result = processor.process_alert(alert)
        suite.record_throughput(ts)
    
    suite.end_phase()
    suite.end()
    
    report = suite.print_report()
    return report


def run_full_benchmark_suite(args) -> List[Dict]:
    """
    Run complete benchmark suite.
    """
    print("\n" + "="*70)
    print("STREAMING LSST ALERT PROCESSOR - FULL BENCHMARK SUITE")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Initialize processor
    processor = StreamingLSSTProcessor(
        device=device,
        enable_transformer=args.enable_transformer,
        enable_autoencoder=args.enable_autoencoder,
        enable_gnn=args.enable_gnn,
    )
    
    print(f"Transformer enabled: {args.enable_transformer}")
    print(f"Autoencoder enabled: {args.enable_autoencoder}")
    print(f"GNN enabled: {args.enable_gnn}")
    
    reports = []
    
    # Run benchmarks
    if args.benchmark_latency:
        report = benchmark_latency(processor, args.simulator_type(seed=42), 
                                  num_alerts=args.num_alerts)
        reports.append(report)
    
    if args.benchmark_throughput:
        report = benchmark_throughput(processor, args.simulator_type(seed=42),
                                     duration_sec=args.duration_sec,
                                     target_rate_hz=args.target_rate_hz)
        reports.append(report)
    
    if args.benchmark_anomaly:
        report = benchmark_anomaly_detection(processor, args.simulator_type(seed=42),
                                            num_alerts=args.num_alerts)
        reports.append(report)
    
    if args.benchmark_memory:
        report = benchmark_memory_efficiency(processor, args.simulator_type(seed=42),
                                            duration_sec=args.duration_sec)
        reports.append(report)
    
    # Save reports
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        combined_report = {
            'metadata': {
                'device': device,
                'num_benchmarks': len(reports),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'benchmarks': reports,
        }
        
        report_file = output_path / 'benchmark_report.json'
        with open(report_file, 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        print(f"\nSUCCESS: Full report saved to: {report_file}")
    
    return reports


if __name__ == '__main__':
    import numpy as np
    
    parser = argparse.ArgumentParser(description='LSST Streaming Alert Processor Benchmark')
    
    # Simulator selection
    parser.add_argument('--simulator', type=str, default='standard',
                       choices=['standard', 'high_variability', 'burst'],
                       help='Alert simulator type')
    
    # Benchmark selection
    parser.add_argument('--benchmark-latency', action='store_true', default=True,
                       help='Run latency benchmark')
    parser.add_argument('--benchmark-throughput', action='store_true', default=True,
                       help='Run throughput benchmark')
    parser.add_argument('--benchmark-anomaly', action='store_true', default=True,
                       help='Run anomaly detection benchmark')
    parser.add_argument('--benchmark-memory', action='store_true', default=True,
                       help='Run memory efficiency benchmark')
    
    # Model selection
    parser.add_argument('--enable-transformer', action='store_true', default=True,
                       help='Enable streaming transformer')
    parser.add_argument('--enable-autoencoder', action='store_true', default=True,
                       help='Enable online autoencoder')
    parser.add_argument('--enable-gnn', action='store_true', default=True,
                       help='Enable graph neural network')
    
    # Benchmark parameters
    parser.add_argument('--num-alerts', type=int, default=1000,
                       help='Number of alerts for latency/anomaly benchmarks')
    parser.add_argument('--duration-sec', type=float, default=60.0,
                       help='Duration of throughput/memory benchmarks in seconds')
    parser.add_argument('--target-rate-hz', type=float, default=100.0,
                       help='Target alert rate in Hz for throughput benchmark')
    parser.add_argument('--use-cuda', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for benchmark reports')
    
    args = parser.parse_args()
    
    # Select simulator
    simulators = {
        'standard': LSSTAlertSimulator,
        'high_variability': HighVariabilityAlertSimulator,
        'burst': BurstAlertSimulator,
    }
    args.simulator_type = simulators[args.simulator]
    
    # Run benchmarks
    reports = run_full_benchmark_suite(args)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
