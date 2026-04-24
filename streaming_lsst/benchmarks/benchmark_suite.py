"""
Comprehensive benchmarking suite for streaming LSST alert processing.
Measures latency, throughput, memory usage, and model performance.
"""

import torch
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import json
from datetime import datetime


class LatencyBenchmark:
    """Measure end-to-end latency for alert processing."""
    
    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)
        self.timestamp_pairs = []
    
    def record(self, start_time: float, end_time: float):
        """Record latency in milliseconds."""
        latency_ms = (end_time - start_time) * 1000
        self.latencies.append(latency_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        lats = list(self.latencies)
        return {
            'latency_min_ms': min(lats),
            'latency_p50_ms': np.percentile(lats, 50),
            'latency_p95_ms': np.percentile(lats, 95),
            'latency_p99_ms': np.percentile(lats, 99),
            'latency_max_ms': max(lats),
            'latency_mean_ms': np.mean(lats),
            'latency_std_ms': np.std(lats),
        }


class ThroughputBenchmark:
    """Measure alert processing throughput."""
    
    def __init__(self, window_size: int = 100):
        self.timestamps = deque(maxlen=window_size)
        self.alert_count = 0
    
    def record_alert(self, timestamp: float):
        """Record alert processing."""
        self.timestamps.append(timestamp)
        self.alert_count += 1
    
    def get_throughput_hz(self) -> float:
        """Get current throughput in alerts/second."""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span <= 0:
            return 0.0
        
        return len(self.timestamps) / time_span
    
    def get_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        return {
            'throughput_hz': self.get_throughput_hz(),
            'total_alerts': self.alert_count,
        }


class MemoryBenchmark:
    """Monitor memory usage during streaming."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = deque(maxlen=1000)
        self.peak_memory_mb = 0
    
    def sample(self):
        """Sample current memory usage."""
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        self.memory_samples.append(mem_mb)
        self.peak_memory_mb = max(self.peak_memory_mb, mem_mb)
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.memory_samples:
            return {}
        
        mems = list(self.memory_samples)
        return {
            'memory_current_mb': mems[-1],
            'memory_mean_mb': np.mean(mems),
            'memory_peak_mb': self.peak_memory_mb,
            'memory_std_mb': np.std(mems),
        }


class AnomalyDetectionBenchmark:
    """Measure anomaly detection performance (precision, recall, F1)."""
    
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        self.predictions = []
        self.ground_truth = []
    
    def record(self, prediction: bool, ground_truth: bool):
        """Record prediction vs ground truth."""
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        
        if prediction and ground_truth:
            self.true_positives += 1
        elif prediction and not ground_truth:
            self.false_positives += 1
        elif not prediction and ground_truth:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get detection performance metrics."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        
        if total == 0:
            return {}
        
        accuracy = (self.true_positives + self.true_negatives) / total
        
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
        }


class StreamingBenchmarkSuite:
    """Comprehensive benchmarking suite for streaming systems."""
    
    def __init__(self, name: str = "StreamingLSST"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
        self.latency = LatencyBenchmark()
        self.throughput = ThroughputBenchmark()
        self.memory = MemoryBenchmark()
        self.anomaly = AnomalyDetectionBenchmark()
        
        self.phase_timers = {}  # {phase_name: [start_time, end_time]}
        self.current_phase = None
    
    def start(self):
        """Start benchmark."""
        self.start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    def end(self):
        """End benchmark."""
        self.end_time = time.perf_counter()
    
    def start_phase(self, phase_name: str):
        """Start timing a processing phase."""
        self.current_phase = phase_name
        self.phase_timers[phase_name] = [time.perf_counter(), None]
    
    def end_phase(self):
        """End timing current phase."""
        if self.current_phase:
            self.phase_timers[self.current_phase][1] = time.perf_counter()
    
    def record_latency(self, start_time: float, end_time: float):
        """Record alert processing latency."""
        self.latency.record(start_time, end_time)
    
    def record_throughput(self, timestamp: float):
        """Record alert processing for throughput."""
        self.throughput.record_alert(timestamp)
        self.memory.sample()
    
    def record_detection(self, prediction: bool, ground_truth: bool):
        """Record anomaly detection result."""
        self.anomaly.record(prediction, ground_truth)
    
    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        # Phase breakdown
        phase_breakdown = {}
        for phase, (start, end) in self.phase_timers.items():
            if end:
                phase_breakdown[phase] = {
                    'duration_sec': end - start,
                    'fraction_of_total': (end - start) / (total_time + 1e-8),
                }
        
        report = {
            'benchmark_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'total_duration_sec': total_time,
            **self.latency.get_stats(),
            **self.throughput.get_stats(),
            **self.memory.get_stats(),
            **self.anomaly.get_stats(),
            'phase_breakdown': phase_breakdown,
        }
        
        return report
    
    def print_report(self, verbose: bool = True):
        """Print formatted benchmark report."""
        report = self.get_comprehensive_report()
        
        print(f"\n{'='*60}")
        print(f"Benchmark Report: {report['benchmark_name']}")
        print(f"{'='*60}\n")
        
        print(f"Total Duration: {report['total_duration_sec']:.2f}s")
        
        print(f"\n--- Latency Metrics (ms) ---")
        for key in ['latency_min_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms', 'latency_max_ms', 'latency_mean_ms']:
            if key in report:
                print(f"{key:25s}: {report[key]:10.3f}")
        
        print(f"\n--- Throughput Metrics ---")
        print(f"{'throughput_hz':25s}: {report.get('throughput_hz', 0):10.2f} alerts/sec")
        print(f"{'total_alerts':25s}: {report.get('total_alerts', 0):10.0f}")
        
        print(f"\n--- Memory Metrics (MB) ---")
        for key in ['memory_current_mb', 'memory_mean_mb', 'memory_peak_mb']:
            if key in report:
                print(f"{key:25s}: {report[key]:10.2f}")
        
        print(f"\n--- Anomaly Detection Metrics ---")
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            if key in report:
                print(f"{key:25s}: {report[key]:10.4f}")
        
        if verbose and report.get('phase_breakdown'):
            print(f"\n--- Phase Breakdown ---")
            for phase, stats in report['phase_breakdown'].items():
                print(f"{phase:25s}: {stats['duration_sec']:8.3f}s ({stats['fraction_of_total']*100:5.1f}%)")
        
        print(f"\n{'='*60}\n")
        
        return report
    
    def save_report(self, filepath: str):
        """Save report to JSON file."""
        report = self.get_comprehensive_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {filepath}")
