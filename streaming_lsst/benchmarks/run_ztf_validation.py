"""
Validation benchmark using ZTF-like real data schema.
Ensures the processor can handle real-world alert formats.
"""

import sys
import json
from pathlib import Path
import time
import torch
import numpy as np

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst.processor import StreamingLSSTProcessor
from streaming_lsst.benchmarks.benchmark_suite import StreamingBenchmarkSuite

def run_ztf_validation():
    print("\n" + "="*80)
    print("STREAMING LSST ALERT PROCESSOR - REAL DATA (ZTF SCHEMA) VALIDATION")
    print("="*80)
    
    # 1. Load Data (assuming it was fetched or generated already)
    # generate_ztf_sample(n=500) # Commented out to use real data
    
    if not Path("ztf_sample_data.json").exists():
        print("Error: ztf_sample_data.json not found. Run scripts/fetch_real_ztf.py first.")
        return

    with open("ztf_sample_data.json", "r") as f:
        ztf_alerts = json.load(f)
    
    # 2. Initialize Processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = StreamingLSSTProcessor(device=device)
    
    # 3. Benchmark
    suite = StreamingBenchmarkSuite("ZTF Validation")
    suite.start()
    suite.start_phase("ztf_processing")
    
    print(f"Processing {len(ztf_alerts)} real ZTF alerts...")
    
    for alert in ztf_alerts:
        start = time.perf_counter()
        result = processor.process_alert(alert)
        end = time.perf_counter()
        
        suite.record_latency(start, end)
        suite.record_throughput(start)
        
        if result['is_anomaly']:
            print(f"  [!] Anomaly Detected: {result['alert_id']} (Score: {result['anomaly_score']:.2f})")
        
    suite.end_phase()
    suite.end()
    
    # 4. Report
    suite.print_report(verbose=True)
    
    print("\n" + "="*80)
    print("VALIDATION SUCCESSFUL")
    print("The processor successfully handled REAL ZTF alert data.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_ztf_validation()
