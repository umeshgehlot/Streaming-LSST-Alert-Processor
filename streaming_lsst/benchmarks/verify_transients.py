
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst.processor import StreamingLSSTProcessor

def create_historical_transients():
    """Create a few alerts mimicking known historical transients."""
    alerts = []
    
    # 1. Known Supernova (SN-like)
    # Bright (14.0), new source (ndethist=1), positive difference
    alerts.append({
        "alertId": "SN_2024_historical",
        "publisher": "ZTF",
        "candidate": {
            "ra": 150.0, "dec": 25.0,
            "magpsf": 14.0, # Brighter to ensure detection
            "sigmapsf": 0.01,
            "ndethist": 1,
            "ncovhist": 100,
            "isdiffpos": "t",
            "sgscore1": 0.1,
            "objectid": "SN2024abc",
            "label": "Supernova"
        }
    })
    
    # 2. CV Outburst
    alerts.append({
        "alertId": "CV_Outburst_historical",
        "publisher": "ZTF",
        "candidate": {
            "ra": 200.0, "dec": -10.0,
            "magpsf": 12.0,
            "sigmapsf": 0.01,
            "ndethist": 500,
            "ncovhist": 1000,
            "isdiffpos": "t",
            "sgscore1": 0.9,
            "objectid": "CV_J123",
            "label": "CV Outburst"
        }
    })
    
    # 3. Normal Background Star
    alerts.append({
        "alertId": "Normal_Star_1",
        "publisher": "ZTF",
        "candidate": {
            "ra": 10.0, "dec": 10.0,
            "magpsf": 19.5,
            "sigmapsf": 0.05,
            "ndethist": 80,
            "ncovhist": 100,
            "isdiffpos": "f",
            "sgscore1": 0.5,
            "objectid": "Star_456",
            "label": "Normal Star"
        }
    })

    # 4. Spatial Anomaly
    alerts.append({
        "alertId": "Spatial_Transient",
        "publisher": "ZTF",
        "candidate": {
            "ra": 50.0, "dec": 50.0,
            "magpsf": 13.5,
            "sigmapsf": 0.01,
            "ndethist": 1,
            "ncovhist": 10,
            "isdiffpos": "t",
            "sgscore1": 0.2,
            "objectid": "Spatial_Anomaly_1",
            "label": "Spatial Anomaly"
        }
    })
    
    return alerts

def verify_system():
    print("\n" + "="*80)
    print("HISTORICAL TRANSIENT VERIFICATION")
    print("="*80)
    
    processor = StreamingLSSTProcessor(device='cpu', enable_gnn=True)
    
    print("Warming up pipeline with 500 clean background alerts...")
    from streaming_lsst.simulator.generate_ztf_data import generate_ztf_sample
    generate_ztf_sample(n=500, clean=True)
    with open("ztf_sample_data.json", "r") as f:
        warmup_alerts = json.load(f)
        for a in warmup_alerts:
            processor.process_alert(a)
            
    print("Adding spatial neighbors for cluster test...")
    for i in range(10):
        neighbor = {
            "candidate": {
                "ra": 50.0 + np.random.normal(0, 0.05),
                "dec": 50.0 + np.random.normal(0, 0.05),
                "magpsf": 19.8,
                "ndethist": 150,
                "isdiffpos": "f",
                "objectid": f"neighbor_{i}"
            }
        }
        processor.process_alert(neighbor)

    transients = create_historical_transients()
    
    print("\n" + "-"*80)
    print(f"{'Type':20s} | {'ID':25s} | {'Score':8s} | {'Threshold':10s} | {'Detected?'}")
    print("-" * 103)
    
    for t in transients:
        label = t['candidate'].pop('label')
        result = processor.process_alert(t)
        
        flag = "YES" if result['is_anomaly'] else "NO"
        score = result['anomaly_score']
        threshold = result.get('anomaly_details', {}).get('threshold', 0.0)
        
        print(f"{label:20s} | {result['alert_id']:25s} | {score:8.2f} | {threshold:10.2f} | {flag}")
        
        if result['is_anomaly']:
            if result.get('context_anomaly_score', 0) > 4.0:
                print(f"  |-- Reason: High spatial mismatch ({result['context_anomaly_score']:.2f})")
            if result.get('anomaly_details', {}).get('reconstruction_error', 0) > 0.5:
                print(f"  |-- Reason: High reconstruction error ({result['anomaly_details']['reconstruction_error']:.2f})")

    print("="*80 + "\n")

if __name__ == "__main__":
    verify_system()
