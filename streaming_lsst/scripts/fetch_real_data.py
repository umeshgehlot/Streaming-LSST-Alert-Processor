"""
Real Astronomical Data Downloader for Streaming LSST Alert Processor.

Downloads labeled ZTF alert data from Fink Broker API.
Fink provides real ZTF alerts with all 16 features needed by our pipeline,
plus classification labels for ground truth.

Usage:
    py scripts/fetch_real_data.py
    py scripts/fetch_real_data.py --num-per-class 5000
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FINK_API = "https://api.ztf.fink-portal.org/api/v1"

# Classes that represent ANOMALIES (transients we want to detect)
ANOMALY_CLASSES = [
    "SN candidate",
    "Early SN Ia candidate",
    "(TNS) SN Ia",
    "(TNS) SN II",
    "(TNS) SN",
    "(TNS) SN I",
    "(TNS) SLSN-I",
    "(TNS) SLSN-II",
    "(TNS) AGN",
    "(TNS) CV",
    "(TNS) Nova",
    "Kilonova candidate",
    "Ambiguous",
]

# Classes that represent NORMAL objects (non-anomalous)
NORMAL_CLASSES = [
    "Solar System candidate",
    "Solar System MPC",
    "(SIMBAD) Star",
    "(SIMBAD) EB*",
    "(SIMBAD) RRLyr",
    "(SIMBAD) Cepheid",
    "(SIMBAD) LPV*",
    "(SIMBAD) QSO",
    "(SIMBAD) Galaxy",
    "(SIMBAD) AGN",
]

# The 16 columns our pipeline needs (Fink uses "i:" prefix)
NEEDED_COLUMNS = [
    "i:objectId", "i:ra", "i:dec",
    "i:magpsf", "i:sigmapsf",
    "i:ndethist", "i:sgscore1", "i:distpsnr1",
    "i:diffmaglim", "i:isdiffpos",
    "i:jd", "i:fid",
]


def get_available_classes():
    """Fetch all available Fink classification classes."""
    r = requests.get(f"{FINK_API}/classes", timeout=30)
    r.raise_for_status()
    data = r.json()
    all_classes = []
    for category, class_list in data.items():
        if isinstance(class_list, list):
            all_classes.extend(class_list)
    print(f"  Fink has {len(all_classes)} classes across {len(data)} categories")
    return all_classes


def fetch_class(class_name, n=1000):
    """Fetch latest alerts for one Fink class."""
    payload = {
        "class": class_name,
        "n": str(min(n, 10000)),
        "output-format": "json",
    }
    try:
        r = requests.post(f"{FINK_API}/latests", json=payload, timeout=120)
        r.raise_for_status()
        if len(r.content) < 10:
            return pd.DataFrame()
        return pd.read_json(BytesIO(r.content))
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


def fink_to_alert(row, fink_class, is_anomaly):
    """Convert a Fink row to our pipeline's alert format."""
    try:
        oid = str(row.get("i:objectId", row.get("objectId", "unk")))
        return {
            "alertId": f"fink_{oid}",
            "source": "fink",
            "fink_class": fink_class,
            "is_ground_truth_anomaly": is_anomaly,
            "anomaly_type": fink_class,
            "candidate": {
                "ra": float(row.get("i:ra", row.get("ra", 0))),
                "dec": float(row.get("i:dec", row.get("dec", 0))),
                "magpsf": float(row.get("i:magpsf", row.get("magpsf", 19.0))),
                "sigmapsf": float(row.get("i:sigmapsf", row.get("sigmapsf", 0.05))),
                "ndethist": int(row.get("i:ndethist", row.get("ndethist", 1))),
                "ncovhist": int(row.get("i:ncovhist", 10)),
                "isdiffpos": str(row.get("i:isdiffpos", "t")),
                "sgscore1": float(row.get("i:sgscore1", row.get("sgscore1", 0.5))),
                "distpsnr1": float(row.get("i:distpsnr1", row.get("distpsnr1", 1.0))),
                "diffmaglim": float(row.get("i:diffmaglim", row.get("diffmaglim", 20.5))),
                "objectid": oid,
                "flux": 10 ** (-float(row.get("i:magpsf", 19.0)) / 2.5),
                "fluxerr": 0.1 * 10 ** (-float(row.get("i:magpsf", 19.0)) / 2.5),
                "ranr": float(row.get("i:ra", 0)),
                "decnr": float(row.get("i:dec", 0)),
                "rmag": float(row.get("i:magpsf", 18.0)),
                "imag": float(row.get("i:magpsf", 17.5)),
                "zmag": float(row.get("i:magpsf", 17.0)),
            },
            "prv_candidates": [],
        }
    except Exception:
        return None


def download_all(num_per_class=2000, output_dir="data/real_alerts"):
    """Download labeled alerts from Fink and save as unified JSON."""
    
    output_path = os.path.join(
        str(PROJECT_ROOT), "streaming_lsst", output_dir
    )
    os.makedirs(output_path, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("FINK BROKER - Downloading Real ZTF Alerts")
    print("=" * 70)
    
    available = get_available_classes()
    all_alerts = []
    
    # Download anomaly classes
    print("\n--- ANOMALY CLASSES (transients) ---")
    for cls in ANOMALY_CLASSES:
        if cls not in available:
            print(f"  [{cls}] not available, skipping")
            continue
        print(f"  [{cls}] fetching up to {num_per_class}...")
        df = fetch_class(cls, n=num_per_class)
        if not df.empty:
            for _, row in df.iterrows():
                alert = fink_to_alert(row, cls, is_anomaly=True)
                if alert:
                    all_alerts.append(alert)
            print(f"    -> got {len(df)} alerts")
        time.sleep(1)
    
    # Download normal classes
    print("\n--- NORMAL CLASSES (non-transient) ---")
    for cls in NORMAL_CLASSES:
        if cls not in available:
            print(f"  [{cls}] not available, skipping")
            continue
        print(f"  [{cls}] fetching up to {num_per_class}...")
        df = fetch_class(cls, n=num_per_class)
        if not df.empty:
            for _, row in df.iterrows():
                alert = fink_to_alert(row, cls, is_anomaly=False)
                if alert:
                    all_alerts.append(alert)
            print(f"    -> got {len(df)} alerts")
        time.sleep(1)
    
    if not all_alerts:
        print("\nERROR: No data downloaded!")
        return ""
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(all_alerts)
    
    # Save
    out_file = os.path.join(output_path, "unified_real_alerts.json")
    with open(out_file, "w") as f:
        json.dump(all_alerts, f, indent=2)
    
    # Summary
    total = len(all_alerts)
    anomalies = sum(1 for a in all_alerts if a["is_ground_truth_anomaly"])
    classes = {}
    for a in all_alerts:
        c = a["fink_class"]
        classes[c] = classes.get(c, 0) + 1
    
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  Total alerts:  {total}")
    print(f"  Anomalies:     {anomalies} ({100*anomalies/total:.1f}%)")
    print(f"  Normal:        {total-anomalies} ({100*(total-anomalies)/total:.1f}%)")
    print(f"\n  Per-class breakdown:")
    for c, n in sorted(classes.items(), key=lambda x: -x[1]):
        label = "ANOMALY" if any(a["fink_class"]==c and a["is_ground_truth_anomaly"] for a in all_alerts) else "NORMAL"
        print(f"    {c:35s} {n:6d}  [{label}]")
    print(f"\n  Saved to: {out_file}")
    print("=" * 70)
    
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download real ZTF data from Fink")
    parser.add_argument("--num-per-class", type=int, default=2000,
                        help="Max alerts per class")
    parser.add_argument("--output-dir", type=str, default="data/real_alerts",
                        help="Output directory")
    args = parser.parse_args()
    
    download_all(args.num_per_class, args.output_dir)


if __name__ == "__main__":
    main()
