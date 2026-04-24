
import json
import pandas as pd
from alerce.core import Alerce
from pathlib import Path
import time

def fetch_real_ztf_sample(limit=50):
    print(f"Fetching {limit} real ZTF objects using ALeRCE client...")
    client = Alerce()
    
    try:
        # 1. Query objects (paginated)
        objects = client.query_objects(
            survey="ztf",
            format="pandas",
            page_size=limit
        )
        
        print(f"  Successfully fetched {len(objects)} objects.")
        
        real_alerts = []
        for i, (idx, obj) in enumerate(objects.iterrows()):
            oid = obj['oid']
            print(f"  Fetching detections for {oid} ({i+1}/{len(objects)})...")
            try:
                # 2. Get latest detection for each object
                detections = client.query_detections(
                    oid=oid,
                    survey="ztf",
                    format="pandas"
                )
                
                if detections.empty:
                    print(f"    No detections found for {oid}")
                    continue
                    
                # Take the latest detection (usually the first row in ALeRCE)
                det = detections.iloc[0]
                
                # Map to our internal alert schema
                mapped = {
                    "alertId": str(det.get('candid', det.name)),
                    "publisher": "ZTF",
                    "candidate": {
                        "ra": float(det.get('ra', 0)),
                        "dec": float(det.get('dec', 0)),
                        "magpsf": float(det.get('magpsf', 20)),
                        "sigmapsf": float(det.get('sigmapsf', 0.05)),
                        "diffmaglim": float(det.get('diffmaglim', 20.5)),
                        "ndethist": int(obj.get('ndethist', 1)),
                        "ncovhist": int(det.get('ncovhist', 0)),
                        "jdstarthist": float(obj.get('firstmjd', 2459000.5)),
                        "sgscore1": float(det.get('sgscore1', 0.5)),
                        "isdiffpos": "t" if det.get('isdiffpos', 1) == 1 else "f",
                        "objectid": str(oid)
                    },
                    "prv_candidates": []
                }
                real_alerts.append(mapped)
                
                # Be polite to the API
                if i % 5 == 0: time.sleep(0.5)
                
            except Exception as inner_e:
                print(f"    Error fetching detections for {oid}: {inner_e}")
            
        # Save to file
        output_path = "ztf_sample_data.json"
        with open(output_path, "w") as f:
            json.dump(real_alerts, f, indent=2)
            
        print(f"\nSUCCESS: Replaced ztf_sample_data.json with {len(real_alerts)} real ZTF alerts.")
        
    except Exception as e:
        print(f"Error using ALeRCE client: {e}")

if __name__ == "__main__":
    fetch_real_ztf_sample(limit=50) # Reduced to 50 for speed
