import json
import numpy as np

def generate_ztf_sample(n=100, clean=False):
    alerts = []
    for i in range(n):
        # 3 Types of Alerts:
        # 1. Normal Background (80%)
        # 2. Simple Anomaly (10%) - Bright, New (Easy to catch with rules)
        # 3. Complex Anomaly (10%) - Subtle, Old (Hard to catch with rules)
        
        r = np.random.random()
        if r < 0.8:
            atype = 'normal'
        elif r < 0.9:
            atype = 'simple_anomaly'
        else:
            atype = 'complex_anomaly'
            
        is_anomaly = (atype != 'normal')
        
        # Mapping to ZTF-like schema
        alert = {
            "alertId": f"ztf_alert_{i}",
            "publisher": "ZTF",
            "is_ground_truth_anomaly": is_anomaly,
            "anomaly_type": atype,
            "candidate": {
                "ra": np.random.uniform(0, 360),
                "dec": np.random.uniform(-90, 90),
                "sigmapsf": np.random.exponential(0.02),
                "diffmaglim": 20.5,
                "ncovhist": np.random.randint(10, 100),
                "jdstarthist": 2459000.5,
                "sgscore1": np.random.normal(0.5, 0.1),
            },
            "prv_candidates": []
        }
        
        cand = alert["candidate"]
        if atype == 'normal':
            cand["magpsf"] = np.random.normal(19.0, 0.1)
            cand["ndethist"] = np.random.randint(50, 100)
            cand["isdiffpos"] = "f"
        elif atype == 'simple_anomaly':
            cand["magpsf"] = np.random.normal(16.0, 0.5)
            cand["ndethist"] = np.random.randint(1, 5)
            cand["isdiffpos"] = "t"
        else: # complex_anomaly
            # Subtle change: mag is normal, but ndethist is weird or isdiffpos is 't' unexpectedly
            cand["magpsf"] = np.random.normal(18.8, 0.05) # Very close to normal
            cand["ndethist"] = np.random.randint(50, 100) # Looks like a normal star
            cand["isdiffpos"] = "t" # Only difference is this flag
            cand["sigmapsf"] = 0.5 # High uncertainty/noise
            
        alerts.append(alert)
    
    with open("ztf_sample_data.json", "w") as f:
        json.dump(alerts, f, indent=2)

if __name__ == "__main__":
    generate_ztf_sample()
