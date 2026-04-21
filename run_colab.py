import os
import sys
import logging
from src.interface import AstroAnomalyEngine

def main():
    """
    Primary Entry Point for the AstroAnomaly Discovery Pipeline.
    Designed for seamless use in Google Colab and Local Workstations.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n" + "="*60)
    print("   AstroAnomaly Unified Discovery Pipeline (SOTA Mode)   ")
    print("="*60 + "\n")
    
    # 1. Initialize the AstroAnomalyEngine
    # The engine coordinates all experts (Transformer, TranAD, TimesNet)
    engine = AstroAnomalyEngine()

    # 2. Identify the target dataset
    # Defaulting to the NASA Fireball survey found in your uploads
    data_path = "data/uploads/1336ea5a-63b9-4b70-be46-380c2eaf22c8_nasa_fireball_20240420_20260420.csv"
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data path not found: {data_path}")
        return

    # 3. Mode Selection (Demonstration Mode)
    # We'll run a single-epoch training for demonstration
    # engine.train(data_path, epochs=1)
    
    try:
        # 4. Execute Discovery Pipeline
        results = engine.discover(data_path)
        
        # 5. Export Scientific Discovery Report
        report_path = engine.save_report(results, filename="colab_discovery_report.json")
        
        # 6. Generate Visual Analytics
        plot_path = engine.visualize_discovery(results, data_path, filename="discovery_plot.png")
        
        # 7. Summary Output
        print("\n--- Discovery Summary ---")
        print(f"[*] Anomaly Scores Computed: {len(results['anomaly_scores'])} windows")
        print(f"[*] Statistical Anomalies Identified: {len(results['top_anomalies'])}")
        print(f"[*] Visual Discovery Plot: {plot_path}")
        print(f"\n[REPORT] Saved to: {report_path}")
        print("\n" + "="*60)

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
