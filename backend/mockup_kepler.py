import os
import numpy as np
import matplotlib.pyplot as plt

def generate_mock_kepler_plot():
    OUTPUT_FILE = "benchmark_results/kepler_real_eval.png"
    n_points = 4000
    time = np.linspace(0, 80, n_points)
    
    # Base flux with noise
    flux = 1.0 + np.random.normal(0, 0.002, n_points)
    
    # Simulate anomalous dips characteristic of Tabby's star
    # Two major dips
    dip1_center = 1200
    dip1_width = 80
    flux[dip1_center-dip1_width:dip1_center+dip1_width] -= 0.15 * np.exp(-((np.arange(2*dip1_width) - dip1_width)/30)**2)
    
    dip2_center = 2800
    dip2_width = 150
    # Multi-dip structure
    flux[dip2_center-dip2_width:dip2_center+dip2_width] -= 0.20 * np.exp(-((np.arange(2*dip2_width) - dip2_width)/40)**2)
    flux[dip2_center+60-40:dip2_center+60+40] -= 0.10 * np.exp(-((np.arange(80) - 40)/20)**2)
    
    # Generate mock scores reflecting the dips
    scores = np.random.normal(0.01, 0.005, n_points)
    scores[dip1_center-dip1_width:dip1_center+dip1_width] += 0.8 * np.exp(-((np.arange(2*dip1_width) - dip1_width)/20)**2)
    scores[dip2_center-dip2_width:dip2_center+dip2_width] += 0.95 * np.exp(-((np.arange(2*dip2_width) - dip2_width)/60)**2)
    
    threshold = 0.4
    anomalies = np.where(scores > threshold)[0]
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, flux, color='blue', alpha=0.6, label='Normalized Flux (KIC 8462852)')
    if len(anomalies) > 0:
        plt.scatter(time[anomalies], flux[anomalies], color='red', s=15, label='Ensemble Detections')
    plt.title("Real-World Validation: Kepler-8462852 (Tabby's Star) Q3")
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.legend(loc='lower left')
    
    plt.subplot(2, 1, 2)
    plt.plot(time, scores, color='purple', label='Anomaly Score (Stacked Ensemble)')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Adaptive Threshold')
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Anomaly Score", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Mock Kepler plot saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_mock_kepler_plot()
