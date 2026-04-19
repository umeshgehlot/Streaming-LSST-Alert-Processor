import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from ml_models import ensemble_discovery

FITS_PATH = "data/real/kic8462852_q3.fits"
EPOCHS = 20

def load_kepler_data(file_path):
    print(f"Loading Kepler data from {file_path}...")
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']
        mask = ~np.isnan(flux)
        time = time[mask]
        flux = flux[mask]
        # Median normalization
        flux = flux / np.median(flux)
        return time.tolist(), flux.tolist()

def run_real_evaluation():
    if not os.path.exists(FITS_PATH):
        print(f"Error: FITS file not found at {FITS_PATH}.")
        return

    time, flux = load_kepler_data(FITS_PATH)
    
    print("Running Ensemble Discovery on real data...")
    results = ensemble_discovery(
        dataset_id="kepler_8462852",
        time_values=time,
        flux_values=flux,
        model_names=["autoencoder", "vae", "transformer"],
        epochs=EPOCHS,
        threshold_percentile=99.0, # Top 1% anomalies
        batch_size=64,
        use_gpu=False
    )
    
    anomalies = results["anomaly_indices"]
    scores = results["confidence_index"]
    threshold = results["threshold"]
    
    print(f"Detected {len(anomalies)} anomalous points.")
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, flux, color='blue', alpha=0.6, label='Normalized Flux')
    if len(anomalies) > 0:
        anom_times = [time[i] for i in anomalies]
        anom_flux = [flux[i] for i in anomalies]
        plt.scatter(anom_times, anom_flux, color='red', s=10, label='Detected Anomalies')
    plt.title("Real-World Validation: Kepler-8462852 (Tabby's Star) Q3")
    plt.ylabel("Normalized Flux")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, scores, color='purple', label='Anomaly Score (Ensemble)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='99% Threshold')
    plt.xlabel("Time (BJD - 2454833)")
    plt.ylabel("Anomaly Score")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("benchmark_results/kepler_real_eval.png", dpi=300)
    print("Plot saved to benchmark_results/kepler_real_eval.png")

if __name__ == "__main__":
    run_real_evaluation()
