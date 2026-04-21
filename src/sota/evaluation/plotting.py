import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging

def plot_anomaly_discovery(flux: np.ndarray, scores: np.ndarray, top_anomalies: list, output_path: str = None):
    """
    Generates a professional-grade astronomical anomaly discovery plot.
    Shows the light-curve, the anomaly score, and highlights detected regions.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # 1. Light Curve Plot
    ax1.plot(flux, color='#1f77b4', linewidth=1.5, label='Observed Flux')
    ax1.set_ylabel('Normalized Flux', fontsize=12, fontweight='bold')
    ax1.set_title('AstroAnomaly Discovery: Light Curve Analysis', fontsize=14, fontweight='bold', pad=15)
    
    # Shade Anomaly Regions
    # We use the top_anomalies indices to shade
    for idx in top_anomalies:
        ax1.axvspan(idx, idx + 32, color='#d62728', alpha=0.3, label='Discovered Anomaly' if idx == top_anomalies[0] else "")

    ax1.legend(loc='upper right')

    # 2. Anomaly Score Plot
    # Scores are usually shorter due to sliding window (N - win_size + 1)
    # We pad them back for alignment
    padded_scores = np.zeros_like(flux)
    padded_scores[:len(scores)] = scores
    
    ax2.fill_between(range(len(padded_scores)), padded_scores, color='#ff7f0e', alpha=0.5, label='Ensemble Anomaly Score')
    ax2.plot(padded_scores, color='#ff7f0e', linewidth=1)
    ax2.axhline(y=np.percentile(scores, 99), color='#d62728', linestyle='--', alpha=0.7, label='99th Percentile Threshold')
    
    ax2.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (Observation Step)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Discovery plot saved to {output_path}")
    
    return fig

if __name__ == "__main__":
    # Test plot with synthetic data
    t = np.linspace(0, 100, 1000)
    flux = np.sin(t)
    flux[500:520] += 5.0 # Add anomaly
    scores = np.abs(np.diff(flux, append=flux[-1]))
    plot_anomaly_discovery(flux, scores, [500], "test_plot.png")
