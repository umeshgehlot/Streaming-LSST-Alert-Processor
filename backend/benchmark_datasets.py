"""
Benchmark dataset generator for astronomical anomaly detection evaluation.

Generates synthetic light curves with known ground-truth anomaly labels
for rigorous evaluation of detection methods. Each dataset mimics realistic
astronomical phenomena with controlled anomaly injection.
"""

import numpy as np


def _normalize_zscore(flux: np.ndarray) -> np.ndarray:
    mean = np.mean(flux)
    std = np.std(flux)
    if std < 1e-12:
        std = 1.0
    return (flux - mean) / std


def generate_dataset_a(n_points: int = 2000, seed: int = 42) -> dict:
    """Sinusoidal variable star + sudden flux spikes (point anomalies).

    Mimics a pulsating variable star (e.g., RR Lyrae) with injected
    fast transient events such as flares or cosmic ray hits.
    """
    rng = np.random.RandomState(seed)
    time = np.linspace(0, 200, n_points)
    period = 12.5
    flux = 1.0 + 0.15 * np.sin(2 * np.pi * time / period)
    flux += 0.02 * rng.randn(n_points)

    labels = np.zeros(n_points, dtype=int)
    anomaly_count = max(3, int(n_points * 0.04))
    spike_indices = rng.choice(n_points, size=anomaly_count, replace=False)
    for idx in spike_indices:
        amplitude = rng.uniform(0.4, 1.2)
        sign = rng.choice([-1, 1])
        width = rng.randint(1, 4)
        start = max(0, idx - width // 2)
        end = min(n_points, idx + width // 2 + 1)
        flux[start:end] += sign * amplitude
        labels[start:end] = 1

    return {
        "name": "A_PointAnomalies",
        "description": "Sinusoidal variable star with sudden flux spikes",
        "time": time,
        "flux": _normalize_zscore(flux),
        "labels": labels,
        "anomaly_rate": float(np.mean(labels)),
    }


def generate_dataset_b(n_points: int = 2000, seed: int = 123) -> dict:
    """Periodic variable star + gradual dimming events (contextual anomalies).

    Mimics an eclipsing binary system with injected anomalous dimming
    episodes (e.g., exoplanet transit-like events at unexpected times).
    """
    rng = np.random.RandomState(seed)
    time = np.linspace(0, 300, n_points)
    period = 15.0
    flux = 1.0 + 0.1 * np.sin(2 * np.pi * time / period)
    flux += 0.05 * np.sin(2 * np.pi * time / (period * 0.5))
    flux += 0.015 * rng.randn(n_points)

    labels = np.zeros(n_points, dtype=int)
    n_dips = 5
    for _ in range(n_dips):
        center = rng.randint(100, n_points - 100)
        width = rng.randint(15, 40)
        depth = rng.uniform(0.3, 0.6)
        start = max(0, center - width // 2)
        end = min(n_points, center + width // 2)
        gaussian_dip = depth * np.exp(-0.5 * ((np.arange(start, end) - center) / (width / 4)) ** 2)
        flux[start:end] -= gaussian_dip
        labels[start:end] = 1

    return {
        "name": "B_ContextualAnomalies",
        "description": "Periodic star with gradual dimming events",
        "time": time,
        "flux": _normalize_zscore(flux),
        "labels": labels,
        "anomaly_rate": float(np.mean(labels)),
    }


def generate_dataset_c(n_points: int = 2000, seed: int = 456) -> dict:
    """Eclipsing binary pattern + injected supernovae-like transients.

    Simulates regular eclipsing binary dips with injected bright
    transients resembling supernovae light curves (rapid rise, slow decay).
    """
    rng = np.random.RandomState(seed)
    time = np.linspace(0, 250, n_points)
    period = 20.0
    phase = (time % period) / period
    flux = np.where(
        (phase > 0.45) & (phase < 0.55),
        0.7 + 0.3 * np.cos(np.pi * (phase - 0.5) / 0.05) ** 2,
        1.0,
    )
    flux += 0.02 * rng.randn(n_points)

    labels = np.zeros(n_points, dtype=int)
    n_transients = 4
    for _ in range(n_transients):
        onset = rng.randint(50, n_points - 80)
        rise_time = rng.randint(3, 6)
        decay_time = rng.randint(20, 50)
        peak_amplitude = rng.uniform(0.5, 1.5)
        total_width = rise_time + decay_time
        end = min(n_points, onset + total_width)
        for j in range(onset, end):
            t_rel = j - onset
            if t_rel < rise_time:
                flux[j] += peak_amplitude * (t_rel / rise_time)
            else:
                flux[j] += peak_amplitude * np.exp(-(t_rel - rise_time) / (decay_time / 3))
            labels[j] = 1

    return {
        "name": "C_SupernovaTransients",
        "description": "Eclipsing binary with supernovae-like transients",
        "time": time,
        "flux": _normalize_zscore(flux),
        "labels": labels,
        "anomaly_rate": float(np.mean(labels)),
    }


def generate_dataset_d(n_points: int = 2000, seed: int = 789) -> dict:
    """Flat baseline + fast radio burst-style impulses.

    Simulates a quiet stellar source with extremely short, high-energy
    burst events similar to fast radio bursts or magnetar flares.
    """
    rng = np.random.RandomState(seed)
    time = np.linspace(0, 400, n_points)
    flux = 1.0 + 0.01 * rng.randn(n_points)
    trend = 0.0001 * (time - 200) ** 2 / 200
    flux += trend * 0.02

    labels = np.zeros(n_points, dtype=int)
    n_bursts = rng.randint(6, 12)
    burst_indices = rng.choice(range(20, n_points - 20), size=n_bursts, replace=False)
    for idx in burst_indices:
        amplitude = rng.uniform(0.8, 3.0)
        width = rng.randint(1, 5)
        start = max(0, idx - width)
        end = min(n_points, idx + width + 1)
        for j in range(start, end):
            dist = abs(j - idx)
            flux[j] += amplitude * np.exp(-dist ** 2 / max(1, width))
        labels[start:end] = 1

    return {
        "name": "D_FastBursts",
        "description": "Flat baseline with fast radio burst-style impulses",
        "time": time,
        "flux": _normalize_zscore(flux),
        "labels": labels,
        "anomaly_rate": float(np.mean(labels)),
    }


def generate_dataset_e(n_points: int = 2000, seed: int = 1024) -> dict:
    """Multi-periodic signal + chaotic segments (collective anomalies).

    Simulates a multi-mode pulsator with injected segments of chaotic
    behavior representing stellar instability or instrumental glitches.
    """
    rng = np.random.RandomState(seed)
    time = np.linspace(0, 350, n_points)
    flux = (
        1.0
        + 0.08 * np.sin(2 * np.pi * time / 10)
        + 0.05 * np.sin(2 * np.pi * time / 25)
        + 0.03 * np.sin(2 * np.pi * time / 50)
    )
    flux += 0.015 * rng.randn(n_points)

    labels = np.zeros(n_points, dtype=int)
    n_segments = 4
    for _ in range(n_segments):
        start = rng.randint(50, n_points - 80)
        length = rng.randint(25, 60)
        end = min(n_points, start + length)
        chaotic = 0.3 * rng.randn(end - start)
        chaotic += 0.2 * np.sin(2 * np.pi * np.arange(end - start) / rng.uniform(2, 5))
        flux[start:end] += chaotic
        labels[start:end] = 1

    return {
        "name": "E_CollectiveAnomalies",
        "description": "Multi-periodic signal with chaotic segments",
        "time": time,
        "flux": _normalize_zscore(flux),
        "labels": labels,
        "anomaly_rate": float(np.mean(labels)),
    }


def get_all_benchmark_datasets() -> list[dict]:
    """Return all 5 benchmark datasets with ground-truth labels."""
    return [
        generate_dataset_a(),
        generate_dataset_b(),
        generate_dataset_c(),
        generate_dataset_d(),
        generate_dataset_e(),
    ]


if __name__ == "__main__":
    datasets = get_all_benchmark_datasets()
    for ds in datasets:
        anomaly_pct = ds["anomaly_rate"] * 100
        print(f"{ds['name']:30s} | {len(ds['time']):5d} pts | {anomaly_pct:5.1f}% anomalies | {ds['description']}")
