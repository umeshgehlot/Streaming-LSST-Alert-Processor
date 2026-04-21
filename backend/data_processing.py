import sys
from pathlib import Path
# Ensure imports resolve (Add both backend and project root)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# SOTA INTEGRATION (MODULARIZED)
try:
    from src.sota.processing import SotaDataService
    # Note: sota_periodogram is now integrated into SotaDataService or processing
    SOTA_AVAILABLE = True
except ImportError:
    SOTA_AVAILABLE = False
    logging.warning("src.sota.processing not found. Falling back to basic preprocessing.")

def _to_unix_seconds(series: pd.Series) -> pd.Series:
    """Converts datetime series to unix seconds."""
    return series.astype("int64") // 10**9

def load_and_preprocess_csv(raw_bytes: bytes, recent_only: bool = True, recent_years: int = 2) -> dict:
    """
    Expert-level preprocessing. Uses Gaussian Process Augmentation (GPA) if available,
    otherwise falls back to robust linear interpolation.
    """
    dataframe = pd.read_csv(BytesIO(raw_bytes))
    lower_name_map = {column.lower().strip(): column for column in dataframe.columns}
    
    if "time" not in lower_name_map or "flux" not in lower_name_map:
        raise ValueError("CSV must contain 'time' and 'flux' columns")
    
    # Identify key columns
    time_col = lower_name_map["time"]
    flux_col = lower_name_map["flux"]
    err_col = lower_name_map.get("flux_err") or lower_name_map.get("error")
    band_col = lower_name_map.get("band") or lower_name_map.get("passband")
    
    dataframe = dataframe.copy()
    dataframe["time"] = pd.to_numeric(dataframe[time_col], errors="coerce")
    dataframe["flux"] = pd.to_numeric(dataframe[flux_col], errors="coerce")
    
    if err_col:
        dataframe["flux_error"] = pd.to_numeric(dataframe[err_col], errors="coerce")
    if band_col:
        dataframe["band"] = dataframe[band_col]
        
    dataframe = dataframe.dropna(subset=["time", "flux"]).sort_values("time")
    original_points = len(dataframe)
    
    if dataframe.empty:
        raise ValueError("No valid rows found after preprocessing")

    # [UPGRADE] SOTA Gaussian Process Augmentation
    if SOTA_AVAILABLE:
        try:
            # Avocado GPA handles irregular sampling and produces high-cadence 256-point windows
            res = SotaDataService.process_with_gpa(dataframe)
            return {
                "time_values": res["time"],
                "flux_values": res["flux"],
                "normalized_flux": res["normalized_flux"],
                "points": res["points"],
                "normalized_points": res["normalized_points"],
                "meta": {
                    "original_points": int(original_points),
                    "processed_points": len(res["time"]),
                    "method": "State-of-the-Art (Avocado GPA)",
                    "status": "Enhanced (Physical Reconstruction)"
                }
            }
        except Exception as e:
            logging.error(f"SOTA GPA processing failed: {e}. Falling back to basic mode.")

    # [FALLBACK] Basic Preprocessing (Linear Interpolation)
    dataframe["flux"] = dataframe["flux"].interpolate(method="linear").ffill().bfill()
    mean = dataframe["flux"].mean()
    std = dataframe["flux"].std() if dataframe["flux"].std() > 0 else 1.0
    normalized_flux = (dataframe["flux"] - mean) / std
    
    points = [
        {"time": float(t), "flux": float(f)}
        for t, f in zip(dataframe["time"], dataframe["flux"])
    ]
    normalized_points = [
        {"time": float(t), "flux": float(f)}
        for t, f in zip(dataframe["time"], normalized_flux)
    ]
    
    return {
        "time_values": dataframe["time"].tolist(),
        "flux_values": dataframe["flux"].tolist(),
        "normalized_flux": normalized_flux.tolist(),
        "points": points,
        "normalized_points": normalized_points,
        "meta": {
            "original_points": int(original_points),
            "processed_points": int(len(dataframe)),
            "method": "Baseline (Linear Interpolation)",
            "status": "Degraded (Geometric Only)"
        }
    }

def lomb_scargle_periodogram(
    time_values: list[float],
    flux_values: list[float],
    min_frequency: float = 0.01,
    max_frequency: float = 2.0,
    steps: int = 300,
) -> dict:
    """Computes the Lomb-Scargle power spectrum."""
    if SOTA_AVAILABLE:
        try:
            return sota_periodogram(time_values, flux_values)
        except Exception as e:
            logging.error(f"SOTA Periodogram failed: {e}")

    # Baseline implementation
    t = np.asarray(time_values, dtype=np.float64)
    y = np.asarray(flux_values, dtype=np.float64)
    y = y - np.mean(y)
    frequencies = np.linspace(min_frequency, max_frequency, max(50, steps))
    angular = 2.0 * np.pi * frequencies
    powers = np.zeros_like(frequencies)
    
    # Standard Scargle (Slow loop)
    for index, omega in enumerate(angular):
        tau = np.arctan2(np.sum(np.sin(2 * omega * t)), np.sum(np.cos(2 * omega * t))) / (2 * omega + 1e-12)
        cos_term = np.cos(omega * (t - tau))
        sin_term = np.sin(omega * (t - tau))
        numerator = (np.sum(y * cos_term) ** 2) / (np.sum(cos_term**2) + 1e-12)
        numerator += (np.sum(y * sin_term) ** 2) / (np.sum(sin_term**2) + 1e-12)
        powers[index] = 0.5 * numerator / (np.var(y) + 1e-12)
        
    peak_idx = int(np.argmax(powers))
    return {
        "frequency": frequencies.astype(float).tolist(),
        "power": powers.astype(float).tolist(),
        "peak_frequency": float(frequencies[peak_idx]),
        "peak_period": float(1.0 / max(frequencies[peak_idx], 1e-12)),
    }
