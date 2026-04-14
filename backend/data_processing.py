from io import BytesIO

import numpy as np
import pandas as pd


def _to_unix_seconds(series: pd.Series) -> pd.Series:
    return series.astype("int64") // 10**9


def load_and_preprocess_csv(raw_bytes: bytes, recent_only: bool = True, recent_years: int = 2) -> dict:
    dataframe = pd.read_csv(BytesIO(raw_bytes))
    lower_name_map = {column.lower().strip(): column for column in dataframe.columns}
    if "time" not in lower_name_map or "flux" not in lower_name_map:
        raise ValueError("CSV must contain 'time' and 'flux' columns")
    dataframe = dataframe[[lower_name_map["time"], lower_name_map["flux"]]].copy()
    dataframe.columns = ["time", "flux"]
    dataframe["flux"] = pd.to_numeric(dataframe["flux"], errors="coerce")
    original_points = len(dataframe)
    time_as_datetime = pd.to_datetime(dataframe["time"], errors="coerce", utc=True)
    datetime_ratio = float(time_as_datetime.notna().mean())
    start_time = None
    end_time = None
    time_mode = "numeric"
    if datetime_ratio >= 0.7:
        time_mode = "datetime"
        dataframe["time"] = time_as_datetime
        dataframe = dataframe.dropna(subset=["time"]).sort_values("time")
        if recent_only:
            cutoff = dataframe["time"].max() - pd.DateOffset(years=recent_years)
            dataframe = dataframe[dataframe["time"] >= cutoff]
        dataframe["flux"] = dataframe["flux"].interpolate(method="linear").ffill().bfill()
        dataframe["time"] = dataframe["time"].ffill().bfill()
        start_time = dataframe["time"].min().isoformat() if not dataframe.empty else None
        end_time = dataframe["time"].max().isoformat() if not dataframe.empty else None
        numeric_time = _to_unix_seconds(dataframe["time"])
    else:
        dataframe["time"] = pd.to_numeric(dataframe["time"], errors="coerce")
        dataframe = dataframe.sort_values("time")
        dataframe["time"] = dataframe["time"].interpolate(method="linear").ffill().bfill()
        dataframe["flux"] = dataframe["flux"].interpolate(method="linear").ffill().bfill()
        numeric_time = dataframe["time"].astype(float)
    dataframe["flux"] = dataframe["flux"].interpolate(method="linear").ffill().bfill()
    if dataframe.empty:
        raise ValueError("No valid rows found after preprocessing")
    mean = dataframe["flux"].mean()
    std = dataframe["flux"].std()
    if std == 0:
        std = 1.0
    normalized_flux = (dataframe["flux"] - mean) / std
    points = [
        {"time": float(time), "flux": float(flux)}
        for time, flux in zip(numeric_time, dataframe["flux"])
    ]
    normalized_points = [
        {"time": float(time), "flux": float(flux)}
        for time, flux in zip(numeric_time, normalized_flux)
    ]
    return {
        "time_values": numeric_time.astype(float).tolist(),
        "flux_values": dataframe["flux"].astype(float).tolist(),
        "normalized_flux": normalized_flux.astype(float).tolist(),
        "points": points,
        "normalized_points": normalized_points,
        "meta": {
            "original_points": int(original_points),
            "processed_points": int(len(dataframe)),
            "recent_only": bool(recent_only),
            "recent_years": int(recent_years),
            "time_mode": time_mode,
            "start_time": start_time,
            "end_time": end_time,
        },
    }


def lomb_scargle_periodogram(
    time_values: list[float],
    flux_values: list[float],
    min_frequency: float = 0.01,
    max_frequency: float = 2.0,
    steps: int = 300,
) -> dict:
    t = np.asarray(time_values, dtype=np.float64)
    y = np.asarray(flux_values, dtype=np.float64)
    y = y - np.mean(y)
    frequencies = np.linspace(min_frequency, max_frequency, max(50, steps))
    angular = 2.0 * np.pi * frequencies
    powers = np.zeros_like(frequencies)
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
