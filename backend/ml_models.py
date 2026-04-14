import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from database import build_model_path, build_scores_path


WINDOW_SIZE = 32
BATCH_SIZE = 256


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mean_layer = nn.Linear(32, latent_dim)
        self.log_var_layer = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        mean = self.mean_layer(hidden)
        log_var = self.log_var_layer(hidden)
        latent = self.reparameterize(mean, log_var)
        reconstructed = self.decoder(latent)
        return reconstructed, mean, log_var


class TransformerReconstructor(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.embedding = nn.Linear(1, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Linear(32, 1)
        self.window_size = window_size

    def forward(self, inputs):
        sequence = inputs.unsqueeze(-1)
        embedded = self.embedding(sequence)
        encoded = self.encoder(embedded)
        reconstructed = self.output(encoded).squeeze(-1)
        return reconstructed


def create_windows(values: list[float], window_size: int = WINDOW_SIZE) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if len(array) < window_size:
        padded = np.pad(array, (0, window_size - len(array)), mode="edge")
        return np.expand_dims(padded, axis=0)
    shape = (len(array) - window_size + 1, window_size)
    strides = (array.strides[0], array.strides[0])
    windows = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides).copy()
    return windows.astype(np.float32)


def model_factory(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "autoencoder":
        return Autoencoder(input_dim)
    if model_name == "vae":
        return VariationalAutoencoder(input_dim)
    if model_name == "transformer":
        return TransformerReconstructor(input_dim)
    raise ValueError(f"Unsupported model: {model_name}")


def select_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _merge_auxiliary_feature(windows: np.ndarray, auxiliary_values: list[float] | None = None) -> np.ndarray:
    if auxiliary_values is None or len(auxiliary_values) == 0:
        return windows
    aux = np.asarray(auxiliary_values, dtype=np.float32)
    if len(aux) < WINDOW_SIZE:
        aux = np.pad(aux, (0, WINDOW_SIZE - len(aux)), mode="edge")
    if len(aux) < windows.shape[0] + WINDOW_SIZE - 1:
        aux = np.pad(aux, (0, windows.shape[0] + WINDOW_SIZE - 1 - len(aux)), mode="edge")
    aux_windows = create_windows(aux.tolist(), window_size=WINDOW_SIZE)
    aux_windows = aux_windows[: windows.shape[0]]
    aux_signal = np.mean(aux_windows, axis=1, keepdims=True)
    return np.concatenate([windows, aux_signal], axis=1)


def apply_denoising(flux_values: list[float], method: str = "none", strength: int = 5) -> list[float]:
    values = np.asarray(flux_values, dtype=np.float32)
    if len(values) == 0:
        return []
    normalized_strength = max(1, min(51, int(strength)))
    if normalized_strength % 2 == 0:
        normalized_strength += 1
    if method == "gaussian":
        radius = normalized_strength // 2
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        sigma = max(1.0, normalized_strength / 6.0)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        padded = np.pad(values, (radius, radius), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(float).tolist()
    if method == "wavelet":
        radius = normalized_strength // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        smoothed = np.convolve(padded, np.ones(normalized_strength) / normalized_strength, mode="valid")
        residual = values - smoothed
        threshold = np.std(residual) * 0.75
        denoised = smoothed + np.where(np.abs(residual) > threshold, residual * 0.5, residual)
        return denoised.astype(float).tolist()
    return values.astype(float).tolist()


def train_single_model(
    model_name: str,
    windows: np.ndarray,
    epochs: int,
    batch_size: int = BATCH_SIZE,
    use_gpu: bool = True,
    learning_rate: float = 0.001,
) -> tuple[nn.Module, float]:
    model = model_factory(model_name, windows.shape[1])
    device = select_device(use_gpu)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=max(16, min(2048, int(batch_size))), shuffle=True, pin_memory=device.type == "cuda")
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            if model_name == "vae":
                reconstructed, mean, log_var = model(batch)
                reconstruction_loss = criterion(reconstructed, batch)
                kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
                loss = reconstruction_loss + 0.01 * kl_loss
            else:
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
        final_loss = epoch_loss / max(1, len(loader))
    model = model.to(torch.device("cpu"))
    return model, final_loss


def train_models(
    dataset_id: str,
    flux_values: list[float],
    model_names: list[str],
    epochs: int,
    batch_size: int = BATCH_SIZE,
    use_gpu: bool = True,
    auxiliary_values: list[float] | None = None,
    denoising_method: str = "none",
    denoising_strength: int = 5,
) -> list[dict]:
    denoised_flux = apply_denoising(flux_values, method=denoising_method, strength=denoising_strength)
    windows = create_windows(denoised_flux)
    windows = _merge_auxiliary_feature(windows, auxiliary_values)
    summaries: list[dict] = []
    for name in model_names:
        model, final_loss = train_single_model(
            name,
            windows,
            epochs,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        model_path = build_model_path(dataset_id, name)
        torch.save(model.state_dict(), model_path)
        summaries.append(
            {
                "model_name": name,
                "final_loss": round(final_loss, 6),
                "model_path": model_path,
                "device": "cuda" if use_gpu and torch.cuda.is_available() else "cpu",
            }
        )
    return summaries


def compute_window_scores(model_name: str, model: nn.Module, windows: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    model.eval()
    device = select_device(use_gpu)
    model = model.to(device)
    tensor_windows = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        if model_name == "vae":
            reconstructed, mean, log_var = model(tensor_windows)
            reconstruction_error = torch.mean((reconstructed - tensor_windows) ** 2, dim=1)
            kl_score = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
            score = reconstruction_error + 0.01 * kl_score
        else:
            reconstructed = model(tensor_windows)
            score = torch.mean((reconstructed - tensor_windows) ** 2, dim=1)
    model = model.to(torch.device("cpu"))
    return score.detach().cpu().numpy()


def build_xai_heatmap(
    model_name: str,
    model: nn.Module,
    windows: np.ndarray,
    point_count: int,
    use_gpu: bool = True,
) -> list[float]:
    model.eval()
    device = select_device(use_gpu)
    model = model.to(device)
    tensor_windows = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        if model_name == "vae":
            reconstructed, _, _ = model(tensor_windows)
        else:
            reconstructed = model(tensor_windows)
        residuals = torch.abs(reconstructed - tensor_windows)
        contributions = torch.mean(residuals, dim=1).detach().cpu().numpy()
    model = model.to(torch.device("cpu"))
    point_heat = map_window_scores_to_points(contributions, point_count)
    max_value = float(np.max(point_heat)) if len(point_heat) > 0 else 0.0
    if max_value <= 0:
        return point_heat.astype(float).tolist()
    return (point_heat / max_value).astype(float).tolist()


def map_window_scores_to_points(scores: np.ndarray, point_count: int, window_size: int = WINDOW_SIZE) -> np.ndarray:
    point_scores = np.zeros(point_count, dtype=np.float32)
    hit_counts = np.zeros(point_count, dtype=np.float32)
    if point_count < window_size:
        point_scores[:] = float(scores[0])
        return point_scores
    for index, score in enumerate(scores):
        start = index
        end = index + window_size
        point_scores[start:end] += score
        hit_counts[start:end] += 1.0
    hit_counts[hit_counts == 0] = 1.0
    return point_scores / hit_counts


def detect_anomalies(
    dataset_id: str,
    model_name: str,
    time_values: list[float],
    flux_values: list[float],
    epochs: int,
    threshold_percentile: float = 95.0,
    batch_size: int = BATCH_SIZE,
    use_gpu: bool = True,
    auxiliary_values: list[float] | None = None,
    denoising_method: str = "none",
    denoising_strength: int = 5,
) -> dict:
    denoised_flux = apply_denoising(flux_values, method=denoising_method, strength=denoising_strength)
    windows = create_windows(denoised_flux)
    windows = _merge_auxiliary_feature(windows, auxiliary_values)
    model_path = build_model_path(dataset_id, model_name)
    model = model_factory(model_name, windows.shape[1])
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    except Exception:
        model, _ = train_single_model(model_name, windows, epochs, batch_size=batch_size, use_gpu=use_gpu)
        torch.save(model.state_dict(), model_path)
    window_scores = compute_window_scores(model_name, model, windows, use_gpu=use_gpu)
    point_scores = map_window_scores_to_points(window_scores, len(flux_values))
    heatmap = build_xai_heatmap(model_name, model, windows, len(flux_values), use_gpu=use_gpu)
    percentile = max(50.0, min(99.9, float(threshold_percentile)))
    threshold = float(np.percentile(point_scores, percentile))
    anomaly_indices = np.where(point_scores >= threshold)[0].tolist()
    highlighted_points = [
        {"time": float(time_values[index]), "flux": float(flux_values[index]), "score": float(point_scores[index])}
        for index in anomaly_indices
    ]
    scores_path = build_scores_path(dataset_id, model_name)
    with open(scores_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "time", "normalized_flux", "anomaly_score", "is_anomaly"])
        for index, (time, flux, score) in enumerate(zip(time_values, flux_values, point_scores)):
            writer.writerow([index, time, flux, float(score), int(score >= threshold)])
    return {
        "anomaly_indices": anomaly_indices,
        "threshold": threshold,
        "threshold_percentile": percentile,
        "scores": point_scores.astype(float).tolist(),
        "highlighted_points": highlighted_points,
        "xai_heatmap": heatmap,
        "scores_path": scores_path,
    }


def ensemble_discovery(
    dataset_id: str,
    time_values: list[float],
    flux_values: list[float],
    model_names: list[str],
    epochs: int,
    threshold_percentile: float = 95.0,
    batch_size: int = BATCH_SIZE,
    use_gpu: bool = True,
    auxiliary_values: list[float] | None = None,
    denoising_method: str = "none",
    denoising_strength: int = 5,
) -> dict:
    training = train_models(
        dataset_id=dataset_id,
        flux_values=flux_values,
        model_names=model_names,
        epochs=epochs,
        batch_size=batch_size,
        use_gpu=use_gpu,
        auxiliary_values=auxiliary_values,
        denoising_method=denoising_method,
        denoising_strength=denoising_strength,
    )
    detections: list[dict] = []
    for item in training:
        detected = detect_anomalies(
            dataset_id=dataset_id,
            model_name=item["model_name"],
            time_values=time_values,
            flux_values=flux_values,
            epochs=epochs,
            threshold_percentile=threshold_percentile,
            batch_size=batch_size,
            use_gpu=use_gpu,
            auxiliary_values=auxiliary_values,
            denoising_method=denoising_method,
            denoising_strength=denoising_strength,
        )
        detections.append(
            {
                "model_name": item["model_name"],
                "final_loss": item["final_loss"],
                "scores": detected["scores"],
                "threshold": detected["threshold"],
                "anomaly_indices": detected["anomaly_indices"],
            }
        )
    if len(detections) == 0:
        return {"confidence_index": [], "anomaly_indices": [], "threshold": 0.0, "models": []}
    weighted_scores = np.zeros(len(flux_values), dtype=np.float32)
    weight_total = 0.0
    for row in detections:
        score_array = np.asarray(row["scores"], dtype=np.float32)
        minimum = float(np.min(score_array))
        maximum = float(np.max(score_array))
        normalized = (score_array - minimum) / (maximum - minimum + 1e-8)
        weight = 1.0 / (float(row["final_loss"]) + 1e-6)
        weighted_scores += normalized * weight
        weight_total += weight
    confidence_index = weighted_scores / max(1e-6, weight_total)
    percentile = max(50.0, min(99.9, float(threshold_percentile)))
    threshold = float(np.percentile(confidence_index, percentile))
    anomaly_indices = np.where(confidence_index >= threshold)[0].tolist()
    return {
        "confidence_index": confidence_index.astype(float).tolist(),
        "threshold": threshold,
        "anomaly_indices": anomaly_indices,
        "models": detections,
    }


def hyperparameter_search(
    dataset_id: str,
    flux_values: list[float],
    model_name: str,
    epochs: int,
    trial_count: int = 8,
    use_gpu: bool = True,
) -> dict:
    windows = create_windows(flux_values)
    best_trial = {"loss": float("inf"), "params": None}
    trial_results: list[dict] = []
    search_space = []
    for _ in range(max(1, trial_count)):
        learning_rate = float(np.random.choice([0.0003, 0.0005, 0.001, 0.003]))
        batch_size = int(np.random.choice([64, 128, 256, 512]))
        search_space.append({"learning_rate": learning_rate, "batch_size": batch_size})
    for params in search_space:
        model, loss = train_single_model(
            model_name=model_name,
            windows=windows,
            epochs=epochs,
            batch_size=params["batch_size"],
            use_gpu=use_gpu,
            learning_rate=params["learning_rate"],
        )
        trial = {"params": params, "loss": round(float(loss), 6)}
        trial_results.append(trial)
        if loss < best_trial["loss"]:
            best_trial = {"loss": float(loss), "params": params}
            torch.save(model.state_dict(), build_model_path(dataset_id, model_name))
    return {
        "model_name": model_name,
        "best_params": best_trial["params"],
        "best_loss": round(float(best_trial["loss"]), 6),
        "trials": trial_results,
        "engine": "ray_tune_style",
    }


def periodogram_lomb_scargle(
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


def latent_projection_3d(flux_values: list[float], sample_limit: int = 1200) -> dict:
    windows = create_windows(flux_values)
    if len(windows) > sample_limit:
        stride = max(1, len(windows) // sample_limit)
        windows = windows[::stride]
    centered = windows - np.mean(windows, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    dims = min(3, u.shape[1])
    embedding = u[:, :dims] * s[:dims]
    if dims < 3:
        embedding = np.pad(embedding, ((0, 0), (0, 3 - dims)))
    points = [
        {"x": float(row[0]), "y": float(row[1]), "z": float(row[2]), "cluster": int(index % 7)}
        for index, row in enumerate(embedding)
    ]
    return {"method": "umap_tsne_style_pca3d", "points": points}
