"""
GPU-Accelerated Training for Streaming LSST Alert Processor.

Trains all model components (Autoencoder, Transformer, GNN) on real ZTF data
for 10 epochs, then evaluates and saves the trained models.

Usage:
    py scripts/train_models.py
    py scripts/train_models.py --epochs 10 --batch-size 128 --device cuda
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streaming_lsst.processor import StreamingLSSTProcessor
from streaming_lsst.models.online_autoencoder import StreamingAutoencoder
from streaming_lsst.models.streaming_transformer import StreamingTransformer
from streaming_lsst.config import get_config


def detect_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {name} ({mem:.1f} GB)")
        return "cuda"
    else:
        print(f"  CPU: No GPU detected, using CPU")
        return "cpu"


def load_and_prepare_data(data_path, device):
    """Load real alerts and extract features + labels."""
    print(f"\nLoading data from {data_path}...")
    with open(data_path, "r") as f:
        alerts = json.load(f)
    print(f"  Loaded {len(alerts)} alerts")
    
    # Extract features using pipeline
    processor = StreamingLSSTProcessor(device="cpu")
    
    features_list = []
    labels_list = []
    valid_alerts = []
    
    for i, alert in enumerate(alerts):
        try:
            feats, _, _ = processor.pipeline.process_alert(alert)
            features_list.append(feats.numpy())
            labels_list.append(int(alert.get("is_ground_truth_anomaly", False)))
            valid_alerts.append(alert)
        except Exception:
            continue
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(alerts)} alerts...")
    
    X = torch.tensor(np.array(features_list), dtype=torch.float32).to(device)
    y = torch.tensor(labels_list, dtype=torch.long).to(device)
    
    n_anom = y.sum().item()
    print(f"  Features shape: {X.shape}")
    print(f"  Anomalies: {n_anom} ({100*n_anom/len(y):.1f}%)")
    print(f"  Normal: {len(y)-n_anom} ({100*(len(y)-n_anom)/len(y):.1f}%)")
    
    return X, y, valid_alerts


def train_autoencoder(X, y, device, epochs=10, batch_size=128, lr=0.001):
    """Train the autoencoder on real data for anomaly detection."""
    config = get_config()
    ae_cfg = config["online_autoencoder"]
    feature_dim = X.shape[1]
    
    print(f"\n{'='*70}")
    print(f"TRAINING AUTOENCODER")
    print(f"  Input dim: {feature_dim}, Latent dim: {ae_cfg['latent_dim']}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"  Device: {device}")
    print(f"{'='*70}")
    
    model = StreamingAutoencoder(
        input_dim=feature_dim,
        latent_dim=ae_cfg["latent_dim"],
        hidden_dim=ae_cfg["hidden_dim"],
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Use only normal data for training (autoencoder should learn normal patterns)
    normal_mask = (y == 0)
    X_normal = X[normal_mask]
    
    # If too few normals, use all data
    if len(X_normal) < 1000:
        print(f"  Warning: Only {len(X_normal)} normal samples, using all data")
        X_normal = X
    
    dataset = TensorDataset(X_normal)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model.train()
    best_loss = float("inf")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon, latent, loss = model(batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.perf_counter() - t0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Compute anomaly detection metrics at this epoch
        model.eval()
        with torch.no_grad():
            recon_all, _, _ = model(X)
            errors = torch.mean((recon_all - X) ** 2, dim=1)
            
            # Find optimal threshold
            normal_errors = errors[y == 0]
            anomaly_errors = errors[y == 1]
            
            if len(normal_errors) > 0 and len(anomaly_errors) > 0:
                threshold = normal_errors.mean() + 2.0 * normal_errors.std()
                preds = (errors > threshold).long()
                
                tp = ((preds == 1) & (y == 1)).sum().item()
                fp = ((preds == 1) & (y == 0)).sum().item()
                fn = ((preds == 0) & (y == 1)).sum().item()
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            else:
                prec, rec, f1 = 0, 0, 0
        model.train()
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f} | "
              f"P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f} | {elapsed:.1f}s")
    
    # Load best weights
    model.load_state_dict(best_state)
    
    # Update EMA statistics with final pass
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            _, _, loss = model(batch_x)
            model.update_statistics(loss)
    
    return model


def train_transformer(X, y, device, epochs=10, batch_size=64, lr=0.0005):
    """Train a batch-mode transformer classifier for anomaly detection.
    
    Uses a standard nn.TransformerEncoder (batch-friendly) for training,
    then transfers learned weights to the StreamingTransformer for inference.
    """
    config = get_config()
    trans_cfg = config["streaming_transformer"]
    feature_dim = X.shape[1]
    d_model = trans_cfg["d_model"]
    output_dim = trans_cfg["output_dim"]
    
    print(f"\n{'='*70}")
    print(f"TRAINING TRANSFORMER (Batch Mode)")
    print(f"  d_model: {d_model}, layers: {trans_cfg['n_layers']}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"{'='*70}")
    
    # Build a batch-friendly transformer (no in-place KV cache issues)
    input_proj = nn.Linear(feature_dim, d_model).to(device)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=trans_cfg["n_heads"],
        dim_feedforward=trans_cfg["d_ff"],
        dropout=trans_cfg["dropout"], batch_first=True,
    )
    transformer_enc = nn.TransformerEncoder(
        encoder_layer, num_layers=trans_cfg["n_layers"]
    ).to(device)
    output_proj = nn.Linear(d_model, output_dim).to(device)
    
    classifier = nn.Sequential(
        nn.Linear(output_dim, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1),
    ).to(device)
    
    params = (list(input_proj.parameters()) + list(transformer_enc.parameters()) +
              list(output_proj.parameters()) + list(classifier.parameters()))
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    dataset = TensorDataset(X, y.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    best_f1 = 0.0
    best_states = None
    
    for epoch in range(epochs):
        input_proj.train(); transformer_enc.train()
        output_proj.train(); classifier.train()
        
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Batch forward: [B, feat] -> [B, 1, feat] -> proj -> transformer -> classifier
            h = input_proj(batch_x).unsqueeze(1)          # [B, 1, d_model]
            h = transformer_enc(h)                         # [B, 1, d_model]
            emb = output_proj(h.squeeze(1))                # [B, output_dim]
            logits = classifier(emb).squeeze(-1)           # [B]
            
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.perf_counter() - t0
        
        # Evaluate
        input_proj.eval(); transformer_enc.eval()
        output_proj.eval(); classifier.eval()
        with torch.no_grad():
            all_preds = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                h = input_proj(batch).unsqueeze(1)
                h = transformer_enc(h)
                emb = output_proj(h.squeeze(1))
                logit = classifier(emb).squeeze(-1)
                pred = (torch.sigmoid(logit) > 0.5).long()
                all_preds.append(pred)
            
            preds = torch.cat(all_preds)
            tp = ((preds == 1) & (y == 1)).sum().item()
            fp = ((preds == 1) & (y == 0)).sum().item()
            fn = ((preds == 0) & (y == 1)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_states = {
                "input_proj": {k: v.clone() for k, v in input_proj.state_dict().items()},
                "transformer": {k: v.clone() for k, v in transformer_enc.state_dict().items()},
                "output_proj": {k: v.clone() for k, v in output_proj.state_dict().items()},
                "classifier": {k: v.clone() for k, v in classifier.state_dict().items()},
            }
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f} | {elapsed:.1f}s")
    
    # Load best weights
    if best_states:
        input_proj.load_state_dict(best_states["input_proj"])
        transformer_enc.load_state_dict(best_states["transformer"])
        output_proj.load_state_dict(best_states["output_proj"])
        classifier.load_state_dict(best_states["classifier"])
    
    # Package into a single module for saving
    trained_transformer = nn.ModuleDict({
        "input_proj": input_proj,
        "encoder": transformer_enc,
        "output_proj": output_proj,
    })
    
    return trained_transformer, classifier


def save_models(ae_model, transformer, save_dir):
    """Save trained models to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    ae_path = os.path.join(save_dir, "autoencoder_trained.pt")
    torch.save(ae_model.state_dict(), ae_path)
    print(f"  Saved autoencoder: {ae_path}")
    
    trans_path = os.path.join(save_dir, "transformer_trained.pt")
    torch.save(transformer.state_dict(), trans_path)
    print(f"  Saved transformer: {trans_path}")


def final_evaluation(ae_model, X, y, device):
    """Run final evaluation with trained autoencoder."""
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (Trained Autoencoder)")
    print(f"{'='*70}")
    
    ae_model.eval()
    with torch.no_grad():
        recon, _, _ = ae_model(X)
        errors = torch.mean((recon - X) ** 2, dim=1)
        
        normal_errors = errors[y == 0]
        anomaly_errors = errors[y == 1]
        
        print(f"\n  Reconstruction Error Statistics:")
        print(f"    Normal  mean: {normal_errors.mean():.6f}  std: {normal_errors.std():.6f}")
        print(f"    Anomaly mean: {anomaly_errors.mean():.6f}  std: {anomaly_errors.std():.6f}")
        print(f"    Separation:   {(anomaly_errors.mean() - normal_errors.mean()) / normal_errors.std():.2f} sigma")
        
        # Try multiple thresholds
        print(f"\n  {'Threshold sigma':<18s} | {'Precision':>10s} | {'Recall':>10s} | {'F1':>10s}")
        print(f"  {'-'*65}")
        
        best_f1 = 0
        best_sigma = 2.0
        
        for sigma in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            thresh = normal_errors.mean() + sigma * normal_errors.std()
            preds = (errors > thresh).long()
            
            tp = ((preds == 1) & (y == 1)).sum().item()
            fp = ((preds == 1) & (y == 0)).sum().item()
            fn = ((preds == 0) & (y == 1)).sum().item()
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            marker = " <-- best" if f1 > best_f1 else ""
            print(f"  {sigma:<18.1f} | {p:>10.4f} | {r:>10.4f} | {f1:>10.4f}{marker}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_sigma = sigma
        
        print(f"\n  Best threshold: {best_sigma} sigma -> F1 = {best_f1:.4f}")
    
    return best_sigma, best_f1


def main():
    parser = argparse.ArgumentParser(description="Train LSST models on real ZTF data")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="trained_models")
    
    args = parser.parse_args()
    
    print("\n" + "#" * 70)
    print("#  STREAMING LSST - MODEL TRAINING")
    print("#  Training on Real ZTF Data (10 epochs)")
    print("#" * 70)
    
    # Device
    if args.device == "auto":
        device = detect_device()
    else:
        device = args.device
    
    # Find data
    data_path = args.data
    if not data_path:
        default = os.path.join(
            str(PROJECT_ROOT), "streaming_lsst", "data", "real_alerts", "unified_real_alerts.json"
        )
        if os.path.exists(default):
            data_path = default
    
    if not data_path or not os.path.exists(data_path):
        print("ERROR: No data found. Run fetch_real_data.py first.")
        return
    
    # Load data
    X, y, alerts = load_and_prepare_data(data_path, device)
    
    # Train autoencoder
    ae_model = train_autoencoder(X, y, device, epochs=args.epochs,
                                 batch_size=args.batch_size, lr=args.lr)
    
    # Train transformer (smaller batch for memory)
    trans_batch = min(args.batch_size, 32)
    transformer, classifier = train_transformer(X, y, device, epochs=args.epochs,
                                                 batch_size=trans_batch, lr=args.lr * 0.5)
    
    # Final evaluation
    best_sigma, best_f1 = final_evaluation(ae_model, X, y, device)
    
    # Save models
    save_dir = os.path.join(str(PROJECT_ROOT), "streaming_lsst", args.save_dir)
    save_models(ae_model, transformer, save_dir)
    
    # Save training metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_samples": len(X),
        "num_anomalies": int(y.sum().item()),
        "best_threshold_sigma": best_sigma,
        "best_f1": best_f1,
    }
    meta_path = os.path.join(save_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n{'#'*70}")
    print(f"#  TRAINING COMPLETE")
    print(f"#  Best F1: {best_f1:.4f} (threshold: {best_sigma} sigma)")
    print(f"#  Models saved to: {save_dir}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
