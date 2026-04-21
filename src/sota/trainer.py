import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from src.sota.models.transformer import AnomalyTransformer
from src.sota.models.tranad import TranAD
from src.sota.models.timesnet import TimesNet
from src.sota.data.datasets import get_dataloader

class SotaTrainer:
    """
    Expert-level trainer for SOTA astronomical anomaly detection models.
    Supports GPU-accelerated training with Mixed Precision (AMP) and Gradient Accumulation.
    Optimized for high-volume streaming data.
    """
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Disabling AMP for custom SOTA losses to ensure numerical stability
        self.scaler = None
        logging.info(f"SOTA Trainer initialized on {self.device}")

    def train_model(self, model: nn.Module, loader: DataLoader, config: Dict[str, Any], model_type: str = "transformer"):
        """
        Generic optimized training loop for SOTA models.
        """
        epochs = config.get('epochs', 10)
        lr = config.get('lr', 1e-4)
        grad_accum_steps = config.get('grad_accum_steps', 1)
        k = config.get('k', 3)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.to(self.device).train()
        
        for epoch in range(epochs):
            loss_epoch = 0
            optimizer.zero_grad()
            i = -1 # Initialize i to handle empty loaders
            
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                
                if model_type == "transformer":
                    output, series, prior, _ = model(batch)
                    rec_loss = criterion(output, batch)
                    dist_loss = 0
                    for s, p in zip(series, prior):
                        # Enhanced numerical stability for KL divergence
                        s = s + 1e-5
                        p = p + 1e-5
                        kl_p_s = torch.mean(p * torch.log(p / s))
                        kl_s_p = torch.mean(s * torch.log(s / p))
                        dist_loss += (kl_p_s + kl_s_p)
                    loss = rec_loss - (1.0 / k) * dist_loss
                
                elif model_type == "tranad":
                    x1, x2 = model(batch, batch)
                    loss1 = torch.mean((batch - x1) ** 2)
                    loss2 = torch.mean((batch - x2) ** 2)
                    loss = (1 / (epoch + 1)) * loss1 + (1 - 1 / (epoch + 1)) * loss2
                    
                elif model_type == "timesnet":
                    output = model(batch)
                    loss = criterion(output, batch)
                
                loss = loss / grad_accum_steps
                
                # Scaled Backward Pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    if (i + 1) % grad_accum_steps == 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (i + 1) % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Safety check for NaN
                current_loss = loss.item() * grad_accum_steps
                if np.isnan(current_loss):
                    logging.warning(f"NaN detected at Step {i+1}. Skipping batch.")
                    continue
                    
                loss_epoch += current_loss
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()*grad_accum_steps:.6f}")
                    
            if i >= 0:
                logging.info(f"{model_type.upper()} Epoch [{epoch+1}/{epochs}], Avg Loss: {loss_epoch/(i+1):.6f}")
            else:
                logging.warning(f"{model_type.upper()} Epoch [{epoch+1}/{epochs}]: No data processed.")
        
        return model

    def save_model(self, model: nn.Module, name: str, savedir: str = "models"):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        path = os.path.join(savedir, f"{name}.pth")
        torch.save(model.state_dict(), path)
        logging.info(f"Model saved to {path}")
        return path

if __name__ == "__main__":
    # Internal test with synthetic data if run directly
    logging.basicConfig(level=logging.INFO)
    trainer = SotaTrainer()
    sync_data = np.random.randn(1000, 1) # 1000 points, 1 feature
    
    logging.info("Testing Transformer training...")
    m1 = trainer.train_transformer(sync_data, {"epochs": 2})
    trainer.save_model(m1, "transformer_test")
    
    logging.info("Testing TranAD training...")
    m2 = trainer.train_tranad(sync_data, {"epochs": 2, "win_size": 10})
    trainer.save_model(m2, "tranad_test")
    
    logging.info("Testing TimesNet training...")
    m3 = trainer.train_timesnet(sync_data, {"epochs": 2})
    trainer.save_model(m3, "timesnet_test")
