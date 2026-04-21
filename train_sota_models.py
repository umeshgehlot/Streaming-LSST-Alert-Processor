import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from src.sota.trainer import SotaTrainer

# Configuration
CONFIG = {
    "transformer": {"epochs": 20, "win_size": 32, "batch_size": 128, "lr": 1e-4},
    "tranad": {"epochs": 25, "win_size": 10, "batch_size": 128, "lr": 1e-3},
    "timesnet": {"epochs": 20, "win_size": 32, "batch_size": 128, "lr": 1e-4},
}

def load_dataset():
    """
    Loads the NASA fireball dataset or synthetic data if missing.
    """
    data_path = "data/uploads/1336ea5a-63b9-4b70-be46-380c2eaf22c8_nasa_fireball_20240420_20260420.csv"
    if os.path.exists(data_path):
        logging.info(f"Loading real data from {data_path}")
        df = pd.read_csv(data_path)
        # Use energy or velocity as the feature
        if 'energy' in df.columns:
            data = df['energy'].values.reshape(-1, 1)
        else:
            data = df.iloc[:, 1].values.reshape(-1, 1)
    else:
        logging.warning("No real data found. Using synthetic astronomical signals.")
        t = np.linspace(0, 100, 2000)
        data = (np.sin(t) + np.random.normal(0, 0.1, 2000)).reshape(-1, 1)
    
    # Simple normalization
    data = (data - np.mean(data)) / (np.std(data) + 1e-9)
    return data

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting SOTA Model Training on Local GPU...")

    data = load_dataset()
    trainer = SotaTrainer()
    
    # 1. Train Anomaly Transformer
    logging.info("--- Training Anomaly Transformer ---")
    transformer_model = trainer.train_transformer(data, CONFIG["transformer"])
    trainer.save_model(transformer_model, "anomaly_transformer")

    # 2. Train TranAD
    logging.info("--- Training TranAD ---")
    tranad_model = trainer.train_tranad(data, CONFIG["tranad"])
    trainer.save_model(tranad_model, "tranad")

    # 3. Train TimesNet
    logging.info("--- Training TimesNet ---")
    timesnet_model = trainer.train_timesnet(data, CONFIG["timesnet"])
    trainer.save_model(timesnet_model, "timesnet")

    logging.info("All SOTA models trained and saved to models/ directory.")

if __name__ == "__main__":
    main()
