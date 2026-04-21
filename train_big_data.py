import os
import logging
import torch
from src.sota.models.transformer import AnomalyTransformer
from src.sota.models.tranad import TranAD
from src.sota.models.timesnet import TimesNet
from src.sota.trainer import SotaTrainer
from src.sota.data.streaming import get_streaming_dataloader

# Configuration for Big Data Training
CONFIG = {
    "transformer": {"epochs": 5, "win_size": 32, "batch_size": 512, "lr": 5e-5, "grad_accum_steps": 4},
    "tranad": {"epochs": 5, "win_size": 10, "batch_size": 512, "lr": 5e-4, "grad_accum_steps": 4},
    "timesnet": {"epochs": 5, "win_size": 32, "batch_size": 512, "lr": 5e-5, "grad_accum_steps": 4},
}

DATA_PATH = "data/uploads/1336ea5a-63b9-4b70-be46-380c2eaf22c8_nasa_fireball_20240420_20260420.csv"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- STARTING PETABYTE-SCALE (BIG DATA) SOTA TRAINING ---")
    
    if not os.path.exists(DATA_PATH):
        logging.error(f"Streaming data not found at {DATA_PATH}. Please provide a path to a large-scale CSV.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SotaTrainer(device=device)
    
    # In a real "Petabyte" scenario, we'd use multiple GPUs and a more complex partition strategy.
    # Here we demonstrate the streaming pipeline capability.
    
    # 1. Anomaly Transformer
    logging.info("Training Anomaly Transformer with Mixed Precision...")
    model_t = AnomalyTransformer(win_size=32, enc_in=1, c_out=1)
    loader_t = get_streaming_dataloader(DATA_PATH, batch_size=CONFIG["transformer"]["batch_size"], window_size=32)
    trainer.train_model(model_t, loader_t, CONFIG["transformer"], model_type="transformer")
    trainer.save_model(model_t, "transformer_big_data")

    # 2. TranAD
    logging.info("Training TranAD with Mixed Precision...")
    model_tr = TranAD(feats=1, window=10)
    loader_tr = get_streaming_dataloader(DATA_PATH, batch_size=CONFIG["tranad"]["batch_size"], window_size=10)
    trainer.train_model(model_tr, loader_tr, CONFIG["tranad"], model_type="tranad")
    trainer.save_model(model_tr, "tranad_big_data")

    # 3. TimesNet
    logging.info("Training TimesNet with Mixed Precision...")
    model_ti = TimesNet(enc_in=1, c_out=1, seq_len=32)
    loader_ti = get_streaming_dataloader(DATA_PATH, batch_size=CONFIG["timesnet"]["batch_size"], window_size=32)
    trainer.train_model(model_ti, loader_ti, CONFIG["timesnet"], model_type="timesnet")
    trainer.save_model(model_ti, "timesnet_big_data")

    logging.info("Petabyte-scale training run complete.")

if __name__ == "__main__":
    main()
