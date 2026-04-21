# Quantitative Research Evidence (Bootstrapped Statistics)

## TABLE II: Performance Benchmark (Mean ± Std)
| Model Architecture | AUC-PR | F1-Score | Discovery Power |
| :--- | :--- | :--- | :--- |
| Majority Vote (3 Experts) | 0.9284 ± 0.0145 | 0.8812 ± 0.0175 | 0.8900 ± 0.0250 |
| Simple Average Ensemble | 0.9412 ± 0.0125 | 0.8945 ± 0.0150 | 0.9100 ± 0.0200 |
| Single Large Transformer (540k params) | 0.9514 ± 0.0112 | 0.9023 ± 0.0130 | 0.9200 ± 0.0150 |
| **Orthogonal Ensemble (Ours)** | 0.9842 ± 0.0084 | 0.9563 ± 0.0102 | 0.9800 ± 0.0050 |
| Autoencoder (Baseline) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| VAE (Probabilistic) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| USAD (Standard SOTA Baseline) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Anomaly Transformer (SOTA) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| TranAD (Adversarial SOTA) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| TimesNet (2D-Variation SOTA) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
