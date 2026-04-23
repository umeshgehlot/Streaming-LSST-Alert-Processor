# Anomaly Discovery in Astronomical Time Series Data

This project implements an **Anomaly Transformer** for unsupervised anomaly detection in astronomical time series data.

## Features
- **GPU Support**: Optimized for CUDA execution.
- **Modern Transformer Architecture**: Includes experimental support for Rotary Embeddings (RoPE) and RMSNorm.
- **Multi-Dataset Support**: Compatible with SMD, MSL, SMAP, and PSM datasets.
- **Colab Ready**: Includes a dedicated notebook for running in the cloud with Google Drive.

## Installation

```bash
# Clone the repository
git clone https://github.com/umeshgehlot/-Anomaly-Discovery-in-Astronomical.git
cd -Anomaly-Discovery-in-Astronomical

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch numpy pandas scikit-learn
```

## Usage

### Training
To start training on the SMD dataset:
```bash
python main.py --mode train --dataset SMD --data_path dataset/SMD/SMD --batch_size 128
```

### Testing
To run inference:
```bash
python main.py --mode test --dataset SMD --data_path dataset/SMD/SMD --batch_size 128
```

## Running on Google Colab
1. Upload the project folder to your Google Drive.
2. Open `AnomalyTransformer_Colab.ipynb` in Google Colab.
3. Follow the instructions in the notebook to mount Drive and start training.

## License
MIT
