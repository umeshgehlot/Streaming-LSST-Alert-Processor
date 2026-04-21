import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class AnomalyDataset(Dataset):
    """
    General purpose sliding window dataset for time-series anomaly detection.
    """
    def __init__(self, data: np.ndarray, window_size: int = 32, stride: int = 1):
        """
        Args:
            data: Input array of shape (L, D)
            window_size: Size of the sliding window
            stride: Stride of the window
        """
        self.data = torch.from_numpy(data).float()
        self.window_size = window_size
        self.stride = stride
        
        # Pre-calculating window starts
        self.window_starts = np.arange(0, len(data) - window_size + 1, stride)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        window = self.data[start : start + self.window_size]
        return window

def get_dataloader(data: np.ndarray, batch_size: int = 32, window_size: int = 32, shuffle: bool = True):
    dataset = AnomalyDataset(data, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=(len(dataset) > batch_size))
