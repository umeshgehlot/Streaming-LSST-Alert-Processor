import torch
from torch.utils.data import IterableDataset, DataLoader
import polars as pl
import numpy as np
import os
import logging
from typing import Iterator

class StreamingAstronomicalDataset(IterableDataset):
    """
    High-performance streaming dataset for 'Petabyte-scale' astronomical data.
    Uses Polars read_csv_batched for robust out-of-core data streaming.
    """
    def __init__(self, file_path: str, chunk_size: int = 100000, window_size: int = 32, feature_col: str = 'flux'):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.feature_col = feature_col
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Streams windows from the file chunk by chunk.
        """
        try:
            # Use read_csv_batched for a more stable streaming experience on Windows
            reader = pl.read_csv_batched(self.file_path, batch_size=self.chunk_size)
            
            batches = reader.next_batches(100) # Get a set of batches
            while batches:
                for chunk in batches:
                    if len(chunk) == 0:
                        continue
                        
                    # Extract the feature column as a numpy array
                    data = chunk[self.feature_col].to_numpy().astype(np.float32)
                    
                    # Zero-mean normalization per chunk
                    data = (data - data.mean()) / (data.std() + 1e-9)
                    
                    # Generate sliding windows
                    for i in range(0, len(data) - self.window_size + 1):
                        yield torch.from_numpy(data[i : i + self.window_size]).unsqueeze(-1)
                
                batches = reader.next_batches(100)
                    
        except Exception as e:
            logging.error(f"Streaming error: {e}")

def get_streaming_dataloader(file_path: str, batch_size: int = 256, window_size: int = 32):
    dataset = StreamingAstronomicalDataset(file_path, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
