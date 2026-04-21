import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict

class PLAsTiCCProcessor:
    """
    Expert-level processor for the Photometric LSST Astronomical Time-Series 
    Classification Challenge (PLAsTiCC) dataset.
    
    This class handles multi-band photometry, irregular sampling, and prepares 
    data for deep learning (PyTorch) unsupervised anomaly detection.
    """
    
    def __init__(self, seq_len: int = 256):
        self.seq_len = seq_len
        # Mapping of passbands: 0=u, 1=g, 2=r, 3=i, 4=z, 5=Y
        self.passbands = [0, 1, 2, 3, 4, 5]
        self.pb_names = ['u', 'g', 'r', 'i', 'z', 'Y']
        
    def load_data(self, meta_path: str, lc_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads PLAsTiCC metadata and light curve files.
        """
        print(f"Loading metadata from {meta_path}...")
        df_meta = pd.read_csv(meta_path)
        
        print(f"Loading light curves from {lc_path}...")
        # Note: In production, large LC files should be loaded in chunks or via Parquet.
        df_lc = pd.read_csv(lc_path)
        
        return df_meta, df_lc

    def preprocess_all(self, df_meta: pd.DataFrame, df_lc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Main pipeline: Pivot, Normalize, and Pad/Truncate.
        Returns: (X, dt, final_meta)
        """
        # 1. Filter light curves to only include those in metadata
        df_lc = df_lc[df_lc['object_id'].isin(df_meta['object_id'])]
        
        # 2. Group by object and passband to normalize flux
        # We use a robust scaler (median/IQR) for flux given the extreme range of transients
        df_lc['flux_norm'] = df_lc.groupby(['object_id', 'passband'])['flux'].transform(
            lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25) + 1e-6)
        )
        
        processed_data = []
        object_ids = df_meta['object_id'].tolist()
        
        print(f"Processing {len(object_ids)} light curves...")
        
        for obj_id in object_ids:
            obj_lc = df_lc[df_lc['object_id'] == obj_id].sort_values('mjd')
            
            if obj_lc.empty:
                continue
                
            # Create features: [flux_u, flux_g, flux_r, flux_i, flux_z, flux_Y, dt]
            # Since sampling is irregular across bands, we pivot to a 'global' time axis
            # and interpolate missing bands or use zero-filling for unsupervised reconstruction.
            
            # Pivot to get passbands as columns
            pivoted = obj_lc.pivot_table(index='mjd', columns='passband', values='flux_norm')
            
            # Ensure all passbands exist
            for pb in self.passbands:
                if pb not in pivoted.columns:
                    pivoted[pb] = 0.0
            
            pivoted = pivoted[self.passbands] # Reorder
            pivoted = pivoted.fillna(0.0) # Zero fill for unobserved bands at that MJD
            
            # Calculate time deltas
            mjd_indices = pivoted.index.values
            dt = np.diff(mjd_indices, prepend=mjd_indices[0])
            
            # Pad or Truncate
            features = pivoted.values # (N, 6)
            
            if len(features) > self.seq_len:
                features = features[:self.seq_len]
                dt_seq = dt[:self.seq_len]
            else:
                pad_width = self.seq_len - len(features)
                features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
                dt_seq = np.pad(dt, (0, pad_width), mode='constant')
            
            processed_data.append({
                'object_id': obj_id,
                'features': features,
                'dt': dt_seq
            })

        X = np.array([d['features'] for d in processed_data])
        DT = np.array([d['dt'] for d in processed_data])
        
        return X, DT, df_meta

    def isolate_anomalous_pool(self, df_meta: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Isolates rare classes to serve as a 'Blind Discovery' test set.
        Target mapping based on PLAsTiCC classification.
        """
        anomaly_map = {
            'TDE': 15,          # Tidal Disruption Events
            'AGN': 88,          # Active Galactic Nuclei
            'SLSN-I': 95,       # Superluminous Supernovae Type I
            'Kilonova': 64,     # Kilonova (extremely rare)
            'SNIa-91bg': 67     # Peculiar SNIa
        }
        
        # 'Normal' pool would typically be SNIa (90), SNII (42), RRL (92), etc.
        normal_classes = [90, 42, 92, 65, 16, 53, 62]
        
        pools = {
            'normal': df_meta[df_meta['target'].isin(normal_classes)],
            'anomalies': df_meta[df_meta['target'].isin(list(anomaly_map.values()))]
        }
        
        for name, cls_id in anomaly_map.items():
            pools[name] = df_meta[df_meta['target'] == cls_id]
            
        print("\nDataset Composition:")
        for key, df in pools.items():
            print(f" - {key}: {len(df)} objects")
            
        return pools

# Example Usage Template
if __name__ == "__main__":
    processor = PLAsTiCCProcessor(seq_len=256)
    
    # Paths to be updated by user
    META_PATH = "data/plasticc_train_metadata.csv"
    LC_PATH = "data/plasticc_train_lightcurves.csv"
    
    if os.path.exists(META_PATH) and os.path.exists(LC_PATH):
        meta, lc = processor.load_data(META_PATH, LC_PATH)
        
        # Isolate anomalies for zero-positive evaluation
        pools = processor.isolate_anomalous_pool(meta)
        
        # Preprocess normal data for training the unsupervised model
        X_train, dt_train, meta_train = processor.preprocess_all(pools['normal'], lc)
        
        print(f"\nFinal Training Shape: {X_train.shape}") # (N, 256, 6)
    else:
        print("Note: PLAsTiCC files not found at specified paths. Use this script once data is available.")
