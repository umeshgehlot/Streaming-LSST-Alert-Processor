import time
import torch
import numpy as np
import pandas as pd
from alerce.core import Alerce
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import logging

# Configure logging for benchmarking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlerceIngestor:
    """
    High-throughput ALeRCE alert ingestor for real-time transient discovery.
    Demonstrates system capability to handle LSST-scale alert volumes.
    """
    
    def __init__(self, window_size: int = 64, stride: int = 5):
        self.client = Alerce()
        self.window_size = window_size
        self.stride = stride
        # ZTF Passbands: 1 (g), 2 (r)
        self.pb_map = {1: 0, 2: 1}
        
    def fetch_single_lc(self, oid: str) -> Dict:
        """Fetches a single light curve detections."""
        try:
            detections = self.client.query_detections(oid, format="pandas")
            return {'oid': oid, 'data': detections}
        except Exception as e:
            return {'oid': oid, 'error': str(e)}

    def get_latest_alerts(self, count: int = 1000) -> List[str]:
        """Queries ALeRCE for the latest transient OIDs."""
        logger.info(f"Querying ALeRCE for latest {count} transient objects...")
        # Filtering for transients (often using high-prob classifications or just recent)
        objects = self.client.query_objects(
            survey="ztf",
            order_by="firstmjd",
            order_mode="DESC",
            page_size=count,
            format="pandas"
        )
        return objects['oid'].tolist()

    def process_to_tensors(self, oid: str, df: pd.DataFrame) -> torch.Tensor:
        """
        Converts raw photometry to sliding window tensors.
        Output Shape: (num_windows, 2, window_size) - [g, r] bands
        """
        # Sort by MJD
        df = df.sort_values('mjd')
        
        # Pivot to bands
        pivoted = df.pivot_table(index='mjd', columns='fid', values='magpsf')
        
        # Ensure g and r bands exist
        for fid in [1, 2]:
            if fid not in pivoted.columns:
                pivoted[fid] = np.nan
        
        pivoted = pivoted[[1, 2]]
        # Fill missing values with median for a stable base in anomaly detection
        pivoted = pivoted.ffill().bfill().fillna(pivoted.median()).fillna(0)
        
        data = pivoted.values.T # (2, N)
        
        # Create sliding windows
        windows = []
        n_points = data.shape[1]
        
        if n_points < self.window_size:
            # Pad if too short
            pad_width = self.window_size - n_points
            window = np.pad(data, ((0, 0), (0, pad_width)), mode='edge')
            windows.append(window)
        else:
            for start in range(0, n_points - self.window_size + 1, self.stride):
                end = start + self.window_size
                windows.append(data[:, start:end])
                
        return torch.tensor(np.array(windows), dtype=torch.float32)

    def run_ingestion_benchmark(self, target_count: int = 1000, max_workers: int = 20):
        """
        Runs the benchmark to prove throughput scaling.
        """
        oids = self.get_latest_alerts(count=target_count)
        
        start_time = time.time()
        processed_count = 0
        total_windows = 0
        
        logger.info(f"Starting concurrent ingestion with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_oid = {executor.submit(self.fetch_single_lc, oid): oid for oid in oids}
            
            for future in as_completed(future_to_oid):
                res = future.result()
                if 'error' not in res and not res['data'].empty:
                    tensors = self.process_to_tensors(res['oid'], res['data'])
                    total_windows += tensors.shape[0]
                    processed_count += 1
                
                # Periodic reporting
                if processed_count % 100 == 0 and processed_count > 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    logger.info(f"Progress: {processed_count}/{target_count} | Rate: {rate:.2f} alerts/sec")

        end_time = time.time()
        total_elapsed = end_time - start_time
        avg_rate = processed_count / total_elapsed
        
        logger.info("\n" + "="*40)
        logger.info("INGESTION BENCHMARK RESULTS")
        logger.info("="*40)
        logger.info(f"Total Alerts Processed: {processed_count}")
        logger.info(f"Total Windows Generated: {total_windows}")
        logger.info(f"Total Time Taken: {total_elapsed:.2f} seconds")
        logger.info(f"Average Throughput: {avg_rate:.2f} alerts/second")
        logger.info(f"LSST Target Rate: ~115 alerts/second")
        
        if avg_rate >= 115:
            logger.info("RESULT: [SUCCESS] System exceeds LSST-scale real-time requirements.")
        else:
            logger.info("RESULT: [OPTIMIZE] System currently below LSST-scale (likely network bound).")
        logger.info("="*40)

if __name__ == "__main__":
    # Test with 10,000 alerts as requested
    ingestor = AlerceIngestor(window_size=64, stride=5)
    ingestor.run_ingestion_benchmark(target_count=10000, max_workers=25)
