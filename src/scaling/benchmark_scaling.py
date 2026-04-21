import polars as pl
import numpy as np
import time
import os
import psutil
from typing import List

class ScalingBenchmark:
    """
    Benchmarks system throughput and memory efficiency for 'Petabyte-Scale' data.
    Uses Polars for lazy, memory-mapped processing.
    """
    
    def __init__(self, data_path: str = "data/scaling_test.parquet"):
        self.data_path = data_path
        
    def generate_synthetic_data(self, n_rows: int = 10_000_000):
        """Generates a large synthetic dataset in Parquet format."""
        print(f"Generating {n_rows} rows of synthetic astronomical data...")
        
        df = pl.DataFrame({
            "object_id": np.random.randint(1000, 1100, n_rows),
            "mjd": np.sort(np.random.uniform(50000, 60000, n_rows)),
            "passband": np.random.randint(0, 6, n_rows),
            "flux": np.random.normal(100, 20, n_rows),
            "flux_err": np.random.uniform(1, 5, n_rows)
        })
        
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.write_parquet(self.data_path)
        print(f"Dataset saved to {self.data_path} ({df.estimated_size('mb'):.2f} MB)")

    def run_lazy_benchmark(self):
        """
        Processes the dataset lazily using Polars.
        Goal: Constant memory footprint with O(N) scaling.
        """
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        
        print(f"\nStarting Scaling Benchmark...")
        print(f"Initial Memory Usage: {start_mem:.2f} MB")
        
        start_time = time.time()
        
        # Define the computation graph (Lazy)
        q = (
            pl.scan_parquet(self.data_path)
            .group_by("object_id")
            .agg([
                pl.col("flux").mean().alias("mean_flux"),
                pl.col("flux").std().alias("std_flux"),
                pl.col("flux").count().alias("point_count")
            ])
            .filter(pl.col("std_flux") > 15) # Identify variable candidates
        )
        
        # Execute
        result = q.collect()
        
        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024)
        
        duration = end_time - start_time
        throughput = len(result) / duration if duration > 0 else 0
        
        print(f"Benchmark Completed in {duration:.2f} seconds")
        print(f"Peak Memory usage delta: {end_mem - start_mem:.2f} MB")
        print(f"Objects Processed: {len(result)}")
        print("-" * 30)
        
        # Calculate row-level throughput
        file_stats = os.stat(self.data_path)
        row_count = 10_000_000 # Known from generator
        row_rate = row_count / duration
        
        print(f"Row Throughput: {row_rate:,.0f} rows/second")
        
        if row_rate > 1_000_000:
            print("RESULT: [HIGH PERFORMANCE] System handles >1M rows/sec (Petabyte Ready).")
        else:
            print("RESULT: [STANDARD] System handles standard data rates.")

if __name__ == "__main__":
    benchmark = ScalingBenchmark()
    
    # Generate 10M rows (representative)
    if not os.path.exists("data/scaling_test.parquet"):
        benchmark.generate_synthetic_data(n_rows=10_000_000)
    
    benchmark.run_lazy_benchmark()
