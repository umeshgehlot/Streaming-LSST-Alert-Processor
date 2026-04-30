# Usage Guide - Streaming LSST Alert Processor

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Component Examples](#component-examples)
4. [Benchmarking](#benchmarking)
5. [Advanced Configuration](#advanced-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone/navigate to project
cd streaming_lsst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
python -c "import streaming_lsst; print('✓ Installation successful')"

# Run validation tests
python tests.py

# View configuration
python -m streaming_lsst.config balanced
```

## Basic Usage

### Minimal Example

```python
from streaming_lsst import StreamingLSSTProcessor, LSSTAlertSimulator

# Create processor
processor = StreamingLSSTProcessor()

# Create simulator
simulator = LSSTAlertSimulator(anomaly_rate=0.05)

# Process alerts
for alert, timestamp in simulator.stream_alerts(duration_sec=10):
    result = processor.process_alert(alert)
    
    print(f"Alert {result['alert_id']}: "
          f"Latency {result['latency_ms']:.2f}ms, "
          f"Anomaly Score {result['anomaly_score']:.4f}")
```

### Full-Featured Example

```python
from streaming_lsst import (
    StreamingLSSTProcessor,
    LSSTAlertSimulator,
    StreamingBenchmarkSuite
)
import torch

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = StreamingLSSTProcessor(
    device=device,
    enable_transformer=True,
    enable_autoencoder=True,
    enable_gnn=True
)

simulator = LSSTAlertSimulator(seed=42, anomaly_rate=0.1)
benchmark = StreamingBenchmarkSuite("Custom Benchmark")

# Run
benchmark.start()
benchmark.start_phase("processing")

anomalies = []
for alert, ts in simulator.stream_alerts(duration_sec=60):
    result = processor.process_alert(alert)
    
    # Record metrics
    benchmark.record_latency(ts, ts + result['latency_ms']/1000)
    benchmark.record_throughput(ts)
    
    if result['is_anomaly']:
        anomalies.append({
            'id': result['alert_id'],
            'score': result['anomaly_score'],
            'timestamp': ts
        })

benchmark.end_phase()
benchmark.end()

# Report
benchmark.print_report(verbose=True)
benchmark.save_report('results.json')

# Analysis
print(f"\nAnomalies Detected: {len(anomalies)}")
for anom in anomalies[:5]:
    print(f"  - Alert {anom['id']}: score {anom['score']:.4f}")
```

## Component Examples

### 1. Using Streaming Transformer Alone

```python
from streaming_lsst.models import StreamingTransformer
import torch

# Create model
transformer = StreamingTransformer(
    input_dim=16,
    d_model=64,
    n_layers=2,
    n_heads=4,
    output_dim=32
)

# Process stream
for batch in alert_batches:
    # batch shape: [batch_size, 16]
    embeddings = transformer(batch)  # [batch_size, 32]
    
    # Use embeddings downstream
    predictions = classifier(embeddings)
```

### 2. Using Online Autoencoder Alone

```python
from streaming_lsst.models import OnlineAnomalyDetector
import torch

# Create detector
detector = OnlineAnomalyDetector(
    input_dim=16,
    latent_dim=8,
    hidden_dim=32
)

# Process single alerts with online learning
for alert_features in stream:
    result = detector.process_alert(
        alert_features,
        learn_rate=0.01  # Online learning
    )
    
    if result['is_anomaly']:
        print(f"Anomaly detected: {result['anomaly_score']:.4f}")
    
    # Latent representation for clustering
    latent = result['latent']
    cluster_id = kmeans_predictor(latent)
```

### 3. Using Streaming GNN Alone

```python
from streaming_lsst.models import StreamingGNN, StreamingAlertGraph

# Create GNN
gnn = StreamingGNN(
    in_dim=16,
    hidden_dim=32,
    out_dim=32,
    n_layers=2
)

# Manage alert graph
graph = StreamingAlertGraph(gnn, node_dim=16, max_nodes=500)

# Add alerts and relationships
for alert1, alert2 in alert_pairs:
    emb1 = graph.add_alert(alert1.id, alert1.features)
    emb2 = graph.add_alert(alert2.id, alert2.features)
    
    # Add connection if similar
    if cosine_similarity(alert1, alert2) > 0.8:
        graph.add_relation(alert1.id, alert2.id, weight=0.8)

# Get all embeddings with relationship info
embeddings = graph.get_embeddings()  # [num_nodes, 32]
```

### 4. Using Data Pipeline Alone

```python
from streaming_lsst.pipeline import StreamingPipeline

# Create pipeline
pipeline = StreamingPipeline(
    feature_dim=16,
    buffer_size=100,
    normalize=True
)

# Process alerts
for alert in alert_stream:
    features, alert_id, latency = pipeline.process_alert(alert)
    
    # Get batch for model
    batch, ids = pipeline.get_batch(batch_size=32)
    model_output = model(batch)
    
    # Monitor performance
    metrics = pipeline.get_metrics()
    print(f"Throughput: {metrics['throughput_alerts_per_sec']:.1f} alerts/sec")
```

## Benchmarking

### Latency Benchmark

Measure per-alert processing time.

```bash
python -m streaming_lsst.benchmarks.run_benchmarks \
    --benchmark-latency \
    --num-alerts 1000 \
    --output-dir ./results
```

**What it measures:**
- Min/max/P95/P99 latencies
- Latency distribution
- Per-component breakdown

### Throughput Benchmark

Measure sustained throughput at various rates.

```bash
python -m streaming_lsst.benchmarks.run_benchmarks \
    --benchmark-throughput \
    --duration-sec 60 \
    --target-rate-hz 100 \
    --output-dir ./results
```

**What it measures:**
- Sustained alerts/second
- Latency under load
- Throughput scaling

### Anomaly Detection Benchmark

Evaluate anomaly detection performance.

```bash
python -m streaming_lsst.benchmarks.run_benchmarks \
    --benchmark-anomaly \
    --simulator high_variability \
    --num-alerts 1000 \
    --output-dir ./results
```

**What it measures:**
- Precision, Recall, F1
- True/False positives/negatives
- Detection latency

### Memory Efficiency Benchmark

Monitor memory usage under streaming load.

```bash
python -m streaming_lsst.benchmarks.run_benchmarks \
    --benchmark-memory \
    --duration-sec 60 \
    --output-dir ./results
```

**What it measures:**
- Peak memory
- Average memory
- Memory growth rate

### Full Benchmark Suite

Run all benchmarks with comprehensive reporting.

```bash
python -m streaming_lsst.benchmarks.run_benchmarks \
    --simulator burst \
    --benchmark-latency \
    --benchmark-throughput \
    --benchmark-anomaly \
    --benchmark-memory \
    --num-alerts 1000 \
    --duration-sec 60 \
    --output-dir ./benchmark_results \
    --use-cuda
```

## Advanced Configuration

### Using Configuration Presets

```python
from streaming_lsst.config import get_config, print_config

# View preset
print_config('balanced')

# Use minimal preset (fastest)
config = get_config('minimal')
processor = StreamingLSSTProcessor(
    device='cpu',
    enable_transformer=True,
    enable_autoencoder=True,
    enable_gnn=True
)
# Automatically uses lightweight models

# Use maximum preset (most accurate)
config = get_config('maximum')
processor = StreamingLSSTProcessor(
    device='cuda',
    enable_transformer=True,
    enable_autoencoder=True,
    enable_gnn=True
)
```

### Custom Configuration

```python
from streaming_lsst.models import StreamingTransformer, StreamingGNN
from streaming_lsst.pipeline import StreamingPipeline

# Custom pipeline
pipeline = StreamingPipeline(
    feature_dim=32,        # More features
    buffer_size=256,       # Larger buffer
    batch_size=64,         # Larger batches
    normalize=True
)

# Custom transformer
transformer = StreamingTransformer(
    input_dim=32,
    d_model=128,           # Larger
    n_layers=4,            # Deeper
    n_heads=8,
    window_size=64,        # Larger window
    output_dim=64
)

# Custom GNN
gnn = StreamingGNN(
    in_dim=32,
    hidden_dim=64,
    out_dim=64,
    n_layers=3,
    max_nodes=1000
)

processor = StreamingLSSTProcessor()
processor.pipeline = pipeline
processor.transformer = transformer
processor.graph_processor.gnn = gnn
```

## Performance Optimization

### For CPU

```python
# Minimal model
config = get_config('minimal')

# Smaller batches
StreamingPipeline(buffer_size=50, batch_size=8)

# Reduced precision (if available)
transformer = transformer.half()  # float16

# Example
processor = StreamingLSSTProcessor(device='cpu')

# Run
for alert in stream:
    result = processor.process_alert(alert)
    # Typical latency: 5-10ms
```

### For GPU

```python
# Larger models
config = get_config('maximum')

# Larger batches
StreamingPipeline(buffer_size=200, batch_size=128)

# Pin memory
torch.cuda.empty_cache()

# Example
processor = StreamingLSSTProcessor(device='cuda')

# Run
for alert in stream:
    result = processor.process_alert(alert)
    # Typical latency: 0.5-2ms
```

### For Mixed Latency/Throughput

```python
# Balanced configuration
config = get_config('balanced')

# Medium batch size
StreamingPipeline(buffer_size=100, batch_size=32)

# Benchmark to find sweet spot
results = []
for rate in [50, 100, 200, 500]:
    benchmark = run_throughput_benchmark(rate)
    results.append({
        'rate': rate,
        'latency_p95': benchmark['latency_p95_ms'],
        'throughput': benchmark['throughput_hz']
    })

# Choose configuration based on requirements
```

## Troubleshooting

### Issue: Out of Memory

**Solution:**
```python
# Reduce model sizes
config = get_config('minimal')

# Reduce graph size
processor.graph_processor.max_nodes = 100

# Reduce buffer size
pipeline = StreamingPipeline(buffer_size=50, batch_size=8)

# Clear cache periodically
torch.cuda.empty_cache()
processor.reset_models()
```

### Issue: High Latency

**Solution:**
```python
# Check bottleneck
benchmark.print_report(verbose=True)

# Reduce model complexity
streaming_transformer = StreamingTransformer(
    input_dim=16,
    d_model=32,  # Smaller
    n_layers=1,  # Fewer layers
    n_heads=2    # Fewer heads
)

# Disable components not needed
processor = StreamingLSSTProcessor(
    enable_gnn=False  # Skip if not using graph
)
```

### Issue: Low Throughput

**Solution:**
```python
# Enable GPU
processor = StreamingLSSTProcessor(device='cuda')

# Batch processing
batch, ids = pipeline.get_batch(batch_size=64)
results = process_batch(batch)

# Parallelize
import multiprocessing
with multiprocessing.Pool() as pool:
    results = pool.map(processor.process_alert, alerts)
```

### Issue: Inaccurate Anomaly Detection

**Solution:**
```python
# Increase model complexity
detector = OnlineAnomalyDetector(
    input_dim=16,
    latent_dim=16,   # Larger
    hidden_dim=64    # Larger
)

# Adjust threshold
error, is_anomaly = detector.autoencoder.anomaly_score(
    features,
    threshold_sigma=2.0  # Stricter
)

# More training samples
for _ in range(1000):
    alert = simulator.generate_alert()
    detector.process_alert(alert, learn_rate=0.05)
```

### Issue: CUDA Not Available

**Solution:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA support
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Fall back to CPU
device = 'cpu'  # Explicitly specify
```

## Performance Expectations

### Latency
- Transformer only: 1-5 ms
- Autoencoder only: 0.5-2 ms
- GNN only: 1-3 ms
- All components: 3-10 ms

### Throughput
- CPU (single): 100-200 alerts/sec
- CPU (multi): 200-500 alerts/sec
- GPU: 1000+ alerts/sec

### Memory
- Model weights: ~100 KB
- Runtime state: 1-2 MB
- Peak usage: 2-3 MB

## Next Steps

1. **Read** [README.md](README.md) for complete documentation
2. **Run** `python examples.py` to see working examples
3. **Benchmark** `python -m streaming_lsst.benchmarks.run_benchmarks`
4. **Customize** Edit `config.py` for your use case
5. **Integrate** Use components in your application

## Support

- **Validation**: Run `python tests.py`
- **Examples**: Run `python examples.py`
- **Configuration**: Run `python -m streaming_lsst.config balanced`
- **Benchmarks**: See results in `./benchmark_results/benchmark_report.json`

---

**Last Updated**: 2024-04-24
**Version**: 0.1.0
