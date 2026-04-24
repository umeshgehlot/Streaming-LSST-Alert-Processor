"""Streaming LSST Alert Processing Framework."""

from .processor import StreamingLSSTProcessor
from .pipeline.data_pipeline import StreamingPipeline
from .models.streaming_transformer import StreamingTransformer
from .models.online_autoencoder import OnlineAnomalyDetector
from .models.streaming_gnn import StreamingGNN, StreamingAlertGraph
from .simulator.alert_simulator import LSSTAlertSimulator, HighVariabilityAlertSimulator, BurstAlertSimulator
from .benchmarks.benchmark_suite import StreamingBenchmarkSuite

__version__ = '0.1.0'
__all__ = [
    'StreamingLSSTProcessor',
    'StreamingPipeline',
    'StreamingTransformer',
    'OnlineAnomalyDetector',
    'StreamingGNN',
    'StreamingAlertGraph',
    'LSSTAlertSimulator',
    'HighVariabilityAlertSimulator',
    'BurstAlertSimulator',
    'StreamingBenchmarkSuite',
]
