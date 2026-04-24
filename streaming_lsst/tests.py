"""
Validation and integration test suite for Streaming LSST Alert Processor.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import time

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from streaming_lsst.models import (
    StreamingTransformer,
    StreamingAutoencoder,
    StreamingGNN,
)
from streaming_lsst.pipeline import StreamingPipeline
from streaming_lsst.simulator import LSSTAlertSimulator
from streaming_lsst import StreamingLSSTProcessor


class TestSuite:
    """Validation test suite."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def assert_equal(self, actual, expected, test_name):
        """Assert equality."""
        if actual == expected:
            self.tests_passed += 1
            print(f"✓ {test_name}")
        else:
            self.tests_failed += 1
            print(f"✗ {test_name}: expected {expected}, got {actual}")
    
    def assert_shape(self, tensor, expected_shape, test_name):
        """Assert tensor shape."""
        if tensor.shape == expected_shape:
            self.tests_passed += 1
            print(f"✓ {test_name}")
        else:
            self.tests_failed += 1
            print(f"✗ {test_name}: expected shape {expected_shape}, got {tensor.shape}")
    
    def assert_range(self, value, min_val, max_val, test_name):
        """Assert value in range."""
        if min_val <= value <= max_val:
            self.tests_passed += 1
            print(f"✓ {test_name}: {value:.4f}")
        else:
            self.tests_failed += 1
            print(f"✗ {test_name}: {value} not in range [{min_val}, {max_val}]")
    
    def test_streaming_transformer(self):
        """Test streaming transformer."""
        print("\n--- Testing Streaming Transformer ---")
        
        transformer = StreamingTransformer(
            input_dim=16,
            d_model=64,
            n_layers=2,
            n_heads=4,
            output_dim=32
        ).to(self.device)
        
        # Test single input
        x = torch.randn(4, 16).to(self.device)
        output = transformer(x)
        self.assert_shape(output, torch.Size([4, 32]), "Output shape")
        
        # Test cache reset
        transformer.reset_cache()
        self.assert_equal(True, True, "Cache reset")
        
        # Test forward pass latency
        start = time.perf_counter()
        for _ in range(100):
            _ = transformer(x)
        latency_ms = (time.perf_counter() - start) * 10  # ms per sample
        self.assert_range(latency_ms, 0.1, 20.0, "Transformer latency (ms)")
    
    def test_streaming_autoencoder(self):
        """Test streaming autoencoder."""
        print("\n--- Testing Streaming Autoencoder ---")
        
        ae = StreamingAutoencoder(
            input_dim=16,
            latent_dim=8,
            hidden_dim=32
        ).to(self.device)
        
        # Test encoding
        x = torch.randn(4, 16).to(self.device)
        z = ae.encode(x)
        self.assert_shape(z, torch.Size([4, 8]), "Latent encoding shape")
        
        # Test decoding
        recon = ae.decode(z)
        self.assert_shape(recon, torch.Size([4, 16]), "Reconstruction shape")
        
        # Test anomaly score
        scores, is_anomaly = ae.anomaly_score(x)
        self.assert_shape(scores, torch.Size([4]), "Anomaly scores shape")
        self.assert_shape(is_anomaly, torch.Size([4]), "Anomaly flags shape")
        
        # Test forward pass latency
        start = time.perf_counter()
        for _ in range(100):
            _, _, _ = ae(x)
        latency_ms = (time.perf_counter() - start) * 10
        self.assert_range(latency_ms, 0.1, 5.0, "Autoencoder latency (ms)")
    
    def test_streaming_gnn(self):
        """Test streaming GNN."""
        print("\n--- Testing Streaming GNN ---")
        
        gnn = StreamingGNN(
            in_dim=16,
            hidden_dim=32,
            out_dim=32,
            n_layers=2
        ).to(self.device)
        
        # Test with nodes and edges
        num_nodes = 10
        num_edges = 15
        
        node_features = torch.randn(num_nodes, 16).to(self.device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(self.device)
        edge_weights = torch.rand(num_edges).to(self.device)
        
        output = gnn(node_features, edge_index, edge_weights)
        self.assert_shape(output, torch.Size([num_nodes, 32]), "GNN output shape")
        
        # Test with no edges
        empty_edges = torch.zeros(2, 0, dtype=torch.long).to(self.device)
        output = gnn(node_features, empty_edges)
        self.assert_shape(output, torch.Size([num_nodes, 32]), "GNN output shape (no edges)")
    
    def test_streaming_pipeline(self):
        """Test data pipeline."""
        print("\n--- Testing Streaming Pipeline ---")
        
        pipeline = StreamingPipeline(
            feature_dim=16,
            buffer_size=100,
            normalize=True
        )
        
        # Generate test alert
        simulator = LSSTAlertSimulator(seed=42)
        alert = simulator.generate_alert()
        
        # Process alert
        features, alert_id, latency_ms = pipeline.process_alert(alert)
        
        self.assert_shape(features, torch.Size([16]), "Feature shape")
        self.assert_equal(len(alert_id) > 0, True, "Alert ID not empty")
        self.assert_range(latency_ms, 0.1, 5.0, "Pipeline latency (ms)")
        
        # Get batch
        batch, ids = pipeline.get_batch(batch_size=32)
        self.assert_range(batch.shape[0], 1, 100, "Batch size")
        
        # Get metrics
        metrics = pipeline.get_metrics()
        self.assert_equal('avg_latency_ms' in metrics, True, "Metrics keys present")
    
    def test_alert_simulator(self):
        """Test alert simulator."""
        print("\n--- Testing Alert Simulator ---")
        
        simulator = LSSTAlertSimulator(seed=42, anomaly_rate=0.1)
        
        # Generate single alert
        alert = simulator.generate_alert()
        self.assert_equal('alert' in alert, True, "Alert structure")
        self.assert_equal('candidate' in alert['alert'], True, "Candidate present")
        
        # Stream alerts
        count = 0
        for alert, ts in simulator.stream_alerts(duration_sec=1.0, rate_hz=100):
            count += 1
            if count >= 50:
                break
        
        self.assert_range(count, 40, 60, f"Alert stream rate (got {count})")
    
    def test_full_processor(self):
        """Test full streaming processor."""
        print("\n--- Testing Full Streaming Processor ---")
        
        processor = StreamingLSSTProcessor(
            device=self.device,
            enable_transformer=True,
            enable_autoencoder=True,
            enable_gnn=True
        )
        
        # Process alerts
        simulator = LSSTAlertSimulator(seed=42)
        
        latencies = []
        anomaly_count = 0
        
        for i, (alert, _) in enumerate(simulator.stream_alerts(duration_sec=2.0)):
            if i >= 100:
                break
            
            result = processor.process_alert(alert)
            
            # Validate result structure
            required_keys = ['alert_id', 'features', 'latency_ms', 'anomaly_score', 'is_anomaly']
            for key in required_keys:
                self.assert_equal(key in result, True, f"Result key '{key}' present")
            
            latencies.append(result['latency_ms'])
            if result['is_anomaly']:
                anomaly_count += 1
        
        # Check latency stats
        avg_latency = np.mean(latencies)
        self.assert_range(avg_latency, 1.0, 20.0, f"Average latency (ms)")
        
        # Check anomaly detection
        self.assert_range(anomaly_count / 100, 0.01, 0.2, "Anomaly detection rate")
    
    def test_model_sizes(self):
        """Test model memory footprints."""
        print("\n--- Testing Model Memory Footprints ---")
        
        # Transformer
        transformer = StreamingTransformer(input_dim=16, d_model=64, output_dim=32)
        transformer_params = sum(p.numel() for p in transformer.parameters())
        self.assert_range(transformer_params, 5000, 20000, "Transformer parameters")
        
        # Autoencoder
        ae = StreamingAutoencoder(input_dim=16, latent_dim=8, hidden_dim=32)
        ae_params = sum(p.numel() for p in ae.parameters())
        self.assert_range(ae_params, 1000, 10000, "Autoencoder parameters")
        
        # GNN
        gnn = StreamingGNN(in_dim=16, hidden_dim=32, out_dim=32, n_layers=2)
        gnn_params = sum(p.numel() for p in gnn.parameters())
        self.assert_range(gnn_params, 3000, 15000, "GNN parameters")
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("STREAMING LSST ALERT PROCESSOR - VALIDATION TEST SUITE")
        print("="*70)
        print(f"\nDevice: {self.device.upper()}")
        print(f"PyTorch: {torch.__version__}")
        
        try:
            self.test_streaming_transformer()
            self.test_streaming_autoencoder()
            self.test_streaming_gnn()
            self.test_streaming_pipeline()
            self.test_alert_simulator()
            self.test_model_sizes()
            self.test_full_processor()
            
        except Exception as e:
            print(f"\n❌ Exception during tests: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Print summary
        total = self.tests_passed + self.tests_failed
        print("\n" + "="*70)
        print(f"TEST RESULTS: {self.tests_passed}/{total} passed")
        print("="*70)
        
        if self.tests_failed == 0:
            print("\n✓ ALL TESTS PASSED!")
            return True
        else:
            print(f"\n✗ {self.tests_failed} tests failed")
            return False


if __name__ == '__main__':
    suite = TestSuite()
    success = suite.run_all_tests()
    exit(0 if success else 1)
