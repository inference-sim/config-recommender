"""Integration tests for the recommender system."""

import unittest
from unittest.mock import patch, MagicMock
from config_recommender import GPURecommender


SAMPLE_OUTPUT = """
💡 Inferred precision from model config: bf16

=== Configuration ===
Model: Qwen/Qwen2.5-7B
GPU: 1x H100
Precision: bf16
Input/Output: 2000/256 tokens
Target: throughput

Fetching model configuration...
Model: 7615283200.0B parameters, 28 layers

=== Performance Analysis ===
Best Latency (concurrency=1):
  TTFT: 36.9 ms
  ITL: 2.2 ms
  E2E: 0.59 s

Best Throughput (concurrency=256):
  Output: 9411.1 tokens/s
  Input: 54260.3 tokens/s
  Requests: 15.61 req/s
  Bottleneck: Memory

=== Roofline Analysis ===
Hardware Ops/Byte Ratio: 275.1 ops/byte
Prefill Arithmetic Intensity: 42973.2 ops/byte
Decode Arithmetic Intensity: 21.5 ops/byte
Prefill Phase: Compute Bound
Decode Phase: Memory Bound

=== Concurrency Analysis ===
KV Cache Memory Limit: 479 concurrent requests
Prefill Compute Limit: 7 concurrent requests
Decode Capacity Limit: 11 concurrent requests
Theoretical Overall Limit: 7 concurrent requests
Empirical Optimal Concurrency: 16 concurrent requests
"""


class TestGPURecommender(unittest.TestCase):
    """Test cases for GPU recommender."""
    
    @patch('subprocess.run')
    def test_recommend_gpus_throughput(self, mock_run):
        """Test GPU recommendation based on throughput."""
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = SAMPLE_OUTPUT
        mock_run.return_value = mock_result
        
        # Create recommender and get recommendations
        recommender = GPURecommender()
        models = ["test-model"]
        gpus = ["H100", "L40"]
        
        recommendations = recommender.recommend_gpus(
            models=models,
            gpus=gpus,
            num_gpus=1,
            metric="throughput"
        )
        
        # Check results
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].model, "test-model")
        # Both would have same mock output, so first one wins
        self.assertIn(recommendations[0].recommended_gpu, ["H100", "L40"])
    
    @patch('subprocess.run')
    def test_recommend_gpus_latency(self, mock_run):
        """Test GPU recommendation based on latency."""
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = SAMPLE_OUTPUT
        mock_run.return_value = mock_result
        
        # Create recommender and get recommendations
        recommender = GPURecommender()
        models = ["test-model"]
        gpus = ["H100"]
        
        recommendations = recommender.recommend_gpus(
            models=models,
            gpus=gpus,
            num_gpus=1,
            metric="latency"
        )
        
        # Check results
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].model, "test-model")
        self.assertEqual(recommendations[0].recommended_gpu, "H100")


if __name__ == "__main__":
    unittest.main()
