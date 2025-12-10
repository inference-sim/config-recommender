"""Tests for the BentoML output parser."""

import unittest
from config_recommender.parser import parse_bentoml_output


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


class TestParser(unittest.TestCase):
    """Test cases for BentoML output parser."""
    
    def test_parse_basic_config(self):
        """Test parsing of basic configuration."""
        result = parse_bentoml_output(
            SAMPLE_OUTPUT,
            model="Qwen/Qwen2.5-7B",
            gpu="H100",
            num_gpus=1,
            input_len=2000,
            output_len=256
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.config.model, "Qwen/Qwen2.5-7B")
        self.assertEqual(result.config.gpu, "H100")
        self.assertEqual(result.config.num_gpus, 1)
        self.assertEqual(result.config.precision, "bf16")
        self.assertEqual(result.config.input_len, 2000)
        self.assertEqual(result.config.output_len, 256)
    
    def test_parse_model_info(self):
        """Test parsing of model information."""
        result = parse_bentoml_output(
            SAMPLE_OUTPUT,
            model="Qwen/Qwen2.5-7B",
            gpu="H100"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.model_parameters, "7615283200.0B")
        self.assertEqual(result.model_layers, 28)
    
    def test_parse_latency_metrics(self):
        """Test parsing of latency metrics."""
        result = parse_bentoml_output(
            SAMPLE_OUTPUT,
            model="Qwen/Qwen2.5-7B",
            gpu="H100"
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.metrics)
        self.assertEqual(result.metrics.ttft_ms, 36.9)
        self.assertEqual(result.metrics.itl_ms, 2.2)
        self.assertEqual(result.metrics.e2e_s, 0.59)
    
    def test_parse_throughput_metrics(self):
        """Test parsing of throughput metrics."""
        result = parse_bentoml_output(
            SAMPLE_OUTPUT,
            model="Qwen/Qwen2.5-7B",
            gpu="H100"
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.metrics)
        self.assertEqual(result.metrics.output_tokens_per_s, 9411.1)
        self.assertEqual(result.metrics.input_tokens_per_s, 54260.3)
        self.assertEqual(result.metrics.requests_per_s, 15.61)
        self.assertEqual(result.metrics.bottleneck, "Memory")
    
    def test_parse_roofline_analysis(self):
        """Test parsing of roofline analysis."""
        result = parse_bentoml_output(
            SAMPLE_OUTPUT,
            model="Qwen/Qwen2.5-7B",
            gpu="H100"
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.metrics)
        self.assertEqual(result.metrics.hardware_ops_per_byte, 275.1)
        self.assertEqual(result.metrics.prefill_arithmetic_intensity, 42973.2)
        self.assertEqual(result.metrics.decode_arithmetic_intensity, 21.5)
        self.assertEqual(result.metrics.prefill_phase, "Compute Bound")
        self.assertEqual(result.metrics.decode_phase, "Memory Bound")
    
    def test_parse_concurrency_analysis(self):
        """Test parsing of concurrency analysis."""
        result = parse_bentoml_output(
            SAMPLE_OUTPUT,
            model="Qwen/Qwen2.5-7B",
            gpu="H100"
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.metrics)
        self.assertEqual(result.metrics.kv_cache_memory_limit, 479)
        self.assertEqual(result.metrics.prefill_compute_limit, 7)
        self.assertEqual(result.metrics.decode_capacity_limit, 11)
        self.assertEqual(result.metrics.theoretical_overall_limit, 7)
        self.assertEqual(result.metrics.empirical_optimal_concurrency, 16)


if __name__ == "__main__":
    unittest.main()
