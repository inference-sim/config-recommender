#!/usr/bin/env python3
"""
Demo script showing how to use the GPU recommender with mock data.

This script demonstrates the API without requiring actual llm-optimizer installation.
"""

from config_recommender.parser import parse_bentoml_output
from config_recommender.models import GPURecommendation


# Sample output from llm-optimizer for demonstration
SAMPLE_H200_OUTPUT = """
💡 Inferred precision from model config: bf16

=== Configuration ===
Model: Qwen/Qwen2.5-7B
GPU: 1x H200
Precision: bf16
Input/Output: 2000/256 tokens
Target: throughput

Fetching model configuration...
Model: 7615283200.0B parameters, 28 layers

=== Performance Analysis ===
Best Latency (concurrency=1):
  TTFT: 30.5 ms
  ITL: 1.8 ms
  E2E: 0.49 s

Best Throughput (concurrency=256):
  Output: 12500.0 tokens/s
  Input: 72000.0 tokens/s
  Requests: 20.5 req/s
  Bottleneck: Memory

=== Roofline Analysis ===
Hardware Ops/Byte Ratio: 350.0 ops/byte
Prefill Arithmetic Intensity: 50000.0 ops/byte
Decode Arithmetic Intensity: 25.0 ops/byte
Prefill Phase: Compute Bound
Decode Phase: Memory Bound

=== Concurrency Analysis ===
KV Cache Memory Limit: 600 concurrent requests
Prefill Compute Limit: 10 concurrent requests
Decode Capacity Limit: 15 concurrent requests
Theoretical Overall Limit: 10 concurrent requests
Empirical Optimal Concurrency: 20 concurrent requests
"""

SAMPLE_L40_OUTPUT = """
💡 Inferred precision from model config: bf16

=== Configuration ===
Model: Qwen/Qwen2.5-7B
GPU: 1x L40
Precision: bf16
Input/Output: 2000/256 tokens
Target: throughput

Fetching model configuration...
Model: 7615283200.0B parameters, 28 layers

=== Performance Analysis ===
Best Latency (concurrency=1):
  TTFT: 45.0 ms
  ITL: 3.5 ms
  E2E: 0.95 s

Best Throughput (concurrency=256):
  Output: 6500.0 tokens/s
  Input: 38000.0 tokens/s
  Requests: 10.8 req/s
  Bottleneck: Memory

=== Roofline Analysis ===
Hardware Ops/Byte Ratio: 180.0 ops/byte
Prefill Arithmetic Intensity: 30000.0 ops/byte
Decode Arithmetic Intensity: 15.0 ops/byte
Prefill Phase: Compute Bound
Decode Phase: Memory Bound

=== Concurrency Analysis ===
KV Cache Memory Limit: 300 concurrent requests
Prefill Compute Limit: 5 concurrent requests
Decode Capacity Limit: 8 concurrent requests
Theoretical Overall Limit: 5 concurrent requests
Empirical Optimal Concurrency: 10 concurrent requests
"""


def demo():
    """Run demo with mock data."""
    print("=" * 80)
    print("GPU RECOMMENDATION ENGINE - DEMO")
    print("=" * 80)
    print("\nThis demo shows how the recommendation engine works using sample data.\n")
    
    # Parse mock data
    model = "Qwen/Qwen2.5-7B"
    
    analysis_h200 = parse_bentoml_output(SAMPLE_H200_OUTPUT, model, "H200")
    analysis_l40 = parse_bentoml_output(SAMPLE_L40_OUTPUT, model, "L40")
    
    # Print individual analyses
    print("\n" + "=" * 80)
    print(f"Analysis for {model} on H200")
    print("=" * 80)
    print(analysis_h200)
    
    print("\n" + "=" * 80)
    print(f"Analysis for {model} on L40")
    print("=" * 80)
    print(analysis_l40)
    
    # Compare and recommend
    print("\n" + "=" * 80)
    print("RECOMMENDATION (based on throughput)")
    print("=" * 80)
    
    h200_throughput = analysis_h200.metrics.output_tokens_per_s
    l40_throughput = analysis_l40.metrics.output_tokens_per_s
    
    print(f"\nH200 throughput: {h200_throughput:.1f} tokens/s")
    print(f"L40 throughput: {l40_throughput:.1f} tokens/s")
    
    if h200_throughput > l40_throughput:
        recommended = "H200"
        improvement = ((h200_throughput - l40_throughput) / l40_throughput) * 100
    else:
        recommended = "L40"
        improvement = ((l40_throughput - h200_throughput) / h200_throughput) * 100
    
    print(f"\n✓ Recommended GPU: {recommended}")
    print(f"  Performance improvement: {improvement:.1f}% higher throughput")
    
    # Create recommendation object
    recommendation = GPURecommendation(
        model=model,
        recommended_gpu=recommended,
        all_analyses=[analysis_h200, analysis_l40]
    )
    
    print("\n" + "=" * 80)
    print("FULL RECOMMENDATION OBJECT")
    print("=" * 80)
    print(recommendation)
    
    print("\n" + "=" * 80)
    print("To use with real data, run:")
    print("  python -m config_recommender --models <model> --gpus H200 L40")
    print("=" * 80)


if __name__ == "__main__":
    demo()
