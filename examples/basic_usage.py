#!/usr/bin/env python3
"""Example: Basic GPU recommendation using the Python API."""

from config_recommender import (
    ModelArchitecture,
    GPUSpec,
    GPURecommender,
)


def main():
    """Run basic GPU recommendation example."""
    
    # Define a model using HuggingFace identifier
    # Model details are automatically fetched from HuggingFace
    model = ModelArchitecture(
        name="Qwen/Qwen2.5-7B",  # HuggingFace model identifier
    )
    
    # Define available GPUs
    gpus = [
        GPUSpec(
            name="NVIDIA A100 80GB",
            memory_gb=80.0,
            memory_bandwidth_gb_s=2039.0,
            tflops_fp16=312.0,
            tflops_fp32=156.0,
            cost_per_hour=3.67,
        ),
        GPUSpec(
            name="NVIDIA H100 80GB",
            memory_gb=80.0,
            memory_bandwidth_gb_s=3350.0,
            tflops_fp16=989.0,
            tflops_fp32=494.5,
            cost_per_hour=4.76,
        ),
        GPUSpec(
            name="NVIDIA V100 32GB",
            memory_gb=32.0,
            memory_bandwidth_gb_s=900.0,
            tflops_fp16=125.0,
            tflops_fp32=62.5,
            cost_per_hour=2.48,
        ),
    ]
    
    # Get recommendation
    print("=" * 60)
    print("GPU Recommendation Example")
    print("=" * 60)
    
    recommender = GPURecommender()
    result = recommender.recommend_gpu(model, gpus)
    
    print(f"\nModel: {result.model_name}")
    print(f"Recommended GPU: {result.recommended_gpu}")
    
    if result.performance:
        print(f"\nPerformance Estimates:")
        print(f"  Throughput: {result.performance.tokens_per_second:.2f} tokens/sec")
        print(f"  Latency: {result.performance.latency_ms_per_token:.2f} ms/token")
        print(f"  Memory Usage: {result.performance.memory_required_gb:.2f} GB")
        print(f"    - Weights: {result.performance.memory_weights_gb:.2f} GB")
        print(f"    - KV Cache: {result.performance.memory_kv_cache_gb:.2f} GB")
        print(f"    - Activations: {result.performance.memory_activation_gb:.2f} GB")
        print(f"  Bottleneck: {'Compute' if result.performance.compute_bound else 'Memory Bandwidth'}")
    
    print(f"\nReasoning: {result.reasoning}")
    
    print(f"\nAll Compatible GPUs ({len(result.all_compatible_gpus)}):")
    for i, gpu_info in enumerate(result.all_compatible_gpus, 1):
        print(f"  {i}. {gpu_info['gpu_name']}")
        print(f"     Throughput: {gpu_info['tokens_per_second']:.2f} tokens/sec")
        print(f"     Latency: {gpu_info['latency_ms_per_token']:.2f} ms/token")
        print(f"     Cost: ${gpu_info['cost_per_hour']:.2f}/hour")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
