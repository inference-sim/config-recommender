#!/usr/bin/env python3
"""Example: Advanced GPU recommendation with multiple models and latency constraints."""

from config_recommender import (GPURecommender, GPUSpec, ModelArchitecture,
                                SyntheticBenchmarkEstimator)


def main():
    """Run advanced GPU recommendation example."""

    # Define multiple models using HuggingFace identifiers
    # Model details are automatically fetched from HuggingFace
    models = [
        ModelArchitecture(name="Qwen/Qwen2.5-7B"),
        ModelArchitecture(name="mistralai/Mixtral-8x7B-v0.1"),
        ModelArchitecture(name="ibm-granite/granite-3.0-8b-base"),
    ]

    # Example: Override parameters for a gated model when you know the specs
    # but don't have a HuggingFace token
    # This is useful for models like Llama where you have the architecture info
    gated_model = ModelArchitecture(
        name="meta-llama/Llama-2-7b-hf",  # This would normally require HF_TOKEN
        # Provide manual overrides so it works without token
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=4096,
    )
    models.append(gated_model)

    # Define GPU fleet
    gpus = [
        GPUSpec(
            name="NVIDIA H100 80GB",
            memory_gb=80.0,
            memory_bandwidth_gb_s=3350.0,
            tflops_fp16=989.0,
            tflops_fp32=494.5,
            cost_per_hour=4.76,
        ),
        GPUSpec(
            name="NVIDIA A100 80GB",
            memory_gb=80.0,
            memory_bandwidth_gb_s=2039.0,
            tflops_fp16=312.0,
            tflops_fp32=156.0,
            cost_per_hour=3.67,
        ),
        GPUSpec(
            name="NVIDIA A100 40GB",
            memory_gb=40.0,
            memory_bandwidth_gb_s=1555.0,
            tflops_fp16=312.0,
            tflops_fp32=156.0,
            cost_per_hour=2.93,
        ),
        GPUSpec(
            name="NVIDIA V100 32GB",
            memory_gb=32.0,
            memory_bandwidth_gb_s=900.0,
            tflops_fp16=125.0,
            tflops_fp32=62.5,
            cost_per_hour=2.48,
        ),
        GPUSpec(
            name="NVIDIA T4 16GB",
            memory_gb=16.0,
            memory_bandwidth_gb_s=300.0,
            tflops_fp16=65.0,
            tflops_fp32=8.1,
            cost_per_hour=0.526,
        ),
    ]

    print("=" * 80)
    print("Advanced GPU Recommendation Example")
    print("=" * 80)

    # Example 1: Basic recommendations for all models
    print("\n### Example 1: Basic Recommendations ###\n")

    recommender = GPURecommender()
    results = recommender.recommend_for_models(models, gpus)

    for result in results:
        print(f"Model: {result.model_name}")
        print(f"  Recommended GPU: {result.recommended_gpu}")
        if result.performance:
            print(
                f"  Throughput: {result.performance.tokens_per_second:.2f} tokens/sec"
            )
            print(
                f"  Inter-token Latency: {result.performance.intertoken_latency_ms:.2f} ms/token"
            )
            print(f"  Memory: {result.performance.memory_required_gb:.2f} GB")
        print()

    # Example 2: With latency constraint
    print("\n### Example 2: With 10ms Latency Constraint ###\n")

    recommender_latency = GPURecommender(latency_bound_ms=10.0)
    results_latency = recommender_latency.recommend_for_models(models, gpus)

    for result in results_latency:
        print(f"Model: {result.model_name}")
        print(
            f"  Recommended GPU: {result.recommended_gpu or 'None (latency constraint not met)'}"
        )
        if result.performance:
            print(
                f"  Inter-token Latency: {result.performance.intertoken_latency_ms:.2f} ms/token (meets <10ms requirement)"
            )
        print()

    # Example 3: Custom estimator configuration (FP32)
    print("\n### Example 3: FP32 Precision (vs default FP16) ###\n")

    estimator_fp32 = SyntheticBenchmarkEstimator(
        precision_bytes=4,  # FP32
    )
    recommender_fp32 = GPURecommender(estimator=estimator_fp32)

    # Compare FP16 vs FP32 for one model
    model = models[0]  # llama-2-7b
    result_fp16 = recommender.recommend_gpu(model, gpus)
    result_fp32 = recommender_fp32.recommend_gpu(model, gpus)

    print(f"Model: {model.name}")
    print(f"\nFP16 (default):")
    print(f"  Memory: {result_fp16.performance.memory_weights_gb:.2f} GB weights")
    print(f"  Recommended: {result_fp16.recommended_gpu}")

    print(f"\nFP32 (higher precision):")
    print(f"  Memory: {result_fp32.performance.memory_weights_gb:.2f} GB weights")
    print(f"  Recommended: {result_fp32.recommended_gpu}")

    # Example 4: Custom sequence length
    print("\n\n### Example 4: Impact of Sequence Length ###\n")

    model = models[1]  # mistral-7b with 8K max

    for seq_len in [512, 2048, 8192]:
        result = recommender.recommend_gpu(model, gpus, sequence_length=seq_len)
        print(f"Sequence Length: {seq_len}")
        print(f"  KV Cache: {result.performance.memory_kv_cache_gb:.2f} GB")
        print(f"  Total Memory: {result.performance.memory_required_gb:.2f} GB")
        print(f"  Recommended: {result.recommended_gpu}")
        print()

    # Example 5: Cost-performance analysis
    print("\n### Example 5: Cost-Performance Analysis ###\n")

    model = models[0]  # llama-2-7b
    result = recommender.recommend_gpu(model, gpus)

    print(f"Model: {model.name}\n")
    print(f"{'GPU':<20} {'Tokens/sec':>12} {'$/hour':>10} {'Tokens/$':>12}")
    print("-" * 56)

    for gpu_info in result.all_compatible_gpus:
        tokens_per_sec = gpu_info["tokens_per_second"]
        cost = gpu_info["cost_per_hour"]
        tokens_per_dollar = (tokens_per_sec * 3600) / cost  # tokens per dollar

        print(
            f"{gpu_info['gpu_name']:<20} {tokens_per_sec:>12.2f} {cost:>10.2f} {tokens_per_dollar:>12.0f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
