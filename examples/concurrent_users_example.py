#!/usr/bin/env python3
"""Example: GPU recommendation with concurrent users using the Python API.

This example demonstrates how increasing concurrent users affects GPU recommendations,
particularly for KV cache memory requirements and tensor parallelism.
"""

from config_recommender import (
    GPURecommender,
    GPUSpec,
    ModelArchitecture,
    SyntheticBenchmarkEstimator,
)


def main():
    """Run concurrent users example."""

    # Define a model using HuggingFace identifier
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
    ]

    print("=" * 80)
    print("GPU Recommendation with Concurrent Users Example")
    print("=" * 80)
    print(f"\nModel: {model.name}")
    print(f"GPUs: {', '.join(gpu.name for gpu in gpus)}")
    print()

    # Test with different numbers of concurrent users
    concurrent_users_scenarios = [1, 5, 10, 20, 50]

    for concurrent_users in concurrent_users_scenarios:
        print(f"\n{'─' * 80}")
        print(f"Scenario: {concurrent_users} concurrent user(s)")
        print(f"{'─' * 80}")

        # Create estimator with concurrent users
        estimator = SyntheticBenchmarkEstimator(
            batch_size=1,
            precision_bytes=2,  # FP16
            concurrent_users=concurrent_users,
        )

        # Create recommender
        recommender = GPURecommender(estimator=estimator)

        # Get recommendation
        result = recommender.recommend_gpu(model, gpus)

        if result.recommended_gpu and result.performance:
            tp_size = result.performance.tensor_parallel_size

            # Calculate total cost
            cost_per_hour = 0
            if result.all_compatible_gpus:
                cost_per_hour = result.all_compatible_gpus[0].get("cost_per_hour", 0)
            total_cost = cost_per_hour * tp_size

            print(f"\nRecommendation:")
            if tp_size > 1:
                print(
                    f"  Configuration: {tp_size}x {result.recommended_gpu} (Tensor Parallel)"
                )
            else:
                print(f"  Configuration: {result.recommended_gpu}")

            print(f"\nMemory Breakdown (per GPU):")
            print(
                f"  Weights:   {result.performance.memory_weights_gb:>6.2f} GB (constant)"
            )
            print(
                f"  KV Cache:  {result.performance.memory_kv_cache_gb:>6.2f} GB (scales with users)"
            )
            print(
                f"  Total:     {result.performance.memory_required_gb:>6.2f} GB / {gpus[0].memory_gb:.2f} GB"
            )

            print(f"\nPerformance:")
            print(
                f"  Throughput: {result.performance.tokens_per_second:.2f} tokens/sec"
            )
            print(
                f"  Latency:    {result.performance.intertoken_latency_ms:.2f} ms/token"
            )

            print(f"\nCost:")
            if tp_size > 1:
                if result.all_compatible_gpus:
                    cost_per_hour = result.all_compatible_gpus[0].get('cost_per_hour', 0)
                    print(f"  Per GPU:  ${cost_per_hour:.2f}/hour")
                    print(f"  Total:    ${cost_per_hour * tp_size:.2f}/hour ({tp_size} GPUs)")
                else:
                    print(f"  Total:    ${total_cost:.2f}/hour ({tp_size} GPUs)")
            else:
                if result.all_compatible_gpus:
                    cost_per_hour = result.all_compatible_gpus[0].get('cost_per_hour', 0)
                    print(f"  ${cost_per_hour:.2f}/hour")
                else:
                    print(f"  ${total_cost:.2f}/hour")
        else:
            print("\n⚠️  No compatible GPU found for this configuration!")

    print(f"\n{'=' * 80}")
    print("\nKey Observations:")
    print("  • Model weights remain constant regardless of concurrent users")
    print("  • KV cache scales linearly with number of concurrent users")
    print("  • More concurrent users → higher memory requirements")
    print("  • Higher memory requirements → may trigger tensor parallelism")
    print("  • Tensor parallelism shards both weights and KV cache across GPUs")
    print("=" * 80)


if __name__ == "__main__":
    main()
