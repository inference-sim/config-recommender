#!/usr/bin/env python3
"""Example: Tensor Parallelism recommendations for large models."""

from config_recommender import GPURecommender, GPUSpec, ModelArchitecture


def main():
    """Demonstrate TP recommendations for Mixtral-8x7B."""

    # Mixtral-8x7B is a large model that doesn't fit on a single GPU
    model = ModelArchitecture(
        name="mistralai/Mixtral-8x7B-v0.1",
    )

    # Define available GPUs - using different GPU types to demonstrate TP behavior
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
    ]

    # Get recommendation
    print("=" * 80)
    print("Tensor Parallelism Recommendation Example")
    print("=" * 80)

    recommender = GPURecommender()
    result = recommender.recommend_gpu(model, gpus)

    print(f"\nModel: {result.model_name}")
    print(f"Recommended Configuration: {result.recommended_gpu}")

    if result.performance:
        tp_size = result.performance.tensor_parallel_size
        if tp_size > 1:
            print(f"Tensor Parallel Size: {tp_size} GPUs")
            print(
                f"\nThis model requires {tp_size} GPUs working together "
                f"using tensor parallelism."
            )
            # Calculate total model memory (before TP sharding)
            total_memory = (
                result.performance.memory_weights_gb
                + result.performance.memory_kv_cache_gb
            ) * tp_size
            print(
                f"Each GPU will hold {result.performance.memory_required_gb:.2f} GB "
                f"of the model (total: ~{total_memory:.2f} GB distributed)."
            )
        else:
            print("Tensor Parallel Size: 1 GPU (single GPU inference)")

        print(f"\nPerformance Estimates:")
        print(f"  Throughput: {result.performance.tokens_per_second:.2f} tokens/sec")
        print(
            f"  Inter-token Latency: {result.performance.intertoken_latency_ms:.2f} ms/token"
        )
        print(
            f"  Memory per GPU: {result.performance.memory_required_gb:.2f} GB / "
            f"{gpus[0].memory_gb:.2f} GB"
        )

    print(f"\nReasoning: {result.reasoning}")

    print(f"\nAll Compatible TP Configurations ({len(result.all_compatible_gpus)}):")
    for i, gpu_info in enumerate(result.all_compatible_gpus[:8], 1):  # Show top 8
        tp = gpu_info.get("tensor_parallel_size", 1)
        print(f"\n  {i}. {tp}x {gpu_info['gpu_name']} (TP={tp})")
        print(f"     Throughput: {gpu_info['tokens_per_second']:.2f} tokens/sec")
        print(
            f"     Inter-token Latency: {gpu_info['intertoken_latency_ms']:.2f} ms/token"
        )
        print(
            f"     Memory per GPU: {gpu_info['memory_required_gb']:.2f} GB / "
            f"{gpu_info['memory_available_gb']:.2f} GB"
        )
        print(f"     Total Cost: ${gpu_info['cost_per_hour'] * tp:.2f}/hour")

    print("\n" + "=" * 80)
    print("\nKey Insights:")
    print("- Tensor Parallelism (TP) splits the model across multiple GPUs")
    print("- A100 40GB requires higher TP (more GPUs) to fit the model compared to 80GB variants")
    print("- H100 achieves better performance than A100 due to higher compute and bandwidth")
    print("- A100 80GB offers lower cost alternative when performance is less critical")
    print("- Higher TP values use more GPUs but can achieve better throughput")
    print("- However, TP introduces communication overhead between GPUs")
    print("- The tool evaluates TP values of 2, 4, and 8 to find the best option")
    print("=" * 80)


if __name__ == "__main__":
    main()
