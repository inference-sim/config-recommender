#!/usr/bin/env python3
"""Example demonstrating GPU library usage and custom GPU specs.

This example shows:
1. Using GPUs from the preloaded library
2. Creating custom GPU specs
3. Overriding library GPUs with custom parameters
"""

from config_recommender import (
    GPURecommender,
    ModelArchitecture,
    create_custom_gpu,
    get_gpu_specs,
)


def main():
    """Demonstrate GPU library and custom GPU functionality."""
    
    # Example 1: Using GPUs from the library
    print("=" * 70)
    print("Example 1: Using GPUs from the preloaded library")
    print("=" * 70)
    
    # Get specific GPUs from library
    library_gpus = get_gpu_specs(["H100", "A100-80GB", "L40"])
    
    print(f"\nLoaded {len(library_gpus)} GPUs from library:")
    for gpu in library_gpus:
        print(f"  - {gpu.name}: {gpu.memory_gb}GB, {gpu.tflops_fp16} TFLOPS FP16")
    
    # Example 2: Creating a custom GPU
    print("\n" + "=" * 70)
    print("Example 2: Creating a custom GPU spec")
    print("=" * 70)
    
    custom_gpu = create_custom_gpu(
        name="My Custom GPU",
        memory_gb=256.0,
        memory_bandwidth_gb_s=8000.0,
        tflops_fp16=2000.0,
        tflops_fp32=1000.0,
        cost_per_hour=12.0,
    )
    
    print(f"\nCreated custom GPU: {custom_gpu.name}")
    print(f"  Memory: {custom_gpu.memory_gb}GB")
    print(f"  Bandwidth: {custom_gpu.memory_bandwidth_gb_s} GB/s")
    print(f"  FP16 Performance: {custom_gpu.tflops_fp16} TFLOPS")
    print(f"  Cost: ${custom_gpu.cost_per_hour}/hour")
    
    # Example 3: Combining library and custom GPUs
    print("\n" + "=" * 70)
    print("Example 3: Combining library and custom GPUs for recommendation")
    print("=" * 70)
    
    # Create a model (using manual specs to avoid HuggingFace API call)
    model = ModelArchitecture(
        name="Example 7B Model",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=4096,
    )
    
    print(f"\nModel: {model.name}")
    print(f"  Parameters: {model.get_num_parameters()}B")
    
    # Combine library GPUs with custom GPU
    all_gpus = library_gpus + [custom_gpu]
    
    # Get recommendation
    recommender = GPURecommender()
    result = recommender.recommend_gpu(model, all_gpus)
    
    print(f"\nRecommended GPU: {result.recommended_gpu}")
    if result.performance:
        print(f"  Throughput: {result.performance.tokens_per_second:.2f} tokens/sec")
        print(f"  Latency: {result.performance.intertoken_latency_ms:.2f} ms/token")
        print(f"  Memory Required: {result.performance.memory_required_gb:.2f} GB")
    
    print(f"\nAll compatible GPUs ({len(result.all_compatible_gpus)}):")
    for gpu_info in result.all_compatible_gpus[:5]:  # Show top 5
        print(f"  - {gpu_info['gpu_name']}: {gpu_info['tokens_per_second']:.2f} tokens/sec")
    
    # Example 4: Override library GPU specs
    print("\n" + "=" * 70)
    print("Example 4: Using library GPU as base and customizing")
    print("=" * 70)
    
    # Get a library GPU and modify it
    from config_recommender import GPUSpec
    
    # Create a modified version of H100 with different cost
    h100_custom = GPUSpec(
        name="NVIDIA H100 80GB (Custom Pricing)",
        memory_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        tflops_fp16=989.0,
        tflops_fp32=494.5,
        cost_per_hour=6.0,  # Custom pricing
    )
    
    print(f"\nCustom H100 variant: {h100_custom.name}")
    print(f"  Modified cost: ${h100_custom.cost_per_hour}/hour (vs. standard $4.76)")
    
    print("\n" + "=" * 70)
    print("Summary: GPU library provides flexibility to:")
    print("  ✓ Use preloaded specs for common GPUs (H100, H200, A100, L40, etc.)")
    print("  ✓ Create fully custom GPU specifications")
    print("  ✓ Override/modify library GPU specs as needed")
    print("  ✓ Combine library and custom GPUs in recommendations")
    print("=" * 70)


if __name__ == "__main__":
    main()
