#!/usr/bin/env python3
"""Example: Model Selection - Choosing the best model for your available GPUs.

This example demonstrates a common real-world scenario:
- You have specific GPU hardware available
- You need to choose which model to deploy
- You want to understand performance trade-offs between different model sizes
- You need to make data-driven architecture decisions

Use Cases:
1. Selecting the right model size (7B vs 13B vs 70B) for your hardware
2. Comparing different model architectures (dense vs MoE)
3. Evaluating the performance impact of model choices
4. Making cost-effective deployment decisions
"""

from config_recommender import (
    GPURecommender,
    ModelArchitecture,
    get_gpu_specs,
)


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def compare_models_on_gpu(models, gpu_name, recommender):
    """Compare how different models perform on a specific GPU."""
    print(f"\n{'Model':<40} {'Fits?':<8} {'Tokens/sec':<15} {'Latency (ms)':<15}")
    print("-" * 80)

    results = []
    for model in models:
        # Get recommendation using just this one GPU
        gpu = get_gpu_specs([gpu_name])
        result = recommender.recommend_gpu(model, gpu)

        fits = result.performance is not None
        tokens_per_sec = result.performance.tokens_per_second if fits else 0
        latency = result.performance.intertoken_latency_ms if fits else float('inf')

        results.append({
            'model': model.name,
            'fits': fits,
            'tokens_per_sec': tokens_per_sec,
            'latency': latency,
            'result': result
        })

        fit_status = "✓ Yes" if fits else "✗ No"
        tokens_display = f"{tokens_per_sec:.2f}" if fits else "N/A"
        latency_display = f"{latency:.2f}" if fits else "N/A"

        print(f"{model.name:<40} {fit_status:<8} {tokens_display:<15} {latency_display:<15}")

    return results


def main():
    """Demonstrate model selection scenarios."""

    print_section_header("Model Selection Example: Choosing the Right Model for Your GPUs")

    # Scenario: You're building an LLM application and need to choose a model
    # You have A100 80GB GPUs available in your infrastructure

    # Define candidate models (different sizes and architectures)
    candidate_models = [
        # Small models - fast but less capable
        ModelArchitecture(name="Qwen/Qwen2.5-7B"),

        # Medium models - good balance
        ModelArchitecture(
            name="Meta Llama 3 8B",  # Using manual spec since it's gated
            num_parameters=8.0,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=128256,
            max_sequence_length=8192,
        ),

        # Large dense models - high quality but slower
        ModelArchitecture(
            name="Meta Llama 3 70B",  # Using manual spec since it's gated
            num_parameters=70.0,
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            vocab_size=128256,
            max_sequence_length=8192,
        ),

        # MoE model - good quality, unique memory characteristics
        ModelArchitecture(name="mistralai/Mixtral-8x7B-v0.1"),
    ]

    recommender = GPURecommender()

    # =========================================================================
    # Scenario 1: What models can run on a single A100 80GB?
    # =========================================================================
    print_section_header("Scenario 1: Model Comparison on Single A100 80GB")

    print("You have A100 80GB GPUs available. Which model should you deploy?")
    print("\nComparing models:")

    a100_results = compare_models_on_gpu(candidate_models, "A100-80GB", recommender)

    # Find best performing model that fits
    fitting_models = [r for r in a100_results if r['fits']]
    if fitting_models:
        best_model = max(fitting_models, key=lambda x: x['tokens_per_sec'])
        print(f"\n→ Best choice: {best_model['model']}")
        print(f"  Reason: Highest throughput ({best_model['tokens_per_sec']:.2f} tokens/sec) among models that fit")

    # =========================================================================
    # Scenario 2: What if we have H100s? Does it change our choice?
    # =========================================================================
    print_section_header("Scenario 2: Same Models on H100 80GB (Newer Hardware)")

    print("If you upgrade to H100 80GB GPUs, how does it affect model selection?")
    print("\nComparing models:")

    h100_results = compare_models_on_gpu(candidate_models, "H100", recommender)

    # Show improvement from A100 to H100
    print("\n→ Performance Improvement (A100 → H100):")
    print(f"\n{'Model':<40} {'A100 (tok/s)':<15} {'H100 (tok/s)':<15} {'Speedup':<10}")
    print("-" * 80)

    for a100_r, h100_r in zip(a100_results, h100_results):
        if a100_r['fits'] and h100_r['fits']:
            speedup = h100_r['tokens_per_sec'] / a100_r['tokens_per_sec']
            print(f"{a100_r['model']:<40} {a100_r['tokens_per_sec']:<15.2f} "
                  f"{h100_r['tokens_per_sec']:<15.2f} {speedup:.2f}x")

    # =========================================================================
    # Scenario 3: Model quality vs performance tradeoff
    # =========================================================================
    print_section_header("Scenario 3: Quality vs Performance Tradeoff Analysis")

    print("Decision framework for choosing between models:\n")

    # Focusing on models that fit on A100
    models_to_compare = [
        ("Qwen/Qwen2.5-7B", "Small (7B)", "Lower quality, highest speed, lowest cost"),
        ("Meta Llama 3 8B", "Medium (8B)", "Good quality, good speed, moderate cost"),
        ("mistralai/Mixtral-8x7B-v0.1", "Large MoE (8x7B)", "High quality, lower speed, requires multiple GPUs"),
    ]

    print(f"{'Model':<40} {'Category':<20} {'Use Case':<40}")
    print("-" * 100)

    for model_name, category, use_case in models_to_compare:
        print(f"{model_name:<40} {category:<20} {use_case:<40}")

    # =========================================================================
    # Scenario 4: Latency-sensitive application
    # =========================================================================
    print_section_header("Scenario 4: Latency-Sensitive Application (e.g., Chatbot)")

    print("For real-time chat applications, you need <20ms inter-token latency.")
    print("Which models meet this requirement on A100 80GB?\n")

    latency_bound = 20.0
    recommender_latency = GPURecommender(latency_bound_ms=latency_bound)

    print(f"{'Model':<40} {'Latency (ms)':<15} {'Meets Requirement?':<20}")
    print("-" * 75)

    for result in a100_results:
        if result['fits']:
            meets_req = result['latency'] < latency_bound
            status = "✓ Yes" if meets_req else "✗ No"
            print(f"{result['model']:<40} {result['latency']:<15.2f} {status:<20}")

    # =========================================================================
    # Scenario 5: Batch processing optimization
    # =========================================================================
    print_section_header("Scenario 5: Batch Processing vs Real-Time Serving")

    print("Different batch sizes affect throughput and latency.\n")
    print("Note: This example uses default batch size. For production,")
    print("you would test with actual batch sizes expected in your workload.\n")

    # Compare a small and large model
    small_model = candidate_models[0]  # Qwen 7B
    large_model = candidate_models[2]  # Llama 70B

    print("Trade-off example:")
    print(f"- {small_model.name}: Fast responses, good for real-time serving")
    print(f"- {large_model.name}: Higher quality, better for offline batch processing")

    # =========================================================================
    # Scenario 6: Multi-model deployment
    # =========================================================================
    print_section_header("Scenario 6: Multi-Model Deployment Strategy")

    print("Consider deploying multiple models for different use cases:\n")

    deployment_strategy = [
        {
            'use_case': 'Simple queries & classification',
            'model': 'Qwen/Qwen2.5-7B',
            'gpu': 'A100-80GB',
            'reason': 'Fast, cost-effective for simple tasks'
        },
        {
            'use_case': 'Complex reasoning & generation',
            'model': 'mistralai/Mixtral-8x7B-v0.1',
            'gpu': '2x A100-80GB',
            'reason': 'Higher quality for demanding tasks'
        },
    ]

    print(f"{'Use Case':<35} {'Model':<35} {'Resources':<20}")
    print("-" * 90)

    for strategy in deployment_strategy:
        print(f"{strategy['use_case']:<35} {strategy['model']:<35} {strategy['gpu']:<20}")
        print(f"  → {strategy['reason']}")
        print()

    # =========================================================================
    # Summary and Recommendations
    # =========================================================================
    print_section_header("Summary: Decision Framework")

    print("""
Key Considerations for Model Selection:

1. **Hardware Constraints**
   - Check which models fit in available GPU memory
   - Consider tensor parallelism for larger models

2. **Performance Requirements**
   - Real-time serving: Prioritize low latency (<20ms)
   - Batch processing: Prioritize high throughput

3. **Quality vs Speed**
   - Smaller models (7-8B): Fast but less capable
   - MoE models (Mixtral): Good quality-speed balance
   - Large models (70B+): Best quality but slower

4. **Cost Optimization**
   - Smaller models: Lower GPU requirements, lower cost
   - Consider multi-model strategy: Fast model for simple tasks,
     large model for complex tasks

5. **Scaling Strategy**
   - Start with smaller model, monitor quality
   - Upgrade to larger model if quality is insufficient
   - Use model router to direct queries to appropriate model

Next Steps:
- Test these models on your specific workload
- Measure actual quality metrics (accuracy, user satisfaction)
- Monitor production performance and costs
- Iterate based on real-world feedback
    """)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
