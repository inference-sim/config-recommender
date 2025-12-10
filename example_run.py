#!/usr/bin/env python3
"""
Example script to run GPU recommendations for the models specified in the issue.

Models:
- gpt-oss-120b
- Qwen3-8B-FP8-dynamic
- Llama-3.3-70B-Instruct
- granite-4.0-h-small
- Mixtral-8x7B-v0.1

GPUs: H200, L40
"""

from config_recommender import GPURecommender

# Define models and GPUs as specified in the issue
MODELS = [
    "gpt-oss-120b",
    "Qwen3-8B-FP8-dynamic",
    "meta-llama/Llama-3.3-70B-Instruct",  # Using standard HF naming
    "granite-4.0-h-small",
    "mistralai/Mixtral-8x7B-v0.1",  # Using standard HF naming
]

GPUS = ["H200", "L40"]


def main():
    """Run the recommendation engine."""
    print("GPU Recommendation Engine")
    print("=" * 80)
    print(f"\nModels to analyze: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model}")
    print(f"\nGPU types: {', '.join(GPUS)}")
    print()
    
    # Create recommender with default settings
    recommender = GPURecommender(input_len=2000, output_len=256)
    
    # Get recommendations optimizing for throughput
    recommendations = recommender.recommend_gpus(
        models=MODELS,
        gpus=GPUS,
        num_gpus=1,
        metric="throughput"
    )
    
    # Print results
    recommender.print_recommendations(recommendations)
    
    # Save to JSON
    import json
    from config_recommender.cli import save_recommendations_json
    
    output_file = "gpu_recommendations.json"
    save_recommendations_json(recommendations, output_file)
    print(f"\n✓ Recommendations saved to {output_file}")


if __name__ == "__main__":
    main()
