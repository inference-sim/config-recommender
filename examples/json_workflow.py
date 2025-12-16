#!/usr/bin/env python3
"""Example: JSON workflow - load from files, process, save results."""

import argparse
import json
from pathlib import Path

from config_recommender import GPURecommender, GPUSpec, ModelArchitecture


def load_models_from_json(filepath):
    """Load models from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [ModelArchitecture(**model_data) for model_data in data]


def load_gpus_from_json(filepath):
    """Load GPUs from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [GPUSpec(**gpu_data) for gpu_data in data]


def main():
    """Run JSON workflow example."""

    parser = argparse.ArgumentParser(
        description="JSON workflow for GPU recommendations"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable printing the summary (enabled by default)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("JSON Workflow Example")
    print("=" * 80)

    # Setup paths
    examples_dir = Path(__file__).parent
    models_file = examples_dir / "models.json"
    gpus_file = examples_dir / "custom_gpus.json"
    output_file = examples_dir / "output_recommendations.json"

    print(f"\nLoading models from: {models_file}")
    models = load_models_from_json(models_file)
    print(f"Loaded {len(models)} models: {[m.name for m in models]}")

    print(f"\nLoading GPUs from: {gpus_file}")
    gpus = load_gpus_from_json(gpus_file)
    print(f"Loaded {len(gpus)} GPU types: {[g.name for g in gpus]}")

    # Get recommendations
    print("\nGenerating recommendations...")
    recommender = GPURecommender()
    results = recommender.recommend_for_models(models, gpus)

    # Convert to dict for JSON
    output_data = {
        "recommendations": [result.to_dict() for result in results],
        "metadata": {
            "num_models": len(models),
            "num_gpu_types": len(gpus),
        },
    }

    # Save to file
    print(f"\nSaving recommendations to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully saved recommendations!")

    # Print summary (unless disabled)
    if not args.no_summary:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        for result in results:
            status = "✓" if result.recommended_gpu else "✗"
            print(
                f"{status} {result.model_name:<20} -> {result.recommended_gpu or 'No compatible GPU'}"
            )

        print(f"\nFull results available in: {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()
