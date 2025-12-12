"""Command-line interface for GPU recommendation."""

import argparse
import json
import sys
from typing import List

from .estimator import SyntheticBenchmarkEstimator
from .models import GPUSpec, ModelArchitecture
from .recommender import GPURecommender


def load_models_from_json(filepath: str) -> List[ModelArchitecture]:
    """Load model architectures from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    models = []
    for model_data in data:
        models.append(ModelArchitecture(**model_data))
    return models


def load_gpus_from_json(filepath: str) -> List[GPUSpec]:
    """Load GPU specs from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    gpus = []
    for gpu_data in data:
        gpus.append(GPUSpec(**gpu_data))
    return gpus


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GPU Recommendation Engine for ML Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommend GPUs for models using example data files
  config-recommender --models examples/models.json --gpus examples/gpus.json

  # With latency constraint and custom parameters
  config-recommender --models examples/models.json --gpus examples/gpus.json \\
      --latency-bound 10 --precision fp16

  # With concurrent users (for multi-user server scenarios)
  config-recommender --models examples/models.json --gpus examples/gpus.json \\
      --concurrent-users 10

  # Output to file
  config-recommender --models examples/models.json --gpus examples/gpus.json \\
      --output recommendations.json
        """,
    )

    parser.add_argument(
        "--models",
        required=True,
        help="Path to JSON file containing model architectures",
    )

    parser.add_argument(
        "--gpus", required=True, help="Path to JSON file containing GPU specifications"
    )

    parser.add_argument("--output", help="Path to output JSON file (default: stdout)")

    parser.add_argument(
        "--latency-bound",
        type=float,
        help="Maximum acceptable latency per token in milliseconds",
    )

    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Inference precision (default: fp16)",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Sequence length (default: use model max_sequence_length)",
    )

    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=1,
        help="Number of concurrent users hitting the server at once (default: 1)",
    )

    args = parser.parse_args()

    try:
        # Load models and GPUs
        models = load_models_from_json(args.models)
        gpus = load_gpus_from_json(args.gpus)

        if not models:
            print("Error: No models found in input file", file=sys.stderr)
            sys.exit(1)

        if not gpus:
            print("Error: No GPUs found in input file", file=sys.stderr)
            sys.exit(1)

        # Use concurrent_users for KV cache calculations (accounts for multiple concurrent requests)
        precision_bytes = 2 if args.precision == "fp16" else 4
        estimator = SyntheticBenchmarkEstimator(
            precision_bytes=precision_bytes,
            concurrent_users=args.concurrent_users,
        )

        # Create recommender
        recommender = GPURecommender(
            estimator=estimator,
            latency_bound_ms=args.latency_bound,
        )

        # Get recommendations
        results = recommender.recommend_for_models(
            models=models,
            available_gpus=gpus,
            sequence_length=args.sequence_length,
        )

        # Convert to dict for JSON output
        output_data = {
            "recommendations": [result.to_dict() for result in results],
            "parameters": {
                "precision": args.precision,
                "latency_bound_ms": args.latency_bound,
                "sequence_length": args.sequence_length,
                "concurrent_users": args.concurrent_users,
            },
        }

        # Output results (always pretty-printed)
        json_output = json.dumps(output_data, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"Recommendations written to {args.output}")
        else:
            print(json_output)

        # Print summary to stderr
        print(f"\n=== Summary ===", file=sys.stderr)
        print(
            f"Evaluated {len(models)} model(s) against {len(gpus)} GPU type(s)",
            file=sys.stderr,
        )
        for result in results:
            if result.recommended_gpu:
                # Include TP size if using tensor parallelism
                if result.performance and result.performance.tensor_parallel_size > 1:
                    tp_size = result.performance.tensor_parallel_size
                    # Get cost from compatible GPUs
                    cost_info = ""
                    if result.all_compatible_gpus:
                        cost_per_hour = result.all_compatible_gpus[0].get("cost_per_hour")
                        if cost_per_hour is not None:
                            total_cost = cost_per_hour * tp_size
                            cost_info = f" (${total_cost:.2f}/hr)"
                    print(
                        f"  {result.model_name}: {tp_size}x{result.recommended_gpu} "
                        f"(TP={tp_size}){cost_info}",
                        file=sys.stderr,
                    )
                else:
                    # Get cost for single GPU
                    cost_info = ""
                    if result.all_compatible_gpus:
                        cost_per_hour = result.all_compatible_gpus[0].get("cost_per_hour")
                        if cost_per_hour is not None:
                            cost_info = f" (${cost_per_hour:.2f}/hr)"
                    print(
                        f"  {result.model_name}: {result.recommended_gpu}{cost_info}",
                        file=sys.stderr,
                    )
            else:
                print(f"  {result.model_name}: No compatible GPU", file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
