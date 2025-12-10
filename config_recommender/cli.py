#!/usr/bin/env python3
"""
Command-line interface for GPU recommendation engine.

Example usage:
    python -m config_recommender.cli --models "Qwen/Qwen2.5-7B" "meta-llama/Llama-3.3-70B-Instruct" --gpus H200 L40
"""

import argparse
import json
import sys
from typing import List

from .recommender import GPURecommender
from .models import GPURecommendation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GPU recommendation engine for model inference"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to analyze"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="List of GPU types to consider (e.g., H200 L40 H100)"
    )
    
    parser.add_argument(
        "--input-len",
        type=int,
        default=2000,
        help="Input token length for benchmarking (default: 2000)"
    )
    
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Output token length for benchmarking (default: 256)"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    
    parser.add_argument(
        "--metric",
        choices=["throughput", "latency"],
        default="throughput",
        help="Metric to optimize for (default: throughput)"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        help="Output recommendations to JSON file"
    )
    
    args = parser.parse_args()
    
    # Create recommender
    recommender = GPURecommender(
        input_len=args.input_len,
        output_len=args.output_len
    )
    
    # Get recommendations
    print(f"\nAnalyzing {len(args.models)} models across {len(args.gpus)} GPU types...")
    print(f"Models: {', '.join(args.models)}")
    print(f"GPUs: {', '.join(args.gpus)}")
    print(f"Optimizing for: {args.metric}")
    
    recommendations = recommender.recommend_gpus(
        models=args.models,
        gpus=args.gpus,
        num_gpus=args.num_gpus,
        metric=args.metric
    )
    
    # Print results
    recommender.print_recommendations(recommendations)
    
    # Save to JSON if requested
    if args.output_json:
        save_recommendations_json(recommendations, args.output_json)
        print(f"\nRecommendations saved to {args.output_json}")
    
    return 0


def save_recommendations_json(recommendations: List[GPURecommendation], filepath: str):
    """Save recommendations to a JSON file."""
    data = []
    for rec in recommendations:
        rec_data = {
            "model": rec.model,
            "recommended_gpu": rec.recommended_gpu,
            "analyses": []
        }
        
        for analysis in rec.all_analyses:
            analysis_data = {
                "config": {
                    "model": analysis.config.model,
                    "gpu": analysis.config.gpu,
                    "num_gpus": analysis.config.num_gpus,
                    "precision": analysis.config.precision,
                    "input_len": analysis.config.input_len,
                    "output_len": analysis.config.output_len,
                },
                "model_parameters": analysis.model_parameters,
                "model_layers": analysis.model_layers,
            }
            
            if analysis.metrics:
                analysis_data["metrics"] = {
                    "ttft_ms": analysis.metrics.ttft_ms,
                    "itl_ms": analysis.metrics.itl_ms,
                    "e2e_s": analysis.metrics.e2e_s,
                    "output_tokens_per_s": analysis.metrics.output_tokens_per_s,
                    "input_tokens_per_s": analysis.metrics.input_tokens_per_s,
                    "requests_per_s": analysis.metrics.requests_per_s,
                    "bottleneck": analysis.metrics.bottleneck,
                    "hardware_ops_per_byte": analysis.metrics.hardware_ops_per_byte,
                    "prefill_arithmetic_intensity": analysis.metrics.prefill_arithmetic_intensity,
                    "decode_arithmetic_intensity": analysis.metrics.decode_arithmetic_intensity,
                    "prefill_phase": analysis.metrics.prefill_phase,
                    "decode_phase": analysis.metrics.decode_phase,
                    "kv_cache_memory_limit": analysis.metrics.kv_cache_memory_limit,
                    "prefill_compute_limit": analysis.metrics.prefill_compute_limit,
                    "decode_capacity_limit": analysis.metrics.decode_capacity_limit,
                    "theoretical_overall_limit": analysis.metrics.theoretical_overall_limit,
                    "empirical_optimal_concurrency": analysis.metrics.empirical_optimal_concurrency,
                }
            
            rec_data["analyses"].append(analysis_data)
        
        data.append(rec_data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
