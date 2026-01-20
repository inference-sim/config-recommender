#!/usr/bin/env python3
"""Example client demonstrating how to use the GPU Config Recommender API.

This script shows various ways to interact with the API endpoints.
"""

import requests
from typing import List, Dict, Any, Optional


class GPUConfigRecommenderClient:
    """Client for interacting with the GPU Config Recommender API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api"

    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy.

        Returns:
            Health status information
        """
        response = requests.get(f"{self.api_base}/health")
        response.raise_for_status()
        return response.json()

    def list_available_gpus(self) -> List[str]:
        """List all available GPU keys in the library.

        Returns:
            List of GPU key strings
        """
        response = requests.get(f"{self.api_base}/gpu-library/available")
        response.raise_for_status()
        return response.json()

    def get_gpu_specs(self, gpu_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get GPU specifications from the library.

        Args:
            gpu_keys: Optional list of GPU keys (returns all if None)

        Returns:
            List of GPU specifications
        """
        payload = {"gpu_keys": gpu_keys}
        response = requests.post(f"{self.api_base}/gpu-library", json=payload)
        response.raise_for_status()
        return response.json()

    def validate_model(
        self,
        model_name: str,
        hf_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate a HuggingFace model.

        Args:
            model_name: HuggingFace model identifier
            hf_token: Optional HuggingFace token for gated models

        Returns:
            Validation results with model metadata
        """
        payload = {
            "model_name": model_name,
            "hf_token": hf_token
        }
        response = requests.post(f"{self.api_base}/models/validate", json=payload)
        response.raise_for_status()
        return response.json()

    def get_recommendation(
        self,
        model_name: str,
        available_gpus: List[Dict[str, Any]],
        sequence_length: Optional[int] = None,
        latency_bound_ms: Optional[float] = None,
        hf_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get GPU recommendation for a model.

        Args:
            model_name: HuggingFace model identifier
            available_gpus: List of GPU specifications
            sequence_length: Optional sequence length
            latency_bound_ms: Optional latency constraint
            hf_token: Optional HuggingFace token

        Returns:
            Recommendation with performance estimates
        """
        payload = {
            "model": {
                "name": model_name,
                "hf_token": hf_token
            },
            "available_gpus": available_gpus,
            "sequence_length": sequence_length,
            "latency_bound_ms": latency_bound_ms
        }
        response = requests.post(f"{self.api_base}/recommendations", json=payload)
        response.raise_for_status()
        return response.json()


def example_1_health_check():
    """Example 1: Check API health."""
    print("\n" + "=" * 70)
    print("Example 1: Health Check")
    print("=" * 70)

    client = GPUConfigRecommenderClient()
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Version: {health['version']}")


def example_2_list_gpus():
    """Example 2: List available GPUs."""
    print("\n" + "=" * 70)
    print("Example 2: List Available GPUs")
    print("=" * 70)

    client = GPUConfigRecommenderClient()
    gpus = client.list_available_gpus()
    print(f"Available GPUs ({len(gpus)}):")
    for gpu in gpus:
        print(f"  - {gpu}")


def example_3_get_gpu_specs():
    """Example 3: Get GPU specifications."""
    print("\n" + "=" * 70)
    print("Example 3: Get GPU Specifications")
    print("=" * 70)

    client = GPUConfigRecommenderClient()
    specs = client.get_gpu_specs(["H100", "A100-80GB"])

    for spec in specs:
        print(f"\n{spec['name']}:")
        print(f"  Memory: {spec['memory_gb']} GB")
        print(f"  Bandwidth: {spec['memory_bandwidth_gb_s']} GB/s")
        print(f"  FP16 TFLOPS: {spec['tflops_fp16']}")
        print(f"  Cost: ${spec['cost_per_hour']}/hour")


def example_4_validate_model():
    """Example 4: Validate a model."""
    print("\n" + "=" * 70)
    print("Example 4: Validate Model")
    print("=" * 70)

    client = GPUConfigRecommenderClient()
    result = client.validate_model("Qwen/Qwen2.5-7B")

    if result['valid']:
        print(f"✓ Model validated: {result['model_name']}")
        print(f"  Parameters: {result['num_parameters']:.2f}B")
        print(f"  Max sequence length: {result['max_sequence_length']}")
    else:
        print(f"✗ Validation failed: {result['error']}")


def example_5_simple_recommendation():
    """Example 5: Simple recommendation."""
    print("\n" + "=" * 70)
    print("Example 5: Simple Recommendation")
    print("=" * 70)

    client = GPUConfigRecommenderClient()

    # Get GPU specs from library
    gpus = client.get_gpu_specs(["H100", "A100-80GB", "L40"])

    # Get recommendation
    result = client.get_recommendation(
        model_name="Qwen/Qwen2.5-7B",
        available_gpus=gpus
    )

    print(f"Model: {result['model_name']}")
    print(f"Recommended GPU: {result['recommended_gpu']}")

    if result['performance']:
        perf = result['performance']
        print(f"\nPerformance:")
        print(f"  Throughput: {perf['tokens_per_second']:.2f} tokens/sec")
        print(f"  Latency: {perf['intertoken_latency_ms']:.2f} ms/token")
        print(f"  Memory: {perf['memory_required_gb']:.2f} GB")

    print(f"\nReasoning: {result['reasoning']}")
    print(f"\nCompatible GPUs: {len(result['all_compatible_gpus'])}")


def example_6_recommendation_with_constraints():
    """Example 6: Recommendation with latency constraint."""
    print("\n" + "=" * 70)
    print("Example 6: Recommendation with Latency Constraint")
    print("=" * 70)

    client = GPUConfigRecommenderClient()

    # Get GPU specs
    gpus = client.get_gpu_specs(["H100", "A100-80GB", "L40", "L4"])

    # Get recommendation with latency constraint
    result = client.get_recommendation(
        model_name="Qwen/Qwen2.5-7B",
        available_gpus=gpus,
        sequence_length=2048,
        latency_bound_ms=1.0  # Max 1ms per token
    )

    print(f"Model: {result['model_name']}")
    print(f"Latency Constraint: ≤ 1.0 ms/token")
    print(f"Recommended GPU: {result['recommended_gpu']}")

    if result['performance']:
        perf = result['performance']
        print(f"\nPerformance:")
        print(f"  Throughput: {perf['tokens_per_second']:.2f} tokens/sec")
        print(f"  Latency: {perf['intertoken_latency_ms']:.2f} ms/token")

    print(f"\nReasoning: {result['reasoning']}")


def example_7_custom_gpu():
    """Example 7: Recommendation with custom GPU."""
    print("\n" + "=" * 70)
    print("Example 7: Recommendation with Custom GPU")
    print("=" * 70)

    client = GPUConfigRecommenderClient()

    # Define custom GPU
    custom_gpus = [
        {
            "name": "Custom GPU",
            "memory_gb": 100.0,
            "memory_bandwidth_gb_s": 4000.0,
            "tflops_fp16": 2000.0,
            "tflops_fp32": 1000.0,
            "cost_per_hour": 5.0
        }
    ]

    # Get recommendation
    result = client.get_recommendation(
        model_name="Qwen/Qwen2.5-7B",
        available_gpus=custom_gpus
    )

    print(f"Model: {result['model_name']}")
    print(f"Recommended GPU: {result['recommended_gpu']}")

    if result['performance']:
        perf = result['performance']
        print(f"\nPerformance:")
        print(f"  Throughput: {perf['tokens_per_second']:.2f} tokens/sec")
        print(f"  Latency: {perf['intertoken_latency_ms']:.2f} ms/token")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GPU Config Recommender API - Example Client")
    print("=" * 70)

    try:
        example_1_health_check()
        example_2_list_gpus()
        example_3_get_gpu_specs()
        example_4_validate_model()
        example_5_simple_recommendation()
        example_6_recommendation_with_constraints()
        example_7_custom_gpu()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API. Is the server running?")
        print("Start the server with: ./start.sh")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
