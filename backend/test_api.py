#!/usr/bin/env python3
"""Simple test script to verify the API endpoints work correctly.

This script tests all major endpoints of the GPU Config Recommender API.
Run this after starting the backend server to verify everything is working.

Usage:
    python test_api.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
from typing import Dict, Any

import requests


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_result(success: bool, message: str):
    """Print a test result."""
    status = "✓" if success else "✗"
    print(f"{status} {message}")


def test_health(base_url: str) -> bool:
    """Test the health check endpoint."""
    print_section("Testing Health Check")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        response.raise_for_status()
        data = response.json()

        success = data.get("status") == "healthy"
        print_result(success, f"Health check: {data}")
        return success
    except Exception as e:
        print_result(False, f"Health check failed: {e}")
        return False


def test_list_gpus(base_url: str) -> bool:
    """Test listing available GPU keys."""
    print_section("Testing List Available GPUs")
    try:
        response = requests.get(f"{base_url}/api/gpu-library/available", timeout=5)
        response.raise_for_status()
        gpus = response.json()

        success = isinstance(gpus, list) and len(gpus) > 0
        print_result(success, f"Found {len(gpus)} GPUs: {gpus}")
        return success
    except Exception as e:
        print_result(False, f"List GPUs failed: {e}")
        return False


def test_get_gpu_specs(base_url: str) -> bool:
    """Test getting GPU specifications."""
    print_section("Testing Get GPU Specs")
    try:
        payload = {"gpu_keys": ["H100", "A100-80GB"]}
        response = requests.post(
            f"{base_url}/api/gpu-library",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        specs = response.json()

        success = isinstance(specs, list) and len(specs) == 2
        print_result(success, f"Retrieved {len(specs)} GPU specs")

        if success:
            for spec in specs:
                print(f"  - {spec['name']}: {spec['memory_gb']}GB, "
                      f"{spec['tflops_fp16']} TFLOPS FP16")
        return success
    except Exception as e:
        print_result(False, f"Get GPU specs failed: {e}")
        return False


def test_validate_model(base_url: str) -> bool:
    """Test model validation endpoint."""
    print_section("Testing Model Validation")
    try:
        payload = {"model_name": "Qwen/Qwen2.5-7B"}
        response = requests.post(
            f"{base_url}/api/models/validate",
            json=payload,
            timeout=30  # Model validation can take longer
        )
        response.raise_for_status()
        result = response.json()

        success = result.get("valid", False)
        if success:
            print_result(True, f"Model validated: {result['model_name']}")
            print(f"    Parameters: {result.get('num_parameters', 'N/A')}B")
            print(f"    Max sequence length: {result.get('max_sequence_length', 'N/A')}")
        else:
            print_result(False, f"Model validation failed: {result.get('error', 'Unknown error')}")

        return success
    except Exception as e:
        print_result(False, f"Model validation failed: {e}")
        return False


def test_recommendation(base_url: str) -> bool:
    """Test the recommendation endpoint."""
    print_section("Testing GPU Recommendation")
    try:
        payload = {
            "model": {"name": "Qwen/Qwen2.5-7B"},
            "available_gpus": [
                {
                    "name": "NVIDIA H100 80GB",
                    "memory_gb": 80.0,
                    "memory_bandwidth_gb_s": 3350.0,
                    "tflops_fp16": 1979.0,
                    "tflops_fp32": 989.0,
                    "cost_per_hour": 4.76
                },
                {
                    "name": "NVIDIA A100 80GB",
                    "memory_gb": 80.0,
                    "memory_bandwidth_gb_s": 2039.0,
                    "tflops_fp16": 312.0,
                    "tflops_fp32": 156.0,
                    "cost_per_hour": 3.67
                }
            ],
            "sequence_length": 2048
        }

        response = requests.post(
            f"{base_url}/api/recommendations",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        success = result.get("recommended_gpu") is not None
        if success:
            print_result(True, f"Recommendation received")
            print(f"    Model: {result['model_name']}")
            print(f"    Recommended GPU: {result['recommended_gpu']}")

            if result.get('performance'):
                perf = result['performance']
                print(f"    Throughput: {perf['tokens_per_second']:.2f} tokens/sec")
                print(f"    Latency: {perf['intertoken_latency_ms']:.2f} ms/token")
                print(f"    Memory: {perf['memory_required_gb']:.2f} GB")

            print(f"    Reasoning: {result['reasoning'][:100]}...")
            print(f"    Compatible GPUs: {len(result['all_compatible_gpus'])}")
        else:
            print_result(False, f"No GPU recommendation returned")

        return success
    except Exception as e:
        print_result(False, f"Recommendation failed: {e}")
        return False


def main():
    """Run all API tests."""
    parser = argparse.ArgumentParser(description="Test GPU Config Recommender API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip('/')

    print(f"\nTesting API at: {base_url}")

    # Run all tests
    tests = [
        ("Health Check", test_health),
        ("List GPUs", test_list_gpus),
        ("Get GPU Specs", test_get_gpu_specs),
        ("Validate Model", test_validate_model),
        ("GPU Recommendation", test_recommendation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func(base_url)
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            sys.exit(1)

    # Print summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        print_result(result, name)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
