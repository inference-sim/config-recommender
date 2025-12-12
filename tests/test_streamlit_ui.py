"""Tests for Streamlit UI functionality.

These tests validate the core components and data flow of the Streamlit application
without requiring a full browser/UI test environment.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config_recommender import GPURecommender, GPUSpec, ModelArchitecture
from config_recommender.estimator import SyntheticBenchmarkEstimator


@pytest.fixture
def sample_models():
    """Sample models for testing."""
    return [
        ModelArchitecture(
            name="test-7b",
            num_parameters=7.0,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_sequence_length=2048,
        ),
    ]


@pytest.fixture
def sample_gpus():
    """Sample GPUs for testing."""
    return [
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


def test_estimator_configuration():
    """Test that estimator can be configured with custom parameters."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2,
        memory_overhead_factor=1.2,
        concurrent_users=1,
    )

    assert estimator.precision_bytes == 2
    assert estimator.memory_overhead_factor == 1.2
    assert estimator.concurrent_users == 1


def test_recommendations_generation(sample_models, sample_gpus):
    """Test that recommendations can be generated successfully."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2, memory_overhead_factor=1.2, concurrent_users=1
    )
    recommender = GPURecommender(estimator=estimator)

    results = recommender.recommend_for_models(sample_models, sample_gpus)

    assert len(results) == len(sample_models)
    assert all(hasattr(r, "model_name") for r in results)
    assert all(hasattr(r, "recommended_gpu") for r in results)
    assert all(hasattr(r, "performance") for r in results)
    assert all(hasattr(r, "reasoning") for r in results)


def test_recommendations_with_latency_bound(sample_models, sample_gpus):
    """Test recommendations with latency constraint."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2, memory_overhead_factor=1.2, concurrent_users=1
    )
    recommender = GPURecommender(estimator=estimator, latency_bound_ms=10.0)

    results = recommender.recommend_for_models(sample_models, sample_gpus)

    assert len(results) == len(sample_models)
    # If a GPU is recommended, it should meet the latency bound
    for result in results:
        if result.performance and result.recommended_gpu:
            assert result.performance.intertoken_latency_ms <= 10.0


def test_recommendation_to_dict(sample_models, sample_gpus):
    """Test that recommendations can be serialized to dict."""
    estimator = SyntheticBenchmarkEstimator()
    recommender = GPURecommender(estimator=estimator)

    results = recommender.recommend_for_models(sample_models, sample_gpus)

    for result in results:
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "model_name" in result_dict
        assert "recommended_gpu" in result_dict
        assert "reasoning" in result_dict
        assert "all_compatible_gpus" in result_dict


def test_export_to_json(sample_models, sample_gpus):
    """Test that recommendations can be exported to JSON."""
    estimator = SyntheticBenchmarkEstimator()
    recommender = GPURecommender(estimator=estimator)

    results = recommender.recommend_for_models(sample_models, sample_gpus)

    # Convert to JSON-serializable format
    json_data = {"recommendations": [rec.to_dict() for rec in results]}
    json_str = json.dumps(json_data, indent=2)

    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert "recommendations" in parsed
    assert len(parsed["recommendations"]) == len(results)


def test_export_to_dataframe(sample_models, sample_gpus):
    """Test that recommendations can be converted to DataFrame for CSV export."""
    estimator = SyntheticBenchmarkEstimator()
    recommender = GPURecommender(estimator=estimator)

    results = recommender.recommend_for_models(sample_models, sample_gpus)

    # Create DataFrame similar to what the UI does
    table_data = []
    for rec in results:
        row = {
            "Model": rec.model_name,
            "Recommended GPU": rec.recommended_gpu or "None",
            "Throughput (tok/s)": (
                f"{rec.performance.tokens_per_second:.1f}" if rec.performance else "N/A"
            ),
            "Latency (ms)": (
                f"{rec.performance.intertoken_latency_ms:.2f}" if rec.performance else "N/A"
            ),
            "Memory (GB)": f"{rec.performance.memory_required_gb:.1f}" if rec.performance else "N/A",
            "Fits": "✅" if rec.performance and rec.performance.fits_in_memory else "❌",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Verify DataFrame structure
    assert len(df) == len(results)
    assert "Model" in df.columns
    assert "Recommended GPU" in df.columns
    assert "Throughput (tok/s)" in df.columns
    assert "Latency (ms)" in df.columns


def test_multiple_models_recommendations(sample_gpus):
    """Test recommendations for multiple models."""
    models = [
        ModelArchitecture(
            name="small-1b",
            num_parameters=1.0,
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=32000,
            max_sequence_length=2048,
        ),
        ModelArchitecture(
            name="medium-7b",
            num_parameters=7.0,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            max_sequence_length=2048,
        ),
    ]

    estimator = SyntheticBenchmarkEstimator()
    recommender = GPURecommender(estimator=estimator)

    results = recommender.recommend_for_models(models, sample_gpus)

    assert len(results) == 2
    # All models should fit on the provided GPUs
    assert all(r.recommended_gpu is not None for r in results)
    assert all(r.performance is not None for r in results)


def test_concurrent_users_impact(sample_models, sample_gpus):
    """Test that concurrent users parameter affects memory requirements."""
    # Test with 1 user
    estimator_1user = SyntheticBenchmarkEstimator(concurrent_users=1)
    recommender_1user = GPURecommender(estimator=estimator_1user)
    results_1user = recommender_1user.recommend_for_models(sample_models, sample_gpus)

    # Test with 10 users
    estimator_10users = SyntheticBenchmarkEstimator(concurrent_users=10)
    recommender_10users = GPURecommender(estimator=estimator_10users)
    results_10users = recommender_10users.recommend_for_models(sample_models, sample_gpus)

    # Memory should be higher with more concurrent users
    mem_1user = results_1user[0].performance.memory_required_gb
    mem_10users = results_10users[0].performance.memory_required_gb

    assert mem_10users > mem_1user, "More concurrent users should require more memory"


def test_filter_and_sort_logic(sample_models, sample_gpus):
    """Test the filtering and sorting logic used in the UI."""
    estimator = SyntheticBenchmarkEstimator()
    recommender = GPURecommender(estimator=estimator)

    # Add more models for testing
    models = sample_models + [
        ModelArchitecture(
            name="large-70b",
            num_parameters=70.0,
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            vocab_size=32000,
            max_sequence_length=4096,
        ),
    ]

    results = recommender.recommend_for_models(models, sample_gpus)

    # Create DataFrame
    table_data = []
    for rec in results:
        row = {
            "Model": rec.model_name,
            "Recommended GPU": rec.recommended_gpu or "None",
            "Throughput": rec.performance.tokens_per_second if rec.performance else 0,
            "Latency": rec.performance.intertoken_latency_ms if rec.performance else 0,
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Test sorting by throughput
    sorted_df = df.sort_values(by="Throughput", ascending=False)
    assert sorted_df.iloc[0]["Throughput"] >= sorted_df.iloc[-1]["Throughput"]

    # Test filtering
    filtered_df = df[df["Recommended GPU"] != "None"]
    assert all(filtered_df["Recommended GPU"] != "None")
