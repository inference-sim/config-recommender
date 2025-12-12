"""Tests for sequence length UI functionality.

This test validates that the sequence length parameter is properly integrated
into the Streamlit UI and recommendation workflow.
"""

import pytest

from config_recommender import GPURecommender, GPUSpec, ModelArchitecture
from config_recommender.estimator import SyntheticBenchmarkEstimator


@pytest.fixture
def test_model():
    """Create a test model with known max_sequence_length."""
    return ModelArchitecture(
        name="test-7b",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=2048,
    )


@pytest.fixture
def test_gpu():
    """Create a test GPU."""
    return GPUSpec(
        name="NVIDIA A100 80GB",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
        cost_per_hour=3.67,
    )


def test_sequence_length_default(test_model, test_gpu):
    """Test that default sequence length uses model's max_sequence_length."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2, concurrent_users=1)
    recommender = GPURecommender(estimator=estimator)
    
    # When sequence_length is None, should use model's max_sequence_length (2048)
    result = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=None)
    
    assert result.recommended_gpu is not None
    assert result.performance is not None
    # With 2048 tokens and 1 user, KV cache should be relatively small
    kv_cache_default = result.performance.memory_kv_cache_gb
    assert kv_cache_default > 0


def test_sequence_length_custom(test_model, test_gpu):
    """Test that custom sequence length affects KV cache memory."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2, concurrent_users=1)
    recommender = GPURecommender(estimator=estimator)
    
    # Test with custom sequence length (double the default)
    result_custom = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=4096)
    
    assert result_custom.recommended_gpu is not None
    assert result_custom.performance is not None
    kv_cache_custom = result_custom.performance.memory_kv_cache_gb
    assert kv_cache_custom > 0


def test_sequence_length_scaling(test_model, test_gpu):
    """Test that KV cache scales linearly with sequence length."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2, concurrent_users=1)
    recommender = GPURecommender(estimator=estimator)
    
    # Get results for two different sequence lengths
    result_2048 = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=2048)
    result_4096 = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=4096)
    
    kv_2048 = result_2048.performance.memory_kv_cache_gb
    kv_4096 = result_4096.performance.memory_kv_cache_gb
    
    # KV cache should scale linearly with sequence length
    # Doubling sequence length should approximately double KV cache
    ratio = kv_4096 / kv_2048
    assert 1.9 < ratio < 2.1, f"Expected ratio ~2.0, got {ratio}"


def test_sequence_length_affects_total_memory(test_model, test_gpu):
    """Test that sequence length affects total memory requirement."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2, concurrent_users=1)
    recommender = GPURecommender(estimator=estimator)
    
    result_short = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=512)
    result_long = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=8192)
    
    # Total memory should increase with longer sequence
    assert result_long.performance.memory_required_gb > result_short.performance.memory_required_gb
    
    # But weights should remain the same
    assert result_long.performance.memory_weights_gb == result_short.performance.memory_weights_gb


def test_sequence_length_with_recommend_for_models(test_model, test_gpu):
    """Test sequence length works with recommend_for_models."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2, concurrent_users=1)
    recommender = GPURecommender(estimator=estimator)
    
    # Test with custom sequence length for batch recommendation
    results = recommender.recommend_for_models([test_model], [test_gpu], sequence_length=4096)
    
    assert len(results) == 1
    assert results[0].recommended_gpu is not None
    assert results[0].performance.memory_kv_cache_gb > 0


def test_sequence_length_zero_uses_default(test_model, test_gpu):
    """Test that sequence_length=0 or None uses model default."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2, concurrent_users=1)
    recommender = GPURecommender(estimator=estimator)
    
    result_none = recommender.recommend_gpu(test_model, [test_gpu], sequence_length=None)
    result_default = recommender.recommend_gpu(test_model, [test_gpu])
    
    # Both should produce the same results (using model's max_sequence_length)
    assert result_none.performance.memory_kv_cache_gb == result_default.performance.memory_kv_cache_gb
