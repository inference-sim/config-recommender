"""Unit tests for KV cache calculations with max_model_len."""

import pytest

from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.models import GPUSpec, ModelArchitecture
from config_recommender.recommender import GPURecommender


@pytest.fixture
def test_model():
    """Test model for KV cache tests."""
    return ModelArchitecture(
        name="test-model-7b",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=2048,
    )


@pytest.fixture
def test_gpu():
    """Test GPU for KV cache tests."""
    return GPUSpec(
        name="NVIDIA A100",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
        cost_per_hour=3.67,
    )


def test_kv_cache_uses_batch_size_one(test_model):
    """Test that KV cache is calculated with batch_size=1 for single request."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2,
    )
    
    # Calculate KV cache for max sequence length
    kv_cache_gb = estimator.estimate_memory_kv_cache(test_model, 2048)
    
    # Should be calculated for batch_size=1 (single request)
    # Verify by checking against model's internal calculation
    expected_kv_cache = test_model.get_kv_cache_gb(2048, batch_size=1)
    assert kv_cache_gb == expected_kv_cache


def test_kv_cache_increases_with_sequence_length(test_model):
    """Test that KV cache memory increases with sequence length."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2,
    )
    
    # Calculate KV cache for different sequence lengths
    kv_cache_512 = estimator.estimate_memory_kv_cache(test_model, 512)
    kv_cache_2048 = estimator.estimate_memory_kv_cache(test_model, 2048)
    
    # KV cache should scale with sequence length
    assert kv_cache_2048 > kv_cache_512
    # Should be approximately 4x (2048/512)
    assert abs(kv_cache_2048 / kv_cache_512 - 4.0) < 0.1


def test_total_memory_uses_max_model_len(test_model):
    """Test that total memory calculation uses model's max sequence length by default."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2,
    )
    
    # Get total memory without specifying sequence length
    memory = estimator.estimate_total_memory(test_model)
    
    # Should use model's max_sequence_length (2048)
    expected_kv_cache = test_model.get_kv_cache_gb(2048, batch_size=1)
    
    # KV cache should match expected
    assert memory["kv_cache_gb"] == expected_kv_cache
    
    # Total memory should be (weights + kv_cache) * overhead
    expected_total = (memory["weights_gb"] + expected_kv_cache) * estimator.memory_overhead_factor
    assert abs(memory["total_gb"] - expected_total) < 0.01


def test_performance_with_max_model_len(test_model, test_gpu):
    """Test that performance estimates use max_model_len for KV cache."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2,
    )
    
    # Get performance estimate using model's max sequence length
    perf = estimator.estimate_performance(test_model, test_gpu, 2048)
    
    # KV cache should be for batch_size=1
    expected_kv_cache = test_model.get_kv_cache_gb(2048, batch_size=1)
    assert abs(perf.memory_kv_cache_gb - expected_kv_cache) < 0.01


def test_memory_scales_with_sequence_length_not_batch(test_model):
    """Test that memory scales with sequence length, not batch size."""
    estimator = SyntheticBenchmarkEstimator(
        precision_bytes=2,
    )
    
    memory_512 = estimator.estimate_total_memory(test_model, sequence_length=512)
    memory_2048 = estimator.estimate_total_memory(test_model, sequence_length=2048)
    
    # Total memory should increase with sequence length
    assert memory_2048["total_gb"] > memory_512["total_gb"]
    # KV cache should scale with sequence length
    assert memory_2048["kv_cache_gb"] > memory_512["kv_cache_gb"]
    # Weights should remain the same
    assert memory_2048["weights_gb"] == memory_512["weights_gb"]
