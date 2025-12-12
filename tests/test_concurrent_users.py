"""Unit tests for concurrent users functionality."""

import pytest

from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.models import GPUSpec, ModelArchitecture
from config_recommender.recommender import GPURecommender


@pytest.fixture
def test_model():
    """Test model for concurrent users tests."""
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
    """Test GPU for concurrent users tests."""
    return GPUSpec(
        name="NVIDIA A100",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
        cost_per_hour=3.67,
    )


def test_estimator_with_concurrent_users():
    """Test that estimator accepts concurrent_users parameter."""
    estimator = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=10,
    )
    assert estimator.concurrent_users == 10


def test_estimator_default_concurrent_users():
    """Test that concurrent_users defaults to 1."""
    estimator = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
    )
    assert estimator.concurrent_users == 1


def test_kv_cache_increases_with_concurrent_users(test_model):
    """Test that KV cache memory increases with concurrent users."""
    # Estimator with 1 concurrent user
    estimator_1 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=1,
    )
    
    # Estimator with 10 concurrent users
    estimator_10 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=10,
    )
    
    # Calculate KV cache for both
    kv_cache_1 = estimator_1.estimate_memory_kv_cache(test_model, 2048)
    kv_cache_10 = estimator_10.estimate_memory_kv_cache(test_model, 2048)
    
    # KV cache should scale with concurrent users
    assert kv_cache_10 > kv_cache_1
    # Should be approximately 10x (allowing for floating point precision)
    assert abs(kv_cache_10 / kv_cache_1 - 10.0) < 0.1


def test_total_memory_increases_with_concurrent_users(test_model):
    """Test that total memory increases with concurrent users."""
    estimator_1 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=1,
    )
    
    estimator_5 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=5,
    )
    
    memory_1 = estimator_1.estimate_total_memory(test_model, 2048)
    memory_5 = estimator_5.estimate_total_memory(test_model, 2048)
    
    # Total memory should increase with concurrent users
    assert memory_5["total_gb"] > memory_1["total_gb"]
    # KV cache should scale with concurrent users
    assert memory_5["kv_cache_gb"] > memory_1["kv_cache_gb"]
    # Weights should remain the same
    assert memory_5["weights_gb"] == memory_1["weights_gb"]


def test_performance_with_concurrent_users(test_model, test_gpu):
    """Test that performance estimates change with concurrent users."""
    estimator_1 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=1,
    )
    
    estimator_10 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=10,
    )
    
    perf_1 = estimator_1.estimate_performance(test_model, test_gpu, 2048)
    perf_10 = estimator_10.estimate_performance(test_model, test_gpu, 2048)
    
    # With more concurrent users, memory required increases
    assert perf_10.memory_required_gb > perf_1.memory_required_gb
    # KV cache should increase
    assert perf_10.memory_kv_cache_gb > perf_1.memory_kv_cache_gb


def test_concurrent_users_may_require_tensor_parallelism(test_model):
    """Test that high concurrent users may require tensor parallelism."""
    # Create a GPU with limited memory
    small_gpu = GPUSpec(
        name="NVIDIA V100",
        memory_gb=32.0,
        memory_bandwidth_gb_s=900.0,
        tflops_fp16=125.0,
        tflops_fp32=62.5,
        cost_per_hour=2.48,
    )
    
    # With 1 concurrent user, model should fit
    estimator_1 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=1,
    )
    
    recommender_1 = GPURecommender(estimator=estimator_1)
    result_1 = recommender_1.recommend_gpu(test_model, [small_gpu], sequence_length=2048)
    
    # With many concurrent users, model may not fit in single GPU
    estimator_100 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=100,
    )
    
    recommender_100 = GPURecommender(estimator=estimator_100)
    result_100 = recommender_100.recommend_gpu(test_model, [small_gpu], sequence_length=2048)
    
    # With low concurrent users, should have lower or equal TP size compared to high concurrent users
    # The key is that higher concurrent users leads to higher memory requirements
    if result_1.performance and result_100.performance:
        # If both fit, high concurrent users should require higher or equal TP
        if result_1.performance.fits_in_memory and result_100.performance.fits_in_memory:
            assert result_100.performance.tensor_parallel_size >= result_1.performance.tensor_parallel_size


def test_concurrent_users_batch_size_independence(test_model):
    """Test that concurrent_users and batch_size are independent."""
    # batch_size is for processing, concurrent_users is for KV cache
    estimator = SyntheticBenchmarkEstimator(
        batch_size=4,
        precision_bytes=2,
        concurrent_users=10,
    )
    
    # Both should be stored independently
    assert estimator.batch_size == 4
    assert estimator.concurrent_users == 10
    
    # KV cache should use concurrent_users
    kv_cache = estimator.estimate_memory_kv_cache(test_model, 2048)
    
    # Create another estimator with only concurrent_users set
    estimator2 = SyntheticBenchmarkEstimator(
        batch_size=1,
        precision_bytes=2,
        concurrent_users=10,
    )
    
    kv_cache2 = estimator2.estimate_memory_kv_cache(test_model, 2048)
    
    # KV cache should be the same (uses concurrent_users, not batch_size)
    assert kv_cache == kv_cache2
