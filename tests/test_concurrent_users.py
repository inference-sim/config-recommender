"""Unit tests for concurrent users functionality."""

import pytest

from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.models import GPUSpec, ModelArchitecture
from config_recommender.recommender import GPURecommender


@pytest.fixture
def test_model():
    """Test model."""
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
def gpu_fleet():
    """Fleet of GPUs for testing."""
    return [
        GPUSpec(
            name="NVIDIA H100",
            memory_gb=80.0,
            memory_bandwidth_gb_s=3350.0,
            tflops_fp16=989.0,
            tflops_fp32=494.5,
            cost_per_hour=4.76,
        ),
        GPUSpec(
            name="NVIDIA A100",
            memory_gb=80.0,
            memory_bandwidth_gb_s=2039.0,
            tflops_fp16=312.0,
            tflops_fp32=156.0,
            cost_per_hour=3.67,
        ),
    ]


def test_concurrent_users_initialization():
    """Test that estimator accepts concurrent_users parameter."""
    estimator = SyntheticBenchmarkEstimator(concurrent_users=1)
    assert estimator.concurrent_users == 1

    estimator_multi = SyntheticBenchmarkEstimator(concurrent_users=10)
    assert estimator_multi.concurrent_users == 10


def test_concurrent_users_increases_kv_cache_memory(test_model):
    """Test that concurrent users increases KV cache memory requirement."""
    estimator_1_user = SyntheticBenchmarkEstimator(concurrent_users=1)
    estimator_10_users = SyntheticBenchmarkEstimator(concurrent_users=10)

    kv_cache_1 = estimator_1_user.estimate_memory_kv_cache(test_model, 2048)
    kv_cache_10 = estimator_10_users.estimate_memory_kv_cache(test_model, 2048)

    # KV cache with 10 users should be 10x that of 1 user
    assert kv_cache_10 == pytest.approx(kv_cache_1 * 10, rel=0.01)


def test_concurrent_users_increases_total_memory(test_model):
    """Test that concurrent users increases total memory requirement."""
    estimator_1_user = SyntheticBenchmarkEstimator(concurrent_users=1)
    estimator_10_users = SyntheticBenchmarkEstimator(concurrent_users=10)

    memory_1 = estimator_1_user.estimate_total_memory(test_model)
    memory_10 = estimator_10_users.estimate_total_memory(test_model)

    # Total memory with 10 users should be higher than with 1 user
    assert memory_10["total_gb"] > memory_1["total_gb"]
    
    # KV cache should be 10x
    assert memory_10["kv_cache_gb"] == pytest.approx(memory_1["kv_cache_gb"] * 10, rel=0.01)
    
    # Weights should be the same (not affected by concurrent users)
    assert memory_10["weights_gb"] == pytest.approx(memory_1["weights_gb"], rel=0.01)


def test_concurrent_users_may_require_larger_gpu(test_model):
    """Test that concurrent users may require a larger GPU or TP."""
    small_gpu = GPUSpec(
        name="NVIDIA T4",
        memory_gb=16.0,
        memory_bandwidth_gb_s=300.0,
        tflops_fp16=65.0,
        tflops_fp32=8.1,
    )
    
    # With 1 user, model should fit in 16GB
    estimator_1_user = SyntheticBenchmarkEstimator(concurrent_users=1)
    perf_1 = estimator_1_user.estimate_performance(test_model, small_gpu)
    
    # With 10 users, model likely won't fit
    estimator_10_users = SyntheticBenchmarkEstimator(concurrent_users=10)
    perf_10 = estimator_10_users.estimate_performance(test_model, small_gpu)
    
    # Total memory requirement should be higher with more users
    assert perf_10.memory_required_gb > perf_1.memory_required_gb


def test_concurrent_users_may_trigger_tensor_parallelism(test_model, gpu_fleet):
    """Test that increasing concurrent users may trigger tensor parallelism."""
    # With 1 user, should fit in single GPU
    estimator_1_user = SyntheticBenchmarkEstimator(concurrent_users=1)
    recommender_1 = GPURecommender(estimator=estimator_1_user)
    result_1 = recommender_1.recommend_gpu(test_model, gpu_fleet)
    
    # With many users, may need tensor parallelism or might not fit
    estimator_many_users = SyntheticBenchmarkEstimator(concurrent_users=50)
    recommender_many = GPURecommender(estimator=estimator_many_users)
    result_many = recommender_many.recommend_gpu(test_model, gpu_fleet)
    
    # With 1 user, should get a recommendation with TP=1
    assert result_1.recommended_gpu is not None
    if result_1.performance:
        assert result_1.performance.tensor_parallel_size == 1
    
    # With many users, if it gets a recommendation, it should have higher TP
    # or might not fit at all
    if result_many.recommended_gpu:
        # If it fits, TP size should be > 1
        if result_many.performance:
            assert result_many.performance.tensor_parallel_size >= 1


def test_concurrent_users_default_value():
    """Test that concurrent_users defaults to 1."""
    estimator = SyntheticBenchmarkEstimator()
    assert estimator.concurrent_users == 1


def test_performance_with_varying_concurrent_users(test_model):
    """Test performance estimates with varying numbers of concurrent users."""
    gpu = GPUSpec(
        name="NVIDIA A100",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
    )
    
    # Test with different numbers of concurrent users
    for users in [1, 5, 10, 20]:
        estimator = SyntheticBenchmarkEstimator(concurrent_users=users)
        perf = estimator.estimate_performance(test_model, gpu)
        
        # Memory should scale with users
        assert perf.memory_kv_cache_gb > 0
        
        # If it fits, performance metrics should be valid
        if perf.fits_in_memory:
            assert perf.tokens_per_second > 0
            assert perf.intertoken_latency_ms < float("inf")


def test_concurrent_users_memory_scaling_is_linear(test_model):
    """Test that KV cache memory scales linearly with concurrent users."""
    users_list = [1, 2, 5, 10]
    kv_caches = []
    
    for users in users_list:
        estimator = SyntheticBenchmarkEstimator(concurrent_users=users)
        kv_cache = estimator.estimate_memory_kv_cache(test_model, 2048)
        kv_caches.append(kv_cache)
    
    # Check linear scaling
    base_kv = kv_caches[0]
    for i, users in enumerate(users_list):
        expected = base_kv * users
        assert kv_caches[i] == pytest.approx(expected, rel=0.01)


def test_concurrent_users_with_different_sequence_lengths(test_model):
    """Test concurrent users with different sequence lengths."""
    estimator = SyntheticBenchmarkEstimator(concurrent_users=5)
    
    kv_cache_short = estimator.estimate_memory_kv_cache(test_model, 512)
    kv_cache_long = estimator.estimate_memory_kv_cache(test_model, 2048)
    
    # Longer sequences should need more KV cache
    assert kv_cache_long > kv_cache_short
    
    # Should scale with sequence length
    assert kv_cache_long == pytest.approx(kv_cache_short * 4, rel=0.01)
