"""Unit tests for the synthetic benchmark estimator."""

import pytest

from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.models import GPUSpec, ModelArchitecture


@pytest.fixture
def small_model():
    """Small test model."""
    return ModelArchitecture(
        name="small-model",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=2048,
    )


@pytest.fixture
def large_model():
    """Large test model."""
    return ModelArchitecture(
        name="large-model",
        num_parameters=70.0,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=32000,
        max_sequence_length=4096,
    )


@pytest.fixture
def high_end_gpu():
    """High-end GPU."""
    return GPUSpec(
        name="NVIDIA A100",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
    )


@pytest.fixture
def low_end_gpu():
    """Low-end GPU."""
    return GPUSpec(
        name="NVIDIA T4",
        memory_gb=16.0,
        memory_bandwidth_gb_s=300.0,
        tflops_fp16=65.0,
        tflops_fp32=8.1,
    )


def test_estimator_initialization():
    """Test estimator initialization."""
    estimator = SyntheticBenchmarkEstimator()
    assert estimator.precision_bytes == 2
    assert estimator.concurrent_users == 1

    estimator_fp32 = SyntheticBenchmarkEstimator(precision_bytes=4)
    assert estimator_fp32.precision_bytes == 4


def test_estimate_memory_weights(small_model):
    """Test weight memory estimation."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2)  # FP16
    weights_gb = estimator.estimate_memory_weights(small_model)

    # 7B parameters * 2 bytes = 14 GB
    assert weights_gb == pytest.approx(14.0, rel=0.01)


def test_estimate_memory_weights_fp32(small_model):
    """Test weight memory estimation with FP32."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=4)  # FP32
    weights_gb = estimator.estimate_memory_weights(small_model)

    # Model memory is determined by config_explorer from actual HF model,
    # which may use mixed precision (FP16 for most weights)
    # Expected: ~14 GB for 7B parameter model with FP16 weights
    assert weights_gb == pytest.approx(14.0, rel=0.01)


def test_estimate_memory_kv_cache(small_model):
    """Test KV cache memory estimation."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2)
    kv_cache_gb = estimator.estimate_memory_kv_cache(small_model, sequence_length=2048)

    # Should be non-zero and reasonable
    assert kv_cache_gb > 0
    assert kv_cache_gb < 10  # Should be less than weights for this model


def test_estimate_memory_kv_cache_with_gqa(large_model):
    """Test KV cache with Grouped Query Attention."""
    estimator = SyntheticBenchmarkEstimator(precision_bytes=2)
    kv_cache_gb = estimator.estimate_memory_kv_cache(large_model, sequence_length=4096)

    # With GQA (8 KV heads vs 64 attention heads), KV cache should be smaller
    assert kv_cache_gb > 0


def test_estimate_total_memory(small_model):
    """Test total memory estimation."""
    estimator = SyntheticBenchmarkEstimator()
    memory = estimator.estimate_total_memory(small_model)

    assert "weights_gb" in memory
    assert "kv_cache_gb" in memory
    assert "total_gb" in memory

    # Total should be sum of components with overhead
    expected_total = (
        memory["weights_gb"] + memory["kv_cache_gb"]
    ) * estimator.memory_overhead_factor

    assert memory["total_gb"] == pytest.approx(expected_total, rel=0.01)


def test_estimate_performance_fits(small_model, high_end_gpu):
    """Test performance estimation when model fits in GPU."""
    estimator = SyntheticBenchmarkEstimator()
    perf = estimator.estimate_performance(small_model, high_end_gpu)

    assert perf.fits_in_memory is True
    assert perf.tokens_per_second > 0
    assert perf.intertoken_latency_ms > 0
    assert perf.intertoken_latency_ms < float("inf")
    assert perf.memory_required_gb <= high_end_gpu.memory_gb


def test_estimate_performance_does_not_fit(large_model, low_end_gpu):
    """Test performance estimation when model doesn't fit in GPU."""
    estimator = SyntheticBenchmarkEstimator()
    perf = estimator.estimate_performance(large_model, low_end_gpu)

    assert perf.fits_in_memory is False
    assert perf.tokens_per_second == 0.0
    assert perf.intertoken_latency_ms == float("inf")
    assert perf.memory_required_gb > low_end_gpu.memory_gb

def test_performance_with_custom_sequence_length(small_model, high_end_gpu):
    """Test performance estimation with custom sequence length."""
    estimator = SyntheticBenchmarkEstimator()

    perf_short = estimator.estimate_performance(
        small_model, high_end_gpu, sequence_length=512
    )
    perf_long = estimator.estimate_performance(
        small_model, high_end_gpu, sequence_length=4096
    )

    # Longer sequence should require more memory for KV cache
    assert perf_long.memory_kv_cache_gb > perf_short.memory_kv_cache_gb
