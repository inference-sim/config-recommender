"""Unit tests for tensor parallelism recommendations."""

import pytest

from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.models import GPUSpec, ModelArchitecture
from config_recommender.recommender import GPURecommender


@pytest.fixture
def very_large_model():
    """Very large model that doesn't fit on a single GPU."""
    return ModelArchitecture(
        name="large-mixtral",
        num_parameters=50.0,  # 50B parameters (~100GB in FP16)
        num_layers=60,
        hidden_size=8192,
        num_attention_heads=64,
        vocab_size=32000,
        max_sequence_length=2048,
    )


@pytest.fixture
def h100_gpu():
    """H100 GPU spec."""
    return GPUSpec(
        name="NVIDIA H100 80GB",
        memory_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        tflops_fp16=989.0,
        tflops_fp32=494.5,
        cost_per_hour=4.76,
    )


def test_tp_performance_estimation(very_large_model, h100_gpu):
    """Test that TP performance estimation works correctly."""
    estimator = SyntheticBenchmarkEstimator()
    
    # Single GPU - should not fit
    perf_single = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=1
    )
    assert not perf_single.fits_in_memory
    assert perf_single.tensor_parallel_size == 1
    
    # TP=2 - should fit with less memory per GPU
    perf_tp2 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=2
    )
    assert perf_tp2.fits_in_memory
    assert perf_tp2.tensor_parallel_size == 2
    assert perf_tp2.memory_required_gb < perf_single.memory_required_gb
    
    # TP=4 - should fit with even less memory per GPU
    perf_tp4 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=4
    )
    assert perf_tp4.fits_in_memory
    assert perf_tp4.tensor_parallel_size == 4
    assert perf_tp4.memory_required_gb < perf_tp2.memory_required_gb


def test_tp_throughput_scales(very_large_model, h100_gpu):
    """Test that throughput increases with TP (despite overhead)."""
    estimator = SyntheticBenchmarkEstimator()
    
    perf_tp2 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=2
    )
    perf_tp4 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=4
    )
    perf_tp8 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=8
    )
    
    # Higher TP should have higher throughput (even with overhead)
    # But not perfectly linear due to overhead
    assert perf_tp4.tokens_per_second > perf_tp2.tokens_per_second
    assert perf_tp8.tokens_per_second > perf_tp4.tokens_per_second


def test_tp_overhead_reduces_performance(very_large_model, h100_gpu):
    """Test that TP overhead reduces performance compared to ideal scaling."""
    estimator = SyntheticBenchmarkEstimator()
    
    perf_tp2 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=2
    )
    perf_tp8 = estimator.estimate_performance(
        very_large_model, h100_gpu, tensor_parallel_size=8
    )
    
    # Due to overhead, TP=8 should be less than 4x TP=2
    # With 5% overhead per rank: TP=2 has 0.95x efficiency, TP=8 has 0.65x efficiency
    # So ratio should be less than 4.0
    ratio = perf_tp8.tokens_per_second / perf_tp2.tokens_per_second
    assert ratio < 4.0
    assert ratio > 2.0  # But still provides some benefit


def test_recommend_tp_for_large_model(very_large_model, h100_gpu):
    """Test that recommender suggests TP for large models."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(very_large_model, [h100_gpu])
    
    # Should get a recommendation with TP
    assert result.recommended_gpu is not None
    assert result.performance is not None
    assert result.performance.tensor_parallel_size > 1
    assert result.performance.fits_in_memory
    
    # Reasoning should mention TP
    assert "TP=" in result.reasoning or "x " in result.reasoning


def test_recommend_best_tp_value(very_large_model, h100_gpu):
    """Test that recommender selects the best TP value."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(very_large_model, [h100_gpu])
    
    # Should recommend the TP value with highest throughput
    assert result.performance is not None
    
    # All compatible configs should be in results
    assert len(result.all_compatible_gpus) >= 2  # At least TP=2 and another
    
    # First (recommended) should have highest throughput
    recommended_throughput = result.all_compatible_gpus[0]["tokens_per_second"]
    for config in result.all_compatible_gpus[1:]:
        assert recommended_throughput >= config["tokens_per_second"]


def test_tp_values_in_range(very_large_model, h100_gpu):
    """Test that TP values are in the expected range (2, 4, 8)."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(very_large_model, [h100_gpu])
    
    # All TP values should be in the range [2, 4, 8]
    for config in result.all_compatible_gpus:
        tp_size = config.get("tensor_parallel_size", 1)
        assert tp_size in [2, 4, 8]


def test_model_fits_single_gpu_no_tp(h100_gpu):
    """Test that small models that fit on single GPU don't use TP."""
    small_model = ModelArchitecture(
        name="small-7b",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=2048,
    )
    
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, [h100_gpu])
    
    # Should recommend single GPU (TP=1)
    assert result.performance.tensor_parallel_size == 1
    assert "TP=" not in result.reasoning


def test_tp_json_serialization(very_large_model, h100_gpu):
    """Test that TP information is included in JSON output."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(very_large_model, [h100_gpu])
    
    result_dict = result.to_dict()
    
    # Performance should include tensor_parallel_size
    assert "performance" in result_dict
    assert result_dict["performance"] is not None
    assert "tensor_parallel_size" in result_dict["performance"]
    assert result_dict["performance"]["tensor_parallel_size"] > 1
    
    # All compatible GPUs should include tensor_parallel_size
    for config in result_dict["all_compatible_gpus"]:
        assert "tensor_parallel_size" in config


def test_extremely_large_model_no_tp_fits():
    """Test behavior when even with TP=8, model doesn't fit."""
    huge_model = ModelArchitecture(
        name="huge-model",
        num_parameters=500.0,  # 500B parameters
        num_layers=120,
        hidden_size=16384,
        num_attention_heads=128,
        vocab_size=32000,
        max_sequence_length=2048,
    )
    
    small_gpu = GPUSpec(
        name="NVIDIA T4 16GB",
        memory_gb=16.0,
        memory_bandwidth_gb_s=300.0,
        tflops_fp16=65.0,
        tflops_fp32=8.1,
    )
    
    recommender = GPURecommender()
    result = recommender.recommend_gpu(huge_model, [small_gpu])
    
    # Should not find a recommendation
    assert result.recommended_gpu is None
    assert result.performance is None
    assert "tensor parallelism" in result.reasoning.lower()
