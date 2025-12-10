"""Unit tests for GPU recommender."""

import pytest
from config_recommender.models import ModelArchitecture, GPUSpec
from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.recommender import GPURecommender, RecommendationResult


@pytest.fixture
def small_model():
    """Small test model."""
    return ModelArchitecture(
        name="small-7b",
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
        name="large-70b",
        num_parameters=70.0,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        vocab_size=32000,
        max_sequence_length=4096,
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
        GPUSpec(
            name="NVIDIA V100",
            memory_gb=32.0,
            memory_bandwidth_gb_s=900.0,
            tflops_fp16=125.0,
            tflops_fp32=62.5,
            cost_per_hour=2.48,
        ),
        GPUSpec(
            name="NVIDIA T4",
            memory_gb=16.0,
            memory_bandwidth_gb_s=300.0,
            tflops_fp16=65.0,
            tflops_fp32=8.1,
            cost_per_hour=0.526,
        ),
    ]


def test_recommender_initialization():
    """Test recommender initialization."""
    recommender = GPURecommender()
    assert recommender.estimator is not None
    assert recommender.latency_bound_ms is None
    
    recommender_with_latency = GPURecommender(latency_bound_ms=10.0)
    assert recommender_with_latency.latency_bound_ms == 10.0


def test_recommend_gpu_basic(small_model, gpu_fleet):
    """Test basic GPU recommendation."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    assert isinstance(result, RecommendationResult)
    assert result.model_name == "small-7b"
    assert result.recommended_gpu is not None
    assert result.performance is not None
    assert len(result.all_compatible_gpus) > 0
    assert len(result.reasoning) > 0


def test_recommend_gpu_selects_best_performance(small_model, gpu_fleet):
    """Test that recommender selects GPU with best performance."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    # Should select a GPU (model fits in all GPUs with 32GB+)
    assert result.recommended_gpu is not None
    
    # The recommended GPU should have highest tokens/sec among compatible ones
    if len(result.all_compatible_gpus) > 1:
        recommended_perf = result.all_compatible_gpus[0]["tokens_per_second"]
        for gpu_info in result.all_compatible_gpus[1:]:
            assert recommended_perf >= gpu_info["tokens_per_second"]


def test_recommend_gpu_no_compatible(large_model):
    """Test recommendation when no GPU can fit the model."""
    # Small GPUs that can't fit 70B model
    small_gpus = [
        GPUSpec(
            name="NVIDIA T4",
            memory_gb=16.0,
            memory_bandwidth_gb_s=300.0,
            tflops_fp16=65.0,
            tflops_fp32=8.1,
        ),
    ]
    
    recommender = GPURecommender()
    result = recommender.recommend_gpu(large_model, small_gpus)
    
    assert result.recommended_gpu is None
    assert result.performance is None
    assert "No compatible GPU" in result.reasoning


def test_recommend_gpu_with_latency_bound(small_model, gpu_fleet):
    """Test recommendation with latency constraint."""
    # Very strict latency bound that might not be met
    recommender = GPURecommender(latency_bound_ms=0.01)
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    # Either no GPU meets the requirement, or the selected one does
    if result.recommended_gpu:
        assert result.performance.latency_ms_per_token <= 0.01
    else:
        assert "latency requirement" in result.reasoning.lower()


def test_recommend_for_models(small_model, large_model, gpu_fleet):
    """Test recommendation for multiple models."""
    models = [small_model, large_model]
    recommender = GPURecommender()
    results = recommender.recommend_for_models(models, gpu_fleet)
    
    assert len(results) == 2
    assert results[0].model_name == "small-7b"
    assert results[1].model_name == "large-70b"
    
    # Small model should get a recommendation
    assert results[0].recommended_gpu is not None


def test_recommendation_result_to_dict(small_model, gpu_fleet):
    """Test converting recommendation result to dictionary."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert "model_name" in result_dict
    assert "recommended_gpu" in result_dict
    assert "performance" in result_dict
    assert "all_compatible_gpus" in result_dict
    assert "reasoning" in result_dict


def test_recommendation_preserves_memory_info(small_model, gpu_fleet):
    """Test that memory information is included in results."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    for gpu_info in result.all_compatible_gpus:
        assert "memory_required_gb" in gpu_info
        assert "memory_available_gb" in gpu_info
        assert gpu_info["memory_required_gb"] <= gpu_info["memory_available_gb"]


def test_recommendation_includes_cost_info(small_model, gpu_fleet):
    """Test that cost information is included when available."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    for gpu_info in result.all_compatible_gpus:
        assert "cost_per_hour" in gpu_info
        # Cost should match the original GPU spec
        original_gpu = next(g for g in gpu_fleet if g.name == gpu_info["gpu_name"])
        assert gpu_info["cost_per_hour"] == original_gpu.cost_per_hour


def test_recommendation_reasoning_quality(small_model, gpu_fleet):
    """Test that reasoning contains useful information."""
    recommender = GPURecommender()
    result = recommender.recommend_gpu(small_model, gpu_fleet)
    
    reasoning = result.reasoning.lower()
    
    # Should mention the selected GPU
    if result.recommended_gpu:
        assert result.recommended_gpu.lower() in reasoning
        assert "throughput" in reasoning or "tokens" in reasoning
        assert "latency" in reasoning or "ms" in reasoning


def test_custom_sequence_length(small_model, gpu_fleet):
    """Test recommendation with custom sequence length."""
    recommender = GPURecommender()
    
    result_short = recommender.recommend_gpu(small_model, gpu_fleet, sequence_length=512)
    result_long = recommender.recommend_gpu(small_model, gpu_fleet, sequence_length=4096)
    
    # Both should get recommendations (small model)
    assert result_short.recommended_gpu is not None
    assert result_long.recommended_gpu is not None
    
    # Longer sequence needs more memory
    assert result_long.performance.memory_required_gb >= result_short.performance.memory_required_gb
