"""Unit tests for data models."""

import pytest
from config_recommender.models import ModelArchitecture, GPUSpec


def test_model_architecture_creation():
    """Test creating a ModelArchitecture."""
    model = ModelArchitecture(
        name="test-model",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        max_sequence_length=2048,
    )
    
    assert model.name == "test-model"
    assert model.num_parameters == 7.0
    assert model.num_layers == 32
    assert model.num_kv_heads == 32  # Should default to num_attention_heads


def test_model_architecture_with_gqa():
    """Test ModelArchitecture with Grouped Query Attention."""
    model = ModelArchitecture(
        name="gqa-model",
        num_parameters=7.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=32000,
    )
    
    assert model.num_kv_heads == 8
    assert model.num_kv_heads != model.num_attention_heads


def test_gpu_spec_creation():
    """Test creating a GPUSpec."""
    gpu = GPUSpec(
        name="NVIDIA A100",
        memory_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        tflops_fp16=312.0,
        tflops_fp32=156.0,
        cost_per_hour=3.67,
    )
    
    assert gpu.name == "NVIDIA A100"
    assert gpu.memory_gb == 80.0
    assert gpu.cost_per_hour == 3.67


def test_gpu_spec_without_cost():
    """Test GPUSpec without cost information."""
    gpu = GPUSpec(
        name="NVIDIA V100",
        memory_gb=32.0,
        memory_bandwidth_gb_s=900.0,
        tflops_fp16=125.0,
        tflops_fp32=62.5,
    )
    
    assert gpu.cost_per_hour is None
