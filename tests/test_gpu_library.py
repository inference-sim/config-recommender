"""Unit tests for GPU library module."""

import pytest

from config_recommender.gpu_library import (
    create_custom_gpu,
    get_gpu_from_library,
    get_gpu_specs,
    list_available_gpus,
)
from config_recommender.models import GPUSpec


def test_list_available_gpus():
    """Test listing available GPUs in the library."""
    gpus = list_available_gpus()
    
    assert isinstance(gpus, list)
    assert len(gpus) > 0
    
    # Check that priority GPUs are in the library
    assert "H100" in gpus
    assert "H200" in gpus
    assert "A100-80GB" in gpus
    assert "L40" in gpus


def test_get_gpu_from_library():
    """Test getting a specific GPU from the library."""
    # Test valid GPU
    h100 = get_gpu_from_library("H100")
    assert h100 is not None
    assert isinstance(h100, GPUSpec)
    assert "H100" in h100.name
    assert h100.memory_gb == 80.0
    
    # Test another GPU
    a100 = get_gpu_from_library("A100-80GB")
    assert a100 is not None
    assert "A100" in a100.name
    assert a100.memory_gb == 80.0
    
    # Test invalid GPU
    invalid = get_gpu_from_library("NonExistentGPU")
    assert invalid is None


def test_get_gpu_specs_all():
    """Test getting all GPU specs."""
    all_gpus = get_gpu_specs()
    
    assert isinstance(all_gpus, list)
    assert len(all_gpus) > 0
    assert all(isinstance(gpu, GPUSpec) for gpu in all_gpus)


def test_get_gpu_specs_specific():
    """Test getting specific GPU specs."""
    gpu_keys = ["H100", "A100-80GB", "L40"]
    gpus = get_gpu_specs(gpu_keys)
    
    assert len(gpus) == 3
    assert all(isinstance(gpu, GPUSpec) for gpu in gpus)
    
    # Check that we got the right GPUs
    gpu_names = [gpu.name for gpu in gpus]
    assert any("H100" in name for name in gpu_names)
    assert any("A100" in name for name in gpu_names)
    assert any("L40" in name for name in gpu_names)


def test_get_gpu_specs_invalid_key():
    """Test that invalid GPU key raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        get_gpu_specs(["InvalidGPU"])
    
    assert "not found in library" in str(exc_info.value)
    assert "Available GPUs" in str(exc_info.value)


def test_create_custom_gpu():
    """Test creating a custom GPU spec."""
    custom_gpu = create_custom_gpu(
        name="Custom Test GPU",
        memory_gb=100.0,
        memory_bandwidth_gb_s=5000.0,
        tflops_fp16=1000.0,
        tflops_fp32=500.0,
        cost_per_hour=10.0,
    )
    
    assert isinstance(custom_gpu, GPUSpec)
    assert custom_gpu.name == "Custom Test GPU"
    assert custom_gpu.memory_gb == 100.0
    assert custom_gpu.memory_bandwidth_gb_s == 5000.0
    assert custom_gpu.tflops_fp16 == 1000.0
    assert custom_gpu.tflops_fp32 == 500.0
    assert custom_gpu.cost_per_hour == 10.0


def test_create_custom_gpu_no_cost():
    """Test creating a custom GPU without cost."""
    custom_gpu = create_custom_gpu(
        name="Test GPU No Cost",
        memory_gb=50.0,
        memory_bandwidth_gb_s=2500.0,
        tflops_fp16=500.0,
        tflops_fp32=250.0,
    )
    
    assert custom_gpu.cost_per_hour is None


def test_h100_specs():
    """Test H100 GPU specifications."""
    h100 = get_gpu_from_library("H100")
    
    assert h100.memory_gb == 80.0
    assert h100.memory_bandwidth_gb_s == 3350.0
    assert h100.tflops_fp16 == 989.0
    assert h100.tflops_fp32 == 494.5


def test_h200_specs():
    """Test H200 GPU specifications."""
    h200 = get_gpu_from_library("H200")
    
    assert h200.memory_gb == 141.0
    assert h200.memory_bandwidth_gb_s == 4800.0
    assert h200.tflops_fp16 == 989.0
    assert h200.tflops_fp32 == 494.5


def test_a100_80gb_specs():
    """Test A100 80GB GPU specifications."""
    a100 = get_gpu_from_library("A100-80GB")
    
    assert a100.memory_gb == 80.0
    assert a100.memory_bandwidth_gb_s == 2039.0
    assert a100.tflops_fp16 == 312.0
    assert a100.tflops_fp32 == 156.0


def test_l40_specs():
    """Test L40 GPU specifications."""
    l40 = get_gpu_from_library("L40")
    
    assert l40.memory_gb == 48.0
    assert l40.memory_bandwidth_gb_s == 864.0
    assert l40.tflops_fp16 == 362.0
    assert l40.tflops_fp32 == 181.0
