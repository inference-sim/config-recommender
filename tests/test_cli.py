"""Unit tests for CLI module."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from config_recommender.cli import load_gpus_from_json, load_models_from_json


def test_load_models_from_json(tmp_path):
    """Test loading models from JSON file."""
    models_data = [
        {
            "name": "test-model",
            "num_parameters": 7.0,
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "max_sequence_length": 2048,
        }
    ]

    # Create temp file
    models_file = tmp_path / "models.json"
    with open(models_file, "w") as f:
        json.dump(models_data, f)

    # Load models
    models = load_models_from_json(models_file)

    assert len(models) == 1
    assert models[0].name == "test-model"
    assert models[0].num_parameters == 7.0


def test_load_gpus_from_json(tmp_path):
    """Test loading GPUs from JSON file."""
    gpus_data = [
        {
            "name": "NVIDIA Test GPU",
            "memory_gb": 80.0,
            "memory_bandwidth_gb_s": 2000.0,
            "tflops_fp16": 300.0,
            "tflops_fp32": 150.0,
        }
    ]

    # Create temp file
    gpus_file = tmp_path / "gpus.json"
    with open(gpus_file, "w") as f:
        json.dump(gpus_data, f)

    # Load GPUs
    gpus = load_gpus_from_json(gpus_file)

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA Test GPU"
    assert gpus[0].memory_gb == 80.0


def test_load_models_with_optional_fields(tmp_path):
    """Test loading models with optional fields."""
    models_data = [
        {
            "name": "gqa-model",
            "num_parameters": 7.0,
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "vocab_size": 32000,
        }
    ]

    models_file = tmp_path / "models.json"
    with open(models_file, "w") as f:
        json.dump(models_data, f)

    models = load_models_from_json(models_file)

    assert models[0].num_kv_heads == 8


def test_load_gpus_with_cost(tmp_path):
    """Test loading GPUs with cost information."""
    gpus_data = [
        {
            "name": "NVIDIA Test GPU",
            "memory_gb": 80.0,
            "memory_bandwidth_gb_s": 2000.0,
            "tflops_fp16": 300.0,
            "tflops_fp32": 150.0,
            "cost_per_hour": 3.50,
        }
    ]

    gpus_file = tmp_path / "gpus.json"
    with open(gpus_file, "w") as f:
        json.dump(gpus_data, f)

    gpus = load_gpus_from_json(gpus_file)

    assert gpus[0].cost_per_hour == 3.50


def test_example_files_exist():
    """Test that example files exist and are valid."""
    examples_dir = Path(__file__).parent.parent / "examples"

    # Check models.json exists and is valid
    models_file = examples_dir / "models.json"
    assert models_file.exists(), "examples/models.json should exist"

    models = load_models_from_json(models_file)
    assert len(models) > 0, "examples/models.json should contain at least one model"

    # Check gpus.json exists and is valid
    gpus_file = examples_dir / "gpus.json"
    assert gpus_file.exists(), "examples/gpus.json should exist"

    gpus = load_gpus_from_json(gpus_file)
    assert len(gpus) > 0, "examples/gpus.json should contain at least one GPU"


def test_cli_list_gpus():
    """Test CLI --list-gpus option."""
    result = subprocess.run(
        [sys.executable, "-m", "config_recommender.cli", "--list-gpus"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Available GPUs in the library:" in result.stdout
    assert "H100" in result.stdout
    assert "H200" in result.stdout
    assert "A100-80GB" in result.stdout
    assert "L40" in result.stdout


def test_cli_gpu_library_selection(tmp_path):
    """Test CLI with --gpu-library option."""
    # Create a simple model file
    models_data = [
        {
            "name": "test-model",
            "num_parameters": 7.0,
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
    ]
    
    models_file = tmp_path / "models.json"
    with open(models_file, "w") as f:
        json.dump(models_data, f)
    
    output_file = tmp_path / "output.json"
    
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "config_recommender.cli",
            "--models", str(models_file),
            "--gpu-library", "H100", "A100-80GB",
            "--output", str(output_file),
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert output_file.exists()
    
    # Check output
    with open(output_file) as f:
        output_data = json.load(f)
    
    assert "recommendations" in output_data
    assert len(output_data["recommendations"]) > 0


def test_cli_extend_gpus(tmp_path):
    """Test CLI with --extend-gpus option."""
    # Create model file
    models_data = [
        {
            "name": "test-model",
            "num_parameters": 7.0,
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
    ]
    
    models_file = tmp_path / "models.json"
    with open(models_file, "w") as f:
        json.dump(models_data, f)
    
    # Create custom GPU file
    custom_gpus_data = [
        {
            "name": "Custom GPU",
            "memory_gb": 100.0,
            "memory_bandwidth_gb_s": 5000.0,
            "tflops_fp16": 1000.0,
            "tflops_fp32": 500.0,
        }
    ]
    
    custom_gpus_file = tmp_path / "custom_gpus.json"
    with open(custom_gpus_file, "w") as f:
        json.dump(custom_gpus_data, f)
    
    output_file = tmp_path / "output.json"
    
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "config_recommender.cli",
            "--models", str(models_file),
            "--gpu-library", "H100",
            "--extend-gpus", str(custom_gpus_file),
            "--output", str(output_file),
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Extended GPU list with 1 custom GPU(s)" in result.stderr
