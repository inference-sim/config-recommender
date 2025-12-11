"""Unit tests for CLI module."""

import json
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
