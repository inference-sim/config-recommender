# Config Recommender

GPU recommendation engine for ML inference with synthetic benchmark estimation.

## Overview

Config Recommender is a Python library that recommends optimal GPU configurations for running machine learning models, particularly large language models (LLMs). It uses synthetic benchmark estimation based on model architecture parameters and GPU specifications to predict performance without requiring actual hardware testing.

## Features

- **Deterministic Synthetic Benchmarking**: Estimates performance using architecture-derived computations (FLOPs, memory footprints)
- **Memory Analysis**: Calculates memory requirements for weights, KV cache, and activations (WIP)
- **Performance Prediction**: Estimates throughput (tokens/sec) and latency (ms/token) for each GPU
- **Smart GPU Selection**: Filters compatible GPUs and selects optimal based on customizable objectives
- **Clean Python API**: Easy-to-use programmatic interface
- **CLI Tool**: Command-line interface for quick recommendations
- **Machine-Readable Output**: JSON format for integration with other tools

## Installation

### Setting up a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Installing the Package

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Or install with development dependencies (testing, linting)
pip install -e ".[dev]"
```

The project uses modern Python packaging with `pyproject.toml`. Dependencies are also available in `requirements.txt` for compatibility.

## Quick Start

### Streamlit Web UI (Recommended)

The easiest way to use the GPU Recommendation Engine is through the interactive web UI:

```bash
# Run the Streamlit UI
streamlit run streamlit_app.py

# The UI will open in your browser at http://localhost:8501
```

**Features:**
- 📝 Add models and GPUs via forms or JSON upload
- ⚙️ Configure performance parameters (precision, batch size, latency bounds)
- 📊 View recommendations in sortable, filterable tables
- 🔍 Detailed per-model breakdown (memory, FLOPs, bottleneck)
- 💾 Export results to JSON or CSV
- ♿ Accessible and responsive design

### Command-Line Interface

```bash
# Using the GPU library (recommended for common GPUs)
config-recommender --models examples/models.json --gpu-library H100 A100-80GB L40

# List available GPUs in the library
config-recommender --list-gpus

# Traditional usage with custom GPU JSON file
config-recommender --models examples/models.json --gpus examples/gpus.json

# Combine library GPUs with custom GPUs
config-recommender --models examples/models.json --gpu-library H100 A100-80GB \
    --extend-gpus custom_gpus.json

# With latency constraint
config-recommender --models examples/models.json --gpu-library H100 A100-80GB \
    --latency-bound 10

# Save output to file
config-recommender --models examples/models.json --gpu-library H100 L40 \
    --output recommendations.json

# Custom parameters
config-recommender --models examples/models.json --gpu-library H100 \
    --precision fp16 --concurrent-users 10
```

### Python API

```python
from config_recommender import (
    ModelArchitecture,
    GPURecommender,
    get_gpu_specs,
    create_custom_gpu,
)

# Define a model using HuggingFace identifier
# No need to manually specify model parameters - they're fetched automatically
model = ModelArchitecture(
    name="Qwen/Qwen2.5-7B",  # HuggingFace model identifier
)

# Option 1: Use GPUs from the preloaded library (recommended)
gpus = get_gpu_specs(["H100", "A100-80GB", "L40"])

# Option 2: Create custom GPU specs
custom_gpu = create_custom_gpu(
    name="My Custom GPU",
    memory_gb=100.0,
    memory_bandwidth_gb_s=5000.0,
    tflops_fp16=1000.0,
    tflops_fp32=500.0,
    cost_per_hour=8.0,
)
gpus.append(custom_gpu)

# Option 3: Define GPUs manually (legacy method, still supported)
from config_recommender import GPUSpec

manual_gpu = GPUSpec(
    name="NVIDIA A100 80GB",
    memory_gb=80.0,
    memory_bandwidth_gb_s=2039.0,
    tflops_fp16=312.0,
    tflops_fp32=156.0,
    cost_per_hour=3.67,
)

# Get recommendation
recommender = GPURecommender()
result = recommender.recommend_gpu(model, gpus)

print(f"Recommended GPU: {result.recommended_gpu}")
print(f"Throughput: {result.performance.tokens_per_second:.2f} tokens/sec")
print(f"Latency: {result.performance.intertoken_latency_ms:.2f} ms/token")
print(f"Reasoning: {result.reasoning}")
```

## How It Works

### Model Information Fetching

The recommendation engine automatically fetches model architecture details from HuggingFace:

- Model parameters are retrieved using the `config_explorer` library from [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark)
- No need to manually specify model architecture parameters
- Simply provide the HuggingFace model identifier (e.g., `"mistralai/Mixtral-8x7B-v0.1"`)
- For gated models, set the `HF_TOKEN` environment variable

### Synthetic Benchmark Estimation

The recommendation engine estimates performance using:

1. **Memory Requirements** (via config_explorer):
   - **Weights**: Accurate model size from HuggingFace safetensors
   - **KV Cache**: Precise calculation accounting for attention type (MHA/GQA/MQA/MLA)
   - **Activations** (WIP): Estimated based on batch size, sequence length, and hidden dimensions

2. **Performance Estimation**:
   - **Compute-bound throughput**: Based on FLOPs required per token and GPU's peak FP16 performance
   - **Memory-bound throughput**: Based on bytes read per token and GPU's memory bandwidth
   - **Actual throughput**: Limited by the bottleneck (compute or memory)

3. **GPU Selection**:
   - Filter GPUs that can fit the model in memory
   - Apply latency constraints if specified
   - Select GPU with highest tokens/sec
   - Use cost as tiebreaker when available

### GPU Library

The recommendation engine includes a preloaded library of well-known GPUs with accurate specifications:

**Available GPUs:**
- **NVIDIA H100 80GB**: High-performance GPU for demanding workloads
- **NVIDIA H200 141GB**: Extended memory variant with superior bandwidth
- **NVIDIA A100 80GB/40GB**: Widely-used data center GPUs
- **NVIDIA L40 48GB**: Mid-range GPU optimized for inference
- **NVIDIA L4 24GB**: Cost-effective option for smaller models
- **NVIDIA V100 32GB**: Previous generation GPU
- **NVIDIA T4 16GB**: Entry-level GPU

**Using the GPU Library:**
- **CLI**: Use `--gpu-library H100 A100-80GB L40` to select specific GPUs
- **Python**: Use `get_gpu_specs(["H100", "A100-80GB"])` to load from library
- **UI**: Select GPUs from dropdown menu in the Streamlit interface

**Flexibility:**
- Override library specs by creating custom GPUSpec objects
- Extend library with custom GPUs using `--extend-gpus` (CLI) or `create_custom_gpu()` (Python)
- Mix library and custom GPUs in the same recommendation

See `examples/gpu_library_usage.py` for detailed examples.

## Input Formats

### Model Architecture JSON

Models are now specified using HuggingFace identifiers:

```json
[
  {
    "name": "mistralai/Mixtral-8x7B-v0.1"
  },
  {
    "name": "Qwen/Qwen2.5-7B"
  },
  {
    "name": "ibm-granite/granite-3.0-8b-base"
  }
]
```

For gated models like Llama, set the `HF_TOKEN` environment variable:
```bash
export HF_TOKEN=your_huggingface_token
```

### GPU Specification JSON

```json
[
  {
    "name": "NVIDIA A100 80GB",
    "memory_gb": 80.0,
    "memory_bandwidth_gb_s": 2039.0,
    "tflops_fp16": 312.0,
    "tflops_fp32": 156.0,
    "cost_per_hour": 3.67
  }
]
```

## Example Output

```json
{
  "recommendations": [
    {
      "model_name": "mistralai/Mixtral-8x7B-v0.1",
      "recommended_gpu": "NVIDIA H100 80GB",
      "performance": {
        "tokens_per_second": 1234.56,
        "intertoken_latency_ms": 0.81,
        "memory_required_gb": 18.5,
        "fits_in_memory": true
      },
      "reasoning": "Selected NVIDIA H100 80GB for mistralai/Mixtral-8x7B-v0.1. Throughput: 1234.56 tokens/sec...",
      "all_compatible_gpus": [...]
    }
  ]
}
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=config_recommender tests/

# Run specific test file
pytest tests/test_recommender.py
```

## Examples

See the `examples/` directory for sample model and GPU configuration files:
- `examples/models.json`: Sample model architectures (Llama-2, Mistral)
- `examples/gpus.json`: Sample GPU specifications (A100, H100, V100, T4, L4)
- `examples/basic_usage.py`: A simple, basic usage of the Python API, single model recommendation
- `examples/advanced_usage.py`: An advanced usage, multiple models recommendations
- `examples/tensor_parallelism_example.py`: Demonstrates a model requiring TP>1
- `examples/gpu_library_usage.py`: **NEW** - Demonstrates GPU library usage and custom GPU creation

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- Code follows existing style
- New features include tests and documentation

## References

Inspired by synthetic benchmark approaches in:
- [BentoML's llm-optimizer](https://github.com/bentoml/llm-optimizer)
- FLOPs-based performance estimation techniques
