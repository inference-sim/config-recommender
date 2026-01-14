# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Config Recommender is a GPU recommendation engine for ML inference that uses synthetic benchmark estimation. It provides three interfaces: a Python API, a CLI tool, and an interactive Streamlit web UI. The engine estimates performance using BentoML's llm-optimizer library for roofline analysis and automatically fetches model architecture details from HuggingFace.

## Key Architecture

### Core Components

**Models and GPU Specs** (`config_recommender/models.py`)
- `ModelArchitecture`: Represents ML models via HuggingFace identifiers. Automatically fetches architecture details from HuggingFace using config_explorer library. Supports optional parameter overrides for gated models.
- `GPUSpec`: GPU specifications (name, memory, bandwidth, TFLOPS, cost).

**GPU Library** (`config_recommender/gpu_library.py`)
- Preloaded library with common NVIDIA GPUs (H100, H200, A100, L40, L4, V100, T4)
- `get_gpu_specs()`: Load GPUs from library by name
- `create_custom_gpu()`: Create custom GPU specifications

**Estimator** (`config_recommender/estimator.py`)
- `SyntheticBenchmarkEstimator`: Performs roofline analysis to estimate throughput (tokens/sec) and latency (ms/token)
- Uses BentoML's llm-optimizer for detailed FLOP calculations and memory bandwidth analysis
- `PerformanceEstimate`: Output dataclass with throughput, latency, memory requirements

**Recommender** (`config_recommender/recommender.py`)
- `GPURecommender`: Main recommendation engine
- `recommend_gpu()`: Returns `RecommendationResult` with recommended GPU, performance metrics, all compatible GPUs, and reasoning explanation
- Filters GPUs that fit model in memory, applies latency constraints, selects by highest throughput with cost tiebreaker

### Interfaces

**CLI** (`config_recommender/cli.py`)
- Entry point: `config-recommender` command
- Supports: GPU library selection, custom GPU files, extend-gpus option, latency bounds, output to JSON

**Streamlit UI** (`streamlit_app.py`)
- Run: `streamlit run streamlit_app.py`
- Three tabs: Models, GPUs, Recommendations
- Supports JSON upload, manual entry, filtering, sorting, export (JSON/CSV)
- Configurable parameters: precision, latency bounds, memory overhead

**Python API** (`config_recommender/__init__.py`)
- Public exports for direct programmatic use

## Dependencies

**Key External Libraries**
- `config_explorer` (from llm-d-benchmark): Fetches model architecture details from HuggingFace and calculates memory requirements
- `llm-optimizer` (from BentoML): Performs roofline analysis for performance estimation
- `streamlit`: Web UI framework
- `pandas`: Data handling and export

**Dev Dependencies**
- `pytest`, `pytest-cov`: Testing
- `black`, `isort`: Code formatting
- `flake8`: Linting

## Common Commands

```bash
# Development setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run Streamlit UI
streamlit run streamlit_app.py

# CLI usage
config-recommender --models examples/models.json --gpu-library H100 A100-80GB
config-recommender --list-gpus

# Run tests
pytest tests/
pytest tests/test_recommender.py
pytest --cov=config_recommender tests/

# Code formatting and linting
black config_recommender/ tests/
isort config_recommender/ tests/
flake8 config_recommender/ tests/
```

## Important Implementation Details

**Model Information Fetching**
- Models are specified by HuggingFace identifier (e.g., `"Qwen/Qwen2.5-7B"`)
- Architecture details and model weights are fetched automatically via `config_explorer`
- For gated models, set `HF_TOKEN` environment variable
- Parameters can be manually overridden in `ModelArchitecture` (useful when token unavailable)

**Performance Estimation Process**
1. Calculate memory requirements (weights, KV cache, activations)
2. Determine KV cache requirements per concurrent user (based on attention type: MHA/GQA/MQA/MLA)
3. Apply roofline analysis: determine if workload is compute-bound or memory-bound
4. Calculate accurate FLOPS based on transformer architecture (attention + MLP)
5. Estimate throughput using arithmetic intensity and GPU specifications

**GPU Selection Logic**
- Filter GPUs that can fit model weights + KV cache in memory
- Apply optional latency constraint (rejects GPUs that can't meet it)
- Select GPU with highest throughput
- Use cost per hour as tiebreaker when available

**Tensor Parallelism**
- `config_explorer.find_possible_tp()` determines valid TP sizes based on model memory requirements
- Used when recommending configurations for larger models

## Testing Strategy

Tests are organized in `tests/` directory:
- `test_models.py`: Model loading and architecture fetching
- `test_recommender.py`: Core recommendation logic
- `test_estimator.py`: Performance estimation accuracy
- `test_gpu_library.py`: GPU library functionality
- `test_cli.py`: CLI argument parsing and integration
- `test_tensor_parallelism.py`: TP size calculations
- `test_concurrent_users.py`: Multi-user KV cache handling
- `test_streamlit_ui.py`: Streamlit component tests
- `test_sequence_length_ui.py`: Sequence length configuration in UI

Run full test suite with coverage: `pytest --cov=config_recommender tests/`

## Code Style

Follow standards from `.github/copilot-instructions.md`:
- PEP 8 style guidelines
- Type hints for functions and return values
- Docstrings for public functions, classes, and modules
- Single-purpose functions, minimal comments (code self-documenting)
- Clean error handling and logging

Project uses:
- Black: line-length 100, target Python 3.11+
- isort: black profile, line-length 100
- flake8: default rules

## Input/Output Formats

**Model Configuration (JSON)**
```json
[{"name": "mistralai/Mixtral-8x7B-v0.1"}, {"name": "Qwen/Qwen2.5-7B"}]
```

**GPU Configuration (JSON)**
```json
[{
  "name": "NVIDIA A100 80GB",
  "memory_gb": 80.0,
  "memory_bandwidth_gb_s": 2039.0,
  "tflops_fp16": 312.0,
  "tflops_fp32": 156.0,
  "cost_per_hour": 3.67
}]
```

**Recommendation Output (JSON)**
```json
{
  "recommendations": [{
    "model_name": "mistralai/Mixtral-8x7B-v0.1",
    "recommended_gpu": "NVIDIA H100 80GB",
    "performance": {
      "tokens_per_second": 1234.56,
      "intertoken_latency_ms": 0.81,
      "memory_required_gb": 18.5,
      "fits_in_memory": true
    },
    "reasoning": "...",
    "all_compatible_gpus": [...]
  }]
}
```
