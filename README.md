# Config Recommender

GPU recommendation engine for model inference workloads. This tool uses BentoML's `llm-optimizer` to analyze model performance across different GPU types and recommends the best GPU for each model.

## Features

- Automatic benchmark estimation using BentoML's llm-optimizer
- Support for multiple models and GPU types
- Performance analysis including:
  - Latency metrics (TTFT, ITL, E2E)
  - Throughput metrics (tokens/s, requests/s)
  - Roofline analysis (compute vs memory bound)
  - Concurrency analysis
- Flexible recommendation criteria (throughput or latency optimization)
- JSON export for results

## Installation

```bash
pip install -r requirements.txt
```

This will install `llm-optimizer` from the BentoML GitHub repository. Note that `llm-optimizer` is not available on PyPI and must be installed directly from source.

Alternatively, you can install llm-optimizer manually:

```bash
pip install -e git+https://github.com/bentoml/llm-optimizer.git#egg=llm-optimizer
```

## Usage

### Command Line Interface

Basic usage:

```bash
python -m config_recommender --models "Qwen/Qwen2.5-7B" "meta-llama/Llama-3.3-70B-Instruct" --gpus H200 L40
```

With custom parameters:

```bash
python -m config_recommender \
  --models "Qwen/Qwen2.5-7B" "meta-llama/Llama-3.3-70B-Instruct" \
  --gpus H200 L40 H100 \
  --input-len 2000 \
  --output-len 256 \
  --num-gpus 1 \
  --metric throughput \
  --output-json recommendations.json
```

### Python API

```python
from config_recommender import GPURecommender

# Create recommender
recommender = GPURecommender(input_len=2000, output_len=256)

# Define models and GPUs
models = [
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.3-70B-Instruct"
]
gpus = ["H200", "L40"]

# Get recommendations
recommendations = recommender.recommend_gpus(
    models=models,
    gpus=gpus,
    num_gpus=1,
    metric="throughput"
)

# Print results
recommender.print_recommendations(recommendations)
```

### Example Script

Run the example script for the models specified in the issue:

```bash
python example_run.py
```

This will analyze:
- gpt-oss-120b
- Qwen3-8B-FP8-dynamic
- Llama-3.3-70B-Instruct
- granite-4.0-h-small
- Mixtral-8x7B-v0.1

Across H200 and L40 GPUs.

## Output Format

The tool provides detailed output including:

### Configuration
- Model name
- GPU type and count
- Precision (bf16, fp8, etc.)
- Input/output token lengths

### Performance Analysis
- **Best Latency**: TTFT, ITL, E2E metrics
- **Best Throughput**: Output/input tokens per second, requests per second
- **Roofline Analysis**: Hardware ops/byte ratio, arithmetic intensity, phase analysis
- **Concurrency Analysis**: KV cache limits, compute limits, optimal concurrency

### Recommendation
- Best GPU for each model based on optimization criteria
- Complete performance data for all GPU types

## Architecture

The tool consists of several modules:

- `estimator.py`: Wrapper for BentoML llm-optimizer CLI
- `parser.py`: Parses llm-optimizer output into structured data
- `models.py`: Data structures for configuration and performance metrics
- `recommender.py`: Main recommendation engine
- `cli.py`: Command-line interface

## Requirements

- Python 3.8+
- llm-optimizer (installed via requirements.txt)

## License

MIT
