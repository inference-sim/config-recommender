# GPU Recommendation Engine - Implementation Summary

## Overview

This implementation provides a complete GPU recommendation engine that uses BentoML's `llm-optimizer` to analyze model performance across different GPU types and recommends the best GPU for each model based on synthetic benchmarks.

## Components

### Core Modules

1. **models.py** - Data structures
   - `ModelConfig`: Configuration for a model/GPU combination
   - `PerformanceMetrics`: All performance metrics (latency, throughput, roofline, concurrency)
   - `PerformanceAnalysis`: Complete analysis for a model/GPU pair
   - `GPURecommendation`: Recommendation result with all analyses

2. **parser.py** - Output parser
   - `parse_bentoml_output()`: Parses llm-optimizer output into structured data
   - Extracts configuration and performance analysis sections as specified

3. **estimator.py** - BentoML integration
   - `BentoMLEstimator`: Wrapper for llm-optimizer CLI
   - Calls: `llm-optimizer estimate --model <model> --gpu <gpu> --input-len <len> --output-len <len> --num-gpus <n>`

4. **recommender.py** - Main recommendation engine
   - `GPURecommender`: Analyzes all model/GPU pairs
   - `recommend_gpus()`: Returns best GPU for each model based on metric (throughput or latency)

5. **cli.py** - Command-line interface
   - Full CLI with argparse for easy usage
   - Supports JSON output

### Scripts

- **demo.py** - Standalone demo with mock data (no llm-optimizer needed)
- **example_run.py** - Example for the specific models mentioned in the issue

## Usage Examples

### Using the CLI

```bash
# Basic usage
python -m config_recommender \
  --models "Qwen/Qwen2.5-7B" "meta-llama/Llama-3.3-70B-Instruct" \
  --gpus H200 L40

# With all options
python -m config_recommender \
  --models "Qwen/Qwen2.5-7B" \
  --gpus H200 L40 H100 \
  --input-len 2000 \
  --output-len 256 \
  --num-gpus 1 \
  --metric throughput \
  --output-json recommendations.json
```

### Using the Python API

```python
from config_recommender import GPURecommender

# Create recommender
recommender = GPURecommender(input_len=2000, output_len=256)

# Define models and GPUs
models = [
    "gpt-oss-120b",
    "Qwen3-8B-FP8-dynamic",
    "meta-llama/Llama-3.3-70B-Instruct",
    "granite-4.0-h-small",
    "mistralai/Mixtral-8x7B-v0.1",
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

### Running the Example Script

```bash
python example_run.py
```

This will analyze all the models specified in the issue across H200 and L40 GPUs.

### Running the Demo

```bash
python demo.py
```

This shows how the system works with mock data, without requiring llm-optimizer installation.

## Data Flow

1. **Input**: List of models + List of GPUs
2. **Estimation**: For each model/GPU pair, call `llm-optimizer estimate`
3. **Parsing**: Extract configuration and performance analysis from output
4. **Storage**: Store in `PerformanceAnalysis` objects
5. **Recommendation**: Compare analyses and select best GPU based on metric
6. **Output**: `GPURecommendation` objects with complete data

## Extracted Metrics

### Configuration Section
- Model name
- GPU type and count
- Precision (bf16, fp8, etc.)
- Input/output token lengths

### Performance Analysis Section

**Best Latency (concurrency=1)**:
- TTFT (Time to First Token)
- ITL (Inter-Token Latency)
- E2E (End-to-End latency)

**Best Throughput (concurrency=256)**:
- Output tokens per second
- Input tokens per second
- Requests per second
- Bottleneck type (Memory/Compute)

**Roofline Analysis**:
- Hardware Ops/Byte Ratio
- Prefill Arithmetic Intensity
- Decode Arithmetic Intensity
- Prefill Phase (Compute/Memory Bound)
- Decode Phase (Compute/Memory Bound)

**Concurrency Analysis**:
- KV Cache Memory Limit
- Prefill Compute Limit
- Decode Capacity Limit
- Theoretical Overall Limit
- Empirical Optimal Concurrency

## Testing

Run all tests:
```bash
python -m unittest discover tests -v
```

Tests include:
- Parser tests with real llm-optimizer output
- Integration tests with mocked subprocess calls
- All tests pass and provide good coverage

## Dependencies

- `llm-optimizer>=0.1.0` (BentoML's LLM optimizer tool)

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- The implementation is simple and in Python as requested
- Extracts only Configuration and Performance Analysis sections as specified
- Supports all the models mentioned in the issue
- Works with H200 and L40 GPUs (and any other GPU type supported by llm-optimizer)
- No security vulnerabilities detected
- Compatible with Python 3.8+
