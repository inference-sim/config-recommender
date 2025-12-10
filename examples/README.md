# Examples

This directory contains example files and scripts demonstrating how to use the GPU recommendation engine.

## Files

### Data Files

- **`models.json`**: Sample model architectures including Llama-2 and Mistral models
- **`gpus.json`**: Sample GPU specifications including NVIDIA A100, H100, V100, T4, and L4

### Python Scripts

- **`basic_usage.py`**: Simple example showing basic API usage
- **`advanced_usage.py`**: Comprehensive examples showing:
  - Multiple models recommendation
  - Latency constraints
  - FP16 vs FP32 comparison
  - Sequence length impact
  - Cost-performance analysis
- **`json_workflow.py`**: Example of loading from JSON files and saving results

## Running Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

This demonstrates the simplest use case: recommending a GPU for a single model.

### Advanced Usage

```bash
python examples/advanced_usage.py
```

This shows more sophisticated scenarios including:
- Batch processing multiple models
- Applying latency constraints
- Comparing different precision settings
- Analyzing cost-performance tradeoffs

### JSON Workflow

```bash
python examples/json_workflow.py
```

This demonstrates a complete end-to-end workflow:
1. Load models and GPUs from JSON files
2. Generate recommendations
3. Save results to a JSON file

### CLI Usage

```bash
# Basic recommendation
python -m config_recommender.cli --models examples/models.json --gpus examples/gpus.json

# With options
python -m config_recommender.cli \
    --models examples/models.json \
    --gpus examples/gpus.json \
    --latency-bound 10 \
    --precision fp16 \
    --pretty

# Save to file
python -m config_recommender.cli \
    --models examples/models.json \
    --gpus examples/gpus.json \
    --output recommendations.json
```

## Input Format Examples

### Model Architecture

The `models.json` file contains an array of model specifications:

```json
[
  {
    "name": "llama-2-7b",
    "num_parameters": 7.0,
    "num_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "vocab_size": 32000,
    "max_sequence_length": 4096
  }
]
```

Required fields:
- `name`: Model identifier
- `num_parameters`: Number of parameters in billions
- `num_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension size
- `num_attention_heads`: Number of attention heads
- `vocab_size`: Vocabulary size

Optional fields:
- `max_sequence_length`: Maximum sequence length (default: 2048)
- `num_kv_heads`: Number of KV heads for GQA/MQA (default: same as num_attention_heads)

### GPU Specification

The `gpus.json` file contains an array of GPU specifications:

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

Required fields:
- `name`: GPU identifier
- `memory_gb`: Total GPU memory in GB
- `memory_bandwidth_gb_s`: Memory bandwidth in GB/s
- `tflops_fp16`: Peak FP16 TFLOPS
- `tflops_fp32`: Peak FP32 TFLOPS

Optional fields:
- `cost_per_hour`: Cost per hour (used for cost-performance analysis)

## Expected Output

The recommendation engine outputs JSON with the following structure:

```json
{
  "recommendations": [
    {
      "model_name": "llama-2-7b",
      "recommended_gpu": "NVIDIA H100 80GB",
      "performance": {
        "tokens_per_second": 239.29,
        "latency_ms_per_token": 4.18,
        "memory_required_gb": 24.0,
        "fits_in_memory": true,
        "compute_bound": false
      },
      "reasoning": "Selected NVIDIA H100 80GB for llama-2-7b. Throughput: 239.29 tokens/sec...",
      "all_compatible_gpus": [...]
    }
  ]
}
```

## Creating Your Own Examples

To create your own model or GPU configurations:

1. Create a JSON file with your model specifications (see `models.json` format)
2. Create a JSON file with your GPU specifications (see `gpus.json` format)
3. Run the CLI or use the Python API to get recommendations

Example:

```python
from config_recommender import ModelArchitecture, GPUSpec, GPURecommender

# Define your model
my_model = ModelArchitecture(
    name="my-custom-model",
    num_parameters=13.0,
    num_layers=40,
    hidden_size=5120,
    num_attention_heads=40,
    vocab_size=50000,
)

# Define available GPUs
my_gpus = [...]  # Your GPU specs

# Get recommendation
recommender = GPURecommender()
result = recommender.recommend_gpu(my_model, my_gpus)
print(result.to_dict())
```
