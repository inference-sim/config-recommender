# Examples

This directory contains example files and scripts demonstrating how to use the GPU recommendation engine.

## Files

### Data Files

- **`models.json`**: Sample model architectures including Llama-2 and Mistral models
- **`custom_gpus.json`**: Sample GPU specifications including NVIDIA A100, H100, V100, T4, and L4

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
python -m config_recommender.cli --models examples/models.json --gpus examples/custom_gpus.json

# With options
python -m config_recommender.cli \
    --models examples/models.json \
    --gpus examples/custom_gpus.json \
    --latency-bound 10 \
    --precision fp16 \
    --pretty

# Save to file
python -m config_recommender.cli \
    --models examples/models.json \
    --gpus examples/custom_gpus.json \
    --output recommendations.json
```

## Input Format Examples

### Model Architecture

The `models.json` file contains an array of HuggingFace model identifiers.
Model details are automatically fetched from HuggingFace:

```json
[
  {
    "name": "Qwen/Qwen2.5-7B"
  },
  {
    "name": "mistralai/Mixtral-8x7B-v0.1"
  },
  {
    "name": "ibm-granite/granite-3.0-8b-base"
  }
]
```

Required fields:
- `name`: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B")

Optional fields:
- `hf_token`: HuggingFace token for gated models (defaults to `HF_TOKEN` environment variable)

**Note**: The examples use non-gated models that don't require authentication. For gated models like Llama, set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_huggingface_token
```

### GPU Specification

The `custom_gpus.json` file contains an array of GPU specifications:

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
      "model_name": "Qwen/Qwen2.5-7B",
      "recommended_gpu": "NVIDIA H100 80GB",
      "performance": {
        "tokens_per_second": 239.29,
        "latency_ms_per_token": 4.18,
        "memory_required_gb": 24.0,
        "fits_in_memory": true
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
2. Create a JSON file with your GPU specifications (see `custom_gpus.json` format)
3. Run the CLI or use the Python API to get recommendations

Example:

```python
from config_recommender import ModelArchitecture, GPUSpec, GPURecommender

# Define your model using HuggingFace identifier
my_model = ModelArchitecture(
    name="your-org/your-model-name",  # HuggingFace model identifier
)

# Define available GPUs
my_gpus = [...]  # Your GPU specs

# Get recommendation
recommender = GPURecommender()
result = recommender.recommend_gpu(my_model, my_gpus)
print(result.to_dict())
```
