# Streamlit UI Documentation

## Overview

The GPU Recommendation Engine includes an interactive Streamlit web interface that provides an intuitive way to configure models, GPUs, and generate recommendations with detailed visualizations.

## Getting Started

### Installation

```bash
# Install with Streamlit support (already included in requirements.txt)
pip install -r requirements.txt
pip install -e .
```

### Running the UI

```bash
# Start the Streamlit application
streamlit run streamlit_app.py

# The application will open in your browser at http://localhost:8501
```

### Alternative: Using a specific port

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Features

### 1. **Model Configuration** 🤖

The Models tab allows you to add machine learning models for GPU recommendations:

**Manual Entry:**
- Enter HuggingFace model identifiers (e.g., `Qwen/Qwen2.5-7B`)
- Model details are automatically fetched from HuggingFace
- Advanced override options for gated models without tokens

**JSON Upload:**
- Upload a JSON file with multiple models
- Format: `[{"name": "model-id"}, ...]`

**Example JSON:**
```json
[
  {"name": "Qwen/Qwen2.5-7B"},
  {"name": "mistralai/Mixtral-8x7B-v0.1"}
]
```

### 2. **GPU Configuration** 🖥️

The GPUs tab allows you to define available GPU specifications:

**Manual Entry:**
- GPU name and memory capacity
- Memory bandwidth (GB/s)
- Compute performance (TFLOPS FP16/FP32)
- Cost per hour

**JSON Upload:**
- Upload a JSON file with GPU catalog
- See `examples/gpus.json` for format

**Example JSON:**
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

### 3. **Performance Parameters** ⚙️

Configure estimation parameters in the sidebar:

- **Precision**: FP16 or FP32 (affects memory and bandwidth requirements)
- **Concurrent Users**: Number of simultaneous requests (affects KV cache)
- **Memory Overhead**: Memory overhead factor (default: 1.2x)
- **Max Latency**: Optional latency constraint (ms/token)

**Note:** All performance numbers are estimates based on synthetic benchmarks.

### 4. **Recommendations** 📊

The Recommendations tab displays:

**Summary Table:**
- GPU recommended for each model
- Tensor parallelism (TP) size
- Cost per hour

**Interactive Table:**
- Sortable and filterable results
- Filter by GPU type
- Sort by throughput, latency, or memory

**Detailed Results:**
- Per-model performance metrics
- Memory breakdown per GPU (weights, KV cache)
- List of all compatible GPUs
- Reasoning explanation

**Export Options:**
- Download results as JSON
- Download results as CSV

## Usage Workflow

1. **Add Models**: Navigate to the Models tab and add one or more models
2. **Add GPUs**: Navigate to the GPUs tab and add available GPU options
3. **Configure Parameters**: Adjust performance parameters in the sidebar
4. **Generate Recommendations**: Click "Get Recommendations" in the Recommendations tab
5. **Analyze Results**: Review the summary, filter/sort results, and explore details
6. **Export**: Download results in JSON or CSV format

## Accessibility Features

- **Keyboard Navigation**: Full keyboard support through Streamlit's built-in accessibility
- **Semantic HTML**: Streamlit components use semantic HTML elements
- **Responsive Design**: Works on desktop and tablet devices
- **Clear Visual Hierarchy**: Organized layout with clear section headers

## Tips

- **Use JSON Upload**: For batch processing, upload JSON files with multiple models/GPUs
- **Adjust Efficiency**: Lower compute efficiency for more conservative estimates
- **Set Latency Bounds**: Filter out GPUs that don't meet latency requirements
- **Compare Options**: Use the detailed view to compare performance across GPUs
- **Export Results**: Save recommendations for documentation or further analysis

## Troubleshooting

### Model Loading Fails

If a model fails to load:
- Ensure the HuggingFace model ID is correct
- For gated models, set `HF_TOKEN` environment variable or use manual override
- Check internet connectivity for HuggingFace API access

### Streamlit Port in Use

If port 8501 is already in use:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Page Doesn't Load

- Clear browser cache
- Check firewall settings
- Ensure Streamlit is properly installed: `pip list | grep streamlit`

## Advanced Usage

### Environment Variables

```bash
# Set HuggingFace token for gated models
export HF_TOKEN=your_token_here

# Run with custom configuration
streamlit run streamlit_app.py --server.headless true --server.port 8080
```

### Custom Styling

The UI includes custom CSS for improved aesthetics. Modify the CSS in `streamlit_app.py` to customize appearance.

## Screenshots

See the main README for screenshots demonstrating the UI features.
