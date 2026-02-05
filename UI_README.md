# GPU Recommendation Engine - Modern Web UI

A beautiful, production-ready web interface for the GPU recommendation system built with FastAPI and vanilla HTML/CSS/JavaScript.

## Overview

This is a completely redesigned alternative to the Streamlit application featuring:
- **Modern Design**: Purple gradient theme with Inter font, glassmorphism effects, and smooth animations
- **Production-Ready**: Professional SaaS-style interface that customers can use
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **Workflow-Based**: Intuitive 4-step process (Models → GPUs → Configure → Recommend)
- **Interactive**: Drag-and-drop uploads, collapsible sections, segmented controls
- **Responsive**: Mobile-first design that works on all devices
- **No Framework Dependencies**: Pure HTML/CSS/JavaScript frontend (no React/Vue/Angular)

## Features

- **Model Management**: Add models from HuggingFace, with support for manual parameter overrides
- **GPU Management**:
  - Select from preloaded GPU library
  - Add custom GPUs manually
  - Upload GPU configurations via JSON
- **Performance Configuration**:
  - Precision selection (FP16/FP32)
  - Input/output length configuration
  - Memory overhead factor
  - Latency constraints
- **Recommendations**:
  - Generate GPU recommendations for multiple models
  - View detailed performance metrics
  - Summary tables and detailed results
- **Export**: Download results as JSON or CSV

## Getting Started

### Installation

Make sure you have installed the package with FastAPI dependencies:

```bash
pip install -e ".[dev]"
```

### Running the Application

Start the FastAPI server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at: **http://localhost:8000**

### API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

### Backend (FastAPI)

The backend provides RESTful API endpoints:

- `GET /` - Serve main HTML page
- `GET /health` - Health check endpoint
- `GET /api/models` - List all models
- `POST /api/models` - Add a new model
- `DELETE /api/models/{index}` - Delete a model
- `DELETE /api/models` - Clear all models
- `GET /api/gpus` - List all GPUs
- `POST /api/gpus` - Add a new GPU
- `DELETE /api/gpus/{index}` - Delete a GPU
- `DELETE /api/gpus` - Clear all GPUs
- `GET /api/gpus/library` - Get GPU library
- `POST /api/gpus/library/add` - Add GPUs from library
- `POST /api/recommendations` - Generate recommendations

### Frontend (HTML/CSS/JavaScript)

The frontend is a single-page application with:

- **HTML** (`static/index.html`): Structure and layout
- **CSS** (`static/styles.css`): Modern, responsive styling
- **JavaScript** (`static/app.js`): API interactions and UI logic

### State Management

The application uses in-memory storage for models and GPUs. In a production environment, you would replace this with a database (e.g., PostgreSQL, MongoDB).

## Usage Guide

### Adding Models

1. Go to the **Models** tab
2. Choose input method:
   - **Manual Entry**: Enter HuggingFace model ID (e.g., `Qwen/Qwen2.5-7B`)
   - **JSON Upload**: Upload a JSON file with model specifications
3. Click "Add Model"

### Adding GPUs

1. Go to the **GPUs** tab
2. Choose input method:
   - **GPU Library**: Select from preloaded GPUs (H100, A100, etc.)
   - **Manual Entry**: Enter custom GPU specifications
   - **JSON Upload**: Upload a JSON file with GPU specifications
3. Click "Add GPU" or "Add Selected GPUs"

### Generating Recommendations

1. Add at least one model and one GPU
2. Configure performance parameters in the sidebar:
   - Precision (FP16/FP32)
   - Input/Output lengths
   - Memory overhead factor
   - Maximum latency constraint
3. Go to the **Recommendations** tab
4. Click "Get Recommendations"
5. View results in summary table and detailed cards
6. Export results to JSON or CSV

## Configuration

### Performance Parameters

- **Precision**: Model precision (FP16 = 2 bytes, FP32 = 4 bytes per parameter)
- **Input Length**: Input sequence length for prefill phase (0 = default to 1)
- **Output Length**: Output sequence length for decode phase (0 = default to 1)
- **Memory Overhead Factor**: Memory overhead multiplier (default: 1.2 = 20% overhead)
- **Max Latency**: Maximum acceptable latency per token (0 = no limit)

## API Integration

The FastAPI backend can be used independently for integration with other applications:

### Example: Python Client

```python
import requests

# Add a model
response = requests.post('http://localhost:8000/api/models', json={
    'name': 'Qwen/Qwen2.5-7B'
})

# Add a GPU
response = requests.post('http://localhost:8000/api/gpus', json={
    'name': 'NVIDIA A100 80GB',
    'memory_gb': 80.0,
    'memory_bandwidth_gb_s': 2039.0,
    'tflops_fp16': 312.0,
    'tflops_fp32': 156.0,
    'cost_per_hour': 3.67
})

# Generate recommendations
response = requests.post('http://localhost:8000/api/recommendations', json={
    'models': [{'name': 'Qwen/Qwen2.5-7B'}],
    'gpus': [{
        'name': 'NVIDIA A100 80GB',
        'memory_gb': 80.0,
        'memory_bandwidth_gb_s': 2039.0,
        'tflops_fp16': 312.0,
        'tflops_fp32': 156.0,
        'cost_per_hour': 3.67
    }],
    'precision': 'FP16',
    'memory_overhead': 1.2
})

recommendations = response.json()
```

## Development

### Project Structure

```
.
├── app.py                      # FastAPI backend
├── static/
│   ├── index.html             # Main HTML page
│   ├── styles.css             # Styling
│   └── app.js                 # Frontend logic
├── config_recommender/        # Core library
└── pyproject.toml            # Dependencies
```

### Running in Development Mode

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes.

## Comparison with Streamlit UI

| Feature | FastAPI UI | Streamlit UI |
|---------|-----------|--------------|
| Technology | FastAPI + HTML/JS | Streamlit |
| API | RESTful API | Limited |
| Integration | Easy to integrate | Harder to integrate |
| Customization | Full control | Limited |
| Performance | Fast, lightweight | Heavier |
| State Management | Manual (in-memory) | Automatic |
| Deployment | Standard web app | Streamlit-specific |

## Future Enhancements

Potential improvements for production use:

- [ ] Add database backend (PostgreSQL, MongoDB)
- [ ] User authentication and authorization
- [ ] Session management
- [ ] Batch recommendations
- [ ] Result caching
- [ ] WebSocket support for real-time updates
- [ ] Docker containerization
- [ ] Kubernetes deployment configs
- [ ] Rate limiting
- [ ] API key authentication

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, specify a different port:

```bash
python app.py  # Change port in code
# or
uvicorn app:app --port 8080
```

### CORS Issues

CORS is enabled for all origins by default. In production, restrict to specific origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

### Static Files Not Loading

Ensure the `static/` directory exists and contains all files:
- index.html
- styles.css
- app.js

## License

Same as the parent project (MIT License).
