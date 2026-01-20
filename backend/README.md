# GPU Config Recommender - FastAPI Backend

A production-ready FastAPI backend for the GPU Config Recommender application. This API wraps the `config_recommender` library and provides RESTful endpoints for GPU recommendation, model validation, and GPU library access.

## Features

- **GPU Recommendation**: Get optimal GPU recommendations for ML models with detailed performance estimates
- **Model Validation**: Validate HuggingFace models and check accessibility
- **GPU Library**: Access preloaded specifications for common NVIDIA GPUs
- **Performance Estimation**: Synthetic benchmark estimates using roofline analysis
- **CORS Support**: Configured for cross-origin requests (configurable for production)
- **Comprehensive Error Handling**: Detailed error messages and validation
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## Architecture

```
backend/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   ├── schemas.py             # Pydantic models for validation
│   │   └── routes/
│   │       ├── recommendations.py # Recommendation endpoints
│   │       ├── gpus.py            # GPU library endpoints
│   │       ├── models.py          # Model validation endpoints
│   │       └── health.py          # Health check endpoint
│   └── services/
│       └── recommender_service.py # Business logic layer
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
└── README.md                      # This file
```

## Prerequisites

- Python 3.11+
- The `config_recommender` library (from parent directory)
- Git (for installing config-explorer dependency)

## Setup

### Local Development

1. **Install the config_recommender library first:**

```bash
# From the project root directory
cd /Users/jchen/go/src/llm-d/config-recommender
pip install -e .
```

2. **Install backend dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

3. **Run the development server:**

```bash
# From the backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker image (from project root):**

```bash
# From the project root directory
docker build -f backend/Dockerfile -t gpu-config-recommender-backend .
```

2. **Run the container:**

```bash
docker run -p 8000:8000 gpu-config-recommender-backend
```

3. **With environment variables (for gated models):**

```bash
docker run -p 8000:8000 -e HF_TOKEN=your_token_here gpu-config-recommender-backend
```

### Production Deployment

For production, use a production ASGI server like Gunicorn with Uvicorn workers:

```bash
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

## API Endpoints

### Health Check

**GET `/api/health`**

Returns the health status of the API.

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### GPU Recommendation

**POST `/api/recommendations`**

Get GPU recommendation for a model.

Request body:
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-7B"
  },
  "available_gpus": [
    {
      "name": "NVIDIA H100 80GB",
      "memory_gb": 80.0,
      "memory_bandwidth_gb_s": 3350.0,
      "tflops_fp16": 1979.0,
      "tflops_fp32": 989.0,
      "cost_per_hour": 4.76
    }
  ],
  "sequence_length": 2048,
  "latency_bound_ms": null
}
```

Example:
```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"name": "Qwen/Qwen2.5-7B"},
    "available_gpus": [{
      "name": "NVIDIA H100 80GB",
      "memory_gb": 80.0,
      "memory_bandwidth_gb_s": 3350.0,
      "tflops_fp16": 1979.0,
      "tflops_fp32": 989.0,
      "cost_per_hour": 4.76
    }],
    "sequence_length": 2048
  }'
```

### GPU Library

**POST `/api/gpu-library`**

Get GPU specifications from the preloaded library.

Request body:
```json
{
  "gpu_keys": ["H100", "A100-80GB"]
}
```

Example:
```bash
curl -X POST http://localhost:8000/api/gpu-library \
  -H "Content-Type: application/json" \
  -d '{"gpu_keys": ["H100", "A100-80GB"]}'
```

**GET `/api/gpu-library/available`**

List all available GPU keys.

```bash
curl http://localhost:8000/api/gpu-library/available
```

### Model Validation

**POST `/api/models/validate`**

Validate a HuggingFace model.

Request body:
```json
{
  "model_name": "Qwen/Qwen2.5-7B",
  "hf_token": null
}
```

Example:
```bash
curl -X POST http://localhost:8000/api/models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B"}'
```

## Configuration

### Environment Variables

- `HF_TOKEN`: HuggingFace token for accessing gated models
- `PORT`: Port to run the server on (default: 8000)
- `HOST`: Host to bind to (default: 0.0.0.0)

### CORS Configuration

By default, CORS is configured to allow all origins (`allow_origins=["*"]`). For production, update this in `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## API Documentation

The API provides auto-generated documentation:

- **Swagger UI**: http://localhost:8000/docs
  - Interactive API documentation
  - Try out endpoints directly in the browser
  - View request/response schemas

- **ReDoc**: http://localhost:8000/redoc
  - Alternative documentation format
  - More readable for complex APIs

## Error Handling

The API provides detailed error messages:

### Validation Errors (422)

```json
{
  "detail": "Request validation failed",
  "errors": [
    {
      "field": "model.name",
      "message": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Bad Request (400)

```json
{
  "detail": "GPU 'InvalidGPU' not found in library. Available GPUs: H100, H200, A100-80GB, A100-40GB, L40, L4"
}
```

### Internal Server Error (500)

```json
{
  "detail": "An unexpected error occurred",
  "error": "Error details..."
}
```

## Testing

### Manual Testing with curl

1. **Health check:**
```bash
curl http://localhost:8000/api/health
```

2. **Get available GPUs:**
```bash
curl http://localhost:8000/api/gpu-library/available
```

3. **Get GPU specs:**
```bash
curl -X POST http://localhost:8000/api/gpu-library \
  -H "Content-Type: application/json" \
  -d '{"gpu_keys": ["H100", "A100-80GB"]}'
```

4. **Validate model:**
```bash
curl -X POST http://localhost:8000/api/models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B"}'
```

5. **Get recommendation:**
```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"name": "Qwen/Qwen2.5-7B"},
    "available_gpus": [{
      "name": "NVIDIA H100 80GB",
      "memory_gb": 80.0,
      "memory_bandwidth_gb_s": 3350.0,
      "tflops_fp16": 1979.0,
      "tflops_fp32": 989.0,
      "cost_per_hour": 4.76
    }]
  }'
```

### Testing with Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/api/health")
print(response.json())

# Get recommendation
response = requests.post(
    "http://localhost:8000/api/recommendations",
    json={
        "model": {"name": "Qwen/Qwen2.5-7B"},
        "available_gpus": [{
            "name": "NVIDIA H100 80GB",
            "memory_gb": 80.0,
            "memory_bandwidth_gb_s": 3350.0,
            "tflops_fp16": 1979.0,
            "tflops_fp32": 989.0,
            "cost_per_hour": 4.76
        }]
    }
)
print(response.json())
```

## Performance Considerations

- **Model Loading**: HuggingFace models are fetched on-demand. Consider implementing caching for frequently used models.
- **Concurrent Requests**: Uvicorn handles concurrent requests well. For high traffic, use multiple workers.
- **Memory Usage**: Each recommendation requires loading model configs. Monitor memory usage in production.

## Troubleshooting

### Common Issues

1. **Module not found: config_recommender**
   - Ensure you installed the parent package: `pip install -e ..` from backend directory

2. **Model validation fails**
   - Check if the model name is correct
   - For gated models, provide HF_TOKEN environment variable

3. **Port already in use**
   - Change the port: `uvicorn app.main:app --port 8001`

4. **CORS errors**
   - Check CORS configuration in `app/main.py`
   - Add your frontend origin to allowed origins

## Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Caching layer for model metadata
- [ ] Rate limiting
- [ ] Authentication/Authorization
- [ ] Batch recommendation endpoints
- [ ] Performance metrics and monitoring
- [ ] Database integration for recommendation history

## License

Same as the parent project.

## Support

For issues and questions, please refer to the main project repository.
