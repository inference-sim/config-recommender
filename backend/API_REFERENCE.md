# API Reference Card

Quick reference for all GPU Config Recommender API endpoints.

## Base URL
```
http://localhost:8000
```

---

## Health Check

### GET /api/health

Check service health and version.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

**cURL:**
```bash
curl http://localhost:8000/api/health
```

---

## GPU Library

### GET /api/gpu-library/available

List all available GPU keys in the library.

**Response:**
```json
["H100", "H200", "A100-80GB", "A100-40GB", "L40", "L4"]
```

**cURL:**
```bash
curl http://localhost:8000/api/gpu-library/available
```

---

### POST /api/gpu-library

Get GPU specifications from the library.

**Request:**
```json
{
  "gpu_keys": ["H100", "A100-80GB"]  // Optional, returns all if null
}
```

**Response:**
```json
[
  {
    "name": "NVIDIA H100 80GB",
    "memory_gb": 80.0,
    "memory_bandwidth_gb_s": 3350.0,
    "tflops_fp16": 1979.0,
    "tflops_fp32": 989.0,
    "cost_per_hour": 4.76
  }
]
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/gpu-library \
  -H "Content-Type: application/json" \
  -d '{"gpu_keys": ["H100", "A100-80GB"]}'
```

---

## Model Validation

### POST /api/models/validate

Validate a HuggingFace model and retrieve metadata.

**Request:**
```json
{
  "model_name": "Qwen/Qwen2.5-7B",
  "hf_token": null  // Optional, for gated models
}
```

**Response (Success):**
```json
{
  "valid": true,
  "model_name": "Qwen/Qwen2.5-7B",
  "num_parameters": 7.61,
  "max_sequence_length": 32768,
  "error": null,
  "is_gated": false
}
```

**Response (Failure):**
```json
{
  "valid": false,
  "model_name": "invalid/model",
  "num_parameters": null,
  "max_sequence_length": null,
  "error": "Could not load model...",
  "is_gated": false
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B"}'
```

---

## GPU Recommendation

### POST /api/recommendations

Get optimal GPU recommendation for a model.

**Request:**
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-7B",
    "hf_token": null,
    // Optional overrides for gated models:
    "num_parameters": null,
    "num_layers": null,
    "hidden_size": null,
    "num_attention_heads": null,
    "vocab_size": null,
    "max_sequence_length": null,
    "num_kv_heads": null
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
  "sequence_length": 2048,        // Optional
  "latency_bound_ms": null        // Optional
}
```

**Response:**
```json
{
  "model_name": "Qwen/Qwen2.5-7B",
  "recommended_gpu": "NVIDIA H100 80GB",
  "performance": {
    "tokens_per_second": 1234.56,
    "intertoken_latency_ms": 0.81,
    "memory_required_gb": 18.5,
    "fits_in_memory": true,
    "tensor_parallel_size": 1
  },
  "reasoning": "Selected NVIDIA H100 80GB for Qwen/Qwen2.5-7B. Throughput: 1234.56 tokens/sec. Inter-token Latency: 0.81 ms/token. Memory usage: 18.5 GB / 80.0 GB. Performance is limited by hardware capabilities.",
  "all_compatible_gpus": [
    {
      "gpu_name": "NVIDIA H100 80GB",
      "fits": true,
      "tokens_per_second": 1234.56,
      "intertoken_latency_ms": 0.81,
      "memory_required_gb": 18.5,
      "memory_available_gb": 80.0,
      "cost_per_hour": 4.76,
      "tensor_parallel_size": 1,
      "meets_latency_requirement": null
    }
  ]
}
```

**cURL:**
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

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "GPU 'InvalidGPU' not found in library..."
}
```

### 422 Validation Error
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

### 500 Internal Server Error
```json
{
  "detail": "An unexpected error occurred",
  "error": "Error details..."
}
```

---

## Common Use Cases

### 1. Get Recommendation with Library GPUs

```bash
# Step 1: Get available GPUs
curl http://localhost:8000/api/gpu-library/available

# Step 2: Get GPU specs
curl -X POST http://localhost:8000/api/gpu-library \
  -H "Content-Type: application/json" \
  -d '{"gpu_keys": ["H100", "A100-80GB"]}'

# Step 3: Get recommendation
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"name": "Qwen/Qwen2.5-7B"},
    "available_gpus": [/* GPU specs from step 2 */]
  }'
```

### 2. Validate Model Before Recommendation

```bash
# Step 1: Validate model
curl -X POST http://localhost:8000/api/models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B"}'

# Step 2: If valid, get recommendation
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### 3. Recommendation with Latency Constraint

```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"name": "Qwen/Qwen2.5-7B"},
    "available_gpus": [{...}],
    "latency_bound_ms": 1.0
  }'
```

### 4. Custom GPU Recommendation

```bash
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"name": "Qwen/Qwen2.5-7B"},
    "available_gpus": [{
      "name": "My Custom GPU",
      "memory_gb": 100.0,
      "memory_bandwidth_gb_s": 4000.0,
      "tflops_fp16": 2000.0,
      "tflops_fp32": 1000.0,
      "cost_per_hour": 5.0
    }]
  }'
```

---

## Rate Limits

Currently no rate limits enforced. Consider implementing for production.

---

## Authentication

Currently no authentication required. Consider implementing API keys or JWT for production.

---

## Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Support

For issues and questions, refer to README.md or the main project repository.
