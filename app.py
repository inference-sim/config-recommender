#!/usr/bin/env python3
"""FastAPI backend for GPU Recommendation Engine.

This application provides a REST API for GPU recommendation,
allowing users to manage models and GPUs, configure parameters,
and generate recommendations.
"""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config_recommender import GPURecommender, GPUSpec, ModelArchitecture
from config_recommender.estimator import SyntheticBenchmarkEstimator
from config_recommender.gpu_library import get_gpu_specs, list_available_gpus

app = FastAPI(
    title="GPU Recommendation Engine API",
    description="REST API for GPU recommendation using synthetic benchmark estimation",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for request/response
class ModelRequest(BaseModel):
    name: str
    num_parameters: Optional[float] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    vocab_size: Optional[int] = None


class GPUSpecRequest(BaseModel):
    name: str
    memory_gb: float = Field(gt=0)
    memory_bandwidth_gb_s: float = Field(gt=0)
    tflops_fp16: float = Field(gt=0)
    tflops_fp32: float = Field(gt=0)
    cost_per_hour: Optional[float] = Field(default=None, ge=0)


class RecommendationRequest(BaseModel):
    models: List[ModelRequest]
    gpus: List[GPUSpecRequest]
    precision: str = Field(default="FP16", pattern="^(FP16|FP32)$")
    input_length: Optional[int] = Field(default=None, gt=0)
    output_length: Optional[int] = Field(default=None, gt=0)
    memory_overhead: float = Field(default=1.2, ge=1.0, le=2.0)
    latency_bound_ms: Optional[float] = Field(default=None, gt=0)


class PerformanceResponse(BaseModel):
    tokens_per_second: float
    intertoken_latency_ms: float
    memory_required_gb: float
    memory_weights_gb: float
    memory_kv_cache_gb: float
    fits_in_memory: bool
    tensor_parallel_size: int


class RecommendationResponse(BaseModel):
    model_name: str
    recommended_gpu: Optional[str]
    performance: Optional[PerformanceResponse]
    reasoning: str
    all_compatible_gpus: List[Dict]


# In-memory storage (replace with database in production)
stored_models: List[ModelArchitecture] = []
stored_gpus: List[GPUSpec] = []


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Model endpoints
@app.get("/api/models")
async def get_models():
    """Get all stored models."""
    return {
        "models": [
            {
                "name": m.name,
                "num_parameters": m.get_num_parameters(),
                "num_layers": m.num_layers if hasattr(m, "num_layers") else None,
                "hidden_size": m.hidden_size if hasattr(m, "hidden_size") else None,
            }
            for m in stored_models
        ]
    }


@app.post("/api/models")
async def add_model(model_request: ModelRequest):
    """Add a new model."""
    try:
        # Build model kwargs
        model_kwargs = {"name": model_request.name}
        if model_request.num_parameters:
            model_kwargs.update(
                {
                    "num_parameters": model_request.num_parameters,
                    "num_layers": model_request.num_layers,
                    "hidden_size": model_request.hidden_size,
                    "num_attention_heads": model_request.num_attention_heads,
                    "vocab_size": model_request.vocab_size,
                }
            )

        model = ModelArchitecture(**model_kwargs)
        stored_models.append(model)
        return {
            "message": f"Model {model_request.name} added successfully",
            "model": {
                "name": model.name,
                "num_parameters": model.get_num_parameters(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/models/{index}")
async def delete_model(index: int):
    """Delete a model by index."""
    if 0 <= index < len(stored_models):
        removed = stored_models.pop(index)
        return {"message": f"Model {removed.name} deleted successfully"}
    raise HTTPException(status_code=404, detail="Model not found")


@app.delete("/api/models")
async def clear_models():
    """Clear all models."""
    stored_models.clear()
    return {"message": "All models cleared"}


# GPU endpoints
@app.get("/api/gpus")
async def get_gpus():
    """Get all stored GPUs."""
    return {
        "gpus": [
            {
                "name": g.name,
                "memory_gb": g.memory_gb,
                "memory_bandwidth_gb_s": g.memory_bandwidth_gb_s,
                "tflops_fp16": g.tflops_fp16,
                "tflops_fp32": g.tflops_fp32,
                "cost_per_hour": g.cost_per_hour,
            }
            for g in stored_gpus
        ]
    }


@app.post("/api/gpus")
async def add_gpu(gpu_request: GPUSpecRequest):
    """Add a new GPU."""
    try:
        gpu = GPUSpec(
            name=gpu_request.name,
            memory_gb=gpu_request.memory_gb,
            memory_bandwidth_gb_s=gpu_request.memory_bandwidth_gb_s,
            tflops_fp16=gpu_request.tflops_fp16,
            tflops_fp32=gpu_request.tflops_fp32,
            cost_per_hour=gpu_request.cost_per_hour,
        )
        stored_gpus.append(gpu)
        return {"message": f"GPU {gpu_request.name} added successfully", "gpu": gpu_request.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/gpus/{index}")
async def delete_gpu(index: int):
    """Delete a GPU by index."""
    if 0 <= index < len(stored_gpus):
        removed = stored_gpus.pop(index)
        return {"message": f"GPU {removed.name} deleted successfully"}
    raise HTTPException(status_code=404, detail="GPU not found")


@app.delete("/api/gpus")
async def clear_gpus():
    """Clear all GPUs."""
    stored_gpus.clear()
    return {"message": "All GPUs cleared"}


@app.get("/api/gpus/library")
async def get_gpu_library():
    """Get available GPUs from the preloaded library."""
    try:
        gpu_keys = list_available_gpus()
        all_gpu_specs = get_gpu_specs()

        library = []
        for i, gpu_key in enumerate(gpu_keys):
            gpu_spec = all_gpu_specs[i]
            library.append(
                {
                    "key": gpu_key,
                    "name": gpu_spec.name,
                    "memory_gb": gpu_spec.memory_gb,
                    "memory_bandwidth_gb_s": gpu_spec.memory_bandwidth_gb_s,
                    "tflops_fp16": gpu_spec.tflops_fp16,
                    "tflops_fp32": gpu_spec.tflops_fp32,
                    "cost_per_hour": gpu_spec.cost_per_hour,
                }
            )
        return {"library": library}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gpus/library/add")
async def add_gpus_from_library(keys: List[str]):
    """Add GPUs from the library by their keys."""
    try:
        gpu_keys = list_available_gpus()
        all_gpu_specs = get_gpu_specs()

        added_count = 0
        for key in keys:
            if key in gpu_keys:
                idx = gpu_keys.index(key)
                gpu_spec = all_gpu_specs[idx]
                # Check if GPU is already in the list
                if not any(g.name == gpu_spec.name for g in stored_gpus):
                    stored_gpus.append(gpu_spec)
                    added_count += 1

        return {"message": f"Added {added_count} GPU(s) from library"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Recommendation endpoint
@app.post("/api/recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """Generate GPU recommendations."""
    try:
        # Convert request models to ModelArchitecture
        models = []
        for model_req in request.models:
            model_kwargs = {"name": model_req.name}
            if model_req.num_parameters:
                model_kwargs.update(
                    {
                        "num_parameters": model_req.num_parameters,
                        "num_layers": model_req.num_layers,
                        "hidden_size": model_req.hidden_size,
                        "num_attention_heads": model_req.num_attention_heads,
                        "vocab_size": model_req.vocab_size,
                    }
                )
            models.append(ModelArchitecture(**model_kwargs))

        # Convert request GPUs to GPUSpec
        gpus = [
            GPUSpec(
                name=gpu_req.name,
                memory_gb=gpu_req.memory_gb,
                memory_bandwidth_gb_s=gpu_req.memory_bandwidth_gb_s,
                tflops_fp16=gpu_req.tflops_fp16,
                tflops_fp32=gpu_req.tflops_fp32,
                cost_per_hour=gpu_req.cost_per_hour,
            )
            for gpu_req in request.gpus
        ]

        # Setup estimator and recommender
        precision_bytes = 2 if request.precision == "FP16" else 4
        estimator = SyntheticBenchmarkEstimator(
            precision_bytes=precision_bytes,
            memory_overhead_factor=request.memory_overhead,
            input_length=request.input_length,
            output_length=request.output_length,
        )
        recommender = GPURecommender(estimator=estimator, latency_bound_ms=request.latency_bound_ms)

        # Calculate sequence length for KV cache
        if request.input_length and request.output_length:
            sequence_length = request.input_length + request.output_length
        else:
            sequence_length = None

        # Generate recommendations
        results = recommender.recommend_for_models(models, gpus, sequence_length=sequence_length)

        # Convert results to response format
        recommendations = []
        for rec in results:
            perf_response = None
            if rec.performance:
                perf_response = PerformanceResponse(
                    tokens_per_second=rec.performance.tokens_per_second,
                    intertoken_latency_ms=rec.performance.intertoken_latency_ms,
                    memory_required_gb=rec.performance.memory_required_gb,
                    memory_weights_gb=rec.performance.memory_weights_gb,
                    memory_kv_cache_gb=rec.performance.memory_kv_cache_gb,
                    fits_in_memory=rec.performance.fits_in_memory,
                    tensor_parallel_size=rec.performance.tensor_parallel_size,
                )

            recommendations.append(
                RecommendationResponse(
                    model_name=rec.model_name,
                    recommended_gpu=rec.recommended_gpu,
                    performance=perf_response,
                    reasoning=rec.reasoning,
                    all_compatible_gpus=rec.all_compatible_gpus,
                )
            )

        return {
            "recommendations": recommendations,
            "config": {
                "precision": request.precision,
                "input_length": request.input_length,
                "output_length": request.output_length,
                "memory_overhead": request.memory_overhead,
                "latency_bound_ms": request.latency_bound_ms,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
