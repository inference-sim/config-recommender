"""Simple FastAPI backend for GPU Recommendation Engine Web UI."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path to import config_recommender
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_recommender import (
    ModelArchitecture,
    GPUSpec,
    GPURecommender,
    SyntheticBenchmarkEstimator,
    get_gpu_specs,
    list_available_gpus,
)

app = FastAPI(
    title="GPU Recommendation Engine API",
    description="Simple API for GPU recommendation",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RecommendationRequest(BaseModel):
    model_names: List[str]
    gpu_names: List[str]
    precision_bytes: int = 2
    memory_overhead_factor: float = 1.2
    latency_bound_ms: Optional[float] = None
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    sequence_length: Optional[int] = None

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "message": "GPU Recommendation Engine API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """Generate GPU recommendations."""
    try:
        # Create models
        models = []
        for model_name in request.model_names:
            try:
                model = ModelArchitecture(name=model_name)
                models.append(model)
            except Exception as e:
                print(f"Warning: Could not load model {model_name}: {e}")
                continue

        if not models:
            raise HTTPException(status_code=400, detail="No valid models provided")

        # Get GPUs from library
        gpus = []
        gpu_keys = list_available_gpus()
        all_gpu_specs = get_gpu_specs()

        for gpu_name in request.gpu_names:
            # Try to find in library
            for i, key in enumerate(gpu_keys):
                if gpu_name in all_gpu_specs[i].name or key in gpu_name:
                    gpus.append(all_gpu_specs[i])
                    break

        if not gpus:
            raise HTTPException(status_code=400, detail="No valid GPUs provided")

        # Create estimator and recommender
        estimator = SyntheticBenchmarkEstimator(
            precision_bytes=request.precision_bytes,
            memory_overhead_factor=request.memory_overhead_factor,
            input_length=request.input_length,
            output_length=request.output_length,
        )

        recommender = GPURecommender(
            estimator=estimator,
            latency_bound_ms=request.latency_bound_ms,
        )

        # Generate recommendations
        results = recommender.recommend_for_models(
            models, gpus, sequence_length=request.sequence_length
        )

        # Format response
        recommendations = []
        for result in results:
            rec_dict = {
                "model_name": result.model_name,
                "recommended_gpu": result.recommended_gpu,
                "reasoning": result.reasoning,
                "performance": None,
                "all_compatible_gpus": []
            }

            import math
            # Helper to handle inf/nan values
            def safe_float(val):
                if val is None:
                    return None
                if math.isinf(val) or math.isnan(val):
                    return None
                return val

            if result.performance:
                rec_dict["performance"] = {
                    "tokens_per_second": safe_float(result.performance.tokens_per_second),
                    "intertoken_latency_ms": safe_float(result.performance.intertoken_latency_ms),
                    "memory_required_gb": safe_float(result.performance.memory_required_gb),
                    "memory_weights_gb": safe_float(result.performance.memory_weights_gb),
                    "memory_kv_cache_gb": safe_float(result.performance.memory_kv_cache_gb),
                    "fits_in_memory": result.performance.fits_in_memory,
                    "tensor_parallel_size": result.performance.tensor_parallel_size,
                }

            # Clean all_compatible_gpus to remove inf/nan values
            cleaned_compatible = []
            for gpu_info in result.all_compatible_gpus:
                cleaned_gpu = {}
                for key, value in gpu_info.items():
                    if isinstance(value, float):
                        cleaned_gpu[key] = safe_float(value)
                    else:
                        cleaned_gpu[key] = value
                cleaned_compatible.append(cleaned_gpu)

            rec_dict["all_compatible_gpus"] = cleaned_compatible
            recommendations.append(rec_dict)

        return {"recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (the web UI)
app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting GPU Recommendation Engine Web UI...")
    print("Open your browser to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Made with Bob
