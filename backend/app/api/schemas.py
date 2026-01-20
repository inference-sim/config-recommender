"""Pydantic schemas for API request/response validation.

These schemas define the structure of data flowing in and out of the API endpoints.
They provide automatic validation, serialization, and documentation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelArchitectureRequest(BaseModel):
    """Request schema for model architecture.

    Maps to config_recommender.models.ModelArchitecture but uses Pydantic
    for API validation and serialization.
    """

    name: str = Field(
        ...,
        description="HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-7B')",
        example="Qwen/Qwen2.5-7B"
    )
    hf_token: Optional[str] = Field(
        None,
        description="HuggingFace token for gated models"
    )

    # Optional override fields for gated models without token
    num_parameters: Optional[float] = Field(
        None,
        description="Total number of parameters in billions",
        gt=0
    )
    num_layers: Optional[int] = Field(
        None,
        description="Number of transformer layers",
        gt=0
    )
    hidden_size: Optional[int] = Field(
        None,
        description="Hidden dimension size",
        gt=0
    )
    num_attention_heads: Optional[int] = Field(
        None,
        description="Number of attention heads",
        gt=0
    )
    vocab_size: Optional[int] = Field(
        None,
        description="Vocabulary size",
        gt=0
    )
    max_sequence_length: Optional[int] = Field(
        None,
        description="Maximum sequence length",
        gt=0
    )
    num_kv_heads: Optional[int] = Field(
        None,
        description="Number of key-value heads for GQA/MQA",
        gt=0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Qwen/Qwen2.5-7B",
                "hf_token": None
            }
        }


class GPUSpecRequest(BaseModel):
    """Request schema for GPU specifications.

    Maps to config_recommender.models.GPUSpec.
    """

    name: str = Field(
        ...,
        description="Name/model of the GPU",
        example="NVIDIA H100 80GB"
    )
    memory_gb: float = Field(
        ...,
        description="Total GPU memory in GB",
        gt=0,
        example=80.0
    )
    memory_bandwidth_gb_s: float = Field(
        ...,
        description="Memory bandwidth in GB/s",
        gt=0,
        example=3350.0
    )
    tflops_fp16: float = Field(
        ...,
        description="Peak FP16 TFLOPS",
        gt=0,
        example=1979.0
    )
    tflops_fp32: float = Field(
        ...,
        description="Peak FP32 TFLOPS",
        gt=0,
        example=989.0
    )
    cost_per_hour: Optional[float] = Field(
        None,
        description="Estimated cost per hour",
        gt=0,
        example=4.76
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "NVIDIA H100 80GB",
                "memory_gb": 80.0,
                "memory_bandwidth_gb_s": 3350.0,
                "tflops_fp16": 1979.0,
                "tflops_fp32": 989.0,
                "cost_per_hour": 4.76
            }
        }


class GPUSpecResponse(GPUSpecRequest):
    """Response schema for GPU specifications (same as request)."""
    pass


class RecommendationRequest(BaseModel):
    """Request schema for GPU recommendation.

    Includes model specifications, available GPUs, and optional parameters.
    """

    model: ModelArchitectureRequest = Field(
        ...,
        description="Model architecture to run"
    )
    available_gpus: List[GPUSpecRequest] = Field(
        ...,
        description="List of available GPU specifications",
        min_length=1
    )
    sequence_length: Optional[int] = Field(
        None,
        description="Sequence length in tokens (uses model's max if not specified)",
        gt=0
    )
    latency_bound_ms: Optional[float] = Field(
        None,
        description="Maximum acceptable latency per token in milliseconds",
        gt=0
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                "latency_bound_ms": None
            }
        }


class GPULibraryRequest(BaseModel):
    """Request schema for fetching GPUs from the library."""

    gpu_keys: Optional[List[str]] = Field(
        None,
        description="List of GPU keys to retrieve (returns all if not specified)",
        example=["H100", "A100-80GB"]
    )


class PerformanceEstimateResponse(BaseModel):
    """Response schema for performance estimates.

    Maps to config_recommender.estimator.PerformanceEstimate.
    """

    tokens_per_second: float = Field(
        ...,
        description="Estimated throughput in tokens per second"
    )
    intertoken_latency_ms: float = Field(
        ...,
        description="Estimated inter-token latency in milliseconds"
    )
    memory_required_gb: float = Field(
        ...,
        description="Total memory required in GB"
    )
    fits_in_memory: bool = Field(
        ...,
        description="Whether the model fits in GPU memory"
    )
    tensor_parallel_size: int = Field(
        ...,
        description="Number of GPUs used for tensor parallelism"
    )


class CompatibleGPUInfo(BaseModel):
    """Information about a compatible GPU."""

    gpu_name: str = Field(..., description="Name of the GPU")
    fits: bool = Field(..., description="Whether model fits in memory")
    tokens_per_second: Optional[float] = Field(None, description="Throughput in tokens/sec")
    intertoken_latency_ms: Optional[float] = Field(None, description="Latency in ms/token")
    memory_required_gb: Optional[float] = Field(None, description="Memory required in GB")
    memory_available_gb: Optional[float] = Field(None, description="Memory available in GB")
    cost_per_hour: Optional[float] = Field(None, description="Cost per hour")
    tensor_parallel_size: Optional[int] = Field(None, description="Tensor parallelism size")
    meets_latency_requirement: Optional[bool] = Field(None, description="Meets latency bound")


class RecommendationResponse(BaseModel):
    """Response schema for GPU recommendation.

    Maps to config_recommender.recommender.RecommendationResult.
    """

    model_name: str = Field(
        ...,
        description="Name of the model"
    )
    recommended_gpu: Optional[str] = Field(
        None,
        description="Name of recommended GPU (None if no GPU can fit the model)"
    )
    performance: Optional[PerformanceEstimateResponse] = Field(
        None,
        description="Performance estimate on recommended GPU"
    )
    all_compatible_gpus: List[CompatibleGPUInfo] = Field(
        ...,
        description="List of all compatible GPUs with their performance"
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of the recommendation"
    )


class ModelValidationRequest(BaseModel):
    """Request schema for validating a HuggingFace model."""

    model_name: str = Field(
        ...,
        description="HuggingFace model identifier",
        example="Qwen/Qwen2.5-7B"
    )
    hf_token: Optional[str] = Field(
        None,
        description="HuggingFace token for gated models"
    )


class ModelValidationResponse(BaseModel):
    """Response schema for model validation."""

    valid: bool = Field(..., description="Whether the model is accessible")
    model_name: str = Field(..., description="Model identifier")
    num_parameters: Optional[float] = Field(None, description="Number of parameters in billions")
    max_sequence_length: Optional[int] = Field(None, description="Maximum sequence length")
    error: Optional[str] = Field(None, description="Error message if validation failed")
    is_gated: Optional[bool] = Field(None, description="Whether the model is gated")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status", example="healthy")
    version: str = Field(..., description="API version", example="0.1.0")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")
