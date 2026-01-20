"""Service layer for GPU recommendation logic.

This module wraps the config_recommender library and provides business logic
for the API endpoints. It handles conversion between API schemas and library models.
"""

from typing import List, Optional

from config_recommender import (
    GPURecommender,
    GPUSpec,
    ModelArchitecture,
    RecommendationResult,
    SyntheticBenchmarkEstimator,
)
from config_recommender.gpu_library import list_available_gpus, get_gpu_specs

from ..api.schemas import (
    ModelArchitectureRequest,
    GPUSpecRequest,
    RecommendationResponse,
    PerformanceEstimateResponse,
    CompatibleGPUInfo,
    ModelValidationResponse,
    GPUSpecResponse,
)


class RecommenderService:
    """Service for GPU recommendation operations."""

    @staticmethod
    def convert_model_request_to_architecture(
        model_request: ModelArchitectureRequest,
    ) -> ModelArchitecture:
        """Convert API model request to ModelArchitecture object.

        Args:
            model_request: Pydantic model from API request

        Returns:
            ModelArchitecture object for config_recommender library

        Raises:
            ValueError: If model cannot be loaded or validated
        """
        return ModelArchitecture(
            name=model_request.name,
            hf_token=model_request.hf_token,
            num_parameters=model_request.num_parameters,
            num_layers=model_request.num_layers,
            hidden_size=model_request.hidden_size,
            num_attention_heads=model_request.num_attention_heads,
            vocab_size=model_request.vocab_size,
            max_sequence_length=model_request.max_sequence_length,
            num_kv_heads=model_request.num_kv_heads,
        )

    @staticmethod
    def convert_gpu_request_to_spec(gpu_request: GPUSpecRequest) -> GPUSpec:
        """Convert API GPU request to GPUSpec object.

        Args:
            gpu_request: Pydantic model from API request

        Returns:
            GPUSpec object for config_recommender library
        """
        return GPUSpec(
            name=gpu_request.name,
            memory_gb=gpu_request.memory_gb,
            memory_bandwidth_gb_s=gpu_request.memory_bandwidth_gb_s,
            tflops_fp16=gpu_request.tflops_fp16,
            tflops_fp32=gpu_request.tflops_fp32,
            cost_per_hour=gpu_request.cost_per_hour,
        )

    @staticmethod
    def convert_gpu_spec_to_response(gpu_spec: GPUSpec) -> GPUSpecResponse:
        """Convert GPUSpec to API response schema.

        Args:
            gpu_spec: GPUSpec from config_recommender library

        Returns:
            GPUSpecResponse for API
        """
        return GPUSpecResponse(
            name=gpu_spec.name,
            memory_gb=gpu_spec.memory_gb,
            memory_bandwidth_gb_s=gpu_spec.memory_bandwidth_gb_s,
            tflops_fp16=gpu_spec.tflops_fp16,
            tflops_fp32=gpu_spec.tflops_fp32,
            cost_per_hour=gpu_spec.cost_per_hour,
        )

    @staticmethod
    def sanitize_float(value: float) -> float:
        """Sanitize float values to ensure JSON compliance.

        Replaces inf/-inf/nan with None-safe values.

        Args:
            value: Float value to sanitize

        Returns:
            Sanitized float value (0.0 for inf/nan)
        """
        import math
        if math.isinf(value) or math.isnan(value):
            return 0.0
        return value

    @staticmethod
    def convert_recommendation_to_response(
        result: RecommendationResult,
    ) -> RecommendationResponse:
        """Convert RecommendationResult to API response schema.

        Args:
            result: RecommendationResult from config_recommender library

        Returns:
            RecommendationResponse for API
        """
        # Convert performance estimate if available
        performance = None
        if result.performance:
            performance = PerformanceEstimateResponse(
                tokens_per_second=RecommenderService.sanitize_float(
                    result.performance.tokens_per_second
                ),
                intertoken_latency_ms=RecommenderService.sanitize_float(
                    result.performance.intertoken_latency_ms
                ),
                memory_required_gb=RecommenderService.sanitize_float(
                    result.performance.memory_required_gb
                ),
                fits_in_memory=result.performance.fits_in_memory,
                tensor_parallel_size=result.performance.tensor_parallel_size,
            )

        # Convert compatible GPUs info with sanitized floats
        compatible_gpus = []
        for gpu_info in result.all_compatible_gpus:
            sanitized_info = {
                'gpu_name': gpu_info['gpu_name'],
                'fits': gpu_info['fits'],
                'tokens_per_second': RecommenderService.sanitize_float(gpu_info['tokens_per_second'])
                    if gpu_info.get('tokens_per_second') is not None else None,
                'intertoken_latency_ms': RecommenderService.sanitize_float(gpu_info['intertoken_latency_ms'])
                    if gpu_info.get('intertoken_latency_ms') is not None else None,
                'memory_required_gb': RecommenderService.sanitize_float(gpu_info['memory_required_gb'])
                    if gpu_info.get('memory_required_gb') is not None else None,
                'memory_available_gb': gpu_info.get('memory_available_gb'),
                'cost_per_hour': gpu_info.get('cost_per_hour'),
                'tensor_parallel_size': gpu_info.get('tensor_parallel_size'),
                'meets_latency_requirement': gpu_info.get('meets_latency_requirement'),
            }
            compatible_gpus.append(CompatibleGPUInfo(**sanitized_info))

        return RecommendationResponse(
            model_name=result.model_name,
            recommended_gpu=result.recommended_gpu,
            performance=performance,
            all_compatible_gpus=compatible_gpus,
            reasoning=result.reasoning,
        )

    @staticmethod
    def get_recommendation(
        model_request: ModelArchitectureRequest,
        gpu_requests: List[GPUSpecRequest],
        sequence_length: Optional[int] = None,
        latency_bound_ms: Optional[float] = None,
    ) -> RecommendationResponse:
        """Get GPU recommendation for a model.

        Args:
            model_request: Model architecture specification
            gpu_requests: List of available GPU specifications
            sequence_length: Optional sequence length override
            latency_bound_ms: Optional latency constraint

        Returns:
            RecommendationResponse with recommendation details

        Raises:
            ValueError: If model or GPU specifications are invalid
        """
        # Convert API models to library models
        model = RecommenderService.convert_model_request_to_architecture(model_request)
        gpus = [
            RecommenderService.convert_gpu_request_to_spec(gpu_req)
            for gpu_req in gpu_requests
        ]

        # Create recommender with optional latency bound
        estimator = SyntheticBenchmarkEstimator()
        recommender = GPURecommender(
            estimator=estimator, latency_bound_ms=latency_bound_ms
        )

        # Get recommendation
        result = recommender.recommend_gpu(model, gpus, sequence_length)

        # Convert to API response
        return RecommenderService.convert_recommendation_to_response(result)

    @staticmethod
    def validate_model(
        model_name: str, hf_token: Optional[str] = None
    ) -> ModelValidationResponse:
        """Validate that a HuggingFace model is accessible.

        Args:
            model_name: HuggingFace model identifier
            hf_token: Optional HuggingFace token

        Returns:
            ModelValidationResponse with validation results
        """
        try:
            # Try to load the model architecture
            model = ModelArchitecture(name=model_name, hf_token=hf_token)

            # If successful, extract information
            return ModelValidationResponse(
                valid=True,
                model_name=model_name,
                num_parameters=model.get_num_parameters(),
                max_sequence_length=model.get_max_sequence_length(),
                error=None,
                is_gated=False,
            )
        except ValueError as e:
            # Model validation failed
            error_msg = str(e)
            is_gated = "gated" in error_msg.lower() or "authentication" in error_msg.lower()

            return ModelValidationResponse(
                valid=False,
                model_name=model_name,
                num_parameters=None,
                max_sequence_length=None,
                error=error_msg,
                is_gated=is_gated,
            )
        except Exception as e:
            # Unexpected error
            return ModelValidationResponse(
                valid=False,
                model_name=model_name,
                num_parameters=None,
                max_sequence_length=None,
                error=f"Unexpected error: {str(e)}",
                is_gated=False,
            )

    @staticmethod
    def get_gpu_library(gpu_keys: Optional[List[str]] = None) -> List[GPUSpecResponse]:
        """Get GPU specifications from the library.

        Args:
            gpu_keys: Optional list of GPU keys to retrieve (returns all if None)

        Returns:
            List of GPUSpecResponse objects

        Raises:
            ValueError: If any specified GPU key is not found
        """
        # Get GPU specs from library
        gpu_specs = get_gpu_specs(gpu_keys)

        # Convert to API response format
        return [
            RecommenderService.convert_gpu_spec_to_response(spec) for spec in gpu_specs
        ]

    @staticmethod
    def list_available_gpu_keys() -> List[str]:
        """Get list of available GPU keys in the library.

        Returns:
            List of GPU key strings
        """
        return list_available_gpus()
