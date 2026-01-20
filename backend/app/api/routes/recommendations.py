"""Recommendation endpoints for GPU recommendation API."""

from fastapi import APIRouter, HTTPException, status

from ...services.recommender_service import RecommenderService
from ..schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post(
    "",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get GPU recommendation for a model",
    description=(
        "Analyzes a model's requirements and recommends the optimal GPU from "
        "the available options. Returns performance estimates and reasoning."
    ),
)
async def create_recommendation(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Get GPU recommendation for a model.

    This endpoint performs the following:
    1. Validates the model architecture (fetches from HuggingFace if needed)
    2. Estimates performance on each available GPU
    3. Filters GPUs by memory requirements and optional latency constraints
    4. Selects the best GPU based on throughput and cost
    5. Returns detailed performance metrics and reasoning

    Args:
        request: Recommendation request with model and GPU specifications

    Returns:
        RecommendationResponse with recommended GPU and performance details

    Raises:
        HTTPException: If model validation fails or invalid parameters provided
    """
    try:
        result = RecommenderService.get_recommendation(
            model_request=request.model,
            gpu_requests=request.available_gpus,
            sequence_length=request.sequence_length,
            latency_bound_ms=request.latency_bound_ms,
        )
        return result
    except ValueError as e:
        # Model or GPU validation error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during recommendation: {str(e)}",
        )
