"""Model validation endpoints for GPU recommendation API."""

from fastapi import APIRouter, HTTPException, status

from ...services.recommender_service import RecommenderService
from ..schemas import (
    ModelValidationRequest,
    ModelValidationResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.post(
    "/validate",
    response_model=ModelValidationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Validate a HuggingFace model",
    description=(
        "Validates that a HuggingFace model exists and is accessible. "
        "Returns model metadata if successful, or error details if validation fails."
    ),
)
async def validate_model(request: ModelValidationRequest) -> ModelValidationResponse:
    """Validate a HuggingFace model.

    This endpoint checks if a model can be loaded from HuggingFace and
    returns basic information about it (number of parameters, max sequence length).

    For gated models, provide a HuggingFace token with the appropriate permissions.

    Args:
        request: Model validation request with model name and optional token

    Returns:
        ModelValidationResponse with validation results and model metadata

    Raises:
        HTTPException: If an unexpected error occurs during validation
    """
    try:
        return RecommenderService.validate_model(
            model_name=request.model_name,
            hf_token=request.hf_token,
        )
    except Exception as e:
        # This should rarely happen since validate_model catches most errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during validation: {str(e)}",
        )
