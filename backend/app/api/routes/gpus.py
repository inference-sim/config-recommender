"""GPU library endpoints for GPU recommendation API."""

from typing import List

from fastapi import APIRouter, HTTPException, status

from ...services.recommender_service import RecommenderService
from ..schemas import (
    GPULibraryRequest,
    GPUSpecResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/gpu-library", tags=["gpus"])


@router.post(
    "",
    response_model=List[GPUSpecResponse],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid GPU key"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get GPU specifications from library",
    description=(
        "Retrieves GPU specifications from the preloaded library. "
        "If gpu_keys is not provided, returns all available GPUs."
    ),
)
async def get_gpu_library(request: GPULibraryRequest) -> List[GPUSpecResponse]:
    """Get GPU specifications from the library.

    The library includes preloaded specifications for common NVIDIA GPUs:
    - H100, H200
    - A100-80GB, A100-40GB
    - L40, L4

    Args:
        request: Optional list of GPU keys to retrieve

    Returns:
        List of GPU specifications

    Raises:
        HTTPException: If any specified GPU key is not found in the library
    """
    try:
        return RecommenderService.get_gpu_library(request.gpu_keys)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get(
    "/available",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="List available GPU keys",
    description="Returns a list of all available GPU keys in the library.",
)
async def list_available_gpus() -> List[str]:
    """List all available GPU keys in the library.

    Returns:
        List of GPU key strings (e.g., ["H100", "A100-80GB", "L40"])
    """
    try:
        return RecommenderService.list_available_gpu_keys()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )
