"""Health check endpoint for GPU recommendation API."""

from fastapi import APIRouter, status

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns the health status and version of the API.",
)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns basic information about the API service status and version.

    Returns:
        HealthResponse with status and version information
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
    )
