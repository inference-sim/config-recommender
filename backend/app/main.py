"""FastAPI application for GPU Config Recommender.

This module creates and configures the FastAPI application with all routes,
middleware, and error handling.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .api.routes import recommendations, gpus, models, health

# Create FastAPI application
app = FastAPI(
    title="GPU Config Recommender API",
    description=(
        "A production-ready API for recommending optimal GPU configurations "
        "for ML model inference. Provides synthetic benchmark estimates and "
        "detailed performance analysis."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS middleware
# In production, replace ["*"] with specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed information.

    Args:
        request: The incoming request
        exc: The validation exception

    Returns:
        JSONResponse with validation error details
    """
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions globally.

    Args:
        request: The incoming request
        exc: The exception

    Returns:
        JSONResponse with error details
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc),
        },
    )


# Include API routers
app.include_router(health.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(gpus.router, prefix="/api")
app.include_router(models.router, prefix="/api")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information.

    Returns:
        Basic API information and links to documentation
    """
    return {
        "message": "GPU Config Recommender API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    # Run the application
    # In production, use a process manager like systemd or supervisord
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info",
    )
