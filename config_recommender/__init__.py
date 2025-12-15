"""Config Recommender - GPU recommendation engine for ML inference."""

from .estimator import SyntheticBenchmarkEstimator
from .gpu_library import (
    create_custom_gpu,
    get_gpu_from_library,
    get_gpu_specs,
    list_available_gpus,
)
from .models import GPUSpec, ModelArchitecture
from .recommender import GPURecommender, RecommendationResult

__version__ = "0.1.0"

__all__ = [
    "ModelArchitecture",
    "GPUSpec",
    "SyntheticBenchmarkEstimator",
    "GPURecommender",
    "RecommendationResult",
    "get_gpu_from_library",
    "list_available_gpus",
    "get_gpu_specs",
    "create_custom_gpu",
]
