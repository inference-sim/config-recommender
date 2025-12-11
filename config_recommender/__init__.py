"""Config Recommender - GPU recommendation engine for ML inference."""

from .estimator import SyntheticBenchmarkEstimator
from .models import GPUSpec, ModelArchitecture
from .recommender import GPURecommender, RecommendationResult

__version__ = "0.1.0"

__all__ = [
    "ModelArchitecture",
    "GPUSpec",
    "SyntheticBenchmarkEstimator",
    "GPURecommender",
    "RecommendationResult",
]
