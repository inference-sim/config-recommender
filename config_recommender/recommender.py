"""GPU recommendation engine."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from config_explorer.capacity_planner import find_possible_tp

from .estimator import PerformanceEstimate, SyntheticBenchmarkEstimator
from .models import GPUSpec, ModelArchitecture


@dataclass
class RecommendationResult:
    """Result of GPU recommendation for a model.

    Attributes:
        model_name: Name of the model
        recommended_gpu: Name of recommended GPU (None if no GPU can fit the model)
        performance: Performance estimate on recommended GPU
        all_compatible_gpus: List of all compatible GPUs with their performance
        reasoning: Human-readable explanation of the recommendation
    """

    model_name: str
    recommended_gpu: Optional[str]
    performance: Optional[PerformanceEstimate]
    all_compatible_gpus: List[Dict[str, Any]]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_name": self.model_name,
            "recommended_gpu": self.recommended_gpu,
            "reasoning": self.reasoning,
            "all_compatible_gpus": self.all_compatible_gpus,
        }

        if self.performance:
            result["performance"] = asdict(self.performance)
        else:
            result["performance"] = None

        return result


class GPURecommender:
    """Recommends optimal GPU for running ML models.

    Given a list of models and available GPUs, recommends the best GPU
    for each model based on synthetic performance estimates.
    """

    def __init__(
        self,
        estimator: Optional[SyntheticBenchmarkEstimator] = None,
        latency_bound_ms: Optional[float] = None,
    ):
        """Initialize the recommender.

        Args:
            estimator: Benchmark estimator (creates default if None)
            latency_bound_ms: Maximum acceptable latency per token in ms (optional)
        """
        self.estimator = estimator or SyntheticBenchmarkEstimator()
        self.latency_bound_ms = latency_bound_ms

    def recommend_gpu(
        self,
        model: ModelArchitecture,
        available_gpus: List[GPUSpec],
        sequence_length: Optional[int] = None,
    ) -> RecommendationResult:
        """Recommend the best GPU for a given model.

        Selection criteria:
        1. Filter GPUs that can fit the model (memory constraint)
        2. If latency_bound is set, filter by latency
        3. Select GPU with highest tokens/sec
        4. If tied, prefer lower cost (if cost data available)

        Args:
            model: Model architecture
            available_gpus: List of available GPU specs
            sequence_length: Sequence length (defaults to model's max_sequence_length)

        Returns:
            RecommendationResult with the recommendation and reasoning
        """
        if sequence_length is None:
            sequence_length = model.get_max_sequence_length()

        # Evaluate all GPUs
        gpu_evaluations = []
        for gpu in available_gpus:
            perf = self.estimator.estimate_performance(model, gpu, sequence_length)
            gpu_evaluations.append(
                {
                    "gpu": gpu,
                    "performance": perf,
                }
            )

        # Filter: GPUs that fit the model
        compatible_gpus = [
            eval_data for eval_data in gpu_evaluations if eval_data["performance"].fits_in_memory
        ]

        if not compatible_gpus:
            # No single GPU can fit this model - try tensor parallelism
            # Get possible TP sizes from capacity_planner
            tp_sizes_to_try = []
            if model._model_config is not None:
                # Get TP sizes from capacity planner (excludes 1 since we already tried it)
                all_tp_sizes = find_possible_tp(model._model_config)
                # Use all TP sizes from capacity planner
                tp_sizes_to_try = [tp for tp in all_tp_sizes if tp > 1]
            else:
                # Fallback to default range if model config not available
                tp_sizes_to_try = [2, 4, 8]

            tp_candidates = []
            for tp_size in tp_sizes_to_try:
                for gpu in available_gpus:
                    perf = self.estimator.estimate_performance(
                        model, gpu, sequence_length, tensor_parallel_size=tp_size
                    )
                    if perf.fits_in_memory:
                        tp_candidates.append(
                            {
                                "gpu": gpu,
                                "performance": perf,
                                "tp_size": tp_size,
                            }
                        )

            if not tp_candidates:
                # Even with TP, model doesn't fit
                all_gpus_info = [
                    {
                        "gpu_name": eval_data["gpu"].name,
                        "fits": False,
                        "memory_required_gb": eval_data["performance"].memory_required_gb,
                        "memory_available_gb": eval_data["gpu"].memory_gb,
                        "tensor_parallel_size": 1,
                    }
                    for eval_data in gpu_evaluations
                ]

                reasoning = (
                    f"No compatible GPU found even with tensor parallelism. "
                    f"Model requires "
                    f"{gpu_evaluations[0]['performance'].memory_required_gb:.2f} "
                    f"GB per GPU, but largest GPU has "
                    f"{max(g['gpu'].memory_gb for g in gpu_evaluations):.2f} GB."
                )

                return RecommendationResult(
                    model_name=model.name,
                    recommended_gpu=None,
                    performance=None,
                    all_compatible_gpus=all_gpus_info,
                    reasoning=reasoning,
                )

            # We have TP candidates - use them as compatible_gpus
            compatible_gpus = tp_candidates

        # Filter by latency bound if specified
        if self.latency_bound_ms is not None:
            latency_filtered = [
                eval_data
                for eval_data in compatible_gpus
                if eval_data["performance"].intertoken_latency_ms <= self.latency_bound_ms
            ]

            if not latency_filtered:
                reasoning = (
                    f"No GPU meets latency requirement of "
                    f"{self.latency_bound_ms} ms/token. "
                    f"Best achievable: "
                    f"{min(e['performance'].intertoken_latency_ms for e in compatible_gpus):.2f} "
                    f"ms/token."
                )

                all_gpus_info = [
                    {
                        "gpu_name": eval_data["gpu"].name,
                        "fits": True,
                        "tokens_per_second": eval_data["performance"].tokens_per_second,
                        "intertoken_latency_ms": eval_data["performance"].intertoken_latency_ms,
                        "meets_latency_requirement": False,
                    }
                    for eval_data in compatible_gpus
                ]

                return RecommendationResult(
                    model_name=model.name,
                    recommended_gpu=None,
                    performance=None,
                    all_compatible_gpus=all_gpus_info,
                    reasoning=reasoning,
                )

            compatible_gpus = latency_filtered

        # Sort by tokens/sec (descending), then by cost (ascending for tiebreaker)
        # Using negative tokens_per_second to sort descending, positive cost to sort ascending
        compatible_gpus.sort(
            key=lambda x: (
                -x[
                    "performance"
                ].tokens_per_second,  # Higher is better (use negative for ascending sort)
                (
                    x["gpu"].cost_per_hour if x["gpu"].cost_per_hour is not None else float("inf")
                ),  # Lower is better
            ),
        )

        # Select the best GPU
        best_eval = compatible_gpus[0]
        best_gpu = best_eval["gpu"]
        best_perf = best_eval["performance"]

        # Build reasoning
        if best_perf.tensor_parallel_size > 1:
            reasoning_parts = [
                f"Selected {best_perf.tensor_parallel_size}x {best_gpu.name} "
                f"(TP={best_perf.tensor_parallel_size}) for {model.name}.",
                f"Throughput: {best_perf.tokens_per_second:.2f} tokens/sec.",
                f"Inter-token Latency: {best_perf.intertoken_latency_ms:.2f} " f"ms/token.",
                f"Memory usage per GPU: {best_perf.memory_required_gb:.2f} GB / "
                f"{best_gpu.memory_gb:.2f} GB.",
            ]
        else:
            reasoning_parts = [
                f"Selected {best_gpu.name} for {model.name}.",
                f"Throughput: {best_perf.tokens_per_second:.2f} tokens/sec.",
                f"Inter-token Latency: {best_perf.intertoken_latency_ms:.2f} " f"ms/token.",
                f"Memory usage: {best_perf.memory_required_gb:.2f} GB / "
                f"{best_gpu.memory_gb:.2f} GB.",
            ]

        reasoning_parts.append("Performance is limited by hardware capabilities.")

        if len(compatible_gpus) > 1:
            second_best = compatible_gpus[1]
            second_perf = second_best["performance"]
            if second_perf.tensor_parallel_size > 1:
                reasoning_parts.append(
                    f"Next best: {second_perf.tensor_parallel_size}x {second_best['gpu'].name} "
                    f"({second_perf.tokens_per_second:.2f} tokens/sec)."
                )
            else:
                reasoning_parts.append(
                    f"Next best: {second_best['gpu'].name} "
                    f"({second_perf.tokens_per_second:.2f} tokens/sec)."
                )

        reasoning = " ".join(reasoning_parts)

        # Build all compatible GPUs info
        all_gpus_info = [
            {
                "gpu_name": eval_data["gpu"].name,
                "fits": True,
                "tokens_per_second": eval_data["performance"].tokens_per_second,
                "intertoken_latency_ms": eval_data["performance"].intertoken_latency_ms,
                "memory_required_gb": eval_data["performance"].memory_required_gb,
                "memory_available_gb": eval_data["gpu"].memory_gb,
                "cost_per_hour": eval_data["gpu"].cost_per_hour,
                "tensor_parallel_size": eval_data["performance"].tensor_parallel_size,
            }
            for eval_data in compatible_gpus
        ]

        return RecommendationResult(
            model_name=model.name,
            recommended_gpu=best_gpu.name,
            performance=best_perf,
            all_compatible_gpus=all_gpus_info,
            reasoning=reasoning,
        )

    def recommend_for_models(
        self,
        models: List[ModelArchitecture],
        available_gpus: List[GPUSpec],
        sequence_length: Optional[int] = None,
    ) -> List[RecommendationResult]:
        """Recommend GPUs for multiple models.

        Args:
            models: List of model architectures
            available_gpus: List of available GPU specs
            sequence_length: Sequence length (uses each model's default if None)

        Returns:
            List of RecommendationResult, one per model
        """
        results = []
        for model in models:
            result = self.recommend_gpu(model, available_gpus, sequence_length)
            results.append(result)
        return results
