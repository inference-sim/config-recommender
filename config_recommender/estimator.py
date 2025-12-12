"""Synthetic benchmark estimator for model-GPU performance prediction."""

from dataclasses import dataclass
from typing import Optional

from .models import GPUSpec, ModelArchitecture

# Constants for performance estimation
# For transformer inference forward pass, approximately 2 FLOPs per parameter
# (1 multiply + 1 add for each weight in matrix multiplications)
FLOPS_PER_PARAM = 2

# Intermediate activations in transformer models typically require ~4x the hidden size
# due to feedforward network expansions (commonly 4 * hidden_size in FFN)
ACTIVATION_MULTIPLIER = 4

# Memory bandwidth utilization factor for weight reading
# In practice, not all weights are read from memory each token due to caching,
# but this provides a conservative estimate. Real-world: ~0.5-0.7 for prefill, ~1.0 for decode
MEMORY_READ_FACTOR = 1.0

# Tensor parallelism overhead factor per additional GPU
# Each additional GPU in TP introduces communication overhead for activations
# This is a conservative estimate: ~5% overhead per TP rank beyond 1
# For TP=2: 5% overhead, TP=4: 15% overhead, TP=8: 35% overhead
TP_OVERHEAD_PER_RANK = 0.05


@dataclass
class PerformanceEstimate:
    """Performance estimates for a model on a specific GPU.

    Attributes:
        tokens_per_second: Estimated throughput in tokens/second
        intertoken_latency_ms: Estimated inter-token latency in ms
            (time per token during generation)
        memory_required_gb: Total memory required in GB (per GPU)
        memory_weights_gb: Memory for model weights in GB (per GPU)
        memory_kv_cache_gb: Memory for KV cache in GB (per GPU)
        fits_in_memory: Whether the model fits in GPU memory
        compute_bound: Whether inference is compute-bound vs memory-bound
        tensor_parallel_size: Number of GPUs for tensor parallelism
            (1 for single GPU)
    """

    tokens_per_second: float
    intertoken_latency_ms: float
    memory_required_gb: float
    memory_weights_gb: float
    memory_kv_cache_gb: float
    fits_in_memory: bool
    compute_bound: bool
    tensor_parallel_size: int = 1


class SyntheticBenchmarkEstimator:
    """Estimates model performance on GPUs using synthetic benchmarks.

    Based on architecture parameters and GPU specs, estimates memory requirements,
    throughput, and latency using deterministic calculations similar to FLOPs analysis.
    """

    def __init__(
        self,
        batch_size: int = 1,
        precision_bytes: int = 2,  # FP16 = 2 bytes, FP32 = 4 bytes
        memory_overhead_factor: float = 1.2,  # 20% overhead for fragmentation, etc.
        compute_efficiency: float = 0.5,  # Utilization efficiency (50% of peak)
        concurrent_users: int = 1,  # Number of concurrent users hitting the server at once (affects KV cache memory requirements)
    ):
        """Initialize the estimator.

        Args:
            batch_size: Batch size for inference
            precision_bytes: Bytes per parameter (2 for FP16, 4 for FP32)
            memory_overhead_factor: Multiplier for memory overhead
            compute_efficiency: Fraction of peak compute actually achieved
            concurrent_users: Number of concurrent users hitting the server at once (affects KV cache memory requirements)
        """
        self.batch_size = batch_size
        self.precision_bytes = precision_bytes
        self.memory_overhead_factor = memory_overhead_factor
        self.compute_efficiency = compute_efficiency
        self.concurrent_users = concurrent_users

    def estimate_memory_weights(self, model: ModelArchitecture) -> float:
        """Estimate memory required for model weights in GB.

        Uses config_explorer library for accurate model memory requirements.

        Args:
            model: Model architecture

        Returns:
            Memory required in GB
        """
        return model.get_model_memory_gb()

    def estimate_memory_kv_cache(self, model: ModelArchitecture, sequence_length: int) -> float:
        """Estimate memory required for KV cache in GB.

        Uses config_explorer library for HF models, or falls back to calculation.
        Uses concurrent_users to account for multiple concurrent requests hitting the server.

        Args:
            model: Model architecture
            sequence_length: Sequence length to cache

        Returns:
            Memory required in GB
        """
        # ModelArchitecture handles both HF and manual modes internally
        # Use concurrent_users as batch_size for KV cache to account for concurrent requests
        return model.get_kv_cache_gb(sequence_length, self.concurrent_users)

    def estimate_memory_activation(self, model: ModelArchitecture) -> float:
        """Estimate memory required for activations in GB.

        This is a rough estimate based on batch size and model size.
        Activations scale with batch_size * sequence_length * hidden_size.
        Uses concurrent_users to account for multiple concurrent requests.

        Args:
            model: Model architecture

        Returns:
            Memory required in GB
        """
        # Try to get accurate KV cache detail from HF
        kv_detail = model.get_kv_cache_detail(model.get_max_sequence_length(), self.concurrent_users)
        if kv_detail:
            activation_elements = (
                self.concurrent_users
                * model.get_max_sequence_length()
                * kv_detail.hidden_size
                * kv_detail.num_hidden_layers
                * ACTIVATION_MULTIPLIER
            )
            activation_bytes = activation_elements * kv_detail.precision_in_bytes
            activation_gb = activation_bytes / (1024**3)
            return activation_gb

        # Fallback to manual calculation
        if model.hidden_size and model.num_layers:
            activation_elements = (
                self.concurrent_users
                * model.get_max_sequence_length()
                * model.hidden_size
                * model.num_layers
                * ACTIVATION_MULTIPLIER
            )
            activation_bytes = activation_elements * self.precision_bytes
            activation_gb = activation_bytes / (1024**3)
            return activation_gb

        return 0.0

    def estimate_total_memory(
        self, model: ModelArchitecture, sequence_length: Optional[int] = None
    ) -> dict:
        """Estimate total memory required for model inference.

        Args:
            model: Model architecture
            sequence_length: Sequence length (defaults to model's max_sequence_length)

        Returns:
            Dictionary with memory breakdown
        """
        if sequence_length is None:
            sequence_length = model.get_max_sequence_length()

        weights_gb = self.estimate_memory_weights(model)
        kv_cache_gb = self.estimate_memory_kv_cache(model, sequence_length)

        total_gb = (weights_gb + kv_cache_gb) * self.memory_overhead_factor

        return {
            "weights_gb": weights_gb,
            "kv_cache_gb": kv_cache_gb,
            "total_gb": total_gb,
        }

    def estimate_flops_per_token(self, model: ModelArchitecture) -> float:
        """Estimate FLOPs required per token for inference.

        For transformer models in inference:
        - Attention: 2 * num_layers * hidden_size^2 * (4 for QKV projections + attention)
        - FFN: 2 * num_layers * hidden_size * ffn_hidden_size (typically 4 * hidden_size)
        - Total ≈ FLOPS_PER_PARAM * num_params (forward pass only, simplified)

        Args:
            model: Model architecture

        Returns:
            FLOPs per token
        """
        # Simplified: ~FLOPS_PER_PARAM FLOPs per parameter for forward pass
        # num_parameters is in billions
        flops_per_token = FLOPS_PER_PARAM * model.get_num_parameters() * 1e9
        return flops_per_token

    def estimate_performance(
        self,
        model: ModelArchitecture,
        gpu: GPUSpec,
        sequence_length: Optional[int] = None,
        tensor_parallel_size: int = 1,
    ) -> PerformanceEstimate:
        """Estimate performance of a model on a specific GPU.

        Args:
            model: Model architecture
            gpu: GPU specification
            sequence_length: Sequence length (defaults to model's max_sequence_length)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)

        Returns:
            PerformanceEstimate object
        """
        if sequence_length is None:
            sequence_length = model.get_max_sequence_length()

        # Memory estimates - weights are split across TP GPUs
        memory_breakdown = self.estimate_total_memory(model, sequence_length)

        # With tensor parallelism, model weights are sharded across GPUs
        # KV cache is also sharded (each GPU handles a subset of attention heads)
        weights_per_gpu = memory_breakdown["weights_gb"] / tensor_parallel_size
        kv_cache_per_gpu = memory_breakdown["kv_cache_gb"] / tensor_parallel_size

        # Memory overhead still applies per GPU
        memory_required = (weights_per_gpu + kv_cache_per_gpu) * self.memory_overhead_factor
        fits_in_memory = memory_required <= gpu.memory_gb

        # FLOPs and throughput estimates
        flops_per_token = self.estimate_flops_per_token(model)

        # Use FP16 TFLOPs for performance (assuming FP16 inference)
        # With TP, compute is distributed across GPUs
        peak_flops = gpu.tflops_fp16 * 1e12 * tensor_parallel_size  # Total compute across all GPUs
        effective_flops = peak_flops * self.compute_efficiency

        # Compute-bound throughput
        compute_tokens_per_second = effective_flops / flops_per_token

        # Memory-bandwidth-bound throughput
        # With TP, each GPU reads its portion of weights
        # All GPUs process the same token in parallel, reading from their own memory
        bytes_per_token_per_gpu = (
            model.get_num_parameters()
            * 1e9
            * self.precision_bytes
            * MEMORY_READ_FACTOR
            / tensor_parallel_size
        )
        # Each GPU reads its portion; they all work on the same token in parallel
        # Throughput is limited by slowest GPU (all same, so just use one)
        memory_tokens_per_second = (gpu.memory_bandwidth_gb_s * 1e9) / bytes_per_token_per_gpu

        # Actual throughput is limited by the bottleneck
        tokens_per_second = min(compute_tokens_per_second, memory_tokens_per_second)
        compute_bound = compute_tokens_per_second < memory_tokens_per_second

        # Apply TP communication overhead (reduces throughput)
        if tensor_parallel_size > 1:
            tp_overhead = 1.0 - (TP_OVERHEAD_PER_RANK * (tensor_parallel_size - 1))
            tokens_per_second *= tp_overhead

        # If doesn't fit in memory, throughput is 0
        if not fits_in_memory:
            tokens_per_second = 0.0

        # Inter-token latency (ms per token during generation)
        intertoken_latency_ms = (
            (1000.0 / tokens_per_second) if tokens_per_second > 0 else float("inf")
        )

        return PerformanceEstimate(
            tokens_per_second=tokens_per_second,
            intertoken_latency_ms=intertoken_latency_ms,
            memory_required_gb=memory_required,
            memory_weights_gb=weights_per_gpu,
            memory_kv_cache_gb=kv_cache_per_gpu,
            fits_in_memory=fits_in_memory,
            compute_bound=compute_bound,
            tensor_parallel_size=tensor_parallel_size,
        )
