"""Synthetic benchmark estimator for model-GPU performance prediction."""

from dataclasses import dataclass
from typing import Optional

from .models import ModelArchitecture, GPUSpec


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


@dataclass
class PerformanceEstimate:
    """Performance estimates for a model on a specific GPU.
    
    Attributes:
        tokens_per_second: Estimated throughput in tokens/second
        latency_ms_per_token: Estimated latency in ms per token
        memory_required_gb: Total memory required in GB
        memory_weights_gb: Memory for model weights in GB
        memory_kv_cache_gb: Memory for KV cache in GB
        memory_activation_gb: Memory for activations in GB
        fits_in_memory: Whether the model fits in GPU memory
        compute_bound: Whether inference is compute-bound vs memory-bound
    """
    tokens_per_second: float
    latency_ms_per_token: float
    memory_required_gb: float
    memory_weights_gb: float
    memory_kv_cache_gb: float
    memory_activation_gb: float
    fits_in_memory: bool
    compute_bound: bool


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
    ):
        """Initialize the estimator.
        
        Args:
            batch_size: Batch size for inference
            precision_bytes: Bytes per parameter (2 for FP16, 4 for FP32)
            memory_overhead_factor: Multiplier for memory overhead
            compute_efficiency: Fraction of peak compute actually achieved
        """
        self.batch_size = batch_size
        self.precision_bytes = precision_bytes
        self.memory_overhead_factor = memory_overhead_factor
        self.compute_efficiency = compute_efficiency
    
    def estimate_memory_weights(self, model: ModelArchitecture) -> float:
        """Estimate memory required for model weights in GB.
        
        Args:
            model: Model architecture
            
        Returns:
            Memory required in GB
        """
        # num_parameters is in billions
        params_in_billions = model.num_parameters
        bytes_per_param = self.precision_bytes
        memory_gb = params_in_billions * bytes_per_param
        return memory_gb
    
    def estimate_memory_kv_cache(
        self, 
        model: ModelArchitecture, 
        sequence_length: int
    ) -> float:
        """Estimate memory required for KV cache in GB.
        
        The KV cache stores keys and values for each layer and each token.
        Size = 2 (K and V) * num_layers * num_kv_heads * head_dim * seq_len * batch_size * precision
        
        Args:
            model: Model architecture
            sequence_length: Sequence length to cache
            
        Returns:
            Memory required in GB
        """
        head_dim = model.hidden_size // model.num_attention_heads
        num_kv_heads = model.num_kv_heads if model.num_kv_heads else model.num_attention_heads
        
        # 2 for K and V
        kv_cache_elements = (
            2 * model.num_layers * num_kv_heads * head_dim * 
            sequence_length * self.batch_size
        )
        
        kv_cache_bytes = kv_cache_elements * self.precision_bytes
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        return kv_cache_gb
    
    def estimate_memory_activation(self, model: ModelArchitecture) -> float:
        """Estimate memory required for activations in GB.
        
        This is a rough estimate based on batch size and model size.
        Activations scale with batch_size * sequence_length * hidden_size.
        
        Args:
            model: Model architecture
            
        Returns:
            Memory required in GB
        """
        # Rough estimate: batch_size * seq_len * hidden_size * num_layers * ACTIVATION_MULTIPLIER
        # ACTIVATION_MULTIPLIER accounts for FFN intermediate activations
        activation_elements = (
            self.batch_size * model.max_sequence_length * 
            model.hidden_size * model.num_layers * ACTIVATION_MULTIPLIER
        )
        activation_bytes = activation_elements * self.precision_bytes
        activation_gb = activation_bytes / (1024 ** 3)
        return activation_gb
    
    def estimate_total_memory(
        self, 
        model: ModelArchitecture, 
        sequence_length: Optional[int] = None
    ) -> dict:
        """Estimate total memory required for model inference.
        
        Args:
            model: Model architecture
            sequence_length: Sequence length (defaults to model.max_sequence_length)
            
        Returns:
            Dictionary with memory breakdown
        """
        if sequence_length is None:
            sequence_length = model.max_sequence_length
        
        weights_gb = self.estimate_memory_weights(model)
        kv_cache_gb = self.estimate_memory_kv_cache(model, sequence_length)
        activation_gb = self.estimate_memory_activation(model)
        
        total_gb = (weights_gb + kv_cache_gb + activation_gb) * self.memory_overhead_factor
        
        return {
            "weights_gb": weights_gb,
            "kv_cache_gb": kv_cache_gb,
            "activation_gb": activation_gb,
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
        flops_per_token = FLOPS_PER_PARAM * model.num_parameters * 1e9
        return flops_per_token
    
    def estimate_performance(
        self, 
        model: ModelArchitecture, 
        gpu: GPUSpec,
        sequence_length: Optional[int] = None
    ) -> PerformanceEstimate:
        """Estimate performance of a model on a specific GPU.
        
        Args:
            model: Model architecture
            gpu: GPU specification
            sequence_length: Sequence length (defaults to model.max_sequence_length)
            
        Returns:
            PerformanceEstimate object
        """
        if sequence_length is None:
            sequence_length = model.max_sequence_length
        
        # Memory estimates
        memory_breakdown = self.estimate_total_memory(model, sequence_length)
        memory_required = memory_breakdown["total_gb"]
        fits_in_memory = memory_required <= gpu.memory_gb
        
        # FLOPs and throughput estimates
        flops_per_token = self.estimate_flops_per_token(model)
        
        # Use FP16 TFLOPs for performance (assuming FP16 inference)
        peak_flops = gpu.tflops_fp16 * 1e12  # Convert to FLOPs
        effective_flops = peak_flops * self.compute_efficiency
        
        # Compute-bound throughput
        compute_tokens_per_second = effective_flops / flops_per_token
        
        # Memory-bandwidth-bound throughput
        # Simplified model: need to read model weights for each token
        # Note: This is a conservative estimate. In practice, inference engines
        # cache frequently accessed weights, reducing memory bandwidth requirements.
        # MEMORY_READ_FACTOR can be tuned based on caching effectiveness.
        bytes_per_token = model.num_parameters * 1e9 * self.precision_bytes * MEMORY_READ_FACTOR
        memory_tokens_per_second = (gpu.memory_bandwidth_gb_s * 1e9) / bytes_per_token
        
        # Actual throughput is limited by the bottleneck
        tokens_per_second = min(compute_tokens_per_second, memory_tokens_per_second)
        compute_bound = compute_tokens_per_second < memory_tokens_per_second
        
        # If doesn't fit in memory, throughput is 0
        if not fits_in_memory:
            tokens_per_second = 0.0
        
        # Latency (ms per token)
        latency_ms_per_token = (1000.0 / tokens_per_second) if tokens_per_second > 0 else float('inf')
        
        return PerformanceEstimate(
            tokens_per_second=tokens_per_second,
            latency_ms_per_token=latency_ms_per_token,
            memory_required_gb=memory_required,
            memory_weights_gb=memory_breakdown["weights_gb"],
            memory_kv_cache_gb=memory_breakdown["kv_cache_gb"],
            memory_activation_gb=memory_breakdown["activation_gb"],
            fits_in_memory=fits_in_memory,
            compute_bound=compute_bound,
        )
