"""Synthetic benchmark estimator for model-GPU performance prediction.

This module uses BentoML's llm-optimizer library for FLOPs-based performance estimation.
The llm-optimizer library provides production-grade roofline analysis for LLM inference.
"""

from dataclasses import dataclass
from typing import Optional

from llm_optimizer.common import ModelConfig as BentoModelConfig
from llm_optimizer.performance import estimate_llm_performance
from llm_optimizer.resources import GPUResourceManager, ModelMemoryCalculator

from .models import GPUSpec, ModelArchitecture

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
        tensor_parallel_size: Number of GPUs for tensor parallelism
            (1 for single GPU)
    """

    tokens_per_second: float
    intertoken_latency_ms: float
    memory_required_gb: float
    memory_weights_gb: float
    memory_kv_cache_gb: float
    fits_in_memory: bool
    tensor_parallel_size: int = 1


class SyntheticBenchmarkEstimator:
    """Estimates model performance on GPUs using BentoML's llm-optimizer.

    This estimator leverages BentoML's production-grade roofline analysis for
    accurate FLOPs-based performance estimation. It maintains compatibility with
    the existing API while using industry-standard calculations underneath.
    """

    def __init__(
        self,
        precision_bytes: int = 2,  # FP16 = 2 bytes, FP32 = 4 bytes
        memory_overhead_factor: float = 1.2,  # 20% overhead for fragmentation, etc.
        concurrent_users: int = 1,  # Number of concurrent users hitting the server at once (affects KV cache memory requirements)
    ):
        """Initialize the estimator.

        Args:
            precision_bytes: Bytes per parameter (2 for FP16, 4 for FP32)
            memory_overhead_factor: Multiplier for memory overhead
            concurrent_users: Number of concurrent users hitting the server at once (affects KV cache memory requirements)
        """
        self.precision_bytes = precision_bytes
        self.memory_overhead_factor = memory_overhead_factor
        self.concurrent_users = concurrent_users
        
        # Initialize BentoML resource managers
        self._gpu_manager = GPUResourceManager()
        self._memory_calculator = ModelMemoryCalculator()

    def _convert_to_bento_model_config(self, model: ModelArchitecture) -> BentoModelConfig:
        """Convert our ModelArchitecture to BentoML's ModelConfig.
        
        Args:
            model: Our model architecture
            
        Returns:
            BentoML's ModelConfig object
        """
        # Get model configuration details
        num_params_billions = model.get_num_parameters()
        num_params = int(num_params_billions * 1e9)
        
        # Use manual fields if available, otherwise use HF config
        if model.num_layers is not None:
            num_layers = model.num_layers
            hidden_dim = model.hidden_size
            num_heads = model.num_attention_heads
            num_kv_heads = model.num_kv_heads if model.num_kv_heads else num_heads
            vocab_size = model.vocab_size if model.vocab_size else 32000
        elif model._model_config is not None:
            # Extract from HF config
            config = model._model_config
            num_layers = config.num_hidden_layers
            hidden_dim = config.hidden_size
            num_heads = config.num_attention_heads
            # Handle different naming conventions for KV heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            vocab_size = getattr(config, 'vocab_size', 32000)
        else:
            # Fallback defaults (shouldn't happen if model is properly initialized)
            raise ValueError(f"Could not extract architecture details from model {model.name}")
        
        # Determine precision from bytes
        precision_map = {2: "fp16", 4: "fp32", 1: "fp8"}
        inferred_precision = precision_map.get(self.precision_bytes, "fp16")
        
        return BentoModelConfig(
            num_params=num_params,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            inferred_precision=inferred_precision,
        )

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

    def estimate_performance(
        self,
        model: ModelArchitecture,
        gpu: GPUSpec,
        sequence_length: Optional[int] = None,
        tensor_parallel_size: int = 1,
    ) -> PerformanceEstimate:
        """Estimate performance of a model on a specific GPU using BentoML's roofline analysis.

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

        # Convert model to BentoML format
        bento_model_config = self._convert_to_bento_model_config(model)
        
        # Determine precision string from bytes
        precision_map = {2: "fp16", 4: "fp32", 1: "fp8"}
        precision = precision_map.get(self.precision_bytes, "fp16")
        
        # Map our GPU name to BentoML's GPU names
        # BentoML uses simplified names like "H100", "A100", etc.
        gpu_name_mapping = {
            "NVIDIA H100": "H100",
            "NVIDIA H100 80GB": "H100",
            "NVIDIA H200": "H200",
            "NVIDIA H200 141GB": "H200",
            "NVIDIA A100": "A100",
            "NVIDIA A100 80GB": "A100",
            "NVIDIA A100 40GB": "A100",
            "NVIDIA L40": "L40",
            "NVIDIA L40 48GB": "L40",
            "NVIDIA L4": "L4",
            "NVIDIA L4 24GB": "L4",
            "NVIDIA V100": "A100",  # Fallback to A100 specs if V100 not supported
            "NVIDIA V100 32GB": "A100",
            "NVIDIA T4": "A100",  # Fallback to A100 specs if T4 not supported
            "NVIDIA T4 16GB": "A100",
        }
        
        # Try to find a matching GPU name, otherwise use the name as-is
        bento_gpu_name = gpu_name_mapping.get(gpu.name, gpu.name.replace("NVIDIA ", ""))
        
        try:
            # Use BentoML's estimate_llm_performance function
            # We use decode phase for single-token generation (which is what we're estimating)
            # input_length=1 for decode phase, output_length=1 for single token
            bento_result = estimate_llm_performance(
                num_gpus=tensor_parallel_size,
                gpu_name=bento_gpu_name,
                model_config=bento_model_config,
                precision=precision,
                concurrency=self.concurrent_users,
                input_length=1,  # Decode phase generates one token at a time
                output_length=1,  # Estimating per-token latency
                mfu_prefill=0.45,  # Model FLOPs utilization for prefill
                mfu_decode=0.30,   # Model FLOPs utilization for decode
                vram_util_factor=1.0 / self.memory_overhead_factor,  # Convert our overhead to utilization
            )
            
            # Extract results from BentoML's PerformanceResult
            tokens_per_second = bento_result.output_throughput_tps
            intertoken_latency_ms = bento_result.itl_ms
            
            # Calculate memory breakdown for compatibility and accuracy
            # We use our own memory calculation because it accounts for the specific sequence_length
            weights_per_gpu = self.estimate_memory_weights(model) / tensor_parallel_size
            kv_cache_per_gpu = self.estimate_memory_kv_cache(model, sequence_length) / tensor_parallel_size
            memory_required_gb = (weights_per_gpu + kv_cache_per_gpu) * self.memory_overhead_factor
            
            # Check if model fits in GPU memory
            fits_in_memory = memory_required_gb <= gpu.memory_gb
            
        except (ValueError, KeyError) as e:
            # If BentoML doesn't support this GPU, fall back to our own calculation
            # This maintains backward compatibility
            memory_breakdown = self.estimate_total_memory(model, sequence_length)
            
            # With tensor parallelism, model weights are sharded across GPUs
            # KV cache is also sharded (each GPU handles a subset of attention heads)
            weights_per_gpu = memory_breakdown["weights_gb"] / tensor_parallel_size
            kv_cache_per_gpu = memory_breakdown["kv_cache_gb"] / tensor_parallel_size
            
            # Memory overhead still applies per GPU
            memory_required_gb = (weights_per_gpu + kv_cache_per_gpu) * self.memory_overhead_factor
            fits_in_memory = memory_required_gb <= gpu.memory_gb
            
            # Simplified FLOPs calculation: ~2 FLOPs per parameter for forward pass
            flops_per_token = 2 * model.get_num_parameters() * 1e9
            
            # Use FP16 TFLOPs for performance (assuming FP16 inference)
            # With TP, compute is distributed across GPUs
            peak_flops = gpu.tflops_fp16 * 1e12 * tensor_parallel_size
            
            # Compute-bound throughput
            compute_tokens_per_second = peak_flops / flops_per_token
            
            # Memory-bandwidth-bound throughput
            bytes_per_token_per_gpu = (
                model.get_num_parameters()
                * 1e9
                * self.precision_bytes
                / tensor_parallel_size
            )
            memory_tokens_per_second = (gpu.memory_bandwidth_gb_s * 1e9) / bytes_per_token_per_gpu
            
            # Actual throughput is limited by the bottleneck
            tokens_per_second = min(compute_tokens_per_second, memory_tokens_per_second)
            
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
            memory_required_gb=memory_required_gb,
            memory_weights_gb=weights_per_gpu,
            memory_kv_cache_gb=kv_cache_per_gpu,
            fits_in_memory=fits_in_memory,
            tensor_parallel_size=tensor_parallel_size,
        )
