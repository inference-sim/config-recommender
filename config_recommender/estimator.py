"""Synthetic benchmark estimator for model-GPU performance prediction.

This module uses BentoML's llm-optimizer library for FLOPs-based performance estimation.
The llm-optimizer library provides production-grade roofline analysis for LLM inference,
implementing detailed FLOPS calculations and memory access patterns based on academic
literature and industry best practices.

Key features from BentoML's llm-optimizer:
- Accurate transformer FLOPS calculation (attention + MLP)
- Roofline analysis to determine compute vs memory bottlenecks
- Support for modern GPU architectures (H100, H200, A100, L40, etc.)
- Proper handling of prefill vs decode phases

References:
- BentoML llm-optimizer: https://github.com/bentoml/llm-optimizer
- Roofline analysis: https://jax-ml.github.io/scaling-book/roofline/
- "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv:2402.16363)
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

# Prefill speedup factor for fallback estimation
# Prefill phase is typically faster than decode due to parallel processing of tokens
# This is a conservative estimate used only when BentoML estimation is not available
# Typical range: 1.5-3.0x faster than decode, depending on hardware and batch size
PREFILL_SPEEDUP_FACTOR = 2.0


@dataclass
class PerformanceEstimate:
    """Performance estimates for a model on a specific GPU.

    Attributes:
        tokens_per_second: Estimated throughput in tokens/second
        intertoken_latency_ms: Estimated inter-token latency in ms
            (time per token during generation)
        ttft_ms: Time to first token in ms (prefill latency)
        e2e_latency_s: End-to-end latency in seconds
        memory_required_gb: Total memory required in GB (per GPU)
        memory_weights_gb: Memory for model weights in GB (per GPU)
        memory_kv_cache_gb: Memory for KV cache in GB (per GPU)
        fits_in_memory: Whether the model fits in GPU memory
        tensor_parallel_size: Number of GPUs for tensor parallelism
            (1 for single GPU)
    """

    tokens_per_second: float
    intertoken_latency_ms: float
    ttft_ms: float
    e2e_latency_s: float
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
        input_length: Optional[int] = None,  # Input sequence length for BentoML estimation
        output_length: Optional[int] = None,  # Output sequence length for BentoML estimation
    ):
        """Initialize the estimator.

        Args:
            precision_bytes: Bytes per parameter (2 for FP16, 4 for FP32)
            memory_overhead_factor: Multiplier for memory overhead
            input_length: Input sequence length for BentoML's performance estimation (default: 1 for per-token decode)
            output_length: Output sequence length for BentoML's performance estimation (default: 1 for per-token decode)
        """
        self.precision_bytes = precision_bytes
        self.memory_overhead_factor = memory_overhead_factor
        self.input_length = input_length if input_length is not None else 1
        self.output_length = output_length if output_length is not None else 1
        
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
        Calculates KV cache for a single request at the given sequence length (batch_size=1).

        Args:
            model: Model architecture
            sequence_length: Sequence length to cache

        Returns:
            Memory required in GB
        """
        # ModelArchitecture handles both HF and manual modes internally
        # Use batch_size=1 for KV cache (single request at max_model_len)
        return model.get_kv_cache_gb(sequence_length, batch_size=1)

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

        This method integrates BentoML's llm-optimizer library for production-grade performance
        estimation while maintaining backward compatibility with the existing API. It uses:
        
        1. BentoML's roofline analysis for compute/memory bottleneck detection
        2. Our own memory calculation for sequence-length-specific estimates
        3. Fallback to simplified estimation if GPU not supported by BentoML
        
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

        # Convert our model representation to BentoML's ModelConfig format
        bento_model_config = self._convert_to_bento_model_config(model)
        
        # Map precision bytes to string format expected by BentoML
        precision_map = {2: "fp16", 4: "fp32", 1: "fp8"}
        precision = precision_map.get(self.precision_bytes, "fp16")
        
        # Map our GPU names to BentoML's naming convention
        # BentoML uses simplified names like "H100", "A100", etc.
        # For GPUs not directly supported by BentoML, we intentionally use None
        # to trigger the fallback calculation which uses our own GPU specs
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
            # V100 and T4 not in BentoML's GPU list - will use fallback calculation
            # which uses our actual GPU specs for accurate results
        }
        
        bento_gpu_name = gpu_name_mapping.get(gpu.name)
        
        # If GPU is not in mapping, try to extract a simple name (e.g., "H100" from "NVIDIA H100")
        if bento_gpu_name is None:
            bento_gpu_name = gpu.name.replace("NVIDIA ", "").split()[0]
        
        try:
            # Only attempt BentoML estimation if we have a known GPU mapping
            if gpu.name not in gpu_name_mapping and bento_gpu_name not in ["H100", "H200", "A100", "L40", "L4"]:
                # Force fallback for unsupported GPUs to use accurate custom specs
                raise ValueError(f"GPU {gpu.name} not directly supported by BentoML, using custom calculation")
            
            # Use BentoML's estimate_llm_performance for accurate roofline analysis
            # This provides:
            # - Detailed FLOPS calculation (attention + MLP)
            # - Memory bandwidth analysis
            # - Arithmetic intensity computation
            # - Automatic bottleneck detection (compute vs memory bound)
            bento_result = estimate_llm_performance(
                num_gpus=tensor_parallel_size,
                gpu_name=bento_gpu_name,
                model_config=bento_model_config,
                precision=precision,
                concurrency=1,  # Single request for KV cache estimation
                input_length=self.input_length,  # Input sequence length (prefill phase)
                output_length=self.output_length,  # Output sequence length (decode phase)
                mfu_prefill=0.45,  # Model FLOPs Utilization for prefill (typical: 0.3-0.5)
                mfu_decode=0.30,   # Model FLOPs Utilization for decode (typical: 0.2-0.4)
                vram_util_factor=1.0 / self.memory_overhead_factor,  # Convert our overhead to utilization
            )
            
            # Extract results from BentoML's PerformanceResult
            # Use exactly what BentoML provides: Throughput, TTFT, ITL, and E2E latency
            tokens_per_second = bento_result.output_throughput_tps
            intertoken_latency_ms = bento_result.itl_ms
            ttft_ms = bento_result.ttft_ms
            e2e_latency_s = bento_result.e2e_latency_s
            
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
            
            # For fallback calculation, estimate TTFT and E2E latency
            # NOTE: This is a simplified estimation used only for GPUs not supported by BentoML
            # Real-world prefill behavior is more complex and depends on:
            # - Hardware parallelism (tensor cores, memory bandwidth)
            # - Batch size and sequence length
            # - Attention mechanism implementation
            # These estimates should be treated as rough approximations
            if tokens_per_second > 0:
                # TTFT is the prefill time for input_length tokens
                # Prefill is faster due to parallel token processing vs sequential decode
                ttft_ms = (self.input_length / (tokens_per_second * PREFILL_SPEEDUP_FACTOR)) * 1000.0
                # E2E latency = prefill time + decode time for all output tokens
                decode_time_s = self.output_length / tokens_per_second
                e2e_latency_s = (ttft_ms / 1000.0) + decode_time_s
            else:
                ttft_ms = float("inf")
                e2e_latency_s = float("inf")
                e2e_latency_s = float("inf")

        return PerformanceEstimate(
            tokens_per_second=tokens_per_second,
            intertoken_latency_ms=intertoken_latency_ms,
            ttft_ms=ttft_ms,
            e2e_latency_s=e2e_latency_s,
            memory_required_gb=memory_required_gb,
            memory_weights_gb=weights_per_gpu,
            memory_kv_cache_gb=kv_cache_per_gpu,
            fits_in_memory=fits_in_memory,
            tensor_parallel_size=tensor_parallel_size,
        )
