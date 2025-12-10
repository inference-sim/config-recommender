"""Data models for GPU specs and model architectures."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config_explorer.capacity_planner import KVCacheDetail
    from huggingface_hub import ModelInfo
    from transformers import AutoConfig

from config_explorer.capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    model_memory_req,
    kv_cache_req,
    max_context_len,
    model_total_params,
    KVCacheDetail,
)
from huggingface_hub import ModelInfo
from transformers import AutoConfig


@dataclass
class ModelArchitecture:
    """Represents a machine learning model's architecture details.
    
    Model details are automatically fetched from HuggingFace using the model identifier.
    For gated models, set the HF_TOKEN environment variable.
    
    You can optionally override specific parameters (e.g., for gated models where you
    know the specs but don't have a token).
    
    Attributes:
        name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B")
        hf_token: Optional HuggingFace token (defaults to HF_TOKEN environment variable)
        
        # Optional override fields (for gated models without token):
        num_parameters: Total number of parameters in billions (overrides HF value)
        num_layers: Number of transformer layers (overrides HF value)
        hidden_size: Hidden dimension size (overrides HF value)
        num_attention_heads: Number of attention heads (overrides HF value)
        vocab_size: Vocabulary size (overrides HF value)
        max_sequence_length: Maximum sequence length (overrides HF value)
        num_kv_heads: Number of key-value heads for GQA/MQA (overrides HF value)
    """
    name: str
    hf_token: Optional[str] = None
    
    # Optional override fields (for gated models without token)
    num_parameters: Optional[float] = None  # in billions
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_sequence_length: Optional[int] = None
    num_kv_heads: Optional[int] = None
    
    # Private fields - not part of constructor, initialized in __post_init__
    _model_info: Optional[ModelInfo] = field(default=None, init=False, repr=False)
    _model_config: Optional[AutoConfig] = field(default=None, init=False, repr=False)
    _kv_cache_detail: Optional[KVCacheDetail] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize model info from HuggingFace."""
        # Use environment variable if token not provided
        if self.hf_token is None:
            self.hf_token = os.environ.get('HF_TOKEN')
        
        # Try to fetch from HuggingFace (may fail for gated models without token)
        try:
            self._model_info = get_model_info_from_hf(self.name, self.hf_token)
            self._model_config = get_model_config_from_hf(self.name, self.hf_token)
        except Exception as e:
            # If fetching fails and we have manual overrides, that's okay
            if self.num_parameters is not None:
                # Validate we have enough manual specifications
                required_fields = ['num_layers', 'hidden_size', 'num_attention_heads', 'vocab_size']
                missing = [f for f in required_fields if getattr(self, f) is None]
                if missing:
                    raise ValueError(
                        f"Failed to fetch model info for '{self.name}': {e}\n"
                        f"When using manual overrides, you must provide: num_parameters, num_layers, "
                        f"hidden_size, num_attention_heads, vocab_size. Missing: {', '.join(missing)}"
                    )
                # Set default num_kv_heads if not provided
                if self.num_kv_heads is None:
                    self.num_kv_heads = self.num_attention_heads
            else:
                raise ValueError(
                    f"Failed to fetch model info for '{self.name}': {e}\n"
                    f"For gated models, either set HF_TOKEN environment variable or provide "
                    f"manual parameters (num_parameters, num_layers, hidden_size, "
                    f"num_attention_heads, vocab_size)."
                )
    
    def get_num_parameters(self) -> float:
        """Get total number of parameters in billions."""
        # Use manual override if provided
        if self.num_parameters is not None:
            return self.num_parameters
        # Otherwise use HF value
        if self._model_info is not None:
            return model_total_params(self._model_info) / 1e9
        return 0.0
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        # Use manual override if provided
        if self.max_sequence_length is not None:
            return self.max_sequence_length
        # Otherwise use HF value
        if self._model_config is not None:
            return max_context_len(self._model_config)
        return 2048
    
    def get_model_memory_gb(self) -> float:
        """Get model memory requirement in GB."""
        # If we have HF info, use it (unless overridden)
        if self._model_info is not None and self._model_config is not None and self.num_parameters is None:
            return model_memory_req(self._model_info, self._model_config)
        # Fallback: simple calculation using num_parameters
        if self.num_parameters is not None:
            # Assume FP16 (2 bytes per parameter)
            return self.num_parameters * 2.0
        return 0.0
    
    def get_kv_cache_gb(self, context_len: int, batch_size: int = 1) -> float:
        """Get KV cache memory requirement in GB.
        
        Args:
            context_len: Context/sequence length in tokens
            batch_size: Batch size (default: 1)
            
        Returns:
            KV cache memory in GB
        """
        # If we have HF info and no manual overrides, use it
        if (self._model_info is not None and self._model_config is not None and 
            self.num_layers is None):
            return kv_cache_req(self._model_info, self._model_config, context_len, batch_size)
        
        # Fallback: manual calculation
        if (self.num_layers is not None and self.hidden_size is not None and 
            self.num_attention_heads is not None):
            head_dim = self.hidden_size // self.num_attention_heads
            num_kv_heads = self.num_kv_heads if self.num_kv_heads else self.num_attention_heads
            # 2 for K and V, 2 bytes for FP16
            kv_cache_bytes = (
                2 * self.num_layers * num_kv_heads * head_dim * 
                context_len * batch_size * 2
            )
            return kv_cache_bytes / (1024 ** 3)
        return 0.0
    
    def get_kv_cache_detail(self, context_len: int, batch_size: int = 1) -> Optional[KVCacheDetail]:
        """Get detailed KV cache information.
        
        Only available when HF info was successfully fetched.
        
        Args:
            context_len: Context/sequence length in tokens
            batch_size: Batch size (default: 1)
            
        Returns:
            KVCacheDetail object or None if HF info not available
        """
        if self._model_info is None or self._model_config is None:
            return None
        
        if self._kv_cache_detail is None or \
           self._kv_cache_detail.context_len != context_len or \
           self._kv_cache_detail.batch_size != batch_size:
            self._kv_cache_detail = KVCacheDetail(
                self._model_info, 
                self._model_config, 
                context_len, 
                batch_size
            )
        return self._kv_cache_detail


@dataclass
class GPUSpec:
    """Represents GPU hardware specifications.
    
    Attributes:
        name: Name/model of the GPU
        memory_gb: Total GPU memory in GB
        memory_bandwidth_gb_s: Memory bandwidth in GB/s
        tflops_fp16: Peak FP16 TFLOPS
        tflops_fp32: Peak FP32 TFLOPS
        cost_per_hour: Estimated cost per hour (optional)
    """
    name: str
    memory_gb: float
    memory_bandwidth_gb_s: float
    tflops_fp16: float
    tflops_fp32: float
    cost_per_hour: Optional[float] = None
