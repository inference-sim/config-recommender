"""Data models for GPU specs and model architectures."""

import os
from dataclasses import dataclass
from typing import Optional

try:
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
    HAS_CONFIG_EXPLORER = True
except ImportError:
    HAS_CONFIG_EXPLORER = False


@dataclass
class ModelArchitecture:
    """Represents a machine learning model's architecture details.
    
    Supports two modes:
    1. Auto-fetch from HuggingFace (recommended): Just provide the `name` (HF model identifier)
    2. Manual specification (fallback/offline): Provide all architecture parameters
    
    For gated models in auto-fetch mode, set the HF_TOKEN environment variable.
    
    Attributes:
        name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B") or custom name
        hf_token: Optional HuggingFace token (defaults to HF_TOKEN environment variable)
        
        # Manual specification fields (optional, for offline/testing):
        num_parameters: Total number of parameters in billions
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        max_sequence_length: Maximum sequence length
        num_kv_heads: Number of key-value heads (for GQA/MQA), defaults to num_attention_heads
        
        # Internal cached properties
        _model_info: Cached HuggingFace model info
        _model_config: Cached HuggingFace model config
        _kv_cache_detail: Cached KV cache detail object
        _use_hf: Whether using HuggingFace auto-fetch
    """
    name: str
    hf_token: Optional[str] = None
    
    # Manual specification fields (optional)
    num_parameters: Optional[float] = None  # in billions
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_sequence_length: int = 2048
    num_kv_heads: Optional[int] = None
    
    # Cached properties from HuggingFace
    _model_info: Optional['ModelInfo'] = None
    _model_config: Optional['AutoConfig'] = None
    _kv_cache_detail: Optional['KVCacheDetail'] = None
    _use_hf: bool = False
    
    def __post_init__(self):
        """Initialize model info from HuggingFace or validate manual spec."""
        # Use environment variable if token not provided
        if self.hf_token is None:
            self.hf_token = os.environ.get('HF_TOKEN')
        
        # Determine mode: auto-fetch or manual
        # If any manual field is provided, use manual mode
        if self.num_parameters is not None:
            # Manual mode - validate that we have minimum required fields
            if self.num_kv_heads is None:
                self.num_kv_heads = self.num_attention_heads
            self._use_hf = False
        else:
            # Auto-fetch mode - try to fetch from HuggingFace
            if not HAS_CONFIG_EXPLORER:
                raise ValueError(
                    f"config_explorer is not installed. Either install it or provide manual "
                    f"model specifications (num_parameters, num_layers, etc.)"
                )
            
            try:
                self._model_info = get_model_info_from_hf(self.name, self.hf_token)
                self._model_config = get_model_config_from_hf(self.name, self.hf_token)
                self._use_hf = True
            except Exception as e:
                raise ValueError(
                    f"Failed to fetch model info for '{self.name}': {e}\n"
                    f"If this is expected (offline mode), provide manual model specifications."
                )
    
    def get_num_parameters(self) -> float:
        """Get total number of parameters in billions."""
        if self._use_hf and self._model_info is not None:
            return model_total_params(self._model_info) / 1e9
        return self.num_parameters or 0.0
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length."""
        if self._use_hf and self._model_config is not None:
            return max_context_len(self._model_config)
        return self.max_sequence_length
    
    def get_model_memory_gb(self) -> float:
        """Get model memory requirement in GB."""
        if self._use_hf and self._model_info is not None and self._model_config is not None:
            return model_memory_req(self._model_info, self._model_config)
        # Fallback: simple calculation for manual mode
        if self.num_parameters is not None:
            # Assume FP16 (2 bytes per parameter)
            return self.num_parameters * 2.0
        return 0.0
    
    def get_kv_cache_gb(self, context_len: int, batch_size: int = 1) -> float:
        """Get KV cache memory requirement in GB."""
        if self._use_hf and self._model_info is not None and self._model_config is not None:
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
    
    def get_kv_cache_detail(self, context_len: int = 1, batch_size: int = 1) -> Optional['KVCacheDetail']:
        """Get detailed KV cache information."""
        if not self._use_hf:
            return None  # Not available in manual mode
        
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
