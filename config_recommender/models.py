"""Data models for GPU specs and model architectures."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArchitecture:
    """Represents a machine learning model's architecture details.
    
    Attributes:
        name: Name/identifier of the model
        num_parameters: Total number of parameters in billions
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        max_sequence_length: Maximum sequence length
        num_kv_heads: Number of key-value heads (for GQA/MQA), defaults to num_attention_heads
    """
    name: str
    num_parameters: float  # in billions
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    max_sequence_length: int = 2048
    num_kv_heads: Optional[int] = None
    
    def __post_init__(self):
        """Set default num_kv_heads if not provided."""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads


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
