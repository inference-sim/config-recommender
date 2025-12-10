"""Data models for GPU recommendations."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for a model and GPU combination."""
    model: str
    gpu: str
    num_gpus: int
    precision: str
    input_len: int
    output_len: int
    
    def __str__(self) -> str:
        return f"{self.model} on {self.num_gpus}x {self.gpu} ({self.precision})"


@dataclass
class PerformanceMetrics:
    """Performance metrics from benchmarking."""
    # Best Latency metrics
    ttft_ms: Optional[float] = None  # Time to first token
    itl_ms: Optional[float] = None  # Inter-token latency
    e2e_s: Optional[float] = None  # End-to-end latency
    
    # Best Throughput metrics
    output_tokens_per_s: Optional[float] = None
    input_tokens_per_s: Optional[float] = None
    requests_per_s: Optional[float] = None
    bottleneck: Optional[str] = None
    
    # Roofline Analysis
    hardware_ops_per_byte: Optional[float] = None
    prefill_arithmetic_intensity: Optional[float] = None
    decode_arithmetic_intensity: Optional[float] = None
    prefill_phase: Optional[str] = None
    decode_phase: Optional[str] = None
    
    # Concurrency Analysis
    kv_cache_memory_limit: Optional[int] = None
    prefill_compute_limit: Optional[int] = None
    decode_capacity_limit: Optional[int] = None
    theoretical_overall_limit: Optional[int] = None
    empirical_optimal_concurrency: Optional[int] = None


@dataclass
class PerformanceAnalysis:
    """Complete performance analysis for a model/GPU combination."""
    config: ModelConfig
    model_parameters: Optional[str] = None
    model_layers: Optional[int] = None
    metrics: Optional[PerformanceMetrics] = None
    
    def __str__(self) -> str:
        lines = [
            "=== Configuration ===",
            f"Model: {self.config.model}",
            f"GPU: {self.config.num_gpus}x {self.config.gpu}",
            f"Precision: {self.config.precision}",
            f"Input/Output: {self.config.input_len}/{self.config.output_len} tokens",
        ]
        
        if self.metrics:
            lines.append("\n=== Performance Analysis ===")
            if self.metrics.ttft_ms is not None:
                lines.append(f"Best Latency (concurrency=1):")
                lines.append(f"  TTFT: {self.metrics.ttft_ms} ms")
                lines.append(f"  ITL: {self.metrics.itl_ms} ms")
                lines.append(f"  E2E: {self.metrics.e2e_s} s")
            
            if self.metrics.output_tokens_per_s is not None:
                lines.append(f"Best Throughput (concurrency=256):")
                lines.append(f"  Output: {self.metrics.output_tokens_per_s} tokens/s")
                lines.append(f"  Input: {self.metrics.input_tokens_per_s} tokens/s")
                lines.append(f"  Requests: {self.metrics.requests_per_s} req/s")
                lines.append(f"  Bottleneck: {self.metrics.bottleneck}")
        
        return "\n".join(lines)


@dataclass
class GPURecommendation:
    """Recommendation for best GPU for a model."""
    model: str
    recommended_gpu: str
    all_analyses: List[PerformanceAnalysis]
    
    def get_best_analysis(self) -> Optional[PerformanceAnalysis]:
        """Get the performance analysis for the recommended GPU."""
        for analysis in self.all_analyses:
            if analysis.config.gpu == self.recommended_gpu:
                return analysis
        return None
    
    def __str__(self) -> str:
        best = self.get_best_analysis()
        return f"Model: {self.model}\nRecommended GPU: {self.recommended_gpu}\n{best if best else 'No analysis available'}"
