"""Parser for BentoML llm-optimizer output."""

import re
from typing import Optional
from .models import ModelConfig, PerformanceMetrics, PerformanceAnalysis


def parse_bentoml_output(output: str, model: str, gpu: str, num_gpus: int = 1, 
                         input_len: int = 2000, output_len: int = 256) -> Optional[PerformanceAnalysis]:
    """
    Parse the output from llm-optimizer estimate command.
    
    Args:
        output: The raw output string from llm-optimizer
        model: Model name
        gpu: GPU type
        num_gpus: Number of GPUs
        input_len: Input token length
        output_len: Output token length
        
    Returns:
        PerformanceAnalysis object or None if parsing fails
    """
    try:
        # Extract precision
        precision_match = re.search(r'Inferred precision from model config:\s*(\w+)', output)
        precision = precision_match.group(1) if precision_match else "unknown"
        
        # Create config
        config = ModelConfig(
            model=model,
            gpu=gpu,
            num_gpus=num_gpus,
            precision=precision,
            input_len=input_len,
            output_len=output_len
        )
        
        # Extract model info
        model_params = None
        model_layers = None
        model_info_match = re.search(r'Model:\s*([\d.]+[BMK]?)\s*parameters,\s*(\d+)\s*layers', output)
        if model_info_match:
            model_params = model_info_match.group(1)
            model_layers = int(model_info_match.group(2))
        
        # Initialize metrics
        metrics = PerformanceMetrics()
        
        # Parse Best Latency section
        ttft_match = re.search(r'TTFT:\s*([\d.]+)\s*ms', output)
        if ttft_match:
            metrics.ttft_ms = float(ttft_match.group(1))
            
        itl_match = re.search(r'ITL:\s*([\d.]+)\s*ms', output)
        if itl_match:
            metrics.itl_ms = float(itl_match.group(1))
            
        e2e_match = re.search(r'E2E:\s*([\d.]+)\s*s', output)
        if e2e_match:
            metrics.e2e_s = float(e2e_match.group(1))
        
        # Parse Best Throughput section
        output_tokens_match = re.search(r'Output:\s*([\d.]+)\s*tokens/s', output)
        if output_tokens_match:
            metrics.output_tokens_per_s = float(output_tokens_match.group(1))
            
        input_tokens_match = re.search(r'Input:\s*([\d.]+)\s*tokens/s', output)
        if input_tokens_match:
            metrics.input_tokens_per_s = float(input_tokens_match.group(1))
            
        requests_match = re.search(r'Requests:\s*([\d.]+)\s*req/s', output)
        if requests_match:
            metrics.requests_per_s = float(requests_match.group(1))
            
        bottleneck_match = re.search(r'Bottleneck:\s*(\w+)', output)
        if bottleneck_match:
            metrics.bottleneck = bottleneck_match.group(1)
        
        # Parse Roofline Analysis
        hw_ops_match = re.search(r'Hardware Ops/Byte Ratio:\s*([\d.]+)\s*ops/byte', output)
        if hw_ops_match:
            metrics.hardware_ops_per_byte = float(hw_ops_match.group(1))
            
        prefill_ai_match = re.search(r'Prefill Arithmetic Intensity:\s*([\d.]+)\s*ops/byte', output)
        if prefill_ai_match:
            metrics.prefill_arithmetic_intensity = float(prefill_ai_match.group(1))
            
        decode_ai_match = re.search(r'Decode Arithmetic Intensity:\s*([\d.]+)\s*ops/byte', output)
        if decode_ai_match:
            metrics.decode_arithmetic_intensity = float(decode_ai_match.group(1))
            
        prefill_phase_match = re.search(r'Prefill Phase:\s*(.+?)(?:\n|$)', output)
        if prefill_phase_match:
            metrics.prefill_phase = prefill_phase_match.group(1).strip()
            
        decode_phase_match = re.search(r'Decode Phase:\s*(.+?)(?:\n|$)', output)
        if decode_phase_match:
            metrics.decode_phase = decode_phase_match.group(1).strip()
        
        # Parse Concurrency Analysis
        kv_cache_match = re.search(r'KV Cache Memory Limit:\s*(\d+)\s*concurrent requests', output)
        if kv_cache_match:
            metrics.kv_cache_memory_limit = int(kv_cache_match.group(1))
            
        prefill_compute_match = re.search(r'Prefill Compute Limit:\s*(\d+)\s*concurrent requests', output)
        if prefill_compute_match:
            metrics.prefill_compute_limit = int(prefill_compute_match.group(1))
            
        decode_capacity_match = re.search(r'Decode Capacity Limit:\s*(\d+)\s*concurrent requests', output)
        if decode_capacity_match:
            metrics.decode_capacity_limit = int(decode_capacity_match.group(1))
            
        theoretical_limit_match = re.search(r'Theoretical Overall Limit:\s*(\d+)\s*concurrent requests', output)
        if theoretical_limit_match:
            metrics.theoretical_overall_limit = int(theoretical_limit_match.group(1))
            
        empirical_concurrency_match = re.search(r'Empirical Optimal Concurrency:\s*(\d+)\s*concurrent requests', output)
        if empirical_concurrency_match:
            metrics.empirical_optimal_concurrency = int(empirical_concurrency_match.group(1))
        
        return PerformanceAnalysis(
            config=config,
            model_parameters=model_params,
            model_layers=model_layers,
            metrics=metrics
        )
    except Exception as e:
        print(f"Error parsing BentoML output: {e}")
        return None
