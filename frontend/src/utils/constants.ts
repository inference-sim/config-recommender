import type { GPUSpec } from '@/types';

// Preloaded GPU library from the backend
export const GPU_LIBRARY: GPUSpec[] = [
  {
    name: 'NVIDIA H100 80GB',
    memory_gb: 80,
    memory_bandwidth_gb_s: 2039,
    tflops_fp16: 312,
    tflops_fp32: 156,
    cost_per_hour: 2.38,
  },
  {
    name: 'NVIDIA H200 141GB',
    memory_gb: 141,
    memory_bandwidth_gb_s: 4800,
    tflops_fp16: 532,
    tflops_fp32: 266,
    cost_per_hour: 3.5,
  },
  {
    name: 'NVIDIA A100 80GB',
    memory_gb: 80,
    memory_bandwidth_gb_s: 2039,
    tflops_fp16: 312,
    tflops_fp32: 156,
    cost_per_hour: 3.67,
  },
  {
    name: 'NVIDIA A100 40GB',
    memory_gb: 40,
    memory_bandwidth_gb_s: 1555,
    tflops_fp16: 312,
    tflops_fp32: 156,
    cost_per_hour: 2.0,
  },
  {
    name: 'NVIDIA L40',
    memory_gb: 48,
    memory_bandwidth_gb_s: 864,
    tflops_fp16: 362,
    tflops_fp32: 181,
    cost_per_hour: 1.08,
  },
  {
    name: 'NVIDIA L4',
    memory_gb: 24,
    memory_bandwidth_gb_s: 300,
    tflops_fp16: 121,
    tflops_fp32: 60,
    cost_per_hour: 0.5,
  },
  {
    name: 'NVIDIA V100',
    memory_gb: 32,
    memory_bandwidth_gb_s: 900,
    tflops_fp16: 125,
    tflops_fp32: 62,
    cost_per_hour: 0.85,
  },
];

export const DEFAULT_PARAMETERS = {
  precision: 'fp16' as const,
  input_length: null,
  output_length: null,
  memory_overhead_factor: 1.2,
  latency_bound_ms: null,
};
