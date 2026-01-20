// Core data types for the GPU Config Recommender frontend

export interface ModelArchitecture {
  name: string;
  num_parameters: number;
  num_layers: number;
  hidden_size: number;
  num_attention_heads: number;
  vocab_size: number;
  max_sequence_length: number;
  architecture?: string;
}

export interface AddedModel extends ModelArchitecture {
  id: string;
  addedAt: Date;
  fetchedFrom: 'hf' | 'manual' | 'json';
}

export interface GPUSpec {
  name: string;
  memory_gb: number;
  memory_bandwidth_gb_s: number;
  tflops_fp16: number;
  tflops_fp32: number;
  cost_per_hour?: number;
}

export interface AddedGPU extends GPUSpec {
  id: string;
  addedAt: Date;
  source: 'library' | 'custom' | 'json';
}

export interface PerformanceEstimate {
  tokens_per_second: number;
  intertoken_latency_ms: number;
  memory_required_gb: number;
  fits_in_memory: boolean;
  tensor_parallel_size: number;
}

export interface CompatibleGPUInfo {
  gpu_name: string;
  fits: boolean;
  tokens_per_second?: number;
  intertoken_latency_ms?: number;
  memory_required_gb?: number;
  memory_available_gb?: number;
  cost_per_hour?: number;
  tensor_parallel_size?: number;
  meets_latency_requirement?: boolean;
}

export interface RecommendationResult {
  model_name: string;
  recommended_gpu: string | null;
  performance: PerformanceEstimate | null;
  all_compatible_gpus: CompatibleGPUInfo[];
  reasoning: string;
}

export interface EstimationParameters {
  precision: 'fp16' | 'fp32';
  input_length: number | null;
  output_length: number | null;
  memory_overhead_factor: number;
  latency_bound_ms: number | null;
}

export interface RecommendationRequest {
  models: Array<{ name: string }>;
  gpus: GPUSpec[];
  precision_bytes: number;
  memory_overhead_factor: number;
  latency_bound_ms?: number | null;
  input_length?: number | null;
  output_length?: number | null;
}

export interface RecommendationResponse {
  recommendations: RecommendationResult[];
}

export interface UIState {
  currentTab: 'dashboard' | 'recommend' | 'models' | 'gpus' | 'compare';
  isLoading: boolean;
  error: string | null;
}
