import { apiClient } from './api';
import type {
  AddedModel,
  AddedGPU,
  RecommendationResponse,
  EstimationParameters,
} from '@/types';

export const recommendationService = {
  async generateRecommendations(
    models: AddedModel[],
    gpus: AddedGPU[],
    parameters: EstimationParameters
  ): Promise<RecommendationResponse> {
    // Backend currently only supports single model per request
    // So we'll make multiple requests and combine results
    const allRecommendations = [];

    for (const model of models) {
      // Calculate sequence_length from input + output
      let sequence_length: number | null = null;
      if (parameters.input_length && parameters.output_length) {
        sequence_length = parameters.input_length + parameters.output_length;
      } else if (parameters.input_length) {
        sequence_length = parameters.input_length;
      } else if (parameters.output_length) {
        sequence_length = parameters.output_length;
      }

      const request = {
        model: {
          name: model.name,
        },
        available_gpus: gpus.map((g) => ({
          name: g.name,
          memory_gb: g.memory_gb,
          memory_bandwidth_gb_s: g.memory_bandwidth_gb_s,
          tflops_fp16: g.tflops_fp16,
          tflops_fp32: g.tflops_fp32,
          cost_per_hour: g.cost_per_hour,
        })),
        sequence_length: sequence_length,
        latency_bound_ms: parameters.latency_bound_ms || null,
      };

      const response = await apiClient.post('/recommendations', request);
      allRecommendations.push(response.data);
    }

    return {
      recommendations: allRecommendations,
    };
  },

  exportJSON(data: unknown): Blob {
    return new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
  },

  exportCSV(data: unknown[]): Blob {
    // Basic CSV export implementation
    const headers = Object.keys(data[0] || {});
    const rows = data.map((row) => headers.map((h) => (row as any)[h]).join(','));
    const csv = [headers.join(','), ...rows].join('\n');
    return new Blob([csv], { type: 'text/csv' });
  },
};
